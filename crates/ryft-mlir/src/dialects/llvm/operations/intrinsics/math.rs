use crate::{
    AttributeRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value,
    ValueRef, mlir_op,
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Acos);

/// Constructs a new detached `llvm.intr.acos` operation.
pub fn intr_acos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedAcosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ACOS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_acos`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Asin);

/// Constructs a new detached `llvm.intr.asin` operation.
pub fn intr_asin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedAsinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ASIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_asin`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedAtan2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ATAN2_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_atan2`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Atan);

/// Constructs a new detached `llvm.intr.atan` operation.
pub fn intr_atan<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedAtanOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ATAN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_atan`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `is_int_min_poison` attribute.
    fn is_int_min_poison(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("is_int_min_poison")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "is_int_min_poison",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Abs);

/// Constructs a new detached `llvm.intr.abs` operation.
pub fn intr_abs<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_int_min_poison: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedAbsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ABS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_int_min_poison", is_int_min_poison);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_abs`"))
    })
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
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Assume);

/// Constructs a new detached `llvm.intr.assume` operation.
pub fn intr_assume<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    condition: V0,
    location: L,
) -> Result<DetachedAssumeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ASSUME_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_attribute("op_bundle_sizes", context.dense_i32_array_attribute(&[])?);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_assume`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(BitReverse);

/// Constructs a new detached `llvm.intr.bitreverse` operation.
pub fn intr_bit_reverse<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedBitReverseOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(BIT_REVERSE_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_bit_reverse`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(ByteSwap);

/// Constructs a new detached `llvm.intr.bswap` operation.
pub fn intr_bswap<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedByteSwapOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(BYTE_SWAP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_bswap`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedCopySignOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COPY_SIGN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_copy_sign`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Cos);

/// Constructs a new detached `llvm.intr.cos` operation.
pub fn intr_cos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedCosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_cos`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Cosh);

/// Constructs a new detached `llvm.intr.cosh` operation.
pub fn intr_cosh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedCoshOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COSH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_cosh`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `is_zero_poison` attribute.
    fn is_zero_poison(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("is_zero_poison")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "is_zero_poison",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CountLeadingZeros);

/// Constructs a new detached `llvm.intr.ctlz` operation.
pub fn intr_ctlz<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_zero_poison: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedCountLeadingZerosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COUNT_LEADING_ZEROS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_zero_poison", is_zero_poison);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ctlz`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `is_zero_poison` attribute.
    fn is_zero_poison(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("is_zero_poison")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "is_zero_poison",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CountTrailingZeros);

/// Constructs a new detached `llvm.intr.cttz` operation.
pub fn intr_count_trailing_zeros<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    is_zero_poison: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedCountTrailingZerosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(COUNT_TRAILING_ZEROS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("is_zero_poison", is_zero_poison);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_count_trailing_zeros`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(CtPop);

/// Constructs a new detached `llvm.intr.ctpop` operation.
pub fn intr_ct_pop<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedCtPopOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(CT_POP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ct_pop`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Exp10);

/// Constructs a new detached `llvm.intr.exp10` operation.
pub fn intr_exp10<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedExp10Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EXP10_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_exp10`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Exp2);

/// Constructs a new detached `llvm.intr.exp2` operation.
pub fn intr_exp2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedExp2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EXP2_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_exp2`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Exp);

/// Constructs a new detached `llvm.intr.exp` operation.
pub fn intr_exp<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedExpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EXP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_exp`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `expected` operand.
    fn expected(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedExpectOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EXPECT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(expected);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_expect`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `expected` operand.
    fn expected(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `prob` attribute.
    fn prob(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("prob")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "prob",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedExpectWithProbabilityOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(EXPECT_WITH_PROBABILITY_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(expected);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("prob", prob);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_expect_with_probability`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Fabs);

/// Constructs a new detached `llvm.intr.fabs` operation.
pub fn intr_fabs<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFabsOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FABS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fabs`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Fceil);

/// Constructs a new detached `llvm.intr.ceil` operation.
pub fn intr_ceil<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFceilOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FCEIL_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ceil`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Ffloor);

/// Constructs a new detached `llvm.intr.floor` operation.
pub fn intr_floor<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFfloorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FFLOOR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_floor`"))
    })
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
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `second` operand.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `third` operand.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FMA_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fma`"))
    })
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
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `second` operand.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `third` operand.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedFmulAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FMUL_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fmuladd`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Ftrunc);

/// Constructs a new detached `llvm.intr.trunc` operation.
pub fn intr_trunc<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFtruncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_trunc`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(FractionExp);

/// Constructs a new detached `llvm.intr.frexp` operation.
pub fn intr_frexp<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedFractionExpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FRACTION_EXP_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_frexp`"))
    })
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
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `second` operand.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `third` operand.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedFshlOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FSHL_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fshl`"))
    })
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
    fn first(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `second` operand.
    fn second(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `third` operand.
    fn third(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedFshrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(FSHR_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_fshr`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(IsConstant);

/// Constructs a new detached `llvm.intr.is.constant` operation.
pub fn intr_is_constant<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedIsConstantOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(IS_CONSTANT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_is_constant`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `bit` attribute.
    fn bit(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("bit")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "bit",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(IsFpclass);

/// Constructs a new detached `llvm.intr.is.fpclass` operation.
pub fn intr_is_fpclass<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    bit: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedIsFpclassOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(IS_FPCLASS_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("bit", bit);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_is_fpclass`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Llrint);

/// Constructs a new detached `llvm.intr.llrint` operation.
pub fn intr_llrint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedLlrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LLRINT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_llrint`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Llround);

/// Constructs a new detached `llvm.intr.llround` operation.
pub fn intr_llround<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedLlroundOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LLROUND_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_llround`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `power` operand.
    fn power(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedLoadExpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LOAD_EXP_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(power);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ldexp`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Log10);

/// Constructs a new detached `llvm.intr.log10` operation.
pub fn intr_log10<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedLog10Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LOG10_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_log10`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Log2);

/// Constructs a new detached `llvm.intr.log2` operation.
pub fn intr_log2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedLog2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LOG2_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_log2`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Log);

/// Constructs a new detached `llvm.intr.log` operation.
pub fn intr_log<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedLogOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LOG_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_log`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Lrint);

/// Constructs a new detached `llvm.intr.lrint` operation.
pub fn intr_lrint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedLrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LRINT_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_lrint`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Lround);

/// Constructs a new detached `llvm.intr.lround` operation.
pub fn intr_lround<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedLroundOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LROUND_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_lround`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMaxNumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MAX_NUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_maxnum`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMaximumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MAXIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_maximum`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMinNumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MIN_NUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_min_num`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMinimumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MINIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_minimum`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(NearbyInt);

/// Constructs a new detached `llvm.intr.nearbyint` operation.
pub fn intr_nearby_int<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedNearbyIntOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(NEARBY_INT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_nearby_int`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `power` operand.
    fn power(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedPowIOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(POW_I_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(power);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_powi`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedPowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(POW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_pow`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Rint);

/// Constructs a new detached `llvm.intr.rint` operation.
pub fn intr_rint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedRintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(RINT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_rint`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(RoundEven);

/// Constructs a new detached `llvm.intr.roundeven` operation.
pub fn intr_round_even<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedRoundEvenOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ROUND_EVEN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_round_even`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Round);

/// Constructs a new detached `llvm.intr.round` operation.
pub fn intr_round<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedRoundOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(ROUND_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_round`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSaddSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SADD_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sadd_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSaddWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sadd_with_overflow`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedScmpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SCMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_scmp`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_smax`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_smin`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSmulWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_smul_with_overflow`"))
    })
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
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(SsaCopy);

/// Constructs a new detached `llvm.intr.ssa.copy` operation.
pub fn intr_ssa_copy<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    operand: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedSsaCopyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SSA_COPY_OPERATION_NAME, location);
    builder = builder.add_operand(operand);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ssa_copy`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSshlSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SSHL_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sshl_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSsubSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SSUB_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ssub_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedSsubWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SSUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ssub_with_overflow`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Sin);

/// Constructs a new detached `llvm.intr.sin` operation.
pub fn intr_sin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedSinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sin`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Sincos);

/// Constructs a new detached `llvm.intr.sincos` operation.
pub fn intr_sincos<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    value: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedSincosOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SINCOS_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sincos`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Sinh);

/// Constructs a new detached `llvm.intr.sinh` operation.
pub fn intr_sinh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedSinhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SINH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sinh`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Sqrt);

/// Constructs a new detached `llvm.intr.sqrt` operation.
pub fn intr_sqrt<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedSqrtOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(SQRT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_sqrt`"))
    })
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
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(StepVector);

/// Constructs a new detached `llvm.intr.stepvector` operation.
pub fn intr_stepvector<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> Result<DetachedStepVectorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(STEP_VECTOR_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_stepvector`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Tan);

/// Constructs a new detached `llvm.intr.tan` operation.
pub fn intr_tan<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedTanOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(TAN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_tan`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `fastmathFlags` attribute.
    fn fastmath_flags(&self) -> Result<Option<AttributeRef<'c, 't>>, Error> {
        self.attribute("fastmathFlags")
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Tanh);

/// Constructs a new detached `llvm.intr.tanh` operation.
pub fn intr_tanh<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedTanhOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(TANH_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_tanh`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUaddSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UADD_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_uadd_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUaddWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_uadd_with_overflow`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUcmpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UCMP_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ucmp`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_umax`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_umin`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUmulWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(UMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_umul_with_overflow`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUshlSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(USHL_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ushl_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUsubSatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(USUB_SAT_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_usub_sat`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedUsubWithOverflowOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(USUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_usub_with_overflow`"))
    })
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::{Attribute, Block, Context, DialectHandle, Operation, Type};

    use super::*;

    #[test]
    fn test_intr_acos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_acos(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.acos");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_acos_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_acos_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.acos(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_asin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_asin(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.asin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_asin_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_asin_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.asin(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_atan2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_atan2(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.atan2");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_atan2_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_atan2_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.atan2(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_atan() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_atan(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.atan");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_atan_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_atan_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.atan(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_abs() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let is_int_min_poison = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_abs(arg_0, i32_type, is_int_min_poison, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.is_int_min_poison().unwrap(), is_int_min_poison);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.abs");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_abs_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_abs_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.abs\"(%arg0) <{is_int_min_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_assume() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i1_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_assume(arg_0, location).unwrap();
                assert_eq!(op.condition().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.assume");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_assume_test",
                    func::FuncAttributes { arguments: vec![i1_type.into()], results: vec![], ..Default::default() },
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
                  func.func @llvm_intr_assume_test(%arg0: i1) {
                    llvm.intr.assume %arg0 : i1
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_bit_reverse() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_bit_reverse(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.bitreverse");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_bitreverse_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_bitreverse_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.bitreverse(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_bswap() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_bswap(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.bswap");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_bswap_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_bswap_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.bswap(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_copy_sign() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_copy_sign(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.copysign");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_copysign_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_copysign_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.copysign(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_cos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_cos(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.cos");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_cos_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_cos_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.cos(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_cosh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_cosh(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.cosh");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_cosh_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_cosh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.cosh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ctlz() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let is_zero_poison = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_ctlz(arg_0, i32_type, is_zero_poison, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.is_zero_poison().unwrap(), is_zero_poison);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ctlz");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ctlz_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ctlz_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.ctlz\"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_count_trailing_zeros() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let is_zero_poison = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_count_trailing_zeros(arg_0, i32_type, is_zero_poison, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.is_zero_poison().unwrap(), is_zero_poison);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.cttz");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_cttz_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_cttz_test(%arg0: i32) -> i32 {
                    %0 = \"llvm.intr.cttz\"(%arg0) <{is_zero_poison = false}> : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ct_pop() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_ct_pop(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ctpop");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ctpop_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ctpop_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.ctpop(%arg0) : (i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp10() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_exp10(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.exp10");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_exp10_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_exp10_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp10(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_exp2(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.exp2");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_exp2_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_exp2_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp2(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_exp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_exp(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.exp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_exp_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_exp_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.exp(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_expect() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_expect(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.expected().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.expect");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_expect_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_expect_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.expect %arg0, %arg1 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_expect_with_probability() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f64_type = context.float64_type();
        let prob = context.float_attribute(f64_type, 5.000000e-01).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_expect_with_probability(arg_0, arg_1, i32_type, prob, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.expected().unwrap(), arg_1);
                assert_eq!(op.prob().unwrap(), prob);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.expect.with.probability");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_expect_with_probability_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_expect_with_probability_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.expect.with.probability %arg0, %arg1, 5.000000e-01 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fabs() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_fabs(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.fabs");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_fabs_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_fabs_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.fabs(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ceil() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_ceil(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ceil");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ceil_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_ceil_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.ceil(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_floor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_floor(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.floor");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_floor_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_floor_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.floor(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fma() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_fma(arg_0, arg_1, arg_2, f32_type, None, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.fma");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_fma_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_fma_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.fma(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fmuladd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                    (f32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_fmuladd(arg_0, arg_1, arg_2, f32_type, None, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.fmuladd");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_fmuladd_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_fmuladd_test(%arg0: f32, %arg1: f32, %arg2: f32) -> f32 {
                    %0 = llvm.intr.fmuladd(%arg0, %arg1, %arg2) : (f32, f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_trunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_trunc(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.trunc");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_trunc_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_trunc_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.trunc(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_frexp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[f32_type.as_ref(), i32_type.as_ref()], false).unwrap();
                let op = intr_frexp(arg_0, result_type, None, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.frexp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_frexp_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_frexp_test(%arg0: f32) -> !llvm.struct<(f32, i32)> {
                    %0 = llvm.intr.frexp(%arg0) : (f32) -> !llvm.struct<(f32, i32)>
                    return %0 : !llvm.struct<(f32, i32)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fshl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_fshl(arg_0, arg_1, arg_2, i32_type, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.fshl");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_fshl_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_fshl_test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = llvm.intr.fshl(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_fshr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_fshr(arg_0, arg_1, arg_2, i32_type, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.fshr");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_fshr_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_fshr_test(%arg0: i32, %arg1: i32, %arg2: i32) -> i32 {
                    %0 = llvm.intr.fshr(%arg0, %arg1, %arg2) : (i32, i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_is_constant() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_is_constant(arg_0, i1_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i1_type);
                assert_eq!(op.operation_name(), "llvm.intr.is.constant");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_is_constant_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
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
                  func.func @llvm_intr_is_constant_test(%arg0: i32) -> i1 {
                    %0 = \"llvm.intr.is.constant\"(%arg0) : (i32) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_is_fpclass() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let bit = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_is_fpclass(arg_0, i1_type, bit, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.bit().unwrap(), bit);
                assert_eq!(op.output_type().unwrap(), i1_type);
                assert_eq!(op.operation_name(), "llvm.intr.is.fpclass");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_is_fpclass_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
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
                  func.func @llvm_intr_is_fpclass_test(%arg0: f32) -> i1 {
                    %0 = \"llvm.intr.is.fpclass\"(%arg0) <{bit = 1 : i32}> : (f32) -> i1
                    return %0 : i1
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_llrint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_llrint(arg_0, i64_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.llrint");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_llrint_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![i64_type.into()],
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
                  func.func @llvm_intr_llrint_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.llrint(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_llround() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_llround(arg_0, i64_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.llround");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_llround_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![i64_type.into()],
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
                  func.func @llvm_intr_llround_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.llround(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ldexp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_ldexp(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.power().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ldexp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ldexp_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), i32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_ldexp_test(%arg0: f32, %arg1: i32) -> f32 {
                    %0 = llvm.intr.ldexp(%arg0, %arg1) : (f32, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log10() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_log10(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.log10");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_log10_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_log10_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log10(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_log2(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.log2");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_log2_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_log2_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log2(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_log() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_log(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.log");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_log_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_log_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.log(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lrint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_lrint(arg_0, i64_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.lrint");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_lrint_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![i64_type.into()],
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
                  func.func @llvm_intr_lrint_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.lrint(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lround() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_lround(arg_0, i64_type, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.lround");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_lround_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![i64_type.into()],
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
                  func.func @llvm_intr_lround_test(%arg0: f32) -> i64 {
                    %0 = llvm.intr.lround(%arg0) : (f32) -> i64
                    return %0 : i64
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_maxnum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_maxnum(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.maxnum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_maxnum_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_maxnum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.maxnum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_maximum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_maximum(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.maximum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_maximum_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_maximum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.maximum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_min_num() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_min_num(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.minnum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_minnum_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_minnum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.minnum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_minimum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_minimum(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.minimum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_minimum_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_minimum_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.minimum(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_nearby_int() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_nearby_int(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.nearbyint");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_nearbyint_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_nearbyint_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.nearbyint(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_powi() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_powi(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.power().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.powi");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_powi_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), i32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_powi_test(%arg0: f32, %arg1: i32) -> f32 {
                    %0 = llvm.intr.powi(%arg0, %arg1) : (f32, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_pow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_pow(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.pow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_pow_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_pow_test(%arg0: f32, %arg1: f32) -> f32 {
                    %0 = llvm.intr.pow(%arg0, %arg1) : (f32, f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_rint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_rint(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.rint");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_rint_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_rint_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.rint(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_round_even() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_round_even(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.roundeven");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_roundeven_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_roundeven_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.roundeven(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_round() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_round(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.round");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_round_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_round_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.round(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sadd_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_sadd_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.sadd.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sadd_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_sadd_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.sadd.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sadd_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_sadd_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.sadd.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sadd_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_sadd_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.sadd.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_scmp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_scmp(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.scmp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_scmp_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_scmp_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.scmp(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_smax(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.smax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_smax_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_smax_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.smax(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_smin(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.smin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_smin_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_smin_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.smin(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_smul_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_smul_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.smul.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_smul_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_smul_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.smul.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssa_copy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_ssa_copy(arg_0, i32_type, location).unwrap();
                assert_eq!(op.operand_value(0).unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ssa.copy");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ssa_copy_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ssa_copy_test(%arg0: i32) -> i32 {
                    %0 = llvm.intr.ssa.copy %arg0 : i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sshl_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_sshl_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.sshl.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sshl_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_sshl_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.sshl.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssub_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_ssub_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ssub.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ssub_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ssub_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ssub.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ssub_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_ssub_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.ssub.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ssub_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_ssub_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.ssub.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_sin(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.sin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sin_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_sin_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sin(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sincos() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[f32_type.as_ref(), f32_type.as_ref()], false).unwrap();
                let op = intr_sincos(arg_0, result_type, None, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.sincos");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sincos_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_sincos_test(%arg0: f32) -> !llvm.struct<(f32, f32)> {
                    %0 = llvm.intr.sincos(%arg0) : (f32) -> !llvm.struct<(f32, f32)>
                    return %0 : !llvm.struct<(f32, f32)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sinh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_sinh(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.sinh");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sinh_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_sinh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sinh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_sqrt() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_sqrt(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.sqrt");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_sqrt_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_sqrt_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sqrt(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stepvector() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_stepvector(vector_i32_type.as_ref(), location).unwrap();
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.stepvector");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_stepvector_test",
                    func::FuncAttributes {
                        arguments: vec![],
                        results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_stepvector_test() -> vector<4xi32> {
                    %0 = llvm.intr.stepvector : vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_tan() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_tan(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.tan");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_tan_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_tan_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.tan(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_tanh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_tanh(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.tanh");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_tanh_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into()],
                        results: vec![f32_type.into()],
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
                  func.func @llvm_intr_tanh_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.tanh(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_uadd_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_uadd_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.uadd.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_uadd_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_uadd_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.uadd.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_uadd_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_uadd_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.uadd.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_uadd_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_uadd_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.uadd.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ucmp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_ucmp(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ucmp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ucmp_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ucmp_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ucmp(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_umax(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.umax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_umax_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_umax_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.umax(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_umin(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.umin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_umin_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_umin_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.umin(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_umul_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_umul_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.umul.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_umul_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_umul_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.umul.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ushl_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_ushl_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.ushl.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ushl_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_ushl_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.ushl.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_usub_sat() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_usub_sat(arg_0, arg_1, i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.usub.sat");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_usub_sat_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![i32_type.into()],
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
                  func.func @llvm_intr_usub_sat_test(%arg0: i32, %arg1: i32) -> i32 {
                    %0 = llvm.intr.usub.sat(%arg0, %arg1) : (i32, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_usub_with_overflow() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i1_type = context.signless_integer_type(1);
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let result_type =
                    context.llvm_literal_struct_type(&[i32_type.as_ref(), i1_type.as_ref()], false).unwrap();
                let op = intr_usub_with_overflow(arg_0, arg_1, result_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), result_type);
                assert_eq!(op.operation_name(), "llvm.intr.usub.with.overflow");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_usub_with_overflow_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![result_type.into()],
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
                  func.func @llvm_intr_usub_with_overflow_test(%arg0: i32, %arg1: i32) -> !llvm.struct<(i32, i1)> {
                    %0 = \"llvm.intr.usub.with.overflow\"(%arg0, %arg1) : (i32, i32) -> !llvm.struct<(i32, i1)>
                    return %0 : !llvm.struct<(i32, i1)>
                  }
                }
            "},
        );
    }
}
