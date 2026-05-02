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

/// Canonical MLIR operation name for [`AnnotationOperation`].
pub const ANNOTATION_OPERATION_NAME: &str = "llvm.intr.annotation";

/// Operation trait for `llvm.intr.annotation`.
pub trait AnnotationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ANNOTATION_OPERATION_NAME
    }

    /// Returns the `integer` operand.
    fn integer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `line` operand.
    fn line(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Annotation);

/// Constructs a new detached `llvm.intr.annotation` operation.
pub fn intr_annotation<
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
    integer: V0,
    annotation: V1,
    file_name: V2,
    line: V3,
    result_type: T0,
    location: L,
) -> DetachedAnnotationOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(integer);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_annotation`")
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
pub fn intr_bitreverse<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::intr_bitreverse`")
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

/// Canonical MLIR operation name for [`ConstrainedFaddOperation`].
pub const CONSTRAINED_FADD_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fadd";

/// Operation trait for `llvm.intr.experimental.constrained.fadd`.
pub trait ConstrainedFaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FADD_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFadd);

/// Constructs a new detached `llvm.intr.experimental.constrained.fadd` operation.
pub fn intr_experimental_constrained_fadd<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFaddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fadd`")
}

/// Canonical MLIR operation name for [`ConstrainedFdivOperation`].
pub const CONSTRAINED_FDIV_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fdiv";

/// Operation trait for `llvm.intr.experimental.constrained.fdiv`.
pub trait ConstrainedFdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FDIV_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFdiv);

/// Constructs a new detached `llvm.intr.experimental.constrained.fdiv` operation.
pub fn intr_experimental_constrained_fdiv<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFdivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FDIV_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fdiv`")
}

/// Canonical MLIR operation name for [`ConstrainedFmaOperation`].
pub const CONSTRAINED_FMA_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fma";

/// Operation trait for `llvm.intr.experimental.constrained.fma`.
pub trait ConstrainedFmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FMA_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `argument_2` operand.
    fn argument_2(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFma);

/// Constructs a new detached `llvm.intr.experimental.constrained.fma` operation.
pub fn intr_experimental_constrained_fma<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    argument_2: V2,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFmaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FMA_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_operand(argument_2);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fma`")
}

/// Canonical MLIR operation name for [`ConstrainedFmulAddOperation`].
pub const CONSTRAINED_FMUL_ADD_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fmuladd";

/// Operation trait for `llvm.intr.experimental.constrained.fmuladd`.
pub trait ConstrainedFmulAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FMUL_ADD_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `argument_2` operand.
    fn argument_2(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFmulAdd);

/// Constructs a new detached `llvm.intr.experimental.constrained.fmuladd` operation.
pub fn intr_experimental_constrained_fmuladd<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    argument_2: V2,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFmulAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FMUL_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_operand(argument_2);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fmuladd`")
}

/// Canonical MLIR operation name for [`ConstrainedFmulOperation`].
pub const CONSTRAINED_FMUL_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fmul";

/// Operation trait for `llvm.intr.experimental.constrained.fmul`.
pub trait ConstrainedFmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FMUL_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFmul);

/// Constructs a new detached `llvm.intr.experimental.constrained.fmul` operation.
pub fn intr_experimental_constrained_fmul<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fmul`")
}

/// Canonical MLIR operation name for [`ConstrainedFpextOperation`].
pub const CONSTRAINED_FPEXT_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fpext";

/// Operation trait for `llvm.intr.experimental.constrained.fpext`.
pub trait ConstrainedFpextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FPEXT_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFpext);

/// Constructs a new detached `llvm.intr.experimental.constrained.fpext` operation.
pub fn intr_experimental_constrained_fpext<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    result_type: T0,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFpextOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FPEXT_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fpext`")
}

/// Canonical MLIR operation name for [`ConstrainedFptruncOperation`].
pub const CONSTRAINED_FPTRUNC_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fptrunc";

/// Operation trait for `llvm.intr.experimental.constrained.fptrunc`.
pub trait ConstrainedFptruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FPTRUNC_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFptrunc);

/// Constructs a new detached `llvm.intr.experimental.constrained.fptrunc` operation.
pub fn intr_experimental_constrained_fptrunc<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFptruncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FPTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fptrunc`")
}

/// Canonical MLIR operation name for [`ConstrainedFremOperation`].
pub const CONSTRAINED_FREM_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.frem";

/// Operation trait for `llvm.intr.experimental.constrained.frem`.
pub trait ConstrainedFremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FREM_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFrem);

/// Constructs a new detached `llvm.intr.experimental.constrained.frem` operation.
pub fn intr_experimental_constrained_frem<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFremOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FREM_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_frem`")
}

/// Canonical MLIR operation name for [`ConstrainedFsubOperation`].
pub const CONSTRAINED_FSUB_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.fsub";

/// Operation trait for `llvm.intr.experimental.constrained.fsub`.
pub trait ConstrainedFsubOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_FSUB_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `argument_1` operand.
    fn argument_1(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedFsub);

/// Constructs a new detached `llvm.intr.experimental.constrained.fsub` operation.
pub fn intr_experimental_constrained_fsub<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    argument_1: V1,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedFsubOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_FSUB_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_operand(argument_1);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_fsub`")
}

/// Canonical MLIR operation name for [`ConstrainedSitoFpOperation`].
pub const CONSTRAINED_SITO_FP_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.sitofp";

/// Operation trait for `llvm.intr.experimental.constrained.sitofp`.
pub trait ConstrainedSitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_SITO_FP_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedSitoFp);

/// Constructs a new detached `llvm.intr.experimental.constrained.sitofp` operation.
pub fn intr_experimental_constrained_sitofp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedSitoFpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_SITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_sitofp`")
}

/// Canonical MLIR operation name for [`ConstrainedUitoFpOperation`].
pub const CONSTRAINED_UITO_FP_OPERATION_NAME: &str = "llvm.intr.experimental.constrained.uitofp";

/// Operation trait for `llvm.intr.experimental.constrained.uitofp`.
pub trait ConstrainedUitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONSTRAINED_UITO_FP_OPERATION_NAME
    }

    /// Returns the `argument_0` operand.
    fn argument_0(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `roundingmode` attribute.
    fn roundingmode(&self) -> AttributeRef<'c, 't> {
        self.attribute("roundingmode").unwrap()
    }

    /// Returns the `fpExceptionBehavior` attribute.
    fn fp_exception_behavior(&self) -> AttributeRef<'c, 't> {
        self.attribute("fpExceptionBehavior").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ConstrainedUitoFp);

/// Constructs a new detached `llvm.intr.experimental.constrained.uitofp` operation.
pub fn intr_experimental_constrained_uitofp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    argument_0: V0,
    result_type: T0,
    roundingmode: AttributeRef<'c, 't>,
    fp_exception_behavior: AttributeRef<'c, 't>,
    location: L,
) -> DetachedConstrainedUitoFpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(CONSTRAINED_UITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(argument_0);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("roundingmode", roundingmode);
    builder = builder.add_attribute("fpExceptionBehavior", fp_exception_behavior);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_constrained_uitofp`")
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
pub fn intr_copysign<
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
        .expect("invalid arguments to `llvm::intr_copysign`")
}

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
pub fn intr_cttz<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::intr_cttz`")
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
pub fn intr_ctpop<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::intr_ctpop`")
}

/// Canonical MLIR operation name for [`DbgDeclareOperation`].
pub const DBG_DECLARE_OPERATION_NAME: &str = "llvm.intr.dbg.declare";

/// Operation trait for `llvm.intr.dbg.declare`.
pub trait DbgDeclareOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_DECLARE_OPERATION_NAME
    }

    /// Returns the `address` operand.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> AttributeRef<'c, 't> {
        self.attribute("varInfo").unwrap()
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> AttributeRef<'c, 't> {
        self.attribute("locationExpr").unwrap()
    }
}

mlir_op!(DbgDeclare);

/// Constructs a new detached `llvm.intr.dbg.declare` operation.
pub fn intr_dbg_declare<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    address: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgDeclareOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_DECLARE_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_declare`")
}

/// Canonical MLIR operation name for [`DbgLabelOperation`].
pub const DBG_LABEL_OPERATION_NAME: &str = "llvm.intr.dbg.label";

/// Operation trait for `llvm.intr.dbg.label`.
pub trait DbgLabelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_LABEL_OPERATION_NAME
    }

    /// Returns the `label` attribute.
    fn label(&self) -> AttributeRef<'c, 't> {
        self.attribute("label").unwrap()
    }
}

mlir_op!(DbgLabel);

/// Constructs a new detached `llvm.intr.dbg.label` operation.
pub fn intr_dbg_label<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    label: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgLabelOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_LABEL_OPERATION_NAME, location);
    builder = builder.add_attribute("label", label);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_label`")
}

/// Canonical MLIR operation name for [`DbgValueOperation`].
pub const DBG_VALUE_OPERATION_NAME: &str = "llvm.intr.dbg.value";

/// Operation trait for `llvm.intr.dbg.value`.
pub trait DbgValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DBG_VALUE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `varInfo` attribute.
    fn var_info(&self) -> AttributeRef<'c, 't> {
        self.attribute("varInfo").unwrap()
    }

    /// Returns the `locationExpr` attribute.
    fn location_expr(&self) -> AttributeRef<'c, 't> {
        self.attribute("locationExpr").unwrap()
    }
}

mlir_op!(DbgValue);

/// Constructs a new detached `llvm.intr.dbg.value` operation.
pub fn intr_dbg_value<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V0,
    var_info: AttributeRef<'c, 't>,
    location_expr: AttributeRef<'c, 't>,
    location: L,
) -> DetachedDbgValueOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(DBG_VALUE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_attribute("varInfo", var_info);
    builder = builder.add_attribute("locationExpr", location_expr);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_dbg_value`")
}

/// Canonical MLIR operation name for [`DebugTrapOperation`].
pub const DEBUG_TRAP_OPERATION_NAME: &str = "llvm.intr.debugtrap";

/// Operation trait for `llvm.intr.debugtrap`.
pub trait DebugTrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DEBUG_TRAP_OPERATION_NAME
    }
}

mlir_op!(DebugTrap);

/// Constructs a new detached `llvm.intr.debugtrap` operation.
pub fn intr_debugtrap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedDebugTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let builder = OperationBuilder::new(DEBUG_TRAP_OPERATION_NAME, location);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_debugtrap`")
}

/// Canonical MLIR operation name for [`EhTypeidForOperation`].
pub const EH_TYPEID_FOR_OPERATION_NAME: &str = "llvm.intr.eh.typeid.for";

/// Operation trait for `llvm.intr.eh.typeid.for`.
pub trait EhTypeidForOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EH_TYPEID_FOR_OPERATION_NAME
    }

    /// Returns the `type_info` operand.
    fn type_info(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(EhTypeidFor);

/// Constructs a new detached `llvm.intr.eh.typeid.for` operation.
pub fn intr_eh_typeid_for<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    type_info: V0,
    result_type: T0,
    location: L,
) -> DetachedEhTypeidForOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(EH_TYPEID_FOR_OPERATION_NAME, location);
    builder = builder.add_operand(type_info);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_eh_typeid_for`")
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

/// Canonical MLIR operation name for [`FakeUseOperation`].
pub const FAKE_USE_OPERATION_NAME: &str = "llvm.intr.fake.use";

/// Operation trait for `llvm.intr.fake.use`.
pub trait FakeUseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FAKE_USE_OPERATION_NAME
    }

    /// Returns the variadic arguments.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(FakeUse);

/// Constructs a new detached `llvm.intr.fake.use` operation.
pub fn intr_fake_use<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    arguments: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedFakeUseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FAKE_USE_OPERATION_NAME, location);
    builder = builder.add_operands(arguments);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_fake_use`")
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
    }
}

mlir_op!(FractionExp);

/// Constructs a new detached `llvm.intr.frexp` operation.
pub fn intr_frexp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    first_result_type: T0,
    second_result_type: T1,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedFractionExpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(FRACTION_EXP_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

/// Canonical MLIR operation name for [`GetActiveLaneMaskOperation`].
pub const GET_ACTIVE_LANE_MASK_OPERATION_NAME: &str = "llvm.intr.get.active.lane.mask";

/// Operation trait for `llvm.intr.get.active.lane.mask`.
pub trait GetActiveLaneMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GET_ACTIVE_LANE_MASK_OPERATION_NAME
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `bound` operand.
    fn bound(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(GetActiveLaneMask);

/// Constructs a new detached `llvm.intr.get.active.lane.mask` operation.
pub fn intr_get_active_lane_mask<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    base: V0,
    bound: V1,
    result_type: T0,
    location: L,
) -> DetachedGetActiveLaneMaskOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(GET_ACTIVE_LANE_MASK_OPERATION_NAME, location);
    builder = builder.add_operand(base);
    builder = builder.add_operand(bound);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_get_active_lane_mask`")
}

/// Canonical MLIR operation name for [`InvariantEndOperation`].
pub const INVARIANT_END_OPERATION_NAME: &str = "llvm.intr.invariant.end";

/// Operation trait for `llvm.intr.invariant.end`.
pub trait InvariantEndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INVARIANT_END_OPERATION_NAME
    }

    /// Returns the `start` operand.
    fn start(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `size` attribute.
    fn size(&self) -> AttributeRef<'c, 't> {
        self.attribute("size").unwrap()
    }
}

mlir_op!(InvariantEnd);

/// Constructs a new detached `llvm.intr.invariant.end` operation.
pub fn intr_invariant_end<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    start: V0,
    pointer: V1,
    size: AttributeRef<'c, 't>,
    location: L,
) -> DetachedInvariantEndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INVARIANT_END_OPERATION_NAME, location);
    builder = builder.add_operand(start);
    builder = builder.add_operand(pointer);
    builder = builder.add_attribute("size", size);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_invariant_end`")
}

/// Canonical MLIR operation name for [`InvariantStartOperation`].
pub const INVARIANT_START_OPERATION_NAME: &str = "llvm.intr.invariant.start";

/// Operation trait for `llvm.intr.invariant.start`.
pub trait InvariantStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INVARIANT_START_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `size` attribute.
    fn size(&self) -> AttributeRef<'c, 't> {
        self.attribute("size").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(InvariantStart);

/// Constructs a new detached `llvm.intr.invariant.start` operation.
pub fn intr_invariant_start<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    result_type: T0,
    size: AttributeRef<'c, 't>,
    location: L,
) -> DetachedInvariantStartOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(INVARIANT_START_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("size", size);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_invariant_start`")
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

/// Canonical MLIR operation name for [`LaunderInvariantGroupOperation`].
pub const LAUNDER_INVARIANT_GROUP_OPERATION_NAME: &str = "llvm.intr.launder.invariant.group";

/// Operation trait for `llvm.intr.launder.invariant.group`.
pub trait LaunderInvariantGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LAUNDER_INVARIANT_GROUP_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(LaunderInvariantGroup);

/// Constructs a new detached `llvm.intr.launder.invariant.group` operation.
pub fn intr_launder_invariant_group<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V0,
    result_type: T0,
    location: L,
) -> DetachedLaunderInvariantGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LAUNDER_INVARIANT_GROUP_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_launder_invariant_group`")
}

/// Canonical MLIR operation name for [`LifetimeEndOperation`].
pub const LIFETIME_END_OPERATION_NAME: &str = "llvm.intr.lifetime.end";

/// Operation trait for `llvm.intr.lifetime.end`.
pub trait LifetimeEndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LIFETIME_END_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(LifetimeEnd);

/// Constructs a new detached `llvm.intr.lifetime.end` operation.
pub fn intr_lifetime_end<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> DetachedLifetimeEndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LIFETIME_END_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_lifetime_end`")
}

/// Canonical MLIR operation name for [`LifetimeStartOperation`].
pub const LIFETIME_START_OPERATION_NAME: &str = "llvm.intr.lifetime.start";

/// Operation trait for `llvm.intr.lifetime.start`.
pub trait LifetimeStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LIFETIME_START_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(LifetimeStart);

/// Constructs a new detached `llvm.intr.lifetime.start` operation.
pub fn intr_lifetime_start<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> DetachedLifetimeStartOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(LIFETIME_START_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_lifetime_start`")
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

/// Canonical MLIR operation name for [`MaskedLoadOperation`].
pub const MASKED_LOAD_OPERATION_NAME: &str = "llvm.intr.masked.load";

/// Operation trait for `llvm.intr.masked.load`.
pub trait MaskedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_LOAD_OPERATION_NAME
    }

    /// Returns the `data` operand.
    fn data(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> AttributeRef<'c, 't> {
        self.attribute("alignment").unwrap()
    }

    /// Returns the `nontemporal` attribute.
    fn nontemporal(&self) -> AttributeRef<'c, 't> {
        self.attribute("nontemporal").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MaskedLoad);

/// Constructs a new detached `llvm.intr.masked.load` operation.
pub fn intr_masked_load<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    data: V0,
    mask: V1,
    result_type: T0,
    alignment: AttributeRef<'c, 't>,
    nontemporal: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMaskedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(data);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("alignment", alignment);
    builder = builder.add_attribute("nontemporal", nontemporal);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_load`")
}

/// Canonical MLIR operation name for [`MaskedStoreOperation`].
pub const MASKED_STORE_OPERATION_NAME: &str = "llvm.intr.masked.store";

/// Operation trait for `llvm.intr.masked.store`.
pub trait MaskedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_STORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `data` operand.
    fn data(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> AttributeRef<'c, 't> {
        self.attribute("alignment").unwrap()
    }
}

mlir_op!(MaskedStore);

/// Constructs a new detached `llvm.intr.masked.store` operation.
pub fn intr_masked_store<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    data: V1,
    mask: V2,
    alignment: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMaskedStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(data);
    builder = builder.add_operand(mask);
    builder = builder.add_attribute("alignment", alignment);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_store`")
}

/// Canonical MLIR operation name for [`MatrixColumnMajorLoadOperation`].
pub const MATRIX_COLUMN_MAJOR_LOAD_OPERATION_NAME: &str = "llvm.intr.matrix.column.major.load";

/// Operation trait for `llvm.intr.matrix.column.major.load`.
pub trait MatrixColumnMajorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MATRIX_COLUMN_MAJOR_LOAD_OPERATION_NAME
    }

    /// Returns the `data` operand.
    fn data(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> AttributeRef<'c, 't> {
        self.attribute("rows").unwrap()
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> AttributeRef<'c, 't> {
        self.attribute("columns").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MatrixColumnMajorLoad);

/// Constructs a new detached `llvm.intr.matrix.column.major.load` operation.
pub fn intr_matrix_column_major_load<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    data: V0,
    stride: V1,
    result_type: T0,
    is_volatile: AttributeRef<'c, 't>,
    rows: AttributeRef<'c, 't>,
    columns: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMatrixColumnMajorLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MATRIX_COLUMN_MAJOR_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(data);
    builder = builder.add_operand(stride);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_matrix_column_major_load`")
}

/// Canonical MLIR operation name for [`MatrixColumnMajorStoreOperation`].
pub const MATRIX_COLUMN_MAJOR_STORE_OPERATION_NAME: &str = "llvm.intr.matrix.column.major.store";

/// Operation trait for `llvm.intr.matrix.column.major.store`.
pub trait MatrixColumnMajorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MATRIX_COLUMN_MAJOR_STORE_OPERATION_NAME
    }

    /// Returns the `matrix` operand.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `data` operand.
    fn data(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> AttributeRef<'c, 't> {
        self.attribute("rows").unwrap()
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> AttributeRef<'c, 't> {
        self.attribute("columns").unwrap()
    }
}

mlir_op!(MatrixColumnMajorStore);

/// Constructs a new detached `llvm.intr.matrix.column.major.store` operation.
pub fn intr_matrix_column_major_store<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    matrix: V0,
    data: V1,
    stride: V2,
    is_volatile: AttributeRef<'c, 't>,
    rows: AttributeRef<'c, 't>,
    columns: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMatrixColumnMajorStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MATRIX_COLUMN_MAJOR_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(matrix);
    builder = builder.add_operand(data);
    builder = builder.add_operand(stride);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_matrix_column_major_store`")
}

/// Canonical MLIR operation name for [`MatrixMultiplyOperation`].
pub const MATRIX_MULTIPLY_OPERATION_NAME: &str = "llvm.intr.matrix.multiply";

/// Operation trait for `llvm.intr.matrix.multiply`.
pub trait MatrixMultiplyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MATRIX_MULTIPLY_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `lhs_rows` attribute.
    fn lhs_rows(&self) -> AttributeRef<'c, 't> {
        self.attribute("lhs_rows").unwrap()
    }

    /// Returns the `lhs_columns` attribute.
    fn lhs_columns(&self) -> AttributeRef<'c, 't> {
        self.attribute("lhs_columns").unwrap()
    }

    /// Returns the `rhs_columns` attribute.
    fn rhs_columns(&self) -> AttributeRef<'c, 't> {
        self.attribute("rhs_columns").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MatrixMultiply);

/// Constructs a new detached `llvm.intr.matrix.multiply` operation.
pub fn intr_matrix_multiply<
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
    lhs_rows: AttributeRef<'c, 't>,
    lhs_columns: AttributeRef<'c, 't>,
    rhs_columns: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMatrixMultiplyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MATRIX_MULTIPLY_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("lhs_rows", lhs_rows);
    builder = builder.add_attribute("lhs_columns", lhs_columns);
    builder = builder.add_attribute("rhs_columns", rhs_columns);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_matrix_multiply`")
}

/// Canonical MLIR operation name for [`MatrixTransposeOperation`].
pub const MATRIX_TRANSPOSE_OPERATION_NAME: &str = "llvm.intr.matrix.transpose";

/// Operation trait for `llvm.intr.matrix.transpose`.
pub trait MatrixTransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MATRIX_TRANSPOSE_OPERATION_NAME
    }

    /// Returns the `matrix` operand.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> AttributeRef<'c, 't> {
        self.attribute("rows").unwrap()
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> AttributeRef<'c, 't> {
        self.attribute("columns").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MatrixTranspose);

/// Constructs a new detached `llvm.intr.matrix.transpose` operation.
pub fn intr_matrix_transpose<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    matrix: V0,
    result_type: T0,
    rows: AttributeRef<'c, 't>,
    columns: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMatrixTransposeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MATRIX_TRANSPOSE_OPERATION_NAME, location);
    builder = builder.add_operand(matrix);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_matrix_transpose`")
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

/// Canonical MLIR operation name for [`MemcpyInlineOperation`].
pub const MEMCPY_INLINE_OPERATION_NAME: &str = "llvm.intr.memcpy.inline";

/// Operation trait for `llvm.intr.memcpy.inline`.
pub trait MemcpyInlineOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMCPY_INLINE_OPERATION_NAME
    }

    /// Returns the `destination` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `len` attribute.
    fn len(&self) -> AttributeRef<'c, 't> {
        self.attribute("len").unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }
}

mlir_op!(MemcpyInline);

/// Constructs a new detached `llvm.intr.memcpy.inline` operation.
pub fn intr_memcpy_inline<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    destination: V0,
    source: V1,
    len: AttributeRef<'c, 't>,
    is_volatile: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMemcpyInlineOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MEMCPY_INLINE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_attribute("len", len);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_memcpy_inline`")
}

/// Canonical MLIR operation name for [`MemcpyOperation`].
pub const MEMCPY_OPERATION_NAME: &str = "llvm.intr.memcpy";

/// Operation trait for `llvm.intr.memcpy`.
pub trait MemcpyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMCPY_OPERATION_NAME
    }

    /// Returns the `destination` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `length` operand.
    fn length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }
}

mlir_op!(Memcpy);

/// Constructs a new detached `llvm.intr.memcpy` operation.
pub fn intr_memcpy<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    destination: V0,
    source: V1,
    length: V2,
    is_volatile: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMemcpyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MEMCPY_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_memcpy`")
}

/// Canonical MLIR operation name for [`MemmoveOperation`].
pub const MEMMOVE_OPERATION_NAME: &str = "llvm.intr.memmove";

/// Operation trait for `llvm.intr.memmove`.
pub trait MemmoveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMMOVE_OPERATION_NAME
    }

    /// Returns the `destination` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `length` operand.
    fn length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }
}

mlir_op!(Memmove);

/// Constructs a new detached `llvm.intr.memmove` operation.
pub fn intr_memmove<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    destination: V0,
    source: V1,
    length: V2,
    is_volatile: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMemmoveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MEMMOVE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_memmove`")
}

/// Canonical MLIR operation name for [`MemsetInlineOperation`].
pub const MEMSET_INLINE_OPERATION_NAME: &str = "llvm.intr.memset.inline";

/// Operation trait for `llvm.intr.memset.inline`.
pub trait MemsetInlineOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMSET_INLINE_OPERATION_NAME
    }

    /// Returns the `destination` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `len` attribute.
    fn len(&self) -> AttributeRef<'c, 't> {
        self.attribute("len").unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }
}

mlir_op!(MemsetInline);

/// Constructs a new detached `llvm.intr.memset.inline` operation.
pub fn intr_memset_inline<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    destination: V0,
    value: V1,
    len: AttributeRef<'c, 't>,
    is_volatile: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMemsetInlineOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MEMSET_INLINE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(value);
    builder = builder.add_attribute("len", len);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_memset_inline`")
}

/// Canonical MLIR operation name for [`MemsetOperation`].
pub const MEMSET_OPERATION_NAME: &str = "llvm.intr.memset";

/// Operation trait for `llvm.intr.memset`.
pub trait MemsetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMSET_OPERATION_NAME
    }

    /// Returns the `destination` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `length` operand.
    fn length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> AttributeRef<'c, 't> {
        self.attribute("isVolatile").unwrap()
    }
}

mlir_op!(Memset);

/// Constructs a new detached `llvm.intr.memset` operation.
pub fn intr_memset<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    destination: V0,
    value: V1,
    length: V2,
    is_volatile: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMemsetOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MEMSET_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(value);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_memset`")
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
pub fn intr_minnum<
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
        .expect("invalid arguments to `llvm::intr_minnum`")
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
pub const NEARBYINT_OPERATION_NAME: &str = "llvm.intr.nearbyint";

/// Operation trait for `llvm.intr.nearbyint`.
pub trait NearbyintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NEARBYINT_OPERATION_NAME
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

mlir_op!(Nearbyint);

/// Constructs a new detached `llvm.intr.nearbyint` operation.
pub fn intr_nearbyint<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedNearbyintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(NEARBYINT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_nearbyint`")
}

/// Canonical MLIR operation name for [`NoAliasScopeDeclOperation`].
pub const NO_ALIAS_SCOPE_DECL_OPERATION_NAME: &str = "llvm.intr.experimental.noalias.scope.decl";

/// Operation trait for `llvm.intr.experimental.noalias.scope.decl`.
pub trait NoAliasScopeDeclOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NO_ALIAS_SCOPE_DECL_OPERATION_NAME
    }

    /// Returns the `scope` attribute.
    fn scope(&self) -> AttributeRef<'c, 't> {
        self.attribute("scope").unwrap()
    }
}

mlir_op!(NoAliasScopeDecl);

/// Constructs a new detached `llvm.intr.experimental.noalias.scope.decl` operation.
pub fn intr_experimental_noalias_scope_decl<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    scope: AttributeRef<'c, 't>,
    location: L,
) -> DetachedNoAliasScopeDeclOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(NO_ALIAS_SCOPE_DECL_OPERATION_NAME, location);
    builder = builder.add_attribute("scope", scope);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_noalias_scope_decl`")
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

/// Canonical MLIR operation name for [`PrefetchOperation`].
pub const PREFETCH_OPERATION_NAME: &str = "llvm.intr.prefetch";

/// Operation trait for `llvm.intr.prefetch`.
pub trait PrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PREFETCH_OPERATION_NAME
    }

    /// Returns the `address` operand.
    fn address(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rw` attribute.
    fn rw(&self) -> AttributeRef<'c, 't> {
        self.attribute("rw").unwrap()
    }

    /// Returns the `hint` attribute.
    fn hint(&self) -> AttributeRef<'c, 't> {
        self.attribute("hint").unwrap()
    }

    /// Returns the `cache` attribute.
    fn cache(&self) -> AttributeRef<'c, 't> {
        self.attribute("cache").unwrap()
    }
}

mlir_op!(Prefetch);

/// Constructs a new detached `llvm.intr.prefetch` operation.
pub fn intr_prefetch<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    address: V0,
    rw: AttributeRef<'c, 't>,
    hint: AttributeRef<'c, 't>,
    cache: AttributeRef<'c, 't>,
    location: L,
) -> DetachedPrefetchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(PREFETCH_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_attribute("rw", rw);
    builder = builder.add_attribute("hint", hint);
    builder = builder.add_attribute("cache", cache);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_prefetch`")
}

/// Canonical MLIR operation name for [`PtrAnnotationOperation`].
pub const PTR_ANNOTATION_OPERATION_NAME: &str = "llvm.intr.ptr.annotation";

/// Operation trait for `llvm.intr.ptr.annotation`.
pub trait PtrAnnotationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PTR_ANNOTATION_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `line` operand.
    fn line(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `attribute` operand.
    fn attribute(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(PtrAnnotation);

/// Constructs a new detached `llvm.intr.ptr.annotation` operation.
pub fn intr_ptr_annotation<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V0,
    annotation: V1,
    file_name: V2,
    line: V3,
    attribute: V4,
    result_type: T0,
    location: L,
) -> DetachedPtrAnnotationOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(PTR_ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_operand(attribute);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ptr_annotation`")
}

/// Canonical MLIR operation name for [`PtrMaskOperation`].
pub const PTR_MASK_OPERATION_NAME: &str = "llvm.intr.ptrmask";

/// Operation trait for `llvm.intr.ptrmask`.
pub trait PtrMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PTR_MASK_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(PtrMask);

/// Constructs a new detached `llvm.intr.ptrmask` operation.
pub fn intr_ptrmask<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V0,
    mask: V1,
    result_type: T0,
    location: L,
) -> DetachedPtrMaskOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(PTR_MASK_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ptrmask`")
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
pub fn intr_roundeven<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
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
        .expect("invalid arguments to `llvm::intr_roundeven`")
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedSaddWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedSmulWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedSsubWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SSUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
    }
}

mlir_op!(Sincos);

/// Constructs a new detached `llvm.intr.sincos` operation.
pub fn intr_sincos<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    first_result_type: T0,
    second_result_type: T1,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedSincosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(SINCOS_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

/// Canonical MLIR operation name for [`StackRestoreOperation`].
pub const STACK_RESTORE_OPERATION_NAME: &str = "llvm.intr.stackrestore";

/// Operation trait for `llvm.intr.stackrestore`.
pub trait StackRestoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        STACK_RESTORE_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(StackRestore);

/// Constructs a new detached `llvm.intr.stackrestore` operation.
pub fn intr_stackrestore<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> DetachedStackRestoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(STACK_RESTORE_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_stackrestore`")
}

/// Canonical MLIR operation name for [`StackSaveOperation`].
pub const STACK_SAVE_OPERATION_NAME: &str = "llvm.intr.stacksave";

/// Operation trait for `llvm.intr.stacksave`.
pub trait StackSaveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        STACK_SAVE_OPERATION_NAME
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(StackSave);

/// Constructs a new detached `llvm.intr.stacksave` operation.
pub fn intr_stacksave<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> DetachedStackSaveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(STACK_SAVE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_stacksave`")
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

/// Canonical MLIR operation name for [`StripInvariantGroupOperation`].
pub const STRIP_INVARIANT_GROUP_OPERATION_NAME: &str = "llvm.intr.strip.invariant.group";

/// Operation trait for `llvm.intr.strip.invariant.group`.
pub trait StripInvariantGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        STRIP_INVARIANT_GROUP_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(StripInvariantGroup);

/// Constructs a new detached `llvm.intr.strip.invariant.group` operation.
pub fn intr_strip_invariant_group<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    result_type: T0,
    location: L,
) -> DetachedStripInvariantGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(STRIP_INVARIANT_GROUP_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_strip_invariant_group`")
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

/// Canonical MLIR operation name for [`ThreadlocalAddressOperation`].
pub const THREADLOCAL_ADDRESS_OPERATION_NAME: &str = "llvm.intr.threadlocal.address";

/// Operation trait for `llvm.intr.threadlocal.address`.
pub trait ThreadlocalAddressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        THREADLOCAL_ADDRESS_OPERATION_NAME
    }

    /// Returns the `global` operand.
    fn global(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(ThreadlocalAddress);

/// Constructs a new detached `llvm.intr.threadlocal.address` operation.
pub fn intr_threadlocal_address<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    global: V0,
    result_type: T0,
    location: L,
) -> DetachedThreadlocalAddressOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(THREADLOCAL_ADDRESS_OPERATION_NAME, location);
    builder = builder.add_operand(global);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_threadlocal_address`")
}

/// Canonical MLIR operation name for [`TrapOperation`].
pub const TRAP_OPERATION_NAME: &str = "llvm.intr.trap";

/// Operation trait for `llvm.intr.trap`.
pub trait TrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TRAP_OPERATION_NAME
    }
}

mlir_op!(Trap);

/// Constructs a new detached `llvm.intr.trap` operation.
pub fn intr_trap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let builder = OperationBuilder::new(TRAP_OPERATION_NAME, location);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_trap`")
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedUaddWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UADD_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_uadd_with_overflow`")
}

/// Canonical MLIR operation name for [`UbsanTrapOperation`].
pub const UBSAN_TRAP_OPERATION_NAME: &str = "llvm.intr.ubsantrap";

/// Operation trait for `llvm.intr.ubsantrap`.
pub trait UbsanTrapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        UBSAN_TRAP_OPERATION_NAME
    }

    /// Returns the `failureKind` attribute.
    fn failure_kind(&self) -> AttributeRef<'c, 't> {
        self.attribute("failureKind").unwrap()
    }
}

mlir_op!(UbsanTrap);

/// Constructs a new detached `llvm.intr.ubsantrap` operation.
pub fn intr_ubsantrap<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    failure_kind: AttributeRef<'c, 't>,
    location: L,
) -> DetachedUbsanTrapOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UBSAN_TRAP_OPERATION_NAME, location);
    builder = builder.add_attribute("failureKind", failure_kind);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_ubsantrap`")
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedUmulWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(UMUL_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
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

    /// Returns this operation's result type at index 0.
    fn first_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }

    /// Returns this operation's result type at index 1.
    fn second_output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(1).unwrap()
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
    T1: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    lhs: V0,
    rhs: V1,
    first_result_type: T0,
    second_result_type: T1,
    location: L,
) -> DetachedUsubWithOverflowOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(USUB_WITH_OVERFLOW_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(first_result_type);
    builder = builder.add_result(second_result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_usub_with_overflow`")
}

/// Canonical MLIR operation name for [`VpashrOperation`].
pub const VPASHR_OPERATION_NAME: &str = "llvm.intr.vp.ashr";

/// Operation trait for `llvm.intr.vp.ashr`.
pub trait VpashrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPASHR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpashr);

/// Constructs a new detached `llvm.intr.vp.ashr` operation.
pub fn intr_vp_ashr<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpashrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPASHR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_ashr`")
}

/// Canonical MLIR operation name for [`VpaddOperation`].
pub const VPADD_OPERATION_NAME: &str = "llvm.intr.vp.add";

/// Operation trait for `llvm.intr.vp.add`.
pub trait VpaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPADD_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpadd);

/// Constructs a new detached `llvm.intr.vp.add` operation.
pub fn intr_vp_add<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpaddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPADD_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_add`")
}

/// Canonical MLIR operation name for [`VpandOperation`].
pub const VPAND_OPERATION_NAME: &str = "llvm.intr.vp.and";

/// Operation trait for `llvm.intr.vp.and`.
pub trait VpandOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPAND_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpand);

/// Constructs a new detached `llvm.intr.vp.and` operation.
pub fn intr_vp_and<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpandOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPAND_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_and`")
}

/// Canonical MLIR operation name for [`VpfaddOperation`].
pub const VPFADD_OPERATION_NAME: &str = "llvm.intr.vp.fadd";

/// Operation trait for `llvm.intr.vp.fadd`.
pub trait VpfaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFADD_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfadd);

/// Constructs a new detached `llvm.intr.vp.fadd` operation.
pub fn intr_vp_fadd<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpfaddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFADD_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fadd`")
}

/// Canonical MLIR operation name for [`VpfdivOperation`].
pub const VPFDIV_OPERATION_NAME: &str = "llvm.intr.vp.fdiv";

/// Operation trait for `llvm.intr.vp.fdiv`.
pub trait VpfdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfdiv);

/// Constructs a new detached `llvm.intr.vp.fdiv` operation.
pub fn intr_vp_fdiv<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpfdivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fdiv`")
}

/// Canonical MLIR operation name for [`VpfmulAddOperation`].
pub const VPFMUL_ADD_OPERATION_NAME: &str = "llvm.intr.vp.fmuladd";

/// Operation trait for `llvm.intr.vp.fmuladd`.
pub trait VpfmulAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFMUL_ADD_OPERATION_NAME
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

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpfmulAdd);

/// Constructs a new detached `llvm.intr.vp.fmuladd` operation.
pub fn intr_vp_fmuladd<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    mask: V3,
    explicit_vector_length: V4,
    result_type: T0,
    location: L,
) -> DetachedVpfmulAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFMUL_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fmuladd`")
}

/// Canonical MLIR operation name for [`VpfmulOperation`].
pub const VPFMUL_OPERATION_NAME: &str = "llvm.intr.vp.fmul";

/// Operation trait for `llvm.intr.vp.fmul`.
pub trait VpfmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFMUL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfmul);

/// Constructs a new detached `llvm.intr.vp.fmul` operation.
pub fn intr_vp_fmul<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpfmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFMUL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fmul`")
}

/// Canonical MLIR operation name for [`VpfnegOperation`].
pub const VPFNEG_OPERATION_NAME: &str = "llvm.intr.vp.fneg";

/// Operation trait for `llvm.intr.vp.fneg`.
pub trait VpfnegOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFNEG_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfneg);

/// Constructs a new detached `llvm.intr.vp.fneg` operation.
pub fn intr_vp_fneg<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpfnegOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFNEG_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fneg`")
}

/// Canonical MLIR operation name for [`VpfpextOperation`].
pub const VPFPEXT_OPERATION_NAME: &str = "llvm.intr.vp.fpext";

/// Operation trait for `llvm.intr.vp.fpext`.
pub trait VpfpextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFPEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfpext);

/// Constructs a new detached `llvm.intr.vp.fpext` operation.
pub fn intr_vp_fpext<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpfpextOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFPEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fpext`")
}

/// Canonical MLIR operation name for [`VpfptoSiOperation`].
pub const VPFPTO_SI_OPERATION_NAME: &str = "llvm.intr.vp.fptosi";

/// Operation trait for `llvm.intr.vp.fptosi`.
pub trait VpfptoSiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFPTO_SI_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpfptoSi);

/// Constructs a new detached `llvm.intr.vp.fptosi` operation.
pub fn intr_vp_fptosi<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpfptoSiOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFPTO_SI_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fptosi`")
}

/// Canonical MLIR operation name for [`VpfptoUiOperation`].
pub const VPFPTO_UI_OPERATION_NAME: &str = "llvm.intr.vp.fptoui";

/// Operation trait for `llvm.intr.vp.fptoui`.
pub trait VpfptoUiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFPTO_UI_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpfptoUi);

/// Constructs a new detached `llvm.intr.vp.fptoui` operation.
pub fn intr_vp_fptoui<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpfptoUiOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFPTO_UI_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fptoui`")
}

/// Canonical MLIR operation name for [`VpfptruncOperation`].
pub const VPFPTRUNC_OPERATION_NAME: &str = "llvm.intr.vp.fptrunc";

/// Operation trait for `llvm.intr.vp.fptrunc`.
pub trait VpfptruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFPTRUNC_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfptrunc);

/// Constructs a new detached `llvm.intr.vp.fptrunc` operation.
pub fn intr_vp_fptrunc<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpfptruncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFPTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fptrunc`")
}

/// Canonical MLIR operation name for [`VpfremOperation`].
pub const VPFREM_OPERATION_NAME: &str = "llvm.intr.vp.frem";

/// Operation trait for `llvm.intr.vp.frem`.
pub trait VpfremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfrem);

/// Constructs a new detached `llvm.intr.vp.frem` operation.
pub fn intr_vp_frem<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpfremOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_frem`")
}

/// Canonical MLIR operation name for [`VpfsubOperation`].
pub const VPFSUB_OPERATION_NAME: &str = "llvm.intr.vp.fsub";

/// Operation trait for `llvm.intr.vp.fsub`.
pub trait VpfsubOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFSUB_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfsub);

/// Constructs a new detached `llvm.intr.vp.fsub` operation.
pub fn intr_vp_fsub<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpfsubOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFSUB_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fsub`")
}

/// Canonical MLIR operation name for [`VpfmaOperation`].
pub const VPFMA_OPERATION_NAME: &str = "llvm.intr.vp.fma";

/// Operation trait for `llvm.intr.vp.fma`.
pub trait VpfmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPFMA_OPERATION_NAME
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

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpfma);

/// Constructs a new detached `llvm.intr.vp.fma` operation.
pub fn intr_vp_fma<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first: V0,
    second: V1,
    third: V2,
    mask: V3,
    explicit_vector_length: V4,
    result_type: T0,
    location: L,
) -> DetachedVpfmaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPFMA_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_fma`")
}

/// Canonical MLIR operation name for [`VpintToPtrOperation`].
pub const VPINT_TO_PTR_OPERATION_NAME: &str = "llvm.intr.vp.inttoptr";

/// Operation trait for `llvm.intr.vp.inttoptr`.
pub trait VpintToPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPINT_TO_PTR_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpintToPtr);

/// Constructs a new detached `llvm.intr.vp.inttoptr` operation.
pub fn intr_vp_inttoptr<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpintToPtrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPINT_TO_PTR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_inttoptr`")
}

/// Canonical MLIR operation name for [`VplshrOperation`].
pub const VPLSHR_OPERATION_NAME: &str = "llvm.intr.vp.lshr";

/// Operation trait for `llvm.intr.vp.lshr`.
pub trait VplshrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPLSHR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vplshr);

/// Constructs a new detached `llvm.intr.vp.lshr` operation.
pub fn intr_vp_lshr<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVplshrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPLSHR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_lshr`")
}

/// Canonical MLIR operation name for [`VploadOperation`].
pub const VPLOAD_OPERATION_NAME: &str = "llvm.intr.vp.load";

/// Operation trait for `llvm.intr.vp.load`.
pub trait VploadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPLOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpload);

/// Constructs a new detached `llvm.intr.vp.load` operation.
pub fn intr_vp_load<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVploadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPLOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_load`")
}

/// Canonical MLIR operation name for [`VpmergeMinOperation`].
pub const VPMERGE_MIN_OPERATION_NAME: &str = "llvm.intr.vp.merge";

/// Operation trait for `llvm.intr.vp.merge`.
pub trait VpmergeMinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPMERGE_MIN_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `true_value` operand.
    fn true_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `false_value` operand.
    fn false_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpmergeMin);

/// Constructs a new detached `llvm.intr.vp.merge` operation.
pub fn intr_vp_merge<
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
    condition: V0,
    true_value: V1,
    false_value: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpmergeMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPMERGE_MIN_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operand(true_value);
    builder = builder.add_operand(false_value);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_merge`")
}

/// Canonical MLIR operation name for [`VpmulOperation`].
pub const VPMUL_OPERATION_NAME: &str = "llvm.intr.vp.mul";

/// Operation trait for `llvm.intr.vp.mul`.
pub trait VpmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPMUL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpmul);

/// Constructs a new detached `llvm.intr.vp.mul` operation.
pub fn intr_vp_mul<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPMUL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_mul`")
}

/// Canonical MLIR operation name for [`VporOperation`].
pub const VPOR_OPERATION_NAME: &str = "llvm.intr.vp.or";

/// Operation trait for `llvm.intr.vp.or`.
pub trait VporOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPOR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpor);

/// Constructs a new detached `llvm.intr.vp.or` operation.
pub fn intr_vp_or<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVporOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPOR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_or`")
}

/// Canonical MLIR operation name for [`VpptrToIntOperation`].
pub const VPPTR_TO_INT_OPERATION_NAME: &str = "llvm.intr.vp.ptrtoint";

/// Operation trait for `llvm.intr.vp.ptrtoint`.
pub trait VpptrToIntOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPPTR_TO_INT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpptrToInt);

/// Constructs a new detached `llvm.intr.vp.ptrtoint` operation.
pub fn intr_vp_ptrtoint<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpptrToIntOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPPTR_TO_INT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_ptrtoint`")
}

/// Canonical MLIR operation name for [`VpreduceAddOperation`].
pub const VPREDUCE_ADD_OPERATION_NAME: &str = "llvm.intr.vp.reduce.add";

/// Operation trait for `llvm.intr.vp.reduce.add`.
pub trait VpreduceAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_ADD_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceAdd);

/// Constructs a new detached `llvm.intr.vp.reduce.add` operation.
pub fn intr_vp_reduce_add<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_add`")
}

/// Canonical MLIR operation name for [`VpreduceAndOperation`].
pub const VPREDUCE_AND_OPERATION_NAME: &str = "llvm.intr.vp.reduce.and";

/// Operation trait for `llvm.intr.vp.reduce.and`.
pub trait VpreduceAndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_AND_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceAnd);

/// Constructs a new detached `llvm.intr.vp.reduce.and` operation.
pub fn intr_vp_reduce_and<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceAndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_AND_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_and`")
}

/// Canonical MLIR operation name for [`VpreduceFaddOperation`].
pub const VPREDUCE_FADD_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fadd";

/// Operation trait for `llvm.intr.vp.reduce.fadd`.
pub trait VpreduceFaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_FADD_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceFadd);

/// Constructs a new detached `llvm.intr.vp.reduce.fadd` operation.
pub fn intr_vp_reduce_fadd<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceFaddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_fadd`")
}

/// Canonical MLIR operation name for [`VpreduceFmaxOperation`].
pub const VPREDUCE_FMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmax";

/// Operation trait for `llvm.intr.vp.reduce.fmax`.
pub trait VpreduceFmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_FMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceFmax);

/// Constructs a new detached `llvm.intr.vp.reduce.fmax` operation.
pub fn intr_vp_reduce_fmax<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceFmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_FMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_fmax`")
}

/// Canonical MLIR operation name for [`VpreduceFminOperation`].
pub const VPREDUCE_FMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmin";

/// Operation trait for `llvm.intr.vp.reduce.fmin`.
pub trait VpreduceFminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_FMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceFmin);

/// Constructs a new detached `llvm.intr.vp.reduce.fmin` operation.
pub fn intr_vp_reduce_fmin<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceFminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_FMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_fmin`")
}

/// Canonical MLIR operation name for [`VpreduceFmulOperation`].
pub const VPREDUCE_FMUL_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmul";

/// Operation trait for `llvm.intr.vp.reduce.fmul`.
pub trait VpreduceFmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_FMUL_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceFmul);

/// Constructs a new detached `llvm.intr.vp.reduce.fmul` operation.
pub fn intr_vp_reduce_fmul<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceFmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_fmul`")
}

/// Canonical MLIR operation name for [`VpreduceMulOperation`].
pub const VPREDUCE_MUL_OPERATION_NAME: &str = "llvm.intr.vp.reduce.mul";

/// Operation trait for `llvm.intr.vp.reduce.mul`.
pub trait VpreduceMulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_MUL_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceMul);

/// Constructs a new detached `llvm.intr.vp.reduce.mul` operation.
pub fn intr_vp_reduce_mul<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceMulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_MUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_mul`")
}

/// Canonical MLIR operation name for [`VpreduceOrOperation`].
pub const VPREDUCE_OR_OPERATION_NAME: &str = "llvm.intr.vp.reduce.or";

/// Operation trait for `llvm.intr.vp.reduce.or`.
pub trait VpreduceOrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_OR_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceOr);

/// Constructs a new detached `llvm.intr.vp.reduce.or` operation.
pub fn intr_vp_reduce_or<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceOrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_OR_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_or`")
}

/// Canonical MLIR operation name for [`VpreduceSmaxOperation`].
pub const VPREDUCE_SMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.smax";

/// Operation trait for `llvm.intr.vp.reduce.smax`.
pub trait VpreduceSmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_SMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceSmax);

/// Constructs a new detached `llvm.intr.vp.reduce.smax` operation.
pub fn intr_vp_reduce_smax<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceSmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_smax`")
}

/// Canonical MLIR operation name for [`VpreduceSminOperation`].
pub const VPREDUCE_SMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.smin";

/// Operation trait for `llvm.intr.vp.reduce.smin`.
pub trait VpreduceSminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_SMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceSmin);

/// Constructs a new detached `llvm.intr.vp.reduce.smin` operation.
pub fn intr_vp_reduce_smin<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceSminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_smin`")
}

/// Canonical MLIR operation name for [`VpreduceUmaxOperation`].
pub const VPREDUCE_UMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.umax";

/// Operation trait for `llvm.intr.vp.reduce.umax`.
pub trait VpreduceUmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_UMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceUmax);

/// Constructs a new detached `llvm.intr.vp.reduce.umax` operation.
pub fn intr_vp_reduce_umax<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceUmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_umax`")
}

/// Canonical MLIR operation name for [`VpreduceUminOperation`].
pub const VPREDUCE_UMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.umin";

/// Operation trait for `llvm.intr.vp.reduce.umin`.
pub trait VpreduceUminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_UMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceUmin);

/// Constructs a new detached `llvm.intr.vp.reduce.umin` operation.
pub fn intr_vp_reduce_umin<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceUminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_umin`")
}

/// Canonical MLIR operation name for [`VpreduceXorOperation`].
pub const VPREDUCE_XOR_OPERATION_NAME: &str = "llvm.intr.vp.reduce.xor";

/// Operation trait for `llvm.intr.vp.reduce.xor`.
pub trait VpreduceXorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPREDUCE_XOR_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpreduceXor);

/// Constructs a new detached `llvm.intr.vp.reduce.xor` operation.
pub fn intr_vp_reduce_xor<
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
    start_value: V0,
    value: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpreduceXorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPREDUCE_XOR_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_reduce_xor`")
}

/// Canonical MLIR operation name for [`VpsdivOperation`].
pub const VPSDIV_OPERATION_NAME: &str = "llvm.intr.vp.sdiv";

/// Operation trait for `llvm.intr.vp.sdiv`.
pub trait VpsdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsdiv);

/// Constructs a new detached `llvm.intr.vp.sdiv` operation.
pub fn intr_vp_sdiv<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpsdivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_sdiv`")
}

/// Canonical MLIR operation name for [`VpsextOperation`].
pub const VPSEXT_OPERATION_NAME: &str = "llvm.intr.vp.sext";

/// Operation trait for `llvm.intr.vp.sext`.
pub trait VpsextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsext);

/// Constructs a new detached `llvm.intr.vp.sext` operation.
pub fn intr_vp_sext<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpsextOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_sext`")
}

/// Canonical MLIR operation name for [`VpsitoFpOperation`].
pub const VPSITO_FP_OPERATION_NAME: &str = "llvm.intr.vp.sitofp";

/// Operation trait for `llvm.intr.vp.sitofp`.
pub trait VpsitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSITO_FP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpsitoFp);

/// Constructs a new detached `llvm.intr.vp.sitofp` operation.
pub fn intr_vp_sitofp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpsitoFpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_sitofp`")
}

/// Canonical MLIR operation name for [`VpsmaxOperation`].
pub const VPSMAX_OPERATION_NAME: &str = "llvm.intr.vp.smax";

/// Operation trait for `llvm.intr.vp.smax`.
pub trait VpsmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsmax);

/// Constructs a new detached `llvm.intr.vp.smax` operation.
pub fn intr_vp_smax<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpsmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_smax`")
}

/// Canonical MLIR operation name for [`VpsminOperation`].
pub const VPSMIN_OPERATION_NAME: &str = "llvm.intr.vp.smin";

/// Operation trait for `llvm.intr.vp.smin`.
pub trait VpsminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsmin);

/// Constructs a new detached `llvm.intr.vp.smin` operation.
pub fn intr_vp_smin<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpsminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_smin`")
}

/// Canonical MLIR operation name for [`VpsremOperation`].
pub const VPSREM_OPERATION_NAME: &str = "llvm.intr.vp.srem";

/// Operation trait for `llvm.intr.vp.srem`.
pub trait VpsremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsrem);

/// Constructs a new detached `llvm.intr.vp.srem` operation.
pub fn intr_vp_srem<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpsremOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_srem`")
}

/// Canonical MLIR operation name for [`VpselectMinOperation`].
pub const VPSELECT_MIN_OPERATION_NAME: &str = "llvm.intr.vp.select";

/// Operation trait for `llvm.intr.vp.select`.
pub trait VpselectMinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSELECT_MIN_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `true_value` operand.
    fn true_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `false_value` operand.
    fn false_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpselectMin);

/// Constructs a new detached `llvm.intr.vp.select` operation.
pub fn intr_vp_select<
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
    condition: V0,
    true_value: V1,
    false_value: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpselectMinOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSELECT_MIN_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operand(true_value);
    builder = builder.add_operand(false_value);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_select`")
}

/// Canonical MLIR operation name for [`VpshlOperation`].
pub const VPSHL_OPERATION_NAME: &str = "llvm.intr.vp.shl";

/// Operation trait for `llvm.intr.vp.shl`.
pub trait VpshlOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSHL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpshl);

/// Constructs a new detached `llvm.intr.vp.shl` operation.
pub fn intr_vp_shl<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpshlOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSHL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_shl`")
}

/// Canonical MLIR operation name for [`VpstoreOperation`].
pub const VPSTORE_OPERATION_NAME: &str = "llvm.intr.vp.store";

/// Operation trait for `llvm.intr.vp.store`.
pub trait VpstoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSTORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }
}

mlir_op!(Vpstore);

/// Constructs a new detached `llvm.intr.vp.store` operation.
pub fn intr_vp_store<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    pointer: V1,
    mask: V2,
    explicit_vector_length: V3,
    location: L,
) -> DetachedVpstoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSTORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_store`")
}

/// Canonical MLIR operation name for [`VpstridedLoadOperation`].
pub const VPSTRIDED_LOAD_OPERATION_NAME: &str = "llvm.intr.experimental.vp.strided.load";

/// Operation trait for `llvm.intr.experimental.vp.strided.load`.
pub trait VpstridedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSTRIDED_LOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpstridedLoad);

/// Constructs a new detached `llvm.intr.experimental.vp.strided.load` operation.
pub fn intr_experimental_vp_strided_load<
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
    pointer: V0,
    stride: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpstridedLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSTRIDED_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(stride);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_vp_strided_load`")
}

/// Canonical MLIR operation name for [`VpstridedStoreOperation`].
pub const VPSTRIDED_STORE_OPERATION_NAME: &str = "llvm.intr.experimental.vp.strided.store";

/// Operation trait for `llvm.intr.experimental.vp.strided.store`.
pub trait VpstridedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSTRIDED_STORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }
}

mlir_op!(VpstridedStore);

/// Constructs a new detached `llvm.intr.experimental.vp.strided.store` operation.
pub fn intr_experimental_vp_strided_store<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    pointer: V1,
    stride: V2,
    mask: V3,
    explicit_vector_length: V4,
    location: L,
) -> DetachedVpstridedStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSTRIDED_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(stride);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_experimental_vp_strided_store`")
}

/// Canonical MLIR operation name for [`VpsubOperation`].
pub const VPSUB_OPERATION_NAME: &str = "llvm.intr.vp.sub";

/// Operation trait for `llvm.intr.vp.sub`.
pub trait VpsubOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPSUB_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpsub);

/// Constructs a new detached `llvm.intr.vp.sub` operation.
pub fn intr_vp_sub<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpsubOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPSUB_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_sub`")
}

/// Canonical MLIR operation name for [`VptruncOperation`].
pub const VPTRUNC_OPERATION_NAME: &str = "llvm.intr.vp.trunc";

/// Operation trait for `llvm.intr.vp.trunc`.
pub trait VptruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPTRUNC_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vptrunc);

/// Constructs a new detached `llvm.intr.vp.trunc` operation.
pub fn intr_vp_trunc<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVptruncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_trunc`")
}

/// Canonical MLIR operation name for [`VpudivOperation`].
pub const VPUDIV_OPERATION_NAME: &str = "llvm.intr.vp.udiv";

/// Operation trait for `llvm.intr.vp.udiv`.
pub trait VpudivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPUDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpudiv);

/// Constructs a new detached `llvm.intr.vp.udiv` operation.
pub fn intr_vp_udiv<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpudivOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPUDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_udiv`")
}

/// Canonical MLIR operation name for [`VpuitoFpOperation`].
pub const VPUITO_FP_OPERATION_NAME: &str = "llvm.intr.vp.uitofp";

/// Operation trait for `llvm.intr.vp.uitofp`.
pub trait VpuitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPUITO_FP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VpuitoFp);

/// Constructs a new detached `llvm.intr.vp.uitofp` operation.
pub fn intr_vp_uitofp<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpuitoFpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPUITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_uitofp`")
}

/// Canonical MLIR operation name for [`VpumaxOperation`].
pub const VPUMAX_OPERATION_NAME: &str = "llvm.intr.vp.umax";

/// Operation trait for `llvm.intr.vp.umax`.
pub trait VpumaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPUMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpumax);

/// Constructs a new detached `llvm.intr.vp.umax` operation.
pub fn intr_vp_umax<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpumaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPUMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_umax`")
}

/// Canonical MLIR operation name for [`VpuminOperation`].
pub const VPUMIN_OPERATION_NAME: &str = "llvm.intr.vp.umin";

/// Operation trait for `llvm.intr.vp.umin`.
pub trait VpuminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPUMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpumin);

/// Constructs a new detached `llvm.intr.vp.umin` operation.
pub fn intr_vp_umin<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpuminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPUMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_umin`")
}

/// Canonical MLIR operation name for [`VpuremOperation`].
pub const VPUREM_OPERATION_NAME: &str = "llvm.intr.vp.urem";

/// Operation trait for `llvm.intr.vp.urem`.
pub trait VpuremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPUREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpurem);

/// Constructs a new detached `llvm.intr.vp.urem` operation.
pub fn intr_vp_urem<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpuremOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPUREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_urem`")
}

/// Canonical MLIR operation name for [`VpxorOperation`].
pub const VPXOR_OPERATION_NAME: &str = "llvm.intr.vp.xor";

/// Operation trait for `llvm.intr.vp.xor`.
pub trait VpxorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPXOR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpxor);

/// Constructs a new detached `llvm.intr.vp.xor` operation.
pub fn intr_vp_xor<
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
    lhs: V0,
    rhs: V1,
    mask: V2,
    explicit_vector_length: V3,
    result_type: T0,
    location: L,
) -> DetachedVpxorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPXOR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_xor`")
}

/// Canonical MLIR operation name for [`VpzextOperation`].
pub const VPZEXT_OPERATION_NAME: &str = "llvm.intr.vp.zext";

/// Operation trait for `llvm.intr.vp.zext`.
pub trait VpzextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VPZEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vpzext);

/// Constructs a new detached `llvm.intr.vp.zext` operation.
pub fn intr_vp_zext<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V0,
    mask: V1,
    explicit_vector_length: V2,
    result_type: T0,
    location: L,
) -> DetachedVpzextOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VPZEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vp_zext`")
}

/// Canonical MLIR operation name for [`VaCopyOperation`].
pub const VA_COPY_OPERATION_NAME: &str = "llvm.intr.vacopy";

/// Operation trait for `llvm.intr.vacopy`.
pub trait VaCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VA_COPY_OPERATION_NAME
    }

    /// Returns the `destination_list` operand.
    fn destination_list(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source_list` operand.
    fn source_list(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(VaCopy);

/// Constructs a new detached `llvm.intr.vacopy` operation.
pub fn intr_vacopy<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    destination_list: V0,
    source_list: V1,
    location: L,
) -> DetachedVaCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VA_COPY_OPERATION_NAME, location);
    builder = builder.add_operand(destination_list);
    builder = builder.add_operand(source_list);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vacopy`")
}

/// Canonical MLIR operation name for [`VaEndOperation`].
pub const VA_END_OPERATION_NAME: &str = "llvm.intr.vaend";

/// Operation trait for `llvm.intr.vaend`.
pub trait VaEndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VA_END_OPERATION_NAME
    }

    /// Returns the `argument_list` operand.
    fn argument_list(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(VaEnd);

/// Constructs a new detached `llvm.intr.vaend` operation.
pub fn intr_vaend<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument_list: V0,
    location: L,
) -> DetachedVaEndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VA_END_OPERATION_NAME, location);
    builder = builder.add_operand(argument_list);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vaend`")
}

/// Canonical MLIR operation name for [`VaStartOperation`].
pub const VA_START_OPERATION_NAME: &str = "llvm.intr.vastart";

/// Operation trait for `llvm.intr.vastart`.
pub trait VaStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VA_START_OPERATION_NAME
    }

    /// Returns the `argument_list` operand.
    fn argument_list(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(VaStart);

/// Constructs a new detached `llvm.intr.vastart` operation.
pub fn intr_vastart<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument_list: V0,
    location: L,
) -> DetachedVaStartOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VA_START_OPERATION_NAME, location);
    builder = builder.add_operand(argument_list);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vastart`")
}

/// Canonical MLIR operation name for [`VarAnnotationOperation`].
pub const VAR_ANNOTATION_OPERATION_NAME: &str = "llvm.intr.var.annotation";

/// Operation trait for `llvm.intr.var.annotation`.
pub trait VarAnnotationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VAR_ANNOTATION_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `line` operand.
    fn line(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the `attribute` operand.
    fn attribute(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(4).unwrap()
    }
}

mlir_op!(VarAnnotation);

/// Constructs a new detached `llvm.intr.var.annotation` operation.
pub fn intr_var_annotation<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    V3: Value<'v, 'c, 't>,
    V4: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    annotation: V1,
    file_name: V2,
    line: V3,
    attribute: V4,
    location: L,
) -> DetachedVarAnnotationOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VAR_ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_operand(attribute);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_var_annotation`")
}

/// Canonical MLIR operation name for [`MaskedCompressstoreOperation`].
pub const MASKED_COMPRESSSTORE_OPERATION_NAME: &str = "llvm.intr.masked.compressstore";

/// Operation trait for `llvm.intr.masked.compressstore`.
pub trait MaskedCompressstoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_COMPRESSSTORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }
}

mlir_op!(MaskedCompressstore);

/// Constructs a new detached `llvm.intr.masked.compressstore` operation.
pub fn intr_masked_compressstore<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    pointer: V1,
    mask: V2,
    location: L,
) -> DetachedMaskedCompressstoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_COMPRESSSTORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_compressstore`")
}

/// Canonical MLIR operation name for [`MaskedExpandloadOperation`].
pub const MASKED_EXPANDLOAD_OPERATION_NAME: &str = "llvm.intr.masked.expandload";

/// Operation trait for `llvm.intr.masked.expandload`.
pub trait MaskedExpandloadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_EXPANDLOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `passthru` operand.
    fn passthru(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MaskedExpandload);

/// Constructs a new detached `llvm.intr.masked.expandload` operation.
pub fn intr_masked_expandload<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointer: V0,
    mask: V1,
    passthru: V2,
    result_type: T0,
    location: L,
) -> DetachedMaskedExpandloadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_EXPANDLOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(passthru);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_expandload`")
}

/// Canonical MLIR operation name for [`MaskedGatherOperation`].
pub const MASKED_GATHER_OPERATION_NAME: &str = "llvm.intr.masked.gather";

/// Operation trait for `llvm.intr.masked.gather`.
pub trait MaskedGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_GATHER_OPERATION_NAME
    }

    /// Returns the `pointers` operand.
    fn pointers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> AttributeRef<'c, 't> {
        self.attribute("alignment").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(MaskedGather);

/// Constructs a new detached `llvm.intr.masked.gather` operation.
pub fn intr_masked_gather<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    pointers: V0,
    mask: V1,
    result_type: T0,
    alignment: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMaskedGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_GATHER_OPERATION_NAME, location);
    builder = builder.add_operand(pointers);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("alignment", alignment);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_gather`")
}

/// Canonical MLIR operation name for [`MaskedScatterOperation`].
pub const MASKED_SCATTER_OPERATION_NAME: &str = "llvm.intr.masked.scatter";

/// Operation trait for `llvm.intr.masked.scatter`.
pub trait MaskedScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_SCATTER_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pointers` operand.
    fn pointers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> AttributeRef<'c, 't> {
        self.attribute("alignment").unwrap()
    }
}

mlir_op!(MaskedScatter);

/// Constructs a new detached `llvm.intr.masked.scatter` operation.
pub fn intr_masked_scatter<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    V2: Value<'v, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: V0,
    pointers: V1,
    mask: V2,
    alignment: AttributeRef<'c, 't>,
    location: L,
) -> DetachedMaskedScatterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(MASKED_SCATTER_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointers);
    builder = builder.add_operand(mask);
    builder = builder.add_attribute("alignment", alignment);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_masked_scatter`")
}

/// Canonical MLIR operation name for [`VectorDeinterleave2Operation`].
pub const VECTOR_DEINTERLEAVE2_OPERATION_NAME: &str = "llvm.intr.vector.deinterleave2";

/// Operation trait for `llvm.intr.vector.deinterleave2`.
pub trait VectorDeinterleave2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_DEINTERLEAVE2_OPERATION_NAME
    }

    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VectorDeinterleave2);

/// Constructs a new detached `llvm.intr.vector.deinterleave2` operation.
pub fn intr_vector_deinterleave2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    vector: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorDeinterleave2Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_DEINTERLEAVE2_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_deinterleave2`")
}

/// Canonical MLIR operation name for [`VectorExtractOperation`].
pub const VECTOR_EXTRACT_OPERATION_NAME: &str = "llvm.intr.vector.extract";

/// Operation trait for `llvm.intr.vector.extract`.
pub trait VectorExtractOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_EXTRACT_OPERATION_NAME
    }

    /// Returns the `source_vector` operand.
    fn source_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `pos` attribute.
    fn pos(&self) -> AttributeRef<'c, 't> {
        self.attribute("pos").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VectorExtract);

/// Constructs a new detached `llvm.intr.vector.extract` operation.
pub fn intr_vector_extract<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    source_vector: V0,
    result_type: T0,
    pos: AttributeRef<'c, 't>,
    location: L,
) -> DetachedVectorExtractOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_EXTRACT_OPERATION_NAME, location);
    builder = builder.add_operand(source_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("pos", pos);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_extract`")
}

/// Canonical MLIR operation name for [`VectorInsertOperation`].
pub const VECTOR_INSERT_OPERATION_NAME: &str = "llvm.intr.vector.insert";

/// Operation trait for `llvm.intr.vector.insert`.
pub trait VectorInsertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_INSERT_OPERATION_NAME
    }

    /// Returns the `destination_vector` operand.
    fn destination_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source_vector` operand.
    fn source_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `pos` attribute.
    fn pos(&self) -> AttributeRef<'c, 't> {
        self.attribute("pos").unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VectorInsert);

/// Constructs a new detached `llvm.intr.vector.insert` operation.
pub fn intr_vector_insert<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    destination_vector: V0,
    source_vector: V1,
    result_type: T0,
    pos: AttributeRef<'c, 't>,
    location: L,
) -> DetachedVectorInsertOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_INSERT_OPERATION_NAME, location);
    builder = builder.add_operand(destination_vector);
    builder = builder.add_operand(source_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("pos", pos);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_insert`")
}

/// Canonical MLIR operation name for [`VectorInterleave2Operation`].
pub const VECTOR_INTERLEAVE2_OPERATION_NAME: &str = "llvm.intr.vector.interleave2";

/// Operation trait for `llvm.intr.vector.interleave2`.
pub trait VectorInterleave2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_INTERLEAVE2_OPERATION_NAME
    }

    /// Returns the `first_vector` operand.
    fn first_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `second_vector` operand.
    fn second_vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(VectorInterleave2);

/// Constructs a new detached `llvm.intr.vector.interleave2` operation.
pub fn intr_vector_interleave2<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    first_vector: V0,
    second_vector: V1,
    result_type: T0,
    location: L,
) -> DetachedVectorInterleave2Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_INTERLEAVE2_OPERATION_NAME, location);
    builder = builder.add_operand(first_vector);
    builder = builder.add_operand(second_vector);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_interleave2`")
}

/// Canonical MLIR operation name for [`VectorReduceAddOperation`].
pub const VECTOR_REDUCE_ADD_OPERATION_NAME: &str = "llvm.intr.vector.reduce.add";

/// Operation trait for `llvm.intr.vector.reduce.add`.
pub trait VectorReduceAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_ADD_OPERATION_NAME
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

mlir_op!(VectorReduceAdd);

/// Constructs a new detached `llvm.intr.vector.reduce.add` operation.
pub fn intr_vector_reduce_add<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceAddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_add`")
}

/// Canonical MLIR operation name for [`VectorReduceAndOperation`].
pub const VECTOR_REDUCE_AND_OPERATION_NAME: &str = "llvm.intr.vector.reduce.and";

/// Operation trait for `llvm.intr.vector.reduce.and`.
pub trait VectorReduceAndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_AND_OPERATION_NAME
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

mlir_op!(VectorReduceAnd);

/// Constructs a new detached `llvm.intr.vector.reduce.and` operation.
pub fn intr_vector_reduce_and<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceAndOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_AND_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_and`")
}

/// Canonical MLIR operation name for [`VectorReduceFaddOperation`].
pub const VECTOR_REDUCE_FADD_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fadd";

/// Operation trait for `llvm.intr.vector.reduce.fadd`.
pub trait VectorReduceFaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FADD_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
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

mlir_op!(VectorReduceFadd);

/// Constructs a new detached `llvm.intr.vector.reduce.fadd` operation.
pub fn intr_vector_reduce_fadd<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    start_value: V0,
    input: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFaddOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fadd`")
}

/// Canonical MLIR operation name for [`VectorReduceFmaxOperation`].
pub const VECTOR_REDUCE_FMAX_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fmax";

/// Operation trait for `llvm.intr.vector.reduce.fmax`.
pub trait VectorReduceFmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FMAX_OPERATION_NAME
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

mlir_op!(VectorReduceFmax);

/// Constructs a new detached `llvm.intr.vector.reduce.fmax` operation.
pub fn intr_vector_reduce_fmax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fmax`")
}

/// Canonical MLIR operation name for [`VectorReduceFmaximumOperation`].
pub const VECTOR_REDUCE_FMAXIMUM_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fmaximum";

/// Operation trait for `llvm.intr.vector.reduce.fmaximum`.
pub trait VectorReduceFmaximumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FMAXIMUM_OPERATION_NAME
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

mlir_op!(VectorReduceFmaximum);

/// Constructs a new detached `llvm.intr.vector.reduce.fmaximum` operation.
pub fn intr_vector_reduce_fmaximum<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFmaximumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMAXIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fmaximum`")
}

/// Canonical MLIR operation name for [`VectorReduceFminOperation`].
pub const VECTOR_REDUCE_FMIN_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fmin";

/// Operation trait for `llvm.intr.vector.reduce.fmin`.
pub trait VectorReduceFminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FMIN_OPERATION_NAME
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

mlir_op!(VectorReduceFmin);

/// Constructs a new detached `llvm.intr.vector.reduce.fmin` operation.
pub fn intr_vector_reduce_fmin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fmin`")
}

/// Canonical MLIR operation name for [`VectorReduceFminimumOperation`].
pub const VECTOR_REDUCE_FMINIMUM_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fminimum";

/// Operation trait for `llvm.intr.vector.reduce.fminimum`.
pub trait VectorReduceFminimumOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FMINIMUM_OPERATION_NAME
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

mlir_op!(VectorReduceFminimum);

/// Constructs a new detached `llvm.intr.vector.reduce.fminimum` operation.
pub fn intr_vector_reduce_fminimum<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFminimumOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMINIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fminimum`")
}

/// Canonical MLIR operation name for [`VectorReduceFmulOperation`].
pub const VECTOR_REDUCE_FMUL_OPERATION_NAME: &str = "llvm.intr.vector.reduce.fmul";

/// Operation trait for `llvm.intr.vector.reduce.fmul`.
pub trait VectorReduceFmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_FMUL_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
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

mlir_op!(VectorReduceFmul);

/// Constructs a new detached `llvm.intr.vector.reduce.fmul` operation.
pub fn intr_vector_reduce_fmul<
    'v,
    'c: 'v,
    't: 'c,
    V0: Value<'v, 'c, 't>,
    V1: Value<'v, 'c, 't>,
    T0: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    start_value: V0,
    input: V1,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorReduceFmulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_fmul`")
}

/// Canonical MLIR operation name for [`VectorReduceMulOperation`].
pub const VECTOR_REDUCE_MUL_OPERATION_NAME: &str = "llvm.intr.vector.reduce.mul";

/// Operation trait for `llvm.intr.vector.reduce.mul`.
pub trait VectorReduceMulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_MUL_OPERATION_NAME
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

mlir_op!(VectorReduceMul);

/// Constructs a new detached `llvm.intr.vector.reduce.mul` operation.
pub fn intr_vector_reduce_mul<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceMulOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_MUL_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_mul`")
}

/// Canonical MLIR operation name for [`VectorReduceOrOperation`].
pub const VECTOR_REDUCE_OR_OPERATION_NAME: &str = "llvm.intr.vector.reduce.or";

/// Operation trait for `llvm.intr.vector.reduce.or`.
pub trait VectorReduceOrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_OR_OPERATION_NAME
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

mlir_op!(VectorReduceOr);

/// Constructs a new detached `llvm.intr.vector.reduce.or` operation.
pub fn intr_vector_reduce_or<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceOrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_OR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_or`")
}

/// Canonical MLIR operation name for [`VectorReduceSmaxOperation`].
pub const VECTOR_REDUCE_SMAX_OPERATION_NAME: &str = "llvm.intr.vector.reduce.smax";

/// Operation trait for `llvm.intr.vector.reduce.smax`.
pub trait VectorReduceSmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_SMAX_OPERATION_NAME
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

mlir_op!(VectorReduceSmax);

/// Constructs a new detached `llvm.intr.vector.reduce.smax` operation.
pub fn intr_vector_reduce_smax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceSmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_smax`")
}

/// Canonical MLIR operation name for [`VectorReduceSminOperation`].
pub const VECTOR_REDUCE_SMIN_OPERATION_NAME: &str = "llvm.intr.vector.reduce.smin";

/// Operation trait for `llvm.intr.vector.reduce.smin`.
pub trait VectorReduceSminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_SMIN_OPERATION_NAME
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

mlir_op!(VectorReduceSmin);

/// Constructs a new detached `llvm.intr.vector.reduce.smin` operation.
pub fn intr_vector_reduce_smin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceSminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_smin`")
}

/// Canonical MLIR operation name for [`VectorReduceUmaxOperation`].
pub const VECTOR_REDUCE_UMAX_OPERATION_NAME: &str = "llvm.intr.vector.reduce.umax";

/// Operation trait for `llvm.intr.vector.reduce.umax`.
pub trait VectorReduceUmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_UMAX_OPERATION_NAME
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

mlir_op!(VectorReduceUmax);

/// Constructs a new detached `llvm.intr.vector.reduce.umax` operation.
pub fn intr_vector_reduce_umax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceUmaxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_umax`")
}

/// Canonical MLIR operation name for [`VectorReduceUminOperation`].
pub const VECTOR_REDUCE_UMIN_OPERATION_NAME: &str = "llvm.intr.vector.reduce.umin";

/// Operation trait for `llvm.intr.vector.reduce.umin`.
pub trait VectorReduceUminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_UMIN_OPERATION_NAME
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

mlir_op!(VectorReduceUmin);

/// Constructs a new detached `llvm.intr.vector.reduce.umin` operation.
pub fn intr_vector_reduce_umin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceUminOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_umin`")
}

/// Canonical MLIR operation name for [`VectorReduceXorOperation`].
pub const VECTOR_REDUCE_XOR_OPERATION_NAME: &str = "llvm.intr.vector.reduce.xor";

/// Operation trait for `llvm.intr.vector.reduce.xor`.
pub trait VectorReduceXorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VECTOR_REDUCE_XOR_OPERATION_NAME
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

mlir_op!(VectorReduceXor);

/// Constructs a new detached `llvm.intr.vector.reduce.xor` operation.
pub fn intr_vector_reduce_xor<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> DetachedVectorReduceXorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_XOR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vector_reduce_xor`")
}

/// Canonical MLIR operation name for [`VscaleOperation`].
pub const VSCALE_OPERATION_NAME: &str = "llvm.intr.vscale";

/// Operation trait for `llvm.intr.vscale`.
pub trait VscaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VSCALE_OPERATION_NAME
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> TypeRef<'c, 't> {
        self.result_type(0).unwrap()
    }
}

mlir_op!(Vscale);

/// Constructs a new detached `llvm.intr.vscale` operation.
pub fn intr_vscale<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> DetachedVscaleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm());
    let mut builder = OperationBuilder::new(VSCALE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `llvm::intr_vscale`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Operation, dialects::func};

    use super::*;

    #[test]
    fn test_sin() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        module.body().append_operation({
            let mut block = context.block(&[(f32_type, location)]);
            let input = block.argument(0).unwrap();
            let op = intr_sin(input, f32_type, None, location);
            assert_eq!(op.operation_name(), "llvm.intr.sin");
            assert_eq!(op.input(), input);
            assert_eq!(op.operands().count(), 1);
            assert_eq!(op.results().count(), 1);
            assert_eq!(op.output_type(), f32_type);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_sin_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into()],
                    results: vec![f32_type.into()],
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
                  func.func @llvm_intr_sin_test(%arg0: f32) -> f32 {
                    %0 = llvm.intr.sin(%arg0) : (f32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }
}
