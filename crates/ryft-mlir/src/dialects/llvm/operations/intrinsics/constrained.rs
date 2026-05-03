use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef,
    mlir_op,
};

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
pub fn intr_experimental_constrained_sito_fp<
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
        .expect("invalid arguments to `llvm::intr_experimental_constrained_sito_fp`")
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
pub fn intr_experimental_constrained_uito_fp<
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
        .expect("invalid arguments to `llvm::intr_experimental_constrained_uito_fp`")
}
