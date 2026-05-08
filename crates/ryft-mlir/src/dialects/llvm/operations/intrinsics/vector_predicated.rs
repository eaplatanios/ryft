use crate::{
    DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`VpAshrOperation`].
pub const VP_ASHR_OPERATION_NAME: &str = "llvm.intr.vp.ashr";

/// Operation trait for `llvm.intr.vp.ashr`.
pub trait VpAshrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_ASHR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpAshr);

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
) -> Result<DetachedVpAshrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_ASHR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_ashr`"))
    })
}

/// Canonical MLIR operation name for [`VpAddOperation`].
pub const VP_ADD_OPERATION_NAME: &str = "llvm.intr.vp.add";

/// Operation trait for `llvm.intr.vp.add`.
pub trait VpAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_ADD_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpAdd);

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
) -> Result<DetachedVpAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_add`"))
    })
}

/// Canonical MLIR operation name for [`VpAndOperation`].
pub const VP_AND_OPERATION_NAME: &str = "llvm.intr.vp.and";

/// Operation trait for `llvm.intr.vp.and`.
pub trait VpAndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_AND_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpAnd);

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
) -> Result<DetachedVpAndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_AND_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_and`"))
    })
}

/// Canonical MLIR operation name for [`VpFaddOperation`].
pub const VP_FADD_OPERATION_NAME: &str = "llvm.intr.vp.fadd";

/// Operation trait for `llvm.intr.vp.fadd`.
pub trait VpFaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FADD_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFadd);

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
) -> Result<DetachedVpFaddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fadd`"))
    })
}

/// Canonical MLIR operation name for [`VpFdivOperation`].
pub const VP_FDIV_OPERATION_NAME: &str = "llvm.intr.vp.fdiv";

/// Operation trait for `llvm.intr.vp.fdiv`.
pub trait VpFdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFdiv);

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
) -> Result<DetachedVpFdivOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fdiv`"))
    })
}

/// Canonical MLIR operation name for [`VpFmulAddOperation`].
pub const VP_FMUL_ADD_OPERATION_NAME: &str = "llvm.intr.vp.fmuladd";

/// Operation trait for `llvm.intr.vp.fmuladd`.
pub trait VpFmulAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FMUL_ADD_OPERATION_NAME
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

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFmulAdd);

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
) -> Result<DetachedVpFmulAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FMUL_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fmuladd`"))
    })
}

/// Canonical MLIR operation name for [`VpFmulOperation`].
pub const VP_FMUL_OPERATION_NAME: &str = "llvm.intr.vp.fmul";

/// Operation trait for `llvm.intr.vp.fmul`.
pub trait VpFmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FMUL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFmul);

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
) -> Result<DetachedVpFmulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fmul`"))
    })
}

/// Canonical MLIR operation name for [`VpFnegOperation`].
pub const VP_FNEG_OPERATION_NAME: &str = "llvm.intr.vp.fneg";

/// Operation trait for `llvm.intr.vp.fneg`.
pub trait VpFnegOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FNEG_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFneg);

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
) -> Result<DetachedVpFnegOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FNEG_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fneg`"))
    })
}

/// Canonical MLIR operation name for [`VpFpextOperation`].
pub const VP_FPEXT_OPERATION_NAME: &str = "llvm.intr.vp.fpext";

/// Operation trait for `llvm.intr.vp.fpext`.
pub trait VpFpextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FPEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFpext);

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
) -> Result<DetachedVpFpextOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FPEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fpext`"))
    })
}

/// Canonical MLIR operation name for [`VpFptoSiOperation`].
pub const VP_FPTO_SI_OPERATION_NAME: &str = "llvm.intr.vp.fptosi";

/// Operation trait for `llvm.intr.vp.fptosi`.
pub trait VpFptoSiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FPTO_SI_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFptoSi);

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
) -> Result<DetachedVpFptoSiOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FPTO_SI_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fptosi`"))
    })
}

/// Canonical MLIR operation name for [`VpFptoUiOperation`].
pub const VP_FPTO_UI_OPERATION_NAME: &str = "llvm.intr.vp.fptoui";

/// Operation trait for `llvm.intr.vp.fptoui`.
pub trait VpFptoUiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FPTO_UI_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFptoUi);

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
) -> Result<DetachedVpFptoUiOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FPTO_UI_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fptoui`"))
    })
}

/// Canonical MLIR operation name for [`VpFptruncOperation`].
pub const VP_FPTRUNC_OPERATION_NAME: &str = "llvm.intr.vp.fptrunc";

/// Operation trait for `llvm.intr.vp.fptrunc`.
pub trait VpFptruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FPTRUNC_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFptrunc);

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
) -> Result<DetachedVpFptruncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FPTRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fptrunc`"))
    })
}

/// Canonical MLIR operation name for [`VpFremOperation`].
pub const VP_FREM_OPERATION_NAME: &str = "llvm.intr.vp.frem";

/// Operation trait for `llvm.intr.vp.frem`.
pub trait VpFremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFrem);

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
) -> Result<DetachedVpFremOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_frem`"))
    })
}

/// Canonical MLIR operation name for [`VpFsubOperation`].
pub const VP_FSUB_OPERATION_NAME: &str = "llvm.intr.vp.fsub";

/// Operation trait for `llvm.intr.vp.fsub`.
pub trait VpFsubOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FSUB_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFsub);

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
) -> Result<DetachedVpFsubOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FSUB_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fsub`"))
    })
}

/// Canonical MLIR operation name for [`VpFmaOperation`].
pub const VP_FMA_OPERATION_NAME: &str = "llvm.intr.vp.fma";

/// Operation trait for `llvm.intr.vp.fma`.
pub trait VpFmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_FMA_OPERATION_NAME
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

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpFma);

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
) -> Result<DetachedVpFmaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_FMA_OPERATION_NAME, location);
    builder = builder.add_operand(first);
    builder = builder.add_operand(second);
    builder = builder.add_operand(third);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_fma`"))
    })
}

/// Canonical MLIR operation name for [`VpIntToPtrOperation`].
pub const VP_INT_TO_PTR_OPERATION_NAME: &str = "llvm.intr.vp.inttoptr";

/// Operation trait for `llvm.intr.vp.inttoptr`.
pub trait VpIntToPtrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_INT_TO_PTR_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpIntToPtr);

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
) -> Result<DetachedVpIntToPtrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_INT_TO_PTR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_inttoptr`"))
    })
}

/// Canonical MLIR operation name for [`VpLshrOperation`].
pub const VP_LSHR_OPERATION_NAME: &str = "llvm.intr.vp.lshr";

/// Operation trait for `llvm.intr.vp.lshr`.
pub trait VpLshrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_LSHR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpLshr);

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
) -> Result<DetachedVpLshrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_LSHR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_lshr`"))
    })
}

/// Canonical MLIR operation name for [`VpLoadOperation`].
pub const VP_LOAD_OPERATION_NAME: &str = "llvm.intr.vp.load";

/// Operation trait for `llvm.intr.vp.load`.
pub trait VpLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_LOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpLoad);

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
) -> Result<DetachedVpLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_load`"))
    })
}

/// Canonical MLIR operation name for [`VpMergeMinOperation`].
pub const VP_MERGE_MIN_OPERATION_NAME: &str = "llvm.intr.vp.merge";

/// Operation trait for `llvm.intr.vp.merge`.
pub trait VpMergeMinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_MERGE_MIN_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `true_value` operand.
    fn true_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `false_value` operand.
    fn false_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpMergeMin);

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
) -> Result<DetachedVpMergeMinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_MERGE_MIN_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operand(true_value);
    builder = builder.add_operand(false_value);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_merge`"))
    })
}

/// Canonical MLIR operation name for [`VpMulOperation`].
pub const VP_MUL_OPERATION_NAME: &str = "llvm.intr.vp.mul";

/// Operation trait for `llvm.intr.vp.mul`.
pub trait VpMulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_MUL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpMul);

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
) -> Result<DetachedVpMulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_MUL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_mul`"))
    })
}

/// Canonical MLIR operation name for [`VpOrOperation`].
pub const VP_OR_OPERATION_NAME: &str = "llvm.intr.vp.or";

/// Operation trait for `llvm.intr.vp.or`.
pub trait VpOrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_OR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpOr);

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
) -> Result<DetachedVpOrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_OR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_or`"))
    })
}

/// Canonical MLIR operation name for [`VpPtrToIntOperation`].
pub const VP_PTR_TO_INT_OPERATION_NAME: &str = "llvm.intr.vp.ptrtoint";

/// Operation trait for `llvm.intr.vp.ptrtoint`.
pub trait VpPtrToIntOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_PTR_TO_INT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpPtrToInt);

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
) -> Result<DetachedVpPtrToIntOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_PTR_TO_INT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_ptrtoint`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceAddOperation`].
pub const VP_REDUCE_ADD_OPERATION_NAME: &str = "llvm.intr.vp.reduce.add";

/// Operation trait for `llvm.intr.vp.reduce.add`.
pub trait VpReduceAddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_ADD_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceAdd);

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
) -> Result<DetachedVpReduceAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_add`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceAndOperation`].
pub const VP_REDUCE_AND_OPERATION_NAME: &str = "llvm.intr.vp.reduce.and";

/// Operation trait for `llvm.intr.vp.reduce.and`.
pub trait VpReduceAndOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_AND_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceAnd);

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
) -> Result<DetachedVpReduceAndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_AND_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_and`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceFaddOperation`].
pub const VP_REDUCE_FADD_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fadd";

/// Operation trait for `llvm.intr.vp.reduce.fadd`.
pub trait VpReduceFaddOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_FADD_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceFadd);

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
) -> Result<DetachedVpReduceFaddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_fadd`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceFmaxOperation`].
pub const VP_REDUCE_FMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmax";

/// Operation trait for `llvm.intr.vp.reduce.fmax`.
pub trait VpReduceFmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_FMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceFmax);

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
) -> Result<DetachedVpReduceFmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_FMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_fmax`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceFminOperation`].
pub const VP_REDUCE_FMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmin";

/// Operation trait for `llvm.intr.vp.reduce.fmin`.
pub trait VpReduceFminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_FMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceFmin);

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
) -> Result<DetachedVpReduceFminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_FMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_fmin`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceFmulOperation`].
pub const VP_REDUCE_FMUL_OPERATION_NAME: &str = "llvm.intr.vp.reduce.fmul";

/// Operation trait for `llvm.intr.vp.reduce.fmul`.
pub trait VpReduceFmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_FMUL_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceFmul);

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
) -> Result<DetachedVpReduceFmulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_fmul`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceMulOperation`].
pub const VP_REDUCE_MUL_OPERATION_NAME: &str = "llvm.intr.vp.reduce.mul";

/// Operation trait for `llvm.intr.vp.reduce.mul`.
pub trait VpReduceMulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_MUL_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceMul);

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
) -> Result<DetachedVpReduceMulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_MUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_mul`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceOrOperation`].
pub const VP_REDUCE_OR_OPERATION_NAME: &str = "llvm.intr.vp.reduce.or";

/// Operation trait for `llvm.intr.vp.reduce.or`.
pub trait VpReduceOrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_OR_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceOr);

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
) -> Result<DetachedVpReduceOrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_OR_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_or`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceSmaxOperation`].
pub const VP_REDUCE_SMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.smax";

/// Operation trait for `llvm.intr.vp.reduce.smax`.
pub trait VpReduceSmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_SMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceSmax);

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
) -> Result<DetachedVpReduceSmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_smax`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceSminOperation`].
pub const VP_REDUCE_SMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.smin";

/// Operation trait for `llvm.intr.vp.reduce.smin`.
pub trait VpReduceSminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_SMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceSmin);

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
) -> Result<DetachedVpReduceSminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_smin`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceUmaxOperation`].
pub const VP_REDUCE_UMAX_OPERATION_NAME: &str = "llvm.intr.vp.reduce.umax";

/// Operation trait for `llvm.intr.vp.reduce.umax`.
pub trait VpReduceUmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_UMAX_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceUmax);

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
) -> Result<DetachedVpReduceUmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_umax`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceUminOperation`].
pub const VP_REDUCE_UMIN_OPERATION_NAME: &str = "llvm.intr.vp.reduce.umin";

/// Operation trait for `llvm.intr.vp.reduce.umin`.
pub trait VpReduceUminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_UMIN_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceUmin);

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
) -> Result<DetachedVpReduceUminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_umin`"))
    })
}

/// Canonical MLIR operation name for [`VpReduceXorOperation`].
pub const VP_REDUCE_XOR_OPERATION_NAME: &str = "llvm.intr.vp.reduce.xor";

/// Operation trait for `llvm.intr.vp.reduce.xor`.
pub trait VpReduceXorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_REDUCE_XOR_OPERATION_NAME
    }

    /// Returns the `start_value` operand.
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpReduceXor);

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
) -> Result<DetachedVpReduceXorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_REDUCE_XOR_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(value);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_reduce_xor`"))
    })
}

/// Canonical MLIR operation name for [`VpSdivOperation`].
pub const VP_SDIV_OPERATION_NAME: &str = "llvm.intr.vp.sdiv";

/// Operation trait for `llvm.intr.vp.sdiv`.
pub trait VpSdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSdiv);

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
) -> Result<DetachedVpSdivOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_sdiv`"))
    })
}

/// Canonical MLIR operation name for [`VpSextOperation`].
pub const VP_SEXT_OPERATION_NAME: &str = "llvm.intr.vp.sext";

/// Operation trait for `llvm.intr.vp.sext`.
pub trait VpSextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSext);

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
) -> Result<DetachedVpSextOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_sext`"))
    })
}

/// Canonical MLIR operation name for [`VpSitoFpOperation`].
pub const VP_SITO_FP_OPERATION_NAME: &str = "llvm.intr.vp.sitofp";

/// Operation trait for `llvm.intr.vp.sitofp`.
pub trait VpSitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SITO_FP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSitoFp);

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
) -> Result<DetachedVpSitoFpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_sitofp`"))
    })
}

/// Canonical MLIR operation name for [`VpSmaxOperation`].
pub const VP_SMAX_OPERATION_NAME: &str = "llvm.intr.vp.smax";

/// Operation trait for `llvm.intr.vp.smax`.
pub trait VpSmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSmax);

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
) -> Result<DetachedVpSmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_smax`"))
    })
}

/// Canonical MLIR operation name for [`VpSminOperation`].
pub const VP_SMIN_OPERATION_NAME: &str = "llvm.intr.vp.smin";

/// Operation trait for `llvm.intr.vp.smin`.
pub trait VpSminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSmin);

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
) -> Result<DetachedVpSminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_smin`"))
    })
}

/// Canonical MLIR operation name for [`VpSremOperation`].
pub const VP_SREM_OPERATION_NAME: &str = "llvm.intr.vp.srem";

/// Operation trait for `llvm.intr.vp.srem`.
pub trait VpSremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSrem);

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
) -> Result<DetachedVpSremOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_srem`"))
    })
}

/// Canonical MLIR operation name for [`VpSelectMinOperation`].
pub const VP_SELECT_MIN_OPERATION_NAME: &str = "llvm.intr.vp.select";

/// Operation trait for `llvm.intr.vp.select`.
pub trait VpSelectMinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SELECT_MIN_OPERATION_NAME
    }

    /// Returns the `condition` operand.
    fn condition(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `true_value` operand.
    fn true_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `false_value` operand.
    fn false_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSelectMin);

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
) -> Result<DetachedVpSelectMinOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SELECT_MIN_OPERATION_NAME, location);
    builder = builder.add_operand(condition);
    builder = builder.add_operand(true_value);
    builder = builder.add_operand(false_value);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_select`"))
    })
}

/// Canonical MLIR operation name for [`VpShlOperation`].
pub const VP_SHL_OPERATION_NAME: &str = "llvm.intr.vp.shl";

/// Operation trait for `llvm.intr.vp.shl`.
pub trait VpShlOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SHL_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpShl);

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
) -> Result<DetachedVpShlOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SHL_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_shl`"))
    })
}

/// Canonical MLIR operation name for [`VpStoreOperation`].
pub const VP_STORE_OPERATION_NAME: &str = "llvm.intr.vp.store";

/// Operation trait for `llvm.intr.vp.store`.
pub trait VpStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_STORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }
}

mlir_op!(VpStore);

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
) -> Result<DetachedVpStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_store`"))
    })
}

/// Canonical MLIR operation name for [`VpStridedLoadOperation`].
pub const VP_STRIDED_LOAD_OPERATION_NAME: &str = "llvm.intr.experimental.vp.strided.load";

/// Operation trait for `llvm.intr.experimental.vp.strided.load`.
pub trait VpStridedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_STRIDED_LOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpStridedLoad);

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
) -> Result<DetachedVpStridedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_STRIDED_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(stride);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_experimental_vp_strided_load`"))
    })
}

/// Canonical MLIR operation name for [`VpStridedStoreOperation`].
pub const VP_STRIDED_STORE_OPERATION_NAME: &str = "llvm.intr.experimental.vp.strided.store";

/// Operation trait for `llvm.intr.experimental.vp.strided.store`.
pub trait VpStridedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_STRIDED_STORE_OPERATION_NAME
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
    }
}

mlir_op!(VpStridedStore);

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
) -> Result<DetachedVpStridedStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_STRIDED_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(stride);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_experimental_vp_strided_store`"))
    })
}

/// Canonical MLIR operation name for [`VpSubOperation`].
pub const VP_SUB_OPERATION_NAME: &str = "llvm.intr.vp.sub";

/// Operation trait for `llvm.intr.vp.sub`.
pub trait VpSubOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_SUB_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpSub);

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
) -> Result<DetachedVpSubOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_SUB_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_sub`"))
    })
}

/// Canonical MLIR operation name for [`VpTruncOperation`].
pub const VP_TRUNC_OPERATION_NAME: &str = "llvm.intr.vp.trunc";

/// Operation trait for `llvm.intr.vp.trunc`.
pub trait VpTruncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_TRUNC_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpTrunc);

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
) -> Result<DetachedVpTruncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_TRUNC_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_trunc`"))
    })
}

/// Canonical MLIR operation name for [`VpUdivOperation`].
pub const VP_UDIV_OPERATION_NAME: &str = "llvm.intr.vp.udiv";

/// Operation trait for `llvm.intr.vp.udiv`.
pub trait VpUdivOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_UDIV_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpUdiv);

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
) -> Result<DetachedVpUdivOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_UDIV_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_udiv`"))
    })
}

/// Canonical MLIR operation name for [`VpUitoFpOperation`].
pub const VP_UITO_FP_OPERATION_NAME: &str = "llvm.intr.vp.uitofp";

/// Operation trait for `llvm.intr.vp.uitofp`.
pub trait VpUitoFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_UITO_FP_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpUitoFp);

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
) -> Result<DetachedVpUitoFpOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_UITO_FP_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_uitofp`"))
    })
}

/// Canonical MLIR operation name for [`VpUmaxOperation`].
pub const VP_UMAX_OPERATION_NAME: &str = "llvm.intr.vp.umax";

/// Operation trait for `llvm.intr.vp.umax`.
pub trait VpUmaxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_UMAX_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpUmax);

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
) -> Result<DetachedVpUmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_umax`"))
    })
}

/// Canonical MLIR operation name for [`VpUminOperation`].
pub const VP_UMIN_OPERATION_NAME: &str = "llvm.intr.vp.umin";

/// Operation trait for `llvm.intr.vp.umin`.
pub trait VpUminOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_UMIN_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpUmin);

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
) -> Result<DetachedVpUminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_umin`"))
    })
}

/// Canonical MLIR operation name for [`VpUremOperation`].
pub const VP_UREM_OPERATION_NAME: &str = "llvm.intr.vp.urem";

/// Operation trait for `llvm.intr.vp.urem`.
pub trait VpUremOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_UREM_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpUrem);

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
) -> Result<DetachedVpUremOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_UREM_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_urem`"))
    })
}

/// Canonical MLIR operation name for [`VpXorOperation`].
pub const VP_XOR_OPERATION_NAME: &str = "llvm.intr.vp.xor";

/// Operation trait for `llvm.intr.vp.xor`.
pub trait VpXorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_XOR_OPERATION_NAME
    }

    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpXor);

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
) -> Result<DetachedVpXorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_XOR_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_xor`"))
    })
}

/// Canonical MLIR operation name for [`VpZextOperation`].
pub const VP_ZEXT_OPERATION_NAME: &str = "llvm.intr.vp.zext";

/// Operation trait for `llvm.intr.vp.zext`.
pub trait VpZextOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VP_ZEXT_OPERATION_NAME
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `explicit_vector_length` operand.
    fn explicit_vector_length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VpZext);

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
) -> Result<DetachedVpZextOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VP_ZEXT_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(explicit_vector_length);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vp_zext`"))
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
    fn test_intr_vp_ashr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_ashr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.ashr");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_ashr_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_ashr_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.ashr\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_add(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.add");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_add_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_add_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.add\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_and(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.and");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_and_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_and_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.and\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_fadd(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fadd");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fadd_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fadd_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fadd\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fdiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_fdiv(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fdiv");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fdiv_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fdiv_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fdiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fmuladd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let arg_4 = block.argument(4).unwrap();
                let op = intr_vp_fmuladd(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.mask().unwrap(), arg_3);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_4);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fmuladd");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fmuladd_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fmuladd_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: vector<4xi1>, %arg4: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fmuladd\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_fmul(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fmul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fmul_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fmul_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fmul\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fneg() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_fneg(arg_0, arg_1, arg_2, vector_f32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fneg");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fneg_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fneg_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fneg\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fpext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_fpext(arg_0, arg_1, arg_2, vector_f32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fpext");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fpext_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fpext_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fpext\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptosi() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_fptosi(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fptosi");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fptosi_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_fptosi_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.fptosi\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptoui() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_fptoui(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fptoui");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fptoui_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_fptoui_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.fptoui\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fptrunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_fptrunc(arg_0, arg_1, arg_2, vector_f32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fptrunc");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fptrunc_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fptrunc_test(%arg0: vector<4xf32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fptrunc\"(%arg0, %arg1, %arg2) : (vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_frem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_frem(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.frem");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_frem_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_frem_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.frem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fsub() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_fsub(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fsub");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fsub_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fsub_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fsub\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_fma() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let arg_4 = block.argument(4).unwrap();
                let op = intr_vp_fma(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location).unwrap();
                assert_eq!(op.first().unwrap(), arg_0);
                assert_eq!(op.second().unwrap(), arg_1);
                assert_eq!(op.third().unwrap(), arg_2);
                assert_eq!(op.mask().unwrap(), arg_3);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_4);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.fma");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_fma_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            vector_f32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_fma_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>, %arg2: vector<4xf32>, %arg3: vector<4xi1>, %arg4: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.fma\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xf32>, vector<4xf32>, vector<4xf32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_inttoptr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i64_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_inttoptr(arg_0, arg_1, arg_2, vector_pointer_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.inttoptr");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_inttoptr_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i64_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_pointer_type.into()],
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
                  func.func @llvm_intr_vp_inttoptr_test(%arg0: vector<4xi64>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4x!llvm.ptr> {
                    %0 = \"llvm.intr.vp.inttoptr\"(%arg0, %arg1, %arg2) : (vector<4xi64>, vector<4xi1>, i32) -> vector<4x!llvm.ptr>
                    return %0 : vector<4x!llvm.ptr>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_lshr() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_lshr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.lshr");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_lshr_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_lshr_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.lshr\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_load(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.load");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_load_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_load_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.load\"(%arg0, %arg1, %arg2) : (!llvm.ptr, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_merge() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (mask_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_merge(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.condition().unwrap(), arg_0);
                assert_eq!(op.true_value().unwrap(), arg_1);
                assert_eq!(op.false_value().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.merge");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_merge_test",
                    func::FuncAttributes {
                        arguments: vec![
                            mask_type.into(),
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_merge_test(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.merge\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi1>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_mul(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.mul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_mul_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_mul_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.mul\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_or(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.or");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_or_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_or_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.or\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_ptrtoint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_pointer_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_ptrtoint(arg_0, arg_1, arg_2, vector_i64_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i64_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.ptrtoint");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_ptrtoint_test",
                    func::FuncAttributes {
                        arguments: vec![vector_pointer_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_i64_type.into()],
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
                  func.func @llvm_intr_vp_ptrtoint_test(%arg0: vector<4x!llvm.ptr>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi64> {
                    %0 = \"llvm.intr.vp.ptrtoint\"(%arg0, %arg1, %arg2) : (vector<4x!llvm.ptr>, vector<4xi1>, i32) -> vector<4xi64>
                    return %0 : vector<4xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_add(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.add");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_add_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_add_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.add\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_and(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.and");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_and_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_and_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.and\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_fadd(arg_0, arg_1, arg_2, arg_3, f32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fadd");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_fadd_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fadd_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fadd\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_fmax(arg_0, arg_1, arg_2, arg_3, f32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_fmax_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmax_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmax\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_fmin(arg_0, arg_1, arg_2, arg_3, f32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_fmin_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmin_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmin\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (f32_type.as_ref(), location),
                    (vector_f32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_fmul(arg_0, arg_1, arg_2, arg_3, f32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_fmul_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_fmul_test(%arg0: f32, %arg1: vector<4xf32>, %arg2: vector<4xi1>, %arg3: i32) -> f32 {
                    %0 = \"llvm.intr.vp.reduce.fmul\"(%arg0, %arg1, %arg2, %arg3) : (f32, vector<4xf32>, vector<4xi1>, i32) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_mul(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.mul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_mul_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_mul_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.mul\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_or(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.or");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_or_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_or_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.or\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_smax(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_smax_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_smax_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.smax\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_smin(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_smin_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_smin_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.smin\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_umax(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_umax_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_umax_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.umax\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_umin(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_umin_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_umin_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.umin\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_reduce_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_reduce_xor(arg_0, arg_1, arg_2, arg_3, i32_type, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.xor");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_reduce_xor_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_reduce_xor_test(%arg0: i32, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> i32 {
                    %0 = \"llvm.intr.vp.reduce.xor\"(%arg0, %arg1, %arg2, %arg3) : (i32, vector<4xi32>, vector<4xi1>, i32) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sdiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_sdiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.sdiv");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_sdiv_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_sdiv_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sdiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_sext(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.sext");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_sext_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_sext_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sext\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sitofp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_sitofp(arg_0, arg_1, arg_2, vector_f32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.sitofp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_sitofp_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_sitofp_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.sitofp\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_smax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.smax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_smax_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_smax_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.smax\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_smin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.smin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_smin_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_smin_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.smin\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_srem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_srem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.srem");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_srem_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_srem_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.srem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_select() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (mask_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_select(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.condition().unwrap(), arg_0);
                assert_eq!(op.true_value().unwrap(), arg_1);
                assert_eq!(op.false_value().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.select");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_select_test",
                    func::FuncAttributes {
                        arguments: vec![
                            mask_type.into(),
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_select_test(%arg0: vector<4xi1>, %arg1: vector<4xi32>, %arg2: vector<4xi32>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.select\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi1>, vector<4xi32>, vector<4xi32>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_shl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_shl(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.shl");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_shl_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_shl_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.shl\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_store(arg_0, arg_1, arg_2, arg_3, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.pointer().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.operation_name(), "llvm.intr.vp.store");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_vp_store_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>, %arg3: i32) {
                    \"llvm.intr.vp.store\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, !llvm.ptr, vector<4xi1>, i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_vp_strided_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op =
                    intr_experimental_vp_strided_load(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.stride().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.load");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_experimental_vp_strided_load_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i64_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_experimental_vp_strided_load_test(%arg0: !llvm.ptr, %arg1: i64, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.experimental.vp.strided.load\"(%arg0, %arg1, %arg2, %arg3) : (!llvm.ptr, i64, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_vp_strided_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let arg_4 = block.argument(4).unwrap();
                let op = intr_experimental_vp_strided_store(arg_0, arg_1, arg_2, arg_3, arg_4, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.pointer().unwrap(), arg_1);
                assert_eq!(op.stride().unwrap(), arg_2);
                assert_eq!(op.mask().unwrap(), arg_3);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_4);
                assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.store");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_experimental_vp_strided_store_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            pointer_type.into(),
                            i64_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_experimental_vp_strided_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: i64, %arg3: vector<4xi1>, %arg4: i32) {
                    \"llvm.intr.experimental.vp.strided.store\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (vector<4xi32>, !llvm.ptr, i64, vector<4xi1>, i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_sub() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_sub(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.sub");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_sub_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_sub_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.sub\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_trunc() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_trunc(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.trunc");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_trunc_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_trunc_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.trunc\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_udiv() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_udiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.udiv");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_udiv_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_udiv_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.udiv\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_uitofp() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_uitofp(arg_0, arg_1, arg_2, vector_f32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.uitofp");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_uitofp_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                        results: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vp_uitofp_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xf32> {
                    %0 = \"llvm.intr.vp.uitofp\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_umax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.umax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_umax_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_umax_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.umax\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_umin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.umin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_umin_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_umin_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.umin\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_urem() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_urem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.urem");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_urem_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_urem_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.urem\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let op = intr_vp_xor(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location).unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_3);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.xor");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 4);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_xor_test",
                    func::FuncAttributes {
                        arguments: vec![
                            vector_i32_type.into(),
                            vector_i32_type.into(),
                            mask_type.into(),
                            i32_type.into(),
                        ],
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
                  func.func @llvm_intr_vp_xor_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>, %arg2: vector<4xi1>, %arg3: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.xor\"(%arg0, %arg1, %arg2, %arg3) : (vector<4xi32>, vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vp_zext() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_vp_zext(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.explicit_vector_length().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vp.zext");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vp_zext_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
                  func.func @llvm_intr_vp_zext_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.zext\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }
}
