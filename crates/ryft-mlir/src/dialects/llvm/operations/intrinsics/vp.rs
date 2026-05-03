use crate::{
    DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef, mlir_op,
};

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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_ashr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.ashr");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_ashr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_add(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.add");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_add_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_and(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.and");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_and_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fadd(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fadd");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fadd_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fdiv(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fdiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fdiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fmuladd(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fmuladd");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
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
                block.into(),
                location,
            )
        });
        assert!(module.verify());
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fmul(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fmul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fmul_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fneg(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fneg");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fneg_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fpext(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fpext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fpext_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptosi(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptosi");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptosi_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptoui(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptoui");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptoui_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_f32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_fptrunc(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fptrunc");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fptrunc_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_frem(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.frem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_frem_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fsub(arg_0, arg_1, arg_2, arg_3, vector_f32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fsub");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_fsub_test",
                func::FuncAttributes {
                    arguments: vec![vector_f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_fma(arg_0, arg_1, arg_2, arg_3, arg_4, vector_f32_type, location);
            assert_eq!(op.first(), arg_0);
            assert_eq!(op.second(), arg_1);
            assert_eq!(op.third(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.fma");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
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
                block.into(),
                location,
            )
        });
        assert!(module.verify());
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i64_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_inttoptr(arg_0, arg_1, arg_2, vector_pointer_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_pointer_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.inttoptr");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_inttoptr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i64_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_pointer_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_lshr(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.lshr");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_lshr_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_load(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.load");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_merge(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.condition(), arg_0);
            assert_eq!(op.true_value(), arg_1);
            assert_eq!(op.false_value(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.merge");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_merge_test",
                func::FuncAttributes {
                    arguments: vec![mask_type.into(), vector_i32_type.into(), vector_i32_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_mul(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.mul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_mul_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_or(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.or");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_or_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i64_type = context.parse_type("vector<4xi64>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_pointer_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_ptrtoint(arg_0, arg_1, arg_2, vector_i64_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i64_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.ptrtoint");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_ptrtoint_test",
                func::FuncAttributes {
                    arguments: vec![vector_pointer_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i64_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_add(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.add");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_add_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_and(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.and");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_and_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_fadd(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fadd");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fadd_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_fmax(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmax_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_fmin(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmin_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_fmul(arg_0, arg_1, arg_2, arg_3, f32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.fmul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_fmul_test",
                func::FuncAttributes {
                    arguments: vec![f32_type.into(), vector_f32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_mul(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.mul");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_mul_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_or(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.or");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_or_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_smax(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_smax_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_smin(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.smin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_smin_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_umax(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_umax_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_umin(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.umin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_umin_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_reduce_xor(arg_0, arg_1, arg_2, arg_3, i32_type, location);
            assert_eq!(op.start_value(), arg_0);
            assert_eq!(op.value(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.reduce.xor");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_reduce_xor_test",
                func::FuncAttributes {
                    arguments: vec![i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_sdiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sdiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sdiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_sext(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sext_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_sitofp(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sitofp");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sitofp_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_smax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.smax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_smax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_smin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.smin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_smin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_srem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.srem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_srem_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_select(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.condition(), arg_0);
            assert_eq!(op.true_value(), arg_1);
            assert_eq!(op.false_value(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.select");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_select_test",
                func::FuncAttributes {
                    arguments: vec![mask_type.into(), vector_i32_type.into(), vector_i32_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_shl(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.shl");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_shl_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_store(arg_0, arg_1, arg_2, arg_3, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.operation_name(), "llvm.intr.vp.store");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
            func::func(
                "llvm_intr_vp_store_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into(), i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_experimental_vp_strided_load(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.pointer(), arg_0);
            assert_eq!(op.stride(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.load");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_experimental_vp_strided_load_test",
                func::FuncAttributes {
                    arguments: vec![pointer_type.into(), i64_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_experimental_vp_strided_store(arg_0, arg_1, arg_2, arg_3, arg_4, location);
            assert_eq!(op.value(), arg_0);
            assert_eq!(op.pointer(), arg_1);
            assert_eq!(op.stride(), arg_2);
            assert_eq!(op.mask(), arg_3);
            assert_eq!(op.explicit_vector_length(), arg_4);
            assert_eq!(op.operation_name(), "llvm.intr.experimental.vp.strided.store");
            assert_eq!(op.operands().count(), 5);
            assert_eq!(op.results().count(), 0);
            block.append_operation(op);
            block.append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location));
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
                block.into(),
                location,
            )
        });
        assert!(module.verify());
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_sub(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.sub");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_sub_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_trunc(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.trunc");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_trunc_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_udiv(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.udiv");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_udiv_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_uitofp(arg_0, arg_1, arg_2, vector_f32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_f32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.uitofp");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_uitofp_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_f32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_umax(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.umax");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_umax_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_umin(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.umin");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_umin_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_urem(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.urem");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_urem_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
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
            let op = intr_vp_xor(arg_0, arg_1, arg_2, arg_3, vector_i32_type, location);
            assert_eq!(op.lhs(), arg_0);
            assert_eq!(op.rhs(), arg_1);
            assert_eq!(op.mask(), arg_2);
            assert_eq!(op.explicit_vector_length(), arg_3);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.xor");
            assert_eq!(op.operands().count(), 4);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_xor_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
        context.load_dialect(DialectHandle::llvm());
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module.body().append_operation({
            let mut block = context.block(&[
                (vector_i32_type.as_ref(), location),
                (mask_type.as_ref(), location),
                (i32_type.as_ref(), location),
            ]);
            let arg_0 = block.argument(0).unwrap();
            let arg_1 = block.argument(1).unwrap();
            let arg_2 = block.argument(2).unwrap();
            let op = intr_vp_zext(arg_0, arg_1, arg_2, vector_i32_type, location);
            assert_eq!(op.input(), arg_0);
            assert_eq!(op.mask(), arg_1);
            assert_eq!(op.explicit_vector_length(), arg_2);
            assert_eq!(op.output_type(), vector_i32_type);
            assert_eq!(op.operation_name(), "llvm.intr.vp.zext");
            assert_eq!(op.operands().count(), 3);
            assert_eq!(op.results().count(), 1);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "llvm_intr_vp_zext_test",
                func::FuncAttributes {
                    arguments: vec![vector_i32_type.into(), mask_type.into(), i32_type.into()],
                    results: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vp_zext_test(%arg0: vector<4xi32>, %arg1: vector<4xi1>, %arg2: i32) -> vector<4xi32> {
                    %0 = \"llvm.intr.vp.zext\"(%arg0, %arg1, %arg2) : (vector<4xi32>, vector<4xi1>, i32) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }
}
