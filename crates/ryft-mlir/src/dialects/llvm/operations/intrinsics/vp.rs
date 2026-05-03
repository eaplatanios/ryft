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
