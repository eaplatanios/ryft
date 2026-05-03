use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef,
    mlir_op,
};

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
