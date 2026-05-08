use crate::{
    AttributeRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value,
    ValueRef, mlir_op,
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
    fn vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorDeinterleave2);

/// Constructs a new detached `llvm.intr.vector.deinterleave2` operation.
pub fn intr_vector_deinterleave2<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    vector: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorDeinterleave2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_DEINTERLEAVE2_OPERATION_NAME, location);
    builder = builder.add_operand(vector);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_deinterleave2`"))
    })
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
    fn source_vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `pos` attribute.
    fn pos(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("pos")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "pos",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorExtract);

/// Constructs a new detached `llvm.intr.vector.extract` operation.
pub fn intr_vector_extract<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    source_vector: V0,
    result_type: T0,
    pos: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorExtractOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_EXTRACT_OPERATION_NAME, location);
    builder = builder.add_operand(source_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("pos", pos);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_extract`"))
    })
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
    fn destination_vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source_vector` operand.
    fn source_vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `pos` attribute.
    fn pos(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("pos")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "pos",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedVectorInsertOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_INSERT_OPERATION_NAME, location);
    builder = builder.add_operand(destination_vector);
    builder = builder.add_operand(source_vector);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("pos", pos);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_insert`"))
    })
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
    fn first_vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `second_vector` operand.
    fn second_vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedVectorInterleave2Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_INTERLEAVE2_OPERATION_NAME, location);
    builder = builder.add_operand(first_vector);
    builder = builder.add_operand(second_vector);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_interleave2`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceAdd);

/// Constructs a new detached `llvm.intr.vector.reduce.add` operation.
pub fn intr_vector_reduce_add<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceAddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_ADD_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_add`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceAnd);

/// Constructs a new detached `llvm.intr.vector.reduce.and` operation.
pub fn intr_vector_reduce_and<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceAndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_AND_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_and`"))
    })
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
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedVectorReduceFaddOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FADD_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fadd`"))
    })
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

mlir_op!(VectorReduceFmax);

/// Constructs a new detached `llvm.intr.vector.reduce.fmax` operation.
pub fn intr_vector_reduce_fmax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorReduceFmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fmax`"))
    })
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

mlir_op!(VectorReduceFmaximum);

/// Constructs a new detached `llvm.intr.vector.reduce.fmaximum` operation.
pub fn intr_vector_reduce_fmaximum<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorReduceFmaximumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMAXIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fmaximum`"))
    })
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

mlir_op!(VectorReduceFmin);

/// Constructs a new detached `llvm.intr.vector.reduce.fmin` operation.
pub fn intr_vector_reduce_fmin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorReduceFminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fmin`"))
    })
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

mlir_op!(VectorReduceFminimum);

/// Constructs a new detached `llvm.intr.vector.reduce.fminimum` operation.
pub fn intr_vector_reduce_fminimum<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    fastmath_flags: Option<AttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorReduceFminimumOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMINIMUM_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fminimum`"))
    })
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
    fn start_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedVectorReduceFmulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_FMUL_OPERATION_NAME, location);
    builder = builder.add_operand(start_value);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    if let Some(fastmath_flags) = fastmath_flags {
        builder = builder.add_attribute("fastmathFlags", fastmath_flags);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_fmul`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceMul);

/// Constructs a new detached `llvm.intr.vector.reduce.mul` operation.
pub fn intr_vector_reduce_mul<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceMulOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_MUL_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_mul`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceOr);

/// Constructs a new detached `llvm.intr.vector.reduce.or` operation.
pub fn intr_vector_reduce_or<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceOrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_OR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_or`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceSmax);

/// Constructs a new detached `llvm.intr.vector.reduce.smax` operation.
pub fn intr_vector_reduce_smax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceSmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_SMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_smax`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceSmin);

/// Constructs a new detached `llvm.intr.vector.reduce.smin` operation.
pub fn intr_vector_reduce_smin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceSminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_SMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_smin`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceUmax);

/// Constructs a new detached `llvm.intr.vector.reduce.umax` operation.
pub fn intr_vector_reduce_umax<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceUmaxOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_UMAX_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_umax`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceUmin);

/// Constructs a new detached `llvm.intr.vector.reduce.umin` operation.
pub fn intr_vector_reduce_umin<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceUminOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_UMIN_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_umin`"))
    })
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
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(VectorReduceXor);

/// Constructs a new detached `llvm.intr.vector.reduce.xor` operation.
pub fn intr_vector_reduce_xor<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    input: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedVectorReduceXorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VECTOR_REDUCE_XOR_OPERATION_NAME, location);
    builder = builder.add_operand(input);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vector_reduce_xor`"))
    })
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
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(Vscale);

/// Constructs a new detached `llvm.intr.vscale` operation.
pub fn intr_vscale<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> Result<DetachedVscaleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VSCALE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_vscale`"))
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
    fn test_intr_vector_deinterleave2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector8_i32_type = context.parse_type("vector<8xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector8_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_deinterleave2(arg_0, vector_i32_type, location).unwrap();
                assert_eq!(op.vector().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.deinterleave2");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_deinterleave2_test",
                    func::FuncAttributes {
                        arguments: vec![vector8_i32_type.into()],
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
                  func.func @llvm_intr_vector_deinterleave2_test(%arg0: vector<8xi32>) -> vector<4xi32> {
                    %0 = \"llvm.intr.vector.deinterleave2\"(%arg0) : (vector<8xi32>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_extract() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let pos = context.integer_attribute(i64_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_extract(arg_0, vector_i32_type, pos, location).unwrap();
                assert_eq!(op.source_vector().unwrap(), arg_0);
                assert_eq!(op.pos().unwrap(), pos);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.extract");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_extract_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_extract_test(%arg0: vector<4xi32>) -> vector<4xi32> {
                    %0 = llvm.intr.vector.extract %arg0[1] : vector<4xi32> from vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_insert() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let pos = context.integer_attribute(i64_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(vector_i32_type.as_ref(), location), (vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_vector_insert(arg_0, arg_1, vector_i32_type, pos, location).unwrap();
                assert_eq!(op.destination_vector().unwrap(), arg_0);
                assert_eq!(op.source_vector().unwrap(), arg_1);
                assert_eq!(op.pos().unwrap(), pos);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.insert");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_insert_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_insert_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<4xi32> {
                    %0 = llvm.intr.vector.insert %arg1, %arg0[1] : vector<4xi32> into vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_interleave2() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector8_i32_type = context.parse_type("vector<8xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(vector_i32_type.as_ref(), location), (vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_vector_interleave2(arg_0, arg_1, vector8_i32_type, location).unwrap();
                assert_eq!(op.first_vector().unwrap(), arg_0);
                assert_eq!(op.second_vector().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), vector8_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.interleave2");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_interleave2_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), vector_i32_type.into()],
                        results: vec![vector8_i32_type.into()],
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
                  func.func @llvm_intr_vector_interleave2_test(%arg0: vector<4xi32>, %arg1: vector<4xi32>) -> vector<8xi32> {
                    %0 = \"llvm.intr.vector.interleave2\"(%arg0, %arg1) : (vector<4xi32>, vector<4xi32>) -> vector<8xi32>
                    return %0 : vector<8xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_add() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_add(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.add");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_add_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_add_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.add\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_and() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_and(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.and");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_and_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_and_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.and\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fadd() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_vector_reduce_fadd(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.input().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fadd");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fadd_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fadd_test(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
                    %0 = \"llvm.intr.vector.reduce.fadd\"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_fmax(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fmax_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmax_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmax(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmaximum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_fmaximum(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmaximum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fmaximum_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmaximum_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmaximum(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_fmin(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fmin_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmin_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fmin(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fminimum() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_fminimum(arg_0, f32_type, None, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fminimum");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fminimum_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fminimum_test(%arg0: vector<4xf32>) -> f32 {
                    %0 = llvm.intr.vector.reduce.fminimum(%arg0) : (vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_fmul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let f32_type = context.float32_type();
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_vector_reduce_fmul(arg_0, arg_1, f32_type, None, location).unwrap();
                assert_eq!(op.start_value().unwrap(), arg_0);
                assert_eq!(op.input().unwrap(), arg_1);
                assert_eq!(op.fastmath_flags().unwrap().unwrap().to_string(), "#llvm.fastmath<none>");
                assert_eq!(op.output_type().unwrap(), f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.fmul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_fmul_test",
                    func::FuncAttributes {
                        arguments: vec![f32_type.into(), vector_f32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_fmul_test(%arg0: f32, %arg1: vector<4xf32>) -> f32 {
                    %0 = \"llvm.intr.vector.reduce.fmul\"(%arg0, %arg1) <{fastmathFlags = #llvm.fastmath<none>}> : (f32, vector<4xf32>) -> f32
                    return %0 : f32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_mul() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_mul(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.mul");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_mul_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_mul_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.mul\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_or() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_or(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.or");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_or_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_or_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.or\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_smax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_smax(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.smax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_smax_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_smax_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.smax\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_smin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_smin(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.smin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_smin_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_smin_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.smin\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_umax() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_umax(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.umax");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_umax_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_umax_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.umax\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_umin() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_umin(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.umin");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_umin_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_umin_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.umin\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vector_reduce_xor() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_vector_reduce_xor(arg_0, i32_type, location).unwrap();
                assert_eq!(op.input().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vector.reduce.xor");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vector_reduce_xor_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into()],
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
                  func.func @llvm_intr_vector_reduce_xor_test(%arg0: vector<4xi32>) -> i32 {
                    %0 = \"llvm.intr.vector.reduce.xor\"(%arg0) : (vector<4xi32>) -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_vscale() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_vscale(i32_type.as_ref(), location).unwrap();
                assert_eq!(op.output_type().unwrap(), i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.vscale");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_vscale_test",
                    func::FuncAttributes { arguments: vec![], results: vec![i32_type.into()], ..Default::default() },
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
                  func.func @llvm_intr_vscale_test() -> i32 {
                    %0 = \"llvm.intr.vscale\"() : () -> i32
                    return %0 : i32
                  }
                }
            "},
        );
    }
}
