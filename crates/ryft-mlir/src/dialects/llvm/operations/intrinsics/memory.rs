use crate::{
    AttributeRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, Type, TypeRef, Value,
    ValueRef, mlir_op,
};

/// Canonical MLIR operation name for [`GetActiveLaneMaskOperation`].
pub const GET_ACTIVE_LANE_MASK_OPERATION_NAME: &str = "llvm.intr.get.active.lane.mask";

/// Operation trait for `llvm.intr.get.active.lane.mask`.
pub trait GetActiveLaneMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GET_ACTIVE_LANE_MASK_OPERATION_NAME
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `bound` operand.
    fn bound(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedGetActiveLaneMaskOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(GET_ACTIVE_LANE_MASK_OPERATION_NAME, location);
    builder = builder.add_operand(base);
    builder = builder.add_operand(bound);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_get_active_lane_mask`"))
    })
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
    fn start(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `size` attribute.
    fn size(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("size")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "size",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(InvariantEnd);

/// Constructs a new detached `llvm.intr.invariant.end` operation.
pub fn intr_invariant_end<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    start: V0,
    pointer: V1,
    size: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedInvariantEndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(INVARIANT_END_OPERATION_NAME, location);
    builder = builder.add_operand(start);
    builder = builder.add_operand(pointer);
    builder = builder.add_attribute("size", size);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_invariant_end`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `size` attribute.
    fn size(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("size")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "size",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(InvariantStart);

/// Constructs a new detached `llvm.intr.invariant.start` operation.
pub fn intr_invariant_start<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    result_type: T0,
    size: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedInvariantStartOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(INVARIANT_START_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("size", size);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_invariant_start`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedLaunderInvariantGroupOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LAUNDER_INVARIANT_GROUP_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_launder_invariant_group`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(LifetimeEnd);

/// Constructs a new detached `llvm.intr.lifetime.end` operation.
pub fn intr_lifetime_end<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> Result<DetachedLifetimeEndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LIFETIME_END_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_lifetime_end`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(LifetimeStart);

/// Constructs a new detached `llvm.intr.lifetime.start` operation.
pub fn intr_lifetime_start<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> Result<DetachedLifetimeStartOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(LIFETIME_START_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_lifetime_start`"))
    })
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
    fn data(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("alignment")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "alignment",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `nontemporal` attribute.
    fn nontemporal(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("nontemporal")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "nontemporal",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMaskedLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(data);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("alignment", alignment);
    builder = builder.add_attribute("nontemporal", nontemporal);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_load`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `data` operand.
    fn data(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("alignment")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "alignment",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMaskedStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(data);
    builder = builder.add_operand(mask);
    builder = builder.add_attribute("alignment", alignment);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_store`"))
    })
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
    fn data(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("rows")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "rows",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("columns")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "columns",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMatrixColumnMajorLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MATRIX_COLUMN_MAJOR_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(data);
    builder = builder.add_operand(stride);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_matrix_column_major_load`"))
    })
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
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `data` operand.
    fn data(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `stride` operand.
    fn stride(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("rows")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "rows",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("columns")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "columns",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMatrixColumnMajorStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MATRIX_COLUMN_MAJOR_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(matrix);
    builder = builder.add_operand(data);
    builder = builder.add_operand(stride);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_matrix_column_major_store`"))
    })
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
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `lhs_rows` attribute.
    fn lhs_rows(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("lhs_rows")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "lhs_rows",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `lhs_columns` attribute.
    fn lhs_columns(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("lhs_columns")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "lhs_columns",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `rhs_columns` attribute.
    fn rhs_columns(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("rhs_columns")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "rhs_columns",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMatrixMultiplyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MATRIX_MULTIPLY_OPERATION_NAME, location);
    builder = builder.add_operand(lhs);
    builder = builder.add_operand(rhs);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("lhs_rows", lhs_rows);
    builder = builder.add_attribute("lhs_columns", lhs_columns);
    builder = builder.add_attribute("rhs_columns", rhs_columns);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_matrix_multiply`"))
    })
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
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rows` attribute.
    fn rows(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("rows")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "rows",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `columns` attribute.
    fn columns(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("columns")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "columns",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMatrixTransposeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MATRIX_TRANSPOSE_OPERATION_NAME, location);
    builder = builder.add_operand(matrix);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("rows", rows);
    builder = builder.add_attribute("columns", columns);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_matrix_transpose`"))
    })
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
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `len` attribute.
    fn len(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("len")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "len",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMemcpyInlineOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MEMCPY_INLINE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_attribute("len", len);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_memcpy_inline`"))
    })
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
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `length` operand.
    fn length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMemcpyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MEMCPY_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_memcpy`"))
    })
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
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `length` operand.
    fn length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMemmoveOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MEMMOVE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(source);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_memmove`"))
    })
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
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `len` attribute.
    fn len(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("len")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "len",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMemsetInlineOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MEMSET_INLINE_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(value);
    builder = builder.add_attribute("len", len);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_memset_inline`"))
    })
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
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `length` operand.
    fn length(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `isVolatile` attribute.
    fn is_volatile(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("isVolatile")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "isVolatile",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMemsetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MEMSET_OPERATION_NAME, location);
    builder = builder.add_operand(destination);
    builder = builder.add_operand(value);
    builder = builder.add_operand(length);
    builder = builder.add_attribute("isVolatile", is_volatile);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_memset`"))
    })
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
    fn scope(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("scope")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "scope",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(NoAliasScopeDecl);

/// Constructs a new detached `llvm.intr.experimental.noalias.scope.decl` operation.
pub fn intr_experimental_noalias_scope_decl<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    scope: AttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedNoAliasScopeDeclOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(NO_ALIAS_SCOPE_DECL_OPERATION_NAME, location);
    builder = builder.add_attribute("scope", scope);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_experimental_noalias_scope_decl`"))
    })
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
    fn address(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rw` attribute.
    fn rw(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("rw")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "rw",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `hint` attribute.
    fn hint(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("hint")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "hint",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `cache` attribute.
    fn cache(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("cache")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "cache",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedPrefetchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(PREFETCH_OPERATION_NAME, location);
    builder = builder.add_operand(address);
    builder = builder.add_attribute("rw", rw);
    builder = builder.add_attribute("hint", hint);
    builder = builder.add_attribute("cache", cache);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_prefetch`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `annotation` operand.
    fn annotation(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `file_name` operand.
    fn file_name(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `line` operand.
    fn line(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the `attribute` operand.
    fn attribute(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(4)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedPtrAnnotationOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(PTR_ANNOTATION_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(annotation);
    builder = builder.add_operand(file_name);
    builder = builder.add_operand(line);
    builder = builder.add_operand(attribute);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ptr_annotation`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedPtrMaskOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(PTR_MASK_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_ptrmask`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(StackRestore);

/// Constructs a new detached `llvm.intr.stackrestore` operation.
pub fn intr_stack_restore<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    location: L,
) -> Result<DetachedStackRestoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(STACK_RESTORE_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_stack_restore`"))
    })
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
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(StackSave);

/// Constructs a new detached `llvm.intr.stacksave` operation.
pub fn intr_stack_save<'v, 'c: 'v, 't: 'c, T0: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T0,
    location: L,
) -> Result<DetachedStackSaveOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(STACK_SAVE_OPERATION_NAME, location);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_stack_save`"))
    })
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
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(StripInvariantGroup);

/// Constructs a new detached `llvm.intr.strip.invariant.group` operation.
pub fn intr_strip_invariant_group<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    pointer: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedStripInvariantGroupOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(STRIP_INVARIANT_GROUP_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_strip_invariant_group`"))
    })
}

/// Canonical MLIR operation name for [`ThreadlocalAddressOperation`].
pub const THREAD_LOCAL_ADDRESS_OPERATION_NAME: &str = "llvm.intr.threadlocal.address";

/// Operation trait for `llvm.intr.threadlocal.address`.
pub trait ThreadLocalAddressOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        THREAD_LOCAL_ADDRESS_OPERATION_NAME
    }

    /// Returns the `global` operand.
    fn global(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(ThreadLocalAddress);

/// Constructs a new detached `llvm.intr.threadlocal.address` operation.
pub fn intr_thread_local_address<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, T0: Type<'c, 't>, L: Location<'c, 't>>(
    global: V0,
    result_type: T0,
    location: L,
) -> Result<DetachedThreadLocalAddressOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(THREAD_LOCAL_ADDRESS_OPERATION_NAME, location);
    builder = builder.add_operand(global);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_thread_local_address`"))
    })
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
    fn destination_list(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source_list` operand.
    fn source_list(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(VaCopy);

/// Constructs a new detached `llvm.intr.vacopy` operation.
pub fn intr_va_copy<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, V1: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    destination_list: V0,
    source_list: V1,
    location: L,
) -> Result<DetachedVaCopyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VA_COPY_OPERATION_NAME, location);
    builder = builder.add_operand(destination_list);
    builder = builder.add_operand(source_list);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_va_copy`"))
    })
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
    fn argument_list(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(VaEnd);

/// Constructs a new detached `llvm.intr.vaend` operation.
pub fn intr_va_end<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument_list: V0,
    location: L,
) -> Result<DetachedVaEndOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VA_END_OPERATION_NAME, location);
    builder = builder.add_operand(argument_list);
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_va_end`"))
    })
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
    fn argument_list(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(VaStart);

/// Constructs a new detached `llvm.intr.vastart` operation.
pub fn intr_va_start<'v, 'c: 'v, 't: 'c, V0: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    argument_list: V0,
    location: L,
) -> Result<DetachedVaStartOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(VA_START_OPERATION_NAME, location);
    builder = builder.add_operand(argument_list);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_va_start`"))
    })
}

/// Canonical MLIR operation name for [`MaskedCompressstoreOperation`].
pub const MASKED_COMPRESS_STORE_OPERATION_NAME: &str = "llvm.intr.masked.compressstore";

/// Operation trait for `llvm.intr.masked.compressstore`.
pub trait MaskedCompressStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_COMPRESS_STORE_OPERATION_NAME
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
}

mlir_op!(MaskedCompressStore);

/// Constructs a new detached `llvm.intr.masked.compressstore` operation.
pub fn intr_masked_compress_store<
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
) -> Result<DetachedMaskedCompressStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_COMPRESS_STORE_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_compress_store`"))
    })
}

/// Canonical MLIR operation name for [`MaskedExpandloadOperation`].
pub const MASKED_EXPAND_LOAD_OPERATION_NAME: &str = "llvm.intr.masked.expandload";

/// Operation trait for `llvm.intr.masked.expandload`.
pub trait MaskedExpandLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the canonical MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MASKED_EXPAND_LOAD_OPERATION_NAME
    }

    /// Returns the `pointer` operand.
    fn pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `passthru` operand.
    fn passthru(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
    }
}

mlir_op!(MaskedExpandLoad);

/// Constructs a new detached `llvm.intr.masked.expandload` operation.
pub fn intr_masked_expand_load<
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
) -> Result<DetachedMaskedExpandLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_EXPAND_LOAD_OPERATION_NAME, location);
    builder = builder.add_operand(pointer);
    builder = builder.add_operand(mask);
    builder = builder.add_operand(passthru);
    builder = builder.add_result(result_type);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_expandload`"))
    })
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
    fn pointers(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("alignment")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "alignment",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns this operation's result type.
    fn output_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.result_type(0)
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
) -> Result<DetachedMaskedGatherOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_GATHER_OPERATION_NAME, location);
    builder = builder.add_operand(pointers);
    builder = builder.add_operand(mask);
    builder = builder.add_result(result_type);
    builder = builder.add_attribute("alignment", alignment);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_gather`"))
    })
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
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `pointers` operand.
    fn pointers(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `mask` operand.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `alignment` attribute.
    fn alignment(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute("alignment")?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                "alignment",
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
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
) -> Result<DetachedMaskedScatterOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::llvm()?)?;
    let mut builder = OperationBuilder::new(MASKED_SCATTER_OPERATION_NAME, location);
    builder = builder.add_operand(value);
    builder = builder.add_operand(pointers);
    builder = builder.add_operand(mask);
    builder = builder.add_attribute("alignment", alignment);
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `llvm::intr_masked_scatter`"))
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
    fn test_intr_get_active_lane_mask() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(i32_type.as_ref(), location), (i32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_get_active_lane_mask(arg_0, arg_1, mask_type, location).unwrap();
                assert_eq!(op.base().unwrap(), arg_0);
                assert_eq!(op.bound().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), mask_type);
                assert_eq!(op.operation_name(), "llvm.intr.get.active.lane.mask");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_get_active_lane_mask_test",
                    func::FuncAttributes {
                        arguments: vec![i32_type.into(), i32_type.into()],
                        results: vec![mask_type.into()],
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
                  func.func @llvm_intr_get_active_lane_mask_test(%arg0: i32, %arg1: i32) -> vector<4xi1> {
                    %0 = llvm.intr.get.active.lane.mask %arg0, %arg1 : i32, i32 to vector<4xi1>
                    return %0 : vector<4xi1>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_invariant_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let size = context.integer_attribute(i64_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_invariant_end(arg_0, arg_1, size, location).unwrap();
                assert_eq!(op.start().unwrap(), arg_0);
                assert_eq!(op.pointer().unwrap(), arg_1);
                assert_eq!(op.size().unwrap(), size);
                assert_eq!(op.operation_name(), "llvm.intr.invariant.end");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_invariant_end_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), pointer_type.into()],
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
                  func.func @llvm_intr_invariant_end_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    llvm.intr.invariant.end %arg0, 1, %arg1 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_invariant_start() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let size = context.integer_attribute(i64_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_invariant_start(arg_0, pointer_type, size, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.size().unwrap(), size);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.invariant.start");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_invariant_start_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_invariant_start_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.invariant.start 1, %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_launder_invariant_group() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_launder_invariant_group(arg_0, pointer_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.launder.invariant.group");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_launder_invariant_group_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_launder_invariant_group_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.launder.invariant.group %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lifetime_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_lifetime_end(arg_0, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.lifetime.end");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_lifetime_end_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_lifetime_end_test(%arg0: !llvm.ptr) {
                    llvm.intr.lifetime.end %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_lifetime_start() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_lifetime_start(arg_0, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.lifetime.start");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_lifetime_start_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_lifetime_start_test(%arg0: !llvm.ptr) {
                    llvm.intr.lifetime.start %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        let nontemporal = context.unit_attribute().as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (mask_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_masked_load(arg_0, arg_1, vector_i32_type, alignment, nontemporal, location).unwrap();
                assert_eq!(op.data().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.alignment().unwrap(), alignment);
                assert_eq!(op.nontemporal().unwrap(), nontemporal);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.masked.load");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_masked_load_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), mask_type.into()],
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
                  func.func @llvm_intr_masked_load_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>) -> vector<4xi32> {
                    %0 = llvm.intr.masked.load %arg0, %arg1 {alignment = 1 : i32, nontemporal} : (!llvm.ptr, vector<4xi1>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_masked_store(arg_0, arg_1, arg_2, alignment, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.data().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.alignment().unwrap(), alignment);
                assert_eq!(op.operation_name(), "llvm.intr.masked.store");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_masked_store_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into()],
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
                  func.func @llvm_intr_masked_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>) {
                    llvm.intr.masked.store %arg0, %arg1, %arg2 {alignment = 1 : i32} : vector<4xi32>, vector<4xi1> into !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_column_major_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (i64_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op =
                    intr_matrix_column_major_load(arg_0, arg_1, vector_i32_type, is_volatile, rows, columns, location)
                        .unwrap();
                assert_eq!(op.data().unwrap(), arg_0);
                assert_eq!(op.stride().unwrap(), arg_1);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.rows().unwrap(), rows);
                assert_eq!(op.columns().unwrap(), columns);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.matrix.column.major.load");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_matrix_column_major_load_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i64_type.into()],
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
                  func.func @llvm_intr_matrix_column_major_load_test(%arg0: !llvm.ptr, %arg1: i64) -> vector<4xi32> {
                    %0 = llvm.intr.matrix.column.major.load %arg0, <stride = %arg1> {columns = 1 : i32, isVolatile = false, rows = 1 : i32} : vector<4xi32> from !llvm.ptr stride i64
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_column_major_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op =
                    intr_matrix_column_major_store(arg_0, arg_1, arg_2, is_volatile, rows, columns, location).unwrap();
                assert_eq!(op.matrix().unwrap(), arg_0);
                assert_eq!(op.data().unwrap(), arg_1);
                assert_eq!(op.stride().unwrap(), arg_2);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.rows().unwrap(), rows);
                assert_eq!(op.columns().unwrap(), columns);
                assert_eq!(op.operation_name(), "llvm.intr.matrix.column.major.store");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_matrix_column_major_store_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), pointer_type.into(), i64_type.into()],
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
                  func.func @llvm_intr_matrix_column_major_store_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: i64) {
                    llvm.intr.matrix.column.major.store %arg0, %arg1, <stride = %arg2> {columns = 1 : i32, isVolatile = false, rows = 1 : i32} : vector<4xi32> to !llvm.ptr stride i64
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_multiply() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        let lhs_rows = context.integer_attribute(i32_type, 1).as_ref();
        let lhs_columns = context.integer_attribute(i32_type, 1).as_ref();
        let rhs_columns = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(vector_f32_type.as_ref(), location), (vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op =
                    intr_matrix_multiply(arg_0, arg_1, vector_f32_type, lhs_rows, lhs_columns, rhs_columns, location)
                        .unwrap();
                assert_eq!(op.lhs().unwrap(), arg_0);
                assert_eq!(op.rhs().unwrap(), arg_1);
                assert_eq!(op.lhs_rows().unwrap(), lhs_rows);
                assert_eq!(op.lhs_columns().unwrap(), lhs_columns);
                assert_eq!(op.rhs_columns().unwrap(), rhs_columns);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.matrix.multiply");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_matrix_multiply_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into(), vector_f32_type.into()],
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
                  func.func @llvm_intr_matrix_multiply_test(%arg0: vector<4xf32>, %arg1: vector<4xf32>) -> vector<4xf32> {
                    %0 = llvm.intr.matrix.multiply %arg0, %arg1 {lhs_columns = 1 : i32, lhs_rows = 1 : i32, rhs_columns = 1 : i32} : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_matrix_transpose() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let vector_f32_type = context.parse_type("vector<4xf32>").unwrap();
        let rows = context.integer_attribute(i32_type, 1).as_ref();
        let columns = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(vector_f32_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_matrix_transpose(arg_0, vector_f32_type, rows, columns, location).unwrap();
                assert_eq!(op.matrix().unwrap(), arg_0);
                assert_eq!(op.rows().unwrap(), rows);
                assert_eq!(op.columns().unwrap(), columns);
                assert_eq!(op.output_type().unwrap(), vector_f32_type);
                assert_eq!(op.operation_name(), "llvm.intr.matrix.transpose");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_matrix_transpose_test",
                    func::FuncAttributes {
                        arguments: vec![vector_f32_type.into()],
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
                  func.func @llvm_intr_matrix_transpose_test(%arg0: vector<4xf32>) -> vector<4xf32> {
                    %0 = llvm.intr.matrix.transpose %arg0 {columns = 1 : i32, rows = 1 : i32} : vector<4xf32> into vector<4xf32>
                    return %0 : vector<4xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memcpy_inline() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let len = context.integer_attribute(i64_type, 1).as_ref();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_memcpy_inline(arg_0, arg_1, len, is_volatile, location).unwrap();
                assert_eq!(op.destination().unwrap(), arg_0);
                assert_eq!(op.source().unwrap(), arg_1);
                assert_eq!(op.len().unwrap(), len);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.operation_name(), "llvm.intr.memcpy.inline");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_memcpy_inline_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), pointer_type.into()],
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
                  func.func @llvm_intr_memcpy_inline_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    \"llvm.intr.memcpy.inline\"(%arg0, %arg1) <{isVolatile = false, len = 1 : i64}> : (!llvm.ptr, !llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memcpy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_memcpy(arg_0, arg_1, arg_2, is_volatile, location).unwrap();
                assert_eq!(op.destination().unwrap(), arg_0);
                assert_eq!(op.source().unwrap(), arg_1);
                assert_eq!(op.length().unwrap(), arg_2);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.operation_name(), "llvm.intr.memcpy");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_memcpy_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), pointer_type.into(), i64_type.into()],
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
                  func.func @llvm_intr_memcpy_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
                    \"llvm.intr.memcpy\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memmove() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_memmove(arg_0, arg_1, arg_2, is_volatile, location).unwrap();
                assert_eq!(op.destination().unwrap(), arg_0);
                assert_eq!(op.source().unwrap(), arg_1);
                assert_eq!(op.length().unwrap(), arg_2);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.operation_name(), "llvm.intr.memmove");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_memmove_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), pointer_type.into(), i64_type.into()],
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
                  func.func @llvm_intr_memmove_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: i64) {
                    \"llvm.intr.memmove\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memset_inline() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i8_type = context.signless_integer_type(8);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let len = context.integer_attribute(i64_type, 1).as_ref();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (i8_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_memset_inline(arg_0, arg_1, len, is_volatile, location).unwrap();
                assert_eq!(op.destination().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.len().unwrap(), len);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.operation_name(), "llvm.intr.memset.inline");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_memset_inline_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i8_type.into()],
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
                  func.func @llvm_intr_memset_inline_test(%arg0: !llvm.ptr, %arg1: i8) {
                    \"llvm.intr.memset.inline\"(%arg0, %arg1) <{isVolatile = false, len = 1 : i64}> : (!llvm.ptr, i8) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_memset() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i8_type = context.signless_integer_type(8);
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let is_volatile = context.boolean_attribute(false).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (i8_type.as_ref(), location),
                    (i64_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_memset(arg_0, arg_1, arg_2, is_volatile, location).unwrap();
                assert_eq!(op.destination().unwrap(), arg_0);
                assert_eq!(op.value().unwrap(), arg_1);
                assert_eq!(op.length().unwrap(), arg_2);
                assert_eq!(op.is_volatile().unwrap(), is_volatile);
                assert_eq!(op.operation_name(), "llvm.intr.memset");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_memset_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i8_type.into(), i64_type.into()],
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
                  func.func @llvm_intr_memset_test(%arg0: !llvm.ptr, %arg1: i8, %arg2: i64) {
                    \"llvm.intr.memset\"(%arg0, %arg1, %arg2) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_experimental_noalias_scope_decl() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let scope = context.parse_attribute(r#"#llvm.alias_scope<id = "scope", domain = <id = "domain">>"#).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_experimental_noalias_scope_decl(scope, location).unwrap();
                assert_eq!(op.scope().unwrap(), scope);
                assert_eq!(op.operation_name(), "llvm.intr.experimental.noalias.scope.decl");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_experimental_noalias_scope_decl_test",
                    func::FuncAttributes { arguments: vec![], results: vec![], ..Default::default() },
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
                #alias_scope_domain = #llvm.alias_scope_domain<id = \"domain\">
                #alias_scope = #llvm.alias_scope<id = \"scope\", domain = #alias_scope_domain>
                module {
                  func.func @llvm_intr_experimental_noalias_scope_decl_test() {
                    llvm.intr.experimental.noalias.scope.decl #alias_scope
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_prefetch() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        let rw = context.integer_attribute(i32_type, 1).as_ref();
        let hint = context.integer_attribute(i32_type, 1).as_ref();
        let cache = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_prefetch(arg_0, rw, hint, cache, location).unwrap();
                assert_eq!(op.address().unwrap(), arg_0);
                assert_eq!(op.rw().unwrap(), rw);
                assert_eq!(op.hint().unwrap(), hint);
                assert_eq!(op.cache().unwrap(), cache);
                assert_eq!(op.operation_name(), "llvm.intr.prefetch");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_prefetch_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_prefetch_test(%arg0: !llvm.ptr) {
                    \"llvm.intr.prefetch\"(%arg0) <{cache = 1 : i32, hint = 1 : i32, rw = 1 : i32}> : (!llvm.ptr) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ptr_annotation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                    (i32_type.as_ref(), location),
                    (pointer_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let arg_3 = block.argument(3).unwrap();
                let arg_4 = block.argument(4).unwrap();
                let op = intr_ptr_annotation(arg_0, arg_1, arg_2, arg_3, arg_4, pointer_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.annotation().unwrap(), arg_1);
                assert_eq!(op.file_name().unwrap(), arg_2);
                assert_eq!(op.line().unwrap(), arg_3);
                assert_eq!(PtrAnnotationOperation::attribute(&op).unwrap(), arg_4);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.ptr.annotation");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 5);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ptr_annotation_test",
                    func::FuncAttributes {
                        arguments: vec![
                            pointer_type.into(),
                            pointer_type.into(),
                            pointer_type.into(),
                            i32_type.into(),
                            pointer_type.into(),
                        ],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_ptr_annotation_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr, %arg2: !llvm.ptr, %arg3: i32, %arg4: !llvm.ptr) -> !llvm.ptr {
                    %0 = \"llvm.intr.ptr.annotation\"(%arg0, %arg1, %arg2, %arg3, %arg4) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_ptrmask() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i64_type = context.signless_integer_type(64);
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (i64_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_ptrmask(arg_0, arg_1, pointer_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.ptrmask");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_ptrmask_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), i64_type.into()],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_ptrmask_test(%arg0: !llvm.ptr, %arg1: i64) -> !llvm.ptr {
                    %0 = llvm.intr.ptrmask %arg0, %arg1 : (!llvm.ptr, i64) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stack_restore() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_stack_restore(arg_0, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.stackrestore");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_stackrestore_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_stackrestore_test(%arg0: !llvm.ptr) {
                    llvm.intr.stackrestore %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_stack_save() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block_with_no_arguments();
                let op = intr_stack_save(pointer_type.as_ref(), location).unwrap();
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.stacksave");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_stacksave_test",
                    func::FuncAttributes {
                        arguments: vec![],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_stacksave_test() -> !llvm.ptr {
                    %0 = llvm.intr.stacksave : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_strip_invariant_group() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_strip_invariant_group(arg_0, pointer_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.strip.invariant.group");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_strip_invariant_group_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_strip_invariant_group_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = llvm.intr.strip.invariant.group %arg0 : !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_thread_local_address() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_thread_local_address(arg_0, pointer_type, location).unwrap();
                assert_eq!(op.global().unwrap(), arg_0);
                assert_eq!(op.output_type().unwrap(), pointer_type);
                assert_eq!(op.operation_name(), "llvm.intr.threadlocal.address");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_threadlocal_address_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
                        results: vec![pointer_type.into()],
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
                  func.func @llvm_intr_threadlocal_address_test(%arg0: !llvm.ptr) -> !llvm.ptr {
                    %0 = \"llvm.intr.threadlocal.address\"(%arg0) : (!llvm.ptr) -> !llvm.ptr
                    return %0 : !llvm.ptr
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_va_copy() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location), (pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_va_copy(arg_0, arg_1, location).unwrap();
                assert_eq!(op.destination_list().unwrap(), arg_0);
                assert_eq!(op.source_list().unwrap(), arg_1);
                assert_eq!(op.operation_name(), "llvm.intr.vacopy");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_vacopy_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), pointer_type.into()],
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
                  func.func @llvm_intr_vacopy_test(%arg0: !llvm.ptr, %arg1: !llvm.ptr) {
                    llvm.intr.vacopy %arg1 to %arg0 : !llvm.ptr, !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_va_end() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_va_end(arg_0, location).unwrap();
                assert_eq!(op.argument_list().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.vaend");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_vaend_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_vaend_test(%arg0: !llvm.ptr) {
                    llvm.intr.vaend %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_va_start() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(0).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(pointer_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let op = intr_va_start(arg_0, location).unwrap();
                assert_eq!(op.argument_list().unwrap(), arg_0);
                assert_eq!(op.operation_name(), "llvm.intr.vastart");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_vastart_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into()],
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
                  func.func @llvm_intr_vastart_test(%arg0: !llvm.ptr) {
                    llvm.intr.vastart %arg0 : !llvm.ptr
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_compress_store() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
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
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_masked_compress_store(arg_0, arg_1, arg_2, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.pointer().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.operation_name(), "llvm.intr.masked.compressstore");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_masked_compressstore_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), pointer_type.into(), mask_type.into()],
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
                  func.func @llvm_intr_masked_compressstore_test(%arg0: vector<4xi32>, %arg1: !llvm.ptr, %arg2: vector<4xi1>) {
                    \"llvm.intr.masked.compressstore\"(%arg0, %arg1, %arg2) : (vector<4xi32>, !llvm.ptr, vector<4xi1>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_expand_load() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
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
                    (vector_i32_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_masked_expand_load(arg_0, arg_1, arg_2, vector_i32_type, location).unwrap();
                assert_eq!(op.pointer().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.passthru().unwrap(), arg_2);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.masked.expandload");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_masked_expandload_test",
                    func::FuncAttributes {
                        arguments: vec![pointer_type.into(), mask_type.into(), vector_i32_type.into()],
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
                  func.func @llvm_intr_masked_expandload_test(%arg0: !llvm.ptr, %arg1: vector<4xi1>, %arg2: vector<4xi32>) -> vector<4xi32> {
                    %0 = \"llvm.intr.masked.expandload\"(%arg0, %arg1, %arg2) : (!llvm.ptr, vector<4xi1>, vector<4xi32>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_gather() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block =
                    context.block(&[(vector_pointer_type.as_ref(), location), (mask_type.as_ref(), location)]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let op = intr_masked_gather(arg_0, arg_1, vector_i32_type, alignment, location).unwrap();
                assert_eq!(op.pointers().unwrap(), arg_0);
                assert_eq!(op.mask().unwrap(), arg_1);
                assert_eq!(op.alignment().unwrap(), alignment);
                assert_eq!(op.output_type().unwrap(), vector_i32_type);
                assert_eq!(op.operation_name(), "llvm.intr.masked.gather");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 2);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                let op = block.append_operation(op).unwrap();
                block.append_operation(func::r#return(&[op.result(0).unwrap()], location).unwrap()).unwrap();
                func::func(
                    "llvm_intr_masked_gather_test",
                    func::FuncAttributes {
                        arguments: vec![vector_pointer_type.into(), mask_type.into()],
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
                  func.func @llvm_intr_masked_gather_test(%arg0: vector<4x!llvm.ptr>, %arg1: vector<4xi1>) -> vector<4xi32> {
                    %0 = llvm.intr.masked.gather %arg0, %arg1 {alignment = 1 : i32} : (vector<4x!llvm.ptr>, vector<4xi1>) -> vector<4xi32>
                    return %0 : vector<4xi32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_intr_masked_scatter() {
        let context = Context::new();
        context.load_dialect(DialectHandle::llvm().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let i32_type = context.signless_integer_type(32);
        let mask_type = context.parse_type("vector<4xi1>").unwrap();
        let vector_i32_type = context.parse_type("vector<4xi32>").unwrap();
        let vector_pointer_type = context.parse_type("vector<4x!llvm.ptr>").unwrap();
        let alignment = context.integer_attribute(i32_type, 1).as_ref();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[
                    (vector_i32_type.as_ref(), location),
                    (vector_pointer_type.as_ref(), location),
                    (mask_type.as_ref(), location),
                ]);
                let arg_0 = block.argument(0).unwrap();
                let arg_1 = block.argument(1).unwrap();
                let arg_2 = block.argument(2).unwrap();
                let op = intr_masked_scatter(arg_0, arg_1, arg_2, alignment, location).unwrap();
                assert_eq!(op.value().unwrap(), arg_0);
                assert_eq!(op.pointers().unwrap(), arg_1);
                assert_eq!(op.mask().unwrap(), arg_2);
                assert_eq!(op.alignment().unwrap(), alignment);
                assert_eq!(op.operation_name(), "llvm.intr.masked.scatter");
                assert_eq!(op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 3);
                assert_eq!(op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                block.append_operation(op).unwrap();
                block
                    .append_operation(func::r#return::<crate::ValueRef<'_, '_, '_>, _>(&[], location).unwrap())
                    .unwrap();
                func::func(
                    "llvm_intr_masked_scatter_test",
                    func::FuncAttributes {
                        arguments: vec![vector_i32_type.into(), vector_pointer_type.into(), mask_type.into()],
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
                  func.func @llvm_intr_masked_scatter_test(%arg0: vector<4xi32>, %arg1: vector<4x!llvm.ptr>, %arg2: vector<4xi1>) {
                    llvm.intr.masked.scatter %arg0, %arg1, %arg2 {alignment = 1 : i32} : vector<4xi32>, vector<4xi1> into vector<4x!llvm.ptr>
                    return
                  }
                }
            "},
        );
    }
}
