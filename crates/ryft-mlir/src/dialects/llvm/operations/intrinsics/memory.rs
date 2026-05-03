use crate::{
    AttributeRef, DetachedOp, DialectHandle, Location, Operation, OperationBuilder, Type, TypeRef, Value, ValueRef,
    mlir_op,
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
