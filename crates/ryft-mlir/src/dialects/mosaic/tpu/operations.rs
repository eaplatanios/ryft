use crate::{
    Attribute, BooleanAttributeRef, DenseBooleanArrayAttributeRef, DenseInteger32ArrayAttributeRef,
    DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, Error, IntegerAttributeRef, Location, Operation,
    OperationBuilder, OperationResultRef, StringAttributeRef, TypeAttributeRef, TypeRef, ValueRef, mlir_op,
    mlir_op_trait,
};

use super::attributes::{
    ContractPrecisionAttributeRef, DotDimensionNumbersAttributeRef, PackFormatAttributeRef, ReductionKindAttributeRef,
    RoundingModeAttributeRef,
};

/// Name of the [`Attribute`] that stores the `dim` value.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Name of the [`Attribute`] that stores the `kind` value.
pub const KIND_ATTRIBUTE: &str = "kind";

/// Mosaic TPU [`Operation`] for `tpu.all_reduce` that reduces a vector across one dimension.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `dim` attribute.
    fn dim(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIM_ATTRIBUTE)
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> Result<ReductionKindAttributeRef<'c, 't>, Error> {
        self.attribute(KIND_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                KIND_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, ZeroRegions);
mlir_op_trait!(AllReduce, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.all_reduce`.
pub fn all_reduce<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    dim: IntegerAttributeRef<'c, 't>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedAllReduceOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.all_reduce", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIM_ATTRIBUTE, dim);
    builder = builder.add_attribute(KIND_ATTRIBUTE, kind);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedAllReduceOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `axis` value.
pub const AXIS_ATTRIBUTE: &str = "axis";

/// Mosaic TPU [`Operation`] for `tpu.reduce_index` that reduces vector indices across one dimension.
pub trait ReduceIndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `axis` attribute.
    fn axis(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(AXIS_ATTRIBUTE)
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> Result<ReductionKindAttributeRef<'c, 't>, Error> {
        self.attribute(KIND_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                KIND_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(ReduceIndex);
mlir_op_trait!(ReduceIndex, ZeroRegions);
mlir_op_trait!(ReduceIndex, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reduce_index`.
pub fn reduce_index<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    axis: IntegerAttributeRef<'c, 't>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedReduceIndexOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.reduce_index", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(AXIS_ATTRIBUTE, axis);
    builder = builder.add_attribute(KIND_ATTRIBUTE, kind);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedReduceIndexOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.scan` that computes a vector scan using a Mosaic TPU reduction kind.
pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() > 1 { self.operand_value(1).map(Some) } else { Ok(None) }
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> Result<ReductionKindAttributeRef<'c, 't>, Error> {
        self.attribute(KIND_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                KIND_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Scan);
mlir_op_trait!(Scan, ZeroRegions);
mlir_op_trait!(Scan, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.scan`.
pub fn scan<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    mask: Option<ValueRef<'o, 'c, 't>>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedScanOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.scan", location);
    let mut operands = Vec::new();
    operands.push(input);
    if let Some(mask) = mask {
        operands.push(mask);
    } else {
    }
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(KIND_ATTRIBUTE, kind);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedScanOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `descending` value.
pub const DESCENDING_ATTRIBUTE: &str = "descending";

/// Mosaic TPU [`Operation`] for `tpu.sort` that sorts key/value vectors.
pub trait SortOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `keys` operand.
    fn keys(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `values` operand.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() > 2 { self.operand_value(2).map(Some) } else { Ok(None) }
    }

    /// Returns the `descending` attribute.
    fn descending(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = DESCENDING_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }

    /// Returns the `output_mask` result.
    fn output_mask(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the `sorted_keys` result.
    fn sorted_keys(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }

    /// Returns the `sorted_values` result.
    fn sorted_values(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(2)
    }
}

mlir_op!(Sort);
mlir_op_trait!(Sort, ZeroRegions);
mlir_op_trait!(Sort, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sort`.
pub fn sort<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    keys: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    mask: Option<ValueRef<'o, 'c, 't>>,
    descending: Option<BooleanAttributeRef<'c, 't>>,
    output_mask_type: TypeRef<'c, 't>,
    sorted_keys_type: TypeRef<'c, 't>,
    sorted_values_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSortOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sort", location);
    let mut operands = Vec::new();
    operands.push(keys);
    operands.push(values);
    if let Some(mask) = mask {
        operands.push(mask);
    } else {
    }
    builder = builder.add_operands(&operands);
    if let Some(descending) = descending {
        builder = builder.add_attribute(DESCENDING_ATTRIBUTE, descending);
    }
    builder = builder.add_result(output_mask_type);
    builder = builder.add_result(sorted_keys_type);
    builder = builder.add_result(sorted_values_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSortOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores Mosaic TPU operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the [`Attribute`] that stores the `sublane_mask` value.
pub const SUBLANE_MASK_ATTRIBUTE: &str = "sublane_mask";

/// Name of the [`Attribute`] that stores the `sublane_stride` value.
pub const SUBLANE_STRIDE_ATTRIBUTE: &str = "sublane_stride";

/// Name of the [`Attribute`] that stores the `add` value.
pub const ADD_ATTRIBUTE: &str = "add";

/// Mosaic TPU [`Operation`] for `tpu.store` that stores a native TPU vector register into memory.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> Result<DenseBooleanArrayAttributeRef<'c, 't>, Error> {
        self.dense_boolean_array_attribute(SUBLANE_MASK_ATTRIBUTE)
    }

    /// Returns the `sublane_stride` attribute.
    fn sublane_stride(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = SUBLANE_STRIDE_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().integer_attribute(self.context().signless_integer_type(32), 1)))
    }

    /// Returns the `add` attribute.
    fn add(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = ADD_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.store`.
pub fn store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_stride: Option<IntegerAttributeRef<'c, 't>>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedStoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.store", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(value_to_store);
    operand_segment_sizes.push(1);
    operands.push(base);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(indices);
    operand_segment_sizes.push(indices.len() as i32);
    if let Some(mask) = mask {
        operands.push(mask);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    if let Some(sublane_stride) = sublane_stride {
        builder = builder.add_attribute(SUBLANE_STRIDE_ATTRIBUTE, sublane_stride);
    }
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedStoreOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.load` that loads a native TPU vector register from memory.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index)).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> Result<DenseBooleanArrayAttributeRef<'c, 't>, Error> {
        self.dense_boolean_array_attribute(SUBLANE_MASK_ATTRIBUTE)
    }

    /// Returns the `sublane_stride` attribute.
    fn sublane_stride(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = SUBLANE_STRIDE_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().integer_attribute(self.context().signless_integer_type(32), 1)))
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.load`.
pub fn load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_stride: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedLoadOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.load", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    if let Some(sublane_stride) = sublane_stride {
        builder = builder.add_attribute(SUBLANE_STRIDE_ATTRIBUTE, sublane_stride);
    }
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedLoadOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `strides` value.
pub const STRIDES_ATTRIBUTE: &str = "strides";

/// Mosaic TPU [`Operation`] for `tpu.vector_store` that stores a vector into memory.
pub trait VectorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(STRIDES_ATTRIBUTE)
    }

    /// Returns the `add` attribute.
    fn add(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = ADD_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(VectorStore);
mlir_op_trait!(VectorStore, ZeroRegions);
mlir_op_trait!(VectorStore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_store`.
pub fn vector_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorStoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.vector_store", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(value_to_store);
    operand_segment_sizes.push(1);
    operands.push(base);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(indices);
    operand_segment_sizes.push(indices.len() as i32);
    if let Some(mask) = mask {
        operands.push(mask);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedVectorStoreOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.vector_load` that loads a vector from memory.
pub trait VectorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(STRIDES_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(VectorLoad);
mlir_op_trait!(VectorLoad, ZeroRegions);
mlir_op_trait!(VectorLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_load`.
pub fn vector_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorLoadOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.vector_load", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(base);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(indices);
    operand_segment_sizes.push(indices.len() as i32);
    if let Some(mask) = mask {
        operands.push(mask);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedVectorLoadOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.strided_load` that loads a vector using explicit strides.
pub trait StridedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index)).collect()
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(STRIDES_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(StridedLoad);
mlir_op_trait!(StridedLoad, ZeroRegions);
mlir_op_trait!(StridedLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.strided_load`.
pub fn strided_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedStridedLoadOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.strided_load", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedStridedLoadOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.strided_store` that stores a vector using explicit strides.
pub trait StridedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(2);
        (0..count).map(|index| self.operand_value(2 + index)).collect()
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(STRIDES_ATTRIBUTE)
    }
}

mlir_op!(StridedStore);
mlir_op_trait!(StridedStore, ZeroRegions);
mlir_op_trait!(StridedStore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.strided_store`.
pub fn strided_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedStridedStoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.strided_store", location);
    let mut operands = Vec::new();
    operands.push(value_to_store);
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedStridedStoreOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `sublane_offsets` value.
pub const SUBLANE_OFFSETS_ATTRIBUTE: &str = "sublane_offsets";

/// Mosaic TPU [`Operation`] for `tpu.shuffled_load` that loads a vector using sublane offsets.
pub trait ShuffledLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index)).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> Result<DenseBooleanArrayAttributeRef<'c, 't>, Error> {
        self.dense_boolean_array_attribute(SUBLANE_MASK_ATTRIBUTE)
    }

    /// Returns the `sublane_offsets` attribute.
    fn sublane_offsets(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(SUBLANE_OFFSETS_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(ShuffledLoad);
mlir_op_trait!(ShuffledLoad, ZeroRegions);
mlir_op_trait!(ShuffledLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.shuffled_load`.
pub fn shuffled_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_offsets: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedShuffledLoadOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.shuffled_load", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    builder = builder.add_attribute(SUBLANE_OFFSETS_ATTRIBUTE, sublane_offsets);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedShuffledLoadOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.shuffled_store` that stores a vector using sublane offsets.
pub trait ShuffledStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(2);
        (0..count).map(|index| self.operand_value(2 + index)).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> Result<DenseBooleanArrayAttributeRef<'c, 't>, Error> {
        self.dense_boolean_array_attribute(SUBLANE_MASK_ATTRIBUTE)
    }

    /// Returns the `sublane_offsets` attribute.
    fn sublane_offsets(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(SUBLANE_OFFSETS_ATTRIBUTE)
    }
}

mlir_op!(ShuffledStore);
mlir_op_trait!(ShuffledStore, ZeroRegions);
mlir_op_trait!(ShuffledStore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.shuffled_store`.
pub fn shuffled_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_offsets: DenseInteger32ArrayAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedShuffledStoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.shuffled_store", location);
    let mut operands = Vec::new();
    operands.push(value_to_store);
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    builder = builder.add_attribute(SUBLANE_OFFSETS_ATTRIBUTE, sublane_offsets);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedShuffledStoreOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.vector_load_idx` that loads a vector using vector index operands.
pub trait VectorLoadIdxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `value` result.
    fn value(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(VectorLoadIdx);
mlir_op_trait!(VectorLoadIdx, ZeroRegions);
mlir_op_trait!(VectorLoadIdx, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_load_idx`.
pub fn vector_load_idx<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    value_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorLoadIdxOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.vector_load_idx", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(base);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(indices);
    operand_segment_sizes.push(indices.len() as i32);
    if let Some(mask) = mask {
        operands.push(mask);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_result(value_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedVectorLoadIdxOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.vector_store_idx` that stores a vector using vector index operands.
pub trait VectorStoreIdxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `add` attribute.
    fn add(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = ADD_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(VectorStoreIdx);
mlir_op_trait!(VectorStoreIdx, ZeroRegions);
mlir_op_trait!(VectorStoreIdx, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_store_idx`.
pub fn vector_store_idx<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedVectorStoreIdxOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.vector_store_idx", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(value_to_store);
    operand_segment_sizes.push(1);
    operands.push(base);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(indices);
    operand_segment_sizes.push(indices.len() as i32);
    if let Some(mask) = mask {
        operands.push(mask);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedVectorStoreIdxOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `amount` value.
pub const AMOUNT_ATTRIBUTE: &str = "amount";

/// Name of the [`Attribute`] that stores the `dimension` value.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Name of the [`Attribute`] that stores the `stride` value.
pub const STRIDE_ATTRIBUTE: &str = "stride";

/// Name of the [`Attribute`] that stores the `stride_dimension` value.
pub const STRIDE_DIMENSION_ATTRIBUTE: &str = "stride_dimension";

/// Mosaic TPU [`Operation`] for `tpu.rotate` that rotates a vector by a static amount.
pub trait RotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `amount` attribute.
    fn amount(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(AMOUNT_ATTRIBUTE)
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the `stride` attribute.
    fn stride(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(STRIDE_ATTRIBUTE) { self.integer_attribute(STRIDE_ATTRIBUTE).map(Some) } else { Ok(None) }
    }

    /// Returns the `stride_dimension` attribute.
    fn stride_dimension(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(STRIDE_DIMENSION_ATTRIBUTE) {
            self.integer_attribute(STRIDE_DIMENSION_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Rotate);
mlir_op_trait!(Rotate, ZeroRegions);
mlir_op_trait!(Rotate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.rotate`.
pub fn rotate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    amount: IntegerAttributeRef<'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    stride: Option<IntegerAttributeRef<'c, 't>>,
    stride_dimension: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedRotateOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.rotate", location);
    let mut operands = Vec::new();
    operands.push(value);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(AMOUNT_ATTRIBUTE, amount);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    if let Some(stride) = stride {
        builder = builder.add_attribute(STRIDE_ATTRIBUTE, stride);
    }
    if let Some(stride_dimension) = stride_dimension {
        builder = builder.add_attribute(STRIDE_DIMENSION_ATTRIBUTE, stride_dimension);
    }
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedRotateOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.dynamic_rotate` that rotates a vector by a dynamic amount.
pub trait DynamicRotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the `stride` attribute.
    fn stride(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(STRIDE_ATTRIBUTE) { self.integer_attribute(STRIDE_ATTRIBUTE).map(Some) } else { Ok(None) }
    }

    /// Returns the `stride_dimension` attribute.
    fn stride_dimension(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(STRIDE_DIMENSION_ATTRIBUTE) {
            self.integer_attribute(STRIDE_DIMENSION_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(DynamicRotate);
mlir_op_trait!(DynamicRotate, ZeroRegions);
mlir_op_trait!(DynamicRotate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.dynamic_rotate`.
pub fn dynamic_rotate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    stride: Option<IntegerAttributeRef<'c, 't>>,
    stride_dimension: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedDynamicRotateOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.dynamic_rotate", location);
    let mut operands = Vec::new();
    operands.push(value);
    operands.push(amount);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    if let Some(stride) = stride {
        builder = builder.add_attribute(STRIDE_ATTRIBUTE, stride);
    }
    if let Some(stride_dimension) = stride_dimension {
        builder = builder.add_attribute(STRIDE_DIMENSION_ATTRIBUTE, stride_dimension);
    }
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedDynamicRotateOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.scan_count` that counts duplicate occurrences in a vector scan.
pub trait ScanCountOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in_mask` operand.
    fn in_mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `values` operand.
    fn values(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `out_mask` result.
    fn out_mask(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the `counts` result.
    fn counts(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }
}

mlir_op!(ScanCount);
mlir_op_trait!(ScanCount, ZeroRegions);
mlir_op_trait!(ScanCount, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.scan_count`.
pub fn scan_count<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    in_mask: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    out_mask_type: TypeRef<'c, 't>,
    counts_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedScanCountOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.scan_count", location);
    let mut operands = Vec::new();
    operands.push(in_mask);
    operands.push(values);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(out_mask_type);
    builder = builder.add_result(counts_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedScanCountOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `dimensions` value.
pub const DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// Mosaic TPU [`Operation`] for `tpu.iota` that creates a vector iota.
pub trait IotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `dimensions` attribute.
    fn dimensions(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(DIMENSIONS_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Iota);
mlir_op_trait!(Iota, ZeroOperands);
mlir_op_trait!(Iota, ZeroRegions);
mlir_op_trait!(Iota, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.iota`.
pub fn iota<'c, 't: 'c, L: Location<'c, 't>>(
    dimensions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedIotaOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.iota", location);
    builder = builder.add_attribute(DIMENSIONS_ATTRIBUTE, dimensions);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedIotaOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.reshape` that reshapes a TPU vector.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reshape`.
pub fn reshape<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedReshapeOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.reshape", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedReshapeOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `times` value.
pub const TIMES_ATTRIBUTE: &str = "times";

/// Mosaic TPU [`Operation`] for `tpu.repeat` that repeats values along a vector dimension.
pub trait RepeatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the `times` attribute.
    fn times(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(TIMES_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Repeat);
mlir_op_trait!(Repeat, ZeroRegions);
mlir_op_trait!(Repeat, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.repeat`.
pub fn repeat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    times: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedRepeatOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.repeat", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_attribute(TIMES_ATTRIBUTE, times);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedRepeatOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `lane` value.
pub const LANE_ATTRIBUTE: &str = "lane";

/// Mosaic TPU [`Operation`] for `tpu.broadcast_in_sublanes` that broadcasts a lane value within each sublane.
pub trait BroadcastInSublanesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `lane` attribute.
    fn lane(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(LANE_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(BroadcastInSublanes);
mlir_op_trait!(BroadcastInSublanes, ZeroRegions);
mlir_op_trait!(BroadcastInSublanes, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.broadcast_in_sublanes`.
pub fn broadcast_in_sublanes<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    lane: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedBroadcastInSublanesOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.broadcast_in_sublanes", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(LANE_ATTRIBUTE, lane);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedBroadcastInSublanesOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `index` value.
pub const INDEX_ATTRIBUTE: &str = "index";

/// Name of the [`Attribute`] that stores the `pack_format` value.
pub const PACK_FORMAT_ATTRIBUTE: &str = "pack_format";

/// Name of the [`Attribute`] that stores the `integer_extended` value.
pub const INTEGER_EXTENDED_ATTRIBUTE: &str = "integer_extended";

/// Name of the [`Attribute`] that stores the `unsigned_integers` value.
pub const UNSIGNED_INTEGERS_ATTRIBUTE: &str = "unsigned_integers";

/// Mosaic TPU [`Operation`] for `tpu.unpack_subelements` that unpacks subelements from a packed vector.
pub trait UnpackSubelementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `index` attribute.
    fn index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(INDEX_ATTRIBUTE)
    }

    /// Returns the `pack_format` attribute.
    fn pack_format(&self) -> Result<PackFormatAttributeRef<'c, 't>, Error> {
        self.attribute(PACK_FORMAT_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                PACK_FORMAT_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `integer_extended` attribute.
    fn integer_extended(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = INTEGER_EXTENDED_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(true)))
    }

    /// Returns the `unsigned_integers` attribute.
    fn unsigned_integers(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = UNSIGNED_INTEGERS_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(UnpackSubelements);
mlir_op_trait!(UnpackSubelements, ZeroRegions);
mlir_op_trait!(UnpackSubelements, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unpack_subelements`.
pub fn unpack_subelements<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    index: IntegerAttributeRef<'c, 't>,
    pack_format: PackFormatAttributeRef<'c, 't>,
    integer_extended: Option<BooleanAttributeRef<'c, 't>>,
    unsigned_integers: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedUnpackSubelementsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.unpack_subelements", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(INDEX_ATTRIBUTE, index);
    builder = builder.add_attribute(PACK_FORMAT_ATTRIBUTE, pack_format);
    if let Some(integer_extended) = integer_extended {
        builder = builder.add_attribute(INTEGER_EXTENDED_ATTRIBUTE, integer_extended);
    }
    if let Some(unsigned_integers) = unsigned_integers {
        builder = builder.add_attribute(UNSIGNED_INTEGERS_ATTRIBUTE, unsigned_integers);
    }
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedUnpackSubelementsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `positions` value.
pub const POSITIONS_ATTRIBUTE: &str = "positions";

/// Mosaic TPU [`Operation`] for `tpu.pack_subelements` that packs subelements from multiple vector registers.
pub trait PackSubelementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `positions` attribute.
    fn positions(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(POSITIONS_ATTRIBUTE)
    }

    /// Returns the `pack_format` attribute.
    fn pack_format(&self) -> Result<PackFormatAttributeRef<'c, 't>, Error> {
        self.attribute(PACK_FORMAT_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                PACK_FORMAT_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `unsigned_integers` attribute.
    fn unsigned_integers(&self) -> Result<Option<BooleanAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(UNSIGNED_INTEGERS_ATTRIBUTE) {
            self.boolean_attribute(UNSIGNED_INTEGERS_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(PackSubelements);
mlir_op_trait!(PackSubelements, ZeroRegions);
mlir_op_trait!(PackSubelements, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_subelements`.
pub fn pack_subelements<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    positions: DenseInteger32ArrayAttributeRef<'c, 't>,
    pack_format: PackFormatAttributeRef<'c, 't>,
    unsigned_integers: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedPackSubelementsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.pack_subelements", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(POSITIONS_ATTRIBUTE, positions);
    builder = builder.add_attribute(PACK_FORMAT_ATTRIBUTE, pack_format);
    if let Some(unsigned_integers) = unsigned_integers {
        builder = builder.add_attribute(UNSIGNED_INTEGERS_ATTRIBUTE, unsigned_integers);
    }
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedPackSubelementsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `target_type` value.
pub const TARGET_TYPE_ATTRIBUTE: &str = "target_type";

/// Mosaic TPU [`Operation`] for `tpu.pack_elementwise` that packs vectors elementwise.
pub trait PackElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `target_type` attribute.
    fn target_type(&self) -> Result<TypeAttributeRef<'c, 't>, Error> {
        self.type_attribute(TARGET_TYPE_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(PackElementwise);
mlir_op_trait!(PackElementwise, ZeroRegions);
mlir_op_trait!(PackElementwise, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_elementwise`.
pub fn pack_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    target_type: TypeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedPackElementwiseOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.pack_elementwise", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(TARGET_TYPE_ATTRIBUTE, target_type);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedPackElementwiseOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `source_type` value.
pub const SOURCE_TYPE_ATTRIBUTE: &str = "source_type";

/// Mosaic TPU [`Operation`] for `tpu.unpack_elementwise` that unpacks a vector elementwise.
pub trait UnpackElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `source_type` attribute.
    fn source_type(&self) -> Result<TypeAttributeRef<'c, 't>, Error> {
        self.type_attribute(SOURCE_TYPE_ATTRIBUTE)
    }

    /// Returns the `index` attribute.
    fn index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(INDEX_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(UnpackElementwise);
mlir_op_trait!(UnpackElementwise, ZeroRegions);
mlir_op_trait!(UnpackElementwise, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unpack_elementwise`.
pub fn unpack_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    source_type: TypeAttributeRef<'c, 't>,
    index: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedUnpackElementwiseOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.unpack_elementwise", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SOURCE_TYPE_ATTRIBUTE, source_type);
    builder = builder.add_attribute(INDEX_ATTRIBUTE, index);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedUnpackElementwiseOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.relayout` that changes a vector register layout.
pub trait RelayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Relayout);
mlir_op_trait!(Relayout, ZeroRegions);
mlir_op_trait!(Relayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.relayout`.
pub fn relayout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedRelayoutOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.relayout", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedRelayoutOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.pack_vmsk` that packs TPU vector masks.
pub trait PackMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `positions` attribute.
    fn positions(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(POSITIONS_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(PackMask);
mlir_op_trait!(PackMask, ZeroRegions);
mlir_op_trait!(PackMask, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_vmsk`.
pub fn pack_mask<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    positions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedPackMaskOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.pack_vmsk", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(POSITIONS_ATTRIBUTE, positions);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedPackMaskOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `indices` value.
pub const INDICES_ATTRIBUTE: &str = "indices";

/// Mosaic TPU [`Operation`] for `tpu.gather` that gathers values from a vector.
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` attribute.
    fn indices(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(INDICES_ATTRIBUTE)
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Gather);
mlir_op_trait!(Gather, ZeroRegions);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.gather`.
pub fn gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    indices: DenseInteger32ArrayAttributeRef<'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGatherOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.gather", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(INDICES_ATTRIBUTE, indices);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedGatherOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.dynamic_gather` that gathers values using dynamic vector indices.
pub trait DynamicGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` operand.
    fn indices(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `dimensions` attribute.
    fn dimensions(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(DIMENSIONS_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(DynamicGather);
mlir_op_trait!(DynamicGather, ZeroRegions);
mlir_op_trait!(DynamicGather, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.dynamic_gather`.
pub fn dynamic_gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    indices: ValueRef<'o, 'c, 't>,
    dimensions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedDynamicGatherOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.dynamic_gather", location);
    let mut operands = Vec::new();
    operands.push(source);
    operands.push(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSIONS_ATTRIBUTE, dimensions);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedDynamicGatherOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `rounding_mode` value.
pub const ROUNDING_MODE_ATTRIBUTE: &str = "rounding_mode";

/// Mosaic TPU [`Operation`] for `tpu.fptosi` that converts floating-point values to signed integers.
pub trait FpToSiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> Result<RoundingModeAttributeRef<'c, 't>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ROUNDING_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(FpToSi);
mlir_op_trait!(FpToSi, ZeroRegions);
mlir_op_trait!(FpToSi, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fptosi`.
pub fn fp_to_si<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedFpToSiOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.fptosi", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedFpToSiOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.fptoui` that converts floating-point values to unsigned integers.
pub trait FpToUiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> Result<RoundingModeAttributeRef<'c, 't>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ROUNDING_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(FpToUi);
mlir_op_trait!(FpToUi, ZeroRegions);
mlir_op_trait!(FpToUi, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fptoui`.
pub fn fp_to_ui<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedFpToUiOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.fptoui", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedFpToUiOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sitofp` that converts signed integer values to floating-point values.
pub trait SiToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> Result<RoundingModeAttributeRef<'c, 't>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ROUNDING_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(SiToFp);
mlir_op_trait!(SiToFp, ZeroRegions);
mlir_op_trait!(SiToFp, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sitofp`.
pub fn si_to_fp<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSiToFpOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sitofp", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSiToFpOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.uitofp` that converts unsigned integer values to floating-point values.
pub trait UiToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> Result<RoundingModeAttributeRef<'c, 't>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ROUNDING_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(UiToFp);
mlir_op_trait!(UiToFp, ZeroRegions);
mlir_op_trait!(UiToFp, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.uitofp`.
pub fn ui_to_fp<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedUiToFpOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.uitofp", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedUiToFpOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.extf` that extends floating-point values.
pub trait ExtFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `out` result.
    fn out(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(ExtF);
mlir_op_trait!(ExtF, ZeroRegions);
mlir_op_trait!(ExtF, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.extf`.
pub fn ext_f<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    out_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedExtFOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.extf", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(out_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedExtFOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.truncf` that truncates floating-point values.
pub trait TruncFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> Result<RoundingModeAttributeRef<'c, 't>, Error> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                ROUNDING_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the `out` result.
    fn out(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(TruncF);
mlir_op_trait!(TruncF, ZeroRegions);
mlir_op_trait!(TruncF, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.truncf`.
pub fn trunc_f<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    out_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedTruncFOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.truncf", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(out_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTruncFOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `transpose_lhs` value.
pub const TRANSPOSE_LHS_ATTRIBUTE: &str = "transpose_lhs";

/// Name of the [`Attribute`] that stores the `transpose_rhs` value.
pub const TRANSPOSE_RHS_ATTRIBUTE: &str = "transpose_rhs";

/// Name of the [`Attribute`] that stores the `precision` value.
pub const PRECISION_ATTRIBUTE: &str = "precision";

/// Name of the [`Attribute`] that stores the `dimension_numbers` value.
pub const DIMENSION_NUMBERS_ATTRIBUTE: &str = "dimension_numbers";

/// Mosaic TPU [`Operation`] for `tpu.matmul` that computes a TPU matrix multiplication.
pub trait MatmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `acc` operand.
    fn acc(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `transpose_lhs` attribute.
    fn transpose_lhs(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = TRANSPOSE_LHS_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }

    /// Returns the `transpose_rhs` attribute.
    fn transpose_rhs(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = TRANSPOSE_RHS_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }

    /// Returns the `precision` attribute.
    fn precision(&self) -> Result<Option<ContractPrecisionAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(PRECISION_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns the `dimension_numbers` attribute.
    fn dimension_numbers(&self) -> Result<Option<DotDimensionNumbersAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(DIMENSION_NUMBERS_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Matmul);
mlir_op_trait!(Matmul, ZeroRegions);
mlir_op_trait!(Matmul, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul`.
pub fn matmul<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    rhs: ValueRef<'o, 'c, 't>,
    acc: ValueRef<'o, 'c, 't>,
    transpose_lhs: Option<BooleanAttributeRef<'c, 't>>,
    transpose_rhs: Option<BooleanAttributeRef<'c, 't>>,
    precision: Option<ContractPrecisionAttributeRef<'c, 't>>,
    dimension_numbers: Option<DotDimensionNumbersAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMatmulOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.matmul", location);
    let mut operands = Vec::new();
    operands.push(lhs);
    operands.push(rhs);
    operands.push(acc);
    builder = builder.add_operands(&operands);
    if let Some(transpose_lhs) = transpose_lhs {
        builder = builder.add_attribute(TRANSPOSE_LHS_ATTRIBUTE, transpose_lhs);
    }
    if let Some(transpose_rhs) = transpose_rhs {
        builder = builder.add_attribute(TRANSPOSE_RHS_ATTRIBUTE, transpose_rhs);
    }
    if let Some(precision) = precision {
        builder = builder.add_attribute(PRECISION_ATTRIBUTE, precision);
    }
    if let Some(dimension_numbers) = dimension_numbers {
        builder = builder.add_attribute(DIMENSION_NUMBERS_ATTRIBUTE, dimension_numbers);
    }
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMatmulOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `mxu_index` value.
pub const MXU_INDEX_ATTRIBUTE: &str = "mxu_index";

/// Name of the [`Attribute`] that stores the `staging_register` value.
pub const STAGING_REGISTER_ATTRIBUTE: &str = "staging_register";

/// Name of the [`Attribute`] that stores the `transpose` value.
pub const TRANSPOSE_ATTRIBUTE: &str = "transpose";

/// Mosaic TPU [`Operation`] for `tpu.matmul_push_rhs` that pushes a matrix-multiply RHS value.
pub trait MatmulPushRhsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(MXU_INDEX_ATTRIBUTE)
    }

    /// Returns the `staging_register` attribute.
    fn staging_register(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = STAGING_REGISTER_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().integer_attribute(self.context().signless_integer_type(32), 0)))
    }

    /// Returns the `transpose` attribute.
    fn transpose(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = TRANSPOSE_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(MatmulPushRhs);
mlir_op_trait!(MatmulPushRhs, ZeroRegions);
mlir_op_trait!(MatmulPushRhs, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_push_rhs`.
pub fn matmul_push_rhs<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    rhs: ValueRef<'o, 'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    staging_register: Option<IntegerAttributeRef<'c, 't>>,
    transpose: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedMatmulPushRhsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.matmul_push_rhs", location);
    let mut operands = Vec::new();
    operands.push(rhs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(MXU_INDEX_ATTRIBUTE, mxu_index);
    if let Some(staging_register) = staging_register {
        builder = builder.add_attribute(STAGING_REGISTER_ATTRIBUTE, staging_register);
    }
    if let Some(transpose) = transpose {
        builder = builder.add_attribute(TRANSPOSE_ATTRIBUTE, transpose);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMatmulPushRhsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `acc` value.
pub const ACC_ATTRIBUTE: &str = "acc";

/// Name of the [`Attribute`] that stores the `load_staged_rhs` value.
pub const LOAD_STAGED_RHS_ATTRIBUTE: &str = "load_staged_rhs";

/// Mosaic TPU [`Operation`] for `tpu.matmul_acc_lhs` that accumulates a matrix-multiply LHS value.
pub trait MatmulAccLhsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `acc` attribute.
    fn acc(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(ACC_ATTRIBUTE)
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(MXU_INDEX_ATTRIBUTE)
    }

    /// Returns the `load_staged_rhs` attribute.
    fn load_staged_rhs(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(LOAD_STAGED_RHS_ATTRIBUTE) {
            self.integer_attribute(LOAD_STAGED_RHS_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }
}

mlir_op!(MatmulAccLhs);
mlir_op_trait!(MatmulAccLhs, ZeroRegions);
mlir_op_trait!(MatmulAccLhs, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_acc_lhs`.
pub fn matmul_acc_lhs<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    acc: IntegerAttributeRef<'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    load_staged_rhs: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedMatmulAccLhsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.matmul_acc_lhs", location);
    let mut operands = Vec::new();
    operands.push(lhs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ACC_ATTRIBUTE, acc);
    builder = builder.add_attribute(MXU_INDEX_ATTRIBUTE, mxu_index);
    if let Some(load_staged_rhs) = load_staged_rhs {
        builder = builder.add_attribute(LOAD_STAGED_RHS_ATTRIBUTE, load_staged_rhs);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMatmulAccLhsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.matmul_pop` that pops a matrix-multiply accumulator value.
pub trait MatmulPopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `acc` attribute.
    fn acc(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(ACC_ATTRIBUTE)
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(MXU_INDEX_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MatmulPop);
mlir_op_trait!(MatmulPop, ZeroOperands);
mlir_op_trait!(MatmulPop, ZeroRegions);
mlir_op_trait!(MatmulPop, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_pop`.
pub fn matmul_pop<'c, 't: 'c, L: Location<'c, 't>>(
    acc: IntegerAttributeRef<'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMatmulPopOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.matmul_pop", location);
    builder = builder.add_attribute(ACC_ATTRIBUTE, acc);
    builder = builder.add_attribute(MXU_INDEX_ATTRIBUTE, mxu_index);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMatmulPopOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.concatenate` that concatenates vector values.
pub trait ConcatenateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Concatenate);
mlir_op_trait!(Concatenate, ZeroRegions);
mlir_op_trait!(Concatenate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.concatenate`.
pub fn concatenate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    dimension: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedConcatenateOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.concatenate", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedConcatenateOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.bitcast` that bitcasts a value.
pub trait BitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Bitcast);
mlir_op_trait!(Bitcast, ZeroRegions);
mlir_op_trait!(Bitcast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.bitcast`.
pub fn bitcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedBitcastOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.bitcast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedBitcastOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.bitcast_vreg` that bitcasts a native TPU vector register.
pub trait BitcastVregOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(BitcastVreg);
mlir_op_trait!(BitcastVreg, ZeroRegions);
mlir_op_trait!(BitcastVreg, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.bitcast_vreg`.
pub fn bitcast_vreg<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedBitcastVregOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.bitcast_vreg", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedBitcastVregOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.weird` that computes the Mosaic TPU weird predicate operation.
pub trait WeirdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Weird);
mlir_op_trait!(Weird, ZeroRegions);
mlir_op_trait!(Weird, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.weird`.
pub fn weird<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedWeirdOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.weird", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedWeirdOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `approx` value.
pub const APPROX_ATTRIBUTE: &str = "approx";

/// Name of the [`Attribute`] that stores the `full_range` value.
pub const FULL_RANGE_ATTRIBUTE: &str = "full_range";

/// Mosaic TPU [`Operation`] for `tpu.reciprocal` that computes reciprocal values.
pub trait ReciprocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `approx` attribute.
    fn approx(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = APPROX_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }

    /// Returns the `full_range` attribute.
    fn full_range(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = FULL_RANGE_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(true)))
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(Reciprocal);
mlir_op_trait!(Reciprocal, ZeroRegions);
mlir_op_trait!(Reciprocal, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reciprocal`.
pub fn reciprocal<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    approx: Option<BooleanAttributeRef<'c, 't>>,
    full_range: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedReciprocalOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.reciprocal", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    if let Some(approx) = approx {
        builder = builder.add_attribute(APPROX_ATTRIBUTE, approx);
    }
    if let Some(full_range) = full_range {
        builder = builder.add_attribute(FULL_RANGE_ATTRIBUTE, full_range);
    }
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedReciprocalOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.stochastic_convert` that stochastically converts floating-point vector values.
pub trait StochasticConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `random` operand.
    fn random(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(StochasticConvert);
mlir_op_trait!(StochasticConvert, ZeroRegions);
mlir_op_trait!(StochasticConvert, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.stochastic_convert`.
pub fn stochastic_convert<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    random: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedStochasticConvertOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.stochastic_convert", location);
    let mut operands = Vec::new();
    operands.push(input);
    operands.push(random);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedStochasticConvertOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `dst_type` value.
pub const DST_TYPE_ATTRIBUTE: &str = "dst_type";

/// Mosaic TPU [`Operation`] for `tpu.stochastic_convert_elementwise` that stochastically converts values elementwise.
pub trait StochasticConvertElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `random` operand.
    fn random(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `dst_type` attribute.
    fn dst_type(&self) -> Result<TypeAttributeRef<'c, 't>, Error> {
        self.type_attribute(DST_TYPE_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(StochasticConvertElementwise);
mlir_op_trait!(StochasticConvertElementwise, ZeroRegions);
mlir_op_trait!(StochasticConvertElementwise, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.stochastic_convert_elementwise`.
pub fn stochastic_convert_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    random: ValueRef<'o, 'c, 't>,
    dst_type: TypeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedStochasticConvertElementwiseOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.stochastic_convert_elementwise", location);
    let mut operands = Vec::new();
    operands.push(input);
    operands.push(random);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DST_TYPE_ATTRIBUTE, dst_type);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedStochasticConvertElementwiseOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.roll_vectors` that rolls multiple vectors into one vector.
pub trait RollVectorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operands.
    fn input(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(RollVectors);
mlir_op_trait!(RollVectors, ZeroRegions);
mlir_op_trait!(RollVectors, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.roll_vectors`.
pub fn roll_vectors<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: &[ValueRef<'o, 'c, 't>],
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedRollVectorsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.roll_vectors", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedRollVectorsOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.unroll_vectors` that unrolls one vector into multiple vectors.
pub trait UnrollVectorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `output` results.
    fn output(&self) -> Result<Vec<OperationResultRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| self.result(index)).collect()
    }
}

mlir_op!(UnrollVectors);
mlir_op_trait!(UnrollVectors, ZeroRegions);
mlir_op_trait!(UnrollVectors, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unroll_vectors`.
pub fn unroll_vectors<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> Result<DetachedUnrollVectorsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.unroll_vectors", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_results(result_types);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedUnrollVectorsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.create_mask` that creates a vector mask from index bounds.
pub trait CreateMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `low` operands.
    fn low(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0) / 2;
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `high` operands.
    fn high(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0) / 2;
        (0..count).map(|index| self.operand_value(0 + count + index)).collect()
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CreateMask);
mlir_op_trait!(CreateMask, ZeroRegions);
mlir_op_trait!(CreateMask, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.create_mask`.
pub fn create_mask<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    low: &[ValueRef<'o, 'c, 't>],
    high: &[ValueRef<'o, 'c, 't>],
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedCreateMaskOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.create_mask", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(low);
    operands.extend_from_slice(high);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedCreateMaskOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `from` value.
pub const FROM_ATTRIBUTE: &str = "from";

/// Name of the [`Attribute`] that stores the `to` value.
pub const TO_ATTRIBUTE: &str = "to";

/// Mosaic TPU [`Operation`] for `tpu.create_subelement_mask` that creates a mask over contiguous subelement rows.
pub trait CreateSubelementMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `from` attribute.
    fn r#from(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(FROM_ATTRIBUTE)
    }

    /// Returns the `to` attribute.
    fn to(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(TO_ATTRIBUTE)
    }

    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(CreateSubelementMask);
mlir_op_trait!(CreateSubelementMask, ZeroOperands);
mlir_op_trait!(CreateSubelementMask, ZeroRegions);
mlir_op_trait!(CreateSubelementMask, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.create_subelement_mask`.
pub fn create_subelement_mask<'c, 't: 'c, L: Location<'c, 't>>(
    r#from: IntegerAttributeRef<'c, 't>,
    to: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedCreateSubelementMaskOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.create_subelement_mask", location);
    builder = builder.add_attribute(FROM_ATTRIBUTE, r#from);
    builder = builder.add_attribute(TO_ATTRIBUTE, to);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedCreateSubelementMaskOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `multiple` value.
pub const MULTIPLE_ATTRIBUTE: &str = "multiple";

/// Mosaic TPU [`Operation`] for `tpu.assume_multiple` that assumes a scalar value is a multiple.
pub trait AssumeMultipleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `multiple` attribute.
    fn multiple(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(MULTIPLE_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(AssumeMultiple);
mlir_op_trait!(AssumeMultiple, ZeroRegions);
mlir_op_trait!(AssumeMultiple, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.assume_multiple`.
pub fn assume_multiple<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    multiple: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedAssumeMultipleOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.assume_multiple", location);
    let mut operands = Vec::new();
    operands.push(value);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(MULTIPLE_ATTRIBUTE, multiple);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedAssumeMultipleOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.memref_slice` that slices a memref.
pub trait MemRefSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `mem_ref` operand.
    fn mem_ref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `base_idx` operands.
    fn base_idx(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the `dynamic_sizes` operands.
    fn dynamic_sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MemRefSlice);
mlir_op_trait!(MemRefSlice, ZeroRegions);
mlir_op_trait!(MemRefSlice, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_slice`.
pub fn mem_ref_slice<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    mem_ref: ValueRef<'o, 'c, 't>,
    base_idx: &[ValueRef<'o, 'c, 't>],
    dynamic_sizes: &[ValueRef<'o, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMemRefSliceOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.memref_slice", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(mem_ref);
    operand_segment_sizes.push(1);
    operands.extend_from_slice(base_idx);
    operand_segment_sizes.push(base_idx.len() as i32);
    operands.extend_from_slice(dynamic_sizes);
    operand_segment_sizes.push(dynamic_sizes.len() as i32);
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMemRefSliceOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.memref_squeeze` that squeezes a memref.
pub trait MemRefSqueezeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MemRefSqueeze);
mlir_op_trait!(MemRefSqueeze, ZeroRegions);
mlir_op_trait!(MemRefSqueeze, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_squeeze`.
pub fn mem_ref_squeeze<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMemRefSqueezeOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.memref_squeeze", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMemRefSqueezeOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.memref_reshape` that reshapes a memref.
pub trait MemRefReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MemRefReshape);
mlir_op_trait!(MemRefReshape, ZeroRegions);
mlir_op_trait!(MemRefReshape, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_reshape`.
pub fn mem_ref_reshape<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMemRefReshapeOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.memref_reshape", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMemRefReshapeOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.memref_bitcast` that bitcasts a memref.
pub trait MemRefBitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MemRefBitcast);
mlir_op_trait!(MemRefBitcast, ZeroRegions);
mlir_op_trait!(MemRefBitcast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_bitcast`.
pub fn mem_ref_bitcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMemRefBitcastOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.memref_bitcast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMemRefBitcastOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.reinterpret_cast` that reinterprets a memref type.
pub trait ReinterpretCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(ReinterpretCast);
mlir_op_trait!(ReinterpretCast, ZeroRegions);
mlir_op_trait!(ReinterpretCast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reinterpret_cast`.
pub fn reinterpret_cast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedReinterpretCastOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.reinterpret_cast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedReinterpretCastOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.assume_layout` that asserts the layout of a value.
pub trait AssumeLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(AssumeLayout);
mlir_op_trait!(AssumeLayout, ZeroRegions);
mlir_op_trait!(AssumeLayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.assume_layout`.
pub fn assume_layout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedAssumeLayoutOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.assume_layout", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedAssumeLayoutOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.erase_memref_layout` that erases a memref layout attribute.
pub trait EraseLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `operand` operand.
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(EraseLayout);
mlir_op_trait!(EraseLayout, ZeroRegions);
mlir_op_trait!(EraseLayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.erase_memref_layout`.
pub fn erase_layout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    operand: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedEraseLayoutOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.erase_memref_layout", location);
    let mut operands = Vec::new();
    operands.push(operand);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedEraseLayoutOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.device_id` that returns the current TPU device identifier.
pub trait DeviceIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(DeviceId);
mlir_op_trait!(DeviceId, ZeroOperands);
mlir_op_trait!(DeviceId, ZeroRegions);
mlir_op_trait!(DeviceId, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.device_id`.
pub fn device_id<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedDeviceIdOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.device_id", location);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedDeviceIdOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sem_read` that reads a TPU semaphore value.
pub trait SemaphoreReadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(SemaphoreRead);
mlir_op_trait!(SemaphoreRead, ZeroRegions);
mlir_op_trait!(SemaphoreRead, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_read`.
pub fn semaphore_read<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSemaphoreReadOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sem_read", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSemaphoreReadOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sem_wait` that waits on a TPU semaphore.
pub trait SemaphoreWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(SemaphoreWait);
mlir_op_trait!(SemaphoreWait, ZeroRegions);
mlir_op_trait!(SemaphoreWait, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_wait`.
pub fn semaphore_wait<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    location: L,
) -> Result<DetachedSemaphoreWaitOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sem_wait", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    operands.push(amount);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSemaphoreWaitOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sem_alloc` that allocates a TPU semaphore.
pub trait AllocaSemaphoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(AllocaSemaphore);
mlir_op_trait!(AllocaSemaphore, ZeroOperands);
mlir_op_trait!(AllocaSemaphore, ZeroRegions);
mlir_op_trait!(AllocaSemaphore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_alloc`.
pub fn alloca_semaphore<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedAllocaSemaphoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sem_alloc", location);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedAllocaSemaphoreOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sem_barrier` that returns the TPU barrier semaphore.
pub trait GetBarrierSemaphoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` result.
    fn semaphore(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(GetBarrierSemaphore);
mlir_op_trait!(GetBarrierSemaphore, ZeroOperands);
mlir_op_trait!(GetBarrierSemaphore, ZeroRegions);
mlir_op_trait!(GetBarrierSemaphore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_barrier`.
pub fn get_barrier_semaphore<'c, 't: 'c, L: Location<'c, 't>>(
    semaphore_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetBarrierSemaphoreOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sem_barrier", location);
    builder = builder.add_result(semaphore_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedGetBarrierSemaphoreOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.sem_signal` that signals a TPU semaphore.
pub trait SemaphoreSignalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }
}

mlir_op!(SemaphoreSignal);
mlir_op_trait!(SemaphoreSignal, ZeroRegions);
mlir_op_trait!(SemaphoreSignal, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_signal`.
pub fn semaphore_signal<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    device_id: Option<ValueRef<'o, 'c, 't>>,
    core_id: Option<ValueRef<'o, 'c, 't>>,
    location: L,
) -> Result<DetachedSemaphoreSignalOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sem_signal", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(semaphore);
    operand_segment_sizes.push(1);
    operands.push(amount);
    operand_segment_sizes.push(1);
    if let Some(device_id) = device_id {
        operands.push(device_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    if let Some(core_id) = core_id {
        operands.push(core_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSemaphoreSignalOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.barrier` that synchronizes TPU vector subcores.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `barrier_id` operand.
    fn barrier_id(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.barrier`.
pub fn barrier<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    barrier_id: ValueRef<'o, 'c, 't>,
    location: L,
) -> Result<DetachedBarrierOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.barrier", location);
    let mut operands = Vec::new();
    operands.push(barrier_id);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedBarrierOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `priority` value.
pub const PRIORITY_ATTRIBUTE: &str = "priority";

/// Name of the [`Attribute`] that stores the `strict_ordering` value.
pub const STRICT_ORDERING_ATTRIBUTE: &str = "strict_ordering";

/// Mosaic TPU [`Operation`] for `tpu.enqueue_dma` that enqueues a TPU DMA transfer.
pub trait EnqueueDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the optional `source_semaphore` operand.
    fn source_semaphore(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `target` operand.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `target_semaphore` operand.
    fn target_semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 4)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 5)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `priority` attribute.
    fn priority(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = PRIORITY_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().integer_attribute(self.context().signless_integer_type(32), 0)))
    }

    /// Returns the `strict_ordering` attribute.
    fn strict_ordering(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = STRICT_ORDERING_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(EnqueueDma);
mlir_op_trait!(EnqueueDma, ZeroRegions);
mlir_op_trait!(EnqueueDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.enqueue_dma`.
pub fn enqueue_dma<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    source_semaphore: Option<ValueRef<'o, 'c, 't>>,
    target: ValueRef<'o, 'c, 't>,
    target_semaphore: ValueRef<'o, 'c, 't>,
    device_id: Option<ValueRef<'o, 'c, 't>>,
    core_id: Option<ValueRef<'o, 'c, 't>>,
    priority: Option<IntegerAttributeRef<'c, 't>>,
    strict_ordering: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedEnqueueDmaOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.enqueue_dma", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(source);
    operand_segment_sizes.push(1);
    if let Some(source_semaphore) = source_semaphore {
        operands.push(source_semaphore);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    operands.push(target);
    operand_segment_sizes.push(1);
    operands.push(target_semaphore);
    operand_segment_sizes.push(1);
    if let Some(device_id) = device_id {
        operands.push(device_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    if let Some(core_id) = core_id {
        operands.push(core_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(priority) = priority {
        builder = builder.add_attribute(PRIORITY_ATTRIBUTE, priority);
    }
    if let Some(strict_ordering) = strict_ordering {
        builder = builder.add_attribute(STRICT_ORDERING_ATTRIBUTE, strict_ordering);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedEnqueueDmaOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.enqueue_indirect_dma` that enqueues an indirect TPU DMA transfer.
pub trait EnqueueIndirectDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `target` operand.
    fn target(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `offsets` operand.
    fn offsets(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the optional `offset_filter` operand.
    fn offset_filter(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() > 4 { self.operand_value(4).map(Some) } else { Ok(None) }
    }

    /// Returns the `add` attribute.
    fn add(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = ADD_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(EnqueueIndirectDma);
mlir_op_trait!(EnqueueIndirectDma, ZeroRegions);
mlir_op_trait!(EnqueueIndirectDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.enqueue_indirect_dma`.
pub fn enqueue_indirect_dma<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    target: ValueRef<'o, 'c, 't>,
    offsets: ValueRef<'o, 'c, 't>,
    semaphore: ValueRef<'o, 'c, 't>,
    offset_filter: Option<ValueRef<'o, 'c, 't>>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedEnqueueIndirectDmaOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.enqueue_indirect_dma", location);
    let mut operands = Vec::new();
    operands.push(source);
    operands.push(target);
    operands.push(offsets);
    operands.push(semaphore);
    if let Some(offset_filter) = offset_filter {
        operands.push(offset_filter);
    } else {
    }
    builder = builder.add_operands(&operands);
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedEnqueueIndirectDmaOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.wait_dma2` that waits for a TPU DMA transfer.
pub trait WaitDma2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `src` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the `dst` operand.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 4)?;
        match range.len() {
            0 => Ok(None),
            1 => self.operand_value(range.start).map(Some),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the `strict_ordering` attribute.
    fn strict_ordering(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = STRICT_ORDERING_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(WaitDma2);
mlir_op_trait!(WaitDma2, ZeroRegions);
mlir_op_trait!(WaitDma2, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.wait_dma2`.
pub fn wait_dma2<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    destination: ValueRef<'o, 'c, 't>,
    device_id: Option<ValueRef<'o, 'c, 't>>,
    core_id: Option<ValueRef<'o, 'c, 't>>,
    strict_ordering: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedWaitDma2Operation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.wait_dma2", location);
    let mut operands = Vec::new();
    let mut operand_segment_sizes = Vec::new();
    operands.push(semaphore);
    operand_segment_sizes.push(1);
    operands.push(source);
    operand_segment_sizes.push(1);
    operands.push(destination);
    operand_segment_sizes.push(1);
    if let Some(device_id) = device_id {
        operands.push(device_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    if let Some(core_id) = core_id {
        operands.push(core_id);
        operand_segment_sizes.push(1);
    } else {
        operand_segment_sizes.push(0);
    }
    builder = builder.add_operands(&operands);
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes)?;
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(strict_ordering) = strict_ordering {
        builder = builder.add_attribute(STRICT_ORDERING_ATTRIBUTE, strict_ordering);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedWaitDma2Operation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.wait_indirect_dma` that waits for an indirect TPU DMA transfer.
pub trait WaitIndirectDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `src` operand.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `dst` operand.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }
}

mlir_op!(WaitIndirectDma);
mlir_op_trait!(WaitIndirectDma, ZeroRegions);
mlir_op_trait!(WaitIndirectDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.wait_indirect_dma`.
pub fn wait_indirect_dma<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    destination: ValueRef<'o, 'c, 't>,
    location: L,
) -> Result<DetachedWaitIndirectDmaOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.wait_indirect_dma", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    operands.push(source);
    operands.push(destination);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedWaitIndirectDmaOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.region` that contains a Mosaic TPU region.
pub trait RegionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation results produced by the region.
    fn result_values(&self) -> Result<Vec<OperationResultRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| self.result(index)).collect()
    }
}

mlir_op!(Region);
mlir_op_trait!(Region, ZeroOperands);
mlir_op_trait!(Region, OneRegion);
mlir_op_trait!(Region, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.region`.
pub fn region<'c, 't: 'c, L: Location<'c, 't>>(
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedRegionOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.region", location);
    builder = builder.add_results(result_types);
    builder = builder.add_region(region);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedRegionOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `message` value.
pub const MESSAGE_ATTRIBUTE: &str = "message";

/// Name of the [`Attribute`] that stores the `level` value.
pub const LEVEL_ATTRIBUTE: &str = "level";

/// Mosaic TPU [`Operation`] for `tpu.trace` that contains a traced Mosaic TPU region.
pub trait TraceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `message` attribute.
    fn message(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(MESSAGE_ATTRIBUTE)
    }

    /// Returns the `level` attribute.
    fn level(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(LEVEL_ATTRIBUTE)
    }

    /// Returns the operation results produced by the trace region.
    fn result_values(&self) -> Result<Vec<OperationResultRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| self.result(index)).collect()
    }
}

mlir_op!(Trace);
mlir_op_trait!(Trace, ZeroOperands);
mlir_op_trait!(Trace, OneRegion);
mlir_op_trait!(Trace, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace`.
pub fn trace<'c, 't: 'c, L: Location<'c, 't>>(
    message: StringAttributeRef<'c, 't>,
    level: IntegerAttributeRef<'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedTraceOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.trace", location);
    builder = builder.add_attribute(MESSAGE_ATTRIBUTE, message);
    builder = builder.add_attribute(LEVEL_ATTRIBUTE, level);
    builder = builder.add_results(result_types);
    builder = builder.add_region(region);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTraceOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.trace_start` that starts a Mosaic TPU trace section.
pub trait TraceStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `message` attribute.
    fn message(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(MESSAGE_ATTRIBUTE)
    }

    /// Returns the `level` attribute.
    fn level(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(LEVEL_ATTRIBUTE)
    }
}

mlir_op!(TraceStart);
mlir_op_trait!(TraceStart, ZeroOperands);
mlir_op_trait!(TraceStart, ZeroRegions);
mlir_op_trait!(TraceStart, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_start`.
pub fn trace_start<'c, 't: 'c, L: Location<'c, 't>>(
    message: StringAttributeRef<'c, 't>,
    level: IntegerAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedTraceStartOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.trace_start", location);
    builder = builder.add_attribute(MESSAGE_ATTRIBUTE, message);
    builder = builder.add_attribute(LEVEL_ATTRIBUTE, level);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTraceStartOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.trace_stop` that stops a Mosaic TPU trace section.
pub trait TraceStopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(TraceStop);
mlir_op_trait!(TraceStop, ZeroOperands);
mlir_op_trait!(TraceStop, ZeroRegions);
mlir_op_trait!(TraceStop, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_stop`.
pub fn trace_stop<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> Result<DetachedTraceStopOperation<'c, 't>, Error> {
    let builder = OperationBuilder::new("tpu.trace_stop", location);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTraceStopOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `label` value.
pub const LABEL_ATTRIBUTE: &str = "label";

/// Mosaic TPU [`Operation`] for `tpu.trace_value` that emits a scalar trace value.
pub trait TraceValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `label` attribute.
    fn label(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(LABEL_ATTRIBUTE)
    }
}

mlir_op!(TraceValue);
mlir_op_trait!(TraceValue, ZeroRegions);
mlir_op_trait!(TraceValue, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_value`.
pub fn trace_value<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    label: StringAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedTraceValueOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.trace_value", location);
    let mut operands = Vec::new();
    operands.push(value);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(LABEL_ATTRIBUTE, label);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTraceValueOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.yield` that terminates a Mosaic TPU region.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `results` operands.
    fn results(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.yield`.
pub fn r#yield<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    results: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.yield", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(results);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedYieldOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.delay` that delays TPU execution.
pub trait DelayOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `nanos` operand.
    fn nanoseconds(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(Delay);
mlir_op_trait!(Delay, ZeroRegions);
mlir_op_trait!(Delay, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.delay`.
pub fn delay<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    nanoseconds: ValueRef<'o, 'c, 't>,
    location: L,
) -> Result<DetachedDelayOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.delay", location);
    let mut operands = Vec::new();
    operands.push(nanoseconds);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedDelayOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.mask_cast` that casts a TPU mask register to a different packing.
pub trait MaskCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MaskCast);
mlir_op_trait!(MaskCast, ZeroRegions);
mlir_op_trait!(MaskCast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.mask_cast`.
pub fn mask_cast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMaskCastOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.mask_cast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedMaskCastOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.iteration_bound` that returns a TPU iteration bound.
pub trait GetIterationBoundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `dim` attribute.
    fn dim(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIM_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(GetIterationBound);
mlir_op_trait!(GetIterationBound, ZeroOperands);
mlir_op_trait!(GetIterationBound, ZeroRegions);
mlir_op_trait!(GetIterationBound, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.iteration_bound`.
pub fn get_iteration_bound<'c, 't: 'c, L: Location<'c, 't>>(
    dim: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetIterationBoundOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.iteration_bound", location);
    builder = builder.add_attribute(DIM_ATTRIBUTE, dim);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedGetIterationBoundOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.internal_scratch` that returns internal TPU scratch memory.
pub trait GetInternalScratchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(GetInternalScratch);
mlir_op_trait!(GetInternalScratch, ZeroOperands);
mlir_op_trait!(GetInternalScratch, ZeroRegions);
mlir_op_trait!(GetInternalScratch, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.internal_scratch`.
pub fn get_internal_scratch<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedGetInternalScratchOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.internal_scratch", location);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedGetInternalScratchOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.prng_set_seed_32` that sets the TPU 32-bit PRNG seed.
pub trait PrngSeed32Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `seeds` operands.
    fn seeds(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }
}

mlir_op!(PrngSeed32);
mlir_op_trait!(PrngSeed32, ZeroRegions);
mlir_op_trait!(PrngSeed32, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.prng_set_seed_32`.
pub fn prng_set_seed_32<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    seeds: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> Result<DetachedPrngSeed32Operation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.prng_set_seed_32", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(seeds);
    builder = builder.add_operands(&operands);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedPrngSeed32Operation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.prng_random_bits` that returns TPU PRNG random bits.
pub trait PrngRandomBitsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `output` result.
    fn output(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }
}

mlir_op!(PrngRandomBits);
mlir_op_trait!(PrngRandomBits, ZeroOperands);
mlir_op_trait!(PrngRandomBits, ZeroRegions);
mlir_op_trait!(PrngRandomBits, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.prng_random_bits`.
pub fn prng_random_bits<'c, 't: 'c, L: Location<'c, 't>>(
    output_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedPrngRandomBitsOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.prng_random_bits", location);
    builder = builder.add_result(output_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedPrngRandomBitsOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `pattern` value.
pub const PATTERN_ATTRIBUTE: &str = "pattern";

/// Mosaic TPU [`Operation`] for `tpu.sublane_shuffle` that shuffles two TPU vector registers by sublane.
pub trait SublaneShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `pattern` attribute.
    fn pattern(&self) -> Result<DenseInteger32ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_32_array_attribute(PATTERN_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(SublaneShuffle);
mlir_op_trait!(SublaneShuffle, ZeroRegions);
mlir_op_trait!(SublaneShuffle, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sublane_shuffle`.
pub fn sublane_shuffle<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    rhs: ValueRef<'o, 'c, 't>,
    pattern: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSublaneShuffleOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.sublane_shuffle", location);
    let mut operands = Vec::new();
    operands.push(lhs);
    operands.push(rhs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(PATTERN_ATTRIBUTE, pattern);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedSublaneShuffleOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `permutation` value.
pub const PERMUTATION_ATTRIBUTE: &str = "permutation";

/// Mosaic TPU [`Operation`] for `tpu.transpose` that transposes a vector.
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `vector` operand.
    fn vector(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `permutation` attribute.
    fn permutation(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(PERMUTATION_ATTRIBUTE)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, ZeroRegions);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.transpose`.
pub fn transpose<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    vector: ValueRef<'o, 'c, 't>,
    permutation: DenseInteger64ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedTransposeOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.transpose", location);
    let mut operands = Vec::new();
    operands.push(vector);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(PERMUTATION_ATTRIBUTE, permutation);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedTransposeOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `tag` value.
pub const TAG_ATTRIBUTE: &str = "tag";

/// Name of the [`Attribute`] that stores the `formatted` value.
pub const FORMATTED_ATTRIBUTE: &str = "formatted";

/// Mosaic TPU [`Operation`] for `tpu.log` that logs scalar values from TPU execution.
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `inputs` operands.
    fn inputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index)).collect()
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(TAG_ATTRIBUTE)
    }

    /// Returns the `formatted` attribute.
    fn formatted(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        Ok(({
            let attribute_name = FORMATTED_ATTRIBUTE;
            self.attribute(attribute_name)?
                .map(|attribute| {
                    attribute.cast().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .unwrap_or_else(|| self.context().boolean_attribute(false)))
    }
}

mlir_op!(Log);
mlir_op_trait!(Log, ZeroRegions);
mlir_op_trait!(Log, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.log`.
pub fn log<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'o, 'c, 't>],
    tag: StringAttributeRef<'c, 't>,
    formatted: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> Result<DetachedLogOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.log", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(inputs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(TAG_ATTRIBUTE, tag);
    if let Some(formatted) = formatted {
        builder = builder.add_attribute(FORMATTED_ATTRIBUTE, formatted);
    }
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedLogOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Name of the [`Attribute`] that stores the `shape` value.
pub const SHAPE_ATTRIBUTE: &str = "shape";

/// Mosaic TPU [`Operation`] for `tpu.log_buffer` that logs a memory buffer from TPU execution.
pub trait LogBufferOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `shape` attribute.
    fn shape(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(SHAPE_ATTRIBUTE)
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(TAG_ATTRIBUTE)
    }
}

mlir_op!(LogBuffer);
mlir_op_trait!(LogBuffer, ZeroRegions);
mlir_op_trait!(LogBuffer, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.log_buffer`.
pub fn log_buffer<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    shape: DenseInteger64ArrayAttributeRef<'c, 't>,
    tag: StringAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedLogBufferOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.log_buffer", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SHAPE_ATTRIBUTE, shape);
    builder = builder.add_attribute(TAG_ATTRIBUTE, tag);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedLogBufferOperation>() }.ok_or_else(|| Error::internal("invalid operation cast"))
}

/// Mosaic TPU [`Operation`] for `tpu.fetch_and_add_sync` that synchronously fetches and increments SMEM.
pub trait FetchAndAddSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let count = self.operand_count().saturating_sub(3);
        (0..count).map(|index| self.operand_value(1 + index)).collect()
    }

    /// Returns the `value` operand.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let count = self.operand_count().saturating_sub(3);
        self.operand_value(1 + count + 0)
    }

    /// Returns the `core_id` operand.
    fn core_id(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let count = self.operand_count().saturating_sub(3);
        self.operand_value(1 + count + 1)
    }

    /// Returns the `result` result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(FetchAndAddSync);
mlir_op_trait!(FetchAndAddSync, ZeroRegions);
mlir_op_trait!(FetchAndAddSync, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fetch_and_add_sync`.
pub fn fetch_and_add_sync<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    value: ValueRef<'o, 'c, 't>,
    core_id: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedFetchAndAddSyncOperation<'c, 't>, Error> {
    let mut builder = OperationBuilder::new("tpu.fetch_and_add_sync", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    operands.push(value);
    operands.push(core_id);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build()?;
    unsafe { operation.cast::<DetachedFetchAndAddSyncOperation>() }
        .ok_or_else(|| Error::internal("invalid operation cast"))
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::mosaic::tpu::attributes::{ContractPrecision, PackFormat, ReductionKind, RoundingMode};
    use crate::{Block, Context, DialectHandle, OneRegion, Operation, Region, Type, TypeRef, Value};

    use super::*;

    struct TestTypes<'c, 't> {
        i1: TypeRef<'c, 't>,
        i32: TypeRef<'c, 't>,
    }

    impl<'c, 't> TestTypes<'c, 't> {
        fn new(context: &'c Context<'t>) -> Self {
            let i1_type = context.signless_integer_type(1);
            let i32_type = context.signless_integer_type(32);

            Self { i1: i1_type.as_ref(), i32: i32_type.as_ref() }
        }
    }

    struct TestAttributes<'c, 't> {
        zero: IntegerAttributeRef<'c, 't>,
        one: IntegerAttributeRef<'c, 't>,
        two: IntegerAttributeRef<'c, 't>,
        boolean: BooleanAttributeRef<'c, 't>,
        false_boolean: BooleanAttributeRef<'c, 't>,
        dense_bool: DenseBooleanArrayAttributeRef<'c, 't>,
        dense_i32: DenseInteger32ArrayAttributeRef<'c, 't>,
        dense_i64: DenseInteger64ArrayAttributeRef<'c, 't>,
        kind: ReductionKindAttributeRef<'c, 't>,
        pack_format: PackFormatAttributeRef<'c, 't>,
        rounding_mode: RoundingModeAttributeRef<'c, 't>,
        precision: ContractPrecisionAttributeRef<'c, 't>,
        dimension_numbers: DotDimensionNumbersAttributeRef<'c, 't>,
        string: StringAttributeRef<'c, 't>,
        type_attribute: TypeAttributeRef<'c, 't>,
    }

    impl<'c, 't> TestAttributes<'c, 't> {
        fn new(context: &'c Context<'t>, types: &TestTypes<'c, 't>) -> Self {
            Self {
                zero: context.integer_attribute(context.signless_integer_type(32), 0),
                one: context.integer_attribute(context.signless_integer_type(32), 1),
                two: context.integer_attribute(context.signless_integer_type(32), 2),
                boolean: context.boolean_attribute(true),
                false_boolean: context.boolean_attribute(false),
                dense_bool: context.dense_bool_array_attribute(&[true, false]).unwrap(),
                dense_i32: context.dense_i32_array_attribute(&[0, 1]).unwrap(),
                dense_i64: context.dense_i64_array_attribute(&[0, 1]).unwrap(),
                kind: context.mosaic_tpu_reduction_kind_attribute(ReductionKind::Sum).unwrap(),
                pack_format: context.mosaic_tpu_pack_format_attribute(PackFormat::Compressed).unwrap(),
                rounding_mode: context.mosaic_tpu_rounding_mode_attribute(RoundingMode::ToNearestEven).unwrap(),
                precision: context.mosaic_tpu_contract_precision_attribute(ContractPrecision::BFloat16).unwrap(),
                dimension_numbers: context
                    .mosaic_tpu_dot_dimension_numbers_attribute(&[1], &[0], &[0], &[1], &[0, 1], &[], &[])
                    .unwrap(),
                string: context.string_attribute("message"),
                type_attribute: context.type_attribute(types.i32),
            }
        }
    }

    macro_rules! mosaic_tpu_operation_test {
        ($test_name:ident, |$context:ident, $location:ident, $values:ident, $types:ident, $attributes:ident| $body:block $(,)?) => {
            #[test]
            fn $test_name() {
                let $context = Context::new();
                $context.load_dialect(DialectHandle::mosaic_tpu().unwrap()).unwrap();
                let $location = $context.unknown_location();
                let $types = TestTypes::new(&$context);
                let $attributes = TestAttributes::new(&$context, &$types);
                let block = $context.block(&[
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                ]);
                let $values = (0..8).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();

                $body
            }
        };
    }

    mosaic_tpu_operation_test!(test_all_reduce_operation, |_context, location, values, types, attributes| {
        let operation = all_reduce(values[0], attributes.zero, attributes.kind, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.all_reduce"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.dim().unwrap(), attributes.zero);
        assert_eq!(operation.kind().unwrap().value().unwrap(), ReductionKind::Sum);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_reduce_index_operation, |_context, location, values, types, attributes| {
        let operation = reduce_index(values[0], attributes.one, attributes.kind, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.reduce_index"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.axis().unwrap(), attributes.one);
        assert_eq!(operation.kind().unwrap().value().unwrap(), ReductionKind::Sum);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_scan_operation, |_context, location, values, types, attributes| {
        let operation = scan(values[0], Some(values[1]), attributes.kind, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.scan"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.mask().unwrap(), Some(values[1]));
        assert_eq!(operation.kind().unwrap().value().unwrap(), ReductionKind::Sum);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_sort_operation, |_context, location, values, types, attributes| {
        let operation = sort(
            values[0],
            values[1],
            Some(values[2]),
            Some(attributes.boolean),
            types.i1,
            types.i32,
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sort"));
        assert_eq!(operation.keys().unwrap(), values[0]);
        assert_eq!(operation.values().unwrap(), values[1]);
        assert_eq!(operation.mask().unwrap(), Some(values[2]));
        assert!(operation.descending().unwrap().value());
        assert_eq!(operation.output_mask().unwrap(), operation.as_ref().result(0).unwrap());
        assert_eq!(operation.sorted_keys().unwrap(), operation.result(1).unwrap());
        assert_eq!(operation.sorted_values().unwrap(), operation.result(2).unwrap());
    });

    mosaic_tpu_operation_test!(test_store_operation, |_context, location, values, _types, attributes| {
        let operation = store(
            values[0],
            values[1],
            &[values[2], values[3]],
            Some(values[4]),
            attributes.dense_bool,
            Some(attributes.two),
            Some(attributes.boolean),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.store"));
        assert_eq!(operation.value_to_store().unwrap(), values[0]);
        assert_eq!(operation.base().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.mask().unwrap(), Some(values[4]));
        assert_eq!(operation.sublane_mask().unwrap().values().collect::<Vec<_>>(), vec![true, false]);
        assert_eq!(operation.sublane_stride().unwrap(), attributes.two);
        assert!(operation.add().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_load_operation, |_context, location, values, types, attributes| {
        let operation =
            load(values[0], &[values[1], values[2]], attributes.dense_bool, Some(attributes.two), types.i32, location)
                .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.load"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.sublane_mask().unwrap().values().collect::<Vec<_>>(), vec![true, false]);
        assert_eq!(operation.sublane_stride().unwrap(), attributes.two);
    });

    mosaic_tpu_operation_test!(test_vector_store_operation, |_context, location, values, _types, attributes| {
        let operation = vector_store(
            values[0],
            values[1],
            &[values[2], values[3]],
            Some(values[4]),
            attributes.dense_i32,
            Some(attributes.boolean),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.vector_store"));
        assert_eq!(operation.value_to_store().unwrap(), values[0]);
        assert_eq!(operation.base().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.mask().unwrap(), Some(values[4]));
        assert_eq!(operation.strides().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert!(operation.add().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_vector_load_operation, |_context, location, values, types, attributes| {
        let operation =
            vector_load(values[0], &[values[1], values[2]], Some(values[3]), attributes.dense_i32, types.i32, location)
                .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.vector_load"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.mask().unwrap(), Some(values[3]));
        assert_eq!(operation.strides().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_strided_load_operation, |_context, location, values, types, attributes| {
        let operation =
            strided_load(values[0], &[values[1], values[2]], attributes.dense_i32, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.strided_load"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.strides().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_strided_store_operation, |_context, location, values, _types, attributes| {
        let operation =
            strided_store(values[0], values[1], &[values[2], values[3]], attributes.dense_i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.strided_store"));
        assert_eq!(operation.value_to_store().unwrap(), values[0]);
        assert_eq!(operation.base().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.strides().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_shuffled_load_operation, |_context, location, values, types, attributes| {
        let operation = shuffled_load(
            values[0],
            &[values[1], values[2]],
            attributes.dense_bool,
            attributes.dense_i32,
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.shuffled_load"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.sublane_mask().unwrap().values().collect::<Vec<_>>(), vec![true, false]);
        assert_eq!(operation.sublane_offsets().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_shuffled_store_operation, |_context, location, values, _types, attributes| {
        let operation = shuffled_store(
            values[0],
            values[1],
            &[values[2], values[3]],
            attributes.dense_bool,
            attributes.dense_i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.shuffled_store"));
        assert_eq!(operation.value_to_store().unwrap(), values[0]);
        assert_eq!(operation.base().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.sublane_mask().unwrap().values().collect::<Vec<_>>(), vec![true, false]);
        assert_eq!(operation.sublane_offsets().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_vector_load_idx_operation, |_context, location, values, types, _attributes| {
        let operation =
            vector_load_idx(values[0], &[values[1], values[2]], Some(values[3]), types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.vector_load_idx"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.mask().unwrap(), Some(values[3]));
        assert_eq!(operation.value().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_vector_store_idx_operation, |_context, location, values, _types, attributes| {
        let operation = vector_store_idx(
            values[0],
            values[1],
            &[values[2], values[3]],
            Some(values[4]),
            Some(attributes.boolean),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.vector_store_idx"));
        assert_eq!(operation.value_to_store().unwrap(), values[0]);
        assert_eq!(operation.base().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.mask().unwrap(), Some(values[4]));
        assert!(operation.add().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_rotate_operation, |_context, location, values, types, attributes| {
        let operation = rotate(
            values[0],
            attributes.one,
            attributes.two,
            Some(attributes.one),
            Some(attributes.zero),
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.rotate"));
        assert_eq!(operation.value().unwrap(), values[0]);
        assert_eq!(operation.amount().unwrap(), attributes.one);
        assert_eq!(operation.dimension().unwrap(), attributes.two);
        assert_eq!(operation.stride().unwrap(), Some(attributes.one));
        assert_eq!(operation.stride_dimension().unwrap(), Some(attributes.zero));
    });

    mosaic_tpu_operation_test!(test_dynamic_rotate_operation, |_context, location, values, types, attributes| {
        let operation = dynamic_rotate(
            values[0],
            values[1],
            attributes.two,
            Some(attributes.one),
            Some(attributes.zero),
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.dynamic_rotate"));
        assert_eq!(operation.value().unwrap(), values[0]);
        assert_eq!(operation.amount().unwrap(), values[1]);
        assert_eq!(operation.dimension().unwrap(), attributes.two);
        assert_eq!(operation.stride().unwrap(), Some(attributes.one));
        assert_eq!(operation.stride_dimension().unwrap(), Some(attributes.zero));
    });

    mosaic_tpu_operation_test!(test_scan_count_operation, |_context, location, values, types, _attributes| {
        let operation = scan_count(values[0], values[1], types.i1, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.scan_count"));
        assert_eq!(operation.in_mask().unwrap(), values[0]);
        assert_eq!(operation.values().unwrap(), values[1]);
        assert_eq!(operation.out_mask().unwrap(), operation.as_ref().result(0).unwrap());
        assert_eq!(operation.counts().unwrap(), operation.result(1).unwrap());
    });

    mosaic_tpu_operation_test!(test_iota_operation, |_context, location, _values, types, attributes| {
        let operation = iota(attributes.dense_i32, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.iota"));
        assert_eq!(operation.dimensions().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_reshape_operation, |_context, location, values, types, _attributes| {
        let operation = reshape(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.reshape"));
        assert_eq!(operation.source().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_repeat_operation, |_context, location, values, types, attributes| {
        let operation = repeat(values[0], attributes.one, attributes.two, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.repeat"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.dimension().unwrap(), attributes.one);
        assert_eq!(operation.times().unwrap(), attributes.two);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(
        test_broadcast_in_sublanes_operation,
        |_context, location, values, types, attributes| {
            let operation = broadcast_in_sublanes(values[0], attributes.one, types.i32, location).unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.broadcast_in_sublanes"));
            assert_eq!(operation.source().unwrap(), values[0]);
            assert_eq!(operation.lane().unwrap(), attributes.one);
            assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
        }
    );

    mosaic_tpu_operation_test!(test_unpack_subelements_operation, |_context, location, values, types, attributes| {
        let operation = unpack_subelements(
            values[0],
            attributes.one,
            attributes.pack_format,
            Some(attributes.boolean),
            Some(attributes.false_boolean),
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.unpack_subelements"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.index().unwrap(), attributes.one);
        assert_eq!(operation.pack_format().unwrap().value().unwrap(), PackFormat::Compressed);
        assert!(operation.integer_extended().unwrap().value());
        assert!(!operation.unsigned_integers().unwrap().value());
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_pack_subelements_operation, |_context, location, values, types, attributes| {
        let operation = pack_subelements(
            &[values[0], values[1]],
            attributes.dense_i32,
            attributes.pack_format,
            Some(attributes.boolean),
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.pack_subelements"));
        assert_eq!(operation.sources().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.positions().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.pack_format().unwrap().value().unwrap(), PackFormat::Compressed);
        assert_eq!(operation.unsigned_integers().unwrap().map(|attribute| attribute.value()), Some(true));
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_pack_elementwise_operation, |_context, location, values, types, attributes| {
        let operation =
            pack_elementwise(&[values[0], values[1]], attributes.type_attribute, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.pack_elementwise"));
        assert_eq!(operation.sources().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.sources().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_unpack_elementwise_operation, |_context, location, values, types, attributes| {
        let operation =
            unpack_elementwise(values[0], attributes.type_attribute, attributes.one, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.unpack_elementwise"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.index().unwrap(), attributes.one);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_relayout_operation, |_context, location, values, types, _attributes| {
        let operation = relayout(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.relayout"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_pack_mask_operation, |_context, location, values, types, attributes| {
        let operation = pack_mask(&[values[0], values[1]], attributes.dense_i32, types.i1, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.pack_vmsk"));
        assert_eq!(operation.sources().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.positions().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_gather_operation, |_context, location, values, types, attributes| {
        let operation = gather(values[0], attributes.dense_i32, attributes.one, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.gather"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.dimension().unwrap(), attributes.one);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_dynamic_gather_operation, |_context, location, values, types, attributes| {
        let operation = dynamic_gather(values[0], values[1], attributes.dense_i32, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.dynamic_gather"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), values[1]);
        assert_eq!(operation.dimensions().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_fp_to_si_operation, |_context, location, values, types, attributes| {
        let operation = fp_to_si(values[0], attributes.rounding_mode, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.fptosi"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.rounding_mode().unwrap().value().unwrap(), RoundingMode::ToNearestEven);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_fp_to_ui_operation, |_context, location, values, types, attributes| {
        let operation = fp_to_ui(values[0], attributes.rounding_mode, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.fptoui"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.rounding_mode().unwrap().value().unwrap(), RoundingMode::ToNearestEven);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_si_to_fp_operation, |_context, location, values, types, attributes| {
        let operation = si_to_fp(values[0], attributes.rounding_mode, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sitofp"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.rounding_mode().unwrap().value().unwrap(), RoundingMode::ToNearestEven);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_ui_to_fp_operation, |_context, location, values, types, attributes| {
        let operation = ui_to_fp(values[0], attributes.rounding_mode, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.uitofp"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.rounding_mode().unwrap().value().unwrap(), RoundingMode::ToNearestEven);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_ext_f_operation, |_context, location, values, types, _attributes| {
        let operation = ext_f(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.extf"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.out().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_trunc_f_operation, |_context, location, values, types, attributes| {
        let operation = trunc_f(values[0], attributes.rounding_mode, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.truncf"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.rounding_mode().unwrap().value().unwrap(), RoundingMode::ToNearestEven);
        assert_eq!(operation.out().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_matmul_operation, |_context, location, values, types, attributes| {
        let operation = matmul(
            values[0],
            values[1],
            values[2],
            Some(attributes.boolean),
            Some(attributes.false_boolean),
            Some(attributes.precision),
            Some(attributes.dimension_numbers),
            types.i32,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.matmul"));
        assert_eq!(operation.lhs().unwrap(), values[0]);
        assert_eq!(operation.rhs().unwrap(), values[1]);
        assert_eq!(operation.acc().unwrap(), values[2]);
        assert!(operation.transpose_lhs().unwrap().value());
        assert!(!operation.transpose_rhs().unwrap().value());
        assert_eq!(
            operation.precision().unwrap().map(|attribute| attribute.value().unwrap()),
            Some(ContractPrecision::BFloat16)
        );
        assert!(operation.dimension_numbers().unwrap().is_some());
    });

    mosaic_tpu_operation_test!(test_matmul_push_rhs_operation, |_context, location, values, _types, attributes| {
        let operation =
            matmul_push_rhs(values[0], attributes.one, Some(attributes.two), Some(attributes.boolean), location)
                .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.matmul_push_rhs"));
        assert_eq!(operation.rhs().unwrap(), values[0]);
        assert_eq!(operation.mxu_index().unwrap(), attributes.one);
        assert_eq!(operation.staging_register().unwrap(), attributes.two);
        assert!(operation.transpose().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_matmul_acc_lhs_operation, |_context, location, values, _types, attributes| {
        let operation =
            matmul_acc_lhs(values[0], attributes.one, attributes.two, Some(attributes.zero), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.matmul_acc_lhs"));
        assert_eq!(operation.lhs().unwrap(), values[0]);
        assert_eq!(operation.acc().unwrap(), attributes.one);
        assert_eq!(operation.mxu_index().unwrap(), attributes.two);
        assert_eq!(operation.load_staged_rhs().unwrap(), Some(attributes.zero));
    });

    mosaic_tpu_operation_test!(test_matmul_pop_operation, |_context, location, _values, types, attributes| {
        let operation = matmul_pop(attributes.one, attributes.two, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.matmul_pop"));
        assert_eq!(operation.acc().unwrap(), attributes.one);
        assert_eq!(operation.mxu_index().unwrap(), attributes.two);
    });

    mosaic_tpu_operation_test!(test_concatenate_operation, |_context, location, values, types, attributes| {
        let operation = concatenate(&[values[0], values[1]], attributes.one, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.concatenate"));
        assert_eq!(operation.sources().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.dimension().unwrap(), attributes.one);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_bitcast_operation, |_context, location, values, types, _attributes| {
        let operation = bitcast(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.bitcast"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_bitcast_vreg_operation, |_context, location, values, types, _attributes| {
        let operation = bitcast_vreg(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.bitcast_vreg"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_weird_operation, |_context, location, values, types, _attributes| {
        let operation = weird(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.weird"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_reciprocal_operation, |_context, location, values, types, attributes| {
        let operation =
            reciprocal(values[0], Some(attributes.boolean), Some(attributes.false_boolean), types.i32, location)
                .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.reciprocal"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert!(operation.approx().unwrap().value());
        assert!(!operation.full_range().unwrap().value());
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_stochastic_convert_operation, |_context, location, values, types, _attributes| {
        let operation = stochastic_convert(values[0], values[1], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.stochastic_convert"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.random().unwrap(), values[1]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(
        test_stochastic_convert_elementwise_operation,
        |_context, location, values, types, attributes| {
            let operation =
                stochastic_convert_elementwise(values[0], values[1], attributes.type_attribute, types.i32, location)
                    .unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.stochastic_convert_elementwise"));
            assert_eq!(operation.input().unwrap(), values[0]);
            assert_eq!(operation.random().unwrap(), values[1]);
            assert_eq!(operation.input().unwrap(), values[0]);
            assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
        },
    );

    mosaic_tpu_operation_test!(test_roll_vectors_operation, |_context, location, values, types, _attributes| {
        let operation = roll_vectors(&[values[0], values[1]], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.roll_vectors"));
        assert_eq!(operation.input().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_unroll_vectors_operation, |_context, location, values, types, _attributes| {
        let operation = unroll_vectors(values[0], &[types.i32, types.i32], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.unroll_vectors"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(
            operation.output().unwrap(),
            vec![operation.as_ref().result(0).unwrap(), operation.result(1).unwrap(),]
        );
    });

    mosaic_tpu_operation_test!(test_create_mask_operation, |_context, location, values, types, _attributes| {
        let operation = create_mask(&[values[0], values[1]], &[values[2], values[3]], types.i1, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.create_mask"));
        assert_eq!(operation.low().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.high().unwrap(), vec![values[2], values[3]]);
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(
        test_create_subelement_mask_operation,
        |_context, location, _values, types, attributes| {
            let operation = create_subelement_mask(attributes.zero, attributes.two, types.i1, location).unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.create_subelement_mask"));
            assert_eq!(operation.r#from().unwrap(), attributes.zero);
            assert_eq!(operation.to().unwrap(), attributes.two);
            assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
        }
    );

    mosaic_tpu_operation_test!(test_assume_multiple_operation, |_context, location, values, types, attributes| {
        let operation = assume_multiple(values[0], attributes.two, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.assume_multiple"));
        assert_eq!(operation.value().unwrap(), values[0]);
        assert_eq!(operation.multiple().unwrap(), attributes.two);
    });

    mosaic_tpu_operation_test!(test_mem_ref_slice_operation, |_context, location, values, types, _attributes| {
        let operation = mem_ref_slice(values[0], &[values[1], values[2]], &[values[3]], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.memref_slice"));
        assert_eq!(operation.mem_ref().unwrap(), values[0]);
        assert_eq!(operation.base_idx().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.dynamic_sizes().unwrap(), vec![values[3]]);
    });

    mosaic_tpu_operation_test!(test_mem_ref_squeeze_operation, |_context, location, values, types, _attributes| {
        let operation = mem_ref_squeeze(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.memref_squeeze"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_mem_ref_reshape_operation, |_context, location, values, types, _attributes| {
        let operation = mem_ref_reshape(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.memref_reshape"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_mem_ref_bitcast_operation, |_context, location, values, types, _attributes| {
        let operation = mem_ref_bitcast(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.memref_bitcast"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_reinterpret_cast_operation, |_context, location, values, types, _attributes| {
        let operation = reinterpret_cast(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.reinterpret_cast"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_assume_layout_operation, |_context, location, values, types, _attributes| {
        let operation = assume_layout(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.assume_layout"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_erase_layout_operation, |_context, location, values, types, _attributes| {
        let operation = erase_layout(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.erase_memref_layout"));
        assert_eq!(operation.operand_value(0).unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_device_id_operation, |_context, location, _values, types, _attributes| {
        let operation = device_id(types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.device_id"));
    });

    mosaic_tpu_operation_test!(test_semaphore_read_operation, |_context, location, values, types, _attributes| {
        let operation = semaphore_read(values[0], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sem_read"));
        assert_eq!(operation.semaphore().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_semaphore_wait_operation, |_context, location, values, _types, _attributes| {
        let operation = semaphore_wait(values[0], values[1], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sem_wait"));
        assert_eq!(operation.semaphore().unwrap(), values[0]);
        assert_eq!(operation.amount().unwrap(), values[1]);
    });

    mosaic_tpu_operation_test!(test_alloca_semaphore_operation, |_context, location, _values, types, _attributes| {
        let operation = alloca_semaphore(types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sem_alloc"));
    });

    mosaic_tpu_operation_test!(
        test_get_barrier_semaphore_operation,
        |_context, location, _values, types, _attributes| {
            let operation = get_barrier_semaphore(types.i32, location).unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.sem_barrier"));
            assert_eq!(operation.semaphore().unwrap(), operation.as_ref().result(0).unwrap());
        }
    );

    mosaic_tpu_operation_test!(test_semaphore_signal_operation, |_context, location, values, _types, _attributes| {
        let operation = semaphore_signal(values[0], values[1], Some(values[2]), Some(values[3]), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sem_signal"));
        assert_eq!(operation.semaphore().unwrap(), values[0]);
        assert_eq!(operation.amount().unwrap(), values[1]);
        assert_eq!(operation.device_id().unwrap(), Some(values[2]));
        assert_eq!(operation.core_id().unwrap(), Some(values[3]));
    });

    mosaic_tpu_operation_test!(test_barrier_operation, |_context, location, values, _types, _attributes| {
        let operation = barrier(values[0], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.barrier"));
        assert_eq!(operation.barrier_id().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_enqueue_dma_operation, |_context, location, values, _types, attributes| {
        let operation = enqueue_dma(
            values[0],
            Some(values[1]),
            values[2],
            values[3],
            Some(values[4]),
            Some(values[5]),
            Some(attributes.two),
            Some(attributes.boolean),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.enqueue_dma"));
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.source_semaphore().unwrap(), Some(values[1]));
        assert_eq!(operation.target().unwrap(), values[2]);
        assert_eq!(operation.target_semaphore().unwrap(), values[3]);
        assert_eq!(operation.device_id().unwrap(), Some(values[4]));
        assert_eq!(operation.core_id().unwrap(), Some(values[5]));
        assert_eq!(operation.priority().unwrap(), attributes.two);
        assert!(operation.strict_ordering().unwrap().value());
    });

    mosaic_tpu_operation_test!(
        test_enqueue_indirect_dma_operation,
        |_context, location, values, _types, attributes| {
            let operation = enqueue_indirect_dma(
                values[0],
                values[1],
                values[2],
                values[3],
                Some(values[4]),
                Some(attributes.boolean),
                location,
            )
            .unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.enqueue_indirect_dma"));
            assert_eq!(operation.source().unwrap(), values[0]);
            assert_eq!(operation.target().unwrap(), values[1]);
            assert_eq!(operation.offsets().unwrap(), values[2]);
            assert_eq!(operation.semaphore().unwrap(), values[3]);
            assert_eq!(operation.offset_filter().unwrap(), Some(values[4]));
            assert!(operation.add().unwrap().value());
        }
    );

    mosaic_tpu_operation_test!(test_wait_dma2_operation, |_context, location, values, _types, attributes| {
        let operation = wait_dma2(
            values[0],
            values[1],
            values[2],
            Some(values[3]),
            Some(values[4]),
            Some(attributes.boolean),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.wait_dma2"));
        assert_eq!(operation.semaphore().unwrap(), values[0]);
        assert_eq!(operation.source().unwrap(), values[1]);
        assert_eq!(operation.destination().unwrap(), values[2]);
        assert_eq!(operation.device_id().unwrap(), Some(values[3]));
        assert_eq!(operation.core_id().unwrap(), Some(values[4]));
        assert!(operation.strict_ordering().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_wait_indirect_dma_operation, |_context, location, values, _types, _attributes| {
        let operation = wait_indirect_dma(values[0], values[1], values[2], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.wait_indirect_dma"));
        assert_eq!(operation.semaphore().unwrap(), values[0]);
        assert_eq!(operation.source().unwrap(), values[1]);
        assert_eq!(operation.destination().unwrap(), values[2]);
    });

    mosaic_tpu_operation_test!(test_region_operation, |context, location, _values, types, _attributes| {
        let operation = region(&[types.i32], context.region(), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.region"));
        assert_eq!(operation.result_values().unwrap(), vec![operation.as_ref().result(0).unwrap()]);
        assert!(operation.body_region().unwrap().is_empty());
    });

    mosaic_tpu_operation_test!(test_trace_operation, |context, location, _values, types, attributes| {
        let operation = trace(attributes.string, attributes.one, &[types.i32], context.region(), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.trace"));
        assert_eq!(operation.message().unwrap().string().as_str(), Ok("message"));
        assert_eq!(operation.level().unwrap(), attributes.one);
        assert_eq!(operation.result_values().unwrap(), vec![operation.as_ref().result(0).unwrap()]);
        assert!(operation.body_region().unwrap().is_empty());
    });

    mosaic_tpu_operation_test!(test_trace_start_operation, |_context, location, _values, _types, attributes| {
        let operation = trace_start(attributes.string, attributes.one, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.trace_start"));
        assert_eq!(operation.message().unwrap().string().as_str(), Ok("message"));
        assert_eq!(operation.level().unwrap(), attributes.one);
    });

    mosaic_tpu_operation_test!(test_trace_stop_operation, |_context, location, _values, _types, _attributes| {
        let operation = trace_stop(location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.trace_stop"));
    });

    mosaic_tpu_operation_test!(test_trace_value_operation, |_context, location, values, _types, attributes| {
        let operation = trace_value(values[0], attributes.string, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.trace_value"));
        assert_eq!(operation.value().unwrap(), values[0]);
        assert_eq!(operation.label().unwrap().string().as_str(), Ok("message"));
    });

    mosaic_tpu_operation_test!(test_yield_operation, |_context, location, values, _types, _attributes| {
        let operation = r#yield(&[values[0], values[1]], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.yield"));
        assert_eq!(operation.operand_values().collect::<Result<Vec<_>, _>>().unwrap(), vec![values[0], values[1]]);
    });

    mosaic_tpu_operation_test!(test_delay_operation, |_context, location, values, _types, _attributes| {
        let operation = delay(values[0], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.delay"));
        assert_eq!(operation.nanoseconds().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_mask_cast_operation, |_context, location, values, types, _attributes| {
        let operation = mask_cast(values[0], types.i1, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.mask_cast"));
        assert_eq!(operation.input().unwrap(), values[0]);
    });

    mosaic_tpu_operation_test!(test_get_iteration_bound_operation, |_context, location, _values, types, attributes| {
        let operation = get_iteration_bound(attributes.one, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.iteration_bound"));
        assert_eq!(operation.dim().unwrap(), attributes.one);
    });

    mosaic_tpu_operation_test!(
        test_get_internal_scratch_operation,
        |_context, location, _values, types, _attributes| {
            let operation = get_internal_scratch(types.i32, location).unwrap();

            assert_eq!(operation.name().as_str(), Ok("tpu.internal_scratch"));
        }
    );

    mosaic_tpu_operation_test!(test_prng_seed32_operation, |_context, location, values, _types, _attributes| {
        let operation = prng_set_seed_32(&[values[0], values[1]], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.prng_set_seed_32"));
        assert_eq!(operation.seeds().unwrap(), vec![values[0], values[1]]);
    });

    mosaic_tpu_operation_test!(test_prng_random_bits_operation, |_context, location, _values, types, _attributes| {
        let operation = prng_random_bits(types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.prng_random_bits"));
        assert_eq!(operation.output().unwrap(), operation.as_ref().result(0).unwrap());
    });

    mosaic_tpu_operation_test!(test_sublane_shuffle_operation, |_context, location, values, types, attributes| {
        let operation = sublane_shuffle(values[0], values[1], attributes.dense_i32, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.sublane_shuffle"));
        assert_eq!(operation.lhs().unwrap(), values[0]);
        assert_eq!(operation.rhs().unwrap(), values[1]);
        assert_eq!(operation.pattern().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_transpose_operation, |_context, location, values, types, attributes| {
        let operation = transpose(values[0], attributes.dense_i64, types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.transpose"));
        assert_eq!(operation.vector().unwrap(), values[0]);
        assert_eq!(operation.permutation().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
    });

    mosaic_tpu_operation_test!(test_log_operation, |_context, location, values, _types, attributes| {
        let operation = log(&[values[0], values[1]], attributes.string, Some(attributes.boolean), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.log"));
        assert_eq!(operation.inputs().unwrap(), vec![values[0], values[1]]);
        assert_eq!(operation.tag().unwrap().string().as_str(), Ok("message"));
        assert!(operation.formatted().unwrap().value());
    });

    mosaic_tpu_operation_test!(test_log_buffer_operation, |_context, location, values, _types, attributes| {
        let operation = log_buffer(values[0], attributes.dense_i64, attributes.string, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.log_buffer"));
        assert_eq!(operation.input().unwrap(), values[0]);
        assert_eq!(operation.shape().unwrap().values().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(operation.tag().unwrap().string().as_str(), Ok("message"));
    });

    mosaic_tpu_operation_test!(test_fetch_and_add_sync_operation, |_context, location, values, types, _attributes| {
        let operation =
            fetch_and_add_sync(values[0], &[values[1], values[2]], values[3], values[4], types.i32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("tpu.fetch_and_add_sync"));
        assert_eq!(operation.base().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), vec![values[1], values[2]]);
        assert_eq!(operation.value().unwrap(), values[3]);
        assert_eq!(operation.core_id().unwrap(), values[4]);
    });
}
