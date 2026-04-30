use crate::{
    Attribute, BooleanAttributeRef, DenseBooleanArrayAttributeRef, DenseInteger32ArrayAttributeRef,
    DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, IntegerAttributeRef, Location, Operation,
    OperationBuilder, OperationResultRef, StringAttributeRef, TypeAttributeRef, TypeRef, ValueRef, mlir_op,
    mlir_op_trait,
};

use super::attributes::{
    ContractPrecisionAttributeRef, DotDimensionNumbersAttributeRef, PackFormatAttributeRef, ReductionKindAttributeRef,
    RoundingModeAttributeRef,
};

/// Name of the [`Attribute`] that stores Mosaic TPU operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the [`Attribute`] that stores the `dim` value.
pub const DIM_ATTRIBUTE: &str = "dim";

/// Name of the [`Attribute`] that stores the `kind` value.
pub const KIND_ATTRIBUTE: &str = "kind";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.all_reduce`.
pub fn all_reduce<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    dim: IntegerAttributeRef<'c, 't>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAllReduceOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.all_reduce", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIM_ATTRIBUTE, dim);
    builder = builder.add_attribute(KIND_ATTRIBUTE, kind);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedAllReduceOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.all_reduce` that reduces a vector across one dimension.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `dim` attribute.
    fn dim(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIM_ATTRIBUTE}` attribute in `tpu.all_reduce`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIM_ATTRIBUTE}` attribute in `tpu.all_reduce`"))
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> ReductionKindAttributeRef<'c, 't> {
        self.attribute(KIND_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{KIND_ATTRIBUTE}` attribute in `tpu.all_reduce`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{KIND_ATTRIBUTE}` attribute in `tpu.all_reduce`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, ZeroRegions);
mlir_op_trait!(AllReduce, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `axis` value.
pub const AXIS_ATTRIBUTE: &str = "axis";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reduce_index`.
pub fn reduce_index<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    axis: IntegerAttributeRef<'c, 't>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReduceIndexOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.reduce_index", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(AXIS_ATTRIBUTE, axis);
    builder = builder.add_attribute(KIND_ATTRIBUTE, kind);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedReduceIndexOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.reduce_index` that reduces vector indices across one dimension.
pub trait ReduceIndexOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `axis` attribute.
    fn axis(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AXIS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{AXIS_ATTRIBUTE}` attribute in `tpu.reduce_index`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{AXIS_ATTRIBUTE}` attribute in `tpu.reduce_index`"))
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> ReductionKindAttributeRef<'c, 't> {
        self.attribute(KIND_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{KIND_ATTRIBUTE}` attribute in `tpu.reduce_index`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{KIND_ATTRIBUTE}` attribute in `tpu.reduce_index`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ReduceIndex);
mlir_op_trait!(ReduceIndex, ZeroRegions);
mlir_op_trait!(ReduceIndex, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.scan`.
pub fn scan<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    mask: Option<ValueRef<'o, 'c, 't>>,
    kind: ReductionKindAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedScanOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedScanOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.scan` that computes a vector scan using a Mosaic TPU reduction kind.
pub trait ScanOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        if self.operand_count() > 1 { self.operand_value(1) } else { None }
    }

    /// Returns the `kind` attribute.
    fn kind(&self) -> ReductionKindAttributeRef<'c, 't> {
        self.attribute(KIND_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{KIND_ATTRIBUTE}` attribute in `tpu.scan`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{KIND_ATTRIBUTE}` attribute in `tpu.scan`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Scan);
mlir_op_trait!(Scan, ZeroRegions);
mlir_op_trait!(Scan, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `descending` value.
pub const DESCENDING_ATTRIBUTE: &str = "descending";

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
) -> DetachedSortOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSortOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sort` that sorts key/value vectors.
pub trait SortOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `keys` operand.
    fn keys(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `values` operand.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        if self.operand_count() > 2 { self.operand_value(2) } else { None }
    }

    /// Returns the `descending` attribute.
    fn descending(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(DESCENDING_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{DESCENDING_ATTRIBUTE}` attribute in `tpu.sort`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }

    /// Returns the `output_mask` result.
    fn output_mask(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }

    /// Returns the `sorted_keys` result.
    fn sorted_keys(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 1).unwrap()
    }

    /// Returns the `sorted_values` result.
    fn sorted_values(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 2).unwrap()
    }
}

mlir_op!(Sort);
mlir_op_trait!(Sort, ZeroRegions);
mlir_op_trait!(Sort, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `sublane_mask` value.
pub const SUBLANE_MASK_ATTRIBUTE: &str = "sublane_mask";

/// Name of the [`Attribute`] that stores the `sublane_stride` value.
pub const SUBLANE_STRIDE_ATTRIBUTE: &str = "sublane_stride";

/// Name of the [`Attribute`] that stores the `add` value.
pub const ADD_ATTRIBUTE: &str = "add";

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
) -> DetachedStoreOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    if let Some(sublane_stride) = sublane_stride {
        builder = builder.add_attribute(SUBLANE_STRIDE_ATTRIBUTE, sublane_stride);
    }
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedStoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.store` that stores a native TPU vector register into memory.
pub trait StoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> DenseBooleanArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_MASK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.store`"))
    }

    /// Returns the `sublane_stride` attribute.
    fn sublane_stride(&self) -> IntegerAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(SUBLANE_STRIDE_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{SUBLANE_STRIDE_ATTRIBUTE}` attribute in `tpu.store`"))
        } else {
            self.context().integer_attribute(self.context().signless_integer_type(32), 1)
        }
    }

    /// Returns the `add` attribute.
    fn add(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(ADD_ATTRIBUTE) {
            attribute.cast().unwrap_or_else(|| panic!("invalid `{ADD_ATTRIBUTE}` attribute in `tpu.store`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(Store);
mlir_op_trait!(Store, ZeroRegions);
mlir_op_trait!(Store, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.load`.
pub fn load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_stride: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedLoadOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedLoadOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.load` that loads a native TPU vector register from memory.
pub trait LoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index).unwrap()).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> DenseBooleanArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_MASK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.load`"))
    }

    /// Returns the `sublane_stride` attribute.
    fn sublane_stride(&self) -> IntegerAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(SUBLANE_STRIDE_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{SUBLANE_STRIDE_ATTRIBUTE}` attribute in `tpu.load`"))
        } else {
            self.context().integer_attribute(self.context().signless_integer_type(32), 1)
        }
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Load);
mlir_op_trait!(Load, ZeroRegions);
mlir_op_trait!(Load, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `strides` value.
pub const STRIDES_ATTRIBUTE: &str = "strides";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_store`.
pub fn vector_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorStoreOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedVectorStoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.vector_store` that stores a vector into memory.
pub trait VectorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(STRIDES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{STRIDES_ATTRIBUTE}` attribute in `tpu.vector_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{STRIDES_ATTRIBUTE}` attribute in `tpu.vector_store`"))
    }

    /// Returns the `add` attribute.
    fn add(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(ADD_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{ADD_ATTRIBUTE}` attribute in `tpu.vector_store`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(VectorStore);
mlir_op_trait!(VectorStore, ZeroRegions);
mlir_op_trait!(VectorStore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_load`.
pub fn vector_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedVectorLoadOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedVectorLoadOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.vector_load` that loads a vector from memory.
pub trait VectorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(1).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(2).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(STRIDES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{STRIDES_ATTRIBUTE}` attribute in `tpu.vector_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{STRIDES_ATTRIBUTE}` attribute in `tpu.vector_load`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(VectorLoad);
mlir_op_trait!(VectorLoad, ZeroRegions);
mlir_op_trait!(VectorLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.strided_load`.
pub fn strided_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedStridedLoadOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.strided_load", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedStridedLoadOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.strided_load` that loads a vector using explicit strides.
pub trait StridedLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index).unwrap()).collect()
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(STRIDES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{STRIDES_ATTRIBUTE}` attribute in `tpu.strided_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{STRIDES_ATTRIBUTE}` attribute in `tpu.strided_load`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(StridedLoad);
mlir_op_trait!(StridedLoad, ZeroRegions);
mlir_op_trait!(StridedLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.strided_store`.
pub fn strided_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    strides: DenseInteger32ArrayAttributeRef<'c, 't>,
    location: L,
) -> DetachedStridedStoreOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.strided_store", location);
    let mut operands = Vec::new();
    operands.push(value_to_store);
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(STRIDES_ATTRIBUTE, strides);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedStridedStoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.strided_store` that stores a vector using explicit strides.
pub trait StridedStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(2);
        (0..count).map(|index| self.operand_value(2 + index).unwrap()).collect()
    }

    /// Returns the `strides` attribute.
    fn strides(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(STRIDES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{STRIDES_ATTRIBUTE}` attribute in `tpu.strided_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{STRIDES_ATTRIBUTE}` attribute in `tpu.strided_store`"))
    }
}

mlir_op!(StridedStore);
mlir_op_trait!(StridedStore, ZeroRegions);
mlir_op_trait!(StridedStore, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `sublane_offsets` value.
pub const SUBLANE_OFFSETS_ATTRIBUTE: &str = "sublane_offsets";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.shuffled_load`.
pub fn shuffled_load<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_offsets: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedShuffledLoadOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.shuffled_load", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    builder = builder.add_attribute(SUBLANE_OFFSETS_ATTRIBUTE, sublane_offsets);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedShuffledLoadOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.shuffled_load` that loads a vector using sublane offsets.
pub trait ShuffledLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(1);
        (0..count).map(|index| self.operand_value(1 + index).unwrap()).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> DenseBooleanArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_MASK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.shuffled_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.shuffled_load`"))
    }

    /// Returns the `sublane_offsets` attribute.
    fn sublane_offsets(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_OFFSETS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_OFFSETS_ATTRIBUTE}` attribute in `tpu.shuffled_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_OFFSETS_ATTRIBUTE}` attribute in `tpu.shuffled_load`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ShuffledLoad);
mlir_op_trait!(ShuffledLoad, ZeroRegions);
mlir_op_trait!(ShuffledLoad, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.shuffled_store`.
pub fn shuffled_store<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    sublane_mask: DenseBooleanArrayAttributeRef<'c, 't>,
    sublane_offsets: DenseInteger32ArrayAttributeRef<'c, 't>,
    location: L,
) -> DetachedShuffledStoreOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.shuffled_store", location);
    let mut operands = Vec::new();
    operands.push(value_to_store);
    operands.push(base);
    operands.extend_from_slice(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SUBLANE_MASK_ATTRIBUTE, sublane_mask);
    builder = builder.add_attribute(SUBLANE_OFFSETS_ATTRIBUTE, sublane_offsets);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedShuffledStoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.shuffled_store` that stores a vector using sublane offsets.
pub trait ShuffledStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(2);
        (0..count).map(|index| self.operand_value(2 + index).unwrap()).collect()
    }

    /// Returns the `sublane_mask` attribute.
    fn sublane_mask(&self) -> DenseBooleanArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_MASK_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.shuffled_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_MASK_ATTRIBUTE}` attribute in `tpu.shuffled_store`"))
    }

    /// Returns the `sublane_offsets` attribute.
    fn sublane_offsets(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(SUBLANE_OFFSETS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SUBLANE_OFFSETS_ATTRIBUTE}` attribute in `tpu.shuffled_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SUBLANE_OFFSETS_ATTRIBUTE}` attribute in `tpu.shuffled_store`"))
    }
}

mlir_op!(ShuffledStore);
mlir_op_trait!(ShuffledStore, ZeroRegions);
mlir_op_trait!(ShuffledStore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_load_idx`.
pub fn vector_load_idx<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    value_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedVectorLoadIdxOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_result(value_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedVectorLoadIdxOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.vector_load_idx` that loads a vector using vector index operands.
pub trait VectorLoadIdxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(1).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(2).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `value` result.
    fn value(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(VectorLoadIdx);
mlir_op_trait!(VectorLoadIdx, ZeroRegions);
mlir_op_trait!(VectorLoadIdx, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.vector_store_idx`.
pub fn vector_store_idx<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'o, 'c, 't>,
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    mask: Option<ValueRef<'o, 'c, 't>>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedVectorStoreIdxOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(add) = add {
        builder = builder.add_attribute(ADD_ATTRIBUTE, add);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedVectorStoreIdxOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.vector_store_idx` that stores a vector using vector index operands.
pub trait VectorStoreIdxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value_to_store` operand.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the optional `mask` operand.
    fn mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `add` attribute.
    fn add(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(ADD_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{ADD_ATTRIBUTE}` attribute in `tpu.vector_store_idx`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(VectorStoreIdx);
mlir_op_trait!(VectorStoreIdx, ZeroRegions);
mlir_op_trait!(VectorStoreIdx, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `amount` value.
pub const AMOUNT_ATTRIBUTE: &str = "amount";

/// Name of the [`Attribute`] that stores the `dimension` value.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Name of the [`Attribute`] that stores the `stride` value.
pub const STRIDE_ATTRIBUTE: &str = "stride";

/// Name of the [`Attribute`] that stores the `stride_dimension` value.
pub const STRIDE_DIMENSION_ATTRIBUTE: &str = "stride_dimension";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.rotate`.
pub fn rotate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    amount: IntegerAttributeRef<'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    stride: Option<IntegerAttributeRef<'c, 't>>,
    stride_dimension: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedRotateOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedRotateOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.rotate` that rotates a vector by a static amount.
pub trait RotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `amount` attribute.
    fn amount(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(AMOUNT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{AMOUNT_ATTRIBUTE}` attribute in `tpu.rotate`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{AMOUNT_ATTRIBUTE}` attribute in `tpu.rotate`"))
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `tpu.rotate`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `tpu.rotate`"))
    }

    /// Returns the `stride` attribute.
    fn stride(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(STRIDE_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `stride_dimension` attribute.
    fn stride_dimension(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(STRIDE_DIMENSION_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Rotate);
mlir_op_trait!(Rotate, ZeroRegions);
mlir_op_trait!(Rotate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.dynamic_rotate`.
pub fn dynamic_rotate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    stride: Option<IntegerAttributeRef<'c, 't>>,
    stride_dimension: Option<IntegerAttributeRef<'c, 't>>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDynamicRotateOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedDynamicRotateOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.dynamic_rotate` that rotates a vector by a dynamic amount.
pub trait DynamicRotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `tpu.dynamic_rotate`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `tpu.dynamic_rotate`"))
    }

    /// Returns the `stride` attribute.
    fn stride(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(STRIDE_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `stride_dimension` attribute.
    fn stride_dimension(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(STRIDE_DIMENSION_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DynamicRotate);
mlir_op_trait!(DynamicRotate, ZeroRegions);
mlir_op_trait!(DynamicRotate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.scan_count`.
pub fn scan_count<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    in_mask: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    out_mask_type: TypeRef<'c, 't>,
    counts_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedScanCountOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.scan_count", location);
    let mut operands = Vec::new();
    operands.push(in_mask);
    operands.push(values);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(out_mask_type);
    builder = builder.add_result(counts_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedScanCountOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.scan_count` that counts duplicate occurrences in a vector scan.
pub trait ScanCountOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in_mask` operand.
    fn in_mask(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `values` operand.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `out_mask` result.
    fn out_mask(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }

    /// Returns the `counts` result.
    fn counts(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 1).unwrap()
    }
}

mlir_op!(ScanCount);
mlir_op_trait!(ScanCount, ZeroRegions);
mlir_op_trait!(ScanCount, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `dimensions` value.
pub const DIMENSIONS_ATTRIBUTE: &str = "dimensions";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.iota`.
pub fn iota<'c, 't: 'c, L: Location<'c, 't>>(
    dimensions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedIotaOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.iota", location);
    builder = builder.add_attribute(DIMENSIONS_ATTRIBUTE, dimensions);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedIotaOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.iota` that creates a vector iota.
pub trait IotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `dimensions` attribute.
    fn dimensions(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(DIMENSIONS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSIONS_ATTRIBUTE}` attribute in `tpu.iota`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSIONS_ATTRIBUTE}` attribute in `tpu.iota`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Iota);
mlir_op_trait!(Iota, ZeroOperands);
mlir_op_trait!(Iota, ZeroRegions);
mlir_op_trait!(Iota, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reshape`.
pub fn reshape<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReshapeOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.reshape", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedReshapeOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.reshape` that reshapes a TPU vector.
pub trait ReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Reshape);
mlir_op_trait!(Reshape, ZeroRegions);
mlir_op_trait!(Reshape, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `times` value.
pub const TIMES_ATTRIBUTE: &str = "times";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.repeat`.
pub fn repeat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    times: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedRepeatOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.repeat", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_attribute(TIMES_ATTRIBUTE, times);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedRepeatOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.repeat` that repeats values along a vector dimension.
pub trait RepeatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `tpu.repeat`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `tpu.repeat`"))
    }

    /// Returns the `times` attribute.
    fn times(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(TIMES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TIMES_ATTRIBUTE}` attribute in `tpu.repeat`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TIMES_ATTRIBUTE}` attribute in `tpu.repeat`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Repeat);
mlir_op_trait!(Repeat, ZeroRegions);
mlir_op_trait!(Repeat, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `lane` value.
pub const LANE_ATTRIBUTE: &str = "lane";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.broadcast_in_sublanes`.
pub fn broadcast_in_sublanes<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    lane: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBroadcastInSublanesOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.broadcast_in_sublanes", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(LANE_ATTRIBUTE, lane);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedBroadcastInSublanesOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.broadcast_in_sublanes` that broadcasts a lane value within each sublane.
pub trait BroadcastInSublanesOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `lane` attribute.
    fn lane(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(LANE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{LANE_ATTRIBUTE}` attribute in `tpu.broadcast_in_sublanes`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{LANE_ATTRIBUTE}` attribute in `tpu.broadcast_in_sublanes`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(BroadcastInSublanes);
mlir_op_trait!(BroadcastInSublanes, ZeroRegions);
mlir_op_trait!(BroadcastInSublanes, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `index` value.
pub const INDEX_ATTRIBUTE: &str = "index";

/// Name of the [`Attribute`] that stores the `pack_format` value.
pub const PACK_FORMAT_ATTRIBUTE: &str = "pack_format";

/// Name of the [`Attribute`] that stores the `integer_extended` value.
pub const INTEGER_EXTENDED_ATTRIBUTE: &str = "integer_extended";

/// Name of the [`Attribute`] that stores the `unsigned_integers` value.
pub const UNSIGNED_INTEGERS_ATTRIBUTE: &str = "unsigned_integers";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unpack_subelements`.
pub fn unpack_subelements<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    index: IntegerAttributeRef<'c, 't>,
    pack_format: PackFormatAttributeRef<'c, 't>,
    integer_extended: Option<BooleanAttributeRef<'c, 't>>,
    unsigned_integers: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedUnpackSubelementsOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedUnpackSubelementsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.unpack_subelements` that unpacks subelements from a packed vector.
pub trait UnpackSubelementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `index` attribute.
    fn index(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(INDEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{INDEX_ATTRIBUTE}` attribute in `tpu.unpack_subelements`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{INDEX_ATTRIBUTE}` attribute in `tpu.unpack_subelements`"))
    }

    /// Returns the `pack_format` attribute.
    fn pack_format(&self) -> PackFormatAttributeRef<'c, 't> {
        self.attribute(PACK_FORMAT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{PACK_FORMAT_ATTRIBUTE}` attribute in `tpu.unpack_subelements`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{PACK_FORMAT_ATTRIBUTE}` attribute in `tpu.unpack_subelements`"))
    }

    /// Returns the `integer_extended` attribute.
    fn integer_extended(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(INTEGER_EXTENDED_ATTRIBUTE) {
            attribute.cast().unwrap_or_else(|| {
                panic!("invalid `{INTEGER_EXTENDED_ATTRIBUTE}` attribute in `tpu.unpack_subelements`")
            })
        } else {
            self.context().boolean_attribute(true)
        }
    }

    /// Returns the `unsigned_integers` attribute.
    fn unsigned_integers(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(UNSIGNED_INTEGERS_ATTRIBUTE) {
            attribute.cast().unwrap_or_else(|| {
                panic!("invalid `{UNSIGNED_INTEGERS_ATTRIBUTE}` attribute in `tpu.unpack_subelements`")
            })
        } else {
            self.context().boolean_attribute(false)
        }
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(UnpackSubelements);
mlir_op_trait!(UnpackSubelements, ZeroRegions);
mlir_op_trait!(UnpackSubelements, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `positions` value.
pub const POSITIONS_ATTRIBUTE: &str = "positions";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_subelements`.
pub fn pack_subelements<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    positions: DenseInteger32ArrayAttributeRef<'c, 't>,
    pack_format: PackFormatAttributeRef<'c, 't>,
    unsigned_integers: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPackSubelementsOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedPackSubelementsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.pack_subelements` that packs subelements from multiple vector registers.
pub trait PackSubelementsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `positions` attribute.
    fn positions(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(POSITIONS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{POSITIONS_ATTRIBUTE}` attribute in `tpu.pack_subelements`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{POSITIONS_ATTRIBUTE}` attribute in `tpu.pack_subelements`"))
    }

    /// Returns the `pack_format` attribute.
    fn pack_format(&self) -> PackFormatAttributeRef<'c, 't> {
        self.attribute(PACK_FORMAT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{PACK_FORMAT_ATTRIBUTE}` attribute in `tpu.pack_subelements`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{PACK_FORMAT_ATTRIBUTE}` attribute in `tpu.pack_subelements`"))
    }

    /// Returns the `unsigned_integers` attribute.
    fn unsigned_integers(&self) -> Option<BooleanAttributeRef<'c, 't>> {
        self.attribute(UNSIGNED_INTEGERS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PackSubelements);
mlir_op_trait!(PackSubelements, ZeroRegions);
mlir_op_trait!(PackSubelements, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `target_type` value.
pub const TARGET_TYPE_ATTRIBUTE: &str = "target_type";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_elementwise`.
pub fn pack_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    target_type: TypeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPackElementwiseOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.pack_elementwise", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(TARGET_TYPE_ATTRIBUTE, target_type);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedPackElementwiseOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.pack_elementwise` that packs vectors elementwise.
pub trait PackElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `target_type` attribute.
    fn target_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(TARGET_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TARGET_TYPE_ATTRIBUTE}` attribute in `tpu.pack_elementwise`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TARGET_TYPE_ATTRIBUTE}` attribute in `tpu.pack_elementwise`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PackElementwise);
mlir_op_trait!(PackElementwise, ZeroRegions);
mlir_op_trait!(PackElementwise, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `source_type` value.
pub const SOURCE_TYPE_ATTRIBUTE: &str = "source_type";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unpack_elementwise`.
pub fn unpack_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    source_type: TypeAttributeRef<'c, 't>,
    index: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedUnpackElementwiseOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.unpack_elementwise", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SOURCE_TYPE_ATTRIBUTE, source_type);
    builder = builder.add_attribute(INDEX_ATTRIBUTE, index);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedUnpackElementwiseOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.unpack_elementwise` that unpacks a vector elementwise.
pub trait UnpackElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `source_type` attribute.
    fn source_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(SOURCE_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SOURCE_TYPE_ATTRIBUTE}` attribute in `tpu.unpack_elementwise`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SOURCE_TYPE_ATTRIBUTE}` attribute in `tpu.unpack_elementwise`"))
    }

    /// Returns the `index` attribute.
    fn index(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(INDEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{INDEX_ATTRIBUTE}` attribute in `tpu.unpack_elementwise`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{INDEX_ATTRIBUTE}` attribute in `tpu.unpack_elementwise`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(UnpackElementwise);
mlir_op_trait!(UnpackElementwise, ZeroRegions);
mlir_op_trait!(UnpackElementwise, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.relayout`.
pub fn relayout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedRelayoutOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.relayout", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedRelayoutOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.relayout` that changes a vector register layout.
pub trait RelayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Relayout);
mlir_op_trait!(Relayout, ZeroRegions);
mlir_op_trait!(Relayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.pack_vmsk`.
pub fn pack_mask<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    positions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPackMaskOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.pack_vmsk", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(POSITIONS_ATTRIBUTE, positions);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedPackMaskOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.pack_vmsk` that packs TPU vector masks.
pub trait PackMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `positions` attribute.
    fn positions(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(POSITIONS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{POSITIONS_ATTRIBUTE}` attribute in `tpu.pack_vmsk`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{POSITIONS_ATTRIBUTE}` attribute in `tpu.pack_vmsk`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PackMask);
mlir_op_trait!(PackMask, ZeroRegions);
mlir_op_trait!(PackMask, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `indices` value.
pub const INDICES_ATTRIBUTE: &str = "indices";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.gather`.
pub fn gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    indices: DenseInteger32ArrayAttributeRef<'c, 't>,
    dimension: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGatherOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.gather", location);
    let mut operands = Vec::new();
    operands.push(source);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(INDICES_ATTRIBUTE, indices);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedGatherOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.gather` that gathers values from a vector.
pub trait GatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` attribute.
    fn indices(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(INDICES_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{INDICES_ATTRIBUTE}` attribute in `tpu.gather`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{INDICES_ATTRIBUTE}` attribute in `tpu.gather`"))
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `tpu.gather`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `tpu.gather`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Gather);
mlir_op_trait!(Gather, ZeroRegions);
mlir_op_trait!(Gather, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.dynamic_gather`.
pub fn dynamic_gather<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    indices: ValueRef<'o, 'c, 't>,
    dimensions: DenseInteger32ArrayAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDynamicGatherOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.dynamic_gather", location);
    let mut operands = Vec::new();
    operands.push(source);
    operands.push(indices);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSIONS_ATTRIBUTE, dimensions);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedDynamicGatherOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.dynamic_gather` that gathers values using dynamic vector indices.
pub trait DynamicGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` operand.
    fn indices(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `dimensions` attribute.
    fn dimensions(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(DIMENSIONS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSIONS_ATTRIBUTE}` attribute in `tpu.dynamic_gather`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSIONS_ATTRIBUTE}` attribute in `tpu.dynamic_gather`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DynamicGather);
mlir_op_trait!(DynamicGather, ZeroRegions);
mlir_op_trait!(DynamicGather, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `rounding_mode` value.
pub const ROUNDING_MODE_ATTRIBUTE: &str = "rounding_mode";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fptosi`.
pub fn fp_to_si<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedFpToSiOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.fptosi", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedFpToSiOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.fptosi` that converts floating-point values to signed integers.
pub trait FpToSiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> RoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.fptosi`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.fptosi`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(FpToSi);
mlir_op_trait!(FpToSi, ZeroRegions);
mlir_op_trait!(FpToSi, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fptoui`.
pub fn fp_to_ui<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedFpToUiOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.fptoui", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedFpToUiOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.fptoui` that converts floating-point values to unsigned integers.
pub trait FpToUiOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> RoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.fptoui`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.fptoui`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(FpToUi);
mlir_op_trait!(FpToUi, ZeroRegions);
mlir_op_trait!(FpToUi, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sitofp`.
pub fn si_to_fp<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSiToFpOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sitofp", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSiToFpOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sitofp` that converts signed integer values to floating-point values.
pub trait SiToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> RoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.sitofp`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.sitofp`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(SiToFp);
mlir_op_trait!(SiToFp, ZeroRegions);
mlir_op_trait!(SiToFp, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.uitofp`.
pub fn ui_to_fp<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedUiToFpOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.uitofp", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedUiToFpOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.uitofp` that converts unsigned integer values to floating-point values.
pub trait UiToFpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> RoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.uitofp`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.uitofp`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(UiToFp);
mlir_op_trait!(UiToFp, ZeroRegions);
mlir_op_trait!(UiToFp, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.extf`.
pub fn ext_f<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    out_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedExtFOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.extf", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(out_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedExtFOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.extf` that extends floating-point values.
pub trait ExtFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `out` result.
    fn out(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ExtF);
mlir_op_trait!(ExtF, ZeroRegions);
mlir_op_trait!(ExtF, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.truncf`.
pub fn trunc_f<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    rounding_mode: RoundingModeAttributeRef<'c, 't>,
    out_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTruncFOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.truncf", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ROUNDING_MODE_ATTRIBUTE, rounding_mode);
    builder = builder.add_result(out_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTruncFOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.truncf` that truncates floating-point values.
pub trait TruncFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `in` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rounding_mode` attribute.
    fn rounding_mode(&self) -> RoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.truncf`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ROUNDING_MODE_ATTRIBUTE}` attribute in `tpu.truncf`"))
    }

    /// Returns the `out` result.
    fn out(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(TruncF);
mlir_op_trait!(TruncF, ZeroRegions);
mlir_op_trait!(TruncF, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `transpose_lhs` value.
pub const TRANSPOSE_LHS_ATTRIBUTE: &str = "transpose_lhs";

/// Name of the [`Attribute`] that stores the `transpose_rhs` value.
pub const TRANSPOSE_RHS_ATTRIBUTE: &str = "transpose_rhs";

/// Name of the [`Attribute`] that stores the `precision` value.
pub const PRECISION_ATTRIBUTE: &str = "precision";

/// Name of the [`Attribute`] that stores the `dimension_numbers` value.
pub const DIMENSION_NUMBERS_ATTRIBUTE: &str = "dimension_numbers";

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
) -> DetachedMatmulOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMatmulOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.matmul` that computes a TPU matrix multiplication.
pub trait MatmulOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `acc` operand.
    fn acc(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `transpose_lhs` attribute.
    fn transpose_lhs(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(TRANSPOSE_LHS_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{TRANSPOSE_LHS_ATTRIBUTE}` attribute in `tpu.matmul`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }

    /// Returns the `transpose_rhs` attribute.
    fn transpose_rhs(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(TRANSPOSE_RHS_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{TRANSPOSE_RHS_ATTRIBUTE}` attribute in `tpu.matmul`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }

    /// Returns the `precision` attribute.
    fn precision(&self) -> Option<ContractPrecisionAttributeRef<'c, 't>> {
        self.attribute(PRECISION_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `dimension_numbers` attribute.
    fn dimension_numbers(&self) -> Option<DotDimensionNumbersAttributeRef<'c, 't>> {
        self.attribute(DIMENSION_NUMBERS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Matmul);
mlir_op_trait!(Matmul, ZeroRegions);
mlir_op_trait!(Matmul, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `mxu_index` value.
pub const MXU_INDEX_ATTRIBUTE: &str = "mxu_index";

/// Name of the [`Attribute`] that stores the `staging_register` value.
pub const STAGING_REGISTER_ATTRIBUTE: &str = "staging_register";

/// Name of the [`Attribute`] that stores the `transpose` value.
pub const TRANSPOSE_ATTRIBUTE: &str = "transpose";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_push_rhs`.
pub fn matmul_push_rhs<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    rhs: ValueRef<'o, 'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    staging_register: Option<IntegerAttributeRef<'c, 't>>,
    transpose: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedMatmulPushRhsOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMatmulPushRhsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.matmul_push_rhs` that pushes a matrix-multiply RHS value.
pub trait MatmulPushRhsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(MXU_INDEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_push_rhs`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_push_rhs`"))
    }

    /// Returns the `staging_register` attribute.
    fn staging_register(&self) -> IntegerAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(STAGING_REGISTER_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{STAGING_REGISTER_ATTRIBUTE}` attribute in `tpu.matmul_push_rhs`"))
        } else {
            self.context().integer_attribute(self.context().signless_integer_type(32), 0)
        }
    }

    /// Returns the `transpose` attribute.
    fn transpose(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(TRANSPOSE_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{TRANSPOSE_ATTRIBUTE}` attribute in `tpu.matmul_push_rhs`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(MatmulPushRhs);
mlir_op_trait!(MatmulPushRhs, ZeroRegions);
mlir_op_trait!(MatmulPushRhs, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `acc` value.
pub const ACC_ATTRIBUTE: &str = "acc";

/// Name of the [`Attribute`] that stores the `load_staged_rhs` value.
pub const LOAD_STAGED_RHS_ATTRIBUTE: &str = "load_staged_rhs";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_acc_lhs`.
pub fn matmul_acc_lhs<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    acc: IntegerAttributeRef<'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    load_staged_rhs: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedMatmulAccLhsOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.matmul_acc_lhs", location);
    let mut operands = Vec::new();
    operands.push(lhs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(ACC_ATTRIBUTE, acc);
    builder = builder.add_attribute(MXU_INDEX_ATTRIBUTE, mxu_index);
    if let Some(load_staged_rhs) = load_staged_rhs {
        builder = builder.add_attribute(LOAD_STAGED_RHS_ATTRIBUTE, load_staged_rhs);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMatmulAccLhsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.matmul_acc_lhs` that accumulates a matrix-multiply LHS value.
pub trait MatmulAccLhsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `acc` attribute.
    fn acc(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(ACC_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ACC_ATTRIBUTE}` attribute in `tpu.matmul_acc_lhs`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ACC_ATTRIBUTE}` attribute in `tpu.matmul_acc_lhs`"))
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(MXU_INDEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_acc_lhs`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_acc_lhs`"))
    }

    /// Returns the `load_staged_rhs` attribute.
    fn load_staged_rhs(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(LOAD_STAGED_RHS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

mlir_op!(MatmulAccLhs);
mlir_op_trait!(MatmulAccLhs, ZeroRegions);
mlir_op_trait!(MatmulAccLhs, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.matmul_pop`.
pub fn matmul_pop<'c, 't: 'c, L: Location<'c, 't>>(
    acc: IntegerAttributeRef<'c, 't>,
    mxu_index: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMatmulPopOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.matmul_pop", location);
    builder = builder.add_attribute(ACC_ATTRIBUTE, acc);
    builder = builder.add_attribute(MXU_INDEX_ATTRIBUTE, mxu_index);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMatmulPopOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.matmul_pop` that pops a matrix-multiply accumulator value.
pub trait MatmulPopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `acc` attribute.
    fn acc(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(ACC_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ACC_ATTRIBUTE}` attribute in `tpu.matmul_pop`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ACC_ATTRIBUTE}` attribute in `tpu.matmul_pop`"))
    }

    /// Returns the `mxu_index` attribute.
    fn mxu_index(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(MXU_INDEX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_pop`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MXU_INDEX_ATTRIBUTE}` attribute in `tpu.matmul_pop`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MatmulPop);
mlir_op_trait!(MatmulPop, ZeroOperands);
mlir_op_trait!(MatmulPop, ZeroRegions);
mlir_op_trait!(MatmulPop, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.concatenate`.
pub fn concatenate<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    sources: &[ValueRef<'o, 'c, 't>],
    dimension: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedConcatenateOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.concatenate", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(sources);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DIMENSION_ATTRIBUTE, dimension);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedConcatenateOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.concatenate` that concatenates vector values.
pub trait ConcatenateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `sources` operands.
    fn sources(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `dimension` attribute.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `tpu.concatenate`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `tpu.concatenate`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Concatenate);
mlir_op_trait!(Concatenate, ZeroRegions);
mlir_op_trait!(Concatenate, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.bitcast`.
pub fn bitcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBitcastOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.bitcast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedBitcastOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.bitcast` that bitcasts a value.
pub trait BitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Bitcast);
mlir_op_trait!(Bitcast, ZeroRegions);
mlir_op_trait!(Bitcast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.bitcast_vreg`.
pub fn bitcast_vreg<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBitcastVregOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.bitcast_vreg", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedBitcastVregOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.bitcast_vreg` that bitcasts a native TPU vector register.
pub trait BitcastVregOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(BitcastVreg);
mlir_op_trait!(BitcastVreg, ZeroRegions);
mlir_op_trait!(BitcastVreg, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.weird`.
pub fn weird<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWeirdOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.weird", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedWeirdOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.weird` that computes the Mosaic TPU weird predicate operation.
pub trait WeirdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Weird);
mlir_op_trait!(Weird, ZeroRegions);
mlir_op_trait!(Weird, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `approx` value.
pub const APPROX_ATTRIBUTE: &str = "approx";

/// Name of the [`Attribute`] that stores the `full_range` value.
pub const FULL_RANGE_ATTRIBUTE: &str = "full_range";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reciprocal`.
pub fn reciprocal<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    approx: Option<BooleanAttributeRef<'c, 't>>,
    full_range: Option<BooleanAttributeRef<'c, 't>>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReciprocalOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedReciprocalOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.reciprocal` that computes reciprocal values.
pub trait ReciprocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `approx` attribute.
    fn approx(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(APPROX_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{APPROX_ATTRIBUTE}` attribute in `tpu.reciprocal`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }

    /// Returns the `full_range` attribute.
    fn full_range(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(FULL_RANGE_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{FULL_RANGE_ATTRIBUTE}` attribute in `tpu.reciprocal`"))
        } else {
            self.context().boolean_attribute(true)
        }
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Reciprocal);
mlir_op_trait!(Reciprocal, ZeroRegions);
mlir_op_trait!(Reciprocal, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.stochastic_convert`.
pub fn stochastic_convert<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    random: ValueRef<'o, 'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedStochasticConvertOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.stochastic_convert", location);
    let mut operands = Vec::new();
    operands.push(input);
    operands.push(random);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedStochasticConvertOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.stochastic_convert` that stochastically converts floating-point vector values.
pub trait StochasticConvertOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `random` operand.
    fn random(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(StochasticConvert);
mlir_op_trait!(StochasticConvert, ZeroRegions);
mlir_op_trait!(StochasticConvert, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `dst_type` value.
pub const DST_TYPE_ATTRIBUTE: &str = "dst_type";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.stochastic_convert_elementwise`.
pub fn stochastic_convert_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    random: ValueRef<'o, 'c, 't>,
    dst_type: TypeAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedStochasticConvertElementwiseOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.stochastic_convert_elementwise", location);
    let mut operands = Vec::new();
    operands.push(input);
    operands.push(random);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(DST_TYPE_ATTRIBUTE, dst_type);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedStochasticConvertElementwiseOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.stochastic_convert_elementwise` that stochastically converts values elementwise.
pub trait StochasticConvertElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `random` operand.
    fn random(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `dst_type` attribute.
    fn dst_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(DST_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{DST_TYPE_ATTRIBUTE}` attribute in `tpu.stochastic_convert_elementwise`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{DST_TYPE_ATTRIBUTE}` attribute in `tpu.stochastic_convert_elementwise`")
            })
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(StochasticConvertElementwise);
mlir_op_trait!(StochasticConvertElementwise, ZeroRegions);
mlir_op_trait!(StochasticConvertElementwise, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.roll_vectors`.
pub fn roll_vectors<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: &[ValueRef<'o, 'c, 't>],
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedRollVectorsOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.roll_vectors", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedRollVectorsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.roll_vectors` that rolls multiple vectors into one vector.
pub trait RollVectorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operands.
    fn input(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(RollVectors);
mlir_op_trait!(RollVectors, ZeroRegions);
mlir_op_trait!(RollVectors, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.unroll_vectors`.
pub fn unroll_vectors<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedUnrollVectorsOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.unroll_vectors", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_results(result_types);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedUnrollVectorsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.unroll_vectors` that unrolls one vector into multiple vectors.
pub trait UnrollVectorsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `output` results.
    fn output(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| Operation::result(self, index).unwrap()).collect()
    }
}

mlir_op!(UnrollVectors);
mlir_op_trait!(UnrollVectors, ZeroRegions);
mlir_op_trait!(UnrollVectors, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.create_mask`.
pub fn create_mask<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    low: &[ValueRef<'o, 'c, 't>],
    high: &[ValueRef<'o, 'c, 't>],
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedCreateMaskOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.create_mask", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(low);
    operands.extend_from_slice(high);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedCreateMaskOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.create_mask` that creates a vector mask from index bounds.
pub trait CreateMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `low` operands.
    fn low(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0) / 2;
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `high` operands.
    fn high(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0) / 2;
        (0..count).map(|index| self.operand_value(0 + count + index).unwrap()).collect()
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(CreateMask);
mlir_op_trait!(CreateMask, ZeroRegions);
mlir_op_trait!(CreateMask, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `from` value.
pub const FROM_ATTRIBUTE: &str = "from";

/// Name of the [`Attribute`] that stores the `to` value.
pub const TO_ATTRIBUTE: &str = "to";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.create_subelement_mask`.
pub fn create_subelement_mask<'c, 't: 'c, L: Location<'c, 't>>(
    r#from: IntegerAttributeRef<'c, 't>,
    to: IntegerAttributeRef<'c, 't>,
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedCreateSubelementMaskOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.create_subelement_mask", location);
    builder = builder.add_attribute(FROM_ATTRIBUTE, r#from);
    builder = builder.add_attribute(TO_ATTRIBUTE, to);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedCreateSubelementMaskOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.create_subelement_mask` that creates a mask over contiguous subelement rows.
pub trait CreateSubelementMaskOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `from` attribute.
    fn r#from(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(FROM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{FROM_ATTRIBUTE}` attribute in `tpu.create_subelement_mask`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{FROM_ATTRIBUTE}` attribute in `tpu.create_subelement_mask`"))
    }

    /// Returns the `to` attribute.
    fn to(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(TO_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TO_ATTRIBUTE}` attribute in `tpu.create_subelement_mask`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TO_ATTRIBUTE}` attribute in `tpu.create_subelement_mask`"))
    }

    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(CreateSubelementMask);
mlir_op_trait!(CreateSubelementMask, ZeroOperands);
mlir_op_trait!(CreateSubelementMask, ZeroRegions);
mlir_op_trait!(CreateSubelementMask, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `multiple` value.
pub const MULTIPLE_ATTRIBUTE: &str = "multiple";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.assume_multiple`.
pub fn assume_multiple<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    multiple: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAssumeMultipleOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.assume_multiple", location);
    let mut operands = Vec::new();
    operands.push(value);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(MULTIPLE_ATTRIBUTE, multiple);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedAssumeMultipleOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.assume_multiple` that assumes a scalar value is a multiple.
pub trait AssumeMultipleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `multiple` attribute.
    fn multiple(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(MULTIPLE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MULTIPLE_ATTRIBUTE}` attribute in `tpu.assume_multiple`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MULTIPLE_ATTRIBUTE}` attribute in `tpu.assume_multiple`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AssumeMultiple);
mlir_op_trait!(AssumeMultiple, ZeroRegions);
mlir_op_trait!(AssumeMultiple, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_slice`.
pub fn mem_ref_slice<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    mem_ref: ValueRef<'o, 'c, 't>,
    base_idx: &[ValueRef<'o, 'c, 't>],
    dynamic_sizes: &[ValueRef<'o, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMemRefSliceOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMemRefSliceOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.memref_slice` that slices a memref.
pub trait MemRefSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `mem_ref` operand.
    fn mem_ref(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `base_idx` operands.
    fn base_idx(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(1).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the `dynamic_sizes` operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        let count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(offset + index).unwrap()).collect()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MemRefSlice);
mlir_op_trait!(MemRefSlice, ZeroRegions);
mlir_op_trait!(MemRefSlice, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_squeeze`.
pub fn mem_ref_squeeze<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMemRefSqueezeOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.memref_squeeze", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMemRefSqueezeOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.memref_squeeze` that squeezes a memref.
pub trait MemRefSqueezeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MemRefSqueeze);
mlir_op_trait!(MemRefSqueeze, ZeroRegions);
mlir_op_trait!(MemRefSqueeze, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_reshape`.
pub fn mem_ref_reshape<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMemRefReshapeOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.memref_reshape", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMemRefReshapeOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.memref_reshape` that reshapes a memref.
pub trait MemRefReshapeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MemRefReshape);
mlir_op_trait!(MemRefReshape, ZeroRegions);
mlir_op_trait!(MemRefReshape, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.memref_bitcast`.
pub fn mem_ref_bitcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMemRefBitcastOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.memref_bitcast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMemRefBitcastOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.memref_bitcast` that bitcasts a memref.
pub trait MemRefBitcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MemRefBitcast);
mlir_op_trait!(MemRefBitcast, ZeroRegions);
mlir_op_trait!(MemRefBitcast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.reinterpret_cast`.
pub fn reinterpret_cast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReinterpretCastOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.reinterpret_cast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedReinterpretCastOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.reinterpret_cast` that reinterprets a memref type.
pub trait ReinterpretCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ReinterpretCast);
mlir_op_trait!(ReinterpretCast, ZeroRegions);
mlir_op_trait!(ReinterpretCast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.assume_layout`.
pub fn assume_layout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAssumeLayoutOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.assume_layout", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedAssumeLayoutOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.assume_layout` that asserts the layout of a value.
pub trait AssumeLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AssumeLayout);
mlir_op_trait!(AssumeLayout, ZeroRegions);
mlir_op_trait!(AssumeLayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.erase_memref_layout`.
pub fn erase_layout<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    operand: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedEraseLayoutOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.erase_memref_layout", location);
    let mut operands = Vec::new();
    operands.push(operand);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedEraseLayoutOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.erase_memref_layout` that erases a memref layout attribute.
pub trait EraseLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `operand` operand.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(EraseLayout);
mlir_op_trait!(EraseLayout, ZeroRegions);
mlir_op_trait!(EraseLayout, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.device_id`.
pub fn device_id<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedDeviceIdOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.device_id", location);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedDeviceIdOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.device_id` that returns the current TPU device identifier.
pub trait DeviceIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DeviceId);
mlir_op_trait!(DeviceId, ZeroOperands);
mlir_op_trait!(DeviceId, ZeroRegions);
mlir_op_trait!(DeviceId, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_read`.
pub fn semaphore_read<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSemaphoreReadOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sem_read", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSemaphoreReadOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sem_read` that reads a TPU semaphore value.
pub trait SemaphoreReadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(SemaphoreRead);
mlir_op_trait!(SemaphoreRead, ZeroRegions);
mlir_op_trait!(SemaphoreRead, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_wait`.
pub fn semaphore_wait<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    location: L,
) -> DetachedSemaphoreWaitOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sem_wait", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    operands.push(amount);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSemaphoreWaitOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sem_wait` that waits on a TPU semaphore.
pub trait SemaphoreWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(SemaphoreWait);
mlir_op_trait!(SemaphoreWait, ZeroRegions);
mlir_op_trait!(SemaphoreWait, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_alloc`.
pub fn alloca_semaphore<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAllocaSemaphoreOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sem_alloc", location);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedAllocaSemaphoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sem_alloc` that allocates a TPU semaphore.
pub trait AllocaSemaphoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AllocaSemaphore);
mlir_op_trait!(AllocaSemaphore, ZeroOperands);
mlir_op_trait!(AllocaSemaphore, ZeroRegions);
mlir_op_trait!(AllocaSemaphore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_barrier`.
pub fn get_barrier_semaphore<'c, 't: 'c, L: Location<'c, 't>>(
    semaphore_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGetBarrierSemaphoreOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sem_barrier", location);
    builder = builder.add_result(semaphore_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedGetBarrierSemaphoreOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sem_barrier` that returns the TPU barrier semaphore.
pub trait GetBarrierSemaphoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` result.
    fn semaphore(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(GetBarrierSemaphore);
mlir_op_trait!(GetBarrierSemaphore, ZeroOperands);
mlir_op_trait!(GetBarrierSemaphore, ZeroRegions);
mlir_op_trait!(GetBarrierSemaphore, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sem_signal`.
pub fn semaphore_signal<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    amount: ValueRef<'o, 'c, 't>,
    device_id: Option<ValueRef<'o, 'c, 't>>,
    core_id: Option<ValueRef<'o, 'c, 't>>,
    location: L,
) -> DetachedSemaphoreSignalOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSemaphoreSignalOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sem_signal` that signals a TPU semaphore.
pub trait SemaphoreSignalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `amount` operand.
    fn amount(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(2).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }
}

mlir_op!(SemaphoreSignal);
mlir_op_trait!(SemaphoreSignal, ZeroRegions);
mlir_op_trait!(SemaphoreSignal, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.barrier`.
pub fn barrier<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    barrier_id: ValueRef<'o, 'c, 't>,
    location: L,
) -> DetachedBarrierOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.barrier", location);
    let mut operands = Vec::new();
    operands.push(barrier_id);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedBarrierOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.barrier` that synchronizes TPU vector subcores.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `barrier_id` operand.
    fn barrier_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `priority` value.
pub const PRIORITY_ATTRIBUTE: &str = "priority";

/// Name of the [`Attribute`] that stores the `strict_ordering` value.
pub const STRICT_ORDERING_ATTRIBUTE: &str = "strict_ordering";

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
) -> DetachedEnqueueDmaOperation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(priority) = priority {
        builder = builder.add_attribute(PRIORITY_ATTRIBUTE, priority);
    }
    if let Some(strict_ordering) = strict_ordering {
        builder = builder.add_attribute(STRICT_ORDERING_ATTRIBUTE, strict_ordering);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedEnqueueDmaOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.enqueue_dma` that enqueues a TPU DMA transfer.
pub trait EnqueueDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the optional `source_semaphore` operand.
    fn source_semaphore(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(1).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `target` operand.
    fn target(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `target_semaphore` operand.
    fn target_semaphore(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..4).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(4).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..5).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(5).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `priority` attribute.
    fn priority(&self) -> IntegerAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(PRIORITY_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{PRIORITY_ATTRIBUTE}` attribute in `tpu.enqueue_dma`"))
        } else {
            self.context().integer_attribute(self.context().signless_integer_type(32), 0)
        }
    }

    /// Returns the `strict_ordering` attribute.
    fn strict_ordering(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(STRICT_ORDERING_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{STRICT_ORDERING_ATTRIBUTE}` attribute in `tpu.enqueue_dma`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(EnqueueDma);
mlir_op_trait!(EnqueueDma, ZeroRegions);
mlir_op_trait!(EnqueueDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.enqueue_indirect_dma`.
pub fn enqueue_indirect_dma<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    target: ValueRef<'o, 'c, 't>,
    offsets: ValueRef<'o, 'c, 't>,
    semaphore: ValueRef<'o, 'c, 't>,
    offset_filter: Option<ValueRef<'o, 'c, 't>>,
    add: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedEnqueueIndirectDmaOperation<'c, 't> {
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
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedEnqueueIndirectDmaOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.enqueue_indirect_dma` that enqueues an indirect TPU DMA transfer.
pub trait EnqueueIndirectDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `source` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `target` operand.
    fn target(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `offsets` operand.
    fn offsets(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the optional `offset_filter` operand.
    fn offset_filter(&self) -> Option<ValueRef<'o, 'c, 't>> {
        if self.operand_count() > 4 { self.operand_value(4) } else { None }
    }

    /// Returns the `add` attribute.
    fn add(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(ADD_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{ADD_ATTRIBUTE}` attribute in `tpu.enqueue_indirect_dma`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(EnqueueIndirectDma);
mlir_op_trait!(EnqueueIndirectDma, ZeroRegions);
mlir_op_trait!(EnqueueIndirectDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.wait_dma2`.
pub fn wait_dma2<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    destination: ValueRef<'o, 'c, 't>,
    device_id: Option<ValueRef<'o, 'c, 't>>,
    core_id: Option<ValueRef<'o, 'c, 't>>,
    strict_ordering: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedWaitDma2Operation<'c, 't> {
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
    let operand_segment_sizes = builder.context().dense_i32_array_attribute(&operand_segment_sizes).unwrap();
    builder = builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, operand_segment_sizes);
    if let Some(strict_ordering) = strict_ordering {
        builder = builder.add_attribute(STRICT_ORDERING_ATTRIBUTE, strict_ordering);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedWaitDma2Operation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.wait_dma2` that waits for a TPU DMA transfer.
pub trait WaitDma2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..0).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `src` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..1).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the `dst` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..2).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        self.operand_value(offset).unwrap()
    }

    /// Returns the optional `device_id` operand.
    fn device_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..3).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(3).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the optional `core_id` operand.
    fn core_id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let offset = (0..4).map(|segment| sizes.get(segment).copied().unwrap_or(0).max(0) as usize).sum::<usize>();
        if sizes.get(4).copied().unwrap_or(0) > 0 { self.operand_value(offset) } else { None }
    }

    /// Returns the `strict_ordering` attribute.
    fn strict_ordering(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(STRICT_ORDERING_ATTRIBUTE) {
            attribute
                .cast()
                .unwrap_or_else(|| panic!("invalid `{STRICT_ORDERING_ATTRIBUTE}` attribute in `tpu.wait_dma2`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(WaitDma2);
mlir_op_trait!(WaitDma2, ZeroRegions);
mlir_op_trait!(WaitDma2, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.wait_indirect_dma`.
pub fn wait_indirect_dma<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    semaphore: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    destination: ValueRef<'o, 'c, 't>,
    location: L,
) -> DetachedWaitIndirectDmaOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.wait_indirect_dma", location);
    let mut operands = Vec::new();
    operands.push(semaphore);
    operands.push(source);
    operands.push(destination);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedWaitIndirectDmaOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.wait_indirect_dma` that waits for an indirect TPU DMA transfer.
pub trait WaitIndirectDmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `semaphore` operand.
    fn semaphore(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `src` operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `dst` operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }
}

mlir_op!(WaitIndirectDma);
mlir_op_trait!(WaitIndirectDma, ZeroRegions);
mlir_op_trait!(WaitIndirectDma, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.region`.
pub fn region<'c, 't: 'c, L: Location<'c, 't>>(
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedRegionOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.region", location);
    builder = builder.add_results(result_types);
    builder = builder.add_region(region);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedRegionOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.region` that contains a Mosaic TPU region.
pub trait RegionOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation results produced by the region.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| Operation::result(self, index).unwrap()).collect()
    }
}

mlir_op!(Region);
mlir_op_trait!(Region, ZeroOperands);
mlir_op_trait!(Region, OneRegion);
mlir_op_trait!(Region, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `message` value.
pub const MESSAGE_ATTRIBUTE: &str = "message";

/// Name of the [`Attribute`] that stores the `level` value.
pub const LEVEL_ATTRIBUTE: &str = "level";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace`.
pub fn trace<'c, 't: 'c, L: Location<'c, 't>>(
    message: StringAttributeRef<'c, 't>,
    level: IntegerAttributeRef<'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedTraceOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.trace", location);
    builder = builder.add_attribute(MESSAGE_ATTRIBUTE, message);
    builder = builder.add_attribute(LEVEL_ATTRIBUTE, level);
    builder = builder.add_results(result_types);
    builder = builder.add_region(region);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTraceOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.trace` that contains a traced Mosaic TPU region.
pub trait TraceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `message` attribute.
    fn message(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(MESSAGE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MESSAGE_ATTRIBUTE}` attribute in `tpu.trace`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MESSAGE_ATTRIBUTE}` attribute in `tpu.trace`"))
    }

    /// Returns the `level` attribute.
    fn level(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(LEVEL_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{LEVEL_ATTRIBUTE}` attribute in `tpu.trace`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{LEVEL_ATTRIBUTE}` attribute in `tpu.trace`"))
    }

    /// Returns the operation results produced by the trace region.
    fn result_values(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| Operation::result(self, index).unwrap()).collect()
    }
}

mlir_op!(Trace);
mlir_op_trait!(Trace, ZeroOperands);
mlir_op_trait!(Trace, OneRegion);
mlir_op_trait!(Trace, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_start`.
pub fn trace_start<'c, 't: 'c, L: Location<'c, 't>>(
    message: StringAttributeRef<'c, 't>,
    level: IntegerAttributeRef<'c, 't>,
    location: L,
) -> DetachedTraceStartOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.trace_start", location);
    builder = builder.add_attribute(MESSAGE_ATTRIBUTE, message);
    builder = builder.add_attribute(LEVEL_ATTRIBUTE, level);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTraceStartOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.trace_start` that starts a Mosaic TPU trace section.
pub trait TraceStartOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `message` attribute.
    fn message(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(MESSAGE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MESSAGE_ATTRIBUTE}` attribute in `tpu.trace_start`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MESSAGE_ATTRIBUTE}` attribute in `tpu.trace_start`"))
    }

    /// Returns the `level` attribute.
    fn level(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(LEVEL_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{LEVEL_ATTRIBUTE}` attribute in `tpu.trace_start`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{LEVEL_ATTRIBUTE}` attribute in `tpu.trace_start`"))
    }
}

mlir_op!(TraceStart);
mlir_op_trait!(TraceStart, ZeroOperands);
mlir_op_trait!(TraceStart, ZeroRegions);
mlir_op_trait!(TraceStart, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_stop`.
pub fn trace_stop<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedTraceStopOperation<'c, 't> {
    let builder = OperationBuilder::new("tpu.trace_stop", location);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTraceStopOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.trace_stop` that stops a Mosaic TPU trace section.
pub trait TraceStopOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(TraceStop);
mlir_op_trait!(TraceStop, ZeroOperands);
mlir_op_trait!(TraceStop, ZeroRegions);
mlir_op_trait!(TraceStop, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `label` value.
pub const LABEL_ATTRIBUTE: &str = "label";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.trace_value`.
pub fn trace_value<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    label: StringAttributeRef<'c, 't>,
    location: L,
) -> DetachedTraceValueOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.trace_value", location);
    let mut operands = Vec::new();
    operands.push(value);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(LABEL_ATTRIBUTE, label);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTraceValueOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.trace_value` that emits a scalar trace value.
pub trait TraceValueOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `label` attribute.
    fn label(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(LABEL_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{LABEL_ATTRIBUTE}` attribute in `tpu.trace_value`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{LABEL_ATTRIBUTE}` attribute in `tpu.trace_value`"))
    }
}

mlir_op!(TraceValue);
mlir_op_trait!(TraceValue, ZeroRegions);
mlir_op_trait!(TraceValue, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.yield`.
pub fn r#yield<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    results: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.yield", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(results);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedYieldOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.yield` that terminates a Mosaic TPU region.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `results` operands.
    fn results(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.delay`.
pub fn delay<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    nanoseconds: ValueRef<'o, 'c, 't>,
    location: L,
) -> DetachedDelayOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.delay", location);
    let mut operands = Vec::new();
    operands.push(nanoseconds);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedDelayOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.delay` that delays TPU execution.
pub trait DelayOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `nanos` operand.
    fn nanoseconds(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(Delay);
mlir_op_trait!(Delay, ZeroRegions);
mlir_op_trait!(Delay, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.mask_cast`.
pub fn mask_cast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMaskCastOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.mask_cast", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedMaskCastOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.mask_cast` that casts a TPU mask register to a different packing.
pub trait MaskCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MaskCast);
mlir_op_trait!(MaskCast, ZeroRegions);
mlir_op_trait!(MaskCast, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.iteration_bound`.
pub fn get_iteration_bound<'c, 't: 'c, L: Location<'c, 't>>(
    dim: IntegerAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGetIterationBoundOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.iteration_bound", location);
    builder = builder.add_attribute(DIM_ATTRIBUTE, dim);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedGetIterationBoundOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.iteration_bound` that returns a TPU iteration bound.
pub trait GetIterationBoundOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `dim` attribute.
    fn dim(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIM_ATTRIBUTE}` attribute in `tpu.iteration_bound`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIM_ATTRIBUTE}` attribute in `tpu.iteration_bound`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(GetIterationBound);
mlir_op_trait!(GetIterationBound, ZeroOperands);
mlir_op_trait!(GetIterationBound, ZeroRegions);
mlir_op_trait!(GetIterationBound, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.internal_scratch`.
pub fn get_internal_scratch<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedGetInternalScratchOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.internal_scratch", location);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedGetInternalScratchOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.internal_scratch` that returns internal TPU scratch memory.
pub trait GetInternalScratchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(GetInternalScratch);
mlir_op_trait!(GetInternalScratch, ZeroOperands);
mlir_op_trait!(GetInternalScratch, ZeroRegions);
mlir_op_trait!(GetInternalScratch, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.prng_set_seed_32`.
pub fn prng_set_seed_32<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    seeds: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> DetachedPrngSeed32Operation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.prng_set_seed_32", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(seeds);
    builder = builder.add_operands(&operands);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedPrngSeed32Operation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.prng_set_seed_32` that sets the TPU 32-bit PRNG seed.
pub trait PrngSeed32Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `seeds` operands.
    fn seeds(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }
}

mlir_op!(PrngSeed32);
mlir_op_trait!(PrngSeed32, ZeroRegions);
mlir_op_trait!(PrngSeed32, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.prng_random_bits`.
pub fn prng_random_bits<'c, 't: 'c, L: Location<'c, 't>>(
    output_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedPrngRandomBitsOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.prng_random_bits", location);
    builder = builder.add_result(output_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedPrngRandomBitsOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.prng_random_bits` that returns TPU PRNG random bits.
pub trait PrngRandomBitsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `output` result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(PrngRandomBits);
mlir_op_trait!(PrngRandomBits, ZeroOperands);
mlir_op_trait!(PrngRandomBits, ZeroRegions);
mlir_op_trait!(PrngRandomBits, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `pattern` value.
pub const PATTERN_ATTRIBUTE: &str = "pattern";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.sublane_shuffle`.
pub fn sublane_shuffle<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    rhs: ValueRef<'o, 'c, 't>,
    pattern: DenseInteger32ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSublaneShuffleOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.sublane_shuffle", location);
    let mut operands = Vec::new();
    operands.push(lhs);
    operands.push(rhs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(PATTERN_ATTRIBUTE, pattern);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedSublaneShuffleOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.sublane_shuffle` that shuffles two TPU vector registers by sublane.
pub trait SublaneShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `lhs` operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `rhs` operand.
    fn rhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `pattern` attribute.
    fn pattern(&self) -> DenseInteger32ArrayAttributeRef<'c, 't> {
        self.attribute(PATTERN_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{PATTERN_ATTRIBUTE}` attribute in `tpu.sublane_shuffle`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{PATTERN_ATTRIBUTE}` attribute in `tpu.sublane_shuffle`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(SublaneShuffle);
mlir_op_trait!(SublaneShuffle, ZeroRegions);
mlir_op_trait!(SublaneShuffle, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `permutation` value.
pub const PERMUTATION_ATTRIBUTE: &str = "permutation";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.transpose`.
pub fn transpose<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    vector: ValueRef<'o, 'c, 't>,
    permutation: DenseInteger64ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTransposeOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.transpose", location);
    let mut operands = Vec::new();
    operands.push(vector);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(PERMUTATION_ATTRIBUTE, permutation);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedTransposeOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.transpose` that transposes a vector.
pub trait TransposeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `vector` operand.
    fn vector(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `permutation` attribute.
    fn permutation(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(PERMUTATION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{PERMUTATION_ATTRIBUTE}` attribute in `tpu.transpose`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{PERMUTATION_ATTRIBUTE}` attribute in `tpu.transpose`"))
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Transpose);
mlir_op_trait!(Transpose, ZeroRegions);
mlir_op_trait!(Transpose, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `tag` value.
pub const TAG_ATTRIBUTE: &str = "tag";

/// Name of the [`Attribute`] that stores the `formatted` value.
pub const FORMATTED_ATTRIBUTE: &str = "formatted";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.log`.
pub fn log<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    inputs: &[ValueRef<'o, 'c, 't>],
    tag: StringAttributeRef<'c, 't>,
    formatted: Option<BooleanAttributeRef<'c, 't>>,
    location: L,
) -> DetachedLogOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.log", location);
    let mut operands = Vec::new();
    operands.extend_from_slice(inputs);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(TAG_ATTRIBUTE, tag);
    if let Some(formatted) = formatted {
        builder = builder.add_attribute(FORMATTED_ATTRIBUTE, formatted);
    }
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedLogOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.log` that logs scalar values from TPU execution.
pub trait LogOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `inputs` operands.
    fn inputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(0);
        (0..count).map(|index| self.operand_value(0 + index).unwrap()).collect()
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(TAG_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TAG_ATTRIBUTE}` attribute in `tpu.log`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TAG_ATTRIBUTE}` attribute in `tpu.log`"))
    }

    /// Returns the `formatted` attribute.
    fn formatted(&self) -> BooleanAttributeRef<'c, 't> {
        if let Some(attribute) = self.attribute(FORMATTED_ATTRIBUTE) {
            attribute.cast().unwrap_or_else(|| panic!("invalid `{FORMATTED_ATTRIBUTE}` attribute in `tpu.log`"))
        } else {
            self.context().boolean_attribute(false)
        }
    }
}

mlir_op!(Log);
mlir_op_trait!(Log, ZeroRegions);
mlir_op_trait!(Log, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the `shape` value.
pub const SHAPE_ATTRIBUTE: &str = "shape";

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.log_buffer`.
pub fn log_buffer<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'o, 'c, 't>,
    shape: DenseInteger64ArrayAttributeRef<'c, 't>,
    tag: StringAttributeRef<'c, 't>,
    location: L,
) -> DetachedLogBufferOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.log_buffer", location);
    let mut operands = Vec::new();
    operands.push(input);
    builder = builder.add_operands(&operands);
    builder = builder.add_attribute(SHAPE_ATTRIBUTE, shape);
    builder = builder.add_attribute(TAG_ATTRIBUTE, tag);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedLogBufferOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.log_buffer` that logs a memory buffer from TPU execution.
pub trait LogBufferOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `input` operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `shape` attribute.
    fn shape(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(SHAPE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SHAPE_ATTRIBUTE}` attribute in `tpu.log_buffer`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SHAPE_ATTRIBUTE}` attribute in `tpu.log_buffer`"))
    }

    /// Returns the `tag` attribute.
    fn tag(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(TAG_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TAG_ATTRIBUTE}` attribute in `tpu.log_buffer`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TAG_ATTRIBUTE}` attribute in `tpu.log_buffer`"))
    }
}

mlir_op!(LogBuffer);
mlir_op_trait!(LogBuffer, ZeroRegions);
mlir_op_trait!(LogBuffer, ZeroSuccessors);

/// Creates a detached Mosaic TPU [`Operation`] for `tpu.fetch_and_add_sync`.
pub fn fetch_and_add_sync<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    base: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    value: ValueRef<'o, 'c, 't>,
    core_id: ValueRef<'o, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedFetchAndAddSyncOperation<'c, 't> {
    let mut builder = OperationBuilder::new("tpu.fetch_and_add_sync", location);
    let mut operands = Vec::new();
    operands.push(base);
    operands.extend_from_slice(indices);
    operands.push(value);
    operands.push(core_id);
    builder = builder.add_operands(&operands);
    builder = builder.add_result(result_type);
    let operation = builder.build().unwrap();
    unsafe { operation.cast::<DetachedFetchAndAddSyncOperation>().unwrap() }
}

/// Mosaic TPU [`Operation`] for `tpu.fetch_and_add_sync` that synchronously fetches and increments SMEM.
pub trait FetchAndAddSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the `base` operand.
    fn base(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `indices` operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let count = self.operand_count().saturating_sub(3);
        (0..count).map(|index| self.operand_value(1 + index).unwrap()).collect()
    }

    /// Returns the `value` operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        let count = self.operand_count().saturating_sub(3);
        self.operand_value(1 + count + 0).unwrap()
    }

    /// Returns the `core_id` operand.
    fn core_id(&self) -> ValueRef<'o, 'c, 't> {
        let count = self.operand_count().saturating_sub(3);
        self.operand_value(1 + count + 1).unwrap()
    }

    /// Returns the `result` result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(FetchAndAddSync);
mlir_op_trait!(FetchAndAddSync, ZeroRegions);
mlir_op_trait!(FetchAndAddSync, ZeroSuccessors);

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::mosaic::tpu::attributes::ReductionKind;
    use crate::{
        Block, Context, DetachedOp, DialectHandle, OneRegion, Operation, OperationBuilder, Region, Type, Value,
    };

    use super::*;

    macro_rules! test_operation_wrapper {
        ($test_name:ident, $operation_type:ident, $operation_name:literal) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                context.load_dialect(DialectHandle::mosaic_tpu());
                let operation = OperationBuilder::new($operation_name, context.unknown_location()).build().unwrap();
                let operation = unsafe { operation.cast::<$operation_type>().unwrap() };
                assert_eq!(operation.name().as_str().unwrap(), $operation_name);
            }
        };
    }

    #[test]
    fn test_all_reduce_constructor_and_accessors() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_tpu());
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32).as_ref();
        let block = context.block(&[(i32_type, location)]);
        let input = block.argument(0).unwrap().as_ref();
        let dim = context.integer_attribute(context.signless_integer_type(64), 0);
        let kind = context.mosaic_tpu_reduction_kind_attribute(ReductionKind::Sum);

        let operation = all_reduce(input, dim, kind, i32_type, location);

        assert_eq!(operation.input(), input);
        assert_eq!(operation.dim(), dim);
        assert_eq!(operation.kind(), kind);
        assert_eq!(operation.output(), Operation::result(&operation, 0).unwrap());
    }

    #[test]
    fn test_store_constructor_and_segmented_accessors() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_tpu());
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32).as_ref();
        let block = context.block(&[
            (i32_type, location),
            (i32_type, location),
            (i32_type, location),
            (i32_type, location),
            (i32_type, location),
        ]);
        let value_to_store = block.argument(0).unwrap().as_ref();
        let base = block.argument(1).unwrap().as_ref();
        let index_0 = block.argument(2).unwrap().as_ref();
        let index_1 = block.argument(3).unwrap().as_ref();
        let mask = block.argument(4).unwrap().as_ref();
        let sublane_mask = context.dense_bool_array_attribute(&[true, false]).unwrap();

        let operation =
            store(value_to_store, base, &[index_0, index_1], Some(mask), sublane_mask, None, None, location);

        assert_eq!(operation.value_to_store(), value_to_store);
        assert_eq!(operation.base(), base);
        assert_eq!(operation.indices(), vec![index_0, index_1]);
        assert_eq!(operation.mask(), Some(mask));
        assert_eq!(operation.sublane_mask(), sublane_mask);
        assert_eq!(operation.sublane_stride().signless_value(), 1);
        assert!(!operation.add().value());
    }

    #[test]
    fn test_region_constructor_and_accessors() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_tpu());
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32).as_ref();

        let operation = region(&[i32_type], context.region(), location);

        assert_eq!(operation.result_values(), vec![Operation::result(&operation, 0).unwrap()]);
        assert!(operation.body_region().is_empty());
    }

    test_operation_wrapper!(test_all_reduce_operation, DetachedAllReduceOperation, "tpu.all_reduce");
    test_operation_wrapper!(test_reduce_index_operation, DetachedReduceIndexOperation, "tpu.reduce_index");
    test_operation_wrapper!(test_scan_operation, DetachedScanOperation, "tpu.scan");
    test_operation_wrapper!(test_sort_operation, DetachedSortOperation, "tpu.sort");
    test_operation_wrapper!(test_store_operation, DetachedStoreOperation, "tpu.store");
    test_operation_wrapper!(test_load_operation, DetachedLoadOperation, "tpu.load");
    test_operation_wrapper!(test_vector_store_operation, DetachedVectorStoreOperation, "tpu.vector_store");
    test_operation_wrapper!(test_vector_load_operation, DetachedVectorLoadOperation, "tpu.vector_load");
    test_operation_wrapper!(test_strided_load_operation, DetachedStridedLoadOperation, "tpu.strided_load");
    test_operation_wrapper!(test_strided_store_operation, DetachedStridedStoreOperation, "tpu.strided_store");
    test_operation_wrapper!(test_shuffled_load_operation, DetachedShuffledLoadOperation, "tpu.shuffled_load");
    test_operation_wrapper!(test_shuffled_store_operation, DetachedShuffledStoreOperation, "tpu.shuffled_store");
    test_operation_wrapper!(test_vector_load_idx_operation, DetachedVectorLoadIdxOperation, "tpu.vector_load_idx");
    test_operation_wrapper!(test_vector_store_idx_operation, DetachedVectorStoreIdxOperation, "tpu.vector_store_idx");
    test_operation_wrapper!(test_rotate_operation, DetachedRotateOperation, "tpu.rotate");
    test_operation_wrapper!(test_dynamic_rotate_operation, DetachedDynamicRotateOperation, "tpu.dynamic_rotate");
    test_operation_wrapper!(test_scan_count_operation, DetachedScanCountOperation, "tpu.scan_count");
    test_operation_wrapper!(test_iota_operation, DetachedIotaOperation, "tpu.iota");
    test_operation_wrapper!(test_reshape_operation, DetachedReshapeOperation, "tpu.reshape");
    test_operation_wrapper!(test_repeat_operation, DetachedRepeatOperation, "tpu.repeat");
    test_operation_wrapper!(
        test_broadcast_in_sublanes_operation,
        DetachedBroadcastInSublanesOperation,
        "tpu.broadcast_in_sublanes"
    );
    test_operation_wrapper!(
        test_unpack_subelements_operation,
        DetachedUnpackSubelementsOperation,
        "tpu.unpack_subelements"
    );
    test_operation_wrapper!(test_pack_subelements_operation, DetachedPackSubelementsOperation, "tpu.pack_subelements");
    test_operation_wrapper!(test_pack_elementwise_operation, DetachedPackElementwiseOperation, "tpu.pack_elementwise");
    test_operation_wrapper!(
        test_unpack_elementwise_operation,
        DetachedUnpackElementwiseOperation,
        "tpu.unpack_elementwise"
    );
    test_operation_wrapper!(test_relayout_operation, DetachedRelayoutOperation, "tpu.relayout");
    test_operation_wrapper!(test_pack_mask_operation, DetachedPackMaskOperation, "tpu.pack_vmsk");
    test_operation_wrapper!(test_gather_operation, DetachedGatherOperation, "tpu.gather");
    test_operation_wrapper!(test_dynamic_gather_operation, DetachedDynamicGatherOperation, "tpu.dynamic_gather");
    test_operation_wrapper!(test_fp_to_si_operation, DetachedFpToSiOperation, "tpu.fptosi");
    test_operation_wrapper!(test_fp_to_ui_operation, DetachedFpToUiOperation, "tpu.fptoui");
    test_operation_wrapper!(test_si_to_fp_operation, DetachedSiToFpOperation, "tpu.sitofp");
    test_operation_wrapper!(test_ui_to_fp_operation, DetachedUiToFpOperation, "tpu.uitofp");
    test_operation_wrapper!(test_ext_f_operation, DetachedExtFOperation, "tpu.extf");
    test_operation_wrapper!(test_trunc_f_operation, DetachedTruncFOperation, "tpu.truncf");
    test_operation_wrapper!(test_matmul_operation, DetachedMatmulOperation, "tpu.matmul");
    test_operation_wrapper!(test_matmul_push_rhs_operation, DetachedMatmulPushRhsOperation, "tpu.matmul_push_rhs");
    test_operation_wrapper!(test_matmul_acc_lhs_operation, DetachedMatmulAccLhsOperation, "tpu.matmul_acc_lhs");
    test_operation_wrapper!(test_matmul_pop_operation, DetachedMatmulPopOperation, "tpu.matmul_pop");
    test_operation_wrapper!(test_concatenate_operation, DetachedConcatenateOperation, "tpu.concatenate");
    test_operation_wrapper!(test_bitcast_operation, DetachedBitcastOperation, "tpu.bitcast");
    test_operation_wrapper!(test_bitcast_vreg_operation, DetachedBitcastVregOperation, "tpu.bitcast_vreg");
    test_operation_wrapper!(test_weird_operation, DetachedWeirdOperation, "tpu.weird");
    test_operation_wrapper!(test_reciprocal_operation, DetachedReciprocalOperation, "tpu.reciprocal");
    test_operation_wrapper!(
        test_stochastic_convert_operation,
        DetachedStochasticConvertOperation,
        "tpu.stochastic_convert"
    );
    test_operation_wrapper!(
        test_stochastic_convert_elementwise_operation,
        DetachedStochasticConvertElementwiseOperation,
        "tpu.stochastic_convert_elementwise"
    );
    test_operation_wrapper!(test_roll_vectors_operation, DetachedRollVectorsOperation, "tpu.roll_vectors");
    test_operation_wrapper!(test_unroll_vectors_operation, DetachedUnrollVectorsOperation, "tpu.unroll_vectors");
    test_operation_wrapper!(test_create_mask_operation, DetachedCreateMaskOperation, "tpu.create_mask");
    test_operation_wrapper!(
        test_create_subelement_mask_operation,
        DetachedCreateSubelementMaskOperation,
        "tpu.create_subelement_mask"
    );
    test_operation_wrapper!(test_assume_multiple_operation, DetachedAssumeMultipleOperation, "tpu.assume_multiple");
    test_operation_wrapper!(test_memref_slice_operation, DetachedMemRefSliceOperation, "tpu.memref_slice");
    test_operation_wrapper!(test_memref_squeeze_operation, DetachedMemRefSqueezeOperation, "tpu.memref_squeeze");
    test_operation_wrapper!(test_memref_reshape_operation, DetachedMemRefReshapeOperation, "tpu.memref_reshape");
    test_operation_wrapper!(test_memref_bitcast_operation, DetachedMemRefBitcastOperation, "tpu.memref_bitcast");
    test_operation_wrapper!(test_reinterpret_cast_operation, DetachedReinterpretCastOperation, "tpu.reinterpret_cast");
    test_operation_wrapper!(test_assume_layout_operation, DetachedAssumeLayoutOperation, "tpu.assume_layout");
    test_operation_wrapper!(test_erase_layout_operation, DetachedEraseLayoutOperation, "tpu.erase_memref_layout");
    test_operation_wrapper!(test_device_id_operation, DetachedDeviceIdOperation, "tpu.device_id");
    test_operation_wrapper!(test_semaphore_read_operation, DetachedSemaphoreReadOperation, "tpu.sem_read");
    test_operation_wrapper!(test_semaphore_wait_operation, DetachedSemaphoreWaitOperation, "tpu.sem_wait");
    test_operation_wrapper!(test_alloca_semaphore_operation, DetachedAllocaSemaphoreOperation, "tpu.sem_alloc");
    test_operation_wrapper!(
        test_get_barrier_semaphore_operation,
        DetachedGetBarrierSemaphoreOperation,
        "tpu.sem_barrier"
    );
    test_operation_wrapper!(test_semaphore_signal_operation, DetachedSemaphoreSignalOperation, "tpu.sem_signal");
    test_operation_wrapper!(test_barrier_operation, DetachedBarrierOperation, "tpu.barrier");
    test_operation_wrapper!(test_enqueue_dma_operation, DetachedEnqueueDmaOperation, "tpu.enqueue_dma");
    test_operation_wrapper!(
        test_enqueue_indirect_dma_operation,
        DetachedEnqueueIndirectDmaOperation,
        "tpu.enqueue_indirect_dma"
    );
    test_operation_wrapper!(test_wait_dma2_operation, DetachedWaitDma2Operation, "tpu.wait_dma2");
    test_operation_wrapper!(
        test_wait_indirect_dma_operation,
        DetachedWaitIndirectDmaOperation,
        "tpu.wait_indirect_dma"
    );
    test_operation_wrapper!(test_region_operation, DetachedRegionOperation, "tpu.region");
    test_operation_wrapper!(test_trace_operation, DetachedTraceOperation, "tpu.trace");
    test_operation_wrapper!(test_trace_start_operation, DetachedTraceStartOperation, "tpu.trace_start");
    test_operation_wrapper!(test_trace_stop_operation, DetachedTraceStopOperation, "tpu.trace_stop");
    test_operation_wrapper!(test_trace_value_operation, DetachedTraceValueOperation, "tpu.trace_value");
    test_operation_wrapper!(test_yield_operation, DetachedYieldOperation, "tpu.yield");
    test_operation_wrapper!(test_delay_operation, DetachedDelayOperation, "tpu.delay");
    test_operation_wrapper!(test_mask_cast_operation, DetachedMaskCastOperation, "tpu.mask_cast");
    test_operation_wrapper!(
        test_get_iteration_bound_operation,
        DetachedGetIterationBoundOperation,
        "tpu.iteration_bound"
    );
    test_operation_wrapper!(
        test_get_internal_scratch_operation,
        DetachedGetInternalScratchOperation,
        "tpu.internal_scratch"
    );
    test_operation_wrapper!(test_prng_seed32_operation, DetachedPrngSeed32Operation, "tpu.prng_set_seed_32");
    test_operation_wrapper!(test_prng_random_bits_operation, DetachedPrngRandomBitsOperation, "tpu.prng_random_bits");
    test_operation_wrapper!(test_sublane_shuffle_operation, DetachedSublaneShuffleOperation, "tpu.sublane_shuffle");
    test_operation_wrapper!(test_transpose_operation, DetachedTransposeOperation, "tpu.transpose");
    test_operation_wrapper!(test_log_operation, DetachedLogOperation, "tpu.log");
    test_operation_wrapper!(test_log_buffer_operation, DetachedLogBufferOperation, "tpu.log_buffer");
    test_operation_wrapper!(
        test_fetch_and_add_sync_operation,
        DetachedFetchAndAddSyncOperation,
        "tpu.fetch_and_add_sync"
    );
}
