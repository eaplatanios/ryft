use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DialectHandle,
    IntegerAttributeRef, Location, Operation, OperationBuilder, OperationResultRef, TypeRef, ValueRef, mlir_op,
    mlir_op_trait,
};

use super::attributes::{RcpRoundingMode, RcpRoundingModeAttributeRef};

/// Name of the NVGPU `transpose` attribute.
pub const TRANSPOSE_ATTRIBUTE: &str = "transpose";

/// Name of the NVGPU `numTiles` attribute.
pub const NUM_TILES_ATTRIBUTE: &str = "numTiles";

/// Operation trait for `nvgpu.ldmatrix`.
pub trait LdMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref operand.
    fn src_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the source memref index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (1..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns whether the loaded matrix is transposed.
    fn transpose(&self) -> bool {
        self.attribute(TRANSPOSE_ATTRIBUTE).unwrap().cast::<BooleanAttributeRef>().unwrap().value()
    }

    /// Returns the number of loaded matrix tiles.
    fn num_tiles(&self) -> i64 {
        self.attribute(NUM_TILES_ATTRIBUTE).unwrap().cast::<IntegerAttributeRef>().unwrap().signless_value()
    }

    /// Returns the loaded matrix fragment result.
    fn matrix(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(LdMatrix);
mlir_op_trait!(LdMatrix, OneResult);
mlir_op_trait!(LdMatrix, ZeroRegions);
mlir_op_trait!(LdMatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`LdMatrixOperation`] at the specified [`Location`].
pub fn ldmatrix<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src_memref: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    transpose: bool,
    num_tiles: i32,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedLdMatrixOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.ldmatrix", location)
        .add_operand(src_memref)
        .add_operands(indices)
        .add_attribute(TRANSPOSE_ATTRIBUTE, context.boolean_attribute(transpose))
        .add_attribute(
            NUM_TILES_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), num_tiles as i64),
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::ldmatrix`")
}

/// Name of the NVGPU `mmaShape` attribute.
pub const MMA_SHAPE_ATTRIBUTE: &str = "mmaShape";

/// Name of the NVGPU `tf32Enabled` unit attribute.
pub const TF32_ENABLED_ATTRIBUTE: &str = "tf32Enabled";

/// Operation trait for `nvgpu.mma.sync`.
pub trait MmaSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left matrix operand.
    fn matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right matrix operand.
    fn matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the accumulator matrix operand.
    fn matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the warp-level MMA shape.
    fn mma_shape(&self) -> Vec<i64> {
        self.attribute(MMA_SHAPE_ATTRIBUTE)
            .unwrap()
            .cast::<ArrayAttributeRef>()
            .unwrap()
            .elements()
            .map(|attribute| attribute.cast::<IntegerAttributeRef>().unwrap().signless_value())
            .collect()
    }

    /// Returns whether TF32 execution is enabled.
    fn tf32_enabled(&self) -> bool {
        self.attribute(TF32_ENABLED_ATTRIBUTE).is_some()
    }

    /// Returns the MMA accumulator result.
    fn matrix_d(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MmaSync);
mlir_op_trait!(MmaSync, OneResult);
mlir_op_trait!(MmaSync, ZeroRegions);
mlir_op_trait!(MmaSync, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaSyncOperation`] at the specified [`Location`].
pub fn mma_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    matrix_a: ValueRef<'v, 'c, 't>,
    matrix_b: ValueRef<'v, 'c, 't>,
    matrix_c: ValueRef<'v, 'c, 't>,
    mma_shape: &[i64],
    tf32_enabled: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMmaSyncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let mma_shape_type = context.signless_integer_type(64);
    let mma_shape_elements =
        mma_shape.iter().map(|value| context.integer_attribute(mma_shape_type, *value)).collect::<Vec<_>>();
    let mut builder = OperationBuilder::new("nvgpu.mma.sync", location)
        .add_operands(&[matrix_a, matrix_b, matrix_c])
        .add_attribute(MMA_SHAPE_ATTRIBUTE, context.array_attribute(&mma_shape_elements));
    if tf32_enabled {
        builder = builder.add_attribute(TF32_ENABLED_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mma_sync`")
}

/// Name of the NVGPU `sparsitySelector` attribute.
pub const SPARSITY_SELECTOR_ATTRIBUTE: &str = "sparsitySelector";

/// Operation trait for `nvgpu.mma.sp.sync`.
pub trait MmaSparseSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the left sparse matrix operand.
    fn matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the right matrix operand.
    fn matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the accumulator matrix operand.
    fn matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the sparse metadata operand.
    fn sparse_metadata(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the warp-level MMA shape.
    fn mma_shape(&self) -> Vec<i64> {
        self.attribute(MMA_SHAPE_ATTRIBUTE)
            .unwrap()
            .cast::<ArrayAttributeRef>()
            .unwrap()
            .elements()
            .map(|attribute| attribute.cast::<IntegerAttributeRef>().unwrap().signless_value())
            .collect()
    }

    /// Returns the sparsity selector value.
    fn sparsity_selector(&self) -> i64 {
        self.attribute(SPARSITY_SELECTOR_ATTRIBUTE)
            .unwrap()
            .cast::<IntegerAttributeRef>()
            .unwrap()
            .signless_value()
    }

    /// Returns whether TF32 execution is enabled.
    fn tf32_enabled(&self) -> bool {
        self.attribute(TF32_ENABLED_ATTRIBUTE).is_some()
    }

    /// Returns the sparse MMA accumulator result.
    fn matrix_d(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MmaSparseSync);
mlir_op_trait!(MmaSparseSync, OneResult);
mlir_op_trait!(MmaSparseSync, ZeroRegions);
mlir_op_trait!(MmaSparseSync, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaSparseSyncOperation`] at the specified [`Location`].
pub fn mma_sparse_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    matrix_a: ValueRef<'v, 'c, 't>,
    matrix_b: ValueRef<'v, 'c, 't>,
    matrix_c: ValueRef<'v, 'c, 't>,
    sparse_metadata: ValueRef<'v, 'c, 't>,
    mma_shape: &[i64],
    sparsity_selector: i32,
    tf32_enabled: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMmaSparseSyncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let mma_shape_type = context.signless_integer_type(64);
    let mma_shape_elements =
        mma_shape.iter().map(|value| context.integer_attribute(mma_shape_type, *value)).collect::<Vec<_>>();
    let mut builder = OperationBuilder::new("nvgpu.mma.sp.sync", location)
        .add_operands(&[matrix_a, matrix_b, matrix_c, sparse_metadata])
        .add_attribute(MMA_SHAPE_ATTRIBUTE, context.array_attribute(&mma_shape_elements))
        .add_attribute(
            SPARSITY_SELECTOR_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), sparsity_selector as i64),
        );
    if tf32_enabled {
        builder = builder.add_attribute(TF32_ENABLED_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mma_sparse_sync`")
}

/// Name of the NVGPU `dstElements` attribute.
pub const DST_ELEMENTS_ATTRIBUTE: &str = "dstElements";

/// Name of the NVGPU `bypassL1` unit attribute.
pub const BYPASS_L1_ATTRIBUTE: &str = "bypassL1";

/// Name of the attribute storing operand segment sizes for variadic NVGPU operations.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Operation trait for `nvgpu.device_async_copy`.
pub trait DeviceAsyncCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the destination memref operand.
    fn dst(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination index operands.
    fn dst_indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        (1..1 + segment_sizes[1]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the source memref operand.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        self.operand_value(1 + segment_sizes[1]).unwrap()
    }

    /// Returns the source index operands.
    fn src_indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = 1 + segment_sizes[1] + 1;
        (start..start + segment_sizes[3]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the optional source element-count operand.
    fn src_elements(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        if segment_sizes[4] == 0 { None } else { self.operand_value(segment_sizes.iter().take(4).sum::<usize>()) }
    }

    /// Returns the destination element count attribute.
    fn dst_elements(&self) -> i64 {
        self.attribute(DST_ELEMENTS_ATTRIBUTE)
            .unwrap()
            .cast::<IntegerAttributeRef>()
            .unwrap()
            .signless_value()
    }

    /// Returns whether L1 bypass is requested.
    fn bypass_l1(&self) -> bool {
        self.attribute(BYPASS_L1_ATTRIBUTE).is_some()
    }

    /// Returns the asynchronous copy token.
    fn async_token(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DeviceAsyncCopy);
mlir_op_trait!(DeviceAsyncCopy, OneResult);
mlir_op_trait!(DeviceAsyncCopy, ZeroRegions);
mlir_op_trait!(DeviceAsyncCopy, ZeroSuccessors);

/// Constructs a new detached/owned [`DeviceAsyncCopyOperation`] at the specified [`Location`].
pub fn device_async_copy<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    dst: ValueRef<'v, 'c, 't>,
    dst_indices: &[ValueRef<'v, 'c, 't>],
    src: ValueRef<'v, 'c, 't>,
    src_indices: &[ValueRef<'v, 'c, 't>],
    dst_elements: i64,
    src_elements: Option<ValueRef<'v, 'c, 't>>,
    bypass_l1: bool,
    location: L,
) -> DetachedDeviceAsyncCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let segment_sizes = [1, dst_indices.len() as i32, 1, src_indices.len() as i32, i32::from(src_elements.is_some())];
    let mut builder = OperationBuilder::new("nvgpu.device_async_copy", location)
        .add_operand(dst)
        .add_operands(dst_indices)
        .add_operand(src)
        .add_operands(src_indices)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap())
        .add_attribute(DST_ELEMENTS_ATTRIBUTE, context.integer_attribute(context.index_type(), dst_elements));
    if let Some(src_elements) = src_elements {
        builder = builder.add_operand(src_elements);
    }
    if bypass_l1 {
        builder = builder.add_attribute(BYPASS_L1_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(context.nvgpu_device_async_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::device_async_copy`")
}

/// Operation trait for `nvgpu.device_async_create_group`.
pub trait DeviceAsyncCreateGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input asynchronous tokens.
    fn input_tokens(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the group asynchronous token.
    fn async_token(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(DeviceAsyncCreateGroup);
mlir_op_trait!(DeviceAsyncCreateGroup, OneResult);
mlir_op_trait!(DeviceAsyncCreateGroup, ZeroRegions);
mlir_op_trait!(DeviceAsyncCreateGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`DeviceAsyncCreateGroupOperation`] at the specified [`Location`].
pub fn device_async_create_group<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input_tokens: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedDeviceAsyncCreateGroupOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.device_async_create_group", location)
        .add_operands(input_tokens)
        .add_result(context.nvgpu_device_async_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::device_async_create_group`")
}

/// Name of the NVGPU `numGroups` attribute.
pub const NUM_GROUPS_ATTRIBUTE: &str = "numGroups";

/// Operation trait for `nvgpu.device_async_wait`.
pub trait DeviceAsyncWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the asynchronous dependency token.
    fn async_dependency(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional maximum number of incomplete groups.
    fn num_groups(&self) -> Option<i64> {
        self.attribute(NUM_GROUPS_ATTRIBUTE)
            .map(|attribute| attribute.cast::<IntegerAttributeRef>().unwrap().signless_value())
    }
}

mlir_op!(DeviceAsyncWait);
mlir_op_trait!(DeviceAsyncWait, ZeroRegions);
mlir_op_trait!(DeviceAsyncWait, ZeroSuccessors);

/// Constructs a new detached/owned [`DeviceAsyncWaitOperation`] at the specified [`Location`].
pub fn device_async_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    async_dependency: ValueRef<'v, 'c, 't>,
    num_groups: Option<i32>,
    location: L,
) -> DetachedDeviceAsyncWaitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let mut builder = OperationBuilder::new("nvgpu.device_async_wait", location).add_operand(async_dependency);
    if let Some(num_groups) = num_groups {
        builder = builder.add_attribute(
            NUM_GROUPS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), num_groups as i64),
        );
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::device_async_wait`")
}

/// Operation trait for `nvgpu.mbarrier.create`.
pub trait MBarrierCreateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the created mbarrier group.
    fn barriers(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MBarrierCreate);
mlir_op_trait!(MBarrierCreate, OneResult);
mlir_op_trait!(MBarrierCreate, ZeroOperands);
mlir_op_trait!(MBarrierCreate, ZeroRegions);
mlir_op_trait!(MBarrierCreate, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierCreateOperation`] at the specified [`Location`].
pub fn mbarrier_create<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMBarrierCreateOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.create", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_create`")
}

/// Operation trait for `nvgpu.mbarrier.get`.
pub trait MBarrierGetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mbarrier pointer result.
    fn mbarrier_pointer(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MBarrierGet);
mlir_op_trait!(MBarrierGet, OneResult);
mlir_op_trait!(MBarrierGet, ZeroRegions);
mlir_op_trait!(MBarrierGet, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierGetOperation`] at the specified [`Location`].
pub fn mbarrier_get<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMBarrierGetOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.get", location)
        .add_operands(&[barriers, mbar_id])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_get`")
}

/// Operation trait for `nvgpu.mbarrier.init`.
pub trait MBarrierInitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the expected participant count operand.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional predicate operand.
    fn predicate(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(3)
    }
}

mlir_op!(MBarrierInit);
mlir_op_trait!(MBarrierInit, ZeroRegions);
mlir_op_trait!(MBarrierInit, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierInitOperation`] at the specified [`Location`].
pub fn mbarrier_init<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    count: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    predicate: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedMBarrierInitOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    let mut builder = OperationBuilder::new("nvgpu.mbarrier.init", location).add_operands(&[barriers, count, mbar_id]);
    if let Some(predicate) = predicate {
        builder = builder.add_operand(predicate);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_init`")
}

/// Operation trait for `nvgpu.mbarrier.test.wait`.
pub trait MBarrierTestWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mbarrier token operand.
    fn token(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the completion test result.
    fn wait_complete(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MBarrierTestWait);
mlir_op_trait!(MBarrierTestWait, OneResult);
mlir_op_trait!(MBarrierTestWait, ZeroRegions);
mlir_op_trait!(MBarrierTestWait, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierTestWaitOperation`] at the specified [`Location`].
pub fn mbarrier_test_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    token: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedMBarrierTestWaitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.test.wait", location)
        .add_operands(&[barriers, token, mbar_id])
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_test_wait`")
}

/// Operation trait for `nvgpu.mbarrier.arrive`.
pub trait MBarrierArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the produced mbarrier token.
    fn token(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MBarrierArrive);
mlir_op_trait!(MBarrierArrive, OneResult);
mlir_op_trait!(MBarrierArrive, ZeroRegions);
mlir_op_trait!(MBarrierArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierArriveOperation`] at the specified [`Location`].
pub fn mbarrier_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedMBarrierArriveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.arrive", location)
        .add_operands(&[barriers, mbar_id])
        .add_result(context.nvgpu_mbarrier_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_arrive`")
}

/// Operation trait for `nvgpu.mbarrier.arrive.nocomplete`.
pub trait MBarrierArriveNoCompleteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the non-completing arrival count operand.
    fn count(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the produced mbarrier token.
    fn token(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MBarrierArriveNoComplete);
mlir_op_trait!(MBarrierArriveNoComplete, OneResult);
mlir_op_trait!(MBarrierArriveNoComplete, ZeroRegions);
mlir_op_trait!(MBarrierArriveNoComplete, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierArriveNoCompleteOperation`] at the specified [`Location`].
pub fn mbarrier_arrive_nocomplete<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    count: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedMBarrierArriveNoCompleteOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.arrive.nocomplete", location)
        .add_operands(&[barriers, mbar_id, count])
        .add_result(context.nvgpu_mbarrier_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_arrive_nocomplete`")
}

/// Operation trait for `nvgpu.mbarrier.arrive.expect_tx`.
pub trait MBarrierArriveExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the expected transaction count operand.
    fn tx_count(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the optional predicate operand.
    fn predicate(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(3)
    }
}

mlir_op!(MBarrierArriveExpectTx);
mlir_op_trait!(MBarrierArriveExpectTx, ZeroRegions);
mlir_op_trait!(MBarrierArriveExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierArriveExpectTxOperation`] at the specified [`Location`].
pub fn mbarrier_arrive_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    tx_count: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    predicate: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedMBarrierArriveExpectTxOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    let mut builder =
        OperationBuilder::new("nvgpu.mbarrier.arrive.expect_tx", location).add_operands(&[barriers, tx_count, mbar_id]);
    if let Some(predicate) = predicate {
        builder = builder.add_operand(predicate);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_arrive_expect_tx`")
}

/// Operation trait for `nvgpu.mbarrier.try_wait.parity`.
pub trait MBarrierTryWaitParityOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the phase parity operand.
    fn phase_parity(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the tick timeout operand.
    fn ticks(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }
}

mlir_op!(MBarrierTryWaitParity);
mlir_op_trait!(MBarrierTryWaitParity, ZeroRegions);
mlir_op_trait!(MBarrierTryWaitParity, ZeroSuccessors);

/// Constructs a new detached/owned [`MBarrierTryWaitParityOperation`] at the specified [`Location`].
pub fn mbarrier_try_wait_parity<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barriers: ValueRef<'v, 'c, 't>,
    phase_parity: ValueRef<'v, 'c, 't>,
    ticks: ValueRef<'v, 'c, 't>,
    mbar_id: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedMBarrierTryWaitParityOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.mbarrier.try_wait.parity", location)
        .add_operands(&[barriers, phase_parity, ticks, mbar_id])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::mbarrier_try_wait_parity`")
}

/// Operation trait for `nvgpu.tma.fence.descriptor`.
pub trait TmaFenceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor map descriptor operand.
    fn tensor_map_descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(TmaFence);
mlir_op_trait!(TmaFence, ZeroRegions);
mlir_op_trait!(TmaFence, ZeroSuccessors);

/// Constructs a new detached/owned [`TmaFenceOperation`] at the specified [`Location`].
pub fn tma_fence<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor_map_descriptor: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedTmaFenceOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.tma.fence.descriptor", location)
        .add_operand(tensor_map_descriptor)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::tma_fence`")
}

/// Operation trait for `nvgpu.tma.prefetch.descriptor`.
pub trait TmaPrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor map descriptor operand.
    fn tensor_map_descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional predicate operand.
    fn predicate(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(1)
    }
}

mlir_op!(TmaPrefetch);
mlir_op_trait!(TmaPrefetch, ZeroRegions);
mlir_op_trait!(TmaPrefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`TmaPrefetchOperation`] at the specified [`Location`].
pub fn tma_prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor_map_descriptor: ValueRef<'v, 'c, 't>,
    predicate: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedTmaPrefetchOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    let mut builder =
        OperationBuilder::new("nvgpu.tma.prefetch.descriptor", location).add_operand(tensor_map_descriptor);
    if let Some(predicate) = predicate {
        builder = builder.add_operand(predicate);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::tma_prefetch`")
}

/// Operation trait for `nvgpu.tma.async.load`.
pub trait TmaAsyncLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the destination memref operand.
    fn dst(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the mbarrier group operand.
    fn barriers(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the tensor map descriptor operand.
    fn tensor_map_descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the TMA coordinate operands.
    fn coordinates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        let start = 3;
        (start..start + segment_sizes[3]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the mbarrier index operand.
    fn mbar_id(&self) -> ValueRef<'o, 'c, 't> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        self.operand_value(3 + segment_sizes[3]).unwrap()
    }

    /// Returns the optional multicast mask operand.
    fn multicast_mask(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        if segment_sizes[5] == 0 { None } else { self.operand_value(segment_sizes.iter().take(5).sum::<usize>()) }
    }

    /// Returns the optional predicate operand.
    fn predicate(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        if segment_sizes[6] == 0 { None } else { self.operand_value(segment_sizes.iter().take(6).sum::<usize>()) }
    }
}

mlir_op!(TmaAsyncLoad);
mlir_op_trait!(TmaAsyncLoad, ZeroRegions);
mlir_op_trait!(TmaAsyncLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`TmaAsyncLoadOperation`] at the specified [`Location`].
pub fn tma_async_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    dst: ValueRef<'v, 'c, 't>,
    barriers: ValueRef<'v, 'c, 't>,
    tensor_map_descriptor: ValueRef<'v, 'c, 't>,
    coordinates: &[ValueRef<'v, 'c, 't>],
    mbar_id: ValueRef<'v, 'c, 't>,
    multicast_mask: Option<ValueRef<'v, 'c, 't>>,
    predicate: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedTmaAsyncLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let segment_sizes =
        [1, 1, 1, coordinates.len() as i32, 1, i32::from(multicast_mask.is_some()), i32::from(predicate.is_some())];
    let mut builder = OperationBuilder::new("nvgpu.tma.async.load", location)
        .add_operands(&[dst, barriers, tensor_map_descriptor])
        .add_operands(coordinates)
        .add_operand(mbar_id)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap());
    if let Some(multicast_mask) = multicast_mask {
        builder = builder.add_operand(multicast_mask);
    }
    if let Some(predicate) = predicate {
        builder = builder.add_operand(predicate);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::tma_async_load`")
}

/// Operation trait for `nvgpu.tma.async.store`.
pub trait TmaAsyncStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref operand.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor map descriptor operand.
    fn tensor_map_descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the TMA coordinate operands.
    fn coordinates(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        (2..2 + segment_sizes[2]).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the optional predicate operand.
    fn predicate(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let segment_sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .unwrap()
            .cast::<DenseInteger32ArrayAttributeRef>()
            .unwrap()
            .values()
            .map(|value| value as usize)
            .collect::<Vec<_>>();
        if segment_sizes[3] == 0 { None } else { self.operand_value(segment_sizes.iter().take(3).sum::<usize>()) }
    }
}

mlir_op!(TmaAsyncStore);
mlir_op_trait!(TmaAsyncStore, ZeroRegions);
mlir_op_trait!(TmaAsyncStore, ZeroSuccessors);

/// Constructs a new detached/owned [`TmaAsyncStoreOperation`] at the specified [`Location`].
pub fn tma_async_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'v, 'c, 't>,
    tensor_map_descriptor: ValueRef<'v, 'c, 't>,
    coordinates: &[ValueRef<'v, 'c, 't>],
    predicate: Option<ValueRef<'v, 'c, 't>>,
    location: L,
) -> DetachedTmaAsyncStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let segment_sizes = [1, 1, coordinates.len() as i32, i32::from(predicate.is_some())];
    let mut builder = OperationBuilder::new("nvgpu.tma.async.store", location)
        .add_operands(&[src, tensor_map_descriptor])
        .add_operands(coordinates)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&segment_sizes).unwrap());
    if let Some(predicate) = predicate {
        builder = builder.add_operand(predicate);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::tma_async_store`")
}

/// Operation trait for `nvgpu.tma.create.descriptor`.
pub trait TmaCreateDescriptorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor operand.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the box dimension operands.
    fn box_dimensions(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the tensor map result.
    fn tensor_map(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(TmaCreateDescriptor);
mlir_op_trait!(TmaCreateDescriptor, OneResult);
mlir_op_trait!(TmaCreateDescriptor, ZeroRegions);
mlir_op_trait!(TmaCreateDescriptor, ZeroSuccessors);

/// Constructs a new detached/owned [`TmaCreateDescriptorOperation`] at the specified [`Location`].
pub fn tma_create_descriptor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    box_dimensions: &[ValueRef<'v, 'c, 't>],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTmaCreateDescriptorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.tma.create.descriptor", location)
        .add_operand(tensor)
        .add_operands(box_dimensions)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::tma_create_descriptor`")
}

/// Operation trait for `nvgpu.warpgroup.generate.descriptor`.
pub trait WarpgroupGenerateDescriptorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor operand.
    fn tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor map descriptor operand.
    fn tensor_map(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the warpgroup descriptor result.
    fn descriptor(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(WarpgroupGenerateDescriptor);
mlir_op_trait!(WarpgroupGenerateDescriptor, OneResult);
mlir_op_trait!(WarpgroupGenerateDescriptor, ZeroRegions);
mlir_op_trait!(WarpgroupGenerateDescriptor, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpgroupGenerateDescriptorOperation`] at the specified [`Location`].
pub fn warpgroup_generate_descriptor<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tensor: ValueRef<'v, 'c, 't>,
    tensor_map: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWarpgroupGenerateDescriptorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.warpgroup.generate.descriptor", location)
        .add_operands(&[tensor, tensor_map])
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::warpgroup_generate_descriptor`")
}

/// Name of the NVGPU `waitGroup` attribute.
pub const WAIT_GROUP_ATTRIBUTE: &str = "waitGroup";

/// Name of the NVGPU `transposeA` unit attribute.
pub const TRANSPOSE_A_ATTRIBUTE: &str = "transposeA";

/// Name of the NVGPU `transposeB` unit attribute.
pub const TRANSPOSE_B_ATTRIBUTE: &str = "transposeB";

/// Operation trait for `nvgpu.warpgroup.mma`.
pub trait WarpgroupMmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the first warpgroup matrix descriptor.
    fn descriptor_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the second warpgroup matrix descriptor.
    fn descriptor_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the accumulator input.
    fn matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the wait-group count.
    fn wait_group(&self) -> i64 {
        self.attribute(WAIT_GROUP_ATTRIBUTE)
            .map(|attribute| attribute.cast::<IntegerAttributeRef>().unwrap().signless_value())
            .unwrap_or(1)
    }

    /// Returns whether descriptor A is transposed.
    fn transpose_a(&self) -> bool {
        self.attribute(TRANSPOSE_A_ATTRIBUTE).is_some()
    }

    /// Returns whether descriptor B is transposed.
    fn transpose_b(&self) -> bool {
        self.attribute(TRANSPOSE_B_ATTRIBUTE).is_some()
    }

    /// Returns the accumulator output.
    fn matrix_d(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(WarpgroupMma);
mlir_op_trait!(WarpgroupMma, OneResult);
mlir_op_trait!(WarpgroupMma, ZeroRegions);
mlir_op_trait!(WarpgroupMma, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpgroupMmaOperation`] at the specified [`Location`].
pub fn warpgroup_mma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    descriptor_a: ValueRef<'v, 'c, 't>,
    descriptor_b: ValueRef<'v, 'c, 't>,
    matrix_c: ValueRef<'v, 'c, 't>,
    wait_group: Option<i64>,
    transpose_a: bool,
    transpose_b: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWarpgroupMmaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let mut builder =
        OperationBuilder::new("nvgpu.warpgroup.mma", location).add_operands(&[descriptor_a, descriptor_b, matrix_c]);
    if let Some(wait_group) = wait_group {
        builder = builder.add_attribute(
            WAIT_GROUP_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), wait_group),
        );
    }
    if transpose_a {
        builder = builder.add_attribute(TRANSPOSE_A_ATTRIBUTE, context.unit_attribute());
    }
    if transpose_b {
        builder = builder.add_attribute(TRANSPOSE_B_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::warpgroup_mma`")
}

/// Operation trait for `nvgpu.warpgroup.mma.store`.
pub trait WarpgroupMmaStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator operand.
    fn matrix_d(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref operand.
    fn dst_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(WarpgroupMmaStore);
mlir_op_trait!(WarpgroupMmaStore, ZeroRegions);
mlir_op_trait!(WarpgroupMmaStore, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpgroupMmaStoreOperation`] at the specified [`Location`].
pub fn warpgroup_mma_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    matrix_d: ValueRef<'v, 'c, 't>,
    dst_memref: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedWarpgroupMmaStoreOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.warpgroup.mma.store", location)
        .add_operands(&[matrix_d, dst_memref])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::warpgroup_mma_store`")
}

/// Operation trait for `nvgpu.warpgroup.mma.init.accumulator`.
pub trait WarpgroupMmaInitAccumulatorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the initialized accumulator result.
    fn matrix_c(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(WarpgroupMmaInitAccumulator);
mlir_op_trait!(WarpgroupMmaInitAccumulator, OneResult);
mlir_op_trait!(WarpgroupMmaInitAccumulator, ZeroOperands);
mlir_op_trait!(WarpgroupMmaInitAccumulator, ZeroRegions);
mlir_op_trait!(WarpgroupMmaInitAccumulator, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpgroupMmaInitAccumulatorOperation`] at the specified [`Location`].
pub fn warpgroup_mma_init_accumulator<'c, 't: 'c, L: Location<'c, 't>>(
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWarpgroupMmaInitAccumulatorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::nvgpu());
    OperationBuilder::new("nvgpu.warpgroup.mma.init.accumulator", location)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::warpgroup_mma_init_accumulator`")
}

/// Name of the NVGPU `rounding` attribute.
pub const ROUNDING_ATTRIBUTE: &str = "rounding";

/// Name of the NVGPU `ftz` unit attribute.
pub const FTZ_ATTRIBUTE: &str = "ftz";

/// Operation trait for `nvgpu.rcp`.
pub trait RcpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input vector operand.
    fn input(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reciprocal rounding mode.
    fn rounding(&self) -> RcpRoundingModeAttributeRef<'c, 't> {
        self.attribute(ROUNDING_ATTRIBUTE).unwrap().cast::<RcpRoundingModeAttributeRef>().unwrap()
    }

    /// Returns whether flush-to-zero behavior is enabled.
    fn ftz(&self) -> bool {
        self.attribute(FTZ_ATTRIBUTE).is_some()
    }

    /// Returns the reciprocal result.
    fn output(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Rcp);
mlir_op_trait!(Rcp, OneResult);
mlir_op_trait!(Rcp, ZeroRegions);
mlir_op_trait!(Rcp, ZeroSuccessors);

/// Constructs a new detached/owned [`RcpOperation`] at the specified [`Location`].
pub fn rcp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    input: ValueRef<'v, 'c, 't>,
    rounding: RcpRoundingMode,
    ftz: bool,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedRcpOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvgpu());
    let mut builder = OperationBuilder::new("nvgpu.rcp", location)
        .add_operand(input)
        .add_attribute(ROUNDING_ATTRIBUTE, context.nvgpu_rcp_rounding_mode_attribute(rounding));
    if ftz {
        builder = builder.add_attribute(FTZ_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `nvgpu::rcp`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::gpu::AddressSpace;
    use crate::{Block, Context, DialectHandle, Operation, Size, Type, TypeRef, Value, ValueRef, VectorTypeDimension};

    use super::super::attributes::{
        TensorMapInterleaveKind, TensorMapL2PromoKind, TensorMapOobKind, TensorMapSwizzleKind,
    };

    use super::*;

    /// Common types used by NVGPU operation wrapper tests.
    #[derive(Copy, Clone)]
    struct TestTypes<'c, 't> {
        /// One-bit signless integer type.
        i1: TypeRef<'c, 't>,

        /// Sixteen-bit signless integer type.
        i16: TypeRef<'c, 't>,

        /// Sixty-four-bit signless integer type.
        i64: TypeRef<'c, 't>,

        /// Index type.
        index: TypeRef<'c, 't>,

        /// Shared-memory 16-bit floating-point memref type.
        shared_f16_memref: TypeRef<'c, 't>,

        /// Global 16-bit floating-point memref type.
        global_f16_memref: TypeRef<'c, 't>,

        /// Global 32-bit floating-point memref type.
        global_f32_memref: TypeRef<'c, 't>,

        /// Unranked 32-bit floating-point memref type.
        unranked_f32_memref: TypeRef<'c, 't>,

        /// Warp-level vector matrix fragment type.
        vector_f16_4x2: TypeRef<'c, 't>,

        /// Warp-level vector matrix fragment type.
        vector_f16_2x2: TypeRef<'c, 't>,

        /// Warp-level accumulator vector type.
        vector_f32_2x2: TypeRef<'c, 't>,

        /// Sparse MMA metadata vector type.
        sparse_metadata: TypeRef<'c, 't>,

        /// NVGPU device asynchronous token type.
        device_async_token: TypeRef<'c, 't>,

        /// NVGPU mbarrier group type.
        mbarrier_group: TypeRef<'c, 't>,

        /// NVGPU mbarrier token type.
        mbarrier_token: TypeRef<'c, 't>,

        /// NVGPU tensor map descriptor type.
        tensor_map_descriptor: TypeRef<'c, 't>,

        /// NVGPU warpgroup matrix descriptor type.
        warpgroup_descriptor: TypeRef<'c, 't>,

        /// Second NVGPU warpgroup matrix descriptor type.
        warpgroup_descriptor_b: TypeRef<'c, 't>,

        /// Warpgroup accumulator destination memref type.
        warpgroup_f32_memref: TypeRef<'c, 't>,

        /// NVGPU warpgroup accumulator type.
        warpgroup_accumulator: TypeRef<'c, 't>,
    }

    impl<'c, 't> TestTypes<'c, 't> {
        /// Builds the common test type set in `context`.
        fn new(context: &'c Context<'t>, location: impl crate::Location<'c, 't>) -> Self {
            context.load_dialect(DialectHandle::gpu());
            context.load_dialect(DialectHandle::nvgpu());
            let i1 = context.signless_integer_type(1);
            let i16 = context.signless_integer_type(16);
            let i64 = context.signless_integer_type(64);
            let index = context.index_type();
            let f16 = context.float16_type();
            let f32 = context.float32_type();
            let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup);
            let shared_memory_space = context.integer_attribute(context.signless_integer_type(64), 3);
            let shared_f16_memref = context
                .mem_ref_type(
                    f16,
                    &[Size::Static(32), Size::Static(32)],
                    None,
                    Some(shared_memory_space.as_ref()),
                    location,
                )
                .unwrap();
            let global_f16_memref = context.mem_ref_type(f16, &[Size::Static(64)], None, None, location).unwrap();
            let global_f32_memref = context
                .mem_ref_type(
                    f32,
                    &[Size::Static(32), Size::Static(32)],
                    None,
                    Some(shared_memory_space.as_ref()),
                    location,
                )
                .unwrap();
            let unranked_f32_memref = context.unranked_mem_ref_type(f32, None, location).unwrap();
            let vector_f16_4x2 = context
                .vector_type(f16, &[VectorTypeDimension::Fixed(4), VectorTypeDimension::Fixed(2)], location)
                .unwrap();
            let vector_f16_2x2 = context
                .vector_type(f16, &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(2)], location)
                .unwrap();
            let vector_f32_2x2 = context
                .vector_type(f32, &[VectorTypeDimension::Fixed(2), VectorTypeDimension::Fixed(2)], location)
                .unwrap();
            let sparse_metadata = context.vector_type(i16, &[VectorTypeDimension::Fixed(2)], location).unwrap();
            let mbarrier_group = context.nvgpu_mbarrier_group_type(workgroup, 4);
            let tensor_map_descriptor = context.nvgpu_tensor_map_descriptor_type(
                global_f32_memref,
                TensorMapSwizzleKind::Swizzle128B,
                TensorMapL2PromoKind::None,
                TensorMapOobKind::Zero,
                TensorMapInterleaveKind::None,
            );
            let warpgroup_a_memref = context
                .mem_ref_type(
                    f16,
                    &[Size::Static(64), Size::Static(64)],
                    None,
                    Some(shared_memory_space.as_ref()),
                    location,
                )
                .unwrap();
            let warpgroup_b_memref = context
                .mem_ref_type(
                    f16,
                    &[Size::Static(64), Size::Static(128)],
                    None,
                    Some(shared_memory_space.as_ref()),
                    location,
                )
                .unwrap();
            let warpgroup_f32_memref = context
                .mem_ref_type(
                    f32,
                    &[Size::Static(64), Size::Static(128)],
                    None,
                    Some(shared_memory_space.as_ref()),
                    location,
                )
                .unwrap();
            let warpgroup_descriptor = context.nvgpu_warpgroup_matrix_descriptor_type(warpgroup_a_memref);
            let warpgroup_descriptor_b = context.nvgpu_warpgroup_matrix_descriptor_type(warpgroup_b_memref);
            let warpgroup_fragmented = context
                .vector_type(f32, &[VectorTypeDimension::Fixed(64), VectorTypeDimension::Fixed(128)], location)
                .unwrap();
            let warpgroup_accumulator = context.nvgpu_warpgroup_accumulator_type(warpgroup_fragmented);

            Self {
                i1: i1.as_ref(),
                i16: i16.as_ref(),
                i64: i64.as_ref(),
                index: index.as_ref(),
                shared_f16_memref: shared_f16_memref.as_ref(),
                global_f16_memref: global_f16_memref.as_ref(),
                global_f32_memref: global_f32_memref.as_ref(),
                unranked_f32_memref: unranked_f32_memref.as_ref(),
                vector_f16_4x2: vector_f16_4x2.as_ref(),
                vector_f16_2x2: vector_f16_2x2.as_ref(),
                vector_f32_2x2: vector_f32_2x2.as_ref(),
                sparse_metadata: sparse_metadata.as_ref(),
                device_async_token: context.nvgpu_device_async_token_type().as_ref(),
                mbarrier_group: mbarrier_group.as_ref(),
                mbarrier_token: context.nvgpu_mbarrier_token_type().as_ref(),
                tensor_map_descriptor: tensor_map_descriptor.as_ref(),
                warpgroup_descriptor: warpgroup_descriptor.as_ref(),
                warpgroup_descriptor_b: warpgroup_descriptor_b.as_ref(),
                warpgroup_f32_memref: warpgroup_f32_memref.as_ref(),
                warpgroup_accumulator: warpgroup_accumulator.as_ref(),
            }
        }
    }

    /// Common block argument values used by NVGPU operation wrapper tests.
    #[derive(Copy, Clone)]
    struct TestValues<'v, 'c, 't> {
        /// Shared-memory 16-bit floating-point memref value.
        shared_f16_memref: ValueRef<'v, 'c, 't>,

        /// Global 16-bit floating-point memref value.
        global_f16_memref: ValueRef<'v, 'c, 't>,

        /// Global 32-bit floating-point memref value.
        global_f32_memref: ValueRef<'v, 'c, 't>,

        /// Unranked 32-bit floating-point memref value.
        unranked_f32_memref: ValueRef<'v, 'c, 't>,

        /// First index value.
        index_0: ValueRef<'v, 'c, 't>,

        /// Second index value.
        index_1: ValueRef<'v, 'c, 't>,

        /// Third index value.
        index_2: ValueRef<'v, 'c, 't>,

        /// Predicate value.
        predicate: ValueRef<'v, 'c, 't>,

        /// Sixteen-bit integer value.
        i16: ValueRef<'v, 'c, 't>,

        /// First 16-bit floating-point vector value.
        vector_a: ValueRef<'v, 'c, 't>,

        /// Second 16-bit floating-point vector value.
        vector_b: ValueRef<'v, 'c, 't>,

        /// Accumulator vector value.
        vector_c: ValueRef<'v, 'c, 't>,

        /// Sparse MMA metadata value.
        sparse_metadata: ValueRef<'v, 'c, 't>,

        /// Device asynchronous token value.
        device_async_token: ValueRef<'v, 'c, 't>,

        /// Mbarrier group value.
        mbarrier_group: ValueRef<'v, 'c, 't>,

        /// Mbarrier token value.
        mbarrier_token: ValueRef<'v, 'c, 't>,

        /// Tensor map descriptor value.
        tensor_map_descriptor: ValueRef<'v, 'c, 't>,

        /// First warpgroup descriptor value.
        warpgroup_descriptor_a: ValueRef<'v, 'c, 't>,

        /// Second warpgroup descriptor value.
        warpgroup_descriptor_b: ValueRef<'v, 'c, 't>,

        /// Warpgroup accumulator value.
        warpgroup_accumulator: ValueRef<'v, 'c, 't>,
    }

    impl<'v, 'c, 't> TestValues<'v, 'c, 't> {
        /// Builds the common test values from `block`.
        fn new(block: &'v impl Block<'v, 'c, 't>) -> Self {
            Self {
                shared_f16_memref: block.argument(0).unwrap().as_ref(),
                global_f16_memref: block.argument(1).unwrap().as_ref(),
                global_f32_memref: block.argument(2).unwrap().as_ref(),
                unranked_f32_memref: block.argument(3).unwrap().as_ref(),
                index_0: block.argument(4).unwrap().as_ref(),
                index_1: block.argument(5).unwrap().as_ref(),
                index_2: block.argument(6).unwrap().as_ref(),
                predicate: block.argument(7).unwrap().as_ref(),
                i16: block.argument(8).unwrap().as_ref(),
                vector_a: block.argument(9).unwrap().as_ref(),
                vector_b: block.argument(10).unwrap().as_ref(),
                vector_c: block.argument(11).unwrap().as_ref(),
                sparse_metadata: block.argument(12).unwrap().as_ref(),
                device_async_token: block.argument(13).unwrap().as_ref(),
                mbarrier_group: block.argument(14).unwrap().as_ref(),
                mbarrier_token: block.argument(15).unwrap().as_ref(),
                tensor_map_descriptor: block.argument(16).unwrap().as_ref(),
                warpgroup_descriptor_a: block.argument(17).unwrap().as_ref(),
                warpgroup_descriptor_b: block.argument(18).unwrap().as_ref(),
                warpgroup_accumulator: block.argument(19).unwrap().as_ref(),
            }
        }
    }

    macro_rules! nvgpu_operation_test {
        ($test_name:ident, |$context:ident, $location:ident, $values:ident, $types:ident| $body:block $(,)?) => {
            #[test]
            fn $test_name() {
                let $context = Context::new();
                let $location = $context.unknown_location();
                let $types = TestTypes::new(&$context, $location);
                let block = $context.block(&[
                    ($types.shared_f16_memref, $location),
                    ($types.global_f16_memref, $location),
                    ($types.global_f32_memref, $location),
                    ($types.unranked_f32_memref, $location),
                    ($types.index, $location),
                    ($types.index, $location),
                    ($types.index, $location),
                    ($types.i1, $location),
                    ($types.i16, $location),
                    ($types.vector_f16_4x2, $location),
                    ($types.vector_f16_2x2, $location),
                    ($types.vector_f32_2x2, $location),
                    ($types.sparse_metadata, $location),
                    ($types.device_async_token, $location),
                    ($types.mbarrier_group, $location),
                    ($types.mbarrier_token, $location),
                    ($types.tensor_map_descriptor, $location),
                    ($types.warpgroup_descriptor, $location),
                    ($types.warpgroup_descriptor_b, $location),
                    ($types.warpgroup_accumulator, $location),
                ]);
                let $values = TestValues::new(&block);

                $body
            }
        };
    }

    nvgpu_operation_test!(test_ldmatrix_operation, |_context, location, values, types| {
        let operation = ldmatrix(
            values.shared_f16_memref,
            &[values.index_0, values.index_1],
            true,
            4,
            types.vector_f16_4x2,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.ldmatrix"));
        assert_eq!(operation.src_memref(), values.shared_f16_memref);
        assert_eq!(operation.indices(), vec![values.index_0, values.index_1]);
        assert!(operation.transpose());
        assert_eq!(operation.num_tiles(), 4);
        assert_eq!(operation.matrix().r#type(), types.vector_f16_4x2);
    });

    nvgpu_operation_test!(test_mma_sync_operation, |_context, location, values, types| {
        let operation = mma_sync(
            values.vector_a,
            values.vector_b,
            values.vector_c,
            &[16, 8, 16],
            true,
            types.vector_f32_2x2,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mma.sync"));
        assert_eq!(operation.matrix_a(), values.vector_a);
        assert_eq!(operation.matrix_b(), values.vector_b);
        assert_eq!(operation.matrix_c(), values.vector_c);
        assert_eq!(operation.mma_shape(), vec![16, 8, 16]);
        assert!(operation.tf32_enabled());
        assert_eq!(operation.matrix_d().r#type(), types.vector_f32_2x2);
    });

    nvgpu_operation_test!(test_mma_sparse_sync_operation, |_context, location, values, types| {
        let operation = mma_sparse_sync(
            values.vector_a,
            values.vector_b,
            values.vector_c,
            values.sparse_metadata,
            &[16, 8, 32],
            1,
            true,
            types.vector_f32_2x2,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mma.sp.sync"));
        assert_eq!(operation.matrix_a(), values.vector_a);
        assert_eq!(operation.matrix_b(), values.vector_b);
        assert_eq!(operation.matrix_c(), values.vector_c);
        assert_eq!(operation.sparse_metadata(), values.sparse_metadata);
        assert_eq!(operation.mma_shape(), vec![16, 8, 32]);
        assert_eq!(operation.sparsity_selector(), 1);
        assert!(operation.tf32_enabled());
        assert_eq!(operation.matrix_d().r#type(), types.vector_f32_2x2);
    });

    nvgpu_operation_test!(test_device_async_copy_operation, |_context, location, values, types| {
        let operation = device_async_copy(
            values.shared_f16_memref,
            &[values.index_0, values.index_1],
            values.global_f16_memref,
            &[values.index_2],
            4,
            Some(values.index_0),
            true,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.device_async_copy"));
        assert_eq!(operation.dst(), values.shared_f16_memref);
        assert_eq!(operation.dst_indices(), vec![values.index_0, values.index_1]);
        assert_eq!(operation.src(), values.global_f16_memref);
        assert_eq!(operation.src_indices(), vec![values.index_2]);
        assert_eq!(operation.src_elements(), Some(values.index_0));
        assert_eq!(operation.dst_elements(), 4);
        assert!(operation.bypass_l1());
        assert_eq!(operation.async_token().r#type(), types.device_async_token);
    });

    nvgpu_operation_test!(test_device_async_create_group_operation, |_context, location, values, types| {
        let operation = device_async_create_group(&[values.device_async_token], location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.device_async_create_group"));
        assert_eq!(operation.input_tokens(), vec![values.device_async_token]);
        assert_eq!(operation.async_token().r#type(), types.device_async_token);
    });

    nvgpu_operation_test!(test_device_async_wait_operation, |_context, location, values, _types| {
        let operation = device_async_wait(values.device_async_token, Some(2), location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.device_async_wait"));
        assert_eq!(operation.async_dependency(), values.device_async_token);
        assert_eq!(operation.num_groups(), Some(2));
    });

    nvgpu_operation_test!(test_mbarrier_create_operation, |_context, location, _values, types| {
        let operation = mbarrier_create(types.mbarrier_group, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.create"));
        assert_eq!(operation.barriers().r#type(), types.mbarrier_group);
    });

    nvgpu_operation_test!(test_mbarrier_get_operation, |_context, location, values, types| {
        let operation = mbarrier_get(values.mbarrier_group, values.index_0, types.i64, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.get"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.mbarrier_pointer().r#type(), types.i64);
    });

    nvgpu_operation_test!(test_mbarrier_init_operation, |_context, location, values, _types| {
        let operation =
            mbarrier_init(values.mbarrier_group, values.index_1, values.index_0, Some(values.predicate), location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.init"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.count(), values.index_1);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.predicate(), Some(values.predicate));
    });

    nvgpu_operation_test!(test_mbarrier_test_wait_operation, |_context, location, values, types| {
        let operation = mbarrier_test_wait(values.mbarrier_group, values.mbarrier_token, values.index_0, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.test.wait"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.token(), values.mbarrier_token);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.wait_complete().r#type(), types.i1);
    });

    nvgpu_operation_test!(test_mbarrier_arrive_operation, |_context, location, values, types| {
        let operation = mbarrier_arrive(values.mbarrier_group, values.index_0, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.arrive"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.token().r#type(), types.mbarrier_token);
    });

    nvgpu_operation_test!(test_mbarrier_arrive_nocomplete_operation, |_context, location, values, types| {
        let operation = mbarrier_arrive_nocomplete(values.mbarrier_group, values.index_0, values.index_1, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.arrive.nocomplete"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.count(), values.index_1);
        assert_eq!(operation.token().r#type(), types.mbarrier_token);
    });

    nvgpu_operation_test!(test_mbarrier_arrive_expect_tx_operation, |_context, location, values, _types| {
        let operation = mbarrier_arrive_expect_tx(
            values.mbarrier_group,
            values.index_2,
            values.index_0,
            Some(values.predicate),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.arrive.expect_tx"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.tx_count(), values.index_2);
        assert_eq!(operation.mbar_id(), values.index_0);
        assert_eq!(operation.predicate(), Some(values.predicate));
    });

    nvgpu_operation_test!(test_mbarrier_try_wait_parity_operation, |_context, location, values, _types| {
        let operation =
            mbarrier_try_wait_parity(values.mbarrier_group, values.predicate, values.index_2, values.index_0, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.mbarrier.try_wait.parity"));
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.phase_parity(), values.predicate);
        assert_eq!(operation.ticks(), values.index_2);
        assert_eq!(operation.mbar_id(), values.index_0);
    });

    nvgpu_operation_test!(test_tma_fence_operation, |_context, location, values, _types| {
        let operation = tma_fence(values.tensor_map_descriptor, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.tma.fence.descriptor"));
        assert_eq!(operation.tensor_map_descriptor(), values.tensor_map_descriptor);
    });

    nvgpu_operation_test!(test_tma_prefetch_operation, |_context, location, values, _types| {
        let operation = tma_prefetch(values.tensor_map_descriptor, Some(values.predicate), location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.tma.prefetch.descriptor"));
        assert_eq!(operation.tensor_map_descriptor(), values.tensor_map_descriptor);
        assert_eq!(operation.predicate(), Some(values.predicate));
    });

    nvgpu_operation_test!(test_tma_async_load_operation, |_context, location, values, _types| {
        let operation = tma_async_load(
            values.shared_f16_memref,
            values.mbarrier_group,
            values.tensor_map_descriptor,
            &[values.index_0, values.index_1],
            values.index_2,
            Some(values.i16),
            Some(values.predicate),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.tma.async.load"));
        assert_eq!(operation.dst(), values.shared_f16_memref);
        assert_eq!(operation.barriers(), values.mbarrier_group);
        assert_eq!(operation.tensor_map_descriptor(), values.tensor_map_descriptor);
        assert_eq!(operation.coordinates(), vec![values.index_0, values.index_1]);
        assert_eq!(operation.mbar_id(), values.index_2);
        assert_eq!(operation.multicast_mask(), Some(values.i16));
        assert_eq!(operation.predicate(), Some(values.predicate));
    });

    nvgpu_operation_test!(test_tma_async_store_operation, |_context, location, values, _types| {
        let operation = tma_async_store(
            values.global_f32_memref,
            values.tensor_map_descriptor,
            &[values.index_0, values.index_1],
            Some(values.predicate),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.tma.async.store"));
        assert_eq!(operation.src(), values.global_f32_memref);
        assert_eq!(operation.tensor_map_descriptor(), values.tensor_map_descriptor);
        assert_eq!(operation.coordinates(), vec![values.index_0, values.index_1]);
        assert_eq!(operation.predicate(), Some(values.predicate));
    });

    nvgpu_operation_test!(test_tma_create_descriptor_operation, |_context, location, values, types| {
        let operation = tma_create_descriptor(
            values.unranked_f32_memref,
            &[values.index_0, values.index_1],
            types.tensor_map_descriptor,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.tma.create.descriptor"));
        assert_eq!(operation.tensor(), values.unranked_f32_memref);
        assert_eq!(operation.box_dimensions(), vec![values.index_0, values.index_1]);
        assert_eq!(operation.tensor_map().r#type(), types.tensor_map_descriptor);
    });

    nvgpu_operation_test!(test_warpgroup_generate_descriptor_operation, |_context, location, values, types| {
        let operation = warpgroup_generate_descriptor(
            values.shared_f16_memref,
            values.tensor_map_descriptor,
            types.warpgroup_descriptor,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.warpgroup.generate.descriptor"));
        assert_eq!(operation.tensor(), values.shared_f16_memref);
        assert_eq!(operation.tensor_map(), values.tensor_map_descriptor);
        assert_eq!(operation.descriptor().r#type(), types.warpgroup_descriptor);
    });

    nvgpu_operation_test!(test_warpgroup_mma_operation, |_context, location, values, types| {
        let operation = warpgroup_mma(
            values.warpgroup_descriptor_a,
            values.warpgroup_descriptor_b,
            values.warpgroup_accumulator,
            Some(2),
            true,
            true,
            types.warpgroup_accumulator,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("nvgpu.warpgroup.mma"));
        assert_eq!(operation.descriptor_a(), values.warpgroup_descriptor_a);
        assert_eq!(operation.descriptor_b(), values.warpgroup_descriptor_b);
        assert_eq!(operation.matrix_c(), values.warpgroup_accumulator);
        assert_eq!(operation.wait_group(), 2);
        assert!(operation.transpose_a());
        assert!(operation.transpose_b());
        assert_eq!(operation.matrix_d().r#type(), types.warpgroup_accumulator);
    });

    nvgpu_operation_test!(test_warpgroup_mma_store_operation, |_context, location, values, _types| {
        let operation = warpgroup_mma_store(values.warpgroup_accumulator, values.global_f32_memref, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.warpgroup.mma.store"));
        assert_eq!(operation.matrix_d(), values.warpgroup_accumulator);
        assert_eq!(operation.dst_memref(), values.global_f32_memref);
    });

    nvgpu_operation_test!(test_warpgroup_mma_init_accumulator_operation, |_context, location, _values, types| {
        let operation = warpgroup_mma_init_accumulator(types.warpgroup_accumulator, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.warpgroup.mma.init.accumulator"));
        assert_eq!(operation.matrix_c().r#type(), types.warpgroup_accumulator);
    });

    nvgpu_operation_test!(test_rcp_operation, |_context, location, values, types| {
        let operation = rcp(values.vector_c, RcpRoundingMode::Approx, true, types.vector_f32_2x2, location);

        assert_eq!(operation.name().as_str(), Ok("nvgpu.rcp"));
        assert_eq!(operation.input(), values.vector_c);
        assert_eq!(operation.rounding().value(), RcpRoundingMode::Approx);
        assert!(operation.ftz());
        assert_eq!(operation.output().r#type(), types.vector_f32_2x2);
    });

    #[test]
    fn test_rcp_operation_module_verification() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        context.load_dialect(DialectHandle::nvgpu());
        let location = context.unknown_location();
        let module = context.module(location);
        let vector_type =
            context.vector_type(context.float32_type(), &[VectorTypeDimension::Fixed(2)], location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(vector_type, location)]);
            let operation =
                rcp(block.argument(0).unwrap().as_ref(), RcpRoundingMode::Approx, true, vector_type.as_ref(), location);
            assert_eq!(operation.input(), block.argument(0).unwrap().as_ref());
            assert_eq!(operation.rounding().value(), RcpRoundingMode::Approx);
            assert!(operation.ftz());
            let operation = block.append_operation(operation);
            block.append_operation(func::r#return(&[operation.result(0).unwrap()], location));
            func::func(
                "nvgpu_rcp",
                func::FuncAttributes {
                    arguments: vec![vector_type.into()],
                    results: vec![vector_type.into()],
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
                  func.func @nvgpu_rcp(%arg0: vector<2xf32>) -> vector<2xf32> {
                    %0 = nvgpu.rcp %arg0{rounding = approx, ftz} : vector<2xf32>
                    return %0 : vector<2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_nvgpu_operations_module_verification() {
        let context = Context::new();
        context.load_dialect(DialectHandle::func());
        context.load_dialect(DialectHandle::gpu());
        context.load_dialect(DialectHandle::nvgpu());
        let location = context.unknown_location();
        let module = context.module(location);
        let types = TestTypes::new(&context, location);
        module.body().append_operation({
            let mut block = context.block(&[
                (types.shared_f16_memref, location),
                (types.global_f16_memref, location),
                (types.global_f32_memref, location),
                (types.unranked_f32_memref, location),
                (types.index, location),
                (types.index, location),
                (types.index, location),
                (types.i1, location),
                (types.i16, location),
                (types.vector_f16_4x2, location),
                (types.vector_f16_2x2, location),
                (types.vector_f32_2x2, location),
                (types.sparse_metadata, location),
                (types.device_async_token, location),
                (types.mbarrier_group, location),
                (types.mbarrier_token, location),
                (types.tensor_map_descriptor, location),
                (types.warpgroup_descriptor, location),
                (types.warpgroup_descriptor_b, location),
                (types.warpgroup_accumulator, location),
                (types.warpgroup_f32_memref, location),
            ]);
            let shared_f16_memref = block.argument(0).unwrap().as_ref();
            let global_f16_memref = block.argument(1).unwrap().as_ref();
            let global_f32_memref = block.argument(2).unwrap().as_ref();
            let unranked_f32_memref = block.argument(3).unwrap().as_ref();
            let index_0 = block.argument(4).unwrap().as_ref();
            let index_1 = block.argument(5).unwrap().as_ref();
            let index_2 = block.argument(6).unwrap().as_ref();
            let predicate = block.argument(7).unwrap().as_ref();
            let i16 = block.argument(8).unwrap().as_ref();
            let vector_a = block.argument(9).unwrap().as_ref();
            let vector_b = block.argument(10).unwrap().as_ref();
            let vector_c = block.argument(11).unwrap().as_ref();
            let sparse_metadata = block.argument(12).unwrap().as_ref();
            let device_async_token = block.argument(13).unwrap().as_ref();
            let mbarrier_group = block.argument(14).unwrap().as_ref();
            let mbarrier_token = block.argument(15).unwrap().as_ref();
            let tensor_map_descriptor = block.argument(16).unwrap().as_ref();
            let warpgroup_descriptor_a = block.argument(17).unwrap().as_ref();
            let warpgroup_descriptor_b = block.argument(18).unwrap().as_ref();
            let warpgroup_accumulator = block.argument(19).unwrap().as_ref();
            let warpgroup_f32_memref = block.argument(20).unwrap().as_ref();
            let operation = ldmatrix(shared_f16_memref, &[index_0, index_1], true, 4, types.vector_f16_4x2, location);
            assert_eq!(operation.src_memref(), shared_f16_memref);
            assert_eq!(operation.indices(), vec![index_0, index_1]);
            assert!(operation.transpose());
            assert_eq!(operation.num_tiles(), 4);
            assert_eq!(operation.matrix().r#type(), types.vector_f16_4x2);
            block.append_operation(operation);

            let operation = mma_sync(vector_a, vector_b, vector_c, &[16, 8, 16], false, types.vector_f32_2x2, location);
            assert_eq!(operation.matrix_a(), vector_a);
            assert_eq!(operation.matrix_b(), vector_b);
            assert_eq!(operation.matrix_c(), vector_c);
            assert_eq!(operation.mma_shape(), vec![16, 8, 16]);
            assert!(!operation.tf32_enabled());
            assert_eq!(operation.matrix_d().r#type(), types.vector_f32_2x2);
            block.append_operation(operation);

            let operation = mma_sparse_sync(
                vector_a,
                vector_a,
                vector_b,
                sparse_metadata,
                &[16, 8, 32],
                1,
                false,
                types.vector_f16_2x2,
                location,
            );
            assert_eq!(operation.matrix_a(), vector_a);
            assert_eq!(operation.matrix_b(), vector_a);
            assert_eq!(operation.matrix_c(), vector_b);
            assert_eq!(operation.sparse_metadata(), sparse_metadata);
            assert_eq!(operation.mma_shape(), vec![16, 8, 32]);
            assert_eq!(operation.sparsity_selector(), 1);
            assert!(!operation.tf32_enabled());
            assert_eq!(operation.matrix_d().r#type(), types.vector_f16_2x2);
            block.append_operation(operation);

            let operation = device_async_copy(
                shared_f16_memref,
                &[index_0, index_1],
                global_f16_memref,
                &[index_2],
                4,
                Some(index_0),
                false,
                location,
            );
            assert_eq!(operation.dst(), shared_f16_memref);
            assert_eq!(operation.dst_indices(), vec![index_0, index_1]);
            assert_eq!(operation.src(), global_f16_memref);
            assert_eq!(operation.src_indices(), vec![index_2]);
            assert_eq!(operation.dst_elements(), 4);
            assert_eq!(operation.src_elements(), Some(index_0));
            assert!(!operation.bypass_l1());
            assert_eq!(operation.async_token().r#type(), types.device_async_token);
            block.append_operation(operation);

            let operation = device_async_create_group(&[device_async_token], location);
            assert_eq!(operation.input_tokens(), vec![device_async_token]);
            assert_eq!(operation.async_token().r#type(), types.device_async_token);
            block.append_operation(operation);

            let operation = device_async_wait(device_async_token, Some(2), location);
            assert_eq!(operation.async_dependency(), device_async_token);
            assert_eq!(operation.num_groups(), Some(2));
            block.append_operation(operation);

            let operation = mbarrier_create(types.mbarrier_group, location);
            assert_eq!(operation.barriers().r#type(), types.mbarrier_group);
            block.append_operation(operation);

            let operation = mbarrier_get(mbarrier_group, index_0, types.i64, location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.mbarrier_pointer().r#type(), types.i64);
            block.append_operation(operation);

            let operation = mbarrier_init(mbarrier_group, index_1, index_0, Some(predicate), location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.count(), index_1);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.predicate(), Some(predicate));
            block.append_operation(operation);

            let operation = mbarrier_test_wait(mbarrier_group, mbarrier_token, index_0, location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.token(), mbarrier_token);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.wait_complete().r#type(), types.i1);
            block.append_operation(operation);

            let operation = mbarrier_arrive(mbarrier_group, index_0, location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.token().r#type(), types.mbarrier_token);
            block.append_operation(operation);

            let operation = mbarrier_arrive_nocomplete(mbarrier_group, index_0, index_1, location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.count(), index_1);
            assert_eq!(operation.token().r#type(), types.mbarrier_token);
            block.append_operation(operation);

            let operation = mbarrier_arrive_expect_tx(mbarrier_group, index_2, index_0, Some(predicate), location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.tx_count(), index_2);
            assert_eq!(operation.mbar_id(), index_0);
            assert_eq!(operation.predicate(), Some(predicate));
            block.append_operation(operation);

            let operation = mbarrier_try_wait_parity(mbarrier_group, predicate, index_2, index_0, location);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.phase_parity(), predicate);
            assert_eq!(operation.ticks(), index_2);
            assert_eq!(operation.mbar_id(), index_0);
            block.append_operation(operation);

            let operation = tma_fence(tensor_map_descriptor, location);
            assert_eq!(operation.tensor_map_descriptor(), tensor_map_descriptor);
            block.append_operation(operation);

            let operation = tma_prefetch(tensor_map_descriptor, Some(predicate), location);
            assert_eq!(operation.tensor_map_descriptor(), tensor_map_descriptor);
            assert_eq!(operation.predicate(), Some(predicate));
            block.append_operation(operation);

            let operation = tma_async_load(
                global_f32_memref,
                mbarrier_group,
                tensor_map_descriptor,
                &[index_0, index_1],
                index_2,
                Some(i16),
                Some(predicate),
                location,
            );
            assert_eq!(operation.dst(), global_f32_memref);
            assert_eq!(operation.barriers(), mbarrier_group);
            assert_eq!(operation.tensor_map_descriptor(), tensor_map_descriptor);
            assert_eq!(operation.coordinates(), vec![index_0, index_1]);
            assert_eq!(operation.mbar_id(), index_2);
            assert_eq!(operation.multicast_mask(), Some(i16));
            assert_eq!(operation.predicate(), Some(predicate));
            block.append_operation(operation);

            let operation = tma_async_store(
                global_f32_memref,
                tensor_map_descriptor,
                &[index_0, index_1],
                Some(predicate),
                location,
            );
            assert_eq!(operation.src(), global_f32_memref);
            assert_eq!(operation.tensor_map_descriptor(), tensor_map_descriptor);
            assert_eq!(operation.coordinates(), vec![index_0, index_1]);
            assert_eq!(operation.predicate(), Some(predicate));
            block.append_operation(operation);

            let operation =
                tma_create_descriptor(unranked_f32_memref, &[index_0, index_1], types.tensor_map_descriptor, location);
            assert_eq!(operation.tensor(), unranked_f32_memref);
            assert_eq!(operation.box_dimensions(), vec![index_0, index_1]);
            assert_eq!(operation.tensor_map().r#type(), types.tensor_map_descriptor);
            block.append_operation(operation);

            let operation = warpgroup_generate_descriptor(
                shared_f16_memref,
                tensor_map_descriptor,
                types.warpgroup_descriptor,
                location,
            );
            assert_eq!(operation.tensor(), shared_f16_memref);
            assert_eq!(operation.tensor_map(), tensor_map_descriptor);
            assert_eq!(operation.descriptor().r#type(), types.warpgroup_descriptor);
            block.append_operation(operation);

            let operation = warpgroup_mma(
                warpgroup_descriptor_a,
                warpgroup_descriptor_b,
                warpgroup_accumulator,
                Some(2),
                true,
                true,
                types.warpgroup_accumulator,
                location,
            );
            assert_eq!(operation.descriptor_a(), warpgroup_descriptor_a);
            assert_eq!(operation.descriptor_b(), warpgroup_descriptor_b);
            assert_eq!(operation.matrix_c(), warpgroup_accumulator);
            assert_eq!(operation.wait_group(), 2);
            assert!(operation.transpose_a());
            assert!(operation.transpose_b());
            assert_eq!(operation.matrix_d().r#type(), types.warpgroup_accumulator);
            block.append_operation(operation);

            let operation = warpgroup_mma_store(warpgroup_accumulator, warpgroup_f32_memref, location);
            assert_eq!(operation.matrix_d(), warpgroup_accumulator);
            assert_eq!(operation.dst_memref(), warpgroup_f32_memref);
            block.append_operation(operation);

            let operation = warpgroup_mma_init_accumulator(types.warpgroup_accumulator, location);
            assert_eq!(operation.matrix_c().r#type(), types.warpgroup_accumulator);
            block.append_operation(operation);

            let operation = rcp(vector_c, RcpRoundingMode::Approx, true, types.vector_f32_2x2, location);
            assert_eq!(operation.input(), vector_c);
            assert_eq!(operation.rounding().value(), RcpRoundingMode::Approx);
            assert!(operation.ftz());
            assert_eq!(operation.output().r#type(), types.vector_f32_2x2);
            block.append_operation(operation);

            block.append_operation(func::r#return(&[] as &[ValueRef], location));
            func::func(
                "nvgpu_operations",
                func::FuncAttributes {
                    arguments: vec![
                        types.shared_f16_memref.into(),
                        types.global_f16_memref.into(),
                        types.global_f32_memref.into(),
                        types.unranked_f32_memref.into(),
                        types.index.into(),
                        types.index.into(),
                        types.index.into(),
                        types.i1.into(),
                        types.i16.into(),
                        types.vector_f16_4x2.into(),
                        types.vector_f16_2x2.into(),
                        types.vector_f32_2x2.into(),
                        types.sparse_metadata.into(),
                        types.device_async_token.into(),
                        types.mbarrier_group.into(),
                        types.mbarrier_token.into(),
                        types.tensor_map_descriptor.into(),
                        types.warpgroup_descriptor.into(),
                        types.warpgroup_descriptor_b.into(),
                        types.warpgroup_accumulator.into(),
                        types.warpgroup_f32_memref.into(),
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
                  func.func @nvgpu_operations(%arg0: memref<32x32xf16, 3>, %arg1: memref<64xf16>, %arg2: memref<32x32xf32, 3>, %arg3: memref<*xf32>, %arg4: index, %arg5: index, %arg6: index, %arg7: i1, %arg8: i16, %arg9: vector<4x2xf16>, %arg10: vector<2x2xf16>, %arg11: vector<2x2xf32>, %arg12: vector<2xi16>, %arg13: !nvgpu.device.async.token, %arg14: !nvgpu.mbarrier.group<memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>, %arg15: !nvgpu.mbarrier.token, %arg16: !nvgpu.tensormap.descriptor<tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, %arg17: !nvgpu.warpgroup.descriptor<tensor = memref<64x64xf16, 3>>, %arg18: !nvgpu.warpgroup.descriptor<tensor = memref<64x128xf16, 3>>, %arg19: !nvgpu.warpgroup.accumulator<fragmented = vector<64x128xf32>>, %arg20: memref<64x128xf32, 3>) {
                    %0 = nvgpu.ldmatrix %arg0[%arg4, %arg5] {numTiles = 4 : i32, transpose = true} : memref<32x32xf16, 3> -> vector<4x2xf16>
                    %1 = nvgpu.mma.sync(%arg9, %arg10, %arg11) {mmaShape = [16, 8, 16]} : (vector<4x2xf16>, vector<2x2xf16>, vector<2x2xf32>) -> vector<2x2xf32>
                    %2 = nvgpu.mma.sp.sync(%arg9, %arg9, %arg10) metadata(%arg12) {mmaShape = [16, 8, 32], sparsitySelector = 1 : i32} : (vector<4x2xf16>, vector<4x2xf16>, vector<2x2xf16>) -> vector<2x2xf16>
                    %3 = nvgpu.device_async_copy %arg1[%arg6], %arg0[%arg4, %arg5], 4, %arg4 : memref<64xf16> to memref<32x32xf16, 3>
                    %4 = nvgpu.device_async_create_group %arg13
                    nvgpu.device_async_wait %arg13 {numGroups = 2 : i32}
                    %5 = nvgpu.mbarrier.create -> <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>
                    %6 = nvgpu.mbarrier.get %arg14[%arg4] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4> -> i64
                    nvgpu.mbarrier.init %arg14[%arg4], %arg5, predicate = %arg7 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>
                    %7 = nvgpu.mbarrier.test.wait %arg14[%arg4], %arg15 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>, !nvgpu.mbarrier.token
                    %8 = nvgpu.mbarrier.arrive %arg14[%arg4] : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4> -> !nvgpu.mbarrier.token
                    %9 = nvgpu.mbarrier.arrive.nocomplete %arg14[%arg4], %arg5 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4> -> !nvgpu.mbarrier.token
                    nvgpu.mbarrier.arrive.expect_tx %arg14[%arg4], %arg6, predicate = %arg7 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>
                    nvgpu.mbarrier.try_wait.parity %arg14[%arg4], %arg7, %arg6 : <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4>
                    nvgpu.tma.fence.descriptor %arg16 : <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
                    nvgpu.tma.prefetch.descriptor %arg16, predicate = %arg7 : <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
                    nvgpu.tma.async.load %arg16[%arg4, %arg5], %arg14[%arg6] to %arg2 multicast_mask = %arg8, predicate = %arg7 : <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>, <memorySpace = #gpu.address_space<workgroup>, num_barriers = 4> -> memref<32x32xf32, 3>
                    nvgpu.tma.async.store %arg2 to %arg16[%arg4, %arg5], predicate = %arg7 : memref<32x32xf32, 3> -> <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
                    %10 = nvgpu.tma.create.descriptor %arg3 box[%arg4, %arg5] : memref<*xf32> -> <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none>
                    %11 = nvgpu.warpgroup.generate.descriptor %arg0, %arg16 : memref<32x32xf16, 3>, <tensor = memref<32x32xf32, 3>, swizzle = swizzle_128b, l2promo = none, oob = zero, interleave = none> -> <tensor = memref<64x64xf16, 3>>
                    %12 = nvgpu.warpgroup.mma %arg17, %arg18, %arg19 {transposeA, transposeB, waitGroup = 2 : i64} : <tensor = memref<64x64xf16, 3>>, <tensor = memref<64x128xf16, 3>>, <fragmented = vector<64x128xf32>> -> <fragmented = vector<64x128xf32>>
                    nvgpu.warpgroup.mma.store %arg19, %arg20 : <fragmented = vector<64x128xf32>> to memref<64x128xf32, 3>
                    %13 = nvgpu.warpgroup.mma.init.accumulator -> <fragmented = vector<64x128xf32>>
                    %14 = nvgpu.rcp %arg11{rounding = approx, ftz} : vector<2x2xf32>
                    return
                  }
                }
            "},
        );
    }
}
