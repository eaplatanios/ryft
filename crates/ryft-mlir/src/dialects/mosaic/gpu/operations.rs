use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, DenseInteger32ArrayAttributeRef,
    DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion, DialectHandle, IntegerAttributeRef, Location,
    Operation, OperationBuilder, OperationResultRef, RegionRef, StringAttributeRef, TypeRef, ValueRef, mlir_op,
    mlir_op_trait,
};

use super::attributes::{
    AtomicOpType, AtomicOpTypeAttributeRef, CopyPartitionAttributeRef, MultimemLoadReductionType,
    MultimemLoadReductionTypeAttributeRef, OobFillMode, OobFillModeAttributeRef, TiledLayoutAttributeRef, TmaReduction,
    TmaReductionAttributeRef, WgStridedFragLayoutAttributeRef,
};

/// Name of the [`Attribute`] that stores an arrival count.
pub const ARRIVAL_COUNT_ATTRIBUTE: &str = "arrival_count";

/// Name of the [`Attribute`] that stores the number of barriers.
pub const NUM_BARRIERS_ATTRIBUTE: &str = "num_barriers";

/// Name of the [`Attribute`] that indicates whether a barrier orders tensor-core operations.
pub const ORDERS_TENSOR_CORE_ATTRIBUTE: &str = "orders_tensor_core";

/// Mosaic GPU [`Operation`] that initializes barrier objects at a shared-memory location.
pub trait InitializeBarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory base pointer.
    fn base_pointer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the number of arriving threads expected by each barrier.
    fn arrival_count(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(ARRIVAL_COUNT_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{ARRIVAL_COUNT_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{ARRIVAL_COUNT_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
    }

    /// Returns the number of barriers initialized by this operation.
    fn num_barriers(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(NUM_BARRIERS_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{NUM_BARRIERS_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{NUM_BARRIERS_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
    }

    /// Returns whether initialized barriers order tensor-core operations.
    fn orders_tensor_core(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(ORDERS_TENSOR_CORE_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{ORDERS_TENSOR_CORE_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{ORDERS_TENSOR_CORE_ATTRIBUTE}` attribute in `mosaic_gpu.initialize_barrier`")
            })
    }
}

mlir_op!(InitializeBarrier);
mlir_op_trait!(InitializeBarrier, ZeroRegions);
mlir_op_trait!(InitializeBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`InitializeBarrierOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn initialize_barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    base_pointer: ValueRef<'v, 'c, 't>,
    arrival_count: i64,
    num_barriers: i64,
    orders_tensor_core: bool,
    location: L,
) -> DetachedInitializeBarrierOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.initialize_barrier", location)
        .add_operand(base_pointer)
        .add_attribute(
            ARRIVAL_COUNT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), arrival_count),
        )
        .add_attribute(
            NUM_BARRIERS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), num_barriers),
        )
        .add_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE, context.boolean_attribute(orders_tensor_core))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::initialize_barrier`")
}

/// Mosaic GPU [`Operation`] that arrives at a barrier.
pub trait ArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns whether this arrive operation orders tensor-core operations.
    fn orders_tensor_core(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(ORDERS_TENSOR_CORE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{ORDERS_TENSOR_CORE_ATTRIBUTE}` attribute in `mosaic_gpu.arrive`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{ORDERS_TENSOR_CORE_ATTRIBUTE}` attribute in `mosaic_gpu.arrive`"))
    }
}

mlir_op!(Arrive);
mlir_op_trait!(Arrive, ZeroRegions);
mlir_op_trait!(Arrive, ZeroSuccessors);

/// Constructs a new detached/owned [`ArriveOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    orders_tensor_core: bool,
    location: L,
) -> DetachedArriveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.arrive", location)
        .add_operand(barrier)
        .add_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE, context.boolean_attribute(orders_tensor_core))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::arrive`")
}

/// Name of the [`Attribute`] that stores the expected byte-transfer count.
pub const EXPECT_TX_ATTRIBUTE: &str = "expect_tx";

/// Mosaic GPU [`Operation`] that arrives at a barrier and sets an expected transfer count.
pub trait ArriveExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the expected byte-transfer count.
    fn expect_tx(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(EXPECT_TX_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{EXPECT_TX_ATTRIBUTE}` attribute in `mosaic_gpu.arrive_expect_tx`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{EXPECT_TX_ATTRIBUTE}` attribute in `mosaic_gpu.arrive_expect_tx`"))
    }
}

mlir_op!(ArriveExpectTx);
mlir_op_trait!(ArriveExpectTx, ZeroRegions);
mlir_op_trait!(ArriveExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`ArriveExpectTxOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn arrive_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    expect_tx: i64,
    location: L,
) -> DetachedArriveExpectTxOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.arrive_expect_tx", location)
        .add_operand(barrier)
        .add_attribute(EXPECT_TX_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), expect_tx))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::arrive_expect_tx`")
}

/// Mosaic GPU [`Operation`] that waits for a barrier parity.
pub trait WaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the parity value.
    fn parity(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(Wait);
mlir_op_trait!(Wait, ZeroRegions);
mlir_op_trait!(Wait, ZeroSuccessors);

/// Constructs a new detached/owned [`WaitOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    parity: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedWaitOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.wait", location)
        .add_operand(barrier)
        .add_operand(parity)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::wait`")
}

/// Mosaic GPU [`Operation`] that tries to claim a new cluster work unit.
pub trait TryClusterCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the cancellation-result shared-memory buffer.
    fn cancellation_result(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the completion barrier.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }
}

mlir_op!(TryClusterCancel);
mlir_op_trait!(TryClusterCancel, ZeroRegions);
mlir_op_trait!(TryClusterCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`TryClusterCancelOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn try_cluster_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    cancellation_result: ValueRef<'v, 'c, 't>,
    barrier: ValueRef<'v, 'c, 't>,
    predicate: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedTryClusterCancelOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.try_cluster_cancel", location)
        .add_operand(cancellation_result)
        .add_operand(barrier)
        .add_operand(predicate)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::try_cluster_cancel`")
}

/// Mosaic GPU [`Operation`] that decodes the result of a cluster-cancel request.
pub trait QueryClusterCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the cancellation-result shared-memory buffer.
    fn cancellation_result(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the claimed cluster X coordinate.
    fn x(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }

    /// Returns the claimed cluster Y coordinate.
    fn y(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 1).unwrap()
    }

    /// Returns the claimed cluster Z coordinate.
    fn z(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 2).unwrap()
    }

    /// Returns whether the cluster-cancel request succeeded.
    fn success(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 3).unwrap()
    }
}

mlir_op!(QueryClusterCancel);
mlir_op_trait!(QueryClusterCancel, ZeroRegions);
mlir_op_trait!(QueryClusterCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`QueryClusterCancelOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn query_cluster_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    cancellation_result: ValueRef<'v, 'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedQueryClusterCancelOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.query_cluster_cancel", location)
        .add_operand(cancellation_result)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::query_cluster_cancel`")
}

/// Name of the [`Attribute`] that stores Mosaic GPU operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Name of the [`Attribute`] that stores Mosaic GPU slice lengths.
pub const SLICE_LENGTHS_ATTRIBUTE: &str = "slice_lengths";

/// Name of the [`Attribute`] that stores Mosaic GPU collective dimensions.
pub const COLLECTIVE_ATTRIBUTE: &str = "collective";

/// Name of the [`Attribute`] that stores the leader-tracking copy partition strategy.
pub const LEADER_TRACKED_ATTRIBUTE: &str = "leader_tracked";

/// Name of the [`Attribute`] that stores the out-of-bounds fill mode.
pub const OOB_FILL_MODE_ATTRIBUTE: &str = "oob_fill_mode";

/// Mosaic GPU [`Operation`] that schedules an asynchronous global-to-shared memory load.
pub trait AsyncLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the completion barrier.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let count = sizes.get(3).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(3 + index).unwrap()).collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let index_count = sizes.get(3).copied().unwrap_or(0).max(0) as usize;
        self.operand_value(3 + index_count).unwrap()
    }

    /// Returns the source slice lengths.
    fn slice_lengths(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(SLICE_LENGTHS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
    }

    /// Returns the collective cluster dimensions.
    fn collective(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{COLLECTIVE_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{COLLECTIVE_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
    }

    /// Returns the optional leader-tracking copy partition strategy.
    fn leader_tracked(&self) -> Option<CopyPartitionAttributeRef<'c, 't>> {
        self.attribute(LEADER_TRACKED_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the out-of-bounds fill mode.
    fn oob_fill_mode(&self) -> OobFillModeAttributeRef<'c, 't> {
        self.attribute(OOB_FILL_MODE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{OOB_FILL_MODE_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{OOB_FILL_MODE_ATTRIBUTE}` attribute in `mosaic_gpu.async_load`"))
    }
}

mlir_op!(AsyncLoad);
mlir_op_trait!(AsyncLoad, ZeroRegions);
mlir_op_trait!(AsyncLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncLoadOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    barrier: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    slice_lengths: &[i64],
    collective: ArrayAttributeRef<'c, 't>,
    leader_tracked: Option<CopyPartitionAttributeRef<'c, 't>>,
    oob_fill_mode: OobFillMode,
    location: L,
) -> DetachedAsyncLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.async_load", location)
        .add_operand(source)
        .add_operand(destination)
        .add_operand(barrier)
        .add_operands(indices)
        .add_operand(predicate)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    1,
                    1,
                    i32::try_from(indices.len()).expect("too many `mosaic_gpu.async_load` indices"),
                    1,
                ])
                .unwrap(),
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths).unwrap())
        .add_attribute(COLLECTIVE_ATTRIBUTE, collective)
        .add_attribute(OOB_FILL_MODE_ATTRIBUTE, context.mosaic_gpu_oob_fill_mode_attribute(oob_fill_mode));
    if let Some(leader_tracked) = leader_tracked {
        builder = builder.add_attribute(LEADER_TRACKED_ATTRIBUTE, leader_tracked);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_load`")
}

/// Mosaic GPU [`Operation`] that schedules an asynchronous global-memory prefetch.
pub trait AsyncPrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let count = sizes.get(1).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(1 + index).unwrap()).collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let index_count = sizes.get(1).copied().unwrap_or(0).max(0) as usize;
        self.operand_value(1 + index_count).unwrap()
    }

    /// Returns the source slice lengths.
    fn slice_lengths(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(SLICE_LENGTHS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_prefetch`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_prefetch`"))
    }

    /// Returns the collective cluster dimensions.
    fn collective(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{COLLECTIVE_ATTRIBUTE}` attribute in `mosaic_gpu.async_prefetch`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{COLLECTIVE_ATTRIBUTE}` attribute in `mosaic_gpu.async_prefetch`"))
    }
}

mlir_op!(AsyncPrefetch);
mlir_op_trait!(AsyncPrefetch, ZeroRegions);
mlir_op_trait!(AsyncPrefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncPrefetchOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    slice_lengths: &[i64],
    collective: ArrayAttributeRef<'c, 't>,
    location: L,
) -> DetachedAsyncPrefetchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_prefetch", location)
        .add_operand(source)
        .add_operands(indices)
        .add_operand(predicate)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    i32::try_from(indices.len()).expect("too many `mosaic_gpu.async_prefetch` indices"),
                    1,
                ])
                .unwrap(),
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths).unwrap())
        .add_attribute(COLLECTIVE_ATTRIBUTE, collective)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_prefetch`")
}

/// Name of the [`Attribute`] that stores an optional TMA reduction operation.
pub const REDUCTION_OP_ATTRIBUTE: &str = "reduction_op";

/// Name of the [`Attribute`] that stores whether an async store commits its group.
pub const COMMIT_GROUP_ATTRIBUTE: &str = "commit_group";

/// Mosaic GPU [`Operation`] that schedules an asynchronous shared-to-global memory store.
pub trait AsyncStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        (0..count).map(|index| self.operand_value(2 + index).unwrap()).collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let index_count = sizes.get(2).copied().unwrap_or(0).max(0) as usize;
        self.operand_value(2 + index_count).unwrap()
    }

    /// Returns the destination slice lengths.
    fn slice_lengths(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(SLICE_LENGTHS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{SLICE_LENGTHS_ATTRIBUTE}` attribute in `mosaic_gpu.async_store`"))
    }

    /// Returns whether this async store commits its group.
    fn commit_group(&self) -> Option<BooleanAttributeRef<'c, 't>> {
        self.attribute(COMMIT_GROUP_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional TMA reduction operation.
    fn reduction_op(&self) -> Option<TmaReductionAttributeRef<'c, 't>> {
        self.attribute(REDUCTION_OP_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }
}

mlir_op!(AsyncStore);
mlir_op_trait!(AsyncStore, ZeroRegions);
mlir_op_trait!(AsyncStore, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    slice_lengths: &[i64],
    commit_group: Option<bool>,
    reduction_op: Option<TmaReduction>,
    location: L,
) -> DetachedAsyncStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.async_store", location)
        .add_operand(source)
        .add_operand(destination)
        .add_operands(indices)
        .add_operand(predicate)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    1,
                    i32::try_from(indices.len()).expect("too many `mosaic_gpu.async_store` indices"),
                    1,
                ])
                .unwrap(),
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths).unwrap());
    if let Some(commit_group) = commit_group {
        builder = builder.add_attribute(COMMIT_GROUP_ATTRIBUTE, context.boolean_attribute(commit_group));
    }
    if let Some(reduction_op) = reduction_op {
        builder =
            builder.add_attribute(REDUCTION_OP_ATTRIBUTE, context.mosaic_gpu_tma_reduction_attribute(reduction_op));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_store`")
}

/// Name of the [`Attribute`] that stores an optimization request.
pub const OPTIMIZED_ATTRIBUTE: &str = "optimized";

/// Mosaic GPU [`Operation`] that reads a non-contiguous memref slice into a vector.
pub trait VectorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns whether an optimized lowering is required.
    fn optimized(&self) -> Option<BooleanAttributeRef<'c, 't>> {
        self.attribute(OPTIMIZED_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the loaded vector.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(VectorLoad);
mlir_op_trait!(VectorLoad, OneOperand);
mlir_op_trait!(VectorLoad, OneResult);
mlir_op_trait!(VectorLoad, ZeroRegions);
mlir_op_trait!(VectorLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorLoadOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn vector_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    optimized: Option<bool>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedVectorLoadOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.vector_load", location).add_operand(source);
    if let Some(optimized) = optimized {
        builder = builder.add_attribute(OPTIMIZED_ATTRIBUTE, context.boolean_attribute(optimized));
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::vector_load`")
}

/// Name of the [`Attribute`] that stores a multimem load reduction type.
pub const REDUCTION_TYPE_ATTRIBUTE: &str = "reduction_type";

/// Mosaic GPU [`Operation`] that loads from multicast memory and reduces the loaded values.
pub trait MultimemLoadReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the multicast source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reduction type.
    fn reduction_type(&self) -> MultimemLoadReductionTypeAttributeRef<'c, 't> {
        self.attribute(REDUCTION_TYPE_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{REDUCTION_TYPE_ATTRIBUTE}` attribute in `mosaic_gpu.multimem_load_reduce`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{REDUCTION_TYPE_ATTRIBUTE}` attribute in `mosaic_gpu.multimem_load_reduce`")
            })
    }

    /// Returns the reduced vector.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(MultimemLoadReduce);
mlir_op_trait!(MultimemLoadReduce, OneOperand);
mlir_op_trait!(MultimemLoadReduce, OneResult);
mlir_op_trait!(MultimemLoadReduce, ZeroRegions);
mlir_op_trait!(MultimemLoadReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`MultimemLoadReduceOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn multimem_load_reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reduction_type: MultimemLoadReductionType,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedMultimemLoadReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.multimem_load_reduce", location)
        .add_operand(source)
        .add_attribute(
            REDUCTION_TYPE_ATTRIBUTE,
            context.mosaic_gpu_multimem_load_reduction_type_attribute(reduction_type),
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::multimem_load_reduce`")
}

/// Name of the [`Attribute`] that stores an atomic store operation type.
pub const ATOMIC_TYPE_ATTRIBUTE: &str = "atomic_type";

/// Name of the [`Attribute`] that indicates whether multimem store instructions are used.
pub const MULTIMEM_ATTRIBUTE: &str = "multimem";

/// Mosaic GPU [`Operation`] that writes a vector to a non-contiguous memref slice.
pub trait VectorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the vector to store.
    fn value_to_store(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns whether an optimized lowering is required.
    fn optimized(&self) -> Option<BooleanAttributeRef<'c, 't>> {
        self.attribute(OPTIMIZED_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional atomic store operation type.
    fn atomic_type(&self) -> Option<AtomicOpTypeAttributeRef<'c, 't>> {
        self.attribute(ATOMIC_TYPE_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns whether this store uses multimem instructions.
    fn multimem(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(MULTIMEM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{MULTIMEM_ATTRIBUTE}` attribute in `mosaic_gpu.vector_store`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{MULTIMEM_ATTRIBUTE}` attribute in `mosaic_gpu.vector_store`"))
    }
}

mlir_op!(VectorStore);
mlir_op_trait!(VectorStore, ZeroRegions);
mlir_op_trait!(VectorStore, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorStoreOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn vector_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    optimized: Option<bool>,
    atomic_type: Option<AtomicOpType>,
    multimem: bool,
    location: L,
) -> DetachedVectorStoreOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.vector_store", location)
        .add_operand(value_to_store)
        .add_operand(destination)
        .add_attribute(MULTIMEM_ATTRIBUTE, context.boolean_attribute(multimem));
    if let Some(optimized) = optimized {
        builder = builder.add_attribute(OPTIMIZED_ATTRIBUTE, context.boolean_attribute(optimized));
    }
    if let Some(atomic_type) = atomic_type {
        builder =
            builder.add_attribute(ATOMIC_TYPE_ATTRIBUTE, context.mosaic_gpu_atomic_op_type_attribute(atomic_type));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::vector_store`")
}

/// Name of the [`Attribute`] that stores a Mosaic GPU layout.
pub const NEW_LAYOUT_ATTRIBUTE: &str = "new_layout";

/// Mosaic GPU [`Operation`] that casts a vector to a new fragment layout.
pub trait LayoutCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input vector.
    fn x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the strided layout when this operation casts to one.
    fn strided_layout(&self) -> Option<WgStridedFragLayoutAttributeRef<'c, 't>> {
        self.attribute(NEW_LAYOUT_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the tiled layout when this operation casts to one.
    fn tiled_layout(&self) -> Option<TiledLayoutAttributeRef<'c, 't>> {
        self.attribute(NEW_LAYOUT_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the cast result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(LayoutCast);
mlir_op_trait!(LayoutCast, OneOperand);
mlir_op_trait!(LayoutCast, OneResult);
mlir_op_trait!(LayoutCast, ZeroRegions);
mlir_op_trait!(LayoutCast, ZeroSuccessors);

/// Constructs a new detached/owned [`LayoutCastOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn layout_cast<'v, 'c: 'v, 't: 'c, A: Attribute<'c, 't>, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    new_layout: A,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedLayoutCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.layout_cast", location)
        .add_operand(x)
        .add_attribute(NEW_LAYOUT_ATTRIBUTE, new_layout)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::layout_cast`")
}

/// Mosaic GPU [`Operation`] that casts a TMEM memref to a new TMEM layout.
pub trait TmemLayoutCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the TMEM memref.
    fn r#ref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the new tiled layout.
    fn new_layout(&self) -> TiledLayoutAttributeRef<'c, 't> {
        self.attribute(NEW_LAYOUT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{NEW_LAYOUT_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_layout_cast`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{NEW_LAYOUT_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_layout_cast`"))
    }

    /// Returns the cast result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(TmemLayoutCast);
mlir_op_trait!(TmemLayoutCast, OneOperand);
mlir_op_trait!(TmemLayoutCast, OneResult);
mlir_op_trait!(TmemLayoutCast, ZeroRegions);
mlir_op_trait!(TmemLayoutCast, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemLayoutCastOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tmem_layout_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    r#ref: ValueRef<'v, 'c, 't>,
    new_layout: TiledLayoutAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTmemLayoutCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.tmem_layout_cast", location)
        .add_operand(r#ref)
        .add_attribute(NEW_LAYOUT_ATTRIBUTE, new_layout)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tmem_layout_cast`")
}

/// Name of the [`Attribute`] that stores broadcast dimensions.
pub const BROADCAST_DIMENSIONS_ATTRIBUTE: &str = "broadcast_dimensions";

/// Mosaic GPU [`Operation`] that broadcasts a vector to a new shape.
pub trait BroadcastInDimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input vector.
    fn operand(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the dimensions that map operand dimensions to result dimensions.
    fn broadcast_dimensions(&self) -> DenseInteger64ArrayAttributeRef<'c, 't> {
        self.attribute(BROADCAST_DIMENSIONS_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{BROADCAST_DIMENSIONS_ATTRIBUTE}` attribute in `mosaic_gpu.broadcast_in_dim`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{BROADCAST_DIMENSIONS_ATTRIBUTE}` attribute in `mosaic_gpu.broadcast_in_dim`")
            })
    }

    /// Returns the broadcast result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(BroadcastInDim);
mlir_op_trait!(BroadcastInDim, OneOperand);
mlir_op_trait!(BroadcastInDim, OneResult);
mlir_op_trait!(BroadcastInDim, ZeroRegions);
mlir_op_trait!(BroadcastInDim, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastInDimOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn broadcast_in_dim<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operand: ValueRef<'v, 'c, 't>,
    broadcast_dimensions: &[i64],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBroadcastInDimOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.broadcast_in_dim", location)
        .add_operand(operand)
        .add_attribute(BROADCAST_DIMENSIONS_ATTRIBUTE, context.dense_i64_array_attribute(broadcast_dimensions).unwrap())
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::broadcast_in_dim`")
}

/// Mosaic GPU [`Operation`] that reinterprets a memref with a new shape or layout.
pub trait ReinterpretCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the reinterpreted memref.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(ReinterpretCast);
mlir_op_trait!(ReinterpretCast, OneOperand);
mlir_op_trait!(ReinterpretCast, OneResult);
mlir_op_trait!(ReinterpretCast, ZeroRegions);
mlir_op_trait!(ReinterpretCast, ZeroSuccessors);

/// Constructs a new detached/owned [`ReinterpretCastOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reinterpret_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedReinterpretCastOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.reinterpret_cast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::reinterpret_cast`")
}

/// Name of the [`Attribute`] that stores a byte or tensor-memory column offset.
pub const OFFSET_ATTRIBUTE: &str = "offset";

/// Name of the [`Attribute`] that stores an optional alias identifier.
pub const ALIAS_ID_ATTRIBUTE: &str = "alias_id";

/// Mosaic GPU [`Operation`] that constructs a shared-memory memref from an offset.
pub trait SliceSmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the byte offset.
    fn offset(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(OFFSET_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{OFFSET_ATTRIBUTE}` attribute in `mosaic_gpu.slice_smem`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{OFFSET_ATTRIBUTE}` attribute in `mosaic_gpu.slice_smem`"))
    }

    /// Returns the optional alias identifier.
    fn alias_id(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(ALIAS_ID_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the sliced shared-memory memref.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(SliceSmem);
mlir_op_trait!(SliceSmem, ZeroOperands);
mlir_op_trait!(SliceSmem, OneResult);
mlir_op_trait!(SliceSmem, ZeroRegions);
mlir_op_trait!(SliceSmem, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceSmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn slice_smem<'c, 't: 'c, L: Location<'c, 't>>(
    offset: i64,
    alias_id: Option<i64>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSliceSmemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.slice_smem", location)
        .add_attribute(OFFSET_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), offset));
    if let Some(alias_id) = alias_id {
        builder = builder
            .add_attribute(ALIAS_ID_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), alias_id));
    }
    builder
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::slice_smem`")
}

/// Mosaic GPU [`Operation`] that schedules warpgroup matrix multiply-accumulate work.
pub trait WgmmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator vector.
    fn accumulator(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `a` operand.
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `b` operand.
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the accumulator result.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(Wgmma);
mlir_op_trait!(Wgmma, OneResult);
mlir_op_trait!(Wgmma, ZeroRegions);
mlir_op_trait!(Wgmma, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn wgmma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    accumulator: ValueRef<'v, 'c, 't>,
    a: ValueRef<'v, 'c, 't>,
    b: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWgmmaOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.wgmma", location)
        .add_operand(accumulator)
        .add_operand(a)
        .add_operand(b)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::wgmma`")
}

/// Name of the [`Attribute`] that stores whether collective tensor-core work is used.
pub const COLLECTIVE_MMA_ATTRIBUTE: &str = "collective";

/// Mosaic GPU [`Operation`] that schedules a `tcgen05.mma` matrix multiply-accumulate.
pub trait TcGen05MmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator memref.
    fn accumulator(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the `a` operand.
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the `b` operand.
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the accumulate flag operand.
    fn accumulate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(3).unwrap()
    }

    /// Returns the optional `a` scale memref.
    fn a_scale(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        if sizes.get(4).copied().unwrap_or(0) > 0 { self.operand_value(4) } else { None }
    }

    /// Returns the optional `b` scale memref.
    fn b_scale(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let a_scale_count = sizes.get(4).copied().unwrap_or(0).max(0) as usize;
        if sizes.get(5).copied().unwrap_or(0) > 0 { self.operand_value(4 + a_scale_count) } else { None }
    }

    /// Returns the optional sparse metadata memref for the `a` operand.
    fn a_sparse_metadata(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = self
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect::<Vec<_>>())
            .unwrap_or_default();
        let a_scale_count = sizes.get(4).copied().unwrap_or(0).max(0) as usize;
        let b_scale_count = sizes.get(5).copied().unwrap_or(0).max(0) as usize;
        if sizes.get(6).copied().unwrap_or(0) > 0 {
            self.operand_value(4 + a_scale_count + b_scale_count)
        } else {
            None
        }
    }

    /// Returns whether the MMA operation is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_MMA_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{COLLECTIVE_MMA_ATTRIBUTE}` attribute in `mosaic_gpu.tcgen05_mma`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{COLLECTIVE_MMA_ATTRIBUTE}` attribute in `mosaic_gpu.tcgen05_mma`"))
    }
}

mlir_op!(TcGen05Mma);
mlir_op_trait!(TcGen05Mma, ZeroRegions);
mlir_op_trait!(TcGen05Mma, ZeroSuccessors);

/// Constructs a new detached/owned [`TcGen05MmaOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tcgen05_mma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    accumulator: ValueRef<'v, 'c, 't>,
    a: ValueRef<'v, 'c, 't>,
    b: ValueRef<'v, 'c, 't>,
    accumulate: ValueRef<'v, 'c, 't>,
    a_scale: Option<ValueRef<'v, 'c, 't>>,
    b_scale: Option<ValueRef<'v, 'c, 't>>,
    a_sparse_metadata: Option<ValueRef<'v, 'c, 't>>,
    collective: bool,
    location: L,
) -> DetachedTcGen05MmaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    let mut builder = OperationBuilder::new("mosaic_gpu.tcgen05_mma", location)
        .add_operand(accumulator)
        .add_operand(a)
        .add_operand(b)
        .add_operand(accumulate);
    if let Some(a_scale) = a_scale {
        builder = builder.add_operand(a_scale);
    }
    if let Some(b_scale) = b_scale {
        builder = builder.add_operand(b_scale);
    }
    if let Some(a_sparse_metadata) = a_sparse_metadata {
        builder = builder.add_operand(a_sparse_metadata);
    }
    builder
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context
                .dense_i32_array_attribute(&[
                    1,
                    1,
                    1,
                    1,
                    i32::from(a_scale.is_some()),
                    i32::from(b_scale.is_some()),
                    i32::from(a_sparse_metadata.is_some()),
                ])
                .unwrap(),
        )
        .add_attribute(COLLECTIVE_MMA_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tcgen05_mma`")
}

/// Mosaic GPU [`Operation`] that prevents compiler motion across a barrier.
pub trait OptimizationBarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns all barrier operands.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns all barrier results.
    fn results(&self) -> Vec<OperationResultRef<'o, 'c, 't>> {
        (0..self.result_count()).map(|index| Operation::result(self, index).unwrap()).collect()
    }
}

mlir_op!(OptimizationBarrier);
mlir_op_trait!(OptimizationBarrier, ZeroRegions);
mlir_op_trait!(OptimizationBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`OptimizationBarrierOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn optimization_barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    location: L,
) -> DetachedOptimizationBarrierOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.optimization_barrier", location)
        .add_operands(operands)
        .add_results(result_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::optimization_barrier`")
}

/// Mosaic GPU [`Operation`] that terminates a custom primitive region.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned operands.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn r#return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.return", location)
        .add_operands(operands)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::return`")
}

/// Name of the [`Attribute`] that stores custom primitive input layouts.
pub const IN_LAYOUTS_ATTRIBUTE: &str = "in_layouts";

/// Name of the [`Attribute`] that stores custom primitive input transforms.
pub const IN_TRANSFORMS_ATTRIBUTE: &str = "in_transforms";

/// Name of the [`Attribute`] that stores custom primitive output layouts.
pub const OUT_LAYOUTS_ATTRIBUTE: &str = "out_layouts";

/// Mosaic GPU [`Operation`] that defines a custom Mosaic GPU primitive.
pub trait CustomPrimitiveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the custom primitive operands.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the input layouts.
    fn in_layouts(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(IN_LAYOUTS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{IN_LAYOUTS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{IN_LAYOUTS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
    }

    /// Returns the input transforms.
    fn in_transforms(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(IN_TRANSFORMS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{IN_TRANSFORMS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{IN_TRANSFORMS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
    }

    /// Returns the output layouts.
    fn out_layouts(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(OUT_LAYOUTS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{OUT_LAYOUTS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{OUT_LAYOUTS_ATTRIBUTE}` attribute in `mosaic_gpu.custom_primitive`"))
    }

    /// Returns the custom primitive body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(CustomPrimitive);
mlir_op_trait!(CustomPrimitive, OneRegion);
mlir_op_trait!(CustomPrimitive, ZeroSuccessors);

/// Constructs a new detached/owned [`CustomPrimitiveOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn custom_primitive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    in_layouts: ArrayAttributeRef<'c, 't>,
    in_transforms: ArrayAttributeRef<'c, 't>,
    out_layouts: ArrayAttributeRef<'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedCustomPrimitiveOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.custom_primitive", location)
        .add_operands(operands)
        .add_attribute(IN_LAYOUTS_ATTRIBUTE, in_layouts)
        .add_attribute(IN_TRANSFORMS_ATTRIBUTE, in_transforms)
        .add_attribute(OUT_LAYOUTS_ATTRIBUTE, out_layouts)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::custom_primitive`")
}

/// Mosaic GPU [`Operation`] that evaluates a block in parallel on all warps.
pub trait WarpMapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values captured by this warp map.
    fn operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        (0..self.operand_count()).map(|index| self.operand_value(index).unwrap()).collect()
    }

    /// Returns the warp-map region.
    fn region(&self) -> RegionRef<'o, 'c, 't> {
        Operation::region(self, 0).unwrap()
    }
}

mlir_op!(WarpMap);
mlir_op_trait!(WarpMap, OneRegion);
mlir_op_trait!(WarpMap, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpMapOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn warp_map<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedWarpMapOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.warp_map", location)
        .add_operands(operands)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::warp_map`")
}

/// Name of the [`Attribute`] that stores shared-memory transforms.
pub const TRANSFORMS_ATTRIBUTE: &str = "transforms";

/// Mosaic GPU [`Operation`] that attaches transforms to a memref without changing the memref.
pub trait WithTransformsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input memref.
    fn r#ref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the transforms.
    fn transforms(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(TRANSFORMS_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{TRANSFORMS_ATTRIBUTE}` attribute in `mosaic_gpu.with_transforms`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{TRANSFORMS_ATTRIBUTE}` attribute in `mosaic_gpu.with_transforms`"))
    }

    /// Returns the transformed memref.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(WithTransforms);
mlir_op_trait!(WithTransforms, OneOperand);
mlir_op_trait!(WithTransforms, OneResult);
mlir_op_trait!(WithTransforms, ZeroRegions);
mlir_op_trait!(WithTransforms, ZeroSuccessors);

/// Constructs a new detached/owned [`WithTransformsOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn with_transforms<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    r#ref: ValueRef<'v, 'c, 't>,
    transforms: ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedWithTransformsOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.with_transforms", location)
        .add_operand(r#ref)
        .add_attribute(TRANSFORMS_ATTRIBUTE, transforms)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::with_transforms`")
}

/// Name of the [`Attribute`] that stores whether an operation is collective.
pub const COLLECTIVE_TMEM_ATTRIBUTE: &str = "collective";

/// Name of the [`Attribute`] that stores a tensor-memory packing factor.
pub const PACKING_ATTRIBUTE: &str = "packing";

/// Mosaic GPU [`Operation`] that allocates tensor memory.
pub trait TmemAllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory pointer used to store the allocation pointer.
    fn smem_ptr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns whether the allocation is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_alloc`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_alloc`"))
    }

    /// Returns the tensor-memory packing factor.
    fn packing(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(PACKING_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{PACKING_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_alloc`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{PACKING_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_alloc`"))
    }

    /// Returns the allocated tensor-memory memref.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(TmemAlloc);
mlir_op_trait!(TmemAlloc, OneOperand);
mlir_op_trait!(TmemAlloc, OneResult);
mlir_op_trait!(TmemAlloc, ZeroRegions);
mlir_op_trait!(TmemAlloc, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemAllocOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tmem_alloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    smem_ptr: ValueRef<'v, 'c, 't>,
    collective: bool,
    packing: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedTmemAllocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.tmem_alloc", location)
        .add_operand(smem_ptr)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .add_attribute(PACKING_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), packing))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tmem_alloc`")
}

/// Mosaic GPU [`Operation`] that relinquishes tensor-memory allocation permission.
pub trait TmemRelinquishAllocPermitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns whether this applies to collective tensor-memory allocations.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_relinquish_alloc_permit`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tmem_relinquish_alloc_permit`")
            })
    }
}

mlir_op!(TmemRelinquishAllocPermit);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroOperands);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroRegions);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemRelinquishAllocPermitOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tmem_relinquish_alloc_permit<'c, 't: 'c, L: Location<'c, 't>>(
    collective: bool,
    location: L,
) -> DetachedTmemRelinquishAllocPermitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.tmem_relinquish_alloc_permit", location)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tmem_relinquish_alloc_permit`")
}

/// Mosaic GPU [`Operation`] that deallocates tensor memory.
pub trait TmemDeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor-memory memref.
    fn tmem_ref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(TmemDealloc);
mlir_op_trait!(TmemDealloc, ZeroRegions);
mlir_op_trait!(TmemDealloc, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemDeallocOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tmem_dealloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tmem_ref: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedTmemDeallocOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.tmem_dealloc", location)
        .add_operand(tmem_ref)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tmem_dealloc`")
}

/// Mosaic GPU [`Operation`] that copies tensor memory into registers asynchronously.
pub trait AsyncLoadTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor-memory source.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the loaded vector.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(AsyncLoadTmem);
mlir_op_trait!(AsyncLoadTmem, OneOperand);
mlir_op_trait!(AsyncLoadTmem, OneResult);
mlir_op_trait!(AsyncLoadTmem, ZeroRegions);
mlir_op_trait!(AsyncLoadTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncLoadTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_load_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAsyncLoadTmemOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_load_tmem", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_load_tmem`")
}

/// Mosaic GPU [`Operation`] that copies registers into tensor memory asynchronously.
pub trait AsyncStoreTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source vector.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor-memory destination.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }
}

mlir_op!(AsyncStoreTmem);
mlir_op_trait!(AsyncStoreTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_store_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedAsyncStoreTmemOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_store_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_store_tmem`")
}

/// Mosaic GPU [`Operation`] that copies shared memory into tensor memory asynchronously.
pub trait AsyncStoreSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory source.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor-memory destination.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.async_store_smem_to_tmem`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.async_store_smem_to_tmem`")
            })
    }
}

mlir_op!(AsyncStoreSmemToTmem);
mlir_op_trait!(AsyncStoreSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreSmemToTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_store_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> DetachedAsyncStoreSmemToTmemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_store_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_store_smem_to_tmem`")
}

/// Mosaic GPU [`Operation`] that copies sparse metadata from shared memory into tensor memory asynchronously.
pub trait AsyncStoreSparseMetadataSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory sparse metadata source.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor-memory sparse metadata destination.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!(
                    "missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in \
                     `mosaic_gpu.async_store_sparse_metadata_smem_to_tmem`"
                )
            })
            .cast()
            .unwrap_or_else(|| {
                panic!(
                    "invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in \
                     `mosaic_gpu.async_store_sparse_metadata_smem_to_tmem`"
                )
            })
    }
}

mlir_op!(AsyncStoreSparseMetadataSmemToTmem);
mlir_op_trait!(AsyncStoreSparseMetadataSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreSparseMetadataSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreSparseMetadataSmemToTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_store_sparse_metadata_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> DetachedAsyncStoreSparseMetadataSmemToTmemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_store_sparse_metadata_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_store_sparse_metadata_smem_to_tmem`")
}

/// Mosaic GPU [`Operation`] that copies MMA scales from shared memory into tensor memory asynchronously.
pub trait AsyncStoreScalesSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory scales source.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor-memory scales destination.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!(
                    "missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in \
                     `mosaic_gpu.async_store_scales_smem_to_tmem`"
                )
            })
            .cast()
            .unwrap_or_else(|| {
                panic!(
                    "invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in \
                     `mosaic_gpu.async_store_scales_smem_to_tmem`"
                )
            })
    }
}

mlir_op!(AsyncStoreScalesSmemToTmem);
mlir_op_trait!(AsyncStoreScalesSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreScalesSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreScalesSmemToTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn async_store_scales_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> DetachedAsyncStoreScalesSmemToTmemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.async_store_scales_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::async_store_scales_smem_to_tmem`")
}

/// Mosaic GPU [`Operation`] that slices a tensor-memory memref.
pub trait SliceTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor-memory memref.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the tensor-memory column offset.
    fn offset(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(OFFSET_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{OFFSET_ATTRIBUTE}` attribute in `mosaic_gpu.slice_tmem`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{OFFSET_ATTRIBUTE}` attribute in `mosaic_gpu.slice_tmem`"))
    }

    /// Returns the sliced tensor-memory memref.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(SliceTmem);
mlir_op_trait!(SliceTmem, OneOperand);
mlir_op_trait!(SliceTmem, OneResult);
mlir_op_trait!(SliceTmem, ZeroRegions);
mlir_op_trait!(SliceTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceTmemOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn slice_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    offset: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedSliceTmemOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.slice_tmem", location)
        .add_operand(source)
        .add_attribute(OFFSET_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), offset))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::slice_tmem`")
}

/// Mosaic GPU [`Operation`] that makes a barrier track prior async `tcgen05` operations.
pub trait TcGen05CommitArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns whether this commit-arrive operation is collective.
    fn collective(&self) -> BooleanAttributeRef<'c, 't> {
        self.attribute(COLLECTIVE_TMEM_ATTRIBUTE)
            .unwrap_or_else(|| {
                panic!("missing `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tcgen05_commit_arrive`")
            })
            .cast()
            .unwrap_or_else(|| {
                panic!("invalid `{COLLECTIVE_TMEM_ATTRIBUTE}` attribute in `mosaic_gpu.tcgen05_commit_arrive`")
            })
    }
}

mlir_op!(TcGen05CommitArrive);
mlir_op_trait!(TcGen05CommitArrive, ZeroRegions);
mlir_op_trait!(TcGen05CommitArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`TcGen05CommitArriveOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn tcgen05_commit_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> DetachedTcGen05CommitArriveOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.tcgen05_commit_arrive", location)
        .add_operand(barrier)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::tcgen05_commit_arrive`")
}

/// Name of the [`Attribute`] that stores a debug format string.
pub const FORMAT_ATTRIBUTE: &str = "format";

/// Mosaic GPU [`Operation`] that prints a value from inside a Mosaic GPU kernel.
pub trait DebugPrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the format string.
    fn format(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(FORMAT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{FORMAT_ATTRIBUTE}` attribute in `mosaic_gpu.debug_print`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{FORMAT_ATTRIBUTE}` attribute in `mosaic_gpu.debug_print`"))
    }

    /// Returns the value to print.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(DebugPrint);
mlir_op_trait!(DebugPrint, ZeroRegions);
mlir_op_trait!(DebugPrint, ZeroSuccessors);

/// Constructs a new detached/owned [`DebugPrintOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn debug_print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    format: &str,
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedDebugPrintOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.debug_print", location)
        .add_operand(value)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::debug_print`")
}

/// Mosaic GPU [`Operation`] that prints the layout of a value.
pub trait PrintLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the format string.
    fn format(&self) -> StringAttributeRef<'c, 't> {
        self.attribute(FORMAT_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{FORMAT_ATTRIBUTE}` attribute in `mosaic_gpu.print_layout`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{FORMAT_ATTRIBUTE}` attribute in `mosaic_gpu.print_layout`"))
    }

    /// Returns the value whose layout is printed.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(PrintLayout);
mlir_op_trait!(PrintLayout, ZeroRegions);
mlir_op_trait!(PrintLayout, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintLayoutOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn print_layout<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    format: &str,
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> DetachedPrintLayoutOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.print_layout", location)
        .add_operand(value)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::print_layout`")
}

/// Name of the [`Attribute`] that stores an iota dimension.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Mosaic GPU [`Operation`] that creates a broadcasted iota vector.
pub trait BroadcastedIotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the iota dimension.
    fn dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(DIMENSION_ATTRIBUTE)
            .unwrap_or_else(|| panic!("missing `{DIMENSION_ATTRIBUTE}` attribute in `mosaic_gpu.broadcasted_iota`"))
            .cast()
            .unwrap_or_else(|| panic!("invalid `{DIMENSION_ATTRIBUTE}` attribute in `mosaic_gpu.broadcasted_iota`"))
    }

    /// Returns the iota vector.
    fn result(&self) -> OperationResultRef<'o, 'c, 't> {
        Operation::result(self, 0).unwrap()
    }
}

mlir_op!(BroadcastedIota);
mlir_op_trait!(BroadcastedIota, ZeroOperands);
mlir_op_trait!(BroadcastedIota, OneResult);
mlir_op_trait!(BroadcastedIota, ZeroRegions);
mlir_op_trait!(BroadcastedIota, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastedIotaOperation`] at the specified [`Location`].
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn broadcasted_iota<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: i64,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedBroadcastedIotaOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu());
    OperationBuilder::new("mosaic_gpu.broadcasted_iota", location)
        .add_attribute(DIMENSION_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), dimension))
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `mosaic_gpu::broadcasted_iota`")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{
        Attribute, AttributeRef, Block, Context, DialectHandle, Operation, Region, Size, Type, TypeRef, Value,
    };

    use super::super::attributes::{
        AtomicOpType, CopyPartitionAttributeRef, MultimemLoadReductionType, OobFillMode, TmaReduction,
    };

    use super::*;

    /// Common scalar and tensor types used by Mosaic GPU operation wrapper tests.
    #[derive(Copy, Clone)]
    struct TestTypes<'c, 't> {
        /// One-bit signless integer type.
        i1: TypeRef<'c, 't>,

        /// 32-bit signless integer type.
        i32: TypeRef<'c, 't>,

        /// 64-bit signless integer type.
        i64: TypeRef<'c, 't>,

        /// 32-bit floating-point tensor type.
        tensor_f32: TypeRef<'c, 't>,
    }

    impl<'c, 't> TestTypes<'c, 't> {
        /// Builds the common test type set in `context`.
        fn new(context: &'c Context<'t>, location: impl crate::Location<'c, 't>) -> Self {
            let i1_type = context.signless_integer_type(1);
            let i32_type = context.signless_integer_type(32);
            let i64_type = context.signless_integer_type(64);
            let f32_type = context.float32_type();
            let tensor_f32_type = context.tensor_type(f32_type, &[Size::Static(4)], None, location).unwrap();

            Self {
                i1: i1_type.as_ref(),
                i32: i32_type.as_ref(),
                i64: i64_type.as_ref(),
                tensor_f32: tensor_f32_type.as_ref(),
            }
        }
    }

    macro_rules! mosaic_gpu_operation_test {
        ($test_name:ident, |$context:ident, $location:ident, $values:ident, $types:ident| $body:block $(,)?) => {
            #[test]
            fn $test_name() {
                let $context = Context::new();
                $context.load_dialect(DialectHandle::mosaic_gpu());
                let $location = $context.unknown_location();
                let $types = TestTypes::new(&$context, $location);
                let block = $context.block(&[
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i32, $location),
                    ($types.i1, $location),
                    ($types.tensor_f32, $location),
                    ($types.tensor_f32, $location),
                    ($types.i64, $location),
                    ($types.tensor_f32, $location),
                ]);
                let $values = (0..8).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();

                $body
            }
        };
    }

    mosaic_gpu_operation_test!(test_initialize_barrier_operation, |_context, location, values, _types| {
        let operation = initialize_barrier(values[0], 4, 2, true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.initialize_barrier"));
        assert_eq!(operation.base_pointer(), values[0]);
        assert_eq!(operation.arrival_count().signless_value(), 4);
        assert_eq!(operation.num_barriers().signless_value(), 2);
        assert!(operation.orders_tensor_core().value());
    });

    mosaic_gpu_operation_test!(test_arrive_operation, |_context, location, values, _types| {
        let operation = arrive(values[0], true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.arrive"));
        assert_eq!(operation.barrier(), values[0]);
        assert!(operation.orders_tensor_core().value());
    });

    mosaic_gpu_operation_test!(test_arrive_expect_tx_operation, |_context, location, values, _types| {
        let operation = arrive_expect_tx(values[0], 128, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.arrive_expect_tx"));
        assert_eq!(operation.barrier(), values[0]);
        assert_eq!(operation.expect_tx().signless_value(), 128);
    });

    mosaic_gpu_operation_test!(test_wait_operation, |_context, location, values, _types| {
        let operation = wait(values[0], values[1], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.wait"));
        assert_eq!(operation.barrier(), values[0]);
        assert_eq!(operation.parity(), values[1]);
    });

    mosaic_gpu_operation_test!(test_try_cluster_cancel_operation, |_context, location, values, _types| {
        let operation = try_cluster_cancel(values[0], values[1], values[3], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.try_cluster_cancel"));
        assert_eq!(operation.cancellation_result(), values[0]);
        assert_eq!(operation.barrier(), values[1]);
        assert_eq!(operation.predicate(), values[3]);
    });

    mosaic_gpu_operation_test!(test_query_cluster_cancel_operation, |_context, location, values, types| {
        let operation = query_cluster_cancel(values[0], &[types.i32, types.i32, types.i32, types.i1], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.query_cluster_cancel"));
        assert_eq!(operation.cancellation_result(), values[0]);
        assert_eq!(operation.x().r#type(), types.i32);
        assert_eq!(operation.y().r#type(), types.i32);
        assert_eq!(operation.z().r#type(), types.i32);
        assert_eq!(operation.success().r#type(), types.i1);
    });

    mosaic_gpu_operation_test!(test_async_load_operation, |context, location, values, _types| {
        let collective = context.array_attribute(&[] as &[AttributeRef]);
        let leader_tracked =
            context.mosaic_gpu_copy_partitioned_attribute(1).cast::<CopyPartitionAttributeRef>().unwrap();
        let operation = async_load(
            values[4],
            values[5],
            values[0],
            &[values[1], values[2]],
            values[3],
            &[16, 32],
            collective,
            Some(leader_tracked),
            OobFillMode::Zeros,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_load"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.destination(), values[5]);
        assert_eq!(operation.barrier(), values[0]);
        assert_eq!(operation.indices(), vec![values[1], values[2]]);
        assert_eq!(operation.predicate(), values[3]);
        assert_eq!(operation.slice_lengths().values().collect::<Vec<_>>(), vec![16, 32]);
        assert!(operation.collective().is_empty());
        assert!(operation.leader_tracked().is_some());
        assert_eq!(operation.oob_fill_mode().value(), OobFillMode::Zeros);
    });

    mosaic_gpu_operation_test!(test_async_prefetch_operation, |context, location, values, _types| {
        let collective = context.array_attribute(&[] as &[AttributeRef]);
        let operation = async_prefetch(values[4], &[values[1], values[2]], values[3], &[16, 32], collective, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_prefetch"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.indices(), vec![values[1], values[2]]);
        assert_eq!(operation.predicate(), values[3]);
        assert_eq!(operation.slice_lengths().values().collect::<Vec<_>>(), vec![16, 32]);
        assert!(operation.collective().is_empty());
    });

    mosaic_gpu_operation_test!(test_async_store_operation, |_context, location, values, _types| {
        let operation = async_store(
            values[4],
            values[5],
            &[values[1], values[2]],
            values[3],
            &[16, 32],
            Some(true),
            Some(TmaReduction::Add),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_store"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.destination(), values[5]);
        assert_eq!(operation.indices(), vec![values[1], values[2]]);
        assert_eq!(operation.predicate(), values[3]);
        assert_eq!(operation.slice_lengths().values().collect::<Vec<_>>(), vec![16, 32]);
        assert_eq!(operation.commit_group().map(|attribute| attribute.value()), Some(true));
        assert_eq!(operation.reduction_op().map(|attribute| attribute.value()), Some(TmaReduction::Add));
    });

    mosaic_gpu_operation_test!(test_vector_load_operation, |_context, location, values, types| {
        let operation = vector_load(values[4], Some(true), types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.vector_load"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.optimized().map(|attribute| attribute.value()), Some(true));
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_multimem_load_reduce_operation, |_context, location, values, types| {
        let operation = multimem_load_reduce(values[4], MultimemLoadReductionType::Add, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.multimem_load_reduce"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.reduction_type().value(), MultimemLoadReductionType::Add);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_vector_store_operation, |_context, location, values, _types| {
        let operation = vector_store(values[4], values[5], Some(true), Some(AtomicOpType::Add), true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.vector_store"));
        assert_eq!(operation.value_to_store(), values[4]);
        assert_eq!(operation.destination(), values[5]);
        assert_eq!(operation.optimized().map(|attribute| attribute.value()), Some(true));
        assert_eq!(operation.atomic_type().map(|attribute| attribute.value()), Some(AtomicOpType::Add));
        assert!(operation.multimem().value());
    });

    mosaic_gpu_operation_test!(test_layout_cast_operation, |context, location, values, types| {
        let layout =
            context.mosaic_gpu_wg_strided_frag_layout_attribute(context.dense_i64_array_attribute(&[4]).unwrap(), 1);
        let operation = layout_cast(values[4], layout, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.layout_cast"));
        assert_eq!(operation.x(), values[4]);
        assert!(operation.strided_layout().is_some());
        assert!(operation.tiled_layout().is_none());
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_tmem_layout_cast_operation, |context, location, values, types| {
        let empty = context.array_attribute(&[] as &[AttributeRef]);
        let layout = context.mosaic_gpu_tiled_layout_attribute(empty, empty, empty, 0);
        let operation = tmem_layout_cast(values[4], layout, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tmem_layout_cast"));
        assert_eq!(operation.r#ref(), values[4]);
        assert_eq!(operation.new_layout(), layout);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_broadcast_in_dim_operation, |_context, location, values, types| {
        let operation = broadcast_in_dim(values[4], &[0], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.broadcast_in_dim"));
        assert_eq!(BroadcastInDimOperation::operand(&operation), values[4]);
        assert_eq!(operation.broadcast_dimensions().values().collect::<Vec<_>>(), vec![0]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_reinterpret_cast_operation, |_context, location, values, types| {
        let operation = reinterpret_cast(values[4], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.reinterpret_cast"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_slice_smem_operation, |_context, location, _values, types| {
        let operation = slice_smem(16, Some(2), types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.slice_smem"));
        assert_eq!(operation.offset().signless_value(), 16);
        assert_eq!(operation.alias_id().map(|attribute| attribute.signless_value()), Some(2));
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_wgmma_operation, |_context, location, values, types| {
        let operation = wgmma(values[4], values[5], values[7], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.wgmma"));
        assert_eq!(operation.accumulator(), values[4]);
        assert_eq!(operation.a(), values[5]);
        assert_eq!(operation.b(), values[7]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_tcgen05_mma_operation, |_context, location, values, _types| {
        let operation = tcgen05_mma(
            values[4],
            values[5],
            values[7],
            values[3],
            Some(values[1]),
            Some(values[2]),
            Some(values[6]),
            true,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tcgen05_mma"));
        assert_eq!(operation.accumulator(), values[4]);
        assert_eq!(operation.a(), values[5]);
        assert_eq!(operation.b(), values[7]);
        assert_eq!(operation.accumulate(), values[3]);
        assert_eq!(operation.a_scale(), Some(values[1]));
        assert_eq!(operation.b_scale(), Some(values[2]));
        assert_eq!(operation.a_sparse_metadata(), Some(values[6]));
        assert!(operation.collective().value());
    });

    mosaic_gpu_operation_test!(test_optimization_barrier_operation, |_context, location, values, types| {
        let operation = optimization_barrier(&[values[4], values[5]], &[types.tensor_f32, types.tensor_f32], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.optimization_barrier"));
        assert_eq!(OptimizationBarrierOperation::operands(&operation), vec![values[4], values[5]]);
        assert_eq!(
            OptimizationBarrierOperation::results(&operation)
                .iter()
                .map(|result| result.r#type())
                .collect::<Vec<_>>(),
            vec![types.tensor_f32, types.tensor_f32,]
        );
    });

    mosaic_gpu_operation_test!(test_return_operation, |_context, location, values, _types| {
        let operation = r#return(&[values[4], values[5]], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.return"));
        assert_eq!(ReturnOperation::operands(&operation), vec![values[4], values[5]]);
    });

    mosaic_gpu_operation_test!(test_custom_primitive_operation, |context, location, values, types| {
        let empty = context.array_attribute(&[] as &[AttributeRef]);
        let operation =
            custom_primitive(&[values[4]], empty, empty, empty, &[types.tensor_f32], context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.custom_primitive"));
        assert_eq!(CustomPrimitiveOperation::operands(&operation), vec![values[4]]);
        assert!(operation.in_layouts().is_empty());
        assert!(operation.in_transforms().is_empty());
        assert!(operation.out_layouts().is_empty());
        assert_eq!(operation.body().blocks().count(), 0);
    });

    mosaic_gpu_operation_test!(test_warp_map_operation, |context, location, values, _types| {
        let operation = warp_map(&[values[4]], context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.warp_map"));
        assert_eq!(WarpMapOperation::operands(&operation), vec![values[4]]);
        assert_eq!(WarpMapOperation::region(&operation).blocks().count(), 0);
    });

    mosaic_gpu_operation_test!(test_with_transforms_operation, |context, location, values, types| {
        let transforms = context.array_attribute(&[context.mosaic_gpu_tile_transform_attribute(&[16]).as_ref()]);
        let operation = with_transforms(values[4], transforms, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.with_transforms"));
        assert_eq!(operation.r#ref(), values[4]);
        assert_eq!(operation.transforms().len(), 1);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_tmem_alloc_operation, |_context, location, values, types| {
        let operation = tmem_alloc(values[4], true, 4, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tmem_alloc"));
        assert_eq!(operation.smem_ptr(), values[4]);
        assert!(operation.collective().value());
        assert_eq!(operation.packing().signless_value(), 4);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_tmem_relinquish_alloc_permit_operation, |_context, location, _values, _types| {
        let operation = tmem_relinquish_alloc_permit(true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tmem_relinquish_alloc_permit"));
        assert!(operation.collective().value());
    });

    mosaic_gpu_operation_test!(test_tmem_dealloc_operation, |_context, location, values, _types| {
        let operation = tmem_dealloc(values[4], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tmem_dealloc"));
        assert_eq!(operation.tmem_ref(), values[4]);
    });

    mosaic_gpu_operation_test!(test_async_load_tmem_operation, |_context, location, values, types| {
        let operation = async_load_tmem(values[4], types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_load_tmem"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_async_store_tmem_operation, |_context, location, values, _types| {
        let operation = async_store_tmem(values[4], values[5], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_store_tmem"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.destination(), values[5]);
    });

    mosaic_gpu_operation_test!(test_async_store_smem_to_tmem_operation, |_context, location, values, _types| {
        let operation = async_store_smem_to_tmem(values[4], values[5], true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_store_smem_to_tmem"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.destination(), values[5]);
        assert!(operation.collective().value());
    });

    mosaic_gpu_operation_test!(
        test_async_store_sparse_metadata_smem_to_tmem_operation,
        |_context, location, values, _types| {
            let operation = async_store_sparse_metadata_smem_to_tmem(values[4], values[5], true, location);

            assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_store_sparse_metadata_smem_to_tmem"));
            assert_eq!(operation.source(), values[4]);
            assert_eq!(operation.destination(), values[5]);
            assert!(operation.collective().value());
        },
    );

    mosaic_gpu_operation_test!(test_async_store_scales_smem_to_tmem_operation, |_context, location, values, _types| {
        let operation = async_store_scales_smem_to_tmem(values[4], values[5], true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.async_store_scales_smem_to_tmem"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.destination(), values[5]);
        assert!(operation.collective().value());
    },);

    mosaic_gpu_operation_test!(test_slice_tmem_operation, |_context, location, values, types| {
        let operation = slice_tmem(values[4], 32, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.slice_tmem"));
        assert_eq!(operation.source(), values[4]);
        assert_eq!(operation.offset().signless_value(), 32);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });

    mosaic_gpu_operation_test!(test_tcgen05_commit_arrive_operation, |_context, location, values, _types| {
        let operation = tcgen05_commit_arrive(values[0], true, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.tcgen05_commit_arrive"));
        assert_eq!(operation.barrier(), values[0]);
        assert!(operation.collective().value());
    });

    mosaic_gpu_operation_test!(test_debug_print_operation, |_context, location, values, _types| {
        let operation = debug_print("value = {}", values[4], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.debug_print"));
        assert_eq!(operation.format().string().as_str(), Ok("value = {}"));
        assert_eq!(operation.value(), values[4]);
    });

    mosaic_gpu_operation_test!(test_print_layout_operation, |_context, location, values, _types| {
        let operation = print_layout("layout = {}", values[4], location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.print_layout"));
        assert_eq!(operation.format().string().as_str(), Ok("layout = {}"));
        assert_eq!(operation.value(), values[4]);
    });

    mosaic_gpu_operation_test!(test_broadcasted_iota_operation, |_context, location, _values, types| {
        let operation = broadcasted_iota(1, types.tensor_f32, location);

        assert_eq!(operation.name().as_str(), Ok("mosaic_gpu.broadcasted_iota"));
        assert_eq!(operation.dimension().signless_value(), 1);
        assert_eq!(Operation::result(&operation, 0).unwrap().r#type(), types.tensor_f32);
    });
}
