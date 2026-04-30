use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, DenseInteger32ArrayAttributeRef,
    DenseInteger64ArrayAttributeRef, IntegerAttributeRef, Operation, OperationResultRef, RegionRef, StringAttributeRef,
    ValueRef, mlir_op, mlir_op_trait,
};

use super::attributes::{
    AtomicOpTypeAttributeRef, CopyPartitionAttributeRef, MultimemLoadReductionTypeAttributeRef,
    OobFillModeAttributeRef, TiledLayoutAttributeRef, TmaReductionAttributeRef, WgStridedFragLayoutAttributeRef,
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

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DetachedOp, DialectHandle, Operation, OperationBuilder};

    use super::*;

    macro_rules! operation_casting_test {
        ($test_name:ident, $operation_type:ty, $operation_name:literal $(,)?) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                context.load_dialect(DialectHandle::mosaic_gpu());
                let location = context.unknown_location();
                let operation: $operation_type = OperationBuilder::new($operation_name, location)
                    .build()
                    .and_then(|operation| unsafe { operation.cast() })
                    .unwrap();

                assert_eq!(operation.name().as_str(), Ok($operation_name));
            }
        };
    }

    operation_casting_test!(
        test_initialize_barrier_operation,
        DetachedInitializeBarrierOperation<'_, '_>,
        "mosaic_gpu.initialize_barrier",
    );

    operation_casting_test!(test_arrive_operation, DetachedArriveOperation<'_, '_>, "mosaic_gpu.arrive");

    operation_casting_test!(
        test_arrive_expect_tx_operation,
        DetachedArriveExpectTxOperation<'_, '_>,
        "mosaic_gpu.arrive_expect_tx",
    );

    operation_casting_test!(test_wait_operation, DetachedWaitOperation<'_, '_>, "mosaic_gpu.wait");

    operation_casting_test!(
        test_try_cluster_cancel_operation,
        DetachedTryClusterCancelOperation<'_, '_>,
        "mosaic_gpu.try_cluster_cancel",
    );

    operation_casting_test!(
        test_query_cluster_cancel_operation,
        DetachedQueryClusterCancelOperation<'_, '_>,
        "mosaic_gpu.query_cluster_cancel",
    );

    operation_casting_test!(test_async_load_operation, DetachedAsyncLoadOperation<'_, '_>, "mosaic_gpu.async_load",);

    operation_casting_test!(
        test_async_prefetch_operation,
        DetachedAsyncPrefetchOperation<'_, '_>,
        "mosaic_gpu.async_prefetch",
    );

    operation_casting_test!(test_async_store_operation, DetachedAsyncStoreOperation<'_, '_>, "mosaic_gpu.async_store",);

    operation_casting_test!(test_vector_load_operation, DetachedVectorLoadOperation<'_, '_>, "mosaic_gpu.vector_load",);

    operation_casting_test!(
        test_multimem_load_reduce_operation,
        DetachedMultimemLoadReduceOperation<'_, '_>,
        "mosaic_gpu.multimem_load_reduce",
    );

    operation_casting_test!(
        test_vector_store_operation,
        DetachedVectorStoreOperation<'_, '_>,
        "mosaic_gpu.vector_store",
    );

    operation_casting_test!(test_layout_cast_operation, DetachedLayoutCastOperation<'_, '_>, "mosaic_gpu.layout_cast",);

    operation_casting_test!(
        test_tmem_layout_cast_operation,
        DetachedTmemLayoutCastOperation<'_, '_>,
        "mosaic_gpu.tmem_layout_cast",
    );

    operation_casting_test!(
        test_broadcast_in_dim_operation,
        DetachedBroadcastInDimOperation<'_, '_>,
        "mosaic_gpu.broadcast_in_dim",
    );

    operation_casting_test!(
        test_reinterpret_cast_operation,
        DetachedReinterpretCastOperation<'_, '_>,
        "mosaic_gpu.reinterpret_cast",
    );

    operation_casting_test!(test_slice_smem_operation, DetachedSliceSmemOperation<'_, '_>, "mosaic_gpu.slice_smem",);

    operation_casting_test!(test_wgmma_operation, DetachedWgmmaOperation<'_, '_>, "mosaic_gpu.wgmma");

    operation_casting_test!(test_tcgen05_mma_operation, DetachedTcGen05MmaOperation<'_, '_>, "mosaic_gpu.tcgen05_mma",);

    operation_casting_test!(
        test_optimization_barrier_operation,
        DetachedOptimizationBarrierOperation<'_, '_>,
        "mosaic_gpu.optimization_barrier",
    );

    operation_casting_test!(test_return_operation, DetachedReturnOperation<'_, '_>, "mosaic_gpu.return");

    operation_casting_test!(
        test_custom_primitive_operation,
        DetachedCustomPrimitiveOperation<'_, '_>,
        "mosaic_gpu.custom_primitive",
    );

    operation_casting_test!(test_warp_map_operation, DetachedWarpMapOperation<'_, '_>, "mosaic_gpu.warp_map");

    operation_casting_test!(
        test_with_transforms_operation,
        DetachedWithTransformsOperation<'_, '_>,
        "mosaic_gpu.with_transforms",
    );

    operation_casting_test!(test_tmem_alloc_operation, DetachedTmemAllocOperation<'_, '_>, "mosaic_gpu.tmem_alloc",);

    operation_casting_test!(
        test_tmem_relinquish_alloc_permit_operation,
        DetachedTmemRelinquishAllocPermitOperation<'_, '_>,
        "mosaic_gpu.tmem_relinquish_alloc_permit",
    );

    operation_casting_test!(
        test_tmem_dealloc_operation,
        DetachedTmemDeallocOperation<'_, '_>,
        "mosaic_gpu.tmem_dealloc",
    );

    operation_casting_test!(
        test_async_load_tmem_operation,
        DetachedAsyncLoadTmemOperation<'_, '_>,
        "mosaic_gpu.async_load_tmem",
    );

    operation_casting_test!(
        test_async_store_tmem_operation,
        DetachedAsyncStoreTmemOperation<'_, '_>,
        "mosaic_gpu.async_store_tmem",
    );

    operation_casting_test!(
        test_async_store_smem_to_tmem_operation,
        DetachedAsyncStoreSmemToTmemOperation<'_, '_>,
        "mosaic_gpu.async_store_smem_to_tmem",
    );

    operation_casting_test!(
        test_async_store_sparse_metadata_smem_to_tmem_operation,
        DetachedAsyncStoreSparseMetadataSmemToTmemOperation<'_, '_>,
        "mosaic_gpu.async_store_sparse_metadata_smem_to_tmem",
    );

    operation_casting_test!(
        test_async_store_scales_smem_to_tmem_operation,
        DetachedAsyncStoreScalesSmemToTmemOperation<'_, '_>,
        "mosaic_gpu.async_store_scales_smem_to_tmem",
    );

    operation_casting_test!(test_slice_tmem_operation, DetachedSliceTmemOperation<'_, '_>, "mosaic_gpu.slice_tmem",);

    operation_casting_test!(
        test_tcgen05_commit_arrive_operation,
        DetachedTcGen05CommitArriveOperation<'_, '_>,
        "mosaic_gpu.tcgen05_commit_arrive",
    );

    operation_casting_test!(test_debug_print_operation, DetachedDebugPrintOperation<'_, '_>, "mosaic_gpu.debug_print",);

    operation_casting_test!(
        test_print_layout_operation,
        DetachedPrintLayoutOperation<'_, '_>,
        "mosaic_gpu.print_layout",
    );

    operation_casting_test!(
        test_broadcasted_iota_operation,
        DetachedBroadcastedIotaOperation<'_, '_>,
        "mosaic_gpu.broadcasted_iota",
    );
}
