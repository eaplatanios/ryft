//! Typed constructors and accessors for Mosaic GPU operations.
//!
//! Constructors load the dialect and return detached operations for insertion into an MLIR block. Operand segment
//! attributes preserve the positions of optional and variadic operands. Construct a containing module and call
//! [`Operation::verify`] to check the dialect's type, shape, memory-space, and region constraints.
//!
//! These operations describe GPU work; constructing or verifying them does not execute that work or synchronize a GPU.
//! Refer to the [Mosaic GPU definitions] for operation semantics and constraints.
//!
//! [Mosaic GPU definitions]: https://github.com/jax-ml/jax/blob/main/jaxlib/mosaic/dialect/gpu/mosaic_gpu.td

use crate::dialects::mosaic::gpu::attributes::{
    AtomicOpType, AtomicOpTypeAttributeRef, CopyPartitionAttributeRef, Dimension, DimensionAttributeRef,
    MultimemLoadReductionType, MultimemLoadReductionTypeAttributeRef, OobFillMode, OobFillModeAttributeRef,
    TiledLayoutAttributeRef, TmaReduction, TmaReductionAttributeRef, TmemLoadReduction, TmemLoadReductionAttributeRef,
    WgStridedFragLayoutAttributeRef,
};
use crate::macros::{mlir_op, mlir_op_trait};
use crate::{
    ArrayAttributeRef, Attribute, BooleanAttributeRef, DenseInteger64ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, Error, IntegerAttributeRef, Location, Operation, OperationBuilder, OperationResultRef, RegionRef,
    StringAttributeRef, TypeRef, Value, ValueRef,
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
    fn base_pointer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the number of arriving threads expected by each barrier.
    fn arrival_count(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(ARRIVAL_COUNT_ATTRIBUTE)
    }

    /// Returns the number of barriers initialized by this operation.
    fn num_barriers(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(NUM_BARRIERS_ATTRIBUTE)
    }

    /// Returns whether initialized barriers order tensor-core operations.
    fn orders_tensor_core(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE)
    }
}

mlir_op!(InitializeBarrier);
mlir_op_trait!(InitializeBarrier, ZeroRegions);
mlir_op_trait!(InitializeBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`InitializeBarrierOperation`] at the specified [`Location`].
pub fn initialize_barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    base_pointer: ValueRef<'v, 'c, 't>,
    arrival_count: i64,
    num_barriers: i32,
    orders_tensor_core: bool,
    location: L,
) -> Result<DetachedInitializeBarrierOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if arrival_count <= 0 {
        return Err(Error::invalid_argument("expected positive `arrival_count` for `mosaic_gpu.initialize_barrier`"));
    }
    if num_barriers <= 0 {
        return Err(Error::invalid_argument("expected positive `num_barriers` for `mosaic_gpu.initialize_barrier`"));
    }
    OperationBuilder::new("mosaic_gpu.initialize_barrier", location)
        .add_operand(base_pointer)
        .add_attribute(
            ARRIVAL_COUNT_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), arrival_count),
        )
        .add_attribute(
            NUM_BARRIERS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(num_barriers)),
        )
        .add_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE, context.boolean_attribute(orders_tensor_core))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::initialize_barrier`"))
        })
}

/// Mosaic GPU [`Operation`] that arrives at a barrier.
pub trait ArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether this arrive operation orders tensor-core operations.
    fn orders_tensor_core(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE)
    }
}

mlir_op!(Arrive);
mlir_op_trait!(Arrive, ZeroRegions);
mlir_op_trait!(Arrive, ZeroSuccessors);

/// Constructs a new detached/owned [`ArriveOperation`] at the specified [`Location`].
pub fn arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    orders_tensor_core: bool,
    location: L,
) -> Result<DetachedArriveOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.arrive", location)
        .add_operand(barrier)
        .add_attribute(ORDERS_TENSOR_CORE_ATTRIBUTE, context.boolean_attribute(orders_tensor_core))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::arrive`"))
        })
}

/// Mosaic GPU [`Operation`] that arrives at a barrier and sets an expected transfer count.
pub trait ArriveExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the expected byte-transfer count.
    fn expect_tx(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(ArriveExpectTx);
mlir_op_trait!(ArriveExpectTx, ZeroRegions);
mlir_op_trait!(ArriveExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`ArriveExpectTxOperation`] at the specified [`Location`].
///
/// `expect_tx` must be an `i32` value containing a nonnegative byte count at execution time. The count may be computed
/// dynamically; negative counts have undefined behavior in the underlying operation.
pub fn arrive_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    expect_tx: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedArriveExpectTxOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.arrive_expect_tx", location)
        .add_operand(barrier)
        .add_operand(expect_tx)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::arrive_expect_tx`"))
        })
}

/// Mosaic GPU [`Operation`] that waits for a barrier parity.
pub trait WaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the parity value.
    fn parity(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(Wait);
mlir_op_trait!(Wait, ZeroRegions);
mlir_op_trait!(Wait, ZeroSuccessors);

/// Constructs a new detached/owned [`WaitOperation`] at the specified [`Location`].
pub fn wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    parity: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedWaitOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.wait", location)
        .add_operand(barrier)
        .add_operand(parity)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::wait`"))
        })
}

/// Mosaic GPU [`Operation`] that tries to claim a new cluster work unit.
pub trait TryClusterCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the cancellation-result shared-memory buffer.
    fn cancellation_result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the completion barrier.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }
}

mlir_op!(TryClusterCancel);
mlir_op_trait!(TryClusterCancel, ZeroRegions);
mlir_op_trait!(TryClusterCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`TryClusterCancelOperation`] at the specified [`Location`].
pub fn try_cluster_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    cancellation_result: ValueRef<'v, 'c, 't>,
    barrier: ValueRef<'v, 'c, 't>,
    predicate: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTryClusterCancelOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.try_cluster_cancel", location)
        .add_operand(cancellation_result)
        .add_operand(barrier)
        .add_operand(predicate)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::try_cluster_cancel`"))
        })
}

/// Mosaic GPU [`Operation`] that decodes the result of a cluster-cancel request.
pub trait QueryClusterCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the cancellation-result shared-memory buffer.
    fn cancellation_result(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the claimed cluster X coordinate.
    fn x(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(0)
    }

    /// Returns the claimed cluster Y coordinate.
    fn y(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(1)
    }

    /// Returns the claimed cluster Z coordinate.
    fn z(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(2)
    }

    /// Returns whether the cluster-cancel request succeeded.
    fn success(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.result(3)
    }
}

mlir_op!(QueryClusterCancel);
mlir_op_trait!(QueryClusterCancel, ZeroRegions);
mlir_op_trait!(QueryClusterCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`QueryClusterCancelOperation`] at the specified [`Location`].
pub fn query_cluster_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    cancellation_result: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedQueryClusterCancelOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.query_cluster_cancel", location)
        .add_operand(cancellation_result)
        .add_results(&[
            context.signless_integer_type(32),
            context.signless_integer_type(32),
            context.signless_integer_type(32),
            context.signless_integer_type(1),
        ])
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::query_cluster_cancel`"))
        })
}

/// Name of the [`Attribute`] that stores Mosaic GPU operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operandSegmentSizes";

/// Name of the [`Attribute`] that stores Mosaic GPU slice lengths.
pub const SLICE_LENGTHS_ATTRIBUTE: &str = "slice_lengths";

/// Name of the [`Attribute`] that stores Mosaic GPU collective dimensions.
pub const COLLECTIVE_ATTRIBUTE: &str = "collective";

/// Name of the [`Attribute`] that stores the leader-tracking copy partition strategy.
pub const LEADER_TRACKED_ATTRIBUTE: &str = "leader_tracked";

/// Name of the [`Attribute`] that stores the out-of-bounds fill mode.
pub const OOB_FILL_MODE_ATTRIBUTE: &str = "oob_fill_mode";

/// Mosaic GPU [`Operation`] that schedules an asynchronous global-to-shared memory load.
///
/// The source indices and slice lengths describe the transferred tile. The optional barrier records completion, and
/// the optional peer ID selects a global-memory peer. A false predicate suppresses the transfer. Issuing the load does
/// not make its destination immediately ready for use.
pub trait AsyncLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the completion barrier.
    fn barrier(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the optional global-memory peer device ID.
    fn global_memory_peer_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 5)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 4)?;
        if range.len() != 1 {
            return Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            )));
        }
        self.operand_value(range.start)
    }

    /// Returns the source slice lengths.
    fn slice_lengths(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(SLICE_LENGTHS_ATTRIBUTE)
    }

    /// Returns the collective cluster dimensions.
    fn collective(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(COLLECTIVE_ATTRIBUTE)
    }

    /// Returns the optional leader-tracking copy partition strategy.
    fn leader_tracked(&self) -> Result<Option<CopyPartitionAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(LEADER_TRACKED_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns the out-of-bounds fill mode.
    fn oob_fill_mode(&self) -> Result<OobFillModeAttributeRef<'c, 't>, Error> {
        self.attribute(OOB_FILL_MODE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OOB_FILL_MODE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(AsyncLoad);
mlir_op_trait!(AsyncLoad, ZeroRegions);
mlir_op_trait!(AsyncLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncLoadOperation`] at the specified [`Location`].
///
/// `indices` and `slice_lengths` must have equal lengths. `collective` contains Mosaic GPU dimension attributes;
/// `leader_tracked` selects an optional copy partition strategy, and `oob_fill_mode` controls out-of-bounds loads.
pub fn async_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    barrier: Option<ValueRef<'v, 'c, 't>>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    global_memory_peer_id: Option<ValueRef<'v, 'c, 't>>,
    slice_lengths: &[i64],
    collective: ArrayAttributeRef<'c, 't>,
    leader_tracked: Option<CopyPartitionAttributeRef<'c, 't>>,
    oob_fill_mode: OobFillMode,
    location: L,
) -> Result<DetachedAsyncLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if indices.len() != slice_lengths.len() {
        return Err(Error::invalid_argument(
            "expected equal numbers of `indices` and `slice_lengths` for `mosaic_gpu.async_load`",
        ));
    }
    let mut builder = OperationBuilder::new("mosaic_gpu.async_load", location)
        .add_operand(source)
        .add_operand(destination)
        .add_operands(barrier.as_slice())
        .add_operands(indices)
        .add_operand(predicate)
        .add_operands(global_memory_peer_id.as_slice())
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                1,
                i32::from(barrier.is_some()),
                i32::try_from(indices.len())
                    .map_err(|_| Error::invalid_argument("too many `mosaic_gpu.async_load` indices"))?,
                1,
                i32::from(global_memory_peer_id.is_some()),
            ])?,
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths)?)
        .add_attribute(COLLECTIVE_ATTRIBUTE, collective)
        .add_attribute(OOB_FILL_MODE_ATTRIBUTE, context.mosaic_gpu_oob_fill_mode_attribute(oob_fill_mode)?);
    if let Some(leader_tracked) = leader_tracked {
        builder = builder.add_attribute(LEADER_TRACKED_ATTRIBUTE, leader_tracked);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_load`"))
    })
}

/// Mosaic GPU [`Operation`] that schedules an asynchronous global-memory prefetch.
pub trait AsyncPrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns the source slice lengths.
    fn slice_lengths(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(SLICE_LENGTHS_ATTRIBUTE)
    }

    /// Returns the collective cluster dimensions.
    fn collective(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(COLLECTIVE_ATTRIBUTE)
    }
}

mlir_op!(AsyncPrefetch);
mlir_op_trait!(AsyncPrefetch, ZeroRegions);
mlir_op_trait!(AsyncPrefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncPrefetchOperation`] at the specified [`Location`].
pub fn async_prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    slice_lengths: &[i64],
    collective: ArrayAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedAsyncPrefetchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if indices.len() != slice_lengths.len() {
        return Err(Error::invalid_argument(
            "expected equal numbers of `indices` and `slice_lengths` for `mosaic_gpu.async_prefetch`",
        ));
    }
    OperationBuilder::new("mosaic_gpu.async_prefetch", location)
        .add_operand(source)
        .add_operands(indices)
        .add_operand(predicate)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                i32::try_from(indices.len())
                    .map_err(|_| Error::invalid_argument("too many `mosaic_gpu.async_prefetch` indices"))?,
                1,
            ])?,
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths)?)
        .add_attribute(COLLECTIVE_ATTRIBUTE, collective)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_prefetch`"))
        })
}

/// Name of the [`Attribute`] that stores an optional TMA reduction operation.
pub const REDUCTION_OP_ATTRIBUTE: &str = "reduction_op";

/// Name of the [`Attribute`] that stores whether an async store commits its group.
pub const COMMIT_GROUP_ATTRIBUTE: &str = "commit_group";

/// Name of the [`Attribute`] that enables broadcasting a store to all global-memory peers.
pub const IS_GLOBAL_BROADCAST_ATTRIBUTE: &str = "is_global_broadcast";

/// Mosaic GPU [`Operation`] that schedules an asynchronous shared-to-global memory store.
///
/// The index operands and slice lengths describe the destination tile. The optional peer ID selects a global-memory
/// peer; `is_global_broadcast` broadcasts to all peers. `reduction_op` optionally combines transferred values with the
/// destination, and `commit_group` controls whether the transfer group is committed.
pub trait AsyncStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the predicate operand.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns the optional global-memory peer device ID.
    fn global_memory_peer_id(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 4)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns whether the store broadcasts to all global-memory peers.
    fn is_global_broadcast(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(IS_GLOBAL_BROADCAST_ATTRIBUTE)
    }

    /// Returns the destination slice lengths.
    fn slice_lengths(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(SLICE_LENGTHS_ATTRIBUTE)
    }

    /// Returns whether this async store commits its group.
    fn commit_group(&self) -> Result<Option<BooleanAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(COMMIT_GROUP_ATTRIBUTE) {
            self.boolean_attribute(COMMIT_GROUP_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the optional TMA reduction operation.
    fn reduction_op(&self) -> Result<Option<TmaReductionAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(REDUCTION_OP_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }
}

mlir_op!(AsyncStore);
mlir_op_trait!(AsyncStore, ZeroRegions);
mlir_op_trait!(AsyncStore, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreOperation`] at the specified [`Location`].
pub fn async_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    indices: &[ValueRef<'v, 'c, 't>],
    predicate: ValueRef<'v, 'c, 't>,
    global_memory_peer_id: Option<ValueRef<'v, 'c, 't>>,
    slice_lengths: &[i64],
    commit_group: Option<bool>,
    reduction_op: Option<TmaReduction>,
    is_global_broadcast: bool,
    location: L,
) -> Result<DetachedAsyncStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if indices.len() != slice_lengths.len() {
        return Err(Error::invalid_argument(
            "expected equal numbers of `indices` and `slice_lengths` for `mosaic_gpu.async_store`",
        ));
    }
    let mut builder = OperationBuilder::new("mosaic_gpu.async_store", location)
        .add_operand(source)
        .add_operand(destination)
        .add_operands(indices)
        .add_operand(predicate)
        .add_operands(global_memory_peer_id.as_slice())
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                1,
                i32::try_from(indices.len())
                    .map_err(|_| Error::invalid_argument("too many `mosaic_gpu.async_store` indices"))?,
                1,
                i32::from(global_memory_peer_id.is_some()),
            ])?,
        )
        .add_attribute(SLICE_LENGTHS_ATTRIBUTE, context.dense_i64_array_attribute(slice_lengths)?)
        .add_attribute(IS_GLOBAL_BROADCAST_ATTRIBUTE, context.boolean_attribute(is_global_broadcast));
    if let Some(commit_group) = commit_group {
        builder = builder.add_attribute(COMMIT_GROUP_ATTRIBUTE, context.boolean_attribute(commit_group));
    }
    if let Some(reduction_op) = reduction_op {
        builder =
            builder.add_attribute(REDUCTION_OP_ATTRIBUTE, context.mosaic_gpu_tma_reduction_attribute(reduction_op)?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store`"))
    })
}

/// Name of the [`Attribute`] that stores an optimization request.
pub const OPTIMIZED_ATTRIBUTE: &str = "optimized";

/// Mosaic GPU [`Operation`] that reads a non-contiguous memref slice into a vector.
pub trait VectorLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether an optimized lowering is required.
    fn optimized(&self) -> Result<Option<BooleanAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(OPTIMIZED_ATTRIBUTE) {
            self.boolean_attribute(OPTIMIZED_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the loaded vector.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(VectorLoad);
mlir_op_trait!(VectorLoad, OneOperand);
mlir_op_trait!(VectorLoad, OneResult);
mlir_op_trait!(VectorLoad, ZeroRegions);
mlir_op_trait!(VectorLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorLoadOperation`] at the specified [`Location`].
pub fn vector_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    optimized: Option<bool>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorLoadOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let mut builder = OperationBuilder::new("mosaic_gpu.vector_load", location).add_operand(source);
    if let Some(optimized) = optimized {
        builder = builder.add_attribute(OPTIMIZED_ATTRIBUTE, context.boolean_attribute(optimized));
    }
    builder.add_result(result_type).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::vector_load`"))
    })
}

/// Name of the [`Attribute`] that stores a multimem load reduction type.
pub const REDUCTION_TYPE_ATTRIBUTE: &str = "reduction_type";

/// Mosaic GPU [`Operation`] that loads from multicast memory and reduces the loaded values.
pub trait MultimemLoadReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the multicast source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the reduction type.
    fn reduction_type(&self) -> Result<MultimemLoadReductionTypeAttributeRef<'c, 't>, Error> {
        self.attribute(REDUCTION_TYPE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                REDUCTION_TYPE_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the reduced vector.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(MultimemLoadReduce);
mlir_op_trait!(MultimemLoadReduce, OneOperand);
mlir_op_trait!(MultimemLoadReduce, OneResult);
mlir_op_trait!(MultimemLoadReduce, ZeroRegions);
mlir_op_trait!(MultimemLoadReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`MultimemLoadReduceOperation`] at the specified [`Location`].
pub fn multimem_load_reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reduction_type: MultimemLoadReductionType,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedMultimemLoadReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.multimem_load_reduce", location)
        .add_operand(source)
        .add_attribute(
            REDUCTION_TYPE_ATTRIBUTE,
            context.mosaic_gpu_multimem_load_reduction_type_attribute(reduction_type)?,
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::multimem_load_reduce`"))
        })
}

/// Name of the [`Attribute`] that stores an atomic store operation type.
pub const ATOMIC_TYPE_ATTRIBUTE: &str = "atomic_type";

/// Name of the [`Attribute`] that indicates whether multimem store instructions are used.
pub const MULTIMEM_ATTRIBUTE: &str = "multimem";

/// Mosaic GPU [`Operation`] that writes a vector to a non-contiguous memref slice.
pub trait VectorStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the vector to store.
    fn value_to_store(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns whether an optimized lowering is required.
    fn optimized(&self) -> Result<Option<BooleanAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(OPTIMIZED_ATTRIBUTE) {
            self.boolean_attribute(OPTIMIZED_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the optional atomic store operation type.
    fn atomic_type(&self) -> Result<Option<AtomicOpTypeAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(ATOMIC_TYPE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns whether this store uses multimem instructions.
    fn multimem(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(MULTIMEM_ATTRIBUTE)
    }
}

mlir_op!(VectorStore);
mlir_op_trait!(VectorStore, ZeroRegions);
mlir_op_trait!(VectorStore, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorStoreOperation`] at the specified [`Location`].
pub fn vector_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value_to_store: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    optimized: Option<bool>,
    atomic_type: Option<AtomicOpType>,
    multimem: bool,
    location: L,
) -> Result<DetachedVectorStoreOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let mut builder = OperationBuilder::new("mosaic_gpu.vector_store", location)
        .add_operand(value_to_store)
        .add_operand(destination)
        .add_attribute(MULTIMEM_ATTRIBUTE, context.boolean_attribute(multimem));
    if let Some(optimized) = optimized {
        builder = builder.add_attribute(OPTIMIZED_ATTRIBUTE, context.boolean_attribute(optimized));
    }
    if let Some(atomic_type) = atomic_type {
        builder =
            builder.add_attribute(ATOMIC_TYPE_ATTRIBUTE, context.mosaic_gpu_atomic_op_type_attribute(atomic_type)?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::vector_store`"))
    })
}

/// Name of the [`Attribute`] that stores a Mosaic GPU layout.
pub const NEW_LAYOUT_ATTRIBUTE: &str = "new_layout";

/// Mosaic GPU [`Operation`] that casts a vector to a new fragment layout.
pub trait LayoutCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input vector.
    fn x(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the strided layout when this operation casts to one.
    fn strided_layout(&self) -> Result<Option<WgStridedFragLayoutAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(NEW_LAYOUT_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns the tiled layout when this operation casts to one.
    fn tiled_layout(&self) -> Result<Option<TiledLayoutAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(NEW_LAYOUT_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns the cast result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(LayoutCast);
mlir_op_trait!(LayoutCast, OneOperand);
mlir_op_trait!(LayoutCast, OneResult);
mlir_op_trait!(LayoutCast, ZeroRegions);
mlir_op_trait!(LayoutCast, ZeroSuccessors);

/// Constructs a new detached/owned [`LayoutCastOperation`] at the specified [`Location`].
pub fn layout_cast<'v, 'c: 'v, 't: 'c, A: Attribute<'c, 't>, L: Location<'c, 't>>(
    x: ValueRef<'v, 'c, 't>,
    new_layout: A,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedLayoutCastOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.layout_cast", location)
        .add_operand(x)
        .add_attribute(NEW_LAYOUT_ATTRIBUTE, new_layout)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::layout_cast`"))
        })
}

/// Mosaic GPU [`Operation`] that casts a TMEM memref to a new TMEM layout.
pub trait TmemLayoutCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the TMEM memref.
    fn r#ref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the new tiled layout.
    fn new_layout(&self) -> Result<TiledLayoutAttributeRef<'c, 't>, Error> {
        self.attribute(NEW_LAYOUT_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                NEW_LAYOUT_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the cast result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(TmemLayoutCast);
mlir_op_trait!(TmemLayoutCast, OneOperand);
mlir_op_trait!(TmemLayoutCast, OneResult);
mlir_op_trait!(TmemLayoutCast, ZeroRegions);
mlir_op_trait!(TmemLayoutCast, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemLayoutCastOperation`] at the specified [`Location`].
pub fn tmem_layout_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    r#ref: ValueRef<'v, 'c, 't>,
    new_layout: TiledLayoutAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedTmemLayoutCastOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.tmem_layout_cast", location)
        .add_operand(r#ref)
        .add_attribute(NEW_LAYOUT_ATTRIBUTE, new_layout)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::tmem_layout_cast`"))
        })
}

/// Name of the [`Attribute`] that stores broadcast dimensions.
pub const BROADCAST_DIMENSIONS_ATTRIBUTE: &str = "broadcast_dimensions";

/// Mosaic GPU [`Operation`] that broadcasts a vector to a new shape.
pub trait BroadcastInDimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input vector.
    fn operand(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the dimensions that map operand dimensions to result dimensions.
    fn broadcast_dimensions(&self) -> Result<DenseInteger64ArrayAttributeRef<'c, 't>, Error> {
        self.dense_integer_64_array_attribute(BROADCAST_DIMENSIONS_ATTRIBUTE)
    }

    /// Returns the broadcast result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(BroadcastInDim);
mlir_op_trait!(BroadcastInDim, OneOperand);
mlir_op_trait!(BroadcastInDim, OneResult);
mlir_op_trait!(BroadcastInDim, ZeroRegions);
mlir_op_trait!(BroadcastInDim, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastInDimOperation`] at the specified [`Location`].
pub fn broadcast_in_dim<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operand: ValueRef<'v, 'c, 't>,
    broadcast_dimensions: &[i64],
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedBroadcastInDimOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.broadcast_in_dim", location)
        .add_operand(operand)
        .add_attribute(BROADCAST_DIMENSIONS_ATTRIBUTE, context.dense_i64_array_attribute(broadcast_dimensions)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::broadcast_in_dim`"))
        })
}

/// Mosaic GPU [`Operation`] that reinterprets a memref with a new shape or layout.
pub trait ReinterpretCastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the reinterpreted memref.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(ReinterpretCast);
mlir_op_trait!(ReinterpretCast, OneOperand);
mlir_op_trait!(ReinterpretCast, OneResult);
mlir_op_trait!(ReinterpretCast, ZeroRegions);
mlir_op_trait!(ReinterpretCast, ZeroSuccessors);

/// Constructs a new detached/owned [`ReinterpretCastOperation`] at the specified [`Location`].
pub fn reinterpret_cast<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedReinterpretCastOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.reinterpret_cast", location)
        .add_operand(source)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::reinterpret_cast`"))
        })
}

/// Name of the [`Attribute`] that stores a byte or tensor-memory column offset.
pub const OFFSET_ATTRIBUTE: &str = "offset";

/// Name of the [`Attribute`] that stores an optional alias identifier.
pub const ALIAS_ID_ATTRIBUTE: &str = "alias_id";

/// Mosaic GPU [`Operation`] that constructs a shared-memory memref from an offset.
pub trait SliceSmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the byte offset.
    fn offset(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(OFFSET_ATTRIBUTE)
    }

    /// Returns the optional alias identifier.
    fn alias_id(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(ALIAS_ID_ATTRIBUTE) {
            self.integer_attribute(ALIAS_ID_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the sliced shared-memory memref.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(SliceSmem);
mlir_op_trait!(SliceSmem, ZeroOperands);
mlir_op_trait!(SliceSmem, OneResult);
mlir_op_trait!(SliceSmem, ZeroRegions);
mlir_op_trait!(SliceSmem, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceSmemOperation`] at the specified [`Location`].
pub fn slice_smem<'c, 't: 'c, L: Location<'c, 't>>(
    offset: i32,
    alias_id: Option<i64>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSliceSmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if offset < 0 {
        return Err(Error::invalid_argument("expected non-negative `offset` for `mosaic_gpu.slice_smem`"));
    }
    let mut builder = OperationBuilder::new("mosaic_gpu.slice_smem", location).add_attribute(
        OFFSET_ATTRIBUTE,
        context.integer_attribute(context.signless_integer_type(32), i64::from(offset)),
    );
    if let Some(alias_id) = alias_id {
        builder = builder
            .add_attribute(ALIAS_ID_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), alias_id));
    }
    builder.add_result(result_type).build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::slice_smem`"))
    })
}

/// Mosaic GPU [`Operation`] that schedules warpgroup matrix multiply-accumulate work.
pub trait WgmmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator vector.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `a` operand.
    fn a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `b` operand.
    fn b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the accumulator result.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(Wgmma);
mlir_op_trait!(Wgmma, OneResult);
mlir_op_trait!(Wgmma, ZeroRegions);
mlir_op_trait!(Wgmma, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaOperation`] at the specified [`Location`].
pub fn wgmma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    accumulator: ValueRef<'v, 'c, 't>,
    a: ValueRef<'v, 'c, 't>,
    b: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedWgmmaOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    let result_type = accumulator.r#type()?;
    OperationBuilder::new("mosaic_gpu.wgmma", location)
        .add_operand(accumulator)
        .add_operand(a)
        .add_operand(b)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::wgmma`"))
        })
}

/// Name of the [`Attribute`] that stores whether collective tensor-core work is used.
pub const COLLECTIVE_MMA_ATTRIBUTE: &str = "collective";

/// Mosaic GPU [`Operation`] that schedules a `tcgen05.mma` matrix multiply-accumulate.
///
/// The tensor-memory accumulator is updated in place. `accumulate` selects whether its previous contents contribute
/// to the result. Optional scale operands support scaled multiplication, and optional sparse metadata describes the
/// sparsity of `a`. Completion must be tracked with [`TcGen05CommitArriveOperation`] and a barrier wait before using
/// the updated accumulator.
pub trait TcGen05MmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator memref.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the `a` operand.
    fn a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the `b` operand.
    fn b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the accumulate flag operand.
    fn accumulate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the optional `a` scale memref.
    fn a_scale(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
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

    /// Returns the optional `b` scale memref.
    fn b_scale(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
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

    /// Returns the optional sparse metadata memref for the `a` operand.
    fn a_sparse_metadata(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 6)?;
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

    /// Returns whether the MMA operation is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_MMA_ATTRIBUTE)
    }
}

mlir_op!(TcGen05Mma);
mlir_op_trait!(TcGen05Mma, ZeroRegions);
mlir_op_trait!(TcGen05Mma, ZeroSuccessors);

/// Constructs a new detached/owned [`TcGen05MmaOperation`] at the specified [`Location`].
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
) -> Result<DetachedTcGen05MmaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
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
            context.dense_i32_array_attribute(&[
                1,
                1,
                1,
                1,
                i32::from(a_scale.is_some()),
                i32::from(b_scale.is_some()),
                i32::from(a_sparse_metadata.is_some()),
            ])?,
        )
        .add_attribute(COLLECTIVE_MMA_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::tcgen05_mma`"))
        })
}

/// Mosaic GPU [`Operation`] that prevents compiler motion across a barrier.
pub trait OptimizationBarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns all barrier operands.
    fn operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns all barrier results.
    fn results(&self) -> Result<Vec<OperationResultRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| self.result(index)).collect()
    }
}

mlir_op!(OptimizationBarrier);
mlir_op_trait!(OptimizationBarrier, ZeroRegions);
mlir_op_trait!(OptimizationBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`OptimizationBarrierOperation`] at the specified [`Location`].
pub fn optimization_barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedOptimizationBarrierOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    let result_types = operands.iter().map(|operand| operand.r#type()).collect::<Result<Vec<_>, _>>()?;
    OperationBuilder::new("mosaic_gpu.optimization_barrier", location)
        .add_operands(operands)
        .add_results(&result_types)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::optimization_barrier`"))
        })
}

/// Mosaic GPU [`Operation`] that terminates a custom primitive region.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned operands.
    fn operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
pub fn r#return<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    location: L,
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.return", location)
        .add_operands(operands)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::return`"))
        })
}

/// Name of the [`Attribute`] that stores custom primitive input layouts.
pub const IN_LAYOUTS_ATTRIBUTE: &str = "in_layouts";

/// Name of the [`Attribute`] that stores custom primitive input transforms.
pub const IN_TRANSFORMS_ATTRIBUTE: &str = "in_transforms";

/// Name of the [`Attribute`] that stores custom primitive output layouts.
pub const OUT_LAYOUTS_ATTRIBUTE: &str = "out_layouts";

/// Mosaic GPU [`Operation`] that defines a custom Mosaic GPU primitive.
///
/// The body receives the input values through block arguments and terminates with [`ReturnOperation`]. Input layouts,
/// input transforms, and output layouts describe how the primitive crosses the surrounding layout boundary.
pub trait CustomPrimitiveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the custom primitive operands.
    fn operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the input layouts.
    fn in_layouts(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(IN_LAYOUTS_ATTRIBUTE)
    }

    /// Returns the input transforms.
    fn in_transforms(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(IN_TRANSFORMS_ATTRIBUTE)
    }

    /// Returns the output layouts.
    fn out_layouts(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(OUT_LAYOUTS_ATTRIBUTE)
    }

    /// Returns the custom primitive body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }
}

mlir_op!(CustomPrimitive);
mlir_op_trait!(CustomPrimitive, OneRegion);
mlir_op_trait!(CustomPrimitive, ZeroSuccessors);

/// Constructs a new detached/owned [`CustomPrimitiveOperation`] at the specified [`Location`].
pub fn custom_primitive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    in_layouts: ArrayAttributeRef<'c, 't>,
    in_transforms: ArrayAttributeRef<'c, 't>,
    out_layouts: ArrayAttributeRef<'c, 't>,
    result_types: &[TypeRef<'c, 't>],
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedCustomPrimitiveOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.custom_primitive", location)
        .add_operands(operands)
        .add_attribute(IN_LAYOUTS_ATTRIBUTE, in_layouts)
        .add_attribute(IN_TRANSFORMS_ATTRIBUTE, in_transforms)
        .add_attribute(OUT_LAYOUTS_ATTRIBUTE, out_layouts)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::custom_primitive`"))
        })
}

/// Mosaic GPU [`Operation`] that evaluates a block in parallel on all warps.
///
/// The isolated body contains one block whose arguments correspond to the operands. It has no terminator and cannot
/// implicitly capture values from the enclosing region.
pub trait WarpMapOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the values captured by this warp map.
    fn operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the warp-map region.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.as_ref().region(0)
    }
}

mlir_op!(WarpMap);
mlir_op_trait!(WarpMap, OneRegion);
mlir_op_trait!(WarpMap, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpMapOperation`] at the specified [`Location`].
pub fn warp_map<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedWarpMapOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.warp_map", location)
        .add_operands(operands)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::warp_map`"))
        })
}

/// Name of the [`Attribute`] that stores shared-memory transforms.
pub const TRANSFORMS_ATTRIBUTE: &str = "transforms";

/// Mosaic GPU [`Operation`] that attaches transforms to a memref without changing the memref.
pub trait WithTransformsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input memref.
    fn r#ref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the transforms.
    fn transforms(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(TRANSFORMS_ATTRIBUTE)
    }

    /// Returns the transformed memref.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(WithTransforms);
mlir_op_trait!(WithTransforms, OneOperand);
mlir_op_trait!(WithTransforms, OneResult);
mlir_op_trait!(WithTransforms, ZeroRegions);
mlir_op_trait!(WithTransforms, ZeroSuccessors);

/// Constructs a new detached/owned [`WithTransformsOperation`] at the specified [`Location`].
pub fn with_transforms<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    r#ref: ValueRef<'v, 'c, 't>,
    transforms: ArrayAttributeRef<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedWithTransformsOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.with_transforms", location)
        .add_operand(r#ref)
        .add_attribute(TRANSFORMS_ATTRIBUTE, transforms)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::with_transforms`"))
        })
}

/// Name of the [`Attribute`] that stores whether an operation is collective.
pub const COLLECTIVE_TMEM_ATTRIBUTE: &str = "collective";

/// Name of the [`Attribute`] that stores a tensor-memory packing factor.
pub const PACKING_ATTRIBUTE: &str = "packing";

/// Mosaic GPU [`Operation`] that allocates tensor memory.
///
/// The allocation pointer is written to a rank-zero shared-memory `i32` memref. The result is a rank-two tensor-memory
/// memref. `packing` is a positive packing factor; collective allocation coordinates two thread blocks.
pub trait TmemAllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory pointer used to store the allocation pointer.
    fn smem_ptr(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether the allocation is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }

    /// Returns the tensor-memory packing factor.
    fn packing(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(PACKING_ATTRIBUTE)
    }

    /// Returns the allocated tensor-memory memref.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(TmemAlloc);
mlir_op_trait!(TmemAlloc, OneOperand);
mlir_op_trait!(TmemAlloc, OneResult);
mlir_op_trait!(TmemAlloc, ZeroRegions);
mlir_op_trait!(TmemAlloc, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemAllocOperation`] at the specified [`Location`].
pub fn tmem_alloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    smem_ptr: ValueRef<'v, 'c, 't>,
    collective: bool,
    packing: i32,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedTmemAllocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if packing <= 0 {
        return Err(Error::invalid_argument("expected positive `packing` for `mosaic_gpu.tmem_alloc`"));
    }
    OperationBuilder::new("mosaic_gpu.tmem_alloc", location)
        .add_operand(smem_ptr)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .add_attribute(
            PACKING_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(packing)),
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::tmem_alloc`"))
        })
}

/// Mosaic GPU [`Operation`] that relinquishes tensor-memory allocation permission.
///
/// Once a thread executes this operation, its thread block must not issue further tensor-memory allocations.
pub trait TmemRelinquishAllocPermitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns whether this applies to collective tensor-memory allocations.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }
}

mlir_op!(TmemRelinquishAllocPermit);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroOperands);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroRegions);
mlir_op_trait!(TmemRelinquishAllocPermit, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemRelinquishAllocPermitOperation`] at the specified [`Location`].
pub fn tmem_relinquish_alloc_permit<'c, 't: 'c, L: Location<'c, 't>>(
    collective: bool,
    location: L,
) -> Result<DetachedTmemRelinquishAllocPermitOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.tmem_relinquish_alloc_permit", location)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `mosaic_gpu::tmem_relinquish_alloc_permit`")
            })
        })
}

/// Mosaic GPU [`Operation`] that deallocates tensor memory.
pub trait TmemDeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor-memory memref.
    fn tmem_ref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(TmemDealloc);
mlir_op_trait!(TmemDealloc, ZeroRegions);
mlir_op_trait!(TmemDealloc, ZeroSuccessors);

/// Constructs a new detached/owned [`TmemDeallocOperation`] at the specified [`Location`].
pub fn tmem_dealloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    tmem_ref: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedTmemDeallocOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.tmem_dealloc", location).add_operand(tmem_ref).build().and_then(
        |operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::tmem_dealloc`"))
        },
    )
}

/// Name of the [`Attribute`] that selects an optional tensor-memory load reduction.
pub const REDUCE_ATTRIBUTE: &str = "reduce";

/// Mosaic GPU [`Operation`] that copies tensor memory into registers asynchronously.
///
/// Without a reduction, this produces the loaded vector. A reduction also produces a second vector containing the
/// reduced values.
pub trait AsyncLoadTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the tensor-memory source.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the loaded vector followed by the reduced vector, when a reduction was requested.
    fn results(&self) -> Result<Vec<OperationResultRef<'o, 'c, 't>>, Error> {
        (0..self.result_count()).map(|index| self.result(index)).collect()
    }

    /// Returns the optional tensor-memory load reduction.
    fn reduction(&self) -> Result<Option<TmemLoadReductionAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(REDUCE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }
}

mlir_op!(AsyncLoadTmem);
mlir_op_trait!(AsyncLoadTmem, ZeroRegions);
mlir_op_trait!(AsyncLoadTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncLoadTmemOperation`] at the specified [`Location`].
///
/// Result types are inferred from `source`: the loaded vector retains its shape and element type, and an optional
/// reduction produces a second vector with the final dimension removed.
pub fn async_load_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    reduction: Option<TmemLoadReduction>,
    location: L,
) -> Result<DetachedAsyncLoadTmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let mut builder = OperationBuilder::new("mosaic_gpu.async_load_tmem", location)
        .add_operand(source)
        .enable_result_type_inference();
    if let Some(reduction) = reduction {
        builder = builder.add_attribute(REDUCE_ATTRIBUTE, context.mosaic_gpu_tmem_load_reduction_attribute(reduction)?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_load_tmem`"))
    })
}

/// Mosaic GPU [`Operation`] that copies registers into tensor memory asynchronously.
pub trait AsyncStoreTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source vector.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor-memory destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }
}

mlir_op!(AsyncStoreTmem);
mlir_op_trait!(AsyncStoreTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreTmemOperation`] at the specified [`Location`].
pub fn async_store_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedAsyncStoreTmemOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.async_store_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store_tmem`"))
        })
}

/// Mosaic GPU [`Operation`] that copies shared memory into tensor memory asynchronously.
pub trait AsyncStoreSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory source.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor-memory destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }
}

mlir_op!(AsyncStoreSmemToTmem);
mlir_op_trait!(AsyncStoreSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreSmemToTmemOperation`] at the specified [`Location`].
pub fn async_store_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> Result<DetachedAsyncStoreSmemToTmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.async_store_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store_smem_to_tmem`"))
        })
}

/// Mosaic GPU [`Operation`] that copies sparse metadata from shared memory into tensor memory asynchronously.
pub trait AsyncStoreSparseMetadataSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory sparse metadata source.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor-memory sparse metadata destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }
}

mlir_op!(AsyncStoreSparseMetadataSmemToTmem);
mlir_op_trait!(AsyncStoreSparseMetadataSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreSparseMetadataSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreSparseMetadataSmemToTmemOperation`] at the specified [`Location`].
pub fn async_store_sparse_metadata_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> Result<DetachedAsyncStoreSparseMetadataSmemToTmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.async_store_sparse_metadata_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store_sparse_metadata_smem_to_tmem`")
            })
        })
}

/// Mosaic GPU [`Operation`] that copies MMA scales from shared memory into tensor memory asynchronously.
pub trait AsyncStoreScalesSmemToTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the shared-memory scales source.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor-memory scales destination.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns whether this copy is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }
}

mlir_op!(AsyncStoreScalesSmemToTmem);
mlir_op_trait!(AsyncStoreScalesSmemToTmem, ZeroRegions);
mlir_op_trait!(AsyncStoreScalesSmemToTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreScalesSmemToTmemOperation`] at the specified [`Location`].
pub fn async_store_scales_smem_to_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> Result<DetachedAsyncStoreScalesSmemToTmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.async_store_scales_smem_to_tmem", location)
        .add_operand(source)
        .add_operand(destination)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| {
                Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store_scales_smem_to_tmem`")
            })
        })
}

/// Mosaic GPU [`Operation`] that slices a tensor-memory memref.
///
/// The offset is measured in tensor-memory columns and must be a nonnegative multiple of four. An optional 64-bit
/// alias ID distinguishes potentially aliasing allocations.
pub trait SliceTmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source tensor-memory memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the tensor-memory column offset.
    fn offset(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(OFFSET_ATTRIBUTE)
    }

    /// Returns the optional identifier used to distinguish potentially aliasing allocations.
    fn alias_id(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(ALIAS_ID_ATTRIBUTE) {
            self.integer_attribute(ALIAS_ID_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the sliced tensor-memory memref.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(SliceTmem);
mlir_op_trait!(SliceTmem, OneOperand);
mlir_op_trait!(SliceTmem, OneResult);
mlir_op_trait!(SliceTmem, ZeroRegions);
mlir_op_trait!(SliceTmem, ZeroSuccessors);

/// Constructs a new detached/owned [`SliceTmemOperation`] at the specified [`Location`].
pub fn slice_tmem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    offset: i32,
    alias_id: Option<i64>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedSliceTmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if offset < 0 {
        return Err(Error::invalid_argument("expected non-negative `offset` for `mosaic_gpu.slice_tmem`"));
    }
    let mut builder = OperationBuilder::new("mosaic_gpu.slice_tmem", location)
        .add_operand(source)
        .add_attribute(
            OFFSET_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(offset)),
        )
        .add_result(result_type);
    if let Some(alias_id) = alias_id {
        builder = builder
            .add_attribute(ALIAS_ID_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), alias_id));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::slice_tmem`"))
    })
}

/// Mosaic GPU [`Operation`] that makes a barrier track prior async `tcgen05` operations.
pub trait TcGen05CommitArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the barrier memref.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns whether this commit-arrive operation is collective.
    fn collective(&self) -> Result<BooleanAttributeRef<'c, 't>, Error> {
        self.boolean_attribute(COLLECTIVE_TMEM_ATTRIBUTE)
    }
}

mlir_op!(TcGen05CommitArrive);
mlir_op_trait!(TcGen05CommitArrive, ZeroRegions);
mlir_op_trait!(TcGen05CommitArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`TcGen05CommitArriveOperation`] at the specified [`Location`].
pub fn tcgen05_commit_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    barrier: ValueRef<'v, 'c, 't>,
    collective: bool,
    location: L,
) -> Result<DetachedTcGen05CommitArriveOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.tcgen05_commit_arrive", location)
        .add_operand(barrier)
        .add_attribute(COLLECTIVE_TMEM_ATTRIBUTE, context.boolean_attribute(collective))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::tcgen05_commit_arrive`"))
        })
}

/// Name of the [`Attribute`] that stores a debug format string.
pub const FORMAT_ATTRIBUTE: &str = "format";

/// Mosaic GPU [`Operation`] that prints a value from inside a Mosaic GPU kernel.
pub trait DebugPrintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the format string.
    fn format(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(FORMAT_ATTRIBUTE)
    }

    /// Returns the value to print.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(DebugPrint);
mlir_op_trait!(DebugPrint, ZeroRegions);
mlir_op_trait!(DebugPrint, ZeroSuccessors);

/// Constructs a new detached/owned [`DebugPrintOperation`] at the specified [`Location`].
pub fn debug_print<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    format: &str,
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedDebugPrintOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.debug_print", location)
        .add_operand(value)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::debug_print`"))
        })
}

/// Mosaic GPU [`Operation`] that prints the layout of a value.
pub trait PrintLayoutOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the format string.
    fn format(&self) -> Result<StringAttributeRef<'c, 't>, Error> {
        self.string_attribute(FORMAT_ATTRIBUTE)
    }

    /// Returns the value whose layout is printed.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(PrintLayout);
mlir_op_trait!(PrintLayout, ZeroRegions);
mlir_op_trait!(PrintLayout, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintLayoutOperation`] at the specified [`Location`].
pub fn print_layout<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    format: &str,
    value: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedPrintLayoutOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.print_layout", location)
        .add_operand(value)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::print_layout`"))
        })
}

/// Name of the [`Attribute`] that stores an iota dimension.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Mosaic GPU [`Operation`] that creates a broadcasted iota vector.
pub trait BroadcastedIotaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the iota dimension.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the iota vector.
    fn result(&self) -> Result<OperationResultRef<'o, 'c, 't>, Error> {
        self.as_ref().result(0)
    }
}

mlir_op!(BroadcastedIota);
mlir_op_trait!(BroadcastedIota, ZeroOperands);
mlir_op_trait!(BroadcastedIota, OneResult);
mlir_op_trait!(BroadcastedIota, ZeroRegions);
mlir_op_trait!(BroadcastedIota, ZeroSuccessors);

/// Constructs a new detached/owned [`BroadcastedIotaOperation`] at the specified [`Location`].
pub fn broadcasted_iota<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: i32,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedBroadcastedIotaOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if dimension < 0 {
        return Err(Error::invalid_argument("expected non-negative `dimension` for `mosaic_gpu.broadcasted_iota`"));
    }
    OperationBuilder::new("mosaic_gpu.broadcasted_iota", location)
        .add_attribute(
            DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(dimension)),
        )
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::broadcasted_iota`"))
        })
}

/// Mosaic GPU [`Operation`] that computes `lhs @ rhs + accumulator` synchronously.
pub trait MmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the accumulator vector.
    fn accumulator(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the left matrix.
    fn lhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the right matrix.
    fn rhs(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }
}

mlir_op!(Mma);
mlir_op_trait!(Mma, OneResult);
mlir_op_trait!(Mma, ZeroRegions);
mlir_op_trait!(Mma, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaOperation`] at the specified [`Location`].
pub fn mma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    accumulator: ValueRef<'v, 'c, 't>,
    lhs: ValueRef<'v, 'c, 't>,
    rhs: ValueRef<'v, 'c, 't>,
    location: L,
) -> Result<DetachedMmaOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.mma", location)
        .add_operands(&[accumulator, lhs, rhs])
        .add_result(accumulator.r#type()?)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::mma`"))
        })
}

/// Mosaic GPU [`Operation`] that concatenates vectors along a dimension.
pub trait VectorConcatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the concatenation dimension.
    fn dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(DIMENSION_ATTRIBUTE)
    }

    /// Returns the vectors in concatenation order.
    fn operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }
}

mlir_op!(VectorConcat);
mlir_op_trait!(VectorConcat, OneResult);
mlir_op_trait!(VectorConcat, ZeroRegions);
mlir_op_trait!(VectorConcat, ZeroSuccessors);

/// Constructs a new detached/owned [`VectorConcatOperation`] at the specified [`Location`].
pub fn vector_concat<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    dimension: i32,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> Result<DetachedVectorConcatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if operands.is_empty() || dimension < 0 {
        return Err(Error::invalid_argument("expected nonempty vectors and a nonnegative concatenation dimension"));
    }
    OperationBuilder::new("mosaic_gpu.vector_concat", location)
        .add_operands(operands)
        .add_result(result_type)
        .add_attribute(
            DIMENSION_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(dimension)),
        )
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::vector_concat`"))
        })
}

/// Name of the [`Attribute`] that stores the assumed positive divisor.
pub const MULTIPLE_ATTRIBUTE: &str = "multiple";

/// Mosaic GPU [`Operation`] that assumes an integer value is divisible by a positive constant.
pub trait AssumeMultipleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the assumed divisor.
    fn multiple(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(MULTIPLE_ATTRIBUTE)
    }

    /// Returns the value to which the divisibility assumption applies.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(AssumeMultiple);
mlir_op_trait!(AssumeMultiple, OneOperand);
mlir_op_trait!(AssumeMultiple, OneResult);
mlir_op_trait!(AssumeMultiple, ZeroRegions);
mlir_op_trait!(AssumeMultiple, ZeroSuccessors);

/// Constructs a new detached/owned [`AssumeMultipleOperation`] at the specified [`Location`].
///
/// `multiple` must be positive. This records a compiler assumption; it does not check divisibility at execution time.
pub fn assume_multiple<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    multiple: i32,
    location: L,
) -> Result<DetachedAssumeMultipleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    if multiple <= 0 {
        return Err(Error::invalid_argument("expected a positive `multiple`"));
    }
    OperationBuilder::new("mosaic_gpu.assume_multiple", location)
        .add_operand(value)
        .add_result(value.r#type()?)
        .add_attribute(
            MULTIPLE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), i64::from(multiple)),
        )
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::assume_multiple`"))
        })
}

/// Mosaic GPU [`Operation`] that maps a memref to a peer block's shared memory.
pub trait GetClusterRefOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional X coordinate of the peer block.
    fn x(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the optional Y coordinate of the peer block.
    fn y(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }

    /// Returns the optional Z coordinate of the peer block.
    fn z(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 3)?
            .next()
            .map(|index| self.operand_value(index))
            .transpose()
    }
}

mlir_op!(GetClusterRef);
mlir_op_trait!(GetClusterRef, OneResult);
mlir_op_trait!(GetClusterRef, ZeroRegions);
mlir_op_trait!(GetClusterRef, ZeroSuccessors);

/// Constructs a new detached/owned [`GetClusterRefOperation`] at the specified [`Location`].
///
/// `coordinates` contains optional `i32` operands in X, Y, Z order. An absent coordinate keeps the current block's
/// coordinate along that dimension. The result preserves the source shape, element type, and layout while using
/// the cluster shared-memory space.
pub fn get_cluster_ref<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'v, 'c, 't>,
    coordinates: [Option<ValueRef<'v, 'c, 't>>; 3],
    location: L,
) -> Result<DetachedGetClusterRefOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    OperationBuilder::new("mosaic_gpu.get_cluster_ref", location)
        .add_operand(source)
        .add_operands(coordinates[0].as_slice())
        .add_operands(coordinates[1].as_slice())
        .add_operands(coordinates[2].as_slice())
        .enable_result_type_inference()
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[
                1,
                i32::from(coordinates[0].is_some()),
                i32::from(coordinates[1].is_some()),
                i32::from(coordinates[2].is_some()),
            ])?,
        )
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::get_cluster_ref`"))
        })
}

/// Name of the [`Attribute`] that stores the destination cluster dimension.
pub const CLUSTER_DIMENSION_ATTRIBUTE: &str = "cluster_dim";

/// Mosaic GPU [`Operation`] that asynchronously stores register values into a cluster peer's shared memory.
pub trait AsyncStoreSmemOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value being stored.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn destination(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the completion barrier.
    fn barrier(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the destination cluster dimension.
    fn cluster_dimension(&self) -> Result<DimensionAttributeRef<'c, 't>, Error> {
        self.attribute(CLUSTER_DIMENSION_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{CLUSTER_DIMENSION_ATTRIBUTE}` attribute in `{}`",
                self.name(),
            ))
        })
    }

    /// Returns the peer block index within the cluster dimension.
    fn cluster_index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(3)
    }

    /// Returns the optional atomic reduction applied at the destination.
    fn atomic_type(&self) -> Result<Option<AtomicOpTypeAttributeRef<'c, 't>>, Error> {
        Ok(self.attribute(ATOMIC_TYPE_ATTRIBUTE)?.and_then(|attribute| attribute.cast()))
    }

    /// Returns whether an optimized lowering is explicitly requested.
    fn optimized(&self) -> Result<Option<BooleanAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(OPTIMIZED_ATTRIBUTE) {
            self.boolean_attribute(OPTIMIZED_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }
}

mlir_op!(AsyncStoreSmem);
mlir_op_trait!(AsyncStoreSmem, ZeroRegions);
mlir_op_trait!(AsyncStoreSmem, ZeroSuccessors);

/// Constructs a new detached/owned [`AsyncStoreSmemOperation`] at the specified [`Location`].
pub fn async_store_smem<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'v, 'c, 't>,
    destination: ValueRef<'v, 'c, 't>,
    barrier: ValueRef<'v, 'c, 't>,
    cluster_dimension: Dimension,
    cluster_index: ValueRef<'v, 'c, 't>,
    atomic_type: Option<AtomicOpType>,
    optimized: Option<bool>,
    location: L,
) -> Result<DetachedAsyncStoreSmemOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::mosaic_gpu()?)?;
    let mut builder = OperationBuilder::new("mosaic_gpu.async_store_smem", location)
        .add_operands(&[value, destination, barrier, cluster_index])
        .add_attribute(CLUSTER_DIMENSION_ATTRIBUTE, context.mosaic_gpu_dimension_attribute(cluster_dimension)?);
    if let Some(atomic_type) = atomic_type {
        builder =
            builder.add_attribute(ATOMIC_TYPE_ATTRIBUTE, context.mosaic_gpu_atomic_op_type_attribute(atomic_type)?);
    }
    if let Some(optimized) = optimized {
        builder = builder.add_attribute(OPTIMIZED_ATTRIBUTE, context.boolean_attribute(optimized));
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `mosaic_gpu::async_store_smem`"))
    })
}

/// Name of the temporary upstream capability marker. New programs should not emit this operation.
pub const ARRIVE_DYN_EXPECT_TX_SUPPORTED_OPERATION_NAME: &str = "mosaic_gpu.arrive_dyn_expect_tx_supported";

/// Temporary upstream capability marker, exposed for inspecting existing IR only.
pub trait ArriveDynExpectTxSupportedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(ArriveDynExpectTxSupported);
mlir_op_trait!(ArriveDynExpectTxSupported, ZeroOperands);
mlir_op_trait!(ArriveDynExpectTxSupported, ZeroRegions);
mlir_op_trait!(ArriveDynExpectTxSupported, ZeroSuccessors);

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::gpu::AddressSpace;
    use crate::{
        Attribute, Block, Context, DetachedBlock, DialectHandle, Location, Module, Operation, Region, Size, Type,
        TypeRef, Value, VectorTypeDimension,
    };

    use super::*;

    /// Constructs a statically shaped memref fixture with an explicit memory space.
    fn memref_type<'c, 't, L: Location<'c, 't>>(
        context: &'c Context<'t>,
        element_type: TypeRef<'c, 't>,
        shape: &[usize],
        memory_space: Option<crate::AttributeRef<'c, 't>>,
        location: L,
    ) -> TypeRef<'c, 't> {
        let shape = shape.iter().copied().map(Size::Static).collect::<Vec<_>>();
        context.mem_ref_type(element_type, shape.as_slice(), None, memory_space, location).unwrap().as_ref()
    }

    /// Constructs a statically shaped vector fixture.
    fn vector_type<'c, 't, L: Location<'c, 't>>(
        context: &'c Context<'t>,
        element_type: TypeRef<'c, 't>,
        shape: &[usize],
        location: L,
    ) -> TypeRef<'c, 't> {
        let shape = shape.iter().copied().map(VectorTypeDimension::Fixed).collect::<Vec<_>>();
        context.vector_type(element_type, shape.as_slice(), location).unwrap().as_ref()
    }

    /// Inserts a function containing the tested operations and a void terminator.
    fn append_void_function<'c, 't, L: Copy + Location<'c, 't>>(
        module: &Module<'c, 't>,
        name: &str,
        argument_types: &[TypeRef<'c, 't>],
        mut block: DetachedBlock<'c, 't>,
        location: L,
    ) {
        block.append_operation(func::r#return(&[] as &[crate::ValueRef], location).unwrap()).unwrap();
        module
            .body()
            .unwrap()
            .append_operation(
                func::func(
                    name,
                    func::FuncAttributes {
                        arguments: argument_types.iter().copied().map(Into::into).collect(),
                        results: Vec::new(),
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap(),
            )
            .unwrap();
    }

    #[test]
    fn test_initialize_barrier_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let pointer_type = context.llvm_pointer_type(3).unwrap().as_ref();
        let mut block = context.block(&[(pointer_type, location)]);
        let pointer = block.argument(0).unwrap().as_ref();
        let operation = initialize_barrier(pointer, 4, 2, true, location).unwrap();
        assert_eq!(operation.base_pointer().unwrap(), pointer);
        assert_eq!(operation.arrival_count().unwrap().signless_value(), 4);
        assert_eq!(operation.num_barriers().unwrap().signless_value(), 2);
        assert!(operation.orders_tensor_core().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "initialize_barrier", &[pointer_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @initialize_barrier(%arg0: !llvm.ptr<3>) {
                    \"mosaic_gpu.initialize_barrier\"(%arg0) <{arrival_count = 4 : i64, num_barriers = 2 : \
                i32, orders_tensor_core = true}> : (!llvm.ptr<3>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_arrive_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let barrier_type = context.mosaic_gpu_barrier_type(true).unwrap().as_ref();
        let barrier_memref = memref_type(&context, barrier_type, &[], Some(workgroup), location);
        let mut block = context.block(&[(barrier_memref, location)]);
        let barrier = block.argument(0).unwrap().as_ref();
        let operation = arrive(barrier, true, location).unwrap();
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert!(operation.orders_tensor_core().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "arrive", &[barrier_memref], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @arrive(%arg0: memref<!mosaic_gpu.barrier<orders_tensor_core = true>, \
                #gpu.address_space<workgroup>>) {
                    \"mosaic_gpu.arrive\"(%arg0) <{orders_tensor_core = true}> : \
                (memref<!mosaic_gpu.barrier<orders_tensor_core = true>, #gpu.address_space<workgroup>>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_arrive_expect_tx_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let barrier_type = context.mosaic_gpu_barrier_type(false).unwrap().as_ref();
        let barrier_memref = memref_type(&context, barrier_type, &[], Some(workgroup), location);
        let count_type = context.signless_integer_type(32).as_ref();
        let argument_types = [barrier_memref, count_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let barrier = block.argument(0).unwrap().as_ref();
        let count = block.argument(1).unwrap().as_ref();
        let operation = arrive_expect_tx(barrier, count, location).unwrap();
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert_eq!(operation.expect_tx().unwrap(), count);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "arrive_expect_tx", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @arrive_expect_tx(%arg0: memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>, %arg1: i32) {
                    mosaic_gpu.arrive_expect_tx barrier(%arg0 : memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>) %arg1
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_wait_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let barrier_memref = memref_type(
            &context,
            context.mosaic_gpu_barrier_type(false).unwrap().as_ref(),
            &[],
            Some(workgroup),
            location,
        );
        let predicate_type = context.signless_integer_type(1).as_ref();
        let mut block = context.block(&[(barrier_memref, location), (predicate_type, location)]);
        let barrier = block.argument(0).unwrap().as_ref();
        let parity = block.argument(1).unwrap().as_ref();
        let operation = wait(barrier, parity, location).unwrap();
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert_eq!(operation.parity().unwrap(), parity);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "wait", &[barrier_memref, predicate_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @wait(%arg0: memref<!mosaic_gpu.barrier, #gpu.address_space<workgroup>>, %arg1: \
                i1) {
                    mosaic_gpu.wait barrier(%arg0 : memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>) parity(%arg1 : i1)
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_try_cluster_cancel_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let cancellation_type =
            memref_type(&context, context.signless_integer_type(8).as_ref(), &[16], Some(workgroup), location);
        let barrier_type = memref_type(
            &context,
            context.mosaic_gpu_barrier_type(false).unwrap().as_ref(),
            &[],
            Some(workgroup),
            location,
        );
        let predicate_type = context.signless_integer_type(1).as_ref();
        let argument_types = [cancellation_type, barrier_type, predicate_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let cancellation = block.argument(0).unwrap().as_ref();
        let barrier = block.argument(1).unwrap().as_ref();
        let predicate = block.argument(2).unwrap().as_ref();
        let operation = try_cluster_cancel(cancellation, barrier, predicate, location).unwrap();
        assert_eq!(operation.cancellation_result().unwrap(), cancellation);
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert_eq!(operation.predicate().unwrap(), predicate);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "try_cluster_cancel", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @try_cluster_cancel(%arg0: memref<16xi8, #gpu.address_space<workgroup>>, %arg1: \
                memref<!mosaic_gpu.barrier, #gpu.address_space<workgroup>>, %arg2: i1) {
                    \"mosaic_gpu.try_cluster_cancel\"(%arg0, %arg1, %arg2) : (memref<16xi8, \
                #gpu.address_space<workgroup>>, memref<!mosaic_gpu.barrier, #gpu.address_space<workgroup>>, \
                i1) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_query_cluster_cancel_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let cancellation_type =
            memref_type(&context, context.signless_integer_type(8).as_ref(), &[16], Some(workgroup), location);
        let mut block = context.block(&[(cancellation_type, location)]);
        let cancellation = block.argument(0).unwrap().as_ref();
        let operation = query_cluster_cancel(cancellation, location).unwrap();
        assert_eq!(operation.cancellation_result().unwrap(), cancellation);
        assert_eq!(operation.x().unwrap().r#type().unwrap(), context.signless_integer_type(32));
        assert_eq!(operation.y().unwrap().r#type().unwrap(), context.signless_integer_type(32));
        assert_eq!(operation.z().unwrap().r#type().unwrap(), context.signless_integer_type(32));
        assert_eq!(operation.success().unwrap().r#type().unwrap(), context.signless_integer_type(1));
        block.append_operation(operation).unwrap();
        append_void_function(&module, "query_cluster_cancel", &[cancellation_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @query_cluster_cancel(%arg0: memref<16xi8, #gpu.address_space<workgroup>>) {
                    %x, %y, %z, %success = \"mosaic_gpu.query_cluster_cancel\"(%arg0) : (memref<16xi8, \
                #gpu.address_space<workgroup>>) -> (i32, i32, i32, i1)
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_load_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let float_type = context.float32_type().as_ref();
        let source_type = memref_type(&context, float_type, &[8, 16], None, location);
        let destination_type = memref_type(&context, float_type, &[8, 16], Some(workgroup), location);
        let barrier_type = memref_type(
            &context,
            context.mosaic_gpu_barrier_type(false).unwrap().as_ref(),
            &[],
            Some(workgroup),
            location,
        );
        let integer_type = context.signless_integer_type(32).as_ref();
        let predicate_type = context.signless_integer_type(1).as_ref();
        let argument_types = [source_type, destination_type, barrier_type, integer_type, integer_type, predicate_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = (0..6).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let collective = context.array_attribute(&[] as &[crate::AttributeRef]);
        let operation = async_load(
            values[0],
            values[1],
            Some(values[2]),
            &values[3..5],
            values[5],
            None,
            &[8, 16],
            collective,
            None,
            OobFillMode::Zeros,
            location,
        )
        .unwrap();
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.destination().unwrap(), values[1]);
        assert_eq!(operation.barrier().unwrap(), Some(values[2]));
        assert_eq!(operation.indices().unwrap(), values[3..5]);
        assert_eq!(operation.predicate().unwrap(), values[5]);
        assert_eq!(operation.slice_lengths().unwrap().values().collect::<Vec<_>>(), vec![8, 16]);
        assert_eq!(operation.oob_fill_mode().unwrap().value().unwrap(), OobFillMode::Zeros);
        assert_eq!(operation.global_memory_peer_id().unwrap(), None);
        assert_eq!(operation.collective().unwrap(), collective);
        assert_eq!(operation.leader_tracked().unwrap(), None);
        // Omitting the barrier must not shift the indices, predicate, or peer operand.
        let peer_load = async_load(
            values[0],
            values[1],
            None,
            &values[3..5],
            values[5],
            Some(values[3]),
            &[8, 16],
            collective,
            None,
            OobFillMode::Zeros,
            location,
        )
        .unwrap();
        assert_eq!(peer_load.barrier().unwrap(), None);
        assert_eq!(peer_load.indices().unwrap(), values[3..5]);
        assert_eq!(peer_load.predicate().unwrap(), values[5]);
        assert_eq!(peer_load.global_memory_peer_id().unwrap(), Some(values[3]));
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_load", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_load(%arg0: memref<8x16xf32>, %arg1: memref<8x16xf32, \
                #gpu.address_space<workgroup>>, %arg2: memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>, %arg3: i32, %arg4: i32, %arg5: i1) {
                    \"mosaic_gpu.async_load\"(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) <{collective = [], \
                oob_fill_mode = 2 : i32, operandSegmentSizes = array<i32: 1, 1, 1, 2, 1, 0>, slice_lengths = \
                array<i64: 8, 16>}> : (memref<8x16xf32>, memref<8x16xf32, #gpu.address_space<workgroup>>, \
                memref<!mosaic_gpu.barrier, #gpu.address_space<workgroup>>, i32, i32, i1) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_prefetch_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[8, 16], None, location);
        let integer_type = context.signless_integer_type(32).as_ref();
        let predicate_type = context.signless_integer_type(1).as_ref();
        let argument_types = [source_type, integer_type, integer_type, predicate_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = (0..4).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let collective = context.array_attribute(&[] as &[crate::AttributeRef]);
        let operation = async_prefetch(values[0], &values[1..3], values[3], &[8, 16], collective, location).unwrap();
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.indices().unwrap(), values[1..3]);
        assert_eq!(operation.predicate().unwrap(), values[3]);
        assert_eq!(operation.slice_lengths().unwrap().values().collect::<Vec<_>>(), vec![8, 16]);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_prefetch", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_prefetch(%arg0: memref<8x16xf32>, %arg1: i32, %arg2: i32, %arg3: i1) {
                    \"mosaic_gpu.async_prefetch\"(%arg0, %arg1, %arg2, %arg3) <{collective = [], \
                operandSegmentSizes = array<i32: 1, 2, 1>, slice_lengths = array<i64: 8, 16>}> : \
                (memref<8x16xf32>, i32, i32, i1) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let float_type = context.float32_type().as_ref();
        let source_type = memref_type(&context, float_type, &[8, 16], Some(workgroup), location);
        let destination_type = memref_type(&context, float_type, &[8, 16], None, location);
        let integer_type = context.signless_integer_type(32).as_ref();
        let predicate_type = context.signless_integer_type(1).as_ref();
        let argument_types = [source_type, destination_type, integer_type, integer_type, predicate_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = (0..5).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let operation = async_store(
            values[0],
            values[1],
            &values[2..4],
            values[4],
            None,
            &[8, 16],
            Some(true),
            Some(TmaReduction::Add),
            false,
            location,
        )
        .unwrap();
        assert_eq!(operation.source().unwrap(), values[0]);
        assert_eq!(operation.destination().unwrap(), values[1]);
        assert_eq!(operation.indices().unwrap(), values[2..4]);
        assert_eq!(operation.predicate().unwrap(), values[4]);
        assert_eq!(operation.reduction_op().unwrap().unwrap().value().unwrap(), TmaReduction::Add);
        assert_eq!(operation.global_memory_peer_id().unwrap(), None);
        assert!(!operation.is_global_broadcast().unwrap().value());
        let peer_store = async_store(
            values[0],
            values[1],
            &values[2..4],
            values[4],
            Some(values[2]),
            &[8, 16],
            None,
            None,
            true,
            location,
        )
        .unwrap();
        assert_eq!(peer_store.global_memory_peer_id().unwrap(), Some(values[2]));
        assert_eq!(peer_store.indices().unwrap(), values[2..4]);
        assert_eq!(peer_store.predicate().unwrap(), values[4]);
        assert!(peer_store.is_global_broadcast().unwrap().value());
        assert_eq!(peer_store.reduction_op().unwrap(), None);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store(%arg0: memref<8x16xf32, #gpu.address_space<workgroup>>, %arg1: \
                memref<8x16xf32>, %arg2: i32, %arg3: i32, %arg4: i1) {
                    \"mosaic_gpu.async_store\"(%arg0, %arg1, %arg2, %arg3, %arg4) <{commit_group = true, \
                is_global_broadcast = false, operandSegmentSizes = array<i32: 1, 1, 2, 1, 0>, reduction_op = \
                0 : i32, slice_lengths = array<i64: 8, 16>}> : (memref<8x16xf32, \
                #gpu.address_space<workgroup>>, memref<8x16xf32>, i32, i32, i1) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_vector_load_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[4, 8], None, location);
        let result_type = vector_type(&context, context.float32_type().as_ref(), &[4, 8], location);
        let mut block = context.block(&[(source_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let operation = vector_load(source, Some(true), result_type, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.optimized().unwrap().unwrap().value(), true);
        assert_eq!(VectorLoadOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "vector_load", &[source_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @vector_load(%arg0: memref<4x8xf32>) {
                    %0 = \"mosaic_gpu.vector_load\"(%arg0) <{optimized = true}> : (memref<4x8xf32>) -> \
                vector<4x8xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_multimem_load_reduce_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[4, 8], None, location);
        let result_type = vector_type(&context, context.float32_type().as_ref(), &[4, 8], location);
        let mut block = context.block(&[(source_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let operation = multimem_load_reduce(source, MultimemLoadReductionType::Add, result_type, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.reduction_type().unwrap().value().unwrap(), MultimemLoadReductionType::Add);
        assert_eq!(MultimemLoadReduceOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "multimem_load_reduce", &[source_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @multimem_load_reduce(%arg0: memref<4x8xf32>) {
                    %0 = \"mosaic_gpu.multimem_load_reduce\"(%arg0) <{reduction_type = 0 : i32}> : \
                (memref<4x8xf32>) -> vector<4x8xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_vector_store_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let float_type = context.float32_type().as_ref();
        let vector = vector_type(&context, float_type, &[4, 8], location);
        let destination = memref_type(&context, float_type, &[4, 8], None, location);
        let argument_types = [vector, destination];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let value = block.argument(0).unwrap().as_ref();
        let destination_value = block.argument(1).unwrap().as_ref();
        let operation =
            vector_store(value, destination_value, Some(true), Some(AtomicOpType::Add), true, location).unwrap();
        assert_eq!(operation.value_to_store().unwrap(), value);
        assert_eq!(operation.destination().unwrap(), destination_value);
        assert_eq!(operation.atomic_type().unwrap().unwrap().value().unwrap(), AtomicOpType::Add);
        assert!(operation.multimem().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "vector_store", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @vector_store(%arg0: vector<4x8xf32>, %arg1: memref<4x8xf32>) {
                    \"mosaic_gpu.vector_store\"(%arg0, %arg1) <{atomic_type = 0 : i32, multimem = true, \
                optimized = true}> : (vector<4x8xf32>, memref<4x8xf32>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_layout_cast_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let layout = context
            .mosaic_gpu_wg_strided_frag_layout_attribute(context.dense_i64_array_attribute(&[4]).unwrap(), 1)
            .unwrap();
        let mut block = context.block(&[(vector, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = layout_cast(value, layout, vector, location).unwrap();
        assert_eq!(operation.x().unwrap(), value);
        assert_eq!(operation.strided_layout().unwrap(), Some(layout));
        assert!(operation.tiled_layout().unwrap().is_none());
        assert_eq!(LayoutCastOperation::result(&operation).unwrap().r#type().unwrap(), vector);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "layout_cast", &[vector], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @layout_cast(%arg0: vector<4xf32>) {
                    %0 = mosaic_gpu.layout_cast x(%arg0 : vector<4xf32>) {new_layout = \
                #mosaic_gpu.WGStridedFragLayout<[4], 1>}
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tmem_layout_cast_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let tmem_ref = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let empty = context.array_attribute(&[] as &[crate::AttributeRef]);
        let layout = context.mosaic_gpu_tiled_layout_attribute(empty, empty, empty, 0).unwrap();
        let mut block = context.block(&[(tmem_ref, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = tmem_layout_cast(value, layout, tmem_ref, location).unwrap();
        assert_eq!(operation.r#ref().unwrap(), value);
        assert_eq!(operation.new_layout().unwrap(), layout);
        assert_eq!(TmemLayoutCastOperation::result(&operation).unwrap().r#type().unwrap(), tmem_ref);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tmem_layout_cast", &[tmem_ref], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tmem_layout_cast(%arg0: memref<32x32xf32, #mosaic_gpu.tmem>) {
                    %0 = \"mosaic_gpu.tmem_layout_cast\"(%arg0) <{new_layout = #mosaic_gpu.TiledLayout<[], \
                warp_dims = [], lane_dims = [], vector_dim = 0>}> : (memref<32x32xf32, #mosaic_gpu.tmem>) -> \
                memref<32x32xf32, #mosaic_gpu.tmem>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_broadcast_in_dim_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let input = vector_type(&context, context.float32_type().as_ref(), &[1], location);
        let output = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let mut block = context.block(&[(input, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = broadcast_in_dim(value, &[0], output, location).unwrap();
        assert_eq!(BroadcastInDimOperation::operand(&operation).unwrap(), value);
        assert_eq!(operation.broadcast_dimensions().unwrap().values().collect::<Vec<_>>(), vec![0]);
        assert_eq!(BroadcastInDimOperation::result(&operation).unwrap().r#type().unwrap(), output);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "broadcast_in_dim", &[input], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @broadcast_in_dim(%arg0: vector<1xf32>) {
                    %0 = mosaic_gpu.broadcast_in_dim(%arg0 : vector<1xf32>) {broadcast_dimensions = \
                array<i64: 0>} -> vector<4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reinterpret_cast_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[2, 8], None, location);
        let result_type = memref_type(&context, context.float32_type().as_ref(), &[4, 4], None, location);
        let mut block = context.block(&[(source_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let operation = reinterpret_cast(source, result_type, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(ReinterpretCastOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "reinterpret_cast", &[source_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @reinterpret_cast(%arg0: memref<2x8xf32>) {
                    %0 = \"mosaic_gpu.reinterpret_cast\"(%arg0) : (memref<2x8xf32>) -> memref<4x4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_slice_smem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let result_type = memref_type(&context, context.float32_type().as_ref(), &[4, 4], Some(workgroup), location);
        let mut block = context.block_with_no_arguments();
        let without_alias = slice_smem(16, None, result_type, location).unwrap();
        assert_eq!(without_alias.alias_id().unwrap(), None);
        let operation = slice_smem(16, Some(4294967296), result_type, location).unwrap();
        assert_eq!(operation.offset().unwrap().signless_value(), 16);
        assert_eq!(operation.alias_id().unwrap().unwrap().signless_value(), 4294967296);
        assert_eq!(SliceSmemOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "slice_smem", &[], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @slice_smem() {
                    %0 = \"mosaic_gpu.slice_smem\"() <{alias_id = 4294967296 : i64, offset = 16 : i32}> : () \
                -> memref<4x4xf32, #gpu.address_space<workgroup>>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_wgmma_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let accumulator_type = vector_type(&context, context.float32_type().as_ref(), &[64, 8], location);
        let a_type = memref_type(&context, context.float16_type().as_ref(), &[64, 16], Some(workgroup), location);
        let b_type = memref_type(&context, context.float16_type().as_ref(), &[16, 8], Some(workgroup), location);
        let argument_types = [accumulator_type, a_type, b_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = (0..3).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let operation = wgmma(values[0], values[1], values[2], location).unwrap();
        assert_eq!(operation.accumulator().unwrap(), values[0]);
        assert_eq!(operation.a().unwrap(), values[1]);
        assert_eq!(operation.b().unwrap(), values[2]);
        assert_eq!(WgmmaOperation::result(&operation).unwrap().r#type().unwrap(), accumulator_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "wgmma", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @wgmma(%arg0: vector<64x8xf32>, %arg1: memref<64x16xf16, \
                #gpu.address_space<workgroup>>, %arg2: memref<16x8xf16, #gpu.address_space<workgroup>>) {
                    %0 = mosaic_gpu.wgmma accumulator(%arg0 : vector<64x8xf32>) a(%arg1 : memref<64x16xf16, \
                #gpu.address_space<workgroup>>) b(%arg2 : memref<16x8xf16, #gpu.address_space<workgroup>>) -> \
                vector<64x8xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tcgen05_mma_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let accumulator_type = memref_type(&context, context.float32_type().as_ref(), &[32, 8], Some(tmem), location);
        let a_type = memref_type(&context, context.float16_type().as_ref(), &[32, 16], Some(workgroup), location);
        let b_type = memref_type(&context, context.float16_type().as_ref(), &[16, 8], Some(workgroup), location);
        let predicate_type = context.signless_integer_type(1).as_ref();
        let argument_types = [accumulator_type, a_type, b_type, predicate_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = (0..4).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let operation =
            tcgen05_mma(values[0], values[1], values[2], values[3], None, None, None, false, location).unwrap();
        assert_eq!(operation.accumulator().unwrap(), values[0]);
        assert_eq!(operation.a().unwrap(), values[1]);
        assert_eq!(operation.b().unwrap(), values[2]);
        assert_eq!(operation.accumulate().unwrap(), values[3]);
        assert!(operation.a_scale().unwrap().is_none());
        assert!(operation.a_sparse_metadata().unwrap().is_none());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tcgen05_mma", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tcgen05_mma(%arg0: memref<32x8xf32, #mosaic_gpu.tmem>, %arg1: memref<32x16xf16, \
                #gpu.address_space<workgroup>>, %arg2: memref<16x8xf16, #gpu.address_space<workgroup>>, \
                %arg3: i1) {
                    \"mosaic_gpu.tcgen05_mma\"(%arg0, %arg1, %arg2, %arg3) <{collective = false, \
                operandSegmentSizes = array<i32: 1, 1, 1, 1, 0, 0, 0>}> : (memref<32x8xf32, \
                #mosaic_gpu.tmem>, memref<32x16xf16, #gpu.address_space<workgroup>>, memref<16x8xf16, \
                #gpu.address_space<workgroup>>, i1) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_optimization_barrier_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let integer_type = context.signless_integer_type(32).as_ref();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let argument_types = [integer_type, vector];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let values = [block.argument(0).unwrap().as_ref(), block.argument(1).unwrap().as_ref()];
        let operation = optimization_barrier(&values, location).unwrap();
        assert_eq!(operation.operand_values().collect::<Result<Vec<_>, _>>().unwrap(), values);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), integer_type);
        assert_eq!(operation.result(1).unwrap().r#type().unwrap(), vector);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "optimization_barrier", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @optimization_barrier(%arg0: i32, %arg1: vector<4xf32>) {
                    %0:2 = \"mosaic_gpu.optimization_barrier\"(%arg0, %arg1) : (i32, vector<4xf32>) -> (i32, \
                vector<4xf32>)
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_return_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let layout = context
            .mosaic_gpu_wg_strided_frag_layout_attribute(context.dense_i64_array_attribute(&[4]).unwrap(), 1)
            .unwrap();
        let layouts = context.array_attribute(&[layout.as_ref()]);
        let empty = context.array_attribute(&[] as &[crate::AttributeRef]);
        let mut body = context.region();
        let mut body_block = context.block(&[(vector, location)]);
        let result = body_block.argument(0).unwrap().as_ref();
        let operation = r#return(&[result], location).unwrap();
        assert_eq!(operation.operand_values().collect::<Result<Vec<_>, _>>().unwrap(), vec![result]);
        body_block.append_operation(operation).unwrap();
        body.append_block(body_block).unwrap();
        let mut block = context.block(&[(vector, location)]);
        let argument = block.argument(0).unwrap().as_ref();
        block
            .append_operation(
                custom_primitive(&[argument], layouts, empty, layouts, &[vector], body, location).unwrap(),
            )
            .unwrap();
        append_void_function(&module, "return", &[vector], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @return(%arg0: vector<4xf32>) {
                    %0 = \"mosaic_gpu.custom_primitive\"(%arg0) <{in_layouts = \
                [#mosaic_gpu.WGStridedFragLayout<[4], 1>], in_transforms = [], out_layouts = \
                [#mosaic_gpu.WGStridedFragLayout<[4], 1>]}> ({
                    ^bb0(%arg1: vector<4xf32>):
                      mosaic_gpu.return %arg1 : vector<4xf32>
                    }) : (vector<4xf32>) -> vector<4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_custom_primitive_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let layout = context
            .mosaic_gpu_wg_strided_frag_layout_attribute(context.dense_i64_array_attribute(&[4]).unwrap(), 1)
            .unwrap();
        let layouts = context.array_attribute(&[layout.as_ref()]);
        let empty = context.array_attribute(&[] as &[crate::AttributeRef]);
        let mut body = context.region();
        let mut body_block = context.block(&[(vector, location)]);
        let body_argument = body_block.argument(0).unwrap().as_ref();
        body_block.append_operation(r#return(&[body_argument], location).unwrap()).unwrap();
        body.append_block(body_block).unwrap();
        let mut block = context.block(&[(vector, location)]);
        let argument = block.argument(0).unwrap().as_ref();
        let operation = custom_primitive(&[argument], layouts, empty, layouts, &[vector], body, location).unwrap();
        assert_eq!(CustomPrimitiveOperation::operands(&operation).unwrap(), vec![argument]);
        assert_eq!(operation.in_layouts().unwrap(), layouts);
        assert_eq!(operation.out_layouts().unwrap(), layouts);
        assert_eq!(operation.body().unwrap().blocks().unwrap().count(), 1);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "custom_primitive", &[vector], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @custom_primitive(%arg0: vector<4xf32>) {
                    %0 = \"mosaic_gpu.custom_primitive\"(%arg0) <{in_layouts = \
                [#mosaic_gpu.WGStridedFragLayout<[4], 1>], in_transforms = [], out_layouts = \
                [#mosaic_gpu.WGStridedFragLayout<[4], 1>]}> ({
                    ^bb0(%arg1: vector<4xf32>):
                      mosaic_gpu.return %arg1 : vector<4xf32>
                    }) : (vector<4xf32>) -> vector<4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_warp_map_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let integer_type = context.signless_integer_type(32).as_ref();
        let mut region = context.region();
        region.append_block(context.block(&[(integer_type, location)])).unwrap();
        let mut block = context.block(&[(integer_type, location)]);
        let argument = block.argument(0).unwrap().as_ref();
        let operation = warp_map(&[argument], region, location).unwrap();
        assert_eq!(WarpMapOperation::operands(&operation).unwrap(), vec![argument]);
        assert_eq!(WarpMapOperation::region(&operation).unwrap().blocks().unwrap().count(), 1);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "warp_map", &[integer_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @warp_map(%arg0: i32) {
                    \"mosaic_gpu.warp_map\"(%arg0) ({
                    ^bb0(%arg1: i32):
                    }) : (i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_with_transforms_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let memref = memref_type(&context, context.float32_type().as_ref(), &[4, 4], Some(workgroup), location);
        let transform = context.mosaic_gpu_tile_transform_attribute(&[2, 2]).unwrap();
        let transforms = context.array_attribute(&[transform.as_ref()]);
        let mut block = context.block(&[(memref, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = with_transforms(value, transforms, memref, location).unwrap();
        assert_eq!(operation.r#ref().unwrap(), value);
        assert_eq!(operation.transforms().unwrap(), transforms);
        assert_eq!(WithTransformsOperation::result(&operation).unwrap().r#type().unwrap(), memref);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "with_transforms", &[memref], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @with_transforms(%arg0: memref<4x4xf32, #gpu.address_space<workgroup>>) {
                    %0 = \"mosaic_gpu.with_transforms\"(%arg0) <{transforms = [#mosaic_gpu.tile<[2, 2]>]}> : \
                (memref<4x4xf32, #gpu.address_space<workgroup>>) -> memref<4x4xf32, \
                #gpu.address_space<workgroup>>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tmem_alloc_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let smem_pointer =
            memref_type(&context, context.signless_integer_type(32).as_ref(), &[], Some(workgroup), location);
        let result_type =
            memref_type(&context, context.signless_integer_type(8).as_ref(), &[32, 64], Some(tmem), location);
        let mut block = context.block(&[(smem_pointer, location)]);
        let pointer = block.argument(0).unwrap().as_ref();
        let operation = tmem_alloc(pointer, true, 4, result_type, location).unwrap();
        assert_eq!(operation.smem_ptr().unwrap(), pointer);
        assert!(operation.collective().unwrap().value());
        assert_eq!(operation.packing().unwrap().signless_value(), 4);
        assert_eq!(TmemAllocOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tmem_alloc", &[smem_pointer], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tmem_alloc(%arg0: memref<i32, #gpu.address_space<workgroup>>) {
                    %0 = mosaic_gpu.tmem_alloc smem_ptr(%arg0 : memref<i32, #gpu.address_space<workgroup>>) \
                {collective = true, packing = 4 : i32} -> memref<32x64xi8, #mosaic_gpu.tmem>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tmem_relinquish_alloc_permit_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mut block = context.block_with_no_arguments();
        let operation = tmem_relinquish_alloc_permit(true, location).unwrap();
        assert!(operation.collective().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tmem_relinquish_alloc_permit", &[], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tmem_relinquish_alloc_permit() {
                    \"mosaic_gpu.tmem_relinquish_alloc_permit\"() <{collective = true}> : () -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tmem_dealloc_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let tmem_ref = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let mut block = context.block(&[(tmem_ref, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = tmem_dealloc(value, location).unwrap();
        assert_eq!(operation.tmem_ref().unwrap(), value);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tmem_dealloc", &[tmem_ref], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tmem_dealloc(%arg0: memref<32x32xf32, #mosaic_gpu.tmem>) {
                    mosaic_gpu.tmem_dealloc tmem_ref(%arg0 : memref<32x32xf32, #mosaic_gpu.tmem>)
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_load_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let result_type = vector_type(&context, context.float32_type().as_ref(), &[32, 32], location);
        let mut block = context.block(&[(source_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let operation = async_load_tmem(source, None, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(AsyncLoadTmemOperation::results(&operation).unwrap(), vec![operation.result(0).unwrap()]);
        assert!(operation.reduction().unwrap().is_none());
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), result_type);
        let reduced_type = vector_type(&context, context.float32_type().as_ref(), &[32], location);
        let reduction = async_load_tmem(source, Some(TmemLoadReduction::Max), location).unwrap();
        assert_eq!(reduction.reduction().unwrap().unwrap().value(), Ok(TmemLoadReduction::Max));
        assert_eq!(
            AsyncLoadTmemOperation::results(&reduction).unwrap(),
            vec![reduction.result(0).unwrap(), reduction.result(1).unwrap()]
        );
        assert_eq!(reduction.result(1).unwrap().r#type().unwrap(), reduced_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_load_tmem", &[source_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_load_tmem(%arg0: memref<32x32xf32, #mosaic_gpu.tmem>) {
                    %0 = \"mosaic_gpu.async_load_tmem\"(%arg0) : (memref<32x32xf32, #mosaic_gpu.tmem>) -> \
                vector<32x32xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[32, 32], location);
        let destination = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let argument_types = [vector, destination];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let source = block.argument(0).unwrap().as_ref();
        let destination_value = block.argument(1).unwrap().as_ref();
        let operation = async_store_tmem(source, destination_value, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination().unwrap(), destination_value);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store_tmem", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store_tmem(%arg0: vector<32x32xf32>, %arg1: memref<32x32xf32, \
                #mosaic_gpu.tmem>) {
                    \"mosaic_gpu.async_store_tmem\"(%arg0, %arg1) : (vector<32x32xf32>, memref<32x32xf32, \
                #mosaic_gpu.tmem>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_smem_to_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(workgroup), location);
        let destination_type = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let argument_types = [source_type, destination_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let source = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let operation = async_store_smem_to_tmem(source, destination, true, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination().unwrap(), destination);
        assert!(operation.collective().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store_smem_to_tmem", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store_smem_to_tmem(%arg0: memref<32x32xf32, \
                #gpu.address_space<workgroup>>, %arg1: memref<32x32xf32, #mosaic_gpu.tmem>) {
                    \"mosaic_gpu.async_store_smem_to_tmem\"(%arg0, %arg1) <{collective = true}> : \
                (memref<32x32xf32, #gpu.address_space<workgroup>>, memref<32x32xf32, #mosaic_gpu.tmem>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_sparse_metadata_smem_to_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let i2 = context.signless_integer_type(2).as_ref();
        let source_type = memref_type(&context, i2, &[1, 1, 128, 64], Some(workgroup), location);
        let destination_type = memref_type(&context, i2, &[128, 64], Some(tmem), location);
        let argument_types = [source_type, destination_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let source = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let operation = async_store_sparse_metadata_smem_to_tmem(source, destination, true, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination().unwrap(), destination);
        assert!(operation.collective().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store_sparse_metadata_smem_to_tmem", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store_sparse_metadata_smem_to_tmem(%arg0: memref<1x1x128x64xi2, \
                #gpu.address_space<workgroup>>, %arg1: memref<128x64xi2, #mosaic_gpu.tmem>) {
                    \"mosaic_gpu.async_store_sparse_metadata_smem_to_tmem\"(%arg0, %arg1) <{collective = \
                true}> : (memref<1x1x128x64xi2, #gpu.address_space<workgroup>>, memref<128x64xi2, \
                #mosaic_gpu.tmem>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_scales_smem_to_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let scale = context.float8e8m0fnu_type().as_ref();
        let source_type = memref_type(&context, scale, &[1, 1, 32, 16], Some(workgroup), location);
        let destination_type = memref_type(&context, scale, &[128, 4], Some(tmem), location);
        let argument_types = [source_type, destination_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let source = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let operation = async_store_scales_smem_to_tmem(source, destination, true, location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination().unwrap(), destination);
        assert!(operation.collective().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store_scales_smem_to_tmem", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store_scales_smem_to_tmem(%arg0: memref<1x1x32x16xf8E8M0FNU, \
                #gpu.address_space<workgroup>>, %arg1: memref<128x4xf8E8M0FNU, #mosaic_gpu.tmem>) {
                    \"mosaic_gpu.async_store_scales_smem_to_tmem\"(%arg0, %arg1) <{collective = true}> : \
                (memref<1x1x32x16xf8E8M0FNU, #gpu.address_space<workgroup>>, memref<128x4xf8E8M0FNU, \
                #mosaic_gpu.tmem>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_slice_tmem_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tmem = context.mosaic_gpu_tmem_attribute().unwrap().as_ref();
        let source_type = memref_type(&context, context.float32_type().as_ref(), &[32, 64], Some(tmem), location);
        let result_type = memref_type(&context, context.float32_type().as_ref(), &[32, 32], Some(tmem), location);
        let mut block = context.block(&[(source_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let without_alias = slice_tmem(source, 4, None, result_type, location).unwrap();
        assert_eq!(without_alias.alias_id().unwrap(), None);
        let operation = slice_tmem(source, 4, Some(4294967296), result_type, location).unwrap();
        assert_eq!(operation.alias_id().unwrap().unwrap().signless_value(), 4294967296);
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.offset().unwrap().signless_value(), 4);
        assert_eq!(SliceTmemOperation::result(&operation).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "slice_tmem", &[source_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @slice_tmem(%arg0: memref<32x64xf32, #mosaic_gpu.tmem>) {
                    %0 = \"mosaic_gpu.slice_tmem\"(%arg0) <{alias_id = 4294967296 : i64, offset = 4 : i32}> : \
                (memref<32x64xf32, #mosaic_gpu.tmem>) -> memref<32x32xf32, #mosaic_gpu.tmem>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_tcgen05_commit_arrive_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let barrier_type = memref_type(
            &context,
            context.mosaic_gpu_barrier_type(false).unwrap().as_ref(),
            &[],
            Some(workgroup),
            location,
        );
        let mut block = context.block(&[(barrier_type, location)]);
        let barrier = block.argument(0).unwrap().as_ref();
        let operation = tcgen05_commit_arrive(barrier, true, location).unwrap();
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert!(operation.collective().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "tcgen05_commit_arrive", &[barrier_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @tcgen05_commit_arrive(%arg0: memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>) {
                    \"mosaic_gpu.tcgen05_commit_arrive\"(%arg0) <{collective = true}> : \
                (memref<!mosaic_gpu.barrier, #gpu.address_space<workgroup>>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_debug_print_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let mut block = context.block(&[(vector, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = debug_print("value = {}", value, location).unwrap();
        assert_eq!(operation.format().unwrap().string().as_str(), Ok("value = {}"));
        assert_eq!(operation.value().unwrap(), value);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "debug_print", &[vector], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @debug_print(%arg0: vector<4xf32>) {
                    \"mosaic_gpu.debug_print\"(%arg0) <{format = \"value = {}\"}> : (vector<4xf32>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_print_layout_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let mut block = context.block(&[(vector, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = print_layout("layout = {}", value, location).unwrap();
        assert_eq!(operation.format().unwrap().string().as_str(), Ok("layout = {}"));
        assert_eq!(operation.value().unwrap(), value);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "print_layout", &[vector], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @print_layout(%arg0: vector<4xf32>) {
                    \"mosaic_gpu.print_layout\"(%arg0) <{format = \"layout = {}\"}> : (vector<4xf32>) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_broadcasted_iota_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        context.load_dialect(DialectHandle::gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let vector = vector_type(&context, context.signless_integer_type(32).as_ref(), &[4, 8], location);
        let mut block = context.block_with_no_arguments();
        let operation = broadcasted_iota(1, vector, location).unwrap();
        assert_eq!(operation.dimension().unwrap().signless_value(), 1);
        assert_eq!(BroadcastedIotaOperation::result(&operation).unwrap().r#type().unwrap(), vector);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "broadcasted_iota", &[], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @broadcasted_iota() {
                    %0 = \"mosaic_gpu.broadcasted_iota\"() <{dimension = 1 : i32}> : () -> vector<4x8xi32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_mma_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let accumulator_type = vector_type(&context, context.float32_type().as_ref(), &[16, 8], location);
        let left_type = vector_type(&context, context.float16_type().as_ref(), &[16, 16], location);
        let right_type = vector_type(&context, context.float16_type().as_ref(), &[16, 8], location);
        let argument_types = [accumulator_type, left_type, right_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let accumulator = block.argument(0).unwrap().as_ref();
        let left = block.argument(1).unwrap().as_ref();
        let right = block.argument(2).unwrap().as_ref();
        let operation = mma(accumulator, left, right, location).unwrap();
        assert_eq!(operation.accumulator().unwrap(), accumulator);
        assert_eq!(operation.lhs().unwrap(), left);
        assert_eq!(operation.rhs().unwrap(), right);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), accumulator_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "mma", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @mma(%arg0: vector<16x8xf32>, %arg1: vector<16x16xf16>, %arg2: vector<16x8xf16>) {
                    %0 = mosaic_gpu.mma accumulator(%arg0 : vector<16x8xf32>) a(%arg1 : vector<16x16xf16>) \
                b(%arg2 : vector<16x8xf16>) -> vector<16x8xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_vector_concat_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let input_type = vector_type(&context, context.float32_type().as_ref(), &[2], location);
        let result_type = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let mut block = context.block(&[(input_type, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = vector_concat(&[value, value], 0, result_type, location).unwrap();
        assert_eq!(VectorConcatOperation::operands(&operation).unwrap(), vec![value, value]);
        assert_eq!(operation.dimension().unwrap().signless_value(), 0);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), result_type);
        assert!(matches!(vector_concat(&[], 0, result_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected nonempty vectors and a nonnegative concatenation dimension"));
        assert!(matches!(vector_concat(&[value], -1, result_type, location),
            Err(Error::InvalidArgument { message, .. })
                if message == "expected nonempty vectors and a nonnegative concatenation dimension"));
        block.append_operation(operation).unwrap();
        append_void_function(&module, "vector_concat", &[input_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @vector_concat(%arg0: vector<2xf32>) {
                    %0 = mosaic_gpu.vector_concat(%arg0, %arg0 : vector<2xf32>, vector<2xf32>) {dimension = 0 \
                : i32} -> vector<4xf32>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_assume_multiple_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let integer_type = context.signless_integer_type(32).as_ref();
        let mut block = context.block(&[(integer_type, location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = assume_multiple(value, 16, location).unwrap();
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.multiple().unwrap().signless_value(), 16);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), integer_type);
        assert!(matches!(assume_multiple(value, 0, location),
            Err(Error::InvalidArgument { message, .. }) if message == "expected a positive `multiple`"));
        assert!(matches!(assume_multiple(value, -1, location),
            Err(Error::InvalidArgument { message, .. }) if message == "expected a positive `multiple`"));
        block.append_operation(operation).unwrap();
        append_void_function(&module, "assume_multiple", &[integer_type], block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @assume_multiple(%arg0: i32) {
                    %0 = mosaic_gpu.assume_multiple %arg0, 16 : i32
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_get_cluster_ref_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let memory_type = memref_type(&context, context.float32_type().as_ref(), &[4], Some(workgroup), location);
        let cluster = context.mosaic_gpu_smem_cluster_attribute().unwrap().as_ref();
        let result_type = memref_type(&context, context.float32_type().as_ref(), &[4], Some(cluster), location);
        let integer_type = context.signless_integer_type(32).as_ref();
        let argument_types = [memory_type, integer_type];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let source = block.argument(0).unwrap().as_ref();
        let coordinate = block.argument(1).unwrap().as_ref();
        let operation = get_cluster_ref(source, [None, Some(coordinate), None], location).unwrap();
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.x().unwrap(), None);
        assert_eq!(operation.y().unwrap(), Some(coordinate));
        assert_eq!(operation.z().unwrap(), None);
        assert_eq!(operation.result(0).unwrap().r#type().unwrap(), result_type);
        block.append_operation(operation).unwrap();
        append_void_function(&module, "get_cluster_ref", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @get_cluster_ref(%arg0: memref<4xf32, #gpu.address_space<workgroup>>, %arg1: i32) \
                {
                    %0 = \"mosaic_gpu.get_cluster_ref\"(%arg0, %arg1) <{operandSegmentSizes = array<i32: 1, \
                0, 1, 0>}> : (memref<4xf32, #gpu.address_space<workgroup>>, i32) -> memref<4xf32, \
                #mosaic_gpu.smem_cluster>
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_async_store_smem_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let workgroup = context.gpu_address_space_attribute(AddressSpace::Workgroup).unwrap().as_ref();
        let vector = vector_type(&context, context.float32_type().as_ref(), &[4], location);
        let memory = memref_type(&context, context.float32_type().as_ref(), &[4], Some(workgroup), location);
        let barrier = memref_type(
            &context,
            context.mosaic_gpu_barrier_type(false).unwrap().as_ref(),
            &[],
            Some(workgroup),
            location,
        );
        let integer = context.signless_integer_type(32).as_ref();
        let argument_types = [vector, memory, barrier, integer];
        let mut block = context.block(&argument_types.map(|r#type| (r#type, location)));
        let value = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let barrier = block.argument(2).unwrap().as_ref();
        let index = block.argument(3).unwrap().as_ref();
        let operation = async_store_smem(
            value,
            destination,
            barrier,
            Dimension::X,
            index,
            Some(AtomicOpType::Add),
            Some(true),
            location,
        )
        .unwrap();
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.destination().unwrap(), destination);
        assert_eq!(operation.barrier().unwrap(), barrier);
        assert_eq!(operation.cluster_dimension().unwrap().value(), Ok(Dimension::X));
        assert_eq!(operation.cluster_index().unwrap(), index);
        assert_eq!(operation.atomic_type().unwrap().unwrap().value(), Ok(AtomicOpType::Add));
        assert!(operation.optimized().unwrap().unwrap().value());
        block.append_operation(operation).unwrap();
        append_void_function(&module, "async_store_smem", &argument_types, block, location);
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @async_store_smem(%arg0: vector<4xf32>, %arg1: memref<4xf32, \
                #gpu.address_space<workgroup>>, %arg2: memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>, %arg3: i32) {
                    \"mosaic_gpu.async_store_smem\"(%arg0, %arg1, %arg2, %arg3) <{atomic_type = \
                #mosaic_gpu<atomic_op_type add>, cluster_dim = #mosaic_gpu<dimension x>, optimized = true}> : \
                (vector<4xf32>, memref<4xf32, #gpu.address_space<workgroup>>, memref<!mosaic_gpu.barrier, \
                #gpu.address_space<workgroup>>, i32) -> ()
                    return
                  }
                }
            "},
        );
    }

    #[test]
    fn test_arrive_dyn_expect_tx_supported_operation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::mosaic_gpu().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        // This temporary upstream marker has no public constructor. Its wrapper still supports inspecting
        // existing IR, so construct the fixture through the generic builder.
        let operation = OperationBuilder::new(ARRIVE_DYN_EXPECT_TX_SUPPORTED_OPERATION_NAME, location).build().unwrap();
        let operation = unsafe { operation.cast::<DetachedArriveDynExpectTxSupportedOperation>() }.unwrap();
        assert_eq!(operation.operand_count(), 0);
        assert_eq!(operation.result_count(), 0);
        module.body().unwrap().append_operation(operation).unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  \"mosaic_gpu.arrive_dyn_expect_tx_supported\"() : () -> ()
                }
            "},
        );
    }
}
