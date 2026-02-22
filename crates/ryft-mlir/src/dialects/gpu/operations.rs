// TODO(eaplatanios): Clean this up and make sure it is correct.

use crate::{
    ARGUMENT_ATTRIBUTES_ATTRIBUTE, ArrayAttributeRef, Attribute, AttributeRef, DenseInteger32ArrayAttributeRef,
    DetachedOp, DetachedRegion, DialectHandle, FUNCTION_TYPE_ATTRIBUTE, FlatSymbolRefAttributeRef, FromWithContext,
    FunctionTypeRef, HasCallableArgumentAndResultAttributes, IntegerAttributeRef, IntoWithContext, Location, Operation,
    OperationBuilder, RESULT_ATTRIBUTES_ATTRIBUTE, ReturnLike, SYMBOL_NAME_ATTRIBUTE, SYMBOL_VISIBILITY_ATTRIBUTE,
    StringAttributeRef, StringRef, Symbol, SymbolRefAttributeRef, SymbolVisibility, Type, TypeAttributeRef, TypeRef,
    Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::attributes::{
    AddressSpaceAttributeRef, AllReduceOperationAttributeRef, BroadcastTypeAttributeRef, Dimension,
    DimensionAttributeRef, MmaElementwiseOpAttributeRef, Prune2To4SpMatFlagAttributeRef, ShuffleModeAttributeRef,
    SpGemmWorkEstimationOrComputeKindAttributeRef, TransposeModeAttributeRef,
};

/// All known `gpu` operation mnemonics.
pub const OPERATIONS: &[&str] = &[
    "all_reduce",
    "alloc",
    "barrier",
    "binary",
    "block_dim",
    "block_id",
    "cluster_block_id",
    "cluster_dim_blocks",
    "cluster_dim",
    "cluster_id",
    "create_2to4_spmat",
    "create_bsr",
    "create_coo_aos",
    "create_coo",
    "create_csc",
    "create_csr",
    "create_dn_tensor",
    "dealloc",
    "destroy_dn_tensor",
    "destroy_sp_mat",
    "dynamic_shared_memory",
    "func",
    "module",
    "global_id",
    "grid_dim",
    "host_register",
    "host_unregister",
    "lane_id",
    "launch_func",
    "launch",
    "memcpy",
    "memset",
    "num_subgroups",
    "printf",
    "return",
    "rotate",
    "sddmm_buffer_size",
    "sddmm",
    "set_csr_pointers",
    "set_default_device",
    "shuffle",
    "spgemm_copy",
    "spgemm_create_descr",
    "spgemm_destroy_descr",
    "spgemm_work_estimation_or_compute",
    "spmm_buffer_size",
    "spmm",
    "spmv_buffer_size",
    "spmv",
    "spmat_get_size",
    "subgroup_broadcast",
    "subgroup_id",
    "subgroup_mma_compute",
    "subgroup_mma_constant_matrix",
    "subgroup_mma_elementwise",
    "subgroup_mma_extract_thread_local",
    "subgroup_mma_insert_thread_local",
    "subgroup_mma_load_matrix",
    "subgroup_mma_store_matrix",
    "subgroup_reduce",
    "subgroup_size",
    "terminator",
    "thread_id",
    "wait",
    "warp_execute_on_lane_0",
    "yield",
];

pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &'static str = "operand_segment_sizes";
pub const RESULT_SEGMENT_SIZES_ATTRIBUTE: &'static str = "result_segment_sizes";
pub const UPPER_BOUND_ATTRIBUTE: &'static str = "upper_bound";
pub const DIMENSION_ATTRIBUTE: &'static str = "dimension";
pub const KERNEL_ATTRIBUTE: &'static str = "kernel";
pub const MODULE_ATTRIBUTE: &'static str = "module";
pub const FUNCTION_ATTRIBUTE: &'static str = "function";
pub const TARGETS_ATTRIBUTE: &'static str = "targets";
pub const OBJECTS_ATTRIBUTE: &'static str = "objects";
pub const OFFLOADING_HANDLER_ATTRIBUTE: &'static str = "offloadingHandler";
pub const HOST_SHARED_ATTRIBUTE: &'static str = "hostShared";
pub const ADDRESS_SPACES_ATTRIBUTE: &'static str = "address_spaces";
pub const FORMAT_ATTRIBUTE: &'static str = "format";
pub const OP_ATTRIBUTE: &'static str = "op";
pub const OP_TYPE_ATTRIBUTE: &'static str = "opType";
pub const UNIFORM_ATTRIBUTE: &'static str = "uniform";
pub const CLUSTER_SIZE_ATTRIBUTE: &'static str = "cluster_size";
pub const CLUSTER_STRIDE_ATTRIBUTE: &'static str = "cluster_stride";
pub const VALUE_ATTRIBUTE: &'static str = "value";
pub const MODE_ATTRIBUTE: &'static str = "mode";
pub const OFFSET_ATTRIBUTE: &'static str = "offset";
pub const WIDTH_ATTRIBUTE: &'static str = "width";
pub const BROADCAST_TYPE_ATTRIBUTE: &'static str = "broadcast_type";
pub const LEAD_DIMENSION_ATTRIBUTE: &'static str = "leadDimension";
pub const TRANSPOSE_ATTRIBUTE: &'static str = "transpose";
pub const A_TRANSPOSE_ATTRIBUTE: &'static str = "a_transpose";
pub const B_TRANSPOSE_ATTRIBUTE: &'static str = "b_transpose";
pub const WARP_SIZE_ATTRIBUTE: &'static str = "warp_size";
pub const MODE_A_ATTRIBUTE: &'static str = "modeA";
pub const MODE_B_ATTRIBUTE: &'static str = "modeB";
pub const COMPUTE_TYPE_ATTRIBUTE: &'static str = "computeType";
pub const KIND_ATTRIBUTE: &'static str = "kind";
pub const PRUNE_FLAG_ATTRIBUTE: &'static str = "pruneFlag";
pub const KNOWN_BLOCK_SIZE_ATTRIBUTE: &'static str = "known_block_size";
pub const KNOWN_GRID_SIZE_ATTRIBUTE: &'static str = "known_grid_size";
pub const KNOWN_CLUSTER_SIZE_ATTRIBUTE: &'static str = "known_cluster_size";
pub const WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE: &'static str = "workgroup_attrib_attrs";
pub const PRIVATE_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE: &'static str = "private_attrib_attrs";

fn build_gpu_operation<'c, 't: 'c, O: DetachedOp<'c, 'c, 't>>(
    builder: OperationBuilder<'c, 't>,
    rust_path: &'static str,
) -> O {
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .unwrap_or_else(|| panic!("invalid arguments to `gpu::{rust_path}`"))
}

fn add_async_token_result<'c, 't: 'c>(builder: OperationBuilder<'c, 't>, is_async: bool) -> OperationBuilder<'c, 't> {
    if is_async {
        let async_token_type = builder.context().gpu_async_token_type();
        builder.add_result(async_token_type)
    } else {
        builder
    }
}

fn add_optional_attribute<'c, 't: 'c, A: Attribute<'c, 't>>(
    builder: OperationBuilder<'c, 't>,
    name: &'static str,
    attribute: Option<A>,
) -> OperationBuilder<'c, 't> {
    if let Some(attribute) = attribute { builder.add_attribute(name, attribute) } else { builder }
}

fn add_operand_segment_sizes_attribute<'c, 't: 'c>(
    builder: OperationBuilder<'c, 't>,
    sizes: &[i32],
) -> OperationBuilder<'c, 't> {
    let segment_sizes = DenseInteger32ArrayAttributeRef::from_with_context(sizes, builder.context());
    builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment_sizes)
}

fn required_dimension_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    op_name: &str,
) -> Dimension {
    operation
        .attribute(DIMENSION_ATTRIBUTE)
        .and_then(|attribute| attribute.cast::<DimensionAttributeRef>())
        .map(|attribute| attribute.value())
        .unwrap_or_else(|| panic!("invalid '{DIMENSION_ATTRIBUTE}' attribute in `{op_name}`"))
}

fn optional_upper_bound_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
) -> Option<IntegerAttributeRef<'c, 't>> {
    operation
        .attribute(UPPER_BOUND_ATTRIBUTE)
        .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
}

fn required_dense_i32_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    name: &'static str,
    op_name: &str,
) -> Vec<i32> {
    operation
        .attribute(name)
        .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
        .map(Vec::<i32>::from)
        .unwrap_or_else(|| panic!("invalid '{name}' attribute in `{op_name}`"))
}

fn leading_async_dependencies<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    fixed_operand_count: usize,
) -> Vec<ValueRef<'o, 'c, 't>> {
    let async_dependency_count = operation.operand_count().saturating_sub(fixed_operand_count);
    operation.operands().take(async_dependency_count).collect::<Vec<_>>()
}

fn fixed_operand_start<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    fixed_operand_count: usize,
) -> usize {
    operation.operand_count().saturating_sub(fixed_operand_count)
}

fn build_dimension_query_operation<'c, 't: 'c, L: Location<'c, 't>, O: DetachedOp<'c, 'c, 't>>(
    operation_name: &'static str,
    rust_path: &'static str,
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> O {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new(operation_name, location)
        .add_attribute(DIMENSION_ATTRIBUTE, context.gpu_dimension_attribute(dimension))
        .add_result(context.index_type());
    if let Some(upper_bound) = upper_bound {
        builder = builder.add_attribute(UPPER_BOUND_ATTRIBUTE, upper_bound);
    }
    build_gpu_operation(builder, rust_path)
}

fn build_upper_bound_query_operation<'c, 't: 'c, L: Location<'c, 't>, O: DetachedOp<'c, 'c, 't>>(
    operation_name: &'static str,
    rust_path: &'static str,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> O {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new(operation_name, location).add_result(context.index_type());
    if let Some(upper_bound) = upper_bound {
        builder = builder.add_attribute(UPPER_BOUND_ATTRIBUTE, upper_bound);
    }
    build_gpu_operation(builder, rust_path)
}

/// Explicit launch dimensions for GPU launch operations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaunchDimensions<'o, 'c: 'o, 't: 'c> {
    /// Grid dimensions (`x`, `y`, `z`).
    pub grid_size: [ValueRef<'o, 'c, 't>; 3],

    /// Block dimensions (`x`, `y`, `z`).
    pub block_size: [ValueRef<'o, 'c, 't>; 3],

    /// Optional cluster dimensions (`x`, `y`, `z`).
    pub cluster_size: Option<[ValueRef<'o, 'c, 't>; 3]>,
}

/// `gpu.all_reduce` operation.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value being reduced.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the built-in reduction operation, if one is specified.
    fn operation(&self) -> Option<AllReduceOperationAttributeRef<'c, 't>> {
        self.attribute(OP_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AllReduceOperationAttributeRef>())
    }

    /// Returns `true` if the reduction is marked as uniform.
    fn uniform(&self) -> bool {
        self.has_attribute(UNIFORM_ATTRIBUTE)
    }
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, OneResult);

/// Constructs a detached `gpu.all_reduce` operation.
///
/// # Parameters
///
///   - `value`: Value to reduce.
///   - `operation`: Optional built-in reduction operation.
///   - `uniform`: Whether the reduction is marked as uniform.
///   - `reduction_region`: Optional custom reduction region.
///   - `location`: Source location for the created operation.
pub fn all_reduce<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    operation: Option<AllReduceOperationAttributeRef<'c, 't>>,
    uniform: bool,
    reduction_region: Option<DetachedRegion<'c, 't>>,
    location: L,
) -> DetachedAllReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.all_reduce", location).add_operand(value);
    builder = add_optional_attribute(builder, OP_ATTRIBUTE, operation);
    if uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(reduction_region) = reduction_region {
        builder = builder.add_region(reduction_region);
    }
    builder = builder.enable_result_type_inference();
    build_gpu_operation(builder, "all_reduce")
}

/// `gpu.alloc` operation.
pub trait AllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.alloc");
        self.operands().take(sizes[0] as usize).collect::<Vec<_>>()
    }

    /// Returns dynamic size operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.alloc");
        self.operands().skip(sizes[0] as usize).take(sizes[1] as usize).collect::<Vec<_>>()
    }

    /// Returns symbol operands.
    fn symbol_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.alloc");
        self.operands().skip((sizes[0] + sizes[1]) as usize).take(sizes[2] as usize).collect::<Vec<_>>()
    }

    /// Returns the allocated memref result.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }

    /// Returns `true` if host-shared memory was requested.
    fn host_shared(&self) -> bool {
        self.has_attribute(HOST_SHARED_ATTRIBUTE)
    }
}

mlir_op!(Alloc);

/// Constructs a detached `gpu.alloc` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete before allocation.
///   - `dynamic_sizes`: Dynamic dimension sizes for the memref shape.
///   - `symbol_operands`: Symbol operands for the memref layout map.
///   - `memref_type`: Type of the allocated memref result.
///   - `is_async`: Whether to return an async token.
///   - `host_shared`: Whether allocation should be host-shared.
///   - `location`: Source location for the created operation.
pub fn alloc<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dynamic_sizes: &[ValueRef<'o, 'c, 't>],
    symbol_operands: &[ValueRef<'o, 'c, 't>],
    memref_type: T,
    is_async: bool,
    host_shared: bool,
    location: L,
) -> DetachedAllocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.alloc", location)
        .add_operands(async_dependencies)
        .add_operands(dynamic_sizes)
        .add_operands(symbol_operands)
        .add_result(memref_type);
    builder = add_async_token_result(builder, is_async);
    if host_shared {
        builder = builder.add_attribute(HOST_SHARED_ATTRIBUTE, context.unit_attribute());
    }
    builder = add_operand_segment_sizes_attribute(
        builder,
        &[async_dependencies.len() as i32, dynamic_sizes.len() as i32, symbol_operands.len() as i32],
    );
    build_gpu_operation(builder, "alloc")
}

/// `gpu.barrier` operation.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns optional address spaces to fence.
    fn address_spaces(&self) -> Option<Vec<AddressSpaceAttributeRef<'c, 't>>> {
        self.attribute(ADDRESS_SPACES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .map(|attribute| {
                attribute
                    .elements()
                    .filter_map(|element| element.cast::<AddressSpaceAttributeRef>())
                    .collect::<Vec<_>>()
            })
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroOperands);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Constructs a detached `gpu.barrier` operation.
///
/// # Parameters
///
///   - `address_spaces`: Optional list of address spaces to fence.
///   - `location`: Source location for the created operation.
pub fn barrier<'c, 't: 'c, L: Location<'c, 't>>(
    address_spaces: Option<ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedBarrierOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = add_optional_attribute(
        OperationBuilder::new("gpu.barrier", location),
        ADDRESS_SPACES_ATTRIBUTE,
        address_spaces,
    );
    build_gpu_operation(builder, "barrier")
}

/// `gpu.binary` operation.
pub trait BinaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + Symbol<'o, 'c, 't> {
    /// Returns the offloading handler attribute, if present.
    fn offloading_handler(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(OFFLOADING_HANDLER_ATTRIBUTE)
    }

    /// Returns the array of embedded objects.
    fn objects(&self) -> ArrayAttributeRef<'c, 't> {
        self.attribute(OBJECTS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{OBJECTS_ATTRIBUTE}' attribute in `gpu.binary`"))
    }
}

mlir_op!(Binary);
mlir_op_trait!(Binary, Symbol);
mlir_op_trait!(Binary, ZeroRegions);
mlir_op_trait!(Binary, ZeroSuccessors);

/// Constructs a detached `gpu.binary` operation.
///
/// # Parameters
///
///   - `name`: Symbol name of the binary object.
///   - `objects`: Array of `gpu.object` attributes.
///   - `offloading_handler`: Optional offloading handler attribute.
///   - `visibility`: Symbol visibility for the operation.
///   - `location`: Source location for the created operation.
pub fn binary<'c, 't: 'c, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    objects: ArrayAttributeRef<'c, 't>,
    offloading_handler: Option<AttributeRef<'c, 't>>,
    visibility: SymbolVisibility,
    location: L,
) -> DetachedBinaryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.binary", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_attribute(OBJECTS_ATTRIBUTE, objects);
    builder = add_optional_attribute(builder, OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    if visibility != SymbolVisibility::default() {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(visibility));
    }
    build_gpu_operation(builder, "binary")
}

/// `gpu.block_dim` operation.
pub trait BlockDimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.block_dim")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(BlockDim);
mlir_op_trait!(BlockDim, OneResult);
mlir_op_trait!(BlockDim, ZeroOperands);

/// Constructs a detached `gpu.block_dim` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn block_dim<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedBlockDimOperation<'c, 't> {
    build_dimension_query_operation("gpu.block_dim", "block_dim", dimension, upper_bound, location)
}

/// `gpu.block_id` operation.
pub trait BlockIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.block_id")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(BlockId);
mlir_op_trait!(BlockId, OneResult);
mlir_op_trait!(BlockId, ZeroOperands);

/// Constructs a detached `gpu.block_id` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn block_id<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedBlockIdOperation<'c, 't> {
    build_dimension_query_operation("gpu.block_id", "block_id", dimension, upper_bound, location)
}

/// `gpu.cluster_block_id` operation.
pub trait ClusterBlockIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.cluster_block_id")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(ClusterBlockId);
mlir_op_trait!(ClusterBlockId, OneResult);
mlir_op_trait!(ClusterBlockId, ZeroOperands);

/// Constructs a detached `gpu.cluster_block_id` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn cluster_block_id<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedClusterBlockIdOperation<'c, 't> {
    build_dimension_query_operation("gpu.cluster_block_id", "cluster_block_id", dimension, upper_bound, location)
}

/// `gpu.cluster_dim_blocks` operation.
pub trait ClusterDimBlocksOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.cluster_dim_blocks")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(ClusterDimBlocks);
mlir_op_trait!(ClusterDimBlocks, OneResult);
mlir_op_trait!(ClusterDimBlocks, ZeroOperands);

/// Constructs a detached `gpu.cluster_dim_blocks` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn cluster_dim_blocks<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedClusterDimBlocksOperation<'c, 't> {
    build_dimension_query_operation("gpu.cluster_dim_blocks", "cluster_dim_blocks", dimension, upper_bound, location)
}

/// `gpu.cluster_dim` operation.
pub trait ClusterDimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.cluster_dim")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(ClusterDim);
mlir_op_trait!(ClusterDim, OneResult);
mlir_op_trait!(ClusterDim, ZeroOperands);

/// Constructs a detached `gpu.cluster_dim` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn cluster_dim<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedClusterDimOperation<'c, 't> {
    build_dimension_query_operation("gpu.cluster_dim", "cluster_dim", dimension, upper_bound, location)
}

/// `gpu.cluster_id` operation.
pub trait ClusterIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.cluster_id")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(ClusterId);
mlir_op_trait!(ClusterId, OneResult);
mlir_op_trait!(ClusterId, ZeroOperands);

/// Constructs a detached `gpu.cluster_id` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn cluster_id<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedClusterIdOperation<'c, 't> {
    build_dimension_query_operation("gpu.cluster_id", "cluster_id", dimension, upper_bound, location)
}

/// `gpu.create_2to4_spmat` operation.
pub trait Create2To4SpMatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 3)
    }

    /// Returns the row count operand.
    fn rows(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3)).unwrap()
    }

    /// Returns the column count operand.
    fn cols(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 1).unwrap()
    }

    /// Returns the memref operand.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 2).unwrap()
    }

    /// Returns the prune flag attribute.
    fn prune_flag(&self) -> Prune2To4SpMatFlagAttributeRef<'c, 't> {
        self.attribute(PRUNE_FLAG_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<Prune2To4SpMatFlagAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{PRUNE_FLAG_ATTRIBUTE}' attribute in `gpu.create_2to4_spmat`"))
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(Create2To4SpMat);

/// Constructs a detached `gpu.create_2to4_spmat` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `rows`: Number of rows.
///   - `cols`: Number of columns.
///   - `prune_flag`: 2:4 pruning strategy.
///   - `memref`: Dense memref operand.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_2to4_spmat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    cols: ValueRef<'o, 'c, 't>,
    prune_flag: Prune2To4SpMatFlagAttributeRef<'c, 't>,
    memref: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreate2To4SpMatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_2to4_spmat", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(cols)
        .add_operand(memref)
        .add_attribute(PRUNE_FLAG_ATTRIBUTE, prune_flag)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_2to4_spmat")
}

/// `gpu.create_bsr` operation.
pub trait CreateBsrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 8)
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateBsr);

/// Constructs a detached `gpu.create_bsr` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `brows`: Number of block rows.
///   - `bcols`: Number of block columns.
///   - `bnnz`: Number of non-zero blocks.
///   - `r_block_size`: Block row size.
///   - `c_block_size`: Block column size.
///   - `brow_pos`: BSR row-position buffer.
///   - `bcol_indices`: BSR column-index buffer.
///   - `values`: BSR values buffer.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_bsr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    brows: ValueRef<'o, 'c, 't>,
    bcols: ValueRef<'o, 'c, 't>,
    bnnz: ValueRef<'o, 'c, 't>,
    r_block_size: ValueRef<'o, 'c, 't>,
    c_block_size: ValueRef<'o, 'c, 't>,
    brow_pos: ValueRef<'o, 'c, 't>,
    bcol_indices: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreateBsrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_bsr", location)
        .add_operands(async_dependencies)
        .add_operand(brows)
        .add_operand(bcols)
        .add_operand(bnnz)
        .add_operand(r_block_size)
        .add_operand(c_block_size)
        .add_operand(brow_pos)
        .add_operand(bcol_indices)
        .add_operand(values)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_bsr")
}

/// `gpu.create_coo_aos` operation.
pub trait CreateCooAosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 5)
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateCooAos);

/// Constructs a detached `gpu.create_coo_aos` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `rows`: Number of rows.
///   - `cols`: Number of columns.
///   - `nnz`: Number of non-zero elements.
///   - `indices`: AoS coordinate buffer.
///   - `values`: Values buffer.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_coo_aos<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    cols: ValueRef<'o, 'c, 't>,
    nnz: ValueRef<'o, 'c, 't>,
    indices: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreateCooAosOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_coo_aos", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(cols)
        .add_operand(nnz)
        .add_operand(indices)
        .add_operand(values)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_coo_aos")
}

/// `gpu.create_coo` operation.
pub trait CreateCooOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 6)
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateCoo);

/// Constructs a detached `gpu.create_coo` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `rows`: Number of rows.
///   - `cols`: Number of columns.
///   - `nnz`: Number of non-zero elements.
///   - `row_indices`: Row-index buffer.
///   - `col_indices`: Column-index buffer.
///   - `values`: Values buffer.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_coo<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    cols: ValueRef<'o, 'c, 't>,
    nnz: ValueRef<'o, 'c, 't>,
    row_indices: ValueRef<'o, 'c, 't>,
    col_indices: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreateCooOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_coo", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(cols)
        .add_operand(nnz)
        .add_operand(row_indices)
        .add_operand(col_indices)
        .add_operand(values)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_coo")
}

/// `gpu.create_csc` operation.
pub trait CreateCscOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 6)
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateCsc);

/// Constructs a detached `gpu.create_csc` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `rows`: Number of rows.
///   - `cols`: Number of columns.
///   - `nnz`: Number of non-zero elements.
///   - `col_pos`: CSC column-position buffer.
///   - `row_indices`: CSC row-index buffer.
///   - `values`: CSC values buffer.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_csc<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    cols: ValueRef<'o, 'c, 't>,
    nnz: ValueRef<'o, 'c, 't>,
    col_pos: ValueRef<'o, 'c, 't>,
    row_indices: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreateCscOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_csc", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(cols)
        .add_operand(nnz)
        .add_operand(col_pos)
        .add_operand(row_indices)
        .add_operand(values)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_csc")
}

/// `gpu.create_csr` operation.
pub trait CreateCsrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 6)
    }

    /// Returns the sparse matrix handle result.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateCsr);

/// Constructs a detached `gpu.create_csr` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `rows`: Number of rows.
///   - `cols`: Number of columns.
///   - `nnz`: Number of non-zero elements.
///   - `row_pos`: CSR row-position buffer.
///   - `col_indices`: CSR column-index buffer.
///   - `values`: CSR values buffer.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_csr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    cols: ValueRef<'o, 'c, 't>,
    nnz: ValueRef<'o, 'c, 't>,
    row_pos: ValueRef<'o, 'c, 't>,
    col_indices: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedCreateCsrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_csr", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(cols)
        .add_operand(nnz)
        .add_operand(row_pos)
        .add_operand(col_indices)
        .add_operand(values)
        .add_result(context.gpu_sparse_spmat_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "create_csr")
}

/// `gpu.create_dn_tensor` operation.
pub trait CreateDnTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.create_dn_tensor");
        self.operands().take(sizes[0] as usize).collect::<Vec<_>>()
    }

    /// Returns the values buffer operand.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.create_dn_tensor");
        self.operand(sizes[0] as usize).unwrap()
    }

    /// Returns tensor dimension operands.
    fn dimensions(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.create_dn_tensor");
        self.operands().skip((sizes[0] + sizes[1]) as usize).take(sizes[2] as usize).collect::<Vec<_>>()
    }

    /// Returns the dense tensor handle result.
    fn dn_tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateDnTensor);

/// Constructs a detached `gpu.create_dn_tensor` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `values`: Dense values buffer.
///   - `dimensions`: Dense tensor shape dimensions.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn create_dn_tensor<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    values: ValueRef<'o, 'c, 't>,
    dimensions: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> DetachedCreateDnTensorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.create_dn_tensor", location)
        .add_operands(async_dependencies)
        .add_operand(values)
        .add_operands(dimensions)
        .add_result(context.gpu_sparse_dn_tensor_handle_type());
    builder = add_async_token_result(builder, is_async);
    builder =
        add_operand_segment_sizes_attribute(builder, &[async_dependencies.len() as i32, 1, dimensions.len() as i32]);
    build_gpu_operation(builder, "create_dn_tensor")
}

/// `gpu.dealloc` operation.
pub trait DeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 1)
    }

    /// Returns the deallocated memref operand.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 1)).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Dealloc);

/// Constructs a detached `gpu.dealloc` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `memref`: Memref to deallocate.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn dealloc<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    memref: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedDeallocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder =
        OperationBuilder::new("gpu.dealloc", location).add_operands(async_dependencies).add_operand(memref);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "dealloc")
}

/// `gpu.destroy_dn_tensor` operation.
pub trait DestroyDnTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 1)
    }

    /// Returns the dense tensor descriptor operand.
    fn dn_tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 1)).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(DestroyDnTensor);

/// Constructs a detached `gpu.destroy_dn_tensor` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `dn_tensor`: Dense tensor descriptor to destroy.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn destroy_dn_tensor<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dn_tensor: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedDestroyDnTensorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.destroy_dn_tensor", location)
        .add_operands(async_dependencies)
        .add_operand(dn_tensor);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "destroy_dn_tensor")
}

/// `gpu.destroy_sp_mat` operation.
pub trait DestroySpMatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 1)
    }

    /// Returns the sparse matrix descriptor operand.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 1)).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(DestroySpMat);

/// Constructs a detached `gpu.destroy_sp_mat` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `spmat`: Sparse matrix descriptor to destroy.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn destroy_sp_mat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedDestroySpMatOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.destroy_sp_mat", location)
        .add_operands(async_dependencies)
        .add_operand(spmat);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "destroy_sp_mat")
}

/// `gpu.dynamic_shared_memory` operation.
pub trait DynamicSharedMemoryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(DynamicSharedMemory);
mlir_op_trait!(DynamicSharedMemory, OneResult);
mlir_op_trait!(DynamicSharedMemory, ZeroOperands);

/// Constructs a detached `gpu.dynamic_shared_memory` operation.
///
/// # Parameters
///
///   - `result_type`: Result memref type.
///   - `location`: Source location for the created operation.
pub fn dynamic_shared_memory<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    result_type: T,
    location: L,
) -> DetachedDynamicSharedMemoryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.dynamic_shared_memory", location).add_result(result_type);
    build_gpu_operation(builder, "dynamic_shared_memory")
}

/// `gpu.func` operation.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + Symbol<'o, 'c, 't> + HasCallableArgumentAndResultAttributes<'o, 'c, 't>
{
    /// Returns `true` if this function is marked as a kernel.
    fn kernel(&self) -> bool {
        self.has_attribute(KERNEL_ATTRIBUTE)
    }

    /// Returns optional workgroup attribution attributes.
    fn workgroup_attribution_attributes(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }

    /// Returns optional private attribution attributes.
    fn private_attribution_attributes(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(PRIVATE_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }

    /// Returns optional known block-size annotation.
    fn known_block_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
    }

    /// Returns optional known grid-size annotation.
    fn known_grid_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_GRID_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
    }

    /// Returns optional known cluster-size annotation.
    fn known_cluster_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, Callable);
mlir_op_trait!(Func, Function);
mlir_op_trait!(Func, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Func, Symbol);

/// Constructs a detached `gpu.func` operation.
///
/// # Parameters
///
///   - `name`: Symbol name of the GPU function.
///   - `function_type`: Function signature.
///   - `body`: Function body region.
///   - `visibility`: Symbol visibility for the function.
///   - `kernel`: Whether this function is marked as a kernel.
///   - `argument_attributes`: Optional callable argument attributes (`arg_attrs`).
///   - `result_attributes`: Optional callable result attributes (`res_attrs`).
///   - `workgroup_attribution_attributes`: Optional workgroup attribution attrs.
///   - `private_attribution_attributes`: Optional private attribution attrs.
///   - `known_block_size`: Optional known block-size hint.
///   - `known_grid_size`: Optional known grid-size hint.
///   - `known_cluster_size`: Optional known cluster-size hint.
///   - `location`: Source location for the created operation.
pub fn func<'c, 't: 'c, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    function_type: FunctionTypeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    visibility: SymbolVisibility,
    kernel: bool,
    argument_attributes: Option<ArrayAttributeRef<'c, 't>>,
    result_attributes: Option<ArrayAttributeRef<'c, 't>>,
    workgroup_attribution_attributes: Option<ArrayAttributeRef<'c, 't>>,
    private_attribution_attributes: Option<ArrayAttributeRef<'c, 't>>,
    known_block_size: Option<DenseInteger32ArrayAttributeRef<'c, 't>>,
    known_grid_size: Option<DenseInteger32ArrayAttributeRef<'c, 't>>,
    known_cluster_size: Option<DenseInteger32ArrayAttributeRef<'c, 't>>,
    location: L,
) -> DetachedFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_attribute(FUNCTION_TYPE_ATTRIBUTE, context.type_attribute(function_type));

    if visibility != SymbolVisibility::default() {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(visibility));
    }
    if kernel {
        builder = builder.add_attribute(KERNEL_ATTRIBUTE, context.unit_attribute());
    }

    builder = add_optional_attribute(builder, ARGUMENT_ATTRIBUTES_ATTRIBUTE, argument_attributes);
    builder = add_optional_attribute(builder, RESULT_ATTRIBUTES_ATTRIBUTE, result_attributes);
    builder =
        add_optional_attribute(builder, WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE, workgroup_attribution_attributes);
    builder = add_optional_attribute(builder, PRIVATE_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE, private_attribution_attributes);
    builder = add_optional_attribute(builder, KNOWN_BLOCK_SIZE_ATTRIBUTE, known_block_size);
    builder = add_optional_attribute(builder, KNOWN_GRID_SIZE_ATTRIBUTE, known_grid_size);
    builder = add_optional_attribute(builder, KNOWN_CLUSTER_SIZE_ATTRIBUTE, known_cluster_size);
    builder = builder.add_region(body);
    build_gpu_operation(builder, "func")
}

/// `gpu.module` operation.
pub trait ModuleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + Symbol<'o, 'c, 't> {
    /// Returns optional target metadata.
    fn targets(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(TARGETS_ATTRIBUTE).and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
    }

    /// Returns optional offloading handler attribute.
    fn offloading_handler(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(OFFLOADING_HANDLER_ATTRIBUTE)
    }
}

mlir_op!(Module);
mlir_op_trait!(Module, Symbol);
mlir_op_trait!(Module, SymbolTable);

/// Constructs a detached `gpu.module` operation.
///
/// # Parameters
///
///   - `name`: Symbol name of the GPU module.
///   - `body`: Module body region.
///   - `targets`: Optional target metadata.
///   - `offloading_handler`: Optional offloading handler attribute.
///   - `visibility`: Symbol visibility for the operation.
///   - `location`: Source location for the created operation.
pub fn module<'c, 't: 'c, N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    body: DetachedRegion<'c, 't>,
    targets: Option<ArrayAttributeRef<'c, 't>>,
    offloading_handler: Option<AttributeRef<'c, 't>>,
    visibility: SymbolVisibility,
    location: L,
) -> DetachedModuleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.module", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.into_with_context(context))
        .add_region(body);
    builder = add_optional_attribute(builder, TARGETS_ATTRIBUTE, targets);
    builder = add_optional_attribute(builder, OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    if visibility != SymbolVisibility::default() {
        builder = builder.add_attribute(SYMBOL_VISIBILITY_ATTRIBUTE, context.symbol_visibility_attribute(visibility));
    }
    build_gpu_operation(builder, "module")
}

/// `gpu.global_id` operation.
pub trait GlobalIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.global_id")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(GlobalId);
mlir_op_trait!(GlobalId, OneResult);
mlir_op_trait!(GlobalId, ZeroOperands);

/// Constructs a detached `gpu.global_id` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn global_id<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedGlobalIdOperation<'c, 't> {
    build_dimension_query_operation("gpu.global_id", "global_id", dimension, upper_bound, location)
}

/// `gpu.grid_dim` operation.
pub trait GridDimOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.grid_dim")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(GridDim);
mlir_op_trait!(GridDim, OneResult);
mlir_op_trait!(GridDim, ZeroOperands);

/// Constructs a detached `gpu.grid_dim` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn grid_dim<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedGridDimOperation<'c, 't> {
    build_dimension_query_operation("gpu.grid_dim", "grid_dim", dimension, upper_bound, location)
}

/// `gpu.host_register` operation.
pub trait HostRegisterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being registered.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }
}

mlir_op!(HostRegister);
mlir_op_trait!(HostRegister, ZeroRegions);
mlir_op_trait!(HostRegister, ZeroSuccessors);

/// Constructs a detached `gpu.host_register` operation.
///
/// # Parameters
///
///   - `value`: Host memref to map for device access.
///   - `location`: Source location for the created operation.
pub fn host_register<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    location: L,
) -> DetachedHostRegisterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.host_register", location).add_operand(value);
    build_gpu_operation(builder, "host_register")
}

/// `gpu.host_unregister` operation.
pub trait HostUnregisterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memref being unregistered.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }
}

mlir_op!(HostUnregister);
mlir_op_trait!(HostUnregister, ZeroRegions);
mlir_op_trait!(HostUnregister, ZeroSuccessors);

/// Constructs a detached `gpu.host_unregister` operation.
///
/// # Parameters
///
///   - `value`: Host memref to unmap from device access.
///   - `location`: Source location for the created operation.
pub fn host_unregister<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    location: L,
) -> DetachedHostUnregisterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.host_unregister", location).add_operand(value);
    build_gpu_operation(builder, "host_unregister")
}

/// `gpu.lane_id` operation.
pub trait LaneIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(LaneId);
mlir_op_trait!(LaneId, OneResult);
mlir_op_trait!(LaneId, ZeroOperands);

/// Constructs a detached `gpu.lane_id` operation.
///
/// # Parameters
///
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn lane_id<'c, 't: 'c, L: Location<'c, 't>>(
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedLaneIdOperation<'c, 't> {
    build_upper_bound_query_operation("gpu.lane_id", "lane_id", upper_bound, location)
}

/// `gpu.launch_func` operation.
pub trait LaunchFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the kernel symbol reference.
    fn kernel(&self) -> SymbolRefAttributeRef<'c, 't> {
        self.attribute(KERNEL_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<SymbolRefAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{KERNEL_ATTRIBUTE}' attribute in `gpu.launch_func`"))
    }

    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.launch_func");
        self.operands().take(sizes[0] as usize).collect::<Vec<_>>()
    }

    /// Returns kernel operands passed to the launch.
    fn kernel_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.launch_func");
        let prefix = (sizes[0]
            + sizes[1]
            + sizes[2]
            + sizes[3]
            + sizes[4]
            + sizes[5]
            + sizes[6]
            + sizes[7]
            + sizes[8]
            + sizes[9]
            + sizes[10]) as usize;
        self.operands().skip(prefix).take(sizes[11] as usize).collect::<Vec<_>>()
    }

    /// Returns the optional async object operand.
    fn async_object(&self) -> Option<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.launch_func");
        if sizes[12] == 0 {
            return None;
        }
        let prefix = (sizes[0]
            + sizes[1]
            + sizes[2]
            + sizes[3]
            + sizes[4]
            + sizes[5]
            + sizes[6]
            + sizes[7]
            + sizes[8]
            + sizes[9]
            + sizes[10]
            + sizes[11]) as usize;
        self.operand(prefix)
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(LaunchFunc);

/// Constructs a detached `gpu.launch_func` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `kernel`: Symbol reference to the launched kernel.
///   - `dimensions`: Grid, block, and optional cluster dimensions.
///   - `dynamic_shared_memory_size`: Optional dynamic shared-memory size.
///   - `kernel_operands`: Kernel arguments.
///   - `async_object`: Optional async object operand.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn launch_func<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    kernel: SymbolRefAttributeRef<'c, 't>,
    dimensions: LaunchDimensions<'o, 'c, 't>,
    dynamic_shared_memory_size: Option<ValueRef<'o, 'c, 't>>,
    kernel_operands: &[ValueRef<'o, 'c, 't>],
    async_object: Option<ValueRef<'o, 'c, 't>>,
    is_async: bool,
    location: L,
) -> DetachedLaunchFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.launch_func", location)
        .add_operands(async_dependencies)
        .add_operand(dimensions.grid_size[0])
        .add_operand(dimensions.grid_size[1])
        .add_operand(dimensions.grid_size[2])
        .add_operand(dimensions.block_size[0])
        .add_operand(dimensions.block_size[1])
        .add_operand(dimensions.block_size[2])
        .add_attribute(KERNEL_ATTRIBUTE, kernel);

    let cluster_segment_sizes = if let Some(cluster_size) = dimensions.cluster_size {
        builder = builder.add_operand(cluster_size[0]).add_operand(cluster_size[1]).add_operand(cluster_size[2]);
        [1, 1, 1]
    } else {
        [0, 0, 0]
    };

    let dynamic_shared_memory_segment_size = if let Some(dynamic_shared_memory_size) = dynamic_shared_memory_size {
        builder = builder.add_operand(dynamic_shared_memory_size);
        1
    } else {
        0
    };

    builder = builder.add_operands(kernel_operands);

    let async_object_segment_size = if let Some(async_object) = async_object {
        builder = builder.add_operand(async_object);
        1
    } else {
        0
    };

    builder = add_async_token_result(builder, is_async);

    builder = add_operand_segment_sizes_attribute(
        builder,
        &[
            async_dependencies.len() as i32,
            1,
            1,
            1,
            1,
            1,
            1,
            cluster_segment_sizes[0],
            cluster_segment_sizes[1],
            cluster_segment_sizes[2],
            dynamic_shared_memory_segment_size,
            kernel_operands.len() as i32,
            async_object_segment_size,
        ],
    );
    build_gpu_operation(builder, "launch_func")
}

/// `gpu.launch` operation.
pub trait LaunchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        let sizes = required_dense_i32_attribute(self, OPERAND_SEGMENT_SIZES_ATTRIBUTE, "gpu.launch");
        self.operands().take(sizes[0] as usize).collect::<Vec<_>>()
    }

    /// Returns optional enclosing module symbol.
    fn module(&self) -> Option<FlatSymbolRefAttributeRef<'c, 't>> {
        self.attribute(MODULE_ATTRIBUTE).and_then(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>())
    }

    /// Returns optional enclosing function symbol.
    fn function(&self) -> Option<FlatSymbolRefAttributeRef<'c, 't>> {
        self.attribute(FUNCTION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<FlatSymbolRefAttributeRef>())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Launch);

/// Constructs a detached `gpu.launch` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `dimensions`: Grid, block, and optional cluster dimensions.
///   - `dynamic_shared_memory_size`: Optional dynamic shared-memory size.
///   - `module`: Optional module symbol.
///   - `function`: Optional function symbol.
///   - `body`: Launch body region.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn launch<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dimensions: LaunchDimensions<'o, 'c, 't>,
    dynamic_shared_memory_size: Option<ValueRef<'o, 'c, 't>>,
    module: Option<FlatSymbolRefAttributeRef<'c, 't>>,
    function: Option<FlatSymbolRefAttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedLaunchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.launch", location)
        .add_operands(async_dependencies)
        .add_operand(dimensions.grid_size[0])
        .add_operand(dimensions.grid_size[1])
        .add_operand(dimensions.grid_size[2])
        .add_operand(dimensions.block_size[0])
        .add_operand(dimensions.block_size[1])
        .add_operand(dimensions.block_size[2]);

    let cluster_segment_sizes = if let Some(cluster_size) = dimensions.cluster_size {
        builder = builder.add_operand(cluster_size[0]).add_operand(cluster_size[1]).add_operand(cluster_size[2]);
        [1, 1, 1]
    } else {
        [0, 0, 0]
    };

    let dynamic_shared_memory_segment_size = if let Some(dynamic_shared_memory_size) = dynamic_shared_memory_size {
        builder = builder.add_operand(dynamic_shared_memory_size);
        1
    } else {
        0
    };

    builder = add_optional_attribute(builder, MODULE_ATTRIBUTE, module);
    builder = add_optional_attribute(builder, FUNCTION_ATTRIBUTE, function);
    builder = add_async_token_result(builder, is_async);
    builder = add_operand_segment_sizes_attribute(
        builder,
        &[
            async_dependencies.len() as i32,
            1,
            1,
            1,
            1,
            1,
            1,
            cluster_segment_sizes[0],
            cluster_segment_sizes[1],
            cluster_segment_sizes[2],
            dynamic_shared_memory_segment_size,
        ],
    );
    builder = builder.add_region(body);
    build_gpu_operation(builder, "launch")
}

/// `gpu.memcpy` operation.
pub trait MemcpyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 2)
    }

    /// Returns destination operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 2)).unwrap()
    }

    /// Returns source operand.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 2) + 1).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Memcpy);

/// Constructs a detached `gpu.memcpy` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `destination`: Destination memref.
///   - `source`: Source memref.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn memcpy<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedMemcpyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.memcpy", location)
        .add_operands(async_dependencies)
        .add_operand(destination)
        .add_operand(source);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "memcpy")
}

/// `gpu.memset` operation.
pub trait MemsetOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 2)
    }

    /// Returns destination operand.
    fn destination(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 2)).unwrap()
    }

    /// Returns value operand.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 2) + 1).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Memset);

/// Constructs a detached `gpu.memset` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies that must complete first.
///   - `destination`: Destination memref.
///   - `value`: Scalar value to store.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn memset<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    value: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedMemsetOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.memset", location)
        .add_operands(async_dependencies)
        .add_operand(destination)
        .add_operand(value);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "memset")
}

/// `gpu.num_subgroups` operation.
pub trait NumSubgroupsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(NumSubgroups);
mlir_op_trait!(NumSubgroups, OneResult);
mlir_op_trait!(NumSubgroups, ZeroOperands);

/// Constructs a detached `gpu.num_subgroups` operation.
///
/// # Parameters
///
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn num_subgroups<'c, 't: 'c, L: Location<'c, 't>>(
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedNumSubgroupsOperation<'c, 't> {
    build_upper_bound_query_operation("gpu.num_subgroups", "num_subgroups", upper_bound, location)
}

/// `gpu.printf` operation.
pub trait PrintfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the format string.
    fn format(&self) -> StringRef<'c> {
        self.attribute(FORMAT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>())
            .map(|attribute| attribute.string())
            .unwrap_or_else(|| panic!("invalid '{FORMAT_ATTRIBUTE}' attribute in `gpu.printf`"))
    }

    /// Returns print arguments.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().collect::<Vec<_>>()
    }
}

mlir_op!(Printf);
mlir_op_trait!(Printf, ZeroRegions);
mlir_op_trait!(Printf, ZeroSuccessors);

/// Constructs a detached `gpu.printf` operation.
///
/// # Parameters
///
///   - `format`: Format string literal.
///   - `arguments`: Scalar arguments printed by the device.
///   - `location`: Source location for the created operation.
pub fn printf<'o, 'c: 'o, 't: 'c, F: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    format: F,
    arguments: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> DetachedPrintfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.printf", location)
        .add_attribute(FORMAT_ATTRIBUTE, format.into_with_context(context))
        .add_operands(arguments);
    build_gpu_operation(builder, "printf")
}

/// `gpu.return` operation.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + ReturnLike<'o, 'c, 't> {
    /// Returns returned values.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().collect::<Vec<_>>()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a detached `gpu.return` operation.
///
/// # Parameters
///
///   - `values`: Values returned from `gpu.func`.
///   - `location`: Source location for the created operation.
pub fn r#return<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.return", location).add_operands(values);
    build_gpu_operation(builder, "return")
}

/// `gpu.rotate` operation.
pub trait RotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the rotate offset.
    fn offset(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(OFFSET_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{OFFSET_ATTRIBUTE}' attribute in `gpu.rotate`"))
    }

    /// Returns the rotate width.
    fn width(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(WIDTH_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{WIDTH_ATTRIBUTE}' attribute in `gpu.rotate`"))
    }

    /// Returns the rotated value result.
    fn rotate_result(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the validity flag result.
    fn valid(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }
}

mlir_op!(Rotate);

/// Constructs a detached `gpu.rotate` operation.
///
/// # Parameters
///
///   - `value`: Source value.
///   - `offset`: Rotation offset (i32).
///   - `width`: Rotation width (i32).
///   - `location`: Source location for the created operation.
pub fn rotate<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    offset: i32,
    width: i32,
    location: L,
) -> DetachedRotateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let i32_type = context.signless_integer_type(32);
    let builder = OperationBuilder::new("gpu.rotate", location)
        .add_operand(value)
        .add_attribute(OFFSET_ATTRIBUTE, context.integer_attribute(i32_type, offset as i64))
        .add_attribute(WIDTH_ATTRIBUTE, context.integer_attribute(i32_type, width as i64))
        .add_result(value.r#type())
        .add_result(context.signless_integer_type(1));
    build_gpu_operation(builder, "rotate")
}

/// `gpu.shuffle` operation.
pub trait ShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns the shuffle offset operand.
    fn offset(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns the shuffle width operand.
    fn width(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }

    /// Returns the shuffle mode.
    fn mode(&self) -> ShuffleModeAttributeRef<'c, 't> {
        self.attribute(MODE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ShuffleModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_ATTRIBUTE}' attribute in `gpu.shuffle`"))
    }

    /// Returns the shuffled value result.
    fn shuffle_result(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the validity flag result.
    fn valid(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }
}

mlir_op!(Shuffle);

/// Constructs a detached `gpu.shuffle` operation.
///
/// # Parameters
///
///   - `value`: Source value.
///   - `offset`: Shuffle offset (i32 value).
///   - `width`: Shuffle width (i32 value).
///   - `mode`: Shuffle mode attribute.
///   - `location`: Source location for the created operation.
pub fn shuffle<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    offset: ValueRef<'o, 'c, 't>,
    width: ValueRef<'o, 'c, 't>,
    mode: ShuffleModeAttributeRef<'c, 't>,
    location: L,
) -> DetachedShuffleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.shuffle", location)
        .add_operand(value)
        .add_operand(offset)
        .add_operand(width)
        .add_attribute(MODE_ATTRIBUTE, mode)
        .add_result(value.r#type())
        .add_result(context.signless_integer_type(1));
    build_gpu_operation(builder, "shuffle")
}

/// `gpu.set_default_device` operation.
pub trait SetDefaultDeviceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the device index operand.
    fn device_index(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }
}

mlir_op!(SetDefaultDevice);
mlir_op_trait!(SetDefaultDevice, ZeroRegions);
mlir_op_trait!(SetDefaultDevice, ZeroSuccessors);

/// Constructs a detached `gpu.set_default_device` operation.
///
/// # Parameters
///
///   - `device_index`: Device index (i32 value).
///   - `location`: Source location for the created operation.
pub fn set_default_device<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    device_index: V,
    location: L,
) -> DetachedSetDefaultDeviceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.set_default_device", location).add_operand(device_index);
    build_gpu_operation(builder, "set_default_device")
}

/// `gpu.sddmm_buffer_size` operation.
pub trait SddmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 4)
    }

    /// Returns dense matrix A descriptor.
    fn dnmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4)).unwrap()
    }

    /// Returns dense matrix B descriptor.
    fn dnmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 1).unwrap()
    }

    /// Returns sparse matrix C descriptor.
    fn spmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 2).unwrap()
    }

    /// Returns dense matrix D descriptor.
    fn dnmat_d(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 3).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn a_transpose(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(A_TRANSPOSE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{A_TRANSPOSE_ATTRIBUTE}' attribute in `gpu.sddmm_buffer_size`"))
    }

    /// Returns transpose mode for matrix B.
    fn b_transpose(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(B_TRANSPOSE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{B_TRANSPOSE_ATTRIBUTE}' attribute in `gpu.sddmm_buffer_size`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.sddmm_buffer_size`"))
    }

    /// Returns the temporary buffer size.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SddmmBufferSize);

/// Constructs a detached `gpu.sddmm_buffer_size` operation.
pub fn sddmm_buffer_size<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dnmat_a: ValueRef<'o, 'c, 't>,
    dnmat_b: ValueRef<'o, 'c, 't>,
    spmat_c: ValueRef<'o, 'c, 't>,
    dnmat_d: ValueRef<'o, 'c, 't>,
    a_transpose: TransposeModeAttributeRef<'c, 't>,
    b_transpose: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSddmmBufferSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.sddmm_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(dnmat_a)
        .add_operand(dnmat_b)
        .add_operand(spmat_c)
        .add_operand(dnmat_d)
        .add_attribute(A_TRANSPOSE_ATTRIBUTE, a_transpose)
        .add_attribute(B_TRANSPOSE_ATTRIBUTE, b_transpose)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type)
        .add_result(context.index_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "sddmm_buffer_size")
}

/// `gpu.sddmm` operation.
pub trait SddmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 5)
    }

    /// Returns dense matrix A descriptor.
    fn dnmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5)).unwrap()
    }

    /// Returns dense matrix B descriptor.
    fn dnmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 1).unwrap()
    }

    /// Returns sparse matrix C descriptor.
    fn spmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 2).unwrap()
    }

    /// Returns dense matrix D descriptor.
    fn dnmat_d(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 3).unwrap()
    }

    /// Returns temporary buffer descriptor.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 4).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn a_transpose(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(A_TRANSPOSE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{A_TRANSPOSE_ATTRIBUTE}' attribute in `gpu.sddmm`"))
    }

    /// Returns transpose mode for matrix B.
    fn b_transpose(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(B_TRANSPOSE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{B_TRANSPOSE_ATTRIBUTE}' attribute in `gpu.sddmm`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.sddmm`"))
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Sddmm);

/// Constructs a detached `gpu.sddmm` operation.
pub fn sddmm<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dnmat_a: ValueRef<'o, 'c, 't>,
    dnmat_b: ValueRef<'o, 'c, 't>,
    spmat_c: ValueRef<'o, 'c, 't>,
    dnmat_d: ValueRef<'o, 'c, 't>,
    buffer: ValueRef<'o, 'c, 't>,
    a_transpose: TransposeModeAttributeRef<'c, 't>,
    b_transpose: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSddmmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.sddmm", location)
        .add_operands(async_dependencies)
        .add_operand(dnmat_a)
        .add_operand(dnmat_b)
        .add_operand(spmat_c)
        .add_operand(dnmat_d)
        .add_operand(buffer)
        .add_attribute(A_TRANSPOSE_ATTRIBUTE, a_transpose)
        .add_attribute(B_TRANSPOSE_ATTRIBUTE, b_transpose)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "sddmm")
}

/// `gpu.set_csr_pointers` operation.
pub trait SetCsrPointersOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 4)
    }

    /// Returns sparse matrix descriptor.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4)).unwrap()
    }

    /// Returns CSR row-position pointer.
    fn row_pos(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 1).unwrap()
    }

    /// Returns CSR column-index pointer.
    fn col_idx(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 2).unwrap()
    }

    /// Returns CSR values pointer.
    fn values(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 3).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(SetCsrPointers);

/// Constructs a detached `gpu.set_csr_pointers` operation.
pub fn set_csr_pointers<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat: ValueRef<'o, 'c, 't>,
    row_pos: ValueRef<'o, 'c, 't>,
    col_idx: ValueRef<'o, 'c, 't>,
    values: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSetCsrPointersOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.set_csr_pointers", location)
        .add_operands(async_dependencies)
        .add_operand(spmat)
        .add_operand(row_pos)
        .add_operand(col_idx)
        .add_operand(values);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "set_csr_pointers")
}

/// `gpu.spgemm_copy` operation.
pub trait SpgemmCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 4)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4)).unwrap()
    }

    /// Returns sparse matrix B descriptor.
    fn spmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 1).unwrap()
    }

    /// Returns sparse matrix C descriptor.
    fn spmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 2).unwrap()
    }

    /// Returns sparse GEMM descriptor.
    fn spgemm_descr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 3).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spgemm_copy`"))
    }

    /// Returns transpose mode for matrix B.
    fn mode_b(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_B_ATTRIBUTE}' attribute in `gpu.spgemm_copy`"))
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(SpgemmCopy);

/// Constructs a detached `gpu.spgemm_copy` operation.
pub fn spgemm_copy<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    spmat_b: ValueRef<'o, 'c, 't>,
    spmat_c: ValueRef<'o, 'c, 't>,
    spgemm_descr: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    mode_b: TransposeModeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpgemmCopyOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spgemm_copy", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(spmat_b)
        .add_operand(spmat_c)
        .add_operand(spgemm_descr)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(MODE_B_ATTRIBUTE, mode_b);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spgemm_copy")
}

/// `gpu.spgemm_create_descr` operation.
pub trait SpgemmCreateDescrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().collect::<Vec<_>>()
    }

    /// Returns the sparse GEMM descriptor result.
    fn spgemm_descr(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpgemmCreateDescr);

/// Constructs a detached `gpu.spgemm_create_descr` operation.
pub fn spgemm_create_descr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> DetachedSpgemmCreateDescrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spgemm_create_descr", location)
        .add_operands(async_dependencies)
        .add_result(context.gpu_sparse_spgemm_op_handle_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spgemm_create_descr")
}

/// `gpu.spgemm_destroy_descr` operation.
pub trait SpgemmDestroyDescrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 1)
    }

    /// Returns sparse GEMM descriptor operand.
    fn spgemm_descr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 1)).unwrap()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(SpgemmDestroyDescr);

/// Constructs a detached `gpu.spgemm_destroy_descr` operation.
pub fn spgemm_destroy_descr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spgemm_descr: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpgemmDestroyDescrOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spgemm_destroy_descr", location)
        .add_operands(async_dependencies)
        .add_operand(spgemm_descr);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spgemm_destroy_descr")
}

/// `gpu.spgemm_work_estimation_or_compute` operation.
pub trait SpgemmWorkEstimationOrComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 5)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5)).unwrap()
    }

    /// Returns sparse matrix B descriptor.
    fn spmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 1).unwrap()
    }

    /// Returns sparse matrix C descriptor.
    fn spmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 2).unwrap()
    }

    /// Returns sparse GEMM descriptor.
    fn spgemm_descr(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 3).unwrap()
    }

    /// Returns temporary buffer descriptor.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 5) + 4).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spgemm_work_estimation_or_compute`")
            })
    }

    /// Returns transpose mode for matrix B.
    fn mode_b(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{MODE_B_ATTRIBUTE}' attribute in `gpu.spgemm_work_estimation_or_compute`")
            })
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.spgemm_work_estimation_or_compute`")
            })
    }

    /// Returns the operation kind.
    fn kind(&self) -> SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't> {
        self.attribute(KIND_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<SpGemmWorkEstimationOrComputeKindAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{KIND_ATTRIBUTE}' attribute in `gpu.spgemm_work_estimation_or_compute`")
            })
    }

    /// Returns updated buffer size result.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpgemmWorkEstimationOrCompute);

/// Constructs a detached `gpu.spgemm_work_estimation_or_compute` operation.
pub fn spgemm_work_estimation_or_compute<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    spmat_b: ValueRef<'o, 'c, 't>,
    spmat_c: ValueRef<'o, 'c, 't>,
    spgemm_descr: ValueRef<'o, 'c, 't>,
    buffer: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    mode_b: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    kind: SpGemmWorkEstimationOrComputeKindAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpgemmWorkEstimationOrComputeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spgemm_work_estimation_or_compute", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(spmat_b)
        .add_operand(spmat_c)
        .add_operand(spgemm_descr)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(MODE_B_ATTRIBUTE, mode_b)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type)
        .add_attribute(KIND_ATTRIBUTE, kind)
        .add_result(context.index_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spgemm_work_estimation_or_compute")
}

/// `gpu.spmm_buffer_size` operation.
pub trait SpmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 3)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3)).unwrap()
    }

    /// Returns dense matrix B descriptor.
    fn dnmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 1).unwrap()
    }

    /// Returns dense matrix C descriptor.
    fn dnmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 2).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spmm_buffer_size`"))
    }

    /// Returns transpose mode for matrix B.
    fn mode_b(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_B_ATTRIBUTE}' attribute in `gpu.spmm_buffer_size`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.spmm_buffer_size`"))
    }

    /// Returns temporary buffer size result.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpmmBufferSize);

/// Constructs a detached `gpu.spmm_buffer_size` operation.
pub fn spmm_buffer_size<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    dnmat_b: ValueRef<'o, 'c, 't>,
    dnmat_c: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    mode_b: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpmmBufferSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spmm_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(dnmat_b)
        .add_operand(dnmat_c)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(MODE_B_ATTRIBUTE, mode_b)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type)
        .add_result(context.index_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spmm_buffer_size")
}

/// `gpu.spmm` operation.
pub trait SpmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 4)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4)).unwrap()
    }

    /// Returns dense matrix B descriptor.
    fn dnmat_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 1).unwrap()
    }

    /// Returns dense matrix C descriptor.
    fn dnmat_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 2).unwrap()
    }

    /// Returns temporary buffer descriptor.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 3).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spmm`"))
    }

    /// Returns transpose mode for matrix B.
    fn mode_b(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_B_ATTRIBUTE}' attribute in `gpu.spmm`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.spmm`"))
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Spmm);

/// Constructs a detached `gpu.spmm` operation.
pub fn spmm<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    dnmat_b: ValueRef<'o, 'c, 't>,
    dnmat_c: ValueRef<'o, 'c, 't>,
    buffer: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    mode_b: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpmmOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spmm", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(dnmat_b)
        .add_operand(dnmat_c)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(MODE_B_ATTRIBUTE, mode_b)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spmm")
}

/// `gpu.spmv_buffer_size` operation.
pub trait SpmvBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 3)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3)).unwrap()
    }

    /// Returns dense vector X descriptor.
    fn dnvec_x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 1).unwrap()
    }

    /// Returns dense vector Y descriptor.
    fn dnvec_y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 3) + 2).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spmv_buffer_size`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.spmv_buffer_size`"))
    }

    /// Returns temporary buffer size result.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpmvBufferSize);

/// Constructs a detached `gpu.spmv_buffer_size` operation.
pub fn spmv_buffer_size<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    dnvec_x: ValueRef<'o, 'c, 't>,
    dnvec_y: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpmvBufferSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spmv_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(dnvec_x)
        .add_operand(dnvec_y)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type)
        .add_result(context.index_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spmv_buffer_size")
}

/// `gpu.spmv` operation.
pub trait SpmvOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 4)
    }

    /// Returns sparse matrix A descriptor.
    fn spmat_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4)).unwrap()
    }

    /// Returns dense vector X descriptor.
    fn dnvec_x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 1).unwrap()
    }

    /// Returns dense vector Y descriptor.
    fn dnvec_y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 2).unwrap()
    }

    /// Returns temporary buffer descriptor.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 4) + 3).unwrap()
    }

    /// Returns transpose mode for matrix A.
    fn mode_a(&self) -> TransposeModeAttributeRef<'c, 't> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{MODE_A_ATTRIBUTE}' attribute in `gpu.spmv`"))
    }

    /// Returns compute element type.
    fn compute_type(&self) -> TypeAttributeRef<'c, 't> {
        self.attribute(COMPUTE_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{COMPUTE_TYPE_ATTRIBUTE}' attribute in `gpu.spmv`"))
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Spmv);

/// Constructs a detached `gpu.spmv` operation.
pub fn spmv<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat_a: ValueRef<'o, 'c, 't>,
    dnvec_x: ValueRef<'o, 'c, 't>,
    dnvec_y: ValueRef<'o, 'c, 't>,
    buffer: ValueRef<'o, 'c, 't>,
    mode_a: TransposeModeAttributeRef<'c, 't>,
    compute_type: TypeAttributeRef<'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpmvOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spmv", location)
        .add_operands(async_dependencies)
        .add_operand(spmat_a)
        .add_operand(dnvec_x)
        .add_operand(dnvec_y)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, mode_a)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, compute_type);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spmv")
}

/// `gpu.spmat_get_size` operation.
pub trait SpmatGetSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self, 1)
    }

    /// Returns sparse matrix descriptor.
    fn spmat(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(fixed_operand_start(self, 1)).unwrap()
    }

    /// Returns matrix row count.
    fn rows(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns matrix column count.
    fn cols(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }

    /// Returns matrix non-zero count.
    fn nnz(&self) -> ValueRef<'o, 'c, 't> {
        self.result(2).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(3).map(|result| result.as_ref())
    }
}

mlir_op!(SpmatGetSize);

/// Constructs a detached `gpu.spmat_get_size` operation.
pub fn spmat_get_size<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    spmat: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedSpmatGetSizeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.spmat_get_size", location)
        .add_operands(async_dependencies)
        .add_operand(spmat)
        .add_result(context.index_type())
        .add_result(context.index_type())
        .add_result(context.index_type());
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "spmat_get_size")
}

/// `gpu.subgroup_broadcast` operation.
pub trait SubgroupBroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to broadcast.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns optional source lane operand.
    fn lane(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand(1)
    }

    /// Returns broadcast type attribute.
    fn broadcast_type(&self) -> BroadcastTypeAttributeRef<'c, 't> {
        self.attribute(BROADCAST_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BroadcastTypeAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{BROADCAST_TYPE_ATTRIBUTE}' attribute in `gpu.subgroup_broadcast`"))
    }

    /// Returns broadcast result.
    fn result_value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupBroadcast);
mlir_op_trait!(SubgroupBroadcast, OneResult);

/// Constructs a detached `gpu.subgroup_broadcast` operation.
pub fn subgroup_broadcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    lane: Option<ValueRef<'o, 'c, 't>>,
    broadcast_type: BroadcastTypeAttributeRef<'c, 't>,
    location: L,
) -> DetachedSubgroupBroadcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.subgroup_broadcast", location)
        .add_operand(value)
        .add_attribute(BROADCAST_TYPE_ATTRIBUTE, broadcast_type)
        .add_result(value.r#type());
    if let Some(lane) = lane {
        builder = builder.add_operand(lane);
    }
    build_gpu_operation(builder, "subgroup_broadcast")
}

/// `gpu.subgroup_mma_compute` operation.
pub trait SubgroupMmaComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns matrix A operand.
    fn matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns matrix B operand.
    fn matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns accumulator matrix C operand.
    fn matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(2).unwrap()
    }

    /// Returns matrix D result.
    fn matrix_d(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaCompute);
mlir_op_trait!(SubgroupMmaCompute, OneResult);

/// Constructs a detached `gpu.subgroup_mma_compute` operation.
pub fn subgroup_mma_compute<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    matrix_a: ValueRef<'o, 'c, 't>,
    matrix_b: ValueRef<'o, 'c, 't>,
    matrix_c: ValueRef<'o, 'c, 't>,
    location: L,
) -> DetachedSubgroupMmaComputeOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.subgroup_mma_compute", location)
        .add_operand(matrix_a)
        .add_operand(matrix_b)
        .add_operand(matrix_c)
        .add_result(matrix_c.r#type());
    build_gpu_operation(builder, "subgroup_mma_compute")
}

/// `gpu.subgroup_mma_constant_matrix` operation.
pub trait SubgroupMmaConstantMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constant matrix payload.
    fn value_attribute(&self) -> AttributeRef<'c, 't> {
        self.attribute(VALUE_ATTRIBUTE)
            .unwrap_or_else(|| panic!("invalid '{VALUE_ATTRIBUTE}' attribute in `gpu.subgroup_mma_constant_matrix`"))
    }

    /// Returns matrix result.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaConstantMatrix);
mlir_op_trait!(SubgroupMmaConstantMatrix, OneResult);
mlir_op_trait!(SubgroupMmaConstantMatrix, ZeroOperands);

/// Constructs a detached `gpu.subgroup_mma_constant_matrix` operation.
pub fn subgroup_mma_constant_matrix<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    value: AttributeRef<'c, 't>,
    result_type: T,
    location: L,
) -> DetachedSubgroupMmaConstantMatrixOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.subgroup_mma_constant_matrix", location)
        .add_attribute(VALUE_ATTRIBUTE, value)
        .add_result(result_type);
    build_gpu_operation(builder, "subgroup_mma_constant_matrix")
}

/// `gpu.subgroup_mma_elementwise` operation.
pub trait SubgroupMmaElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns left-hand-side matrix operand.
    fn lhs(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns optional right-hand-side matrix operand.
    fn rhs(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand(1)
    }

    /// Returns elementwise operation kind.
    fn op_type(&self) -> MmaElementwiseOpAttributeRef<'c, 't> {
        self.attribute(OP_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MmaElementwiseOpAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{OP_TYPE_ATTRIBUTE}' attribute in `gpu.subgroup_mma_elementwise`"))
    }

    /// Returns elementwise result.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaElementwise);
mlir_op_trait!(SubgroupMmaElementwise, OneResult);

/// Constructs a detached `gpu.subgroup_mma_elementwise` operation.
pub fn subgroup_mma_elementwise<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lhs: ValueRef<'o, 'c, 't>,
    rhs: Option<ValueRef<'o, 'c, 't>>,
    op_type: MmaElementwiseOpAttributeRef<'c, 't>,
    location: L,
) -> DetachedSubgroupMmaElementwiseOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_elementwise", location)
        .add_operand(lhs)
        .add_attribute(OP_TYPE_ATTRIBUTE, op_type)
        .add_result(lhs.r#type());
    if let Some(rhs) = rhs {
        builder = builder.add_operand(rhs);
    }
    build_gpu_operation(builder, "subgroup_mma_elementwise")
}

/// `gpu.subgroup_mma_extract_thread_local` operation.
pub trait SubgroupMmaExtractThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns source MMA matrix operand.
    fn src_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns optional element index operand.
    fn id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand(1)
    }

    /// Returns extracted thread-local result.
    fn result_value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaExtractThreadLocal);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, OneResult);

/// Constructs a detached `gpu.subgroup_mma_extract_thread_local` operation.
pub fn subgroup_mma_extract_thread_local<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    src_matrix: ValueRef<'o, 'c, 't>,
    id: Option<ValueRef<'o, 'c, 't>>,
    result_type: T,
    location: L,
) -> DetachedSubgroupMmaExtractThreadLocalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_extract_thread_local", location)
        .add_operand(src_matrix)
        .add_result(result_type);
    if let Some(id) = id {
        builder = builder.add_operand(id);
    }
    build_gpu_operation(builder, "subgroup_mma_extract_thread_local")
}

/// `gpu.subgroup_mma_insert_thread_local` operation.
pub trait SubgroupMmaInsertThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns source thread-local value operand.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns destination MMA matrix operand.
    fn dst_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns optional element index operand.
    fn id(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand(2)
    }

    /// Returns updated matrix result.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaInsertThreadLocal);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, OneResult);

/// Constructs a detached `gpu.subgroup_mma_insert_thread_local` operation.
pub fn subgroup_mma_insert_thread_local<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'o, 'c, 't>,
    dst_matrix: ValueRef<'o, 'c, 't>,
    id: Option<ValueRef<'o, 'c, 't>>,
    location: L,
) -> DetachedSubgroupMmaInsertThreadLocalOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_insert_thread_local", location)
        .add_operand(src)
        .add_operand(dst_matrix)
        .add_result(dst_matrix.r#type());
    if let Some(id) = id {
        builder = builder.add_operand(id);
    }
    build_gpu_operation(builder, "subgroup_mma_insert_thread_local")
}

/// `gpu.subgroup_mma_load_matrix` operation.
pub trait SubgroupMmaLoadMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns source memref operand.
    fn src_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns memref index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().skip(1).collect::<Vec<_>>()
    }

    /// Returns leading-dimension attribute.
    fn lead_dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(LEAD_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{LEAD_DIMENSION_ATTRIBUTE}' attribute in `gpu.subgroup_mma_load_matrix`")
            })
    }

    /// Returns optional transpose mode attribute.
    fn transpose(&self) -> Option<TransposeModeAttributeRef<'c, 't>> {
        self.attribute(TRANSPOSE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<TransposeModeAttributeRef>())
    }

    /// Returns loaded MMA matrix result.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaLoadMatrix);
mlir_op_trait!(SubgroupMmaLoadMatrix, OneResult);

/// Constructs a detached `gpu.subgroup_mma_load_matrix` operation.
pub fn subgroup_mma_load_matrix<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    src_memref: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    lead_dimension: i32,
    transpose: Option<TransposeModeAttributeRef<'c, 't>>,
    result_type: T,
    location: L,
) -> DetachedSubgroupMmaLoadMatrixOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let i32_type = context.signless_integer_type(32);
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_load_matrix", location)
        .add_operand(src_memref)
        .add_operands(indices)
        .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(i32_type, lead_dimension as i64))
        .add_result(result_type);
    builder = add_optional_attribute(builder, TRANSPOSE_ATTRIBUTE, transpose);
    build_gpu_operation(builder, "subgroup_mma_load_matrix")
}

/// `gpu.subgroup_mma_store_matrix` operation.
pub trait SubgroupMmaStoreMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns source MMA matrix operand.
    fn src(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns destination memref operand.
    fn dst_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(1).unwrap()
    }

    /// Returns memref index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().skip(2).collect::<Vec<_>>()
    }

    /// Returns leading-dimension attribute.
    fn lead_dimension(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(LEAD_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| {
                panic!("invalid '{LEAD_DIMENSION_ATTRIBUTE}' attribute in `gpu.subgroup_mma_store_matrix`")
            })
    }
}

mlir_op!(SubgroupMmaStoreMatrix);

/// Constructs a detached `gpu.subgroup_mma_store_matrix` operation.
pub fn subgroup_mma_store_matrix<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    src: ValueRef<'o, 'c, 't>,
    dst_memref: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    lead_dimension: i32,
    location: L,
) -> DetachedSubgroupMmaStoreMatrixOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let i32_type = context.signless_integer_type(32);
    let builder = OperationBuilder::new("gpu.subgroup_mma_store_matrix", location)
        .add_operand(src)
        .add_operand(dst_memref)
        .add_operands(indices)
        .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(i32_type, lead_dimension as i64));
    build_gpu_operation(builder, "subgroup_mma_store_matrix")
}

/// `gpu.subgroup_reduce` operation.
pub trait SubgroupReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns reduction input value.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns reduction operation kind.
    fn operation(&self) -> AllReduceOperationAttributeRef<'c, 't> {
        self.attribute(OP_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AllReduceOperationAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{OP_ATTRIBUTE}' attribute in `gpu.subgroup_reduce`"))
    }

    /// Returns optional subgroup cluster size.
    fn cluster_size(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(CLUSTER_SIZE_ATTRIBUTE).and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
    }

    /// Returns optional subgroup cluster stride.
    fn cluster_stride(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        self.attribute(CLUSTER_STRIDE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
    }

    /// Returns whether this reduction is uniform.
    fn uniform(&self) -> bool {
        self.has_attribute(UNIFORM_ATTRIBUTE)
    }

    /// Returns reduction result.
    fn result_value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupReduce);
mlir_op_trait!(SubgroupReduce, OneResult);

/// Constructs a detached `gpu.subgroup_reduce` operation.
pub fn subgroup_reduce<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    operation: AllReduceOperationAttributeRef<'c, 't>,
    cluster_size: Option<i32>,
    cluster_stride: Option<i32>,
    uniform: bool,
    location: L,
) -> DetachedSubgroupReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let i32_type = context.signless_integer_type(32);
    let mut builder = OperationBuilder::new("gpu.subgroup_reduce", location)
        .add_operand(value)
        .add_attribute(OP_ATTRIBUTE, operation)
        .add_result(value.r#type());
    if let Some(cluster_size) = cluster_size {
        builder =
            builder.add_attribute(CLUSTER_SIZE_ATTRIBUTE, context.integer_attribute(i32_type, cluster_size as i64));
    }
    if let Some(cluster_stride) = cluster_stride {
        builder =
            builder.add_attribute(CLUSTER_STRIDE_ATTRIBUTE, context.integer_attribute(i32_type, cluster_stride as i64));
    }
    if uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    build_gpu_operation(builder, "subgroup_reduce")
}

/// `gpu.subgroup_id` operation.
pub trait SubgroupIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(SubgroupId);
mlir_op_trait!(SubgroupId, OneResult);
mlir_op_trait!(SubgroupId, ZeroOperands);

/// Constructs a detached `gpu.subgroup_id` operation.
///
/// # Parameters
///
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn subgroup_id<'c, 't: 'c, L: Location<'c, 't>>(
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedSubgroupIdOperation<'c, 't> {
    build_upper_bound_query_operation("gpu.subgroup_id", "subgroup_id", upper_bound, location)
}

/// `gpu.subgroup_size` operation.
pub trait SubgroupSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(SubgroupSize);
mlir_op_trait!(SubgroupSize, OneResult);
mlir_op_trait!(SubgroupSize, ZeroOperands);

/// Constructs a detached `gpu.subgroup_size` operation.
///
/// # Parameters
///
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn subgroup_size<'c, 't: 'c, L: Location<'c, 't>>(
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedSubgroupSizeOperation<'c, 't> {
    build_upper_bound_query_operation("gpu.subgroup_size", "subgroup_size", upper_bound, location)
}

/// `gpu.terminator` operation.
pub trait TerminatorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Terminator);
mlir_op_trait!(Terminator, ZeroOperands);
mlir_op_trait!(Terminator, ZeroRegions);
mlir_op_trait!(Terminator, ZeroSuccessors);

/// Constructs a detached `gpu.terminator` operation.
///
/// # Parameters
///
///   - `location`: Source location for the created operation.
pub fn terminator<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedTerminatorOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.terminator", location);
    build_gpu_operation(builder, "terminator")
}

/// `gpu.thread_id` operation.
pub trait ThreadIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the queried dimension.
    fn dimension(&self) -> Dimension {
        required_dimension_attribute(self, "gpu.thread_id")
    }

    /// Returns the optional upper bound.
    fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_upper_bound_attribute(self)
    }
}

mlir_op!(ThreadId);
mlir_op_trait!(ThreadId, OneResult);
mlir_op_trait!(ThreadId, ZeroOperands);

/// Constructs a detached `gpu.thread_id` operation.
///
/// # Parameters
///
///   - `dimension`: Dimension (`x`, `y`, or `z`) to query.
///   - `upper_bound`: Optional upper-bound hint.
///   - `location`: Source location for the created operation.
pub fn thread_id<'c, 't: 'c, L: Location<'c, 't>>(
    dimension: Dimension,
    upper_bound: Option<IntegerAttributeRef<'c, 't>>,
    location: L,
) -> DetachedThreadIdOperation<'c, 't> {
    build_dimension_query_operation("gpu.thread_id", "thread_id", dimension, upper_bound, location)
}

/// `gpu.warp_execute_on_lane_0` operation.
pub trait WarpExecuteOnLane0Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns lane ID operand.
    fn lane_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand(0).unwrap()
    }

    /// Returns forwarded warp operands.
    fn args(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().skip(1).collect::<Vec<_>>()
    }

    /// Returns warp size attribute.
    fn warp_size(&self) -> IntegerAttributeRef<'c, 't> {
        self.attribute(WARP_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .unwrap_or_else(|| panic!("invalid '{WARP_SIZE_ATTRIBUTE}' attribute in `gpu.warp_execute_on_lane_0`"))
    }

    /// Returns warp region.
    fn body(&self) -> crate::RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }

    /// Returns lane-0 results available to all lanes.
    fn warp_results(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        Operation::results(self).map(|result| result.as_ref()).collect::<Vec<_>>()
    }
}

mlir_op!(WarpExecuteOnLane0);

/// Constructs a detached `gpu.warp_execute_on_lane_0` operation.
pub fn warp_execute_on_lane_0<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lane_id: ValueRef<'o, 'c, 't>,
    args: &[ValueRef<'o, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    warp_size: i64,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedWarpExecuteOnLane0Operation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let i64_type = context.signless_integer_type(64);
    let builder = OperationBuilder::new("gpu.warp_execute_on_lane_0", location)
        .add_operand(lane_id)
        .add_operands(args)
        .add_attribute(WARP_SIZE_ATTRIBUTE, context.integer_attribute(i64_type, warp_size))
        .add_results(result_types)
        .add_region(body);
    build_gpu_operation(builder, "warp_execute_on_lane_0")
}

/// `gpu.wait` operation.
pub trait WaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().collect::<Vec<_>>()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(0).map(|result| result.as_ref())
    }
}

mlir_op!(Wait);
mlir_op_trait!(Wait, ZeroRegions);
mlir_op_trait!(Wait, ZeroSuccessors);

/// Constructs a detached `gpu.wait` operation.
///
/// # Parameters
///
///   - `async_dependencies`: Async dependencies to synchronize on.
///   - `is_async`: Whether to return an async token.
///   - `location`: Source location for the created operation.
pub fn wait<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> DetachedWaitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.wait", location).add_operands(async_dependencies);
    builder = add_async_token_result(builder, is_async);
    build_gpu_operation(builder, "wait")
}

/// `gpu.yield` operation.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + ReturnLike<'o, 'c, 't> {
    /// Returns yielded values.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operands().collect::<Vec<_>>()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a detached `gpu.yield` operation.
///
/// # Parameters
///
///   - `values`: Values yielded to the enclosing GPU op.
///   - `location`: Source location for the created operation.
pub fn r#yield<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    values: &[ValueRef<'o, 'c, 't>],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let builder = OperationBuilder::new("gpu.yield", location).add_operands(values);
    build_gpu_operation(builder, "yield")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Block, Context, Operation, Type};

    use super::*;

    #[test]
    fn test_gpu_operation_count() {
        assert_eq!(OPERATIONS.len(), 66);
    }

    #[test]
    fn test_thread_id_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let upper_bound = context.integer_attribute(context.index_type(), 1024);

        let op = thread_id(Dimension::X, Some(upper_bound), location);
        assert_eq!(op.dimension(), Dimension::X);
        assert_eq!(op.upper_bound().unwrap().signless_value(), 1024);
        assert_eq!(op.result_count(), 1);
    }

    #[test]
    fn test_alloc_operand_segment_sizes() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let memref_type = context
            .mem_ref_type(context.float32_type(), &[crate::Size::Static(4)], None, None, location)
            .unwrap();
        let block = context.block(&[(index_type, location), (index_type, location), (index_type, location)]);

        let async_dependencies = vec![block.argument(0).unwrap().as_ref()];
        let dynamic_sizes = vec![block.argument(1).unwrap().as_ref()];
        let symbol_operands = vec![block.argument(2).unwrap().as_ref()];

        let op = alloc(
            async_dependencies.as_slice(),
            dynamic_sizes.as_slice(),
            symbol_operands.as_slice(),
            memref_type,
            true,
            true,
            location,
        );

        let sizes = op
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap();
        assert_eq!(sizes, vec![1, 1, 1]);
        assert!(op.host_shared());
        assert_eq!(op.result_count(), 2);
    }

    #[test]
    fn test_launch_operand_segment_sizes() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);

        let arguments = (0..10).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let async_dependencies = vec![arguments[0]];
        let dimensions = LaunchDimensions {
            grid_size: [arguments[1], arguments[2], arguments[3]],
            block_size: [arguments[4], arguments[5], arguments[6]],
            cluster_size: Some([arguments[7], arguments[8], arguments[9]]),
        };
        let body = context.region();

        let op = launch(async_dependencies.as_slice(), dimensions, None, None, None, body, true, location);

        let sizes = op
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap();
        assert_eq!(sizes, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
        assert_eq!(op.result_count(), 1);
    }

    #[test]
    fn test_launch_func_operand_segment_sizes() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);

        let arguments = (0..12).map(|index| block.argument(index).unwrap().as_ref()).collect::<Vec<_>>();
        let async_dependencies = vec![arguments[0]];
        let dimensions = LaunchDimensions {
            grid_size: [arguments[1], arguments[2], arguments[3]],
            block_size: [arguments[4], arguments[5], arguments[6]],
            cluster_size: Some([arguments[7], arguments[8], arguments[9]]),
        };
        let kernel_operands = vec![arguments[10]];

        let op = launch_func(
            async_dependencies.as_slice(),
            context.symbol_ref_attribute("my_module".into(), &[context.flat_symbol_ref_attribute("my_kernel")]),
            dimensions,
            Some(arguments[11]),
            kernel_operands.as_slice(),
            None,
            true,
            location,
        );

        let sizes = op
            .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(Vec::<i32>::from)
            .unwrap();
        assert_eq!(sizes, vec![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0]);
        assert_eq!(op.result_count(), 1);
    }

    #[test]
    fn test_subgroup_reduce_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location)]);
        let value = block.argument(0).unwrap().as_ref();

        let op = subgroup_reduce(
            value,
            context.gpu_all_reduce_operation_attribute(crate::dialects::gpu::attributes::AllReduceOperation::Add),
            Some(8),
            Some(2),
            true,
            location,
        );

        assert_eq!(op.operation().value(), crate::dialects::gpu::attributes::AllReduceOperation::Add);
        assert_eq!(op.cluster_size().unwrap().signless_value(), 8);
        assert_eq!(op.cluster_stride().unwrap().signless_value(), 2);
        assert!(op.uniform());
    }

    #[test]
    fn test_spmat_get_size_results() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[
            (context.gpu_async_token_type().as_ref(), location),
            (context.gpu_sparse_spmat_handle_type().as_ref(), location),
        ]);

        let async_dependencies = vec![block.argument(0).unwrap().as_ref()];
        let spmat = block.argument(1).unwrap().as_ref();
        let op = spmat_get_size(async_dependencies.as_slice(), spmat, true, location);

        assert_eq!(op.result_count(), 4);
        assert_eq!(op.rows().r#type(), context.index_type().as_ref());
        assert_eq!(op.cols().r#type(), context.index_type().as_ref());
        assert_eq!(op.nnz().r#type(), context.index_type().as_ref());
        assert!(op.async_token().is_some());
    }

    #[test]
    fn test_warp_execute_on_lane_0_attributes() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let f32_type = context.float32_type();
        let block = context.block(&[(i32_type.as_ref(), location), (f32_type.as_ref(), location)]);
        let lane_id = block.argument(0).unwrap().as_ref();
        let args = vec![block.argument(1).unwrap().as_ref()];

        let op =
            warp_execute_on_lane_0(lane_id, args.as_slice(), &[f32_type.as_ref()], 32, context.region(), location);

        assert_eq!(op.warp_size().signless_value(), 32);
        assert_eq!(op.result_count(), 1);
    }
}
