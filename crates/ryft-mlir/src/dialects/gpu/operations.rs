use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, DenseInteger32ArrayAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, FlatSymbolRefAttributeRef, Function, IntegerAttributeRef, Location, OneRegion, Operation,
    OperationBuilder, RegionRef, StringAttributeRef, StringRef, Symbol, SymbolTable, Type, TypeAttributeRef, TypeRef,
    Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::attributes::{
    AddressSpace, AddressSpaceAttributeRef, AllReduceOperationKind, AllReduceOperationKindAttributeRef, BroadcastType,
    BroadcastTypeAttributeRef, Dimension, DimensionAttributeRef, MatrixTransposeMode, MatrixTransposeModeAttributeRef,
    MmaElementwiseOperation, MmaElementwiseOperationAttributeRef, Prune2To4SparseMatrixFlag,
    Prune2To4SparseMatrixFlagAttributeRef, ShuffleMode, ShuffleModeAttributeRef, SpGemmWorkKind,
    SpGemmWorkKindAttributeRef,
};

/// A three-dimensional bundle of GPU launch operands.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Dim3<'o, 'c: 'o, 't: 'c> {
    /// X dimension value.
    pub x: ValueRef<'o, 'c, 't>,

    /// Y dimension value.
    pub y: ValueRef<'o, 'c, 't>,

    /// Z dimension value.
    pub z: ValueRef<'o, 'c, 't>,
}

impl<'o, 'c: 'o, 't: 'c> Dim3<'o, 'c, 't> {
    /// Returns the dimension values as an array.
    pub fn values(&self) -> [ValueRef<'o, 'c, 't>; 3] {
        [self.x, self.y, self.z]
    }
}

fn required_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    attribute_name: &str,
    operation_name: &str,
) -> AttributeRef<'c, 't> {
    operation
        .attribute(attribute_name)
        .unwrap_or_else(|| panic!("invalid '{attribute_name}' attribute in `{operation_name}`"))
}

fn optional_unit_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O, attribute_name: &str) -> bool {
    operation.attribute(attribute_name).is_some()
}

fn integer_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    attribute_name: &str,
    operation_name: &str,
) -> IntegerAttributeRef<'c, 't> {
    required_attribute(operation, attribute_name, operation_name)
        .cast()
        .unwrap_or_else(|| panic!("invalid '{attribute_name}' attribute in `{operation_name}`"))
}

fn optional_integer_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    attribute_name: &str,
) -> Option<IntegerAttributeRef<'c, 't>> {
    operation.attribute(attribute_name).and_then(|attribute| attribute.cast())
}

fn matrix_transpose_mode<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    attribute_name: &str,
    operation_name: &str,
) -> MatrixTransposeMode {
    required_attribute(operation, attribute_name, operation_name)
        .cast::<MatrixTransposeModeAttributeRef>()
        .map(|attribute| attribute.value())
        .unwrap_or_else(|| panic!("invalid `{attribute_name}` attribute in `{operation_name}`"))
}

fn type_attribute<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    attribute_name: &str,
    operation_name: &str,
) -> TypeRef<'c, 't> {
    required_attribute(operation, attribute_name, operation_name)
        .cast::<TypeAttributeRef>()
        .map(|attribute| attribute.r#type())
        .unwrap_or_else(|| panic!("invalid `{attribute_name}` attribute in `{operation_name}`"))
}

fn operand_segment_sizes<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O) -> Vec<usize> {
    operation
        .attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE)
        .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
        .map(|attribute| attribute.values().map(|value| value as usize).collect())
        .unwrap_or_else(|| panic!("invalid '{OPERAND_SEGMENT_SIZES_ATTRIBUTE}' attribute in `{}`", operation.name()))
}

fn result_segment_sizes<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O) -> Vec<usize> {
    operation
        .attribute(RESULT_SEGMENT_SIZES_ATTRIBUTE)
        .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
        .map(|attribute| attribute.values().map(|value| value as usize).collect())
        .unwrap_or_else(|| panic!("invalid '{RESULT_SEGMENT_SIZES_ATTRIBUTE}' attribute in `{}`", operation.name()))
}

fn operand_segment<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    segment_index: usize,
) -> Vec<ValueRef<'o, 'c, 't>> {
    let segment_sizes = operand_segment_sizes(operation);
    let start = segment_sizes.iter().take(segment_index).sum::<usize>();
    operation.operand_values().skip(start).take(segment_sizes[segment_index]).collect()
}

fn result_segment<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
    segment_index: usize,
) -> Vec<ValueRef<'o, 'c, 't>> {
    let segment_sizes = result_segment_sizes(operation);
    let start = segment_sizes.iter().take(segment_index).sum::<usize>();
    operation
        .results()
        .skip(start)
        .take(segment_sizes[segment_index])
        .map(|result| result.as_ref())
        .collect()
}

fn leading_async_dependency_count<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O) -> usize {
    let token_type = operation.context().gpu_async_token_type().as_ref();
    operation.operand_values().take_while(|operand| operand.r#type() == token_type).count()
}

fn leading_async_dependencies<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(
    operation: &O,
) -> Vec<ValueRef<'o, 'c, 't>> {
    operation.operand_values().take(leading_async_dependency_count(operation)).collect()
}

fn optional_async_token<'o, 'c: 'o, 't: 'c, O: Operation<'o, 'c, 't>>(operation: &O) -> Option<ValueRef<'o, 'c, 't>> {
    operation.result(0).map(|result| result.as_ref())
}

fn add_optional_async_result<'c, 't: 'c>(
    builder: OperationBuilder<'c, 't>,
    is_async: bool,
) -> OperationBuilder<'c, 't> {
    if is_async {
        let async_token_type = builder.context().gpu_async_token_type();
        builder.add_result(async_token_type)
    } else {
        builder
    }
}

fn add_operand_segments<'c, 't: 'c>(
    builder: OperationBuilder<'c, 't>,
    segment_sizes: &[usize],
) -> OperationBuilder<'c, 't> {
    let segment_sizes = segment_sizes.iter().map(|size| *size as i32).collect::<Vec<_>>();
    let segment_sizes = builder.context().dense_i32_array_attribute(segment_sizes.as_slice()).unwrap();
    builder.add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment_sizes)
}

/// Name of the [`Attribute`] that stores a GPU dimension.
pub const DIMENSION_ATTRIBUTE: &str = "dimension";

/// Name of the [`Attribute`] that stores an optional index upper bound.
pub const UPPER_BOUND_ATTRIBUTE: &str = "upper_bound";

macro_rules! gpu_dimension_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $doc:literal $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            ///
            /// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
            /// for more information.
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns the GPU dimension queried by this operation.
                fn dimension(&self) -> Dimension {
                    required_attribute(self, DIMENSION_ATTRIBUTE, $operation_name)
                        .cast::<DimensionAttributeRef>()
                        .map(|attribute| attribute.value())
                        .unwrap_or_else(|| panic!("invalid '{DIMENSION_ATTRIBUTE}' attribute in `{}`", $operation_name))
                }

                /// Returns the optional upper bound associated with this operation.
                fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
                    optional_integer_attribute(self, UPPER_BOUND_ATTRIBUTE)
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, OneResult);
            mlir_op_trait!($name, ZeroOperands);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached/owned [`"]
            #[doc = stringify!($name)]
            #[doc = "Operation`] at the specified [`Location`]."]
            pub fn $function_name<'c, 't: 'c, L: Location<'c, 't>>(
                dimension: Dimension,
                upper_bound: Option<usize>,
                location: L,
            ) -> [<Detached $name Operation>]<'c, 't> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu());
                let builder = OperationBuilder::new($operation_name, location)
                    .add_attribute(DIMENSION_ATTRIBUTE, context.gpu_dimension_attribute(dimension))
                    .add_result(context.index_type());
                let builder = if let Some(upper_bound) = upper_bound {
                    builder.add_attribute(
                        UPPER_BOUND_ATTRIBUTE,
                        context.integer_attribute(context.index_type(), upper_bound as i64),
                    )
                } else {
                    builder
                };
                builder
                    .build()
                    .and_then(|operation| unsafe { operation.cast() })
                    .expect(concat!("invalid arguments to `gpu::", stringify!($function_name), "`"))
            }
        }
    };
}

macro_rules! gpu_upper_bound_index_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $doc:literal $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            ///
            /// Refer to the [official MLIR GPU dialect documentation](https://mlir.llvm.org/docs/Dialects/GPU/)
            /// for more information.
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns the optional upper bound associated with this operation.
                fn upper_bound(&self) -> Option<IntegerAttributeRef<'c, 't>> {
                    optional_integer_attribute(self, UPPER_BOUND_ATTRIBUTE)
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, OneResult);
            mlir_op_trait!($name, ZeroOperands);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached/owned [`"]
            #[doc = stringify!($name)]
            #[doc = "Operation`] at the specified [`Location`]."]
            pub fn $function_name<'c, 't: 'c, L: Location<'c, 't>>(
                upper_bound: Option<usize>,
                location: L,
            ) -> [<Detached $name Operation>]<'c, 't> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu());
                let builder = OperationBuilder::new($operation_name, location).add_result(context.index_type());
                let builder = if let Some(upper_bound) = upper_bound {
                    builder.add_attribute(
                        UPPER_BOUND_ATTRIBUTE,
                        context.integer_attribute(context.index_type(), upper_bound as i64),
                    )
                } else {
                    builder
                };
                builder
                    .build()
                    .and_then(|operation| unsafe { operation.cast() })
                    .expect(concat!("invalid arguments to `gpu::", stringify!($function_name), "`"))
            }
        }
    };
}

gpu_dimension_operation!(
    ClusterDim,
    cluster_dim,
    "gpu.cluster_dim",
    "GPU operation that returns the number of clusters per grid along a dimension.",
);

gpu_dimension_operation!(
    ClusterDimBlocks,
    cluster_dim_blocks,
    "gpu.cluster_dim_blocks",
    "GPU operation that returns the number of thread blocks in a cluster along a dimension.",
);

gpu_dimension_operation!(
    ClusterId,
    cluster_id,
    "gpu.cluster_id",
    "GPU operation that returns the current cluster identifier along a dimension.",
);

gpu_dimension_operation!(
    ClusterBlockId,
    cluster_block_id,
    "gpu.cluster_block_id",
    "GPU operation that returns the block identifier within a cluster along a dimension.",
);

gpu_dimension_operation!(
    BlockDim,
    block_dim,
    "gpu.block_dim",
    "GPU operation that returns the number of threads in a block along a dimension.",
);

gpu_dimension_operation!(
    BlockId,
    block_id,
    "gpu.block_id",
    "GPU operation that returns the current block identifier along a dimension.",
);

gpu_dimension_operation!(
    GridDim,
    grid_dim,
    "gpu.grid_dim",
    "GPU operation that returns the number of thread blocks in the grid along a dimension.",
);

gpu_dimension_operation!(
    ThreadId,
    thread_id,
    "gpu.thread_id",
    "GPU operation that returns the current thread identifier along a dimension.",
);

gpu_dimension_operation!(
    GlobalId,
    global_id,
    "gpu.global_id",
    "GPU operation that returns the global work item identifier along a dimension.",
);

gpu_upper_bound_index_operation!(
    LaneId,
    lane_id,
    "gpu.lane_id",
    "GPU operation that returns the current lane identifier within a subgroup.",
);

gpu_upper_bound_index_operation!(
    SubgroupId,
    subgroup_id,
    "gpu.subgroup_id",
    "GPU operation that returns the current subgroup identifier within a workgroup.",
);

gpu_upper_bound_index_operation!(
    NumSubgroups,
    num_subgroups,
    "gpu.num_subgroups",
    "GPU operation that returns the number of subgroups in a workgroup.",
);

gpu_upper_bound_index_operation!(
    SubgroupSize,
    subgroup_size,
    "gpu.subgroup_size",
    "GPU operation that returns the number of threads in a subgroup.",
);

/// Name of the [`Attribute`] that marks a GPU function as a kernel.
pub const KERNEL_ATTRIBUTE: &str = "kernel";

/// Name of the [`Attribute`] that stores GPU function known block sizes.
pub const KNOWN_BLOCK_SIZE_ATTRIBUTE: &str = "known_block_size";

/// Name of the [`Attribute`] that stores GPU function known grid sizes.
pub const KNOWN_GRID_SIZE_ATTRIBUTE: &str = "known_grid_size";

/// Name of the [`Attribute`] that stores GPU function known cluster sizes.
pub const KNOWN_CLUSTER_SIZE_ATTRIBUTE: &str = "known_cluster_size";

/// Name of the [`Attribute`] that stores GPU function workgroup attribution attributes.
pub const WORKGROUP_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE: &str = "workgroup_attrib_attrs";

/// Name of the [`Attribute`] that stores GPU function private attribution attributes.
pub const PRIVATE_ATTRIBUTION_ATTRIBUTES_ATTRIBUTE: &str = "private_attrib_attrs";

/// Name of the [`Attribute`] that stores the number of workgroup attributions.
pub const WORKGROUP_ATTRIBUTIONS_ATTRIBUTE: &str = "workgroup_attributions";

/// GPU function operation executable on a GPU device.
pub trait FuncOperation<'o, 'c: 'o, 't: 'c>: Function<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns `true` if this GPU function is a kernel.
    fn is_kernel(&self) -> bool {
        optional_unit_attribute(self, KERNEL_ATTRIBUTE)
    }

    /// Returns the optional known block size hint.
    fn known_block_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
    }

    /// Returns the optional known grid size hint.
    fn known_grid_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_GRID_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
    }

    /// Returns the optional known cluster size hint.
    fn known_cluster_size(&self) -> Option<Vec<i32>> {
        self.attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseInteger32ArrayAttributeRef>())
            .map(|attribute| attribute.values().collect())
    }

    /// Returns the number of workgroup attributions.
    fn workgroup_attribution_count(&self) -> usize {
        optional_integer_attribute(self, WORKGROUP_ATTRIBUTIONS_ATTRIBUTE)
            .map(|attribute| attribute.signless_value() as usize)
            .unwrap_or(0)
    }

    /// Returns the GPU function body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(Func);
mlir_op_trait!(Func, AffineScope);
mlir_op_trait!(Func, AutomaticAllocationScope);
mlir_op_trait!(Func, HasCallableArgumentAndResultAttributes);
mlir_op_trait!(Func, Callable);
mlir_op_trait!(Func, Function);
mlir_op_trait!(Func, IsolatedFromAbove);
mlir_op_trait!(Func, OneRegion);
mlir_op_trait!(Func, Symbol);
mlir_op_trait!(Func, ZeroSuccessors);

/// GPU operation that returns the dynamic shared-memory memref for the current kernel.
pub trait DynamicSharedMemoryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dynamic shared-memory memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(DynamicSharedMemory);
mlir_op_trait!(DynamicSharedMemory, OneResult);
mlir_op_trait!(DynamicSharedMemory, ZeroOperands);
mlir_op_trait!(DynamicSharedMemory, ZeroRegions);
mlir_op_trait!(DynamicSharedMemory, ZeroSuccessors);

/// Constructs a new detached/owned [`DynamicSharedMemoryOperation`] at the specified [`Location`].
pub fn dynamic_shared_memory<'c, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    memref_type: T,
    location: L,
) -> DetachedDynamicSharedMemoryOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.dynamic_shared_memory", location)
        .add_result(memref_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::dynamic_shared_memory`")
}

/// Name of the MLIR attribute storing operation operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// Properties used to construct a [`LaunchFuncOperation`].
#[derive(Clone, Debug)]
pub struct LaunchFuncProperties<'o, 'c: 'o, 't: 'c> {
    /// Async token dependencies.
    pub async_dependencies: Vec<ValueRef<'o, 'c, 't>>,

    /// Fully qualified kernel symbol reference.
    pub kernel: crate::SymbolRefAttributeRef<'c, 't>,

    /// Grid size operands.
    pub grid_size: Dim3<'o, 'c, 't>,

    /// Block size operands.
    pub block_size: Dim3<'o, 'c, 't>,

    /// Optional cluster size operands.
    pub cluster_size: Option<Dim3<'o, 'c, 't>>,

    /// Optional dynamic shared-memory size operand.
    pub dynamic_shared_memory_size: Option<ValueRef<'o, 'c, 't>>,

    /// Kernel argument operands.
    pub kernel_operands: Vec<ValueRef<'o, 'c, 't>>,

    /// Optional async object operand.
    pub async_object: Option<ValueRef<'o, 'c, 't>>,

    /// Whether to return an async token.
    pub is_async: bool,
}

/// GPU operation that launches a named GPU kernel function.
pub trait LaunchFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 0)
    }

    /// Returns the kernel symbol reference.
    fn kernel(&self) -> crate::SymbolRefAttributeRef<'c, 't> {
        required_attribute(self, KERNEL_ATTRIBUTE, "gpu.launch_func")
            .cast()
            .expect("invalid `kernel` attribute in `gpu.launch_func`")
    }

    /// Returns the grid size operands.
    fn grid_size(&self) -> Dim3<'o, 'c, 't> {
        Dim3 { x: operand_segment(self, 1)[0], y: operand_segment(self, 2)[0], z: operand_segment(self, 3)[0] }
    }

    /// Returns the block size operands.
    fn block_size(&self) -> Dim3<'o, 'c, 't> {
        Dim3 { x: operand_segment(self, 4)[0], y: operand_segment(self, 5)[0], z: operand_segment(self, 6)[0] }
    }

    /// Returns the optional cluster size operands.
    fn cluster_size(&self) -> Option<Dim3<'o, 'c, 't>> {
        let x = operand_segment(self, 7);
        let y = operand_segment(self, 8);
        let z = operand_segment(self, 9);
        if x.is_empty() { None } else { Some(Dim3 { x: x[0], y: y[0], z: z[0] }) }
    }

    /// Returns the optional dynamic shared-memory size operand.
    fn dynamic_shared_memory_size(&self) -> Option<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 10).first().copied()
    }

    /// Returns the kernel argument operands.
    fn kernel_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 11)
    }

    /// Returns the optional async object operand.
    fn async_object(&self) -> Option<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 12).first().copied()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(LaunchFunc);
mlir_op_trait!(LaunchFunc, ZeroRegions);
mlir_op_trait!(LaunchFunc, ZeroSuccessors);

/// Constructs a new detached/owned [`LaunchFuncOperation`] at the specified [`Location`].
pub fn launch_func<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    properties: LaunchFuncProperties<'o, 'c, 't>,
    location: L,
) -> DetachedLaunchFuncOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let cluster_size = properties.cluster_size.map(|size| size.values());
    let mut segment_sizes = vec![
        properties.async_dependencies.len(),
        1,
        1,
        1,
        1,
        1,
        1,
        usize::from(cluster_size.is_some()),
        usize::from(cluster_size.is_some()),
        usize::from(cluster_size.is_some()),
        usize::from(properties.dynamic_shared_memory_size.is_some()),
        properties.kernel_operands.len(),
        usize::from(properties.async_object.is_some()),
    ];
    let mut builder = OperationBuilder::new("gpu.launch_func", location)
        .add_operands(properties.async_dependencies.as_slice())
        .add_attribute(KERNEL_ATTRIBUTE, properties.kernel)
        .add_operands(properties.grid_size.values().as_slice())
        .add_operands(properties.block_size.values().as_slice());
    if let Some(cluster_size) = cluster_size {
        builder = builder.add_operands(cluster_size.as_slice());
    }
    if let Some(dynamic_shared_memory_size) = properties.dynamic_shared_memory_size {
        builder = builder.add_operand(dynamic_shared_memory_size);
    }
    builder = builder.add_operands(properties.kernel_operands.as_slice());
    if let Some(async_object) = properties.async_object {
        builder = builder.add_operand(async_object);
    }
    if properties.is_async {
        builder = builder.add_result(context.gpu_async_token_type());
    }
    if !properties.is_async {
        segment_sizes[12] = usize::from(properties.async_object.is_some());
    }
    add_operand_segments(builder, segment_sizes.as_slice())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::launch_func`")
}

/// Properties used to construct a [`LaunchOperation`].
#[derive(Clone, Debug)]
pub struct LaunchProperties<'o, 'c: 'o, 't: 'c> {
    /// Async token dependencies.
    pub async_dependencies: Vec<ValueRef<'o, 'c, 't>>,

    /// Grid size operands.
    pub grid_size: Dim3<'o, 'c, 't>,

    /// Block size operands.
    pub block_size: Dim3<'o, 'c, 't>,

    /// Optional cluster size operands.
    pub cluster_size: Option<Dim3<'o, 'c, 't>>,

    /// Optional dynamic shared-memory size operand.
    pub dynamic_shared_memory_size: Option<ValueRef<'o, 'c, 't>>,

    /// Optional module symbol to use when outlining the launch body.
    pub module: Option<FlatSymbolRefAttributeRef<'c, 't>>,

    /// Optional function symbol to use when outlining the launch body.
    pub function: Option<FlatSymbolRefAttributeRef<'c, 't>>,

    /// Number of workgroup attributions in the launch body.
    pub workgroup_attributions: Option<usize>,

    /// Whether to return an async token.
    pub is_async: bool,
}

/// Name of the [`Attribute`] that stores GPU launch module symbols.
pub const MODULE_ATTRIBUTE: &str = "module";

/// Name of the [`Attribute`] that stores GPU launch function symbols.
pub const FUNCTION_ATTRIBUTE: &str = "function";

/// GPU operation that launches an inline GPU kernel region.
pub trait LaunchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 0)
    }

    /// Returns the grid size operands.
    fn grid_size(&self) -> Dim3<'o, 'c, 't> {
        Dim3 { x: operand_segment(self, 1)[0], y: operand_segment(self, 2)[0], z: operand_segment(self, 3)[0] }
    }

    /// Returns the block size operands.
    fn block_size(&self) -> Dim3<'o, 'c, 't> {
        Dim3 { x: operand_segment(self, 4)[0], y: operand_segment(self, 5)[0], z: operand_segment(self, 6)[0] }
    }

    /// Returns optional cluster size operands.
    fn cluster_size(&self) -> Option<Dim3<'o, 'c, 't>> {
        let x = operand_segment(self, 7);
        let y = operand_segment(self, 8);
        let z = operand_segment(self, 9);
        if x.is_empty() { None } else { Some(Dim3 { x: x[0], y: y[0], z: z[0] }) }
    }

    /// Returns the optional dynamic shared-memory size operand.
    fn dynamic_shared_memory_size(&self) -> Option<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 10).first().copied()
    }

    /// Returns the optional module symbol.
    fn module_symbol(&self) -> Option<FlatSymbolRefAttributeRef<'c, 't>> {
        self.attribute(MODULE_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional function symbol.
    fn function_symbol(&self) -> Option<FlatSymbolRefAttributeRef<'c, 't>> {
        self.attribute(FUNCTION_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the number of workgroup attributions.
    fn workgroup_attribution_count(&self) -> usize {
        optional_integer_attribute(self, WORKGROUP_ATTRIBUTIONS_ATTRIBUTE)
            .map(|attribute| attribute.signless_value() as usize)
            .unwrap_or(0)
    }

    /// Returns the launch body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(Launch);
mlir_op_trait!(Launch, AffineScope);
mlir_op_trait!(Launch, AutomaticAllocationScope);
mlir_op_trait!(Launch, OneRegion);
mlir_op_trait!(Launch, ZeroSuccessors);

/// Constructs a new detached/owned [`LaunchOperation`] at the specified [`Location`].
pub fn launch<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    properties: LaunchProperties<'o, 'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedLaunchOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let cluster_size = properties.cluster_size.map(|size| size.values());
    let segment_sizes = vec![
        properties.async_dependencies.len(),
        1,
        1,
        1,
        1,
        1,
        1,
        usize::from(cluster_size.is_some()),
        usize::from(cluster_size.is_some()),
        usize::from(cluster_size.is_some()),
        usize::from(properties.dynamic_shared_memory_size.is_some()),
    ];
    let mut builder = OperationBuilder::new("gpu.launch", location)
        .add_operands(properties.async_dependencies.as_slice())
        .add_operands(properties.grid_size.values().as_slice())
        .add_operands(properties.block_size.values().as_slice());
    if let Some(cluster_size) = cluster_size {
        builder = builder.add_operands(cluster_size.as_slice());
    }
    if let Some(dynamic_shared_memory_size) = properties.dynamic_shared_memory_size {
        builder = builder.add_operand(dynamic_shared_memory_size);
    }
    if let Some(module) = properties.module {
        builder = builder.add_attribute(MODULE_ATTRIBUTE, module);
    }
    if let Some(function) = properties.function {
        builder = builder.add_attribute(FUNCTION_ATTRIBUTE, function);
    }
    if let Some(workgroup_attributions) = properties.workgroup_attributions {
        builder = builder.add_attribute(
            WORKGROUP_ATTRIBUTIONS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), workgroup_attributions as i64),
        );
    }
    if properties.is_async {
        builder = builder.add_result(context.gpu_async_token_type());
    }
    add_operand_segments(builder.add_region(body), segment_sizes.as_slice())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::launch`")
}

/// Name of the [`Attribute`] that stores a `printf` format string.
pub const FORMAT_ATTRIBUTE: &str = "format";

/// GPU device-side printf operation.
pub trait PrintfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the printf format string.
    fn format(&self) -> StringRef<'c> {
        required_attribute(self, FORMAT_ATTRIBUTE, "gpu.printf")
            .cast::<StringAttributeRef>()
            .map(|attribute| attribute.string())
            .expect("invalid `format` attribute in `gpu.printf`")
    }

    /// Returns the printf argument operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Printf);
mlir_op_trait!(Printf, ZeroRegions);
mlir_op_trait!(Printf, ZeroSuccessors);

/// Constructs a new detached/owned [`PrintfOperation`] at the specified [`Location`].
pub fn printf<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    format: &str,
    arguments: &[V],
    location: L,
) -> DetachedPrintfOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.printf", location)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .add_operands(arguments)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::printf`")
}

/// GPU function return terminator operation.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned values.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Return);
mlir_op_trait!(Return, SingleBlockRegions);
mlir_op_trait!(Return, IsTerminator);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
pub fn r#return<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedReturnOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::return`")
}

/// GPU launch-body terminator operation.
pub trait TerminatorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Terminator);
mlir_op_trait!(Terminator, SingleBlockRegions);
mlir_op_trait!(Terminator, IsTerminator);
mlir_op_trait!(Terminator, ZeroOperands);
mlir_op_trait!(Terminator, ZeroRegions);
mlir_op_trait!(Terminator, ZeroSuccessors);

/// Constructs a new detached/owned [`TerminatorOperation`] at the specified [`Location`].
pub fn terminator<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> DetachedTerminatorOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.terminator", location)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::terminator`")
}

/// GPU region yield operation.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }
}

mlir_op!(Yield);
mlir_op_trait!(Yield, SingleBlockRegions);
mlir_op_trait!(Yield, IsTerminator);
mlir_op_trait!(Yield, ReturnLike);
mlir_op_trait!(Yield, ZeroRegions);
mlir_op_trait!(Yield, ZeroSuccessors);

/// Constructs a new detached/owned [`YieldOperation`] at the specified [`Location`].
pub fn r#yield<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    values: &[V],
    location: L,
) -> DetachedYieldOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::yield`")
}

/// Name of the [`Attribute`] that stores an all-reduce or subgroup-reduce operation kind.
pub const OP_ATTRIBUTE: &str = "op";

/// Name of the [`Attribute`] that marks a collective operation as uniform.
pub const UNIFORM_ATTRIBUTE: &str = "uniform";

/// GPU all-reduce operation across a workgroup.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the value to reduce.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional built-in reduction operation kind.
    fn operation_kind(&self) -> Option<AllReduceOperationKind> {
        self.attribute(OP_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AllReduceOperationKindAttributeRef>())
            .map(|attribute| attribute.value())
    }

    /// Returns `true` if the collective is marked uniform.
    fn is_uniform(&self) -> bool {
        optional_unit_attribute(self, UNIFORM_ATTRIBUTE)
    }

    /// Returns the reduction body region.
    fn body(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, IsolatedFromAbove);
mlir_op_trait!(AllReduce, OneOperand);
mlir_op_trait!(AllReduce, OneRegion);
mlir_op_trait!(AllReduce, OneResult);
mlir_op_trait!(AllReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`AllReduceOperation`] at the specified [`Location`].
pub fn all_reduce<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    operation_kind: Option<AllReduceOperationKind>,
    is_uniform: bool,
    body: DetachedRegion<'c, 't>,
    result_type: TypeRef<'c, 't>,
    location: L,
) -> DetachedAllReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.all_reduce", location).add_operand(value).add_result(result_type);
    if let Some(operation_kind) = operation_kind {
        builder = builder.add_attribute(OP_ATTRIBUTE, context.gpu_all_reduce_operation_kind_attribute(operation_kind));
    }
    if is_uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::all_reduce`")
}

/// Name of the [`Attribute`] that stores subgroup cluster size.
pub const CLUSTER_SIZE_ATTRIBUTE: &str = "cluster_size";

/// Name of the [`Attribute`] that stores subgroup cluster stride.
pub const CLUSTER_STRIDE_ATTRIBUTE: &str = "cluster_stride";

/// GPU subgroup-reduce operation.
pub trait SubgroupReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to reduce.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the built-in reduction operation kind.
    fn operation_kind(&self) -> AllReduceOperationKind {
        required_attribute(self, OP_ATTRIBUTE, "gpu.subgroup_reduce")
            .cast::<AllReduceOperationKindAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `op` attribute in `gpu.subgroup_reduce`")
    }

    /// Returns `true` if the reduction is marked uniform.
    fn is_uniform(&self) -> bool {
        optional_unit_attribute(self, UNIFORM_ATTRIBUTE)
    }

    /// Returns the optional subgroup cluster size.
    fn cluster_size(&self) -> Option<IntegerAttributeRef<'c, 't>> {
        optional_integer_attribute(self, CLUSTER_SIZE_ATTRIBUTE)
    }

    /// Returns the subgroup cluster stride.
    fn cluster_stride(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, CLUSTER_STRIDE_ATTRIBUTE, "gpu.subgroup_reduce")
    }
}

mlir_op!(SubgroupReduce);
mlir_op_trait!(SubgroupReduce, OneOperand);
mlir_op_trait!(SubgroupReduce, OneResult);
mlir_op_trait!(SubgroupReduce, ZeroRegions);
mlir_op_trait!(SubgroupReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupReduceOperation`] at the specified [`Location`].
pub fn subgroup_reduce<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    operation_kind: AllReduceOperationKind,
    is_uniform: bool,
    cluster_size: Option<u32>,
    cluster_stride: u32,
    location: L,
) -> DetachedSubgroupReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.subgroup_reduce", location)
        .add_operand(value)
        .add_attribute(OP_ATTRIBUTE, context.gpu_all_reduce_operation_kind_attribute(operation_kind))
        .add_attribute(
            CLUSTER_STRIDE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), cluster_stride as i64),
        )
        .add_result(value.r#type());
    if is_uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(cluster_size) = cluster_size {
        builder = builder.add_attribute(
            CLUSTER_SIZE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), cluster_size as i64),
        );
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::subgroup_reduce`")
}

/// Name of the [`Attribute`] that stores GPU shuffle mode.
pub const MODE_ATTRIBUTE: &str = "mode";

/// GPU subgroup shuffle operation.
pub trait ShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to shuffle.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the shuffle offset operand.
    fn offset(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the shuffle width operand.
    fn width(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns the shuffle mode.
    fn mode(&self) -> ShuffleMode {
        required_attribute(self, MODE_ATTRIBUTE, "gpu.shuffle")
            .cast::<ShuffleModeAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `mode` attribute in `gpu.shuffle`")
    }

    /// Returns the shuffled value result.
    fn shuffled_value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the validity flag result.
    fn valid(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }
}

mlir_op!(Shuffle);
mlir_op_trait!(Shuffle, ZeroRegions);
mlir_op_trait!(Shuffle, ZeroSuccessors);

/// Constructs a new detached/owned [`ShuffleOperation`] at the specified [`Location`].
pub fn shuffle<
    'value,
    'offset,
    'width,
    'c: 'value + 'offset + 'width,
    't: 'c,
    ValueT: Value<'value, 'c, 't>,
    Offset: Value<'offset, 'c, 't>,
    Width: Value<'width, 'c, 't>,
    L: Location<'c, 't>,
>(
    value: ValueT,
    offset: Offset,
    width: Width,
    mode: ShuffleMode,
    location: L,
) -> DetachedShuffleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.shuffle", location)
        .add_operand(value)
        .add_operand(offset)
        .add_operand(width)
        .add_attribute(MODE_ATTRIBUTE, context.gpu_shuffle_mode_attribute(mode))
        .add_result(value.r#type())
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::shuffle`")
}

/// Name of the [`Attribute`] that stores rotate offset.
pub const OFFSET_ATTRIBUTE: &str = "offset";

/// Name of the [`Attribute`] that stores rotate width.
pub const WIDTH_ATTRIBUTE: &str = "width";

/// GPU subgroup rotate operation.
pub trait RotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to rotate.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the rotation offset attribute.
    fn offset(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, OFFSET_ATTRIBUTE, "gpu.rotate")
    }

    /// Returns the rotation width attribute.
    fn width(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, WIDTH_ATTRIBUTE, "gpu.rotate")
    }

    /// Returns the rotated value result.
    fn rotated_value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the validity flag result.
    fn valid(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }
}

mlir_op!(Rotate);
mlir_op_trait!(Rotate, ZeroRegions);
mlir_op_trait!(Rotate, ZeroSuccessors);

/// Constructs a new detached/owned [`RotateOperation`] at the specified [`Location`].
pub fn rotate<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    value: V,
    offset: i32,
    width: i32,
    location: L,
) -> DetachedRotateOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.rotate", location)
        .add_operand(value)
        .add_attribute(OFFSET_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), offset as i64))
        .add_attribute(WIDTH_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), width as i64))
        .add_result(value.r#type())
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::rotate`")
}

/// Name of the [`Attribute`] that stores GPU barrier memory fence address spaces.
pub const ADDRESS_SPACES_ATTRIBUTE: &str = "address_spaces";

/// GPU workgroup barrier operation.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional memory fence address spaces.
    fn address_spaces(&self) -> Option<Vec<AddressSpace>> {
        self.attribute(ADDRESS_SPACES_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ArrayAttributeRef>())
            .map(|attribute| {
                attribute
                    .elements()
                    .map(|element| {
                        element
                            .cast::<AddressSpaceAttributeRef>()
                            .map(|attribute| attribute.value())
                            .expect("invalid address space in `gpu.barrier`")
                    })
                    .collect()
            })
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroOperands);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Constructs a new detached/owned [`BarrierOperation`] at the specified [`Location`].
pub fn barrier<'c, 't: 'c, L: Location<'c, 't>>(
    address_spaces: Option<&[AddressSpace]>,
    location: L,
) -> DetachedBarrierOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.barrier", location);
    if let Some(address_spaces) = address_spaces {
        let address_spaces = address_spaces
            .iter()
            .map(|address_space| context.gpu_address_space_attribute(*address_space).as_ref())
            .collect::<Vec<_>>();
        builder = builder.add_attribute(ADDRESS_SPACES_ATTRIBUTE, context.array_attribute(address_spaces.as_slice()));
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::barrier`")
}

/// Name of the [`Attribute`] that stores GPU binary target attributes.
pub const TARGETS_ATTRIBUTE: &str = "targets";

/// Name of the [`Attribute`] that stores GPU offloading handlers.
pub const OFFLOADING_HANDLER_ATTRIBUTE: &str = "offloadingHandler";

/// GPU module operation containing code intended to run on a GPU.
pub trait ModuleOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> + Symbol<'o, 'c, 't> + SymbolTable<'o, 'c, 't>
{
    /// Returns optional target attributes.
    fn targets(&self) -> Option<ArrayAttributeRef<'c, 't>> {
        self.attribute(TARGETS_ATTRIBUTE).and_then(|attribute| attribute.cast())
    }

    /// Returns the optional offloading handler attribute.
    fn offloading_handler(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(OFFLOADING_HANDLER_ATTRIBUTE)
    }

    /// Returns the module body region.
    fn body_region(&self) -> RegionRef<'o, 'c, 't> {
        self.region(0).unwrap()
    }
}

mlir_op!(Module);
mlir_op_trait!(Module, HasOnlyGraphRegion);
mlir_op_trait!(Module, IsolatedFromAbove);
mlir_op_trait!(Module, NoRegionArguments);
mlir_op_trait!(Module, NoTerminator);
mlir_op_trait!(Module, OneRegion);
mlir_op_trait!(Module, SingleBlock);
mlir_op_trait!(Module, SingleBlockRegions);
mlir_op_trait!(Module, Symbol);
mlir_op_trait!(Module, SymbolTable);
mlir_op_trait!(Module, ZeroOperands);
mlir_op_trait!(Module, ZeroSuccessors);

/// Constructs a new detached/owned [`ModuleOperation`] at the specified [`Location`].
pub fn module<'c, 't: 'c, L: Location<'c, 't>>(
    name: &str,
    targets: Option<ArrayAttributeRef<'c, 't>>,
    offloading_handler: Option<AttributeRef<'c, 't>>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedModuleOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.module", location)
        .add_attribute(crate::SYMBOL_NAME_ATTRIBUTE, context.string_attribute(name));
    if let Some(targets) = targets {
        builder = builder.add_attribute(TARGETS_ATTRIBUTE, targets);
    }
    if let Some(offloading_handler) = offloading_handler {
        builder = builder.add_attribute(OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    }
    builder
        .add_region(body)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::module`")
}

/// Name of the [`Attribute`] that stores GPU binary objects.
pub const OBJECTS_ATTRIBUTE: &str = "objects";

/// GPU binary operation storing serialized GPU object attributes.
pub trait BinaryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + Symbol<'o, 'c, 't> {
    /// Returns the optional offloading handler attribute.
    fn offloading_handler(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(OFFLOADING_HANDLER_ATTRIBUTE)
    }

    /// Returns the GPU object attribute array.
    fn objects(&self) -> ArrayAttributeRef<'c, 't> {
        required_attribute(self, OBJECTS_ATTRIBUTE, "gpu.binary")
            .cast()
            .expect("invalid `objects` attribute in `gpu.binary`")
    }
}

mlir_op!(Binary);
mlir_op_trait!(Binary, Symbol);
mlir_op_trait!(Binary, ZeroOperands);
mlir_op_trait!(Binary, ZeroRegions);
mlir_op_trait!(Binary, ZeroSuccessors);

/// Constructs a new detached/owned [`BinaryOperation`] at the specified [`Location`].
pub fn binary<'c, 't: 'c, L: Location<'c, 't>>(
    name: &str,
    objects: ArrayAttributeRef<'c, 't>,
    offloading_handler: Option<AttributeRef<'c, 't>>,
    location: L,
) -> DetachedBinaryOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.binary", location)
        .add_attribute(crate::SYMBOL_NAME_ATTRIBUTE, context.string_attribute(name))
        .add_attribute(OBJECTS_ATTRIBUTE, objects);
    if let Some(offloading_handler) = offloading_handler {
        builder = builder.add_attribute(OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    }
    builder
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::binary`")
}

macro_rules! gpu_one_memref_operand_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $method:ident, $doc:literal $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                #[doc = "Returns the memref operand of this operation."]
                fn $method(&self) -> ValueRef<'o, 'c, 't> {
                    self.operand_value(0).unwrap()
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached/owned [`"]
            #[doc = stringify!($name)]
            #[doc = "Operation`] at the specified [`Location`]."]
            pub fn $function_name<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
                value: V,
                location: L,
            ) -> [<Detached $name Operation>]<'c, 't> {
                location.context().load_dialect(DialectHandle::gpu());
                OperationBuilder::new($operation_name, location)
                    .add_operand(value)
                    .build()
                    .and_then(|operation| unsafe { operation.cast() })
                    .expect(concat!("invalid arguments to `gpu::", stringify!($function_name), "`"))
            }
        }
    };
}

gpu_one_memref_operand_operation!(
    HostRegister,
    host_register,
    "gpu.host_register",
    value,
    "GPU operation that registers a host memref for device access.",
);

gpu_one_memref_operand_operation!(
    HostUnregister,
    host_unregister,
    "gpu.host_unregister",
    value,
    "GPU operation that unregisters a host memref from device access.",
);

/// GPU wait operation for async token dependencies.
pub trait WaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(Wait);
mlir_op_trait!(Wait, ZeroRegions);
mlir_op_trait!(Wait, ZeroSuccessors);

/// Constructs a new detached/owned [`WaitOperation`] at the specified [`Location`].
pub fn wait<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[V],
    is_async: bool,
    location: L,
) -> DetachedWaitOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    add_optional_async_result(OperationBuilder::new("gpu.wait", location).add_operands(async_dependencies), is_async)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::wait`")
}

/// Name of the [`Attribute`] that marks host-shared GPU allocation.
pub const HOST_SHARED_ATTRIBUTE: &str = "hostShared";

/// GPU allocation operation.
pub trait AllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 0)
    }

    /// Returns dynamic size operands.
    fn dynamic_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 1)
    }

    /// Returns symbol operands.
    fn symbol_operands(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 2)
    }

    /// Returns `true` if the allocation is host shared.
    fn host_shared(&self) -> bool {
        optional_unit_attribute(self, HOST_SHARED_ATTRIBUTE)
    }

    /// Returns the allocated memref result.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(Alloc);
mlir_op_trait!(Alloc, ZeroRegions);
mlir_op_trait!(Alloc, ZeroSuccessors);

/// Constructs a new detached/owned [`AllocOperation`] at the specified [`Location`].
pub fn alloc<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    dynamic_sizes: &[ValueRef<'o, 'c, 't>],
    symbol_operands: &[ValueRef<'o, 'c, 't>],
    memref_type: T,
    host_shared: bool,
    is_async: bool,
    location: L,
) -> DetachedAllocOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu());
    let mut builder = OperationBuilder::new("gpu.alloc", location)
        .add_operands(async_dependencies)
        .add_operands(dynamic_sizes)
        .add_operands(symbol_operands)
        .add_result(memref_type);
    if host_shared {
        builder = builder.add_attribute(HOST_SHARED_ATTRIBUTE, context.unit_attribute());
    }
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type());
    }
    add_operand_segments(builder, &[async_dependencies.len(), dynamic_sizes.len(), symbol_operands.len()])
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::alloc`")
}

macro_rules! gpu_async_prefix_operation {
    (
        $name:ident,
        $function_name:ident,
        $operation_name:literal,
        $doc:literal,
        operands = { $($method:ident => $index:expr),+ $(,)* } $(,)*
    ) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns async token dependencies.
                fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
                    leading_async_dependencies(self)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> ValueRef<'o, 'c, 't> {
                        self.operand_value(leading_async_dependency_count(self) + $index).unwrap()
                    }
                )+

                /// Returns the optional async token result.
                fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
                    optional_async_token(self)
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);
        }
    };
}

gpu_async_prefix_operation!(
    Dealloc,
    dealloc,
    "gpu.dealloc",
    "GPU operation that deallocates GPU memory.",
    operands = { memref => 0 },
);

gpu_async_prefix_operation!(
    Memcpy,
    memcpy,
    "gpu.memcpy",
    "GPU operation that copies between memrefs.",
    operands = { destination => 0, source => 1 },
);

gpu_async_prefix_operation!(
    Memset,
    memset,
    "gpu.memset",
    "GPU operation that sets a memref to a scalar value.",
    operands = { destination => 0, value => 1 },
);

/// Constructs a new detached/owned [`DeallocOperation`] at the specified [`Location`].
pub fn dealloc<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    memref: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedDeallocOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    add_optional_async_result(
        OperationBuilder::new("gpu.dealloc", location).add_operands(async_dependencies).add_operand(memref),
        is_async,
    )
    .build()
    .and_then(|operation| unsafe { operation.cast() })
    .expect("invalid arguments to `gpu::dealloc`")
}

/// Constructs a new detached/owned [`MemcpyOperation`] at the specified [`Location`].
pub fn memcpy<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedMemcpyOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    add_optional_async_result(
        OperationBuilder::new("gpu.memcpy", location)
            .add_operands(async_dependencies)
            .add_operand(destination)
            .add_operand(source),
        is_async,
    )
    .build()
    .and_then(|operation| unsafe { operation.cast() })
    .expect("invalid arguments to `gpu::memcpy`")
}

/// Constructs a new detached/owned [`MemsetOperation`] at the specified [`Location`].
pub fn memset<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    value: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> DetachedMemsetOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    add_optional_async_result(
        OperationBuilder::new("gpu.memset", location)
            .add_operands(async_dependencies)
            .add_operand(destination)
            .add_operand(value),
        is_async,
    )
    .build()
    .and_then(|operation| unsafe { operation.cast() })
    .expect("invalid arguments to `gpu::memset`")
}

/// GPU operation that sets the default device index.
pub trait SetDefaultDeviceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the device index operand.
    fn device_index(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }
}

mlir_op!(SetDefaultDevice);
mlir_op_trait!(SetDefaultDevice, ZeroRegions);
mlir_op_trait!(SetDefaultDevice, ZeroSuccessors);

/// Constructs a new detached/owned [`SetDefaultDeviceOperation`] at the specified [`Location`].
pub fn set_default_device<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    device_index: V,
    location: L,
) -> DetachedSetDefaultDeviceOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.set_default_device", location)
        .add_operand(device_index)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::set_default_device`")
}

/// Name of the [`Attribute`] that stores MMA leading dimensions.
pub const LEAD_DIMENSION_ATTRIBUTE: &str = "leadDimension";

/// Name of the [`Attribute`] that marks transposed MMA matrix loads and stores.
pub const TRANSPOSE_ATTRIBUTE: &str = "transpose";

/// GPU subgroup MMA matrix load operation.
pub trait SubgroupMmaLoadMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the leading dimension attribute.
    fn lead_dimension(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, LEAD_DIMENSION_ATTRIBUTE, "gpu.subgroup_mma_load_matrix")
    }

    /// Returns `true` if the load is transposed.
    fn transpose(&self) -> bool {
        optional_unit_attribute(self, TRANSPOSE_ATTRIBUTE)
    }

    /// Returns the loaded MMA matrix result.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaLoadMatrix);
mlir_op_trait!(SubgroupMmaLoadMatrix, OneResult);
mlir_op_trait!(SubgroupMmaLoadMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaLoadMatrix, ZeroSuccessors);

/// GPU subgroup MMA matrix store operation.
pub trait SubgroupMmaStoreMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source MMA matrix.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination memref.
    fn destination_memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the leading dimension attribute.
    fn lead_dimension(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, LEAD_DIMENSION_ATTRIBUTE, "gpu.subgroup_mma_store_matrix")
    }

    /// Returns `true` if the store is transposed.
    fn transpose(&self) -> bool {
        optional_unit_attribute(self, TRANSPOSE_ATTRIBUTE)
    }
}

mlir_op!(SubgroupMmaStoreMatrix);
mlir_op_trait!(SubgroupMmaStoreMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaStoreMatrix, ZeroSuccessors);

/// Name of the [`Attribute`] that marks transposed MMA matrix A operands.
pub const A_TRANSPOSE_ATTRIBUTE: &str = "a_transpose";

/// Name of the [`Attribute`] that marks transposed MMA matrix B operands.
pub const B_TRANSPOSE_ATTRIBUTE: &str = "b_transpose";

/// GPU subgroup MMA matrix multiply-accumulate operation.
pub trait SubgroupMmaComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the A operand matrix.
    fn a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the B operand matrix.
    fn b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the C accumulator matrix.
    fn c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(2).unwrap()
    }

    /// Returns `true` if operand A is transposed.
    fn a_transpose(&self) -> bool {
        optional_unit_attribute(self, A_TRANSPOSE_ATTRIBUTE)
    }

    /// Returns `true` if operand B is transposed.
    fn b_transpose(&self) -> bool {
        optional_unit_attribute(self, B_TRANSPOSE_ATTRIBUTE)
    }

    /// Returns the result matrix.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaCompute);
mlir_op_trait!(SubgroupMmaCompute, OneResult);
mlir_op_trait!(SubgroupMmaCompute, ZeroRegions);
mlir_op_trait!(SubgroupMmaCompute, ZeroSuccessors);

/// GPU subgroup MMA constant matrix operation.
pub trait SubgroupMmaConstantMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the scalar value broadcast into the matrix.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the resulting MMA matrix.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaConstantMatrix);
mlir_op_trait!(SubgroupMmaConstantMatrix, OneOperand);
mlir_op_trait!(SubgroupMmaConstantMatrix, OneResult);
mlir_op_trait!(SubgroupMmaConstantMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaConstantMatrix, ZeroSuccessors);

/// GPU subgroup MMA thread-local extract operation.
pub trait SubgroupMmaExtractThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source MMA matrix.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the thread-local index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the extracted scalar.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaExtractThreadLocal);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, OneResult);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, ZeroRegions);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, ZeroSuccessors);

/// GPU subgroup MMA thread-local insert operation.
pub trait SubgroupMmaInsertThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the scalar value to insert.
    fn value(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the destination MMA matrix.
    fn matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(1).unwrap()
    }

    /// Returns the thread-local index operands.
    fn indices(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the resulting MMA matrix.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaInsertThreadLocal);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, OneResult);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, ZeroRegions);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, ZeroSuccessors);

/// Name of the [`Attribute`] that stores the MMA elementwise operation kind.
pub const OP_TYPE_ATTRIBUTE: &str = "opType";

/// GPU subgroup MMA elementwise operation.
pub trait SubgroupMmaElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the MMA matrix operands.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().collect()
    }

    /// Returns the elementwise operation kind.
    fn operation(&self) -> MmaElementwiseOperation {
        required_attribute(self, OP_TYPE_ATTRIBUTE, "gpu.subgroup_mma_elementwise")
            .cast::<MmaElementwiseOperationAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `opType` attribute in `gpu.subgroup_mma_elementwise`")
    }

    /// Returns the resulting MMA matrix.
    fn result_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupMmaElementwise);
mlir_op_trait!(SubgroupMmaElementwise, OneResult);
mlir_op_trait!(SubgroupMmaElementwise, ZeroRegions);
mlir_op_trait!(SubgroupMmaElementwise, ZeroSuccessors);

macro_rules! gpu_sparse_async_operation {
    ($name:ident, $doc:literal, operands = { $($method:ident => $index:expr),* $(,)* } $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns async token dependencies.
                fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
                    leading_async_dependencies(self)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> ValueRef<'o, 'c, 't> {
                        self.operand_value(leading_async_dependency_count(self) + $index).unwrap()
                    }
                )*

                /// Returns the optional async token result.
                fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
                    if self.result_count() > 1 {
                        self.result(self.result_count() - 1).map(|result| result.as_ref())
                    } else {
                        optional_async_token(self)
                    }
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);
        }
    };
}

macro_rules! gpu_sparse_create_sp_mat_operation {
    ($name:ident, $doc:literal, operands = { $($method:ident => $index:expr),+ $(,)* } $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns async token dependencies.
                fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
                    leading_async_dependencies(self)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> ValueRef<'o, 'c, 't> {
                        self.operand_value(leading_async_dependency_count(self) + $index).unwrap()
                    }
                )+

                /// Returns the sparse matrix handle result.
                fn sparse_matrix(&self) -> ValueRef<'o, 'c, 't> {
                    self.result(0).unwrap().as_ref()
                }

                /// Returns the optional async token result.
                fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
                    self.result(1).map(|result| result.as_ref())
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);
        }
    };
}

/// GPU operation that creates a dense tensor sparse handle.
pub trait CreateDnTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 0)
    }

    /// Returns the dense tensor backing memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        operand_segment(self, 1)[0]
    }

    /// Returns dense tensor dimension operands.
    fn dimensions(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 2)
    }

    /// Returns the dense tensor handle result.
    fn dense_tensor(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(CreateDnTensor);
mlir_op_trait!(CreateDnTensor, ZeroRegions);
mlir_op_trait!(CreateDnTensor, ZeroSuccessors);

gpu_sparse_async_operation!(
    DestroyDnTensor,
    "GPU operation that destroys a dense tensor sparse handle.",
    operands = { dense_tensor => 0 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCoo,
    "GPU operation that creates a sparse matrix in COO format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, row_indices => 3, column_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCooAos,
    "GPU operation that creates a sparse matrix in COO AoS format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, indices => 3, values => 4 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCsr,
    "GPU operation that creates a sparse matrix in CSR format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, row_positions => 3, column_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCsc,
    "GPU operation that creates a sparse matrix in CSC format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, column_positions => 3, row_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateBsr,
    "GPU operation that creates a sparse matrix in BSR format.",
    operands = {
        block_rows => 0,
        block_columns => 1,
        block_non_zero_count => 2,
        row_block_size => 3,
        column_block_size => 4,
        block_row_positions => 5,
        block_column_indices => 6,
        values => 7,
    },
);

/// Name of the [`Attribute`] that stores the 2-to-4 sparse matrix pruning flag.
pub const PRUNE_FLAG_ATTRIBUTE: &str = "pruneFlag";

/// GPU operation that creates a sparse matrix with 2-to-4 sparsity.
pub trait Create2To4SpMatOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns the row count operand.
    fn rows(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns the column count operand.
    fn columns(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns the pruning flag.
    fn prune_flag(&self) -> Prune2To4SparseMatrixFlag {
        required_attribute(self, PRUNE_FLAG_ATTRIBUTE, "gpu.create_2to4_spmat")
            .cast::<Prune2To4SparseMatrixFlagAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `pruneFlag` attribute in `gpu.create_2to4_spmat`")
    }

    /// Returns the dense backing memref.
    fn memref(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the sparse matrix result.
    fn sparse_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(Create2To4SpMat);
mlir_op_trait!(Create2To4SpMat, ZeroRegions);
mlir_op_trait!(Create2To4SpMat, ZeroSuccessors);

gpu_sparse_async_operation!(
    DestroySpMat,
    "GPU operation that destroys a sparse matrix handle.",
    operands = { sparse_matrix => 0 },
);

/// Name of the [`Attribute`] that stores the sparse matrix A transpose mode.
pub const MODE_A_ATTRIBUTE: &str = "modeA";

/// Name of the [`Attribute`] that stores the sparse matrix B transpose mode.
pub const MODE_B_ATTRIBUTE: &str = "modeB";

/// Name of the [`Attribute`] that stores sparse operation compute type.
pub const COMPUTE_TYPE_ATTRIBUTE: &str = "computeType";

/// GPU SpMV buffer-size operation.
pub trait SpmvBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spmv_buffer_size")
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns dense tensor X.
    fn dense_tensor_x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns dense tensor Y.
    fn dense_tensor_y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spmv_buffer_size")
    }

    /// Returns the buffer-size result.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpmvBufferSize);
mlir_op_trait!(SpmvBufferSize, ZeroRegions);
mlir_op_trait!(SpmvBufferSize, ZeroSuccessors);

/// GPU SpMV compute operation.
pub trait SpmvOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spmv")
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns dense tensor X.
    fn dense_tensor_x(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns dense tensor Y.
    fn dense_tensor_y(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the temporary buffer.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 3).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spmv")
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(Spmv);
mlir_op_trait!(Spmv, ZeroRegions);
mlir_op_trait!(Spmv, ZeroSuccessors);

/// Name of the MLIR attribute storing operation result segment sizes.
pub const RESULT_SEGMENT_SIZES_ATTRIBUTE: &str = "result_segment_sizes";

/// GPU SpMM buffer-size operation.
pub trait SpmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spmm_buffer_size")
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.spmm_buffer_size")
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns dense matrix C.
    fn dense_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spmm_buffer_size")
    }

    /// Returns the buffer-size results.
    fn buffer_sizes(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        result_segment(self, 0)
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        result_segment(self, 1).first().copied()
    }
}

mlir_op!(SpmmBufferSize);
mlir_op_trait!(SpmmBufferSize, ZeroRegions);
mlir_op_trait!(SpmmBufferSize, ZeroSuccessors);

/// GPU SpMM compute operation.
pub trait SpmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 0)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spmm")
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.spmm")
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        operand_segment(self, 1)[0]
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        operand_segment(self, 2)[0]
    }

    /// Returns dense matrix C.
    fn dense_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        operand_segment(self, 3)[0]
    }

    /// Returns temporary buffers.
    fn buffers(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        operand_segment(self, 4)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spmm")
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(Spmm);
mlir_op_trait!(Spmm, ZeroRegions);
mlir_op_trait!(Spmm, ZeroSuccessors);

/// GPU SDDMM buffer-size operation.
pub trait SddmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns dense matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.sddmm_buffer_size")
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.sddmm_buffer_size")
    }

    /// Returns dense matrix A.
    fn dense_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns sparse matrix C.
    fn sparse_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.sddmm_buffer_size")
    }

    /// Returns the buffer-size result.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SddmmBufferSize);
mlir_op_trait!(SddmmBufferSize, ZeroRegions);
mlir_op_trait!(SddmmBufferSize, ZeroSuccessors);

/// GPU SDDMM compute operation.
pub trait SddmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns dense matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.sddmm")
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.sddmm")
    }

    /// Returns dense matrix A.
    fn dense_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns sparse matrix C.
    fn sparse_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the temporary buffer.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 3).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.sddmm")
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(Sddmm);
mlir_op_trait!(Sddmm, ZeroRegions);
mlir_op_trait!(Sddmm, ZeroSuccessors);

/// GPU operation that creates a SpGEMM descriptor.
pub trait SpGemmCreateDescrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns the SpGEMM descriptor result.
    fn descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpGemmCreateDescr);
mlir_op_trait!(SpGemmCreateDescr, ZeroRegions);
mlir_op_trait!(SpGemmCreateDescr, ZeroSuccessors);

gpu_sparse_async_operation!(
    SpGemmDestroyDescr,
    "GPU operation that destroys a SpGEMM descriptor.",
    operands = { descriptor => 0 },
);

/// Name of the [`Attribute`] that stores SpGEMM work kind.
pub const KIND_ATTRIBUTE: &str = "kind";

/// GPU SpGEMM work-estimation or compute operation.
pub trait SpGemmWorkEstimationOrComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns the SpGEMM descriptor operand.
    fn descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spgemm_work_estimation_or_compute")
    }

    /// Returns sparse matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.spgemm_work_estimation_or_compute")
    }

    /// Returns the sparse matrix A operand.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns the sparse matrix B operand.
    fn sparse_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the sparse matrix C operand.
    fn sparse_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 3).unwrap()
    }

    /// Returns the current buffer-size operand.
    fn buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 4).unwrap()
    }

    /// Returns the temporary buffer operand.
    fn buffer(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 5).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spgemm_work_estimation_or_compute")
    }

    /// Returns the SpGEMM work kind.
    fn kind(&self) -> SpGemmWorkKind {
        required_attribute(self, KIND_ATTRIBUTE, "gpu.spgemm_work_estimation_or_compute")
            .cast::<SpGemmWorkKindAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `kind` attribute in `gpu.spgemm_work_estimation_or_compute`")
    }

    /// Returns the new buffer-size result.
    fn new_buffer_size(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(1).map(|result| result.as_ref())
    }
}

mlir_op!(SpGemmWorkEstimationOrCompute);
mlir_op_trait!(SpGemmWorkEstimationOrCompute, ZeroRegions);
mlir_op_trait!(SpGemmWorkEstimationOrCompute, ZeroSuccessors);

/// GPU SpGEMM copy operation.
pub trait SpGemmCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_A_ATTRIBUTE, "gpu.spgemm_copy")
    }

    /// Returns sparse matrix B transpose mode.
    fn mode_b(&self) -> MatrixTransposeMode {
        matrix_transpose_mode(self, MODE_B_ATTRIBUTE, "gpu.spgemm_copy")
    }

    /// Returns the SpGEMM descriptor operand.
    fn descriptor(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns the sparse matrix A operand.
    fn sparse_matrix_a(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 1).unwrap()
    }

    /// Returns the sparse matrix B operand.
    fn sparse_matrix_b(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 2).unwrap()
    }

    /// Returns the sparse matrix C operand.
    fn sparse_matrix_c(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self) + 3).unwrap()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> TypeRef<'c, 't> {
        type_attribute(self, COMPUTE_TYPE_ATTRIBUTE, "gpu.spgemm_copy")
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        optional_async_token(self)
    }
}

mlir_op!(SpGemmCopy);
mlir_op_trait!(SpGemmCopy, ZeroRegions);
mlir_op_trait!(SpGemmCopy, ZeroSuccessors);

/// GPU sparse matrix get-size operation.
pub trait SpMatGetSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        leading_async_dependencies(self)
    }

    /// Returns the sparse matrix operand.
    fn sparse_matrix(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(leading_async_dependency_count(self)).unwrap()
    }

    /// Returns the sparse matrix row-count result.
    fn rows(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }

    /// Returns the sparse matrix column-count result.
    fn columns(&self) -> ValueRef<'o, 'c, 't> {
        self.result(1).unwrap().as_ref()
    }

    /// Returns the sparse matrix non-zero-count result.
    fn non_zero_count(&self) -> ValueRef<'o, 'c, 't> {
        self.result(2).unwrap().as_ref()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.result(3).map(|result| result.as_ref())
    }
}

mlir_op!(SpMatGetSize);
mlir_op_trait!(SpMatGetSize, ZeroRegions);
mlir_op_trait!(SpMatGetSize, ZeroSuccessors);

gpu_sparse_async_operation!(
    SetCsrPointers,
    "GPU operation that sets CSR pointers for a sparse matrix.",
    operands = { sparse_matrix => 0, positions => 1, coordinates => 2, values => 3 },
);

/// Name of the [`Attribute`] that stores warp size.
pub const WARP_SIZE_ATTRIBUTE: &str = "warp_size";

/// GPU operation that bridges vector code and SIMT execution by running a region on lane 0.
pub trait WarpExecuteOnLane0Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the lane identifier operand.
    fn lane_id(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the warp size attribute.
    fn warp_size(&self) -> IntegerAttributeRef<'c, 't> {
        integer_attribute(self, WARP_SIZE_ATTRIBUTE, "gpu.warp_execute_on_lane_0")
    }

    /// Returns operands passed into the lane-0 region.
    fn arguments(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.operand_values().skip(1).collect()
    }

    /// Returns values produced for the surrounding SIMT region.
    fn outputs(&self) -> Vec<ValueRef<'o, 'c, 't>> {
        self.results().map(|result| result.as_ref()).collect()
    }

    /// Returns the lane-0 region.
    fn region(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(WarpExecuteOnLane0);
mlir_op_trait!(WarpExecuteOnLane0, OneRegion);
mlir_op_trait!(WarpExecuteOnLane0, SingleBlockRegions);
mlir_op_trait!(WarpExecuteOnLane0, ZeroSuccessors);

/// Name of the [`Attribute`] that stores subgroup broadcast type.
pub const BROADCAST_TYPE_ATTRIBUTE: &str = "broadcast_type";

/// GPU subgroup broadcast operation.
pub trait SubgroupBroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn source(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the optional lane operand.
    fn lane(&self) -> Option<ValueRef<'o, 'c, 't>> {
        self.operand_value(1)
    }

    /// Returns the broadcast type.
    fn broadcast_type(&self) -> BroadcastType {
        required_attribute(self, BROADCAST_TYPE_ATTRIBUTE, "gpu.subgroup_broadcast")
            .cast::<BroadcastTypeAttributeRef>()
            .map(|attribute| attribute.value())
            .expect("invalid `broadcast_type` attribute in `gpu.subgroup_broadcast`")
    }

    /// Returns the broadcast result.
    fn output(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(SubgroupBroadcast);
mlir_op_trait!(SubgroupBroadcast, OneResult);
mlir_op_trait!(SubgroupBroadcast, ZeroRegions);
mlir_op_trait!(SubgroupBroadcast, ZeroSuccessors);

/// GPU subgroup ballot operation.
pub trait BallotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the predicate operand.
    fn predicate(&self) -> ValueRef<'o, 'c, 't> {
        self.operand_value(0).unwrap()
    }

    /// Returns the ballot mask result.
    fn mask(&self) -> ValueRef<'o, 'c, 't> {
        self.result(0).unwrap().as_ref()
    }
}

mlir_op!(Ballot);
mlir_op_trait!(Ballot, OneOperand);
mlir_op_trait!(Ballot, OneResult);
mlir_op_trait!(Ballot, ZeroRegions);
mlir_op_trait!(Ballot, ZeroSuccessors);

/// Constructs a new detached/owned [`BallotOperation`] at the specified [`Location`].
pub fn ballot<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    predicate: V,
    result_type: T,
    location: L,
) -> DetachedBallotOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::gpu());
    OperationBuilder::new("gpu.ballot", location)
        .add_operand(predicate)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `gpu::ballot`")
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::gpu::attributes::ObjectFormat;
    use crate::dialects::gpu::types::MmaMatrixOperand;
    use crate::{Attribute, Block, Context, DetachedOp, OneResult, Operation, OperationBuilder, Region, Size, Type};

    use super::*;

    fn build_detached_operation<'c, 't: 'c, O: DetachedOp<'c, 'c, 't>>(builder: OperationBuilder<'c, 't>) -> O {
        builder.build().and_then(|operation| unsafe { operation.cast::<O>() }).unwrap()
    }

    macro_rules! gpu_dimension_operation_test {
        ($test_name:ident, $function_name:ident, $operation_name:literal, $dimension:expr $(,)?) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let operation = $function_name($dimension, Some(128), location);

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.dimension(), $dimension);
                assert_eq!(operation.upper_bound().map(|attribute| attribute.signless_value()), Some(128));
                assert_eq!(operation.output_type(), context.index_type());
            }
        };
    }

    macro_rules! gpu_upper_bound_index_operation_test {
        ($test_name:ident, $function_name:ident, $operation_name:literal $(,)?) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let operation = $function_name(Some(128), location);

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.upper_bound().map(|attribute| attribute.signless_value()), Some(128));
                assert_eq!(operation.output_type(), context.index_type());
            }
        };
    }

    macro_rules! gpu_sparse_async_operation_test {
        (
            $test_name:ident,
            $operation_type:ident,
            $operation_name:literal,
            operand_count = $operand_count:expr,
            operands = { $($method:ident => $index:expr),* $(,)* } $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let token_type = context.gpu_async_token_type().as_ref();
                let index_type = context.index_type().as_ref();
                let mut arguments = vec![(token_type, location)];
                arguments.extend((0..$operand_count).map(|_| (index_type, location)));
                let block = context.block(arguments.as_slice());
                let token = block.argument(0).unwrap();
                let operands = (1..=$operand_count).map(|index| block.argument(index).unwrap()).collect::<Vec<_>>();
                let operation: $operation_type<'_, '_> = build_detached_operation(
                    OperationBuilder::new($operation_name, location)
                        .add_operand(token)
                        .add_operands(operands.as_slice())
                        .add_result(context.gpu_async_token_type()),
                );

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.async_dependencies(), vec![token]);
                $(assert_eq!(operation.$method(), operands[$index]);)*
                assert!(operation.async_token().is_some());
            }
        };
    }

    macro_rules! gpu_sparse_create_sp_mat_operation_test {
        (
            $test_name:ident,
            $operation_type:ident,
            $operation_name:literal,
            operand_count = $operand_count:expr,
            operands = { $($method:ident => $index:expr),+ $(,)* } $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let token_type = context.gpu_async_token_type().as_ref();
                let index_type = context.index_type().as_ref();
                let mut arguments = vec![(token_type, location)];
                arguments.extend((0..$operand_count).map(|_| (index_type, location)));
                let block = context.block(arguments.as_slice());
                let token = block.argument(0).unwrap();
                let operands = (1..=$operand_count).map(|index| block.argument(index).unwrap()).collect::<Vec<_>>();
                let operation: $operation_type<'_, '_> = build_detached_operation(
                    OperationBuilder::new($operation_name, location)
                        .add_operand(token)
                        .add_operands(operands.as_slice())
                        .add_result(context.gpu_sparse_sp_mat_handle_type())
                        .add_result(context.gpu_async_token_type()),
                );

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.async_dependencies(), vec![token]);
                $(assert_eq!(operation.$method(), operands[$index]);)+
                assert_eq!(operation.sparse_matrix().r#type(), context.gpu_sparse_sp_mat_handle_type());
                assert!(operation.async_token().is_some());
            }
        };
    }

    gpu_dimension_operation_test!(test_cluster_dim_operation, cluster_dim, "gpu.cluster_dim", Dimension::X);
    gpu_dimension_operation_test!(
        test_cluster_dim_blocks_operation,
        cluster_dim_blocks,
        "gpu.cluster_dim_blocks",
        Dimension::Y,
    );
    gpu_dimension_operation_test!(test_cluster_id_operation, cluster_id, "gpu.cluster_id", Dimension::Z);
    gpu_dimension_operation_test!(
        test_cluster_block_id_operation,
        cluster_block_id,
        "gpu.cluster_block_id",
        Dimension::X
    );
    gpu_dimension_operation_test!(test_block_dim_operation, block_dim, "gpu.block_dim", Dimension::Y);
    gpu_dimension_operation_test!(test_block_id_operation, block_id, "gpu.block_id", Dimension::Z);
    gpu_dimension_operation_test!(test_grid_dim_operation, grid_dim, "gpu.grid_dim", Dimension::X);
    gpu_dimension_operation_test!(test_thread_id_operation, thread_id, "gpu.thread_id", Dimension::Y);
    gpu_dimension_operation_test!(test_global_id_operation, global_id, "gpu.global_id", Dimension::Z);

    gpu_upper_bound_index_operation_test!(test_lane_id_operation, lane_id, "gpu.lane_id");
    gpu_upper_bound_index_operation_test!(test_subgroup_id_operation, subgroup_id, "gpu.subgroup_id");
    gpu_upper_bound_index_operation_test!(test_num_subgroups_operation, num_subgroups, "gpu.num_subgroups");
    gpu_upper_bound_index_operation_test!(test_subgroup_size_operation, subgroup_size, "gpu.subgroup_size");

    #[test]
    fn test_func_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        context.load_dialect(DialectHandle::gpu());
        let operation: DetachedFuncOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.func", location)
                .add_attribute(KERNEL_ATTRIBUTE, context.unit_attribute())
                .add_attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&[1, 2, 3]).unwrap())
                .add_attribute(KNOWN_GRID_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&[4, 5, 6]).unwrap())
                .add_attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&[7, 8, 9]).unwrap())
                .add_attribute(
                    WORKGROUP_ATTRIBUTIONS_ATTRIBUTE,
                    context.integer_attribute(context.signless_integer_type(64), 2),
                )
                .add_region(context.region()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.func"));
        assert!(operation.is_kernel());
        assert_eq!(operation.known_block_size(), Some(vec![1, 2, 3]));
        assert_eq!(operation.known_grid_size(), Some(vec![4, 5, 6]));
        assert_eq!(operation.known_cluster_size(), Some(vec![7, 8, 9]));
        assert_eq!(operation.workgroup_attribution_count(), 2);
        assert_eq!(operation.body().blocks().count(), 0);
    }

    #[test]
    fn test_dynamic_shared_memory_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let operation = dynamic_shared_memory(memref_type, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.dynamic_shared_memory"));
        assert_eq!(operation.memref().r#type(), memref_type);
        assert_eq!(operation.output_type(), memref_type);
    }

    #[test]
    fn test_launch_func_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let token_type = context.gpu_async_token_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type.as_ref(), location),
            (index_type.as_ref(), location),
            (index_type.as_ref(), location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let one = block.argument(1).unwrap().as_ref();
        let grid_size = Dim3 { x: one, y: one, z: one };
        let block_size = Dim3 { x: one, y: one, z: one };
        let cluster_size = Dim3 { x: one, y: one, z: one };
        let dynamic_shared_memory_size = block.argument(2).unwrap().as_ref();
        let kernel_operand = block.argument(3).unwrap().as_ref();
        let kernel = context.symbol_ref_attribute("kernels".into(), &[context.flat_symbol_ref_attribute("kernel")]);
        let operation = launch_func(
            LaunchFuncProperties {
                async_dependencies: vec![token],
                kernel,
                grid_size,
                block_size,
                cluster_size: Some(cluster_size),
                dynamic_shared_memory_size: Some(dynamic_shared_memory_size),
                kernel_operands: vec![kernel_operand],
                async_object: Some(token),
                is_async: true,
            },
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.launch_func"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.kernel(), kernel);
        assert_eq!(operation.grid_size(), grid_size);
        assert_eq!(operation.block_size(), block_size);
        assert_eq!(operation.cluster_size(), Some(cluster_size));
        assert_eq!(operation.dynamic_shared_memory_size(), Some(dynamic_shared_memory_size));
        assert_eq!(operation.kernel_operands(), vec![kernel_operand]);
        assert_eq!(operation.async_object(), Some(token));
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_launch_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let token_type = context.gpu_async_token_type().as_ref();
        let block =
            context.block(&[(token_type, location), (index_type.as_ref(), location), (index_type.as_ref(), location)]);
        let token = block.argument(0).unwrap().as_ref();
        let one = block.argument(1).unwrap().as_ref();
        let dynamic_shared_memory_size = block.argument(2).unwrap().as_ref();
        let grid_size = Dim3 { x: one, y: one, z: one };
        let block_size = Dim3 { x: one, y: one, z: one };
        let cluster_size = Dim3 { x: one, y: one, z: one };
        let module = context.flat_symbol_ref_attribute("kernels");
        let function = context.flat_symbol_ref_attribute("kernel");
        let operation = launch(
            LaunchProperties {
                async_dependencies: vec![token],
                grid_size,
                block_size,
                cluster_size: Some(cluster_size),
                dynamic_shared_memory_size: Some(dynamic_shared_memory_size),
                module: Some(module),
                function: Some(function),
                workgroup_attributions: Some(2),
                is_async: true,
            },
            context.region(),
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.launch"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.grid_size(), grid_size);
        assert_eq!(operation.block_size(), block_size);
        assert_eq!(operation.cluster_size(), Some(cluster_size));
        assert_eq!(operation.dynamic_shared_memory_size(), Some(dynamic_shared_memory_size));
        assert_eq!(operation.module_symbol(), Some(module));
        assert_eq!(operation.function_symbol(), Some(function));
        assert_eq!(operation.workgroup_attribution_count(), 2);
        assert_eq!(operation.body().blocks().count(), 0);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_printf_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location)]);
        let argument = block.argument(0).unwrap();
        let operation = printf("value: %d", &[argument], location);

        assert_eq!(operation.name().as_str(), Ok("gpu.printf"));
        assert_eq!(operation.format().as_str(), Ok("value: %d"));
        assert_eq!(operation.arguments(), vec![argument]);
    }

    #[test]
    fn test_return_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation = r#return(&[value], location);

        assert_eq!(operation.name().as_str(), Ok("gpu.return"));
        assert_eq!(operation.values(), vec![value]);
    }

    #[test]
    fn test_terminator_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = terminator(location);

        assert_eq!(operation.name().as_str(), Ok("gpu.terminator"));
        assert_eq!(operation.operand_count(), 0);
        assert_eq!(operation.result_count(), 0);
    }

    #[test]
    fn test_yield_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation = r#yield(&[value], location);

        assert_eq!(operation.name().as_str(), Ok("gpu.yield"));
        assert_eq!(operation.values(), vec![value]);
    }

    #[test]
    fn test_all_reduce_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation =
            all_reduce(value, Some(AllReduceOperationKind::Add), true, context.region(), value.r#type(), location);

        assert_eq!(operation.name().as_str(), Ok("gpu.all_reduce"));
        assert_eq!(operation.value(), value);
        assert_eq!(operation.operation_kind(), Some(AllReduceOperationKind::Add));
        assert!(operation.is_uniform());
        assert_eq!(operation.body().blocks().count(), 0);
        assert_eq!(operation.result_count(), 1);
    }

    #[test]
    fn test_subgroup_reduce_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation =
            subgroup_reduce(value, AllReduceOperationKind::MaximumSignedInteger, true, Some(4), 2, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_reduce"));
        assert_eq!(operation.value(), value);
        assert_eq!(operation.operation_kind(), AllReduceOperationKind::MaximumSignedInteger);
        assert!(operation.is_uniform());
        assert_eq!(operation.cluster_size().map(|attribute| attribute.signless_value()), Some(4));
        assert_eq!(operation.cluster_stride().signless_value(), 2);
    }

    #[test]
    fn test_shuffle_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location), (i32_type, location), (i32_type, location)]);
        let operation = shuffle(
            block.argument(0).unwrap(),
            block.argument(1).unwrap(),
            block.argument(2).unwrap(),
            ShuffleMode::Xor,
            location,
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.shuffle"));
        assert_eq!(operation.value(), block.argument(0).unwrap());
        assert_eq!(operation.offset(), block.argument(1).unwrap());
        assert_eq!(operation.width(), block.argument(2).unwrap());
        assert_eq!(operation.mode(), ShuffleMode::Xor);
        assert_eq!(operation.shuffled_value().r#type(), i32_type);
        assert_eq!(operation.valid().r#type(), context.signless_integer_type(1));
    }

    #[test]
    fn test_rotate_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = rotate(value, 1, 32, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.rotate"));
        assert_eq!(operation.value(), value);
        assert_eq!(operation.offset().signless_value(), 1);
        assert_eq!(operation.width().signless_value(), 32);
        assert_eq!(operation.rotated_value().r#type(), i32_type);
        assert_eq!(operation.valid().r#type(), context.signless_integer_type(1));
    }

    #[test]
    fn test_barrier_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = barrier(Some(&[AddressSpace::Workgroup, AddressSpace::Private]), location);

        assert_eq!(operation.name().as_str(), Ok("gpu.barrier"));
        assert_eq!(operation.address_spaces(), Some(vec![AddressSpace::Workgroup, AddressSpace::Private]));
    }

    #[test]
    fn test_module_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let target = context.string_attribute("sm_90");
        let targets = context.array_attribute(&[target]);
        let offloading_handler = context.string_attribute("handler").as_ref();
        let operation = module("kernels", Some(targets), Some(offloading_handler), context.region(), location);

        assert_eq!(operation.name().as_str(), Ok("gpu.module"));
        assert_eq!(operation.targets(), Some(targets));
        assert_eq!(operation.offloading_handler(), Some(offloading_handler));
        assert_eq!(operation.region(0).unwrap().blocks().count(), 0);
    }

    #[test]
    fn test_binary_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let target = context.string_attribute("sm_90");
        let object = context.gpu_object_attribute(target, ObjectFormat::Binary, "object", None, None);
        let objects = context.array_attribute(&[object]);
        let offloading_handler = context.string_attribute("handler").as_ref();
        let operation = binary("binary", objects, Some(offloading_handler), location);

        assert_eq!(operation.name().as_str(), Ok("gpu.binary"));
        assert_eq!(operation.objects(), objects);
        assert_eq!(operation.offloading_handler(), Some(offloading_handler));
    }

    #[test]
    fn test_host_register_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(memref_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = host_register(value, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.host_register"));
        assert_eq!(operation.value(), value);
    }

    #[test]
    fn test_host_unregister_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(memref_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = host_unregister(value, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.host_unregister"));
        assert_eq!(operation.value(), value);
    }

    #[test]
    fn test_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let block = context.block(&[(token_type, location)]);
        let dependency = block.argument(0).unwrap();
        let operation = wait(&[dependency], true, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.wait"));
        assert_eq!(operation.async_dependencies(), vec![dependency]);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_alloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (index_type, location), (index_type, location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let dynamic_size = block.argument(1).unwrap().as_ref();
        let symbol_operand = block.argument(2).unwrap().as_ref();
        let operation = alloc(&[dependency], &[dynamic_size], &[symbol_operand], memref_type, true, true, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.alloc"));
        assert_eq!(operation.async_dependencies(), vec![dependency]);
        assert_eq!(operation.dynamic_sizes(), vec![dynamic_size]);
        assert_eq!(operation.symbol_operands(), vec![symbol_operand]);
        assert!(operation.host_shared());
        assert_eq!(operation.memref().r#type(), memref_type);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_dealloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (memref_type.as_ref(), location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let memref = block.argument(1).unwrap().as_ref();
        let operation = dealloc(&[dependency], memref, true, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.dealloc"));
        assert_eq!(operation.async_dependencies(), vec![dependency]);
        assert_eq!(operation.memref(), memref);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_memcpy_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[
            (token_type, location),
            (memref_type.as_ref(), location),
            (memref_type.as_ref(), location),
        ]);
        let dependency = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let source = block.argument(2).unwrap().as_ref();
        let operation = memcpy(&[dependency], destination, source, true, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.memcpy"));
        assert_eq!(operation.async_dependencies(), vec![dependency]);
        assert_eq!(operation.destination(), destination);
        assert_eq!(operation.source(), source);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_memset_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (memref_type.as_ref(), location), (index_type, location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let value = block.argument(2).unwrap().as_ref();
        let operation = memset(&[dependency], destination, value, true, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.memset"));
        assert_eq!(operation.async_dependencies(), vec![dependency]);
        assert_eq!(operation.destination(), destination);
        assert_eq!(operation.value(), value);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_set_default_device_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.index_type(), location)]);
        let device_index = block.argument(0).unwrap();
        let operation = set_default_device(device_index, location);

        assert_eq!(operation.name().as_str(), Ok("gpu.set_default_device"));
        assert_eq!(operation.device_index(), device_index);
    }

    #[test]
    fn test_subgroup_mma_load_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let index_type = context.index_type().as_ref();
        let matrix_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        let block = context.block(&[(memref_type.as_ref(), location), (index_type, location), (index_type, location)]);
        let source_memref = block.argument(0).unwrap();
        let index_0 = block.argument(1).unwrap();
        let index_1 = block.argument(2).unwrap();
        let operation: DetachedSubgroupMmaLoadMatrixOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_load_matrix", location)
                .add_operand(source_memref)
                .add_operand(index_0)
                .add_operand(index_1)
                .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), 16))
                .add_attribute(TRANSPOSE_ATTRIBUTE, context.unit_attribute())
                .add_result(matrix_type),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_load_matrix"));
        assert_eq!(operation.source_memref(), source_memref);
        assert_eq!(operation.indices(), vec![index_0, index_1]);
        assert_eq!(operation.lead_dimension().signless_value(), 16);
        assert!(operation.transpose());
        assert_eq!(operation.matrix().r#type(), matrix_type);
    }

    #[test]
    fn test_subgroup_mma_store_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let index_type = context.index_type().as_ref();
        let matrix_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::C);
        let block = context.block(&[
            (matrix_type.as_ref(), location),
            (memref_type.as_ref(), location),
            (index_type, location),
            (index_type, location),
        ]);
        let source = block.argument(0).unwrap();
        let destination_memref = block.argument(1).unwrap();
        let index_0 = block.argument(2).unwrap();
        let index_1 = block.argument(3).unwrap();
        let operation: DetachedSubgroupMmaStoreMatrixOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_store_matrix", location)
                .add_operand(source)
                .add_operand(destination_memref)
                .add_operand(index_0)
                .add_operand(index_1)
                .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), 16))
                .add_attribute(TRANSPOSE_ATTRIBUTE, context.unit_attribute()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_store_matrix"));
        assert_eq!(operation.source(), source);
        assert_eq!(operation.destination_memref(), destination_memref);
        assert_eq!(operation.indices(), vec![index_0, index_1]);
        assert_eq!(operation.lead_dimension().signless_value(), 16);
        assert!(operation.transpose());
    }

    #[test]
    fn test_subgroup_mma_compute_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let a_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A);
        let b_type = context.gpu_mma_matrix_type([8, 16], context.float32_type(), MmaMatrixOperand::B);
        let c_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C);
        let block =
            context.block(&[(a_type.as_ref(), location), (b_type.as_ref(), location), (c_type.as_ref(), location)]);
        let a = block.argument(0).unwrap();
        let b = block.argument(1).unwrap();
        let c = block.argument(2).unwrap();
        let operation: DetachedSubgroupMmaComputeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_compute", location)
                .add_operand(a)
                .add_operand(b)
                .add_operand(c)
                .add_attribute(A_TRANSPOSE_ATTRIBUTE, context.unit_attribute())
                .add_attribute(B_TRANSPOSE_ATTRIBUTE, context.unit_attribute())
                .add_result(c_type),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_compute"));
        assert_eq!(operation.a(), a);
        assert_eq!(operation.b(), b);
        assert_eq!(operation.c(), c);
        assert!(operation.a_transpose());
        assert!(operation.b_transpose());
        assert_eq!(operation.result_matrix().r#type(), c_type);
    }

    #[test]
    fn test_subgroup_mma_constant_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C);
        let block = context.block(&[(context.float32_type().as_ref(), location)]);
        let value = block.argument(0).unwrap();
        let operation: DetachedSubgroupMmaConstantMatrixOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_constant_matrix", location)
                .add_operand(value)
                .add_result(matrix_type),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_constant_matrix"));
        assert_eq!(operation.value(), value);
        assert_eq!(operation.matrix().r#type(), matrix_type);
    }

    #[test]
    fn test_subgroup_mma_extract_thread_local_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C);
        let index_type = context.index_type().as_ref();
        let block = context.block(&[(matrix_type.as_ref(), location), (index_type, location), (index_type, location)]);
        let matrix = block.argument(0).unwrap();
        let index_0 = block.argument(1).unwrap();
        let index_1 = block.argument(2).unwrap();
        let operation: DetachedSubgroupMmaExtractThreadLocalOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_extract_thread_local", location)
                .add_operand(matrix)
                .add_operand(index_0)
                .add_operand(index_1)
                .add_result(context.float32_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_extract_thread_local"));
        assert_eq!(operation.matrix(), matrix);
        assert_eq!(operation.indices(), vec![index_0, index_1]);
        assert_eq!(operation.value().r#type(), context.float32_type());
    }

    #[test]
    fn test_subgroup_mma_insert_thread_local_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C);
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (context.float32_type().as_ref(), location),
            (matrix_type.as_ref(), location),
            (index_type, location),
            (index_type, location),
        ]);
        let value = block.argument(0).unwrap();
        let matrix = block.argument(1).unwrap();
        let index_0 = block.argument(2).unwrap();
        let index_1 = block.argument(3).unwrap();
        let operation: DetachedSubgroupMmaInsertThreadLocalOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_insert_thread_local", location)
                .add_operand(value)
                .add_operand(matrix)
                .add_operand(index_0)
                .add_operand(index_1)
                .add_result(matrix_type),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_insert_thread_local"));
        assert_eq!(operation.value(), value);
        assert_eq!(operation.matrix(), matrix);
        assert_eq!(operation.indices(), vec![index_0, index_1]);
        assert_eq!(operation.result_matrix().r#type(), matrix_type);
    }

    #[test]
    fn test_subgroup_mma_elementwise_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C);
        let block = context.block(&[(matrix_type.as_ref(), location), (matrix_type.as_ref(), location)]);
        let lhs = block.argument(0).unwrap();
        let rhs = block.argument(1).unwrap();
        let operation: DetachedSubgroupMmaElementwiseOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_mma_elementwise", location)
                .add_operand(lhs)
                .add_operand(rhs)
                .add_attribute(
                    OP_TYPE_ATTRIBUTE,
                    context.gpu_mma_elementwise_operation_attribute(MmaElementwiseOperation::AddFloat),
                )
                .add_result(matrix_type),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_elementwise"));
        assert_eq!(operation.arguments(), vec![lhs, rhs]);
        assert_eq!(operation.operation(), MmaElementwiseOperation::AddFloat);
        assert_eq!(operation.result_matrix().r#type(), matrix_type);
    }

    #[test]
    fn test_create_dn_tensor_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let memref = block.argument(1).unwrap();
        let dimension = block.argument(2).unwrap();
        let stride = block.argument(3).unwrap();
        let operation: DetachedCreateDnTensorOperation<'_, '_> = build_detached_operation(add_operand_segments(
            OperationBuilder::new("gpu.create_dn_tensor", location)
                .add_operand(token)
                .add_operand(memref)
                .add_operand(dimension)
                .add_operand(stride)
                .add_result(context.gpu_sparse_dn_tensor_handle_type())
                .add_result(context.gpu_async_token_type()),
            &[1, 1, 2],
        ));

        assert_eq!(operation.name().as_str(), Ok("gpu.create_dn_tensor"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.memref(), memref);
        assert_eq!(operation.dimensions(), vec![dimension, stride]);
        assert_eq!(operation.dense_tensor().r#type(), context.gpu_sparse_dn_tensor_handle_type());
        assert!(operation.async_token().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_destroy_dn_tensor_operation,
        DetachedDestroyDnTensorOperation,
        "gpu.destroy_dn_tensor",
        operand_count = 1,
        operands = { dense_tensor => 0 },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_coo_operation,
        DetachedCreateCooOperation,
        "gpu.create_coo",
        operand_count = 6,
        operands = {
            rows => 0,
            columns => 1,
            non_zero_count => 2,
            row_indices => 3,
            column_indices => 4,
            values => 5,
        },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_coo_aos_operation,
        DetachedCreateCooAosOperation,
        "gpu.create_coo_aos",
        operand_count = 5,
        operands = { rows => 0, columns => 1, non_zero_count => 2, indices => 3, values => 4 },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_csr_operation,
        DetachedCreateCsrOperation,
        "gpu.create_csr",
        operand_count = 6,
        operands = {
            rows => 0,
            columns => 1,
            non_zero_count => 2,
            row_positions => 3,
            column_indices => 4,
            values => 5,
        },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_csc_operation,
        DetachedCreateCscOperation,
        "gpu.create_csc",
        operand_count = 6,
        operands = {
            rows => 0,
            columns => 1,
            non_zero_count => 2,
            column_positions => 3,
            row_indices => 4,
            values => 5,
        },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_bsr_operation,
        DetachedCreateBsrOperation,
        "gpu.create_bsr",
        operand_count = 8,
        operands = {
            block_rows => 0,
            block_columns => 1,
            block_non_zero_count => 2,
            row_block_size => 3,
            column_block_size => 4,
            block_row_positions => 5,
            block_column_indices => 6,
            values => 7,
        },
    );

    #[test]
    fn test_create_2_to_4_sp_mat_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let rows = block.argument(1).unwrap();
        let columns = block.argument(2).unwrap();
        let memref = block.argument(3).unwrap();
        let operation: DetachedCreate2To4SpMatOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.create_2to4_spmat", location)
                .add_operand(token)
                .add_operand(rows)
                .add_operand(columns)
                .add_operand(memref)
                .add_attribute(
                    PRUNE_FLAG_ATTRIBUTE,
                    context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(Prune2To4SparseMatrixFlag::PruneAndCheck),
                )
                .add_result(context.gpu_sparse_sp_mat_handle_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.create_2to4_spmat"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.rows(), rows);
        assert_eq!(operation.columns(), columns);
        assert_eq!(operation.prune_flag(), Prune2To4SparseMatrixFlag::PruneAndCheck);
        assert_eq!(operation.memref(), memref);
        assert_eq!(operation.sparse_matrix().r#type(), context.gpu_sparse_sp_mat_handle_type());
        assert!(operation.async_token().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_destroy_sp_mat_operation,
        DetachedDestroySpMatOperation,
        "gpu.destroy_sp_mat",
        operand_count = 1,
        operands = { sparse_matrix => 0 },
    );

    #[test]
    fn test_spmv_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let sparse_matrix_a = block.argument(1).unwrap();
        let dense_tensor_x = block.argument(2).unwrap();
        let dense_tensor_y = block.argument(3).unwrap();
        let operation: DetachedSpmvBufferSizeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spmv_buffer_size", location)
                .add_operand(token)
                .add_operand(sparse_matrix_a)
                .add_operand(dense_tensor_x)
                .add_operand(dense_tensor_y)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.index_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spmv_buffer_size"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.dense_tensor_x(), dense_tensor_x);
        assert_eq!(operation.dense_tensor_y(), dense_tensor_y);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert_eq!(operation.buffer_size().r#type(), context.index_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_spmv_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let sparse_matrix_a = block.argument(1).unwrap();
        let dense_tensor_x = block.argument(2).unwrap();
        let dense_tensor_y = block.argument(3).unwrap();
        let buffer = block.argument(4).unwrap();
        let operation: DetachedSpmvOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spmv", location)
                .add_operand(token)
                .add_operand(sparse_matrix_a)
                .add_operand(dense_tensor_x)
                .add_operand(dense_tensor_y)
                .add_operand(buffer)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spmv"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.dense_tensor_x(), dense_tensor_x);
        assert_eq!(operation.dense_tensor_y(), dense_tensor_y);
        assert_eq!(operation.buffer(), buffer);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_spmm_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let sparse_matrix_a = block.argument(1).unwrap();
        let dense_matrix_b = block.argument(2).unwrap();
        let dense_matrix_c = block.argument(3).unwrap();
        let operation: DetachedSpmmBufferSizeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spmm_buffer_size", location)
                .add_operand(token)
                .add_operand(sparse_matrix_a)
                .add_operand(dense_matrix_b)
                .add_operand(dense_matrix_c)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_attribute(RESULT_SEGMENT_SIZES_ATTRIBUTE, context.dense_i32_array_attribute(&[2, 1]).unwrap())
                .add_result(context.index_type())
                .add_result(context.index_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spmm_buffer_size"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.dense_matrix_b(), dense_matrix_b);
        assert_eq!(operation.dense_matrix_c(), dense_matrix_c);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert_eq!(operation.buffer_sizes().len(), 2);
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_spmm_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let sparse_matrix_a = block.argument(1).unwrap();
        let dense_matrix_b = block.argument(2).unwrap();
        let dense_matrix_c = block.argument(3).unwrap();
        let buffer_0 = block.argument(4).unwrap();
        let buffer_1 = block.argument(5).unwrap();
        let operation: DetachedSpmmOperation<'_, '_> = build_detached_operation(add_operand_segments(
            OperationBuilder::new("gpu.spmm", location)
                .add_operand(token)
                .add_operand(sparse_matrix_a)
                .add_operand(dense_matrix_b)
                .add_operand(dense_matrix_c)
                .add_operand(buffer_0)
                .add_operand(buffer_1)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.gpu_async_token_type()),
            &[1, 1, 1, 1, 2],
        ));

        assert_eq!(operation.name().as_str(), Ok("gpu.spmm"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.dense_matrix_b(), dense_matrix_b);
        assert_eq!(operation.dense_matrix_c(), dense_matrix_c);
        assert_eq!(operation.buffers(), vec![buffer_0, buffer_1]);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_sddmm_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let dense_matrix_a = block.argument(1).unwrap();
        let dense_matrix_b = block.argument(2).unwrap();
        let sparse_matrix_c = block.argument(3).unwrap();
        let operation: DetachedSddmmBufferSizeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.sddmm_buffer_size", location)
                .add_operand(token)
                .add_operand(dense_matrix_a)
                .add_operand(dense_matrix_b)
                .add_operand(sparse_matrix_c)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.index_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.sddmm_buffer_size"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.dense_matrix_a(), dense_matrix_a);
        assert_eq!(operation.dense_matrix_b(), dense_matrix_b);
        assert_eq!(operation.sparse_matrix_c(), sparse_matrix_c);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert_eq!(operation.buffer_size().r#type(), context.index_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_sddmm_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let dense_matrix_a = block.argument(1).unwrap();
        let dense_matrix_b = block.argument(2).unwrap();
        let sparse_matrix_c = block.argument(3).unwrap();
        let buffer = block.argument(4).unwrap();
        let operation: DetachedSddmmOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.sddmm", location)
                .add_operand(token)
                .add_operand(dense_matrix_a)
                .add_operand(dense_matrix_b)
                .add_operand(sparse_matrix_c)
                .add_operand(buffer)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.sddmm"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.dense_matrix_a(), dense_matrix_a);
        assert_eq!(operation.dense_matrix_b(), dense_matrix_b);
        assert_eq!(operation.sparse_matrix_c(), sparse_matrix_c);
        assert_eq!(operation.buffer(), buffer);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_sp_gemm_create_descr_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let block = context.block(&[(token_type, location)]);
        let token = block.argument(0).unwrap();
        let operation: DetachedSpGemmCreateDescrOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spgemm_create_descr", location)
                .add_operand(token)
                .add_result(context.gpu_sparse_sp_gemm_operation_handle_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_create_descr"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.descriptor().r#type(), context.gpu_sparse_sp_gemm_operation_handle_type());
        assert!(operation.async_token().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_sp_gemm_destroy_descr_operation,
        DetachedSpGemmDestroyDescrOperation,
        "gpu.spgemm_destroy_descr",
        operand_count = 1,
        operands = { descriptor => 0 },
    );

    #[test]
    fn test_sp_gemm_work_estimation_or_compute_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let descriptor = block.argument(1).unwrap();
        let sparse_matrix_a = block.argument(2).unwrap();
        let sparse_matrix_b = block.argument(3).unwrap();
        let sparse_matrix_c = block.argument(4).unwrap();
        let buffer_size = block.argument(5).unwrap();
        let buffer = block.argument(6).unwrap();
        let operation: DetachedSpGemmWorkEstimationOrComputeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spgemm_work_estimation_or_compute", location)
                .add_operand(token)
                .add_operand(descriptor)
                .add_operand(sparse_matrix_a)
                .add_operand(sparse_matrix_b)
                .add_operand(sparse_matrix_c)
                .add_operand(buffer_size)
                .add_operand(buffer)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_attribute(KIND_ATTRIBUTE, context.gpu_sp_gemm_work_kind_attribute(SpGemmWorkKind::Compute))
                .add_result(context.index_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_work_estimation_or_compute"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.descriptor(), descriptor);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.sparse_matrix_b(), sparse_matrix_b);
        assert_eq!(operation.sparse_matrix_c(), sparse_matrix_c);
        assert_eq!(operation.buffer_size(), buffer_size);
        assert_eq!(operation.buffer(), buffer);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert_eq!(operation.kind(), SpGemmWorkKind::Compute);
        assert_eq!(operation.new_buffer_size().r#type(), context.index_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_sp_gemm_copy_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap();
        let descriptor = block.argument(1).unwrap();
        let sparse_matrix_a = block.argument(2).unwrap();
        let sparse_matrix_b = block.argument(3).unwrap();
        let sparse_matrix_c = block.argument(4).unwrap();
        let operation: DetachedSpGemmCopyOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spgemm_copy", location)
                .add_operand(token)
                .add_operand(descriptor)
                .add_operand(sparse_matrix_a)
                .add_operand(sparse_matrix_b)
                .add_operand(sparse_matrix_c)
                .add_attribute(
                    MODE_A_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::NonTranspose),
                )
                .add_attribute(
                    MODE_B_ATTRIBUTE,
                    context.gpu_matrix_transpose_mode_attribute(MatrixTransposeMode::Transpose),
                )
                .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(context.float32_type()))
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_copy"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.descriptor(), descriptor);
        assert_eq!(operation.mode_a(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a(), sparse_matrix_a);
        assert_eq!(operation.sparse_matrix_b(), sparse_matrix_b);
        assert_eq!(operation.sparse_matrix_c(), sparse_matrix_c);
        assert_eq!(operation.compute_type(), context.float32_type());
        assert!(operation.async_token().is_some());
    }

    #[test]
    fn test_sp_mat_get_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[(token_type, location), (index_type, location)]);
        let token = block.argument(0).unwrap();
        let sparse_matrix = block.argument(1).unwrap();
        let operation: DetachedSpMatGetSizeOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.spmat_get_size", location)
                .add_operand(token)
                .add_operand(sparse_matrix)
                .add_result(context.index_type())
                .add_result(context.index_type())
                .add_result(context.index_type())
                .add_result(context.gpu_async_token_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.spmat_get_size"));
        assert_eq!(operation.async_dependencies(), vec![token]);
        assert_eq!(operation.sparse_matrix(), sparse_matrix);
        assert_eq!(operation.rows().r#type(), context.index_type());
        assert_eq!(operation.columns().r#type(), context.index_type());
        assert_eq!(operation.non_zero_count().r#type(), context.index_type());
        assert!(operation.async_token().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_set_csr_pointers_operation,
        DetachedSetCsrPointersOperation,
        "gpu.set_csr_pointers",
        operand_count = 4,
        operands = {
            sparse_matrix => 0,
            positions => 1,
            coordinates => 2,
            values => 3,
        },
    );

    #[test]
    fn test_warp_execute_on_lane_0_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location), (index_type, location), (index_type, location)]);
        let lane_id = block.argument(0).unwrap();
        let argument_0 = block.argument(1).unwrap();
        let argument_1 = block.argument(2).unwrap();
        let operation: DetachedWarpExecuteOnLane0Operation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.warp_execute_on_lane_0", location)
                .add_operand(lane_id)
                .add_operand(argument_0)
                .add_operand(argument_1)
                .add_attribute(WARP_SIZE_ATTRIBUTE, context.integer_attribute(context.index_type(), 32))
                .add_result(context.index_type())
                .add_result(context.index_type())
                .add_region(context.region()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.warp_execute_on_lane_0"));
        assert_eq!(operation.lane_id(), lane_id);
        assert_eq!(operation.warp_size().signless_value(), 32);
        assert_eq!(operation.arguments(), vec![argument_0, argument_1]);
        assert_eq!(operation.outputs().len(), 2);
        assert_eq!(WarpExecuteOnLane0Operation::region(&operation).blocks().count(), 0);
    }

    #[test]
    fn test_subgroup_broadcast_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location), (index_type, location)]);
        let source = block.argument(0).unwrap();
        let lane = block.argument(1).unwrap();
        let operation: DetachedSubgroupBroadcastOperation<'_, '_> = build_detached_operation(
            OperationBuilder::new("gpu.subgroup_broadcast", location)
                .add_operand(source)
                .add_operand(lane)
                .add_attribute(
                    BROADCAST_TYPE_ATTRIBUTE,
                    context.gpu_broadcast_type_attribute(BroadcastType::SpecificLane),
                )
                .add_result(context.index_type()),
        );

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_broadcast"));
        assert_eq!(operation.source(), source);
        assert_eq!(operation.lane(), Some(lane.as_ref()));
        assert_eq!(operation.broadcast_type(), BroadcastType::SpecificLane);
        assert_eq!(SubgroupBroadcastOperation::output(&operation).r#type(), context.index_type());
    }

    #[test]
    fn test_ballot_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(1), location)]);
        let predicate = block.argument(0).unwrap();
        let operation = ballot(predicate, context.signless_integer_type(32), location);

        assert_eq!(operation.name().as_str(), Ok("gpu.ballot"));
        assert_eq!(operation.predicate(), predicate);
        assert_eq!(operation.mask().r#type(), context.signless_integer_type(32));
    }
}
