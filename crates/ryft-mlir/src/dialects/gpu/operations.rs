use crate::{
    ArrayAttributeRef, Attribute, AttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error,
    FUNCTION_TYPE_ATTRIBUTE, FlatSymbolRefAttributeRef, Function, IntegerAttributeRef, Location, OneRegion, Operation,
    OperationBuilder, RegionRef, SYMBOL_NAME_ATTRIBUTE, StringAttributeRef, StringRef, Symbol, SymbolTable,
    TryIntoWithContext, Type, TypeRef, Value, ValueRef, mlir_op, mlir_op_trait,
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
                fn dimension(&self) -> Result<Dimension, Error> {
                    self.attribute(DIMENSION_ATTRIBUTE)
                        .and_then(|attribute| attribute.cast::<DimensionAttributeRef>())
                        .ok_or_else(|| {
                            Error::invalid_argument(format!(
                                "missing or invalid `{DIMENSION_ATTRIBUTE}` attribute in `{}`",
                                self.name().as_str().unwrap_or("<unknown>"),
                            ))
                        })?.value()
                }

                /// Returns the optional upper bound associated with this operation.
                fn upper_bound(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
                    if self.has_attribute(UPPER_BOUND_ATTRIBUTE) {
                        self.integer_attribute(UPPER_BOUND_ATTRIBUTE).map(Some)
                    } else {
                        Ok(None)
                    }
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
            ) -> Result<[<Detached $name Operation>]<'c, 't>, Error> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu()?);
                let builder = OperationBuilder::new($operation_name, location)
                    .add_attribute(DIMENSION_ATTRIBUTE, context.gpu_dimension_attribute(dimension)?)
                    .add_result(context.index_type());
                let builder = if let Some(upper_bound) = upper_bound {
                    builder.add_attribute(
                        UPPER_BOUND_ATTRIBUTE,
                        context.integer_attribute(context.index_type(), upper_bound as i64),
                    )
                } else {
                    builder
                };
                builder.build().and_then(|operation| unsafe {
                    operation.cast().ok_or_else(|| {
                        Error::invalid_argument(concat!(
                            "invalid arguments to `gpu::",
                            stringify!($function_name),
                            "`",
                        ))
                    })
                })
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
                fn upper_bound(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
                    if self.has_attribute(UPPER_BOUND_ATTRIBUTE) {
                        self.integer_attribute(UPPER_BOUND_ATTRIBUTE).map(Some)
                    } else {
                        Ok(None)
                    }
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
            ) -> Result<[<Detached $name Operation>]<'c, 't>, Error> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu()?);
                let builder = OperationBuilder::new($operation_name, location).add_result(context.index_type());
                let builder = if let Some(upper_bound) = upper_bound {
                    builder.add_attribute(
                        UPPER_BOUND_ATTRIBUTE,
                        context.integer_attribute(context.index_type(), upper_bound as i64),
                    )
                } else {
                    builder
                };
                builder.build().and_then(|operation| unsafe {
                    operation.cast().ok_or_else(|| {
                        Error::invalid_argument(concat!(
                            "invalid arguments to `gpu::",
                            stringify!($function_name),
                            "`",
                        ))
                    })
                })
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
        self.attribute(KERNEL_ATTRIBUTE).is_some()
    }

    /// Returns the optional known block size hint.
    fn known_block_size(&self) -> Result<Option<Vec<i32>>, Error> {
        if self.has_attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE) {
            self.dense_integer_32_array_attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE)
                .map(|attribute| Some(attribute.values().collect()))
        } else {
            Ok(None)
        }
    }

    /// Returns the optional known grid size hint.
    fn known_grid_size(&self) -> Result<Option<Vec<i32>>, Error> {
        if self.has_attribute(KNOWN_GRID_SIZE_ATTRIBUTE) {
            self.dense_integer_32_array_attribute(KNOWN_GRID_SIZE_ATTRIBUTE)
                .map(|attribute| Some(attribute.values().collect()))
        } else {
            Ok(None)
        }
    }

    /// Returns the optional known cluster size hint.
    fn known_cluster_size(&self) -> Result<Option<Vec<i32>>, Error> {
        if self.has_attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE) {
            self.dense_integer_32_array_attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE)
                .map(|attribute| Some(attribute.values().collect()))
        } else {
            Ok(None)
        }
    }

    /// Returns the number of workgroup attributions.
    fn workgroup_attribution_count(&self) -> Result<usize, Error> {
        if self.has_attribute(WORKGROUP_ATTRIBUTIONS_ATTRIBUTE) {
            usize::try_from(self.integer_attribute(WORKGROUP_ATTRIBUTIONS_ATTRIBUTE)?.signless_value())
                .map_err(|_| Error::invalid_argument("invalid `workgroup_attributions` attribute in `gpu.func`"))
        } else {
            Ok(0)
        }
    }

    /// Returns the GPU function body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
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

/// Properties used to construct a [`FuncOperation`].
#[derive(Clone, Debug, Default)]
pub struct FuncProperties<'c, 't> {
    /// Function argument types.
    pub arguments: Vec<TypeRef<'c, 't>>,

    /// Function result types.
    pub results: Vec<TypeRef<'c, 't>>,

    /// Whether the function is a host-launchable GPU kernel.
    pub is_kernel: bool,

    /// Optional known block-size launch hint.
    pub known_block_size: Option<[i32; 3]>,

    /// Optional known grid-size launch hint.
    pub known_grid_size: Option<[i32; 3]>,

    /// Optional known cluster-size launch hint.
    pub known_cluster_size: Option<[i32; 3]>,

    /// Number of workgroup attribution block arguments following the function arguments.
    pub workgroup_attribution_count: usize,
}

/// Constructs a new detached/owned [`FuncOperation`] at the specified [`Location`].
pub fn func<'c, 't: 'c, N: TryIntoWithContext<'c, 't, StringAttributeRef<'c, 't>>, L: Location<'c, 't>>(
    name: N,
    properties: FuncProperties<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.func", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, name.try_into_with_context(context)?)
        .add_attribute(
            FUNCTION_TYPE_ATTRIBUTE,
            context.type_attribute(context.function_type(&properties.arguments, &properties.results)),
        );
    if properties.is_kernel {
        builder = builder.add_attribute(KERNEL_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(known_block_size) = properties.known_block_size {
        builder =
            builder.add_attribute(KNOWN_BLOCK_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&known_block_size)?);
    }
    if let Some(known_grid_size) = properties.known_grid_size {
        builder =
            builder.add_attribute(KNOWN_GRID_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&known_grid_size)?);
    }
    if let Some(known_cluster_size) = properties.known_cluster_size {
        builder = builder
            .add_attribute(KNOWN_CLUSTER_SIZE_ATTRIBUTE, context.dense_i32_array_attribute(&known_cluster_size)?);
    }
    if properties.workgroup_attribution_count > 0 {
        builder = builder.add_attribute(
            WORKGROUP_ATTRIBUTIONS_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(64), properties.workgroup_attribution_count as i64),
        );
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::func`"))
    })
}

/// GPU operation that returns the dynamic shared-memory memref for the current kernel.
pub trait DynamicSharedMemoryOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the dynamic shared-memory memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedDynamicSharedMemoryOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.dynamic_shared_memory", location)
        .add_result(memref_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::dynamic_shared_memory`"))
        })
}

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

/// Name of the MLIR attribute storing operation operand segment sizes.
pub const OPERAND_SEGMENT_SIZES_ATTRIBUTE: &str = "operand_segment_sizes";

/// GPU operation that launches a named GPU kernel function.
pub trait LaunchFuncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dependency_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        self.operand_values().take(dependency_count).collect()
    }

    /// Returns the kernel symbol reference.
    fn kernel(&self) -> Result<crate::SymbolRefAttributeRef<'c, 't>, Error> {
        self.symbol_ref_attribute(KERNEL_ATTRIBUTE)
    }

    /// Returns the grid size operands.
    fn grid_size(&self) -> Result<Dim3<'o, 'c, 't>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            if range.len() != 1 {
                return Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                )));
            }
            self.operand_value(range.start)
        };
        Ok(Dim3 { x: operand(1)?, y: operand(2)?, z: operand(3)? })
    }

    /// Returns the block size operands.
    fn block_size(&self) -> Result<Dim3<'o, 'c, 't>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            if range.len() != 1 {
                return Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                )));
            }
            self.operand_value(range.start)
        };
        Ok(Dim3 { x: operand(4)?, y: operand(5)?, z: operand(6)? })
    }

    /// Returns the optional cluster size operands.
    fn cluster_size(&self) -> Result<Option<Dim3<'o, 'c, 't>>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            match range.len() {
                0 => Ok(None),
                1 => self.operand_value(range.start).map(Some),
                _ => Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                ))),
            }
        };
        match (operand(7)?, operand(8)?, operand(9)?) {
            (None, None, None) => Ok(None),
            (Some(x), Some(y), Some(z)) => Ok(Some(Dim3 { x, y, z })),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the optional dynamic shared-memory size operand.
    fn dynamic_shared_memory_size(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 10)?;
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

    /// Returns the kernel argument operands.
    fn kernel_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 11)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the optional async object operand.
    fn async_object(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 12)?;
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

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(LaunchFunc);
mlir_op_trait!(LaunchFunc, ZeroRegions);
mlir_op_trait!(LaunchFunc, ZeroSuccessors);

/// Constructs a new detached/owned [`LaunchFuncOperation`] at the specified [`Location`].
pub fn launch_func<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    properties: LaunchFuncProperties<'o, 'c, 't>,
    location: L,
) -> Result<DetachedLaunchFuncOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
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
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    if !properties.is_async {
        segment_sizes[12] = usize::from(properties.async_object.is_some());
    }
    let segment_sizes = segment_sizes.iter().map(|size| *size as i32).collect::<Vec<_>>();
    let segment_sizes = context.dense_i32_array_attribute(segment_sizes.as_slice())?;
    builder
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment_sizes)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::launch_func`"))
        })
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
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dependency_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        self.operand_values().take(dependency_count).collect()
    }

    /// Returns the grid size operands.
    fn grid_size(&self) -> Result<Dim3<'o, 'c, 't>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            if range.len() != 1 {
                return Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                )));
            }
            self.operand_value(range.start)
        };
        Ok(Dim3 { x: operand(1)?, y: operand(2)?, z: operand(3)? })
    }

    /// Returns the block size operands.
    fn block_size(&self) -> Result<Dim3<'o, 'c, 't>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            if range.len() != 1 {
                return Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                )));
            }
            self.operand_value(range.start)
        };
        Ok(Dim3 { x: operand(4)?, y: operand(5)?, z: operand(6)? })
    }

    /// Returns optional cluster size operands.
    fn cluster_size(&self) -> Result<Option<Dim3<'o, 'c, 't>>, Error> {
        let operand = |segment| {
            let range =
                self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment)?;
            match range.len() {
                0 => Ok(None),
                1 => self.operand_value(range.start).map(Some),
                _ => Err(Error::invalid_argument(format!(
                    "invalid `{}` attribute in `{}`",
                    OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                    self.name(),
                ))),
            }
        };
        match (operand(7)?, operand(8)?, operand(9)?) {
            (None, None, None) => Ok(None),
            (Some(x), Some(y), Some(z)) => Ok(Some(Dim3 { x, y, z })),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                OPERAND_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }

    /// Returns the optional dynamic shared-memory size operand.
    fn dynamic_shared_memory_size(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 10)?;
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

    /// Returns the optional module symbol.
    fn module_symbol(&self) -> Result<Option<FlatSymbolRefAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(MODULE_ATTRIBUTE) {
            self.flat_symbol_ref_attribute(MODULE_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the optional function symbol.
    fn function_symbol(&self) -> Result<Option<FlatSymbolRefAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(FUNCTION_ATTRIBUTE) {
            self.flat_symbol_ref_attribute(FUNCTION_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the number of workgroup attributions.
    fn workgroup_attribution_count(&self) -> Result<usize, Error> {
        if self.has_attribute(WORKGROUP_ATTRIBUTIONS_ATTRIBUTE) {
            usize::try_from(self.integer_attribute(WORKGROUP_ATTRIBUTIONS_ATTRIBUTE)?.signless_value())
                .map_err(|_| Error::invalid_argument("invalid `workgroup_attributions` attribute in `gpu.launch_func`"))
        } else {
            Ok(0)
        }
    }

    /// Returns the launch body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.body_region()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
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
) -> Result<DetachedLaunchOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
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
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    let segment_sizes = segment_sizes.iter().map(|size| *size as i32).collect::<Vec<_>>();
    let segment_sizes = context.dense_i32_array_attribute(segment_sizes.as_slice())?;
    builder
        .add_region(body)
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment_sizes)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::launch`"))
        })
}

/// Name of the [`Attribute`] that stores a `printf` format string.
pub const FORMAT_ATTRIBUTE: &str = "format";

/// GPU device-side printf operation.
pub trait PrintfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the printf format string.
    fn format(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(FORMAT_ATTRIBUTE)?.string())
    }

    /// Returns the printf argument operands.
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
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
) -> Result<DetachedPrintfOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.printf", location)
        .add_attribute(FORMAT_ATTRIBUTE, context.string_attribute(format))
        .add_operands(arguments)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::printf`"))
        })
}

/// GPU function return terminator operation.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the returned values.
    fn values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
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
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.return", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::return`"))
        })
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
pub fn terminator<'c, 't: 'c, L: Location<'c, 't>>(location: L) -> Result<DetachedTerminatorOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.terminator", location).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::terminator`"))
    })
}

/// GPU region yield operation.
pub trait YieldOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the yielded values.
    fn values(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
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
) -> Result<DetachedYieldOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.yield", location)
        .add_operands(values)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::yield`"))
        })
}

/// Name of the [`Attribute`] that stores an all-reduce or subgroup-reduce operation kind.
pub const OP_ATTRIBUTE: &str = "op";

/// Name of the [`Attribute`] that marks a collective operation as uniform.
pub const UNIFORM_ATTRIBUTE: &str = "uniform";

/// GPU all-reduce operation across a workgroup.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the value to reduce.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional built-in reduction operation kind.
    fn operation_kind(&self) -> Result<Option<AllReduceOperationKind>, Error> {
        ({
            let attribute_name = OP_ATTRIBUTE;
            self.attribute(attribute_name)
                .map(|attribute| {
                    attribute.cast::<AllReduceOperationKindAttributeRef>().ok_or_else(|| {
                        Error::invalid_argument(format!(
                            "invalid `{}` attribute in `{}`",
                            attribute_name,
                            self.name().as_str().unwrap_or("<unknown>"),
                        ))
                    })
                })
                .transpose()
        })?
        .map(|attribute| attribute.value())
        .transpose()
    }

    /// Returns `true` if the collective is marked uniform.
    fn is_uniform(&self) -> bool {
        self.attribute(UNIFORM_ATTRIBUTE).is_some()
    }

    /// Returns the reduction body region.
    fn body(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
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
) -> Result<DetachedAllReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.all_reduce", location).add_operand(value).add_result(result_type);
    if let Some(operation_kind) = operation_kind {
        builder = builder.add_attribute(OP_ATTRIBUTE, context.gpu_all_reduce_operation_kind_attribute(operation_kind)?);
    }
    if is_uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::all_reduce`"))
    })
}

/// Name of the [`Attribute`] that stores subgroup cluster size.
pub const CLUSTER_SIZE_ATTRIBUTE: &str = "cluster_size";

/// Name of the [`Attribute`] that stores subgroup cluster stride.
pub const CLUSTER_STRIDE_ATTRIBUTE: &str = "cluster_stride";

/// GPU subgroup-reduce operation.
pub trait SubgroupReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to reduce.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the built-in reduction operation kind.
    fn operation_kind(&self) -> Result<AllReduceOperationKind, Error> {
        self.attribute(OP_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<AllReduceOperationKindAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    OP_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns `true` if the reduction is marked uniform.
    fn is_uniform(&self) -> bool {
        self.attribute(UNIFORM_ATTRIBUTE).is_some()
    }

    /// Returns the optional subgroup cluster size.
    fn cluster_size(&self) -> Result<Option<IntegerAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(CLUSTER_SIZE_ATTRIBUTE) {
            self.integer_attribute(CLUSTER_SIZE_ATTRIBUTE).map(Some)
        } else {
            Ok(None)
        }
    }

    /// Returns the subgroup cluster stride.
    fn cluster_stride(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(CLUSTER_STRIDE_ATTRIBUTE)
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
) -> Result<DetachedSubgroupReduceOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.subgroup_reduce", location)
        .add_operand(value)
        .add_attribute(OP_ATTRIBUTE, context.gpu_all_reduce_operation_kind_attribute(operation_kind)?)
        .add_attribute(
            CLUSTER_STRIDE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), cluster_stride as i64),
        )
        .add_result(value.r#type()?);
    if is_uniform {
        builder = builder.add_attribute(UNIFORM_ATTRIBUTE, context.unit_attribute());
    }
    if let Some(cluster_size) = cluster_size {
        builder = builder.add_attribute(
            CLUSTER_SIZE_ATTRIBUTE,
            context.integer_attribute(context.signless_integer_type(32), cluster_size as i64),
        );
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_reduce`"))
    })
}

/// Name of the [`Attribute`] that stores GPU shuffle mode.
pub const MODE_ATTRIBUTE: &str = "mode";

/// GPU subgroup shuffle operation.
pub trait ShuffleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to shuffle.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the shuffle offset operand.
    fn offset(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the shuffle width operand.
    fn width(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns the shuffle mode.
    fn mode(&self) -> Result<ShuffleMode, Error> {
        self.attribute(MODE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ShuffleModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the shuffled value result.
    fn shuffled_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the validity flag result.
    fn valid(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(1)?.as_ref())
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
) -> Result<DetachedShuffleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.shuffle", location)
        .add_operand(value)
        .add_operand(offset)
        .add_operand(width)
        .add_attribute(MODE_ATTRIBUTE, context.gpu_shuffle_mode_attribute(mode)?)
        .add_result(value.r#type()?)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::shuffle`"))
        })
}

/// Name of the [`Attribute`] that stores rotate offset.
pub const OFFSET_ATTRIBUTE: &str = "offset";

/// Name of the [`Attribute`] that stores rotate width.
pub const WIDTH_ATTRIBUTE: &str = "width";

/// GPU subgroup rotate operation.
pub trait RotateOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the value to rotate.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the rotation offset attribute.
    fn offset(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(OFFSET_ATTRIBUTE)
    }

    /// Returns the rotation width attribute.
    fn width(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(WIDTH_ATTRIBUTE)
    }

    /// Returns the rotated value result.
    fn rotated_value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the validity flag result.
    fn valid(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(1)?.as_ref())
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
) -> Result<DetachedRotateOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.rotate", location)
        .add_operand(value)
        .add_attribute(OFFSET_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), offset as i64))
        .add_attribute(WIDTH_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(32), width as i64))
        .add_result(value.r#type()?)
        .add_result(context.signless_integer_type(1))
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::rotate`"))
        })
}

/// Name of the [`Attribute`] that stores GPU barrier memory fence address spaces.
pub const ADDRESS_SPACES_ATTRIBUTE: &str = "address_spaces";

/// GPU workgroup barrier operation.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional memory fence address spaces.
    fn address_spaces(&self) -> Result<Option<Vec<AddressSpace>>, Error> {
        if !self.has_attribute(ADDRESS_SPACES_ATTRIBUTE) {
            return Ok(None);
        }
        self.array_attribute(ADDRESS_SPACES_ATTRIBUTE)?
            .elements()
            .map(|element| {
                element?
                    .cast::<AddressSpaceAttributeRef>()
                    .ok_or_else(|| Error::invalid_argument("invalid address space in `gpu.barrier`"))?
                    .value()
            })
            .collect::<Result<Vec<_>, Error>>()
            .map(Some)
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
) -> Result<DetachedBarrierOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.barrier", location);
    if let Some(address_spaces) = address_spaces {
        let address_spaces = address_spaces
            .iter()
            .map(|address_space| Ok(context.gpu_address_space_attribute(*address_space)?.as_ref()))
            .collect::<Result<Vec<_>, Error>>()?;
        builder = builder.add_attribute(ADDRESS_SPACES_ATTRIBUTE, context.array_attribute(address_spaces.as_slice()));
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::barrier`"))
    })
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
    fn targets(&self) -> Result<Option<ArrayAttributeRef<'c, 't>>, Error> {
        if self.has_attribute(TARGETS_ATTRIBUTE) { self.array_attribute(TARGETS_ATTRIBUTE).map(Some) } else { Ok(None) }
    }

    /// Returns the optional offloading handler attribute.
    fn offloading_handler(&self) -> Option<AttributeRef<'c, 't>> {
        self.attribute(OFFLOADING_HANDLER_ATTRIBUTE)
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
) -> Result<DetachedModuleOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.module", location)
        .add_attribute(crate::SYMBOL_NAME_ATTRIBUTE, context.string_attribute(name));
    if let Some(targets) = targets {
        builder = builder.add_attribute(TARGETS_ATTRIBUTE, targets);
    }
    if let Some(offloading_handler) = offloading_handler {
        builder = builder.add_attribute(OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    }
    builder.add_region(body).build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::module`"))
    })
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
    fn objects(&self) -> Result<ArrayAttributeRef<'c, 't>, Error> {
        self.array_attribute(OBJECTS_ATTRIBUTE)
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
) -> Result<DetachedBinaryOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.binary", location)
        .add_attribute(crate::SYMBOL_NAME_ATTRIBUTE, context.string_attribute(name))
        .add_attribute(OBJECTS_ATTRIBUTE, objects);
    if let Some(offloading_handler) = offloading_handler {
        builder = builder.add_attribute(OFFLOADING_HANDLER_ATTRIBUTE, offloading_handler);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::binary`"))
    })
}

macro_rules! gpu_one_memref_operand_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $method:ident, $doc:literal $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                #[doc = "Returns the memref operand of this operation."]
                fn $method(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
                    self.operand_value(0)
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
            ) -> Result<[<Detached $name Operation>]<'c, 't>, Error> {
                location.context().load_dialect(DialectHandle::gpu()?);
                OperationBuilder::new($operation_name, location)
                    .add_operand(value)
                    .build()
                    .and_then(|operation| unsafe {
                        operation.cast().ok_or_else(|| {
                            Error::invalid_argument(concat!(
                                "invalid arguments to `gpu::",
                                stringify!($function_name),
                                "`",
                            ))
                        })
                    })
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
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
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
) -> Result<DetachedWaitOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.wait", location).add_operands(async_dependencies);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::wait`"))
    })
}

/// Name of the [`Attribute`] that marks host-shared GPU allocation.
pub const HOST_SHARED_ATTRIBUTE: &str = "hostShared";

/// GPU allocation operation.
pub trait AllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dependency_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        self.operand_values().take(dependency_count).collect()
    }

    /// Returns dynamic size operands.
    fn dynamic_sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 1)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns symbol operands.
    fn symbol_operands(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns `true` if the allocation is host shared.
    fn host_shared(&self) -> bool {
        self.attribute(HOST_SHARED_ATTRIBUTE).is_some()
    }

    /// Returns the allocated memref result.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
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
) -> Result<DetachedAllocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.alloc", location)
        .add_operands(async_dependencies)
        .add_operands(dynamic_sizes)
        .add_operands(symbol_operands)
        .add_result(memref_type);
    if host_shared {
        builder = builder.add_attribute(HOST_SHARED_ATTRIBUTE, context.unit_attribute());
    }
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    let segment_sizes = [async_dependencies.len(), dynamic_sizes.len(), symbol_operands.len()]
        .iter()
        .map(|size| *size as i32)
        .collect::<Vec<_>>();
    let segment_sizes = context.dense_i32_array_attribute(segment_sizes.as_slice())?;
    builder
        .add_attribute(OPERAND_SEGMENT_SIZES_ATTRIBUTE, segment_sizes)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::alloc`"))
        })
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
                fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
                    let async_token_type = self.context().gpu_async_token_type()?.as_ref();
                    let mut dependencies = Vec::new();
                    for operand in self.operand_values() {
                        let operand = operand?;
                        if operand.r#type()? != async_token_type {
                            break;
                        }
                        dependencies.push(operand);
                    }
                    Ok(dependencies)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
                        self.operand_value(self.async_dependencies()?.len() + $index)
                    }
                )+

                /// Returns the optional async token result.
                fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
                    if self.result_count() <= 0 {
                        Ok(None)
                    } else {
                        self.result(0).map(|result| Some(result.as_ref()))
                    }
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
) -> Result<DetachedDeallocOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder =
        OperationBuilder::new("gpu.dealloc", location).add_operands(async_dependencies).add_operand(memref);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::dealloc`"))
    })
}

/// Constructs a new detached/owned [`MemcpyOperation`] at the specified [`Location`].
pub fn memcpy<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    source: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedMemcpyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.memcpy", location)
        .add_operands(async_dependencies)
        .add_operand(destination)
        .add_operand(source);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::memcpy`"))
    })
}

/// Constructs a new detached/owned [`MemsetOperation`] at the specified [`Location`].
pub fn memset<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    destination: ValueRef<'o, 'c, 't>,
    value: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedMemsetOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.memset", location)
        .add_operands(async_dependencies)
        .add_operand(destination)
        .add_operand(value);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::memset`"))
    })
}

/// GPU operation that sets the default device index.
pub trait SetDefaultDeviceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the device index operand.
    fn device_index(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }
}

mlir_op!(SetDefaultDevice);
mlir_op_trait!(SetDefaultDevice, ZeroRegions);
mlir_op_trait!(SetDefaultDevice, ZeroSuccessors);

/// Constructs a new detached/owned [`SetDefaultDeviceOperation`] at the specified [`Location`].
pub fn set_default_device<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    device_index: V,
    location: L,
) -> Result<DetachedSetDefaultDeviceOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.set_default_device", location)
        .add_operand(device_index)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::set_default_device`"))
        })
}

/// Name of the [`Attribute`] that stores MMA leading dimensions.
pub const LEAD_DIMENSION_ATTRIBUTE: &str = "leadDimension";

/// Name of the [`Attribute`] that marks transposed MMA matrix loads and stores.
pub const TRANSPOSE_ATTRIBUTE: &str = "transpose";

/// GPU subgroup MMA matrix load operation.
pub trait SubgroupMmaLoadMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source memref.
    fn source_memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the leading dimension attribute.
    fn lead_dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(LEAD_DIMENSION_ATTRIBUTE)
    }

    /// Returns `true` if the load is transposed.
    fn transpose(&self) -> bool {
        self.attribute(TRANSPOSE_ATTRIBUTE).is_some()
    }

    /// Returns the loaded MMA matrix result.
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaLoadMatrix);
mlir_op_trait!(SubgroupMmaLoadMatrix, OneResult);
mlir_op_trait!(SubgroupMmaLoadMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaLoadMatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaLoadMatrixOperation`] at the specified [`Location`].
pub fn subgroup_mma_load_matrix<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    source_memref: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    lead_dimension: i64,
    transpose: bool,
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaLoadMatrixOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_load_matrix", location)
        .add_operand(source_memref)
        .add_operands(indices)
        .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), lead_dimension))
        .add_result(result_type);
    if transpose {
        builder = builder.add_attribute(TRANSPOSE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_load_matrix`"))
    })
}

/// GPU subgroup MMA matrix store operation.
pub trait SubgroupMmaStoreMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source MMA matrix.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination memref.
    fn destination_memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the leading dimension attribute.
    fn lead_dimension(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(LEAD_DIMENSION_ATTRIBUTE)
    }

    /// Returns `true` if the store is transposed.
    fn transpose(&self) -> bool {
        self.attribute(TRANSPOSE_ATTRIBUTE).is_some()
    }
}

mlir_op!(SubgroupMmaStoreMatrix);
mlir_op_trait!(SubgroupMmaStoreMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaStoreMatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaStoreMatrixOperation`] at the specified [`Location`].
pub fn subgroup_mma_store_matrix<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    destination_memref: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    lead_dimension: i64,
    transpose: bool,
    location: L,
) -> Result<DetachedSubgroupMmaStoreMatrixOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_store_matrix", location)
        .add_operand(source)
        .add_operand(destination_memref)
        .add_operands(indices)
        .add_attribute(LEAD_DIMENSION_ATTRIBUTE, context.integer_attribute(context.index_type(), lead_dimension));
    if transpose {
        builder = builder.add_attribute(TRANSPOSE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_store_matrix`"))
    })
}

/// Name of the [`Attribute`] that marks transposed MMA matrix A operands.
pub const A_TRANSPOSE_ATTRIBUTE: &str = "a_transpose";

/// Name of the [`Attribute`] that marks transposed MMA matrix B operands.
pub const B_TRANSPOSE_ATTRIBUTE: &str = "b_transpose";

/// GPU subgroup MMA matrix multiply-accumulate operation.
pub trait SubgroupMmaComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the A operand matrix.
    fn a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the B operand matrix.
    fn b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the C accumulator matrix.
    fn c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(2)
    }

    /// Returns `true` if operand A is transposed.
    fn a_transpose(&self) -> bool {
        self.attribute(A_TRANSPOSE_ATTRIBUTE).is_some()
    }

    /// Returns `true` if operand B is transposed.
    fn b_transpose(&self) -> bool {
        self.attribute(B_TRANSPOSE_ATTRIBUTE).is_some()
    }

    /// Returns the result matrix.
    fn result_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaCompute);
mlir_op_trait!(SubgroupMmaCompute, OneResult);
mlir_op_trait!(SubgroupMmaCompute, ZeroRegions);
mlir_op_trait!(SubgroupMmaCompute, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaComputeOperation`] at the specified [`Location`].
pub fn subgroup_mma_compute<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    a: ValueRef<'o, 'c, 't>,
    b: ValueRef<'o, 'c, 't>,
    c: ValueRef<'o, 'c, 't>,
    a_transpose: bool,
    b_transpose: bool,
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaComputeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.subgroup_mma_compute", location)
        .add_operand(a)
        .add_operand(b)
        .add_operand(c)
        .add_result(result_type);
    if a_transpose {
        builder = builder.add_attribute(A_TRANSPOSE_ATTRIBUTE, context.unit_attribute());
    }
    if b_transpose {
        builder = builder.add_attribute(B_TRANSPOSE_ATTRIBUTE, context.unit_attribute());
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_compute`"))
    })
}

/// GPU subgroup MMA constant matrix operation.
pub trait SubgroupMmaConstantMatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the scalar value broadcast into the matrix.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the resulting MMA matrix.
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaConstantMatrix);
mlir_op_trait!(SubgroupMmaConstantMatrix, OneOperand);
mlir_op_trait!(SubgroupMmaConstantMatrix, OneResult);
mlir_op_trait!(SubgroupMmaConstantMatrix, ZeroRegions);
mlir_op_trait!(SubgroupMmaConstantMatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaConstantMatrixOperation`] at the specified [`Location`].
pub fn subgroup_mma_constant_matrix<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaConstantMatrixOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.subgroup_mma_constant_matrix", location)
        .add_operand(value)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_constant_matrix`"))
        })
}

/// GPU subgroup MMA thread-local extract operation.
pub trait SubgroupMmaExtractThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source MMA matrix.
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the thread-local index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns the extracted scalar.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaExtractThreadLocal);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, OneResult);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, ZeroRegions);
mlir_op_trait!(SubgroupMmaExtractThreadLocal, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaExtractThreadLocalOperation`] at the specified [`Location`].
pub fn subgroup_mma_extract_thread_local<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    matrix: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaExtractThreadLocalOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.subgroup_mma_extract_thread_local", location)
        .add_operand(matrix)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_extract_thread_local`"))
        })
}

/// GPU subgroup MMA thread-local insert operation.
pub trait SubgroupMmaInsertThreadLocalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the scalar value to insert.
    fn value(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the destination MMA matrix.
    fn matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(1)
    }

    /// Returns the thread-local index operands.
    fn indices(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(2).collect()
    }

    /// Returns the resulting MMA matrix.
    fn result_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaInsertThreadLocal);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, OneResult);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, ZeroRegions);
mlir_op_trait!(SubgroupMmaInsertThreadLocal, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaInsertThreadLocalOperation`] at the specified [`Location`].
pub fn subgroup_mma_insert_thread_local<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    value: ValueRef<'o, 'c, 't>,
    matrix: ValueRef<'o, 'c, 't>,
    indices: &[ValueRef<'o, 'c, 't>],
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaInsertThreadLocalOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.subgroup_mma_insert_thread_local", location)
        .add_operand(value)
        .add_operand(matrix)
        .add_operands(indices)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_insert_thread_local`"))
        })
}

/// Name of the [`Attribute`] that stores the MMA elementwise operation kind.
pub const OP_TYPE_ATTRIBUTE: &str = "opType";

/// GPU subgroup MMA elementwise operation.
pub trait SubgroupMmaElementwiseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the MMA matrix operands.
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().collect()
    }

    /// Returns the elementwise operation kind.
    fn operation(&self) -> Result<MmaElementwiseOperation, Error> {
        self.attribute(OP_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MmaElementwiseOperationAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    OP_TYPE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the resulting MMA matrix.
    fn result_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupMmaElementwise);
mlir_op_trait!(SubgroupMmaElementwise, OneResult);
mlir_op_trait!(SubgroupMmaElementwise, ZeroRegions);
mlir_op_trait!(SubgroupMmaElementwise, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupMmaElementwiseOperation`] at the specified [`Location`].
pub fn subgroup_mma_elementwise<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    arguments: &[ValueRef<'o, 'c, 't>],
    operation: MmaElementwiseOperation,
    result_type: T,
    location: L,
) -> Result<DetachedSubgroupMmaElementwiseOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.subgroup_mma_elementwise", location)
        .add_operands(arguments)
        .add_attribute(OP_TYPE_ATTRIBUTE, context.gpu_mma_elementwise_operation_attribute(operation)?)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_mma_elementwise`"))
        })
}

macro_rules! gpu_sparse_async_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $doc:literal, operands = { $($method:ident => $index:expr),* $(,)* } $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns async token dependencies.
                fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
                    let async_token_type = self.context().gpu_async_token_type()?.as_ref();
                    let mut dependencies = Vec::new();
                    for operand in self.operand_values() {
                        let operand = operand?;
                        if operand.r#type()? != async_token_type {
                            break;
                        }
                        dependencies.push(operand);
                    }
                    Ok(dependencies)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
                        self.operand_value(self.async_dependencies()?.len() + $index)
                    }
                )*

                /// Returns the optional async token result.
                fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
                    if self.result_count() > 1 {
                        self.result(self.result_count() - 1).map(|result| Some(result.as_ref()))
                    } else if self.result_count() == 1 {
                        self.result(0).map(|result| Some(result.as_ref()))
                    } else {
                        Ok(None)
                    }
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached/owned [`"]
            #[doc = stringify!($name)]
            #[doc = "Operation`] at the specified [`Location`]."]
            pub fn $function_name<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
                async_dependencies: &[ValueRef<'o, 'c, 't>],
                $($method: ValueRef<'o, 'c, 't>,)*
                is_async: bool,
                location: L,
            ) -> Result<[<Detached $name Operation>]<'c, 't>, Error> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu()?);
                let mut builder = OperationBuilder::new($operation_name, location).add_operands(async_dependencies);
                $(builder = builder.add_operand($method);)*
                if is_async {
                    builder = builder.add_result(context.gpu_async_token_type()?);
                }
                builder.build().and_then(|operation| unsafe {
                    operation.cast().ok_or_else(|| {
                        Error::invalid_argument(concat!(
                            "invalid arguments to `gpu::",
                            stringify!($function_name),
                            "`",
                        ))
                    })
                })
            }
        }
    };
}

macro_rules! gpu_sparse_create_sp_mat_operation {
    ($name:ident, $function_name:ident, $operation_name:literal, $doc:literal, operands = { $($method:ident => $index:expr),+ $(,)* } $(,)*) => {
        paste::paste! {
            #[doc = $doc]
            pub trait [<$name Operation>]<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
                /// Returns async token dependencies.
                fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
                    let async_token_type = self.context().gpu_async_token_type()?.as_ref();
                    let mut dependencies = Vec::new();
                    for operand in self.operand_values() {
                        let operand = operand?;
                        if operand.r#type()? != async_token_type {
                            break;
                        }
                        dependencies.push(operand);
                    }
                    Ok(dependencies)
                }

                $(
                    #[doc = "Returns this operation operand."]
                    fn $method(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
                        self.operand_value(self.async_dependencies()?.len() + $index)
                    }
                )+

                /// Returns the sparse matrix handle result.
                fn sparse_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
                    Ok(self.result(0)?.as_ref())
                }

                /// Returns the optional async token result.
                fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
                    if self.result_count() <= 1 {
                        Ok(None)
                    } else {
                        self.result(1).map(|result| Some(result.as_ref()))
                    }
                }
            }

            mlir_op!($name);
            mlir_op_trait!($name, ZeroRegions);
            mlir_op_trait!($name, ZeroSuccessors);

            #[doc = "Constructs a new detached/owned [`"]
            #[doc = stringify!($name)]
            #[doc = "Operation`] at the specified [`Location`]."]
            pub fn $function_name<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
                async_dependencies: &[ValueRef<'o, 'c, 't>],
                $($method: ValueRef<'o, 'c, 't>,)+
                is_async: bool,
                location: L,
            ) -> Result<[<Detached $name Operation>]<'c, 't>, Error> {
                let context = location.context();
                context.load_dialect(DialectHandle::gpu()?);
                let mut builder = OperationBuilder::new($operation_name, location)
                    .add_operands(async_dependencies)
                    $(.add_operand($method))+
                    .add_result(context.gpu_sparse_sp_mat_handle_type()?);
                if is_async {
                    builder = builder.add_result(context.gpu_async_token_type()?);
                }
                builder.build().and_then(|operation| unsafe {
                    operation.cast().ok_or_else(|| {
                        Error::invalid_argument(concat!(
                            "invalid arguments to `gpu::",
                            stringify!($function_name),
                            "`",
                        ))
                    })
                })
            }
        }
    };
}

/// GPU operation that creates a dense tensor sparse handle.
pub trait CreateDnTensorOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dependency_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        self.operand_values().take(dependency_count).collect()
    }

    /// Returns the dense tensor backing memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns dense tensor dimension operands.
    fn dimensions(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 2)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the dense tensor handle result.
    fn dense_tensor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(CreateDnTensor);
mlir_op_trait!(CreateDnTensor, ZeroRegions);
mlir_op_trait!(CreateDnTensor, ZeroSuccessors);

/// Constructs a new detached/owned [`CreateDnTensorOperation`] at the specified [`Location`].
pub fn create_dn_tensor<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    memref: ValueRef<'o, 'c, 't>,
    dimensions: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> Result<DetachedCreateDnTensorOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.create_dn_tensor", location)
        .add_operands(async_dependencies)
        .add_operand(memref)
        .add_operands(dimensions)
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[async_dependencies.len() as i32, 1, dimensions.len() as i32])?,
        )
        .add_result(context.gpu_sparse_dn_tensor_handle_type()?);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::create_dn_tensor`"))
    })
}

gpu_sparse_async_operation!(
    DestroyDnTensor,
    destroy_dn_tensor,
    "gpu.destroy_dn_tensor",
    "GPU operation that destroys a dense tensor sparse handle.",
    operands = { dense_tensor => 0 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCoo,
    create_coo,
    "gpu.create_coo",
    "GPU operation that creates a sparse matrix in COO format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, row_indices => 3, column_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCooAos,
    create_coo_aos,
    "gpu.create_coo_aos",
    "GPU operation that creates a sparse matrix in COO AoS format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, indices => 3, values => 4 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCsr,
    create_csr,
    "gpu.create_csr",
    "GPU operation that creates a sparse matrix in CSR format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, row_positions => 3, column_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateCsc,
    create_csc,
    "gpu.create_csc",
    "GPU operation that creates a sparse matrix in CSC format.",
    operands = { rows => 0, columns => 1, non_zero_count => 2, column_positions => 3, row_indices => 4, values => 5 },
);

gpu_sparse_create_sp_mat_operation!(
    CreateBsr,
    create_bsr,
    "gpu.create_bsr",
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
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns the row count operand.
    fn rows(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns the column count operand.
    fn columns(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns the pruning flag.
    fn prune_flag(&self) -> Result<Prune2To4SparseMatrixFlag, Error> {
        self.attribute(PRUNE_FLAG_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<Prune2To4SparseMatrixFlagAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    PRUNE_FLAG_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the dense backing memref.
    fn memref(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the sparse matrix result.
    fn sparse_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(Create2To4SpMat);
mlir_op_trait!(Create2To4SpMat, ZeroRegions);
mlir_op_trait!(Create2To4SpMat, ZeroSuccessors);

/// Constructs a new detached/owned [`Create2To4SpMatOperation`] at the specified [`Location`].
pub fn create_2_to_4_sp_mat<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    rows: ValueRef<'o, 'c, 't>,
    columns: ValueRef<'o, 'c, 't>,
    prune_flag: Prune2To4SparseMatrixFlag,
    memref: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedCreate2To4SpMatOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.create_2to4_spmat", location)
        .add_operands(async_dependencies)
        .add_operand(rows)
        .add_operand(columns)
        .add_operand(memref)
        .add_attribute(PRUNE_FLAG_ATTRIBUTE, context.gpu_prune_2_to_4_sparse_matrix_flag_attribute(prune_flag)?)
        .add_result(context.gpu_sparse_sp_mat_handle_type()?);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::create_2_to_4_sp_mat`"))
    })
}

gpu_sparse_async_operation!(
    DestroySpMat,
    destroy_sp_mat,
    "gpu.destroy_sp_mat",
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
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns dense tensor X.
    fn dense_tensor_x(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns dense tensor Y.
    fn dense_tensor_y(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the buffer-size result.
    fn buffer_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SpmvBufferSize);
mlir_op_trait!(SpmvBufferSize, ZeroRegions);
mlir_op_trait!(SpmvBufferSize, ZeroSuccessors);

/// Constructs a new detached/owned [`SpmvBufferSizeOperation`] at the specified [`Location`].
pub fn spmv_buffer_size<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    dense_tensor_x: ValueRef<'o, 'c, 't>,
    dense_tensor_y: ValueRef<'o, 'c, 't>,
    compute_type: T,
    is_async: bool,
    location: L,
) -> Result<DetachedSpmvBufferSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spmv_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(sparse_matrix_a)
        .add_operand(dense_tensor_x)
        .add_operand(dense_tensor_y)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type))
        .add_result(context.index_type());
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::spmv_buffer_size`"))
    })
}

/// GPU SpMV compute operation.
pub trait SpmvOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns dense tensor X.
    fn dense_tensor_x(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns dense tensor Y.
    fn dense_tensor_y(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the temporary buffer.
    fn buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 3)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(Spmv);
mlir_op_trait!(Spmv, ZeroRegions);
mlir_op_trait!(Spmv, ZeroSuccessors);

/// Constructs a new detached/owned [`SpmvOperation`] at the specified [`Location`].
pub fn spmv<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    dense_tensor_x: ValueRef<'o, 'c, 't>,
    dense_tensor_y: ValueRef<'o, 'c, 't>,
    compute_type: T,
    buffer: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedSpmvOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spmv", location)
        .add_operands(async_dependencies)
        .add_operand(sparse_matrix_a)
        .add_operand(dense_tensor_x)
        .add_operand(dense_tensor_y)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type));
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::spmv`"))
    })
}

/// Name of the MLIR attribute storing operation result segment sizes.
pub const RESULT_SEGMENT_SIZES_ATTRIBUTE: &str = "result_segment_sizes";

/// GPU SpMM buffer-size operation.
pub trait SpmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns dense matrix C.
    fn dense_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the buffer-size results.
    fn buffer_sizes(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(RESULT_SEGMENT_SIZES_ATTRIBUTE, 0)?
            .map(|index| self.result(index).map(|result| result.as_ref()))
            .collect()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        let range = self.dense_integer_32_array_attribute_segment_range(RESULT_SEGMENT_SIZES_ATTRIBUTE, 1)?;
        match range.len() {
            0 => Ok(None),
            1 => self.result(range.start).map(|result| Some(result.as_ref())),
            _ => Err(Error::invalid_argument(format!(
                "invalid `{}` attribute in `{}`",
                RESULT_SEGMENT_SIZES_ATTRIBUTE,
                self.name(),
            ))),
        }
    }
}

mlir_op!(SpmmBufferSize);
mlir_op_trait!(SpmmBufferSize, ZeroRegions);
mlir_op_trait!(SpmmBufferSize, ZeroSuccessors);

/// Constructs a new detached/owned [`SpmmBufferSizeOperation`] at the specified [`Location`].
pub fn spmm_buffer_size<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    dense_matrix_b: ValueRef<'o, 'c, 't>,
    dense_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    buffer_size_count: usize,
    is_async: bool,
    location: L,
) -> Result<DetachedSpmmBufferSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spmm_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(sparse_matrix_a)
        .add_operand(dense_matrix_b)
        .add_operand(dense_matrix_c)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type))
        .add_attribute(
            RESULT_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[buffer_size_count as i32, if is_async { 1 } else { 0 }])?,
        );
    for _ in 0..buffer_size_count {
        builder = builder.add_result(context.index_type());
    }
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::spmm_buffer_size`"))
    })
}

/// GPU SpMM compute operation.
pub trait SpmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let dependency_count = self.dense_integer_32_array_attribute_usize_value(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 0)?;
        self.operand_values().take(dependency_count).collect()
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix A.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns dense matrix C.
    fn dense_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
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

    /// Returns temporary buffers.
    fn buffers(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.dense_integer_32_array_attribute_segment_range(OPERAND_SEGMENT_SIZES_ATTRIBUTE, 4)?
            .map(|index| self.operand_value(index))
            .collect()
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(Spmm);
mlir_op_trait!(Spmm, ZeroRegions);
mlir_op_trait!(Spmm, ZeroSuccessors);

/// Constructs a new detached/owned [`SpmmOperation`] at the specified [`Location`].
pub fn spmm<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    dense_matrix_b: ValueRef<'o, 'c, 't>,
    dense_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    buffers: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> Result<DetachedSpmmOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spmm", location)
        .add_operands(async_dependencies)
        .add_operand(sparse_matrix_a)
        .add_operand(dense_matrix_b)
        .add_operand(dense_matrix_c)
        .add_operands(buffers)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type))
        .add_attribute(
            OPERAND_SEGMENT_SIZES_ATTRIBUTE,
            context.dense_i32_array_attribute(&[async_dependencies.len() as i32, 1, 1, 1, buffers.len() as i32])?,
        );
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::spmm`"))
    })
}

/// GPU SDDMM buffer-size operation.
pub trait SddmmBufferSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns dense matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix A.
    fn dense_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns sparse matrix C.
    fn sparse_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the buffer-size result.
    fn buffer_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SddmmBufferSize);
mlir_op_trait!(SddmmBufferSize, ZeroRegions);
mlir_op_trait!(SddmmBufferSize, ZeroSuccessors);

/// Constructs a new detached/owned [`SddmmBufferSizeOperation`] at the specified [`Location`].
pub fn sddmm_buffer_size<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    dense_matrix_a: ValueRef<'o, 'c, 't>,
    dense_matrix_b: ValueRef<'o, 'c, 't>,
    sparse_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    is_async: bool,
    location: L,
) -> Result<DetachedSddmmBufferSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.sddmm_buffer_size", location)
        .add_operands(async_dependencies)
        .add_operand(dense_matrix_a)
        .add_operand(dense_matrix_b)
        .add_operand(sparse_matrix_c)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type))
        .add_result(context.index_type());
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sddmm_buffer_size`"))
    })
}

/// GPU SDDMM compute operation.
pub trait SddmmOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns dense matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns dense matrix A.
    fn dense_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns dense matrix B.
    fn dense_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns sparse matrix C.
    fn sparse_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the temporary buffer.
    fn buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 3)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(Sddmm);
mlir_op_trait!(Sddmm, ZeroRegions);
mlir_op_trait!(Sddmm, ZeroSuccessors);

/// Constructs a new detached/owned [`SddmmOperation`] at the specified [`Location`].
pub fn sddmm<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    dense_matrix_a: ValueRef<'o, 'c, 't>,
    dense_matrix_b: ValueRef<'o, 'c, 't>,
    sparse_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    buffer: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedSddmmOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.sddmm", location)
        .add_operands(async_dependencies)
        .add_operand(dense_matrix_a)
        .add_operand(dense_matrix_b)
        .add_operand(sparse_matrix_c)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type));
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sddmm`"))
    })
}

/// GPU operation that creates a SpGEMM descriptor.
pub trait SpGemmCreateDescrOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns the SpGEMM descriptor result.
    fn descriptor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SpGemmCreateDescr);
mlir_op_trait!(SpGemmCreateDescr, ZeroRegions);
mlir_op_trait!(SpGemmCreateDescr, ZeroSuccessors);

/// Constructs a new detached/owned [`SpGemmCreateDescrOperation`] at the specified [`Location`].
pub fn sp_gemm_create_descr<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    is_async: bool,
    location: L,
) -> Result<DetachedSpGemmCreateDescrOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spgemm_create_descr", location)
        .add_operands(async_dependencies)
        .add_result(context.gpu_sparse_sp_gemm_operation_handle_type()?);
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sp_gemm_create_descr`"))
    })
}

gpu_sparse_async_operation!(
    SpGemmDestroyDescr,
    sp_gemm_destroy_descr,
    "gpu.spgemm_destroy_descr",
    "GPU operation that destroys a SpGEMM descriptor.",
    operands = { descriptor => 0 },
);

/// Name of the [`Attribute`] that stores SpGEMM work kind.
pub const KIND_ATTRIBUTE: &str = "kind";

/// GPU SpGEMM work-estimation or compute operation.
pub trait SpGemmWorkEstimationOrComputeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns the SpGEMM descriptor operand.
    fn descriptor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the sparse matrix A operand.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns the sparse matrix B operand.
    fn sparse_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the sparse matrix C operand.
    fn sparse_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 3)
    }

    /// Returns the current buffer-size operand.
    fn buffer_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 4)
    }

    /// Returns the temporary buffer operand.
    fn buffer(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 5)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the SpGEMM work kind.
    fn kind(&self) -> Result<SpGemmWorkKind, Error> {
        self.attribute(KIND_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<SpGemmWorkKindAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    KIND_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the new buffer-size result.
    fn new_buffer_size(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 1 { Ok(None) } else { self.result(1).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SpGemmWorkEstimationOrCompute);
mlir_op_trait!(SpGemmWorkEstimationOrCompute, ZeroRegions);
mlir_op_trait!(SpGemmWorkEstimationOrCompute, ZeroSuccessors);

/// Constructs a new detached/owned [`SpGemmWorkEstimationOrComputeOperation`] at the specified [`Location`].
pub fn sp_gemm_work_estimation_or_compute<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    descriptor: ValueRef<'o, 'c, 't>,
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    sparse_matrix_b: ValueRef<'o, 'c, 't>,
    sparse_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    buffer_size: ValueRef<'o, 'c, 't>,
    buffer: ValueRef<'o, 'c, 't>,
    kind: SpGemmWorkKind,
    is_async: bool,
    location: L,
) -> Result<DetachedSpGemmWorkEstimationOrComputeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spgemm_work_estimation_or_compute", location)
        .add_operands(async_dependencies)
        .add_operand(descriptor)
        .add_operand(sparse_matrix_a)
        .add_operand(sparse_matrix_b)
        .add_operand(sparse_matrix_c)
        .add_operand(buffer_size)
        .add_operand(buffer)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type))
        .add_attribute(KIND_ATTRIBUTE, context.gpu_sp_gemm_work_kind_attribute(kind)?)
        .add_result(context.index_type());
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sp_gemm_work_estimation_or_compute`"))
    })
}

/// GPU SpGEMM copy operation.
pub trait SpGemmCopyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns sparse matrix A transpose mode.
    fn mode_a(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_A_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_A_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns sparse matrix B transpose mode.
    fn mode_b(&self) -> Result<MatrixTransposeMode, Error> {
        self.attribute(MODE_B_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<MatrixTransposeModeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    MODE_B_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the SpGEMM descriptor operand.
    fn descriptor(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns the sparse matrix A operand.
    fn sparse_matrix_a(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 1)
    }

    /// Returns the sparse matrix B operand.
    fn sparse_matrix_b(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 2)
    }

    /// Returns the sparse matrix C operand.
    fn sparse_matrix_c(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len() + 3)
    }

    /// Returns the compute type attribute.
    fn compute_type(&self) -> Result<TypeRef<'c, 't>, Error> {
        self.type_attribute(COMPUTE_TYPE_ATTRIBUTE)?.r#type()
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 0 { Ok(None) } else { self.result(0).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SpGemmCopy);
mlir_op_trait!(SpGemmCopy, ZeroRegions);
mlir_op_trait!(SpGemmCopy, ZeroSuccessors);

/// Constructs a new detached/owned [`SpGemmCopyOperation`] at the specified [`Location`].
pub fn sp_gemm_copy<'o, 'c: 'o, 't: 'c, T: Type<'c, 't>, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    descriptor: ValueRef<'o, 'c, 't>,
    mode_a: MatrixTransposeMode,
    mode_b: MatrixTransposeMode,
    sparse_matrix_a: ValueRef<'o, 'c, 't>,
    sparse_matrix_b: ValueRef<'o, 'c, 't>,
    sparse_matrix_c: ValueRef<'o, 'c, 't>,
    compute_type: T,
    is_async: bool,
    location: L,
) -> Result<DetachedSpGemmCopyOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spgemm_copy", location)
        .add_operands(async_dependencies)
        .add_operand(descriptor)
        .add_operand(sparse_matrix_a)
        .add_operand(sparse_matrix_b)
        .add_operand(sparse_matrix_c)
        .add_attribute(MODE_A_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_a)?)
        .add_attribute(MODE_B_ATTRIBUTE, context.gpu_matrix_transpose_mode_attribute(mode_b)?)
        .add_attribute(COMPUTE_TYPE_ATTRIBUTE, context.type_attribute(compute_type));
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sp_gemm_copy`"))
    })
}

/// GPU sparse matrix get-size operation.
pub trait SpMatGetSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns async token dependencies.
    fn async_dependencies(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        let async_token_type = self.context().gpu_async_token_type()?.as_ref();
        let mut dependencies = Vec::new();
        for operand in self.operand_values() {
            let operand = operand?;
            if operand.r#type()? != async_token_type {
                break;
            }
            dependencies.push(operand);
        }
        Ok(dependencies)
    }

    /// Returns the sparse matrix operand.
    fn sparse_matrix(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(self.async_dependencies()?.len())
    }

    /// Returns the sparse matrix row-count result.
    fn rows(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }

    /// Returns the sparse matrix column-count result.
    fn columns(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(1)?.as_ref())
    }

    /// Returns the sparse matrix non-zero-count result.
    fn non_zero_count(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(2)?.as_ref())
    }

    /// Returns the optional async token result.
    fn async_token(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.result_count() <= 3 { Ok(None) } else { self.result(3).map(|result| Some(result.as_ref())) }
    }
}

mlir_op!(SpMatGetSize);
mlir_op_trait!(SpMatGetSize, ZeroRegions);
mlir_op_trait!(SpMatGetSize, ZeroSuccessors);

/// Constructs a new detached/owned [`SpMatGetSizeOperation`] at the specified [`Location`].
pub fn sp_mat_get_size<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    async_dependencies: &[ValueRef<'o, 'c, 't>],
    sparse_matrix: ValueRef<'o, 'c, 't>,
    is_async: bool,
    location: L,
) -> Result<DetachedSpMatGetSizeOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.spmat_get_size", location)
        .add_operands(async_dependencies)
        .add_operand(sparse_matrix)
        .add_result(context.index_type())
        .add_result(context.index_type())
        .add_result(context.index_type());
    if is_async {
        builder = builder.add_result(context.gpu_async_token_type()?);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::sp_mat_get_size`"))
    })
}

gpu_sparse_async_operation!(
    SetCsrPointers,
    set_csr_pointers,
    "gpu.set_csr_pointers",
    "GPU operation that sets CSR pointers for a sparse matrix.",
    operands = { sparse_matrix => 0, positions => 1, coordinates => 2, values => 3 },
);

/// Name of the [`Attribute`] that stores warp size.
pub const WARP_SIZE_ATTRIBUTE: &str = "warp_size";

/// GPU operation that bridges vector code and SIMT execution by running a region on lane 0.
pub trait WarpExecuteOnLane0Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> {
    /// Returns the lane identifier operand.
    fn lane_id(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the warp size attribute.
    fn warp_size(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        self.integer_attribute(WARP_SIZE_ATTRIBUTE)
    }

    /// Returns operands passed into the lane-0 region.
    fn arguments(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.operand_values().skip(1).collect()
    }

    /// Returns values produced for the surrounding SIMT region.
    fn outputs(&self) -> Result<Vec<ValueRef<'o, 'c, 't>>, Error> {
        self.results().map(|result| result.map(|result| result.as_ref())).collect()
    }

    /// Returns the lane-0 region.
    fn region(&self) -> Result<RegionRef<'o, 'c, 't>, Error> {
        self.body_region()
    }
}

mlir_op!(WarpExecuteOnLane0);
mlir_op_trait!(WarpExecuteOnLane0, OneRegion);
mlir_op_trait!(WarpExecuteOnLane0, SingleBlockRegions);
mlir_op_trait!(WarpExecuteOnLane0, ZeroSuccessors);

/// Constructs a new detached/owned [`WarpExecuteOnLane0Operation`] at the specified [`Location`].
pub fn warp_execute_on_lane_0<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    lane_id: ValueRef<'o, 'c, 't>,
    warp_size: i64,
    arguments: &[ValueRef<'o, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    region: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedWarpExecuteOnLane0Operation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.warp_execute_on_lane_0", location)
        .add_operand(lane_id)
        .add_operands(arguments)
        .add_attribute(WARP_SIZE_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), warp_size))
        .add_results(result_types)
        .add_region(region)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::warp_execute_on_lane_0`"))
        })
}

/// Name of the [`Attribute`] that stores subgroup broadcast type.
pub const BROADCAST_TYPE_ATTRIBUTE: &str = "broadcast_type";

/// GPU subgroup broadcast operation.
pub trait SubgroupBroadcastOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the source value.
    fn source(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the optional lane operand.
    fn lane(&self) -> Result<Option<ValueRef<'o, 'c, 't>>, Error> {
        if self.operand_count() <= 1 { Ok(None) } else { self.operand_value(1).map(Some) }
    }

    /// Returns the broadcast type.
    fn broadcast_type(&self) -> Result<BroadcastType, Error> {
        self.attribute(BROADCAST_TYPE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BroadcastTypeAttributeRef>())
            .ok_or_else(|| {
                Error::invalid_argument(format!(
                    "missing or invalid `{}` attribute in `{}`",
                    BROADCAST_TYPE_ATTRIBUTE,
                    self.name().as_str().unwrap_or("<unknown>"),
                ))
            })?
            .value()
    }

    /// Returns the broadcast result.
    fn output(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
    }
}

mlir_op!(SubgroupBroadcast);
mlir_op_trait!(SubgroupBroadcast, OneResult);
mlir_op_trait!(SubgroupBroadcast, ZeroRegions);
mlir_op_trait!(SubgroupBroadcast, ZeroSuccessors);

/// Constructs a new detached/owned [`SubgroupBroadcastOperation`] at the specified [`Location`].
pub fn subgroup_broadcast<'o, 'c: 'o, 't: 'c, L: Location<'c, 't>>(
    source: ValueRef<'o, 'c, 't>,
    lane: Option<ValueRef<'o, 'c, 't>>,
    broadcast_type: BroadcastType,
    location: L,
) -> Result<DetachedSubgroupBroadcastOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::gpu()?);
    let mut builder = OperationBuilder::new("gpu.subgroup_broadcast", location)
        .add_operand(source)
        .add_attribute(BROADCAST_TYPE_ATTRIBUTE, context.gpu_broadcast_type_attribute(broadcast_type)?)
        .add_result(source.r#type()?);
    if let Some(lane) = lane {
        builder = builder.add_operand(lane);
    }
    builder.build().and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::subgroup_broadcast`"))
    })
}

/// GPU subgroup ballot operation.
pub trait BallotOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the predicate operand.
    fn predicate(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the ballot mask result.
    fn mask(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        Ok(self.result(0)?.as_ref())
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
) -> Result<DetachedBallotOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::gpu()?);
    OperationBuilder::new("gpu.ballot", location)
        .add_operand(predicate)
        .add_result(result_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `gpu::ballot`"))
        })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::dialects::gpu::attributes::ObjectFormat;
    use crate::dialects::gpu::types::MmaMatrixOperand;
    use crate::{Attribute, Block, Context, OneResult, Operation, Region, Size, Type};

    use super::*;

    macro_rules! gpu_dimension_operation_test {
        ($test_name:ident, $function_name:ident, $operation_name:literal, $dimension:expr $(,)?) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let operation = $function_name($dimension, Some(128), location).unwrap();

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.dimension().unwrap(), $dimension);
                assert_eq!(operation.upper_bound().unwrap().unwrap().signless_value(), 128);
                assert_eq!(operation.output_type().unwrap(), context.index_type());
            }
        };
    }

    macro_rules! gpu_upper_bound_index_operation_test {
        ($test_name:ident, $function_name:ident, $operation_name:literal $(,)?) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let operation = $function_name(Some(128), location).unwrap();

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.upper_bound().unwrap().unwrap().signless_value(), 128);
                assert_eq!(operation.output_type().unwrap(), context.index_type());
            }
        };
    }

    macro_rules! gpu_sparse_async_operation_test {
        (
            $test_name:ident,
            $function_name:ident,
            $operation_name:literal,
            operand_count = $operand_count:expr,
            operands = { $($method:ident => $index:expr),* $(,)* } $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let token_type = context.gpu_async_token_type().unwrap().as_ref();
                let index_type = context.index_type().as_ref();
                let mut arguments = vec![(token_type, location)];
                arguments.extend((0..$operand_count).map(|_| (index_type, location)));
                let block = context.block(arguments.as_slice());
                let token = block.argument(0).unwrap().as_ref();
                let operands = (1..=$operand_count)
                    .map(|index| block.argument(index).unwrap().as_ref())
                    .collect::<Vec<_>>();
                let operation = $function_name(&[token], $(operands[$index],)* true, location).unwrap();

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
                $(assert_eq!(operation.$method().unwrap(), operands[$index]);)*
                assert!(operation.async_token().unwrap().is_some());
            }
        };
    }

    macro_rules! gpu_sparse_create_sp_mat_operation_test {
        (
            $test_name:ident,
            $function_name:ident,
            $operation_name:literal,
            operand_count = $operand_count:expr,
            operands = { $($method:ident => $index:expr),+ $(,)* } $(,)*
        ) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let location = context.unknown_location();
                let token_type = context.gpu_async_token_type().unwrap().as_ref();
                let index_type = context.index_type().as_ref();
                let mut arguments = vec![(token_type, location)];
                arguments.extend((0..$operand_count).map(|_| (index_type, location)));
                let block = context.block(arguments.as_slice());
                let token = block.argument(0).unwrap().as_ref();
                let operands = (1..=$operand_count)
                    .map(|index| block.argument(index).unwrap().as_ref())
                    .collect::<Vec<_>>();
                let operation = $function_name(&[token], $(operands[$index],)+ true, location).unwrap();

                assert_eq!(operation.name().as_str(), Ok($operation_name));
                assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
                $(assert_eq!(operation.$method().unwrap(), operands[$index]);)+
                assert_eq!(
                    operation.sparse_matrix().unwrap().r#type().unwrap(),
                    context.gpu_sparse_sp_mat_handle_type().unwrap()
                );
                assert!(operation.async_token().unwrap().is_some());
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
        let operation = func(
            "kernel",
            FuncProperties {
                is_kernel: true,
                known_block_size: Some([1, 2, 3]),
                known_grid_size: Some([4, 5, 6]),
                known_cluster_size: Some([7, 8, 9]),
                workgroup_attribution_count: 2,
                ..Default::default()
            },
            context.region(),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.func"));
        assert_eq!(operation.function_name().unwrap().as_str(), Ok("kernel"));
        assert!(operation.is_kernel());
        assert_eq!(operation.known_block_size().unwrap(), Some(vec![1, 2, 3]));
        assert_eq!(operation.known_grid_size().unwrap(), Some(vec![4, 5, 6]));
        assert_eq!(operation.known_cluster_size().unwrap(), Some(vec![7, 8, 9]));
        assert_eq!(operation.workgroup_attribution_count().unwrap(), 2);
        assert_eq!(operation.body().unwrap().blocks().count(), 0);
    }

    #[test]
    fn test_dynamic_shared_memory_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let operation = dynamic_shared_memory(memref_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.dynamic_shared_memory"));
        assert_eq!(operation.memref().unwrap().r#type().unwrap(), memref_type);
        assert_eq!(operation.output_type().unwrap(), memref_type);
    }

    #[test]
    fn test_launch_func_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
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
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.launch_func"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.kernel().unwrap(), kernel);
        assert_eq!(operation.grid_size().unwrap(), grid_size);
        assert_eq!(operation.block_size().unwrap(), block_size);
        assert_eq!(operation.cluster_size().unwrap(), Some(cluster_size));
        assert_eq!(operation.dynamic_shared_memory_size().unwrap(), Some(dynamic_shared_memory_size));
        assert_eq!(operation.kernel_operands().unwrap(), vec![kernel_operand]);
        assert_eq!(operation.async_object().unwrap(), Some(token));
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_launch_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
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
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.launch"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.grid_size().unwrap(), grid_size);
        assert_eq!(operation.block_size().unwrap(), block_size);
        assert_eq!(operation.cluster_size().unwrap(), Some(cluster_size));
        assert_eq!(operation.dynamic_shared_memory_size().unwrap(), Some(dynamic_shared_memory_size));
        assert_eq!(operation.module_symbol().unwrap(), Some(module));
        assert_eq!(operation.function_symbol().unwrap(), Some(function));
        assert_eq!(operation.workgroup_attribution_count().unwrap(), 2);
        assert_eq!(operation.body().unwrap().blocks().count(), 0);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_printf_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location)]);
        let argument = block.argument(0).unwrap();
        let operation = printf("value: %d", &[argument], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.printf"));
        assert_eq!(operation.format().unwrap().as_str(), Ok("value: %d"));
        assert_eq!(operation.arguments().unwrap(), vec![argument]);
    }

    #[test]
    fn test_return_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation = r#return(&[value], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.return"));
        assert_eq!(operation.values().unwrap(), vec![value]);
    }

    #[test]
    fn test_terminator_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = terminator(location).unwrap();

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
        let operation = r#yield(&[value], location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.yield"));
        assert_eq!(operation.values().unwrap(), vec![value]);
    }

    #[test]
    fn test_all_reduce_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation = all_reduce(
            value,
            Some(AllReduceOperationKind::Add),
            true,
            context.region(),
            value.r#type().unwrap(),
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.all_reduce"));
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.operation_kind().unwrap(), Some(AllReduceOperationKind::Add));
        assert!(operation.is_uniform());
        assert_eq!(operation.body().unwrap().blocks().count(), 0);
        assert_eq!(operation.result_count(), 1);
    }

    #[test]
    fn test_subgroup_reduce_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(32), location)]);
        let value = block.argument(0).unwrap();
        let operation =
            subgroup_reduce(value, AllReduceOperationKind::MaximumSignedInteger, true, Some(4), 2, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_reduce"));
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.operation_kind().unwrap(), AllReduceOperationKind::MaximumSignedInteger);
        assert!(operation.is_uniform());
        assert_eq!(operation.cluster_size().unwrap().unwrap().signless_value(), 4);
        assert_eq!(operation.cluster_stride().unwrap().signless_value(), 2);
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
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.shuffle"));
        assert_eq!(operation.value().unwrap(), block.argument(0).unwrap());
        assert_eq!(operation.offset().unwrap(), block.argument(1).unwrap());
        assert_eq!(operation.width().unwrap(), block.argument(2).unwrap());
        assert_eq!(operation.mode().unwrap(), ShuffleMode::Xor);
        assert_eq!(operation.value().unwrap(), block.argument(0).unwrap());
        assert_eq!(operation.offset().unwrap(), block.argument(1).unwrap());
    }

    #[test]
    fn test_rotate_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let i32_type = context.signless_integer_type(32);
        let block = context.block(&[(i32_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = rotate(value, 1, 32, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.rotate"));
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.offset().unwrap().signless_value(), 1);
        assert_eq!(operation.width().unwrap().signless_value(), 32);
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.offset().unwrap().signless_value(), 1);
    }

    #[test]
    fn test_barrier_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = barrier(Some(&[AddressSpace::Workgroup, AddressSpace::Private]), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.barrier"));
        assert_eq!(operation.address_spaces().unwrap(), Some(vec![AddressSpace::Workgroup, AddressSpace::Private]));
    }

    #[test]
    fn test_module_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let target = context.string_attribute("sm_90");
        let targets = context.array_attribute(&[target]);
        let offloading_handler = context.string_attribute("handler").as_ref();
        let operation = module("kernels", Some(targets), Some(offloading_handler), context.region(), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.module"));
        assert_eq!(operation.targets().unwrap(), Some(targets));
        assert_eq!(operation.offloading_handler(), Some(offloading_handler));
        assert_eq!(operation.region(0).unwrap().blocks().count(), 0);
    }

    #[test]
    fn test_binary_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let target = context.string_attribute("sm_90");
        let object = context.gpu_object_attribute(target, ObjectFormat::Binary, "object", None, None).unwrap();
        let objects = context.array_attribute(&[object]);
        let offloading_handler = context.string_attribute("handler").as_ref();
        let operation = binary("binary", objects, Some(offloading_handler), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.binary"));
        assert_eq!(operation.objects().unwrap(), objects);
        assert_eq!(operation.offloading_handler(), Some(offloading_handler));
    }

    #[test]
    fn test_host_register_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(memref_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = host_register(value, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.host_register"));
        assert_eq!(operation.value().unwrap(), value);
    }

    #[test]
    fn test_host_unregister_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(memref_type, location)]);
        let value = block.argument(0).unwrap();
        let operation = host_unregister(value, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.host_unregister"));
        assert_eq!(operation.value().unwrap(), value);
    }

    #[test]
    fn test_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let block = context.block(&[(token_type, location)]);
        let dependency = block.argument(0).unwrap();
        let operation = wait(&[dependency], true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.wait"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![dependency]);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_alloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (index_type, location), (index_type, location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let dynamic_size = block.argument(1).unwrap().as_ref();
        let symbol_operand = block.argument(2).unwrap().as_ref();
        let operation =
            alloc(&[dependency], &[dynamic_size], &[symbol_operand], memref_type, true, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.alloc"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![dependency]);
        assert_eq!(operation.dynamic_sizes().unwrap(), vec![dynamic_size]);
        assert_eq!(operation.symbol_operands().unwrap(), vec![symbol_operand]);
        assert!(operation.host_shared());
        assert_eq!(operation.memref().unwrap().r#type().unwrap(), memref_type);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_dealloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (memref_type.as_ref(), location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let memref = block.argument(1).unwrap().as_ref();
        let operation = dealloc(&[dependency], memref, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.dealloc"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![dependency]);
        assert_eq!(operation.memref().unwrap(), memref);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_memcpy_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[
            (token_type, location),
            (memref_type.as_ref(), location),
            (memref_type.as_ref(), location),
        ]);
        let dependency = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let source = block.argument(2).unwrap().as_ref();
        let operation = memcpy(&[dependency], destination, source, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.memcpy"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![dependency]);
        assert_eq!(operation.destination().unwrap(), destination);
        assert_eq!(operation.source().unwrap(), source);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_memset_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let block = context.block(&[(token_type, location), (memref_type.as_ref(), location), (index_type, location)]);
        let dependency = block.argument(0).unwrap().as_ref();
        let destination = block.argument(1).unwrap().as_ref();
        let value = block.argument(2).unwrap().as_ref();
        let operation = memset(&[dependency], destination, value, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.memset"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![dependency]);
        assert_eq!(operation.destination().unwrap(), destination);
        assert_eq!(operation.value().unwrap(), value);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_set_default_device_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.index_type(), location)]);
        let device_index = block.argument(0).unwrap();
        let operation = set_default_device(device_index, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.set_default_device"));
        assert_eq!(operation.device_index().unwrap(), device_index);
    }

    #[test]
    fn test_subgroup_mma_load_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let index_type = context.index_type().as_ref();
        let matrix_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A).unwrap();
        let block = context.block(&[(memref_type.as_ref(), location), (index_type, location), (index_type, location)]);
        let source_memref = block.argument(0).unwrap().as_ref();
        let index_0 = block.argument(1).unwrap().as_ref();
        let index_1 = block.argument(2).unwrap().as_ref();
        let operation =
            subgroup_mma_load_matrix(source_memref, &[index_0, index_1], 16, true, matrix_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_load_matrix"));
        assert_eq!(operation.source_memref().unwrap(), source_memref);
        assert_eq!(operation.indices().unwrap(), vec![index_0, index_1]);
        assert_eq!(operation.lead_dimension().unwrap().signless_value(), 16);
        assert!(operation.transpose());
        assert_eq!(operation.matrix().unwrap().r#type().unwrap(), matrix_type);
    }

    #[test]
    fn test_subgroup_mma_store_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let memref_type = context.mem_ref_type(context.float32_type(), &[Size::Dynamic], None, None, location).unwrap();
        let index_type = context.index_type().as_ref();
        let matrix_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let block = context.block(&[
            (matrix_type.as_ref(), location),
            (memref_type.as_ref(), location),
            (index_type, location),
            (index_type, location),
        ]);
        let source = block.argument(0).unwrap().as_ref();
        let destination_memref = block.argument(1).unwrap().as_ref();
        let index_0 = block.argument(2).unwrap().as_ref();
        let index_1 = block.argument(3).unwrap().as_ref();
        let operation =
            subgroup_mma_store_matrix(source, destination_memref, &[index_0, index_1], 16, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_store_matrix"));
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.destination_memref().unwrap(), destination_memref);
        assert_eq!(operation.indices().unwrap(), vec![index_0, index_1]);
        assert_eq!(operation.lead_dimension().unwrap().signless_value(), 16);
        assert!(operation.transpose());
    }

    #[test]
    fn test_subgroup_mma_compute_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let a_type = context.gpu_mma_matrix_type([16, 8], context.float32_type(), MmaMatrixOperand::A).unwrap();
        let b_type = context.gpu_mma_matrix_type([8, 16], context.float32_type(), MmaMatrixOperand::B).unwrap();
        let c_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let block =
            context.block(&[(a_type.as_ref(), location), (b_type.as_ref(), location), (c_type.as_ref(), location)]);
        let a = block.argument(0).unwrap().as_ref();
        let b = block.argument(1).unwrap().as_ref();
        let c = block.argument(2).unwrap().as_ref();
        let operation = subgroup_mma_compute(a, b, c, true, true, c_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_compute"));
        assert_eq!(operation.a().unwrap(), a);
        assert_eq!(operation.b().unwrap(), b);
        assert_eq!(operation.c().unwrap(), c);
        assert!(operation.a_transpose());
        assert!(operation.b_transpose());
        assert_eq!(operation.result_matrix().unwrap().r#type().unwrap(), c_type);
    }

    #[test]
    fn test_subgroup_mma_constant_matrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let block = context.block(&[(context.float32_type().as_ref(), location)]);
        let value = block.argument(0).unwrap().as_ref();
        let operation = subgroup_mma_constant_matrix(value, matrix_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_constant_matrix"));
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.value().unwrap(), value);
    }

    #[test]
    fn test_subgroup_mma_extract_thread_local_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[(matrix_type.as_ref(), location), (index_type, location), (index_type, location)]);
        let matrix = block.argument(0).unwrap().as_ref();
        let index_0 = block.argument(1).unwrap().as_ref();
        let index_1 = block.argument(2).unwrap().as_ref();
        let operation =
            subgroup_mma_extract_thread_local(matrix, &[index_0, index_1], context.float32_type(), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_extract_thread_local"));
        assert_eq!(operation.matrix().unwrap(), matrix);
        assert_eq!(operation.indices().unwrap(), vec![index_0, index_1]);
        assert_eq!(operation.value().unwrap().r#type().unwrap(), context.float32_type());
    }

    #[test]
    fn test_subgroup_mma_insert_thread_local_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (context.float32_type().as_ref(), location),
            (matrix_type.as_ref(), location),
            (index_type, location),
            (index_type, location),
        ]);
        let value = block.argument(0).unwrap().as_ref();
        let matrix = block.argument(1).unwrap().as_ref();
        let index_0 = block.argument(2).unwrap().as_ref();
        let index_1 = block.argument(3).unwrap().as_ref();
        let operation =
            subgroup_mma_insert_thread_local(value, matrix, &[index_0, index_1], matrix_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_insert_thread_local"));
        assert_eq!(operation.value().unwrap(), value);
        assert_eq!(operation.matrix().unwrap(), matrix);
        assert_eq!(operation.indices().unwrap(), vec![index_0, index_1]);
        assert_eq!(operation.result_matrix().unwrap().r#type().unwrap(), matrix_type);
    }

    #[test]
    fn test_subgroup_mma_elementwise_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let matrix_type = context.gpu_mma_matrix_type([16, 16], context.float32_type(), MmaMatrixOperand::C).unwrap();
        let block = context.block(&[(matrix_type.as_ref(), location), (matrix_type.as_ref(), location)]);
        let lhs = block.argument(0).unwrap().as_ref();
        let rhs = block.argument(1).unwrap().as_ref();
        let operation =
            subgroup_mma_elementwise(&[lhs, rhs], MmaElementwiseOperation::AddFloat, matrix_type, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_mma_elementwise"));
        assert_eq!(operation.arguments().unwrap(), vec![lhs, rhs]);
        assert_eq!(operation.operation().unwrap(), MmaElementwiseOperation::AddFloat);
        assert_eq!(operation.operation().unwrap(), MmaElementwiseOperation::AddFloat);
    }

    #[test]
    fn test_create_dn_tensor_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let memref = block.argument(1).unwrap().as_ref();
        let dimension = block.argument(2).unwrap().as_ref();
        let stride = block.argument(3).unwrap().as_ref();
        let operation = create_dn_tensor(&[token], memref, &[dimension, stride], true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.create_dn_tensor"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.memref().unwrap(), memref);
        assert_eq!(operation.dimensions().unwrap(), vec![dimension, stride]);
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert!(operation.async_token().unwrap().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_destroy_dn_tensor_operation,
        destroy_dn_tensor,
        "gpu.destroy_dn_tensor",
        operand_count = 1,
        operands = { dense_tensor => 0 },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_coo_operation,
        create_coo,
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
        create_coo_aos,
        "gpu.create_coo_aos",
        operand_count = 5,
        operands = { rows => 0, columns => 1, non_zero_count => 2, indices => 3, values => 4 },
    );

    gpu_sparse_create_sp_mat_operation_test!(
        test_create_csr_operation,
        create_csr,
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
        create_csc,
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
        create_bsr,
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
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let rows = block.argument(1).unwrap().as_ref();
        let columns = block.argument(2).unwrap().as_ref();
        let memref = block.argument(3).unwrap().as_ref();
        let operation = create_2_to_4_sp_mat(
            &[token],
            rows,
            columns,
            Prune2To4SparseMatrixFlag::PruneAndCheck,
            memref,
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.create_2to4_spmat"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.rows().unwrap(), rows);
        assert_eq!(operation.columns().unwrap(), columns);
        assert_eq!(operation.prune_flag().unwrap(), Prune2To4SparseMatrixFlag::PruneAndCheck);
        assert_eq!(operation.memref().unwrap(), memref);
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert!(operation.async_token().unwrap().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_destroy_sp_mat_operation,
        destroy_sp_mat,
        "gpu.destroy_sp_mat",
        operand_count = 1,
        operands = { sparse_matrix => 0 },
    );

    #[test]
    fn test_spmv_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let sparse_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_tensor_x = block.argument(2).unwrap().as_ref();
        let dense_tensor_y = block.argument(3).unwrap().as_ref();
        let operation = spmv_buffer_size(
            &[token],
            MatrixTransposeMode::Transpose,
            sparse_matrix_a,
            dense_tensor_x,
            dense_tensor_y,
            context.float32_type(),
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spmv_buffer_size"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.dense_tensor_x().unwrap(), dense_tensor_x);
        assert_eq!(operation.dense_tensor_y().unwrap(), dense_tensor_y);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_spmv_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let sparse_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_tensor_x = block.argument(2).unwrap().as_ref();
        let dense_tensor_y = block.argument(3).unwrap().as_ref();
        let buffer = block.argument(4).unwrap().as_ref();
        let operation = spmv(
            &[token],
            MatrixTransposeMode::NonTranspose,
            sparse_matrix_a,
            dense_tensor_x,
            dense_tensor_y,
            context.float32_type(),
            buffer,
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spmv"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.dense_tensor_x().unwrap(), dense_tensor_x);
        assert_eq!(operation.dense_tensor_y().unwrap(), dense_tensor_y);
        assert_eq!(operation.buffer().unwrap(), buffer);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_spmm_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let sparse_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_matrix_b = block.argument(2).unwrap().as_ref();
        let dense_matrix_c = block.argument(3).unwrap().as_ref();
        let operation = spmm_buffer_size(
            &[token],
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            sparse_matrix_a,
            dense_matrix_b,
            dense_matrix_c,
            context.float32_type(),
            2,
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spmm_buffer_size"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.dense_matrix_b().unwrap(), dense_matrix_b);
        assert_eq!(operation.dense_matrix_c().unwrap(), dense_matrix_c);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert_eq!(operation.buffer_sizes().unwrap().len(), 2);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_spmm_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let sparse_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_matrix_b = block.argument(2).unwrap().as_ref();
        let dense_matrix_c = block.argument(3).unwrap().as_ref();
        let buffer_0 = block.argument(4).unwrap().as_ref();
        let buffer_1 = block.argument(5).unwrap().as_ref();
        let operation = spmm(
            &[token],
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            sparse_matrix_a,
            dense_matrix_b,
            dense_matrix_c,
            context.float32_type(),
            &[buffer_0, buffer_1],
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spmm"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.dense_matrix_b().unwrap(), dense_matrix_b);
        assert_eq!(operation.dense_matrix_c().unwrap(), dense_matrix_c);
        assert_eq!(operation.buffers().unwrap(), vec![buffer_0, buffer_1]);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_sddmm_buffer_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let dense_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_matrix_b = block.argument(2).unwrap().as_ref();
        let sparse_matrix_c = block.argument(3).unwrap().as_ref();
        let operation = sddmm_buffer_size(
            &[token],
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            dense_matrix_a,
            dense_matrix_b,
            sparse_matrix_c,
            context.float32_type(),
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.sddmm_buffer_size"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.dense_matrix_a().unwrap(), dense_matrix_a);
        assert_eq!(operation.dense_matrix_b().unwrap(), dense_matrix_b);
        assert_eq!(operation.sparse_matrix_c().unwrap(), sparse_matrix_c);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_sddmm_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let dense_matrix_a = block.argument(1).unwrap().as_ref();
        let dense_matrix_b = block.argument(2).unwrap().as_ref();
        let sparse_matrix_c = block.argument(3).unwrap().as_ref();
        let buffer = block.argument(4).unwrap().as_ref();
        let operation = sddmm(
            &[token],
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            dense_matrix_a,
            dense_matrix_b,
            sparse_matrix_c,
            context.float32_type(),
            buffer,
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.sddmm"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.dense_matrix_a().unwrap(), dense_matrix_a);
        assert_eq!(operation.dense_matrix_b().unwrap(), dense_matrix_b);
        assert_eq!(operation.sparse_matrix_c().unwrap(), sparse_matrix_c);
        assert_eq!(operation.buffer().unwrap(), buffer);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_sp_gemm_create_descr_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let block = context.block(&[(token_type, location)]);
        let token = block.argument(0).unwrap().as_ref();
        let operation = sp_gemm_create_descr(&[token], true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_create_descr"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(
            operation.descriptor().unwrap().r#type().unwrap(),
            context.gpu_sparse_sp_gemm_operation_handle_type().unwrap()
        );
        assert!(operation.async_token().unwrap().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_sp_gemm_destroy_descr_operation,
        sp_gemm_destroy_descr,
        "gpu.spgemm_destroy_descr",
        operand_count = 1,
        operands = { descriptor => 0 },
    );

    #[test]
    fn test_sp_gemm_work_estimation_or_compute_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
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
        let token = block.argument(0).unwrap().as_ref();
        let descriptor = block.argument(1).unwrap().as_ref();
        let sparse_matrix_a = block.argument(2).unwrap().as_ref();
        let sparse_matrix_b = block.argument(3).unwrap().as_ref();
        let sparse_matrix_c = block.argument(4).unwrap().as_ref();
        let buffer_size = block.argument(5).unwrap().as_ref();
        let buffer = block.argument(6).unwrap().as_ref();
        let operation = sp_gemm_work_estimation_or_compute(
            &[token],
            descriptor,
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            sparse_matrix_a,
            sparse_matrix_b,
            sparse_matrix_c,
            context.float32_type(),
            buffer_size,
            buffer,
            SpGemmWorkKind::Compute,
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_work_estimation_or_compute"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.descriptor().unwrap(), descriptor);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.sparse_matrix_b().unwrap(), sparse_matrix_b);
        assert_eq!(operation.sparse_matrix_c().unwrap(), sparse_matrix_c);
        assert_eq!(operation.buffer_size().unwrap(), buffer_size);
        assert_eq!(operation.buffer().unwrap(), buffer);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert_eq!(operation.kind().unwrap(), SpGemmWorkKind::Compute);
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_sp_gemm_copy_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[
            (token_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
            (index_type, location),
        ]);
        let token = block.argument(0).unwrap().as_ref();
        let descriptor = block.argument(1).unwrap().as_ref();
        let sparse_matrix_a = block.argument(2).unwrap().as_ref();
        let sparse_matrix_b = block.argument(3).unwrap().as_ref();
        let sparse_matrix_c = block.argument(4).unwrap().as_ref();
        let operation = sp_gemm_copy(
            &[token],
            descriptor,
            MatrixTransposeMode::NonTranspose,
            MatrixTransposeMode::Transpose,
            sparse_matrix_a,
            sparse_matrix_b,
            sparse_matrix_c,
            context.float32_type(),
            true,
            location,
        )
        .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spgemm_copy"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.descriptor().unwrap(), descriptor);
        assert_eq!(operation.mode_a().unwrap(), MatrixTransposeMode::NonTranspose);
        assert_eq!(operation.mode_b().unwrap(), MatrixTransposeMode::Transpose);
        assert_eq!(operation.sparse_matrix_a().unwrap(), sparse_matrix_a);
        assert_eq!(operation.sparse_matrix_b().unwrap(), sparse_matrix_b);
        assert_eq!(operation.sparse_matrix_c().unwrap(), sparse_matrix_c);
        assert_eq!(operation.compute_type().unwrap(), context.float32_type());
        assert!(operation.async_token().unwrap().is_some());
    }

    #[test]
    fn test_sp_mat_get_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let token_type = context.gpu_async_token_type().unwrap().as_ref();
        let index_type = context.index_type().as_ref();
        let block = context.block(&[(token_type, location), (index_type, location)]);
        let token = block.argument(0).unwrap().as_ref();
        let sparse_matrix = block.argument(1).unwrap().as_ref();
        let operation = sp_mat_get_size(&[token], sparse_matrix, true, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.spmat_get_size"));
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.sparse_matrix().unwrap(), sparse_matrix);
        assert_eq!(operation.async_dependencies().unwrap(), vec![token]);
        assert_eq!(operation.sparse_matrix().unwrap(), sparse_matrix);
        assert_eq!(operation.rows().unwrap().r#type().unwrap(), context.index_type());
        assert!(operation.async_token().unwrap().is_some());
    }

    gpu_sparse_async_operation_test!(
        test_set_csr_pointers_operation,
        set_csr_pointers,
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
        let lane_id = block.argument(0).unwrap().as_ref();
        let argument_0 = block.argument(1).unwrap().as_ref();
        let argument_1 = block.argument(2).unwrap().as_ref();
        let result_types = [context.index_type().as_ref(), context.index_type().as_ref()];
        let operation =
            warp_execute_on_lane_0(lane_id, 32, &[argument_0, argument_1], &result_types, context.region(), location)
                .unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.warp_execute_on_lane_0"));
        assert_eq!(operation.lane_id().unwrap(), lane_id);
        assert_eq!(operation.warp_size().unwrap().signless_value(), 32);
        assert_eq!(operation.arguments().unwrap(), vec![argument_0, argument_1]);
        assert_eq!(operation.outputs().unwrap().len(), 2);
        assert_eq!(operation.as_ref().region(0).unwrap().blocks().count(), 0);
    }

    #[test]
    fn test_subgroup_broadcast_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let index_type = context.index_type();
        let block = context.block(&[(index_type, location), (index_type, location)]);
        let source = block.argument(0).unwrap().as_ref();
        let lane = block.argument(1).unwrap().as_ref();
        let operation = subgroup_broadcast(source, Some(lane), BroadcastType::SpecificLane, location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.subgroup_broadcast"));
        assert_eq!(operation.source().unwrap(), source);
        assert_eq!(operation.lane().unwrap(), Some(lane.as_ref()));
        assert_eq!(operation.broadcast_type().unwrap(), BroadcastType::SpecificLane);
        assert_eq!(operation.broadcast_type().unwrap(), BroadcastType::SpecificLane);
    }

    #[test]
    fn test_ballot_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let block = context.block(&[(context.signless_integer_type(1), location)]);
        let predicate = block.argument(0).unwrap();
        let operation = ballot(predicate, context.signless_integer_type(32), location).unwrap();

        assert_eq!(operation.name().as_str(), Ok("gpu.ballot"));
        assert_eq!(operation.predicate().unwrap(), predicate);
        assert_eq!(operation.predicate().unwrap(), predicate);
    }
}
