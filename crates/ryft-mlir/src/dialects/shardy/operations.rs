use crate::{
    Attribute, AttributeRef, DetachedOp, DetachedRegion, DialectHandle, Error, Location, Operation, OperationBuilder,
    StringRef, Type, Value, ValueRef, mlir_op, mlir_op_trait,
};

use super::attributes::{
    ManualAxesAttributeRef, MeshAttributeRef, TensorShardingAttributeRef, TensorShardingPerValueAttributeRef,
};

/// Name of the [`Attribute`] that is used to store [`AllGatherOperation::gathering_axes`].
pub const GATHERING_AXES_ATTRIBUTE: &str = "gathering_axes";

/// Name of the [`Attribute`] that is used to store [`AllGatherOperation::out_sharding`],
/// [`AllReduceOperation::out_sharding`], [`AllSliceOperation::out_sharding`], [`AllToAllOperation::out_sharding`],
/// [`ReduceScatterOperation::out_sharding`], [`ReplicatedToUnreducedOperation::out_sharding`], and
/// [`ShardedToUnreducedOperation::out_sharding`].
pub const OUT_SHARDING_ATTRIBUTE: &str = "out_sharding";

/// Shardy [`Operation`] that gathers slices of a tensor across axes.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyall_gather_sdyallgatherop)
/// for more information.
pub trait AllGatherOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`AllGatherOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the gathering-axes payload of this [`AllGatherOperation`].
    fn gathering_axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(GATHERING_AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                GATHERING_AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the output sharding payload of this [`AllGatherOperation`].
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(AllGather);
mlir_op_trait!(AllGather, OneOperand);
mlir_op_trait!(AllGather, OneResult);
mlir_op_trait!(AllGather, ZeroRegions);
mlir_op_trait!(AllGather, ZeroSuccessors);

/// Constructs a new detached/owned [`AllGatherOperation`] at the specified [`Location`].
pub fn all_gather<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    gathering_axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedAllGatherOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.all_gather", location)
        .add_operand(input)
        .add_attribute(GATHERING_AXES_ATTRIBUTE, gathering_axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::all_gather`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`AllReduceOperation::reduction_axes`].
pub const REDUCTION_AXES_ATTRIBUTE: &str = "reduction_axes";

/// Shardy [`Operation`] that reduces a tensor across axes and keeps the same sharding rank.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyall_reduce_sdyallreduceop)
/// for more information.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`AllReduceOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the reduction-axes payload of this [`AllReduceOperation`].
    fn reduction_axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(REDUCTION_AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                REDUCTION_AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the output sharding payload of this [`AllReduceOperation`].
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, OneOperand);
mlir_op_trait!(AllReduce, OneResult);
mlir_op_trait!(AllReduce, ZeroRegions);
mlir_op_trait!(AllReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`AllReduceOperation`] at the specified [`Location`].
pub fn all_reduce<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    reduction_axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedAllReduceOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.all_reduce", location)
        .add_operand(input)
        .add_attribute(REDUCTION_AXES_ATTRIBUTE, reduction_axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::all_reduce`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`AllSliceOperation::slicing_axes`].
pub const SLICING_AXES_ATTRIBUTE: &str = "slicing_axes";

/// Shardy [`Operation`] that slices a tensor along the specified axes.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyall_slice_sdyallsliceop)
/// for more information.
pub trait AllSliceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`AllSliceOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the slicing-axes payload of this [`AllSliceOperation`].
    fn slicing_axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(SLICING_AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                SLICING_AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the output sharding payload of this [`AllSliceOperation`].
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(AllSlice);
mlir_op_trait!(AllSlice, OneOperand);
mlir_op_trait!(AllSlice, OneResult);
mlir_op_trait!(AllSlice, ZeroRegions);
mlir_op_trait!(AllSlice, ZeroSuccessors);

/// Constructs a new detached/owned [`AllSliceOperation`] at the specified [`Location`].
pub fn all_slice<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    slicing_axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedAllSliceOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.all_slice", location)
        .add_operand(input)
        .add_attribute(SLICING_AXES_ATTRIBUTE, slicing_axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::all_slice`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`AllToAllOperation::params`].
pub const PARAMS_ATTRIBUTE: &str = "params";

/// Shardy [`Operation`] that performs a parameterized all-to-all tensor exchange.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyall_to_all_sdyalltoallop)
/// for more information.
pub trait AllToAllOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`AllToAllOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the parameter payload of this [`AllToAllOperation`].
    fn params(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(PARAMS_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                PARAMS_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns the output sharding payload of this [`AllToAllOperation`].
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(AllToAll);
mlir_op_trait!(AllToAll, OneOperand);
mlir_op_trait!(AllToAll, OneResult);
mlir_op_trait!(AllToAll, ZeroRegions);
mlir_op_trait!(AllToAll, ZeroSuccessors);

/// Constructs a new detached/owned [`AllToAllOperation`] at the specified [`Location`].
pub fn all_to_all<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    params: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedAllToAllOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.all_to_all", location)
        .add_operand(input)
        .add_attribute(PARAMS_ATTRIBUTE, params)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::all_to_all`"))
        })
}

/// Shardy [`Operation`] that permutes tensor data between source and target peers.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdycollective_permute_sdycollectivepermuteop)
/// for more information.
pub trait CollectivePermuteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`CollectivePermuteOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns output sharding metadata.
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(CollectivePermute);
mlir_op_trait!(CollectivePermute, OneOperand);
mlir_op_trait!(CollectivePermute, OneResult);
mlir_op_trait!(CollectivePermute, ZeroRegions);
mlir_op_trait!(CollectivePermute, ZeroSuccessors);

/// Constructs a new detached/owned [`CollectivePermuteOperation`] at the specified [`Location`].
pub fn collective_permute<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    input: V,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedCollectivePermuteOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.collective_permute", location)
        .add_operand(input)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::collective_permute`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`ShardingConstraintOperation::sharding`].
pub const SHARDING_ATTRIBUTE: &str = "sharding";

/// Name of the [`Attribute`] that is used to store [`ManualComputationOperation::in_shardings`].
pub const IN_SHARDINGS_ATTRIBUTE: &str = "in_shardings";

/// Name of the [`Attribute`] that is used to store [`ManualComputationOperation::out_shardings`].
pub const OUT_SHARDINGS_ATTRIBUTE: &str = "out_shardings";

/// Name of the [`Attribute`] that is used to store [`ManualComputationOperation::manual_axes`].
pub const MANUAL_AXES_ATTRIBUTE: &str = "manual_axes";

/// Shardy region-based [`Operation`] for manual SPMD computation sections.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdymanual_computation_sdymanualcomputationop)
/// for more information.
pub trait ManualComputationOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the operation body region.
    fn body(&self) -> Result<crate::RegionRef<'o, 'c, 't>, Error> {
        self.region(0)
    }

    /// Returns per-input shardings.
    fn in_shardings(&self) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, Error> {
        self.attribute(IN_SHARDINGS_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                IN_SHARDINGS_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns per-output shardings.
    fn out_shardings(&self) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDINGS_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDINGS_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns axes excluded from automatic propagation in the body.
    fn manual_axes(&self) -> Result<ManualAxesAttributeRef<'c, 't>, Error> {
        self.attribute(MANUAL_AXES_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                MANUAL_AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ManualComputation);
mlir_op_trait!(ManualComputation, IsolatedFromAbove);
mlir_op_trait!(ManualComputation, SingleBlockRegions);
mlir_op_trait!(ManualComputation, ZeroSuccessors);

/// Constructs a new detached/owned [`ManualComputationOperation`] at the specified [`Location`].
pub fn manual_computation<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    tensors: &[V],
    result_types: &[T],
    in_shardings: TensorShardingPerValueAttributeRef<'c, 't>,
    out_shardings: TensorShardingPerValueAttributeRef<'c, 't>,
    manual_axes: ManualAxesAttributeRef<'c, 't>,
    body: DetachedRegion<'c, 't>,
    location: L,
) -> Result<DetachedManualComputationOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.manual_computation", location)
        .add_operands(tensors)
        .add_attribute(IN_SHARDINGS_ATTRIBUTE, in_shardings)
        .add_attribute(OUT_SHARDINGS_ATTRIBUTE, out_shardings)
        .add_attribute(MANUAL_AXES_ATTRIBUTE, manual_axes)
        .add_results(result_types)
        .add_region(body)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::manual_computation`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`MeshOperation::symbol_name`].
pub const SYMBOL_NAME_ATTRIBUTE: &str = "sym_name";

/// Name of the [`Attribute`] that is used to store [`MeshOperation::mesh`].
pub const MESH_ATTRIBUTE: &str = "mesh";

/// Shardy [`Operation`] that defines a named mesh symbol.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdymesh_sdymeshop)
/// for more information.
pub trait MeshOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the symbol name of this [`MeshOperation`].
    fn symbol_name(&self) -> Result<StringRef<'c>, Error> {
        Ok(self.string_attribute(SYMBOL_NAME_ATTRIBUTE)?.string())
    }

    /// Returns the mesh payload of this [`MeshOperation`].
    fn mesh(&self) -> Result<MeshAttributeRef<'c, 't>, Error> {
        self.attribute(MESH_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                MESH_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(Mesh);
mlir_op_trait!(Mesh, Symbol);
mlir_op_trait!(Mesh, ZeroOperands);
mlir_op_trait!(Mesh, ZeroRegions);
mlir_op_trait!(Mesh, ZeroSuccessors);

/// Constructs a new detached/owned [`MeshOperation`] at the specified [`Location`].
pub fn mesh<'c, 't: 'c, L: Location<'c, 't>>(
    symbol_name: &str,
    mesh: MeshAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedMeshOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.mesh", location)
        .add_attribute(SYMBOL_NAME_ATTRIBUTE, context.string_attribute(symbol_name))
        .add_attribute(MESH_ATTRIBUTE, mesh)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::mesh`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`ReduceScatterOperation::reduce_scatter_axes`].
pub const REDUCE_SCATTER_AXES_ATTRIBUTE: &str = "reduce_scatter_axes";

/// Shardy [`Operation`] that combines reduce and scatter along specified axes.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyreduce_scatter_sdyreducescatterop)
/// for more information.
pub trait ReduceScatterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`ReduceScatterOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns reduce-scatter axes metadata.
    fn reduce_scatter_axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(REDUCE_SCATTER_AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                REDUCE_SCATTER_AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns output sharding metadata.
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ReduceScatter);
mlir_op_trait!(ReduceScatter, OneOperand);
mlir_op_trait!(ReduceScatter, OneResult);
mlir_op_trait!(ReduceScatter, ZeroRegions);
mlir_op_trait!(ReduceScatter, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduceScatterOperation`] at the specified [`Location`].
pub fn reduce_scatter<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    reduce_scatter_axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedReduceScatterOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.reduce_scatter", location)
        .add_operand(input)
        .add_attribute(REDUCE_SCATTER_AXES_ATTRIBUTE, reduce_scatter_axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::reduce_scatter`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`ReplicatedToUnreducedOperation::axes`] and
/// [`ShardedToUnreducedOperation::axes`].
pub const AXES_ATTRIBUTE: &str = "axes";

/// Shardy [`Operation`] that moves replicated axes to unreduced axes.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyreplicated_to_unreduced_sdyreplicatedtounreducedop)
/// for more information.
pub trait ReplicatedToUnreducedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`ReplicatedToUnreducedOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns axes metadata.
    fn axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns output sharding metadata.
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ReplicatedToUnreduced);
mlir_op_trait!(ReplicatedToUnreduced, OneOperand);
mlir_op_trait!(ReplicatedToUnreduced, OneResult);
mlir_op_trait!(ReplicatedToUnreduced, ZeroRegions);
mlir_op_trait!(ReplicatedToUnreduced, ZeroSuccessors);

/// Constructs a new detached/owned [`ReplicatedToUnreducedOperation`] at the specified [`Location`].
pub fn replicated_to_unreduced<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedReplicatedToUnreducedOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.replicated_to_unreduced", location)
        .add_operand(input)
        .add_attribute(AXES_ATTRIBUTE, axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::replicated_to_unreduced`"))
        })
}

/// Shardy region terminator [`Operation`] used inside Shardy region-based operations.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdyreturn_sdyreturnop)
/// for more information.
pub trait ReturnOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Return);
mlir_op_trait!(Return, ReturnLike);
mlir_op_trait!(Return, ZeroRegions);
mlir_op_trait!(Return, ZeroSuccessors);

/// Constructs a new detached/owned [`ReturnOperation`] at the specified [`Location`].
pub fn r#return<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    results: &[V],
    location: L,
) -> Result<DetachedReturnOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.return", location)
        .add_operands(results)
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::return`"))
        })
}

/// Shardy [`Operation`] that moves sharded axes to unreduced axes.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdysharded_to_unreduced_sdyshardedtounreducedop)
/// for more information.
pub trait ShardedToUnreducedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the input tensor of this [`ShardedToUnreducedOperation`].
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns axes metadata.
    fn axes(&self) -> Result<AttributeRef<'c, 't>, Error> {
        self.attribute(AXES_ATTRIBUTE)?.ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing `{}` attribute in `{}`",
                AXES_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }

    /// Returns output sharding metadata.
    fn out_sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(OUT_SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                OUT_SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ShardedToUnreduced);
mlir_op_trait!(ShardedToUnreduced, OneOperand);
mlir_op_trait!(ShardedToUnreduced, OneResult);
mlir_op_trait!(ShardedToUnreduced, ZeroRegions);
mlir_op_trait!(ShardedToUnreduced, ZeroSuccessors);

/// Constructs a new detached/owned [`ShardedToUnreducedOperation`] at the specified [`Location`].
pub fn sharded_to_unreduced<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    A: Attribute<'c, 't>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    input: V,
    axes: A,
    out_sharding: TensorShardingAttributeRef<'c, 't>,
    output_type: T,
    location: L,
) -> Result<DetachedShardedToUnreducedOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.sharded_to_unreduced", location)
        .add_operand(input)
        .add_attribute(AXES_ATTRIBUTE, axes)
        .add_attribute(OUT_SHARDING_ATTRIBUTE, out_sharding)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::sharded_to_unreduced`"))
        })
}

/// Shardy [`Operation`] that constrains one value to a requested sharding.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdysharding_constraint_sdyshardingconstraintop)
/// for more information.
pub trait ShardingConstraintOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the constrained input value.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the requested sharding payload.
    fn sharding(&self) -> Result<TensorShardingAttributeRef<'c, 't>, Error> {
        self.attribute(SHARDING_ATTRIBUTE)?.and_then(|attribute| attribute.cast()).ok_or_else(|| {
            Error::invalid_argument(format!(
                "missing or invalid `{}` attribute in `{}`",
                SHARDING_ATTRIBUTE,
                self.name().as_str().unwrap_or("<unknown>"),
            ))
        })
    }
}

mlir_op!(ShardingConstraint);
mlir_op_trait!(ShardingConstraint, OneOperand);
mlir_op_trait!(ShardingConstraint, OneResult);
mlir_op_trait!(ShardingConstraint, ZeroRegions);
mlir_op_trait!(ShardingConstraint, ZeroSuccessors);

/// Constructs a new detached/owned [`ShardingConstraintOperation`] at the specified [`Location`].
pub fn sharding_constraint<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    sharding: TensorShardingAttributeRef<'c, 't>,
    location: L,
) -> Result<DetachedShardingConstraintOperation<'c, 't>, Error> {
    location.context().load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.sharding_constraint", location)
        .add_operand(input)
        .add_attribute(SHARDING_ATTRIBUTE, sharding)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::sharding_constraint`"))
        })
}

/// Name of the [`Attribute`] that is used to store [`ShardingGroupOperation::group_id`].
pub const GROUP_ID_ATTRIBUTE: &str = "group_id";

/// Shardy [`Operation`] that groups a tensor into a sharding-propagation group.
///
/// Refer to the [official Shardy documentation](https://openxla.org/shardy/sdy_dialect#sdysharding_group_sdyshardinggroupop)
/// for more information.
pub trait ShardingGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the group input tensor.
    fn input(&self) -> Result<ValueRef<'o, 'c, 't>, Error> {
        self.operand_value(0)
    }

    /// Returns the group id.
    fn group_id(&self) -> Result<i64, Error> {
        Ok(self.integer_attribute(GROUP_ID_ATTRIBUTE)?.signless_value())
    }
}

mlir_op!(ShardingGroup);
mlir_op_trait!(ShardingGroup, ZeroRegions);
mlir_op_trait!(ShardingGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`ShardingGroupOperation`] at the specified [`Location`].
pub fn sharding_group<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    group_id: i64,
    location: L,
) -> Result<DetachedShardingGroupOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::shardy()?)?;
    OperationBuilder::new("sdy.sharding_group", location)
        .add_operand(input)
        .add_attribute(GROUP_ID_ATTRIBUTE, context.integer_attribute(context.signless_integer_type(64), group_id))
        .build()
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `shardy::sharding_group`"))
        })
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::dialects::func;
    use crate::dialects::shardy::ReductionOperation;
    use crate::{
        Attribute, Block, Context, DialectHandle, Operation, Region, Size, StringRef, Type, TypeAndAttributes,
    };

    use super::*;

    /// Internal helper for building common Shardy attributes used in tests in this module.
    fn test_sharding_attributes<'c, 't>(
        context: &'c Context<'t>,
    ) -> (TensorShardingAttributeRef<'c, 't>, TensorShardingPerValueAttributeRef<'c, 't>, ManualAxesAttributeRef<'c, 't>)
    {
        let mesh_axis = context.shardy_mesh_axis("a", 2).unwrap();
        let mesh = context.shardy_mesh([mesh_axis], &[]).unwrap();
        let axis_ref = context.shardy_axis_ref("a", None).unwrap();
        let dim_sharding = context.shardy_dimension_sharding([axis_ref], true, None).unwrap();
        let tensor_sharding =
            context.shardy_tensor_sharding(mesh, &[dim_sharding], &[], &[], ReductionOperation::Sum).unwrap();
        let tensor_sharding_per_value = context.shardy_tensor_sharding_per_value(&[tensor_sharding]).unwrap();
        let manual_axes = context.shardy_manual_axes(&["a"]).unwrap();
        (tensor_sharding, tensor_sharding_per_value, manual_axes)
    }

    fn test_operation_attribute<'c, 't>(
        context: &'c Context<'t>,
        module_source: &str,
        attribute_name: &str,
    ) -> AttributeRef<'c, 't> {
        context.load_dialect(DialectHandle::func().unwrap()).unwrap();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let module = context.parse_operation(module_source, "shardy_ops_attr_test.mlir").unwrap();
        let module_block = module.region(0).unwrap().blocks().unwrap().next().unwrap().unwrap();
        let function = module_block
            .operations()
            .unwrap()
            .find(|operation| operation.as_ref().unwrap().name().as_str().unwrap() == "func.func")
            .unwrap()
            .unwrap();
        let function_block = function.region(0).unwrap().blocks().unwrap().next().unwrap().unwrap();
        let operation = function_block
            .operations()
            .unwrap()
            .next()
            .expect("failed to find SDY operation when extracting attribute for tests")
            .unwrap();
        operation
            .attribute(attribute_name)
            .unwrap()
            .expect("failed to find SDY operation attribute for tests")
    }

    #[test]
    fn test_all_gather() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        let gathering_axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
                  ) -> tensor<8xf32> {
                    %0 = sdy.all_gather [{}] %arg0 out_sharding=<@mesh, [{"a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            GATHERING_AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let all_gather_op = all_gather(input, gathering_axes, tensor_sharding, tensor_type, location).unwrap();
                assert_eq!(all_gather_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_gather_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_gather_op.input().unwrap(), input);
                assert_eq!(all_gather_op.gathering_axes().unwrap(), gathering_axes);
                assert_eq!(all_gather_op.out_sharding().unwrap(), tensor_sharding);
                assert_eq!(all_gather_op.input().unwrap(), input);
                let all_gather_block = block.append_operation(all_gather_op).unwrap();
                block
                    .append_operation(func::r#return(&[all_gather_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "all_gather_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                tensor_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @all_gather_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}]>}) -> tensor<8xf32> {
                    %0 = sdy.all_gather [{}] %arg0 out_sharding=<mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_reduce() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let mesh_axis = context.shardy_mesh_axis("a", 2).unwrap();
        let mesh = context.shardy_mesh([mesh_axis], &[]).unwrap();
        let empty_dimension_sharding = context.shardy_dimension_sharding([], true, None).unwrap();
        let out_sharding = context
            .shardy_tensor_sharding(mesh, &[empty_dimension_sharding], &[], &[], ReductionOperation::Sum)
            .unwrap();
        let reduction_axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.all_reduce {"a"} %arg0 out_sharding=<@mesh, [{}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            REDUCTION_AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let all_reduce_op = all_reduce(input, reduction_axes, out_sharding, tensor_type, location).unwrap();
                assert_eq!(all_reduce_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_reduce_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_reduce_op.input().unwrap(), input);
                assert_eq!(all_reduce_op.reduction_axes().unwrap(), reduction_axes);
                assert_eq!(all_reduce_op.out_sharding().unwrap(), out_sharding);
                assert_eq!(all_reduce_op.input().unwrap(), input);
                let all_reduce_block = block.append_operation(all_reduce_op).unwrap();
                block
                    .append_operation(func::r#return(&[all_reduce_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "all_reduce_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @all_reduce_test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.all_reduce {\"a\"} %arg0 out_sharding=<mesh<[\"a\"=2]>, [{}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_slice() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        let slicing_axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
                  ) -> tensor<8xf32> {
                    %0 = sdy.all_slice [{}] %arg0 out_sharding=<@mesh, [{"a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            SLICING_AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let all_slice_op = all_slice(input, slicing_axes, tensor_sharding, tensor_type, location).unwrap();
                assert_eq!(all_slice_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_slice_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_slice_op.input().unwrap(), input);
                assert_eq!(all_slice_op.slicing_axes().unwrap(), slicing_axes);
                assert_eq!(all_slice_op.out_sharding().unwrap(), tensor_sharding);
                assert_eq!(all_slice_op.input().unwrap(), input);
                let all_slice_block = block.append_operation(all_slice_op).unwrap();
                block
                    .append_operation(func::r#return(&[all_slice_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "all_slice_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                tensor_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @all_slice_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}]>}) -> tensor<8xf32> {
                    %0 = sdy.all_slice [{}] %arg0 out_sharding=<mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_to_all() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context
            .tensor_type(context.float32_type(), &[Size::Static(8), Size::Static(8)], None, location)
            .unwrap();
        let mesh_axis = context.shardy_mesh_axis("a", 2).unwrap();
        let mesh = context.shardy_mesh([mesh_axis], &[]).unwrap();
        let axis_ref = context.shardy_axis_ref("a", None).unwrap();
        let dimension_sharding = context.shardy_dimension_sharding([axis_ref], true, None).unwrap();
        let empty_dimension_sharding = context.shardy_dimension_sharding([], true, None).unwrap();
        let input_sharding = context
            .shardy_tensor_sharding(
                mesh,
                &[dimension_sharding, empty_dimension_sharding],
                &[],
                &[],
                ReductionOperation::Sum,
            )
            .unwrap();
        let output_sharding = context
            .shardy_tensor_sharding(
                mesh,
                &[empty_dimension_sharding, dimension_sharding],
                &[],
                &[],
                ReductionOperation::Sum,
            )
            .unwrap();
        let params = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}, {}]>}
                  ) -> tensor<8x8xf32> {
                    %0 = sdy.all_to_all [{"a"}: 0->1] %arg0 out_sharding=<@mesh, [{}, {"a"}]> : tensor<8x8xf32>
                    return %0 : tensor<8x8xf32>
                  }
                }
            "#},
            PARAMS_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let all_to_all_op = all_to_all(input, params, output_sharding, tensor_type, location).unwrap();
                assert_eq!(all_to_all_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_to_all_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(all_to_all_op.input().unwrap(), input);
                assert_eq!(all_to_all_op.params().unwrap(), params);
                assert_eq!(all_to_all_op.out_sharding().unwrap(), output_sharding);
                assert_eq!(all_to_all_op.input().unwrap(), input);
                let all_to_all_block = block.append_operation(all_to_all_op).unwrap();
                block
                    .append_operation(func::r#return(&[all_to_all_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "all_to_all_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                input_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @all_to_all_test(%arg0: tensor<8x8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}, {}]>}) -> tensor<8x8xf32> {
                    %0 = sdy.all_to_all [{\"a\"}: 0->1] %arg0 out_sharding=<mesh<[\"a\"=2]>, [{}, {\"a\"}]> : tensor<8x8xf32>
                    return %0 : tensor<8x8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collective_permute() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let collective_permute_op = collective_permute(input, tensor_sharding, tensor_type, location).unwrap();
                assert_eq!(
                    collective_permute_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(
                    collective_permute_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(collective_permute_op.input().unwrap(), input);
                assert_eq!(collective_permute_op.out_sharding().unwrap(), tensor_sharding);
                assert_eq!(collective_permute_op.input().unwrap(), input);
                let collective_permute_block = block.append_operation(collective_permute_op).unwrap();
                block
                    .append_operation(func::r#return(&[collective_permute_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "collective_permute_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                tensor_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @collective_permute_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}]>}) -> tensor<8xf32> {
                    %0 = sdy.collective_permute %arg0 out_sharding=<mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_manual_computation() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let local_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let (_, tensor_sharding_per_value, manual_axes) = test_sharding_attributes(&context);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let mut region = context.region();
                let mut region_block = context.block(&[(local_type, location)]);
                let region_argument = region_block.argument(0).unwrap();
                region_block.append_operation(r#return(&[region_argument], location).unwrap()).unwrap();
                region.append_block(region_block).unwrap();
                let manual_computation_op = manual_computation(
                    &[input],
                    &[tensor_type],
                    tensor_sharding_per_value,
                    tensor_sharding_per_value,
                    manual_axes,
                    region,
                    location,
                )
                .unwrap();
                assert_eq!(
                    manual_computation_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(
                    manual_computation_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(manual_computation_op.in_shardings().unwrap(), tensor_sharding_per_value);
                assert_eq!(manual_computation_op.out_shardings().unwrap(), tensor_sharding_per_value);
                assert_eq!(manual_computation_op.manual_axes().unwrap(), manual_axes);
                let manual_computation_block = block.append_operation(manual_computation_op).unwrap();
                block
                    .append_operation(func::r#return(&[manual_computation_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "manual_computation_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @manual_computation_test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<mesh<[\"a\"=2]>, [{\"a\"}]>] \
                      out_shardings=[<mesh<[\"a\"=2]>, [{\"a\"}]>] manual_axes={\"a\"} (%arg1: tensor<4xf32>) {
                      sdy.return %arg1 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_mesh() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let mesh_axis = context.shardy_mesh_axis("a", 2).unwrap();
        let mesh_attribute = context.shardy_mesh([mesh_axis], &[]).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mesh_op = mesh("mesh", mesh_attribute, location).unwrap();
                assert_eq!(mesh_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(mesh_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(mesh_op.symbol_name().unwrap().as_str().unwrap(), "mesh");
                assert_eq!(mesh_op.mesh().unwrap(), mesh_attribute);
                mesh_op
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  sdy.mesh @mesh = <[\"a\"=2]>
                }
            "},
        );
    }

    #[test]
    fn test_reduce_scatter() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        let reduce_scatter_axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
                  ) -> tensor<8xf32> {
                    %0 = sdy.reduce_scatter [{}] %arg0 out_sharding=<@mesh, [{"a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            REDUCE_SCATTER_AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let reduce_scatter_op =
                    reduce_scatter(input, reduce_scatter_axes, tensor_sharding, tensor_type, location).unwrap();
                assert_eq!(reduce_scatter_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(reduce_scatter_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(reduce_scatter_op.input().unwrap(), input);
                assert_eq!(reduce_scatter_op.reduce_scatter_axes().unwrap(), reduce_scatter_axes);
                assert_eq!(reduce_scatter_op.out_sharding().unwrap(), tensor_sharding);
                assert_eq!(reduce_scatter_op.input().unwrap(), input);
                let reduce_scatter_block = block.append_operation(reduce_scatter_op).unwrap();
                block
                    .append_operation(func::r#return(&[reduce_scatter_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "reduce_scatter_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                tensor_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @reduce_scatter_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}]>}) -> tensor<8xf32> {
                    %0 = sdy.reduce_scatter [{}] %arg0 out_sharding=<mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_replicated_to_unreduced() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let mesh_axis = context.shardy_mesh_axis("a", 2).unwrap();
        let mesh = context.shardy_mesh([mesh_axis], &[]).unwrap();
        let axis_ref = context.shardy_axis_ref("a", None).unwrap();
        let empty_dimension_sharding = context.shardy_dimension_sharding([], true, None).unwrap();
        let input_sharding = context
            .shardy_tensor_sharding(mesh, &[empty_dimension_sharding], &[axis_ref], &[], ReductionOperation::Sum)
            .unwrap();
        let output_sharding = context
            .shardy_tensor_sharding(mesh, &[empty_dimension_sharding], &[], &[axis_ref], ReductionOperation::Sum)
            .unwrap();
        let axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}], replicated={"a"}>}
                  ) -> tensor<8xf32> {
                    %0 = sdy.replicated_to_unreduced {"a"} %arg0 out_sharding=<@mesh, [{}], unreduced={"a"}> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let replicated_to_unreduced_op =
                    replicated_to_unreduced(input, axes, output_sharding, tensor_type, location).unwrap();
                assert_eq!(
                    replicated_to_unreduced_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(
                    replicated_to_unreduced_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(replicated_to_unreduced_op.input().unwrap(), input);
                assert_eq!(replicated_to_unreduced_op.axes().unwrap(), axes);
                assert_eq!(replicated_to_unreduced_op.out_sharding().unwrap(), output_sharding);
                assert_eq!(replicated_to_unreduced_op.input().unwrap(), input);
                let replicated_to_unreduced_block = block.append_operation(replicated_to_unreduced_op).unwrap();
                block
                    .append_operation(
                        func::r#return(&[replicated_to_unreduced_block.result(0).unwrap()], location).unwrap(),
                    )
                    .unwrap();
                func::func(
                    "replicated_to_unreduced_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                input_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @replicated_to_unreduced_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{}], replicated={\"a\"}>}) -> tensor<8xf32> {
                    %0 = sdy.replicated_to_unreduced {\"a\"} %arg0 out_sharding=<mesh<[\"a\"=2]>, [{}], unreduced={\"a\"}> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_return() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let local_type = context.tensor_type(context.float32_type(), &[Size::Static(4)], None, location).unwrap();
        let (_, tensor_sharding_per_value, manual_axes) = test_sharding_attributes(&context);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let mut region = context.region();
                let mut region_block = context.block(&[(local_type, location)]);
                let region_argument = region_block.argument(0).unwrap();
                let return_op = r#return(&[region_argument], location).unwrap();
                assert_eq!(return_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(return_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                region_block.append_operation(return_op).unwrap();
                region.append_block(region_block).unwrap();
                let manual_computation_op = manual_computation(
                    &[input],
                    &[tensor_type],
                    tensor_sharding_per_value,
                    tensor_sharding_per_value,
                    manual_axes,
                    region,
                    location,
                )
                .unwrap();
                let manual_computation_block = block.append_operation(manual_computation_op).unwrap();
                block
                    .append_operation(func::r#return(&[manual_computation_block.result(0).unwrap()], location).unwrap())
                    .unwrap();
                func::func(
                    "return_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @return_test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<mesh<[\"a\"=2]>, [{\"a\"}]>] \
                      out_shardings=[<mesh<[\"a\"=2]>, [{\"a\"}]>] manual_axes={\"a\"} (%arg1: tensor<4xf32>) {
                      sdy.return %arg1 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sharded_to_unreduced() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        let axes = test_operation_attribute(
            &context,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["a"=2]>
                  func.func @test(
                    %arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"a"}]>}
                  ) -> tensor<8xf32> {
                    %0 = sdy.sharded_to_unreduced [{}] %arg0 out_sharding=<@mesh, [{"a"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#},
            AXES_ATTRIBUTE,
        );
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let sharded_to_unreduced_op =
                    sharded_to_unreduced(input, axes, tensor_sharding, tensor_type, location).unwrap();
                assert_eq!(
                    sharded_to_unreduced_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(
                    sharded_to_unreduced_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(sharded_to_unreduced_op.input().unwrap(), input);
                assert_eq!(sharded_to_unreduced_op.axes().unwrap(), axes);
                assert_eq!(sharded_to_unreduced_op.out_sharding().unwrap(), tensor_sharding);
                assert_eq!(sharded_to_unreduced_op.input().unwrap(), input);
                let sharded_to_unreduced_block = block.append_operation(sharded_to_unreduced_op).unwrap();
                block
                    .append_operation(
                        func::r#return(&[sharded_to_unreduced_block.result(0).unwrap()], location).unwrap(),
                    )
                    .unwrap();
                func::func(
                    "sharded_to_unreduced_test",
                    func::FuncAttributes {
                        arguments: vec![TypeAndAttributes {
                            r#type: tensor_type.as_ref(),
                            attributes: Some(HashMap::from([(
                                StringRef::from("sdy.sharding"),
                                tensor_sharding.as_ref(),
                            )])),
                        }],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @sharded_to_unreduced_test(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<mesh<[\"a\"=2]>, [{\"a\"}]>}) -> tensor<8xf32> {
                    %0 = sdy.sharded_to_unreduced [{}] %arg0 out_sharding=<mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sharding_constraint() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        let (tensor_sharding, _, _) = test_sharding_attributes(&context);
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let sharding_constraint_op = sharding_constraint(input, tensor_sharding, location).unwrap();
                assert_eq!(
                    sharding_constraint_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(
                    sharding_constraint_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(),
                    1
                );
                assert_eq!(sharding_constraint_op.input().unwrap(), input);
                assert_eq!(sharding_constraint_op.sharding().unwrap(), tensor_sharding);
                assert_eq!(sharding_constraint_op.input().unwrap(), input);
                let sharding_constraint_block = block.append_operation(sharding_constraint_op).unwrap();
                block
                    .append_operation(
                        func::r#return(&[sharding_constraint_block.result(0).unwrap()], location).unwrap(),
                    )
                    .unwrap();
                func::func(
                    "sharding_constraint_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @sharding_constraint_test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.sharding_constraint %arg0 <mesh<[\"a\"=2]>, [{\"a\"}]> : tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_sharding_group() {
        let context = Context::new();
        context.load_dialect(DialectHandle::shardy().unwrap()).unwrap();
        let location = context.unknown_location();
        let module = context.module(location).unwrap();
        let tensor_type = context.tensor_type(context.float32_type(), &[Size::Static(8)], None, location).unwrap();
        module
            .body()
            .unwrap()
            .append_operation({
                let mut block = context.block(&[(tensor_type, location)]);
                let input = block.argument(0).unwrap();
                let sharding_group_op = sharding_group(input, 7, location).unwrap();
                assert_eq!(sharding_group_op.operands().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 1);
                assert_eq!(sharding_group_op.results().collect::<Result<Vec<_>, _>>().unwrap().into_iter().count(), 0);
                assert_eq!(sharding_group_op.input().unwrap(), input);
                assert_eq!(sharding_group_op.group_id().unwrap(), 7);
                block.append_operation(sharding_group_op).unwrap();
                block.append_operation(func::r#return(&[input], location).unwrap()).unwrap();
                func::func(
                    "sharding_group_test",
                    func::FuncAttributes {
                        arguments: vec![tensor_type.into()],
                        results: vec![tensor_type.into()],
                        ..Default::default()
                    },
                    block.try_into().unwrap(),
                    location,
                )
                .unwrap()
            })
            .unwrap();
        assert!(module.verify().unwrap());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @sharding_group_test(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    sdy.sharding_group %arg0 group_id=7 : tensor<8xf32>
                    return %arg0 : tensor<8xf32>
                  }
                }
            "},
        );
    }
}
