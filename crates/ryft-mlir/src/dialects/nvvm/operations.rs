use crate::{
    AttributeRef, DetachedOp, DetachedOperation, DialectHandle, Error, Location, Operation, OperationBuilder, TypeRef,
    ValueRef, mlir_op, mlir_op_trait,
};

/// Constructs a detached NVVM operation with explicit operands, result types, and attributes.
///
/// This helper is intentionally low-level because many NVVM operations mirror PTX intrinsics with large and evolving
/// operand/property surfaces. The typed constructors below bind the concrete operation name while leaving operands,
/// results, and attributes explicit.
///
/// # Parameters
///
///   - `name`: Fully-qualified MLIR operation name.
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn build_nvvm_operation<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    name: &'static str,
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedOperation<'c, 't>, Error> {
    let context = location.context();
    context.load_dialect(DialectHandle::nvvm()?)?;
    let mut builder = OperationBuilder::new(name, location).add_operands(operands).add_results(result_types);
    for (name, attribute) in attributes {
        builder = builder.add_attribute(*name, *attribute);
    }
    if infer_result_types {
        builder = builder.enable_result_type_inference();
    }
    builder.build()
}

/// Fully-qualified MLIR operation name for `nvvm.addf`.
pub const ADDF_OPERATION: &str = "nvvm.addf";

/// Operation trait for `nvvm.addf`.
pub trait AddfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ADDF_OPERATION
    }
}

mlir_op!(Addf);
mlir_op_trait!(Addf, ZeroRegions);
mlir_op_trait!(Addf, ZeroSuccessors);

/// Constructs a new detached/owned [`AddfOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn addf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedAddfOperation<'c, 't>, Error> {
    build_nvvm_operation(ADDF_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::addf`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.aggr.smem.size`.
pub const READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION: &str = "nvvm.read.ptx.sreg.aggr.smem.size";

/// Operation trait for `nvvm.read.ptx.sreg.aggr.smem.size`.
pub trait ReadPtxSregAggrSmemSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION
    }
}

mlir_op!(ReadPtxSregAggrSmemSize);
mlir_op_trait!(ReadPtxSregAggrSmemSize, ZeroRegions);
mlir_op_trait!(ReadPtxSregAggrSmemSize, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregAggrSmemSizeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_aggr_smem_size<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregAggrSmemSizeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_aggr_smem_size`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.barrier.arrive`.
pub const BARRIER_ARRIVE_OPERATION: &str = "nvvm.barrier.arrive";

/// Operation trait for `nvvm.barrier.arrive`.
pub trait BarrierArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BARRIER_ARRIVE_OPERATION
    }
}

mlir_op!(BarrierArrive);
mlir_op_trait!(BarrierArrive, ZeroRegions);
mlir_op_trait!(BarrierArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`BarrierArriveOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn barrier_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedBarrierArriveOperation<'c, 't>, Error> {
    build_nvvm_operation(BARRIER_ARRIVE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::barrier_arrive`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.barrier`.
pub const BARRIER_OPERATION: &str = "nvvm.barrier";

/// Operation trait for `nvvm.barrier`.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BARRIER_OPERATION
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Constructs a new detached/owned [`BarrierOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedBarrierOperation<'c, 't>, Error> {
    build_nvvm_operation(BARRIER_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::barrier`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ntid.x`.
pub const READ_PTX_SREG_NTID_X_OPERATION: &str = "nvvm.read.ptx.sreg.ntid.x";

/// Operation trait for `nvvm.read.ptx.sreg.ntid.x`.
pub trait ReadPtxSregNtidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NTID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregNtidX);
mlir_op_trait!(ReadPtxSregNtidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregNtidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNtidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ntid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNtidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NTID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ntid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ntid.y`.
pub const READ_PTX_SREG_NTID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.ntid.y";

/// Operation trait for `nvvm.read.ptx.sreg.ntid.y`.
pub trait ReadPtxSregNtidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NTID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregNtidY);
mlir_op_trait!(ReadPtxSregNtidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregNtidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNtidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ntid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNtidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NTID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ntid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ntid.z`.
pub const READ_PTX_SREG_NTID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.ntid.z";

/// Operation trait for `nvvm.read.ptx.sreg.ntid.z`.
pub trait ReadPtxSregNtidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NTID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregNtidZ);
mlir_op_trait!(ReadPtxSregNtidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregNtidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNtidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ntid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNtidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NTID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ntid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ctaid.x`.
pub const READ_PTX_SREG_CTAID_X_OPERATION: &str = "nvvm.read.ptx.sreg.ctaid.x";

/// Operation trait for `nvvm.read.ptx.sreg.ctaid.x`.
pub trait ReadPtxSregCtaidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CTAID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregCtaidX);
mlir_op_trait!(ReadPtxSregCtaidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregCtaidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregCtaidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ctaid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregCtaidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CTAID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ctaid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ctaid.y`.
pub const READ_PTX_SREG_CTAID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.ctaid.y";

/// Operation trait for `nvvm.read.ptx.sreg.ctaid.y`.
pub trait ReadPtxSregCtaidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CTAID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregCtaidY);
mlir_op_trait!(ReadPtxSregCtaidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregCtaidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregCtaidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ctaid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregCtaidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CTAID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ctaid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.ctaid.z`.
pub const READ_PTX_SREG_CTAID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.ctaid.z";

/// Operation trait for `nvvm.read.ptx.sreg.ctaid.z`.
pub trait ReadPtxSregCtaidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CTAID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregCtaidZ);
mlir_op_trait!(ReadPtxSregCtaidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregCtaidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregCtaidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_ctaid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregCtaidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CTAID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_ctaid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.ctaid.x`.
pub const READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.ctaid.x";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.ctaid.x`.
pub trait ReadPtxSregClusterCtaidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterCtaidX);
mlir_op_trait!(ReadPtxSregClusterCtaidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterCtaidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterCtaidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_ctaid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterCtaidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_ctaid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.ctaid.y`.
pub const READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.ctaid.y";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.ctaid.y`.
pub trait ReadPtxSregClusterCtaidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterCtaidY);
mlir_op_trait!(ReadPtxSregClusterCtaidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterCtaidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterCtaidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_ctaid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterCtaidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_ctaid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.ctaid.z`.
pub const READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.ctaid.z";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.ctaid.z`.
pub trait ReadPtxSregClusterCtaidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterCtaidZ);
mlir_op_trait!(ReadPtxSregClusterCtaidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterCtaidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterCtaidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_ctaid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterCtaidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_ctaid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.breakpoint`.
pub const BREAKPOINT_OPERATION: &str = "nvvm.breakpoint";

/// Operation trait for `nvvm.breakpoint`.
pub trait BreakpointOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BREAKPOINT_OPERATION
    }
}

mlir_op!(Breakpoint);
mlir_op_trait!(Breakpoint, ZeroRegions);
mlir_op_trait!(Breakpoint, ZeroSuccessors);

/// Constructs a new detached/owned [`BreakpointOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn breakpoint<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedBreakpointOperation<'c, 't>, Error> {
    build_nvvm_operation(BREAKPOINT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::breakpoint`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.st.bulk`.
pub const ST_BULK_OPERATION: &str = "nvvm.st.bulk";

/// Operation trait for `nvvm.st.bulk`.
pub trait StBulkOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ST_BULK_OPERATION
    }
}

mlir_op!(StBulk);
mlir_op_trait!(StBulk, ZeroRegions);
mlir_op_trait!(StBulk, ZeroSuccessors);

/// Constructs a new detached/owned [`StBulkOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn st_bulk<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedStBulkOperation<'c, 't>, Error> {
    build_nvvm_operation(ST_BULK_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::st_bulk`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.clock64`.
pub const READ_PTX_SREG_CLOCK64_OPERATION: &str = "nvvm.read.ptx.sreg.clock64";

/// Operation trait for `nvvm.read.ptx.sreg.clock64`.
pub trait ReadPtxSregClock64Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLOCK64_OPERATION
    }
}

mlir_op!(ReadPtxSregClock64);
mlir_op_trait!(ReadPtxSregClock64, ZeroRegions);
mlir_op_trait!(ReadPtxSregClock64, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClock64Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_clock64<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClock64Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLOCK64_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_clock64`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.clock`.
pub const READ_PTX_SREG_CLOCK_OPERATION: &str = "nvvm.read.ptx.sreg.clock";

/// Operation trait for `nvvm.read.ptx.sreg.clock`.
pub trait ReadPtxSregClockOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLOCK_OPERATION
    }
}

mlir_op!(ReadPtxSregClock);
mlir_op_trait!(ReadPtxSregClock, ZeroRegions);
mlir_op_trait!(ReadPtxSregClock, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClockOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_clock<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClockOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLOCK_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_clock`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cluster.arrive`.
pub const CLUSTER_ARRIVE_OPERATION: &str = "nvvm.cluster.arrive";

/// Operation trait for `nvvm.cluster.arrive`.
pub trait ClusterArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CLUSTER_ARRIVE_OPERATION
    }
}

mlir_op!(ClusterArrive);
mlir_op_trait!(ClusterArrive, ZeroRegions);
mlir_op_trait!(ClusterArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`ClusterArriveOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cluster_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedClusterArriveOperation<'c, 't>, Error> {
    build_nvvm_operation(CLUSTER_ARRIVE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cluster_arrive`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.cluster.arrive.relaxed`.
pub const CLUSTER_ARRIVE_RELAXED_OPERATION: &str = "nvvm.cluster.arrive.relaxed";

/// Operation trait for `nvvm.cluster.arrive.relaxed`.
pub trait ClusterArriveRelaxedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CLUSTER_ARRIVE_RELAXED_OPERATION
    }
}

mlir_op!(ClusterArriveRelaxed);
mlir_op_trait!(ClusterArriveRelaxed, ZeroRegions);
mlir_op_trait!(ClusterArriveRelaxed, ZeroSuccessors);

/// Constructs a new detached/owned [`ClusterArriveRelaxedOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cluster_arrive_relaxed<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedClusterArriveRelaxedOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CLUSTER_ARRIVE_RELAXED_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cluster_arrive_relaxed`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.nctarank`.
pub const READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.nctarank";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.nctarank`.
pub trait ReadPtxSregClusterNctarankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterNctarank);
mlir_op_trait!(ReadPtxSregClusterNctarank, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterNctarank, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterNctarankOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_nctarank<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterNctarankOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_nctarank`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.nctaid.x`.
pub const READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.nctaid.x";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.nctaid.x`.
pub trait ReadPtxSregClusterNctaidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterNctaidX);
mlir_op_trait!(ReadPtxSregClusterNctaidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterNctaidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterNctaidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_nctaid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterNctaidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_nctaid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.nctaid.y`.
pub const READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.nctaid.y";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.nctaid.y`.
pub trait ReadPtxSregClusterNctaidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterNctaidY);
mlir_op_trait!(ReadPtxSregClusterNctaidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterNctaidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterNctaidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_nctaid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterNctaidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_nctaid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.nctaid.z`.
pub const READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.nctaid.z";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.nctaid.z`.
pub trait ReadPtxSregClusterNctaidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterNctaidZ);
mlir_op_trait!(ReadPtxSregClusterNctaidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterNctaidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterNctaidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_nctaid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterNctaidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_nctaid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nclusterid.x`.
pub const READ_PTX_SREG_NCLUSTERID_X_OPERATION: &str = "nvvm.read.ptx.sreg.nclusterid.x";

/// Operation trait for `nvvm.read.ptx.sreg.nclusterid.x`.
pub trait ReadPtxSregNclusteridXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCLUSTERID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregNclusteridX);
mlir_op_trait!(ReadPtxSregNclusteridX, ZeroRegions);
mlir_op_trait!(ReadPtxSregNclusteridX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNclusteridXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nclusterid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNclusteridXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCLUSTERID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nclusterid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nclusterid.y`.
pub const READ_PTX_SREG_NCLUSTERID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.nclusterid.y";

/// Operation trait for `nvvm.read.ptx.sreg.nclusterid.y`.
pub trait ReadPtxSregNclusteridYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCLUSTERID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregNclusteridY);
mlir_op_trait!(ReadPtxSregNclusteridY, ZeroRegions);
mlir_op_trait!(ReadPtxSregNclusteridY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNclusteridYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nclusterid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNclusteridYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCLUSTERID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nclusterid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nclusterid.z`.
pub const READ_PTX_SREG_NCLUSTERID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.nclusterid.z";

/// Operation trait for `nvvm.read.ptx.sreg.nclusterid.z`.
pub trait ReadPtxSregNclusteridZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCLUSTERID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregNclusteridZ);
mlir_op_trait!(ReadPtxSregNclusteridZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregNclusteridZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNclusteridZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nclusterid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNclusteridZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCLUSTERID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nclusterid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.cluster.ctarank`.
pub const READ_PTX_SREG_CLUSTER_CTARANK_OPERATION: &str = "nvvm.read.ptx.sreg.cluster.ctarank";

/// Operation trait for `nvvm.read.ptx.sreg.cluster.ctarank`.
pub trait ReadPtxSregClusterCtarankOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTER_CTARANK_OPERATION
    }
}

mlir_op!(ReadPtxSregClusterCtarank);
mlir_op_trait!(ReadPtxSregClusterCtarank, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusterCtarank, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusterCtarankOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_cluster_ctarank<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusterCtarankOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTER_CTARANK_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_cluster_ctarank`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.clusterid.x`.
pub const READ_PTX_SREG_CLUSTERID_X_OPERATION: &str = "nvvm.read.ptx.sreg.clusterid.x";

/// Operation trait for `nvvm.read.ptx.sreg.clusterid.x`.
pub trait ReadPtxSregClusteridXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTERID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregClusteridX);
mlir_op_trait!(ReadPtxSregClusteridX, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusteridX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusteridXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_clusterid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusteridXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTERID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_clusterid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.clusterid.y`.
pub const READ_PTX_SREG_CLUSTERID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.clusterid.y";

/// Operation trait for `nvvm.read.ptx.sreg.clusterid.y`.
pub trait ReadPtxSregClusteridYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTERID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregClusteridY);
mlir_op_trait!(ReadPtxSregClusteridY, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusteridY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusteridYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_clusterid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusteridYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTERID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_clusterid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.clusterid.z`.
pub const READ_PTX_SREG_CLUSTERID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.clusterid.z";

/// Operation trait for `nvvm.read.ptx.sreg.clusterid.z`.
pub trait ReadPtxSregClusteridZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_CLUSTERID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregClusteridZ);
mlir_op_trait!(ReadPtxSregClusteridZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregClusteridZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregClusteridZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_clusterid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregClusteridZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_CLUSTERID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_clusterid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.clusterlaunchcontrol.query.cancel`.
pub const CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION: &str = "nvvm.clusterlaunchcontrol.query.cancel";

/// Operation trait for `nvvm.clusterlaunchcontrol.query.cancel`.
pub trait ClusterlaunchcontrolQueryCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION
    }
}

mlir_op!(ClusterlaunchcontrolQueryCancel);
mlir_op_trait!(ClusterlaunchcontrolQueryCancel, ZeroRegions);
mlir_op_trait!(ClusterlaunchcontrolQueryCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`ClusterlaunchcontrolQueryCancelOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn clusterlaunchcontrol_query_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedClusterlaunchcontrolQueryCancelOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::clusterlaunchcontrol_query_cancel`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.clusterlaunchcontrol.try.cancel`.
pub const CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION: &str = "nvvm.clusterlaunchcontrol.try.cancel";

/// Operation trait for `nvvm.clusterlaunchcontrol.try.cancel`.
pub trait ClusterlaunchcontrolTryCancelOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION
    }
}

mlir_op!(ClusterlaunchcontrolTryCancel);
mlir_op_trait!(ClusterlaunchcontrolTryCancel, ZeroRegions);
mlir_op_trait!(ClusterlaunchcontrolTryCancel, ZeroSuccessors);

/// Constructs a new detached/owned [`ClusterlaunchcontrolTryCancelOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn clusterlaunchcontrol_try_cancel<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedClusterlaunchcontrolTryCancelOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::clusterlaunchcontrol_try_cancel`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cluster.wait`.
pub const CLUSTER_WAIT_OPERATION: &str = "nvvm.cluster.wait";

/// Operation trait for `nvvm.cluster.wait`.
pub trait ClusterWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CLUSTER_WAIT_OPERATION
    }
}

mlir_op!(ClusterWait);
mlir_op_trait!(ClusterWait, ZeroRegions);
mlir_op_trait!(ClusterWait, ZeroSuccessors);

/// Constructs a new detached/owned [`ClusterWaitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cluster_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedClusterWaitOperation<'c, 't>, Error> {
    build_nvvm_operation(CLUSTER_WAIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cluster_wait`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.bf16x2.to.f4x2`.
pub const CONVERT_BF16X2_TO_F4X2_OPERATION: &str = "nvvm.convert.bf16x2.to.f4x2";

/// Operation trait for `nvvm.convert.bf16x2.to.f4x2`.
pub trait ConvertBf16x2ToF4x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_BF16X2_TO_F4X2_OPERATION
    }
}

mlir_op!(ConvertBf16x2ToF4x2);
mlir_op_trait!(ConvertBf16x2ToF4x2, ZeroRegions);
mlir_op_trait!(ConvertBf16x2ToF4x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertBf16x2ToF4x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_bf16x2_to_f4x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertBf16x2ToF4x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_BF16X2_TO_F4X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_bf16x2_to_f4x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.bf16x2.to.f6x2`.
pub const CONVERT_BF16X2_TO_F6X2_OPERATION: &str = "nvvm.convert.bf16x2.to.f6x2";

/// Operation trait for `nvvm.convert.bf16x2.to.f6x2`.
pub trait ConvertBf16x2ToF6x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_BF16X2_TO_F6X2_OPERATION
    }
}

mlir_op!(ConvertBf16x2ToF6x2);
mlir_op_trait!(ConvertBf16x2ToF6x2, ZeroRegions);
mlir_op_trait!(ConvertBf16x2ToF6x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertBf16x2ToF6x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_bf16x2_to_f6x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertBf16x2ToF6x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_BF16X2_TO_F6X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_bf16x2_to_f6x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.bf16x2.to.f8x2`.
pub const CONVERT_BF16X2_TO_F8X2_OPERATION: &str = "nvvm.convert.bf16x2.to.f8x2";

/// Operation trait for `nvvm.convert.bf16x2.to.f8x2`.
pub trait ConvertBf16x2ToF8x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_BF16X2_TO_F8X2_OPERATION
    }
}

mlir_op!(ConvertBf16x2ToF8x2);
mlir_op_trait!(ConvertBf16x2ToF8x2, ZeroRegions);
mlir_op_trait!(ConvertBf16x2ToF8x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertBf16x2ToF8x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_bf16x2_to_f8x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertBf16x2ToF8x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_BF16X2_TO_F8X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_bf16x2_to_f8x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.bf16x2.to.s2f6x2`.
pub const CONVERT_BF16X2_TO_S2F6X2_OPERATION: &str = "nvvm.convert.bf16x2.to.s2f6x2";

/// Operation trait for `nvvm.convert.bf16x2.to.s2f6x2`.
pub trait ConvertBf16x2ToS2f6x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_BF16X2_TO_S2F6X2_OPERATION
    }
}

mlir_op!(ConvertBf16x2ToS2f6x2);
mlir_op_trait!(ConvertBf16x2ToS2f6x2, ZeroRegions);
mlir_op_trait!(ConvertBf16x2ToS2f6x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertBf16x2ToS2f6x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_bf16x2_to_s2f6x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertBf16x2ToS2f6x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_BF16X2_TO_S2F6X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_bf16x2_to_s2f6x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f4x2.to.f16x2`.
pub const CONVERT_F4X2_TO_F16X2_OPERATION: &str = "nvvm.convert.f4x2.to.f16x2";

/// Operation trait for `nvvm.convert.f4x2.to.f16x2`.
pub trait ConvertF4x2ToF16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F4X2_TO_F16X2_OPERATION
    }
}

mlir_op!(ConvertF4x2ToF16x2);
mlir_op_trait!(ConvertF4x2ToF16x2, ZeroRegions);
mlir_op_trait!(ConvertF4x2ToF16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF4x2ToF16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f4x2_to_f16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF4x2ToF16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F4X2_TO_F16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f4x2_to_f16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f6x2.to.f16x2`.
pub const CONVERT_F6X2_TO_F16X2_OPERATION: &str = "nvvm.convert.f6x2.to.f16x2";

/// Operation trait for `nvvm.convert.f6x2.to.f16x2`.
pub trait ConvertF6x2ToF16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F6X2_TO_F16X2_OPERATION
    }
}

mlir_op!(ConvertF6x2ToF16x2);
mlir_op_trait!(ConvertF6x2ToF16x2, ZeroRegions);
mlir_op_trait!(ConvertF6x2ToF16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF6x2ToF16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f6x2_to_f16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF6x2ToF16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F6X2_TO_F16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f6x2_to_f16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f8x2.to.bf16x2`.
pub const CONVERT_F8X2_TO_BF16X2_OPERATION: &str = "nvvm.convert.f8x2.to.bf16x2";

/// Operation trait for `nvvm.convert.f8x2.to.bf16x2`.
pub trait ConvertF8x2ToBf16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F8X2_TO_BF16X2_OPERATION
    }
}

mlir_op!(ConvertF8x2ToBf16x2);
mlir_op_trait!(ConvertF8x2ToBf16x2, ZeroRegions);
mlir_op_trait!(ConvertF8x2ToBf16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF8x2ToBf16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f8x2_to_bf16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF8x2ToBf16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F8X2_TO_BF16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f8x2_to_bf16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f8x2.to.f16x2`.
pub const CONVERT_F8X2_TO_F16X2_OPERATION: &str = "nvvm.convert.f8x2.to.f16x2";

/// Operation trait for `nvvm.convert.f8x2.to.f16x2`.
pub trait ConvertF8x2ToF16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F8X2_TO_F16X2_OPERATION
    }
}

mlir_op!(ConvertF8x2ToF16x2);
mlir_op_trait!(ConvertF8x2ToF16x2, ZeroRegions);
mlir_op_trait!(ConvertF8x2ToF16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF8x2ToF16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f8x2_to_f16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF8x2ToF16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F8X2_TO_F16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f8x2_to_f16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f16x2.to.f4x2`.
pub const CONVERT_F16X2_TO_F4X2_OPERATION: &str = "nvvm.convert.f16x2.to.f4x2";

/// Operation trait for `nvvm.convert.f16x2.to.f4x2`.
pub trait ConvertF16x2ToF4x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F16X2_TO_F4X2_OPERATION
    }
}

mlir_op!(ConvertF16x2ToF4x2);
mlir_op_trait!(ConvertF16x2ToF4x2, ZeroRegions);
mlir_op_trait!(ConvertF16x2ToF4x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF16x2ToF4x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f16x2_to_f4x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF16x2ToF4x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F16X2_TO_F4X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f16x2_to_f4x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f16x2.to.f6x2`.
pub const CONVERT_F16X2_TO_F6X2_OPERATION: &str = "nvvm.convert.f16x2.to.f6x2";

/// Operation trait for `nvvm.convert.f16x2.to.f6x2`.
pub trait ConvertF16x2ToF6x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F16X2_TO_F6X2_OPERATION
    }
}

mlir_op!(ConvertF16x2ToF6x2);
mlir_op_trait!(ConvertF16x2ToF6x2, ZeroRegions);
mlir_op_trait!(ConvertF16x2ToF6x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF16x2ToF6x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f16x2_to_f6x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF16x2ToF6x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F16X2_TO_F6X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f16x2_to_f6x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f16x2.to.f8x2`.
pub const CONVERT_F16X2_TO_F8X2_OPERATION: &str = "nvvm.convert.f16x2.to.f8x2";

/// Operation trait for `nvvm.convert.f16x2.to.f8x2`.
pub trait ConvertF16x2ToF8x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F16X2_TO_F8X2_OPERATION
    }
}

mlir_op!(ConvertF16x2ToF8x2);
mlir_op_trait!(ConvertF16x2ToF8x2, ZeroRegions);
mlir_op_trait!(ConvertF16x2ToF8x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF16x2ToF8x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f16x2_to_f8x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF16x2ToF8x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F16X2_TO_F8X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f16x2_to_f8x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.bf16x2`.
pub const CONVERT_F32X2_TO_BF16X2_OPERATION: &str = "nvvm.convert.f32x2.to.bf16x2";

/// Operation trait for `nvvm.convert.f32x2.to.bf16x2`.
pub trait ConvertF32x2ToBf16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_BF16X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToBf16x2);
mlir_op_trait!(ConvertF32x2ToBf16x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToBf16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToBf16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_bf16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToBf16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_BF16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_bf16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.f4x2`.
pub const CONVERT_F32X2_TO_F4X2_OPERATION: &str = "nvvm.convert.f32x2.to.f4x2";

/// Operation trait for `nvvm.convert.f32x2.to.f4x2`.
pub trait ConvertF32x2ToF4x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_F4X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToF4x2);
mlir_op_trait!(ConvertF32x2ToF4x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToF4x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToF4x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_f4x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToF4x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_F4X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_f4x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.f6x2`.
pub const CONVERT_F32X2_TO_F6X2_OPERATION: &str = "nvvm.convert.f32x2.to.f6x2";

/// Operation trait for `nvvm.convert.f32x2.to.f6x2`.
pub trait ConvertF32x2ToF6x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_F6X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToF6x2);
mlir_op_trait!(ConvertF32x2ToF6x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToF6x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToF6x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_f6x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToF6x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_F6X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_f6x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.f8x2`.
pub const CONVERT_F32X2_TO_F8X2_OPERATION: &str = "nvvm.convert.f32x2.to.f8x2";

/// Operation trait for `nvvm.convert.f32x2.to.f8x2`.
pub trait ConvertF32x2ToF8x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_F8X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToF8x2);
mlir_op_trait!(ConvertF32x2ToF8x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToF8x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToF8x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_f8x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToF8x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_F8X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_f8x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.f16x2`.
pub const CONVERT_F32X2_TO_F16X2_OPERATION: &str = "nvvm.convert.f32x2.to.f16x2";

/// Operation trait for `nvvm.convert.f32x2.to.f16x2`.
pub trait ConvertF32x2ToF16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_F16X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToF16x2);
mlir_op_trait!(ConvertF32x2ToF16x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToF16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToF16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_f16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToF16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_F16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_f16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x2.to.s2f6x2`.
pub const CONVERT_F32X2_TO_S2F6X2_OPERATION: &str = "nvvm.convert.f32x2.to.s2f6x2";

/// Operation trait for `nvvm.convert.f32x2.to.s2f6x2`.
pub trait ConvertF32x2ToS2f6x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X2_TO_S2F6X2_OPERATION
    }
}

mlir_op!(ConvertF32x2ToS2f6x2);
mlir_op_trait!(ConvertF32x2ToS2f6x2, ZeroRegions);
mlir_op_trait!(ConvertF32x2ToS2f6x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x2ToS2f6x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x2_to_s2f6x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x2ToS2f6x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X2_TO_S2F6X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x2_to_s2f6x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x4.to.f4x4`.
pub const CONVERT_F32X4_TO_F4X4_OPERATION: &str = "nvvm.convert.f32x4.to.f4x4";

/// Operation trait for `nvvm.convert.f32x4.to.f4x4`.
pub trait ConvertF32x4ToF4x4Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X4_TO_F4X4_OPERATION
    }
}

mlir_op!(ConvertF32x4ToF4x4);
mlir_op_trait!(ConvertF32x4ToF4x4, ZeroRegions);
mlir_op_trait!(ConvertF32x4ToF4x4, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x4ToF4x4Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x4_to_f4x4<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x4ToF4x4Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X4_TO_F4X4_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x4_to_f4x4`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x4.to.f6x4`.
pub const CONVERT_F32X4_TO_F6X4_OPERATION: &str = "nvvm.convert.f32x4.to.f6x4";

/// Operation trait for `nvvm.convert.f32x4.to.f6x4`.
pub trait ConvertF32x4ToF6x4Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X4_TO_F6X4_OPERATION
    }
}

mlir_op!(ConvertF32x4ToF6x4);
mlir_op_trait!(ConvertF32x4ToF6x4, ZeroRegions);
mlir_op_trait!(ConvertF32x4ToF6x4, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x4ToF6x4Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x4_to_f6x4<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x4ToF6x4Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X4_TO_F6X4_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x4_to_f6x4`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.f32x4.to.f8x4`.
pub const CONVERT_F32X4_TO_F8X4_OPERATION: &str = "nvvm.convert.f32x4.to.f8x4";

/// Operation trait for `nvvm.convert.f32x4.to.f8x4`.
pub trait ConvertF32x4ToF8x4Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_F32X4_TO_F8X4_OPERATION
    }
}

mlir_op!(ConvertF32x4ToF8x4);
mlir_op_trait!(ConvertF32x4ToF8x4, ZeroRegions);
mlir_op_trait!(ConvertF32x4ToF8x4, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertF32x4ToF8x4Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_f32x4_to_f8x4<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertF32x4ToF8x4Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_F32X4_TO_F8X4_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_f32x4_to_f8x4`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.float.to.tf32`.
pub const CONVERT_FLOAT_TO_TF32_OPERATION: &str = "nvvm.convert.float.to.tf32";

/// Operation trait for `nvvm.convert.float.to.tf32`.
pub trait ConvertFloatToTf32Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_FLOAT_TO_TF32_OPERATION
    }
}

mlir_op!(ConvertFloatToTf32);
mlir_op_trait!(ConvertFloatToTf32, ZeroRegions);
mlir_op_trait!(ConvertFloatToTf32, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertFloatToTf32Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_float_to_tf32<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertFloatToTf32Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_FLOAT_TO_TF32_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_float_to_tf32`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.convert.s2f6x2.to.bf16x2`.
pub const CONVERT_S2F6X2_TO_BF16X2_OPERATION: &str = "nvvm.convert.s2f6x2.to.bf16x2";

/// Operation trait for `nvvm.convert.s2f6x2.to.bf16x2`.
pub trait ConvertS2f6x2ToBf16x2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CONVERT_S2F6X2_TO_BF16X2_OPERATION
    }
}

mlir_op!(ConvertS2f6x2ToBf16x2);
mlir_op_trait!(ConvertS2f6x2ToBf16x2, ZeroRegions);
mlir_op_trait!(ConvertS2f6x2ToBf16x2, ZeroSuccessors);

/// Constructs a new detached/owned [`ConvertS2f6x2ToBf16x2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn convert_s2f6x2_to_bf16x2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedConvertS2f6x2ToBf16x2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        CONVERT_S2F6X2_TO_BF16X2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::convert_s2f6x2_to_bf16x2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cos`.
pub const COS_OPERATION: &str = "nvvm.cos";

/// Operation trait for `nvvm.cos`.
pub trait CosOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        COS_OPERATION
    }
}

mlir_op!(Cos);
mlir_op_trait!(Cos, ZeroRegions);
mlir_op_trait!(Cos, ZeroSuccessors);

/// Constructs a new detached/owned [`CosOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cos<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCosOperation<'c, 't>, Error> {
    build_nvvm_operation(COS_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cos`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.commit.group`.
pub const CP_ASYNC_BULK_COMMIT_GROUP_OPERATION: &str = "nvvm.cp.async.bulk.commit.group";

/// Operation trait for `nvvm.cp.async.bulk.commit.group`.
pub trait CpAsyncBulkCommitGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_COMMIT_GROUP_OPERATION
    }
}

mlir_op!(CpAsyncBulkCommitGroup);
mlir_op_trait!(CpAsyncBulkCommitGroup, ZeroRegions);
mlir_op_trait!(CpAsyncBulkCommitGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkCommitGroupOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_commit_group<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkCommitGroupOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_COMMIT_GROUP_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_commit_group`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.shared.cluster.global`.
pub const CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION: &str = "nvvm.cp.async.bulk.shared.cluster.global";

/// Operation trait for `nvvm.cp.async.bulk.shared.cluster.global`.
pub trait CpAsyncBulkSharedClusterGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION
    }
}

mlir_op!(CpAsyncBulkSharedClusterGlobal);
mlir_op_trait!(CpAsyncBulkSharedClusterGlobal, ZeroRegions);
mlir_op_trait!(CpAsyncBulkSharedClusterGlobal, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkSharedClusterGlobalOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_shared_cluster_global<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkSharedClusterGlobalOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_shared_cluster_global`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.prefetch`.
pub const CP_ASYNC_BULK_PREFETCH_OPERATION: &str = "nvvm.cp.async.bulk.prefetch";

/// Operation trait for `nvvm.cp.async.bulk.prefetch`.
pub trait CpAsyncBulkPrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_PREFETCH_OPERATION
    }
}

mlir_op!(CpAsyncBulkPrefetch);
mlir_op_trait!(CpAsyncBulkPrefetch, ZeroRegions);
mlir_op_trait!(CpAsyncBulkPrefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkPrefetchOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkPrefetchOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_PREFETCH_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_prefetch`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.global.shared.cta`.
pub const CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION: &str = "nvvm.cp.async.bulk.global.shared.cta";

/// Operation trait for `nvvm.cp.async.bulk.global.shared.cta`.
pub trait CpAsyncBulkGlobalSharedCtaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION
    }
}

mlir_op!(CpAsyncBulkGlobalSharedCta);
mlir_op_trait!(CpAsyncBulkGlobalSharedCta, ZeroRegions);
mlir_op_trait!(CpAsyncBulkGlobalSharedCta, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkGlobalSharedCtaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_global_shared_cta<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkGlobalSharedCtaOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_global_shared_cta`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.shared.cluster.shared.cta`.
pub const CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION: &str = "nvvm.cp.async.bulk.shared.cluster.shared.cta";

/// Operation trait for `nvvm.cp.async.bulk.shared.cluster.shared.cta`.
pub trait CpAsyncBulkSharedClusterSharedCtaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION
    }
}

mlir_op!(CpAsyncBulkSharedClusterSharedCta);
mlir_op_trait!(CpAsyncBulkSharedClusterSharedCta, ZeroRegions);
mlir_op_trait!(CpAsyncBulkSharedClusterSharedCta, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkSharedClusterSharedCtaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_shared_cluster_shared_cta<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkSharedClusterSharedCtaOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| {
            Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_shared_cluster_shared_cta`")
        })
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.tensor.shared.cluster.global`.
pub const CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION: &str =
    "nvvm.cp.async.bulk.tensor.shared.cluster.global";

/// Operation trait for `nvvm.cp.async.bulk.tensor.shared.cluster.global`.
pub trait CpAsyncBulkTensorSharedClusterGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION
    }
}

mlir_op!(CpAsyncBulkTensorSharedClusterGlobal);
mlir_op_trait!(CpAsyncBulkTensorSharedClusterGlobal, ZeroRegions);
mlir_op_trait!(CpAsyncBulkTensorSharedClusterGlobal, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkTensorSharedClusterGlobalOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_tensor_shared_cluster_global<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkTensorSharedClusterGlobalOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| {
            Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_tensor_shared_cluster_global`")
        })
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.tensor.prefetch`.
pub const CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION: &str = "nvvm.cp.async.bulk.tensor.prefetch";

/// Operation trait for `nvvm.cp.async.bulk.tensor.prefetch`.
pub trait CpAsyncBulkTensorPrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION
    }
}

mlir_op!(CpAsyncBulkTensorPrefetch);
mlir_op_trait!(CpAsyncBulkTensorPrefetch, ZeroRegions);
mlir_op_trait!(CpAsyncBulkTensorPrefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkTensorPrefetchOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_tensor_prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkTensorPrefetchOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_tensor_prefetch`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.tensor.reduce`.
pub const CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION: &str = "nvvm.cp.async.bulk.tensor.reduce";

/// Operation trait for `nvvm.cp.async.bulk.tensor.reduce`.
pub trait CpAsyncBulkTensorReduceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION
    }
}

mlir_op!(CpAsyncBulkTensorReduce);
mlir_op_trait!(CpAsyncBulkTensorReduce, ZeroRegions);
mlir_op_trait!(CpAsyncBulkTensorReduce, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkTensorReduceOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_tensor_reduce<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkTensorReduceOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_tensor_reduce`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.tensor.global.shared.cta`.
pub const CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION: &str = "nvvm.cp.async.bulk.tensor.global.shared.cta";

/// Operation trait for `nvvm.cp.async.bulk.tensor.global.shared.cta`.
pub trait CpAsyncBulkTensorGlobalSharedCtaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION
    }
}

mlir_op!(CpAsyncBulkTensorGlobalSharedCta);
mlir_op_trait!(CpAsyncBulkTensorGlobalSharedCta, ZeroRegions);
mlir_op_trait!(CpAsyncBulkTensorGlobalSharedCta, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkTensorGlobalSharedCtaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_tensor_global_shared_cta<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkTensorGlobalSharedCtaOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation.cast().ok_or_else(|| {
            Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_tensor_global_shared_cta`")
        })
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.bulk.wait_group`.
pub const CP_ASYNC_BULK_WAIT_GROUP_OPERATION: &str = "nvvm.cp.async.bulk.wait_group";

/// Operation trait for `nvvm.cp.async.bulk.wait_group`.
pub trait CpAsyncBulkWaitGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_BULK_WAIT_GROUP_OPERATION
    }
}

mlir_op!(CpAsyncBulkWaitGroup);
mlir_op_trait!(CpAsyncBulkWaitGroup, ZeroRegions);
mlir_op_trait!(CpAsyncBulkWaitGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncBulkWaitGroupOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_bulk_wait_group<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncBulkWaitGroupOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_BULK_WAIT_GROUP_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_bulk_wait_group`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.commit.group`.
pub const CP_ASYNC_COMMIT_GROUP_OPERATION: &str = "nvvm.cp.async.commit.group";

/// Operation trait for `nvvm.cp.async.commit.group`.
pub trait CpAsyncCommitGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_COMMIT_GROUP_OPERATION
    }
}

mlir_op!(CpAsyncCommitGroup);
mlir_op_trait!(CpAsyncCommitGroup, ZeroRegions);
mlir_op_trait!(CpAsyncCommitGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncCommitGroupOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_commit_group<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncCommitGroupOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_COMMIT_GROUP_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_commit_group`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.mbarrier.arrive`.
pub const CP_ASYNC_MBARRIER_ARRIVE_OPERATION: &str = "nvvm.cp.async.mbarrier.arrive";

/// Operation trait for `nvvm.cp.async.mbarrier.arrive`.
pub trait CpAsyncMbarrierArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_MBARRIER_ARRIVE_OPERATION
    }
}

mlir_op!(CpAsyncMbarrierArrive);
mlir_op_trait!(CpAsyncMbarrierArrive, ZeroRegions);
mlir_op_trait!(CpAsyncMbarrierArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncMbarrierArriveOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_mbarrier_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncMbarrierArriveOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_MBARRIER_ARRIVE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_mbarrier_arrive`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.shared.global`.
pub const CP_ASYNC_SHARED_GLOBAL_OPERATION: &str = "nvvm.cp.async.shared.global";

/// Operation trait for `nvvm.cp.async.shared.global`.
pub trait CpAsyncSharedGlobalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_SHARED_GLOBAL_OPERATION
    }
}

mlir_op!(CpAsyncSharedGlobal);
mlir_op_trait!(CpAsyncSharedGlobal, ZeroRegions);
mlir_op_trait!(CpAsyncSharedGlobal, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncSharedGlobalOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_shared_global<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncSharedGlobalOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_SHARED_GLOBAL_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_shared_global`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.cp.async.wait.group`.
pub const CP_ASYNC_WAIT_GROUP_OPERATION: &str = "nvvm.cp.async.wait.group";

/// Operation trait for `nvvm.cp.async.wait.group`.
pub trait CpAsyncWaitGroupOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        CP_ASYNC_WAIT_GROUP_OPERATION
    }
}

mlir_op!(CpAsyncWaitGroup);
mlir_op_trait!(CpAsyncWaitGroup, ZeroRegions);
mlir_op_trait!(CpAsyncWaitGroup, ZeroSuccessors);

/// Constructs a new detached/owned [`CpAsyncWaitGroupOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn cp_async_wait_group<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedCpAsyncWaitGroupOperation<'c, 't>, Error> {
    build_nvvm_operation(
        CP_ASYNC_WAIT_GROUP_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::cp_async_wait_group`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.divf`.
pub const DIVF_OPERATION: &str = "nvvm.divf";

/// Operation trait for `nvvm.divf`.
pub trait DivfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DIVF_OPERATION
    }
}

mlir_op!(Divf);
mlir_op_trait!(Divf, ZeroRegions);
mlir_op_trait!(Divf, ZeroSuccessors);

/// Constructs a new detached/owned [`DivfOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn divf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedDivfOperation<'c, 't>, Error> {
    build_nvvm_operation(DIVF_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::divf`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.dot.accumulate.2way`.
pub const DOT_ACCUMULATE_2WAY_OPERATION: &str = "nvvm.dot.accumulate.2way";

/// Operation trait for `nvvm.dot.accumulate.2way`.
pub trait DotAccumulateValue2wayOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DOT_ACCUMULATE_2WAY_OPERATION
    }
}

mlir_op!(DotAccumulateValue2way);
mlir_op_trait!(DotAccumulateValue2way, ZeroRegions);
mlir_op_trait!(DotAccumulateValue2way, ZeroSuccessors);

/// Constructs a new detached/owned [`DotAccumulateValue2wayOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn dot_accumulate_2way<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedDotAccumulateValue2wayOperation<'c, 't>, Error> {
    build_nvvm_operation(
        DOT_ACCUMULATE_2WAY_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::dot_accumulate_2way`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.dot.accumulate.4way`.
pub const DOT_ACCUMULATE_4WAY_OPERATION: &str = "nvvm.dot.accumulate.4way";

/// Operation trait for `nvvm.dot.accumulate.4way`.
pub trait DotAccumulateValue4wayOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        DOT_ACCUMULATE_4WAY_OPERATION
    }
}

mlir_op!(DotAccumulateValue4way);
mlir_op_trait!(DotAccumulateValue4way, ZeroRegions);
mlir_op_trait!(DotAccumulateValue4way, ZeroSuccessors);

/// Constructs a new detached/owned [`DotAccumulateValue4wayOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn dot_accumulate_4way<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedDotAccumulateValue4wayOperation<'c, 't>, Error> {
    build_nvvm_operation(
        DOT_ACCUMULATE_4WAY_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::dot_accumulate_4way`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.dynamic.smem.size`.
pub const READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION: &str = "nvvm.read.ptx.sreg.dynamic.smem.size";

/// Operation trait for `nvvm.read.ptx.sreg.dynamic.smem.size`.
pub trait ReadPtxSregDynamicSmemSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION
    }
}

mlir_op!(ReadPtxSregDynamicSmemSize);
mlir_op_trait!(ReadPtxSregDynamicSmemSize, ZeroRegions);
mlir_op_trait!(ReadPtxSregDynamicSmemSize, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregDynamicSmemSizeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_dynamic_smem_size<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregDynamicSmemSizeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_dynamic_smem_size`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.elect.sync`.
pub const ELECT_SYNC_OPERATION: &str = "nvvm.elect.sync";

/// Operation trait for `nvvm.elect.sync`.
pub trait ElectSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        ELECT_SYNC_OPERATION
    }
}

mlir_op!(ElectSync);
mlir_op_trait!(ElectSync, ZeroRegions);
mlir_op_trait!(ElectSync, ZeroSuccessors);

/// Constructs a new detached/owned [`ElectSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn elect_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedElectSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(ELECT_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::elect_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg0`.
pub const READ_PTX_SREG_ENVREG0_OPERATION: &str = "nvvm.read.ptx.sreg.envreg0";

/// Operation trait for `nvvm.read.ptx.sreg.envreg0`.
pub trait ReadPtxSregEnvreg0Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG0_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg0);
mlir_op_trait!(ReadPtxSregEnvreg0, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg0, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg0Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg0<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg0Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG0_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg0`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg1`.
pub const READ_PTX_SREG_ENVREG1_OPERATION: &str = "nvvm.read.ptx.sreg.envreg1";

/// Operation trait for `nvvm.read.ptx.sreg.envreg1`.
pub trait ReadPtxSregEnvreg1Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG1_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg1);
mlir_op_trait!(ReadPtxSregEnvreg1, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg1, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg1Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg1<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg1Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG1_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg1`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg2`.
pub const READ_PTX_SREG_ENVREG2_OPERATION: &str = "nvvm.read.ptx.sreg.envreg2";

/// Operation trait for `nvvm.read.ptx.sreg.envreg2`.
pub trait ReadPtxSregEnvreg2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG2_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg2);
mlir_op_trait!(ReadPtxSregEnvreg2, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg2, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg2Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG2_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg2`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg3`.
pub const READ_PTX_SREG_ENVREG3_OPERATION: &str = "nvvm.read.ptx.sreg.envreg3";

/// Operation trait for `nvvm.read.ptx.sreg.envreg3`.
pub trait ReadPtxSregEnvreg3Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG3_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg3);
mlir_op_trait!(ReadPtxSregEnvreg3, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg3, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg3Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg3<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg3Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG3_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg3`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg4`.
pub const READ_PTX_SREG_ENVREG4_OPERATION: &str = "nvvm.read.ptx.sreg.envreg4";

/// Operation trait for `nvvm.read.ptx.sreg.envreg4`.
pub trait ReadPtxSregEnvreg4Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG4_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg4);
mlir_op_trait!(ReadPtxSregEnvreg4, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg4, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg4Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg4<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg4Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG4_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg4`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg5`.
pub const READ_PTX_SREG_ENVREG5_OPERATION: &str = "nvvm.read.ptx.sreg.envreg5";

/// Operation trait for `nvvm.read.ptx.sreg.envreg5`.
pub trait ReadPtxSregEnvreg5Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG5_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg5);
mlir_op_trait!(ReadPtxSregEnvreg5, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg5, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg5Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg5<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg5Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG5_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg5`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg6`.
pub const READ_PTX_SREG_ENVREG6_OPERATION: &str = "nvvm.read.ptx.sreg.envreg6";

/// Operation trait for `nvvm.read.ptx.sreg.envreg6`.
pub trait ReadPtxSregEnvreg6Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG6_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg6);
mlir_op_trait!(ReadPtxSregEnvreg6, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg6, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg6Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg6<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg6Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG6_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg6`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg7`.
pub const READ_PTX_SREG_ENVREG7_OPERATION: &str = "nvvm.read.ptx.sreg.envreg7";

/// Operation trait for `nvvm.read.ptx.sreg.envreg7`.
pub trait ReadPtxSregEnvreg7Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG7_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg7);
mlir_op_trait!(ReadPtxSregEnvreg7, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg7, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg7Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg7<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg7Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG7_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg7`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg8`.
pub const READ_PTX_SREG_ENVREG8_OPERATION: &str = "nvvm.read.ptx.sreg.envreg8";

/// Operation trait for `nvvm.read.ptx.sreg.envreg8`.
pub trait ReadPtxSregEnvreg8Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG8_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg8);
mlir_op_trait!(ReadPtxSregEnvreg8, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg8, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg8Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg8<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg8Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG8_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg8`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg9`.
pub const READ_PTX_SREG_ENVREG9_OPERATION: &str = "nvvm.read.ptx.sreg.envreg9";

/// Operation trait for `nvvm.read.ptx.sreg.envreg9`.
pub trait ReadPtxSregEnvreg9Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG9_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg9);
mlir_op_trait!(ReadPtxSregEnvreg9, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg9, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg9Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg9<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg9Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG9_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg9`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg10`.
pub const READ_PTX_SREG_ENVREG10_OPERATION: &str = "nvvm.read.ptx.sreg.envreg10";

/// Operation trait for `nvvm.read.ptx.sreg.envreg10`.
pub trait ReadPtxSregEnvreg10Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG10_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg10);
mlir_op_trait!(ReadPtxSregEnvreg10, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg10, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg10Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg10<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg10Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG10_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg10`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg11`.
pub const READ_PTX_SREG_ENVREG11_OPERATION: &str = "nvvm.read.ptx.sreg.envreg11";

/// Operation trait for `nvvm.read.ptx.sreg.envreg11`.
pub trait ReadPtxSregEnvreg11Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG11_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg11);
mlir_op_trait!(ReadPtxSregEnvreg11, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg11, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg11Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg11<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg11Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG11_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg11`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg12`.
pub const READ_PTX_SREG_ENVREG12_OPERATION: &str = "nvvm.read.ptx.sreg.envreg12";

/// Operation trait for `nvvm.read.ptx.sreg.envreg12`.
pub trait ReadPtxSregEnvreg12Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG12_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg12);
mlir_op_trait!(ReadPtxSregEnvreg12, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg12, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg12Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg12<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg12Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG12_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg12`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg13`.
pub const READ_PTX_SREG_ENVREG13_OPERATION: &str = "nvvm.read.ptx.sreg.envreg13";

/// Operation trait for `nvvm.read.ptx.sreg.envreg13`.
pub trait ReadPtxSregEnvreg13Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG13_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg13);
mlir_op_trait!(ReadPtxSregEnvreg13, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg13, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg13Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg13<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg13Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG13_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg13`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg14`.
pub const READ_PTX_SREG_ENVREG14_OPERATION: &str = "nvvm.read.ptx.sreg.envreg14";

/// Operation trait for `nvvm.read.ptx.sreg.envreg14`.
pub trait ReadPtxSregEnvreg14Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG14_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg14);
mlir_op_trait!(ReadPtxSregEnvreg14, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg14, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg14Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg14<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg14Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG14_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg14`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg15`.
pub const READ_PTX_SREG_ENVREG15_OPERATION: &str = "nvvm.read.ptx.sreg.envreg15";

/// Operation trait for `nvvm.read.ptx.sreg.envreg15`.
pub trait ReadPtxSregEnvreg15Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG15_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg15);
mlir_op_trait!(ReadPtxSregEnvreg15, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg15, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg15Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg15<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg15Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG15_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg15`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg16`.
pub const READ_PTX_SREG_ENVREG16_OPERATION: &str = "nvvm.read.ptx.sreg.envreg16";

/// Operation trait for `nvvm.read.ptx.sreg.envreg16`.
pub trait ReadPtxSregEnvreg16Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG16_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg16);
mlir_op_trait!(ReadPtxSregEnvreg16, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg16, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg16Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg16<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg16Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG16_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg16`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg17`.
pub const READ_PTX_SREG_ENVREG17_OPERATION: &str = "nvvm.read.ptx.sreg.envreg17";

/// Operation trait for `nvvm.read.ptx.sreg.envreg17`.
pub trait ReadPtxSregEnvreg17Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG17_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg17);
mlir_op_trait!(ReadPtxSregEnvreg17, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg17, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg17Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg17<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg17Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG17_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg17`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg18`.
pub const READ_PTX_SREG_ENVREG18_OPERATION: &str = "nvvm.read.ptx.sreg.envreg18";

/// Operation trait for `nvvm.read.ptx.sreg.envreg18`.
pub trait ReadPtxSregEnvreg18Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG18_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg18);
mlir_op_trait!(ReadPtxSregEnvreg18, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg18, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg18Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg18<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg18Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG18_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg18`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg19`.
pub const READ_PTX_SREG_ENVREG19_OPERATION: &str = "nvvm.read.ptx.sreg.envreg19";

/// Operation trait for `nvvm.read.ptx.sreg.envreg19`.
pub trait ReadPtxSregEnvreg19Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG19_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg19);
mlir_op_trait!(ReadPtxSregEnvreg19, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg19, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg19Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg19<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg19Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG19_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg19`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg20`.
pub const READ_PTX_SREG_ENVREG20_OPERATION: &str = "nvvm.read.ptx.sreg.envreg20";

/// Operation trait for `nvvm.read.ptx.sreg.envreg20`.
pub trait ReadPtxSregEnvreg20Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG20_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg20);
mlir_op_trait!(ReadPtxSregEnvreg20, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg20, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg20Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg20<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg20Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG20_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg20`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg21`.
pub const READ_PTX_SREG_ENVREG21_OPERATION: &str = "nvvm.read.ptx.sreg.envreg21";

/// Operation trait for `nvvm.read.ptx.sreg.envreg21`.
pub trait ReadPtxSregEnvreg21Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG21_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg21);
mlir_op_trait!(ReadPtxSregEnvreg21, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg21, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg21Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg21<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg21Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG21_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg21`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg22`.
pub const READ_PTX_SREG_ENVREG22_OPERATION: &str = "nvvm.read.ptx.sreg.envreg22";

/// Operation trait for `nvvm.read.ptx.sreg.envreg22`.
pub trait ReadPtxSregEnvreg22Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG22_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg22);
mlir_op_trait!(ReadPtxSregEnvreg22, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg22, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg22Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg22<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg22Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG22_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg22`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg23`.
pub const READ_PTX_SREG_ENVREG23_OPERATION: &str = "nvvm.read.ptx.sreg.envreg23";

/// Operation trait for `nvvm.read.ptx.sreg.envreg23`.
pub trait ReadPtxSregEnvreg23Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG23_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg23);
mlir_op_trait!(ReadPtxSregEnvreg23, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg23, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg23Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg23<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg23Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG23_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg23`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg24`.
pub const READ_PTX_SREG_ENVREG24_OPERATION: &str = "nvvm.read.ptx.sreg.envreg24";

/// Operation trait for `nvvm.read.ptx.sreg.envreg24`.
pub trait ReadPtxSregEnvreg24Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG24_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg24);
mlir_op_trait!(ReadPtxSregEnvreg24, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg24, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg24Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg24<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg24Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG24_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg24`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg25`.
pub const READ_PTX_SREG_ENVREG25_OPERATION: &str = "nvvm.read.ptx.sreg.envreg25";

/// Operation trait for `nvvm.read.ptx.sreg.envreg25`.
pub trait ReadPtxSregEnvreg25Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG25_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg25);
mlir_op_trait!(ReadPtxSregEnvreg25, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg25, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg25Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg25<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg25Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG25_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg25`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg26`.
pub const READ_PTX_SREG_ENVREG26_OPERATION: &str = "nvvm.read.ptx.sreg.envreg26";

/// Operation trait for `nvvm.read.ptx.sreg.envreg26`.
pub trait ReadPtxSregEnvreg26Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG26_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg26);
mlir_op_trait!(ReadPtxSregEnvreg26, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg26, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg26Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg26<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg26Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG26_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg26`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg27`.
pub const READ_PTX_SREG_ENVREG27_OPERATION: &str = "nvvm.read.ptx.sreg.envreg27";

/// Operation trait for `nvvm.read.ptx.sreg.envreg27`.
pub trait ReadPtxSregEnvreg27Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG27_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg27);
mlir_op_trait!(ReadPtxSregEnvreg27, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg27, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg27Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg27<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg27Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG27_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg27`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg28`.
pub const READ_PTX_SREG_ENVREG28_OPERATION: &str = "nvvm.read.ptx.sreg.envreg28";

/// Operation trait for `nvvm.read.ptx.sreg.envreg28`.
pub trait ReadPtxSregEnvreg28Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG28_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg28);
mlir_op_trait!(ReadPtxSregEnvreg28, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg28, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg28Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg28<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg28Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG28_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg28`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg29`.
pub const READ_PTX_SREG_ENVREG29_OPERATION: &str = "nvvm.read.ptx.sreg.envreg29";

/// Operation trait for `nvvm.read.ptx.sreg.envreg29`.
pub trait ReadPtxSregEnvreg29Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG29_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg29);
mlir_op_trait!(ReadPtxSregEnvreg29, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg29, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg29Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg29<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg29Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG29_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg29`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg30`.
pub const READ_PTX_SREG_ENVREG30_OPERATION: &str = "nvvm.read.ptx.sreg.envreg30";

/// Operation trait for `nvvm.read.ptx.sreg.envreg30`.
pub trait ReadPtxSregEnvreg30Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG30_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg30);
mlir_op_trait!(ReadPtxSregEnvreg30, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg30, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg30Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg30<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg30Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG30_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg30`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.envreg31`.
pub const READ_PTX_SREG_ENVREG31_OPERATION: &str = "nvvm.read.ptx.sreg.envreg31";

/// Operation trait for `nvvm.read.ptx.sreg.envreg31`.
pub trait ReadPtxSregEnvreg31Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_ENVREG31_OPERATION
    }
}

mlir_op!(ReadPtxSregEnvreg31);
mlir_op_trait!(ReadPtxSregEnvreg31, ZeroRegions);
mlir_op_trait!(ReadPtxSregEnvreg31, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregEnvreg31Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_envreg31<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregEnvreg31Operation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_ENVREG31_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_envreg31`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.ex2`.
pub const EX2_OPERATION: &str = "nvvm.ex2";

/// Operation trait for `nvvm.ex2`.
pub trait Ex2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EX2_OPERATION
    }
}

mlir_op!(Ex2);
mlir_op_trait!(Ex2, ZeroRegions);
mlir_op_trait!(Ex2, ZeroSuccessors);

/// Constructs a new detached/owned [`Ex2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn ex2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedEx2Operation<'c, 't>, Error> {
    build_nvvm_operation(EX2_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::ex2`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.exit`.
pub const EXIT_OPERATION: &str = "nvvm.exit";

/// Operation trait for `nvvm.exit`.
pub trait ExitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        EXIT_OPERATION
    }
}

mlir_op!(Exit);
mlir_op_trait!(Exit, ZeroRegions);
mlir_op_trait!(Exit, ZeroSuccessors);

/// Constructs a new detached/owned [`ExitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn exit<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedExitOperation<'c, 't>, Error> {
    build_nvvm_operation(EXIT_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::exit`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.fence.mbarrier.init`.
pub const FENCE_MBARRIER_INIT_OPERATION: &str = "nvvm.fence.mbarrier.init";

/// Operation trait for `nvvm.fence.mbarrier.init`.
pub trait FenceMbarrierInitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_MBARRIER_INIT_OPERATION
    }
}

mlir_op!(FenceMbarrierInit);
mlir_op_trait!(FenceMbarrierInit, ZeroRegions);
mlir_op_trait!(FenceMbarrierInit, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceMbarrierInitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_mbarrier_init<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceMbarrierInitOperation<'c, 't>, Error> {
    build_nvvm_operation(
        FENCE_MBARRIER_INIT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_mbarrier_init`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.proxy.acquire`.
pub const FENCE_PROXY_ACQUIRE_OPERATION: &str = "nvvm.fence.proxy.acquire";

/// Operation trait for `nvvm.fence.proxy.acquire`.
pub trait FenceProxyAcquireOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_PROXY_ACQUIRE_OPERATION
    }
}

mlir_op!(FenceProxyAcquire);
mlir_op_trait!(FenceProxyAcquire, ZeroRegions);
mlir_op_trait!(FenceProxyAcquire, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceProxyAcquireOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_proxy_acquire<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceProxyAcquireOperation<'c, 't>, Error> {
    build_nvvm_operation(
        FENCE_PROXY_ACQUIRE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_proxy_acquire`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.proxy`.
pub const FENCE_PROXY_OPERATION: &str = "nvvm.fence.proxy";

/// Operation trait for `nvvm.fence.proxy`.
pub trait FenceProxyOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_PROXY_OPERATION
    }
}

mlir_op!(FenceProxy);
mlir_op_trait!(FenceProxy, ZeroRegions);
mlir_op_trait!(FenceProxy, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceProxyOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_proxy<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceProxyOperation<'c, 't>, Error> {
    build_nvvm_operation(FENCE_PROXY_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_proxy`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.proxy.release`.
pub const FENCE_PROXY_RELEASE_OPERATION: &str = "nvvm.fence.proxy.release";

/// Operation trait for `nvvm.fence.proxy.release`.
pub trait FenceProxyReleaseOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_PROXY_RELEASE_OPERATION
    }
}

mlir_op!(FenceProxyRelease);
mlir_op_trait!(FenceProxyRelease, ZeroRegions);
mlir_op_trait!(FenceProxyRelease, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceProxyReleaseOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_proxy_release<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceProxyReleaseOperation<'c, 't>, Error> {
    build_nvvm_operation(
        FENCE_PROXY_RELEASE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_proxy_release`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.proxy.sync_restrict`.
pub const FENCE_PROXY_SYNC_RESTRICT_OPERATION: &str = "nvvm.fence.proxy.sync_restrict";

/// Operation trait for `nvvm.fence.proxy.sync_restrict`.
pub trait FenceProxySyncRestrictOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_PROXY_SYNC_RESTRICT_OPERATION
    }
}

mlir_op!(FenceProxySyncRestrict);
mlir_op_trait!(FenceProxySyncRestrict, ZeroRegions);
mlir_op_trait!(FenceProxySyncRestrict, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceProxySyncRestrictOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_proxy_sync_restrict<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceProxySyncRestrictOperation<'c, 't>, Error> {
    build_nvvm_operation(
        FENCE_PROXY_SYNC_RESTRICT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_proxy_sync_restrict`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.sc.cluster`.
pub const FENCE_SC_CLUSTER_OPERATION: &str = "nvvm.fence.sc.cluster";

/// Operation trait for `nvvm.fence.sc.cluster`.
pub trait FenceScClusterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_SC_CLUSTER_OPERATION
    }
}

mlir_op!(FenceScCluster);
mlir_op_trait!(FenceScCluster, ZeroRegions);
mlir_op_trait!(FenceScCluster, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceScClusterOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_sc_cluster<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceScClusterOperation<'c, 't>, Error> {
    build_nvvm_operation(FENCE_SC_CLUSTER_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_sc_cluster`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.fence.sync_restrict`.
pub const FENCE_SYNC_RESTRICT_OPERATION: &str = "nvvm.fence.sync_restrict";

/// Operation trait for `nvvm.fence.sync_restrict`.
pub trait FenceSyncRestrictOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FENCE_SYNC_RESTRICT_OPERATION
    }
}

mlir_op!(FenceSyncRestrict);
mlir_op_trait!(FenceSyncRestrict, ZeroRegions);
mlir_op_trait!(FenceSyncRestrict, ZeroSuccessors);

/// Constructs a new detached/owned [`FenceSyncRestrictOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fence_sync_restrict<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFenceSyncRestrictOperation<'c, 't>, Error> {
    build_nvvm_operation(
        FENCE_SYNC_RESTRICT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fence_sync_restrict`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.fma`.
pub const FMA_OPERATION: &str = "nvvm.fma";

/// Operation trait for `nvvm.fma`.
pub trait FmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        FMA_OPERATION
    }
}

mlir_op!(Fma);
mlir_op_trait!(Fma, ZeroRegions);
mlir_op_trait!(Fma, ZeroSuccessors);

/// Constructs a new detached/owned [`FmaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn fma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedFmaOperation<'c, 't>, Error> {
    build_nvvm_operation(FMA_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::fma`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.globaltimer.lo`.
pub const READ_PTX_SREG_GLOBALTIMER_LO_OPERATION: &str = "nvvm.read.ptx.sreg.globaltimer.lo";

/// Operation trait for `nvvm.read.ptx.sreg.globaltimer.lo`.
pub trait ReadPtxSregGlobaltimerLoOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_GLOBALTIMER_LO_OPERATION
    }
}

mlir_op!(ReadPtxSregGlobaltimerLo);
mlir_op_trait!(ReadPtxSregGlobaltimerLo, ZeroRegions);
mlir_op_trait!(ReadPtxSregGlobaltimerLo, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregGlobaltimerLoOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_globaltimer_lo<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregGlobaltimerLoOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_GLOBALTIMER_LO_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_globaltimer_lo`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.globaltimer`.
pub const READ_PTX_SREG_GLOBALTIMER_OPERATION: &str = "nvvm.read.ptx.sreg.globaltimer";

/// Operation trait for `nvvm.read.ptx.sreg.globaltimer`.
pub trait ReadPtxSregGlobaltimerOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_GLOBALTIMER_OPERATION
    }
}

mlir_op!(ReadPtxSregGlobaltimer);
mlir_op_trait!(ReadPtxSregGlobaltimer, ZeroRegions);
mlir_op_trait!(ReadPtxSregGlobaltimer, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregGlobaltimerOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_globaltimer<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregGlobaltimerOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_GLOBALTIMER_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_globaltimer`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nctaid.x`.
pub const READ_PTX_SREG_NCTAID_X_OPERATION: &str = "nvvm.read.ptx.sreg.nctaid.x";

/// Operation trait for `nvvm.read.ptx.sreg.nctaid.x`.
pub trait ReadPtxSregNctaidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCTAID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregNctaidX);
mlir_op_trait!(ReadPtxSregNctaidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregNctaidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNctaidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nctaid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNctaidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCTAID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nctaid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nctaid.y`.
pub const READ_PTX_SREG_NCTAID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.nctaid.y";

/// Operation trait for `nvvm.read.ptx.sreg.nctaid.y`.
pub trait ReadPtxSregNctaidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCTAID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregNctaidY);
mlir_op_trait!(ReadPtxSregNctaidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregNctaidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNctaidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nctaid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNctaidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCTAID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nctaid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nctaid.z`.
pub const READ_PTX_SREG_NCTAID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.nctaid.z";

/// Operation trait for `nvvm.read.ptx.sreg.nctaid.z`.
pub trait ReadPtxSregNctaidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NCTAID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregNctaidZ);
mlir_op_trait!(ReadPtxSregNctaidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregNctaidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNctaidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nctaid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNctaidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NCTAID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nctaid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.gridid`.
pub const READ_PTX_SREG_GRIDID_OPERATION: &str = "nvvm.read.ptx.sreg.gridid";

/// Operation trait for `nvvm.read.ptx.sreg.gridid`.
pub trait ReadPtxSregGrididOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_GRIDID_OPERATION
    }
}

mlir_op!(ReadPtxSregGridid);
mlir_op_trait!(ReadPtxSregGridid, ZeroRegions);
mlir_op_trait!(ReadPtxSregGridid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregGrididOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_gridid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregGrididOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_GRIDID_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_gridid`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.griddepcontrol`.
pub const GRIDDEPCONTROL_OPERATION: &str = "nvvm.griddepcontrol";

/// Operation trait for `nvvm.griddepcontrol`.
pub trait GriddepcontrolOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        GRIDDEPCONTROL_OPERATION
    }
}

mlir_op!(Griddepcontrol);
mlir_op_trait!(Griddepcontrol, ZeroRegions);
mlir_op_trait!(Griddepcontrol, ZeroSuccessors);

/// Constructs a new detached/owned [`GriddepcontrolOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn griddepcontrol<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedGriddepcontrolOperation<'c, 't>, Error> {
    build_nvvm_operation(GRIDDEPCONTROL_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::griddepcontrol`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.inline_ptx`.
pub const INLINE_PTX_OPERATION: &str = "nvvm.inline_ptx";

/// Operation trait for `nvvm.inline_ptx`.
pub trait InlinePtxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        INLINE_PTX_OPERATION
    }
}

mlir_op!(InlinePtx);
mlir_op_trait!(InlinePtx, ZeroRegions);
mlir_op_trait!(InlinePtx, ZeroSuccessors);

/// Constructs a new detached/owned [`InlinePtxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn inline_ptx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedInlinePtxOperation<'c, 't>, Error> {
    build_nvvm_operation(INLINE_PTX_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::inline_ptx`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.laneid`.
pub const READ_PTX_SREG_LANEID_OPERATION: &str = "nvvm.read.ptx.sreg.laneid";

/// Operation trait for `nvvm.read.ptx.sreg.laneid`.
pub trait ReadPtxSregLaneidOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEID_OPERATION
    }
}

mlir_op!(ReadPtxSregLaneid);
mlir_op_trait!(ReadPtxSregLaneid, ZeroRegions);
mlir_op_trait!(ReadPtxSregLaneid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLaneidOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_laneid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLaneidOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEID_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_laneid`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.lanemask.eq`.
pub const READ_PTX_SREG_LANEMASK_EQ_OPERATION: &str = "nvvm.read.ptx.sreg.lanemask.eq";

/// Operation trait for `nvvm.read.ptx.sreg.lanemask.eq`.
pub trait ReadPtxSregLanemaskEqOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEMASK_EQ_OPERATION
    }
}

mlir_op!(ReadPtxSregLanemaskEq);
mlir_op_trait!(ReadPtxSregLanemaskEq, ZeroRegions);
mlir_op_trait!(ReadPtxSregLanemaskEq, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLanemaskEqOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_lanemask_eq<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLanemaskEqOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEMASK_EQ_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_lanemask_eq`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.lanemask.ge`.
pub const READ_PTX_SREG_LANEMASK_GE_OPERATION: &str = "nvvm.read.ptx.sreg.lanemask.ge";

/// Operation trait for `nvvm.read.ptx.sreg.lanemask.ge`.
pub trait ReadPtxSregLanemaskGeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEMASK_GE_OPERATION
    }
}

mlir_op!(ReadPtxSregLanemaskGe);
mlir_op_trait!(ReadPtxSregLanemaskGe, ZeroRegions);
mlir_op_trait!(ReadPtxSregLanemaskGe, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLanemaskGeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_lanemask_ge<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLanemaskGeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEMASK_GE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_lanemask_ge`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.lanemask.gt`.
pub const READ_PTX_SREG_LANEMASK_GT_OPERATION: &str = "nvvm.read.ptx.sreg.lanemask.gt";

/// Operation trait for `nvvm.read.ptx.sreg.lanemask.gt`.
pub trait ReadPtxSregLanemaskGtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEMASK_GT_OPERATION
    }
}

mlir_op!(ReadPtxSregLanemaskGt);
mlir_op_trait!(ReadPtxSregLanemaskGt, ZeroRegions);
mlir_op_trait!(ReadPtxSregLanemaskGt, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLanemaskGtOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_lanemask_gt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLanemaskGtOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEMASK_GT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_lanemask_gt`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.lanemask.le`.
pub const READ_PTX_SREG_LANEMASK_LE_OPERATION: &str = "nvvm.read.ptx.sreg.lanemask.le";

/// Operation trait for `nvvm.read.ptx.sreg.lanemask.le`.
pub trait ReadPtxSregLanemaskLeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEMASK_LE_OPERATION
    }
}

mlir_op!(ReadPtxSregLanemaskLe);
mlir_op_trait!(ReadPtxSregLanemaskLe, ZeroRegions);
mlir_op_trait!(ReadPtxSregLanemaskLe, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLanemaskLeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_lanemask_le<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLanemaskLeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEMASK_LE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_lanemask_le`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.lanemask.lt`.
pub const READ_PTX_SREG_LANEMASK_LT_OPERATION: &str = "nvvm.read.ptx.sreg.lanemask.lt";

/// Operation trait for `nvvm.read.ptx.sreg.lanemask.lt`.
pub trait ReadPtxSregLanemaskLtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_LANEMASK_LT_OPERATION
    }
}

mlir_op!(ReadPtxSregLanemaskLt);
mlir_op_trait!(ReadPtxSregLanemaskLt, ZeroRegions);
mlir_op_trait!(ReadPtxSregLanemaskLt, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregLanemaskLtOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_lanemask_lt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregLanemaskLtOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_LANEMASK_LT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_lanemask_lt`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.ldmatrix`.
pub const LDMATRIX_OPERATION: &str = "nvvm.ldmatrix";

/// Operation trait for `nvvm.ldmatrix`.
pub trait LdmatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LDMATRIX_OPERATION
    }
}

mlir_op!(Ldmatrix);
mlir_op_trait!(Ldmatrix, ZeroRegions);
mlir_op_trait!(Ldmatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`LdmatrixOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn ldmatrix<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedLdmatrixOperation<'c, 't>, Error> {
    build_nvvm_operation(LDMATRIX_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::ldmatrix`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.log2`.
pub const LOG2_OPERATION: &str = "nvvm.log2";

/// Operation trait for `nvvm.log2`.
pub trait Log2Operation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        LOG2_OPERATION
    }
}

mlir_op!(Log2);
mlir_op_trait!(Log2, ZeroRegions);
mlir_op_trait!(Log2, ZeroSuccessors);

/// Constructs a new detached/owned [`Log2Operation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn log2<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedLog2Operation<'c, 't>, Error> {
    build_nvvm_operation(LOG2_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::log2`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive_drop.expect_tx`.
pub const MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION: &str = "nvvm.mbarrier.arrive_drop.expect_tx";

/// Operation trait for `nvvm.mbarrier.arrive_drop.expect_tx`.
pub trait MbarrierArriveDropExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION
    }
}

mlir_op!(MbarrierArriveDropExpectTx);
mlir_op_trait!(MbarrierArriveDropExpectTx, ZeroRegions);
mlir_op_trait!(MbarrierArriveDropExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveDropExpectTxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive_drop_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveDropExpectTxOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive_drop_expect_tx`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive_drop.nocomplete`.
pub const MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION: &str = "nvvm.mbarrier.arrive_drop.nocomplete";

/// Operation trait for `nvvm.mbarrier.arrive_drop.nocomplete`.
pub trait MbarrierArriveDropNocompleteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION
    }
}

mlir_op!(MbarrierArriveDropNocomplete);
mlir_op_trait!(MbarrierArriveDropNocomplete, ZeroRegions);
mlir_op_trait!(MbarrierArriveDropNocomplete, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveDropNocompleteOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive_drop_nocomplete<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveDropNocompleteOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive_drop_nocomplete`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive_drop`.
pub const MBARRIER_ARRIVE_DROP_OPERATION: &str = "nvvm.mbarrier.arrive_drop";

/// Operation trait for `nvvm.mbarrier.arrive_drop`.
pub trait MbarrierArriveDropOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_DROP_OPERATION
    }
}

mlir_op!(MbarrierArriveDrop);
mlir_op_trait!(MbarrierArriveDrop, ZeroRegions);
mlir_op_trait!(MbarrierArriveDrop, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveDropOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive_drop<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveDropOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_ARRIVE_DROP_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive_drop`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive.expect_tx`.
pub const MBARRIER_ARRIVE_EXPECT_TX_OPERATION: &str = "nvvm.mbarrier.arrive.expect_tx";

/// Operation trait for `nvvm.mbarrier.arrive.expect_tx`.
pub trait MbarrierArriveExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_EXPECT_TX_OPERATION
    }
}

mlir_op!(MbarrierArriveExpectTx);
mlir_op_trait!(MbarrierArriveExpectTx, ZeroRegions);
mlir_op_trait!(MbarrierArriveExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveExpectTxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveExpectTxOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_ARRIVE_EXPECT_TX_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive_expect_tx`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive.nocomplete`.
pub const MBARRIER_ARRIVE_NOCOMPLETE_OPERATION: &str = "nvvm.mbarrier.arrive.nocomplete";

/// Operation trait for `nvvm.mbarrier.arrive.nocomplete`.
pub trait MbarrierArriveNocompleteOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_NOCOMPLETE_OPERATION
    }
}

mlir_op!(MbarrierArriveNocomplete);
mlir_op_trait!(MbarrierArriveNocomplete, ZeroRegions);
mlir_op_trait!(MbarrierArriveNocomplete, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveNocompleteOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive_nocomplete<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveNocompleteOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_ARRIVE_NOCOMPLETE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive_nocomplete`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.arrive`.
pub const MBARRIER_ARRIVE_OPERATION: &str = "nvvm.mbarrier.arrive";

/// Operation trait for `nvvm.mbarrier.arrive`.
pub trait MbarrierArriveOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_ARRIVE_OPERATION
    }
}

mlir_op!(MbarrierArrive);
mlir_op_trait!(MbarrierArrive, ZeroRegions);
mlir_op_trait!(MbarrierArrive, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierArriveOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_arrive<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierArriveOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_ARRIVE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_arrive`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.complete_tx`.
pub const MBARRIER_COMPLETE_TX_OPERATION: &str = "nvvm.mbarrier.complete_tx";

/// Operation trait for `nvvm.mbarrier.complete_tx`.
pub trait MbarrierCompleteTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_COMPLETE_TX_OPERATION
    }
}

mlir_op!(MbarrierCompleteTx);
mlir_op_trait!(MbarrierCompleteTx, ZeroRegions);
mlir_op_trait!(MbarrierCompleteTx, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierCompleteTxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_complete_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierCompleteTxOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_COMPLETE_TX_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_complete_tx`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.expect_tx`.
pub const MBARRIER_EXPECT_TX_OPERATION: &str = "nvvm.mbarrier.expect_tx";

/// Operation trait for `nvvm.mbarrier.expect_tx`.
pub trait MbarrierExpectTxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_EXPECT_TX_OPERATION
    }
}

mlir_op!(MbarrierExpectTx);
mlir_op_trait!(MbarrierExpectTx, ZeroRegions);
mlir_op_trait!(MbarrierExpectTx, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierExpectTxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_expect_tx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierExpectTxOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_EXPECT_TX_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_expect_tx`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.init`.
pub const MBARRIER_INIT_OPERATION: &str = "nvvm.mbarrier.init";

/// Operation trait for `nvvm.mbarrier.init`.
pub trait MbarrierInitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_INIT_OPERATION
    }
}

mlir_op!(MbarrierInit);
mlir_op_trait!(MbarrierInit, ZeroRegions);
mlir_op_trait!(MbarrierInit, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierInitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_init<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierInitOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_INIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_init`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.inval`.
pub const MBARRIER_INVAL_OPERATION: &str = "nvvm.mbarrier.inval";

/// Operation trait for `nvvm.mbarrier.inval`.
pub trait MbarrierInvalOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_INVAL_OPERATION
    }
}

mlir_op!(MbarrierInval);
mlir_op_trait!(MbarrierInval, ZeroRegions);
mlir_op_trait!(MbarrierInval, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierInvalOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_inval<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierInvalOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_INVAL_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_inval`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.test.wait`.
pub const MBARRIER_TEST_WAIT_OPERATION: &str = "nvvm.mbarrier.test.wait";

/// Operation trait for `nvvm.mbarrier.test.wait`.
pub trait MbarrierTestWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_TEST_WAIT_OPERATION
    }
}

mlir_op!(MbarrierTestWait);
mlir_op_trait!(MbarrierTestWait, ZeroRegions);
mlir_op_trait!(MbarrierTestWait, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierTestWaitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_test_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierTestWaitOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_TEST_WAIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_test_wait`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.try_wait`.
pub const MBARRIER_TRY_WAIT_OPERATION: &str = "nvvm.mbarrier.try_wait";

/// Operation trait for `nvvm.mbarrier.try_wait`.
pub trait MbarrierTryWaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_TRY_WAIT_OPERATION
    }
}

mlir_op!(MbarrierTryWait);
mlir_op_trait!(MbarrierTryWait, ZeroRegions);
mlir_op_trait!(MbarrierTryWait, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierTryWaitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_try_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierTryWaitOperation<'c, 't>, Error> {
    build_nvvm_operation(MBARRIER_TRY_WAIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_try_wait`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mbarrier.try_wait.parity`.
pub const MBARRIER_TRY_WAIT_PARITY_OPERATION: &str = "nvvm.mbarrier.try_wait.parity";

/// Operation trait for `nvvm.mbarrier.try_wait.parity`.
pub trait MbarrierTryWaitParityOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MBARRIER_TRY_WAIT_PARITY_OPERATION
    }
}

mlir_op!(MbarrierTryWaitParity);
mlir_op_trait!(MbarrierTryWaitParity, ZeroRegions);
mlir_op_trait!(MbarrierTryWaitParity, ZeroSuccessors);

/// Constructs a new detached/owned [`MbarrierTryWaitParityOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mbarrier_try_wait_parity<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMbarrierTryWaitParityOperation<'c, 't>, Error> {
    build_nvvm_operation(
        MBARRIER_TRY_WAIT_PARITY_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mbarrier_try_wait_parity`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.mapa`.
pub const MAPA_OPERATION: &str = "nvvm.mapa";

/// Operation trait for `nvvm.mapa`.
pub trait MapaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MAPA_OPERATION
    }
}

mlir_op!(Mapa);
mlir_op_trait!(Mapa, ZeroRegions);
mlir_op_trait!(Mapa, ZeroSuccessors);

/// Constructs a new detached/owned [`MapaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mapa<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMapaOperation<'c, 't>, Error> {
    build_nvvm_operation(MAPA_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mapa`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.match.sync`.
pub const MATCH_SYNC_OPERATION: &str = "nvvm.match.sync";

/// Operation trait for `nvvm.match.sync`.
pub trait MatchSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MATCH_SYNC_OPERATION
    }
}

mlir_op!(MatchSync);
mlir_op_trait!(MatchSync, ZeroRegions);
mlir_op_trait!(MatchSync, ZeroSuccessors);

/// Constructs a new detached/owned [`MatchSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn match_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMatchSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(MATCH_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::match_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.memory.barrier`.
pub const MEMORY_BARRIER_OPERATION: &str = "nvvm.memory.barrier";

/// Operation trait for `nvvm.memory.barrier`.
pub trait MemoryBarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MEMORY_BARRIER_OPERATION
    }
}

mlir_op!(MemoryBarrier);
mlir_op_trait!(MemoryBarrier, ZeroRegions);
mlir_op_trait!(MemoryBarrier, ZeroSuccessors);

/// Constructs a new detached/owned [`MemoryBarrierOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn memory_barrier<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMemoryBarrierOperation<'c, 't>, Error> {
    build_nvvm_operation(MEMORY_BARRIER_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::memory_barrier`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mma.block_scale`.
pub const MMA_BLOCK_SCALE_OPERATION: &str = "nvvm.mma.block_scale";

/// Operation trait for `nvvm.mma.block_scale`.
pub trait MmaBlockScaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MMA_BLOCK_SCALE_OPERATION
    }
}

mlir_op!(MmaBlockScale);
mlir_op_trait!(MmaBlockScale, ZeroRegions);
mlir_op_trait!(MmaBlockScale, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaBlockScaleOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mma_block_scale<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMmaBlockScaleOperation<'c, 't>, Error> {
    build_nvvm_operation(MMA_BLOCK_SCALE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mma_block_scale`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mma.sync`.
pub const MMA_SYNC_OPERATION: &str = "nvvm.mma.sync";

/// Operation trait for `nvvm.mma.sync`.
pub trait MmaSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MMA_SYNC_OPERATION
    }
}

mlir_op!(MmaSync);
mlir_op_trait!(MmaSync, ZeroRegions);
mlir_op_trait!(MmaSync, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mma_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMmaSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(MMA_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mma_sync`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.mma.sp.block_scale`.
pub const MMA_SP_BLOCK_SCALE_OPERATION: &str = "nvvm.mma.sp.block_scale";

/// Operation trait for `nvvm.mma.sp.block_scale`.
pub trait MmaSpBlockScaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MMA_SP_BLOCK_SCALE_OPERATION
    }
}

mlir_op!(MmaSpBlockScale);
mlir_op_trait!(MmaSpBlockScale, ZeroRegions);
mlir_op_trait!(MmaSpBlockScale, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaSpBlockScaleOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mma_sp_block_scale<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMmaSpBlockScaleOperation<'c, 't>, Error> {
    build_nvvm_operation(MMA_SP_BLOCK_SCALE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mma_sp_block_scale`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.mma.sp.sync`.
pub const MMA_SP_SYNC_OPERATION: &str = "nvvm.mma.sp.sync";

/// Operation trait for `nvvm.mma.sp.sync`.
pub trait MmaSpSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MMA_SP_SYNC_OPERATION
    }
}

mlir_op!(MmaSpSync);
mlir_op_trait!(MmaSpSync, ZeroRegions);
mlir_op_trait!(MmaSpSync, ZeroSuccessors);

/// Constructs a new detached/owned [`MmaSpSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn mma_sp_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMmaSpSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(MMA_SP_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::mma_sp_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.movmatrix`.
pub const MOVMATRIX_OPERATION: &str = "nvvm.movmatrix";

/// Operation trait for `nvvm.movmatrix`.
pub trait MovmatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        MOVMATRIX_OPERATION
    }
}

mlir_op!(Movmatrix);
mlir_op_trait!(Movmatrix, ZeroRegions);
mlir_op_trait!(Movmatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`MovmatrixOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn movmatrix<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedMovmatrixOperation<'c, 't>, Error> {
    build_nvvm_operation(MOVMATRIX_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::movmatrix`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.nanosleep`.
pub const NANOSLEEP_OPERATION: &str = "nvvm.nanosleep";

/// Operation trait for `nvvm.nanosleep`.
pub trait NanosleepOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        NANOSLEEP_OPERATION
    }
}

mlir_op!(Nanosleep);
mlir_op_trait!(Nanosleep, ZeroRegions);
mlir_op_trait!(Nanosleep, ZeroSuccessors);

/// Constructs a new detached/owned [`NanosleepOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn nanosleep<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedNanosleepOperation<'c, 't>, Error> {
    build_nvvm_operation(NANOSLEEP_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::nanosleep`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.pmevent`.
pub const PMEVENT_OPERATION: &str = "nvvm.pmevent";

/// Operation trait for `nvvm.pmevent`.
pub trait PmeventOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PMEVENT_OPERATION
    }
}

mlir_op!(Pmevent);
mlir_op_trait!(Pmevent, ZeroRegions);
mlir_op_trait!(Pmevent, ZeroSuccessors);

/// Constructs a new detached/owned [`PmeventOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn pmevent<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedPmeventOperation<'c, 't>, Error> {
    build_nvvm_operation(PMEVENT_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::pmevent`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.prmt`.
pub const PRMT_OPERATION: &str = "nvvm.prmt";

/// Operation trait for `nvvm.prmt`.
pub trait PrmtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PRMT_OPERATION
    }
}

mlir_op!(Prmt);
mlir_op_trait!(Prmt, ZeroRegions);
mlir_op_trait!(Prmt, ZeroSuccessors);

/// Constructs a new detached/owned [`PrmtOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn prmt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedPrmtOperation<'c, 't>, Error> {
    build_nvvm_operation(PRMT_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::prmt`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.prefetch`.
pub const PREFETCH_OPERATION: &str = "nvvm.prefetch";

/// Operation trait for `nvvm.prefetch`.
pub trait PrefetchOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        PREFETCH_OPERATION
    }
}

mlir_op!(Prefetch);
mlir_op_trait!(Prefetch, ZeroRegions);
mlir_op_trait!(Prefetch, ZeroSuccessors);

/// Constructs a new detached/owned [`PrefetchOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn prefetch<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedPrefetchOperation<'c, 't>, Error> {
    build_nvvm_operation(PREFETCH_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::prefetch`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.rcp.approx.ftz.f`.
pub const RCP_APPROX_FTZ_F_OPERATION: &str = "nvvm.rcp.approx.ftz.f";

/// Operation trait for `nvvm.rcp.approx.ftz.f`.
pub trait RcpApproxFtzFOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        RCP_APPROX_FTZ_F_OPERATION
    }
}

mlir_op!(RcpApproxFtzF);
mlir_op_trait!(RcpApproxFtzF, ZeroRegions);
mlir_op_trait!(RcpApproxFtzF, ZeroSuccessors);

/// Constructs a new detached/owned [`RcpApproxFtzFOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn rcp_approx_ftz_f<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedRcpApproxFtzFOperation<'c, 't>, Error> {
    build_nvvm_operation(RCP_APPROX_FTZ_F_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::rcp_approx_ftz_f`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.redux.sync`.
pub const REDUX_SYNC_OPERATION: &str = "nvvm.redux.sync";

/// Operation trait for `nvvm.redux.sync`.
pub trait ReduxSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        REDUX_SYNC_OPERATION
    }
}

mlir_op!(ReduxSync);
mlir_op_trait!(ReduxSync, ZeroRegions);
mlir_op_trait!(ReduxSync, ZeroSuccessors);

/// Constructs a new detached/owned [`ReduxSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn redux_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReduxSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(REDUX_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::redux_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.rsqrt`.
pub const RSQRT_OPERATION: &str = "nvvm.rsqrt";

/// Operation trait for `nvvm.rsqrt`.
pub trait RsqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        RSQRT_OPERATION
    }
}

mlir_op!(Rsqrt);
mlir_op_trait!(Rsqrt, ZeroRegions);
mlir_op_trait!(Rsqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`RsqrtOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn rsqrt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedRsqrtOperation<'c, 't>, Error> {
    build_nvvm_operation(RSQRT_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::rsqrt`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.setmaxregister`.
pub const SETMAXREGISTER_OPERATION: &str = "nvvm.setmaxregister";

/// Operation trait for `nvvm.setmaxregister`.
pub trait SetmaxregisterOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SETMAXREGISTER_OPERATION
    }
}

mlir_op!(Setmaxregister);
mlir_op_trait!(Setmaxregister, ZeroRegions);
mlir_op_trait!(Setmaxregister, ZeroSuccessors);

/// Constructs a new detached/owned [`SetmaxregisterOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn setmaxregister<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedSetmaxregisterOperation<'c, 't>, Error> {
    build_nvvm_operation(SETMAXREGISTER_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::setmaxregister`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.shfl.sync`.
pub const SHFL_SYNC_OPERATION: &str = "nvvm.shfl.sync";

/// Operation trait for `nvvm.shfl.sync`.
pub trait ShflSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SHFL_SYNC_OPERATION
    }
}

mlir_op!(ShflSync);
mlir_op_trait!(ShflSync, ZeroRegions);
mlir_op_trait!(ShflSync, ZeroSuccessors);

/// Constructs a new detached/owned [`ShflSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn shfl_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedShflSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(SHFL_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::shfl_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.sin`.
pub const SIN_OPERATION: &str = "nvvm.sin";

/// Operation trait for `nvvm.sin`.
pub trait SinOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SIN_OPERATION
    }
}

mlir_op!(Sin);
mlir_op_trait!(Sin, ZeroRegions);
mlir_op_trait!(Sin, ZeroSuccessors);

/// Constructs a new detached/owned [`SinOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn sin<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedSinOperation<'c, 't>, Error> {
    build_nvvm_operation(SIN_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::sin`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nsmid`.
pub const READ_PTX_SREG_NSMID_OPERATION: &str = "nvvm.read.ptx.sreg.nsmid";

/// Operation trait for `nvvm.read.ptx.sreg.nsmid`.
pub trait ReadPtxSregNsmidOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NSMID_OPERATION
    }
}

mlir_op!(ReadPtxSregNsmid);
mlir_op_trait!(ReadPtxSregNsmid, ZeroRegions);
mlir_op_trait!(ReadPtxSregNsmid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNsmidOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nsmid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNsmidOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NSMID_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nsmid`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.smid`.
pub const READ_PTX_SREG_SMID_OPERATION: &str = "nvvm.read.ptx.sreg.smid";

/// Operation trait for `nvvm.read.ptx.sreg.smid`.
pub trait ReadPtxSregSmidOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_SMID_OPERATION
    }
}

mlir_op!(ReadPtxSregSmid);
mlir_op_trait!(ReadPtxSregSmid, ZeroRegions);
mlir_op_trait!(ReadPtxSregSmid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregSmidOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_smid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregSmidOperation<'c, 't>, Error> {
    build_nvvm_operation(READ_PTX_SREG_SMID_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_smid`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.sqrt.approx`.
pub const SQRT_APPROX_OPERATION: &str = "nvvm.sqrt.approx";

/// Operation trait for `nvvm.sqrt.approx`.
pub trait SqrtApproxOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SQRT_APPROX_OPERATION
    }
}

mlir_op!(SqrtApprox);
mlir_op_trait!(SqrtApprox, ZeroRegions);
mlir_op_trait!(SqrtApprox, ZeroSuccessors);

/// Constructs a new detached/owned [`SqrtApproxOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn sqrt_approx<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedSqrtApproxOperation<'c, 't>, Error> {
    build_nvvm_operation(SQRT_APPROX_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::sqrt_approx`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.sqrt`.
pub const SQRT_OPERATION: &str = "nvvm.sqrt";

/// Operation trait for `nvvm.sqrt`.
pub trait SqrtOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SQRT_OPERATION
    }
}

mlir_op!(Sqrt);
mlir_op_trait!(Sqrt, ZeroRegions);
mlir_op_trait!(Sqrt, ZeroSuccessors);

/// Constructs a new detached/owned [`SqrtOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn sqrt<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedSqrtOperation<'c, 't>, Error> {
    build_nvvm_operation(SQRT_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::sqrt`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.stmatrix`.
pub const STMATRIX_OPERATION: &str = "nvvm.stmatrix";

/// Operation trait for `nvvm.stmatrix`.
pub trait StmatrixOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        STMATRIX_OPERATION
    }
}

mlir_op!(Stmatrix);
mlir_op_trait!(Stmatrix, ZeroRegions);
mlir_op_trait!(Stmatrix, ZeroSuccessors);

/// Constructs a new detached/owned [`StmatrixOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn stmatrix<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedStmatrixOperation<'c, 't>, Error> {
    build_nvvm_operation(STMATRIX_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::stmatrix`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.subf`.
pub const SUBF_OPERATION: &str = "nvvm.subf";

/// Operation trait for `nvvm.subf`.
pub trait SubfOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        SUBF_OPERATION
    }
}

mlir_op!(Subf);
mlir_op_trait!(Subf, ZeroRegions);
mlir_op_trait!(Subf, ZeroSuccessors);

/// Constructs a new detached/owned [`SubfOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn subf<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedSubfOperation<'c, 't>, Error> {
    build_nvvm_operation(SUBF_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::subf`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.bar.warp.sync`.
pub const BAR_WARP_SYNC_OPERATION: &str = "nvvm.bar.warp.sync";

/// Operation trait for `nvvm.bar.warp.sync`.
pub trait BarWarpSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        BAR_WARP_SYNC_OPERATION
    }
}

mlir_op!(BarWarpSync);
mlir_op_trait!(BarWarpSync, ZeroRegions);
mlir_op_trait!(BarWarpSync, ZeroSuccessors);

/// Constructs a new detached/owned [`BarWarpSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn bar_warp_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedBarWarpSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(BAR_WARP_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::bar_warp_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.alloc`.
pub const TCGEN05_ALLOC_OPERATION: &str = "nvvm.tcgen05.alloc";

/// Operation trait for `nvvm.tcgen05.alloc`.
pub trait Tcgen05AllocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_ALLOC_OPERATION
    }
}

mlir_op!(Tcgen05Alloc);
mlir_op_trait!(Tcgen05Alloc, ZeroRegions);
mlir_op_trait!(Tcgen05Alloc, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05AllocOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_alloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05AllocOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_ALLOC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_alloc`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.commit`.
pub const TCGEN05_COMMIT_OPERATION: &str = "nvvm.tcgen05.commit";

/// Operation trait for `nvvm.tcgen05.commit`.
pub trait Tcgen05CommitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_COMMIT_OPERATION
    }
}

mlir_op!(Tcgen05Commit);
mlir_op_trait!(Tcgen05Commit, ZeroRegions);
mlir_op_trait!(Tcgen05Commit, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05CommitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_commit<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05CommitOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_COMMIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_commit`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.cp`.
pub const TCGEN05_CP_OPERATION: &str = "nvvm.tcgen05.cp";

/// Operation trait for `nvvm.tcgen05.cp`.
pub trait Tcgen05CpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_CP_OPERATION
    }
}

mlir_op!(Tcgen05Cp);
mlir_op_trait!(Tcgen05Cp, ZeroRegions);
mlir_op_trait!(Tcgen05Cp, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05CpOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_cp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05CpOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_CP_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_cp`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.dealloc`.
pub const TCGEN05_DEALLOC_OPERATION: &str = "nvvm.tcgen05.dealloc";

/// Operation trait for `nvvm.tcgen05.dealloc`.
pub trait Tcgen05DeallocOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_DEALLOC_OPERATION
    }
}

mlir_op!(Tcgen05Dealloc);
mlir_op_trait!(Tcgen05Dealloc, ZeroRegions);
mlir_op_trait!(Tcgen05Dealloc, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05DeallocOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_dealloc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05DeallocOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_DEALLOC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_dealloc`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.fence`.
pub const TCGEN05_FENCE_OPERATION: &str = "nvvm.tcgen05.fence";

/// Operation trait for `nvvm.tcgen05.fence`.
pub trait Tcgen05FenceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_FENCE_OPERATION
    }
}

mlir_op!(Tcgen05Fence);
mlir_op_trait!(Tcgen05Fence, ZeroRegions);
mlir_op_trait!(Tcgen05Fence, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05FenceOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_fence<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05FenceOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_FENCE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_fence`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.ld`.
pub const TCGEN05_LD_OPERATION: &str = "nvvm.tcgen05.ld";

/// Operation trait for `nvvm.tcgen05.ld`.
pub trait Tcgen05LdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_LD_OPERATION
    }
}

mlir_op!(Tcgen05Ld);
mlir_op_trait!(Tcgen05Ld, ZeroRegions);
mlir_op_trait!(Tcgen05Ld, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05LdOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_ld<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05LdOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_LD_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_ld`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.ld.red`.
pub const TCGEN05_LD_RED_OPERATION: &str = "nvvm.tcgen05.ld.red";

/// Operation trait for `nvvm.tcgen05.ld.red`.
pub trait Tcgen05LdRedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_LD_RED_OPERATION
    }
}

mlir_op!(Tcgen05LdRed);
mlir_op_trait!(Tcgen05LdRed, ZeroRegions);
mlir_op_trait!(Tcgen05LdRed, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05LdRedOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_ld_red<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05LdRedOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_LD_RED_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_ld_red`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma.block_scale`.
pub const TCGEN05_MMA_BLOCK_SCALE_OPERATION: &str = "nvvm.tcgen05.mma.block_scale";

/// Operation trait for `nvvm.tcgen05.mma.block_scale`.
pub trait Tcgen05MmaBlockScaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_BLOCK_SCALE_OPERATION
    }
}

mlir_op!(Tcgen05MmaBlockScale);
mlir_op_trait!(Tcgen05MmaBlockScale, ZeroRegions);
mlir_op_trait!(Tcgen05MmaBlockScale, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaBlockScaleOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_block_scale<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaBlockScaleOperation<'c, 't>, Error> {
    build_nvvm_operation(
        TCGEN05_MMA_BLOCK_SCALE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_block_scale`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma`.
pub const TCGEN05_MMA_OPERATION: &str = "nvvm.tcgen05.mma";

/// Operation trait for `nvvm.tcgen05.mma`.
pub trait Tcgen05MmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_OPERATION
    }
}

mlir_op!(Tcgen05Mma);
mlir_op_trait!(Tcgen05Mma, ZeroRegions);
mlir_op_trait!(Tcgen05Mma, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_MMA_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma.sp.block_scale`.
pub const TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION: &str = "nvvm.tcgen05.mma.sp.block_scale";

/// Operation trait for `nvvm.tcgen05.mma.sp.block_scale`.
pub trait Tcgen05MmaSpBlockScaleOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION
    }
}

mlir_op!(Tcgen05MmaSpBlockScale);
mlir_op_trait!(Tcgen05MmaSpBlockScale, ZeroRegions);
mlir_op_trait!(Tcgen05MmaSpBlockScale, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaSpBlockScaleOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_sp_block_scale<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaSpBlockScaleOperation<'c, 't>, Error> {
    build_nvvm_operation(
        TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_sp_block_scale`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma.sp`.
pub const TCGEN05_MMA_SP_OPERATION: &str = "nvvm.tcgen05.mma.sp";

/// Operation trait for `nvvm.tcgen05.mma.sp`.
pub trait Tcgen05MmaSpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_SP_OPERATION
    }
}

mlir_op!(Tcgen05MmaSp);
mlir_op_trait!(Tcgen05MmaSp, ZeroRegions);
mlir_op_trait!(Tcgen05MmaSp, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaSpOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_sp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaSpOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_MMA_SP_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_sp`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma.ws`.
pub const TCGEN05_MMA_WS_OPERATION: &str = "nvvm.tcgen05.mma.ws";

/// Operation trait for `nvvm.tcgen05.mma.ws`.
pub trait Tcgen05MmaWsOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_WS_OPERATION
    }
}

mlir_op!(Tcgen05MmaWs);
mlir_op_trait!(Tcgen05MmaWs, ZeroRegions);
mlir_op_trait!(Tcgen05MmaWs, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaWsOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_ws<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaWsOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_MMA_WS_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_ws`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma.ws.sp`.
pub const TCGEN05_MMA_WS_SP_OPERATION: &str = "nvvm.tcgen05.mma.ws.sp";

/// Operation trait for `nvvm.tcgen05.mma.ws.sp`.
pub trait Tcgen05MmaWsSpOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_WS_SP_OPERATION
    }
}

mlir_op!(Tcgen05MmaWsSp);
mlir_op_trait!(Tcgen05MmaWsSp, ZeroRegions);
mlir_op_trait!(Tcgen05MmaWsSp, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaWsSpOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_ws_sp<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaWsSpOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_MMA_WS_SP_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_ws_sp`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.mma_smem_desc`.
pub const TCGEN05_MMA_SMEM_DESC_OPERATION: &str = "nvvm.tcgen05.mma_smem_desc";

/// Operation trait for `nvvm.tcgen05.mma_smem_desc`.
pub trait Tcgen05MmaSmemDescOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_MMA_SMEM_DESC_OPERATION
    }
}

mlir_op!(Tcgen05MmaSmemDesc);
mlir_op_trait!(Tcgen05MmaSmemDesc, ZeroRegions);
mlir_op_trait!(Tcgen05MmaSmemDesc, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05MmaSmemDescOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_mma_smem_desc<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05MmaSmemDescOperation<'c, 't>, Error> {
    build_nvvm_operation(
        TCGEN05_MMA_SMEM_DESC_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_mma_smem_desc`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.relinquish_alloc_permit`.
pub const TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION: &str = "nvvm.tcgen05.relinquish_alloc_permit";

/// Operation trait for `nvvm.tcgen05.relinquish_alloc_permit`.
pub trait Tcgen05RelinquishAllocPermitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION
    }
}

mlir_op!(Tcgen05RelinquishAllocPermit);
mlir_op_trait!(Tcgen05RelinquishAllocPermit, ZeroRegions);
mlir_op_trait!(Tcgen05RelinquishAllocPermit, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05RelinquishAllocPermitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_relinquish_alloc_permit<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05RelinquishAllocPermitOperation<'c, 't>, Error> {
    build_nvvm_operation(
        TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_relinquish_alloc_permit`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.shift`.
pub const TCGEN05_SHIFT_OPERATION: &str = "nvvm.tcgen05.shift";

/// Operation trait for `nvvm.tcgen05.shift`.
pub trait Tcgen05ShiftOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_SHIFT_OPERATION
    }
}

mlir_op!(Tcgen05Shift);
mlir_op_trait!(Tcgen05Shift, ZeroRegions);
mlir_op_trait!(Tcgen05Shift, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05ShiftOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_shift<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05ShiftOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_SHIFT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_shift`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.st`.
pub const TCGEN05_ST_OPERATION: &str = "nvvm.tcgen05.st";

/// Operation trait for `nvvm.tcgen05.st`.
pub trait Tcgen05StOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_ST_OPERATION
    }
}

mlir_op!(Tcgen05St);
mlir_op_trait!(Tcgen05St, ZeroRegions);
mlir_op_trait!(Tcgen05St, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05StOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_st<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05StOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_ST_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_st`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tcgen05.wait`.
pub const TCGEN05_WAIT_OPERATION: &str = "nvvm.tcgen05.wait";

/// Operation trait for `nvvm.tcgen05.wait`.
pub trait Tcgen05WaitOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TCGEN05_WAIT_OPERATION
    }
}

mlir_op!(Tcgen05Wait);
mlir_op_trait!(Tcgen05Wait, ZeroRegions);
mlir_op_trait!(Tcgen05Wait, ZeroSuccessors);

/// Constructs a new detached/owned [`Tcgen05WaitOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tcgen05_wait<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTcgen05WaitOperation<'c, 't>, Error> {
    build_nvvm_operation(TCGEN05_WAIT_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tcgen05_wait`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.tensormap.replace`.
pub const TENSORMAP_REPLACE_OPERATION: &str = "nvvm.tensormap.replace";

/// Operation trait for `nvvm.tensormap.replace`.
pub trait TensormapReplaceOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        TENSORMAP_REPLACE_OPERATION
    }
}

mlir_op!(TensormapReplace);
mlir_op_trait!(TensormapReplace, ZeroRegions);
mlir_op_trait!(TensormapReplace, ZeroSuccessors);

/// Constructs a new detached/owned [`TensormapReplaceOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn tensormap_replace<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedTensormapReplaceOperation<'c, 't>, Error> {
    build_nvvm_operation(TENSORMAP_REPLACE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::tensormap_replace`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.tid.x`.
pub const READ_PTX_SREG_TID_X_OPERATION: &str = "nvvm.read.ptx.sreg.tid.x";

/// Operation trait for `nvvm.read.ptx.sreg.tid.x`.
pub trait ReadPtxSregTidXOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_TID_X_OPERATION
    }
}

mlir_op!(ReadPtxSregTidX);
mlir_op_trait!(ReadPtxSregTidX, ZeroRegions);
mlir_op_trait!(ReadPtxSregTidX, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregTidXOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_tid_x<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregTidXOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_TID_X_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_tid_x`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.tid.y`.
pub const READ_PTX_SREG_TID_Y_OPERATION: &str = "nvvm.read.ptx.sreg.tid.y";

/// Operation trait for `nvvm.read.ptx.sreg.tid.y`.
pub trait ReadPtxSregTidYOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_TID_Y_OPERATION
    }
}

mlir_op!(ReadPtxSregTidY);
mlir_op_trait!(ReadPtxSregTidY, ZeroRegions);
mlir_op_trait!(ReadPtxSregTidY, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregTidYOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_tid_y<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregTidYOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_TID_Y_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_tid_y`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.tid.z`.
pub const READ_PTX_SREG_TID_Z_OPERATION: &str = "nvvm.read.ptx.sreg.tid.z";

/// Operation trait for `nvvm.read.ptx.sreg.tid.z`.
pub trait ReadPtxSregTidZOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_TID_Z_OPERATION
    }
}

mlir_op!(ReadPtxSregTidZ);
mlir_op_trait!(ReadPtxSregTidZ, ZeroRegions);
mlir_op_trait!(ReadPtxSregTidZ, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregTidZOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_tid_z<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregTidZOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_TID_Z_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_tid_z`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.total.smem.size`.
pub const READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION: &str = "nvvm.read.ptx.sreg.total.smem.size";

/// Operation trait for `nvvm.read.ptx.sreg.total.smem.size`.
pub trait ReadPtxSregTotalSmemSizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION
    }
}

mlir_op!(ReadPtxSregTotalSmemSize);
mlir_op_trait!(ReadPtxSregTotalSmemSize, ZeroRegions);
mlir_op_trait!(ReadPtxSregTotalSmemSize, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregTotalSmemSizeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_total_smem_size<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregTotalSmemSizeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_total_smem_size`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.vote.sync`.
pub const VOTE_SYNC_OPERATION: &str = "nvvm.vote.sync";

/// Operation trait for `nvvm.vote.sync`.
pub trait VoteSyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        VOTE_SYNC_OPERATION
    }
}

mlir_op!(VoteSync);
mlir_op_trait!(VoteSync, ZeroRegions);
mlir_op_trait!(VoteSync, ZeroSuccessors);

/// Constructs a new detached/owned [`VoteSyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn vote_sync<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedVoteSyncOperation<'c, 't>, Error> {
    build_nvvm_operation(VOTE_SYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::vote_sync`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.wmma.load`.
pub const WMMA_LOAD_OPERATION: &str = "nvvm.wmma.load";

/// Operation trait for `nvvm.wmma.load`.
pub trait WmmaLoadOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WMMA_LOAD_OPERATION
    }
}

mlir_op!(WmmaLoad);
mlir_op_trait!(WmmaLoad, ZeroRegions);
mlir_op_trait!(WmmaLoad, ZeroSuccessors);

/// Constructs a new detached/owned [`WmmaLoadOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wmma_load<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWmmaLoadOperation<'c, 't>, Error> {
    build_nvvm_operation(WMMA_LOAD_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wmma_load`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.wmma.mma`.
pub const WMMA_MMA_OPERATION: &str = "nvvm.wmma.mma";

/// Operation trait for `nvvm.wmma.mma`.
pub trait WmmaMmaOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WMMA_MMA_OPERATION
    }
}

mlir_op!(WmmaMma);
mlir_op_trait!(WmmaMma, ZeroRegions);
mlir_op_trait!(WmmaMma, ZeroSuccessors);

/// Constructs a new detached/owned [`WmmaMmaOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wmma_mma<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWmmaMmaOperation<'c, 't>, Error> {
    build_nvvm_operation(WMMA_MMA_OPERATION, operands, result_types, attributes, infer_result_types, location).and_then(
        |operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wmma_mma`"))
        },
    )
}

/// Fully-qualified MLIR operation name for `nvvm.wmma.store`.
pub const WMMA_STORE_OPERATION: &str = "nvvm.wmma.store";

/// Operation trait for `nvvm.wmma.store`.
pub trait WmmaStoreOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WMMA_STORE_OPERATION
    }
}

mlir_op!(WmmaStore);
mlir_op_trait!(WmmaStore, ZeroRegions);
mlir_op_trait!(WmmaStore, ZeroSuccessors);

/// Constructs a new detached/owned [`WmmaStoreOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wmma_store<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWmmaStoreOperation<'c, 't>, Error> {
    build_nvvm_operation(WMMA_STORE_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wmma_store`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.nwarpid`.
pub const READ_PTX_SREG_NWARPID_OPERATION: &str = "nvvm.read.ptx.sreg.nwarpid";

/// Operation trait for `nvvm.read.ptx.sreg.nwarpid`.
pub trait ReadPtxSregNwarpidOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_NWARPID_OPERATION
    }
}

mlir_op!(ReadPtxSregNwarpid);
mlir_op_trait!(ReadPtxSregNwarpid, ZeroRegions);
mlir_op_trait!(ReadPtxSregNwarpid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregNwarpidOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_nwarpid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregNwarpidOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_NWARPID_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_nwarpid`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.warpid`.
pub const READ_PTX_SREG_WARPID_OPERATION: &str = "nvvm.read.ptx.sreg.warpid";

/// Operation trait for `nvvm.read.ptx.sreg.warpid`.
pub trait ReadPtxSregWarpidOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_WARPID_OPERATION
    }
}

mlir_op!(ReadPtxSregWarpid);
mlir_op_trait!(ReadPtxSregWarpid, ZeroRegions);
mlir_op_trait!(ReadPtxSregWarpid, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregWarpidOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_warpid<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregWarpidOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_WARPID_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_warpid`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.read.ptx.sreg.warpsize`.
pub const READ_PTX_SREG_WARPSIZE_OPERATION: &str = "nvvm.read.ptx.sreg.warpsize";

/// Operation trait for `nvvm.read.ptx.sreg.warpsize`.
pub trait ReadPtxSregWarpsizeOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        READ_PTX_SREG_WARPSIZE_OPERATION
    }
}

mlir_op!(ReadPtxSregWarpsize);
mlir_op_trait!(ReadPtxSregWarpsize, ZeroRegions);
mlir_op_trait!(ReadPtxSregWarpsize, ZeroSuccessors);

/// Constructs a new detached/owned [`ReadPtxSregWarpsizeOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn read_ptx_sreg_warpsize<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedReadPtxSregWarpsizeOperation<'c, 't>, Error> {
    build_nvvm_operation(
        READ_PTX_SREG_WARPSIZE_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::read_ptx_sreg_warpsize`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.wgmma.fence.aligned`.
pub const WGMMA_FENCE_ALIGNED_OPERATION: &str = "nvvm.wgmma.fence.aligned";

/// Operation trait for `nvvm.wgmma.fence.aligned`.
pub trait WgmmaFenceAlignedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WGMMA_FENCE_ALIGNED_OPERATION
    }
}

mlir_op!(WgmmaFenceAligned);
mlir_op_trait!(WgmmaFenceAligned, ZeroRegions);
mlir_op_trait!(WgmmaFenceAligned, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaFenceAlignedOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wgmma_fence_aligned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWgmmaFenceAlignedOperation<'c, 't>, Error> {
    build_nvvm_operation(
        WGMMA_FENCE_ALIGNED_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wgmma_fence_aligned`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.wgmma.commit.group.sync.aligned`.
pub const WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION: &str = "nvvm.wgmma.commit.group.sync.aligned";

/// Operation trait for `nvvm.wgmma.commit.group.sync.aligned`.
pub trait WgmmaCommitGroupSyncAlignedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION
    }
}

mlir_op!(WgmmaCommitGroupSyncAligned);
mlir_op_trait!(WgmmaCommitGroupSyncAligned, ZeroRegions);
mlir_op_trait!(WgmmaCommitGroupSyncAligned, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaCommitGroupSyncAlignedOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wgmma_commit_group_sync_aligned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWgmmaCommitGroupSyncAlignedOperation<'c, 't>, Error> {
    build_nvvm_operation(
        WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wgmma_commit_group_sync_aligned`"))
    })
}

/// Fully-qualified MLIR operation name for `nvvm.wgmma.mma_async`.
pub const WGMMA_MMA_ASYNC_OPERATION: &str = "nvvm.wgmma.mma_async";

/// Operation trait for `nvvm.wgmma.mma_async`.
pub trait WgmmaMmaAsyncOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WGMMA_MMA_ASYNC_OPERATION
    }
}

mlir_op!(WgmmaMmaAsync);
mlir_op_trait!(WgmmaMmaAsync, ZeroRegions);
mlir_op_trait!(WgmmaMmaAsync, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaMmaAsyncOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wgmma_mma_async<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWgmmaMmaAsyncOperation<'c, 't>, Error> {
    build_nvvm_operation(WGMMA_MMA_ASYNC_OPERATION, operands, result_types, attributes, infer_result_types, location)
        .and_then(|operation| unsafe {
            operation
                .cast()
                .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wgmma_mma_async`"))
        })
}

/// Fully-qualified MLIR operation name for `nvvm.wgmma.wait.group.sync.aligned`.
pub const WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION: &str = "nvvm.wgmma.wait.group.sync.aligned";

/// Operation trait for `nvvm.wgmma.wait.group.sync.aligned`.
pub trait WgmmaWaitGroupSyncAlignedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the fully-qualified MLIR operation name.
    fn operation_name(&self) -> &'static str {
        WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION
    }
}

mlir_op!(WgmmaWaitGroupSyncAligned);
mlir_op_trait!(WgmmaWaitGroupSyncAligned, ZeroRegions);
mlir_op_trait!(WgmmaWaitGroupSyncAligned, ZeroSuccessors);

/// Constructs a new detached/owned [`WgmmaWaitGroupSyncAlignedOperation`] at the specified [`Location`].
///
/// # Parameters
///
///   - `operands`: Operation operands in MLIR operand order.
///   - `result_types`: Operation result types in MLIR result order.
///   - `attributes`: Operation attributes keyed by their MLIR names.
///   - `infer_result_types`: Whether to ask MLIR to infer result types during construction.
///   - `location`: Source location attached to the operation.
pub fn wgmma_wait_group_sync_aligned<'v, 'c: 'v, 't: 'c, L: Location<'c, 't>>(
    operands: &[ValueRef<'v, 'c, 't>],
    result_types: &[TypeRef<'c, 't>],
    attributes: &[(&str, AttributeRef<'c, 't>)],
    infer_result_types: bool,
    location: L,
) -> Result<DetachedWgmmaWaitGroupSyncAlignedOperation<'c, 't>, Error> {
    build_nvvm_operation(
        WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION,
        operands,
        result_types,
        attributes,
        infer_result_types,
        location,
    )
    .and_then(|operation| unsafe {
        operation
            .cast()
            .ok_or_else(|| Error::invalid_argument("invalid arguments to `nvvm::wgmma_wait_group_sync_aligned`"))
    })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Context;

    use super::*;

    #[test]
    fn test_addf_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = addf(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(ADDF_OPERATION));
        assert_eq!(operation.name(), context.identifier(ADDF_OPERATION));
        assert_eq!(operation.operation_name(), ADDF_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_aggr_smem_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_aggr_smem_size(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_AGGR_SMEM_SIZE_OPERATION);
    }

    #[test]
    fn test_barrier_arrive_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = barrier_arrive(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(BARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.name(), context.identifier(BARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.operation_name(), BARRIER_ARRIVE_OPERATION);
    }

    #[test]
    fn test_barrier_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = barrier(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(BARRIER_OPERATION));
        assert_eq!(operation.name(), context.identifier(BARRIER_OPERATION));
        assert_eq!(operation.operation_name(), BARRIER_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ntid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ntid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NTID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NTID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NTID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ntid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ntid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NTID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NTID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NTID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ntid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ntid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NTID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NTID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NTID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ctaid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ctaid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CTAID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CTAID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CTAID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ctaid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ctaid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CTAID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CTAID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CTAID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_ctaid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_ctaid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CTAID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CTAID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CTAID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_ctaid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_ctaid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_CTAID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_ctaid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_ctaid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_CTAID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_ctaid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_ctaid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_CTAID_Z_OPERATION);
    }

    #[test]
    fn test_breakpoint_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = breakpoint(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(BREAKPOINT_OPERATION));
        assert_eq!(operation.name(), context.identifier(BREAKPOINT_OPERATION));
        assert_eq!(operation.operation_name(), BREAKPOINT_OPERATION);
    }

    #[test]
    fn test_st_bulk_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = st_bulk(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(ST_BULK_OPERATION));
        assert_eq!(operation.name(), context.identifier(ST_BULK_OPERATION));
        assert_eq!(operation.operation_name(), ST_BULK_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_clock64_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_clock64(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLOCK64_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLOCK64_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLOCK64_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_clock_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_clock(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLOCK_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLOCK_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLOCK_OPERATION);
    }

    #[test]
    fn test_cluster_arrive_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cluster_arrive(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CLUSTER_ARRIVE_OPERATION));
        assert_eq!(operation.name(), context.identifier(CLUSTER_ARRIVE_OPERATION));
        assert_eq!(operation.operation_name(), CLUSTER_ARRIVE_OPERATION);
    }

    #[test]
    fn test_cluster_arrive_relaxed_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cluster_arrive_relaxed(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CLUSTER_ARRIVE_RELAXED_OPERATION));
        assert_eq!(operation.name(), context.identifier(CLUSTER_ARRIVE_RELAXED_OPERATION));
        assert_eq!(operation.operation_name(), CLUSTER_ARRIVE_RELAXED_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_nctarank_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_nctarank(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_NCTARANK_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_nctaid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_nctaid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_NCTAID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_nctaid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_nctaid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_NCTAID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_nctaid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_nctaid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_NCTAID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nclusterid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nclusterid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCLUSTERID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCLUSTERID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCLUSTERID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nclusterid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nclusterid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCLUSTERID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCLUSTERID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCLUSTERID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nclusterid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nclusterid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCLUSTERID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCLUSTERID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCLUSTERID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_cluster_ctarank_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_cluster_ctarank(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTER_CTARANK_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTER_CTARANK_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTER_CTARANK_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_clusterid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_clusterid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTERID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTERID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTERID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_clusterid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_clusterid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTERID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTERID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTERID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_clusterid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_clusterid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_CLUSTERID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_CLUSTERID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_CLUSTERID_Z_OPERATION);
    }

    #[test]
    fn test_clusterlaunchcontrol_query_cancel_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = clusterlaunchcontrol_query_cancel(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION));
        assert_eq!(operation.name(), context.identifier(CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION));
        assert_eq!(operation.operation_name(), CLUSTERLAUNCHCONTROL_QUERY_CANCEL_OPERATION);
    }

    #[test]
    fn test_clusterlaunchcontrol_try_cancel_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = clusterlaunchcontrol_try_cancel(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION));
        assert_eq!(operation.name(), context.identifier(CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION));
        assert_eq!(operation.operation_name(), CLUSTERLAUNCHCONTROL_TRY_CANCEL_OPERATION);
    }

    #[test]
    fn test_cluster_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cluster_wait(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CLUSTER_WAIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(CLUSTER_WAIT_OPERATION));
        assert_eq!(operation.operation_name(), CLUSTER_WAIT_OPERATION);
    }

    #[test]
    fn test_convert_bf16x2_to_f4x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_bf16x2_to_f4x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_BF16X2_TO_F4X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_BF16X2_TO_F4X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_BF16X2_TO_F4X2_OPERATION);
    }

    #[test]
    fn test_convert_bf16x2_to_f6x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_bf16x2_to_f6x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_BF16X2_TO_F6X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_BF16X2_TO_F6X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_BF16X2_TO_F6X2_OPERATION);
    }

    #[test]
    fn test_convert_bf16x2_to_f8x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_bf16x2_to_f8x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_BF16X2_TO_F8X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_BF16X2_TO_F8X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_BF16X2_TO_F8X2_OPERATION);
    }

    #[test]
    fn test_convert_bf16x2_to_s2f6x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_bf16x2_to_s2f6x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_BF16X2_TO_S2F6X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_BF16X2_TO_S2F6X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_BF16X2_TO_S2F6X2_OPERATION);
    }

    #[test]
    fn test_convert_f4x2_to_f16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f4x2_to_f16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F4X2_TO_F16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F4X2_TO_F16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F4X2_TO_F16X2_OPERATION);
    }

    #[test]
    fn test_convert_f6x2_to_f16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f6x2_to_f16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F6X2_TO_F16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F6X2_TO_F16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F6X2_TO_F16X2_OPERATION);
    }

    #[test]
    fn test_convert_f8x2_to_bf16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f8x2_to_bf16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F8X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F8X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F8X2_TO_BF16X2_OPERATION);
    }

    #[test]
    fn test_convert_f8x2_to_f16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f8x2_to_f16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F8X2_TO_F16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F8X2_TO_F16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F8X2_TO_F16X2_OPERATION);
    }

    #[test]
    fn test_convert_f16x2_to_f4x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f16x2_to_f4x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F16X2_TO_F4X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F16X2_TO_F4X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F16X2_TO_F4X2_OPERATION);
    }

    #[test]
    fn test_convert_f16x2_to_f6x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f16x2_to_f6x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F16X2_TO_F6X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F16X2_TO_F6X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F16X2_TO_F6X2_OPERATION);
    }

    #[test]
    fn test_convert_f16x2_to_f8x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f16x2_to_f8x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F16X2_TO_F8X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F16X2_TO_F8X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F16X2_TO_F8X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_bf16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_bf16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_BF16X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_f4x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_f4x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_F4X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_F4X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_F4X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_f6x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_f6x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_F6X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_F6X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_F6X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_f8x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_f8x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_F8X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_F8X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_F8X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_f16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_f16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_F16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_F16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_F16X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x2_to_s2f6x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x2_to_s2f6x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X2_TO_S2F6X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X2_TO_S2F6X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X2_TO_S2F6X2_OPERATION);
    }

    #[test]
    fn test_convert_f32x4_to_f4x4_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x4_to_f4x4(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X4_TO_F4X4_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X4_TO_F4X4_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X4_TO_F4X4_OPERATION);
    }

    #[test]
    fn test_convert_f32x4_to_f6x4_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x4_to_f6x4(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X4_TO_F6X4_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X4_TO_F6X4_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X4_TO_F6X4_OPERATION);
    }

    #[test]
    fn test_convert_f32x4_to_f8x4_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_f32x4_to_f8x4(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_F32X4_TO_F8X4_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_F32X4_TO_F8X4_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_F32X4_TO_F8X4_OPERATION);
    }

    #[test]
    fn test_convert_float_to_tf32_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_float_to_tf32(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_FLOAT_TO_TF32_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_FLOAT_TO_TF32_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_FLOAT_TO_TF32_OPERATION);
    }

    #[test]
    fn test_convert_s2f6x2_to_bf16x2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = convert_s2f6x2_to_bf16x2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CONVERT_S2F6X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.name(), context.identifier(CONVERT_S2F6X2_TO_BF16X2_OPERATION));
        assert_eq!(operation.operation_name(), CONVERT_S2F6X2_TO_BF16X2_OPERATION);
    }

    #[test]
    fn test_cos_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cos(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(COS_OPERATION));
        assert_eq!(operation.name(), context.identifier(COS_OPERATION));
        assert_eq!(operation.operation_name(), COS_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_commit_group_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_commit_group(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_COMMIT_GROUP_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_COMMIT_GROUP_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_COMMIT_GROUP_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_shared_cluster_global_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_shared_cluster_global(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_SHARED_CLUSTER_GLOBAL_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_prefetch_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_prefetch(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_PREFETCH_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_PREFETCH_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_PREFETCH_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_global_shared_cta_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_global_shared_cta(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_GLOBAL_SHARED_CTA_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_shared_cluster_shared_cta_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_shared_cluster_shared_cta(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_SHARED_CLUSTER_SHARED_CTA_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_tensor_shared_cluster_global_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_tensor_shared_cluster_global(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_TENSOR_SHARED_CLUSTER_GLOBAL_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_tensor_prefetch_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_tensor_prefetch(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_TENSOR_PREFETCH_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_tensor_reduce_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_tensor_reduce(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_TENSOR_REDUCE_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_tensor_global_shared_cta_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_tensor_global_shared_cta(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_TENSOR_GLOBAL_SHARED_CTA_OPERATION);
    }

    #[test]
    fn test_cp_async_bulk_wait_group_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_bulk_wait_group(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_BULK_WAIT_GROUP_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_BULK_WAIT_GROUP_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_BULK_WAIT_GROUP_OPERATION);
    }

    #[test]
    fn test_cp_async_commit_group_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_commit_group(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_COMMIT_GROUP_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_COMMIT_GROUP_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_COMMIT_GROUP_OPERATION);
    }

    #[test]
    fn test_cp_async_mbarrier_arrive_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_mbarrier_arrive(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_MBARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_MBARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_MBARRIER_ARRIVE_OPERATION);
    }

    #[test]
    fn test_cp_async_shared_global_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_shared_global(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_SHARED_GLOBAL_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_SHARED_GLOBAL_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_SHARED_GLOBAL_OPERATION);
    }

    #[test]
    fn test_cp_async_wait_group_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = cp_async_wait_group(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(CP_ASYNC_WAIT_GROUP_OPERATION));
        assert_eq!(operation.name(), context.identifier(CP_ASYNC_WAIT_GROUP_OPERATION));
        assert_eq!(operation.operation_name(), CP_ASYNC_WAIT_GROUP_OPERATION);
    }

    #[test]
    fn test_divf_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = divf(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(DIVF_OPERATION));
        assert_eq!(operation.name(), context.identifier(DIVF_OPERATION));
        assert_eq!(operation.operation_name(), DIVF_OPERATION);
    }

    #[test]
    fn test_dot_accumulate_2way_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = dot_accumulate_2way(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(DOT_ACCUMULATE_2WAY_OPERATION));
        assert_eq!(operation.name(), context.identifier(DOT_ACCUMULATE_2WAY_OPERATION));
        assert_eq!(operation.operation_name(), DOT_ACCUMULATE_2WAY_OPERATION);
    }

    #[test]
    fn test_dot_accumulate_4way_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = dot_accumulate_4way(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(DOT_ACCUMULATE_4WAY_OPERATION));
        assert_eq!(operation.name(), context.identifier(DOT_ACCUMULATE_4WAY_OPERATION));
        assert_eq!(operation.operation_name(), DOT_ACCUMULATE_4WAY_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_dynamic_smem_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_dynamic_smem_size(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_DYNAMIC_SMEM_SIZE_OPERATION);
    }

    #[test]
    fn test_elect_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = elect_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(ELECT_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(ELECT_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), ELECT_SYNC_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg0_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg0(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG0_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG0_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG0_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg1_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg1(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG1_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG1_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG1_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG2_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG2_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG2_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg3_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg3(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG3_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG3_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG3_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg4_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg4(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG4_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG4_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG4_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg5_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg5(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG5_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG5_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG5_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg6_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg6(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG6_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG6_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG6_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg7_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg7(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG7_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG7_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG7_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg8_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg8(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG8_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG8_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG8_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg9_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg9(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG9_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG9_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG9_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg10_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg10(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG10_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG10_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG10_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg11_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg11(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG11_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG11_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG11_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg12_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg12(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG12_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG12_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG12_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg13_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg13(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG13_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG13_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG13_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg14_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg14(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG14_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG14_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG14_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg15_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg15(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG15_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG15_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG15_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg16_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg16(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG16_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG16_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG16_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg17_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg17(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG17_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG17_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG17_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg18_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg18(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG18_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG18_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG18_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg19_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg19(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG19_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG19_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG19_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg20_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg20(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG20_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG20_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG20_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg21_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg21(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG21_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG21_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG21_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg22_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg22(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG22_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG22_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG22_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg23_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg23(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG23_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG23_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG23_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg24_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg24(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG24_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG24_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG24_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg25_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg25(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG25_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG25_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG25_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg26_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg26(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG26_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG26_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG26_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg27_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg27(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG27_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG27_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG27_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg28_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg28(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG28_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG28_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG28_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg29_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg29(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG29_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG29_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG29_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg30_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg30(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG30_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG30_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG30_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_envreg31_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_envreg31(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_ENVREG31_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_ENVREG31_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_ENVREG31_OPERATION);
    }

    #[test]
    fn test_ex2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = ex2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(EX2_OPERATION));
        assert_eq!(operation.name(), context.identifier(EX2_OPERATION));
        assert_eq!(operation.operation_name(), EX2_OPERATION);
    }

    #[test]
    fn test_exit_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = exit(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(EXIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(EXIT_OPERATION));
        assert_eq!(operation.operation_name(), EXIT_OPERATION);
    }

    #[test]
    fn test_fence_mbarrier_init_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_mbarrier_init(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_MBARRIER_INIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_MBARRIER_INIT_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_MBARRIER_INIT_OPERATION);
    }

    #[test]
    fn test_fence_proxy_acquire_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_proxy_acquire(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_PROXY_ACQUIRE_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_PROXY_ACQUIRE_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_PROXY_ACQUIRE_OPERATION);
    }

    #[test]
    fn test_fence_proxy_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_proxy(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_PROXY_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_PROXY_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_PROXY_OPERATION);
    }

    #[test]
    fn test_fence_proxy_release_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_proxy_release(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_PROXY_RELEASE_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_PROXY_RELEASE_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_PROXY_RELEASE_OPERATION);
    }

    #[test]
    fn test_fence_proxy_sync_restrict_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_proxy_sync_restrict(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_PROXY_SYNC_RESTRICT_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_PROXY_SYNC_RESTRICT_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_PROXY_SYNC_RESTRICT_OPERATION);
    }

    #[test]
    fn test_fence_sc_cluster_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_sc_cluster(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_SC_CLUSTER_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_SC_CLUSTER_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_SC_CLUSTER_OPERATION);
    }

    #[test]
    fn test_fence_sync_restrict_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fence_sync_restrict(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FENCE_SYNC_RESTRICT_OPERATION));
        assert_eq!(operation.name(), context.identifier(FENCE_SYNC_RESTRICT_OPERATION));
        assert_eq!(operation.operation_name(), FENCE_SYNC_RESTRICT_OPERATION);
    }

    #[test]
    fn test_fma_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = fma(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(FMA_OPERATION));
        assert_eq!(operation.name(), context.identifier(FMA_OPERATION));
        assert_eq!(operation.operation_name(), FMA_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_globaltimer_lo_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_globaltimer_lo(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_GLOBALTIMER_LO_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_GLOBALTIMER_LO_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_GLOBALTIMER_LO_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_globaltimer_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_globaltimer(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_GLOBALTIMER_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_GLOBALTIMER_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_GLOBALTIMER_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nctaid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nctaid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCTAID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCTAID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCTAID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nctaid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nctaid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCTAID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCTAID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCTAID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nctaid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nctaid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NCTAID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NCTAID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NCTAID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_gridid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_gridid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_GRIDID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_GRIDID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_GRIDID_OPERATION);
    }

    #[test]
    fn test_griddepcontrol_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = griddepcontrol(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(GRIDDEPCONTROL_OPERATION));
        assert_eq!(operation.name(), context.identifier(GRIDDEPCONTROL_OPERATION));
        assert_eq!(operation.operation_name(), GRIDDEPCONTROL_OPERATION);
    }

    #[test]
    fn test_inline_ptx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = inline_ptx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(INLINE_PTX_OPERATION));
        assert_eq!(operation.name(), context.identifier(INLINE_PTX_OPERATION));
        assert_eq!(operation.operation_name(), INLINE_PTX_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_laneid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_laneid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEID_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_lanemask_eq_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_lanemask_eq(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEMASK_EQ_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEMASK_EQ_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEMASK_EQ_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_lanemask_ge_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_lanemask_ge(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEMASK_GE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEMASK_GE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEMASK_GE_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_lanemask_gt_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_lanemask_gt(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEMASK_GT_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEMASK_GT_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEMASK_GT_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_lanemask_le_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_lanemask_le(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEMASK_LE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEMASK_LE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEMASK_LE_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_lanemask_lt_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_lanemask_lt(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_LANEMASK_LT_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_LANEMASK_LT_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_LANEMASK_LT_OPERATION);
    }

    #[test]
    fn test_ldmatrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = ldmatrix(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(LDMATRIX_OPERATION));
        assert_eq!(operation.name(), context.identifier(LDMATRIX_OPERATION));
        assert_eq!(operation.operation_name(), LDMATRIX_OPERATION);
    }

    #[test]
    fn test_log2_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = log2(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(LOG2_OPERATION));
        assert_eq!(operation.name(), context.identifier(LOG2_OPERATION));
        assert_eq!(operation.operation_name(), LOG2_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_drop_expect_tx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive_drop_expect_tx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_DROP_EXPECT_TX_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_drop_nocomplete_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive_drop_nocomplete(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_DROP_NOCOMPLETE_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_drop_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive_drop(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_DROP_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_DROP_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_DROP_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_expect_tx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive_expect_tx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_EXPECT_TX_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_EXPECT_TX_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_EXPECT_TX_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_nocomplete_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive_nocomplete(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_NOCOMPLETE_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_NOCOMPLETE_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_NOCOMPLETE_OPERATION);
    }

    #[test]
    fn test_mbarrier_arrive_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_arrive(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_ARRIVE_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_ARRIVE_OPERATION);
    }

    #[test]
    fn test_mbarrier_complete_tx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_complete_tx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_COMPLETE_TX_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_COMPLETE_TX_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_COMPLETE_TX_OPERATION);
    }

    #[test]
    fn test_mbarrier_expect_tx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_expect_tx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_EXPECT_TX_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_EXPECT_TX_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_EXPECT_TX_OPERATION);
    }

    #[test]
    fn test_mbarrier_init_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_init(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_INIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_INIT_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_INIT_OPERATION);
    }

    #[test]
    fn test_mbarrier_inval_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_inval(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_INVAL_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_INVAL_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_INVAL_OPERATION);
    }

    #[test]
    fn test_mbarrier_test_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_test_wait(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_TEST_WAIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_TEST_WAIT_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_TEST_WAIT_OPERATION);
    }

    #[test]
    fn test_mbarrier_try_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_try_wait(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_TRY_WAIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_TRY_WAIT_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_TRY_WAIT_OPERATION);
    }

    #[test]
    fn test_mbarrier_try_wait_parity_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mbarrier_try_wait_parity(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MBARRIER_TRY_WAIT_PARITY_OPERATION));
        assert_eq!(operation.name(), context.identifier(MBARRIER_TRY_WAIT_PARITY_OPERATION));
        assert_eq!(operation.operation_name(), MBARRIER_TRY_WAIT_PARITY_OPERATION);
    }

    #[test]
    fn test_mapa_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mapa(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MAPA_OPERATION));
        assert_eq!(operation.name(), context.identifier(MAPA_OPERATION));
        assert_eq!(operation.operation_name(), MAPA_OPERATION);
    }

    #[test]
    fn test_match_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = match_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MATCH_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(MATCH_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), MATCH_SYNC_OPERATION);
    }

    #[test]
    fn test_memory_barrier_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = memory_barrier(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MEMORY_BARRIER_OPERATION));
        assert_eq!(operation.name(), context.identifier(MEMORY_BARRIER_OPERATION));
        assert_eq!(operation.operation_name(), MEMORY_BARRIER_OPERATION);
    }

    #[test]
    fn test_mma_block_scale_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mma_block_scale(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MMA_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.name(), context.identifier(MMA_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.operation_name(), MMA_BLOCK_SCALE_OPERATION);
    }

    #[test]
    fn test_mma_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mma_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MMA_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(MMA_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), MMA_SYNC_OPERATION);
    }

    #[test]
    fn test_mma_sp_block_scale_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mma_sp_block_scale(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MMA_SP_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.name(), context.identifier(MMA_SP_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.operation_name(), MMA_SP_BLOCK_SCALE_OPERATION);
    }

    #[test]
    fn test_mma_sp_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = mma_sp_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MMA_SP_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(MMA_SP_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), MMA_SP_SYNC_OPERATION);
    }

    #[test]
    fn test_movmatrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = movmatrix(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(MOVMATRIX_OPERATION));
        assert_eq!(operation.name(), context.identifier(MOVMATRIX_OPERATION));
        assert_eq!(operation.operation_name(), MOVMATRIX_OPERATION);
    }

    #[test]
    fn test_nanosleep_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = nanosleep(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(NANOSLEEP_OPERATION));
        assert_eq!(operation.name(), context.identifier(NANOSLEEP_OPERATION));
        assert_eq!(operation.operation_name(), NANOSLEEP_OPERATION);
    }

    #[test]
    fn test_pmevent_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = pmevent(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(PMEVENT_OPERATION));
        assert_eq!(operation.name(), context.identifier(PMEVENT_OPERATION));
        assert_eq!(operation.operation_name(), PMEVENT_OPERATION);
    }

    #[test]
    fn test_prmt_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = prmt(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(PRMT_OPERATION));
        assert_eq!(operation.name(), context.identifier(PRMT_OPERATION));
        assert_eq!(operation.operation_name(), PRMT_OPERATION);
    }

    #[test]
    fn test_prefetch_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = prefetch(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(PREFETCH_OPERATION));
        assert_eq!(operation.name(), context.identifier(PREFETCH_OPERATION));
        assert_eq!(operation.operation_name(), PREFETCH_OPERATION);
    }

    #[test]
    fn test_rcp_approx_ftz_f_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = rcp_approx_ftz_f(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(RCP_APPROX_FTZ_F_OPERATION));
        assert_eq!(operation.name(), context.identifier(RCP_APPROX_FTZ_F_OPERATION));
        assert_eq!(operation.operation_name(), RCP_APPROX_FTZ_F_OPERATION);
    }

    #[test]
    fn test_redux_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = redux_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(REDUX_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(REDUX_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), REDUX_SYNC_OPERATION);
    }

    #[test]
    fn test_rsqrt_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = rsqrt(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(RSQRT_OPERATION));
        assert_eq!(operation.name(), context.identifier(RSQRT_OPERATION));
        assert_eq!(operation.operation_name(), RSQRT_OPERATION);
    }

    #[test]
    fn test_setmaxregister_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = setmaxregister(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SETMAXREGISTER_OPERATION));
        assert_eq!(operation.name(), context.identifier(SETMAXREGISTER_OPERATION));
        assert_eq!(operation.operation_name(), SETMAXREGISTER_OPERATION);
    }

    #[test]
    fn test_shfl_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = shfl_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SHFL_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(SHFL_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), SHFL_SYNC_OPERATION);
    }

    #[test]
    fn test_sin_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = sin(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SIN_OPERATION));
        assert_eq!(operation.name(), context.identifier(SIN_OPERATION));
        assert_eq!(operation.operation_name(), SIN_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nsmid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nsmid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NSMID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NSMID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NSMID_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_smid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_smid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_SMID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_SMID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_SMID_OPERATION);
    }

    #[test]
    fn test_sqrt_approx_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = sqrt_approx(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SQRT_APPROX_OPERATION));
        assert_eq!(operation.name(), context.identifier(SQRT_APPROX_OPERATION));
        assert_eq!(operation.operation_name(), SQRT_APPROX_OPERATION);
    }

    #[test]
    fn test_sqrt_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = sqrt(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SQRT_OPERATION));
        assert_eq!(operation.name(), context.identifier(SQRT_OPERATION));
        assert_eq!(operation.operation_name(), SQRT_OPERATION);
    }

    #[test]
    fn test_stmatrix_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = stmatrix(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(STMATRIX_OPERATION));
        assert_eq!(operation.name(), context.identifier(STMATRIX_OPERATION));
        assert_eq!(operation.operation_name(), STMATRIX_OPERATION);
    }

    #[test]
    fn test_subf_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = subf(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(SUBF_OPERATION));
        assert_eq!(operation.name(), context.identifier(SUBF_OPERATION));
        assert_eq!(operation.operation_name(), SUBF_OPERATION);
    }

    #[test]
    fn test_bar_warp_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = bar_warp_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(BAR_WARP_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(BAR_WARP_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), BAR_WARP_SYNC_OPERATION);
    }

    #[test]
    fn test_tcgen05_alloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_alloc(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_ALLOC_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_ALLOC_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_ALLOC_OPERATION);
    }

    #[test]
    fn test_tcgen05_commit_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_commit(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_COMMIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_COMMIT_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_COMMIT_OPERATION);
    }

    #[test]
    fn test_tcgen05_cp_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_cp(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_CP_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_CP_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_CP_OPERATION);
    }

    #[test]
    fn test_tcgen05_dealloc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_dealloc(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_DEALLOC_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_DEALLOC_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_DEALLOC_OPERATION);
    }

    #[test]
    fn test_tcgen05_fence_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_fence(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_FENCE_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_FENCE_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_FENCE_OPERATION);
    }

    #[test]
    fn test_tcgen05_ld_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_ld(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_LD_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_LD_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_LD_OPERATION);
    }

    #[test]
    fn test_tcgen05_ld_red_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_ld_red(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_LD_RED_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_LD_RED_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_LD_RED_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_block_scale_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_block_scale(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_BLOCK_SCALE_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_sp_block_scale_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_sp_block_scale(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_SP_BLOCK_SCALE_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_sp_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_sp(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_SP_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_SP_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_SP_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_ws_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_ws(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_WS_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_WS_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_WS_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_ws_sp_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_ws_sp(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_WS_SP_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_WS_SP_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_WS_SP_OPERATION);
    }

    #[test]
    fn test_tcgen05_mma_smem_desc_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_mma_smem_desc(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_MMA_SMEM_DESC_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_MMA_SMEM_DESC_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_MMA_SMEM_DESC_OPERATION);
    }

    #[test]
    fn test_tcgen05_relinquish_alloc_permit_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_relinquish_alloc_permit(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_RELINQUISH_ALLOC_PERMIT_OPERATION);
    }

    #[test]
    fn test_tcgen05_shift_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_shift(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_SHIFT_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_SHIFT_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_SHIFT_OPERATION);
    }

    #[test]
    fn test_tcgen05_st_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_st(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_ST_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_ST_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_ST_OPERATION);
    }

    #[test]
    fn test_tcgen05_wait_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tcgen05_wait(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TCGEN05_WAIT_OPERATION));
        assert_eq!(operation.name(), context.identifier(TCGEN05_WAIT_OPERATION));
        assert_eq!(operation.operation_name(), TCGEN05_WAIT_OPERATION);
    }

    #[test]
    fn test_tensormap_replace_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = tensormap_replace(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(TENSORMAP_REPLACE_OPERATION));
        assert_eq!(operation.name(), context.identifier(TENSORMAP_REPLACE_OPERATION));
        assert_eq!(operation.operation_name(), TENSORMAP_REPLACE_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_tid_x_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_tid_x(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_TID_X_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_TID_X_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_TID_X_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_tid_y_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_tid_y(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_TID_Y_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_TID_Y_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_TID_Y_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_tid_z_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_tid_z(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_TID_Z_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_TID_Z_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_TID_Z_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_total_smem_size_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_total_smem_size(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_TOTAL_SMEM_SIZE_OPERATION);
    }

    #[test]
    fn test_vote_sync_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = vote_sync(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(VOTE_SYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(VOTE_SYNC_OPERATION));
        assert_eq!(operation.operation_name(), VOTE_SYNC_OPERATION);
    }

    #[test]
    fn test_wmma_load_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wmma_load(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WMMA_LOAD_OPERATION));
        assert_eq!(operation.name(), context.identifier(WMMA_LOAD_OPERATION));
        assert_eq!(operation.operation_name(), WMMA_LOAD_OPERATION);
    }

    #[test]
    fn test_wmma_mma_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wmma_mma(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WMMA_MMA_OPERATION));
        assert_eq!(operation.name(), context.identifier(WMMA_MMA_OPERATION));
        assert_eq!(operation.operation_name(), WMMA_MMA_OPERATION);
    }

    #[test]
    fn test_wmma_store_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wmma_store(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WMMA_STORE_OPERATION));
        assert_eq!(operation.name(), context.identifier(WMMA_STORE_OPERATION));
        assert_eq!(operation.operation_name(), WMMA_STORE_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_nwarpid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_nwarpid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_NWARPID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_NWARPID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_NWARPID_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_warpid_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_warpid(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_WARPID_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_WARPID_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_WARPID_OPERATION);
    }

    #[test]
    fn test_read_ptx_sreg_warpsize_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = read_ptx_sreg_warpsize(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(READ_PTX_SREG_WARPSIZE_OPERATION));
        assert_eq!(operation.name(), context.identifier(READ_PTX_SREG_WARPSIZE_OPERATION));
        assert_eq!(operation.operation_name(), READ_PTX_SREG_WARPSIZE_OPERATION);
    }

    #[test]
    fn test_wgmma_fence_aligned_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wgmma_fence_aligned(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WGMMA_FENCE_ALIGNED_OPERATION));
        assert_eq!(operation.name(), context.identifier(WGMMA_FENCE_ALIGNED_OPERATION));
        assert_eq!(operation.operation_name(), WGMMA_FENCE_ALIGNED_OPERATION);
    }

    #[test]
    fn test_wgmma_commit_group_sync_aligned_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wgmma_commit_group_sync_aligned(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION));
        assert_eq!(operation.name(), context.identifier(WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION));
        assert_eq!(operation.operation_name(), WGMMA_COMMIT_GROUP_SYNC_ALIGNED_OPERATION);
    }

    #[test]
    fn test_wgmma_mma_async_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wgmma_mma_async(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WGMMA_MMA_ASYNC_OPERATION));
        assert_eq!(operation.name(), context.identifier(WGMMA_MMA_ASYNC_OPERATION));
        assert_eq!(operation.operation_name(), WGMMA_MMA_ASYNC_OPERATION);
    }

    #[test]
    fn test_wgmma_wait_group_sync_aligned_operation() {
        let context = Context::new();
        let location = context.unknown_location();
        let operation = wgmma_wait_group_sync_aligned(&[], &[], &[], false, location).unwrap();
        assert!(context.is_registered(WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION));
        assert_eq!(operation.name(), context.identifier(WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION));
        assert_eq!(operation.operation_name(), WGMMA_WAIT_GROUP_SYNC_ALIGNED_OPERATION);
    }
}
