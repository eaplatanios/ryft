use ryft_xla_sys::bindings::{
    MlirAttribute, stablehloAttributeIsChannelHandle, stablehloChannelHandleGet, stablehloChannelHandleGetHandle,
    stablehloChannelHandleGetType,
};

use crate::{
    Attribute, BooleanAttributeRef, Context, DenseIntegerElementsAttributeRef, DetachedOp, DetachedRegion,
    DialectHandle, IntegerAttributeRef, IntoWithContext, Location, OneRegion, Operation, OperationBuilder, RegionRef,
    Size, StringAttributeRef, StringRef, TensorTypeRef, Type, Value, mlir_op, mlir_op_trait, mlir_subtype_trait_impls,
};

/// Represents the type of a StableHLO communication channel.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(i64)]
pub enum ChannelHandleType {
    /// Unknown channel type used to represent types that may appear in the C API but which do not have
    /// a corresponding representation in the Rust API.
    Unknown = 0,

    /// Device-to-device channel type (e.g., for GPU 0 to GPU 1 communication).
    DeviceToDevice = 1,

    /// Device-to-host channel type (e.g., for GPU to CPU communication).
    DeviceToHost = 2,

    /// Host-to-device channel type (e.g., for CPU to GPU communication).
    HostToDevice = 3,
}

/// StableHLO [`Attribute`] that represents a unique identifier for each [`SendOperation`]/[`RecvOperation`]
/// pair or, optionally, for collective operations (e.g., [`AllToAllOperation`], [`AllReduceOperation`],
/// [`CollectiveBroadcastOperation`], and [`CollectivePermuteOperation`]).
#[derive(Copy, Clone)]
pub struct ChannelHandleAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ChannelHandleAttributeRef<'c, 't> {
    pub fn channel_id(&self) -> Option<usize> {
        let value = unsafe { stablehloChannelHandleGetHandle(self.handle) };
        if value < 0 { None } else { Some(value as usize) }
    }

    pub fn channel_type(&self) -> ChannelHandleType {
        let value = unsafe { stablehloChannelHandleGetType(self.handle) };
        match value {
            1 => ChannelHandleType::DeviceToDevice,
            2 => ChannelHandleType::DeviceToHost,
            _ => ChannelHandleType::Unknown,
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ChannelHandleAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if !handle.ptr.is_null() && unsafe { stablehloAttributeIsChannelHandle(handle) } {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(ChannelHandleAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new StableHLO [`ChannelHandleAttributeRef`] owned by this [`Context`].
    pub fn stable_hlo_channel_handle<'c>(
        &'c self,
        channel_id: Option<usize>,
        channel_type: ChannelHandleType,
    ) -> ChannelHandleAttributeRef<'c, 't> {
        // Make sure that the StableHLO dialect is loaded into the current context to prevent segmentation faults.
        self.load_dialect(DialectHandle::stable_hlo());
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            ChannelHandleAttributeRef::from_c_api(
                stablehloChannelHandleGet(
                    *self.handle.borrow(),
                    channel_id.map(|id| id as i64).unwrap_or(-1),
                    channel_type as i64,
                ),
                &self,
            )
            .unwrap()
        }
    }
}

/// StableHLO [`Operation`] that produces the _partition ID_ of the current process. That ID is an unsigned 32-bit
/// integer indicating which partition the current process is executing on in a multi-partition execution environment.
/// This is useful for sharding and distributed computation patterns where different partitions need to execute
/// different logic.
///
/// In the context of XLA and SPMD (Single Program, Multiple Data) parallelism, partition IDs are used in conjunction
/// with _replica IDs_ to create global device IDs using the following formula:
///
/// ```text
/// global_id = replica_id * partition_count + partition_id
/// ```
///
/// Refer to the [official XLA documentation](https://openxla.org/xla/operation_semantics) for more information.
///
/// # Example
///
/// The following is an example of a [`PartitionIdOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = stablehlo.partition_id : tensor<ui32>
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#partition_id)
/// for more information.
pub trait PartitionIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(PartitionId);
mlir_op_trait!(PartitionId, OneResult);
mlir_op_trait!(PartitionId, ZeroOperands);
mlir_op_trait!(PartitionId, ZeroRegions);
mlir_op_trait!(PartitionId, ZeroSuccessors);

/// Constructs a new detached/owned [`PartitionIdOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`PartitionIdOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn partition_id<'c, 't, L: Location<'c, 't>>(location: L) -> DetachedPartitionIdOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.partition_id", location)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::partition_id`")
}

/// StableHLO [`Operation`] that produces the _replica ID_ of the current process. That ID is an unsigned 32-bit
/// integer indicating which replica the current process is executing on in a multi-replica execution environment.
/// This is useful for data parallelism and distributed computation patterns where different replicas process
/// different data.
///
/// In the context of XLA and SPMD (Single Program, Multiple Data) parallelism, replica IDs are used in conjunction
/// with _partition IDs_ to create global device IDs using the following formula:
///
/// ```text
/// global_id = replica_id * partition_count + partition_id
/// ```
///
/// Refer to the [official XLA documentation](https://openxla.org/xla/operation_semantics) for more information.
///
/// # Example
///
/// The following is an example of a [`ReplicaIdOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = stablehlo.replica_id : tensor<ui32>
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#replica_id)
/// for more information.
pub trait ReplicaIdOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(ReplicaId);
mlir_op_trait!(ReplicaId, OneResult);
mlir_op_trait!(ReplicaId, ZeroOperands);
mlir_op_trait!(ReplicaId, ZeroRegions);
mlir_op_trait!(ReplicaId, ZeroSuccessors);

/// Constructs a new detached/owned [`ReplicaIdOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ReplicaIdOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn replica_id<'c, 't, L: Location<'c, 't>>(location: L) -> DetachedReplicaIdOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.replica_id", location)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::replica_id`")
}

/// StableHLO [`Operation`] that ensures execution ordering of other [`Operation`]s. Specifically, it ensures that the
/// operations producing its inputs/operands will be executed before any operations that depend on any one of their
/// results. The execution of this operation does nothing else. This operation only exists to establish data
/// dependencies from results to inputs.
///
/// Note that you can also create an instance of this operation with no inputs in order to create a fresh
/// [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value (e.g., to use in a [`send`] operation).
///
/// # Example
///
/// The following is an example of an [`AfterAllOperation`] represented using its [`Display`](std::fmt::Display)
/// rendering (in this case for an instance of this operation with two inputs):
///
/// ```mlir
///  %0 = stablehlo.after_all %arg0, %arg1 : !stablehlo.token
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#after_all)
/// for more information.
pub trait AfterAllOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(AfterAll);
mlir_op_trait!(AfterAll, OneResult);
mlir_op_trait!(AfterAll, ZeroRegions);
mlir_op_trait!(AfterAll, ZeroSuccessors);

/// Constructs a new detached/owned [`AfterAllOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`AfterAllOperation`] for more information on the operation semantics.
pub fn after_all<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    location: L,
) -> DetachedAfterAllOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.after_all", location)
        .add_operands(inputs)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::after_all`")
}

/// Name of the [`Attribute`] that is used to store [`HasChannelHandle::channel_id`],
/// and [`HasChannelHandle::channel_type`].
pub const COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE: &'static str = "channel_handle";

/// Trait that represents collective [`Operation`]s that support specifying and operating over specific channels.
pub trait SupportsChannelHandle<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the ID of the channel that this [`Operation`] transfers data over, if one is specified.
    fn channel_id(&self) -> Option<usize> {
        self.attribute(COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ChannelHandleAttributeRef>())
            .and_then(|attribute| attribute.channel_id())
    }

    /// Returns the type of the channel that this [`Operation`] transfers data over, if one is specified.
    fn channel_type(&self) -> Option<ChannelHandleType> {
        self.attribute(COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ChannelHandleAttributeRef>())
            .map(|attribute| attribute.channel_type())
    }
}

/// Trait that represents collective [`Operation`]s that require specifying and operating over specific channels.
pub trait HasChannelHandle<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the ID of the channel that this [`Operation`] transfers data over.
    fn channel_id(&self) -> usize {
        self.attribute(COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ChannelHandleAttributeRef>())
            .and_then(|attribute| attribute.channel_id())
            .expect(&format!(
                "invalid '{COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE}' attribute in StableHLO collective operation",
            ))
    }

    /// Returns the type of the channel that this [`Operation`] transfers data over.
    fn channel_type(&self) -> ChannelHandleType {
        self.attribute(COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<ChannelHandleAttributeRef>())
            .map(|attribute| attribute.channel_type())
            .expect(&format!(
                "invalid '{COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE}' attribute in StableHLO collective operation",
            ))
    }
}

/// Name of the [`Attribute`] that is used to store [`HasSourceTargetPairs::source_target_pairs`].
pub const COLLECTIVE_SOURCE_TARGET_PAIRS_ATTRIBUTE: &'static str = "source_target_pairs";

/// Trait that represents collective [`Operation`]s that have a `source_target_pairs` attribute.
pub trait HasSourceTargetPairs<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns a [`Vec`] that contains source-target device ID pairs for this [`SendRecvOperation`], if
    /// [`SendRecvOperation::is_host_transfer`] is `false`, and an empty vector otherwise.
    fn source_target_pairs(&self) -> Vec<(usize, usize)> {
        self.attribute(COLLECTIVE_SOURCE_TARGET_PAIRS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseIntegerElementsAttributeRef>())
            .map(|attribute| {
                let mut device_indices = unsafe { attribute.i64_elements() };
                let mut source_target_pairs = Vec::new();
                while let (Some(source), Some(target)) = (device_indices.next(), device_indices.next()) {
                    source_target_pairs.push((source as usize, target as usize));
                }
                source_target_pairs
            })
            .unwrap_or(Vec::new())
    }
}

impl<'t> Context<'t> {
    /// Internal helper for constructing the [`DenseIntegerElementsAttributeRef`] that is used to store
    /// [`HasSourceTargetPairs::source_target_pairs`].
    fn stable_hlo_source_target_pairs_attribute<'c, L: Location<'c, 't>>(
        &'c self,
        pairs: &[(usize, usize)],
        location: L,
    ) -> DenseIntegerElementsAttributeRef<'c, 't> {
        let i64_type = self.signless_integer_type(64);
        self.dense_integer_elements_attribute(
            self.tensor_type(i64_type, &[Size::Static(pairs.len()), Size::Static(2)], None, location).unwrap(),
            pairs
                .iter()
                .flat_map(|(source, target)| [*source, *target])
                .map(|id| self.integer_attribute(i64_type, id as i64))
                .collect::<Vec<_>>()
                .as_slice(),
        )
        .unwrap()
    }
}

/// Name of the [`Attribute`] that is used to store [`SendRecvOperation::is_host_transfer`].
pub const SEND_RECV_IS_HOST_TRANSFER_ATTRIBUTE: &'static str = "is_host_transfer";

/// Trait for functionality that is shared between [`SendOperation`] and [`RecvOperation`].
pub trait SendRecvOperation<'o, 'c: 'o, 't: 'c>:
    HasChannelHandle<'o, 'c, 't> + HasSourceTargetPairs<'o, 'c, 't>
{
    /// Returns `true` if this [`SendRecvOperation`] sends data to the host or receives data from the host
    /// (i.e., if it is not a device-to-device transfer).
    fn is_host_transfer(&self) -> bool {
        self.attribute(SEND_RECV_IS_HOST_TRANSFER_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<BooleanAttributeRef>())
            .map(|attribute| attribute.value())
            .expect(&format!(
                "invalid '{SEND_RECV_IS_HOST_TRANSFER_ATTRIBUTE}' attribute in StableHLO send/receive operation",
            ))
    }
}

/// StableHLO [`Operation`] that sends data (i.e., its inputs/operands) over a channel specified by
/// [`HasChannelHandle::channel_id`]. It takes as inputs a variable number of tensors that represent the data to be
/// sent over the channel, followed by a [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value, and
/// returns a new [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value. Tokens are used to control
/// the order in which operations execute. You can create a fresh token by using [`after_all`] with no inputs.
///
/// If [`SendRecvOperation::is_host_transfer`] is `true`, then the operation transfers data to the host. Otherwise, it
/// transfers data between devices based on the contents of [`HasSourceTargetPairs::source_target_pairs`]. Note that the
/// information specified by these two fields duplicates some of the information in [`HasChannelHandle::channel_type`].
/// There are [plans](https://github.com/openxla/stablehlo/issues/666) to only keep one of these two ways to specify
/// that information in the future.
///
/// If [`SendRecvOperation::is_host_transfer`] is `false` and [`HasSourceTargetPairs::source_target_pairs`] is [`None`],
/// then this operation will result in undefined behavior.
///
/// # Example
///
/// The following is an example of a [`SendOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0 = "stablehlo.send"(%arg0, %arg1) <{
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
///   is_host_transfer = false,
///   source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
/// }> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#send) for more information.
pub trait SendOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Send);
mlir_op_trait!(Send, OneResult);
mlir_op_trait!(Send, ZeroRegions);
mlir_op_trait!(Send, ZeroSuccessors);
mlir_op_trait!(Send, @local HasChannelHandle);
mlir_op_trait!(Send, @local HasSourceTargetPairs);
mlir_op_trait!(Send, @local SendRecvOperation);

/// Constructs a new detached/owned [`SendOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`SendOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn send<'v, 'k, 'c: 'v + 'k, 't: 'c, V: Value<'v, 'c, 't>, K: Value<'k, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    token: K,
    channel_id: usize,
    channel_type: ChannelHandleType,
    is_host_transfer: bool,
    source_target_pairs: Option<&[(usize, usize)]>,
    location: L,
) -> DetachedSendOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.send", location)
        .add_operands(inputs)
        .add_operand(token)
        .add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(Some(channel_id), channel_type),
        )
        .add_attribute(SEND_RECV_IS_HOST_TRANSFER_ATTRIBUTE, context.boolean_attribute(is_host_transfer));
    if let Some(source_target_pairs) = source_target_pairs {
        builder = builder.add_attribute(
            COLLECTIVE_SOURCE_TARGET_PAIRS_ATTRIBUTE,
            context.stable_hlo_source_target_pairs_attribute(source_target_pairs, location),
        );
    }
    builder
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::send`")
}

/// StableHLO [`Operation`] that receives data over a channel specified by [`HasChannelHandle::channel_id`]. It takes a
/// input a [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value, and returns the received tensors followed
/// by a new [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value. Tokens are used to control
/// the order in which operations execute. You can create a fresh token by using [`after_all`] with no inputs.
///
/// If [`SendRecvOperation::is_host_transfer`] is `true`, then the operation transfers data from the host. Otherwise, it
/// transfers data between devices based on the contents of [`HasSourceTargetPairs::source_target_pairs`]. Note that the
/// information specified by these two fields duplicates some of the information in [`HasChannelHandle::channel_type`].
/// There are [plans](https://github.com/openxla/stablehlo/issues/666) to only keep one of these two ways to specify
/// that information in the future.
///
/// If [`SendRecvOperation::is_host_transfer`] is `false` and [`HasSourceTargetPairs::source_target_pairs`] is [`None`],
/// then this operation will result in undefined behavior.
///
/// # Example
///
/// The following is an example of a [`RecvOperation`] represented using its [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0:2 = "stablehlo.recv"(%arg0) <{
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
///   is_host_transfer = false,
///   source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
/// }> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#recv) for more information.
pub trait RecvOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {}

mlir_op!(Recv);
mlir_op_trait!(Recv, ZeroRegions);
mlir_op_trait!(Recv, ZeroSuccessors);
mlir_op_trait!(Recv, @local HasChannelHandle);
mlir_op_trait!(Recv, @local HasSourceTargetPairs);
mlir_op_trait!(Recv, @local SendRecvOperation);

/// Constructs a new detached/owned [`RecvOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`RecvOperation`] for more information on the operation semantics.
///
/// Note that `output_types` must only contain the output types for the tensors that are to be received and not the
/// additional [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) that [`RecvOperation`] also returns.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn recv<'k, 'c: 'k, 't: 'c, K: Value<'k, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    token: K,
    channel_id: usize,
    channel_type: ChannelHandleType,
    is_host_transfer: bool,
    source_target_pairs: Option<&[(usize, usize)]>,
    output_types: &[T],
    location: L,
) -> DetachedRecvOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.recv", location)
        .add_operand(token)
        .add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(Some(channel_id), channel_type),
        )
        .add_attribute(SEND_RECV_IS_HOST_TRANSFER_ATTRIBUTE, context.boolean_attribute(is_host_transfer));
    if let Some(source_target_pairs) = source_target_pairs {
        builder = builder.add_attribute(
            COLLECTIVE_SOURCE_TARGET_PAIRS_ATTRIBUTE,
            context.stable_hlo_source_target_pairs_attribute(source_target_pairs, location),
        );
    }
    builder
        .add_results(output_types)
        .add_result(context.stable_hlo_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::recv`")
}

/// Name of the [`Attribute`] that is used to store [`OutfeedOperation::outfeed_config`].
pub const OUTFEED_CONFIG_ATTRIBUTE: &'static str = "outfeed_config";

/// StableHLO [`Operation`] that writes data to a feed that is configured by [`OutfeedOperation::outfeed_config`],
/// where this configuration is implementation-specific. This operation takes as inputs a variable number of tensors
/// that represent the data to be written to the feed, followed by a
/// [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value, and returns a new
/// [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value. Tokens are used to control the order in which
/// operations execute. You can create a fresh token by using [`after_all`] with no inputs.
///
/// # Example
///
/// The following is an example of an [`OutfeedOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %result = "stablehlo.outfeed"(%input, %token) <{
///   outfeed_config = "some_config"
/// }> : (tensor<2x2x2xi64>, !stablehlo.token) -> !stablehlo.token
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#outfeed) for more information.
pub trait OutfeedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the configuration string of this [`OutfeedOperation`].
    fn outfeed_config(&self) -> StringRef<'c> {
        self.attribute(OUTFEED_CONFIG_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
            .expect(&format!("invalid '{OUTFEED_CONFIG_ATTRIBUTE}' attribute in `stable_hlo::outfeed`"))
    }
}

mlir_op!(Outfeed);
mlir_op_trait!(Outfeed, OneResult);
mlir_op_trait!(Outfeed, ZeroRegions);
mlir_op_trait!(Outfeed, ZeroSuccessors);

/// Constructs a new detached/owned [`OutfeedOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`OutfeedOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn outfeed<
    'v,
    'k,
    'c: 'v + 'k,
    't: 'c,
    V: Value<'v, 'c, 't>,
    K: Value<'k, 'c, 't>,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    L: Location<'c, 't>,
>(
    inputs: &[V],
    token: K,
    configuration: N,
    location: L,
) -> DetachedOutfeedOperation<'c, 't> {
    location.context().load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.outfeed", location)
        .add_operands(inputs)
        .add_operand(token)
        .add_attribute(OUTFEED_CONFIG_ATTRIBUTE, configuration.into_with_context(location.context()))
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::outfeed`")
}

/// Name of the [`Attribute`] that is used to store [`InfeedOperation::infeed_config`].
pub const INFEED_CONFIG_ATTRIBUTE: &'static str = "infeed_config";

/// StableHLO [`Operation`] that reads data from a feed that is configured by [`InfeedOperation::infeed_config`],
/// where this configuration is implementation-specific. The operation takes an input
/// [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) and returns the read tensors followed by a new
/// [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) value. Tokens are used to control the order in
/// which operations execute.
///
/// # Example
///
/// The following is an example of an [`InfeedOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// %0:2 = "stablehlo.infeed"(%token) <{
///   infeed_config = "some_config"
/// }> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#infeed) for more information.
pub trait InfeedOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the configuration string of this [`InfeedOperation`].
    fn infeed_config(&self) -> StringRef<'c> {
        self.attribute(INFEED_CONFIG_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<StringAttributeRef>().map(|attribute| attribute.string()))
            .expect(&format!("invalid '{INFEED_CONFIG_ATTRIBUTE}' attribute in `stable_hlo::infeed`"))
    }
}

mlir_op!(Infeed);
mlir_op_trait!(Infeed, ZeroRegions);
mlir_op_trait!(Infeed, ZeroSuccessors);

/// Constructs a new detached/owned [`InfeedOperation`] at the specified [`Location`]. Refer to the documentation of
/// [`InfeedOperation`] for more information on the operation semantics.
///
/// Note that `output_types` must only contain the output types for the tensors that are to be read from the infeed
/// and not the additional [`TokenTypeRef`](crate::dialects::stable_hlo::TokenTypeRef) that [`InfeedOperation`] also
/// returns.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn infeed<
    'v,
    'c: 'v,
    't: 'c,
    V: Value<'v, 'c, 't>,
    N: IntoWithContext<'c, 't, StringAttributeRef<'c, 't>>,
    T: Type<'c, 't>,
    L: Location<'c, 't>,
>(
    token: V,
    configuration: N,
    output_types: &[T],
    location: L,
) -> DetachedInfeedOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    OperationBuilder::new("stablehlo.infeed", location)
        .add_operand(token)
        .add_attribute(INFEED_CONFIG_ATTRIBUTE, configuration.into_with_context(context))
        .add_results(output_types)
        .add_result(context.stable_hlo_token_type())
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::infeed`")
}

/// Name of the [`Attribute`] that is used to store [`HasReplicaGroups::replica_groups`].
pub const COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE: &'static str = "replica_groups";

/// Name of the [`Attribute`] that is used to store [`HasReplicaGroups::use_global_device_ids`].
pub const COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE: &'static str = "use_global_device_ids";

/// Trait that represents collective [`Operation`]s that support specifying and operating over replica groups.
pub trait HasReplicaGroups<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the optional replica groups of this [`Operation`]. This must be non-empty if
    /// [`HasReplicaGroups::use_global_device_ids`] is `true`.
    ///
    /// This a [`Vec`] over replica groups where each group is represented as a [`Vec`] of device IDs. In most cases,
    /// the groups must all have the same size. The only exception is [`all_reduce`] which supports non-uniform replica
    /// groups. These groups determine the order in which the gather operation is performed and which devices
    /// communicate with which other devices during the gather operation.
    fn replica_groups(&self) -> Vec<Vec<usize>> {
        self.attribute(COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<DenseIntegerElementsAttributeRef>())
            .and_then(|attribute| {
                let attribute_type = attribute.r#type().cast::<TensorTypeRef>();
                let device_ids = unsafe { attribute.i64_elements() };
                attribute_type.and_then(|tensor_type| {
                    tensor_type.dimension(1).value().map(|max_group_size| {
                        let mut groups = Vec::new();
                        let mut device_ids = device_ids.peekable();
                        while device_ids.peek().is_some() {
                            // We filter for non-negative values after the chunking below because the value `-1` is
                            // used as a "null" device padding value for when dealing with non-uniform replica groups.
                            groups.push(
                                device_ids
                                    .by_ref()
                                    .take(max_group_size)
                                    .filter(|id| *id >= 0)
                                    .map(|id| id as usize)
                                    .collect(),
                            );
                        }
                        groups
                    })
                })
            })
            .unwrap_or(Vec::new())
    }

    /// Returns `true` if this [`Operation`] uses global device IDs. Defaults to `false` if not specified.
    /// If this is `true`, then [`HasReplicaGroups::replica_groups`] must be non-empty.
    fn use_global_device_ids(&self) -> bool {
        self.attribute(COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE).is_some()
    }
}

impl<'t> Context<'t> {
    /// Internal helper for constructing the [`DenseIntegerElementsAttributeRef`] that is used to store
    /// [`HasReplicaGroups::replica_groups`].
    fn stable_hlo_replica_groups_attribute<'c, L: Location<'c, 't>>(
        &'c self,
        replica_groups: &[&[usize]],
        location: L,
    ) -> DenseIntegerElementsAttributeRef<'c, 't> {
        let i64_type = self.signless_integer_type(64);
        let group_count = replica_groups.len();
        let max_group_size = replica_groups.iter().map(|group| group.len()).max().unwrap();
        let attribute_type = self
            .tensor_type(i64_type, &[Size::Static(group_count), Size::Static(max_group_size)], None, location)
            .unwrap();
        let mut attribute_values = Vec::with_capacity(group_count * max_group_size);
        for group in replica_groups {
            let group_size = group.len();
            for id in *group {
                attribute_values.push(self.integer_attribute(i64_type, *id as i64));
            }
            if group_size < max_group_size {
                for _ in 0..max_group_size - group_size {
                    // `-1` acts as the "null" device ID that is used for padding.
                    attribute_values.push(self.integer_attribute(i64_type, -1));
                }
            }
        }
        self.dense_integer_elements_attribute(attribute_type, attribute_values.as_slice()).unwrap()
    }
}

/// Name of the [`Attribute`] that is used to store [`AllGatherOperation::all_gather_dimension`].
pub const ALL_GATHER_DIMENSION_ATTRIBUTE: &'static str = "all_gather_dim";

/// StableHLO [`Operation`] that gathers values from all processes within each process group in the StableHLO process
/// grid, and concatenates them along the [`AllGatherOperation::all_gather_dimension`] dimension.
///
/// # Example
///
/// The following is an example of an [`AllGatherOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // num_replicas: 2
/// // num_partitions: 1
/// // %operand0@(0, 0): [[1, 2], [3, 4]]
/// // %operand0@(1, 0): [[5, 6], [7, 8]]
/// // %operand1@(0, 0): [[11, 12], [13, 14]]
/// // %operand1@(1, 0): [[15, 16], [17, 18]]
/// %result:2 = "stablehlo.all_gather"(%operand0, %operand1) <{
///   all_gather_dim = 1 : i64,
///   replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
///   // use_global_device_ids = false
/// }> : (tensor<2x2xi64>, tensor<2x2xi64>) -> (tensor<2x4xi64>, tensor<2x4xi64>)
/// // %result0@(0, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
/// // %result0@(1, 0): [[1, 2, 5, 6], [3, 4, 7, 8]]
/// // %result1@(0, 0): [[11, 12, 15, 16], [13, 14, 17, 18]]
/// // %result1@(1, 0): [[11, 12, 15, 16], [13, 14, 17, 18]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#all_gather) for more
/// information (including information on how the process groups are defined).
pub trait AllGatherOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + HasReplicaGroups<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
    /// Returns the dimension along which concatenation occurs in this [`AllGatherOperation`].
    fn all_gather_dimension(&self) -> usize {
        self.attribute(ALL_GATHER_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!("invalid '{ALL_GATHER_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::all_gather`"))
    }
}

mlir_op!(AllGather);
mlir_op_trait!(AllGather, ZeroRegions);
mlir_op_trait!(AllGather, ZeroSuccessors);
mlir_op_trait!(AllGather, @local HasReplicaGroups);
mlir_op_trait!(AllGather, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`AllGatherOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`AllGatherOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn all_gather<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    all_gather_dimension: usize,
    replica_groups: &[&[usize]],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    use_global_device_ids: bool,
    output_types: &[T],
    location: L,
) -> DetachedAllGatherOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let i64_type = context.signless_integer_type(64);
    let mut builder = OperationBuilder::new("stablehlo.all_gather", location)
        .add_operands(inputs)
        .add_attribute(ALL_GATHER_DIMENSION_ATTRIBUTE, context.integer_attribute(i64_type, all_gather_dimension as i64))
        .add_attribute(
            COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE,
            context.stable_hlo_replica_groups_attribute(replica_groups, location),
        );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    if use_global_device_ids {
        builder = builder.add_attribute(COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_results(output_types)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::all_gather`")
}

/// StableHLO [`Operation`] that reduces values across all processes in a process group using a custom reduction
/// function. Within each process group in the StableHLO process grid, this operation applies a reduction function
/// `computation` to the values of the input tensors from each process and produces output tensors that contain the
/// reduced results. This is a collective communication operation where each process contributes its local tensor and
/// receives the same reduced result.
///
/// The `computation` region specifies the binary reduction function that takes two scalar inputs of the element type
/// and returns one scalar output of the same type. Common reduction operations include sum, product, min, and max.
///
/// # Example
///
/// The following is an example of an [`AllReduceOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // num_replicas: 2
/// // num_partitions: 1
/// // %operand0@(0, 0): [1, 2, 3, 4]
/// // %operand0@(1, 0): [5, 6, 7, 8]
/// // %operand1@(0, 0): [9, 10, 11, 12]
/// // %operand1@(1, 0): [13, 14, 15, 16]
/// %result:2 = "stablehlo.all_reduce"(%operand0, %operand0) <{
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>,
///   replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
///   // use_global_device_ids = false
/// }> ({
///   ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///     %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
///     stablehlo.return %0 : tensor<i64>
/// }) : (tensor<4xi64>, tensor<4xi64>) -> (tensor<4xi64>, tensor<4xi64>)
/// // %result0@(0, 0): [6, 8, 10, 12]
/// // %result0@(1, 0): [6, 8, 10, 12]
/// // %result1@(0, 0): [22, 24, 26, 28]
/// // %result1@(1, 0): [22, 24, 26, 28]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#all_reduce)
/// for more information.
pub trait AllReduceOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + HasReplicaGroups<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
}

mlir_op!(AllReduce);
mlir_op_trait!(AllReduce, OneRegion);
mlir_op_trait!(AllReduce, SingleBlock);
mlir_op_trait!(AllReduce, SingleBlockRegions);
mlir_op_trait!(AllReduce, ZeroSuccessors);
mlir_op_trait!(AllReduce, @local HasReplicaGroups);
mlir_op_trait!(AllReduce, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`AllReduceOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`AllReduceOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn all_reduce<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    replica_groups: &[&[usize]],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    use_global_device_ids: bool,
    computation: DetachedRegion<'c, 't>,
    location: L,
) -> DetachedAllReduceOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.all_reduce", location).add_operands(inputs).add_attribute(
        COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE,
        context.stable_hlo_replica_groups_attribute(replica_groups, location),
    );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    if use_global_device_ids {
        builder = builder.add_attribute(COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_region(computation)
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::all_reduce`")
}

/// Name of the [`Attribute`] that is used to store [`AllToAllOperation::split_dimension`].
pub const ALL_TO_ALL_SPLIT_DIMENSION_ATTRIBUTE: &'static str = "split_dimension";

/// Name of the [`Attribute`] that is used to store [`AllToAllOperation::split_count`].
pub const ALL_TO_ALL_SPLIT_COUNT_ATTRIBUTE: &'static str = "split_count";

/// Name of the [`Attribute`] that is used to store [`AllToAllOperation::concatenation_dimension`].
pub const ALL_TO_ALL_CONCATENATION_DIMENSION_ATTRIBUTE: &'static str = "concat_dimension";

/// StableHLO [`Operation`] that scatters and gathers data across all processes in a process group. Within each
/// process group in the StableHLO process grid, this operation splits the values of the input tensors along
/// [`AllToAllOperation::split_dimension`] into [`AllToAllOperation::split_count`] parts, scatters the split parts
/// between processes, concatenates the scattered parts along [`AllToAllOperation::concatenation_dimension`],
/// and produces result tensors. This is a collective communication operation that enables all-to-all data
/// exchange patterns.
///
/// # Example
///
/// The following is an example of an [`AllToAllOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // num_replicas: 2
/// // num_partitions: 1
/// // %arg0@(0, 0): [[1, 2, 3, 4],
/// //                [5, 6, 7, 8]]
/// // %arg0@(1, 0): [[9, 10, 11, 12],
/// //                [13, 14, 15, 16]]
/// // %arg1@(0, 0): [[17, 18, 19, 20],
/// //                [21, 22, 23, 24]]
/// // %arg1@(1, 0): [[25, 26, 27, 28],
/// //                [29, 30, 31, 32]]
/// %result:2 = "stablehlo.all_to_all"(%arg0, %arg1) <{
///   split_dimension = 1 : i64,
///   concat_dimension = 0 : i64,
///   split_count = 2 : i64,
///   replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>
/// }> : (tensor<2x4xi64>, tensor<2x4xi64>) -> (tensor<4x2xi64>, tensor<4x2xi64>)
/// // %result#0@(0, 0): [[1, 2], [5, 6], [9, 10], [13, 14]]
/// // %result#0@(1, 0): [[3, 4], [7, 8], [11, 12], [15, 16]]
/// // %result#1@(0, 0): [[17, 18], [21, 22], [25, 26], [29, 30]]
/// // %result#1@(1, 0): [[19, 20], [23, 24], [27, 28], [31, 32]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#all_to_all)
/// for more information.
pub trait AllToAllOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + HasReplicaGroups<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
    /// Returns the dimension along which operands are split in this [`AllToAllOperation`].
    fn split_dimension(&self) -> usize {
        self.attribute(ALL_TO_ALL_SPLIT_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!("invalid '{ALL_TO_ALL_SPLIT_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::all_to_all`"))
    }

    /// Returns the number of parts each operand is divided into in this [`AllToAllOperation`].
    fn split_count(&self) -> usize {
        self.attribute(ALL_TO_ALL_SPLIT_COUNT_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!("invalid '{ALL_TO_ALL_SPLIT_COUNT_ATTRIBUTE}' attribute in `stable_hlo::all_to_all`"))
    }

    /// Returns the dimension along which scattered parts are concatenated in this [`AllToAllOperation`].
    fn concatenation_dimension(&self) -> usize {
        self.attribute(ALL_TO_ALL_CONCATENATION_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!(
                "invalid '{ALL_TO_ALL_CONCATENATION_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::all_to_all`",
            ))
    }
}

mlir_op!(AllToAll);
mlir_op_trait!(AllToAll, ZeroRegions);
mlir_op_trait!(AllToAll, ZeroSuccessors);
mlir_op_trait!(AllToAll, @local HasReplicaGroups);
mlir_op_trait!(AllToAll, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`AllToAllOperation`] at the specified [`Location`]. Refer to the documentation
/// of [`AllToAllOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn all_to_all<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    inputs: &[V],
    split_dimension: usize,
    split_count: usize,
    concatenation_dimension: usize,
    replica_groups: &[&[usize]],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    use_global_device_ids: bool,
    location: L,
) -> DetachedAllToAllOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let i64_type = context.signless_integer_type(64);
    let mut builder = OperationBuilder::new("stablehlo.all_to_all", location)
        .add_operands(inputs)
        .add_attribute(
            ALL_TO_ALL_SPLIT_DIMENSION_ATTRIBUTE,
            context.integer_attribute(i64_type, split_dimension as i64),
        )
        .add_attribute(ALL_TO_ALL_SPLIT_COUNT_ATTRIBUTE, context.integer_attribute(i64_type, split_count as i64))
        .add_attribute(
            ALL_TO_ALL_CONCATENATION_DIMENSION_ATTRIBUTE,
            context.integer_attribute(i64_type, concatenation_dimension as i64),
        )
        .add_attribute(
            COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE,
            context.stable_hlo_replica_groups_attribute(replica_groups, location),
        );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    if use_global_device_ids {
        builder = builder.add_attribute(COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::all_to_all`")
}

/// StableHLO [`Operation`] that broadcasts data from a source process to target processes within each process group
/// in the StableHLO process grid. This is a collective communication operation that implements broadcasting patterns.
/// This operation has a single input/operand tensor and a single output/result tensor.
///
/// # Example
///
/// The following is an example of a [`CollectiveBroadcastOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
///
/// // num_replicas: 4
/// // num_partitions: 1
/// // %operand@(0, 0): [[1, 2]]
/// // %operand@(1, 0): [[3, 4]]
/// // %operand@(2, 0): [[5, 6]]
/// // %operand@(3, 0): [[7, 8]]
/// %result = "stablehlo.collective_broadcast"(%operand) <{
///   replica_groups = dense<[[2, 1]]> : tensor<1x2xi64>,
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>
/// }> : (tensor1x2xi64>) -> tensor<1x2xi64>
/// // %result@(0, 0): [[0, 0]]
/// // %result@(1, 0): [[5, 6]]
/// // %result@(2, 0): [[5, 6]]
/// // %result@(3, 0): [[0, 0]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#collective_broadcast)
/// for more information.
pub trait CollectiveBroadcastOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + HasReplicaGroups<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
}

mlir_op!(CollectiveBroadcast);
mlir_op_trait!(CollectiveBroadcast, OneResult);
mlir_op_trait!(CollectiveBroadcast, ZeroRegions);
mlir_op_trait!(CollectiveBroadcast, ZeroSuccessors);
mlir_op_trait!(CollectiveBroadcast, @local HasReplicaGroups);
mlir_op_trait!(CollectiveBroadcast, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`CollectiveBroadcastOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CollectiveBroadcastOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn collective_broadcast<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    replica_groups: &[&[usize]],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    location: L,
) -> DetachedCollectiveBroadcastOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder =
        OperationBuilder::new("stablehlo.collective_broadcast", location).add_operand(input).add_attribute(
            COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE,
            context.stable_hlo_replica_groups_attribute(replica_groups, location),
        );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    builder
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::collective_broadcast`")
}

/// StableHLO [`Operation`] that permutes data between processes according to specified
/// [`HasSourceTargetPairs::source_target_pairs`]. Within each process group in the StableHLO process grid,
/// this operation sends the value of the input tensor from the source process to the target process according to
/// [`HasSourceTargetPairs::source_target_pairs`], and produces a result tensor. This is a collective
/// communication operation that enables flexible point-to-point communication patterns between processes.
///
/// # Example
///
/// The following is an example of a [`CollectivePermuteOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // num_replicas: 3
/// // num_partitions: 1
/// // %operand@(0, 0): [[1, 2], [3, 4]]
/// // %operand@(1, 0): [[5, 6], [7, 8]]
/// // %operand@(2, 0): [[9, 10], [11, 12]]
/// %result = "stablehlo.collective_permute"(%operand) <{
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
///   source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
/// }> : (tensor<2x2xi64>) -> tensor<2x2xi64>
/// // %result@(0, 0): [[0, 0], [0, 0]]
/// // %result@(1, 0): [[1, 2], [3, 4]]
/// // %result@(2, 0): [[5, 6], [7, 8]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#collective_permute)
/// for more information.
pub trait CollectivePermuteOperation<'o, 'c: 'o, 't: 'c>:
    HasSourceTargetPairs<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
}

mlir_op!(CollectivePermute);
mlir_op_trait!(CollectivePermute, OneResult);
mlir_op_trait!(CollectivePermute, ZeroRegions);
mlir_op_trait!(CollectivePermute, ZeroSuccessors);
mlir_op_trait!(CollectivePermute, @local HasSourceTargetPairs);
mlir_op_trait!(CollectivePermute, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`CollectivePermuteOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`CollectivePermuteOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn collective_permute<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, L: Location<'c, 't>>(
    input: V,
    source_target_pairs: &[(usize, usize)],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    location: L,
) -> DetachedCollectivePermuteOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.collective_permute", location).add_operand(input).add_attribute(
        COLLECTIVE_SOURCE_TARGET_PAIRS_ATTRIBUTE,
        context.stable_hlo_source_target_pairs_attribute(source_target_pairs, location),
    );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    builder
        .enable_result_type_inference()
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::collective_permute`")
}

/// Name of the [`Attribute`] that is used to store [`ReduceScatterOperation::dimension`].
pub const REDUCE_SCATTER_DIMENSION_ATTRIBUTE: &'static str = "scatter_dimension";

/// StableHLO [`Operation`] that performs a reduction using [`ReduceScatterOperation::computation`] over the values of
/// the operand tensor, across [`HasReplicaGroups::replica_groups`] in the StableHLO process grid, followed by
/// splitting the reduction result along [`ReduceScatterOperation::dimension`] into parts, and scattering those parts
/// between the processes to produce the output tensor.
///
/// # Example
///
/// The following is an example of a [`ReduceScatterOperation`] represented using its
/// [`Display`](std::fmt::Display) rendering:
///
/// ```mlir
/// // num_replicas: 2
/// // num_partitions: 1
/// // %operand@(0, 0): [[1, 2, 3, 4],
/// //                   [5, 6, 7, 8]]
/// // %operand@(1, 0): [[9, 10, 11, 12],
/// //                   [13, 14, 15, 16]]
/// %result = "stablehlo.reduce_scatter"(%operand) <{
///   channel_handle = #stablehlo.channel_handle<handle = 0, type = 0>,
///   replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>,
///   scatter_dimension = 1 : i64
/// }> ({
/// ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
///   %0 = stablehlo.add %arg0, %arg1 : tensor<i64>
///   stablehlo.return %0 : tensor<i64>
/// }) : (tensor<2x4xi64>) -> tensor<2x2xi64>
/// // %result@(0, 0): [[10, 12],
/// //                  [18, 20]]
/// // %result@(1, 0): [[14, 16],
/// //                  [22, 24]]
/// ```
///
/// Refer to the [official StableHLO specification](https://openxla.org/stablehlo/spec#reduce_scatter)
/// for more information.
pub trait ReduceScatterOperation<'o, 'c: 'o, 't: 'c>:
    Operation<'o, 'c, 't> + OneRegion<'o, 'c, 't> + HasReplicaGroups<'o, 'c, 't> + SupportsChannelHandle<'o, 'c, 't>
{
    /// Returns the dimension along which to scatter the reduction result for this [`ReduceScatterOperation`].
    fn dimension(&self) -> usize {
        self.attribute(REDUCE_SCATTER_DIMENSION_ATTRIBUTE)
            .and_then(|attribute| attribute.cast::<IntegerAttributeRef>())
            .map(|attribute| attribute.signless_value() as usize)
            .expect(&format!(
                "invalid '{REDUCE_SCATTER_DIMENSION_ATTRIBUTE}' attribute in `stable_hlo::reduce_scatter`"
            ))
    }

    /// Returns a reference to the [`Region`](crate::Region) that contains the reduction computation
    /// used by this [`ReduceScatterOperation`].
    fn computation(&self) -> RegionRef<'o, 'c, 't> {
        self.body_region()
    }
}

mlir_op!(ReduceScatter);
mlir_op_trait!(ReduceScatter, OneOperand);
mlir_op_trait!(ReduceScatter, OneResult);
mlir_op_trait!(ReduceScatter, OneRegion);
mlir_op_trait!(ReduceScatter, SingleBlock);
mlir_op_trait!(ReduceScatter, SingleBlockRegions);
mlir_op_trait!(ReduceScatter, ZeroSuccessors);
mlir_op_trait!(ReduceScatter, @local HasReplicaGroups);
mlir_op_trait!(ReduceScatter, @local SupportsChannelHandle);

/// Constructs a new detached/owned [`ReduceScatterOperation`] at the specified [`Location`]. Refer to the
/// documentation of [`ReduceScatterOperation`] for more information on the operation semantics.
///
/// Note that if any of the inputs to this function are invalid, it will panic!
pub fn reduce_scatter<'v, 'c: 'v, 't: 'c, V: Value<'v, 'c, 't>, T: Type<'c, 't>, L: Location<'c, 't>>(
    operand: V,
    dimension: usize,
    replica_groups: &[&[usize]],
    channel_id: Option<usize>,
    channel_type: Option<ChannelHandleType>,
    use_global_device_ids: bool,
    computation: DetachedRegion<'c, 't>,
    output_type: T,
    location: L,
) -> DetachedReduceScatterOperation<'c, 't> {
    let context = location.context();
    context.load_dialect(DialectHandle::stable_hlo());
    let mut builder = OperationBuilder::new("stablehlo.reduce_scatter", location)
        .add_operand(operand)
        .add_attribute(
            REDUCE_SCATTER_DIMENSION_ATTRIBUTE,
            location.context().integer_attribute(context.signless_integer_type(64), dimension as i64),
        )
        .add_attribute(
            COLLECTIVE_REPLICA_GROUPS_ATTRIBUTE,
            context.stable_hlo_replica_groups_attribute(replica_groups, location),
        );
    if let Some(channel_id) = channel_id {
        builder = builder.add_attribute(
            COLLECTIVE_CHANNEL_HANDLE_ATTRIBUTE,
            context.stable_hlo_channel_handle(
                Some(channel_id),
                channel_type.expect("channel type is required when channel ID is provided"),
            ),
        );
    }
    if use_global_device_ids {
        builder = builder.add_attribute(COLLECTIVE_USE_GLOBAL_DEVICE_IDS_ATTRIBUTE, context.unit_attribute());
    }
    builder
        .add_region(computation)
        .add_result(output_type)
        .build()
        .and_then(|operation| unsafe { operation.cast() })
        .expect("invalid arguments to `stable_hlo::reduce_scatter`")
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};
    use crate::dialects::{func, stable_hlo};
    use crate::{Block, Context, Operation, Region, Size};

    use super::*;

    #[test]
    fn test_channel_handle_attribute() {
        let context = Context::new();
        let attribute = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToHost);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.channel_id(), Some(42));
        assert_eq!(attribute.channel_type(), ChannelHandleType::DeviceToHost);
        let attribute = context.stable_hlo_channel_handle(Some(24), ChannelHandleType::Unknown);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.channel_id(), Some(24));
        assert_eq!(attribute.channel_type(), ChannelHandleType::Unknown);
    }

    #[test]
    fn test_channel_handle_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToDevice);
        let attribute_2 = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToDevice);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::Unknown);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToDevice);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_channel_handle_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToDevice);
        test_attribute_display_and_debug(attribute, "#stablehlo.channel_handle<handle = 42, type = 1>");
    }

    #[test]
    fn test_channel_handle_attribute_casting() {
        let context = Context::new();
        let attribute = context.stable_hlo_channel_handle(Some(42), ChannelHandleType::DeviceToDevice);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_partition_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let ui32_type = context.unsigned_integer_type(32);
        let tensor_type = context.tensor_type(ui32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = block.append_operation(partition_id(location));
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_partition_id",
                func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_partition_id() -> tensor<ui32> {
                    %0 = stablehlo.partition_id : tensor<ui32>
                    return %0 : tensor<ui32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_replica_id() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let ui32_type = context.unsigned_integer_type(32);
        let tensor_type = context.tensor_type(ui32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block_with_no_arguments();
            let op = block.append_operation(replica_id(location));
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_replica_id",
                func::FuncAttributes { arguments: vec![], results: vec![tensor_type.into()], ..Default::default() },
                block.into(),
                location,
            )
        });
        assert!(module.verify());
        assert_eq!(
            module.to_string(),
            indoc! {"
                module {
                  func.func @test_replica_id() -> tensor<ui32> {
                    %0 = stablehlo.replica_id : tensor<ui32>
                    return %0 : tensor<ui32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_after_all() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.stable_hlo_token_type();
        module.body().append_operation({
            let mut block = context.block(&[(token_type, location), (token_type, location)]);
            let op = after_all(block.arguments().collect::<Vec<_>>().as_slice(), location);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_after_all",
                func::FuncAttributes {
                    arguments: vec![token_type.into(), token_type.into()],
                    results: vec![token_type.into()],
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
                  func.func @test_after_all(%arg0: !stablehlo.token, %arg1: !stablehlo.token) -> !stablehlo.token {
                    %0 = stablehlo.after_all %arg0, %arg1 : !stablehlo.token
                    return %0 : !stablehlo.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_send() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.stable_hlo_token_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type.as_ref(), location), (token_type.as_ref(), location)]);
            let op = send(
                &[block.argument(0).unwrap()],
                block.argument(1).unwrap(),
                0,
                ChannelHandleType::DeviceToDevice,
                false,
                Some(&[(0, 1), (1, 2)]),
                location,
            );
            assert_eq!(op.channel_id(), 0);
            assert_eq!(op.channel_type(), ChannelHandleType::DeviceToDevice);
            assert!(!op.is_host_transfer());
            assert_eq!(op.source_target_pairs(), vec![(0, 1), (1, 2)]);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_send",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), token_type.into()],
                    results: vec![token_type.into()],
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
                  func.func @test_send(%arg0: tensor<2x2xi64>, %arg1: !stablehlo.token) -> !stablehlo.token {
                    %0 = \"stablehlo.send\"(%arg0, %arg1) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>, \
                      is_host_transfer = false, \
                      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>\
                    }> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
                    return %0 : !stablehlo.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_recv() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.stable_hlo_token_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(token_type, location)]);
            let op = recv(
                block.argument(0).unwrap(),
                0,
                ChannelHandleType::DeviceToDevice,
                false,
                Some(&[(0, 1), (1, 2)]),
                &[tensor_type],
                location,
            );
            assert_eq!(op.channel_id(), 0);
            assert_eq!(op.channel_type(), ChannelHandleType::DeviceToDevice);
            assert!(!op.is_host_transfer());
            assert_eq!(op.source_target_pairs(), vec![(0, 1), (1, 2)]);
            let op = block.append_operation(op);
            block.append_operation(func::r#return(op.results().collect::<Vec<_>>().as_slice(), location));
            func::func(
                "test_recv",
                func::FuncAttributes {
                    arguments: vec![token_type.into()],
                    results: vec![tensor_type.into(), token_type.into()],
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
                  func.func @test_recv(%arg0: !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token) {
                    %0:2 = \"stablehlo.recv\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>, \
                      is_host_transfer = false, \
                      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>\
                    }> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
                    return %0#0, %0#1 : tensor<2x2xi64>, !stablehlo.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_outfeed() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.stable_hlo_token_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type.as_ref(), location), (token_type.as_ref(), location)]);
            let op = outfeed(&[block.argument(0).unwrap()], block.argument(1).unwrap(), "config", location);
            assert_eq!(op.outfeed_config().as_str().unwrap(), "config");
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_outfeed",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into(), token_type.into()],
                    results: vec![token_type.into()],
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
                  func.func @test_outfeed(%arg0: tensor<2x2xi64>, %arg1: !stablehlo.token) -> !stablehlo.token {
                    %0 = \"stablehlo.outfeed\"(%arg0, %arg1) <{\
                      outfeed_config = \"config\"\
                    }> : (tensor<2x2xi64>, !stablehlo.token) -> !stablehlo.token
                    return %0 : !stablehlo.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_infeed() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let token_type = context.stable_hlo_token_type();
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(token_type, location)]);
            let op = infeed(block.argument(0).unwrap(), "config", &[tensor_type], location);
            assert_eq!(op.infeed_config().as_str().unwrap(), "config");
            let op = block.append_operation(op);
            block.append_operation(func::r#return(op.results().collect::<Vec<_>>().as_slice(), location));
            func::func(
                "test_infeed",
                func::FuncAttributes {
                    arguments: vec![token_type.into()],
                    results: vec![tensor_type.into(), token_type.into()],
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
                  func.func @test_infeed(%arg0: !stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token) {
                    %0:2 = \"stablehlo.infeed\"(%arg0) <{\
                      infeed_config = \"config\"\
                    }> : (!stablehlo.token) -> (tensor<2x2xi64>, !stablehlo.token)
                    return %0#0, %0#1 : tensor<2x2xi64>, !stablehlo.token
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_gather() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_tensor_type =
            context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let output_tensor_type =
            context.tensor_type(i64_type, &[Size::Static(2), Size::Static(4)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let op = all_gather(
                &[block.argument(0).unwrap()],
                1,
                &[&[0, 2], &[1, 3]],
                Some(0),
                Some(ChannelHandleType::DeviceToDevice),
                true,
                &[output_tensor_type],
                location,
            );
            assert_eq!(op.all_gather_dimension(), 1);
            assert_eq!(op.replica_groups(), vec![vec![0, 2], vec![1, 3]]);
            assert_eq!(op.channel_id(), Some(0));
            assert_eq!(op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            assert!(op.use_global_device_ids());
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_all_gather",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
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
                  func.func @test_all_gather(%arg0: tensor<2x2xi64>) -> tensor<2x4xi64> {
                    %0 = \"stablehlo.all_gather\"(%arg0) <{\
                      all_gather_dim = 1 : i64, \
                      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>, \
                      replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, \
                      use_global_device_ids\
                    }> : (tensor<2x2xi64>) -> tensor<2x4xi64>
                    return %0 : tensor<2x4xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_reduce() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let f32_type = context.float32_type();
        let tensor_type = context.tensor_type(f32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        let scalar_tensor_type = context.tensor_type(f32_type, &[], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let mut computation_region = context.region();
            let mut computation_block =
                context.block(&[(scalar_tensor_type, location), (scalar_tensor_type, location)]);
            let add_op = stable_hlo::add(
                computation_block.argument(0).unwrap(),
                computation_block.argument(1).unwrap(),
                location,
            );
            let add_op = computation_block.append_operation(add_op);
            computation_block.append_operation(stable_hlo::r#return(&[add_op.result(0).unwrap()], location));
            computation_region.append_block(computation_block);
            let computation = computation_region.into();
            let op = all_reduce(
                &[block.argument(0).unwrap()],
                &[&[0, 2], &[1]],
                Some(1),
                Some(ChannelHandleType::DeviceToDevice),
                true,
                computation,
                location,
            );
            assert_eq!(op.replica_groups(), vec![vec![0, 2], vec![1]]);
            assert_eq!(op.channel_id(), Some(1));
            assert_eq!(op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            assert!(op.use_global_device_ids());
            let result = block.append_operation(op);
            block.append_operation(func::r#return(&[result.result(0).unwrap()], location));
            func::func(
                "test_all_reduce",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @test_all_reduce(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
                    %0 = \"stablehlo.all_reduce\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, \
                      replica_groups = dense<[[0, 2], [1, -1]]> : tensor<2x2xi64>, \
                      use_global_device_ids\
                    }> ({
                    ^bb0(%arg1: tensor<f32>, %arg2: tensor<f32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<f32>
                      stablehlo.return %1 : tensor<f32>
                    }) : (tensor<2x2xf32>) -> tensor<2x2xf32>
                    return %0 : tensor<2x2xf32>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_all_to_all() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let input_tensor_type =
            context.tensor_type(i64_type, &[Size::Static(2), Size::Static(4)], None, location).unwrap();
        let output_tensor_type =
            context.tensor_type(i64_type, &[Size::Static(4), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_tensor_type, location)]);
            let op = all_to_all(
                &[block.argument(0).unwrap()],
                1,
                2,
                0,
                &[&[0, 2], &[1, 3]],
                Some(1),
                Some(ChannelHandleType::DeviceToDevice),
                true,
                location,
            );
            assert_eq!(op.split_dimension(), 1);
            assert_eq!(op.split_count(), 2);
            assert_eq!(op.concatenation_dimension(), 0);
            assert_eq!(op.replica_groups(), vec![vec![0, 2], vec![1, 3]]);
            assert_eq!(op.channel_id(), Some(1));
            assert_eq!(op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_all_to_all",
                func::FuncAttributes {
                    arguments: vec![input_tensor_type.into()],
                    results: vec![output_tensor_type.into()],
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
                  func.func @test_all_to_all(%arg0: tensor<2x4xi64>) -> tensor<4x2xi64> {
                    %0 = \"stablehlo.all_to_all\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, \
                      concat_dimension = 0 : i64, \
                      replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, \
                      split_count = 2 : i64, \
                      split_dimension = 1 : i64\
                    }> {use_global_device_ids} : (tensor<2x4xi64>) -> tensor<4x2xi64>
                    return %0 : tensor<4x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collective_broadcast() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(1), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let op = collective_broadcast(
                block.argument(0).unwrap(),
                &[&[0, 2], &[1, 3]],
                Some(1),
                Some(ChannelHandleType::DeviceToDevice),
                location,
            );
            assert_eq!(op.replica_groups(), vec![vec![0, 2], vec![1, 3]]);
            assert_eq!(op.channel_id(), Some(1));
            assert_eq!(op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_collective_broadcast",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @test_collective_broadcast(%arg0: tensor<1x2xi64>) -> tensor<1x2xi64> {
                    %0 = \"stablehlo.collective_broadcast\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, \
                      replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>\
                    }> : (tensor<1x2xi64>) -> tensor<1x2xi64>
                    return %0 : tensor<1x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_collective_permute() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i64_type = context.signless_integer_type(64);
        let tensor_type = context.tensor_type(i64_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(tensor_type, location)]);
            let op = collective_permute(
                block.argument(0).unwrap(),
                &[(0, 1), (1, 2)],
                Some(0),
                Some(ChannelHandleType::DeviceToDevice),
                location,
            );
            assert_eq!(op.source_target_pairs(), vec![(0, 1), (1, 2)]);
            assert_eq!(op.channel_id(), Some(0));
            assert_eq!(op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            let op = block.append_operation(op);
            block.append_operation(func::r#return(&[op.result(0).unwrap()], location));
            func::func(
                "test_collective_permute",
                func::FuncAttributes {
                    arguments: vec![tensor_type.into()],
                    results: vec![tensor_type.into()],
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
                  func.func @test_collective_permute(%arg0: tensor<2x2xi64>) -> tensor<2x2xi64> {
                    %0 = \"stablehlo.collective_permute\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 0, type = 1>, \
                      source_target_pairs = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>\
                    }> : (tensor<2x2xi64>) -> tensor<2x2xi64>
                    return %0 : tensor<2x2xi64>
                  }
                }
            "},
        );
    }

    #[test]
    fn test_reduce_scatter() {
        let context = Context::new();
        let location = context.unknown_location();
        let module = context.module(location);
        let i32_type = context.signless_integer_type(32);
        let input_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(4)], None, location).unwrap();
        let initial_value_type = context.tensor_type(i32_type, &[], None, location).unwrap();
        let output_type = context.tensor_type(i32_type, &[Size::Static(2), Size::Static(2)], None, location).unwrap();
        module.body().append_operation({
            let mut block = context.block(&[(input_type, location)]);
            let input = block.argument(0).unwrap();
            let mut region = context.region();
            let mut region_block = context.block(&[(initial_value_type, location), (initial_value_type, location)]);
            let lhs = region_block.argument(0).unwrap();
            let rhs = region_block.argument(1).unwrap();
            let add_op = stable_hlo::add(lhs, rhs, location);
            let add_op = region_block.append_operation(add_op);
            let return_op = stable_hlo::r#return(&[add_op.result(0).unwrap()], location);
            region_block.append_operation(return_op);
            region.append_block(region_block);
            let reduce_scatter_op = reduce_scatter(
                input,
                1,
                &[&[0, 2], &[1, 3]],
                Some(1),
                Some(ChannelHandleType::DeviceToDevice),
                false,
                region.into(),
                output_type,
                location,
            );
            assert_eq!(reduce_scatter_op.dimension(), 1);
            assert_eq!(reduce_scatter_op.replica_groups(), vec![vec![0, 2], vec![1, 3]]);
            assert_eq!(reduce_scatter_op.channel_id(), Some(1));
            assert_eq!(reduce_scatter_op.channel_type(), Some(ChannelHandleType::DeviceToDevice));
            assert_eq!(reduce_scatter_op.use_global_device_ids(), false);
            assert_eq!(reduce_scatter_op.computation().blocks().count(), 1);
            assert_eq!(reduce_scatter_op.operands().count(), 1);
            assert_eq!(reduce_scatter_op.results().count(), 1);
            let reduce_scatter_op = block.append_operation(reduce_scatter_op);
            block.append_operation(func::r#return(&[reduce_scatter_op.result(0).unwrap()], location));
            func::func(
                "reduce_scatter_test",
                func::FuncAttributes {
                    arguments: vec![input_type.into()],
                    results: vec![output_type.into()],
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
                  func.func @reduce_scatter_test(%arg0: tensor<2x4xi32>) -> tensor<2x2xi32> {
                    %0 = \"stablehlo.reduce_scatter\"(%arg0) <{\
                      channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, \
                      replica_groups = dense<[[0, 2], [1, 3]]> : tensor<2x2xi64>, \
                      scatter_dimension = 1 : i64\
                    }> ({
                    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
                      %1 = stablehlo.add %arg1, %arg2 : tensor<i32>
                      stablehlo.return %1 : tensor<i32>
                    }) : (tensor<2x4xi32>) -> tensor<2x2xi32>
                    return %0 : tensor<2x2xi32>
                  }
                }
            "}
        );
    }
}
