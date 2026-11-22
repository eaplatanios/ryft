//! Moving values between concrete placements, which is the Ryft analogue of JAX's
//! [`device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html). A [`Placement`] names where an
//! array lives concretely (i.e., a [`Sharding`] over the devices of a [`DeviceMesh`], optionally with a [`Memory`]
//! space and a storage [`Layout`]). Moving a value to another placement never changes its elements, shape, or data
//! type, and unlike a [`Reshard`](crate::Reshard), which redistributes a value over the mesh that it already lives
//! on, it can move the value onto the devices of another mesh.
//!
//! The module is organized around the following entities:
//!
//!   - [`PlaceOperation`] is the operation that a placement stages. It carries an optional target, an optional source
//!     placement, and a copy policy that states whether its output may share storage with its input, and it owns the
//!     type inference rule that validates both (e.g., that the target preserves the reduction state of the input and
//!     splits every static extent evenly) along with the interpretation, partial evaluation, batching, differentiation,
//!     and transposition rules of a placement. Without a target, the value stays where it is.
//!   - [`PlacementRequest`] is the target and the source of a [`PlaceOperation`], which is a resolved [`Placement`],
//!     a single [`Device`], or a change of storage in place, and it is the one place that states the contract of each
//!     form, for every path that stages, executes, compiles, or transforms a placement. Frontends stage a request in
//!     the form that they are given, and backends resolve a device or in-place request against a concrete value only
//!     when the placement executes on it, so that the type that the type rule of the request infers is the type that
//!     every path produces.
//!   - [`Placed`] reports the placement that a value concretely lives on. Only values that hold device buffers (e.g.,
//!     the arrays of a backend) have one, because devices and placement over [`MeshAxisType::Auto`] mesh axes are
//!     runtime state that types do not record. Staged values and reference [`Array`]s have none, wrappers such as
//!     tracers report the placement of the value that they wrap, and batched values report that of their batch items.
//!   - [`PlaceDispatch`] places one leaf with an optional target and a given copy policy through the
//!     [`Value::Dispatch`] policy of its value type. Values that dispatch through their domain bind a
//!     [`PlaceOperation`] in their own context, reference [`Array`]s record the placement in their types,
//!     and array IR values place their array members.
//!   - [`Place`] is the value capability through which callers place individual values and the leaves of structured
//!     values, with one target and one source of any form per leaf (i.e., a [`PlacementRequestParameter`]), preserving
//!     their structure and static fields. Its results may share storage with its inputs, and callers that need results
//!     with storage of their own use [`PlaceDispatch`] directly.
//!
//! The source of a placement is informative about devices. It states where the value is expected to live when the
//! placement is created, which transposition needs in order to move the cotangent back to those devices, because types
//! do not record them, and transposition swaps the source and the target. Type inference checks a source against
//! what the input type tracks (i.e., its sharding, memory, and declared layout), and only against it, while backends
//! that execute a placement on a concrete value move it from wherever it actually lives. When a placement omits its
//! source, the rule that executes it on a value that concretely lives somewhere (i.e., interpretation, batching, or
//! differentiation) records the [`Placed`] placement of that value instead, so concrete backend arrays record where
//! they actually live, while staged values leave the source unknown, and a source that names only a memory or a layout
//! is completed with that placement (refer to [`PlacementRequest::resolve_source`]) after the placement is validated
//! as it is given (refer to [`PlaceOperation::infer_output_type`]). Inside batched functions, the placement of a batch
//! item also records how its batch is distributed (refer to [`Placement::batch_shardings`]), so a source inferred from
//! a batch item is lifted back to where the packed value lives, an explicit target places every batch item where it
//! says, and a placement without a target leaves the packed value where it is.

use std::fmt::Display;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::arrays::{
    Array, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayIrBatch, ArrayIrType, ArrayIrValue,
    ArrayType, Device, DeviceMesh, Dimension, Layout, LogicalMesh, Memory, MeshAxisType, Placement, Shape, Sharding,
    ShardingDimension, decode_logical_bytes,
};
use crate::axes::Axis;
use crate::batching::{
    BatchableOperation, BatchingContext, BatchingDriver, BatchingError, BatchingPolicy, BatchingTracer,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationDual, DifferentiationPolicy, DifferentiationTracer};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::Capability;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::sharding::validate_preserved_mesh_axes;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, find_duplicate_paths};
use crate::partial::{PartialTracer, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, OperationProjection, OperationProvider, ProgramError, ProjectedValue,
    RegionInterface, Type, TypeError, Typed, Value, ValueDirectDispatch, ValueDomainDispatch, ValueProjection,
};
use crate::tracing::Tracer;

/// Target or source of a [`PlaceOperation`], which is a resolved [`Placement`], a single [`Device`], or a change of
/// storage in place. This is the one place that states the contract of each form, which type inference, frontends,
/// eager execution, compilation, and the transformation rules of [`PlaceOperation`] share.
///
/// A [`Placement`] names a device mesh and a sharding over it, so the sharding of a target is what the output type
/// records, and the sharding of a source is what the input type must track. A bare device names no mesh, and the mesh
/// of a value is not always known when its placement is staged (e.g., while a function is traced before it is compiled
/// for a device mesh), so the device form keeps the intent "this device" until a backend knows where the value lives.
/// The in-place form names no devices at all, so it changes only the storage of a value wherever the value lives,
/// which is the analogue of a memory space destination of JAX's `device_put` (e.g.,
/// `jax.device_put(x, jax.memory.Space.Host)`), which keeps the devices and the distribution of `x`.
///
/// # Staging and Resolution
///
/// A frontend stages a request in the form that it was given. A bare device stays a device request and a bare memory
/// stays an in-place request, whether or not the staged value wraps a concrete array, and only a request that names a
/// mesh or a sharding is a resolved placement when it is staged. A backend resolves a device or in-place request
/// against a concrete value in exactly one place, when the placement executes on that value (or against the device
/// mesh that a program is compiled for). The type rule of [`PlaceOperation`] is the single owner of what a request
/// does to the type of a value, and executing a placement must produce a value of exactly that type, so that eager
/// execution, the replay of a staged program, compilation, and the transformations of a placement agree. In particular,
/// a request never changes the reduction state of a value on any path, a concrete value included (refer to
/// [`PlaceOperation::infer_output_type`]), so a value whose reduction is pending or complete must be reduced or
/// resharded by the operations that own reductions before a request can drop that state.
///
/// # Device Targets
///
/// As a target, the device form leaves a value that lives wholly on that one device:
///
///   - A value that the input type tracks over a logical mesh of one device keeps that logical mesh and its sharding,
///     laid over the target device.
///   - A value that the input type tracks over a logical mesh of more than one device is gathered onto the device,
///     so the output type is replicated over the canonical logical mesh of one device (i.e.,
///     [`LogicalMesh::single_device`]), which is also the mesh that backends give such a value.
///   - A value without a tracked sharding stays without one.
///   - Without a memory, the output is in the default memory of the device (i.e., [`Memory::Device`]), so a value
///     leaves any other memory (e.g., pinned host memory), as with JAX's eager `device_put` onto a device, and with
///     a memory, it is in that memory (refer to [`Self::target_memory`]). This holds inside compiled functions too.
///   - The output has no unreduced or reduced state. A completed reduction (i.e., a reduced axis of any size) leaves
///     the complete value on every device along its axis, and a pending reduction over mesh axes of size one has a
///     single contribution, so both are complete on one device. A pending reduction over a larger mesh axis (i.e., an
///     unreduced axis) describes contributions on more devices than one, which a placement cannot combine, so such an
///     input is rejected on every path, a concrete array included, and must be reduced first. Transposition cannot give
///     the cotangent of an input that is reduced over a larger mesh axis its dual unreduced state, so differentiating
///     such a placement in reverse mode fails.
///   - A value inside a manual region (i.e., one that varies along manual mesh axes or carries reduction state over
///     them, whatever their sizes) cannot be moved, as with a [`Placement`] target.
///
/// # In-Place Targets
///
/// As a target, the in-place form leaves a value on the devices where it already lives, with the distribution that it
/// already has, and changes only its storage:
///
///   - The output keeps the sharding of the input type exactly, including its reduction state and its variation along
///     manual mesh axes, and backends resolve the request against the value (e.g., against the mesh and the sharding
///     of the concrete array that the placement executes on, or against the placement that a compiled program gives
///     the value).
///   - The output takes the memory and the layout that the request names, and it keeps those of the input otherwise.
///     A request that names neither changes nothing, which an absent target already expresses, so it is rejected.
///   - Nothing moves, so a value inside a manual region is accepted.
///
/// # Sources
///
/// A source states where the input is expected to live. The devices that it names are informative, because a backend
/// moves a concrete value from wherever it actually lives, but the sharding of a placement source must describe what
/// the input type tracks. Transposition turns a source into the destination of the cotangent, so a backend that
/// compiles a program for fixed devices may reject a source that names other devices like such a target, when the
/// program is compiled rather than when it is differentiated. The memory and the layout that a source of any form
/// names are checked against the input type in the same way on every path (i.e., the memory must be the memory of the
/// input type, and the layout must address the shape of the input and be the layout of the input type when that type
/// declares one). The device form states that the input is expected to live on that one device, so it is rejected for
/// an input that its type tracks over a mesh of more than one device. The in-place form states only the storage in
/// which the input is expected, so an in-place source that names neither a memory nor a layout says nothing and is
/// rejected. A path that knows where a value concretely lives completes an in-place source with that placement (refer
/// to [`Self::resolve_source`]), so that a source that names no devices never hides the devices that the concrete
/// placement tells.
///
/// # Transposition
///
/// Transposition moves the cotangent back to where the input lived whenever that is known, and always into the memory
/// of the input, which its cotangent type records. A known source is the destination of the transposed placement for
/// every form of the target, the in-place form included, as JAX returns the cotangent of
/// `jax.device_put(x, jax.memory.Space.Host)` to the devices of `x`. A placement source gets the cotangent dual of its
/// sharding. A device source is accepted only for an input that its type tracks over a mesh of one device, so that
/// device and the logical mesh of the input name where the input lived, and the cotangent returns there with the
/// sharding of the input cotangent, whatever mesh the output cotangent lives on, as JAX returns the cotangent of
/// `jax.device_put(x, target, src=device)` to that device. A device source of an input without a tracked sharding stays
/// a device. An in-place source names no devices, so the cotangent returns in place to the memory of the input. Only
/// where the source is not known, or where the value is inside a manual region, does the transposed placement of an
/// in-place target leave the cotangent where it is and return it to the memory of the input. A transposed placement
/// whose target names no mesh (i.e., that of an in-place source, or of a placement target without a source) leaves the
/// cotangent on the logical mesh where it lives, so it is rejected when that mesh does not have the topology of the
/// mesh of the input, which only a source that names devices can return the cotangent to. For the same reason, the
/// transposed placement of a gathering device target needs a placement source.
///
/// Every form may name a [`Memory`] space and a storage [`Layout`]. An omitted memory of a placement or in-place target
/// keeps the memory of the value, and an omitted memory of a device target selects the default memory of the device
/// (refer to [`Self::target_memory`]). An omitted layout of a target keeps the layout of the value while it still
/// describes each shard of the output (i.e., the buffer that one device holds), and the output declares no layout
/// otherwise (e.g., the strides of a column shard of a matrix that a device target gathers into one whole shard), so
/// that backends store it in their default layout. An omitted memory or layout of a source states nothing about it. A
/// target layout must address the shape of the input. The rules of other operations that move values between memories
/// (e.g., the dynamic padding of arrays, which moves an offset into the memory of the padded array) keep using
/// [`TransferToMemoryOperation`](crate::TransferToMemoryOperation), which changes only the memory of its inputs,
/// without the copy policy and the source of a placement.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum PlacementRequest {
    /// Resolved [`Placement`] (i.e., a device mesh, a sharding over it, and optionally a memory and a layout).
    Resolved(Placement),

    /// Single [`Device`], optionally with a [`Memory`] space and a storage [`Layout`] for the output.
    Device {
        /// [`Device`] that the value is moved to.
        device: Device,

        /// Optional [`Memory`] space of the output. When it is [`None`], a target selects the default memory of the
        /// device (i.e., [`Memory::Device`]) and a source states nothing about the memory of the value.
        memory: Option<Memory>,

        /// Optional storage [`Layout`] of the output. When it is [`None`], a target keeps the layout of the value only
        /// while it still describes the one shard of the output, and a source states nothing about the layout.
        layout: Option<Layout>,
    },

    /// Storage change that leaves the value on its devices with its sharding (i.e., in place).
    InPlace {
        /// Optional [`Memory`] space of the output. When it is [`None`], the value keeps its memory.
        memory: Option<Memory>,

        /// Optional storage [`Layout`] of the output. When it is [`None`], the value keeps its layout.
        layout: Option<Layout>,
    },
}

impl PlacementRequest {
    /// Returns this request without a storage [`Layout`], keeping everything else. An in-place request that named only
    /// a layout therefore names nothing afterwards.
    #[inline]
    pub fn without_layout(self) -> Self {
        match self {
            Self::Resolved(placement) => Self::Resolved(placement.without_layout()),
            Self::Device { device, memory, .. } => Self::Device { device, memory, layout: None },
            Self::InPlace { memory, .. } => Self::InPlace { memory, layout: None },
        }
    }

    /// Returns the [`Placement`] of this request, or [`None`] if it is a [`PlacementRequest::Device`] or a
    /// [`PlacementRequest::InPlace`].
    #[inline]
    pub fn placement(&self) -> Option<&Placement> {
        match self {
            Self::Resolved(placement) => Some(placement),
            Self::Device { .. } | Self::InPlace { .. } => None,
        }
    }

    /// Returns the [`Memory`] space that this request names, or [`None`] if it names none. A source that names a memory
    /// states that the value is expected in it, and [`Self::target_memory`] returns the memory that a target selects.
    #[inline]
    pub fn memory(&self) -> Option<Memory> {
        match self {
            Self::Resolved(placement) => placement.memory(),
            Self::Device { memory, .. } | Self::InPlace { memory, .. } => *memory,
        }
    }

    // TODO(eaplatanios): I don't fully understand the difference between this and `PlacementRequest::memory`.
    /// Returns the [`Memory`] space that this request selects as the target of a placement, or [`None`] if the output
    /// keeps the memory of the value. A [`PlacementRequest::Device`] without a memory selects the default memory of its
    /// device (i.e., [`Memory::Device`]), as JAX's eager `device_put` does for a bare device, so that it moves a value
    /// out of any other memory (e.g., pinned host memory). Every other form selects the memory that it names. This is
    /// the one owner of that rule, which the type rule of [`PlaceOperation`] and the backends that execute a placement
    /// share.
    #[inline]
    pub fn target_memory(&self) -> Option<Memory> {
        match self {
            Self::Device { memory, .. } => Some(memory.unwrap_or(Memory::Device)),
            Self::Resolved(_) | Self::InPlace { .. } => self.memory(),
        }
    }

    /// Returns the storage [`Layout`] that this request selects, or [`None`] if it keeps the layout of the value.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        match self {
            Self::Resolved(placement) => placement.layout(),
            Self::Device { layout, .. } | Self::InPlace { layout, .. } => layout.as_ref(),
        }
    }

    /// Returns whether this request moves a value to other devices or redistributes it, which every form except
    /// [`PlacementRequest::InPlace`] can do.
    #[inline]
    pub fn moves(&self) -> bool {
        !matches!(self, Self::InPlace { .. })
    }

    /// Returns the source that a placement records for a value that concretely lives on `placement`, if anywhere
    /// (refer to [`Placed`]), given the `source` that its caller states, if any. Without a stated source, it is the
    /// concrete placement of the value. An in-place source names only the storage in which the value is expected, so
    /// it is completed with the devices and the sharding of the concrete placement, and becomes that placement with
    /// the memory and the layout that the source names. Every other source names devices itself, so it is returned as
    /// it is. A source that says nothing about devices therefore never makes the transpose of a placement forget the
    /// devices to which the cotangent returns, which the concrete placement alone tells. The rules of
    /// [`PlaceOperation`] that place a value with a concrete placement (i.e., interpretation, batching, and
    /// differentiation) record their sources through this function, and only after they validate the placement as it
    /// is given (refer to [`PlaceOperation::infer_output_type`]), because a completed source can hide a request that
    /// the type rule rejects (e.g., an in-place source that names nothing becomes a valid placement).
    pub fn resolve_source(source: Option<Self>, placement: Option<Placement>) -> Option<Self> {
        match (source, placement) {
            (Some(Self::InPlace { memory, layout }), Some(placement)) => {
                let placement = match memory {
                    Some(memory) => placement.with_memory(memory),
                    None => placement,
                };
                Some(Self::Resolved(match layout {
                    Some(layout) => placement.with_layout(layout),
                    None => placement,
                }))
            }
            (Some(source), _) => Some(source),
            (None, placement) => placement.map(Self::Resolved),
        }
    }
}

impl Display for PlacementRequest {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // A device is rendered by its globally unique ID, as the devices of a placement are.
        match self {
            Self::Resolved(placement) => write!(formatter, "{placement}"),
            Self::Device { device, memory, layout } => {
                write!(formatter, "device({}", device.id())?;
                if let Some(memory) = memory {
                    write!(formatter, ", memory={memory}")?;
                }
                if let Some(layout) = layout {
                    write!(formatter, ", layout={layout}")?;
                }
                write!(formatter, ")")
            }
            Self::InPlace { memory, layout } => {
                write!(formatter, "in_place(")?;
                if let Some(memory) = memory {
                    write!(formatter, "memory={memory}")?;
                }
                if let Some(layout) = layout {
                    let separator = if memory.is_some() { ", " } else { "" };
                    write!(formatter, "{separator}layout={layout}")?;
                }
                write!(formatter, ")")
            }
        }
    }
}

impl From<Placement> for PlacementRequest {
    #[inline]
    fn from(value: Placement) -> Self {
        Self::Resolved(value)
    }
}

impl From<Device> for PlacementRequest {
    #[inline]
    fn from(value: Device) -> Self {
        Self::Device { device: value, memory: None, layout: None }
    }
}

impl From<Memory> for PlacementRequest {
    #[inline]
    fn from(value: Memory) -> Self {
        Self::InPlace { memory: Some(value), layout: None }
    }
}

impl From<Placement> for Option<PlacementRequest> {
    #[inline]
    fn from(value: Placement) -> Self {
        Some(PlacementRequest::from(value))
    }
}

impl From<Device> for Option<PlacementRequest> {
    #[inline]
    fn from(value: Device) -> Self {
        Some(PlacementRequest::from(value))
    }
}

impl From<Memory> for Option<PlacementRequest> {
    #[inline]
    fn from(value: Memory) -> Self {
        Some(PlacementRequest::from(value))
    }
}

/// Target or source leaf accepted by [`Place::place`]. A leaf can be a [`Placement`], a [`PlacementRequest`],
/// a [`Device`], a [`Memory`], or an [`Option`] of any of them. A [`None`] leaf omits its target or source.
/// The trait is sealed to these types.
pub trait PlacementRequestParameter: Clone + Parameter + sealed::Sealed {
    /// Returns this element as the optional target or source of a [`PlaceOperation`].
    fn into_optional_placement_request(self) -> Option<PlacementRequest>;
}

impl<R: Clone + Parameter + sealed::Sealed + Into<PlacementRequest>> PlacementRequestParameter for R {
    #[inline]
    fn into_optional_placement_request(self) -> Option<PlacementRequest> {
        Some(self.into())
    }
}

impl<R: Clone + Parameter + sealed::Sealed + Into<PlacementRequest>> PlacementRequestParameter for Option<R> {
    #[inline]
    fn into_optional_placement_request(self) -> Option<PlacementRequest> {
        self.map(Into::into)
    }
}

/// Seals [`PlacementRequestParameter`] to placement leaves and their optional forms.
mod sealed {
    /// Super-trait of [`PlacementRequestParameter`](super::PlacementRequestParameter) implemented in this module.
    pub trait Sealed {}

    impl Sealed for super::Placement {}

    impl Sealed for super::PlacementRequest {}

    impl Sealed for super::Device {}

    impl Sealed for super::Memory {}

    impl<R: Sealed + Into<super::PlacementRequest>> Sealed for Option<R> {}
}

/// Canonical operation name for [`PlaceOperation`].
pub const PLACE_OPERATION_NAME: &str = "place";

/// [`Operation`] that moves its input to a target [`Placement`], possibly on the devices of another [`DeviceMesh`], to
/// a single [`Device`], or to another storage where it lives (refer to [`PlacementRequest`] for the contract of each
/// form). This operation is the Ryft analogue of JAX's
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html). The array's elements, shape,
/// and data type are unchanged. For a [`Placement`] target, type inference returns the input type with its sharding
/// replaced by the target sharding without its [`MeshAxisType::Auto`] mesh axes (which array types cannot record,
/// refer to [`Sharding::without_auto_axes`]), its memory replaced by the target memory when the target names one, and
/// its layout replaced by the target layout when the target names one. Without a target layout, the output keeps the
/// layout of the input only while it still describes each shard of the output, and declares none otherwise.
///
/// A [`PlacementRequest::Device`] or [`PlacementRequest::InPlace`] target names no mesh, so type inference follows the
/// contract that [`PlacementRequest`] states for it (i.e., a device target keeps the sharding that the input type
/// tracks over a mesh of one device and gathers an input that is tracked over more devices, without reduction state
/// in both cases, and an in-place target keeps the sharding exactly). It replaces the memory of the input type with
/// the memory that the target selects (refer to [`PlacementRequest::target_memory`], which is the default memory of
/// the device for a device target without one), and its layout as for a [`Placement`] target. The rank, reduction
/// state, and split checks of a [`Placement`] target below apply to neither form. Backends resolve both against the
/// value that the placement applies to (e.g., a concrete array, or the device mesh that a program is compiled for,
/// which must consist of the device of a device target alone). These forms serve the placement of values whose devices
/// are not known yet (e.g., while a function is traced before it is compiled).
///
/// The target is optional. Without a target, the value stays where it is (i.e., its devices, sharding, memory, and
/// layout are unchanged), and type inference returns the input type. Such an operation still records its source, which
/// transposition uses to move the cotangent back to it, and its copy policy, so that it can force a copy of a concrete
/// value. An operation without a target, without a source, and whose output may share storage with its input is
/// therefore an identity.
///
/// Unlike a [`ReshardOperation`](crate::ReshardOperation), which redistributes a value over the mesh that it already
/// lives on, a placement may move the value to another mesh, so the target mesh need not define the axes of the input's
/// mesh. A placement still changes only where a value lives and never its reduction state, so the target sharding must
/// restate the unreduced and reduced axes of the input's tracked sharding (and its unreduced kind) exactly, and the
/// target mesh must define each of those axes with the same size and type as the input's mesh (refer to
/// [`Self::infer_output_type`], through which backends also apply every check of this rule but the even split below
/// to the concrete values that they place outside a program). Every shard of a partitioned dimension of the output
/// has the same extent, so for each dimension that the target sharding partitions, the number of shards along it (i.e.,
/// the product of the sizes of the mesh axes that shard it, [`MeshAxisType::Auto`] axes included) must divide its
/// static extent, on every path that this operation takes (i.e., its eager execution through [`Place`], the replay of
/// a staged program, its transformations, and compilation), as JAX's `device_put` requires eagerly and inside `jit`.
/// Unlike JAX, a backend may still split a concrete value unevenly when it places that value outside this operation. A
/// dynamic extent is only known when a backend places a concrete value, so type inference accepts any split of it and
/// backends check it when they place the value.
///
/// A value inside a manual region lives on the devices of that region, so it cannot be moved, and nothing can be moved
/// into such a region either. Types show such values through their variation along [`MeshAxisType::Manual`] mesh axes
/// (refer to [`Sharding::varying_manual_axes`]) or their reduction state over them, so neither the target sharding nor
/// the source sharding or the tracked sharding of an input that is moved to a target may have either. An operation
/// without a target or with an in-place target moves nothing, so it accepts such an input together with a source that
/// describes it (e.g., the concrete placement of the input, which interpretation, batching, and differentiation record
/// as the source), and its transpose leaves the cotangent where it is as well. A source that describes such a value is
/// still rejected for an input outside any manual region. A global array outside any manual region can still be tiled
/// over a manual axis of its mesh, and it is moved like any other array. Backends separately reject placements onto a
/// target that moves values inside active manual regions when they lower a program. A target layout must address the
/// shape of the input (refer to [`ArrayType::validate_layout`]).
///
/// The optional source is informative about devices. It is a [`PlacementRequest`] too, which is a [`Placement`], a
/// single [`Device`] whose mesh is not known (e.g., a device that a caller names for a value whose devices are not
/// known while it is traced), or an in-place request, which names only the storage in which the input is expected
/// (refer to [`PlacementRequest`]). It states where the input is expected to live when the operation is created, and
/// transposition uses it to move the cotangent back there. Type inference checks it against what the input type tracks.
/// The sharding of a [`Placement`] source must have the rank of the input, must not describe a value inside a manual
/// region, and must restate the reduction state of the input exactly (i.e., its unreduced axes, reduced axes, and
/// unreduced kind, which are empty and a sum for an input without a tracked sharding) over a mesh that defines those
/// axes like the input's mesh. When the input type tracks a sharding, the source sharding without its auto axes must
/// also equal it. A device source is rejected for an input that its type tracks over a mesh of more than one device.
/// These checks guarantee that the cotangent that transposition moves back to the source devices has the cotangent
/// type of the input (up to a placement-only broadcast for an input without a tracked sharding). The memory and the
/// layout that a source of any form names are checked against the input type as [`PlacementRequest`] states. Backends
/// that execute the operation on a concrete value move the value from wherever it actually lives, without comparing
/// the devices of that placement with the source on any path, as JAX never compares them either, so a source that
/// names other devices sends the cotangent to the devices that it names.
///
/// The operation also carries a copy policy (refer to [`Self::with_may_alias`]), which is the Ryft analogue of the
/// `copy_semantics` of JAX's `device_put` primitive. By default, the output may share storage with the input (e.g.,
/// a placement that leaves a concrete value where it already lives can return that value), and with `may_alias` set to
/// `false`, the output has storage of its own, which no other value shares. The policy does not affect type inference.
/// It only matters for concrete values, whose storage other handles can observe. A compiled program has no observable
/// buffer identity, because its outputs are buffers that its execution allocates (or donated inputs, which no other
/// handle can observe), so backends ignore the policy when they lower the operation.
///
/// Interpretation moves the value through the [`PlaceDispatch`] primitive of [`Place`] with the operation's target,
/// source, and copy policy. Batching first validates the target and the source against the type of a batch item with
/// the type rule, which therefore rejects on the batched path every request that it rejects for a batch item. It then
/// keeps the copy policy and extends the target and source shardings with an entry for the batch dimension. A target
/// or source that records how its batch is distributed (refer to [`Placement::batch_shardings`]), such as the placement
/// that a batch item reports (refer to [`Placed`]), gets that distribution back. Otherwise, an explicit target
/// replicates the batch dimension, so that it places every batch item where it says, and a source gets the placement
/// that the packed input type records for its batch dimension. This differs from JAX, whose batching rule applies an
/// unchanged target to the packed value positionally, so that a target that splits the first dimension of a batch item
/// splits the batch dimension instead when the batch is mapped at position `0`. An omitted target stays omitted and
/// leaves the packed value where it is, and a device or in-place target and a device source apply to the packed value
/// unchanged (without the layout of a source), because they record no sharding. A target layout does not describe the
/// batch dimension, so it is rejected for a mapped input. An unmapped input (i.e., one that is the same for every batch
/// item) whose target records a split batch is first broadcast along a new batch dimension, which is replicated over
/// the devices where the input lives, so that the placement splits the packed result as recorded, and otherwise it
/// stays unmapped. Differentiation places the tangent with the same target, source, and copy policy as the primal but
/// without the layouts of the target and source, which need not be valid for the element type of the tangent, so an
/// in-place target that names only a layout leaves the tangent where it is without a target. When the operation has no
/// source, both use the concrete placement of the primal as their source, if it has one, and an in-place source is
/// completed with that placement (refer to [`PlacementRequest::resolve_source`]) after the request is validated as it
/// is given, as interpretation and batching also do. Transposition moves the cotangent back to where the input lived
/// whenever the source says where that is, for every form of the target, and rejects a cotangent that it cannot return
/// to the mesh of the input (refer to [`PlacementRequest`]), and it always forces a copy, as JAX's `device_put`
/// transpose does. Like that transpose, which swaps the devices and the sources of `device_put`, it swaps the two ends
/// of the placement, so that transposing twice moves the value to the forward target again, except that an in-place
/// target names no devices and so becomes no source. A device source of an input that its type tracks over a mesh of
/// one device becomes a placement over the logical mesh of the input on that device.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaceOperation {
    /// Optional source [`PlacementRequest`] where the input is expected to live, which is informative only.
    source: Option<PlacementRequest>,

    /// Optional [`PlacementRequest`] that the input is moved to. When it is [`None`], the input stays where it is.
    target: Option<PlacementRequest>,

    /// Boolean indicating whether the output may share storage with the input. When it is `false`, the output must
    /// have storage of its own.
    may_alias: bool,
}

impl PlaceOperation {
    /// Creates a new [`PlaceOperation`] that moves its input to `target` and records `source` as the expected placement
    /// of that input. Each of them is a [`Placement`], a [`PlacementRequest`], a [`Device`], a [`Memory`], or an
    /// optional [`PlacementRequest`]. To pass an `Option<Placement>`, use `placement.map(PlacementRequest::from)`.
    /// A [`None`] target leaves the input where it is, and a [`None`] source means that the expected placement is
    /// unknown. Type inference validates both requests (refer to [`PlacementRequest`]). The output may share storage
    /// with its input (refer to [`Self::with_may_alias`]).
    #[inline]
    pub fn new(target: impl Into<Option<PlacementRequest>>, source: impl Into<Option<PlacementRequest>>) -> Self {
        Self { source: source.into(), target: target.into(), may_alias: true }
    }

    /// Returns this [`PlaceOperation`] with its copy policy replaced by `may_alias`. With `true`, which is the default,
    /// the output may share storage with the input (e.g., when the placement leaves the input where it already lives),
    /// and with `false`, the output has storage of its own, which no other value shares. This is the analogue of the
    /// `copy_semantics` of JAX's `device_put` primitive, where `false` corresponds to a forced copy.
    ///
    /// The copy is made every time that the placement is executed. A placement that forces a copy reports so through
    /// [`Operation::requires_fresh_storage`], and partial evaluation refuses to fold it, or any application whose
    /// executed regions contain it, where it folds applications and the context does not allow copies to fold.
    /// When a program is specialized on a known input of such a placement (refer to
    /// [`Program::partially_evaluate`](crate::Program::partially_evaluate)), the placement therefore stays in
    /// the specialized program, also inside calls, composites, and rematerialized regions, so that every execution of
    /// that program copies, as JAX does when it stages a `device_put` that forces a copy over a constant. A single
    /// evaluation of a known placement (e.g., the primal evaluation of linearization) makes the copy once.
    #[inline]
    pub fn with_may_alias(mut self, may_alias: bool) -> Self {
        self.may_alias = may_alias;
        self
    }

    /// Returns the source [`PlacementRequest`] where the input is expected to live, or [`None`] if it is not known.
    #[inline]
    pub fn source(&self) -> Option<&PlacementRequest> {
        self.source.as_ref()
    }

    /// Returns the [`PlacementRequest`] that the input is moved to, or [`None`] if the input stays where it is.
    #[inline]
    pub fn target(&self) -> Option<&PlacementRequest> {
        self.target.as_ref()
    }

    /// Returns whether the output may share storage with the input (refer to [`Self::with_may_alias`]).
    #[inline]
    pub fn may_alias(&self) -> bool {
        self.may_alias
    }

    /// Returns whether the transposition of this placement, for an input of type `input`, leaves the cotangent on the
    /// devices where it lives and changes at most its memory (refer to [`PlacementRequest`]). That is the case when
    /// this placement moves nothing (i.e., it has no target or an in-place target) and either its source is not known
    /// or its input lives inside a manual region, whose cotangent lives in the same region. Otherwise, the transposed
    /// placement moves the cotangent to the source or keeps it on the mesh of the forward target. This is the one owner
    /// of that rule, which transposition and backends that reject moves of values inside manual regions share (e.g.,
    /// the lowering of a source that names devices inside a manual region, which only matters when transposition
    /// moves the cotangent to it).
    #[inline]
    pub fn transposes_in_place(&self, input: &ArrayType) -> bool {
        !self.target.as_ref().is_some_and(PlacementRequest::moves)
            && (self.source.is_none() || input.sharding().is_some_and(is_manual_region_sharding))
    }

    /// Validates a placement of a value of type `input` onto `target`, which states that the value lives
    /// on `source`, and returns the type of the placed value. This is the type rule of [`PlaceOperation`] (refer to
    /// [`Operation::infer_output_types`]) except for one check, which is that a [`Placement`] target splits each static
    /// extent evenly, and the type rule makes that check after this function. Backends call this function for the
    /// placement of a concrete value outside a program (e.g., a frontend that places an array, or that uploads a host
    /// value to a destination, which it validates as the placement of a value with the type of that host value), so
    /// that such a placement checks everything that the type rule checks, in the same order and with the same errors,
    /// and splits a dimension unevenly only where it places a concrete value. The interpretation, batching, and
    /// differentiation rules of [`PlaceOperation`] also call it to validate a placement as it is given before they
    /// complete its source with the placement of a concrete value (refer to [`PlacementRequest::resolve_source`]),
    /// so that the completed source never hides a request that the type rule rejects.
    ///
    /// `concrete_sharding` is the sharding over every mesh axis of the concrete value that is placed, when it is known
    /// (i.e., the sharding of its [`Placed`] placement, such as the runtime sharding of a backend array), and [`None`]
    /// for a staged value, whose type is all that is known about it. The type of a value omits its placement along
    /// [`MeshAxisType::Auto`] mesh axes, including its reduction state over them, while the concrete sharding records
    /// it, so the checks that the concrete sharding can decide more precisely use it, which are the reduction state of
    /// the input (e.g., a pending reduction over an auto axis, which a target must restate and which a device target
    /// cannot complete) and the shards that a layout of the input or of an in-place output must address. Every other
    /// check, and the output type, uses the type of the input alone, so the output type never depends on it. The
    /// placement of the input along auto axes is never compared with a source, whose devices and auto axes are
    /// informative (e.g., a cotangent can be split along auto axes otherwise than the source of the transposed
    /// placement says). Without a concrete sharding, a source restates the reduction state of the input over the axes
    /// that its type records, and its reduction state over auto axes, which the type cannot describe, is not checked.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] for every request that [`Operation::infer_output_types`] rejects, except for a target
    /// that splits a static extent unevenly, and for every request that the concrete sharding shows to change the
    /// reduction state of the value or to have a layout that does not address its shards.
    pub fn infer_output_type(
        input: &ArrayType,
        concrete_sharding: Option<&Sharding>,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
    ) -> Result<ArrayType, TypeError> {
        // A device or in-place request records no sharding, so only the shardings of the resolved forms are validated
        // below, and the storage that a source of any form names is validated afterwards. An in-place request that
        // names neither a memory nor a layout says nothing, which an absent target or source already expresses.
        if let Some(PlacementRequest::InPlace { memory: None, layout: None }) = source {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` in-place source must name a memory or a layout",
            )));
        }

        if let Some(PlacementRequest::InPlace { memory: None, layout: None }) = target {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` in-place target must name a memory or a layout",
            )));
        }

        let destination = target;
        let target = destination.and_then(PlacementRequest::placement).map(Placement::sharding);
        if let Some(target) = target
            && input.rank() != target.rank()
        {
            return Err(TypeError::invalid(format!(
                "`{}` target sharding rank ({}) does not match the input rank ({})",
                PLACE_OPERATION_NAME,
                target.rank(),
                input.rank(),
            )));
        }

        let source_request = source;
        let source = source_request.and_then(PlacementRequest::placement).map(Placement::sharding);
        if let Some(source) = source
            && source.rank() != input.rank()
        {
            return Err(TypeError::invalid(format!(
                "`{}` source sharding rank ({}) does not match the input rank ({})",
                PLACE_OPERATION_NAME,
                source.rank(),
                input.rank(),
            )));
        }

        // Values of a manual region live on the devices of that region, so they cannot be moved anywhere else, and
        // nothing can be moved into such a region either. Their types show it through their variation along manual
        // axes or their reduction state over them (i.e., per device values or pending reductions across the devices of
        // the region), whereas a global array can be tiled over a manual axis outside any manual region and is moved
        // like any other array. Backends reject placements inside active manual regions separately. A placement without
        // a target leaves its input where it is, so it moves nothing and accepts such an input, together with a source
        // that describes it (e.g., the concrete placement of the input, which differentiation records as the source).
        // So does an in-place target, which changes only the storage of the value on the devices where it lives. A
        // device target moves the value onto one device, so it rejects such an input like a placement target does.
        let is_manual_input = input.sharding().is_some_and(is_manual_region_sharding);
        let moves = destination.is_some_and(PlacementRequest::moves);
        if (moves && is_manual_input)
            || target.is_some_and(is_manual_region_sharding)
            || (source.is_some_and(is_manual_region_sharding) && (moves || !is_manual_input))
        {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes \
                 or carry reduction state over them",
            )));
        }

        // The sharding that is known for the input, which describes every mesh axis exactly when it is the sharding of
        // a concrete value.
        let (known_sharding, is_known_complete) = match concrete_sharding {
            Some(sharding) => (Some(sharding), true),
            None => (input.sharding(), false),
        };

        Self::validate_reduction_state(input, known_sharding, is_known_complete, destination, source_request)?;

        // A given source must also describe where the input lives as far as its type can tell, which excludes the
        // placement over auto axes that types do not record. A device source states that the input lives on that one
        // device, which an input that its type tracks over a mesh of more devices does not.
        if let Some(source) = source
            && let Some(input_sharding) = input.sharding()
        {
            let source = source.without_auto_axes();
            if input_sharding != &source {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{source}` does not match the input type `{input}`",
                )));
            }
        }

        if let Some(PlacementRequest::Device { .. }) = source_request
            && let Some(input_sharding) = input.sharding()
            && input_sharding.mesh().device_count() > 1
        {
            return Err(TypeError::invalid(format!(
                "`{}` device source names one device, but the input type `{}` is tracked over {} devices",
                PLACE_OPERATION_NAME,
                input,
                input_sharding.mesh().device_count(),
            )));
        }

        // The storage of the input is recorded in its type, so the memory and the layout that a source of any form
        // names are checked against it, whichever path stages or executes the placement. A layout must address the
        // shape of the input, and it must be the layout of the input type when that type declares one. This is the only
        // check of a source, so no path compares a source with the physical layouts of concrete buffers, which a type
        // that declares no layout does not describe and which no staged, compiled, or transformed path could inspect.
        if let Some(source) = source_request {
            if let Some(memory) = source.memory()
                && memory != input.memory()
            {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` source memory `{memory}` does not match the input type `{input}`",
                )));
            }

            if let Some(layout) = source.layout() {
                if input.layout().is_some_and(|input_layout| input_layout != layout) {
                    return Err(TypeError::invalid(format!(
                        "`{PLACE_OPERATION_NAME}` source layout `{layout}` does not match the input type `{input}`",
                    )));
                }

                // The sharding of a placement source describes every mesh axis, and so does the concrete sharding of
                // the input, while the sharding that the input type tracks omits the auto axes along which the input
                // may also be split.
                let (sharding, is_complete) = match source.placement() {
                    Some(placement) => (Some(placement.sharding()), true),
                    None => (known_sharding, is_known_complete),
                };

                validate_shard_layout(input, layout, sharding, is_complete).map_err(|error| {
                    TypeError::invalid(format!("`{PLACE_OPERATION_NAME}` source layout is invalid: {error}"))
                })?;
            }
        }

        // Without a target, the value stays where it is, so it keeps its type. A device target names no mesh, so the
        // output keeps the sharding that the input type tracks over a mesh of one device without the reduction state
        // that one device completes, and it is replicated over the canonical mesh of one device for an input that the
        // type tracks over more devices. An in-place target keeps the sharding exactly, and only a placement target
        // replaces it.
        let Some(destination) = destination else {
            return Ok(input.clone());
        };
        let mut output = input.clone();

        if let (PlacementRequest::Device { .. }, Some(input_sharding)) = (destination, input.sharding()) {
            // A device target leaves a value that lives wholly on that device, so the output has no reduction state,
            // which one device completes (refer to `Self::validate_reduction_state`, which rejected the reduction state
            // that it does not complete). A value on a mesh of more devices is gathered onto the device, so it can no
            // longer be distributed over that mesh, and it is replicated over the canonical mesh of one device instead,
            // which is also where backends place it when the placement executes.
            output = if input_sharding.mesh().device_count() > 1 {
                output.with_sharding(Sharding::replicated(LogicalMesh::single_device(), input.rank()))?
            } else {
                input.without_reduction_axes()
            };
        }

        if let Some(target) = target {
            output = output.with_sharding(target.without_auto_axes()).map_err(TypeError::from)?;
        }

        if let Some(memory) = destination.target_memory() {
            output = output.with_memory(memory);
        }

        // A layout describes each shard of a value (i.e., the buffer that one device holds), so a target layout must
        // address each shard of the output as far as the output type determines its shards, so that a malformed
        // request fails here rather than only when a backend materializes or lowers the output. Without a target
        // layout, the output keeps the layout of the input only while it still describes the shards of the output, and
        // it declares no layout otherwise (e.g., the strides of a column shard of a matrix that a device target gathers
        // into one whole shard), in which case backends store it in their default layout. The sharding of a placement
        // target describes every mesh axis, and the output of a device target lives on one device, while the output of
        // an in-place target is distributed like its input, which may also be split along the auto axes that its type
        // omits unless its concrete sharding is known.
        let (sharding, is_complete) = match destination {
            PlacementRequest::Resolved(placement) => (Some(placement.sharding()), true),
            PlacementRequest::Device { .. } => (output.sharding(), true),
            PlacementRequest::InPlace { .. } => (known_sharding, is_known_complete),
        };

        match (destination.layout(), output.layout()) {
            (Some(layout), _) => {
                validate_shard_layout(&output, layout, sharding, is_complete).map_err(|error| {
                    TypeError::invalid(format!("`{PLACE_OPERATION_NAME}` target layout is invalid: {error}"))
                })?;
                output = output.with_layout(layout.clone());
            }
            (None, Some(layout)) if validate_shard_layout(&output, layout, sharding, is_complete).is_err() => {
                output = output.with_layout(None::<Layout>);
            }
            (None, _) => {}
        }

        Ok(output)
    }

    /// Validates that a placement of a value of type `input` onto `target`, which states that the value lives
    /// on `source`, leaves the reduction state of the value unchanged. This is the part of the type rule of
    /// [`PlaceOperation`] that owns reduction state (refer to [`Self::infer_output_type`], through which backends
    /// also apply it to the placement of a concrete value outside a program), so that no path changes reduction state.
    ///
    /// Reduction state records pending (i.e., unreduced) or completed (i.e., reduced) cross-device reductions, and a
    /// placement changes where a value lives but never that state. The sharding of a [`Placement`] target therefore
    /// restates the unreduced axes, the reduced axes, and the unreduced kind of `input` exactly, and so does the
    /// sharding of a [`Placement`] source, which describes the input. The state is restated by axis name, so the target
    /// and source meshes must also define every axis that carries it identically (e.g., an explicit unreduced axis must
    /// not become an auto axis, over which array types record no reduction state, so that the output would silently
    /// lose its pending reduction, nor an axis of another size). A [`PlacementRequest::Device`] target leaves a value
    /// that lives wholly on one device, which completes a reduced axis of any size and an unreduced axis of size one,
    /// whereas an unreduced axis of a larger size describes contributions on more devices than one, which a placement
    /// does not combine, so such an input must be reduced first. A [`PlacementRequest::InPlace`] target keeps the
    /// sharding of `input` exactly, so it restates its reduction state by construction. An input without a tracked
    /// sharding has no reduction state.
    ///
    /// The reduction state of `input` is that of `input_sharding`, which is the concrete sharding of the value
    /// when it is known, and which then also records the reduction state over auto mesh axes that the type omits,
    /// as `is_complete` states, and the sharding that the type of `input` tracks otherwise (refer to
    /// [`Self::infer_output_type`]). A source is compared over the axes that `input_sharding` records, so without a
    /// concrete sharding its reduction state over auto axes is not checked, as its split along them is not.
    ///
    /// # Errors
    ///
    /// Returns a [`TypeError`] for a target or source mesh that does not define an axis that carries reduction state of
    /// `input` identically, for a source or a placement target whose sharding does not restate the reduction state of
    /// `input`, and for a device target of an input that is unreduced over a mesh axis larger than one.
    fn validate_reduction_state(
        input: &ArrayType,
        input_sharding: Option<&Sharding>,
        is_complete: bool,
        target: Option<&PlacementRequest>,
        source: Option<&PlacementRequest>,
    ) -> Result<(), TypeError> {
        let target_sharding = target.and_then(PlacementRequest::placement).map(Placement::sharding);
        let source_sharding = source.and_then(PlacementRequest::placement).map(Placement::sharding);
        if let Some(input_sharding) = input_sharding {
            let reduction_axes = || input_sharding.unreduced_axes().iter().chain(input_sharding.reduced_axes());
            for mesh in source_sharding.into_iter().chain(target_sharding).map(Sharding::mesh) {
                validate_preserved_mesh_axes(PLACE_OPERATION_NAME, input_sharding.mesh(), reduction_axes(), mesh)?;
            }
        }

        let (unreduced_axes, reduced_axes) = match input_sharding {
            Some(sharding) => (sharding.unreduced_axes().clone(), sharding.reduced_axes().clone()),
            None => Default::default(),
        };
        let unreduced_kind = input_sharding.map_or(input.unreduced_kind(), Sharding::unreduced_kind);

        let has_input_reduction_state = |sharding: &Sharding| {
            sharding.unreduced_axes() == &unreduced_axes
                && sharding.reduced_axes() == &reduced_axes
                && sharding.unreduced_kind() == unreduced_kind
        };

        if let Some(source) = source_sharding
            && !(if is_complete {
                has_input_reduction_state(source)
            } else {
                has_input_reduction_state(&source.without_auto_axes())
            })
        {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding `{source}` does not match the reduction state of the input \
                 type `{input}`",
            )));
        }

        if target_sharding.is_some_and(|target| !has_input_reduction_state(target)) {
            return Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding \
                 collective) instead",
            )));
        }

        // Reduction state over manual axes describes a value inside a manual region, which the type rule rejects
        // for a device target before it validates reduction state, whatever the sizes of those axes.
        if let (Some(PlacementRequest::Device { .. }), Some(input_sharding)) = (target, input_sharding) {
            let larger_axes = input_sharding
                .unreduced_axes()
                .iter()
                .filter(|axis| input_sharding.mesh().axis_size(axis).is_some_and(|size| size > 1))
                .map(|axis| format!("`{axis}`"))
                .collect::<Vec<_>>();
            if !larger_axes.is_empty() {
                return Err(TypeError::invalid(format!(
                    "`{}` cannot move a value with reduction state over mesh axes larger than one (i.e., {}) \
                     onto a single device; use a reduction (or the corresponding collective) first",
                    PLACE_OPERATION_NAME,
                    larger_axes.join(", "),
                )));
            }
        }

        Ok(())
    }

    /// Validates that a placement of a value of type `input` onto `target` splits each static extent of the value
    /// evenly, which is the one check of the type rule of [`PlaceOperation`] that [`Self::infer_output_type`] does not
    /// make. Every shard of a partitioned dimension of a value has the same extent, so for each dimension that the
    /// target sharding partitions, the number of shards along it (i.e., the product of the sizes of the mesh axes that
    /// shard it, which includes auto axes, which the output type does not record but along which the value is still
    /// split) must divide its static extent. A dynamic extent is only known when a backend places a concrete value,
    /// which is when it is checked. Only a [`Placement`] target partitions dimensions.
    fn validate_even_splits(input: &ArrayType, target: Option<&PlacementRequest>) -> Result<(), TypeError> {
        let Some(target) = target.and_then(PlacementRequest::placement).map(Placement::sharding) else {
            return Ok(());
        };
        for (dimension, (extent, sharding)) in input.shape().dimensions().iter().zip(target.dimensions()).enumerate() {
            let (Dimension::Static(extent), ShardingDimension::Sharded(axis_names)) = (extent, sharding) else {
                continue;
            };

            // The product is over distinct axes of one mesh, so it never exceeds the device count of that mesh,
            // which fits `usize`.
            let shard_count = axis_names.iter().filter_map(|name| target.mesh().axis_size(name)).product::<usize>();
            if extent % shard_count != 0 {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #{dimension} with extent {extent} into \
                     {shard_count} shards, which do not divide it evenly",
                )));
            }
        }
        Ok(())
    }
}

impl Display for PlaceOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Hash for PlaceOperation {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // A source or a target that is a placement, or that is absent, hashes exactly like an `Option<Placement>`,
        // so that the hashes (and with them the digests) of programs without the other forms do not depend on their
        // representation. A device or an in-place form hashes as a whole after the discriminant of `Some`, and the
        // derived hash of a request starts with the discriminant of its form, which tells the two forms apart.
        let hash = |end: &Option<PlacementRequest>, state: &mut H| match end {
            None => Option::<&Placement>::None.hash(state),
            Some(PlacementRequest::Resolved(placement)) => Some(placement).hash(state),
            Some(end @ (PlacementRequest::Device { .. } | PlacementRequest::InPlace { .. })) => Some(end).hash(state),
        };
        hash(&self.source, state);
        hash(&self.target, state);
        self.may_alias.hash(state);
    }
}

impl Operation for PlaceOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        PLACE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        check_count!("region", region_interfaces, 0, TypeError);
        let input = &input_types[0];
        let output = Self::infer_output_type(input, None, self.target(), self.source())?;
        Self::validate_even_splits(input, self.target())?;
        Ok(vec![output])
    }

    #[inline]
    fn requires_fresh_storage(&self) -> bool {
        !self.may_alias
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        // An omitted source or target is not rendered, and neither is the default copy policy, so that the rendering
        // of a placement stays compact. An identity placement therefore renders as its name alone.
        let formatter = OperationFormatter::new(formatter, indentation, self.name())?;
        if self.source.is_none() && self.target.is_none() && self.may_alias {
            return Ok(());
        }
        formatter.bracketed(|operation| {
            if let Some(source) = &self.source {
                operation.field("source", source)?;
            }
            if let Some(target) = &self.target {
                operation.field("target", target)?;
            }
            if !self.may_alias {
                operation.field("may_alias", false)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for PlaceOperation
where
    C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
{
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // The placement flows through the dispatch primitive of `Place`, which also carries the optional target and the
        // copy policy, so that interpretation over staging values (e.g., program batching, re-tracing, etc.) preserves
        // them, while concrete values move to the target or stay where they are without one.
        check_count!("input", inputs, 1, ProgramError);
        let input = &inputs[0];

        let placement = input.placement();
        if let Some(placement) = &placement {
            // Validate the stated source before completing it with the concrete placement.
            Self::infer_output_type(&input.r#type(), Some(placement.sharding()), self.target(), self.source())?;
        }

        let source = PlacementRequest::resolve_source(self.source.clone(), placement);
        Ok(vec![<C::Value as Value>::Dispatch::place(input, self.target.as_ref(), source, self.may_alias)?])
    }
}

impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for PlaceOperation
where
    C::Operation: From<PlaceOperation>,
{
    // The copy policy is enforced where applications fold, through `Operation::requires_fresh_storage`.
}

impl<C: Context<Type = ArrayType>, P: ArrayExtentBatchingPolicy<C>> BatchableOperation<C, ArrayBatchingPolicy<P>>
    for PlaceOperation
where
    C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        // The batching rule places the packed value, so the target and source shardings gain an entry for the batch
        // dimension. Every batching level records exactly one batch sharding in the placements of the batch items that
        // it reports (refer to `Placed`), so this level takes the innermost recorded batch sharding off the target and
        // the source and puts it back at the batch dimension. That makes the source that a batch item reports for
        // itself (e.g., the source that a placement inside a batched function infers from its input) the placement of
        // the packed input, including the split of the batch dimension over auto mesh axes that the packed input type
        // does not record, and it makes the transposed placement of such a source move the cotangent back there. Every
        // recorded batch sharding names only axes of the mesh of its placement, so it can always be put back.
        //
        // An unmapped input has no batch dimension. When the target records a split batch, the placed value must still
        // be split along it, so the input is first broadcast along a new replicated batch dimension where it lives
        // (e.g., the cotangent of a sum is the same for every batch item, and its transposed placement moves it back to
        // a split source). Otherwise, the input stays unmapped and the records are dropped, because a replicated batch
        // dimension carries no placement. A source that records a split batch does not force a batch dimension by
        // itself, because a source is informative and the value is placed where the target says.
        //
        // An explicit target that records nothing places every batch item where it says, so its batch dimension is
        // replicated (e.g., a replicated target replicates the packed value, also when it is split along mesh axes).
        // This deliberately differs from JAX, whose batching rule binds `device_put` on the packed value with the
        // unchanged target, so that its partition specification applies positionally to the packed dimensions (e.g.,
        // a target `P(x)` for a `[6]` batch item mapped at position 0 of a `[4, 6]` value splits the batch dimension,
        // whereas this rule splits the item dimension). An omitted target stays omitted, so that the packed value stays
        // where it is. A source that records nothing gains the placement that the packed input type records for its
        // batch dimension, so that it keeps describing where the packed input lives as far as its type can tell.
        //
        // A device target or source records no sharding and no batch shardings, so it applies to the packed value
        // unchanged, for mapped and unmapped inputs alike, and the output type of a device target follows the sharding
        // of the packed input. The layout of a source of any form describes a batch item and never the packed value,
        // so it is dropped for a mapped input once the type rule has validated it against a batch item, as a source is
        // informative. An in-place target records no sharding either, so it applies to the packed value unchanged and
        // keeps the sharding of the packed input, which also records nothing about the batch dimension, except that its
        // layout is rejected for a mapped input as for the other forms.
        check_count!("input", inputs, 1, ProgramError);

        // The target and the source describe a batch item, so the type rule validates them against the type of a batch
        // item before this rule rewrites them for the packed value (e.g., before it drops the layout of a source, or
        // broadcasts an unmapped input along a new batch dimension, whose type no longer has the layout of the item),
        // so that the batched placement rejects every request that the placement of a batch item rejects with the same
        // error. An unmapped input is its own batch item.
        self.infer_output_types(&[inputs[0].unbatched_type()], &[])?;

        let mut target = self.target.clone();
        let mut source = self.source.clone();
        let pop_batch_sharding = |end: &mut Option<PlacementRequest>| match end {
            Some(PlacementRequest::Resolved(placement)) => placement.pop_batch_sharding(),
            Some(PlacementRequest::Device { .. } | PlacementRequest::InPlace { .. }) | None => None,
        };
        let target_batch_sharding = pop_batch_sharding(&mut target);
        let source_batch_sharding = pop_batch_sharding(&mut source);
        let is_split = matches!(&target_batch_sharding, Some(ShardingDimension::Sharded(axes)) if !axes.is_empty());
        let input = match inputs[0].batch_axis_position() {
            None if is_split => {
                // The new batch dimension is materialized where the input lives, which can be another mesh than the
                // one whose axes the sharding of the batch of this level names (e.g., a cotangent that lives on the
                // target mesh of a forward placement while the batch is split over the source mesh), so it is
                // replicated there. The placement below then splits it as the target records.
                let context = context.clone().with_axis_sharding(ShardingDimension::Replicated);
                P::match_axis(&context, &inputs[0], Axis::from(0))?
            }
            _ => inputs[0].clone(),
        };

        if let Some(batch_axis) = input.batch_axis_position() {
            if target.as_ref().and_then(PlacementRequest::layout).is_some() {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout \
                         does not describe the batch dimension",
                    ),
                });
            }

            let batched = |placement: &Placement, batch_sharding: ShardingDimension| -> Result<_, BatchingError> {
                let sharding = placement.sharding().batched(batch_axis, batch_sharding)?;
                Ok(placement.clone().with_sharding(sharding).map_err(TypeError::from)?)
            };

            if let Some(PlacementRequest::Resolved(placement)) = &target {
                let batch_sharding = target_batch_sharding.unwrap_or(ShardingDimension::Replicated);
                target = Some(PlacementRequest::Resolved(batched(placement, batch_sharding)?));
            }

            source = match source {
                Some(PlacementRequest::Resolved(placement)) => {
                    let batch_sharding = source_batch_sharding.unwrap_or_else(|| {
                        input
                            .r#type()
                            .sharding()
                            .map_or(ShardingDimension::Replicated, |sharding| sharding.dimensions()[batch_axis].clone())
                    });
                    Some(PlacementRequest::Resolved(batched(&placement.without_layout(), batch_sharding)?))
                }
                Some(source @ PlacementRequest::Device { .. }) => Some(source.without_layout()),
                Some(PlacementRequest::InPlace { memory, .. }) => {
                    // An in-place source that named only a layout names nothing without it, so it is dropped.
                    memory.map(|memory| PlacementRequest::InPlace { memory: Some(memory), layout: None })
                }
                None => None,
            };
        }

        // Placement changes preserve the packed shape and mapped batch axis, and the copy policy applies to the packed
        // value unchanged.
        let packed = input.value();
        let placement = packed.placement();
        if let Some(placement) = &placement {
            Self::infer_output_type(&packed.r#type(), Some(placement.sharding()), target.as_ref(), source.as_ref())?;
        }
        let source = PlacementRequest::resolve_source(source, placement);
        let output = <C::Value as Value>::Dispatch::place(packed, target.as_ref(), source, self.may_alias)?;
        Ok(vec![ArrayBatch::new(output, input.batch_axis())?])
    }
}

impl_differentiable_operation! {
    PlaceOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Placed + Value<Dispatch: PlaceDispatch<ArrayType, C::Value>>,
        C::Operation: From<PlaceOperation>,
    {
        |operation, _context, _driver, inputs| {
            // A placement is linear, so the tangent is placed on the same devices, with the same sharding, in the same
            // memory, and with the same copy policy as the primal, and symbolic zero tangents stay symbolic without
            // allocation. The tangent can have another element type than the primal (e.g., an `f8e8m0fnu` primal has
            // an `f32` tangent), for which the layouts of the target and source can be invalid, so it is placed without
            // them. Derivative types carry no layout either. Without a source, the primal records where it concretely
            // lives as its source, if anywhere, and so does the tangent, because the tangent is placed by the same
            // linear map, whose transpose must move the cotangent back to where the primal lived. An in-place target
            // that names only a layout names nothing without it, so the tangent is placed without a target, which
            // leaves it where it is, as the primal is, and such an in-place source is replaced like an absent one.
            check_count!("input", inputs, 1, ProgramError);
            let may_alias = operation.may_alias();
            let input = inputs[0].primal();
            let placement = input.placement();
            if let Some(placement) = &placement {
                let sharding = Some(placement.sharding());
                PlaceOperation::infer_output_type(&input.r#type(), sharding, operation.target(), operation.source())?;
            }
            let source = PlacementRequest::resolve_source(operation.source().cloned(), placement);
            let primal = <C::Value as Value>::Dispatch::place(input, operation.target(), source.clone(), may_alias)?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => {
                    let names_nothing = |end: &PlacementRequest| {
                        matches!(end, PlacementRequest::InPlace { memory: None, layout: None })
                    };
                    let target = operation.target().cloned().map(PlacementRequest::without_layout);
                    let target = target.filter(|target| !names_nothing(target));
                    let source = source.map(PlacementRequest::without_layout).filter(|source| !names_nothing(source));
                    let placement = tangent.placement();
                    if let Some(placement) = &placement {
                        let r#type = tangent.r#type();
                        let sharding = Some(placement.sharding());
                        PlaceOperation::infer_output_type(&r#type, sharding, target.as_ref(), source.as_ref())?;
                    }
                    let source = PlacementRequest::resolve_source(source, placement);
                    MaybeZero::Value(<C::Value as Value>::Dispatch::place(tangent, target.as_ref(), source, may_alias)?)
                }
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<BroadcastOperation> + From<PlaceOperation>,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // A placement is the identity linear map between two placements, so its transpose moves the output
            // cotangent back to where the input lived, as JAX's `device_put` transpose does by swapping its devices
            // and sources. The cotangent is sharded like the cotangent dual of the input's sharding (i.e., with its
            // unreduced and reduced axes swapped) and lives in the input's memory. It carries no layout, since
            // derivative types carry none. Whatever the copy policy of the forward placement is, the transposed
            // placement forces a copy, as JAX's `device_put` transpose does, so that the input cotangent never shares
            // storage with the output cotangent that the caller provided. Symbolic zero cotangents propagate unchanged.
            // Every placement that the transposed placement derives from the target or the source keeps the batch
            // shardings that it records (refer to `Placement::batch_shardings`), so that the transposed placement of
            // a batch item inside a batched function still moves the cotangent back to how its batch was distributed.
            // A source that splits a static extent unevenly (e.g., an extent of 3 over 2 devices) cannot be the target
            // of the transposed placement, because type inference requires even splits, so transposition fails for it.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);

            // Nothing is validated or staged for a cotangent that is not needed, so that a placement whose
            // transposition would fail cannot fail the derivative of an unrelated input.
            let MaybeZero::Value(cotangent) = &outputs[0] else { return Ok(()) };
            if !accumulators[0].is_needed() {
                return Ok(());
            }
            let input_type = inputs[0].r#type();
            let input_cotangent_type = input_type.cotangent()?;

            // A device target leaves a value without reduction state, so the output cotangent has none either, and the
            // broadcast below gives the input cotangent the cotangent dual of the reduction state of the input back.
            // That is exact over mesh axes of size one, where the one device that holds the cotangent is the only
            // contribution of an unreduced cotangent. The cotangent of an input that is reduced over a larger axis
            // is unreduced over it, and its contributions on the devices along that axis must sum to the output
            // cotangent, which a placement that leaves the complete cotangent on each of them cannot produce.
            let has_device_target = matches!(operation.target(), Some(PlacementRequest::Device { .. }));
            if has_device_target && let Some(sharding) = input_type.sharding() {
                let larger_axes = sharding
                    .reduced_axes()
                    .iter()
                    .filter(|axis| sharding.mesh().axis_size(axis).is_some_and(|size| size > 1))
                    .map(|axis| format!("`{axis}`"))
                    .collect::<Vec<_>>();
                if !larger_axes.is_empty() {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is \
                             reduced over mesh axes larger than one (i.e., {}), because its cotangent is unreduced \
                             over them",
                            larger_axes.join(", "),
                        ),
                    }
                    .into());
                }
            }

            // Returns the end of the transposed placement that corresponds to the end `end` of the forward placement,
            // where the transposed placement does not move the cotangent to it (i.e., the forward target, which becomes
            // the transposed source, and the source of a value inside a manual region). A placement gets the cotangent
            // dual of its sharding, and every form loses its layout and memory, because the storage of the output
            // cotangent is not known. An in-place request names nothing without them, so it has no counterpart.
            let transposed_end = |end: &PlacementRequest| -> Result<Option<PlacementRequest>, TypeError> {
                Ok(match end {
                    PlacementRequest::Resolved(placement) => {
                        let sharding = placement.sharding().cotangent()?;
                        let placement = placement.clone().with_sharding(sharding)?;
                        Some(PlacementRequest::Resolved(placement.without_layout().without_memory()))
                    }
                    PlacementRequest::Device { device, .. } => {
                        Some(PlacementRequest::Device { device: *device, memory: None, layout: None })
                    }
                    PlacementRequest::InPlace { .. } => None,
                })
            };

            // Returns the transposed placement that leaves the cotangent on the devices where it lives and only returns
            // it to the memory of the input when the forward target changed the storage of the input in place. This is
            // the transposed placement of a value inside a manual region, which cannot have been moved and whose
            // cotangent lives in the same region, and of a placement whose source is not known. The forward source
            // stays the source, so that a second transposition records it again.
            let storage_only = |source: Option<&PlacementRequest>, target: Option<&PlacementRequest>| {
                let memory = Some(input_type.memory());
                let target = target.map(|_| PlacementRequest::InPlace { memory, layout: None });
                let source = source.map(transposed_end).transpose()?.flatten();
                Ok::<_, TypeError>(PlaceOperation::new(target, source))
            };

            // A device target gathers an input that its type tracks over a mesh of more than one device onto that one
            // device, so only a source that names the devices of that mesh (i.e., a placement) can return the cotangent
            // to them.
            let is_gathered =
                has_device_target && input_type.sharding().is_some_and(|sharding| sharding.mesh().device_count() > 1);
            let gathered_error = || ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that lives on more \
                     than one device without a source placement that names those devices",
                ),
            };

            // A transposed placement whose target names no mesh (i.e., an in-place source, or no source at all with a
            // placement target) leaves the cotangent on the logical mesh where it lives, and the broadcast below can
            // only give it the type of the input cotangent over a mesh of the same topology. A cotangent on a mesh of
            // another topology than the input's comes from a placement that changed meshes, and only a source that
            // names devices (i.e., a placement, or a device for an input on a mesh of one device) tells where it
            // returns, so such a transposition is rejected rather than staging a placement that no backend can execute.
            let crosses_meshes = matches!(
                (cotangent.r#type().sharding(), input_cotangent_type.sharding()),
                (Some(output), Some(input)) if !output.mesh().has_same_topology(input.mesh()),
            );
            let cross_mesh_error = || ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` across device meshes without a source placement",
                ),
            };

            let transposed = match (operation.source(), operation.target()) {
                (Some(source), target) if operation.transposes_in_place(&input_type) => {
                    storage_only(Some(source), target)?
                }
                (Some(source), target) => {
                    // With a known source, the cotangent moves back to the source in the memory of the input for every
                    // form of the forward target, as JAX's `device_put` transpose swaps its devices and sources. This
                    // includes an in-place target, whose output cotangent can live on other devices than the input did
                    // (e.g., a gradient seed). A placement source gets the cotangent dual of its sharding, a device
                    // source becomes the placement of the input cotangent over the logical mesh of the input on that
                    // device (or stays a device for an input without a tracked sharding), and an in-place source names
                    // no devices, so the cotangent returns in place to the memory that it names, which is the memory of
                    // the input. The forward target becomes the transposed source, so that a second transposition moves
                    // the value to the forward target again, except for an in-place target, which says nothing about
                    // where the output cotangent lives. After a device target, the output cotangent has no reduction
                    // state, so a placement source loses its reduction state too, because a placement cannot change it,
                    // and the broadcast below restores it.
                    let memory = Some(input_type.memory());
                    let transposed_target = match source {
                        PlacementRequest::Resolved(placement) => {
                            let mut sharding = placement.sharding().cotangent().map_err(TypeError::from)?;
                            if has_device_target {
                                sharding = sharding
                                    .with_unreduced_axes(Vec::<String>::new())
                                    .and_then(|sharding| sharding.with_reduced_axes(Vec::<String>::new()))
                                    .map_err(TypeError::from)?;
                            }
                            let placement = placement.clone().with_sharding(sharding).map_err(TypeError::from)?;
                            PlacementRequest::Resolved(placement.without_layout().with_memory(input_type.memory()))
                        }
                        PlacementRequest::Device { device, .. } => {
                            let target_type = if has_device_target {
                                input_cotangent_type.without_reduction_axes()
                            } else {
                                input_cotangent_type.clone()
                            };

                            // The type rule accepts a device source only for an input that its type tracks over a mesh
                            // of one device, so the device and the logical mesh of the input together name where the
                            // input lived, and the cotangent returns there whatever mesh the output cotangent lives on.
                            if let Some(sharding) = target_type.sharding() {
                                let mesh =
                                    DeviceMesh::new(sharding.mesh().clone(), vec![*device]).map_err(TypeError::from)?;
                                let placement = Placement::new(mesh, sharding.clone()).map_err(TypeError::from)?;
                                PlacementRequest::Resolved(placement.with_memory(input_type.memory()))
                            } else {
                                PlacementRequest::Device { device: *device, memory, layout: None }
                            }
                        }
                        PlacementRequest::InPlace { .. } if is_gathered => return Err(gathered_error().into()),
                        PlacementRequest::InPlace { .. } if crosses_meshes => return Err(cross_mesh_error().into()),
                        PlacementRequest::InPlace { .. } => PlacementRequest::InPlace { memory, layout: None },
                    };
                    let transposed_source = target.map(transposed_end).transpose()?.flatten();
                    PlaceOperation::new(transposed_target, transposed_source)
                }
                (None, target @ (None | Some(PlacementRequest::InPlace { .. }))) => {
                    // Without a source, the devices of the input are not known, so an in-place target returns the
                    // cotangent in place to the memory of the input, and a placement without a target and a source is
                    // an identity up to its copy policy, and so is its transpose, which still forces a copy, as JAX's
                    // `device_put` transpose does.
                    storage_only(None, target)?
                }
                (None, Some(PlacementRequest::Resolved(target))) => {
                    // Without a source, the value is not known to have changed meshes, so the cotangent stays on the
                    // target mesh and gets the input's tracked sharding and memory back. This deliberately differs from
                    // JAX, whose transpose swaps the devices and sources of `device_put` and so leaves a cotangent with
                    // an unknown source where it is. Restoring the input's tracked sharding and memory keeps the
                    // cotangent well typed, and it is exact inside a compiled program, which runs on a single mesh.
                    // An input that is tracked over another mesh did change meshes, and its devices are not known.
                    // The forward target becomes the transposed source, as for a known source, so that a second
                    // transposition moves the value to the forward target again, including its placement along auto
                    // mesh axes, which neither type records.
                    if crosses_meshes {
                        return Err(cross_mesh_error().into());
                    }
                    let sharding = match input_cotangent_type.sharding() {
                        Some(sharding) => sharding.clone(),
                        None => target.sharding().cotangent().map_err(TypeError::from)?,
                    };
                    let source = transposed_end(&PlacementRequest::Resolved(target.clone()))?;
                    let placement = target.clone().with_sharding(sharding).map_err(TypeError::from)?;
                    PlaceOperation::new(placement.without_layout().with_memory(input_type.memory()), source)
                }
                (None, Some(PlacementRequest::Device { device, .. })) => {
                    // Without a source, the devices of the input are not known, so the cotangent stays on the device
                    // of the forward target, where it gets the input's memory back. A device target keeps the tracked
                    // sharding of an input on a mesh of one device up to its reduction state, so the output cotangent
                    // has the sharding of the input cotangent up to that state, which the broadcast below restores. A
                    // gathered input cannot get its cotangent back on its devices.
                    if is_gathered {
                        return Err(gathered_error().into());
                    }
                    let memory = Some(input_type.memory());
                    PlaceOperation::new(PlacementRequest::Device { device: *device, memory, layout: None }, None)
                }
            };

            let transposed = transposed.with_may_alias(false);
            let mut placed = context.stage_operation(transposed, Vec::new(), std::slice::from_ref(cotangent))?;
            check_count!("output", placed, 1, ProgramError);
            let placed = placed.remove(0);

            // An input without a tracked sharding receives its exact cotangent type through a placement-only
            // broadcast, as in `ReshardOperation` transposition, and so does an input whose reduction state a
            // device target completed, whose cotangent carries the dual of that state over mesh axes of size one.
            let contribution = if placed.r#type().as_ref() == &input_cotangent_type {
                placed
            } else {
                placed.broadcast(
                    input_cotangent_type.clone(),
                    &(0..input_cotangent_type.shape().rank()).collect::<Vec<_>>(),
                )?
            };

            accumulators[0].accumulate(context, MaybeZero::Value(contribution))?;

            Ok(())
        }
    },
}

impl<O: Operation<Type = ArrayType> + From<PlaceOperation>> OperationProvider<ArrayType, PlaceOperation> for O {
    type Operation = Self;

    #[inline]
    fn provide(request: PlaceOperation, input_types: &[&ArrayType]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        Ok(request.into())
    }
}

impl<O: Operation<Type = ArrayIrType> + OperationProjection<ArrayType, Projected: From<PlaceOperation>>>
    OperationProvider<ArrayIrType, PlaceOperation> for O
{
    type Operation = Self;

    #[inline]
    fn provide(request: PlaceOperation, input_types: &[&ArrayIrType]) -> Result<Self, ProgramError> {
        // Array IR families select the placement through their array projection, so that the provided operation
        // is the canonical array member (e.g., `ArrayIrOperation::Array(ArrayOperation::Place(_))`).
        check_count!("input", input_types, 1, ProgramError);
        let operation: <Self as OperationProjection<ArrayType>>::Projected = request.into();
        Ok(operation.into())
    }
}

/// Reports the [`Placement`] that a value concretely lives on. Devices and placement over [`MeshAxisType::Auto`] mesh
/// axes are runtime state that types do not record, so only the values that hold device buffers have a placement
/// (e.g., the arrays of a backend). Staged values have none, because where they live is decided when their program is
/// compiled and executed, and neither do reference [`Array`]s, which are host values. A value that wraps another value
/// (e.g., a [`PartialTracer`] or a [`DifferentiationTracer`]) reports the placement of the value that it wraps.
/// A batched value (i.e., an [`ArrayBatch`], an [`ArrayIrBatch`], or a [`BatchingTracer`] whose batch reports a
/// placement) reports the placement of the batch items that a batched function observes. That is the placement of its
/// packed value with the batch dimension removed from the sharding, because the placement of that dimension describes
/// how the batch is distributed rather than how each batch item is placed, and without a layout, because a layout of
/// the packed value does not describe a batch item. The placement of the removed dimension is recorded as the innermost
/// batch sharding of the reported placement instead (refer to [`Placement::batch_shardings`]), because a batch item
/// does not live on the whole mesh when its batch is split, and the batching rule of [`PlaceOperation`] puts it back.
/// A replicated value is its own batch item, so it reports the placement of its packed value with a replicated batch
/// recorded. Every batching level thus records exactly one batch sharding, so that nested batching levels each take
/// their own off. When a placement omits its source, the rules of [`PlaceOperation`] that execute it on such a value
/// (i.e., interpretation, batching, and differentiation) use this trait to record its concrete placement.
pub trait Placed {
    /// Returns the [`Placement`] that this value concretely lives on, or [`None`] if it has none.
    fn placement(&self) -> Option<Placement>;
}

impl Placed for Array {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // Reference arrays are host values that no device holds, so they have no placement even when their type records
        // a sharding (which then only describes how a backend would distribute the array).
        None
    }
}

impl<A: Value<Type = ArrayType> + Placed> Placed for ArrayIrValue<A> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // Only array members live on devices. First-class dimensions and references have no placement.
        match self {
            Self::Array(value) => value.placement(),
            Self::Dimension(_) | Self::Reference(_) => None,
        }
    }
}

impl<T: Type, V: Placed> Placed for ProjectedValue<T, V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        self.value().placement()
    }
}

impl<C: Context> Placed for Tracer<C> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A staged value lives wherever the program that computes it places it, which is decided only when that
        // program is compiled and executed.
        None
    }
}

impl<C: Context<Value: Placed>> Placed for PartialTracer<C> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A partial evaluation value lives where its known payload lives, while an unknown value has no placement yet.
        // A poisoned value reports none either, and its deferred error surfaces when the value is used.
        self.value().ok().and_then(|value| match value.value() {
            PartialValue::Known(known) => known.placement(),
            PartialValue::Unknown(_) => None,
        })
    }
}

impl<V: Value<Type = ArrayType> + Placed> Placed for ArrayBatch<V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        unbatched_placement(self.value().placement()?, self.batch_axis_position())
    }
}

impl<V: Value<Type = ArrayIrType> + Placed> Placed for ArrayIrBatch<V> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        unbatched_placement(self.value().placement()?, self.batch_axis_position())
    }
}

impl<C: Context, P: BatchingPolicy<C, Batch: Placed>> Placed for BatchingTracer<C, P> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // The policy-selected batch owns the batch dimension of the packed value, so it also owns how the placement
        // of a batch item follows from the placement of that packed value.
        self.batch().placement()
    }
}

impl<C: Context<Value: Placed>, P: DifferentiationPolicy<C>> Placed for DifferentiationTracer<C, P> {
    #[inline]
    fn placement(&self) -> Option<Placement> {
        // A differentiation tracer lives where its primal lives.
        self.primal().placement()
    }
}

/// Executes one [`PlaceOperation`] for a leaf through its [`Value::Dispatch`] policy. A domain-dispatched leaf binds
/// the operation in its own context. Direct backends can implement this trait on [`ValueDirectDispatch`] for their
/// values. This primitive and [`Place`], which calls it for each leaf, use the source that they are given as is. A
/// source is inferred only by the rules of [`PlaceOperation`] that execute a placement on a value that concretely lives
/// somewhere (i.e., interpretation, batching, and differentiation, refer to [`PlacementRequest::resolve_source`]).
/// Array IR values project the array member, place it, and lift the result back into their original universe.
///
/// Unlike [`Place`], this primitive also takes an optional source, and both the target and the source are
/// [`PlacementRequest`]s, whose forms it handles as their contract states, together with the copy policy of the
/// placement (refer to [`PlaceOperation::with_may_alias`]). Without a target, the value stays where it is (i.e., its
/// devices, sharding, memory, and layout are unchanged), which places a value with its source recorded or, with
/// `may_alias` set to `false`, gives it storage of its own. With `may_alias` set to `false`, implementations for
/// concrete values must return a value whose storage no other value shares, and implementations for staged values must
/// stage the placement with that policy, so that the copy happens when the staged placement executes on concrete
/// values. Implementations for concrete values resolve a device or in-place target against the value that they place,
/// and reference [`Array`]s, which have no devices, record the output type that [`PlaceOperation`] type inference
/// computes for it.
pub trait PlaceDispatch<T: Type, V: Value<Type = T>> {
    /// Moves `input` to `target` and records `source` as its expected placement. A [`None`] target leaves it where it
    /// is. The result may share storage with `input` only when `may_alias` is `true`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::Type`] when [`PlaceOperation`] type inference rejects the target or the source for
    /// the type of `input` (e.g., when their ranks differ from that of `input` or the target changes its reduction
    /// state), and [`ProgramError::InvalidInputCount`] or [`ProgramError::InvalidOutputCount`] when a context binds
    /// the placement with an unexpected number of values. Array IR values return [`ProgramError::Type`] for a member
    /// that is not an array. Backends that execute the placement on concrete values return their own errors (e.g.,
    /// when a transfer to the devices of `target` fails), and staging contexts return the errors of staging it.
    fn place(
        input: &V,
        target: Option<&PlacementRequest>,
        source: Option<PlacementRequest>,
        may_alias: bool,
    ) -> Result<V, ProgramError>;
}

impl PlaceDispatch<ArrayType, Array> for ValueDirectDispatch {
    fn place(
        input: &Array,
        target: Option<&PlacementRequest>,
        source: Option<PlacementRequest>,
        may_alias: bool,
    ) -> Result<Array, ProgramError> {
        // A reference array is a host value without devices, so placing it keeps its elements unchanged. Its type still
        // records the placement, and the output type comes from the `PlaceOperation` type inference rule itself, so
        // that eager evaluation validates the target and source exactly like staged programs do. The layout determines
        // how the physical storage addresses the elements, so the storage is shared only when the layout is unchanged
        // and the copy policy allows it, and it is copied when that policy forbids sharing it. Otherwise, the logical
        // bytes are re-encoded for the output layout into new storage, which preserves the exact element bits
        // (including low-precision encodings and NaN payloads), as in `Cast` for `Array`. Without a target,
        // the output type is the input type, so the storage is shared or copied as the copy policy says.
        //
        // A layout describes each shard of a value, whereas a reference array stores the whole value in one buffer,
        // so it can only materialize a layout that also addresses the whole value. The type rule accepts a layout that
        // addresses each shard of a split value only (e.g., the byte strides of a column shard of a matrix, which
        // overlap the rows of the whole matrix), which a backend that stores shards materializes, and this backend
        // rejects it here with an error that says so, rather than applying it to the whole value. The storage of an
        // array is publicly mutable, so it is decoded with the fallible decoder, which rejects element encodings that
        // are not valid (e.g., a Boolean byte other than `0` and `1`).
        let operation = PlaceOperation::new(target.cloned(), source).with_may_alias(may_alias);
        let mut output_types = operation.infer_output_types(&[input.r#type().into_owned()], &[])?;
        check_count!("output", output_types, 1, ProgramError);
        let output_type = output_types.remove(0);
        if output_type.layout() != input.r#type().layout() {
            if let Err(error) = output_type.validate_layout() {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "the reference `Array` backend stores a whole value in one buffer, so it cannot materialize \
                         the shard layout of `{output_type}` for it: {error}",
                    ),
                });
            }
            Array::from_logical_bytes(output_type, &decode_logical_bytes(&input.r#type(), input.storage_bytes())?)
        } else if may_alias {
            Ok(Array::new_unchecked(output_type, input.shared_storage_bytes().clone()))
        } else {
            Ok(Array::new_unchecked(output_type, Arc::new(input.storage_bytes().to_vec())))
        }
    }
}

impl<
    V: Value<
            Type = ArrayType,
            Dispatch = ValueDomainDispatch,
            Domain: Context<Type = ArrayType, Operation: From<PlaceOperation>>,
        >,
> PlaceDispatch<ArrayType, V> for ValueDomainDispatch
{
    #[inline]
    fn place(
        input: &V,
        target: Option<&PlacementRequest>,
        source: Option<PlacementRequest>,
        may_alias: bool,
    ) -> Result<V, ProgramError> {
        let outputs = input.domain().bind(
            PlaceOperation::new(target.cloned(), source).with_may_alias(may_alias),
            Vec::new(),
            std::slice::from_ref(input),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.into_iter().next().unwrap())
    }
}

impl<
    V: Value<Type = ArrayIrType>
        + ValueProjection<ArrayType, Projected: Value<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, V::Projected>>>,
> PlaceDispatch<ArrayIrType, V> for V::Dispatch
{
    #[inline]
    fn place(
        input: &V,
        target: Option<&PlacementRequest>,
        source: Option<PlacementRequest>,
        may_alias: bool,
    ) -> Result<V, ProgramError> {
        let input = input.clone().into_projected()?;
        let output = <V::Projected as Value>::Dispatch::place(&input, target, source, may_alias)?;
        Ok(V::from_projected(output))
    }
}

/// Source argument accepted by [`Place::place`], which is a borrowed tree of [`PlacementRequestParameter`]s with the
/// same ordered parameter paths as the receiver. `&S` supplies the source of every element of the receiver, and an
/// element whose source is [`None`] (e.g., in a tree of `Option<Device>`s) infers its own. `Option<&S>` supplies or
/// omits a whole tree of [`Placement`]s, so that a bare `None` infers every source without a type annotation.
pub trait PlacementSources<P: Parameter, S: Parameterized<P>> {
    /// Returns the supplied source tree, or [`None`] to infer the source of every element.
    fn as_parameterized(&self) -> Option<&S>;
}

impl<S: Parameterized<Placement>> PlacementSources<Placement, S> for Option<&S> {
    #[inline]
    fn as_parameterized(&self) -> Option<&S> {
        *self
    }
}

impl<P: PlacementRequestParameter, S: Parameterized<P>> PlacementSources<P, S> for &S {
    #[inline]
    fn as_parameterized(&self) -> Option<&S> {
        Some(self)
    }
}

/// Moves a value to a target [`Placement`], possibly on the devices of another [`DeviceMesh`],
/// without changing its elements, shape, or data type. This is the Ryft analogue of
/// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html), and like JAX's `device_put`
/// primitive it works inside transformations, so that a value can be moved within staged programs and functions that
/// are batched or differentiated. Its type records the target sharding (without its [`MeshAxisType::Auto`] mesh axes)
/// and the target [`Memory`] and [`Layout`] when the target names them, while an omitted memory keeps the value's
/// current memory and an omitted layout keeps its layout while that layout still describes each shard. A target can
/// also be any other [`PlacementRequest`] (e.g., a [`Device`], which without a memory also selects the default memory
/// of that device, or a bare [`Memory`], which changes only the memory of a value where it lives), whose forms follow
/// the contract that [`PlacementRequest`] states. Refer to [`PlaceOperation`] for the validation that the target and
/// source go through.
///
/// The source is informative. It states where the value is expected to live when the placement is created. Placement
/// over auto mesh axes and devices are runtime state that types do not record, so transposition uses the source to move
/// the cotangent back to those devices. [`PlaceOperation`] type inference checks the source against what the type of
/// the value tracks, while backends that execute the placement on a concrete value move the value from wherever it
/// actually lives without comparing that placement with the source. Each leaf is placed with the source as it is
/// given, and the rules of [`PlaceOperation`] that execute the placement on a leaf that concretely lives somewhere
/// (refer to [`Placed`]) record that placement when the source is [`None`] and complete an in-place source with it
/// (refer to [`PlacementRequest::resolve_source`]) after they validate the request as it is given, so that a traced
/// value that wraps a concrete array records where that array lives, while staged values leave the source unknown, and
/// this capability rejects every request that the type rule rejects. Reference [`Array`]s have no devices, so they
/// record the output type that [`PlaceOperation`] type inference computes without changing their elements, and they
/// re-encode their storage only for a target layout that differs from theirs. Unlike [`Reshard`](crate::Reshard), which
/// redistributes a value over the mesh that it already lives on, this capability can express placements across meshes.
/// Host scalar types have no placement and do not implement it.
///
/// Every placement through this capability uses the default copy policy of [`PlaceOperation`], so its results may share
/// storage with its inputs (e.g., a reference [`Array`] keeps sharing its storage unless its layout changes). To place
/// a leaf whose result must have storage of its own, call [`PlaceDispatch::place`] with `may_alias` set to `false`
/// instead, which also forces that copy when the placement is staged and later executes on concrete values.
/// [`PlaceDispatch::place`] also places a leaf without a target, which leaves it where it is.
///
/// A single value accepts its target followed by an optional source. A [`Parameterized`] receiver accepts placements
/// with the same structure (i.e., the same ordered parameter paths and containers without children) and preserves its
/// structure and static fields. The universe parameter `T` defaults to the [`Capability`] universe of a single-value
/// receiver, which is [`ArrayType`] for homogeneous arrays and [`ArrayIrType`] for array IR values. The leaf parameter
/// `P` defaults to `Self`, so single-value bounds can use `Place<T>`. A structured receiver implements `Place<T, P>`
/// for the universe `T` of its leaves.
///
/// # Examples
///
/// ```
/// # use ryft_core::{
/// #     Array, ArrayType, DataType, Device, DeviceMesh, LogicalMesh, Memory, MeshAxis, MeshAxisType, Place, Placement,
/// #     Sharding, ShardingDimension, Typed,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit)?])?;
/// let devices = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)])?;
/// let sharding = Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])])?;
/// let target = Placement::new(devices, sharding.clone())?.with_memory(Memory::Host { pinned: true });
/// let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2]), &[1i32, 2])?;
/// let output = input.place(&target, None)?;
/// assert_eq!(output.r#type().sharding(), Some(&sharding));
/// assert_eq!(output.r#type().memory(), Memory::Host { pinned: true });
/// assert_eq!(output.elements::<i32>()?, vec![1, 2]);
///
/// let inputs = (input, output);
/// let targets = (target.clone(), target.clone());
/// let outputs = inputs.place(&targets, &(None, Some(target)))?;
/// assert_eq!(outputs.0, outputs.1);
/// assert_eq!(outputs.place(&targets, Some(&targets))?, outputs);
/// # Ok(())
/// # }
/// ```
pub trait Place<T = <Self as Capability>::Universe, P: Parameter + Capability = Self>: Sized {
    /// Places each leaf on its corresponding target. The targets form a tree of one element type that converts into a
    /// [`PlacementRequest`] (refer to [`PlacementRequestParameter`]), such as [`Placement`]s, [`PlacementRequest`]s of
    /// any form, [`Device`]s, or [`Memory`] spaces, or their [`Option`]s, whose [`None`] leaves the leaf where it is.
    /// The sources follow the same conversion. Pass `None` to infer all sources, `Some(&sources)` to supply a tree of
    /// [`Placement`]s, or a borrowed tree of any [`PlacementRequestParameter`] (e.g., `&(None, Some(device))`) to
    /// supply the sources of its elements and infer those that are [`None`]. Each inferred source comes from the
    /// leaf's [`Placed::placement`], which can itself be unknown.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::Parameter`] with [`ParameterError::DuplicateParameterPaths`] before placing any leaf
    /// if distinct nodes of the receiver have the same path (e.g., the entries of two map keys with the same `Debug`
    /// text), which could not be matched with their targets and sources otherwise, and
    /// [`ProgramError::InvalidArgument`] before placing any leaf if the ordered parameter paths or the node paths
    /// (refer to [`Parameterized::paths`]) of the receiver, targets, or supplied sources differ, and otherwise the
    /// first error that placing a leaf returns (refer to [`PlaceDispatch::place`]), such as a [`ProgramError::Type`]
    /// when [`PlaceOperation`] type inference rejects the target or source of that leaf.
    fn place<Target: PlacementRequestParameter, Source: PlacementRequestParameter>(
        &self,
        targets: &<Self as Parameterized<P>>::To<Target>,
        sources: impl PlacementSources<Source, <Self as Parameterized<P>>::To<Source>>,
    ) -> Result<Self, ProgramError>
    where
        Self: Parameterized<P, Family: ParameterizedFamily<Target> + ParameterizedFamily<Source>>;
}

impl<T, P: Value<Dispatch: PlaceDispatch<P::Type, P>> + Capability<Universe = T>, S: Parameterized<P>> Place<T, P>
    for S
{
    fn place<Target: PlacementRequestParameter, Source: PlacementRequestParameter>(
        &self,
        targets: &S::To<Target>,
        sources: impl PlacementSources<Source, S::To<Source>>,
    ) -> Result<Self, ProgramError>
    where
        S::Family: ParameterizedFamily<Target> + ParameterizedFamily<Source>,
    {
        // Distinct nodes of one tree with the same path (e.g., the entries of two map keys with the same `Debug` text)
        // cannot be told apart by path, so the leaves of the receiver would be matched with those of the targets and
        // the sources by position, which could give one key the placement of another. They are rejected as the tree
        // prefix expansion of `Parameterized::broadcast_to_parameter_structure` rejects them. The targets and the
        // sources must have the node paths of the receiver, so checking the receiver covers them. Two structures with
        // the same leaf paths can still differ in their other nodes (e.g., `[[x], []]` and `[[x]]`), and two
        // structures with the same node paths can still differ in their leaves (e.g., `[x]` and `[[]]`), so both are
        // compared.
        let sources = sources.as_parameterized();
        let paths = self.paths();
        let duplicate_paths = find_duplicate_paths(&paths);
        if !duplicate_paths.is_empty() {
            return Err(ParameterError::DuplicateParameterPaths { paths: duplicate_paths }.into());
        }

        if !self.parameter_paths().eq(targets.parameter_paths()) || paths != targets.paths() {
            return Err(ProgramError::InvalidArgument {
                message: "receiver and `targets` must have the same parameter structure".to_string(),
            });
        }

        if sources
            .is_some_and(|sources| !self.parameter_paths().eq(sources.parameter_paths()) || paths != sources.paths())
        {
            return Err(ProgramError::InvalidArgument {
                message: "receiver and `sources` must have the same parameter structure".to_string(),
            });
        }

        let mut sources = sources.into_iter().flat_map(|sources| sources.parameters());
        let outputs = self
            .parameters()
            .zip(targets.parameters())
            .map(|(input, target)| {
                // The source is placed as it is given, and the rule that executes the placement on a value that
                // concretely lives somewhere completes it with that placement after it validates it (refer to
                // `PlacementRequest::resolve_source`), so that this capability never accepts a request that the type
                // rule rejects.
                let source = sources.next().and_then(|source| source.clone().into_optional_placement_request());
                let target = target.clone().into_optional_placement_request();
                P::Dispatch::place(input, target.as_ref(), source, true)
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(S::from_parameters(self.parameter_structure(), outputs)?)
    }
}

/// Validates `layout` as the storage layout of each shard of a value of type `r#type` that `sharding` distributes, as
/// far as the type determines its shards. A layout describes each shard (i.e., the buffer that one device holds), so
/// its rank must always be the rank of the value. A tiled layout is a permutation of the dimensions with tiles, which
/// is valid for every shape of its rank, so it is validated against the shape of the value. A strided layout holds
/// byte strides that must keep the elements of a shard apart, so they are validated against the extents of a shard
/// when `sharding` determines them (i.e., when every extent is static and split evenly, and `sharding` either describes
/// every mesh axis, which `is_complete` states, or its mesh has no auto axis larger than one, along which a value may
/// be split without its type recording it). Otherwise, only the rank of a strided layout is validated here, and the
/// backend that places the value validates its strides against the shards that it creates. An unconstrained dimension
/// (i.e., `{?}`) is stored replicated (refer to [`ShardingDimension::Unconstrained`]), so its shards have its whole
/// extent, as for a replicated dimension. A value without a tracked sharding is validated against its whole shape.
pub(super) fn validate_shard_layout(
    r#type: &ArrayType,
    layout: &Layout,
    sharding: Option<&Sharding>,
    is_complete: bool,
) -> Result<(), TypeError> {
    let shard_shape = match (layout, sharding) {
        (Layout::Tiled(_), _) | (_, None) => Some(r#type.shape().clone()),
        (Layout::Strided(_), Some(sharding)) => {
            let is_determined = is_complete
                || sharding.mesh().axes().iter().all(|axis| axis.r#type() != MeshAxisType::Auto || axis.size() == 1);
            r#type
                .shape()
                .dimensions()
                .iter()
                .zip(sharding.dimensions())
                .map(|(dimension, sharding_dimension)| match (dimension, sharding_dimension) {
                    (Dimension::Static(extent), ShardingDimension::Replicated | ShardingDimension::Unconstrained) => {
                        Some(Dimension::Static(*extent))
                    }
                    (Dimension::Static(extent), ShardingDimension::Sharded(axis_names)) => {
                        let count =
                            axis_names.iter().filter_map(|name| sharding.mesh().axis_size(name)).product::<usize>();
                        (extent % count == 0).then(|| Dimension::Static(extent / count))
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()
                .filter(|dimensions| is_determined && dimensions.len() == r#type.rank())
                .map(Shape::new)
        }
    };

    match (layout, shard_shape) {
        (_, Some(shape)) => r#type.clone().with_shape(shape).with_layout(layout.clone()).validate_layout(),
        (Layout::Strided(strided), None) if strided.rank() != r#type.rank() => Err(TypeError::invalid(format!(
            "strided layout rank {} does not match array rank {}",
            strided.rank(),
            r#type.rank(),
        ))),
        (_, None) => Ok(()),
    }
}

/// Returns whether `sharding` describes a value inside a manual region (i.e., one that varies along manual mesh axes or
/// carries reduction state over them, whatever their sizes), which lives on the devices of that region and so cannot be
/// moved. A global array that is only tiled over a manual axis of its mesh lives outside any manual region.
fn is_manual_region_sharding(sharding: &Sharding) -> bool {
    !sharding.varying_manual_axes().is_empty()
        || sharding
            .unreduced_axes()
            .iter()
            .chain(sharding.reduced_axes())
            .any(|axis| sharding.mesh().axis_type(axis) == Some(MeshAxisType::Manual))
}

/// Returns the placement of the batch items of a packed value that lives on `placement` and whose batch dimension is
/// at position `batch_axis`, or [`None`] if the sharding of `placement` has no such dimension. The batch dimension is
/// removed from the sharding, because its placement describes how the batch is distributed rather than how each batch
/// item is placed (refer to [`ArrayType::unbatched`]), and so is the layout, which describes the storage of the packed
/// value. The placement of the removed dimension is recorded as the innermost batch sharding of the result instead
/// (refer to [`Placement::batch_shardings`]), because a batch item does not live on the whole mesh when its batch is
/// split, and the batching rule of [`PlaceOperation`] puts it back. A replicated value (i.e., one without a batch
/// dimension) is its own batch item, so `placement` is returned with a replicated batch recorded, which keeps one
/// recorded batch sharding per enclosing batching level, so that each level takes its own off.
fn unbatched_placement(placement: Placement, batch_axis: Option<usize>) -> Option<Placement> {
    let Some(batch_axis) = batch_axis else {
        return placement.with_batch_sharding(ShardingDimension::Replicated).ok();
    };
    let sharding = placement.sharding();
    if batch_axis >= sharding.rank() {
        return None;
    }
    let mut dimensions = sharding.dimensions().to_vec();
    let batch_sharding = dimensions.remove(batch_axis);
    let sharding = sharding.with_dimensions(dimensions).ok()?;
    placement.with_sharding(sharding).ok()?.without_layout().with_batch_sharding(batch_sharding).ok()
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::collections::{BTreeMap, HashSet};
    use std::hash::DefaultHasher;
    use std::rc::Rc;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayIrOperation, ArrayOperation, DataType, Device, DeviceMesh, Dimension, DimensionBounds, DimensionValue,
        DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, Shape, StridedLayout, TiledLayout, UnreducedKind,
    };
    use crate::axes::{NamedAxes, NamedAxis};
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, DifferentiationError, NothingSavable, TransposableOperation,
        TranspositionContext, differentiate_at,
    };
    use crate::macros::{check_operation_partial_evaluation, check_operation_type_inference};
    use crate::operations::comparisons::Compare;
    use crate::operations::constants::constant::Constant;
    use crate::operations::constants::fill::Fill;
    use crate::operations::constants::iota::Iota;
    use crate::operations::constants::one::One;
    use crate::operations::constants::zero::Zero;
    use crate::operations::control_flow::condition::Condition;
    use crate::operations::control_flow::scan::{ScanOptions, scan};
    use crate::operations::control_flow::r#while::{WhileOptions, r#while};
    use crate::operations::reductions::{Reduce, ReductionKind};
    use crate::operations::{CompositeJvpRule, CompositeOperation, LinearCallOperation, RematerializeOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationContext, PartialEvaluationInput, ResidualPolicyReference};
    use crate::programs::{
        AttachedRegions, Concretizable, EffectClasses, EmptyRegionDriver, ProgramBuilder, Provenance, ProvenanceScope,
    };
    use crate::tracing::TracingContext;

    use super::*;

    /// Memory that placement fixtures move values into.
    const PINNED_HOST: Memory = Memory::Host { pinned: true };

    /// Array that dispatches through [`RecordingContext`] and reports `placement` as where it concretely lives.
    #[derive(Clone, Debug)]
    struct PlacedArray {
        array: Array,
        placement: Option<Placement>,
        context: RecordingContext,
    }

    impl Display for PlacedArray {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.array)
        }
    }

    impl Parameter for PlacedArray {}

    impl Typed for PlacedArray {
        type Type = ArrayType;

        fn r#type(&self) -> Cow<'_, ArrayType> {
            self.array.r#type()
        }
    }

    impl Value for PlacedArray {
        type Dispatch = ValueDomainDispatch;
        type Domain = RecordingContext;

        fn domain(&self) -> Self::Domain {
            self.context.clone()
        }
    }

    impl Placed for PlacedArray {
        fn placement(&self) -> Option<Placement> {
            self.placement.clone()
        }
    }

    /// Eager context that records the operations bound in it and interprets them over reference arrays. The outputs
    /// of a placement live on its target, or where its input lives when it has no target, and all other outputs, as
    /// well as those of a placement on a device target, have no placement.
    #[derive(Clone, Debug, Default)]
    struct RecordingContext(Rc<RefCell<Vec<ArrayOperation<Array>>>>);

    impl Domain for RecordingContext {
        type Type = ArrayType;
        type Value = PlacedArray;
        type Constant = Array;
        type Operation = ArrayOperation<Array>;
    }

    impl Context for RecordingContext {
        fn lift(&self, array: Array) -> Result<PlacedArray, ProgramError> {
            Ok(PlacedArray { array, placement: None, context: self.clone() })
        }

        fn bind<'r, O: Into<Self::Operation>, R: Into<AttachedRegions<'r, Self::Constant, Self::Operation>>>(
            &self,
            operation: O,
            regions: R,
            inputs: &[PlacedArray],
        ) -> Result<Vec<PlacedArray>, ProgramError>
        where
            Self::Constant: 'r,
            Self::Operation: 'r,
        {
            let operation = operation.into();
            let placement = match &operation {
                ArrayOperation::Place(place) => match place.target() {
                    Some(PlacementRequest::Resolved(target)) => Some(target.clone()),
                    Some(PlacementRequest::Device { .. }) => None,
                    Some(PlacementRequest::InPlace { memory, .. }) => {
                        inputs.first().and_then(|input| input.placement.clone()).map(|placement| match memory {
                            Some(memory) => placement.with_memory(*memory),
                            None => placement,
                        })
                    }
                    None => inputs.first().and_then(|input| input.placement.clone()),
                },
                _ => None,
            };
            let inputs = inputs.iter().map(|input| input.array.clone()).collect::<Vec<_>>();
            let outputs =
                EagerContext::<Array, ArrayOperation<Array>>::new().bind(operation.clone(), regions, &inputs)?;
            self.0.borrow_mut().push(operation);
            Ok(outputs
                .into_iter()
                .map(|array| PlacedArray { array, placement: placement.clone(), context: self.clone() })
                .collect())
        }

        fn is_eager(&self) -> bool {
            true
        }

        fn provenance(&self) -> Provenance {
            Provenance::unknown()
        }

        fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
            function()
        }

        fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
            function()
        }
    }

    impl NamedAxes for RecordingContext {
        fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
            None
        }

        fn named_axes(&self) -> Vec<(String, NamedAxis)> {
            Vec::new()
        }
    }

    impl Zero<PlacedArray> for RecordingContext {
        fn zero(&self, r#type: &ArrayType) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().zero(r#type)?)
        }
    }

    impl One<PlacedArray> for RecordingContext {
        fn one(&self, r#type: &ArrayType) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().one(r#type)?)
        }
    }

    impl Iota<PlacedArray> for RecordingContext {
        fn iota(&self, r#type: &ArrayType, dimension: usize) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().iota(r#type, dimension)?)
        }
    }

    impl Fill<u64, PlacedArray> for RecordingContext {
        fn fill(&self, r#type: &ArrayType, value: u64) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().fill(r#type, value)?)
        }
    }

    impl Fill<f64, PlacedArray> for RecordingContext {
        fn fill(&self, r#type: &ArrayType, value: f64) -> Result<PlacedArray, ProgramError> {
            self.lift(EagerContext::<Array>::new().fill(r#type, value)?)
        }
    }

    impl Concretizable<bool> for PlacedArray {
        fn concretize(&self) -> Result<bool, ProgramError> {
            self.array.concretize()
        }
    }

    /// Returns a device mesh with the explicit axis `x` of size 2 over the devices 0 and 1.
    fn mesh_x() -> DeviceMesh {
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0), Device::new(1, 0)],
        )
        .unwrap()
    }

    /// Returns a device mesh with the explicit axis `y` of size 4 over the devices 2 through 5, which has another
    /// topology than [`mesh_x`].
    fn mesh_y() -> DeviceMesh {
        DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 4, MeshAxisType::Explicit).unwrap()]).unwrap(),
            (2..6).map(|id| Device::new(id, 0)).collect(),
        )
        .unwrap()
    }

    /// Returns a placement on `mesh` whose sharding over the logical mesh of `mesh` has the provided `dimensions`.
    fn placement(mesh: &DeviceMesh, dimensions: Vec<ShardingDimension>) -> Placement {
        Placement::new(mesh.clone(), Sharding::new(mesh.logical_mesh().clone(), dimensions).unwrap()).unwrap()
    }

    /// Returns where a rank `rank` value inside a manual region over the devices of [`mesh_x`] concretely lives (i.e.,
    /// a replicated placement over the axis `x` as a manual axis, along which the value varies).
    fn manual_region_placement(rank: usize) -> Placement {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        Placement::new(mesh_x(), Sharding::replicated(mesh, rank).with_varying_manual_axes(["x"]).unwrap()).unwrap()
    }

    #[test]
    fn test_placement_request() {
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target_placement = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])
            .with_memory(PINNED_HOST)
            .with_layout(layout.clone());

        // A placement target selects the memory and layout of its placement and renders as that placement.
        let target = PlacementRequest::from(target_placement.clone());
        assert_eq!(target, PlacementRequest::Resolved(target_placement.clone()));
        assert_eq!(target.placement(), Some(&target_placement));
        assert_eq!(target.memory(), Some(PINNED_HOST));
        assert_eq!(target.layout(), Some(&layout));
        assert_eq!(target.to_string(), target_placement.to_string());

        // A device converts into a device target without a memory and a layout, which names no placement.
        let device = Device::new(3, 0);
        let target = PlacementRequest::from(device);
        assert_eq!(target, PlacementRequest::Device { device, memory: None, layout: None });
        assert_eq!(target.placement(), None);
        assert_eq!(target.memory(), None);
        assert_eq!(target.layout(), None);
        assert_eq!(target.to_string(), "device(3)");
        let target = PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        assert_eq!(target.placement(), None);
        assert_eq!(target.memory(), Some(PINNED_HOST));
        assert_eq!(target.layout(), Some(&layout));
        assert_eq!(target.to_string(), "device(3, memory=Host[Pinned], layout=tiled{0})");

        // A memory converts into an in-place target without a layout, which names no devices and no placement.
        let target = PlacementRequest::from(PINNED_HOST);
        assert_eq!(target, PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: None });
        assert_eq!(target.placement(), None);
        assert_eq!(target.memory(), Some(PINNED_HOST));
        assert_eq!(target.layout(), None);
        assert_eq!(target.to_string(), "in_place(memory=Host[Pinned])");
        let target = PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        assert_eq!(target.placement(), None);
        assert_eq!(target.memory(), Some(PINNED_HOST));
        assert_eq!(target.layout(), Some(&layout));
        assert_eq!(target.to_string(), "in_place(memory=Host[Pinned], layout=tiled{0})");
        let target = PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) };
        assert_eq!(target.memory(), None);
        assert_eq!(target.to_string(), "in_place(layout=tiled{0})");
    }

    #[test]
    fn test_placement_request_without_layout() {
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target_placement = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(PINNED_HOST);
        assert_eq!(
            PlacementRequest::from(target_placement.clone().with_layout(layout.clone())).without_layout(),
            PlacementRequest::Resolved(target_placement),
        );
        let device = Device::new(3, 0);
        assert_eq!(
            PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: Some(layout.clone()) }
                .without_layout(),
            PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: None },
        );

        // An in-place target that names only a layout names nothing without it.
        assert_eq!(
            PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) }.without_layout(),
            PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: None },
        );
        assert_eq!(
            PlacementRequest::InPlace { memory: None, layout: Some(layout) }.without_layout(),
            PlacementRequest::InPlace { memory: None, layout: None },
        );
    }

    #[test]
    fn test_placement_request_target_memory() {
        // A device target without a memory selects the default memory of the device, and every other target selects
        // the memory that it names, if any.
        let device = Device::new(0, 0);
        let resolved = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        assert_eq!(PlacementRequest::from(device).target_memory(), Some(Memory::Device));
        assert_eq!(PlacementRequest::from(device).memory(), None);
        assert_eq!(
            PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: None }.target_memory(),
            Some(PINNED_HOST),
        );
        assert_eq!(PlacementRequest::from(resolved.clone()).target_memory(), None);
        assert_eq!(PlacementRequest::from(resolved.with_memory(PINNED_HOST)).target_memory(), Some(PINNED_HOST));
        assert_eq!(PlacementRequest::from(PINNED_HOST).target_memory(), Some(PINNED_HOST));
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        assert_eq!(PlacementRequest::InPlace { memory: None, layout: Some(layout) }.target_memory(), None);
    }

    #[test]
    fn test_placement_request_moves() {
        // Every form except the in-place one can move a value to other devices or redistribute it.
        assert!(PlacementRequest::from(placement(&mesh_x(), vec![ShardingDimension::Replicated])).moves());
        assert!(PlacementRequest::from(Device::new(0, 0)).moves());
        assert!(!PlacementRequest::from(PINNED_HOST).moves());
    }

    #[test]
    fn test_placement_request_resolve_source() {
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let concrete = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(Memory::Device);
        let other = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let device = PlacementRequest::from(Device::new(7, 0));

        // Without a stated source, the source is the concrete placement of the value, if it has one.
        assert_eq!(PlacementRequest::resolve_source(None, None), None);
        assert_eq!(
            PlacementRequest::resolve_source(None, Some(concrete.clone())),
            Some(PlacementRequest::Resolved(concrete.clone())),
        );

        // An in-place source is completed with the devices and the sharding of the concrete placement, keeping the
        // storage that it names, and it stays as it is without a concrete placement.
        let in_place = PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        assert_eq!(
            PlacementRequest::resolve_source(Some(in_place.clone()), Some(concrete.clone())),
            Some(PlacementRequest::Resolved(concrete.clone().with_memory(PINNED_HOST).with_layout(layout.clone()))),
        );
        assert_eq!(
            PlacementRequest::resolve_source(
                Some(PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) }),
                Some(concrete.clone()),
            ),
            Some(PlacementRequest::Resolved(concrete.clone().with_layout(layout))),
        );
        assert_eq!(PlacementRequest::resolve_source(Some(in_place.clone()), None), Some(in_place));

        // Every other source names devices itself, so it is kept, whatever the concrete placement is.
        assert_eq!(
            PlacementRequest::resolve_source(Some(PlacementRequest::Resolved(other.clone())), Some(concrete.clone())),
            Some(PlacementRequest::Resolved(other)),
        );
        assert_eq!(PlacementRequest::resolve_source(Some(device.clone()), Some(concrete)), Some(device));
    }

    #[test]
    fn test_place_operation() {
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let operation = PlaceOperation::new(target.clone(), source.clone());
        assert_eq!(operation.name(), PLACE_OPERATION_NAME);
        assert_eq!(operation.source(), Some(&PlacementRequest::Resolved(source.clone())));
        assert_eq!(operation.target(), Some(&PlacementRequest::Resolved(target.clone())));
        assert!(operation.may_alias());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                ]"
            },
        );

        // An unknown source is not rendered.
        let operation = PlaceOperation::new(target, None);
        assert_eq!(operation.source(), None);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                ]"
            },
        );

        // An omitted target leaves the input where it is and is not rendered either, so that a placement without a
        // target and a source renders as its name alone.
        let operation = PlaceOperation::new(None, source.clone());
        assert_eq!(operation.source(), Some(&PlacementRequest::Resolved(source.clone())));
        assert_eq!(operation.target(), None);
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ]"
            },
        );
        let operation = PlaceOperation::new(None, None);
        assert_eq!(operation.source(), None);
        assert_eq!(operation.target(), None);
        assert_eq!(operation.to_string(), PLACE_OPERATION_NAME);
    }

    #[test]
    fn test_place_operation_device_target() {
        // A device target is rendered by the ID of its device, followed by its memory and layout when it names them.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target =
            PlacementRequest::Device { device: Device::new(3, 0), memory: Some(Memory::Device), layout: Some(layout) };
        let operation = PlaceOperation::new(target.clone(), source.clone()).with_may_alias(false);
        assert_eq!(operation.source(), Some(&PlacementRequest::Resolved(source.clone())));
        assert_eq!(operation.target(), Some(&target));
        assert!(!operation.may_alias());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target=device(3, memory=Device, layout=tiled{0}),
                    may_alias=false,
                ]"
            },
        );
        let operation = PlaceOperation::new(PlacementRequest::from(Device::new(3, 0)), None);
        assert_eq!(operation.target(), Some(&PlacementRequest::from(Device::new(3, 0))));
        assert_eq!(operation.to_string(), "place [target=device(3)]");

        // A bare device converts into a device target or source without a memory and a layout, and a device source is
        // rendered like a device target.
        assert_eq!(PlaceOperation::new(Device::new(3, 0), None), operation);
        let operation =
            PlaceOperation::new(placement(&mesh_x(), vec![ShardingDimension::Replicated]), Device::new(4, 0));
        assert_eq!(operation.source(), Some(&PlacementRequest::from(Device::new(4, 0))));
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source=device(4),
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}]}},
                ]"
            },
        );
        let source = PlacementRequest::Device { device: Device::new(4, 0), memory: Some(PINNED_HOST), layout: None };
        let operation = PlaceOperation::new(Device::new(3, 0), source.clone());
        assert_eq!(operation.source(), Some(&source));
        assert_eq!(operation.to_string(), "place [source=device(4, memory=Host[Pinned]), target=device(3)]");
    }

    #[test]
    fn test_place_operation_in_place_target() {
        // An in-place target is rendered by the storage that it names, and a bare memory converts into one.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target = PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout) };
        let operation = PlaceOperation::new(target.clone(), source.clone()).with_may_alias(false);
        assert_eq!(operation.source(), Some(&PlacementRequest::Resolved(source)));
        assert_eq!(operation.target(), Some(&target));
        assert!(!operation.may_alias());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target=in_place(memory=Host[Pinned], layout=tiled{0}),
                    may_alias=false,
                ]"
            },
        );
        let operation = PlaceOperation::new(PINNED_HOST, None);
        assert_eq!(operation.source(), None);
        assert_eq!(operation.target(), Some(&PlacementRequest::from(PINNED_HOST)));
        assert_eq!(operation.to_string(), "place [target=in_place(memory=Host[Pinned])]");

        // An in-place source is constructed as given, so that type inference can reject it with its own error.
        let operation = PlaceOperation::new(None, PINNED_HOST);
        assert_eq!(operation.source(), Some(&PlacementRequest::from(PINNED_HOST)));
        assert_eq!(operation.to_string(), "place [source=in_place(memory=Host[Pinned])]");
    }

    #[test]
    fn test_place_operation_with_may_alias() {
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let aliasing = PlaceOperation::new(target.clone(), source.clone());
        let copying = aliasing.clone().with_may_alias(false);
        assert!(!copying.may_alias());
        assert_eq!(copying.source(), Some(&PlacementRequest::Resolved(source.clone())));
        assert_eq!(copying.target(), Some(&PlacementRequest::Resolved(target.clone())));
        assert!(copying.clone().with_may_alias(true).may_alias());

        // The copy policy takes part in equality and hashing.
        assert_eq!(copying, copying.clone());
        assert_ne!(copying, aliasing);
        assert_eq!(copying.clone().with_may_alias(true), aliasing);
        let operations = HashSet::from([aliasing.clone(), copying.clone()]);
        assert_eq!(operations.len(), 2);
        assert!(operations.contains(&aliasing));
        assert!(operations.contains(&copying));

        // A forced copy is rendered after the placements, or alone when both are omitted.
        assert_eq!(
            copying.to_string(),
            indoc! {"
                place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    may_alias=false,
                ]"
            },
        );
        assert_eq!(
            PlaceOperation::new(target, None).with_may_alias(false).to_string(),
            indoc! {"
                place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    may_alias=false,
                ]"
            },
        );
        assert_eq!(PlaceOperation::new(None, None).with_may_alias(false).to_string(), "place [may_alias=false]");
    }

    #[test]
    fn test_place_operation_validate_reduction_state() {
        let validate = |input: &ArrayType, target: Option<PlacementRequest>, source: Option<PlacementRequest>| {
            PlaceOperation::validate_reduction_state(input, input.sharding(), false, target.as_ref(), source.as_ref())
                .map_err(|error| error.to_string())
        };
        let mesh = mesh_x();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let unreduced = replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let reduced = replicated.clone().with_reduced_axes(["x"]).unwrap();
        let with_sharding =
            |sharding: &Sharding| ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let request = |mesh: &DeviceMesh, sharding: &Sharding| {
            PlacementRequest::from(Placement::new(mesh.clone(), sharding.clone()).unwrap())
        };
        let other_mesh =
            DeviceMesh::new(mesh.logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let undefined_mesh = mesh_y();
        let undefined_replicated = Sharding::replicated(undefined_mesh.logical_mesh().clone(), 1);
        let device = PlacementRequest::from(Device::new(2, 0));
        let memory = PlacementRequest::from(PINNED_HOST);
        let cannot_change = format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        );
        let undefined_axis = format!(
            "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as an undefined \
             axis",
        );

        // A placement never changes reduction state, so a target over any mesh must restate a pending reduction over
        // an axis that its mesh defines identically, and a device target cannot combine the contributions of the
        // devices along an unreduced axis larger than one. An in-place target keeps the reduction state.
        let input = with_sharding(&unreduced);
        assert_eq!(validate(&input, None, None), Ok(()));
        assert_eq!(validate(&input, Some(request(&other_mesh, &unreduced)), None), Ok(()));
        assert_eq!(validate(&input, Some(memory.clone()), None), Ok(()));
        assert_eq!(validate(&input, Some(request(&mesh, &replicated)), None), Err(cannot_change.clone()));
        assert_eq!(
            validate(&input, Some(request(&undefined_mesh, &undefined_replicated)), None),
            Err(undefined_axis.clone()),
        );
        assert_eq!(
            validate(&input, Some(device.clone()), None),
            Err(format!(
                "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger than one \
                 (i.e., `x`) onto a single device; use a reduction (or the corresponding collective) first",
            )),
        );

        // A completed reduction is complete on one device, so a device target accepts it, while a placement target
        // must still restate it.
        let input = with_sharding(&reduced);
        assert_eq!(validate(&input, Some(device.clone()), None), Ok(()));
        assert_eq!(validate(&input, Some(request(&other_mesh, &replicated)), None), Err(cannot_change));
        assert_eq!(
            validate(&input, Some(request(&undefined_mesh, &undefined_replicated)), None),
            Err(undefined_axis.clone()),
        );

        // A source describes the input, so it restates its reduction state as well, and an input without a tracked
        // sharding has none to restate.
        let input = with_sharding(&unreduced);
        assert_eq!(validate(&input, None, Some(request(&other_mesh, &unreduced))), Ok(()));
        assert_eq!(
            validate(&input, None, Some(request(&mesh, &replicated))),
            Err(format!(
                "`{PLACE_OPERATION_NAME}` source sharding `{replicated}` does not match the reduction state of the \
                 input type `{input}`",
            )),
        );
        assert_eq!(validate(&input, None, Some(request(&undefined_mesh, &undefined_replicated))), Err(undefined_axis));
        let input = ArrayType::new_static(DataType::F32, [4]);
        assert_eq!(validate(&input, Some(device), Some(request(&mesh, &replicated))), Ok(()));
        assert_eq!(
            validate(&input, Some(request(&mesh, &unreduced)), None),
            Err(format!(
                "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding \
                 collective) instead",
            )),
        );
    }

    #[test]
    fn test_place_operation_hash() {
        let hash = |operation: &PlaceOperation| {
            let mut hasher = DefaultHasher::new();
            operation.hash(&mut hasher);
            hasher.finish()
        };

        // An operation with a placement target, or without a target, hashes like its fields with an optional placement
        // as its target, in declaration order, so that the hashes of programs without device targets are stable.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let mut hasher = DefaultHasher::new();
        Some(source.clone()).hash(&mut hasher);
        Some(target.clone()).hash(&mut hasher);
        false.hash(&mut hasher);
        let placed = PlaceOperation::new(target, source.clone()).with_may_alias(false);
        assert_eq!(hash(&placed), hasher.finish());
        let mut hasher = DefaultHasher::new();
        Option::<Placement>::None.hash(&mut hasher);
        Option::<Placement>::None.hash(&mut hasher);
        true.hash(&mut hasher);
        let identity = PlaceOperation::new(None, None);
        assert_eq!(hash(&identity), hasher.finish());

        // A device target takes part in equality and hashing like a placement target.
        let device = PlaceOperation::new(PlacementRequest::from(Device::new(3, 0)), None);
        let other_device = PlaceOperation::new(PlacementRequest::from(Device::new(4, 0)), None);
        assert_eq!(device, device.clone());
        assert_eq!(hash(&device), hash(&device.clone()));
        assert_ne!(device, other_device);
        assert_ne!(hash(&device), hash(&identity));
        let operations = HashSet::from([placed.clone(), identity.clone(), device.clone(), other_device.clone()]);
        assert_eq!(operations.len(), 4);
        assert!(operations.contains(&device));
        assert!(operations.contains(&other_device));

        // A device source takes part in equality and hashing as well, and it hashes like a device target in its
        // position, so that it differs from an absent source and from a placement source.
        let device_source = PlaceOperation::new(None, Device::new(3, 0));
        let mut hasher = DefaultHasher::new();
        Some(PlacementRequest::from(Device::new(3, 0))).hash(&mut hasher);
        Option::<Placement>::None.hash(&mut hasher);
        true.hash(&mut hasher);
        assert_eq!(hash(&device_source), hasher.finish());
        assert_ne!(device_source, identity);
        assert_ne!(hash(&device_source), hash(&identity));
        assert_ne!(hash(&device_source), hash(&PlaceOperation::new(None, source)));
        assert_ne!(hash(&device_source), hash(&PlaceOperation::new(None, Device::new(4, 0))));

        // An in-place target hashes as a whole after the discriminant of `Some`, like a device target, so that it
        // differs from an absent target, from a device target in the same memory, and from another memory.
        let in_place = PlaceOperation::new(PINNED_HOST, None);
        let mut hasher = DefaultHasher::new();
        Option::<Placement>::None.hash(&mut hasher);
        Some(PlacementRequest::from(PINNED_HOST)).hash(&mut hasher);
        true.hash(&mut hasher);
        assert_eq!(hash(&in_place), hasher.finish());
        assert_eq!(in_place, in_place.clone());
        assert_eq!(hash(&in_place), hash(&in_place.clone()));
        let device_in_pinned_host = PlaceOperation::new(
            PlacementRequest::Device { device: Device::new(0, 0), memory: Some(PINNED_HOST), layout: None },
            None,
        );
        assert_ne!(hash(&in_place), hash(&identity));
        assert_ne!(hash(&in_place), hash(&device_in_pinned_host));
        assert_ne!(hash(&in_place), hash(&PlaceOperation::new(Memory::Device, None)));
        let operations = HashSet::from([in_place.clone(), identity.clone(), device_in_pinned_host.clone()]);
        assert_eq!(operations.len(), 3);
        assert!(operations.contains(&in_place));
    }

    #[test]
    fn test_place_operation_type_inference() {
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let x_input_type = input_type.clone().with_sharding(x_sharding.clone()).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);

        // The output adopts the target sharding, also when the input is tracked on another mesh, and keeps its storage
        // unless the target names one.
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), None),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }, {
                input_types = [x_input_type.clone().with_memory(PINNED_HOST)],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST)],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!("`{PLACE_OPERATION_NAME}` target sharding rank (1) does not match the input rank (2)"),
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone().with_memory(PINNED_HOST).with_layout(layout.clone()), None),
            cases = [{
                input_types = [x_input_type.clone()],
                output_types = [input_type
                    .clone()
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST)
                    .with_layout(layout)],
            }],
        );

        // Types cannot record placement over auto axes, so the output drops them from the target sharding.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_target = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(auto_target, None),
            cases = [{
                input_types = [x_input_type.clone()],
                output_types = [input_type.clone().with_sharding(Sharding::replicated(auto_mesh.clone(), 1)).unwrap()],
            }],
        );

        // A target layout must address the output shape.
        let rank_3_layout = Layout::Tiled(TiledLayout::new(vec![0, 1, 2], vec![]));
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone().with_layout(rank_3_layout), None),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: tiled layout rank 3 does not match array \
                     rank 1",
                ),
            }],
        );

        // A source must agree with the tracked sharding of the input without its auto axes.
        let auto_source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let replicated_source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), auto_source),
            cases = [{
                input_types = [input_type.clone().with_sharding(Sharding::replicated(auto_mesh.clone(), 1)).unwrap()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), replicated_source.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }, {
                input_types = [x_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type `{x_input_type}`",
                    replicated_source.sharding(),
                ),
            }],
        );
        let rank_2_source = placement(&mesh_x(), vec![ShardingDimension::Replicated; 2]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), rank_2_source),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{PLACE_OPERATION_NAME}` source sharding rank (2) does not match the input rank (1)"),
            }],
        );

        // A source restates the reduction state of the input exactly, also when the input type tracks no sharding
        // (i.e., when the input has no reduction state), so that every accepted placement can be transposed.
        let unreduced_sharding =
            Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_source = Placement::new(mesh_x(), unreduced_sharding.clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), unreduced_source.clone()),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{unreduced_sharding}` does not match the reduction \
                     state of the input type `{input_type}`",
                ),
            }],
        );
        let max_input_type = input_type
            .clone()
            .with_sharding(unreduced_sharding.clone().with_unreduced_kind(UnreducedKind::Max).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Placement::new(
                    mesh_x(),
                    unreduced_sharding.clone().with_unreduced_kind(UnreducedKind::Max).unwrap(),
                )
                .unwrap(),
                unreduced_source,
            ),
            cases = [{
                input_types = [max_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{unreduced_sharding}` does not match the reduction \
                     state of the input type `{max_input_type}`",
                ),
            }],
        );
        let reduced_sharding =
            Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Placement::new(mesh_x(), reduced_sharding.clone()).unwrap(),
                replicated_source.clone(),
            ),
            cases = [{
                input_types = [input_type.clone().with_sharding(reduced_sharding.clone()).unwrap()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the reduction state of the input \
                     type `{}`",
                    replicated_source.sharding(),
                    input_type.clone().with_sharding(reduced_sharding).unwrap(),
                ),
            }],
        );

        // Values inside manual regions, which vary along manual axes or carry reduction state over them, cannot be
        // moved, and nothing can be moved into such a region either.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_sharding = Sharding::replicated(manual_mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
        let manual_error = format!(
            "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes or carry reduction state \
             over them",
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), None),
            cases = [{
                input_types = [input_type.clone().with_sharding(varying_sharding.clone()).unwrap()],
                error = manual_error.clone(),
            }, {
                input_types = [input_type
                    .clone()
                    .with_sharding(Sharding::replicated(manual_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
                    .unwrap()],
                error = manual_error.clone(),
            }, {
                input_types = [input_type
                    .clone()
                    .with_sharding(Sharding::replicated(manual_mesh.clone(), 1).with_reduced_axes(["x"]).unwrap())
                    .unwrap()],
                error = manual_error.clone(),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Placement::new(mesh_x(), varying_sharding.clone()).unwrap(), None),
            cases = [{
                input_types = [input_type.clone()],
                error = manual_error.clone(),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), Placement::new(mesh_x(), varying_sharding).unwrap()),
            cases = [{
                input_types = [input_type.clone()],
                error = manual_error,
            }],
        );

        // A global array outside any manual region can be tiled over a manual axis of its mesh, and it can be moved
        // like any other array (e.g., onto a single device), also onto a mesh with manual axes.
        let single_device_mesh =
            LogicalMesh::new(vec![MeshAxis::new("d", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let single_device = Placement::new(
            DeviceMesh::new(single_device_mesh.clone(), vec![Device::new(0, 0)]).unwrap(),
            Sharding::replicated(single_device_mesh.clone(), 1),
        )
        .unwrap();
        let manual_tiled_sharding =
            Sharding::new(manual_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                single_device,
                Placement::new(mesh_x(), manual_tiled_sharding.clone()).unwrap(),
            ),
            cases = [{
                input_types = [input_type.clone().with_sharding(manual_tiled_sharding.clone()).unwrap()],
                output_types = [input_type.clone().with_sharding(Sharding::replicated(single_device_mesh, 1)).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(Placement::new(mesh_x(), manual_tiled_sharding.clone()).unwrap(), None),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(manual_tiled_sharding).unwrap()],
            }],
        );

        // A placement restates the reduction state of its input exactly.
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let reduction_error = format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Placement::new(
                    mesh_x(),
                    Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap(),
                )
                .unwrap(),
                None,
            ),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type.clone()],
            }, {
                input_types = [x_input_type],
                error = reduction_error,
            }],
        );

        // Reduction state is restated by axis name, so the target and source meshes must define the axes that carry
        // it like the input's mesh does. Otherwise, the unreduced input would silently lose its pending reduction over
        // an auto axis `x` (over which types record no reduction state), or change it over an axis of another size or
        // over an axis that the target mesh does not define.
        let explicit_axis = MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), None),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as an \
                     undefined axis",
                ),
            }],
        );
        let auto_axis = MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap();
        let wide_axis = MeshAxis::new("x", 4, MeshAxisType::Explicit).unwrap();
        let wide_mesh = LogicalMesh::new(vec![wide_axis.clone()]).unwrap();
        let wide_devices = DeviceMesh::new(wide_mesh.clone(), (2..6).map(|id| Device::new(id, 0)).collect()).unwrap();
        let auto_unreduced =
            Placement::new(mesh_x(), Sharding::replicated(auto_mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap())
                .unwrap();
        let wide_unreduced =
            Placement::new(wide_devices, Sharding::replicated(wide_mesh, 1).with_unreduced_axes(["x"]).unwrap())
                .unwrap();
        let unreduced_placement = Placement::new(mesh_x(), unreduced_input_type.sharding().unwrap().clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(auto_unreduced.clone(), None),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{auto_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(wide_unreduced.clone(), None),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{wide_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(unreduced_placement.clone(), auto_unreduced),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{auto_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(unreduced_placement.clone(), wide_unreduced),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `{explicit_axis}` as \
                     `{wide_axis}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(unreduced_placement.clone(), unreduced_placement),
            cases = [{
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type],
            }],
        );

        // The operation cannot own nested regions.
        assert_eq!(
            PlaceOperation::new(target, None).infer_output_types(
                std::slice::from_ref(&input_type),
                &[RegionInterface::new(vec![], vec![], EffectClasses::NONE)],
            ),
            Err(TypeError::invalid("expected 0 regions but got 1")),
        );
    }

    #[test]
    fn test_place_operation_type_inference_without_target() {
        // Without a target, the value stays where it is, so the output keeps the sharding, memory, and layout of the
        // input, also when the input varies along a manual mesh axis or carries reduction state.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned_input_type =
            input_type.clone().with_sharding(x_sharding).unwrap().with_memory(PINNED_HOST).with_layout(layout);
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(manual_mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, None).with_may_alias(false),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }, {
                input_types = [pinned_input_type.clone()],
                output_types = [pinned_input_type.clone()],
            }, {
                input_types = [varying_input_type.clone()],
                output_types = [varying_input_type.clone()],
            }, {
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type],
            }, {
                input_types = [],
                error = "expected 1 input but got 0",
            }],
        );

        // A placement without a target moves nothing, so it also accepts a source that describes a value inside a
        // manual region (e.g., the concrete placement of such a value, which differentiation infers as its source).
        let varying_source = Placement::new(mesh_x(), varying_input_type.sharding().unwrap().clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, varying_source),
            cases = [{
                input_types = [varying_input_type.clone()],
                output_types = [varying_input_type],
            }],
        );

        // A given source is still validated against the input type.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let replicated_input_type =
            input_type.clone().with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1)).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, source.clone()),
            cases = [{
                input_types = [pinned_input_type.clone()],
                output_types = [pinned_input_type],
            }, {
                input_types = [input_type.clone()],
                output_types = [input_type],
            }, {
                input_types = [replicated_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type \
                     `{replicated_input_type}`",
                    source.sharding(),
                ),
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!("`{PLACE_OPERATION_NAME}` source sharding rank (1) does not match the input rank (2)"),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_device_target() {
        // A device target names no mesh, so the output keeps the sharding that the input type tracks over a mesh of one
        // device, an input that the type tracks over more devices is gathered (i.e., replicated over the canonical mesh
        // of one device), and the output takes the memory and layout of the target. A layout must still address the
        // input shape, and a value inside a manual region still cannot be moved. Reduction state over a mesh axis
        // larger than one cannot live on one device.
        let device = Device::new(0, 0);
        let gathered = Sharding::replicated(LogicalMesh::single_device(), 1);
        let single_device_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let single_device_input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(single_device_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned_input_type = input_type.clone().with_sharding(x_sharding).unwrap().with_memory(PINNED_HOST);
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying_input_type = input_type
            .clone()
            .with_sharding(Sharding::replicated(manual_mesh, 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                PlacementRequest::Device { device, memory: Some(Memory::Device), layout: Some(layout.clone()) },
                None,
            ),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_layout(layout.clone())],
            }, {
                input_types = [single_device_input_type.clone()],
                output_types = [single_device_input_type.with_layout(layout.clone())],
            }, {
                input_types = [pinned_input_type.clone()],
                output_types = [
                    input_type.clone().with_sharding(gathered.clone()).unwrap().with_layout(layout.clone()),
                ],
            }, {
                input_types = [unreduced_input_type],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger than one \
                     (i.e., `x`) onto a single device; use a reduction (or the corresponding collective) first",
                ),
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: tiled layout rank 1 does not match array rank 2",
                ),
            }, {
                input_types = [varying_input_type],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes or carry reduction \
                     state over them",
                ),
            }],
        );

        // Without a memory, the output is in the default memory of the device, as with JAX's eager `device_put` onto a
        // device, so a pinned host input moves to device memory, and without a layout, it keeps the layout of the input
        // while that layout still describes its one shard. A given source is still validated against the input type.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let replicated_input_type =
            input_type.clone().with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1)).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(PlacementRequest::from(device), source.clone()),
            cases = [{
                input_types = [pinned_input_type.clone()],
                output_types = [input_type.clone().with_sharding(gathered.clone()).unwrap()],
            }, {
                input_types = [pinned_input_type.clone().with_layout(layout.clone())],
                output_types = [input_type.clone().with_sharding(gathered).unwrap().with_layout(layout.clone())],
            }, {
                input_types = [replicated_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{}` does not match the input type \
                     `{replicated_input_type}`",
                    source.sharding(),
                ),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_device_target_reduction_state() {
        // A device target leaves a value that lives wholly on one device, where a pending reduction over mesh axes of
        // size one is complete, so the output loses its unreduced and reduced state over such axes (including the kind
        // of its unreduced axes), while the dimensions of a value on a mesh of one device keep their sharding. A
        // completed reduction is complete on every device, so reduced axes of any size are cleared as well, and a value
        // on a mesh of more devices is gathered (i.e., replicated over the canonical mesh of one device). Pending
        // contributions over a larger axis are rejected, also when other axes of the mesh have size one, and so is
        // reduction state over a manual axis of size one, which describes a value inside a manual region.
        let device = Device::new(0, 0);
        let small_mesh = LogicalMesh::new(vec![
            MeshAxis::new("w", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let small_sharded = Sharding::new(small_mesh, vec![ShardingDimension::sharded(["w"])]).unwrap();
        let sharded = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["z"])]).unwrap();
        let gathered = Sharding::replicated(LogicalMesh::single_device(), 1);
        let with_sharding = |sharding: Sharding| input_type.clone().with_sharding(sharding).unwrap();
        let unreduced = small_sharded.clone().with_unreduced_axes(["x"]).unwrap().with_reduced_axes(["y"]).unwrap();
        let maximum = small_sharded
            .clone()
            .with_unreduced_axes(["x", "y"])
            .unwrap()
            .with_unreduced_kind(UnreducedKind::Max)
            .unwrap();
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 1, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual = Sharding::replicated(manual_mesh, 1).with_unreduced_axes(["m"]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(device, None),
            cases = [{
                input_types = [with_sharding(unreduced)],
                output_types = [with_sharding(small_sharded.clone())],
            }, {
                input_types = [with_sharding(maximum)],
                output_types = [with_sharding(small_sharded.clone())],
            }, {
                input_types = [with_sharding(
                    Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x", "z"]).unwrap(),
                )],
                output_types = [with_sharding(gathered.clone())],
            }, {
                input_types = [with_sharding(
                    sharded.with_unreduced_axes(["x"]).unwrap().with_reduced_axes(["y"]).unwrap(),
                )],
                output_types = [with_sharding(gathered)],
            }, {
                input_types = [with_sharding(
                    Sharding::replicated(mesh, 1)
                        .with_unreduced_axes(["x", "z"])
                        .unwrap()
                        .with_reduced_axes(["y"])
                        .unwrap(),
                )],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger than one \
                     (i.e., `z`) onto a single device; use a reduction (or the corresponding collective) first",
                ),
            }, {
                input_types = [with_sharding(manual)],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` cannot move values that vary along manual mesh axes or carry reduction \
                     state over them",
                ),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_device_source() {
        // A device source names no sharding, so it is checked against the memory of the input type and the shape that
        // its layout must address when it names them, with or without a target, and it accepts any sharding over a
        // mesh of one device. It states that the input lives on that one device, so it is rejected for an input that
        // its type tracks over a mesh of more devices.
        let device = Device::new(0, 0);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let single_device_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(single_device_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let split_input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(
                Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        let source = PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: Some(layout) };
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), source.clone()),
            cases = [{
                input_types = [input_type.clone().with_memory(PINNED_HOST)],
                output_types = [
                    input_type.clone().with_sharding(target.sharding().clone()).unwrap().with_memory(PINNED_HOST),
                ],
            }, {
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type \
                     `{input_type}`",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, source),
            cases = [{
                input_types = [input_type.clone().with_memory(PINNED_HOST)],
                output_types = [input_type.clone().with_memory(PINNED_HOST)],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2]).with_memory(PINNED_HOST)],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 1 does not match array \
                     rank 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, device),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4])],
                output_types = [ArrayType::new_static(DataType::F32, [4])],
            }, {
                input_types = [split_input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` device source names one device, but the input type `{split_input_type}` \
                     is tracked over 2 devices",
                ),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_in_place_target() {
        // An in-place target leaves the value where it is, so the output keeps the sharding of the input exactly,
        // including reduction state over mesh axes larger than one and variation along manual mesh axes, and takes the
        // memory and the layout of the target. A layout must still address the input shape.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let unreduced_input_type = input_type
            .clone()
            .with_sharding(
                Sharding::replicated(mesh_x().logical_mesh().clone(), 1)
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_unreduced_kind(UnreducedKind::Max)
                    .unwrap(),
            )
            .unwrap();
        let varying_input_type =
            input_type.clone().with_sharding(manual_region_placement(1).sharding().clone()).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) },
                None,
            ),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_memory(PINNED_HOST).with_layout(layout.clone())],
            }, {
                input_types = [unreduced_input_type.clone()],
                output_types = [unreduced_input_type.clone().with_memory(PINNED_HOST).with_layout(layout.clone())],
            }, {
                input_types = [varying_input_type.clone()],
                output_types = [varying_input_type.clone().with_memory(PINNED_HOST).with_layout(layout.clone())],
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: tiled layout rank 1 does not match array rank 2",
                ),
            }],
        );

        // A target that names only a memory keeps the layout of the input, and a source that describes a value inside
        // a manual region is accepted together with such an input, because nothing moves.
        let pinned_input_type = varying_input_type.clone().with_memory(PINNED_HOST).with_layout(layout.clone());
        check_operation_type_inference!(
            operation = PlaceOperation::new(Memory::Device, manual_region_placement(1)),
            cases = [{
                input_types = [pinned_input_type.clone()],
                output_types = [pinned_input_type.with_memory(Memory::Device)],
            }],
        );

        // An in-place target that names neither a memory nor a layout changes nothing, which an absent target already
        // expresses, so it is rejected.
        check_operation_type_inference!(
            operation = PlaceOperation::new(PlacementRequest::InPlace { memory: None, layout: None }, None),
            cases = [{
                input_types = [input_type.clone()],
                error = format!("`{PLACE_OPERATION_NAME}` in-place target must name a memory or a layout"),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_in_place_source() {
        // An in-place source states the storage in which the input is expected, so the memory and the layout that it
        // names are checked against the input type, with or without a target, and one that names neither says nothing,
        // so it is rejected.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_memory(PINNED_HOST);
        let source = PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        check_operation_type_inference!(
            operation = PlaceOperation::new(Memory::Device, source.clone()),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_memory(Memory::Device)],
            }, {
                input_types = [input_type.clone().with_memory(Memory::Device)],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{}`",
                    input_type.clone().with_memory(Memory::Device),
                ),
            }, {
                input_types = [ArrayType::new_static(DataType::F32, [4, 2]).with_memory(PINNED_HOST)],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 1 does not match array \
                     rank 2",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, PlacementRequest::InPlace { memory: None, layout: None }),
            cases = [{
                input_types = [input_type],
                error = format!("`{PLACE_OPERATION_NAME}` in-place source must name a memory or a layout"),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_source_storage() {
        // The memory and the layout that a source names are checked against the input type in the same way for every
        // form of the source, so that no path that stages a placement in another form accepts a source that another
        // path rejects. A layout must be the layout that the input type declares, if it declares one.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let other_layout = Layout::Strided(StridedLayout::new(vec![4]));
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let resolved = placement(&mesh, vec![ShardingDimension::Replicated]);
        let device = Device::new(0, 0);
        let memory_error = format!(
            "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{input_type}`",
        );
        let declared_input_type = input_type.clone().with_layout(layout.clone());
        let layout_error = format!(
            "`{PLACE_OPERATION_NAME}` source layout `{other_layout}` does not match the input type \
             `{declared_input_type}`",
        );
        for source in [
            PlacementRequest::Resolved(resolved.clone().with_memory(PINNED_HOST)),
            PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: None },
            PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: None },
        ] {
            check_operation_type_inference!(
                operation = PlaceOperation::new(None, source),
                cases = [{
                    input_types = [input_type.clone()],
                    error = memory_error.clone(),
                }, {
                    input_types = [input_type.clone().with_memory(PINNED_HOST)],
                    output_types = [input_type.clone().with_memory(PINNED_HOST)],
                }],
            );
        }
        for source in [
            PlacementRequest::Resolved(resolved.clone().with_layout(other_layout.clone())),
            PlacementRequest::Device { device, memory: None, layout: Some(other_layout.clone()) },
            PlacementRequest::InPlace { memory: None, layout: Some(other_layout.clone()) },
        ] {
            check_operation_type_inference!(
                operation = PlaceOperation::new(None, source),
                cases = [{
                    input_types = [declared_input_type.clone()],
                    error = layout_error.clone(),
                }, {
                    input_types = [input_type.clone()],
                    output_types = [input_type.clone()],
                }],
            );
        }
    }

    #[test]
    fn test_place_operation_type_inference_shard_layouts() {
        // A layout describes each shard, so the byte strides `[8, 4]` of a dense `f32[4, 2]` column shard are accepted
        // for an `f32[4, 4]` value that is split into two column shards, although they would overlap the rows of the
        // whole value, and strides that overlap the elements of a shard are rejected. A tiled layout is valid for any
        // shape of its rank.
        let mesh = mesh_x();
        let columns = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(columns.clone()).unwrap();
        let shard_strides = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let overlapping = Layout::Strided(StridedLayout::new(vec![4, 4]));
        let target = Placement::new(mesh.clone(), columns.clone()).unwrap();
        for target in [PlacementRequest::from(target), PlacementRequest::InPlace { memory: None, layout: None }] {
            let with_layout = |layout: &Layout| match target.clone() {
                PlacementRequest::Resolved(placement) => PlacementRequest::from(placement.with_layout(layout.clone())),
                _ => PlacementRequest::InPlace { memory: None, layout: Some(layout.clone()) },
            };
            check_operation_type_inference!(
                operation = PlaceOperation::new(with_layout(&shard_strides), None),
                cases = [{
                    input_types = [input_type.clone()],
                    output_types = [input_type.clone().with_layout(shard_strides.clone())],
                }],
            );
            assert_eq!(
                PlaceOperation::new(with_layout(&overlapping), None).infer_output_types(&[input_type.clone()], &[]),
                Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: strided layout stride 4 on axis 1 is smaller \
                     than the 16-byte span occupied by more minor axes and may alias array elements",
                ))),
            );
        }
        check_operation_type_inference!(
            operation = PlaceOperation::new(None, PlacementRequest::InPlace { memory: None, layout: Some(shard_strides.clone()) }),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone()],
            }],
        );

        // Without a target layout, the output keeps the layout of the input only while it still describes each shard of
        // the output. The column shard strides describe the shards of a target that keeps the columns split and of an
        // in-place target, but not the one whole shard of a value that a device target gathers or that a placement
        // target replicates, also when it leaves the columns unconstrained (i.e., `{?}`), which backends store
        // replicated, so that output declares no layout, whereas a tiled layout describes every shard.
        let strided_input_type = input_type.clone().with_layout(shard_strides.clone());
        let replicated = placement(&mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let unconstrained = placement(&mesh, vec![ShardingDimension::Replicated, ShardingDimension::Unconstrained]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(unconstrained.clone(), None),
            cases = [{
                input_types = [strided_input_type.clone()],
                output_types = [input_type.clone().with_sharding(unconstrained.sharding().clone()).unwrap()],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(unconstrained.clone().with_layout(shard_strides.clone()), None),
            cases = [{
                input_types = [input_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: strided layout stride 8 on axis 0 is smaller \
                     than the 16-byte span occupied by more minor axes and may alias array elements",
                ),
            }],
        );
        let gathered = Sharding::replicated(LogicalMesh::single_device(), 2);
        let tiled = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        for (target, output_type) in [
            (
                PlacementRequest::from(Placement::new(mesh.clone(), columns.clone()).unwrap()),
                strided_input_type.clone(),
            ),
            (PlacementRequest::from(PINNED_HOST), strided_input_type.clone().with_memory(PINNED_HOST)),
            (
                PlacementRequest::from(replicated.clone()),
                input_type.clone().with_sharding(replicated.sharding().clone()).unwrap(),
            ),
            (PlacementRequest::from(Device::new(0, 0)), input_type.clone().with_sharding(gathered.clone()).unwrap()),
        ] {
            check_operation_type_inference!(
                operation = PlaceOperation::new(target, None),
                cases = [{
                    input_types = [strided_input_type.clone()],
                    output_types = [output_type],
                }],
            );
        }
        check_operation_type_inference!(
            operation = PlaceOperation::new(Device::new(0, 0), None),
            cases = [{
                input_types = [input_type.clone().with_layout(tiled.clone())],
                output_types = [input_type.clone().with_sharding(gathered).unwrap().with_layout(tiled)],
            }],
        );

        // Along an auto axis larger than one, a value may be split without its type recording it, so the shards of an
        // in-place output are not known, and only the rank of a strided layout is checked by the type rule.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_type = ArrayType::new_static(DataType::F32, [4, 4])
            .with_sharding(Sharding::replicated(auto_mesh, 2))
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(PlacementRequest::InPlace { memory: None, layout: Some(overlapping.clone()) }, None),
            cases = [{
                input_types = [auto_type.clone()],
                output_types = [auto_type.clone().with_layout(overlapping.clone())],
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                PlacementRequest::InPlace { memory: None, layout: Some(Layout::Strided(StridedLayout::new(vec![4]))) },
                None,
            ),
            cases = [{
                input_types = [auto_type.clone()],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: strided layout rank 1 does not match array \
                     rank 2",
                ),
            }],
        );

        // The concrete sharding of a value describes every mesh axis, so it determines the shards of a concrete value
        // that is replicated along the auto axis, and an in-place target layout or a source layout that overlaps them
        // is rejected, as the placement of a staged value of the same type accepts it only because its shards are not
        // known.
        let concrete = Sharding::replicated(auto_type.sharding().unwrap().mesh().clone(), 2);
        let in_place = PlacementRequest::InPlace { memory: None, layout: Some(overlapping.clone()) };
        let invalid = |end: &str| {
            Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` {end} layout is invalid: strided layout stride 4 on axis 1 is smaller than \
                 the 16-byte span occupied by more minor axes and may alias array elements",
            )))
        };
        assert_eq!(
            PlaceOperation::infer_output_type(&auto_type, Some(&concrete), Some(&in_place), None),
            invalid("target")
        );
        assert_eq!(
            PlaceOperation::infer_output_type(&auto_type, Some(&concrete), None, Some(&in_place)),
            invalid("source")
        );
        assert_eq!(PlaceOperation::infer_output_type(&auto_type, None, None, Some(&in_place)), Ok(auto_type.clone()));
    }

    #[test]
    fn test_place_operation_infer_output_type_concrete_reduction_state() {
        // The type of a value omits its reduction state over auto axes, which its concrete sharding records. With the
        // concrete sharding, a target that restates that state (e.g., the placement of the value itself) is accepted
        // and changes nothing, and a target that drops it, a device target, and a source that does not restate it are
        // rejected, as for the same state over an explicit axis. Without it, only the state that the type records is
        // checked, so the same requests are judged against a value without reduction state, and a source is compared
        // over the axes that the type records.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(auto_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let input_type =
            ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(auto_mesh.clone(), 0)).unwrap();
        let replicated = Sharding::replicated(auto_mesh, 0);
        let changed = Err(TypeError::invalid(format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        )));
        for (state, concrete) in [
            ("unreduced", replicated.clone().with_unreduced_axes(["x"]).unwrap()),
            ("reduced", replicated.clone().with_reduced_axes(["x"]).unwrap()),
        ] {
            let identity = PlacementRequest::from(Placement::new(mesh.clone(), concrete.clone()).unwrap());
            let plain = PlacementRequest::from(Placement::new(mesh.clone(), replicated.clone()).unwrap());
            let device = PlacementRequest::from(Device::new(0, 0));
            let infer =
                |concrete_sharding: Option<&Sharding>, target: &PlacementRequest, source: Option<&PlacementRequest>| {
                    PlaceOperation::infer_output_type(&input_type, concrete_sharding, Some(target), source)
                };
            assert_eq!(infer(Some(&concrete), &identity, None), Ok(input_type.clone()), "{state}");
            assert_eq!(infer(Some(&concrete), &identity, Some(&identity)), Ok(input_type.clone()), "{state}");
            assert_eq!(infer(Some(&concrete), &plain, None), changed.clone(), "{state}");
            assert_eq!(
                infer(Some(&concrete), &identity, Some(&plain)),
                Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` source sharding `{replicated}` does not match the reduction state of the \
                     input type `{input_type}`",
                ))),
                "{state}",
            );
            let gathered = ArrayType::scalar(DataType::F32)
                .with_sharding(Sharding::replicated(LogicalMesh::single_device(), 0))
                .unwrap();
            let device_result = match state {
                "unreduced" => Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger than one \
                     (i.e., `x`) onto a single device; use a reduction (or the corresponding collective) first",
                ))),
                _ => Ok(gathered.clone()),
            };
            assert_eq!(infer(Some(&concrete), &device, None), device_result, "{state}");
            assert_eq!(infer(None, &identity, None), changed.clone(), "{state}");
            assert_eq!(infer(None, &plain, Some(&identity)), Ok(input_type.clone()), "{state}");
            assert_eq!(infer(None, &device, None), Ok(gathered), "{state}");
        }
    }

    #[test]
    fn test_place_operation_type_inference_shapes() {
        // Rank-0 arrays are placed with rank-0 shardings, which can only replicate them.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let scalar_target = placement(&mesh_y(), vec![]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(scalar_target.clone(), placement(&mesh_x(), vec![])),
            cases = [{
                input_types = [scalar_type.clone()],
                output_types = [scalar_type.clone().with_sharding(scalar_target.sharding().clone()).unwrap()],
            }],
        );

        // Placement needs only the rank of the input, so a dynamic extent keeps its variable and gets the target
        // sharding like a static one.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(2, Some(8)).unwrap(),
            ))]),
        );
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), None),
            cases = [{
                input_types = [dynamic_type.clone()],
                output_types = [dynamic_type.clone().with_sharding(target.sharding().clone()).unwrap()],
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_uneven_splits() {
        // A target can only split a static extent into shards of equal size, also along auto axes, which the output
        // type does not record, and along every mesh axis that shards the dimension together.
        let mixed_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("a", 3, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let mixed_devices = DeviceMesh::new(mixed_mesh.clone(), (0..6).map(|id| Device::new(id, 0)).collect()).unwrap();
        let target = Placement::new(
            mixed_devices,
            Sharding::new(
                mixed_mesh.clone(),
                vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x", "a"])],
            )
            .unwrap(),
        )
        .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(target, None),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5, 12])],
                output_types = [ArrayType::new_static(DataType::I32, [5, 12])
                    .with_sharding(
                        Sharding::new(
                            mixed_mesh,
                            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
                        )
                        .unwrap(),
                    )
                    .unwrap()],
            }, {
                input_types = [ArrayType::new_static(DataType::I32, [5, 4])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #1 with extent 4 into 6 shards, which \
                     do not divide it evenly",
                ),
            }],
        );
        check_operation_type_inference!(
            operation = PlaceOperation::new(placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]), None),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 5 into 2 shards, which \
                     do not divide it evenly",
                ),
            }],
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(
                Placement::new(mesh_x(), Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
                    .unwrap(),
                None,
            ),
            cases = [{
                input_types = [ArrayType::new_static(DataType::I32, [5])],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 5 into 2 shards, which \
                     do not divide it evenly",
                ),
            }],
        );

        // A dynamic extent is only known when a backend places a concrete value, so type inference accepts any split
        // of it, also one that its bounds do not admit evenly.
        let dynamic_type = ArrayType::new(
            DataType::I32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(3, Some(5)).unwrap(),
            ))]),
        );
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        check_operation_type_inference!(
            operation = PlaceOperation::new(target.clone(), None),
            cases = [{
                input_types = [dynamic_type.clone()],
                output_types = [dynamic_type.with_sharding(target.sharding().clone()).unwrap()],
            }],
        );

        // The even split is the one check of the type rule that `PlaceOperation::infer_output_type` does not make, so
        // that backends can split a concrete value unevenly, and it is made after every other check.
        let input_type = ArrayType::new_static(DataType::I32, [5]);
        let target = PlacementRequest::Resolved(placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]));
        assert_eq!(
            PlaceOperation::infer_output_type(&input_type, None, Some(&target), None),
            Ok(input_type.clone().with_sharding(target.placement().unwrap().sharding().clone()).unwrap()),
        );
        let target =
            target.placement().unwrap().clone().with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![])));
        check_operation_type_inference!(
            operation = PlaceOperation::new(target, None),
            cases = [{
                input_types = [input_type],
                error = format!(
                    "`{PLACE_OPERATION_NAME}` target layout is invalid: tiled layout rank 2 does not match array rank \
                     1",
                ),
            }],
        );
    }

    #[test]
    fn test_place_operation_type_inference_unconstrained_target() {
        // A target may leave a dimension unconstrained, which the output type records exactly, and which backends store
        // replicated. Its auto axes are dropped from the output type like those of any other target.
        let input_type = ArrayType::new_static(DataType::F32, [4, 2]);
        let explicit = Sharding::new(
            mesh_x().logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Unconstrained],
        )
        .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(Placement::new(mesh_x(), explicit.clone()).unwrap(), None),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.clone().with_sharding(explicit).unwrap()],
            }],
        );
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto = Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::Unconstrained])
            .unwrap();
        check_operation_type_inference!(
            operation = PlaceOperation::new(Placement::new(mesh_x(), auto.clone()).unwrap(), None),
            cases = [{
                input_types = [input_type.clone()],
                output_types = [input_type.with_sharding(auto.without_auto_axes()).unwrap()],
            }],
        );
    }

    #[test]
    fn test_place_operation_interpretation() {
        // Reference arrays have no devices, so interpretation keeps the payload unchanged while recording the
        // placement in the value's type so that it matches the declared output type.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(PINNED_HOST);
        let operation = PlaceOperation::new(target.clone(), None);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let outputs = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(
            outputs,
            vec![
                Array::from_elements(
                    ArrayType::new_static(DataType::F32, [2])
                        .with_sharding(target.sharding().clone())
                        .unwrap()
                        .with_memory(PINNED_HOST),
                    &[1.0f32, 2.0],
                )
                .unwrap(),
            ],
        );
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));
        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // A forced copy gives the output storage of its own with the same elements.
        let copies = operation
            .clone()
            .with_may_alias(false)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(copies, outputs);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copies[0].shared_storage_bytes()));

        // Rank-0 arrays are placed like any other array.
        let scalar_target = placement(&mesh_x(), vec![]);
        let scalar = Array::scalar(3.0f32).unwrap();
        assert_eq!(
            PlaceOperation::new(scalar_target.clone(), None).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&scalar),
            ),
            Ok(vec![
                Array::from_elements(
                    ArrayType::scalar(DataType::F32).with_sharding(scalar_target.sharding().clone()).unwrap(),
                    &[3.0f32],
                )
                .unwrap(),
            ]),
        );

        // Interpreting the operation in a staging context stages it, also for an input with a dynamic extent.
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new(
                "n",
                DimensionBounds::new(2, Some(8)).unwrap(),
            ))]),
        );
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_context_and_named_axes(
            |context, input| Ok(operation.interpret(context, &EmptyRegionDriver, &[input])?.remove(0)),
            dynamic_type.clone(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            output_type,
            dynamic_type.with_sharding(target.sharding().clone()).unwrap().with_memory(PINNED_HOST),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n] .
                let %1:f32[n][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Host[Pinned]},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_interpretation_without_target() {
        // Without a target, a reference array keeps its type and shares its storage, unless the copy policy forbids
        // sharing it.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let outputs = PlaceOperation::new(None, None)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs, vec![input.clone()]);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));

        // A placement without a target and a source that forces a copy is still staged, so that replaying the program
        // on concrete values copies them.
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&input, None, None, false),
            input.r#type().into_owned(),
        )
        .unwrap();
        assert_eq!(output_type, input.r#type().into_owned());
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [may_alias=false] %0
                in (%1)"
            },
        );
        let output = program.interpret(input.clone()).unwrap();
        assert_eq!(output, input);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_interpretation_device_target() {
        // Reference arrays have no devices, so a device target records its memory and layout in the type, keeps the
        // sharding that is tracked over a mesh of one device, and re-encodes the storage for the new layout.
        let single_device_mesh =
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let x_sharding = Sharding::new(single_device_mesh, vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::U8, [2]).with_sharding(x_sharding).unwrap();
        let input = Array::from_logical_bytes(input_type.clone(), &[10, 20]).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![-1]));
        let device = Device::new(0, 0);
        let target = PlacementRequest::Device { device, memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        assert_eq!(
            PlaceOperation::new(target, None).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![Array::new(input_type.with_memory(PINNED_HOST).with_layout(layout), vec![20, 10]).unwrap()]),
        );

        // Without a memory and a layout, the type is unchanged and the storage is shared.
        let outputs = PlaceOperation::new(PlacementRequest::from(device), None)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs, vec![input.clone()]);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_interpretation_in_place_target() {
        // Reference arrays have no devices, so an in-place target records its memory and layout in the type, keeps the
        // sharding, and re-encodes the storage for the new layout.
        let x_sharding =
            Sharding::new(mesh_x().logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let input_type = ArrayType::new_static(DataType::U8, [2]).with_sharding(x_sharding).unwrap();
        let input = Array::from_logical_bytes(input_type.clone(), &[10, 20]).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![-1]));
        let target = PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) };
        assert_eq!(
            PlaceOperation::new(target, None).interpret(
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Ok(vec![
                Array::new(input_type.clone().with_memory(PINNED_HOST).with_layout(layout), vec![20, 10]).unwrap()
            ]),
        );

        // Without a layout, the storage is shared.
        let outputs = PlaceOperation::new(PINNED_HOST, None)
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs, vec![Array::new(input_type.with_memory(PINNED_HOST), vec![10, 20]).unwrap()]);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), outputs[0].shared_storage_bytes()));

        // Nothing moves, so a value inside a manual region is accepted, and the placement records where the value
        // concretely lives as its source.
        let source = manual_region_placement(0);
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(source.sharding().clone()).unwrap();
        let array = Array::from_elements(input_type.clone(), &[3.0f32]).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array, placement: Some(source.clone()), context: context.clone() };
        let outputs = PlaceOperation::new(PINNED_HOST, None).interpret(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].array, Array::from_elements(input_type.with_memory(PINNED_HOST), &[3.0f32]).unwrap());
        assert_eq!(outputs[0].placement, Some(source.clone().with_memory(PINNED_HOST)));
        assert_eq!(context.0.borrow().as_slice(), &[ArrayOperation::Place(PlaceOperation::new(PINNED_HOST, source))],);
    }

    #[test]
    fn test_place_operation_interpretation_manual_region() {
        // A value inside a manual region cannot be moved, but a placement without a target moves nothing, so it accepts
        // such a value and records where the value concretely lives as its source.
        let source = manual_region_placement(0);
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(source.sharding().clone()).unwrap();
        let array = Array::from_elements(input_type, &[3.0f32]).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(source.clone()), context: context.clone() };
        let outputs = PlaceOperation::new(None, None).interpret(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs[0].array, array);
        assert_eq!(outputs[0].placement, Some(source.clone()));
        assert_eq!(context.0.borrow().as_slice(), &[ArrayOperation::Place(PlaceOperation::new(None, source))]);
    }

    #[test]
    fn test_place_operation_partial_evaluation() {
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(target.clone(), None),
            inputs = [input.clone()],
            expected = input.place(&target, None).unwrap(),
        );

        // A placement that forces a copy stays residual also when its input is known, so that every execution of the
        // specialized program copies.
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(None, None).with_may_alias(false),
            cases = [{
                inputs = [(@known, input.clone())],
                outputs = [(@residual, input.clone())],
                residual_instructions = 1,
            }, {
                inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                outputs = [(@residual, input)],
                residual_instructions = 1,
            }],
        );

        // Known inputs fold through the reference interpreter, which re-encodes them for a target layout.
        let input = Array::from_logical_bytes(ArrayType::new_static(DataType::U8, [2]), &[10, 20]).unwrap();
        let layout = Layout::Strided(StridedLayout::new(vec![-1]));
        check_operation_partial_evaluation!(
            operation = PlaceOperation::new(target.clone().with_layout(layout.clone()), None),
            inputs = [input],
            expected = Array::new(
                ArrayType::new_static(DataType::U8, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_layout(layout),
                vec![20, 10],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias() {
        // Specializing a program on the known input of a placement that forces a copy keeps the placement, with or
        // without a target, so that every execution of the specialized program returns storage of its own.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                Ok(vec![
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x[0], None, None, false)?,
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x[0],
                        Some(&PlacementRequest::from(target.clone())),
                        None,
                        false,
                    )?,
                ])
            },
            vec![input.r#type().into_owned()],
        )
        .unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [may_alias=false] %0
                    %2:f32[2][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                        target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                        may_alias=false,
                    ] %0
                in (%1, %2)"
            },
        );
        assert_eq!(evaluation.inputs(), &[PartialEvaluationInput::Known(input.clone())]);
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![input.clone(), input.place(&target, None).unwrap()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[1].shared_storage_bytes(), second[1].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        // A placement whose output may share storage with its input still folds.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                Ok(vec![<ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x[0],
                    Some(&PlacementRequest::from(target.clone())),
                    None,
                    true,
                )?])
            },
            vec![input.r#type().into_owned()],
        )
        .unwrap();
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![input.place(&target, None).unwrap()]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_device_target() {
        // A placement on a device target follows the copy policy like any other placement. It stays in a specialized
        // program when it forces a copy, and it folds otherwise.
        let target = PlacementRequest::Device { device: Device::new(0, 0), memory: Some(PINNED_HOST), layout: None };
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let expected =
            Array::from_elements(input.r#type().into_owned().with_memory(PINNED_HOST), &[1.0f32, 2.0]).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let program = |may_alias: bool| {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |x: Vec<_>| {
                    Ok(vec![<ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x[0],
                        Some(&target),
                        None,
                        may_alias,
                    )?])
                },
                vec![input.r#type().into_owned()],
            )
            .unwrap();
            program
        };

        let evaluation = program(false).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2]@Host[Pinned] = place [target=device(0, memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![expected.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        let evaluation = program(true).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![expected]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_in_place_target() {
        // A placement on an in-place target follows the copy policy like any other placement. It stays in a
        // specialized program when it forces a copy, and it folds otherwise.
        let target = PlacementRequest::from(PINNED_HOST);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let expected =
            Array::from_elements(input.r#type().into_owned().with_memory(PINNED_HOST), &[1.0f32, 2.0]).unwrap();
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let program = |may_alias: bool| {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |x: Vec<_>| {
                    Ok(vec![<ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x[0],
                        Some(&target),
                        None,
                        may_alias,
                    )?])
                },
                vec![input.r#type().into_owned()],
            )
            .unwrap();
            program
        };

        let evaluation = program(false).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2]@Host[Pinned] = place [target=in_place(memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![expected.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        // Linearization evaluates the known primal once, so the copy is made once in its primal program, and the
        // tangent program places the tangent in place with the same copy policy.
        let linearization = program(false).linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2]@Host[Pinned] = place [target=in_place(memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2]@Host[Pinned] = place [target=in_place(memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );

        let evaluation = program(true).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![expected]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_nested() {
        // The split rule of a `scan` partitions its body with the copy folding setting of specialization, so a copy of
        // a known stacked input stays in the residual scan instead of being made once in the known scan.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                let (carry, copies) = scan(
                    |_, carry, item| {
                        let copy =
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, false)?;
                        Ok((carry + copy.clone(), copy))
                    },
                    x[0].clone(),
                    x[1].clone(),
                    ScanOptions::new(),
                )?;
                Ok(vec![carry, copies])
            },
            vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::F32, [3])],
        )
        .unwrap();
        let stacked = Array::vector(vec![1.0f32, 2.0, 3.0]).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayType::scalar(DataType::F32)),
                PartialValue::Known(stacked.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[3] .
                let %2:f32[], %3:f32[3] = scan [carry_count=1, iteration_count=3, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = place [may_alias=false] %2
                            %4:f32[] = add %1 %3
                        in (%4, %3)
                    },
                ]
                in (%2, %3)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[Array::scalar(0.0f32).unwrap()]).unwrap();
        let second = evaluation.interpret(&context, &[Array::scalar(0.0f32).unwrap()]).unwrap();
        assert_eq!(first, vec![Array::scalar(6.0f32).unwrap(), stacked]);
        assert!(!Arc::ptr_eq(first[1].shared_storage_bytes(), second[1].shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_projected() {
        // An array IR family holds the placement through its projected array member, whose own partial evaluation rule
        // decides whether a placement with a known input folds, so a copy stays residual under specialization there
        // too, while a placement whose output may share storage with its input still folds.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let input_type = ArrayIrType::Array(input.r#type().into_owned());
        let (_, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |x: Vec<_>| {
                Ok(vec![
                    <ValueDomainDispatch as PlaceDispatch<ArrayIrType, _>>::place(&x[0], None, None, false)?,
                    <ValueDomainDispatch as PlaceDispatch<ArrayIrType, _>>::place(&x[0], None, None, true)?,
                ])
            },
            vec![input_type],
        )
        .unwrap();
        let known = ArrayIrValue::Array(input.clone());
        let evaluation = program.partially_evaluate(&[PartialValue::Known(known.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [may_alias=false] %0
                in (%1)"
            },
        );
        assert_eq!(evaluation.inputs(), &[PartialEvaluationInput::Known(known.clone())]);
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![known.clone(), known]);
        let [ArrayIrValue::Array(first), ArrayIrValue::Array(_)] = first.as_slice() else { unreachable!() };
        let [ArrayIrValue::Array(second), ArrayIrValue::Array(_)] = second.as_slice() else { unreachable!() };
        assert!(!Arc::ptr_eq(first.shared_storage_bytes(), second.shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first.shared_storage_bytes(), input.shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_scan() {
        // A `scan` whose inputs are all known folds whole only when its body has no work that must stay residual. A
        // carry that the body copies keeps the whole scan residual under specialization, so that every execution of the
        // specialized program returns storage of its own, as every execution of the original program does.
        let trace = |may_alias: bool| {
            TracingContext::<Array, ArrayOperation<Array>>::trace(
                move |x: Vec<_>| {
                    let (carry, _) = scan(
                        |_, carry, item| {
                            let copy = <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                                &carry, None, None, may_alias,
                            )?;
                            Ok((copy, item))
                        },
                        x[0].clone(),
                        x[1].clone(),
                        ScanOptions::new(),
                    )?;
                    Ok(vec![carry])
                },
                vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::F32, [1])],
            )
            .unwrap()
            .1
        };
        let carry = Array::scalar(4.0f32).unwrap();
        let items = Array::vector(vec![1.0f32]).unwrap();
        let program = trace(false);
        let first = program.interpret(vec![carry.clone(), items.clone()]).unwrap();
        let second = program.interpret(vec![carry.clone(), items.clone()]).unwrap();
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));

        let knowledge = [PartialValue::Known(carry.clone()), PartialValue::Known(items.clone())];
        let evaluation = program.partially_evaluate(&knowledge).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[1] .
                let %2:f32[], %3:f32[1] = scan [carry_count=1, iteration_count=1, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = place [may_alias=false] %1
                        in (%3, %2)
                    },
                ]
                in (%2)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![carry.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));

        // Linearization evaluates the known primal once, so the same copy is made once in its primal program, whose
        // outputs are all known, and the tangent program only copies the tangent.
        let linearization = program.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[1] .
                let %2:f32[], %3:f32[1] = scan [carry_count=1, iteration_count=1, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = place [may_alias=false] %1
                        in (%3, %2)
                    },
                ]
                in (%2)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[1] .
                let %2:f32[], %3:f32[1] = scan [carry_count=1, iteration_count=1, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:f32[] .
                        let %3:f32[] = place [may_alias=false] %1
                        in (%3, %2)
                    },
                ]
                in (%2)"
            },
        );

        // A scan whose body places the carry without forcing a copy still folds whole.
        let evaluation = trace(true).partially_evaluate(&knowledge).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![carry]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_condition() {
        // A `condition` whose inputs are all known but whose predicate selects no branch at partial evaluation time
        // (e.g., the stacked predicate of a `scan` body, which is known to the partition of that body but not
        // constant) folds whole only when neither branch has work that must stay residual. A branch that copies the
        // carry keeps the condition residual in the partition of the body, so the whole scan stays residual under
        // specialization, and every execution of the specialized program returns storage of its own.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                let (carry, _) = scan(
                    |_, carry, predicate| {
                        let next = predicate.condition(
                            |value| {
                                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&value, None, None, false)
                            },
                            Ok,
                            carry,
                        )?;
                        Ok((next, predicate))
                    },
                    x[0].clone(),
                    x[1].clone(),
                    ScanOptions::new(),
                )?;
                Ok(vec![carry])
            },
            vec![ArrayType::scalar(DataType::F32), ArrayType::new_static(DataType::Boolean, [1])],
        )
        .unwrap();
        let carry = Array::scalar(4.0f32).unwrap();
        let predicates = Array::vector(vec![true]).unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(carry.clone()), PartialValue::Known(predicates)])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[], %1:bool[1] .
                let %2:f32[], %3:bool[1] = scan [carry_count=1, iteration_count=1, reverse=false] %0 %1 [
                    body={
                        lambda %0:i64[], %1:f32[], %2:bool[] .
                        let %3:f32[] = condition %2 %1 [
                            true={
                                lambda %0:f32[] .
                                let %1:f32[] = place [may_alias=false] %0
                                in (%1)
                            },
                            false={
                                lambda %0:f32[] .
                                in (%0)
                            },
                        ]
                        in (%3, %2)
                    },
                ]
                in (%2)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![carry]);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_while() {
        // A `while` whose inputs are all known folds whole only when neither its condition nor its body has work that
        // must stay residual. A body that copies a state value keeps the whole loop residual under specialization, so
        // that every execution of the specialized program returns storage of its own.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                let (_, value) = r#while(
                    |(count, _)| count.less_than(&count.context().lift(Array::scalar(1i64)?)?),
                    |(count, value)| {
                        let one = count.context().lift(Array::scalar(1i64)?)?;
                        let copy =
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&value, None, None, false)?;
                        Ok((count + one, copy))
                    },
                    (x[0].clone(), x[1].clone()),
                    WhileOptions::new(),
                )?;
                Ok(vec![value])
            },
            vec![ArrayType::scalar(DataType::I64), ArrayType::scalar(DataType::F32)],
        )
        .unwrap();
        let value = Array::scalar(4.0f32).unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(0i64).unwrap()),
                PartialValue::Known(value.clone()),
            ])
            .unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:i64[], %1:f32[] .
                let %2:i64[], %3:f32[] = while %0 %1 [
                    condition={
                        lambda %0:i64[], %1:f32[] .
                        let %2:i64[] = const 1
                            %3:bool[] = compare [direction=LessThan] %0 %2
                        in (%3)
                    },
                    body={
                        lambda %0:i64[], %1:f32[] .
                        let %2:i64[] = const 1
                            %3:i64[] = add %0 %2
                            %4:f32[] = place [may_alias=false] %1
                        in (%3, %4)
                    },
                ]
                in (%3)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![value]);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_rematerialize() {
        // A `rematerialize` call whose inputs are all known folds whole only when its body has no work that must stay
        // residual. A body that copies its input keeps the whole call residual under specialization, so that every
        // execution of the specialized program returns storage of its own, as every execution of the original program
        // does.
        let program = |may_alias: bool| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let copy = PlaceOperation::new(None, None).with_may_alias(may_alias);
            let output = builder.add_instruction(copy, Vec::new(), vec![input], None).unwrap()[0];
            let body = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]);
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let body = builder.intern_callee(&Arc::new(body.unwrap()), None).unwrap();
            let call = RematerializeOperation::new(ResidualPolicyReference::new(NothingSavable));
            let output = builder.add_instruction(call, vec![body], vec![input], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let input = Array::scalar(4.0f32).unwrap();
        let program_with_copy = program(false);
        let first = program_with_copy.interpret(vec![input.clone()]).unwrap();
        let second = program_with_copy.interpret(vec![input.clone()]).unwrap();
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));

        let evaluation = program_with_copy.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = rematerialize %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = place [may_alias=false] %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![input.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        // Linearization evaluates the known primal once, so the same copy is made once in its primal program, whose
        // outputs are all known, and the tangent program only copies the tangent.
        let linearization = program_with_copy.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = place [may_alias=false] %0
                in (%1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = rematerialize [differentiated=true] %0 [
                    body={
                        lambda %0:f32[] .
                        let %1:f32[] = place [may_alias=false] %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );

        // A call whose body places its input without forcing a copy still folds whole.
        let evaluation = program(true).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![input]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_linear_call() {
        // A `linear_call` whose inputs are all known folds whole only when its forward region has no work that must
        // stay residual. A forward region that copies its input keeps the whole call residual under specialization,
        // while a copy in its transpose region, which a fold does not execute, does not.
        let program = |forward_may_alias: bool, transpose_may_alias: bool| {
            let region = |may_alias: bool| {
                let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
                let input = builder.add_input(ArrayType::scalar(DataType::F32));
                let copy = PlaceOperation::new(None, None).with_may_alias(may_alias);
                let output = builder.add_instruction(copy, Vec::new(), vec![input], None).unwrap()[0];
                Arc::new(
                    builder
                        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder])
                        .unwrap(),
                )
            };
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let forward = builder.intern_callee(&region(forward_may_alias), None).unwrap();
            let transpose = builder.intern_callee(&region(transpose_may_alias), None).unwrap();
            let output = builder
                .add_instruction(LinearCallOperation::new(0), vec![forward, transpose], vec![input], None)
                .unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let input = Array::scalar(4.0f32).unwrap();
        let program_with_copy = program(false, true);
        let first = program_with_copy.interpret(vec![input.clone()]).unwrap();
        let second = program_with_copy.interpret(vec![input.clone()]).unwrap();
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));

        let evaluation = program_with_copy.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = linear_call [residual_count=0] %0 [
                    forward={
                        lambda %0:f32[] .
                        let %1:f32[] = place [may_alias=false] %0
                        in (%1)
                    },
                    transpose={
                        lambda %0:f32[] .
                        let %1:f32[] = place %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![input.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        // Linearization evaluates the known primal once, so the same copy is made once in its primal program, whose
        // outputs are all known, and the tangent program only copies the tangent.
        let linearization = program_with_copy.linearize().unwrap();
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = place [may_alias=false] %0
                in (%1)"
            },
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = linear_call [residual_count=0] %0 [
                    forward={
                        lambda %0:f32[] .
                        let %1:f32[] = place [may_alias=false] %0
                        in (%1)
                    },
                    transpose={
                        lambda %0:f32[] .
                        let %1:f32[] = place %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );

        // A call whose forward region places its input without forcing a copy still folds whole, even when its
        // transpose region forces a copy.
        let evaluation = program(true, true).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![input.clone()]);
        let evaluation = program(true, false).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![input]);
    }

    #[test]
    fn test_place_operation_partial_evaluation_may_alias_composite() {
        // A composite call whose inputs are all known folds whole only when its primal region has no work that must
        // stay residual. A primal region that copies its input keeps the whole call residual under specialization, so
        // that every execution of the specialized program returns storage of its own.
        let program = |may_alias: bool| {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let copy = PlaceOperation::new(None, None).with_may_alias(may_alias);
            let output = builder.add_instruction(copy, Vec::new(), vec![input], None).unwrap()[0];
            let primal = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]);
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F32));
            let primal = builder.intern_callee(&Arc::new(primal.unwrap()), None).unwrap();
            let call =
                CompositeOperation::<Array, ArrayOperation<Array>>::from_rule_regions(CompositeJvpRule::Absent, false);
            let output = builder.add_instruction(call, vec![primal], vec![input], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let input = Array::scalar(4.0f32).unwrap();
        let program_with_copy = program(false);
        let first = program_with_copy.interpret(vec![input.clone()]).unwrap();
        let second = program_with_copy.interpret(vec![input.clone()]).unwrap();
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));

        let evaluation = program_with_copy.partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = composite %0 [
                    primal={
                        lambda %0:f32[] .
                        let %1:f32[] = place [may_alias=false] %0
                        in (%1)
                    },
                ]
                in (%1)"
            },
        );
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let first = evaluation.interpret(&context, &[]).unwrap();
        let second = evaluation.interpret(&context, &[]).unwrap();
        assert_eq!(first, vec![input.clone()]);
        assert_eq!(second, first);
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), second[0].shared_storage_bytes()));
        assert!(!Arc::ptr_eq(first[0].shared_storage_bytes(), input.shared_storage_bytes()));

        // A call whose primal region places its input without forcing a copy still folds whole.
        let evaluation = program(true).partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert!(evaluation.program().instructions().is_empty());
        assert_eq!(evaluation.interpret(&context, &[]).unwrap(), vec![input]);
    }

    #[test]
    fn test_place_operation_requires_fresh_storage() {
        // Only a placement that forces a copy requires storage of its own on every execution, also when an operation
        // family holds it.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        assert!(!PlaceOperation::new(None, None).requires_fresh_storage());
        assert!(!PlaceOperation::new(target.clone(), None).requires_fresh_storage());
        assert!(PlaceOperation::new(None, None).with_may_alias(false).requires_fresh_storage());
        assert!(PlaceOperation::new(target, None).with_may_alias(false).requires_fresh_storage());
        let copy = ArrayOperation::<Array>::from(PlaceOperation::new(None, None).with_may_alias(false));
        assert!(copy.requires_fresh_storage());
        assert!(ArrayIrOperation::<Array>::from(copy).requires_fresh_storage());
    }

    #[test]
    fn test_place_operation_batching() {
        // The batch item is placed with rank-1 shardings, so the lifted placement extends the target sharding with
        // a replicated batch dimension and the source sharding with the placement that the packed input type records
        // for its batch dimension. The target therefore splits the item dimension of the packed value and not its
        // batch dimension, unlike JAX, which applies the target to the packed value positionally.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |item| item.place(&target, Some(&source)),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [2, 4]),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [2, 4])
                .with_sharding(target.sharding().with_inserted_dimension(0, ShardingDimension::Replicated).unwrap())
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                in (%1)"
            },
        );

        // The target applies to each batch item, so it splits the item dimension wherever the batch dimension is, and
        // the batch dimension stays replicated also when it comes after the item dimension.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |item| item.place(&target, Some(&source)),
                    x,
                    BatchAxis::new(1),
                    BatchAxis::new(1),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [4, 2]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4, 2] .
                let %1:f32[4, 2][sharding={mesh<['y'=4:explicit]>, [{'y'}, {}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}, {}]}},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_may_alias() {
        // The packed value is placed with the copy policy of the batch item placement.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                Ok(batch(
                    move |item| {
                        <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                            &item,
                            Some(&PlacementRequest::from(target.clone())),
                            None,
                            false,
                        )
                    },
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            ArrayType::new_static(DataType::F32, [2, 4]),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_axis_sharding() {
        // An explicit target that records no batch sharding places every batch item where it says, so the batch
        // dimension of the packed value is replicated, as in JAX, whatever the placement of the batch dimension that
        // the batching level selects is. A target that records how its batch is distributed (e.g., the placement of
        // another batch item) puts that distribution back instead.
        let target = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let input =
            ArrayBatch::new(Array::matrix(4, 1, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 4)
            .with_axis_sharding(ShardingDimension::sharded(["y"]));
        let outputs = PlaceOperation::new(target.clone(), None)
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [4, 1])
                    .with_sharding(Sharding::replicated(mesh_y().logical_mesh().clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );

        // A target that records the batch sharding `y` splits the batch dimension along `y`, whatever the batching
        // level selects.
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 4)
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let split_target = target.with_batch_sharding(ShardingDimension::sharded(["y"])).unwrap();
        let outputs = PlaceOperation::new(split_target, None)
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [4, 1])
                    .with_sharding(
                        Sharding::new(
                            mesh_y().logical_mesh().clone(),
                            vec![ShardingDimension::sharded(["y"]), ShardingDimension::Replicated],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_batching_sharded_source() {
        // The packed input is sharded along `x` in its batch dimension, so the source of the lifted placement records
        // that placement for the batch dimension and keeps describing where the packed input lives.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let input_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                Ok(batch(
                    move |item| item.place(&target, Some(&source)),
                    x,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            input_type,
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] .
                let %1:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_identity() {
        // The packed input is split along the auto axis `x` in its batch dimension, which its type does not record. A
        // batched function that places each batch item without a target leaves the packed value where it is, so the
        // lifted placement has no target either and keeps that split, while its source is the placement of the packed
        // input.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh, 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let output = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(context.0.borrow().as_slice(), &[ArrayOperation::Place(PlaceOperation::new(None, packed.clone()))],);
        assert_eq!(output.array, array);
        assert_eq!(output.placement, Some(packed));
    }

    #[test]
    fn test_place_operation_batching_replicated_target() {
        // The packed vector is split along the auto axis `x`, which its type does not record. An explicit replicated
        // target for each batch item is not a request to leave it where it is, so the lifted target replicates the
        // batch dimension, and the packed result is replicated on both devices, as in JAX. Omitting the target instead
        // keeps the split.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 1))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let target = Placement::new(mesh_x(), Sharding::replicated(auto_mesh.clone(), 0)).unwrap();
        let replicated =
            batch(|item| item.place(&target, None), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        let kept = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let replicated_placement = Placement::new(mesh_x(), Sharding::replicated(auto_mesh, 1)).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(replicated_placement.clone(), packed.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, packed.clone())),
            ],
        );
        assert_eq!(replicated.array, array);
        assert_eq!(replicated.placement, Some(replicated_placement));
        assert_eq!(kept.array, array);
        assert_eq!(kept.placement, Some(packed));

        // The same holds when the vector is split along the explicit axis `x`, which its type records, so that the
        // batching level is also split along `x`. The explicit replicated target still replicates the packed result.
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(packed.sharding().clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let target = placement(&mesh_x(), vec![]);
        let replicated =
            batch(|item| item.place(&target, None), input.clone(), BatchAxis::new(0), BatchAxis::new(0), None).unwrap();
        let kept = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let replicated_placement = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(replicated_placement.clone(), packed.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, packed.clone())),
            ],
        );
        assert_eq!(
            replicated.array,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(replicated_placement.sharding().clone())
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        assert_eq!(replicated.placement, Some(replicated_placement));
        assert_eq!(kept.array, array);
        assert_eq!(kept.placement, Some(packed));
    }

    #[test]
    fn test_place_operation_batching_unmapped_input() {
        // An unmapped input has no batch dimension, but a target that records a split batch places the packed value
        // with its batch dimension split, so the input is broadcast along a new batch dimension first and the result
        // is mapped.
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        let input = ArrayBatch::replicated(Array::vector(vec![1.0f32, 2.0]).unwrap());
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let outputs = PlaceOperation::new(target.clone(), None).batch(&context, &EmptyRegionDriver, &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(
                        Sharding::new(
                            mesh_x().logical_mesh().clone(),
                            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                &[1.0f32, 2.0, 1.0, 2.0],
            )
            .unwrap(),
        );

        // A batched function stages the broadcast of the unmapped input followed by the placement of the packed value.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, y)| {
                let target = target.clone();
                Ok(batch(
                    move |(_, y): (_, _)| y.place(&target, None),
                    (x, y),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            (ArrayType::new_static(DataType::F32, [2, 4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:f32[4] .
                let %2:f32[2, 4] = broadcast [output_type=f32[2, 4], output_axes=[1]] %1
                    %3:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = place [
                        target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                            sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    ] %2
                in (%3)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_unmapped_input_replicated() {
        // A replicated batch dimension carries no placement, so an unmapped input stays unmapped when the target
        // records nothing or a replicated batch, and the records are dropped. A source is informative, so a source
        // that records a split batch does not give the input a batch dimension either.
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let split_source = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let expected = array.place(&target, None).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let operations = [
            PlaceOperation::new(target.clone(), None),
            PlaceOperation::new(target.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap(), None),
            PlaceOperation::new(target.clone(), split_source),
        ];
        for operation in operations {
            let input = ArrayBatch::replicated(array.clone());
            let outputs = operation.batch(&context, &EmptyRegionDriver, &[input]).unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis_position(), None);
            assert_eq!(outputs[0].value(), &expected);
        }
    }

    #[test]
    fn test_place_operation_batching_unmapped_input_across_meshes() {
        // The batch is split along the explicit axis `x` of the mesh of the target, while the unmapped input lives on
        // another mesh whose explicit axis is `y`. The new batch dimension of the input is therefore replicated where
        // the input lives, because that mesh has no axis `x`, and the placement then splits it along `x` as the target
        // records.
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        let mesh_y = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(2, 0), Device::new(3, 0)],
        )
        .unwrap();
        let source = placement(&mesh_y, vec![ShardingDimension::Replicated]);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh_y.logical_mesh().clone(), 1))
            .unwrap();
        let input = ArrayBatch::replicated(Array::from_elements(input_type.clone(), &[1.0f32, 2.0]).unwrap());
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2)
            .with_axis_sharding(ShardingDimension::sharded(["x"]));
        let outputs = PlaceOperation::new(target.clone(), source.clone())
            .batch(&context, &EmptyRegionDriver, &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), Some(0));
        assert_eq!(
            outputs[0].value(),
            &Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(
                        Sharding::new(
                            mesh_x().logical_mesh().clone(),
                            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                &[1.0f32, 2.0, 1.0, 2.0],
            )
            .unwrap(),
        );

        // A batched function whose batch is split along `x` stages the broadcast of the unmapped input with its batch
        // dimension replicated over the mesh of `y`, followed by the placement that splits it along `x`.
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, y)| {
                let target = target.clone();
                let source = source.clone();
                Ok(batch(
                    move |(_, y): (_, _)| y.place(&target, Some(&source)),
                    (x, y),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            (
                ArrayType::new_static(DataType::F32, [2, 4])
                    .with_sharding(
                        Sharding::new(
                            mesh_x().logical_mesh().clone(),
                            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ArrayType::new_static(DataType::F32, [4])
                    .with_sharding(Sharding::replicated(mesh_y.logical_mesh().clone(), 1))
                    .unwrap(),
            ),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}], \
                    %1:f32[4][sharding={mesh<['y'=2:explicit]>, [{}]}] .
                let %2:f32[2, 4][sharding={mesh<['y'=2:explicit]>, [{}, {}]}] = broadcast [
                    output_type=f32[2, 4][sharding={mesh<['y'=2:explicit]>, [{}, {}]}],
                    output_axes=[1],
                ] %1
                    %3:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}] = place [
                        source={mesh<['y'=2:explicit]>, devices=[2, 3], sharding={mesh<['y'=2:explicit]>, [{}, {}]}},
                        target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    ] %2
                in (%3)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_layouts() {
        // A layout describes every dimension of the placed value, so a mapped input cannot be placed with one, whereas
        // a replicated input keeps its rank and is placed with the unchanged target.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_layout(layout.clone());
        let operation = PlaceOperation::new(target.clone(), None);
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout does \
                     not describe the batch dimension",
                ),
            }),
        );
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let replicated = ArrayBatch::new(array.clone(), BatchAxis::replicated()).unwrap();
        let outputs = operation.batch(&context, &EmptyRegionDriver, &[replicated]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis_position(), None);
        assert_eq!(outputs[0].value(), &array.place(&target, None).unwrap());
        assert_eq!(outputs[0].r#type().layout(), Some(&layout));
    }

    #[test]
    fn test_place_operation_batching_device_target() {
        // A device target records no sharding and no batch sharding, so the packed value is placed on the device with
        // the unchanged target, whether the input is mapped or unmapped, and an unmapped input stays unmapped. The type
        // rule of the target applies to the packed input, so a packed input that is split over the two devices of its
        // mesh is gathered (i.e., replicated over the canonical mesh of one device), and the source gains an entry for
        // the batch dimension as for any other target.
        let target = PlacementRequest::Device { device: Device::new(0, 0), memory: Some(PINNED_HOST), layout: None };
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let mapped_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, y)| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |(x, y): (_, _)| {
                        let place = |value| {
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                                value,
                                Some(&target),
                                Some(PlacementRequest::from(source.clone())),
                                true,
                            )
                        };
                        Ok((place(&x)?, place(&y)?))
                    },
                    (x, y),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    None,
                )?)
            },
            (mapped_type, ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}], %1:f32[4] .
                let %2:f32[2, 4][sharding={mesh<['device'=1:auto]>, [{}, {}]}]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    target=device(0, memory=Host[Pinned]),
                ] %0
                    %3:f32[4]@Host[Pinned] = place [
                        source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}]}},
                        target=device(0, memory=Host[Pinned]),
                    ] %1
                in (%2, %3)"
            },
        );

        // A layout does not describe the batch dimension, so a mapped input cannot be placed with one.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let operation = PlaceOperation::new(
            PlacementRequest::Device { device: Device::new(0, 0), memory: None, layout: Some(layout) },
            None,
        );
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout does \
                     not describe the batch dimension",
                ),
            }),
        );
    }

    #[test]
    fn test_place_operation_batching_source_layouts() {
        // A source is informative, and the layout that it names describes a batch item, which never describes the
        // packed value, so a source of every form drops its layout for a mapped input and keeps it for an unmapped one.
        // An in-place source that named only a layout names nothing without it, so it is dropped. A device source
        // records no sharding and no batch sharding, so it otherwise applies to the packed value unchanged, and a
        // placement source gains the placement that the packed input type records for its batch dimension.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let sources = [
            PlacementRequest::from(
                placement(&mesh_x(), vec![ShardingDimension::Replicated]).with_layout(layout.clone()),
            ),
            PlacementRequest::Device { device: Device::new(1, 0), memory: None, layout: Some(layout.clone()) },
            PlacementRequest::InPlace { memory: Some(Memory::Device), layout: Some(layout.clone()) },
            PlacementRequest::InPlace { memory: None, layout: Some(layout) },
        ];
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, y)| {
                let sources = sources.clone();
                Ok(batch(
                    move |(x, y): (_, _)| {
                        let mut outputs = Vec::new();
                        for source in &sources {
                            for value in [&x, &y] {
                                outputs.push(<ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                                    value,
                                    Some(&PlacementRequest::from(PINNED_HOST)),
                                    Some(source.clone()),
                                    true,
                                )?);
                            }
                        }
                        Ok(outputs)
                    },
                    (x, y),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    [BatchAxis::new(0), BatchAxis::replicated()].repeat(4),
                    None,
                )?)
            },
            (ArrayType::new_static(DataType::F32, [2, 4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:f32[4] .
                let %2:f32[2, 4]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target=in_place(memory=Host[Pinned]),
                ] %0
                    %3:f32[4]@Host[Pinned] = place [
                        source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}]}, \
                            layout=tiled{0}},
                        target=in_place(memory=Host[Pinned]),
                    ] %1
                    %4:f32[2, 4]@Host[Pinned] = place [source=device(1), target=in_place(memory=Host[Pinned])] %0
                    %5:f32[4]@Host[Pinned] = place [\
                        source=device(1, layout=tiled{0}), target=in_place(memory=Host[Pinned])] %1
                    %6:f32[2, 4]@Host[Pinned] = place [\
                        source=in_place(memory=Device), target=in_place(memory=Host[Pinned])] %0
                    %7:f32[4]@Host[Pinned] = place [
                        source=in_place(memory=Device, layout=tiled{0}),
                        target=in_place(memory=Host[Pinned]),
                    ] %1
                    %8:f32[2, 4]@Host[Pinned] = place [target=in_place(memory=Host[Pinned])] %0
                    %9:f32[4]@Host[Pinned] = place [\
                        source=in_place(layout=tiled{0}), target=in_place(memory=Host[Pinned])] %1
                in (%2, %3, %4, %5, %6, %7, %8, %9)"
            },
        );
    }

    #[test]
    fn test_place_operation_batching_invalid_requests() {
        // The target and the source describe a batch item, so the type rule validates them against a batch item
        // before the rule rewrites them for the packed value, and a mapped input is rejected with the error of the
        // placement of a batch item (e.g., a source layout whose rank is not the rank of a batch item, which the rule
        // drops otherwise, or a target layout of another rank, which it rejects with a less specific error otherwise).
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let place = |target: PlacementRequest, source: Option<PlacementRequest>| {
            TracingContext::<Array, ArrayOperation<Array>>::trace(
                |x| {
                    let (target, source) = (target.clone(), source.clone());
                    Ok(batch(
                        move |x| {
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                                &x,
                                Some(&target),
                                source.clone(),
                                true,
                            )
                        },
                        x,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        None,
                    )?)
                },
                ArrayType::new_static(DataType::F32, [2, 4]),
            )
            .map(|_| ())
            .map_err(|error| error.to_string())
        };
        let device = Device::new(1, 0);
        let device_source =
            PlacementRequest::Device { device: Device::new(0, 0), memory: None, layout: Some(rank_2.clone()) };
        let in_place_source = PlacementRequest::InPlace { memory: None, layout: Some(rank_2.clone()) };
        let expected = |end: &str| {
            Err(format!(
                "`{PLACE_OPERATION_NAME}` {end} layout is invalid: tiled layout rank 2 does not match array rank 1"
            ))
        };
        assert_eq!(place(PlacementRequest::from(device), Some(device_source)), expected("source"));
        assert_eq!(place(PlacementRequest::from(device), Some(in_place_source)), expected("source"));
        assert_eq!(place(PlacementRequest::InPlace { memory: None, layout: Some(rank_2) }, None), expected("target"));

        // An unmapped input is its own batch item, so it is validated as given before a target that records a split
        // batch broadcasts it along a new batch dimension, whose type no longer has the layout of the item, and the
        // batched placement reports the error of the placement of the item, as it does without the split batch.
        let item_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![8])));
        let item = Array::from_elements(item_type.clone(), &[1.0f32, 2.0]).unwrap();
        let source =
            PlacementRequest::InPlace { memory: None, layout: Some(Layout::Strided(StridedLayout::new(vec![4]))) };
        let replicated = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let split = replicated.clone().with_batch_sharding(ShardingDimension::sharded(["x"])).unwrap();
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        for target in [replicated, split] {
            let operation = PlaceOperation::new(target, source.clone());
            let direct = operation.infer_output_types(&[item_type.clone()], &[]).map_err(|error| error.to_string());
            assert_eq!(
                direct,
                Err(format!(
                    "`{PLACE_OPERATION_NAME}` source layout `strided{{4}}` does not match the input type `{item_type}`",
                )),
            );
            let batched = operation.batch(&context, &EmptyRegionDriver, &[ArrayBatch::replicated(item.clone())]);
            assert_eq!(batched.map(|_| ()).map_err(|error| error.to_string()), direct.map(|_| ()));
        }
    }

    #[test]
    fn test_place_operation_batching_in_place_target() {
        // An in-place target records no sharding and no batch sharding, so the packed value is placed with the
        // unchanged target, whether the input is mapped or unmapped, and an unmapped input stays unmapped. The output
        // keeps the sharding of the packed input, and the source gains an entry for the batch dimension as for any
        // other target.
        let target = PlacementRequest::from(PINNED_HOST);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let mapped_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, y)| {
                let (target, source) = (target.clone(), source.clone());
                Ok(batch(
                    move |(x, y): (_, _)| {
                        let place = |value| {
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                                value,
                                Some(&target),
                                Some(PlacementRequest::from(source.clone())),
                                true,
                            )
                        };
                        Ok((place(&x)?, place(&y)?))
                    },
                    (x, y),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    (BatchAxis::new(0), BatchAxis::replicated()),
                    None,
                )?)
            },
            (mapped_type, ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}], %1:f32[4] .
                let %2:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}, {}]}},
                    target=in_place(memory=Host[Pinned]),
                ] %0
                    %3:f32[4]@Host[Pinned] = place [
                        source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}]}},
                        target=in_place(memory=Host[Pinned]),
                    ] %1
                in (%2, %3)"
            },
        );

        // A layout does not describe the batch dimension, so a mapped input cannot be placed with one.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let operation = PlaceOperation::new(PlacementRequest::InPlace { memory: None, layout: Some(layout) }, None);
        let context = BatchingContext::<_, ArrayBatchingPolicy>::new(EagerContext::<Array>::new(), 2);
        let mapped =
            ArrayBatch::new(Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(), BatchAxis::new(0)).unwrap();
        assert_eq!(
            operation.batch(&context, &EmptyRegionDriver, &[mapped]),
            Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{PLACE_OPERATION_NAME}` cannot batch a target with an explicit layout, because the layout does \
                     not describe the batch dimension",
                ),
            }),
        );
    }

    #[test]
    fn test_place_operation_batching_manual_region() {
        // A placement without a target leaves the packed value where it is, so it accepts batch items inside a manual
        // region, and the source that it infers from a batch item is lifted back to where the packed value lives.
        let packed = manual_region_placement(1);
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(packed.sharding().clone()).unwrap();
        let array = Array::from_elements(input_type, &[1.0f32, 2.0]).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let output = batch(
            |item| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&item, None, None, true),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.array, array);
        assert_eq!(output.placement, Some(packed.clone()));
        assert_eq!(context.0.borrow().as_slice(), &[ArrayOperation::Place(PlaceOperation::new(None, packed))]);
    }

    #[test]
    fn test_place_operation_differentiation() {
        // A placement is linear, so the tangent is placed with the same target as the primal.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]).with_memory(PINNED_HOST);
        let (primal, tangent) = differentiate_at(Array::vector(vec![1.0f32, 2.0]).unwrap())
            .jvp(Array::vector(vec![3.0f32, 4.0]).unwrap(), |x| x.place(&target, None))
            .unwrap();
        let expected_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(target.sharding().clone())
            .unwrap()
            .with_memory(PINNED_HOST);
        assert_eq!(primal, Array::from_elements(expected_type.clone(), &[1.0f32, 2.0]).unwrap());
        assert_eq!(tangent, Array::from_elements(expected_type.clone(), &[3.0f32, 4.0]).unwrap());

        // Symbolic zero tangents keep the type of the placed primal without staging a placement of their own.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(ArrayType::new_static(DataType::F32, [2]));
        let duals = PlaceOperation::new(target, None)
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(duals[0].tangent().is_zero());
        assert_eq!(duals[0].tangent().r#type().as_ref(), &expected_type);
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_place_operation_differentiation_may_alias() {
        // The primal and the tangent are both placed with the copy policy of the primal placement.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, t)| {
                Ok(differentiate_at(x).jvp(t, |x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        Some(&PlacementRequest::from(target.clone())),
                        None,
                        false,
                    )
                })?)
            },
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    may_alias=false,
                ] %0
                    %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                        target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                        may_alias=false,
                    ] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_layouts() {
        // A one-byte stride is valid for the one-byte elements of an `f8e8m0fnu` primal, but its tangent has `f32`
        // elements, which that stride would overlap, so the tangent is placed without the target layout.
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_layout(Layout::Strided(StridedLayout::new(vec![1])));
        let primal = Array::from_logical_bytes(ArrayType::new_static(DataType::F8E8M0FNU, [2]), &[127, 128]).unwrap();
        let (primal, tangent) = differentiate_at(primal)
            .jvp(Array::vector(vec![3.0f32, 4.0]).unwrap(), |x| x.place(&target, None))
            .unwrap();
        assert_eq!(
            primal,
            Array::from_logical_bytes(
                ArrayType::new_static(DataType::F8E8M0FNU, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_layout(Layout::Strided(StridedLayout::new(vec![1]))),
                &[127, 128],
            )
            .unwrap(),
        );
        assert_eq!(
            tangent,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2]).with_sharding(target.sharding().clone()).unwrap(),
                &[3.0f32, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_differentiation_device_target() {
        // The tangent is placed on the same device, in the same memory, and with the same copy policy as the primal,
        // but without the layout of the target, which need not be valid for the element type of the tangent.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let target =
            PlacementRequest::Device { device: Device::new(0, 0), memory: Some(PINNED_HOST), layout: Some(layout) };
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, t)| {
                Ok(differentiate_at(x).jvp(t, |x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, Some(&target), None, false)
                })?)
            },
            (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4][layout=tiled{0}]@Host[Pinned] = place [\
                    target=device(0, memory=Host[Pinned], layout=tiled{0}), may_alias=false] %0
                    %3:f32[4]@Host[Pinned] = place [target=device(0, memory=Host[Pinned]), may_alias=false] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_in_place_target() {
        // The tangent is placed in place in the same memory and with the same copy policy as the primal, but without
        // the layout of the target, which need not be valid for the element type of the tangent. A target that names
        // only a layout names nothing without it, so the tangent is then placed without a target.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let program = |target: PlacementRequest| {
            let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
                |(x, t)| {
                    Ok(differentiate_at(x).jvp(t, |x| {
                        <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, Some(&target), None, false)
                    })?)
                },
                (ArrayType::new_static(DataType::F32, [4]), ArrayType::new_static(DataType::F32, [4])),
            )
            .unwrap();
            program.to_string()
        };
        assert_eq!(
            program(PlacementRequest::InPlace { memory: Some(PINNED_HOST), layout: Some(layout.clone()) }),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4][layout=tiled{0}]@Host[Pinned] = place [\
                    target=in_place(memory=Host[Pinned], layout=tiled{0}), may_alias=false] %0
                    %3:f32[4]@Host[Pinned] = place [target=in_place(memory=Host[Pinned]), may_alias=false] %1
                in (%2, %3)"
            },
        );
        assert_eq!(
            program(PlacementRequest::InPlace { memory: None, layout: Some(layout) }),
            indoc! {"
                lambda %0:f32[4], %1:f32[4] .
                let %2:f32[4][layout=tiled{0}] = place [target=in_place(layout=tiled{0}), may_alias=false] %0
                    %3:f32[4] = place [may_alias=false] %1
                in (%2, %3)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_manual_region() {
        // A placement without a target moves nothing, so it accepts a value inside a manual region, and the primal and
        // the tangent both record where the primal concretely lives as their source.
        let source = manual_region_placement(0);
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(source.sharding().clone()).unwrap();
        let context = RecordingContext::default();
        let value = |value: f32| PlacedArray {
            array: Array::from_elements(input_type.clone(), &[value]).unwrap(),
            placement: Some(source.clone()),
            context: context.clone(),
        };
        let (primal, tangent) = differentiate_at(value(3.0))
            .jvp(value(5.0), |x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))
            .unwrap();
        assert_eq!(primal.array, Array::from_elements(input_type.clone(), &[3.0f32]).unwrap());
        assert_eq!(tangent.array, Array::from_elements(input_type, &[5.0f32]).unwrap());
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(None, source.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, source)),
            ],
        );
    }

    #[test]
    fn test_place_operation_differentiation_batching() {
        // Batching a differentiated placement places the packed primal and the packed tangent with the same lifted
        // target and source.
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |(x, t)| {
                Ok(batch(
                    move |(item, tangent)| {
                        Ok(differentiate_at(item).jvp(tangent, |item| item.place(&target, Some(&source)))?)
                    },
                    (x, t),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    None,
                )?)
            },
            (ArrayType::new_static(DataType::F32, [2, 4]), ArrayType::new_static(DataType::F32, [2, 4])),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2, 4], %1:f32[2, 4] .
                let %2:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                ] %0
                    %3:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] = place [
                        source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {}]}},
                        target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                            sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    ] %1
                in (%2, %3)"
            },
        );

        // Differentiating a batched placement transposes the lifted placement, so the batched cotangent moves back to
        // the source mesh with the lifted source sharding.
        let input_type = ArrayType::new_static(DataType::F32, [2, 4])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
                )
                .unwrap(),
            )
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (_, pullback) = differentiate_at(input)
            .vjp(|x| {
                Ok(batch(|item| item.place(&target, Some(&source)), x, BatchAxis::new(0), BatchAxis::new(0), None)?)
            })
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[2, 4][sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}] .
                let %1:f32[2, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}, {'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_differentiation_batching_inferred_source() {
        // A gradient taken inside a batched function moves the cotangent back to the devices that the packed input
        // concretely lives on, which the placement infers as its source from the batch item view of that input,
        // also when the target has the topology of the source and no source is given.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
        let target = placement(&other_mesh_x, vec![ShardingDimension::Replicated]);
        let context = RecordingContext::default();
        let input = PlacedArray {
            array: Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(packed.sharding().clone()).unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            placement: Some(packed.clone()),
            context: context.clone(),
        };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();

        // The forward placement records the packed input placement as its source, so the lifted transpose moves the
        // cotangent back to the devices 0 and 1 with the sharding of the packed input, which the source of each batch
        // item records for its batch dimension. The explicit replicated target of each batch item replicates the
        // batch dimension of the forward output, as in JAX, so that is where the transpose moves the cotangent from.
        let target = placement(&other_mesh_x, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]);
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(target.clone(), packed.clone())),
                ArrayOperation::Place(PlaceOperation::new(gradient_placement.clone(), target).with_may_alias(false),),
            ],
        );
        assert_eq!(
            gradient.array,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2]).with_sharding(packed.sharding().clone()).unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_auto_axis() {
        // The packed input is split along the auto axis `x` in its batch dimension on the devices 0 and 1, which its
        // type does not record, and each batch item is moved to the devices 2 and 3 of a mesh with the same axes. The
        // source of each batch item records that split, so the forward placement records the packed input placement as
        // its source, and the transpose moves the cotangent back to the devices 0 and 1 with its batch dimension split
        // along `x` again, rather than replicated.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let target = Placement::new(other_mesh_x.clone(), Sharding::replicated(auto_mesh.clone(), 1)).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let target = Placement::new(other_mesh_x, Sharding::replicated(auto_mesh, 2)).unwrap();
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(target.clone(), packed)),
                ArrayOperation::Place(PlaceOperation::new(gradient_placement.clone(), target).with_may_alias(false),),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_unmapped_cotangent() {
        // The packed input is split along the auto axis `x` in its batch dimension on the devices 0 and 1, which its
        // type does not record, and each batch item is moved to the devices 2 and 3 and summed. The cotangent of the
        // sum is the same for every batch item, so it is unmapped, but the transposed placement moves it back to the
        // source of each batch item, which records the split. The cotangent is therefore broadcast along a new batch
        // dimension and placed with its batch dimension split along `x` again, rather than replicated.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let target = Placement::new(other_mesh_x.clone(), Sharding::replicated(auto_mesh.clone(), 1)).unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray {
            array: Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
            placement: Some(packed.clone()),
            context: context.clone(),
        };
        let gradient = batch(
            |item| Ok(differentiate_at(item).gradient(|x| x.place(&target, None)?.reduce(&[0], ReductionKind::Sum))?),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();

        // The last recorded operation is the transposed placement, which places the broadcast cotangent where the
        // packed input lives.
        let gradient_placement = packed.with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().last(),
            Some(&ArrayOperation::Place(
                PlaceOperation::new(
                    gradient_placement.clone(),
                    Placement::new(other_mesh_x, Sharding::replicated(auto_mesh.clone(), 2)).unwrap(),
                )
                .with_may_alias(false),
            )),
        );
        assert_eq!(
            gradient.array,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(auto_mesh, 2))
                    .unwrap(),
                &[1.0f32, 1.0, 1.0, 1.0],
            )
            .unwrap(),
        );
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_identity() {
        // A gradient taken inside a batched function through a placement of each batch item without a target keeps the
        // primal where it lives (i.e., split along the auto axis `x` in its batch dimension, which its type does not
        // record) and records the batch item view of that placement, which records the split, as the source of the
        // tangent. The transposed placement moves the cotangent to that source, so batching puts the split back, and
        // the gradient lives where the packed input does.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let packed = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
                .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 2))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item)
                    .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(None, packed.clone())),
                ArrayOperation::Place(PlaceOperation::new(gradient_placement.clone(), packed).with_may_alias(false),),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_nested() {
        // The packed input is split along `x` in its outer batch dimension and along `y` in its inner batch dimension,
        // so each batch item records both splits, from the outermost to the innermost, and each batching level puts
        // back its own. The placement of each batch item without a target therefore records the packed input placement
        // as its source, and the transpose moves the cotangent back to that placement.
        let auto_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let devices = DeviceMesh::new(auto_mesh.clone(), (0..4).map(|id| Device::new(id, 0)).collect()).unwrap();
        let packed = Placement::new(
            devices.clone(),
            Sharding::new(
                auto_mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::sharded(["y"]),
                    ShardingDimension::Replicated,
                ],
            )
            .unwrap(),
        )
        .unwrap();
        let array = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2, 1])
                .with_sharding(Sharding::replicated(auto_mesh.clone(), 3))
                .unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let context = RecordingContext::default();
        let input = PlacedArray { array: array.clone(), placement: Some(packed.clone()), context: context.clone() };
        let item_placement = RefCell::new(None);
        let gradient = batch(
            |outer| {
                Ok(batch(
                    |item| {
                        *item_placement.borrow_mut() = item.placement();
                        let (output, pullback) = differentiate_at(item).vjp(|x| {
                            <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true)
                        })?;
                        pullback.apply_materialized(output)
                    },
                    outer,
                    BatchAxis::new(0),
                    BatchAxis::new(0),
                    None,
                )?)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            item_placement.into_inner(),
            Some(
                Placement::new(devices, Sharding::replicated(auto_mesh, 1))
                    .unwrap()
                    .with_batch_sharding(ShardingDimension::sharded(["x"]))
                    .unwrap()
                    .with_batch_sharding(ShardingDimension::sharded(["y"]))
                    .unwrap(),
            ),
        );
        let gradient_placement = packed.clone().with_memory(Memory::Device);
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(None, packed.clone())),
                ArrayOperation::Place(PlaceOperation::new(gradient_placement.clone(), packed).with_may_alias(false),),
            ],
        );
        assert_eq!(gradient.array, array);
        assert_eq!(gradient.placement, Some(gradient_placement));
    }

    #[test]
    fn test_place_operation_differentiation_batching_without_source() {
        // A reference array reports no placement, so a placement of each batch item without a source records none,
        // and its transpose restores the tracked sharding of the batch item on the target mesh. That transposed target
        // records no batch sharding, so batching replicates its batch dimension like that of the forward target, and
        // the gradient is replicated even though the packed input type records a split of its batch dimension along
        // `x`. This pins a remaining limitation. Without a source, nothing records how the batch of the input was
        // distributed (JAX leaves such a cotangent where it is as well).
        let input_type = ArrayType::new_static(DataType::F32, [2, 2])
            .with_sharding(
                Sharding::new(
                    mesh_x().logical_mesh().clone(),
                    vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated],
                )
                .unwrap(),
            )
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let gradient = batch(
            |item| {
                let (output, pullback) = differentiate_at(item).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(output)
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(
            gradient,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2, 2])
                    .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 2))
                    .unwrap(),
                &[1.0f32, 2.0, 3.0, 4.0],
            )
            .unwrap(),
        );
    }

    #[test]
    fn test_place_operation_transposition() {
        // With a known source, the cotangent moves back to the source mesh with the cotangent dual of the source
        // sharding and the memory of the input, and the old target becomes the new source.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let cotangent = pullback
            .apply_materialized(
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap());
        let (pullback, residuals) = pullback.into_transposed_parts().unwrap();
        assert!(residuals.is_empty());
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // Without a source, the cotangent stays on the target mesh and gets the tracked sharding of the input back.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh_x().logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, None)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // An input without a tracked sharding gets its exact cotangent type through a placement-only broadcast.
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, None)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );

        // An input that is tracked over another mesh did change meshes, so its cotangent cannot be moved back to
        // devices that are not known.
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh_y().logical_mesh().clone(), 1))
            .unwrap();
        let input = Array::from_elements(input_type, &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, None)).unwrap();
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` across device meshes without a source placement",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_without_target() {
        // Without a target, the primal stays where it is, and the cotangent moves to the source with the memory of the
        // input and no source of its own. The input type tracks no sharding here, so it gets its exact cotangent type
        // through a placement-only broadcast.
        let source = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input.clone())
            .vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    None,
                    Some(PlacementRequest::from(source.clone())),
                    true,
                )
            })
            .unwrap();
        assert_eq!(output, input);
        let cotangent = Array::vector(vec![5.0f32, 6.0, 7.0, 8.0]).unwrap();
        assert_eq!(pullback.apply_materialized(cotangent.clone()), Ok(cotangent.clone()));
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] = place [
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );

        // Without a target and a source, the placement is an identity, and so is its transpose, which still forces a
        // copy of the cotangent.
        let (_, pullback) = differentiate_at(input)
            .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))
            .unwrap();
        let copied = pullback.apply_materialized(cotangent.clone()).unwrap();
        assert_eq!(copied, cotangent);
        assert!(!Arc::ptr_eq(cotangent.shared_storage_bytes(), copied.shared_storage_bytes()));
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4] = place [may_alias=false] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_device_target() {
        // The input is split over the two devices of its mesh, so the device target gathers it (i.e., replicates it
        // over the canonical mesh of one device). With a source, the cotangent moves back to the source with the
        // cotangent dual of the source sharding and the memory of the input, and the device of the forward target
        // becomes its source, without the memory of the forward target, as the two ends of a placement swap.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let gathered = Sharding::replicated(LogicalMesh::single_device(), 1);
        let target = PlacementRequest::Device { device: Device::new(0, 0), memory: Some(PINNED_HOST), layout: None };
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let place = |source: Option<Placement>| {
            let target = target.clone();
            move |x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(&target),
                    source.map(PlacementRequest::from),
                    true,
                )
            }
        };
        let (output, pullback) = differentiate_at(input.clone()).vjp(place(Some(source.clone()))).unwrap();
        assert_eq!(
            output.r#type().as_ref(),
            &input_type.clone().with_sharding(gathered).unwrap().with_memory(PINNED_HOST),
        );
        let cotangent = pullback
            .apply_materialized(
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['device'=1:auto]>, [{}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source=device(0),
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // Without a source, or with a source that does not name the devices of the input, the devices to which the
        // cotangent must return are not known, so the transposition fails. A device source states that the input lives
        // on one device, which an input that is tracked over two devices does not, so it is already rejected forward.
        assert!(matches!(
            differentiate_at(input.clone()).vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(&target),
                    Some(PlacementRequest::from(Device::new(0, 0))),
                    true,
                )
            }),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message.starts_with("`place` device source names one device"),
        ));
        let error = format!(
            "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that lives on more than one \
             device without a source placement that names those devices",
        );
        for source in [None, Some(PlacementRequest::from(Memory::Device))] {
            let (_, pullback) = differentiate_at(input.clone())
                .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, Some(&target), source, true))
                .unwrap();
            assert!(matches!(
                pullback.into_transposed_parts(),
                Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message })) if message == error,
            ));
        }
    }

    #[test]
    fn test_place_operation_transposition_device_target_reduction_state() {
        // A device target completes the reduction state of its input over mesh axes of size one, so the output
        // cotangent carries no reduction state. The cotangent moves back without it, also when the forward placement
        // records a source whose sharding restates that state, and the cotangent of the input, which carries the dual
        // of that state, gets it back through a placement-only broadcast.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let device_mesh = DeviceMesh::new(mesh.clone(), vec![Device::new(0, 0)]).unwrap();
        let unreduced = Sharding::replicated(mesh.clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let reduced = Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap();
        let transposed = |sharding: &Sharding, source: Option<PlacementRequest>| {
            let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
            let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
            let (output, pullback) = differentiate_at(input)
                .vjp(|x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        Some(&Device::new(1, 0).into()),
                        source.clone(),
                        true,
                    )
                })
                .unwrap();
            assert_eq!(output.r#type().sharding(), Some(&Sharding::replicated(mesh.clone(), 1)));
            let cotangent = pullback
                .apply_materialized(
                    Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
                )
                .unwrap();
            assert_eq!(
                cotangent,
                Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            );
            pullback.into_transposed_parts().unwrap().0.into_program().to_string()
        };

        // Without a source, the cotangent stays on the device of the forward target.
        assert_eq!(
            transposed(&unreduced, None),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [\
                    target=device(1, memory=Device), may_alias=false] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );
        assert_eq!(
            transposed(&reduced, None),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [\
                    target=device(1, memory=Device), may_alias=false] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );

        // With a placement source, the cotangent moves back to the source devices with the sharding of the source
        // without its reduction state, which the output cotangent does not carry.
        assert_eq!(
            transposed(
                &unreduced,
                Some(PlacementRequest::from(Placement::new(device_mesh.clone(), unreduced.clone()).unwrap())),
            ),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );
        assert_eq!(
            transposed(&reduced, Some(PlacementRequest::from(Placement::new(device_mesh, reduced.clone()).unwrap()))),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );

        // With a device source, the cotangent moves back to the source device over the logical mesh of the input, as
        // with a placement source, without the reduction state that the device target completed, which the broadcast
        // restores.
        assert_eq!(
            transposed(&unreduced, Some(PlacementRequest::from(Device::new(0, 0)))),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], reduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );
        assert_eq!(
            transposed(&reduced, Some(PlacementRequest::from(Device::new(0, 0)))),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}] = broadcast [
                        output_type=f32[4][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}],
                        output_axes=[0],
                    ] %1
                in (%2)"
            },
        );

        // A value that is reduced over a larger mesh axis is complete on every device along it, so it can be moved onto
        // one device. Its cotangent is unreduced over that axis, and the contributions of the devices along it must sum
        // to the output cotangent, which a placement cannot produce, so the transposition fails.
        let reduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2]).with_sharding(reduced.clone()).unwrap(),
            &[1.0f32, 2.0],
        )
        .unwrap();
        let (output, pullback) = differentiate_at(input)
            .vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(&Device::new(1, 0).into()),
                    Some(PlacementRequest::from(Placement::new(mesh_x(), reduced.clone()).unwrap())),
                    true,
                )
            })
            .unwrap();
        assert_eq!(output.r#type().sharding(), Some(&Sharding::replicated(LogicalMesh::single_device(), 1)));
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is reduced over \
                     mesh axes larger than one (i.e., `x`), because its cotangent is unreduced over them",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_device_source() {
        // A device source is accepted only for an input that its type tracks over a mesh of one device, so the device
        // and the logical mesh of the input name where the input lived. The cotangent returns there with the sharding
        // of the input cotangent in the memory of the input, whatever mesh the output cotangent lives on, as JAX
        // returns the cotangent of `device_put(x, target, src=device)` to that device, and the forward target becomes
        // the source of the transposed placement in its own form, as the two ends of a placement swap.
        let source = PlacementRequest::Device { device: Device::new(0, 0), memory: Some(Memory::Device), layout: None };
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        let transposed = |input_type: &ArrayType, target: Option<PlacementRequest>| {
            let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
            let (output, pullback) = differentiate_at(input)
                .vjp(|x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        target.as_ref(),
                        Some(source.clone()),
                        true,
                    )
                })
                .unwrap();
            let cotangent =
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
            assert_eq!(
                pullback.apply_materialized(cotangent).unwrap(),
                Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            );
            pullback.into_transposed_parts().unwrap().0.into_program().to_string()
        };

        // A placement target on a mesh with the topology of the input's mesh becomes a placement source.
        let x_mesh = DeviceMesh::new(input_type.sharding().unwrap().mesh().clone(), vec![Device::new(2, 0)]).unwrap();
        let target = placement(&x_mesh, vec![ShardingDimension::Replicated]).with_memory(PINNED_HOST);
        assert_eq!(
            transposed(&input_type, Some(PlacementRequest::from(target))),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] = place [
                    source={mesh<['x'=1:explicit]>, devices=[2], sharding={mesh<['x'=1:explicit]>, [{}]}},
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // A placement target on a logical mesh of another topology leaves the output cotangent on that mesh, and the
        // cotangent still returns to the logical mesh of the input on the source device.
        let y_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(2, 0)],
        )
        .unwrap();
        let target = PlacementRequest::from(placement(&y_mesh, vec![ShardingDimension::sharded(["y"])]));
        assert_eq!(
            transposed(&input_type, Some(target)),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=1:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=1:explicit]>, devices=[2], sharding={mesh<['y'=1:explicit]>, [{'y'}]}},
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // A device target becomes a device source without its memory.
        let target = PlacementRequest::Device { device: Device::new(1, 0), memory: Some(PINNED_HOST), layout: None };
        assert_eq!(
            transposed(&input_type, Some(target.clone())),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // Without a target, the forward placement leaves the value on the source device, so the cotangent moves there
        // without a source of its own.
        assert_eq!(
            transposed(&input_type, None),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // An input without a tracked sharding names no logical mesh, so the device source stays a device.
        assert_eq!(
            transposed(&ArrayType::new_static(DataType::F32, [4]), Some(target)),
            indoc! {"
                lambda %0:f32[4]@Host[Pinned] .
                let %1:f32[4] = place [source=device(1), target=device(0, memory=Device), may_alias=false] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_in_place_target() {
        // An in-place target leaves the value where it is, but the output cotangent can live elsewhere (e.g., a
        // gradient seed), so with a known source, the cotangent moves back to the source in the memory of the input, as
        // for any other target, and the in-place target says nothing about where the output cotangent lives, so the
        // transposed placement has no source. A second transposition keeps the cotangent of the cotangent on that mesh
        // and returns it to the memory of the forward output.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let transposed = |source: Option<Placement>| {
            let (output, pullback) = differentiate_at(input.clone())
                .vjp(|x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        Some(&PlacementRequest::from(PINNED_HOST)),
                        source.map(PlacementRequest::from),
                        true,
                    )
                })
                .unwrap();
            assert_eq!(output.r#type().as_ref(), &input_type.clone().with_memory(PINNED_HOST));
            let cotangent =
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
            assert_eq!(
                pullback.apply_materialized(cotangent).unwrap(),
                Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            );
            pullback.into_transposed_parts().unwrap().0.into_program()
        };

        assert_eq!(
            transposed(Some(source.clone())).to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // A second transposition places the cotangent of the cotangent on the source mesh in the memory of the forward
        // output, and it records the target of the first transposition as its source.
        let output_cotangent =
            Array::from_elements(input_type.cotangent().unwrap().with_memory(PINNED_HOST), &[5.0f32, 6.0, 7.0, 8.0])
                .unwrap();
        let (_, pullback) = differentiate_at(output_cotangent)
            .vjp(|cotangent| {
                let input = cotangent.context().constant(input.clone())?;
                let (_, pullback) = differentiate_at(input).vjp(|x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        Some(&PlacementRequest::from(PINNED_HOST)),
                        Some(PlacementRequest::from(source.clone())),
                        true,
                    )
                })?;
                pullback.apply_materialized(cotangent)
            })
            .unwrap();
        assert_eq!(
            pullback.into_transposed_parts().unwrap().0.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Host[Pinned]},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // Without a source, the devices of the input are not known, so the cotangent stays where it is and only returns
        // to the memory of the input.
        assert_eq!(
            transposed(None).to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [\
                    target=in_place(memory=Device), may_alias=false] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_in_place_source() {
        // An in-place source states the memory in which the input is expected and names no devices, so the cotangent
        // returns in place to that memory, which is the memory of the input, for every form of the forward target, as
        // JAX returns the cotangent of `device_put(x, Space.Device, src=Space.Host)` in host memory. The forward
        // target becomes the source of the transposed placement, except for an in-place target, which names no devices.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0)],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap()
            .with_memory(PINNED_HOST);
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let transposed = |target: PlacementRequest| {
            let (output, pullback) = differentiate_at(input.clone())
                .vjp(|x| {
                    <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                        &x,
                        Some(&target),
                        Some(PlacementRequest::from(PINNED_HOST)),
                        true,
                    )
                })
                .unwrap();
            let cotangent =
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
            assert_eq!(
                pullback.apply_materialized(cotangent).unwrap(),
                Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            );
            pullback.into_transposed_parts().unwrap().0.into_program().to_string()
        };
        assert_eq!(
            transposed(PlacementRequest::from(Memory::Device)),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned] = place [\
                    target=in_place(memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );
        assert_eq!(
            transposed(PlacementRequest::from(Device::new(1, 0))),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{}]}]@Host[Pinned] = place [\
                    source=device(1), target=in_place(memory=Host[Pinned]), may_alias=false] %0
                in (%1)"
            },
        );

        // A placement target on a mesh of another topology leaves the output cotangent on that mesh, and a memory names
        // no mesh to which the cotangent could return, so transposition rejects it as it does without a source, rather
        // than stage a placement that leaves the cotangent on that mesh and a broadcast that relabels its type.
        let target = PlacementRequest::from(placement(&mesh_y(), vec![ShardingDimension::Replicated]));
        let (_, pullback) = differentiate_at(input.clone())
            .vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(&target),
                    Some(PlacementRequest::from(PINNED_HOST)),
                    true,
                )
            })
            .unwrap();
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!(
                    "cannot transpose `{PLACE_OPERATION_NAME}` across device meshes without a source placement",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_may_alias() {
        // The transposed placement forces a copy of the cotangent whatever the copy policy of the forward placement is,
        // so a forward placement that forces a copy itself transposes to the same placement as one that does not.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (output, pullback) = differentiate_at(input)
            .vjp(|x| {
                <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                    &x,
                    Some(&PlacementRequest::from(target.clone())),
                    Some(PlacementRequest::from(source.clone())),
                    false,
                )
            })
            .unwrap();
        let cotangent = pullback
            .apply_materialized(
                Array::from_elements(output.r#type().cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap(),
            )
            .unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_unsharded_input() {
        // An input without a tracked sharding moves back to the source mesh and then gets its exact cotangent type
        // through a placement-only broadcast.
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                    %2:f32[4] = broadcast [output_type=f32[4], output_axes=[0]] %1
                in (%2)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_reduction_state() {
        // The cotangent moves back with the cotangent duals of the source and target shardings, which swap their
        // unreduced and reduced axes, so an unreduced input receives a reduced cotangent and vice versa.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let unreduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced.clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let source = Placement::new(mesh_x(), unreduced.clone()).unwrap();
        let target = Placement::new(other_mesh_x.clone(), unreduced).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[2, 3], \
                        sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}], reduced={'x'}}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        // A reduced input receives an unreduced cotangent.
        let reduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(reduced.clone()).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let source = Placement::new(mesh_x(), reduced.clone()).unwrap();
        let target = Placement::new(other_mesh_x, reduced).unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[2, 3], \
                        sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}], unreduced={'x'}}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_undefined_cotangent() {
        // A value that is unreduced with a kind other than a sum has no cotangent, so the placement cannot be
        // transposed. Transposition contexts already reject such inputs when they create their accumulators, so the
        // accumulators here come from the corresponding sum, which isolates the rejection by the rule itself.
        let other_mesh_x =
            DeviceMesh::new(mesh_x().logical_mesh().clone(), vec![Device::new(2, 0), Device::new(3, 0)]).unwrap();
        let sum = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let maximum = sum.clone().with_unreduced_kind(UnreducedKind::Max).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(maximum.clone()).unwrap();
        let target = Placement::new(other_mesh_x, maximum.clone()).unwrap();
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let sum_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sum).unwrap();
        assert_eq!(
            rule_context.cotangent_accumulators(&[PartialValue::Unknown(input_type.clone())], &[]).err(),
            Some(DifferentiationError::UndefinedCotangentType { primal_type: input_type.to_string() }),
        );
        let accumulators =
            rule_context.cotangent_accumulators(&[PartialValue::Unknown(sum_type.clone())], &[]).unwrap();
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let cotangent = context.input(sum_type.cotangent().unwrap());
        assert_eq!(
            PlaceOperation::new(target, Placement::new(mesh_x(), maximum).unwrap()).transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            ),
            Err(DifferentiationError::UndefinedCotangentType { primal_type: input_type.to_string() }),
        );
    }

    #[test]
    fn test_place_operation_transposition_auto_source() {
        // The source sharding places the input along the auto axis `x`, which the input type does not record, so the
        // cotangent moves back with that runtime placement and its type matches the input cotangent type exactly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap(),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:auto]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_uneven_source() {
        // The source splits an extent of 3 over the 2 devices along the auto axis `x`, which the input type does not
        // record. The transposed placement moves the cotangent back to that source, which type inference rejects
        // because a placement cannot describe an uneven split.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let source = Placement::new(
            mesh_x(),
            Sharding::new(auto_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap(),
        )
        .unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [3]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap(),
            &[1.0f32, 2.0, 3.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        assert!(matches!(
            pullback.into_transposed_parts(),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!(
                    "`{PLACE_OPERATION_NAME}` target sharding splits dimension #0 with extent 3 into 2 shards, which \
                     do not divide it evenly",
                ),
        ));
    }

    #[test]
    fn test_place_operation_transposition_host_memory() {
        // The cotangent returns to the memory of the input, which is pinned host memory here.
        let source = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(Memory::Device);
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [4])
                .with_sharding(source.sharding().clone())
                .unwrap()
                .with_memory(PINNED_HOST),
            &[1.0f32, 2.0, 3.0, 4.0],
        )
        .unwrap();
        let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source))).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{}]}]@Host[Pinned] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], sharding={mesh<['y'=4:explicit]>, [{'y'}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{}]}, memory=Host[Pinned]},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_manual_mesh() {
        // A replicated scalar over a mesh with a manual axis is a global value outside any manual region, so placing it
        // where it already lives without a source transposes to a placement that gives the cotangent its sharding back.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let target = Placement::new(mesh_x(), Sharding::replicated(manual_mesh.clone(), 0)).unwrap();
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(Sharding::replicated(manual_mesh, 0)).unwrap();
        let input = Array::from_elements(input_type.clone(), &[3.0f32]).unwrap();
        let (output, pullback) = differentiate_at(input).vjp(|x| x.place(&target, None)).unwrap();
        assert_eq!(output, Array::from_elements(input_type.clone(), &[3.0f32]).unwrap());
        let cotangent =
            pullback.apply_materialized(Array::from_elements(input_type.clone(), &[5.0f32]).unwrap()).unwrap();
        assert_eq!(cotangent, Array::from_elements(input_type, &[5.0f32]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, []}] .
                let %1:f32[][sharding={mesh<['x'=2:manual]>, []}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:manual]>, []}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:manual]>, []}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_manual_region() {
        // A value inside a manual region cannot have been moved, and neither can its cotangent, so a placement without
        // a target transposes to a placement without a target, whose source is the cotangent dual of the forward
        // source.
        let source = manual_region_placement(0);
        let input_type = ArrayType::scalar(DataType::F32).with_sharding(source.sharding().clone()).unwrap();
        let context = RecordingContext::default();
        let value = |value: f32| PlacedArray {
            array: Array::from_elements(input_type.clone(), &[value]).unwrap(),
            placement: Some(source.clone()),
            context: context.clone(),
        };
        let (output, pullback) = differentiate_at(value(3.0))
            .vjp(|x| <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x, None, None, true))
            .unwrap();
        assert_eq!(output.array, Array::from_elements(input_type.clone(), &[3.0f32]).unwrap());
        let cotangent = pullback.apply_materialized(value(7.0)).unwrap();
        assert_eq!(cotangent.array, Array::from_elements(input_type.cotangent().unwrap(), &[7.0f32]).unwrap());
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] .
                let %1:f32[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_second_order() {
        // The pullback of a placement is itself a placement, so differentiating a function that applies the pullback
        // transposes it again and moves the cotangent of the cotangent to the target of the original placement.
        let source = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]).with_memory(PINNED_HOST);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let output_cotangent_type = input_type
            .with_sharding(target.sharding().clone())
            .unwrap()
            .with_memory(PINNED_HOST)
            .cotangent()
            .unwrap();
        let output_cotangent = Array::from_elements(output_cotangent_type, &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
        let (_, pullback) = differentiate_at(output_cotangent)
            .vjp(|cotangent| {
                let input = cotangent.context().constant(input)?;
                let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, Some(&source)))?;
                pullback.apply_materialized(cotangent)
            })
            .unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] .
                let %1:f32[4][sharding={mesh<['y'=4:explicit]>, [{'y'}]}]@Host[Pinned] = place [
                    source={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                    target={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{'y'}]}, memory=Host[Pinned]},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_second_order_device_forms() {
        // Each transposition swaps the two ends of a placement, so transposing a placement onto a device twice moves
        // the value onto that device again, whether its source is a placement or a device. The value of the second
        // transposition is tracked over a mesh of one device, so the device that becomes its source is completed with
        // that logical mesh, as for every device source of such a value, and both forms give the same placement.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::new(0, 0)],
        )
        .unwrap();
        let source = placement(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(source.sharding().clone()).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let target = PlacementRequest::from(Device::new(1, 0));
        let transposed_twice = |source: PlacementRequest| {
            let output_cotangent =
                Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
            let (_, pullback) = differentiate_at(output_cotangent)
                .vjp(|cotangent| {
                    let input = cotangent.context().constant(input.clone())?;
                    let (_, pullback) = differentiate_at(input).vjp(|x| {
                        <ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(
                            &x,
                            Some(&target),
                            Some(source.clone()),
                            true,
                        )
                    })?;
                    pullback.apply_materialized(cotangent)
                })
                .unwrap();
            pullback.into_transposed_parts().unwrap().0.into_program().to_string()
        };
        let expected = indoc! {"
            lambda %0:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] .
            let %1:f32[4][sharding={mesh<['x'=1:explicit]>, [{'x'}]}] = place [
                source={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{'x'}]}},
                target={mesh<['x'=1:explicit]>, devices=[1], \
                    sharding={mesh<['x'=1:explicit]>, [{'x'}]}, memory=Device},
                may_alias=false,
            ] %0
            in (%1)"
        };
        assert_eq!(transposed_twice(PlacementRequest::from(source.clone())), expected);
        assert_eq!(transposed_twice(PlacementRequest::from(Device::new(0, 0))), expected);
    }

    #[test]
    fn test_place_operation_transposition_second_order_auto_target() {
        // A target without a source splits the value along the auto axis `x`, which neither the input type nor the
        // output type records. The transposed placement records the forward target as its source, so transposing twice
        // moves the value to the forward target again, with its split along `x`.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(auto_mesh.clone(), vec![Device::new(0, 0), Device::new(1, 0)]).unwrap();
        let target = placement(&mesh, vec![ShardingDimension::sharded(["x"])]);
        let input_type =
            ArrayType::new_static(DataType::F32, [4]).with_sharding(Sharding::replicated(auto_mesh, 1)).unwrap();
        let input = Array::from_elements(input_type.clone(), &[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let (_, pullback) = differentiate_at(input.clone()).vjp(|x| x.place(&target, None)).unwrap();
        let (pullback, _) = pullback.into_transposed_parts().unwrap();
        assert_eq!(
            pullback.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] = place [
                    source={mesh<['x'=2:auto]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{'x'}]}},
                    target={mesh<['x'=2:auto]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );

        let output_cotangent = Array::from_elements(input_type.cotangent().unwrap(), &[5.0f32, 6.0, 7.0, 8.0]).unwrap();
        let (_, pullback) = differentiate_at(output_cotangent)
            .vjp(|cotangent| {
                let input = cotangent.context().constant(input.clone())?;
                let (_, pullback) = differentiate_at(input).vjp(|x| x.place(&target, None))?;
                pullback.apply_materialized(cotangent)
            })
            .unwrap();
        assert_eq!(
            pullback.into_transposed_parts().unwrap().0.into_program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] = place [
                    source={mesh<['x'=2:auto]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{}]}},
                    target={mesh<['x'=2:auto]>, devices=[0, 1], sharding={mesh<['x'=2:auto]>, [{'x'}]}, memory=Device},
                    may_alias=false,
                ] %0
                in (%1)"
            },
        );
    }

    #[test]
    fn test_place_operation_transposition_ignored_cotangent() {
        // A placement onto one device of a value that is reduced over a larger axis cannot be transposed, but when the
        // cotangent of its input is not needed (e.g., because only the gradient of another input of a conditional is
        // requested), its transposition validates and stages nothing, so that it cannot fail an unrelated derivative.
        let reduced = Sharding::replicated(mesh_x().logical_mesh().clone(), 1).with_reduced_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(reduced).unwrap();
        let operation = PlaceOperation::new(Device::new(1, 0), None);
        let output_type = operation.infer_output_types(&[input_type.clone()], &[]).unwrap().remove(0);
        let transpose = |needed: bool| {
            let context = TracingContext::<Array, ArrayOperation<Array>>::new();
            let mut rule_context = TranspositionContext::new(context.clone());
            let inputs = [PartialValue::Unknown(input_type.clone())];
            let accumulators = vec![rule_context.cotangent_accumulator(input_type.cotangent().unwrap(), needed)];
            let cotangent = context.input(output_type.cotangent().unwrap());
            let result = operation.transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Value(cotangent)],
                &accumulators,
            );
            (result, context.builder().borrow().instructions().len())
        };
        assert_eq!(transpose(false), (Ok(()), 0));
        assert_eq!(
            transpose(true),
            (
                Err(DifferentiationError::Program(ProgramError::UnsupportedOperation {
                    message: format!(
                        "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is reduced \
                         over mesh axes larger than one (i.e., `x`), because its cotangent is unreduced over them",
                    ),
                })),
                0,
            ),
        );
    }

    #[test]
    fn test_place_operation_transposition_zero_cotangent() {
        // A symbolic zero cotangent propagates unchanged without staging a placement.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input_type = ArrayType::new_static(DataType::F32, [2]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let mut rule_context = TranspositionContext::new(context.clone());
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        PlaceOperation::new(target.clone(), None)
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(input_type.clone().with_sharding(target.sharding().clone()).unwrap())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type);
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_place_operation_provider() {
        // Homogeneous array families provide the placement directly, and array IR families through their array
        // projection.
        let operation = PlaceOperation::new(placement(&mesh_x(), vec![ShardingDimension::Replicated]), None);
        let array_type = ArrayType::new_static(DataType::F32, [2]);
        assert_eq!(
            ArrayOperation::<Array>::provide(operation.clone(), &[&array_type]).unwrap().to_string(),
            operation.to_string(),
        );
        assert_eq!(
            ArrayOperation::<Array>::provide(operation.clone(), &[]).unwrap_err(),
            ProgramError::InvalidInputCount { expected: 1, actual: 0 },
        );
        let ir_type = ArrayIrType::Array(array_type);
        assert!(matches!(
            ArrayIrOperation::<Array>::provide(operation.clone(), &[&ir_type]).unwrap(),
            ArrayIrOperation::Array(ArrayOperation::Place(provided)) if provided == operation,
        ));
    }

    #[test]
    fn test_placed() {
        /// Value that reports a fixed placement.
        struct FixedPlacement(Placement);

        impl Placed for FixedPlacement {
            fn placement(&self) -> Option<Placement> {
                Some(self.0.clone())
            }
        }

        // Reference arrays, staged and unknown values, and batches of reference arrays have no placement.
        let array = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let array_type = array.r#type().into_owned();
        assert_eq!(array.placement(), None);
        let tracer = TracingContext::<Array, ArrayOperation<Array>>::new().input(array_type.clone());
        assert_eq!(tracer.placement(), None);
        let partial = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(PartialTracer::new(partial.clone(), partial.unknown_input(array_type.clone(), 0)).placement(), None);
        let batching = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);
        assert_eq!(batching.lift(array.clone()).unwrap().placement(), None);

        // Wrappers report the placement of the value that they wrap, which a reference array does not have.
        assert_eq!(partial.lift(array.clone()).unwrap().placement(), None);
        let differentiation = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(array.clone()).unwrap(),
            DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new()),
        );
        assert_eq!(differentiation.placement(), None);
        assert_eq!(ProjectedValue::new(array.clone(), array_type.clone()).placement(), None);
        let fixed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        assert_eq!(ProjectedValue::new(FixedPlacement(fixed.clone()), array_type).placement(), Some(fixed));

        // Only the array members of array IR values live on devices.
        assert_eq!(ArrayIrValue::Array(array).placement(), None);
        assert_eq!(ArrayIrValue::<Array>::Dimension(DimensionValue::constant(3).unwrap()).placement(), None);

        // Batched values report the placement of their batch items, which drops the batch dimension from the sharding
        // of the packed value along with its layout, keeps its memory, and records the sharding of the batch dimension
        // as how the batch is distributed. Replicated values report the placement of their packed value with a
        // replicated batch recorded, because each of their batch items is the whole packed value.
        let context = RecordingContext::default();
        let packed = placement(&mesh_x(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
            .with_memory(PINNED_HOST)
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));
        let item = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])
            .with_memory(PINNED_HOST)
            .with_batch_sharding(ShardingDimension::Replicated)
            .unwrap();
        let split_item = placement(&mesh_x(), vec![ShardingDimension::Replicated])
            .with_memory(PINNED_HOST)
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap();
        let replicated_item = packed.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap();
        let packed_array = PlacedArray {
            array: Array::matrix(2, 2, vec![1.0f32, 2.0, 3.0, 4.0]).unwrap(),
            placement: Some(packed.clone()),
            context: context.clone(),
        };
        let mapped = ArrayBatch::new(packed_array.clone(), BatchAxis::new(0)).unwrap();
        assert_eq!(mapped.placement(), Some(item.clone()));
        assert_eq!(ArrayBatch::new(packed_array.clone(), BatchAxis::new(1)).unwrap().placement(), Some(split_item));
        assert_eq!(ArrayBatch::replicated(packed_array.clone()).placement(), Some(replicated_item.clone()));
        let packed_ir_value = ArrayIrValue::Array(packed_array.clone());
        assert_eq!(
            ArrayIrBatch::new(packed_ir_value.clone(), BatchAxis::new(0)).unwrap().placement(),
            Some(item.clone()),
        );
        assert_eq!(ArrayIrBatch::replicated(packed_ir_value).placement(), Some(replicated_item));
        let batching = BatchingContext::<_, ArrayBatchingPolicy>::new(context.clone(), 2);
        assert_eq!(BatchingTracer::new(batching, mapped).placement(), Some(item));
        assert_eq!(
            ArrayBatch::new(context.lift(packed_array.array).unwrap(), BatchAxis::new(0)).unwrap().placement(),
            None,
        );
    }

    #[test]
    fn test_place_dispatch() {
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let expected = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2]).with_sharding(target.sharding().clone()).unwrap(),
            &[1.0f32, 2.0],
        )
        .unwrap();

        // A reference array shares its storage by default and copies it when the copy policy forbids sharing it.
        let shared =
            ValueDirectDispatch::place(&input, Some(&PlacementRequest::from(target.clone())), None, true).unwrap();
        assert_eq!(shared, expected);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), shared.shared_storage_bytes()));
        let copied =
            ValueDirectDispatch::place(&input, Some(&PlacementRequest::from(target.clone())), None, false).unwrap();
        assert_eq!(copied, expected);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));

        // Without a target, a reference array keeps its type and follows the copy policy in the same way.
        let shared = ValueDirectDispatch::place(&input, None, None, true).unwrap();
        assert_eq!(shared, input);
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), shared.shared_storage_bytes()));
        let copied = ValueDirectDispatch::place(&input, None, None, false).unwrap();
        assert_eq!(copied, input);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));
        assert_eq!(
            ValueDirectDispatch::place(&input, None, Some(PlacementRequest::from(placement(&mesh_x(), vec![]))), true),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding rank (0) does not match the input rank (1)",
            )))),
        );

        // Array IR values pass the copy policy on to their array members.
        let ir_input = ArrayIrValue::Array(input.clone());
        let ArrayIrValue::Array(copied) =
            ValueDirectDispatch::place(&ir_input, Some(&PlacementRequest::from(target.clone())), None, false).unwrap()
        else {
            panic!("placing an array member must produce an array member");
        };
        assert_eq!(copied, expected);
        assert!(!Arc::ptr_eq(input.shared_storage_bytes(), copied.shared_storage_bytes()));

        // Values that dispatch through their domain bind the placement with the copy policy.
        let context = RecordingContext::default();
        let source = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let placed = context.lift(Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap()).unwrap();
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        ValueDomainDispatch::place(
            &placed,
            Some(&PlacementRequest::from(target.clone())),
            Some(PlacementRequest::from(source.clone())),
            false,
        )
        .unwrap();
        ValueDomainDispatch::place(&placed, Some(&PlacementRequest::from(target.clone())), None, true).unwrap();
        ValueDomainDispatch::place(&placed, None, None, false).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(target.clone(), source).with_may_alias(false)),
                ArrayOperation::Place(PlaceOperation::new(target, None)),
                ArrayOperation::Place(PlaceOperation::new(None, None).with_may_alias(false)),
            ],
        );
    }

    #[test]
    fn test_place_dispatch_shard_layouts() {
        // A layout describes each shard, so the type rule accepts the byte strides `[8, 4]` of a dense `f32[4, 2]`
        // column shard for an `f32[4, 4]` value that is split into two column shards. A reference array stores the
        // whole value in one buffer, which those strides would overlap, so it rejects them with an error that says so,
        // eagerly and when the program that partial evaluation specializes on a known input executes.
        let columns = Sharding::new(
            mesh_x().logical_mesh().clone(),
            vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])],
        )
        .unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [4, 4]).with_sharding(columns.clone()).unwrap();
        let input =
            Array::from_elements(input_type.clone(), &(1..=16).map(|value| value as f32).collect::<Vec<_>>()).unwrap();
        let strides = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let target = PlacementRequest::from(Placement::new(mesh_x(), columns).unwrap().with_layout(strides.clone()));
        let output_type = input_type.clone().with_layout(strides);
        assert_eq!(
            PlaceOperation::new(target.clone(), None).infer_output_types(&[input_type.clone()], &[]),
            Ok(vec![output_type.clone()]),
        );
        let error = ProgramError::UnsupportedOperation {
            message: format!(
                "the reference `Array` backend stores a whole value in one buffer, so it cannot materialize the shard \
                 layout of `{output_type}` for it: strided layout stride 8 on axis 0 is smaller than the 16-byte span \
                 occupied by more minor axes and may alias array elements",
            ),
        };
        assert_eq!(ValueDirectDispatch::place(&input, Some(&target), None, true), Err(error.clone()));
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |x: Vec<_>| {
                Ok(vec![<ValueDomainDispatch as PlaceDispatch<ArrayType, _>>::place(&x[0], Some(&target), None, true)?])
            },
            vec![input_type],
        )
        .unwrap();
        // Partial evaluation does not fold the placement of the known input, whose interpretation fails, so it stays in
        // the specialized program, which reports the same error when it executes.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(input)]).unwrap();
        assert_eq!(
            evaluation.program().to_string(),
            indoc! {"
                lambda %0:f32[4, 4][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] .
                let %1:f32[4, 4][layout=strided{8,4}][sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], sharding={mesh<['x'=2:explicit]>, [{}, {'x'}]}, \
                        layout=strided{8,4}},
                ] %0
                in (%1)"
            },
        );
        assert_eq!(evaluation.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[]), Err(error));

        // The storage of a reference array is publicly mutable, so a placement that re-encodes it for another layout
        // rejects an element encoding that is not valid with an error rather than a panic, as the upload of the same
        // storage to a backend does.
        let mut input = Array::from_elements(ArrayType::new_static(DataType::Boolean, [2]), &[true, false]).unwrap();
        input.storage_bytes_mut()[0] = 2;
        let reversed =
            PlacementRequest::InPlace { memory: None, layout: Some(Layout::Strided(StridedLayout::new(vec![-1]))) };
        assert_eq!(
            ValueDirectDispatch::place(&input, Some(&reversed), None, true).map_err(|error| error.to_string()),
            Err("array element 0 has invalid `bool` byte encoding [2]".to_string()),
        );
    }

    #[test]
    fn test_placement_sources() {
        let x = placement(&mesh_x(), vec![]);

        // An optional structure of placements supplies the source of every leaf or of none.
        let sources = (x.clone(), vec![x.clone()]);
        assert_eq!(PlacementSources::as_parameterized(&Some(&sources)), Some(&sources));
        assert_eq!(PlacementSources::<Placement, (Placement, Vec<Placement>)>::as_parameterized(&None), None);

        // A structure of optional placements is always supplied, and each of its leaves can omit its own source.
        let sources = (Some(x), vec![None]);
        assert_eq!(PlacementSources::as_parameterized(&&sources), Some(&sources));
    }

    #[test]
    fn test_place() {
        // Staged values have no concrete placement, so placing them without a source stages an unknown source.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |input| input.place(&target, None),
            ArrayType::new_static(DataType::F32, [4]),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayType::new_static(DataType::F32, [4]).with_sharding(target.sharding().clone()).unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                in (%1)"
            },
        );

        // Array IR values place their array members, with a given source staged as is.
        let source = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let (output_type, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| input.place(&target, Some(&source)),
            ArrayIrType::Array(ArrayType::new_static(DataType::F32, [4])),
        )
        .unwrap();
        assert_eq!(
            output_type,
            ArrayIrType::Array(
                ArrayType::new_static(DataType::F32, [4]).with_sharding(target.sharding().clone()).unwrap()
            ),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4] .
                let %1:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    source={mesh<['y'=4:explicit]>, devices=[2, 3, 4, 5], \
                        sharding={mesh<['y'=4:explicit]>, [{}]}},
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                in (%1)"
            },
        );
        assert_eq!(
            ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap()).place(&target, None),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // Reference arrays have no devices, so placing them keeps their payload and records the placement in their
        // type, and they validate the target and source exactly like staged programs do.
        let input = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let output = input.place(&target.clone().with_memory(PINNED_HOST), None).unwrap();
        assert_eq!(
            output,
            Array::from_elements(
                ArrayType::new_static(DataType::F32, [2])
                    .with_sharding(target.sharding().clone())
                    .unwrap()
                    .with_memory(PINNED_HOST),
                &[1.0f32, 2.0],
            )
            .unwrap(),
        );
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
        assert_eq!(
            input.place(&placement(&mesh_x(), vec![]), None),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` target sharding rank (0) does not match the input rank (1)",
            )))),
        );
        assert_eq!(
            input.place(&target, Some(&placement(&mesh_x(), vec![]))),
            Err(ProgramError::Type(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` source sharding rank (0) does not match the input rank (1)",
            )))),
        );
    }

    #[test]
    fn test_place_requests() {
        // A tree of requests can mix every form, and each leaf is placed as its request says. Reference arrays have no
        // devices, so a device target keeps the sharding of its leaf and an in-place target changes only its storage.
        let target = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let targets = (
            PlacementRequest::from(target.clone()),
            PlacementRequest::from(Device::new(0, 0)),
            PlacementRequest::from(PINNED_HOST),
        );
        let (output_type, program) = TracingContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: (_, _, _)| inputs.place(&targets, None),
            (
                ArrayType::new_static(DataType::F32, [4]),
                ArrayType::new_static(DataType::F32, [4]),
                ArrayType::new_static(DataType::F32, [4]),
            ),
        )
        .unwrap();
        assert_eq!(
            output_type,
            (
                ArrayType::new_static(DataType::F32, [4]).with_sharding(target.sharding().clone()).unwrap(),
                ArrayType::new_static(DataType::F32, [4]),
                ArrayType::new_static(DataType::F32, [4]).with_memory(PINNED_HOST),
            ),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[4], %1:f32[4], %2:f32[4] .
                let %3:f32[4][sharding={mesh<['x'=2:explicit]>, [{'x'}]}] = place [
                    target={mesh<['x'=2:explicit]>, devices=[0, 1], \
                        sharding={mesh<['x'=2:explicit]>, [{'x'}]}},
                ] %0
                    %4:f32[4] = place [target=device(0)] %1
                    %5:f32[4]@Host[Pinned] = place [target=in_place(memory=Host[Pinned])] %2
                in (%3, %4, %5)"
            },
        );

        // Trees of resolved placements and of memories place every leaf with their one form.
        let inputs = (Array::vector(vec![1.0f32, 2.0]).unwrap(), Array::vector(vec![3.0f32, 4.0]).unwrap());
        let sharded_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(target.sharding().clone()).unwrap();
        assert_eq!(
            inputs.place(&(target.clone(), target.clone()), None),
            Ok((
                Array::from_elements(sharded_type.clone(), &[1.0f32, 2.0]).unwrap(),
                Array::from_elements(sharded_type, &[3.0f32, 4.0]).unwrap(),
            )),
        );
        let pinned_type = ArrayType::new_static(DataType::F32, [2]).with_memory(PINNED_HOST);
        assert_eq!(
            inputs.place(&(PINNED_HOST, Memory::Device), None),
            Ok((Array::from_elements(pinned_type, &[1.0f32, 2.0]).unwrap(), inputs.1.clone())),
        );
    }

    #[test]
    fn test_place_layouts() {
        // A layout determines how the storage addresses the elements, so a reference array placed with another layout
        // re-encodes its elements instead of reinterpreting its storage. Reversing the byte stride stores the elements
        // in reverse order, and widening it leaves a hole between them, while the elements stay the same.
        let input = Array::from_logical_bytes(ArrayType::new_static(DataType::U8, [2]), &[10, 20]).unwrap();
        let target = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let output_type = |stride| {
            ArrayType::new_static(DataType::U8, [2])
                .with_sharding(target.sharding().clone())
                .unwrap()
                .with_layout(Layout::Strided(StridedLayout::new(vec![stride])))
        };
        let output = input.place(&target.clone().with_layout(Layout::Strided(StridedLayout::new(vec![-1]))), None);
        assert_eq!(output, Array::new(output_type(-1), vec![20, 10]));
        let output = output.unwrap();
        assert_eq!(output.elements::<u8>(), Ok(vec![10, 20]));
        assert_eq!(output.logical_bytes(), vec![10, 20]);
        let output = input.place(&target.clone().with_layout(Layout::Strided(StridedLayout::new(vec![2]))), None);
        assert_eq!(output, Array::new(output_type(2), vec![10, 0, 20]));
        let output = output.unwrap();
        assert_eq!(output.elements::<u8>(), Ok(vec![10, 20]));
        assert_eq!(output.logical_bytes(), vec![10, 20]);

        // An unchanged layout keeps sharing the storage.
        let output = input.place(&target, None).unwrap();
        assert!(Arc::ptr_eq(input.shared_storage_bytes(), output.shared_storage_bytes()));
    }

    #[test]
    fn test_place_source_inference() {
        let context = RecordingContext::default();
        let concrete = placement(&mesh_x(), vec![ShardingDimension::Replicated]);
        let explicit = placement(&mesh_y(), vec![ShardingDimension::Replicated]);
        let target = placement(&mesh_y(), vec![ShardingDimension::sharded(["y"])]);
        let array = Array::vector(vec![1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let placed = PlacedArray { array: array.clone(), placement: Some(concrete), context: context.clone() };
        let unplaced = context.lift(array).unwrap();

        // The capability places every value with the source as it is given. The rules that execute the placement on a
        // value with a concrete placement complete it (refer to `test_place_operation_interpretation`), after they
        // validate the request as it is given, so the binding of an eager context records the given source.
        placed.place(&target, None).unwrap();
        placed.place(&target, Some(&explicit)).unwrap();
        unplaced.place(&target, None).unwrap();
        assert_eq!(
            context.0.borrow().as_slice(),
            &[
                ArrayOperation::Place(PlaceOperation::new(target.clone(), None)),
                ArrayOperation::Place(PlaceOperation::new(target.clone(), explicit)),
                ArrayOperation::Place(PlaceOperation::new(target.clone(), None)),
            ],
        );

        // An in-place source that names nothing is rejected by the type rule, also for a value with a concrete
        // placement, with which it would otherwise be completed into a valid source.
        let empty = Some(PlacementRequest::InPlace { memory: None, layout: None });
        let error = ProgramError::Type(TypeError::invalid(format!(
            "`{PLACE_OPERATION_NAME}` in-place source must name a memory or a layout",
        )));
        let target = Some(PlacementRequest::Resolved(target));
        assert_eq!(placed.place(&target, &empty).map(|output| output.array), Err(error.clone()));
        let operation = PlaceOperation::new(target, empty);
        assert_eq!(
            operation
                .interpret(&context, &EmptyRegionDriver, std::slice::from_ref(&placed))
                .map(|outputs| outputs.len()),
            Err(error),
        );
        assert_eq!(context.0.borrow().len(), 3);
    }

    #[test]
    fn test_place_structured() {
        /// Exercises a single-value caller that specifies its universe.
        fn place_array<V: Value + Place<ArrayType>>(input: &V, target: &Placement) -> Result<V, ProgramError> {
            input.place(target, None)
        }

        /// Only the placeholder structure needs to be cloneable to rebuild this receiver.
        #[derive(ryft_macros::Parameterized)]
        struct Inputs<P: Parameter> {
            /// Vector and scalar groups with different placement ranks.
            values: (P, Vec<P>),

            /// Static metadata supplied by the receiver.
            label: &'static str,
        }

        impl Clone for Inputs<Placeholder> {
            fn clone(&self) -> Self {
                Self { values: self.values.clone(), label: self.label }
            }
        }

        let vector = Array::vector(vec![1.0f32, 2.0]).unwrap();
        let scalar = Array::scalar(3.0f32).unwrap();
        let x = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])]);
        let y = placement(&mesh_y(), vec![]).with_memory(PINNED_HOST);

        // Every leaf is placed on the target at its parameter path, and the rest of the structure is kept.
        let inputs = Inputs { values: (vector.clone(), vec![scalar.clone()]), label: "inputs" };
        let targets = Inputs { values: (x.clone(), vec![y.clone()]), label: "targets" };
        let outputs = inputs.place(&targets, None).unwrap();
        assert_eq!(outputs.label, "inputs");
        assert_eq!(outputs.values, (place_array(&vector, &x).unwrap(), vec![scalar.place(&y, None).unwrap()]));

        // Structures of array IR values place their array members, and their other members cannot be placed.
        let inputs = vec![ArrayIrValue::Array(scalar.clone())];
        assert_eq!(
            inputs.place(&vec![y.clone()], None),
            Ok(vec![ArrayIrValue::Array(scalar.place(&y, None).unwrap())]),
        );
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(1).unwrap());
        assert_eq!(
            vec![dimension].place(&vec![y], None),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );

        // Structures without leaves place nothing.
        assert!(Vec::<Array>::new().place(&Vec::<Placement>::new(), None).unwrap().is_empty());
        assert_eq!(<() as Place<ArrayType, Array>>::place::<Placement, _>(&(), &(), Some(&())), Ok(()));
    }

    #[test]
    fn test_place_structured_sources() {
        let context = RecordingContext::default();
        let x = placement(&mesh_x(), vec![]);
        let y = placement(&mesh_y(), vec![]);
        let array = Array::scalar(1.0f32).unwrap();
        let first = PlacedArray { array: array.clone(), placement: Some(x.clone()), context: context.clone() };
        let second = PlacedArray { array: array.clone(), placement: Some(y.clone()), context: context.clone() };
        let inputs = (first, [second, context.lift(array).unwrap()]);
        let targets = (y.clone(), [x.clone(), x.clone()]);

        // Without sources, every leaf is placed without one, and the rules that execute the placement on a leaf that
        // reports where it concretely lives record that placement (refer to `test_place_operation_interpretation`).
        inputs.place(&targets, None).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(y.clone(), None)),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), None)),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), None)),
            ],
        );

        // A structure of sources gives every leaf its source.
        let sources = (y.clone(), [x.clone(), y.clone()]);
        inputs.place(&targets, Some(&sources)).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(y.clone(), y.clone())),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), x.clone())),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), y.clone())),
            ],
        );

        // A structure of optional sources gives a source only to the leaves that it names a source for.
        inputs.place(&targets, &(None, [Some(x.clone()), None])).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(y.clone(), None)),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), x.clone())),
                ArrayOperation::Place(PlaceOperation::new(x.clone(), None)),
            ],
        );

        // Sources follow the same conversion as targets, so they can be devices, memories, or requests of any form,
        // with or without options, and so can the targets.
        let device = Device::new(3, 0);
        let memory = Memory::Device;
        inputs.place(&(Some(memory), [Some(memory), None]), &(Some(device), [None, Some(device)])).unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(Memory::Device, device)),
                ArrayOperation::Place(PlaceOperation::new(Memory::Device, None)),
                ArrayOperation::Place(PlaceOperation::new(None, device)),
            ],
        );
        // An in-place source is placed as it is given too.
        let in_place = PlacementRequest::from(Memory::Device);
        inputs
            .place(
                &(Some(x.clone()), [None, Some(y.clone())]),
                &(Some(in_place.clone()), [Some(in_place.clone()), None]),
            )
            .unwrap();
        assert_eq!(
            context.0.take(),
            vec![
                ArrayOperation::Place(PlaceOperation::new(x, in_place.clone())),
                ArrayOperation::Place(PlaceOperation::new(None, in_place)),
                ArrayOperation::Place(PlaceOperation::new(y, None)),
            ],
        );
    }

    #[test]
    fn test_place_structured_mismatched_parameter_paths() {
        let context = RecordingContext::default();
        let input = context.lift(Array::scalar(1.0f32).unwrap()).unwrap();
        let inputs = BTreeMap::from([("a", input.clone()), ("b", input)]);
        let x = placement(&mesh_x(), vec![]);
        let targets = BTreeMap::from([("a", x.clone()), ("b", x.clone())]);
        let mismatched = BTreeMap::from([("a", x.clone()), ("c", x)]);
        let target_error = ProgramError::InvalidArgument {
            message: "receiver and `targets` must have the same parameter structure".to_string(),
        };
        let source_error = ProgramError::InvalidArgument {
            message: "receiver and `sources` must have the same parameter structure".to_string(),
        };

        // The targets and the sources must have the parameter paths of the receiver, and no leaf is placed otherwise.
        assert_eq!(inputs.place(&mismatched, None).unwrap_err(), target_error);
        assert_eq!(inputs.place(&targets, Some(&mismatched)).unwrap_err(), source_error);
        let missing = BTreeMap::from([("a", None::<Placement>)]);
        assert_eq!(inputs.place(&targets, &missing).unwrap_err(), source_error);
        let wrong_paths = BTreeMap::from([("a", None::<Placement>), ("c", None)]);
        assert_eq!(inputs.place(&targets, &wrong_paths).unwrap_err(), source_error);
        assert!(context.0.borrow().is_empty());
    }

    #[test]
    fn test_place_structured_duplicate_parameter_paths() {
        /// Map key whose `Debug` text is the same for every key, so that distinct keys have the same parameter path.
        #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
        struct Key(u32);

        impl std::fmt::Debug for Key {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("key")
            }
        }

        // The entries of distinct keys with the same `Debug` text cannot be told apart by path, so a target keyed by
        // another key with that text would be applied to them by position. Such a receiver is rejected before any leaf
        // is placed, with the error of the tree prefix expansion of `Session::place`, whatever its targets are.
        let context = RecordingContext::default();
        let input = context.lift(Array::scalar(1.0f32).unwrap()).unwrap();
        let inputs = BTreeMap::from([(Key(1), input.clone()), (Key(2), input)]);
        let targets = BTreeMap::from([(Key(1), Memory::Device), (Key(9), PINNED_HOST)]);
        let error =
            ProgramError::Parameter(ParameterError::DuplicateParameterPaths { paths: vec!["$[key]".to_string()] });
        assert_eq!(inputs.place(&targets, None).unwrap_err(), error);
        assert_eq!(
            targets
                .broadcast_to_parameter_structure::<BTreeMap<Key, Memory>>(targets.parameter_structure())
                .err(),
            Some(ParameterError::DuplicateParameterPaths { paths: vec!["$[key]".to_string()] }),
        );
        assert!(context.0.borrow().is_empty());
    }

    #[test]
    fn test_place_structured_mismatched_empty_containers() {
        let context = RecordingContext::default();
        let input = context.lift(Array::scalar(1.0f32).unwrap()).unwrap();
        let x = placement(&mesh_x(), vec![]);
        let target_error = ProgramError::InvalidArgument {
            message: "receiver and `targets` must have the same parameter structure".to_string(),
        };
        let source_error = ProgramError::InvalidArgument {
            message: "receiver and `sources` must have the same parameter structure".to_string(),
        };

        // Structures whose leaf paths coincide but that differ in a container without children are different, whether
        // the receiver or the placements have the extra empty sibling.
        let with_empty = vec![vec![input.clone()], Vec::new()];
        let without_empty = vec![vec![input]];
        let targets_with_empty = vec![vec![x.clone()], Vec::new()];
        let targets_without_empty = vec![vec![x.clone()]];
        assert_eq!(with_empty.place(&targets_without_empty, None).unwrap_err(), target_error);
        assert_eq!(without_empty.place(&targets_with_empty, None).unwrap_err(), target_error);
        assert_eq!(with_empty.place(&targets_with_empty, Some(&targets_without_empty)).unwrap_err(), source_error);
        assert_eq!(without_empty.place(&targets_without_empty, Some(&targets_with_empty)).unwrap_err(), source_error);
        assert_eq!(
            without_empty.place(&targets_without_empty, &vec![vec![Some(x.clone())], Vec::new()]).unwrap_err(),
            source_error,
        );
        assert!(context.0.borrow().is_empty());

        // Matching empty containers are accepted and kept.
        assert_eq!(with_empty.place(&targets_with_empty, Some(&targets_with_empty)).unwrap().len(), 2);
        assert_eq!(context.0.take(), vec![ArrayOperation::Place(PlaceOperation::new(x.clone(), x))]);
    }

    #[test]
    fn test_unbatched_placement() {
        let packed = placement(&mesh_x(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::Replicated])
            .with_memory(PINNED_HOST)
            .with_layout(Layout::Tiled(TiledLayout::new(vec![0, 1], vec![])));

        // The batch dimension leaves the sharding and is recorded as how the batch is distributed, the memory is kept,
        // and the layout, which describes the storage of the packed value, is dropped.
        assert_eq!(
            unbatched_placement(packed.clone(), Some(0)),
            Some(
                placement(&mesh_x(), vec![ShardingDimension::Replicated])
                    .with_memory(PINNED_HOST)
                    .with_batch_sharding(ShardingDimension::sharded(["x"]))
                    .unwrap(),
            ),
        );
        assert_eq!(
            unbatched_placement(packed.clone(), Some(1)),
            Some(
                placement(&mesh_x(), vec![ShardingDimension::sharded(["x"])])
                    .with_memory(PINNED_HOST)
                    .with_batch_sharding(ShardingDimension::Replicated)
                    .unwrap(),
            ),
        );

        // A sharding that has no dimension at the batch axis describes no batch items.
        assert_eq!(unbatched_placement(packed.clone(), Some(2)), None);

        // A value without a batch dimension is its own batch item, so it keeps its placement, including its layout,
        // with a replicated batch recorded.
        assert_eq!(
            unbatched_placement(packed.clone(), None),
            Some(packed.clone().with_batch_sharding(ShardingDimension::Replicated).unwrap()),
        );

        // The batch shardings of enclosing batching levels are kept from the outermost to the innermost.
        let item = unbatched_placement(unbatched_placement(packed, Some(0)).unwrap(), Some(0)).unwrap();
        assert_eq!(item.sharding().rank(), 0);
        assert_eq!(item.batch_shardings(), &[ShardingDimension::sharded(["x"]), ShardingDimension::Replicated]);
    }
}
