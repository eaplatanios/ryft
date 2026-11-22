//! Placement of arrays on devices, which is the `ryft` analogue of JAX's
//! [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
//!
//! A placement goes through three representations. An [`ArrayPlacement`] is a request, which names a bare device, a
//! bare memory, a device mesh with a sharding, or a sharding over the [default mesh](Session::device_mesh) of a
//! session, and which may leave the memory and layout unspecified. A core [`PlacementRequest`] is the form in which a
//! placement is recorded, which is a resolved [`Placement`] (i.e., the concrete devices, a sharding with the rank of
//! the value, and the selected memory and layout, which inside a batched function also records how the enclosing
//! batches are distributed), a bare device, or a change of storage in place. The backend then executes the placement on
//! concrete arrays. A runtime [`Array`] moves its buffers with the strategies of `arrays::resharding`, and a traced
//! value records a [`PlaceOperation`], which executes like that wherever its program runs on concrete arrays, and which
//! a compiled function lowers to sharding, layout, and memory constraints over the devices of the mesh that it is
//! compiled for.
//!
//! [`PlacementRequest`] states the one contract of every form, which every path follows. A request that names a mesh
//! or a sharding resolves to a [`Placement`] when it is given, and a bare device or a bare memory is recorded in the
//! form that it was given (i.e., as a [`PlacementRequest::Device`] or a [`PlacementRequest::InPlace`]), whether or not
//! the traced value wraps a concrete array, so that the type rule of [`PlaceOperation`] alone decides the type of the
//! result. Such a request resolves against a concrete array in exactly one place, when the placement executes on that
//! array, through the same resolver that [`Session::place`] uses for an [`Array`] leaf (refer to
//! [`ArrayPlacementTarget::Device`] and [`ArrayPlacementTarget::InPlace`]). A bare device gathers an array on more
//! devices onto the canonical logical mesh of one device ([`LogicalMesh::single_device`]), which is also the mesh of
//! a value that is uploaded without one, and it selects the default memory of the device when it names none. A bare
//! memory leaves a value on the devices where it lives with the distribution that it has and changes only its
//! storage, as `jax.device_put(x, jax.memory.Space.Host)` does. A source is informative about devices on every path,
//! and a source that names only a memory or a layout is completed with the placement where a value concretely lives
//! (refer to [`PlacementRequest::resolve_source`]). Inside a compiled function, every device that a placement names,
//! in its target or in its source, must belong to the program, because transposition turns a source into the target
//! of the cotangent (refer to [`Session::place`]).
//!
//! There are three entry points.
//!
//!   - [`Session::place`] accepts a tree of [`Placeable`] leaves, which are host values that it uploads, runtime
//!     [`Array`]s, and traced values of compiled, differentiated, or batched functions, and it takes requests or
//!     resolved placements through [`PlacementOptions`].
//!   - [`Array::to_placement`] and [`Array::into_placement`] move one runtime [`Array`], the latter donating its
//!     buffers.
//!   - The core [`Place`] capability places any value that implements it, including traced values in every context,
//!     with targets and sources of every form of [`PlacementRequest`] (e.g., resolved [`Placement`]s, [`Device`]s, and
//!     [`Memory`] spaces), which it records without resolving them against a session.
//!
//! The destination, source, donation, and aliasing options of [`Session::place`] are each a tree prefix of the input,
//! which is broadcast over the subtree under each of its leaves, so one destination can place a whole tree. A `None`
//! destination or source leaf omits that option for its subtree, and a prefix leaf over a subtree without leaves
//! places nothing. A request whose sharding is shorter than the rank of a leaf replicates the omitted trailing
//! dimensions, so one request can place leaves of different ranks.
//!
//! Placing runtime arrays issues the transfers and usually returns without waiting for them, so the readiness
//! functions and the readbacks of each resulting [`Array`] (e.g., [`Array::block_until_ready`]) are what report their
//! completion and any failure. A placement can still block the calling thread in the cases that `arrays::resharding`
//! documents (e.g., on the first cross-host rendezvous between two devices of the CUDA plugin, and on host readbacks
//! when the PJRT plugin lacks host to device transfer managers). Every [`Array`] that is placed on a destination, a
//! bare memory included, is [committed](Array::is_committed) to its devices, a host value uploaded without one is
//! uncommitted, and an [`Array`] without a destination keeps its commitment. A traced value stays traced, and these
//! promises apply to the arrays that its placement produces when its program executes.

use std::borrow::Cow;
use std::sync::{Arc, Weak};

use half::{bf16, f16};

use ryft_core::{
    Array as CoreArray, ArrayType, BatchingPolicy, BatchingTracer, Complex, Context, Device, DeviceMesh,
    DifferentiationPolicy, DifferentiationTracer, Dimension, Layout, LogicalMesh, Memory, PLACE_OPERATION_NAME,
    Parameter, Parameterized, ParameterizedFamily, PartialTracer, PlaceDispatch, PlaceOperation, Placed, Placement,
    PlacementRequest, ProjectedValue, Shape, Sharding, ShardingDimension, Tracer, TypeError, Typed, UnreducedKind,
    Value as CoreValue, check_sharding, decode_logical_bytes, f4e2m1fn, f6e2m3fn, f6e3m2fn, f8e3m4, f8e4m3,
    f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2, u4,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, Error as PjrtError};

use crate::arrays::shards::ArrayShard;
use crate::arrays::{Array, ArrayError};
use crate::{Error, FromPjrt, Session, Value};

#[cfg(doc)]
use ryft_core::Place;

// TODO(eaplatanios): Review this module.

/// Placement of an [`Array`] on devices. It specifies a destination for [`Session::place`],
/// [`Array::to_placement`], and [`Array::into_placement`], or an expected source in [`PlacementOptions`]. Its
/// [`ArrayPlacementTarget`] selects the devices and the sharding, and its optional memory and layout select the
/// storage of each shard. Destination and source trees in [`PlacementOptions`] can use `Some(placement)` and
/// `None` together. A `None` leaf omits the option for its branch.
///
/// An [`ArrayPlacement`] is a request that may depend on a [`Session`] (e.g., on its default mesh) and on the array
/// that it applies to. It resolves against them to a [`Placement`], which names the concrete devices, the sharding
/// over them, and the selected memory and layout, and which is what a [`PlaceOperation`]
/// records. A sharding with fewer dimensions than the array leaves the omitted trailing dimensions replicated, as a
/// short JAX partition specification does, so one request can place arrays of different ranks.
///
/// Use [`Self::with_layout`] and [`Self::with_memory`] to select the storage of each shard. Host uploads use device
/// memory and the backend's default layout when these options are omitted. A destination that is a bare device without
/// a memory selects the default memory of that device, as JAX's eager `device_put` does, on every path (refer to
/// [`PlacementRequest::target_memory`]), whereas existing arrays keep their memory under a mesh or sharding destination
/// without a memory. An explicit layout applies to the destination shard buffers, and without one, a destination keeps
/// the layout of an array only while it still describes each destination shard. An [in-place](Self::in_place) placement
/// (e.g., a bare [`Memory`]) selects only the memory and the layout, and it leaves an array on its devices with its
/// sharding (refer to [`ArrayPlacementTarget::InPlace`]).
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub struct ArrayPlacement {
    /// Devices and sharding of the array.
    target: ArrayPlacementTarget,

    /// Physical layout of the shards, or `None` when this placement does not select one.
    layout: Option<Layout>,

    /// Memory in which the shards are stored, or `None` when this placement does not select one.
    memory: Option<Memory>,

    /// Sharding of the batch dimension of each batched value that encloses the array, from the outermost to the
    /// innermost one (refer to [`Self::batch_shardings`]).
    batch_shardings: Vec<ShardingDimension>,
}

impl ArrayPlacement {
    /// Creates a placement on `target` with its layout and memory unspecified.
    #[inline]
    pub fn new(target: ArrayPlacementTarget) -> Self {
        Self { target, layout: None, memory: None, batch_shardings: Vec::new() }
    }

    /// Creates a single-device placement.
    ///
    /// In a distributed run, `device` must be the same on every process, because the result is treated as an array
    /// whose devices every process knows. A choice of device that differs between processes is expressible only by
    /// naming no device (e.g., with [`ArrayPlacementTarget::InPlace`]), which leaves the choice to each process
    /// (refer to [`crate::arrays::resharding`]).
    #[inline]
    pub fn device(device: Device) -> Self {
        Self::new(ArrayPlacementTarget::Device(device))
    }

    /// Creates a placement on an explicit mesh and sharding.
    ///
    /// # Errors
    ///
    /// Returns an [`Error`] if `sharding` refers to a different logical mesh than `mesh`.
    #[inline]
    pub fn mesh(mesh: DeviceMesh, sharding: Sharding) -> Result<Self, Error> {
        check_sharding!(&mesh, &sharding);
        Ok(Self::new(ArrayPlacementTarget::Mesh { mesh, sharding }))
    }

    /// Creates a placement with `sharding` on the session's [default mesh](Session::device_mesh).
    #[inline]
    pub fn sharding(sharding: Sharding) -> Self {
        Self::new(ArrayPlacementTarget::Sharding(sharding))
    }

    /// Creates a placement that leaves an array on its devices with its sharding (refer to
    /// [`ArrayPlacementTarget::InPlace`]), whose memory and layout [`Self::with_memory`] and [`Self::with_layout`]
    /// select. A bare [`Memory`] converts into such a placement with that memory.
    #[inline]
    pub fn in_place() -> Self {
        Self::new(ArrayPlacementTarget::InPlace)
    }

    /// Sets the physical layout of each shard, or the expected layout when used as a source, which is checked against
    /// the type of the array like every other part of a source (i.e., it must address each shard of the array and
    /// equal the layout that the type of the array declares, if any, refer to [`Session::place`]).
    pub fn with_layout(mut self, layout: Layout) -> Self {
        self.layout = Some(layout);
        self
    }

    /// Sets the memory in which the shards are stored, or the expected memory when used as a source.
    pub fn with_memory(mut self, memory: Memory) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Returns the devices and the sharding that this placement selects.
    #[inline]
    pub fn target(&self) -> &ArrayPlacementTarget {
        &self.target
    }

    /// Returns the physical layout that this placement selects, if any.
    #[inline]
    pub fn layout(&self) -> Option<&Layout> {
        self.layout.as_ref()
    }

    /// Returns the memory that this placement selects, if any.
    #[inline]
    pub fn memory(&self) -> Option<Memory> {
        self.memory
    }

    /// Returns the sharding of the batch dimension of each batched value that encloses the array, from the outermost
    /// to the innermost one, which is the record of [`Placement::batch_shardings`]. Inside a batched function, the
    /// placement of a batch item (refer to [`Placed`]) records how its batch is distributed over the devices, and
    /// converting that [`Placement`] into an [`ArrayPlacement`] keeps this record, so that placing a batch item on it
    /// keeps that distribution. The record is empty for every other placement, which places each batch item on the
    /// whole mesh of its target.
    #[inline]
    pub fn batch_shardings(&self) -> &[ShardingDimension] {
        &self.batch_shardings
    }

    /// Resolves this placement against `session` to the [`Placement`] of a value of rank `rank`, which names the
    /// concrete devices, the sharding over them, and the memory and layout that this placement selects, if any. An
    /// [`ArrayPlacementTarget::Device`] target resolves to the device over the canonical logical mesh of one device
    /// (i.e., [`LogicalMesh::single_device`]) and to the replicated sharding over that mesh, which is the sharding that
    /// the type rule of a [`PlacementRequest::Device`] target gives a gathered value. A sharding with fewer dimensions
    /// than `rank` leaves the omitted trailing dimensions replicated, as a JAX partition specification that is shorter
    /// than the rank of an array does, so one request can place values of different ranks (e.g., a sharding along `x`
    /// splits both a vector and a matrix along their first dimension). A sharding with more dimensions than `rank` is
    /// kept as it is, so the resolved sharding then has more dimensions than the value, and the placement of the value
    /// rejects it.
    ///
    /// The resolved placement keeps the [batch record](Self::batch_shardings) of this placement, so a placement that
    /// was converted from a resolved [`Placement`] resolves back to it. Such a placement is padded like any other,
    /// because the batching rule of [`PlaceOperation`] puts each recorded batch sharding
    /// back at the batch dimension of the packed value and not after the last dimension of a batch item, and replicated
    /// dimensions use no mesh axes, so the padding neither moves the record nor conflicts with it.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::MissingMesh`](crate::SessionError::MissingMesh) when an
    /// [`ArrayPlacementTarget::Sharding`] target needs a default mesh and the session has none, or
    /// [`ShardingError::MeshMismatch`](ryft_core::ShardingError::MeshMismatch) when the sharding and the concrete mesh
    /// have different logical topologies. The batch record was validated when it was recorded, and the padding uses no
    /// mesh axes, so neither adds errors. An [`ArrayPlacementTarget::InPlace`] target names no devices, so it only
    /// resolves against an array (refer to [`Self::resolve_for`]), and this function rejects it with a [`TypeError`].
    pub(crate) fn resolve(self, session: &Session, rank: usize) -> Result<Placement, Error> {
        let (mesh, sharding) = match self.target {
            ArrayPlacementTarget::Device(device) => {
                let mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device])?;
                let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
                (mesh, sharding)
            }
            ArrayPlacementTarget::Mesh { mesh, sharding } => (mesh, sharding),
            ArrayPlacementTarget::Sharding(sharding) => (session.device_mesh()?.clone(), sharding),
            // An in-place target names no devices, so only `Self::resolve_for` resolves it, against an array.
            ArrayPlacementTarget::InPlace => {
                return Err(TypeError::invalid(format!(
                    "`{PLACE_OPERATION_NAME}` in-place placement names no devices to resolve",
                ))
                .into());
            }
        };
        let sharding = match rank.checked_sub(sharding.rank()) {
            Some(padding) if padding > 0 => {
                let padding = std::iter::repeat_n(ShardingDimension::replicated(), padding);
                sharding.with_dimensions(sharding.dimensions().iter().cloned().chain(padding).collect())?
            }
            _ => sharding,
        };
        let mut placement = Placement::new(mesh, sharding)?;
        for batch_sharding in self.batch_shardings {
            placement = placement.with_batch_sharding(batch_sharding)?;
        }
        let placement = match self.layout {
            Some(layout) => placement.with_layout(layout),
            None => placement,
        };
        Ok(match self.memory {
            Some(memory) => placement.with_memory(memory),
            None => placement,
        })
    }

    /// Resolves this placement against `session` to the [`Placement`] that it selects as the destination of `array`, as
    /// [`Self::resolve`] does for a value of its rank. An [`ArrayPlacementTarget::Device`] target names a device and
    /// not a mesh, so it resolves as the contract of [`PlacementRequest`] describes (i.e., an array on a mesh of one
    /// device keeps its logical mesh and its sharding, laid over the target device, without the reduction state that
    /// one device completes, and an array on more devices is gathered onto the target device, replicated over the
    /// canonical logical mesh of one device). A device target without a memory selects the default memory of its
    /// device, as [`PlacementRequest::target_memory`] states for every path. An [`ArrayPlacementTarget::InPlace`]
    /// target resolves to the mesh and the sharding of `array`, with the memory and the layout that it names, so the
    /// array keeps its devices and its distribution. The execution of a staged request on an array resolves through
    /// this function as well (refer to [`Self::from_request`]), so that direct and staged placements of an array agree.
    ///
    /// # Errors
    ///
    /// Returns the error of [`Self::validate_destination`] for an [`ArrayPlacementTarget::InPlace`] target that names
    /// neither a memory nor a layout, and the errors of [`Self::resolve`] otherwise.
    pub(crate) fn resolve_for(self, session: &Session, array: &Array) -> Result<Placement, Error> {
        self.validate_destination()?;
        let destination = match self.target {
            ArrayPlacementTarget::Device(device) => {
                let request = PlacementRequest::Device { device, memory: self.memory, layout: None };
                Self { memory: request.target_memory(), ..self }
            }
            _ => self,
        };
        let placement = match destination.target {
            ArrayPlacementTarget::InPlace => {
                let target =
                    ArrayPlacementTarget::Mesh { mesh: array.mesh().clone(), sharding: array.sharding().clone() };
                Self { target, ..destination }.resolve(session, array.sharding().rank())
            }
            ArrayPlacementTarget::Device(device) if array.mesh().devices().len() == 1 => {
                // Every axis of a mesh of one device has size one, so the array has no reduction state over a larger
                // axis, and all of its reduction state is complete on the target device.
                let mesh = DeviceMesh::new(array.mesh().logical_mesh().clone(), vec![device])?;
                let sharding = array
                    .sharding()
                    .clone()
                    .with_unreduced_axes(Vec::<String>::new())?
                    .with_unreduced_kind(UnreducedKind::Sum)?
                    .with_reduced_axes(Vec::<String>::new())?;
                let target = ArrayPlacementTarget::Mesh { mesh, sharding };
                Self { target, ..destination }.resolve(session, array.sharding().rank())
            }
            _ => destination.resolve(session, array.sharding().rank()),
        }?;
        validate_target_rank(&placement, &array.r#type())?;
        Ok(placement)
    }

    /// Returns the [`ArrayPlacement`] through which a staged `request` is placed when a
    /// [`PlaceOperation`] executes on `array`, so that every path that executes a placement
    /// on an array (e.g., eager execution, and the replay of a staged program, of a derivative, or of a batched
    /// function) resolves it in the one place where [`Session::place`] resolves the destination of an [`Array`] leaf
    /// (i.e., [`Array::run_placement_request`], through [`Self::resolve_for`]). This is the one place where a device or
    /// an in-place request meets a concrete array (refer to [`PlacementRequest`]). The request keeps the memory and the
    /// layout that it names, so that [`Self::resolve_for`] selects the memory of a device request without one exactly
    /// as the type rule of the request does.
    pub(crate) fn from_request(request: &PlacementRequest) -> Self {
        let (placement, memory, layout) = match request {
            PlacementRequest::Resolved(placement) => return Self::from(placement.clone()),
            PlacementRequest::Device { device, memory, layout } => (Self::device(*device), *memory, layout),
            PlacementRequest::InPlace { memory, layout } => (Self::in_place(), *memory, layout),
        };
        let placement = match memory {
            Some(memory) => placement.with_memory(memory),
            None => placement,
        };
        match layout {
            Some(layout) => placement.with_layout(layout.clone()),
            None => placement,
        }
    }

    /// Validates this placement as a destination. An [`ArrayPlacementTarget::InPlace`] destination that names neither
    /// a memory nor a layout changes nothing, which no destination already expresses, so it is rejected with a
    /// [`TypeError`], as the type rule of [`PlaceOperation`] rejects such a target.
    fn validate_destination(&self) -> Result<(), Error> {
        match (&self.target, self.memory, &self.layout) {
            (ArrayPlacementTarget::InPlace, None, None) => Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` in-place target must name a memory or a layout",
            ))
            .into()),
            _ => Ok(()),
        }
    }

    /// Validates this placement as a source. An [`ArrayPlacementTarget::InPlace`] source states the memory or the
    /// layout in which a value is expected (refer to [`PlacementRequest`]), so one that names neither says nothing, and
    /// it is rejected with a [`TypeError`], as the type rule of [`PlaceOperation`] rejects
    /// such a source.
    fn validate_source(&self) -> Result<(), Error> {
        match (&self.target, self.memory, &self.layout) {
            (ArrayPlacementTarget::InPlace, None, None) => Err(TypeError::invalid(format!(
                "`{PLACE_OPERATION_NAME}` in-place source must name a memory or a layout",
            ))
            .into()),
            _ => Ok(()),
        }
    }

    /// Returns the [`PlacementRequest`] that this destination or source stages for a traced value of rank `rank`. The
    /// request keeps the form that it was given, whether or not the traced value wraps a concrete array (refer to
    /// [`PlacementRequest`]). An [`ArrayPlacementTarget::Device`] names a device and not a mesh, so it stays a
    /// [`PlacementRequest::Device`], and an [`ArrayPlacementTarget::InPlace`] names no devices, so it stays a
    /// [`PlacementRequest::InPlace`], each with the memory and the layout of this placement, until the staged
    /// placement executes on a concrete array (refer to [`Self::from_request`]) or is compiled for a device mesh.
    /// Every other placement names a mesh or a sharding, so it resolves to the [`Placement`] that [`Self::resolve`]
    /// returns.
    fn into_request(self, session: &Session, rank: usize) -> Result<PlacementRequest, Error> {
        match self.target {
            ArrayPlacementTarget::Device(device) => {
                Ok(PlacementRequest::Device { device, memory: self.memory, layout: self.layout })
            }
            ArrayPlacementTarget::InPlace => Ok(PlacementRequest::InPlace { memory: self.memory, layout: self.layout }),
            _ => Ok(PlacementRequest::Resolved(self.resolve(session, rank)?)),
        }
    }
}

impl From<Device> for ArrayPlacement {
    #[inline]
    fn from(value: Device) -> Self {
        Self::device(value)
    }
}

impl From<Sharding> for ArrayPlacement {
    #[inline]
    fn from(value: Sharding) -> Self {
        Self::sharding(value)
    }
}

impl From<Memory> for ArrayPlacement {
    #[inline]
    fn from(value: Memory) -> Self {
        Self::in_place().with_memory(value)
    }
}

impl From<Placement> for ArrayPlacement {
    #[inline]
    fn from(value: Placement) -> Self {
        let batch_shardings = value.batch_shardings().to_vec();
        let (mesh, sharding, layout, memory) = value.into_parts();
        Self { target: ArrayPlacementTarget::Mesh { mesh, sharding }, layout, memory, batch_shardings }
    }
}

/// Devices and sharding that an [`ArrayPlacement`] selects, or [`ArrayPlacementTarget::InPlace`], which selects the
/// devices and the sharding that the value already has.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayPlacementTarget {
    /// Commit the value to one concrete device. As a destination of [`Session::place`], it leaves a value that lives
    /// wholly on that device, with the contract of [`PlacementRequest::Device`] for concrete arrays, traced values, and
    /// compiled functions alike. [`Array::to_placement`], the physical primitive below [`Session::place`], also
    /// combines the contributions of a value with a pending reduction over a larger mesh axis onto the device, which
    /// the contract rejects.
    ///
    ///   - A value on a mesh of one device keeps its logical mesh and its sharding, laid over the target device.
    ///   - A value on more than one device is gathered onto the device, replicated over the canonical logical mesh of
    ///     one device (i.e., [`LogicalMesh::single_device`]), which is also the type that the staged placement of a
    ///     traced value infers.
    ///   - A completed reduction and a pending reduction over mesh axes of size one are complete on one device, so the
    ///     value loses its unreduced and reduced state. A value with a pending reduction over a larger mesh axis is
    ///     rejected, an [`Array`] leaf included, because a placement does not combine the contributions of several
    ///     devices (refer to [`PlaceOperation::infer_output_type_allowing_uneven_splits`]), so such a value must be reduced first.
    ///   - A value inside a manual region cannot be moved.
    ///
    /// A traced value stages a [`PlacementRequest::Device`], whether or not it wraps a concrete array, which resolves
    /// against the concrete array that it executes on exactly as this destination resolves for an [`Array`] leaf, or
    /// which a compiled function accepts when the device is the only device of the mesh that the function is compiled
    /// for. As a destination without a memory, it also selects the default memory of that device (e.g., it moves a
    /// value out of pinned host memory), as JAX's eager `device_put` does (refer to
    /// [`PlacementRequest::target_memory`]). As a source, it states that the value is expected on that device, which
    /// every leaf stages as a [`PlacementRequest::Device`] source that transposition moves cotangents back to (refer to
    /// [`Session::place`]). The device is informative, so it is not compared with the device of an [`Array`] leaf,
    /// but a value whose type is tracked over a mesh of more than one device is rejected. As a source without a memory,
    /// it accepts a value in any memory.
    Device(Device),

    /// Commit the value to the provided mesh and sharding.
    Mesh {
        /// Concrete destination mesh describing the device topology.
        mesh: DeviceMesh,

        /// Sharding to apply over `mesh`.
        sharding: Sharding,
    },

    /// Use this sharding on the session's [default mesh](Session::device_mesh), whose logical topology must match
    /// the sharding.
    Sharding(Sharding),

    /// Leave the value on the devices where it lives, with the sharding that it has, and change only its storage to
    /// the memory and the layout of the [`ArrayPlacement`], which must name at least one of them. This is the analogue
    /// of a memory space destination of JAX's `device_put` (e.g., `jax.device_put(x, jax.memory.Space.Host)`), and a
    /// bare [`Memory`] converts into it. It follows the contract of [`PlacementRequest::InPlace`].
    ///
    ///   - An [`Array`] keeps its mesh and its sharding, including its reduction state, and its buffers move to the
    ///     requested memory and layout on their own devices.
    ///   - A traced value stages a [`PlacementRequest::InPlace`], whose devices need not be known while it is traced
    ///     (e.g., the input of a function that is compiled for a mesh whose input sharding the function does not
    ///     know), and which resolves against the concrete array that it executes on, or lowers to a layout constraint
    ///     and a memory transfer in a compiled function, also inside a manual region.
    ///   - A host value is uploaded to the first addressable device, as an upload without a destination is, in the
    ///     requested memory and layout.
    ///
    /// As every destination, it [commits](Array::is_committed) its results, as JAX commits the result of a `device_put`
    /// onto a memory space, eagerly and inside `jit`. As a source, it states only the memory or the layout in which the
    /// value is expected, as the source of `jax.device_put(x, Space.Device, src=Space.Host)` does, which is checked
    /// against the type of the value. A value that concretely lives somewhere (e.g., an [`Array`] leaf, or a traced
    /// value that wraps one) completes it with that placement (refer to [`PlacementRequest::resolve_source`]), so that
    /// transposition returns the cotangent to the devices of the value in that storage, and a staged value without
    /// devices keeps it, so that transposition returns the cotangent in place, which it rejects for a cotangent on a
    /// mesh of another topology than the value's. A source that names neither says nothing, so it is rejected.
    InPlace,
}

/// A destination or source leaf in the tree prefixes of [`PlacementOptions`]. Resolved [`Placement`]s,
/// [`ArrayPlacement`] requests, and bare [`Device`]s, [`Sharding`]s, and [`Memory`] spaces (which convert into
/// [`ArrayPlacement`] requests) are accepted as destinations and as sources, as are their [`Option`]s. A bare memory
/// as a source states the memory in which a value is expected (refer to [`ArrayPlacementTarget::InPlace`]). A [`None`]
/// leaf omits the option for its branch of the input tree.
pub trait PlaceableParameter: Parameter {
    /// Returns the placement to apply to this branch, if any.
    fn into_optional_placement(self) -> Option<ArrayPlacement>;
}

impl<P: Parameter + Into<ArrayPlacement>> PlaceableParameter for P {
    #[inline]
    fn into_optional_placement(self) -> Option<ArrayPlacement> {
        Some(self.into())
    }
}

impl<P: Parameter + Into<ArrayPlacement>> PlaceableParameter for Option<P> {
    #[inline]
    fn into_optional_placement(self) -> Option<ArrayPlacement> {
        self.map(Into::into)
    }
}

/// An aliasing leaf in the tree prefix of [`PlacementOptions`]. A `bool` leaf states whether the results of its branch
/// of the input tree may share buffers with other arrays, and so does an `Option<bool>` leaf, which can also leave the
/// option unspecified with [`None`] (refer to [`Session::place`]). So both `with_may_alias(false)` and
/// `with_may_alias(Some(false))` request a forced copy.
pub trait AliasingParameter: Parameter {
    /// Returns the aliasing option of this branch, which is [`None`] when it is unspecified.
    fn into_may_alias(self) -> Option<bool>;
}

impl AliasingParameter for bool {
    #[inline]
    fn into_may_alias(self) -> Option<bool> {
        Some(self)
    }
}

impl AliasingParameter for Option<bool> {
    #[inline]
    fn into_may_alias(self) -> Option<bool> {
        self
    }
}

/// Options of [`Session::place`].
///
/// Each field follows JAX's tree prefix semantics, so when a field is present, its structure is broadcast over the
/// input tree and applied leafwise. Destination and source prefixes accept [`PlaceableParameter`]s, including
/// [`Placement`], [`ArrayPlacement`], and their optional forms, and aliasing prefixes accept [`AliasingParameter`]s
/// (i.e., `bool` and `Option<bool>`). Each prefix has one leaf type. To mix resolved placements and requests within the
/// same prefix, convert the resolved leaves with [`ArrayPlacement::from`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlacementOptions<Target = ArrayPlacement, Source = ArrayPlacement, Donate = bool, MayAlias = Option<bool>> {
    /// Destination placement tree prefix. When absent, host leaves are uploaded to the default local device without
    /// being [committed](Array::is_committed) to it, and [`Array`] leaves and traced leaves preserve their current
    /// placement and commitment.
    target: Option<Target>,

    /// Source placement tree prefix. This is validated for [`Array`] leaves and ignored for host leaves, which do not
    /// carry runtime placement metadata before upload.
    source: Option<Source>,

    /// Donation tree prefix. A donated [`Array`] leaf lets a compiled reshard reuse its buffers for the result. The
    /// donation only takes effect when the leaf is the only handle of its buffers. Only concrete [`Array`] leaves
    /// apply it, and host leaves and traced leaves ignore it. A leaf cannot be donated while its aliasing option is
    /// `Some(true)`.
    donate: Option<Donate>,

    /// Aliasing tree prefix. With `false` or `Some(false)`, a resulting array shares no buffer with any other array,
    /// which copies the buffers that it would otherwise share. Concrete [`Array`] leaves apply it when they are placed,
    /// and traced leaves stage their placements with a forced copy, which applies it when those placements execute on
    /// concrete arrays (e.g., when a batched or differentiated function places concrete arrays). Host leaves ignore
    /// it, because their uploads never share buffers. Refer to [`Session::place`] for more information.
    may_alias: Option<MayAlias>,
}

impl<Target, Source, Donate, MayAlias> PlacementOptions<Target, Source, Donate, MayAlias> {
    /// Creates a new [`PlacementOptions`].
    #[inline]
    pub fn new(
        target: Option<Target>,
        source: Option<Source>,
        donate: Option<Donate>,
        may_alias: Option<MayAlias>,
    ) -> Self {
        Self { target, source, donate, may_alias }
    }

    /// Creates a new [`PlacementOptions`] with all fields unset.
    #[inline]
    pub fn empty() -> Self {
        Self::new(None, None, None, None)
    }

    /// Returns these options with the destination tree prefix replaced by `target`, which can have another type than
    /// the current one. The type of each other option is kept, so this extends typed options such as
    /// [`Self::defaults`] without naming the types of the unrelated options (e.g.,
    /// `PlacementOptions::defaults().with_target(device)`).
    #[inline]
    pub fn with_target<NewTarget>(self, target: NewTarget) -> PlacementOptions<NewTarget, Source, Donate, MayAlias> {
        PlacementOptions::new(Some(target), self.source, self.donate, self.may_alias)
    }

    /// Returns these options with the source tree prefix replaced by `source`, which can have another type than the
    /// current one. The type of each other option is kept, as in [`Self::with_target`].
    #[inline]
    pub fn with_source<NewSource>(self, source: NewSource) -> PlacementOptions<Target, NewSource, Donate, MayAlias> {
        PlacementOptions::new(self.target, Some(source), self.donate, self.may_alias)
    }

    /// Returns these options with the donation tree prefix replaced by `donate`, which can have another type than the
    /// current one. The type of each other option is kept, as in [`Self::with_target`] (e.g.,
    /// `PlacementOptions::to(device).with_donate(true)`).
    #[inline]
    pub fn with_donate<NewDonate>(self, donate: NewDonate) -> PlacementOptions<Target, Source, NewDonate, MayAlias> {
        PlacementOptions::new(self.target, self.source, Some(donate), self.may_alias)
    }

    /// Returns these options with the aliasing tree prefix replaced by `may_alias`, which can have another type than
    /// the current one. The type of each other option is kept, as in [`Self::with_target`], so that, for example,
    /// `PlacementOptions::defaults().with_may_alias(false)` requests a forced copy without a destination and without
    /// naming the type of a destination. The prefix can hold `bool` or `Option<bool>` leaves (refer to
    /// [`AliasingParameter`]).
    #[inline]
    pub fn with_may_alias<NewMayAlias>(
        self,
        may_alias: NewMayAlias,
    ) -> PlacementOptions<Target, Source, Donate, NewMayAlias> {
        PlacementOptions::new(self.target, self.source, self.donate, Some(may_alias))
    }

    /// Returns the destination placement tree prefix, if one was provided.
    #[inline]
    pub fn target(&self) -> Option<&Target> {
        self.target.as_ref()
    }

    /// Returns the source placement tree prefix, if one was provided.
    #[inline]
    pub fn source(&self) -> Option<&Source> {
        self.source.as_ref()
    }

    /// Returns the donation tree prefix, if one was provided.
    #[inline]
    pub fn donate(&self) -> Option<&Donate> {
        self.donate.as_ref()
    }

    /// Returns the aliasing tree prefix, if one was provided.
    #[inline]
    pub fn may_alias(&self) -> Option<&MayAlias> {
        self.may_alias.as_ref()
    }

    /// Returns the destination, source, donation, and aliasing tree prefixes of these options.
    fn into_parts(self) -> (Option<Target>, Option<Source>, Option<Donate>, Option<MayAlias>) {
        (self.target, self.source, self.donate, self.may_alias)
    }
}

impl<Target, Source, Donate, MayAlias> Default for PlacementOptions<Target, Source, Donate, MayAlias> {
    fn default() -> Self {
        Self::empty()
    }
}

impl PlacementOptions<ArrayPlacement, ArrayPlacement, bool, Option<bool>> {
    /// Creates default [`Session::place`] options without requiring generic type inference.
    #[inline]
    pub fn defaults() -> Self {
        Self::empty()
    }
}

impl<Target> PlacementOptions<Target, ArrayPlacement, bool, Option<bool>> {
    /// Creates [`Session::place`] options with only the destination tree prefix `target` (e.g., a [`Device`], a
    /// [`Memory`], an [`ArrayPlacement`], or a tuple of optional placements). The other options are unset, and their
    /// types are fixed as in [`Self::defaults`], so that this needs no generic type annotations.
    #[inline]
    pub fn to(target: Target) -> Self {
        Self::new(Some(target), None, None, None)
    }
}

/// Leaf types accepted by [`Session::place`].
///
/// A [`Placeable`] consumes one input leaf and produces one [`Self::Output`] leaf. `ryft` provides implementations for
/// runtime [`Array`] leaves, host scalars of the [`ArrayElement`](ryft_core::ArrayElement) types that are
/// [`Parameter`]s, and host arrays of the reference backend ([`ryft_core::Array`]), which are uploaded as dense arrays
/// whatever their host layout is, and all of which produce runtime [`Array`] leaves. [`HostOrArray`] holds either kind
/// of these leaves, including scalars of the other element types (e.g., complex scalars), so that one tree can mix
/// them. `ryft` also provides implementations for the traced values of `ryft-core` that can hold arrays of this backend
/// (i.e., [`Tracer`], [`PartialTracer`], [`DifferentiationTracer`], [`BatchingTracer`], and [`ProjectedValue`] over
/// [`ArrayType`], which covers [`CompileTracer`](crate::CompileTracer)), so that the same call works inside compiled
/// functions, differentiation, and batching. A traced leaf stays traced and records its placement through the
/// [`PlaceDispatch`] primitive of the [`Place`] capability (i.e., by staging a
/// [`PlaceOperation`] whose copy policy follows the `may_alias` option of the leaf), and
/// without a destination, that operation has no target and leaves the leaf where it is. Those implementations require
/// the values that the traced values wrap to be ones whose concrete arrays this crate can find (i.e., [`Array`]s,
/// [`Value`]s, host arrays of the reference backend, and the traced values above), so that a traced leaf over arrays
/// of another PJRT client or of another session can be rejected.
///
/// [`Placeable::place`] is the per-leaf primitive of [`Session::place`], like
/// [`ReshardDispatch::reshard`](ryft_core::ReshardDispatch::reshard) is for resharding. It takes the leaf as an
/// ordinary first argument instead of a receiver, so that it does not shadow the [`Place::place`] function of the
/// traced values that also implement [`Place`]. Call it as `P::place(leaf, session, ...)`.
pub trait Placeable: Parameter {
    /// Leaf type that placing this leaf produces, which is [`Array`] for host values and runtime arrays, and the leaf
    /// type itself for traced values.
    type Output: Parameter;

    /// Places `value` using the provided leafwise placement options. [`Session::place`] checks the donation and
    /// aliasing options of every leaf before it places any leaf, so implementations can assume that `donate` is
    /// `false` whenever `may_alias` is `Some(true)`. It also checks that the sharding of `target` matches the mesh
    /// that it resolves against, but every other check of the options (e.g., of `source`, or of the rank of the
    /// sharding of `target`) is made here, because it depends on the leaf.
    ///
    /// # Parameters
    ///
    ///   - `value`: Leaf to place.
    ///   - `session`: [`Session`] that provides the PJRT client used to materialize any needed destination buffers and
    ///     the cache of any compiled reshard executables.
    ///   - `target`: Destination placement for this leaf, if one was specified.
    ///   - `source`: Source placement for this leaf, if one was specified.
    ///   - `donate`: Whether a compiled reshard may reuse the buffers of this leaf for the result.
    ///   - `may_alias`: Whether the result may share buffers with other arrays. `Some(false)` forbids it.
    ///
    /// # Errors
    ///
    /// Implementations return an [`Error`] instead of a placed leaf, and never a partially placed one, when the leaf
    /// cannot be placed by `session` as requested. This covers a leaf whose arrays belong to another PJRT client than
    /// `session`, a traced leaf whose arrays belong to another session (refer to [`Session::place`] for the rules of
    /// concrete and traced leaves), a `target` or `source` that cannot be resolved against `session` (e.g.,
    /// [`SessionError::MissingMesh`](crate::SessionError::MissingMesh) for a sharding without a default mesh), a
    /// `target` or `source` that the type rule of [`PlaceOperation`] rejects for a concrete leaf, except for an uneven
    /// split (i.e., a [`TypeError`] of [`PlaceOperation::infer_output_type_allowing_uneven_splits`]), and a placement that fails to execute or
    /// to stage (e.g., a failed transfer, or a target that the rules of [`PlaceOperation`] reject for a traced leaf).
    /// The leaf is consumed either way, and [`Session::place`] returns the first error of any leaf as its own, after
    /// the leaves before it were placed.
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self::Output, Error>;
}

impl<T: IntoHostArray + Parameter> Placeable for T {
    type Output = Array;

    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        // A host value has no placement to validate and no device buffers to donate or share.
        let _ = (source, donate, may_alias);
        let client = session.client();
        // A value without a destination lives on the first device only because nothing chose any, so it is not
        // committed to that device, and operations move it to the devices of their committed inputs.
        let host = value.into_host_array();
        // A host array holds materialized storage, which only a type with static extents can describe.
        let shape = host.r#type().static_shape().unwrap().as_slice().to_vec();
        let element_type = host.r#type().data_type();

        // The upload of a host value onto a destination is the placement of a value that has no tracked sharding and
        // lives in device memory without a declared layout, which is how a host value is uploaded without a
        // destination, so the type rule of `place` validates the destination against that type before anything is
        // uploaded, except that the destination may split a dimension unevenly (refer to `Session::place`).
        if let Some(target) = &target {
            let request = target.clone().into_request(session, shape.len())?;
            let r#type = ArrayType::new_static(element_type, shape.clone());
            PlaceOperation::infer_output_type_allowing_uneven_splits(&r#type, None, Some(&request), None)?;
        }
        // The upload takes dense row-major bytes, which is how a host array without a layout is stored, so only an
        // array with a layout is decoded into a copy. Its storage can be mutated safely, so the decoding and the upload
        // validate it rather than trust it.
        let bytes = match host.r#type().layout() {
            Some(_) => Cow::Owned(decode_logical_bytes(host.r#type().as_ref(), host.storage_bytes())?),
            None => Cow::Borrowed(host.storage_bytes()),
        };
        // An in-place destination names no devices, so it uploads the value to the same device as an upload without
        // a destination, in the storage that it names, and it commits the result like every other destination, as JAX
        // commits the result of `device_put` onto a memory space.
        // A value without a destination, or with one that names only its storage, lives on a device that this process
        // chose on its own (refer to `Array::is_locally_assigned`).
        let committed = target.is_some();
        let locally_assigned = target.as_ref().is_none_or(|target| target.target == ArrayPlacementTarget::InPlace);
        let default_device = || -> Result<Device, Error> {
            let device = client
                .addressable_devices()?
                .into_iter()
                .next()
                .ok_or_else(|| Error::from(ArrayError::MissingDefaultDevice))?;
            Ok(Device::from_pjrt(device)?)
        };
        let placement = match target {
            Some(destination) if destination.target == ArrayPlacementTarget::InPlace => {
                let target = ArrayPlacementTarget::Device(default_device()?);
                ArrayPlacement { target, ..destination }.resolve(session, shape.len())?
            }
            Some(target) => target.resolve(session, shape.len())?,
            None => ArrayPlacement::device(default_device()?).resolve(session, shape.len())?,
        };
        let r#type = ArrayType::new(element_type, Shape::new(shape.iter().copied().map(Dimension::Static).collect()))
            .with_memory(placement.memory().unwrap_or_default())
            .with_layout(placement.layout().cloned());
        let (mesh, sharding) = (placement.mesh().clone(), placement.sharding().clone());
        Ok(Array::from_host_buffer(session, r#type, mesh, Some(sharding), bytes)?
            .with_committed(committed)
            .with_locally_assigned(locally_assigned))
    }
}

impl Placeable for Array {
    type Output = Array;

    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        // The result belongs to `session` even when it is this leaf or shares its buffers, so associate the leaf with
        // `session` first, which also rejects an array of another PJRT client.
        let value = value.with_session(session)?;
        validate_array_request(session, &value, target.as_ref(), source)?;

        // A forced copy must not return a buffer of this leaf, even one that only the result holds once the placement
        // consumes the leaf. Weak handles of those buffers keep a donation of the leaf possible.
        let buffers: Vec<_> = match may_alias {
            Some(false) => value.addressable_shards().filter_map(ArrayShard::buffer).map(Arc::downgrade).collect(),
            _ => Vec::new(),
        };
        // The destination resolves against the array in the one place that also keeps the locally assigned devices of an
        // array that an in-place destination leaves where it is.
        let output = match target {
            Some(target) => value.run_placement_request(session, target, donate)?,
            None => value,
        };

        // The result is this leaf when the placement does not change, and it otherwise shares the buffers of the
        // shards that stay on their devices. Holding the buffers of this leaf that are still alive makes them shared,
        // so that `into_unaliased` copies exactly those, and not the buffers that the placement created.
        if may_alias == Some(false) {
            let _buffers = buffers.iter().filter_map(Weak::upgrade).collect::<Vec<_>>();
            output.into_unaliased(session)
        } else {
            Ok(output)
        }
    }
}

impl<C: Context<Type = ArrayType>> Placeable for Tracer<C>
where
    Tracer<C>: Placed + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, Tracer<C>>>,
{
    type Output = Self;

    #[inline]
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self, Error> {
        place_traced(value, session, target, source, donate, may_alias)
    }
}

impl<C: Context<Type = ArrayType>> Placeable for PartialTracer<C>
where
    PartialTracer<C>:
        Placed + ConcreteArrays + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, PartialTracer<C>>>,
{
    type Output = Self;

    #[inline]
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self, Error> {
        place_traced(value, session, target, source, donate, may_alias)
    }
}

impl<C: Context<Type = ArrayType>, P: DifferentiationPolicy<C>> Placeable for DifferentiationTracer<C, P>
where
    DifferentiationTracer<C, P>: Placed
        + ConcreteArrays
        + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, DifferentiationTracer<C, P>>>,
{
    type Output = Self;

    #[inline]
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self, Error> {
        place_traced(value, session, target, source, donate, may_alias)
    }
}

impl<C: Context<Type = ArrayType>, P: BatchingPolicy<C>> Placeable for BatchingTracer<C, P>
where
    BatchingTracer<C, P>:
        Placed + ConcreteArrays + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, BatchingTracer<C, P>>>,
{
    type Output = Self;

    #[inline]
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self, Error> {
        place_traced(value, session, target, source, donate, may_alias)
    }
}

impl<V> Placeable for ProjectedValue<ArrayType, V>
where
    ProjectedValue<ArrayType, V>: Placed
        + ConcreteArrays
        + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, ProjectedValue<ArrayType, V>>>,
{
    type Output = Self;

    #[inline]
    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Self, Error> {
        place_traced(value, session, target, source, donate, may_alias)
    }
}

/// Leaf of [`Session::place`] that is either a host value or a runtime [`Array`]. A [`Parameterized`] tree has one leaf
/// type, so a tuple of a host scalar and an [`Array`] is not a tree that [`Session::place`] accepts, whereas a tuple
/// of their [`HostOrArray`] conversions is. Every host value that [`Session::place`] can upload (i.e., host arrays of
/// the reference backend and scalars of every [`ArrayElement`](ryft_core::ArrayElement) type, including the complex
/// and low precision types that are not leaves on their own) and [`Array`]s convert into it with [`From`]. Placing a
/// [`HostOrArray`] places its value as [`Session::place`] places a leaf of that kind, so every option applies to it as
/// it applies to such a leaf, and the result is an [`Array`] in both cases.
///
/// # Example
///
/// ```
/// # use ryft_core::{ArrayType, DataType};
/// # use ryft_xla::{Array, ArrayPlacement, HostOrArray, PlacementOptions, Platform, Session};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let session = Session::builder().platform(Platform::Cpu).build()?;
/// let r#type = ArrayType::new_static(DataType::F32, [2]);
/// let array = Array::from_host_buffer(&session, r#type, None, None, [1f32, 2.].map(f32::to_ne_bytes).concat())?;
/// let device = array.mesh().devices()[0];
///
/// // The destination applies to both leaves, and so would every other option.
/// let options = PlacementOptions::to(ArrayPlacement::device(device));
/// let (scalar, array) = session.place((HostOrArray::from(2.5f32), HostOrArray::from(array)), options)?;
/// assert_eq!(scalar.mesh().devices(), [device]);
/// assert_eq!(scalar.copy_to_host()?.elements::<f32>()?, vec![2.5]);
/// assert_eq!(array.copy_to_host()?.elements::<f32>()?, vec![1., 2.]);
/// # Ok(())
/// # }
/// ```
#[derive(Clone, Debug, Parameter)]
pub enum HostOrArray {
    /// Host value, which is uploaded as a dense array (e.g., a scalar, which is a rank-0 array).
    Host(CoreArray),

    /// Runtime array.
    Array(Array),
}

impl<T: IntoHostArray> From<T> for HostOrArray {
    #[inline]
    fn from(value: T) -> Self {
        Self::Host(value.into_host_array())
    }
}

impl From<Array> for HostOrArray {
    #[inline]
    fn from(value: Array) -> Self {
        Self::Array(value)
    }
}

impl Placeable for HostOrArray {
    type Output = Array;

    fn place(
        value: Self,
        session: &Session,
        target: Option<ArrayPlacement>,
        source: Option<ArrayPlacement>,
        donate: bool,
        may_alias: Option<bool>,
    ) -> Result<Array, Error> {
        match value {
            Self::Host(value) => <CoreArray as Placeable>::place(value, session, target, source, donate, may_alias),
            Self::Array(value) => <Array as Placeable>::place(value, session, target, source, donate, may_alias),
        }
    }
}

impl Array {
    /// Returns an array with the type, the placement, and the commitment of this array whose addressable shard buffers
    /// no other handle holds. This is how a placement whose result must not alias its input (i.e., a placement with
    /// `may_alias` set to `false`) gives that result buffers of its own. Each buffer that another handle also holds
    /// (e.g., the input of a placement that keeps a shard on its device, which the caller still holds) is replaced with
    /// a copy on the same device and in the same memory, and each buffer that only this array holds (e.g., one that the
    /// placement created) is kept. A clone of this array holds all of its buffers, so it makes this function copy every
    /// buffer. A caller that consumed the only handle of the input must therefore hold the buffers of that input while
    /// it calls this function, or the result could return them. The copies are made by
    /// [`Self::with_copied_shard_buffers`], so the execution fence of the result joins the readiness of every copy with
    /// the fence of this array, because a readback reads only one replica of each tile and would otherwise succeed with
    /// a replica whose copy did not fail.
    pub(crate) fn into_unaliased(self, session: &Session) -> Result<Self, Error> {
        // Clones of this array share its parts, and with them every one of its buffers.
        let has_clones = Arc::strong_count(&self.parts) != 1;
        let is_shared = |buffer: &Arc<Buffer<'static>>| has_clones || Arc::strong_count(buffer) != 1;
        if !self.addressable_shards().filter_map(ArrayShard::buffer).any(is_shared) {
            return Ok(self);
        }
        self.with_copied_shard_buffers(session, is_shared)
    }
}

impl Session {
    /// Places `x` on devices. This is the `ryft` analogue of JAX's
    /// [`jax.device_put`](https://docs.jax.dev/en/latest/_autosummary/jax.device_put.html).
    ///
    /// The input `x` may be one supported leaf or a [`Parameterized`] tree of supported leaves (i.e., of
    /// [`Placeable`]s). Any provided `target`, `source`, `donate`, and `may_alias` options follow tree prefix
    /// broadcasting semantics over `x`. The destination and source prefixes accept [`Placement`] or [`ArrayPlacement`]
    /// leaves. Either can be optional, so a tree can mix explicit placements with `None` to omit an option on a branch.
    /// A tree has one leaf type, so a tree that mixes host values and [`Array`]s holds them as [`HostOrArray`]s. A
    /// prefix leaf may also cover only subtrees without leaves, in which case it places nothing (e.g., one of two
    /// destinations for an empty [`Vec`] and a scalar, or a single destination for an empty [`Vec`]). Each option tree
    /// is validated against the nodes of `x` (refer to [`Parameterized::paths`] and
    /// [`Parameterized::broadcast_to_parameter_structure`]), so every node of an option tree must be a node of `x`,
    /// and every node of `x` must be a node of the option tree or lie at or below one of its leaves (e.g., the
    /// destinations `[d]` for an input `[]`, `[[d]]` for `[[x], []]`, `[[d], []]` for `[[x]]`, and `[[]]` for `[]` are
    /// all rejected). Nodes are matched by their paths only, which is intended, so an option tree of another type with
    /// the same paths (e.g., a [`Vec`] for an array, or another derived struct with the same parameter fields)
    /// matches, and so does a node whose static metadata differs from that of `x`. An input without leaves is returned
    /// as it is once its options are validated.
    ///
    /// With no destination, a host leaf goes to the first addressable device and an existing [`Array`] leaf keeps
    /// its current placement. Use [`ArrayPlacement::sharding`] to place a leaf on the session's default mesh. As in
    /// JAX, every [`Array`] that is placed with a destination is [committed](Array::is_committed) to its devices, an
    /// [`Array`] leaf without a destination keeps its commitment, and a host leaf without a destination is uncommitted,
    /// so operations with committed inputs on other devices move it to those devices. Every resulting [`Array`] belongs
    /// to this session, including an [`Array`] leaf that keeps its placement, so an [`Array`] leaf of another PJRT
    /// client is rejected. A traced leaf executes its placement with the session of the arrays that it wraps (e.g.,
    /// the input of a function that is batched or differentiated at such arrays, including every primal, tangent,
    /// and packed batch), which cannot be replaced by this session. A traced leaf whose arrays belong to another PJRT
    /// client is therefore rejected in the same way as such an [`Array`] leaf, and one whose arrays belong to another
    /// session of the same client is rejected with a [`ryft_pjrt::Error::InvalidArgument`] error. Other handles of
    /// this session (e.g., one with another default mesh) are accepted, and the results of a traced leaf stay attached
    /// to the handles of the arrays that it wraps.
    ///
    /// An [`ArrayPlacementTarget::Device`] destination leaves every leaf wholly on that device, as its documentation
    /// describes for concrete and traced leaves alike. An [`Array`] leaf on a mesh of one device keeps its logical mesh
    /// and its sharding over the device, without its reduction state, which one device completes, and an [`Array`]
    /// leaf on more devices is gathered onto the device, which rejects a leaf that is unreduced over a mesh axis larger
    /// than one, because a placement cannot combine the contributions of several devices. Without a memory, such a
    /// destination also selects the default memory of its device for every leaf (e.g., it moves an
    /// array out of pinned host memory), as JAX's eager `device_put` does for a device and as
    /// [`PlacementRequest::target_memory`] states for every path (traced leaves and compiled functions included),
    /// whereas a mesh or sharding destination without a memory keeps the memory of the leaf. Without a layout, a
    /// destination keeps the layout of a leaf only while it still describes each shard of the result, and the result
    /// declares no layout otherwise (refer to [`PlacementRequest`]).
    ///
    /// An [`ArrayPlacementTarget::InPlace`] destination (e.g., a bare [`Memory`]) leaves every leaf on its devices
    /// with its sharding and changes only its memory and layout, as its documentation describes for concrete, traced,
    /// and host leaves. This is how a traced leaf whose devices are not known while it is traced (e.g., the input of a
    /// function that is compiled for a mesh without knowing how that input is split along auto mesh axes) moves to
    /// another memory. Its results are committed like those of every other destination, as JAX commits the result of
    /// `device_put` onto a memory space, eagerly and inside `jit`.
    ///
    /// Inside a batched function, the placement of a batch item (refer to [`Placed`]) records how its batch is
    /// distributed. A destination or source that is such a [`Placement`] keeps that record (refer to
    /// [`ArrayPlacement::batch_shardings`]), so that placing a batch item on its own placement leaves every batch item
    /// on the devices that hold it.
    ///
    /// Traced leaves (e.g., the inputs of a function that is compiled, differentiated, or batched) stay traced, and
    /// their placements resolve against this session and are recorded through the [`PlaceDispatch`] primitive of the
    /// [`Place`] capability, so the same call works inside those transformations. Every leaf follows the type rule of
    /// [`PlaceOperation`], an [`Array`] leaf and a host leaf included, which validate their requests as they are given
    /// with that rule (refer to [`PlaceOperation::infer_output_type_allowing_uneven_splits`]) before they move or upload anything. A host leaf
    /// is validated as the placement of a value without a tracked sharding in device memory, which is how it is
    /// uploaded without a destination, and it ignores its source. So no placement changes the reduction state of a
    /// value or moves a value inside a manual region or into one, and every leaf reports the error that the type rule
    /// reports first. A destination must restate the unreduced and
    /// reduced axes of a leaf over a mesh that defines them identically, so a placement never combines the values of
    /// several devices along unreduced axes, and a host leaf, which has no reduction state, cannot be uploaded onto an
    /// unreduced or reduced destination. The one exception to the restatement is a device destination, which completes
    /// reduction state over mesh axes of size one and accepts or rejects reduction state over larger axes as the
    /// contract of [`PlacementRequest::Device`] describes. [`Array::to_placement`] is the physical primitive below this
    /// function, which combines contributions and drops reduction state when its destination requires it. JAX rejects a
    /// `device_put` that makes a concrete value unreduced too, whereas it accepts one that makes such a value reduced,
    /// which Ryft rejects, and inside `jit` it lowers a change of reduction state to a sharding constraint that
    /// completes the reduction, which Ryft rejects as well. The only check of the type rule that the placement of an
    /// [`Array`] or host leaf does not make is that a destination splits each dimension evenly (refer to
    /// [`Array::to_placement`]), whereas traced leaves follow it.
    ///
    /// A traced leaf without a destination stages a placement without a target, which leaves the value where it is
    /// (i.e., its devices, sharding, memory, and layout are unchanged), as JAX's `device_put` does without a device.
    /// That placement records the provided source. Without one, it records where the value concretely lives wherever
    /// it executes on a value that lives somewhere (e.g., when a differentiated function is evaluated on concrete
    /// arrays, refer to [`Placed`]). Reverse mode differentiation moves the cotangent back to that source. Such a
    /// placement is staged for every traced leaf, including one without a concrete placement and without a source, so
    /// that its copy policy applies wherever the staged program executes on concrete arrays.
    ///
    /// A placement inside a compiled function must target the device mesh that the function is compiled for (refer to
    /// [`LoweringError::PlacementOutsideProgramMesh`](crate::LoweringError::PlacementOutsideProgramMesh)), so a
    /// compiled function cannot move a value to other devices. The devices of a traced leaf that wraps no concrete
    /// array are not known while it is traced, so an [`ArrayPlacementTarget::Device`] destination stages a
    /// [`PlacementRequest::Device`] with its memory and layout, whose type rule selects the default memory of the
    /// device when it names none, as it does for a traced leaf that lives on a mesh of one device. Its type keeps the
    /// sharding of the leaf without the reduction state that one device completes, so it combines with the other values
    /// of the function. A compiled function accepts it only when the device is the only device of the mesh that the
    /// function is compiled for (refer to
    /// [`LoweringError::DeviceOutsideProgramMesh`](crate::LoweringError::DeviceOutsideProgramMesh)), and the outputs of
    /// such a function are committed, as an explicit device commits an [`Array`] eagerly, whereas JAX leaves the
    /// outputs of a `jit` function whose only placement request is a bare device uncommitted. Wherever the staged
    /// placement executes on a concrete array instead (e.g., when a traced program is evaluated on arrays), the device
    /// resolves against that array as it does for an [`Array`] leaf. An [`ArrayPlacementTarget::Device`] source of a
    /// traced leaf whose devices are not known stages a [`PlacementRequest::Device`] source, so that reverse mode
    /// differentiation moves the cotangent back to that device, and transposing the result again moves it to the
    /// destination once more, as JAX's transposition of `device_put` does. Transposition turns a source into the
    /// destination of the cotangent, so inside a compiled function a source that names devices follows the rule of a
    /// destination (i.e., a device source must be the only device of the mesh that the function is compiled for, and a
    /// mesh or sharding source must be over the devices of that mesh, in its order and with its topology). Inside a
    /// manual region (e.g., a `shard_map` body), where no value can be moved, a source must not name devices at all, and
    /// only a source that names a memory or a layout is accepted there, as is a destination that names only those. A
    /// function whose reverse mode could not be compiled is therefore rejected when it is compiled, with
    /// [`LoweringError::UnsupportedOperation`](crate::LoweringError::UnsupportedOperation), whereas JAX elides the
    /// devices of a `device_put` source inside `jit` and accepts it.
    ///
    /// The donation option is only applied to concrete [`Array`] leaves, and it is ignored for host leaves and traced
    /// leaves. The aliasing option is applied to concrete [`Array`] leaves when they are placed, and its leaves can be
    /// `bool`s or `Option<bool>`s (refer to [`AliasingParameter`]), where `false` means `Some(false)` and `true` means
    /// `Some(true)`. With `Some(false)`, the result holds no buffer of the leaf, even when the leaf was the only handle
    /// of its buffers, and the buffers that the placement creates are not copied again. This also holds for a donated
    /// leaf that already lives where it is placed, which gets storage of its own because the explicit request not to
    /// alias wins, whereas JAX applies the donation and returns the storage of the input. A traced leaf whose
    /// `may_alias` option is `Some(false)` stages its placement with a forced copy (refer to
    /// [`PlaceOperation::with_may_alias`]), so that when a batched or
    /// differentiated function is evaluated on concrete arrays, its result shares no buffer with any other array, as
    /// in JAX. A compiled function has no observable buffer identity, so the option has no effect inside one. Host
    /// leaves ignore the option, because their uploads never share buffers. A leaf cannot be donated while its
    /// `may_alias` option is `Some(true)`, which this function reports with [`ArrayError::DonatedArrayMayAlias`] before
    /// it places any leaf, as JAX does.
    ///
    /// The source option states where a leaf is expected to live. Its devices are informative on every path, as in
    /// JAX, so they are never compared with the devices that a leaf actually lives on, and reverse mode
    /// differentiation sends the cotangent to the devices that it names. Its sharding, memory, and layout are checked
    /// by the type rule of [`PlaceOperation`] against the type of an [`Array`] leaf and its runtime sharding, which
    /// records its reduction state and its shards along auto mesh axes that its type omits, exactly as they are for a
    /// traced leaf that wraps a concrete array (refer to [`PlaceOperation::infer_output_type_allowing_uneven_splits`]), as it is given. A
    /// traced leaf that wraps a concrete array then completes a source that names only a memory or a layout with the
    /// concrete placement of that array (refer to [`PlacementRequest::resolve_source`]), so that transposition returns
    /// the cotangent to its devices. An expected layout must therefore address the shape of each shard of the leaf as
    /// far as its runtime sharding (or, for a staged leaf, its type) determines its shards, and it must equal the
    /// layout that the type of the leaf declares, if any. The type rule is the only check of a source, so the physical
    /// layouts of the buffers of a leaf whose type declares no layout are not inspected on any path. A target that
    /// restates a pending or completed reduction of an [`Array`] leaf over an auto axis (e.g., its own placement) is
    /// accepted and changes nothing, and one that drops it is rejected, as for any other reduction state.
    ///
    /// In a run with more than one process, every process must make the same call with trees of the same structure
    /// and the same options, because the placements of the leaves communicate across processes in the order of the
    /// leaves. An error that one process reports before it places a leaf must also be one that every process reports,
    /// so that no process posts a transfer that another one never matches. The checks that run before any leaf is
    /// placed (refer to the errors below) and the checks of a source meet this requirement when every process also uses
    /// a session with the same default mesh, because they depend only on the call, on that mesh, on the devices of the
    /// client, and on the types of the leaves.
    ///
    /// # Parameters
    ///
    ///   - `x`: Leaf or tree of leaves to place.
    ///   - `options`: Destination, expected source, donation, and aliasing options for the leaves of `x`.
    ///
    /// # Errors
    ///
    /// Before it places any leaf, this function checks that every option prefix broadcasts over `x`, that no leaf is
    /// donated while it may alias, and that the sharding of every destination matches its mesh (i.e., the mesh of an
    /// [`ArrayPlacementTarget::Mesh`], or the default mesh of this session for an [`ArrayPlacementTarget::Sharding`],
    /// which reports [`SessionError::MissingMesh`](crate::SessionError::MissingMesh) when the session has none). It
    /// also rejects an [`ArrayPlacementTarget::InPlace`] destination or source that names neither a memory nor a
    /// layout, with a [`TypeError`] whose message is that of the type rule of [`PlaceOperation`] for such a target or
    /// source, whereas one that names a memory or a layout is accepted on every path. Any other error belongs to one
    /// leaf (refer to [`Placeable::place`]), and it is returned when that leaf is placed, after the leaves before it
    /// were placed. Their results are dropped, but the transfers and donations that placing them started are not
    /// undone.
    pub fn place<
        P: Placeable,
        Input: Parameterized<P>,
        TargetPlacement: PlaceableParameter,
        Target: Parameterized<TargetPlacement>,
        SourcePlacement: PlaceableParameter,
        Source: Parameterized<SourcePlacement>,
        Donate: Parameterized<bool>,
        Aliasing: AliasingParameter,
        MayAlias: Parameterized<Aliasing>,
    >(
        &self,
        x: Input,
        options: PlacementOptions<Target, Source, Donate, MayAlias>,
    ) -> Result<<Input as Parameterized<P>>::To<P::Output>, Error>
    where
        <Input as Parameterized<P>>::Family: ParameterizedFamily<P::Output>
            + ParameterizedFamily<Option<ArrayPlacement>>
            + ParameterizedFamily<bool>
            + ParameterizedFamily<Option<bool>>,
        <Target as Parameterized<TargetPlacement>>::Family: ParameterizedFamily<Option<ArrayPlacement>>,
        <Source as Parameterized<SourcePlacement>>::Family: ParameterizedFamily<Option<ArrayPlacement>>,
        <MayAlias as Parameterized<Aliasing>>::Family: ParameterizedFamily<Option<bool>>,
    {
        let structure = x.parameter_structure();
        let parameter_count = structure.parameter_count();
        let (target, source, donate, may_alias) = options.into_parts();

        // An option tree that is one leaf at its root covers every node of the input by construction, so its value is
        // repeated for every leaf of the input without building and matching the paths of the input, which would
        // otherwise dominate the cost of placing a large tree. Every other option tree is broadcast over the input.
        let target_values = match target {
            Some(target) if is_root_parameter(&target) => {
                vec![target.into_parameters().next().unwrap().into_optional_placement(); parameter_count]
            }
            Some(target) => target
                .map_parameters(|placement| placement.into_optional_placement())?
                .broadcast_to_parameter_structure::<Input::To<Option<ArrayPlacement>>>(structure.clone())?
                .into_parameters()
                .collect::<Vec<_>>(),
            None => vec![None; parameter_count],
        };
        let source_values = match source {
            Some(source) if is_root_parameter(&source) => {
                vec![source.into_parameters().next().unwrap().into_optional_placement(); parameter_count]
            }
            Some(source) => source
                .map_parameters(|placement| placement.into_optional_placement())?
                .broadcast_to_parameter_structure::<Input::To<Option<ArrayPlacement>>>(structure.clone())?
                .into_parameters()
                .collect::<Vec<_>>(),
            None => vec![None; parameter_count],
        };
        let donate_values = match donate {
            Some(donate) if is_root_parameter(&donate) => {
                vec![donate.into_parameters().next().unwrap(); parameter_count]
            }
            Some(donate) => donate
                .broadcast_to_parameter_structure::<Input::To<bool>>(structure.clone())?
                .into_parameters()
                .collect::<Vec<_>>(),
            None => vec![false; parameter_count],
        };
        let may_alias_values = match may_alias {
            Some(may_alias) if is_root_parameter(&may_alias) => {
                vec![may_alias.into_parameters().next().unwrap().into_may_alias(); parameter_count]
            }
            Some(may_alias) => may_alias
                .map_parameters(|may_alias| may_alias.into_may_alias())?
                .broadcast_to_parameter_structure::<Input::To<Option<bool>>>(structure.clone())?
                .into_parameters()
                .collect::<Vec<_>>(),
            None => vec![None; parameter_count],
        };

        // Validate what the options of every leaf can tell without the leaf before placing any leaf, so that such an
        // invalid request has no effects. Every kind of leaf resolves its destination against the mesh that the
        // destination names, or against the default mesh of this session for a bare sharding, so those meshes are
        // checked here, and so are an in-place destination and an in-place source that name no storage, which no leaf
        // accepts. Every other source is checked when its leaf is placed, because host leaves ignore sources. Sources
        // are checked first, as the type rule and traced values check them.
        if let Some(index) = donate_values
            .iter()
            .zip(&may_alias_values)
            .position(|(&donate, &may_alias)| donate && may_alias == Some(true))
        {
            let path = x.parameter_paths().nth(index).unwrap();
            return Err(ArrayError::DonatedArrayMayAlias { path }.into());
        }
        for source in source_values.iter().flatten() {
            source.validate_source()?;
        }
        for target in target_values.iter().flatten() {
            match target.target() {
                ArrayPlacementTarget::Device(_) => {}
                ArrayPlacementTarget::Mesh { mesh, sharding } => check_sharding!(mesh, sharding),
                ArrayPlacementTarget::Sharding(sharding) => check_sharding!(self.device_mesh()?, sharding),
                ArrayPlacementTarget::InPlace => target.validate_destination()?,
            }
        }

        let mut output_parameters = Vec::with_capacity(parameter_count);
        let mut target_values = target_values.into_iter();
        let mut source_values = source_values.into_iter();
        let mut donate_values = donate_values.into_iter();
        let mut may_alias_values = may_alias_values.into_iter();
        for parameter in x.into_parameters() {
            output_parameters.push(P::place(
                parameter,
                self,
                target_values.next().unwrap(),
                source_values.next().unwrap(),
                donate_values.next().unwrap(),
                may_alias_values.next().unwrap(),
            )?);
        }
        Input::To::<P::Output>::from_parameters(structure, output_parameters).map_err(Into::into)
    }
}

/// Host value that [`Session::place`] uploads as one dense array (i.e., a host array of the reference backend, or a
/// scalar of any [`ArrayElement`](ryft_core::ArrayElement) type, which is uploaded as a rank-0 array). This trait is
/// the single list of uploadable host values. Every host value that implements it converts into [`HostOrArray`], and
/// every one that is also a [`Parameter`] is a [`Placeable`] leaf on its own.
trait IntoHostArray {
    /// Returns this value as a host array of the reference backend.
    fn into_host_array(self) -> CoreArray;
}

impl IntoHostArray for CoreArray {
    #[inline]
    fn into_host_array(self) -> CoreArray {
        self
    }
}

macro_rules! impl_into_host_array {
    // Implements `IntoHostArray` for the scalar `ArrayElement` type `$ty` as a rank-0 host array.
    ($ty:ty) => {
        impl IntoHostArray for $ty {
            #[inline]
            fn into_host_array(self) -> CoreArray {
                // A rank-0 array of an element type always has a valid shape and encoding.
                CoreArray::scalar(self).unwrap()
            }
        }
    };
}

impl_into_host_array!(bool);
impl_into_host_array!(i1);
impl_into_host_array!(i2);
impl_into_host_array!(i4);
impl_into_host_array!(i8);
impl_into_host_array!(i16);
impl_into_host_array!(i32);
impl_into_host_array!(i64);
impl_into_host_array!(u1);
impl_into_host_array!(u2);
impl_into_host_array!(u4);
impl_into_host_array!(u8);
impl_into_host_array!(u16);
impl_into_host_array!(u32);
impl_into_host_array!(u64);
impl_into_host_array!(f4e2m1fn);
impl_into_host_array!(f6e2m3fn);
impl_into_host_array!(f6e3m2fn);
impl_into_host_array!(f8e3m4);
impl_into_host_array!(f8e4m3);
impl_into_host_array!(f8e4m3fn);
impl_into_host_array!(f8e4m3fnuz);
impl_into_host_array!(f8e4m3b11fnuz);
impl_into_host_array!(f8e5m2);
impl_into_host_array!(f8e5m2fnuz);
impl_into_host_array!(f8e8m0fnu);
impl_into_host_array!(bf16);
impl_into_host_array!(f16);
impl_into_host_array!(f32);
impl_into_host_array!(f64);
impl_into_host_array!(Complex<f32>);
impl_into_host_array!(Complex<f64>);

/// Value that may wrap concrete [`Array`]s, through whose sessions the placement of a traced value executes. A traced
/// value places its arrays with the contexts that it was created in, which hold the sessions of those arrays, so
/// [`Session::place`] uses this trait to reject a traced value over the arrays of another session than the receiving
/// one, and in particular over the arrays of another PJRT client.
trait ConcreteArrays {
    /// Returns every concrete [`Array`] that this value wraps (e.g., both the primal and the tangent of a
    /// differentiated value). Staged values and host values wrap none.
    fn concrete_arrays(&self) -> Vec<&Array>;
}

impl ConcreteArrays for Array {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        vec![self]
    }
}

impl ConcreteArrays for Value {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        match self {
            Self::Array(value) => vec![value],
            Self::Dimension(_) | Self::Reference(_) => Vec::new(),
        }
    }
}

impl ConcreteArrays for CoreArray {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        Vec::new()
    }
}

impl<C: Context> ConcreteArrays for Tracer<C> {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        Vec::new()
    }
}

impl<C: Context<Value: ConcreteArrays>> ConcreteArrays for PartialTracer<C> {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        // An unknown value has no payload yet, and a poisoned value surfaces its deferred error when it is used.
        match self.value().ok().and_then(|value| value.as_known()) {
            Some(value) => value.concrete_arrays(),
            None => Vec::new(),
        }
    }
}

impl<C: Context<Value: ConcreteArrays>, P: DifferentiationPolicy<C>> ConcreteArrays for DifferentiationTracer<C, P> {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        let mut arrays = self.primal().concrete_arrays();
        if let Some(tangent) = self.tangent().as_value() {
            arrays.extend(tangent.concrete_arrays());
        }
        arrays
    }
}

impl<C: Context<Value: ConcreteArrays>, P: BatchingPolicy<C>> ConcreteArrays for BatchingTracer<C, P> {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        P::value(self.batch()).concrete_arrays()
    }
}

impl<V: ConcreteArrays> ConcreteArrays for ProjectedValue<ArrayType, V> {
    #[inline]
    fn concrete_arrays(&self) -> Vec<&Array> {
        self.value().concrete_arrays()
    }
}

/// Validates the destination `target` and the expected `source` of the [`Array`] leaf `array` of [`Session::place`]
/// against the type of `array` with the type rule of [`PlaceOperation`] (refer to
/// [`PlaceOperation::infer_output_type_allowing_uneven_splits`]), as given and before anything is moved, so that [`Session::place`] rejects
/// every request that the type rule rejects, with the error that the type rule reports first, as every path that stages
/// or executes a placement does. This includes the reduction state of `array`, which the placement never changes (e.g.,
/// it never sums pending contributions onto one device), which only [`Array::to_placement`], the physical primitive
/// that it runs on, does when its destination requires it, and a value inside a manual region, which it never moves.
/// The one check of the type rule that it does not make is that the target splits each dimension evenly, which an
/// [`Array`] does not have to (refer to [`Session::place`]).
///
/// The type of `array` omits its placement along auto mesh axes, including a pending or completed reduction over them,
/// which its runtime sharding records, so the request is validated with that runtime sharding as the concrete sharding
/// of the input (refer to [`PlaceOperation::infer_output_type_allowing_uneven_splits`]), as the rules of [`PlaceOperation`] that place a
/// concrete value do. A target that restates the reduction state of `array` over an auto axis (e.g., its own placement)
/// is therefore accepted and changes nothing, while a target that drops it is rejected, and a source layout and an
/// in-place target layout are checked against the shards that the runtime sharding determines.
///
/// The source is validated as it is given, as the core [`Place`] capability validates it, and it is not completed with
/// the placement of `array` (refer to [`PlacementRequest::resolve_source`]), because the placement of a concrete array
/// records nothing for transposition. The devices that the source names are informative, as on every other path and as
/// in JAX, so they are not compared with the devices of `array`, and neither are the physical layouts of its buffers,
/// which no staged, compiled, or transformed path could inspect, and which a process could inspect only for its own
/// buffers.
fn validate_array_request(
    session: &Session,
    array: &Array,
    target: Option<&ArrayPlacement>,
    source: Option<ArrayPlacement>,
) -> Result<(), Error> {
    let r#type = array.r#type();
    let rank = r#type.rank();
    let target = target.map(|target| target.clone().into_request(session, rank)).transpose()?;
    let source = source.map(|source| source.into_request(session, rank)).transpose()?;
    PlaceOperation::infer_output_type_allowing_uneven_splits(&r#type, Some(array.sharding()), target.as_ref(), source.as_ref())?;
    Ok(())
}

/// Validates that the resolved destination `placement` of the concrete array `array` has the rank of that array, with
/// the error of the type rule of [`PlaceOperation`], which owns that check. [`Array::to_placement`] and
/// [`Array::into_placement`] are physical primitives that the type rule does not validate otherwise, and every other
/// placement of an array validates its whole request with the type rule before it resolves its destination. The type
/// rule is only consulted for a mismatched rank, which is the first check that it makes for a placement target, because
/// it also rejects placements that the physical primitives support (e.g., uneven splits).
fn validate_target_rank(placement: &Placement, r#type: &ArrayType) -> Result<(), Error> {
    if placement.sharding().rank() != r#type.rank() {
        PlaceOperation::infer_output_type_allowing_uneven_splits(r#type, None, Some(&PlacementRequest::Resolved(placement.clone())), None)?;
    }
    Ok(())
}

/// Places the traced `value` through its [`PlaceDispatch`] primitive, with `target` and `source` converted against
/// `session` into the [`PlacementRequest`]s that they stage. This implements [`Placeable::place`] for every traced
/// value type, so that they all handle the placement options identically (refer to [`Session::place`] for the resulting
/// behavior).
///
/// Every destination and source is staged in the form that it was given (refer to [`ArrayPlacement::into_request`]),
/// whether or not `value` wraps a concrete array, so a bare device stays a [`PlacementRequest::Device`] and a bare
/// memory stays a [`PlacementRequest::InPlace`], and only a destination or source that names a mesh or a sharding is
/// resolved for the rank of the type of `value` (which is the rank of each batch item for a batched value). A device or
/// in-place request resolves against a concrete array only when the staged placement executes on it (refer to
/// [`ArrayPlacement::from_request`]), and the type rule of [`PlaceOperation`] owns what each request does to the
/// type of `value`, so that the type of the result does not depend on whether `value` wraps a concrete array. Without
/// a source, the rule of [`PlaceOperation`] that places a traced value that wraps a concrete array records the
/// placement of that array (refer to [`Placed`]) as the source, so that reverse mode differentiation moves the
/// cotangent back there, and it completes a source that names only a memory or a layout with that placement rather
/// than replacing it (refer to [`PlacementRequest::resolve_source`]), so that such a source never loses the devices of
/// the value. That rule validates the requests as they are given with the type rule first, so this function does not
/// validate them itself. Without a target, the value is placed without one, which leaves it where it is, with the
/// source if one is provided.
///
/// With `may_alias` set to `Some(false)`, the placement forces a copy (refer to
/// [`PlaceOperation::with_may_alias`]), also without a target and a source, so that its result shares no buffer with
/// any other array once it executes on concrete arrays. Any other `may_alias` stages the placement with the default
/// copy policy. The donation option only applies to concrete [`Array`] leaves, so it is ignored here.
///
/// The placement executes through the contexts of `value` (e.g., its batching or differentiation context), which hold
/// the session of the arrays that `value` wraps, so its results belong to that session and not to `session`. Those
/// contexts cannot be rebuilt over another session (e.g., the partial evaluation context of reverse mode
/// differentiation owns the residual program that the value is part of), so every array that `value` wraps (i.e., its
/// primal, its tangent, its packed batch, and any known partial value, at every level of nesting) must belong to
/// `session`. An array of another PJRT client is rejected with the error of [`Array::with_session`], as a concrete
/// array of that client is, and an array of another session of the same client is rejected with a
/// [`ryft_pjrt::Error::InvalidArgument`] error. The devices of the resolved placements cannot reveal either, because
/// device IDs are only unique within one client. Other handles of the same session (e.g., ones with another default
/// mesh) are accepted, and the results stay attached to the handles of the wrapped arrays.
fn place_traced<V: Placed + ConcreteArrays + CoreValue<Type = ArrayType, Dispatch: PlaceDispatch<ArrayType, V>>>(
    value: V,
    session: &Session,
    target: Option<ArrayPlacement>,
    source: Option<ArrayPlacement>,
    donate: bool,
    may_alias: Option<bool>,
) -> Result<V, Error> {
    let _ = donate;
    for array in value.concrete_arrays() {
        // Only the client check of `Array::with_session` matters here, because the placement executes through the
        // contexts of `value`, which cannot be rebound to `session`.
        array.clone().with_session(session)?;
        if !array.session().is_same_session(session) {
            let message = "cannot place a traced value over arrays of another session";
            return Err(PjrtError::invalid_argument(message).into());
        }
    }
    // The requests are placed as they are given, and the rules of `PlaceOperation` validate them with its type rule
    // before they complete the source with the placement of a value that concretely lives somewhere.
    let rank = value.r#type().rank();
    let source = source.map(|source| source.into_request(session, rank)).transpose()?;
    let target = target.map(|target| target.into_request(session, rank)).transpose()?;
    Ok(V::Dispatch::place(&value, target.as_ref(), source, may_alias != Some(false))?)
}

/// Returns whether `tree` is one leaf at its root (e.g., a bare [`Device`] as the destination of a whole input tree),
/// as opposed to a container, even one with a single leaf, whose leaves lie below the root.
fn is_root_parameter<P: Parameter, T: Parameterized<P>>(tree: &T) -> bool {
    tree.parameter_count() == 1 && tree.named_parameters().next().is_some_and(|(path, _)| path.is_root())
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Add, BatchAxis, DataType, DeviceMesh, DifferentiableType, DimensionBounds, DimensionVariable,
        Error as CoreError, Exp, LogicalMesh, Memory, MeshAxis, MeshAxisType, Operation as CoreOperation,
        ParameterError, Place, PlaceOperation, ProgramError, ShardMapOptions, Sharding, ShardingDimension,
        ShardingError, StridedLayout, TiledLayout, Typed, ValueProjection, batch, differentiate_at, shard_map,
    };
    use ryft_pjrt::BufferType;

    use crate::sessions::tests::{domain_mesh, two_device_mesh};
    use crate::tests::{cpu_client, device_mesh_2x2, other_cpu_client, values_from_bytes, values_to_bytes};
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, CompiledFunctionDispatcher, LoweringError, Options, SessionError,
        StagedFunction, StatefulCompileTracer, ToPjrt, TracedProgram, Type, Value, compile, compile_statefully,
        compile_with_options, jit, stage, trace,
    };

    use super::*;

    // Returns the `f32` values that the devices of `array` hold, in the device order of its mesh.
    fn shard_values(array: &Array) -> Vec<Vec<f32>> {
        array
            .shards()
            .iter()
            .map(|shard| values_from_bytes::<f32>(&shard.copy_to_host_bytes().unwrap()))
            .collect()
    }

    // Returns the native addresses of the buffers that the devices of `array` hold, in the device order of its mesh,
    // once they are ready. Two arrays that are both alive share storage exactly when they share such an address, which
    // the identity of the handles of their buffers does not show.
    fn native_buffer_addresses(array: &Array) -> Vec<usize> {
        array
            .shards()
            .iter()
            .map(|shard| {
                let buffer = shard.buffer().unwrap();
                buffer.ready().unwrap().r#await().unwrap();
                // The buffer is ready and stays alive while `array` does, and its address is only compared and never
                // dereferenced.
                unsafe { buffer.unsafe_pointer().unwrap() }
            })
            .collect()
    }

    // Returns the physical dimensions of the buffer that each device of `array` holds, in the device order of its mesh.
    fn shard_buffer_dimensions(array: &Array) -> Vec<Vec<u64>> {
        array.shards().iter().map(|shard| shard.buffer().unwrap().dimensions().unwrap().to_vec()).collect()
    }

    #[test]
    fn test_array_placement_new() {
        let sharding = Sharding::replicated(device_mesh_2x2().logical_mesh().clone(), 1);
        let target = ArrayPlacementTarget::Sharding(sharding);
        let placement = ArrayPlacement::new(target.clone());
        assert_eq!(placement.target(), &target);
        assert_eq!(placement.layout(), None);
        assert_eq!(placement.memory(), None);
    }

    #[test]
    fn test_array_placement_device() {
        let device = Device::new(3, 0);
        let placement = ArrayPlacement::device(device);
        assert_eq!(placement, ArrayPlacement::new(ArrayPlacementTarget::Device(device)));
        assert_eq!(placement.target(), &ArrayPlacementTarget::Device(device));
        assert_eq!(ArrayPlacement::from(device), placement);
    }

    #[test]
    fn test_array_placement_mesh() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(
            ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap(),
            ArrayPlacement::new(ArrayPlacementTarget::Mesh { mesh: mesh.clone(), sharding }),
        );

        // A sharding over another logical mesh is rejected.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(other_mesh.clone(), 1)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh()),
        );
    }

    #[test]
    fn test_array_placement_sharding() {
        let sharding = Sharding::replicated(device_mesh_2x2().logical_mesh().clone(), 1);
        assert_eq!(
            ArrayPlacement::sharding(sharding.clone()),
            ArrayPlacement::new(ArrayPlacementTarget::Sharding(sharding.clone())),
        );
        assert_eq!(ArrayPlacement::from(sharding.clone()), ArrayPlacement::sharding(sharding));
    }

    #[test]
    fn test_array_placement_in_place() {
        // An in-place placement names no devices, and a bare memory converts into one in that memory.
        let placement = ArrayPlacement::in_place();
        assert_eq!(placement, ArrayPlacement::new(ArrayPlacementTarget::InPlace));
        assert_eq!(placement.layout(), None);
        assert_eq!(placement.memory(), None);
        assert!(placement.batch_shardings().is_empty());
        let memory = Memory::Host { pinned: true };
        assert_eq!(ArrayPlacement::from(memory), ArrayPlacement::in_place().with_memory(memory));
    }

    #[test]
    fn test_array_placement_with_layout() {
        let device = Device::new(3, 0);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let placement = ArrayPlacement::device(device).with_memory(Memory::Device).with_layout(layout.clone());
        assert_eq!(placement.target(), &ArrayPlacementTarget::Device(device));
        assert_eq!(placement.layout(), Some(&layout));
        assert_eq!(placement.memory(), Some(Memory::Device));

        // The last layout replaces an earlier one.
        let other_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        assert_eq!(placement.with_layout(other_layout.clone()).layout(), Some(&other_layout));
    }

    #[test]
    fn test_array_placement_with_memory() {
        let device = Device::new(3, 0);
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let placement = ArrayPlacement::device(device).with_layout(layout.clone()).with_memory(Memory::Device);
        assert_eq!(placement.target(), &ArrayPlacementTarget::Device(device));
        assert_eq!(placement.layout(), Some(&layout));
        assert_eq!(placement.memory(), Some(Memory::Device));

        // The last memory replaces an earlier one, and the order of the storage options does not matter.
        let memory = Memory::Host { pinned: true };
        assert_eq!(placement.clone().with_memory(memory).memory(), Some(memory));
        assert_eq!(placement, ArrayPlacement::device(device).with_memory(Memory::Device).with_layout(layout));
    }

    #[test]
    fn test_array_placement_batch_shardings() {
        let mesh = device_mesh_2x2();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);

        // A request records no enclosing batches, whereas one converted from the placement of a batch item keeps the
        // batch record of that placement, which also makes it differ from a request without that record.
        let request = ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap();
        assert!(request.batch_shardings().is_empty());
        let placement = Placement::new(mesh, sharding).unwrap().with_batch_sharding(ShardingDimension::sharded(["x"]));
        let converted = ArrayPlacement::from(placement.unwrap());
        assert_eq!(converted.batch_shardings(), [ShardingDimension::sharded(["x"])]);
        assert_eq!(converted.target(), request.target());
        assert_ne!(converted, request);
    }

    #[test]
    fn test_array_placement_resolve() {
        let session = Session::builder().client(cpu_client(1)).build().unwrap();

        // A device target resolves to a replicated placement on a mesh with one auto axis of size 1.
        let device = Device::new(3, 0);
        let device_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        assert_eq!(
            ArrayPlacement::device(device).resolve(&session, 2).unwrap(),
            Placement::new(
                DeviceMesh::new(device_mesh.clone(), vec![device]).unwrap(),
                Sharding::replicated(device_mesh, 2),
            )
            .unwrap(),
        );

        // A placement target resolves to its own mesh and sharding.
        let mesh = device_mesh_2x2();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        assert_eq!(
            ArrayPlacement::mesh(mesh.clone(), sharding.clone()).unwrap().resolve(&session, 1).unwrap(),
            Placement::new(mesh.clone(), sharding.clone()).unwrap(),
        );

        // A bare sharding needs a configured mesh with matching topology.
        let placement = ArrayPlacement::sharding(sharding.clone());
        assert!(matches!(placement.clone().resolve(&session, 1), Err(Error::Session(SessionError::MissingMesh))));
        let configured = session.with_device_mesh(mesh.clone());
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        assert_eq!(
            placement
                .with_memory(Memory::Device)
                .with_layout(layout.clone())
                .with_memory(memory)
                .resolve(&configured, 1)
                .unwrap(),
            Placement::new(mesh.clone(), sharding).unwrap().with_memory(memory).with_layout(layout),
        );

        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        let error = ArrayPlacement::sharding(Sharding::replicated(other_mesh.clone(), 1))
            .resolve(&configured, 1)
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Core(error) if matches!(
                error.as_ref(),
                CoreError::Sharding(ShardingError::MeshMismatch { expected, actual })
                    if expected == mesh.logical_mesh() && actual == &other_mesh,
            ),
        ));
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh())
        );

        // A resolved placement converts into a request that resolves back to it, including the batch record of the
        // placement of a batch item.
        let placement = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap()
            .with_batch_sharding(ShardingDimension::sharded(["x"]))
            .unwrap()
            .with_memory(memory);
        assert_eq!(ArrayPlacement::from(placement.clone()).resolve(&session, 1).unwrap(), placement);
    }

    #[test]
    fn test_array_placement_resolve_in_place() {
        // An in-place request names no devices, so it does not resolve without an array.
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let error = ArrayPlacement::from(Memory::Host { pinned: true }).resolve(&session, 1).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place placement names no devices to resolve");
    }

    #[test]
    fn test_array_placement_resolve_for() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // A device target on the device that the array lives on keeps the mesh of the array, even though that mesh
        // names its axis differently from the mesh that a device target resolves to, and its sharding without the
        // pending reduction along its axis of size one, which one device completes.
        let memory = Memory::Host { pinned: true };
        let replicated = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        assert_eq!(
            ArrayPlacement::device(devices[0]).with_memory(memory).resolve_for(&session, &array).unwrap(),
            replicated.clone().with_memory(memory),
        );
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let unreduced_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(
            ArrayPlacement::device(devices[0])
                .with_memory(memory)
                .resolve_for(&session, &unreduced_array)
                .unwrap(),
            replicated.clone().with_memory(memory),
        );

        // Any other device keeps the logical mesh and the sharding of an array on a mesh of one device, laid over
        // that device, so that the array keeps how its dimensions are sharded.
        let other_mesh = DeviceMesh::new(mesh.logical_mesh().clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let sharded_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap(),
            mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(
            ArrayPlacement::device(devices[1])
                .with_memory(memory)
                .resolve_for(&session, &sharded_array)
                .unwrap(),
            Placement::new(other_mesh.clone(), sharding).unwrap().with_memory(memory),
        );

        // An array on more devices resolves like a value of its rank, which gathers it onto the device.
        let two_device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices.clone(),
        )
        .unwrap();
        let split_array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            two_device_mesh,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let other = ArrayPlacement::device(devices[1]).with_memory(memory);
        assert_eq!(other.clone().resolve_for(&session, &split_array).unwrap(), other.resolve(&session, 1).unwrap());

        // A device target without a memory selects the default memory of its device, as JAX's `device_put` does,
        // whereas a sharding target without a memory selects none, so the array keeps its own memory.
        assert_eq!(
            ArrayPlacement::device(devices[0]).resolve_for(&session, &array).unwrap(),
            replicated.clone().with_memory(Memory::Device),
        );
        assert_eq!(
            ArrayPlacement::device(devices[1]).resolve_for(&session, &array).unwrap(),
            Placement::new(other_mesh, Sharding::replicated(mesh.logical_mesh().clone(), 1))
                .unwrap()
                .with_memory(Memory::Device),
        );
        let configured = session.with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(ArrayPlacement::sharding(sharding).resolve_for(&configured, &array).unwrap(), replicated);
    }

    #[test]
    fn test_array_placement_resolve_for_in_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();

        // An in-place target resolves to the mesh and the sharding of the array, with the memory and the layout that
        // it names, and without them the array keeps its own.
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let placement = Placement::new(mesh.clone(), sharding).unwrap();
        assert_eq!(
            ArrayPlacement::from(memory).with_layout(layout.clone()).resolve_for(&session, &array).unwrap(),
            placement.clone().with_memory(memory).with_layout(layout.clone()),
        );
        assert_eq!(
            ArrayPlacement::in_place().with_layout(layout.clone()).resolve_for(&session, &array).unwrap(),
            placement.with_layout(layout),
        );

        // An in-place target that names neither a memory nor a layout changes nothing, so it is rejected.
        let error = ArrayPlacement::in_place().resolve_for(&session, &array).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place target must name a memory or a layout");
    }

    #[test]
    fn test_array_placement_into_request() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

        // A device stays a device target with the memory and layout of the request, also when it is the only device
        // of the default mesh of the session, because the mesh of an unplaced value is not known.
        assert_eq!(
            ArrayPlacement::device(device)
                .with_memory(Memory::Host { pinned: true })
                .with_layout(layout.clone())
                .into_request(&session, 2)
                .unwrap(),
            PlacementRequest::Device { device, memory: Some(Memory::Host { pinned: true }), layout: Some(layout) },
        );
        assert_eq!(
            ArrayPlacement::device(device).into_request(&session, 2).unwrap(),
            PlacementRequest::Device { device, memory: None, layout: None },
        );

        // Every other request resolves as it does for any value of that rank.
        let sharding = ArrayPlacement::sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1));
        assert_eq!(
            sharding.clone().into_request(&session, 2).unwrap(),
            PlacementRequest::Resolved(sharding.resolve(&session, 2).unwrap()),
        );
    }

    #[test]
    fn test_array_placement_into_request_in_place() {
        // An in-place request names no devices, so it stays an in-place target with the memory and the layout of the
        // request.
        let session = Session::builder().client(cpu_client(1)).build().unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        assert_eq!(
            ArrayPlacement::from(Memory::Host { pinned: true })
                .with_layout(layout.clone())
                .into_request(&session, 2)
                .unwrap(),
            PlacementRequest::InPlace { memory: Some(Memory::Host { pinned: true }), layout: Some(layout) },
        );
    }

    #[test]
    fn test_placement_options_new() {
        let target = ArrayPlacement::device(Device::new(3, 0));
        let options =
            PlacementOptions::new(Some(target.clone()), Option::<ArrayPlacement>::None, Some(true), Some(Some(false)));
        assert_eq!(options.target(), Some(&target));
        assert_eq!(options.source(), None);
        assert_eq!(options.donate(), Some(&true));
        assert_eq!(options.may_alias(), Some(&Some(false)));
        assert_eq!(options.into_parts(), (Some(target), None, Some(true), Some(Some(false))));
    }

    #[test]
    fn test_placement_options_empty() {
        let options = PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, Option<bool>>::empty();
        assert_eq!(options.into_parts(), (None, None, None, None));
        assert_eq!(PlacementOptions::defaults(), PlacementOptions::empty());
        assert_eq!(PlacementOptions::default(), PlacementOptions::defaults());
    }

    #[test]
    fn test_placement_options_with_target() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The destination changes its type, and the other options keep theirs.
        let options = PlacementOptions::defaults().with_may_alias(Some(false)).with_target((Some(devices[1]), None));
        assert_eq!(options.clone().into_parts(), (Some((Some(devices[1]), None)), None, None, Some(Some(false))));
        let (placed, uploaded) = session.place((1f32, 2f32), options).unwrap();
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert!(placed.is_committed());
        assert_eq!(uploaded.mesh().devices(), [devices[0]]);
        assert!(!uploaded.is_committed());
    }

    #[test]
    fn test_placement_options_with_source() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let array = session.place(2.5f32, PlacementOptions::to(devices[0])).unwrap();

        // The source changes its type, and it is validated against the type of the array leaf, whereas the devices that
        // it names are informative.
        let options = PlacementOptions::to(devices[1]).with_source(devices[0]);
        assert_eq!(options.clone().into_parts(), (Some(devices[1]), Some(devices[0]), None, None));
        let moved = session.place(array.clone(), options).unwrap();
        assert_eq!(moved.mesh().devices(), [devices[1]]);
        let kept = session.place(array.clone(), PlacementOptions::defaults().with_source(devices[1])).unwrap();
        assert_eq!(kept.mesh().devices(), [devices[0]]);
        let pinned = ArrayPlacement::from(Memory::Host { pinned: true });
        let error = session.place(array.clone(), PlacementOptions::defaults().with_source(pinned)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!("`place` source memory `Host[Pinned]` does not match the input type `{}`", array.r#type()),
        );
    }

    #[test]
    fn test_placement_options_with_donate() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let array = session.place(2.5f32, PlacementOptions::defaults()).unwrap();
        let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());

        // The donation changes its type, and a donated leaf is consumed.
        let options = PlacementOptions::to(device).with_donate(true);
        assert_eq!(options.clone().into_parts(), (Some(device), None, Some(true), None));
        let placed = session.place(array, options).unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
        assert_eq!(shard_values(&placed), vec![vec![2.5]]);
        assert!(buffer.upgrade().is_none());
        let options = PlacementOptions::defaults().with_donate((true, false));
        assert_eq!(options.into_parts(), (None, None, Some((true, false)), None));
    }

    #[test]
    fn test_placement_options_with_may_alias() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = session.place(2.5f32, PlacementOptions::defaults()).unwrap();

        let shares_buffer =
            |placed: &Array| Arc::ptr_eq(placed.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap());

        // A forced copy needs no destination, and the copy shares no buffer with the input.
        let options = PlacementOptions::defaults().with_may_alias(Some(false));
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(Some(false))));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(copied.mesh(), array.mesh());
        assert_eq!(copied.r#type(), array.r#type());
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
        let kept = session.place(array.clone(), PlacementOptions::defaults().with_may_alias(Some(true))).unwrap();
        assert!(shares_buffer(&kept));

        // Plain `bool` aliasing options are accepted as well, and they mean what their `Some` forms mean.
        let options = PlacementOptions::defaults().with_may_alias(false);
        assert_eq!(options.clone().into_parts(), (None, None, None, Some(false)));
        let copied = session.place(array.clone(), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(copied.mesh(), array.mesh());
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
        let kept = session.place(array.clone(), PlacementOptions::defaults().with_may_alias(true)).unwrap();
        assert!(shares_buffer(&kept));

        // So are tuple prefixes of plain `bool` options.
        let options = PlacementOptions::defaults().with_may_alias((false, true));
        let (copied, kept) = session.place((array.clone(), array.clone()), options).unwrap();
        assert!(!shares_buffer(&copied));
        assert_eq!(shard_values(&copied), vec![vec![2.5]]);
        assert!(shares_buffer(&kept));
    }

    #[test]
    fn test_placement_options_to() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(PlacementOptions::to(devices[1]).into_parts(), (Some(devices[1]), None, None, None));

        // The destination alone fixes the types of the other options, whether it is a raw device, a placement request,
        // or a tuple prefix.
        let scalar = session.place(2.5f32, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert!(scalar.is_committed());
        let moved = session.place(scalar, PlacementOptions::to(ArrayPlacement::device(devices[0]))).unwrap();
        assert_eq!(moved.mesh().devices(), [devices[0]]);
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        let (placed, uploaded) = session.place((1f32, 2f32), PlacementOptions::to((Some(devices[1]), None))).unwrap();
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert!(placed.is_committed());
        assert_eq!(uploaded.mesh().devices(), [devices[0]]);
        assert!(!uploaded.is_committed());
    }

    #[test]
    fn test_host_or_array_from() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let array = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let complex = Complex::new(1.5f32, -2.);
        let low_precision = f8e4m3fn::from_f32(0.5).unwrap();

        // Every scalar that a host array can hold converts into a host value, including complex and low precision
        // scalars, which are not leaves on their own, so one tree can place them together with arrays.
        assert!(matches!(
            HostOrArray::from(complex),
            HostOrArray::Host(host) if host == CoreArray::scalar(complex).unwrap(),
        ));
        let (complex_array, low_precision_array, array) = session
            .place(
                (HostOrArray::from(complex), HostOrArray::from(low_precision), HostOrArray::from(array)),
                PlacementOptions::defaults(),
            )
            .unwrap();
        assert_eq!(complex_array.data_type(), DataType::C64);
        assert_eq!(complex_array.copy_to_host().unwrap().elements::<Complex<f32>>().unwrap(), vec![complex]);
        assert_eq!(low_precision_array.data_type(), DataType::F8E4M3FN);
        assert_eq!(low_precision_array.copy_to_host().unwrap().elements::<f8e4m3fn>().unwrap(), vec![low_precision]);
        assert_eq!(shard_values(&array), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_host_or_array_place() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        assert!(matches!(HostOrArray::from(true), HostOrArray::Host(host) if host == CoreArray::scalar(true).unwrap()));
        assert!(matches!(HostOrArray::from(upload()), HostOrArray::Array(_)));

        // One tree mixes host values and arrays, and the destination prefix broadcasts over both kinds of leaves.
        let array = upload();
        let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());
        let options = PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, _>::new(
            Some(ArrayPlacement::device(devices[1])),
            None,
            None,
            Some(Some(false)),
        );
        let (scalar, vector, placed) = session
            .place(
                (
                    HostOrArray::from(2.5f32),
                    HostOrArray::from(CoreArray::vector(vec![3i32, 4]).unwrap()),
                    HostOrArray::from(array),
                ),
                options,
            )
            .unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        assert_eq!(vector.mesh().devices(), [devices[1]]);
        assert_eq!(vector.copy_to_host().unwrap().elements::<i32>().unwrap(), vec![3, 4]);
        assert_eq!(placed.mesh().devices(), [devices[1]]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2.]]);
        assert!(buffer.upgrade().is_none());

        // The source prefix also broadcasts over both kinds of leaves, and only the array leaf validates it.
        let options = |source: ArrayPlacement| {
            PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(None, Some(source), None, None)
        };
        let (scalar, kept) = session
            .place((HostOrArray::from(1u8), HostOrArray::from(upload())), options(ArrayPlacement::device(devices[0])))
            .unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[0]]);
        assert_eq!(scalar.copy_to_host_bytes().unwrap(), vec![1u8]);
        assert_eq!(kept.mesh().devices(), [devices[0]]);
        let pinned = ArrayPlacement::from(Memory::Host { pinned: true });
        let error = session.place((HostOrArray::from(1u8), HostOrArray::from(upload())), options(pinned)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
    }

    #[test]
    fn test_array_into_unaliased() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let buffer_addresses =
            |array: &Array| array.shards().iter().map(|shard| Arc::as_ptr(shard.buffer().unwrap())).collect::<Vec<_>>();

        // An array that is the only handle of its buffers is returned as it is.
        let array = upload();
        let addresses = buffer_addresses(&array);
        let unaliased = array.into_unaliased(&session).unwrap();
        assert_eq!(buffer_addresses(&unaliased), addresses);

        // Only the buffers that another handle shares are copied, and the others are kept.
        let array = upload();
        let addresses = buffer_addresses(&array);
        let native_addresses = native_buffer_addresses(&array);
        let held = Arc::clone(array.shards()[0].buffer().unwrap());
        let unaliased = array.into_unaliased(&session).unwrap();
        assert!(!Arc::ptr_eq(unaliased.shards()[0].buffer().unwrap(), &held));
        assert_ne!(native_buffer_addresses(&unaliased)[0], native_addresses[0]);
        assert_eq!(buffer_addresses(&unaliased)[1], addresses[1]);
        assert_eq!(native_buffer_addresses(&unaliased)[1], native_addresses[1]);
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // A clone shares every buffer, so all of them are copied.
        let array = upload();
        let unaliased = array.clone().into_unaliased(&session).unwrap();
        for (unaliased_shard, shard) in unaliased.shards().iter().zip(array.shards()) {
            assert!(!Arc::ptr_eq(unaliased_shard.buffer().unwrap(), shard.buffer().unwrap()));
        }
        for (unaliased_address, address) in
            native_buffer_addresses(&unaliased).into_iter().zip(native_buffer_addresses(&array))
        {
            assert_ne!(unaliased_address, address);
        }
        assert_eq!(unaliased.r#type(), array.r#type());
        assert_eq!(unaliased.mesh(), array.mesh());
        assert_eq!(unaliased.sharding(), array.sharding());
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // The copies keep the commitment of the array.
        let array = upload().with_committed(false);
        let unaliased = array.clone().into_unaliased(&session).unwrap();
        assert!(!unaliased.is_committed());
        assert!(array.into_unaliased(&session).unwrap().with_committed(true).is_committed());
    }

    #[test]
    fn test_session_place_uploads_host_leaves() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();

        // Without a destination, a host leaf is uploaded to the first local device without being committed to it, also
        // when the session has a mesh.
        let scalar = session.place(2.5f32, PlacementOptions::defaults()).unwrap();
        let flag = session.place(true, PlacementOptions::defaults()).unwrap();
        assert_eq!(scalar.r#type().into_owned().with_sharding(None).unwrap(), ArrayType::scalar(DataType::F32));
        assert_eq!(scalar.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert!(!scalar.is_committed());
        let meshed_session = session.with_device_mesh(domain_mesh(client, "x", 2));
        assert!(!meshed_session.place(2.5f32, PlacementOptions::defaults()).unwrap().is_committed());
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![2.5]);
        assert_eq!(flag.data_type(), DataType::Boolean);
        assert_eq!(flag.copy_to_host_bytes().unwrap(), vec![1u8]);

        // A destination device places the leaf on that device.
        let placed = session
            .place(
                bf16::from_f32(1.5),
                PlacementOptions::new(
                    Some(ArrayPlacement::device(device)),
                    Option::<ArrayPlacement>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), [device]);
        assert!(placed.is_committed());
        assert_eq!(placed.copy_to_host_bytes().unwrap(), bf16::from_f32(1.5).to_bits().to_ne_bytes().to_vec());
    }

    #[test]
    fn test_session_place_uploads_reference_backend_arrays() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let matrix = CoreArray::matrix(2, 3, vec![1f32, 2., 3., 4., 5., 6.]).unwrap();
        let array = session.place(matrix, PlacementOptions::defaults()).unwrap();
        assert_eq!(
            array.r#type().into_owned().with_sharding(None).unwrap(),
            ArrayType::new_static(DataType::F32, [2, 3]),
        );
        assert_eq!(array.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4., 5., 6.]);
    }

    #[test]
    fn test_session_place_validates_host_storage() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();

        // The storage of a host array can be mutated safely, so its upload validates the encoding of its elements
        // rather than trusting it, with and without an explicit layout, and reports an invalid Boolean byte.
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        for r#type in [
            ArrayType::new_static(DataType::Boolean, [1]),
            ArrayType::new_static(DataType::Boolean, [1]).with_layout(layout),
        ] {
            let mut host = CoreArray::new(r#type, vec![1]).unwrap();
            host.storage_bytes_mut()[0] = 2;
            let error = session.place(host, PlacementOptions::defaults()).unwrap_err();
            assert_eq!(error.to_string(), "array element 0 has invalid `bool` byte encoding [2]");
        }
    }

    #[test]
    fn test_session_place_tracks_local_assignment() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = session.target().unwrap().devices()[1].device();
        let pinned = Memory::Host { pinned: true };

        // A host value uploaded without a destination, or with one that names only its storage, lives on a device that
        // this process chose on its own, and a destination that names a device does not.
        let local = session.place(1f32, PlacementOptions::defaults()).unwrap();
        assert!(local.is_locally_assigned());
        assert!(session.place(1f32, PlacementOptions::to(pinned)).unwrap().is_locally_assigned());
        assert!(!session.place(1f32, PlacementOptions::to(device)).unwrap().is_locally_assigned());

        // An array keeps the fact without a destination and with an in-place one, eagerly and when a staged in-place
        // placement executes on it, and a destination that names a device clears it.
        assert!(session.place(local.clone(), PlacementOptions::defaults()).unwrap().is_locally_assigned());
        assert!(session.place(local.clone(), PlacementOptions::to(pinned)).unwrap().is_locally_assigned());
        let staged =
            session.bind(PlaceOperation::new(pinned, None), Vec::new(), &[Value::Array(local.clone())]).unwrap();
        let [Value::Array(staged)] = staged.as_slice() else { panic!("placing an array must produce an array") };
        assert!(staged.is_locally_assigned());
        assert!(!session.place(local, PlacementOptions::to(device)).unwrap().is_locally_assigned());
    }

    #[test]
    fn test_session_place_applies_memory_and_layout() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let values = vec![1f32, 2., 3., 4., 5., 6., 7., 8.];
        let placement = Placement::new(mesh.clone(), sharding.clone())
            .unwrap()
            .with_layout(layout.clone())
            .with_memory(memory);
        let uploaded = session
            .place(
                CoreArray::matrix(4, 2, values.clone()).unwrap(),
                PlacementOptions::<Placement>::new(Some(placement.clone()), None, None, None),
            )
            .unwrap();

        assert_eq!(uploaded.mesh(), &mesh);
        assert_eq!(uploaded.sharding(), &sharding);
        assert_eq!(uploaded.r#type().memory(), memory);
        assert_eq!(uploaded.layout(), Some(&layout));
        for shard in uploaded.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap().kind().unwrap(), "pinned_host");
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        }
        assert_eq!(uploaded.copy_to_host().unwrap().elements::<f32>().unwrap(), values);

        // An existing leaf can change both options without changing its mesh or shard slices.
        let layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let target = ArrayPlacement::sharding(sharding.clone()).with_memory(Memory::Device).with_layout(layout.clone());
        let moved = session
            .place(uploaded, PlacementOptions::<_, _>::new(Some(target), Some(placement), None, None))
            .unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &sharding);
        assert_eq!(moved.r#type().memory(), Memory::Device);
        assert_eq!(moved.layout(), Some(&layout));
        for shard in moved.addressable_shards() {
            let buffer = shard.buffer().unwrap();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
            assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        }
        assert_eq!(moved.copy_to_host().unwrap().elements::<f32>().unwrap(), values);
    }

    #[test]
    fn test_session_place_pads_short_shardings_with_replicated_dimensions() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let logical_mesh = mesh.logical_mesh().clone();
        let split = Sharding::new(logical_mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_matrix = Sharding::new(
            logical_mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let vector = CoreArray::vector(vec![1i32, 2, 3, 4]).unwrap();
        let matrix = CoreArray::matrix(4, 3, (0..12).collect::<Vec<i32>>()).unwrap();
        let shard_elements = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| values_from_bytes::<i32>(&shard.copy_to_host_bytes().unwrap()))
                .collect::<Vec<_>>()
        };

        // One sharding shorter than the rank of a leaf leaves the omitted trailing dimensions replicated, so one
        // destination splits a vector and a matrix along their first dimension, and a rank zero sharding replicates
        // both. The resolved shardings have the rank of each leaf.
        let (placed_vector, placed_matrix) = session
            .place((vector.clone(), matrix.clone()), PlacementOptions::to(ArrayPlacement::sharding(split.clone())))
            .unwrap();
        assert_eq!(placed_vector.sharding(), &split);
        assert_eq!(shard_buffer_dimensions(&placed_vector), vec![vec![2], vec![2]]);
        assert_eq!(shard_elements(&placed_vector), vec![vec![1, 2], vec![3, 4]]);
        assert!(placed_vector.is_committed());
        assert_eq!(placed_matrix.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&placed_matrix), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_elements(&placed_matrix), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);
        assert!(placed_matrix.is_committed());
        let replicated = Sharding::replicated(logical_mesh.clone(), 0);
        let (replicated_vector, replicated_matrix) = session
            .place((vector.clone(), matrix.clone()), PlacementOptions::to(ArrayPlacement::sharding(replicated)))
            .unwrap();
        assert_eq!(replicated_vector.sharding(), &Sharding::replicated(logical_mesh.clone(), 1));
        assert_eq!(shard_buffer_dimensions(&replicated_vector), vec![vec![4], vec![4]]);
        assert_eq!(replicated_matrix.sharding(), &Sharding::replicated(logical_mesh.clone(), 2));
        assert_eq!(shard_buffer_dimensions(&replicated_matrix), vec![vec![4, 3], vec![4, 3]]);
        assert_eq!(shard_elements(&replicated_matrix), vec![(0..12).collect::<Vec<_>>(); 2]);

        // Arrays resolve short destinations and sources in the same way, also from a resolved placement.
        let placement = Placement::new(mesh.clone(), split.clone()).unwrap();
        let resplit = session
            .place(
                replicated_matrix.clone(),
                PlacementOptions::<Placement, ArrayPlacement>::new(
                    Some(placement.clone()),
                    Some(ArrayPlacement::sharding(Sharding::replicated(logical_mesh.clone(), 1))),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(resplit.mesh(), &mesh);
        assert_eq!(resplit.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&resplit), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_elements(&resplit), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);
        assert!(resplit.is_committed());
        let checked = session
            .place(
                resplit.clone(),
                PlacementOptions::<ArrayPlacement, Placement>::new(None, Some(placement), None, None),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), resplit.shards()[0].buffer().unwrap()));

        // A traced leaf resolves a short destination for its rank.
        let input_type = ArrayType::new_static(DataType::I32, [4, 3])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 2))
            .unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| {
                session.place(input, PlacementOptions::to(ArrayPlacement::sharding(split.clone()))).unwrap()
            },
            input_type,
            &session,
            mesh.clone(),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, replicated_matrix).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &split_matrix);
        assert_eq!(shard_buffer_dimensions(&output), vec![vec![2, 3], vec![2, 3]]);
        assert_eq!(shard_elements(&output), vec![(0..6).collect::<Vec<_>>(), (6..12).collect()]);

        // A sharding with more dimensions than a leaf is still rejected, with the error of the type rule of `place`
        // (refer to `test_session_place_reports_type_rule_error_for_destinations_of_higher_rank`).
        let error = session.place(vector, PlacementOptions::to(ArrayPlacement::sharding(split_matrix))).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` target sharding rank (2) does not match the input rank (1)");
    }

    #[test]
    fn test_session_place_reports_type_rule_error_for_destinations_of_higher_rank() {
        // A destination whose sharding has more dimensions than a leaf is rejected with the error of the type rule of
        // `place` for every kind of leaf, which a source reports as the error of a source.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let placement = || ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 3));
        let target = || PlacementOptions::to(placement().unwrap());
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let values = [1f32, 2., 3., 4.];
        let message = format!("`{PLACE_OPERATION_NAME}` target sharding rank (3) does not match the input rank (1)");

        let host = CoreArray::from_elements(r#type.clone(), &values).unwrap();
        assert_eq!(session.place(host, target()).map(|_| ()).map_err(|error| error.to_string()), Err(message.clone()));
        let array = Array::from_host_buffer(&session, r#type.clone(), None, None, values_to_bytes(&values)).unwrap();
        let error = session.place(array.clone(), target()).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), message);
        let traced = RefCell::new(None);
        let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
            |x: ArrayTracer| {
                traced.replace(Some(session.place(x.clone(), target()).map(|_| ()).map_err(|error| error.to_string())));
                x
            },
            r#type,
        );
        assert_eq!(traced.take(), Some(Err(message)));

        let source =
            PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(None, Some(placement().unwrap()), None, None);
        assert_eq!(
            session.place(array, source).map(|_| ()).map_err(|error| error.to_string()),
            Err(format!("`{PLACE_OPERATION_NAME}` source sharding rank (3) does not match the input rank (1)")),
        );
    }

    #[test]
    fn test_session_place_selects_default_memory_for_device_destinations() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]).with_memory(pinned),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(array.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        let is_in_default_memory = |array: &Array| {
            let buffer = array.shards()[0].buffer().unwrap();
            buffer.memory().unwrap() == buffer.device().unwrap().default_memory().unwrap()
        };

        // A bare device destination selects the default memory of that device, as JAX's `device_put` does, both on
        // another device and on the device that the array already lives on.
        for device in [devices[1], devices[0]] {
            let moved = session.place(array.clone(), PlacementOptions::to(device)).unwrap();
            assert_eq!(moved.mesh().devices(), [device]);
            assert_eq!(moved.r#type().memory(), Memory::Device);
            assert!(is_in_default_memory(&moved));
            assert_eq!(shard_values(&moved), vec![vec![1., 2.]]);
            let moved = array.to_placement(&session, device).unwrap();
            assert_eq!(moved.r#type().memory(), Memory::Device);
            assert!(is_in_default_memory(&moved));
        }

        // An explicit memory still applies, and a mesh destination without a memory keeps the memory of the array.
        let moved = session
            .place(array.clone(), PlacementOptions::to(ArrayPlacement::device(devices[1]).with_memory(pinned)))
            .unwrap();
        assert_eq!(moved.r#type().memory(), pinned);
        assert_eq!(moved.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let target = ArrayPlacement::mesh(target_mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap();
        let moved = session.place(array.clone(), PlacementOptions::to(target)).unwrap();
        assert_eq!(moved.mesh(), &target_mesh);
        assert_eq!(moved.r#type().memory(), pinned);

        // A host value placed on a bare device is uploaded to the default memory of that device.
        let uploaded = session.place(3f32, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(uploaded.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&uploaded));

        // A traced value placed on a bare device also selects the default memory of that device.
        let (value, _) = differentiate_at(array.clone())
            .vjp(|x| Ok(session.place(x, PlacementOptions::to(devices[1])).unwrap()))
            .unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        assert_eq!(value.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&value));

        // The core `Place` capability follows the same contract, eagerly and traced, because the type rule of a bare
        // device request selects that memory for every path.
        let moved = array.place(&devices[1], None).unwrap();
        assert_eq!(moved.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&moved));
        let (value, _) = differentiate_at(array.clone()).vjp(|x| x.place(&devices[1], None)).unwrap();
        assert_eq!(value.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&value));

        // A bare device source accepts the memory of an array on that device.
        let checked = session
            .place(
                array.clone(),
                PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(
                    None,
                    Some(ArrayPlacement::device(devices[0])),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        assert_eq!(checked.r#type().memory(), pinned);
    }

    #[test]
    fn test_session_place_moves_arrays_to_memory_destinations() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let memory_kinds = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().to_string())
                .collect::<Vec<_>>()
        };
        let is_in_default_memory = |array: &Array| {
            array.shards().iter().all(|shard| {
                let buffer = shard.buffer().unwrap();
                buffer.memory().unwrap() == buffer.device().unwrap().default_memory().unwrap()
            })
        };

        // A bare memory leaves an array that is split over two devices on its mesh with its sharding and its values,
        // and moves its buffers to that memory. The result is committed, as JAX commits the result of `device_put`
        // onto a memory space.
        let sharded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let moved = session.place(sharded, PlacementOptions::to(pinned)).unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &sharding);
        assert_eq!(moved.r#type().memory(), pinned);
        assert_eq!(memory_kinds(&moved), vec!["pinned_host"; 2]);
        assert_eq!(shard_buffer_dimensions(&moved), vec![vec![2], vec![2]]);
        assert_eq!(shard_values(&moved), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(moved.is_committed());

        // The default memory moves it back.
        let restored = session.place(moved, PlacementOptions::to(Memory::Device)).unwrap();
        assert_eq!(restored.mesh(), &mesh);
        assert_eq!(restored.sharding(), &sharding);
        assert_eq!(restored.r#type().memory(), Memory::Device);
        assert!(is_in_default_memory(&restored));
        assert_eq!(shard_values(&restored), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(restored.is_committed());

        // An uncommitted array on one device stays on that device and is committed, and `Array::to_placement` accepts
        // a bare memory as well.
        let single = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::defaults()).unwrap();
        assert!(!single.is_committed());
        for moved in [
            session.place(single.clone(), PlacementOptions::to(pinned)).unwrap(),
            single.to_placement(&session, pinned).unwrap(),
        ] {
            assert_eq!(moved.mesh(), single.mesh());
            assert_eq!(moved.sharding(), single.sharding());
            assert_eq!(moved.r#type().memory(), pinned);
            assert_eq!(memory_kinds(&moved), vec!["pinned_host"]);
            assert_eq!(shard_values(&moved), vec![vec![1., 2.]]);
            assert!(moved.is_committed());
        }
    }

    #[test]
    fn test_session_place_uploads_host_values_to_memory_destinations() {
        // A bare memory names no devices, so a host value is uploaded to the first addressable device, as without a
        // destination, in that memory, and the result is committed like the result of every destination.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let pinned = Memory::Host { pinned: true };
        let uploaded = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::to(pinned)).unwrap();
        assert_eq!(uploaded.mesh().devices(), [Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()]);
        assert_eq!(uploaded.r#type().memory(), pinned);
        assert_eq!(uploaded.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(uploaded.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2.]);
        assert!(uploaded.is_committed());
    }

    #[test]
    fn test_session_place_combines_memory_destinations_with_other_destinations() {
        // One branch of a tree can change only the memory of its leaves while another moves its leaves to a device.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let options = PlacementOptions::to((ArrayPlacement::from(pinned), ArrayPlacement::device(mesh.devices()[1])));
        let (in_place, on_device) = session.place((array.clone(), array), options).unwrap();
        assert_eq!(in_place.mesh(), &mesh);
        assert_eq!(in_place.sharding(), &sharding);
        assert_eq!(in_place.r#type().memory(), pinned);
        assert_eq!(shard_values(&in_place), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(on_device.mesh().devices(), [mesh.devices()[1]]);
        assert_eq!(on_device.r#type().memory(), Memory::Device);
        assert_eq!(shard_values(&on_device), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_validates_source_storage_on_every_path() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let array = session.place(CoreArray::vector(vec![1f32, 2.]).unwrap(), PlacementOptions::defaults()).unwrap();
        let device = array.mesh().devices()[0];
        let pinned = Memory::Host { pinned: true };
        let options =
            |source: &ArrayPlacement| PlacementOptions::<ArrayPlacement>::new(None, Some(source.clone()), None, None);

        // A source states the storage in which a value is expected, whatever its form, so a source in another memory
        // than the memory of the array is rejected with the error of the type rule of `place` for an array leaf, for a
        // traced leaf of a staged function, and for a traced leaf of a differentiated function that wraps the array.
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{}`",
            array.r#type(),
        );
        for source in [
            ArrayPlacement::from(array.placement().unwrap()).with_memory(pinned),
            ArrayPlacement::device(device).with_memory(pinned),
            ArrayPlacement::from(pinned),
        ] {
            let eager = session.place(array.clone(), options(&source));
            assert_eq!(eager.map(|_| ()).map_err(|error| error.to_string()), Err(error.clone()));
            // A staging context reports the error of the placement that it stages when tracing completes.
            let staged = RefCell::new(None);
            let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| {
                    staged.replace(session.place(x.clone(), options(&source)).err().map(|error| error.to_string()));
                    x
                },
                array.r#type().into_owned(),
            );
            assert_eq!(staged.take().or(traced.err().map(|error| error.to_string())), Some(error.clone()));
            let differentiated = RefCell::new(None);
            differentiate_at(array.clone())
                .jvp(array.clone(), |x| {
                    let result = session.place(x.clone(), options(&source));
                    differentiated.replace(result.err().map(|error| error.to_string()));
                    Ok(x)
                })
                .unwrap();
            assert_eq!(differentiated.take(), Some(error.clone()));
        }

        // A source in the memory of the array is accepted, and a host leaf ignores its source.
        let placed = session.place(array.clone(), options(&ArrayPlacement::from(Memory::Device))).unwrap();
        assert!(Arc::ptr_eq(placed.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
        let uploaded = session.place(3f32, options(&ArrayPlacement::from(pinned))).unwrap();
        assert_eq!(uploaded.r#type().memory(), Memory::Device);

        // An in-place source or destination that names neither a memory nor a layout says nothing, so it is rejected
        // for every kind of leaf before any leaf is placed.
        let error = session.place(array.clone(), options(&ArrayPlacement::in_place())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place source must name a memory or a layout");
        let host_error = session.place(3f32, options(&ArrayPlacement::in_place())).unwrap_err();
        assert_eq!(host_error.to_string(), error.to_string());
        let error = session.place(array, PlacementOptions::to(ArrayPlacement::in_place())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(error.to_string(), "`place` in-place target must name a memory or a layout");
    }

    #[test]
    fn test_session_place_broadcasts_root_placement_over_array_tuple() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let client_devices = client.addressable_devices().unwrap();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(client_devices[0].id().unwrap(), client_devices[0].process_index().unwrap())],
        )
        .unwrap();
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let first_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let first_source_array = Array::from_host_buffer(
            &session,
            first_source_type,
            source_mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 1.0, 2.0, 3.0]).as_slice(),
        )
        .unwrap();
        let second_source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(source_sharding)
            .unwrap();
        let second_source_array = Array::from_host_buffer(
            &session,
            second_source_type,
            source_mesh,
            None,
            values_to_bytes::<f32>(&[10.0, 11.0, 12.0, 13.0]).as_slice(),
        )
        .unwrap();

        let target_mesh = domain_mesh(client, "x", 2);
        let target_sharding =
            Sharding::new(target_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        let moved_arrays = session
            .place(
                (first_source_array, second_source_array),
                PlacementOptions::new(
                    Some(ArrayPlacement::mesh(target_mesh.clone(), target_sharding.clone()).unwrap()),
                    Option::<ArrayPlacement>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap();

        assert_eq!(
            moved_arrays.0.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(
            moved_arrays.1.sharding().visualize().unwrap().render(false),
            indoc! {"
                    ┌─────┬─────┐
                    │  0  │  1  │
                    └─────┴─────┘
                "}
            .trim_end()
            .to_string()
        );
        assert_eq!(shard_values(&moved_arrays.0), vec![vec![0.0, 1.0], vec![2.0, 3.0]]);
        assert_eq!(shard_values(&moved_arrays.1), vec![vec![10.0, 11.0], vec![12.0, 13.0]]);
    }

    #[test]
    fn test_session_place_repeats_root_options_over_large_inputs() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let array = session.place(2.5f32, PlacementOptions::to(devices[0])).unwrap();
        let array_count = 1_000;
        let input = vec![array.clone(); array_count];

        // Options at the root give every leaf what options that repeat them for every leaf give it.
        let root_options = PlacementOptions::new(Some(devices[1]), Some(devices[0]), Some(false), Some(Some(false)));
        let repeated_options = PlacementOptions::new(
            Some(vec![devices[1]; array_count]),
            Some(vec![devices[0]; array_count]),
            Some(vec![false; array_count]),
            Some(vec![Some(false); array_count]),
        );
        let placed = session.place(input.clone(), root_options).unwrap();
        let expected = session.place(input.clone(), repeated_options).unwrap();
        assert_eq!(placed.len(), array_count);
        for (placed, expected) in placed.iter().zip(&expected) {
            assert_eq!(placed.mesh().devices(), [devices[1]]);
            assert_eq!(placed.mesh(), expected.mesh());
            assert_eq!(placed.sharding(), expected.sharding());
            assert_eq!(placed.r#type(), expected.r#type());
            assert!(placed.is_committed());
            assert_eq!(shard_values(placed), vec![vec![2.5]]);
        }

        // The checks of root options that do not depend on paths still run, before and while each leaf is placed.
        let options =
            PlacementOptions::<ArrayPlacement, ArrayPlacement, _, _>::new(None, None, Some(true), Some(Some(true)));
        assert_eq!(
            session.place(input.clone(), options).map(|_| ()).map_err(|error| error.to_string()),
            Err("array `$[0]` cannot be donated while `may_alias` is `Some(true)`".to_string()),
        );
        assert!(matches!(
            session.place(input, PlacementOptions::defaults().with_source(Memory::Host { pinned: true })),
            Err(Error::Core(error)) if matches!(error.as_ref(), CoreError::Type(_)),
        ));
    }

    #[test]
    fn test_session_place_broadcasts_optional_placement_prefixes() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let values = vec![1f32, 2., 3., 4.];
        let host = CoreArray::vector(values.clone()).unwrap();

        // An explicit sharding uses the session mesh. The None prefix sends both leaves to the first local device.
        let (partitioned, [local, other_local]) = session
            .place(
                (host.clone(), [host.clone(), host]),
                PlacementOptions::<_, ArrayPlacement>::new(
                    Some((Some(ArrayPlacement::sharding(sharded.clone())), None)),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(partitioned.mesh(), &mesh);
        assert_eq!(partitioned.sharding(), &sharded);
        assert_eq!(shard_values(&partitioned), vec![vec![1., 2.], vec![3., 4.]]);
        for array in [&local, &other_local] {
            assert_eq!(array.mesh().devices(), &mesh.devices()[..1]);
            assert_eq!(shard_values(array), vec![values.clone()]);
        }

        // For existing arrays, None leaves preserve their destination or omit their source check independently.
        let (moved, kept) = session
            .place(
                (local, partitioned.clone()),
                PlacementOptions::<_, _>::new(
                    Some((Some(Placement::new(mesh.clone(), replicated.clone()).unwrap()), None)),
                    Some((None, Some(Placement::new(mesh.clone(), sharded.clone()).unwrap()))),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(moved.mesh(), &mesh);
        assert_eq!(moved.sharding(), &replicated);
        assert_eq!(shard_values(&moved), vec![values.clone(); 2]);
        assert_eq!(kept.mesh(), partitioned.mesh());
        assert_eq!(kept.sharding(), partitioned.sharding());
        for (kept_shard, source_shard) in kept.shards().iter().zip(partitioned.shards()) {
            assert!(Arc::ptr_eq(kept_shard.buffer().unwrap(), source_shard.buffer().unwrap()));
        }

        // A bare sharding is a placement on the session mesh, also as a prefix leaf.
        let partitioned = session
            .place(
                CoreArray::vector(values.clone()).unwrap(),
                PlacementOptions::<_, ArrayPlacement>::new(Some(Some(sharded.clone())), None, None, None),
            )
            .unwrap();
        assert_eq!(partitioned.mesh(), &mesh);
        assert_eq!(partitioned.sharding(), &sharded);
        assert_eq!(shard_values(&partitioned), vec![vec![1., 2.], vec![3., 4.]]);

        // A bare device is a placement on that device, also as a prefix leaf.
        let device = mesh.devices()[1];
        let placed = session
            .place(
                CoreArray::vector(values.clone()).unwrap(),
                PlacementOptions::<_, ArrayPlacement>::new(Some(Some(device)), None, None, None),
            )
            .unwrap();
        assert_eq!(placed.mesh().devices(), &[device]);
        assert_eq!(shard_values(&placed), vec![values]);
    }

    #[test]
    fn test_session_place_validates_option_prefixes_of_empty_input() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = ArrayPlacement::device(Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap());

        // Root option prefixes cover the input without leaves, which is returned as it is.
        let options = PlacementOptions::new(Some(device.clone()), Some(device.clone()), Some(true), Some(Some(false)));
        assert_eq!(session.place(Vec::<Array>::new(), options).unwrap().len(), 0);
        assert_eq!(session.place(Vec::<f32>::new(), PlacementOptions::defaults()).unwrap().len(), 0);
        assert_eq!(session.place(Vec::<f32>::new(), PlacementOptions::to(device.clone())).unwrap().len(), 0);

        // An option prefix with an element for an input without elements matches no node of that input.
        let options = PlacementOptions::<Vec<ArrayPlacement>>::new(Some(vec![device]), None, None, None);
        assert!(matches!(
            session.place(Vec::<f32>::new(), options),
            Err(Error::Core(error)) if matches!(
                error.as_ref(),
                CoreError::Parameter(ParameterError::UnusedParameters { paths: Some(paths) })
                    if paths == &vec!["$[0]".to_string()],
            ),
        ));
    }

    #[test]
    fn test_session_place_broadcasts_option_prefixes_over_empty_subtrees() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();

        // The first destination covers the empty vector, which places nothing, and the second one places the scalar.
        let (empty, scalar) = session
            .place(
                (Vec::<f32>::new(), 1f32),
                PlacementOptions::<_, ArrayPlacement>::new(
                    Some((ArrayPlacement::device(devices[0]), ArrayPlacement::device(devices[1]))),
                    None,
                    None,
                    None,
                ),
            )
            .unwrap();
        assert_eq!(empty.len(), 0);
        assert_eq!(scalar.mesh().devices(), [devices[1]]);
        assert!(scalar.is_committed());
        assert_eq!(scalar.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1.]);
    }

    #[test]
    fn test_session_place_validates_empty_nodes_of_option_prefixes() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let parameter_error = |result: Result<Vec<Vec<Array>>, Error>| match result {
            Err(Error::Core(error)) => match error.as_ref() {
                CoreError::Parameter(error) => error.clone(),
                error => panic!("expected a parameter error, got `{error}`"),
            },
            result => panic!("expected a parameter error, got `{result:?}`"),
        };
        let missing =
            |path: &str| ParameterError::MissingParameters { expected_count: 1, paths: Some(vec![path.to_string()]) };
        let unused = |path: &str| ParameterError::UnusedParameters { paths: Some(vec![path.to_string()]) };

        // An empty node of the input must be covered by an option leaf or matched by an empty option node.
        assert_eq!(
            parameter_error(session.place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]]]))),
            missing("$[1]"),
        );

        // An empty option node must match an empty node of the input at the same path. This holds for every option.
        let options = PlacementOptions::to(vec![vec![devices[1]], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::defaults().with_source(vec![vec![devices[0]], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::defaults().with_donate(vec![vec![false], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));
        let options = PlacementOptions::defaults().with_may_alias(vec![vec![None], Vec::new()]);
        assert_eq!(parameter_error(session.place(vec![vec![1f32]], options)), unused("$[1]"));

        // An empty option node does not match an input without that node, even when neither has leaves.
        assert_eq!(
            parameter_error(session.place(Vec::<Vec<f32>>::new(), PlacementOptions::to(vec![Vec::<Device>::new()]))),
            unused("$[0]"),
        );

        // Matching empty nodes on both sides are valid, and so is an option leaf over an empty node of the input.
        let placed = session
            .place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]], Vec::new()]))
            .unwrap();
        assert_eq!(placed[0][0].mesh().devices(), [devices[1]]);
        assert_eq!(placed[1].len(), 0);
        let placed = session
            .place(vec![vec![1f32], Vec::new()], PlacementOptions::to(vec![devices[1], devices[0]]))
            .unwrap();
        assert_eq!(placed[0][0].mesh().devices(), [devices[1]]);
        assert_eq!(placed[1].len(), 0);
        assert_eq!(session.place(Vec::<Vec<f32>>::new(), PlacementOptions::to(Vec::<Device>::new())).unwrap().len(), 0);

        // Nodes match by path, so an option tree of another container type with the same paths matches.
        let [placed, empty] = session
            .place([vec![1f32], Vec::new()], PlacementOptions::to(vec![vec![devices[1]], Vec::new()]))
            .unwrap();
        assert_eq!(placed[0].mesh().devices(), [devices[1]]);
        assert_eq!(empty.len(), 0);
    }

    #[test]
    fn test_session_place_donation_and_aliasing() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("source", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let replicated = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let sharded = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let source = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            source_mesh,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_source_buffer = |array: &Array| {
            array
                .shards()
                .iter()
                .map(|shard| Arc::ptr_eq(shard.buffer().unwrap(), source.shards()[0].buffer().unwrap()))
                .collect::<Vec<_>>()
        };
        let options = |target: Option<ArrayPlacement>, donate: bool, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<ArrayPlacement>::None, Some(donate), Some(may_alias))
        };

        // Without a new placement, the result is the leaf itself unless aliasing is forbidden.
        let unchanged = session.place(source.clone(), options(None, false, None)).unwrap();
        assert_eq!(shares_source_buffer(&unchanged), vec![true]);
        let copied = session.place(source.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), source.r#type());
        assert_eq!(copied.mesh(), source.mesh());
        assert_eq!(shares_source_buffer(&copied), vec![false]);
        assert_ne!(native_buffer_addresses(&copied), native_buffer_addresses(&source));
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A forbidden alias copies every shard within its own memory, so an array in host memory stays there.
        let host = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]).with_memory(Memory::Host { pinned: true }),
            source.mesh().clone(),
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let copied = session.place(host.clone(), options(None, false, Some(false))).unwrap();
        assert_eq!(copied.r#type(), host.r#type());
        assert!(!Arc::ptr_eq(copied.shards()[0].buffer().unwrap(), host.shards()[0].buffer().unwrap()));
        assert_ne!(native_buffer_addresses(&copied), native_buffer_addresses(&host));
        assert_eq!(copied.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(shard_values(&copied), vec![vec![1., 2., 3., 4.]]);

        // A shard that stays on its device shares its buffer unless aliasing is forbidden.
        let target = ArrayPlacement::mesh(mesh.clone(), replicated).unwrap();
        let placed = session.place(source.clone(), options(Some(target.clone()), false, None)).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![true, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);
        let placed = session.place(source.clone(), options(Some(target), false, Some(false))).unwrap();
        assert_eq!(shares_source_buffer(&placed), vec![false, false]);
        assert_eq!(shard_values(&placed), vec![vec![1., 2., 3., 4.]; 2]);

        // A donated leaf is consumed. This one shares its buffers with `source`, so the donation does not take effect
        // and `source` stays readable.
        let target = ArrayPlacement::mesh(mesh, sharded.clone()).unwrap();
        let donated = session.place(source.clone(), options(Some(target), true, None)).unwrap();
        assert_eq!(donated.sharding(), &sharded);
        assert_eq!(shard_values(&donated), vec![vec![1., 2.], vec![3., 4.]]);
        assert_eq!(shard_values(&source), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_commits_array_leaves_with_a_destination() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]);
        let uncommitted = Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let options = |target: Option<ArrayPlacement>, may_alias: Option<bool>| {
            PlacementOptions::new(target, Option::<ArrayPlacement>::None, Option::<bool>::None, Some(may_alias))
        };
        assert!(!uncommitted.is_committed());

        // Without a destination, an array leaf keeps its commitment, also when it is copied.
        assert!(!session.place(uncommitted.clone(), options(None, None)).unwrap().is_committed());
        assert!(!session.place(uncommitted.clone(), options(None, Some(false))).unwrap().is_committed());
        let committed = uncommitted.clone().with_committed(true);
        assert!(session.place(committed, options(None, Some(false))).unwrap().is_committed());

        // A destination commits the result, even one that leaves the leaf where it is.
        let device = uncommitted.mesh().devices()[0];
        let placed = session.place(uncommitted.clone(), options(Some(ArrayPlacement::device(device)), None)).unwrap();
        assert_eq!(placed.mesh(), uncommitted.mesh());
        assert!(placed.is_committed());
        let other_device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let placed = session.place(uncommitted, options(Some(ArrayPlacement::device(other_device)), None)).unwrap();
        assert_eq!(placed.mesh().devices(), [other_device]);
        assert!(placed.is_committed());
    }

    #[test]
    fn test_session_place_replicates_array_leaf_on_its_own_device() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(unreduced).unwrap();
        let array =
            Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();

        // A device names neither a mesh nor a sharding, so an array that already lives on that device keeps its mesh
        // but becomes an ordinary replicated array, without the pending reduction along its unreduced axis.
        let target = ArrayPlacement::device(device);
        let placed = session
            .place(array, PlacementOptions::<ArrayPlacement>::new(Some(target), None, None, None))
            .unwrap();
        assert_eq!(placed.mesh(), &mesh);
        assert_eq!(placed.sharding(), &Sharding::replicated(mesh.logical_mesh().clone(), 1));
        assert_eq!(shard_values(&placed.exp().unwrap()), vec![vec![1f32.exp(), 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_moves_array_on_single_device_mesh_onto_device() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let array =
            Array::from_host_buffer(&session, r#type.clone(), source_mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();

        // The array keeps its logical mesh over the device, and its pending reduction along `x`, which has size one,
        // is complete there, so `exp` applies to the result. That is the type that the type rule of a staged device
        // target records.
        let placed = session.place(array, PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(placed.mesh(), &target_mesh);
        assert_eq!(placed.sharding(), &replicated);
        assert_eq!(
            placed.r#type().into_owned(),
            CoreOperation::infer_output_types(&PlaceOperation::new(devices[1], None), &[r#type], &[])
                .unwrap()
                .remove(0),
        );
        assert_eq!(
            placed.r#type().as_ref(),
            &ArrayType::new_static(DataType::F32, [2]).with_sharding(replicated).unwrap()
        );
        assert!(placed.is_committed());
        assert_eq!(shard_values(&placed.exp().unwrap()), vec![vec![1f32.exp(), 2f32.exp()]]);

        // An ordinary array on one device keeps its logical mesh as well, whatever its axes are named (e.g., that of
        // the mesh that an upload without a mesh creates).
        let uploaded = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let placed = session.place(uploaded.clone(), PlacementOptions::to(devices[1])).unwrap();
        assert_eq!(placed.mesh(), &DeviceMesh::new(uploaded.mesh().logical_mesh().clone(), vec![devices[1]]).unwrap());
        assert_eq!(placed.sharding(), uploaded.sharding());
        assert_eq!(placed.r#type(), uploaded.r#type());
        assert_eq!(shard_values(&placed), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_copies_only_handle_of_leaf_that_may_not_alias() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 2);
        let upload = || {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let options = |target: Option<ArrayPlacement>| {
            PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, _>::new(target, None, None, Some(Some(false)))
        };

        // The leaf is the only handle of its buffer, which a forced copy still must not return, whether the placement
        // leaves the leaf where it is or keeps its shard on its device while adding a replica on another device.
        let replicated =
            ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        for (target, values) in [(None, vec![vec![1f32, 2.]]), (Some(replicated), vec![vec![1., 2.]; 2])] {
            let array = upload();
            let buffer = Arc::downgrade(array.shards()[0].buffer().unwrap());
            let placed = session.place(array, options(target)).unwrap();
            assert!(buffer.upgrade().is_none());
            assert_eq!(shard_values(&placed), values);
        }
    }

    #[test]
    fn test_session_place_rejects_donated_leaf_that_may_alias() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();

        // The donation and aliasing options of every leaf are checked before any leaf is placed, and the error names
        // the first leaf that is donated while it may alias.
        let error = session
            .place(
                (array.clone(), array.clone()),
                PlacementOptions::<ArrayPlacement, ArrayPlacement, _, _>::new(
                    None,
                    None,
                    Some((false, true)),
                    Some(Some(true)),
                ),
            )
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::DonatedArrayMayAlias { path } if path.to_string() == "$.1",
            ),
        ));
        assert_eq!(error.to_string(), "array `$.1` cannot be donated while `may_alias` is `Some(true)`");

        // Host leaves are rejected in the same way, even though they ignore both options otherwise.
        let error = session
            .place(
                1f32,
                PlacementOptions::<ArrayPlacement, ArrayPlacement, _, _>::new(None, None, Some(true), Some(Some(true))),
            )
            .unwrap_err();
        assert_eq!(error.to_string(), "array `$` cannot be donated while `may_alias` is `Some(true)`");
    }

    #[test]
    fn test_session_place_validates_destination_meshes_before_placing_any_input() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let options = |target: ArrayPlacement| {
            PlacementOptions::<_, _>::new(
                Some((None, Some(target))),
                Some((Some(ArrayPlacement::from(Memory::Host { pinned: true })), None)),
                None,
                None,
            )
        };

        // The first leaf is not in the memory of its expected source, which is only found when it is placed, whereas
        // the destination of the second leaf needs a default mesh that the session lacks, which is found before any
        // leaf is placed and so is the error that is returned.
        let error = session
            .place((array.clone(), array.clone()), options(ArrayPlacement::sharding(sharding)))
            .unwrap_err();
        assert!(matches!(error, Error::Session(SessionError::MissingMesh)));

        // A destination whose sharding has another topology than its mesh is also found before any leaf is placed.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("z", 4, MeshAxisType::Auto).unwrap()]).unwrap();
        let target = ArrayPlacement::new(ArrayPlacementTarget::Mesh {
            mesh: mesh.clone(),
            sharding: Sharding::replicated(other_mesh.clone(), 1),
        });
        let error = session.place((array.clone(), array.clone()), options(target)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("mesh mismatch; expected `{:?}` but got `{other_mesh:?}`", mesh.logical_mesh()),
        );

        // Any other error of a leaf is found when that leaf is placed, after the leaves before it.
        let target = ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let error = session.place((array.clone(), array), options(target)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
    }

    #[test]
    fn test_session_place_preserves_partially_addressable_array_when_target_is_absent() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let local_device = client.addressable_devices().unwrap().remove(0);
        let local_device_id = local_device.id().unwrap();
        let remote_device_id = local_device_id + 1;
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap(),
            vec![Device::new(local_device_id, local_device.process_index().unwrap()), Device::new(remote_device_id, 1)],
        )
        .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let local_source_buffer = client
            .buffer(values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(), BufferType::F32, [2u64], None, local_device, None)
            .unwrap();
        let source_array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let source_array = Array::from_addressable_buffers(
            &session,
            source_array_type,
            mesh.clone(),
            Some(sharding.clone()),
            vec![local_source_buffer],
        )
        .unwrap();

        let copied_array = session.place(source_array, PlacementOptions::defaults()).unwrap();
        let expected_visualization =
            format!("┌─────┬─────┐\n│{:^5}│{:^5}│\n└─────┴─────┘", local_device_id, remote_device_id);

        assert_eq!(copied_array.addressable_shards().count(), 1);
        assert_eq!(copied_array.sharding().visualize().unwrap().render(false), expected_visualization);
        let copied_shard_bytes = copied_array.device_shard(local_device_id).unwrap().copy_to_host_bytes().unwrap();
        assert_eq!(values_from_bytes::<f32>(copied_shard_bytes.as_slice()), vec![0.0, 1.0]);
        assert!(copied_array.device_shard(remote_device_id).unwrap().buffer().is_none());
    }

    #[test]
    fn test_session_place_treats_source_devices_as_informative_on_every_path() {
        // The devices that a source names are informative, as in JAX, which never compares them with where a value
        // lives. An array on the first device that is placed on the second one with a source on the third one is
        // therefore placed alike eagerly, by the replay of a staged program, and under forward and reverse mode
        // differentiation, and reverse mode sends the cotangent to the devices that the source names.
        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        assert_eq!(input.mesh().devices(), [devices[0]]);
        let options = || {
            PlacementOptions::new(
                Some(ArrayPlacement::device(devices[1])),
                Some(ArrayPlacement::device(devices[2])),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            )
        };
        let observe =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let expected = (vec![devices[1]], input.r#type().into_owned(), vec![vec![1., 2., 3., 4.]]);

        let eager = session.place(input.clone(), options()).unwrap();
        assert_eq!(observe(&eager), expected);

        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| session.place(x, options()).unwrap(), input.r#type().into_owned()).unwrap();
        let Value::Array(staged) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(observe(&staged), expected);

        let (primal, tangent) = differentiate_at(input.clone())
            .jvp(input.clone(), |x| Ok(session.place(x, options()).unwrap()))
            .unwrap();
        assert_eq!(observe(&primal), expected);
        assert_eq!(observe(&tangent), expected);

        let (primal, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
        assert_eq!(observe(&primal), expected);
        let seed = Array::from_host_buffer(
            &session,
            primal.r#type().into_owned(),
            primal.mesh().clone(),
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();
        let cotangent = pullback.apply_materialized(seed).unwrap();
        assert_eq!(
            observe(&cotangent),
            (vec![devices[2]], input.r#type().cotangent().unwrap(), vec![vec![5., 6., 7., 8.]]),
        );
    }

    #[test]
    fn test_session_place_validates_source_storage() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 2);
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 2]).with_layout(layout.clone());
        let source = Placement::new(mesh.clone(), sharding).unwrap();
        let array =
            Array::from_host_buffer(&session, r#type, mesh, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap();
        let place = |source| {
            session.place(
                array.clone(),
                PlacementOptions::<ArrayPlacement, Placement>::new(None, Some(source), None, None),
            )
        };
        let checked = place(source.clone().with_memory(Memory::Device).with_layout(layout.clone())).unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));

        // The memory and the layout that the array declares are those of its type, so they are checked by the type
        // rule of `place`, as they are on every other path (refer to
        // `test_session_place_validates_source_storage_on_every_path`).
        let memory = Memory::Host { pinned: true };
        let error = place(source.clone().with_memory(memory)).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!(
                "`{PLACE_OPERATION_NAME}` source memory `Host[Pinned]` does not match the input type `{}`",
                array.r#type(),
            ),
        );

        let other_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let error = place(source.with_layout(other_layout.clone())).unwrap_err();
        assert!(matches!(&error, Error::Core(error) if matches!(error.as_ref(), CoreError::Type(_))));
        assert_eq!(
            error.to_string(),
            format!(
                "`{PLACE_OPERATION_NAME}` source layout `{other_layout}` does not match the input type `{}`",
                array.r#type(),
            ),
        );
    }

    #[test]
    fn test_session_place_validates_source_layout_against_type_on_every_path() {
        // Records the error of a placement inside a transformed or compiled function, whose error type is not the error
        // type of `Session::place`, so that the path reports that error rather than the one that wraps it.
        fn record<T>(error: &RefCell<Option<String>>, result: Result<T, Error>) -> Result<T, ProgramError> {
            result.map_err(|placement_error| {
                error.replace(Some(placement_error.to_string()));
                ProgramError::InvalidArgument { message: placement_error.to_string() }
            })
        }

        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap();
        let row_major = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let rank_1 = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        let r#type = ArrayType::new_static(DataType::F32, [2, 3]);
        let values = [1f32, 2., 3., 4., 5., 6.];
        let options = |layout: &Layout| {
            PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(
                None,
                Some(ArrayPlacement::in_place().with_layout(layout.clone())),
                None,
                None,
            )
        };
        // Two batch items with the values of `array`, packed along a new leading dimension, whose type, like that of
        // every batch item of a packed value, declares no layout.
        let packed = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2, 3]),
            mesh.clone(),
            None,
            values_to_bytes(&values.repeat(2)),
        )
        .unwrap();

        // Places `array` with the source layout `layout` eagerly, by the replay of a staged program, in a compiled
        // function, under forward and reverse mode differentiation, and for each batch item of `packed` under batching,
        // and returns the values (of one batch item for batching) or the error of each path.
        let place_on_every_path = |array: &Array, layout: &Layout| {
            let error = RefCell::new(None::<String>);
            let observe = |result: Result<Array, String>| {
                result.map(|array| array.copy_to_host().unwrap().elements::<f32>().unwrap())
            };
            let eager = observe(session.place(array.clone(), options(layout)).map_err(|error| error.to_string()));
            let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| record(&error, session.place(x.clone(), options(layout))).unwrap_or(x),
                array.r#type().into_owned(),
            );
            let staged = match (error.take(), traced) {
                (Some(message), _) => Err(message),
                (None, Err(error)) => Err(error.to_string()),
                (None, Ok(traced)) => {
                    match traced.program().interpret_in_context(&session, Value::Array(array.clone())) {
                        Ok(Value::Array(output)) => Ok(output),
                        Ok(_) => Err("placing an array must produce an array".to_string()),
                        Err(error) => Err(error.to_string()),
                    }
                }
            };
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |x: CompileTracer| record(&error, session.place(x.clone(), options(layout))).unwrap_or(x),
                array.r#type().into_owned(),
                &session,
                mesh.clone(),
            );
            let compiled = match (error.take(), compiled) {
                (Some(message), _) => Err(message),
                (None, Err(error)) => Err(error.to_string()),
                (None, Ok(compiled)) => {
                    compiled.executable().execute(&session, array.clone()).map_err(|error| error.to_string())
                }
            };
            let jvp = differentiate_at(array.clone())
                .jvp(array.clone(), |x| record(&error, session.place(x, options(layout))))
                .map(|(primal, _)| primal)
                .map_err(|other| error.take().unwrap_or(other.to_string()));
            let vjp = differentiate_at(array.clone())
                .vjp(|x| record(&error, session.place(x, options(layout))))
                .map(|(primal, _)| primal)
                .map_err(|other| error.take().unwrap_or(other.to_string()));
            let batched: Result<Array, _> = batch(
                |x| record(&error, session.place(x, options(layout))),
                packed.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            );
            let batched = observe(batched.map_err(|other| error.take().unwrap_or(other.to_string()))).map(|values| {
                let (first, second) = values.split_at(values.len() / 2);
                assert_eq!(first, second);
                first.to_vec()
            });
            [eager, observe(staged), observe(compiled), observe(jvp), observe(vjp), batched]
        };

        // A source is checked against the type of a value only, so an expected layout is accepted on every path for an
        // array whose type declares no layout, whatever the physical layouts of its buffers are (here column-major).
        let buffers = client
            .addressable_devices()
            .unwrap()
            .into_iter()
            .map(|device| {
                let memory = device.default_memory().unwrap();
                let layout = Some(column_major.to_pjrt().unwrap());
                client.buffer(values_to_bytes(&values).as_slice(), BufferType::F32, [2u64, 3], None, memory, layout)
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap();
        let undeclared =
            Array::from_addressable_buffers(&session, r#type.clone(), mesh.clone(), None, buffers).unwrap();
        assert_eq!(undeclared.layout(), None);
        assert_eq!(place_on_every_path(&undeclared, &row_major), [0; 6].map(|_| Ok(values.to_vec())));

        // An expected layout that does not address the shape of the value is rejected on every path with the error of
        // the type rule of `place`, which validates the source of a batched placement against a batch item before the
        // batching rule drops its layout.
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 1 does not match array rank 2"
        );
        assert_eq!(place_on_every_path(&undeclared, &rank_1), [0; 6].map(|_| Err(error.clone())));

        // An expected layout that contradicts the layout that the type of an array declares is rejected on every path
        // with the error of the type rule of `place`. The batch items of `packed` declare no layout, so the batched
        // placement accepts it.
        let declared = Array::from_host_buffer(
            &session,
            r#type.with_layout(column_major.clone()),
            mesh.clone(),
            None,
            values_to_bytes(&values),
        )
        .unwrap();
        let error = format!(
            "`{PLACE_OPERATION_NAME}` source layout `{row_major}` does not match the input type `{}`",
            declared.r#type(),
        );
        assert_eq!(
            place_on_every_path(&declared, &row_major),
            [
                Err(error.clone()),
                Err(error.clone()),
                Err(error.clone()),
                Err(error.clone()),
                Err(error),
                Ok(values.to_vec())
            ]
        );
    }

    #[test]
    fn test_session_place_validates_source_layout_against_shape_of_array_without_buffers() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let array = Array::zero(&session, ArrayType::new_static(DataType::Zero, [3]), None, mesh.clone()).unwrap();
        let source = Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let place = |source: ArrayPlacement| {
            session
                .place(array.clone(), PlacementOptions::defaults().with_source(source))
                .map(|placed| placed.r#type().into_owned())
        };

        // A rank-1 array of the zero data type has no buffers, and an expected layout is checked against its shape by
        // the type rule of `place`, as for every other value, whether the source is a placement or a device.
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let message = format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: tiled layout rank 2 does not match array rank 1",
        );
        assert_eq!(
            place(ArrayPlacement::from(source.clone().with_layout(rank_2.clone()))).map_err(|error| error.to_string()),
            Err(message.clone()),
        );
        assert_eq!(
            place(ArrayPlacement::device(mesh.devices()[0]).with_layout(rank_2)).map_err(|error| error.to_string()),
            Err(message),
        );
        let rank_1 = Layout::Tiled(TiledLayout::new(vec![0], vec![]));
        assert_eq!(place(ArrayPlacement::from(source.with_layout(rank_1))).unwrap(), array.r#type().into_owned());
    }

    #[test]
    fn test_session_place_accepts_device_source_for_array_on_that_device() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();

        // An eager upload without a mesh lives on the canonical mesh of one device, as a device destination does.
        let array = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2]),
            None,
            None,
            values_to_bytes(&[1f32, 2.]),
        )
        .unwrap();
        assert_eq!(array.mesh().devices(), [device]);
        let checked = session
            .place(
                array.clone(),
                PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(
                    None,
                    Some(ArrayPlacement::device(device)),
                    None,
                    None,
                ),
            )
            .unwrap();
        assert!(Arc::ptr_eq(checked.shards()[0].buffer().unwrap(), array.shards()[0].buffer().unwrap()));
    }

    #[test]
    fn test_session_place_returns_arrays_of_receiving_session() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let meshed_session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2.])).unwrap()
        };
        let options =
            |target: Option<ArrayPlacement>| PlacementOptions::<ArrayPlacement>::new(target, None, None, None);

        // An array of another session of the same client comes back owned by the receiving session, which has a
        // default mesh that the session of the input lacks, whether the placement keeps the array where it is or not.
        let array = upload(&session);
        for target in [None, Some(ArrayPlacement::device(device))] {
            let placed = meshed_session.place(array.clone(), options(target)).unwrap();
            assert!(placed.session().is_same_session(&meshed_session));
            assert_eq!(placed.session().device_mesh().unwrap(), &mesh);
        }

        // An array of another client is rejected even when it lives on a device with the same ID.
        let other_array = upload(&Session::builder().client(other_cpu_client(1)).build().unwrap());
        assert_eq!(other_array.mesh().devices(), [device]);
        for target in [None, Some(ArrayPlacement::device(device))] {
            assert!(matches!(
                session.place(other_array.clone(), options(target)),
                Err(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            ));
        }
    }

    #[test]
    fn test_session_place_rejects_traced_leaves_over_arrays_of_another_client() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let other_client = other_cpu_client(2);
        let other_session = Session::builder().client(other_client).build().unwrap();
        let other_device = Device::from_pjrt(&other_client.addressable_devices().unwrap()[1]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let other_tangent = Array::from_host_buffer(
            &other_session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[5f32, 6., 7., 8.]),
        )
        .unwrap();
        let is_rejected = |error: &Option<Error>| {
            matches!(
                error,
                Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "cannot associate an array with a session of a different PJRT client",
            )
        };

        // A traced leaf over arrays of another client is rejected like those arrays are, with or without a target,
        // because the devices of the receiving session that a target names are not the devices of those arrays.
        for target in [None, Some(ArrayPlacement::device(other_device))] {
            let options = || PlacementOptions::<ArrayPlacement>::new(target.clone(), None, None, None);

            // A batched leaf wraps the packed array.
            let mut error = None;
            let _: Array = batch(
                |x| match other_session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                },
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert!(is_rejected(&error));

            // A differentiated leaf wraps the primal array through a partially evaluated value.
            let mut error = None;
            let _ = differentiate_at(input.clone())
                .vjp(|x| match other_session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));

            // A differentiated leaf also wraps its tangent, which is checked like its primal, so a tangent of another
            // client is rejected even when the primal is an array of the receiving session.
            let mut error = None;
            let _: (Array, Array) = differentiate_at(input.clone())
                .jvp(other_tangent.clone(), |x| match session.place(x.clone(), options()) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));
        }

        // A staged leaf wraps no arrays, so any session can place it.
        let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
            |input: ArrayTracer| other_session.place(input, PlacementOptions::defaults()).unwrap(),
            ArrayType::new_static(DataType::F32, [2]),
        );
        assert!(traced.is_ok());
    }

    #[test]
    fn test_session_place_rejects_traced_leaves_over_arrays_of_another_session() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let other_session = Session::builder().client(client).build().unwrap();
        let mesh = domain_mesh(client, "x", 1);
        let device = mesh.devices()[0];
        let upload = |session: &Session| {
            let r#type = ArrayType::new_static(DataType::F32, [2, 2]);
            Array::from_host_buffer(session, r#type, None, None, values_to_bytes(&[1f32, 2., 3., 4.])).unwrap()
        };
        let options =
            |target: Option<ArrayPlacement>| PlacementOptions::<ArrayPlacement>::new(target, None, None, None);
        let is_rejected = |error: &Option<Error>| {
            matches!(
                error,
                Some(Error::Pjrt(ryft_pjrt::Error::InvalidArgument { message, .. }))
                    if message == "cannot place a traced value over arrays of another session",
            )
        };

        // A traced leaf places its arrays with the session that they belong to, which cannot become the receiving
        // session, so a traced leaf over arrays of another session of the same client is rejected, with or without
        // a target.
        let input = upload(&other_session);
        for target in [None, Some(ArrayPlacement::device(device))] {
            let mut error = None;
            let _: Array = batch(
                |x| match session.place(x.clone(), options(target.clone())) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                },
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert!(is_rejected(&error));

            let mut error = None;
            let _ = differentiate_at(input.clone())
                .vjp(|x| match session.place(x.clone(), options(target.clone())) {
                    Ok(output) => Ok(output),
                    Err(place_error) => {
                        error = Some(place_error);
                        Ok(x)
                    }
                })
                .unwrap();
            assert!(is_rejected(&error));
        }

        // Another handle of the session of the arrays (e.g., one with a default mesh) places the leaf, whose result
        // stays attached to the handle of those arrays.
        let input = upload(&session);
        let meshed_session = session.with_device_mesh(mesh);
        let output: Array = batch(
            |x| Ok(meshed_session.place(x, options(Some(ArrayPlacement::device(device)))).unwrap()),
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert!(output.session().is_same_session(&session));
        assert!(output.session().device_mesh().is_err());
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_compiled_functions() {
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [4])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();

        // The traced input stays traced, and its placement on the mesh of the compiled function is staged in it.
        let target = ArrayPlacement::sharding(sharding.clone());
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| {
                session
                    .place(input, PlacementOptions::<ArrayPlacement>::new(Some(target), None, None, None))
                    .unwrap()
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_places_traced_values_in_place_in_compiled_functions() {
        // A traced input of a function that is compiled for a mesh of two devices has no concrete placement, and its
        // type does not say how it is split along the auto axis `x`, which only the boundary of the compiled function
        // states, so the traced function cannot name its placement, and a bare memory stages an in-place target. The
        // result lives on the devices of the input, with the distribution of the input, in that memory.
        let client = cpu_client(2);
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let session = Session::builder().client(client).build().unwrap();
        let pinned = Memory::Host { pinned: true };
        let input_type = ArrayType::new_static(DataType::F32, [4]);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile_with_options(
            |input: CompileTracer| session.place(input, PlacementOptions::to(pinned)).unwrap(),
            input_type.clone(),
            &session,
            Options::new(mesh.clone()).with_input_shardings(vec![sharding.clone()]),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}] .
                let %1:f32[4][sharding={mesh<['x'=2:auto]>, [{}]}]@Host[Pinned] = place [\
                    target=in_place(memory=Host[Pinned])] %0
                in (%1)"
            },
        );
        let input = Array::from_host_buffer(
            &session,
            input_type,
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(output.r#type().memory(), pinned);
        for shard in output.shards() {
            assert_eq!(shard.buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
        }
        assert_eq!(shard_values(&output), vec![vec![1., 2.], vec![3., 4.]]);
        assert!(output.is_committed());
    }

    #[test]
    fn test_session_place_places_traced_leaves_on_sole_device_of_compiled_mesh() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let input_type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let session = Session::builder().client(client).build().unwrap();

        // The devices of a traced input are not known while it is traced, so a bare device stages a device target whose
        // type rule selects the default memory of the device, which keeps the sharding of the input on the mesh of the
        // function. That does not depend on the default mesh of the session, which here is either absent or another
        // mesh.
        for session in [session.clone(), session.with_device_mesh(two_device_mesh(client, MeshAxisType::Explicit))] {
            let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
                |input: CompileTracer| session.place(input, PlacementOptions::to(devices[1])).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            assert_eq!(
                compiled.source_program().program().to_string(),
                indoc! {"
                    lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                    let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(1)] %0
                    in (%1)"
                },
            );

            // The output lives on that device in its default memory, and it is committed, as an explicit device
            // commits an array eagerly.
            let input =
                Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                    .unwrap();
            let output = compiled.executable().execute(&session, input).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.r#type().memory(), Memory::Device);
            assert!(output.is_committed());
            let buffer = output.shards()[0].buffer().unwrap().clone();
            assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
            assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
        }
    }

    #[test]
    fn test_session_place_completes_pending_reduction_of_traced_values_on_sole_device_of_compiled_mesh() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();

        // The pending reduction along `x`, which has size one, is complete on the device, so the staged device target
        // records a type without it, `exp` applies to that value, and the compiled function places its output as a
        // direct placement of the input array on the device does.
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(devices[1])).unwrap().exp().unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}], unreduced={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(1)] %0
                    %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = exp %1
                in (%2)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let output = compiled.executable().execute(&session, input.clone()).unwrap();
        let direct = session.place(input, PlacementOptions::to(devices[1])).unwrap().exp().unwrap();
        assert_eq!(
            (output.mesh(), output.sharding(), output.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!((output.mesh(), output.sharding()), (&mesh, &replicated));
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1f32.exp(), 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_combines_traced_values_on_sole_device_with_other_values() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();

        // The placed value keeps the sharding that the input is tracked with over the mesh of the function, so it can
        // be added to another input of that function.
        let compiled: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
            |(x, y): (CompileTracer, CompileTracer)| {
                session.place(x, PlacementOptions::to(device)).unwrap().add(&y).unwrap()
            },
            (input_type.clone(), input_type.clone()),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], \
                    %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [target=device(0)] %0
                    %3:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = add %2 %1
                in (%3)"
            },
        );
        let x = Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
            .unwrap();
        let y =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[3f32, 4.])).unwrap();
        let output = compiled.executable().execute(&session, (x, y)).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![4., 6.]]);
    }

    #[test]
    fn test_session_place_combines_gathered_values_with_uploads_on_their_device() {
        // A value that a device destination gathers lives on the canonical logical mesh of one device, which is also
        // the mesh of an upload without a mesh, so the two combine on that device eagerly and in a staged program
        // that is replayed on arrays, as values with `SingleDeviceSharding`s on one device do in JAX.
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let device = mesh.devices()[0];
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let split_type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding).unwrap();
        let split =
            Array::from_host_buffer(&session, split_type.clone(), mesh, None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let upload = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4]),
            None,
            None,
            values_to_bytes(&[1f32, 1., 1., 1.]),
        )
        .unwrap();
        let canonical_mesh = DeviceMesh::new(LogicalMesh::single_device(), vec![device]).unwrap();
        assert_eq!(upload.mesh(), &canonical_mesh);

        let gathered = session.place(split.clone(), PlacementOptions::to(device)).unwrap();
        assert_eq!(gathered.mesh(), &canonical_mesh);
        let sum = gathered.add(&upload).unwrap();
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));
        let placed_upload = session.place(upload.clone(), PlacementOptions::to(device)).unwrap();
        let sum = gathered.add(&placed_upload).unwrap();
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));

        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(x, y): (ArrayTracer, ArrayTracer)| {
                let x = session.place(x, PlacementOptions::to(device)).unwrap();
                let y = session.place(y, PlacementOptions::to(device)).unwrap();
                x.add(&y).unwrap()
            },
            (split_type, upload.r#type().into_owned()),
        )
        .unwrap();
        let inputs = (Value::Array(split), Value::Array(upload));
        let Value::Array(sum) = traced.program().interpret_in_context(&session, inputs).unwrap() else {
            panic!("adding two arrays must produce an array");
        };
        assert_eq!((sum.mesh(), shard_values(&sum)), (&canonical_mesh, vec![vec![2., 3., 4., 5.]]));
    }

    #[test]
    fn test_session_place_places_traced_values_on_sole_device_through_every_staging_entry_point() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh, 1))
            .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let input =
            Array::from_host_buffer(&session, input_type.clone(), mesh.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let memory = Memory::Host { pinned: true };
        let target = ArrayPlacement::device(device).with_memory(memory);

        // A function that is staged and compiled later traces its input without a concrete placement as well.
        let staged: StagedFunction<ArrayType, ArrayType> = stage(
            |input: CompileTracer| session.place(input, PlacementOptions::to(target.clone())).unwrap(),
            input_type.clone(),
            Options::new(mesh.clone()),
        )
        .unwrap();
        let output = session.compile_staged_function(staged).unwrap().executable().execute(&session, input.clone());
        let output = output.unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // So does a function that is dispatched with `jit`, which traces it on its first call.
        let function: CompiledFunctionDispatcher<_, (), ArrayType, ArrayType> = jit(
            |_, input: CompileTracer| Ok(session.place(input, PlacementOptions::to(target.clone())).unwrap()),
            &session,
            mesh.clone(),
        );
        let output = function.execute((), input.clone()).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);

        // A stateful function traces values of the array IR type universe, which `Session::place` does not accept,
        // so it stages the device target through the `PlaceDispatch` primitive of `Place` directly.
        let device_target = PlacementRequest::Device { device, memory: Some(memory), layout: None };
        let compiled = compile_statefully::<_, Type, Type>(
            |input| {
                Ok(<StatefulCompileTracer as CoreValue>::Dispatch::place(&input, Some(&device_target), None, true)?)
            },
            Type::Array(input_type),
            &session,
            Options::new(mesh.clone()),
        )
        .unwrap();
        let Value::Array(output) = compiled.execute_statefully_and_wait(&session, Value::Array(input)).unwrap() else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert!(output.is_committed());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_rejects_traced_values_on_devices_outside_compiled_mesh() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]);
        let place_error = |device: Device, mesh: &DeviceMesh| {
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |input: CompileTracer| session.place(input, PlacementOptions::to(device)).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            );
            compiled.err().unwrap()
        };

        // A device that is not the device of the mesh of the function is rejected.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let error = place_error(devices[1], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[1] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}] of mesh \
                 `{}`",
                devices[1].id(),
                devices[0].id(),
                mesh.logical_mesh(),
            ),
        );

        // So is a device of a mesh with more than one device, because the value is not placed on that device alone.
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let error = place_error(devices[0], &mesh);
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::DeviceOutsideProgramMesh { device, mesh: program_mesh })
                if device == &devices[0] && program_mesh == &mesh,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "`place` cannot move a value to the device {} inside a program compiled for the devices [{}, {}] of \
                 mesh `{}`",
                devices[0].id(),
                devices[0].id(),
                devices[1].id(),
                mesh.logical_mesh(),
            ),
        );

        // A device source of a value without a tracked sharding is the destination of the cotangent in reverse mode,
        // so it is rejected like the same device target when the forward function is compiled, with or without a
        // target, rather than only when the function is differentiated.
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        for target in [Some(ArrayPlacement::from(Memory::Device)), None] {
            let options = || {
                PlacementOptions::new(
                    target.clone(),
                    Some(ArrayPlacement::device(devices[0])),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                |input: CompileTracer| session.place(input, options()).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            );
            let error = compiled.err().unwrap();
            assert!(matches!(
                &error,
                Error::Lowering(LoweringError::UnsupportedOperation { operation, .. })
                    if operation == PLACE_OPERATION_NAME,
            ));
            assert_eq!(
                error.to_string(),
                format!(
                    "cannot lower `place`: its source names the device {}, which is not the only device of the mesh \
                     `{}` over the devices [{}, {}] that the program is compiled for, and transposition would return \
                     the cotangent to it",
                    devices[0].id(),
                    mesh.logical_mesh(),
                    devices[0].id(),
                    devices[1].id(),
                ),
            );
        }

        // So is a resolved source on devices outside the program mesh, in the forward function and in its reverse
        // mode derivative inside the same compiled function, which would return the cotangent to those devices.
        let first_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let second_mesh = DeviceMesh::new(first_mesh.logical_mesh().clone(), vec![devices[1]]).unwrap();
        let source =
            ArrayPlacement::mesh(second_mesh.clone(), Sharding::replicated(second_mesh.logical_mesh().clone(), 1))
                .unwrap();
        let options = || {
            PlacementOptions::new(
                Some(ArrayPlacement::from(Memory::Device)),
                Some(source.clone()),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            )
        };
        let reverse: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
            |(x, seed): (CompileTracer, CompileTracer)| {
                let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(seed).unwrap()
            },
            (input_type.clone(), input_type.clone()),
            &session,
            first_mesh.clone(),
        );
        let forward: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
            |input: CompileTracer| session.place(input, options()).unwrap(),
            input_type.clone(),
            &session,
            first_mesh.clone(),
        );
        let error = forward.err().unwrap();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::UnsupportedOperation { operation, .. }) if operation == PLACE_OPERATION_NAME,
        ));
        assert_eq!(
            error.to_string(),
            format!(
                "cannot lower `place`: its source names the devices [{}] of mesh `{}`, which are not the devices [{}] \
                 of the mesh `{}` that the program is compiled for, and transposition would return the cotangent to \
                 them",
                devices[1].id(),
                second_mesh.logical_mesh(),
                devices[0].id(),
                first_mesh.logical_mesh(),
            ),
        );
        // The reverse mode program keeps only the transposed placement, whose target is that source.
        let error = reverse.err().unwrap();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::PlacementOutsideProgramMesh { target, mesh: program_mesh })
                if target == &second_mesh && program_mesh == &first_mesh,
        ));
    }

    #[test]
    fn test_session_place_rejects_split_sources_of_traced_values_in_bounded_dynamic_compiled_functions() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Auto);
        let extent = DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap());
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![extent.into()]));
        let place_onto = |target: ArrayPlacement, source: Option<ArrayPlacement>| {
            let options =
                PlacementOptions::new(Some(target), source, Option::<bool>::None, Option::<Option<bool>>::None);
            compile::<_, ArrayType, ArrayType>(
                |input: CompileTracer| session.place(input, options.clone()).unwrap(),
                input_type.clone(),
                &session,
                mesh.clone(),
            )
        };
        let place = |sharding: Sharding| {
            place_onto(
                ArrayPlacement::from(Memory::Device),
                Some(ArrayPlacement::mesh(mesh.clone(), sharding).unwrap()),
            )
        };

        // Bounded dynamic programs run one fully replicated replica per device, so a split target would be ignored, and
        // transposition turns a source into the target of the cotangent, so a split source is rejected like it.
        let error = place(Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .err()
            .unwrap();
        assert!(matches!(
            &error,
            Error::Lowering(LoweringError::UnsupportedOperation { operation, .. }) if operation == PLACE_OPERATION_NAME,
        ));
        assert_eq!(
            error.to_string(),
            "cannot lower `place`: bounded-dynamic multi-device programs currently require fully replicated placement \
             sources because transposition turns them into the targets of cotangents and Shardy rejects dynamic \
             tensors",
        );

        // A replicated source is what every replica holds anyway, and so is a source or a target that leaves a dimension
        // unconstrained, which an array stores replicated, as the unconstrained sharding override of a call is.
        assert!(place(Sharding::replicated(mesh.logical_mesh().clone(), 1)).is_ok());
        let unconstrained = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        assert!(place(unconstrained.clone()).is_ok());
        assert!(place_onto(ArrayPlacement::mesh(mesh.clone(), unconstrained).unwrap(), None).is_ok());
    }

    #[test]
    fn test_session_place_places_traced_values_with_sources_in_compiled_mesh() {
        let client = cpu_client(2);
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let session = Session::builder().client(client).build().unwrap();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]);
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values)).unwrap()
        };

        // A device source on the sole device of the program mesh, and a resolved source over the program mesh, are
        // accepted in the forward function and in its reverse mode derivative inside the same compiled function, whose
        // transposed placement moves the cotangent onto those devices.
        let sources = [
            ArrayPlacement::device(devices[1]),
            ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(logical_mesh, 1)).unwrap(),
        ];
        for source in sources {
            let options = || {
                PlacementOptions::new(
                    Some(ArrayPlacement::from(Memory::Device)),
                    Some(source.clone()),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let forward: CompiledFunction<ArrayType, ArrayType> = compile(
                |x: CompileTracer| session.place(x, options()).unwrap(),
                r#type.clone(),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let output = forward.executable().execute(&session, upload(&[1., 2., 3., 4.])).unwrap();
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);

            let reverse: CompiledFunction<(ArrayType, ArrayType), ArrayType> = compile(
                |(x, seed): (CompileTracer, CompileTracer)| {
                    let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                    pullback.apply_materialized(seed).unwrap()
                },
                (r#type.clone(), r#type.clone()),
                &session,
                mesh.clone(),
            )
            .unwrap();
            let cotangent = reverse
                .executable()
                .execute(&session, (upload(&[1., 2., 3., 4.]), upload(&[5., 6., 7., 8.])))
                .unwrap();
            assert_eq!(cotangent.mesh(), &mesh);
            assert_eq!(shard_values(&cotangent), vec![vec![5., 6., 7., 8.]]);
        }
    }

    #[test]
    fn test_session_place_applies_memory_and_layout_of_devices_to_traced_values_in_compiled_functions() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2, 2]);
        let memory = Memory::Host { pinned: true };
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let target = ArrayPlacement::device(device).with_memory(memory).with_layout(layout.clone());
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| session.place(input, PlacementOptions::to(target)).unwrap(),
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2, 2] .
                let %1:f32[2, 2][layout=tiled{0,1}]@Host[Pinned] = place [\
                    target=device(0, memory=Host[Pinned], layout=tiled{0,1})] %0
                in (%1)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
                .unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.r#type().memory(), memory);
        assert_eq!(output.layout(), Some(&layout));
        let buffer = output.shards()[0].buffer().unwrap().clone();
        assert_eq!(buffer.memory().unwrap().kind().unwrap(), "pinned_host");
        assert_eq!(buffer.specification().unwrap().layout, layout.to_pjrt().unwrap());
        assert!(output.is_committed());
        assert_eq!(output.copy_to_host().unwrap().elements::<f32>().unwrap(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let source_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();

        // The primal moves to the target device, and the pullback moves the cotangent back to the source mesh. The
        // donation option does not apply to the traced leaf, and a forced copy of a moved primal is that move itself.
        let options = PlacementOptions::new(
            Some(ArrayPlacement::device(devices[1])),
            Option::<ArrayPlacement>::None,
            Some(true),
            Some(Some(false)),
        );
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            value.r#type().into_owned(),
            value.mesh().clone(),
            None,
            values_to_bytes(&[4f32, 5., 6.]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &source_mesh);
        assert_eq!(input_cotangent.r#type(), input.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);

        // A device names a device and not a mesh, so a traced leaf over an array on that device resolves the device
        // to the mesh of the array, both as a source and as a destination that keeps the leaf where it is.
        let options = PlacementOptions::new(
            Some(ArrayPlacement::device(devices[1])),
            Some(ArrayPlacement::device(devices[0])),
            Option::<bool>::None,
            Option::<Option<bool>>::None,
        );
        let (value, _) = differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh().devices(), [devices[1]]);
        let options = PlacementOptions::new(
            Some(ArrayPlacement::device(devices[0])),
            Option::<ArrayPlacement>::None,
            Option::<bool>::None,
            Option::<Option<bool>>::None,
        );
        let (value, _) = differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
        assert_eq!(value.mesh(), &source_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
    }

    #[test]
    fn test_session_place_places_traced_values_in_place_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let sharding = Sharding::new(mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let pinned = Memory::Host { pinned: true };
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(sharding.clone()).unwrap();
        let array = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), mesh.clone(), None, values_to_bytes(values)).unwrap()
        };

        // The primal and the tangent both move to the memory on the devices where they live.
        let (primal, tangent) = differentiate_at(array(&[1., 2., 3., 4.]))
            .jvp(array(&[5., 6., 7., 8.]), |x| Ok(session.place(x, PlacementOptions::to(pinned)).unwrap()))
            .unwrap();
        for (output, values) in
            [(&primal, vec![vec![1., 2.], vec![3., 4.]]), (&tangent, vec![vec![5., 6.], vec![7., 8.]])]
        {
            assert_eq!(output.mesh(), &mesh);
            assert_eq!(output.sharding(), &sharding);
            assert_eq!(output.r#type().memory(), pinned);
            assert_eq!(output.shards()[0].buffer().unwrap().memory().unwrap().kind().unwrap(), "pinned_host");
            assert_eq!(shard_values(output), values);
        }

        // The cotangent returns to the memory of the input, on the same devices.
        let (value, pullback) = differentiate_at(array(&[1., 2., 3., 4.]))
            .vjp(|x| Ok(session.place(x, PlacementOptions::to(pinned)).unwrap()))
            .unwrap();
        assert_eq!(value.r#type().memory(), pinned);
        let cotangent = pullback.apply_materialized(value).unwrap();
        assert_eq!(cotangent.mesh(), &mesh);
        assert_eq!(cotangent.sharding(), &sharding);
        assert_eq!(cotangent.r#type().memory(), Memory::Device);
        let buffer = cotangent.shards()[0].buffer().unwrap();
        assert_eq!(buffer.memory().unwrap(), buffer.device().unwrap().default_memory().unwrap());
        assert_eq!(shard_values(&cotangent), vec![vec![1., 2.], vec![3., 4.]]);
    }

    #[test]
    fn test_session_place_completes_pending_reduction_of_traced_values_in_differentiation() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let target_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let replicated = Sharding::replicated(logical_mesh, 1);
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(replicated.clone().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let upload = |values: &[f32]| {
            Array::from_host_buffer(&session, r#type.clone(), source_mesh.clone(), None, values_to_bytes(values))
                .unwrap()
        };
        let (primal, tangent) = (upload(&[1., 2.]), upload(&[3., 4.]));

        // The primal and the tangent of a traced leaf on a mesh of one device move to another device as a direct
        // placement of an array moves them, with their logical mesh over that device and without their pending
        // reduction along `x`, which has size one, so `exp` applies to them.
        let (value, tangent): (Array, Array) = differentiate_at(primal.clone())
            .jvp(tangent, |x| session.place(x, PlacementOptions::to(devices[1])).unwrap().exp())
            .unwrap();
        let direct = session.place(primal, PlacementOptions::to(devices[1])).unwrap().exp().unwrap();
        assert_eq!(
            (value.mesh(), value.sharding(), value.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!((value.mesh(), value.sharding()), (&target_mesh, &replicated));
        assert_eq!(
            (tangent.mesh(), tangent.sharding(), tangent.r#type()),
            (direct.mesh(), direct.sharding(), direct.r#type())
        );
        assert_eq!(shard_values(&value), vec![vec![1f32.exp(), 2f32.exp()]]);
        assert_eq!(shard_values(&tangent), vec![vec![3. * 1f32.exp(), 4. * 2f32.exp()]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_differentiation_without_destination() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let source_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let cotangent_mesh = DeviceMesh::new(logical_mesh, vec![devices[1]]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [3]),
            source_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();

        // Without a destination, the primal stays where it lives, and the pullback moves a cotangent that lives on
        // another device back to the primal, both with the inferred source and with an explicit one.
        for source in [None, Some(ArrayPlacement::device(devices[0]))] {
            let options = PlacementOptions::<ArrayPlacement>::new(None, source, None, None);
            let (value, pullback) =
                differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
            assert_eq!(value.mesh(), &source_mesh);
            assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
            let cotangent = Array::from_host_buffer(
                &session,
                value.r#type().into_owned(),
                cotangent_mesh.clone(),
                None,
                values_to_bytes(&[4f32, 5., 6.]),
            )
            .unwrap();
            let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
            assert_eq!(input_cotangent.mesh(), &source_mesh);
            assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
        }
    }

    #[test]
    fn test_session_place_stages_explicit_source_of_traced_leaves_without_destination() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let source = ArrayPlacement::sharding(sharding);

        // Without a destination, the placement is staged without a target, so the value stays where it is, and with
        // the source, which only the transposition of the placement uses.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| {
                let options = PlacementOptions::<ArrayPlacement>::new(None, Some(source.clone()), None, None);
                session.place(input, options).unwrap()
            },
            r#type,
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{}]}},
                ] %0
                in (%1)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_session_place_stages_explicit_source_of_traced_leaves_without_destination_for_pullback() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let r#type = ArrayType::new_static(DataType::F32, [2]).with_sharding(sharding.clone()).unwrap();
        let source = ArrayPlacement::sharding(sharding);

        // The pullback places the cotangent on the source, with the memory of the primal and without a source of its
        // own, because the forward placement has no target.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let options = PlacementOptions::<ArrayPlacement>::new(None, Some(source.clone()), None, None);
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type),
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    target={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)
            "#}
            .trim_end(),
        );
    }

    #[test]
    fn test_session_place_stages_device_source_of_traced_values_for_pullback() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let meshes = devices
            .iter()
            .map(|&device| DeviceMesh::new(logical_mesh.clone(), vec![device]).unwrap())
            .collect::<Vec<_>>();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(logical_mesh.clone(), 1))
            .unwrap();
        let options =
            || PlacementOptions::to(ArrayPlacement::device(devices[1])).with_source(ArrayPlacement::device(devices[0]));
        let interpret = |traced: &TracedProgram<(ArrayType, ArrayType), ArrayType>, values: [(&[f32], usize); 2]| {
            let [first, second] = values.map(|(values, mesh): (&[f32], usize)| {
                Value::Array(
                    Array::from_host_buffer(
                        &session,
                        r#type.clone(),
                        meshes[mesh].clone(),
                        None,
                        values_to_bytes(values),
                    )
                    .unwrap(),
                )
            });
            let Value::Array(output) = traced.program().interpret_in_context(&session, (first, second)).unwrap() else {
                panic!("placing an array must produce an array");
            };
            output
        };

        // The devices of a traced input are not known, so a bare device source stays a device, and the pullback moves
        // the cotangent from the destination back onto it over the logical mesh of the input, as JAX's transposition of
        // `device_put` returns it to that device. Interpreting
        // the pullback on an input on the first device and a cotangent on the second one returns the gradient on the
        // first device.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at(input).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
                pullback.apply_materialized(cotangent).unwrap()
            },
            (r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source=device(1),
                    target={mesh<['x'=1:explicit]>, devices=[0], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
        let gradient = interpret(&traced, [(&[1., 2.], 0), (&[3., 4.], 1)]);
        assert_eq!(gradient.mesh(), &meshes[0]);
        assert_eq!(gradient.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&gradient), vec![vec![3., 4.]]);

        // Transposing the pullback again swaps the two devices back, so the cotangent of the gradient moves onto the
        // second device, which is the destination of the forward placement.
        let traced: TracedProgram<(ArrayType, ArrayType), ArrayType> = trace(
            |(input, cotangent): (ArrayTracer, ArrayTracer)| {
                let (_, pullback) = differentiate_at((input, cotangent.clone()))
                    .vjp(|(x, cotangent)| {
                        let (_, pullback) = differentiate_at(x).vjp(|x| Ok(session.place(x, options()).unwrap()))?;
                        pullback.apply_materialized(cotangent)
                    })
                    .unwrap();
                pullback.apply_materialized(cotangent).unwrap().1
            },
            (r#type.clone(), r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}], %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %2:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [
                    source={mesh<['x'=1:explicit]>, devices=[0], sharding={mesh<['x'=1:explicit]>, [{}]}},
                    target={mesh<['x'=1:explicit]>, devices=[1], \
                        sharding={mesh<['x'=1:explicit]>, [{}]}, memory=Device},
                    may_alias=false,
                ] %1
                in (%2)"
            },
        );
        let cotangent = interpret(&traced, [(&[1., 2.], 0), (&[5., 6.], 0)]);
        assert_eq!(cotangent.mesh(), &meshes[1]);
        assert_eq!(cotangent.r#type().as_ref(), &r#type);
        assert_eq!(shard_values(&cotangent), vec![vec![5., 6.]]);
    }

    #[test]
    fn test_session_place_keeps_traced_leaves_without_destination_where_they_live() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let first_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[0]]).unwrap();
        let second_mesh = DeviceMesh::new(logical_mesh.clone(), vec![devices[1]]).unwrap();
        let sharding = Sharding::replicated(logical_mesh, 1);
        let r#type = ArrayType::new_static(DataType::F32, [3]).with_sharding(sharding.clone()).unwrap();
        let source = Placement::new(second_mesh.clone(), sharding).unwrap();
        let input = Array::from_host_buffer(
            &session,
            r#type.clone(),
            first_mesh.clone(),
            None,
            values_to_bytes(&[1f32, 2., 3.]),
        )
        .unwrap();
        let options = || PlacementOptions::<ArrayPlacement, Placement>::new(None, Some(source.clone()), None, None);

        // A placement with only a source is staged without a target, so replaying it on an array that lives on the
        // first device leaves the array there, even though the source names the second device.
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|input: ArrayTracer| session.place(input, options()).unwrap(), r#type).unwrap();
        let Value::Array(output) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
            panic!("placing an array must produce an array");
        };
        assert_eq!(output.mesh(), &first_mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3.]]);

        // Its pullback moves the cotangent to the source.
        let (value, pullback) =
            differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options()).unwrap())).unwrap();
        assert_eq!(value.mesh(), &first_mesh);
        assert_eq!(shard_values(&value), vec![vec![1., 2., 3.]]);
        let cotangent = Array::from_host_buffer(
            &session,
            value.r#type().into_owned(),
            first_mesh,
            None,
            values_to_bytes(&[4f32, 5., 6.]),
        )
        .unwrap();
        let input_cotangent = pullback.apply_materialized(cotangent).unwrap();
        assert_eq!(input_cotangent.mesh(), &second_mesh);
        assert_eq!(input_cotangent.r#type(), input.r#type());
        assert_eq!(shard_values(&input_cotangent), vec![vec![4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_compiles_traced_leaves_with_only_a_source() {
        let client = cpu_client(1);
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![device],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2]);

        // A bare device does not determine a placement for a traced value whose devices are not known, so the source
        // stays a device, which only transposition uses, and the placement without a target compiles and leaves the
        // value where it is.
        let source = ArrayPlacement::device(device);
        let compiled: CompiledFunction<ArrayType, ArrayType> = compile(
            |input: CompileTracer| {
                session
                    .place(input, PlacementOptions::<ArrayPlacement>::new(None, Some(source), None, None))
                    .unwrap()
            },
            input_type.clone(),
            &session,
            mesh.clone(),
        )
        .unwrap();
        assert_eq!(
            compiled.source_program().program().to_string(),
            indoc! {"
                lambda %0:f32[2] .
                let %1:f32[2] = place [source=device(0)] %0
                in (%1)"
            },
        );
        let input =
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_copies_traced_leaves_without_destination_and_source() {
        let client = cpu_client(1);
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap()],
        )
        .unwrap();
        let session = Session::builder().client(client).build().unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.logical_mesh().clone(), 1))
            .unwrap();
        let input =
            Array::from_host_buffer(&session, r#type.clone(), mesh, None, values_to_bytes(&[1f32, 2.])).unwrap();

        // A placement without a destination and a source that must not alias its input is still staged, so replaying
        // it while the caller holds the input returns a copy of the buffer of the input.
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |input: ArrayTracer| {
                let options = PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, _>::new(
                    None,
                    None,
                    None,
                    Some(Some(false)),
                );
                session.place(input, options).unwrap()
            },
            r#type,
        )
        .unwrap();
        assert_eq!(
            traced.program().to_string(),
            indoc! {r#"
                lambda %0:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] .
                let %1:f32[2][sharding={mesh<['x'=1:explicit]>, [{}]}] = place [may_alias=false] %0
                in (%1)
            "#}
            .trim_end(),
        );
        let Value::Array(output) =
            traced.program().interpret_in_context(&session, Value::Array(input.clone())).unwrap()
        else {
            panic!("placing an array must produce an array");
        };
        assert!(!Arc::ptr_eq(output.shards()[0].buffer().unwrap(), input.shards()[0].buffer().unwrap()));
        assert_ne!(native_buffer_addresses(&output), native_buffer_addresses(&input));
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_batching() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[1]).unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // The target resolves for the rank of each batch item, and the packed result lives on the target device.
        let options = PlacementOptions::<ArrayPlacement>::new(Some(ArrayPlacement::device(device)), None, None, None);
        let output: Array =
            batch(|x| Ok(session.place(x, options).unwrap()), input, BatchAxis::new(0), BatchAxis::new(0), None)
                .unwrap();
        assert_eq!(output.mesh().devices(), [device]);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_places_traced_leaves_in_batching_on_their_own_placement() {
        let client = cpu_client(2);
        let mesh = domain_mesh(client, "x", 2);
        let session = Session::builder().client(client).build().unwrap();
        let sharding = Sharding::new(
            mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 3]),
            mesh.clone(),
            Some(sharding.clone()),
            values_to_bytes(&[1f32, 2., 3., 4., 5., 6.]),
        )
        .unwrap();

        // Each device holds one batch item, which the placement of a batch item records as the sharding of its batch,
        // so placing every batch item on its own placement leaves each of them on its device.
        let output: Array = batch(
            |x| {
                let placement = x.placement().unwrap();
                assert_eq!(placement.batch_shardings(), [ShardingDimension::sharded(["x"])]);
                Ok(session.place(x, PlacementOptions::<Placement>::new(Some(placement), None, None, None)).unwrap())
            },
            input,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(output.sharding(), &sharding);
        assert_eq!(shard_values(&output), vec![vec![1., 2., 3.], vec![4., 5., 6.]]);
    }

    #[test]
    fn test_session_place_copies_traced_leaves_that_may_not_alias() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let input = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [2, 2]),
            None,
            None,
            values_to_bytes(&[1f32, 2., 3., 4.]),
        )
        .unwrap();
        let shares_input_buffer =
            |array: &Array| Arc::ptr_eq(array.shards()[0].buffer().unwrap(), input.shards()[0].buffer().unwrap());
        let options = |may_alias: Option<bool>| {
            PlacementOptions::<ArrayPlacement, ArrayPlacement, bool, _>::new(None, None, None, Some(may_alias))
        };

        // The caller keeps a live handle of the input, so a placement that leaves each batch item where it is returns
        // the buffer of the input by default and a copy of it when the result must not alias the input.
        for (may_alias, shares_buffer) in [(None, true), (Some(false), false)] {
            let output: Array = batch(
                |x| Ok(session.place(x, options(may_alias)).unwrap()),
                input.clone(),
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();
            assert_eq!(shares_input_buffer(&output), shares_buffer);
            assert_eq!(output.r#type(), input.r#type());
            assert_eq!(shard_values(&output), vec![vec![1., 2., 3., 4.]]);
        }

        // The primal of a differentiated placement follows the same rule.
        for (may_alias, shares_buffer) in [(None, true), (Some(false), false)] {
            let (value, _) =
                differentiate_at(input.clone()).vjp(|x| Ok(session.place(x, options(may_alias)).unwrap())).unwrap();
            assert_eq!(shares_input_buffer(&value), shares_buffer);
            assert_eq!(value.r#type(), input.r#type());
            assert_eq!(shard_values(&value), vec![vec![1., 2., 3., 4.]]);
        }
    }

    #[test]
    fn test_session_place_reports_errors_of_arrays_in_the_order_of_the_type_rule() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let mesh = two_device_mesh(client, MeshAxisType::Explicit);
        let unreduced = Sharding::replicated(mesh.logical_mesh().clone(), 1).with_unreduced_axes(["x"]).unwrap();
        let r#type = ArrayType::new_static(DataType::F32, [4]).with_sharding(unreduced).unwrap();
        let input = Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(&[1f32, 2., 3., 4.]))
            .unwrap();
        let target = ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 1)).unwrap();
        let layout = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));

        // Each target and each source below is rejected, so the error of the eager placement must be the first one that
        // the type rule reports for the requests that the frontend stages for the same array. The target changes the
        // reduction state of the value, except for the empty in-place target.
        let reduction_error = "`place` cannot change reduction state; use a reduction (or the corresponding \
                               collective) instead";
        let cases = [
            (
                target.clone(),
                ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap(),
                "`place` source sharding rank (2) does not match the input rank (1)",
            ),
            (target.clone(), ArrayPlacement::in_place(), "`place` in-place source must name a memory or a layout"),
            (
                ArrayPlacement::in_place(),
                ArrayPlacement::in_place(),
                "`place` in-place source must name a memory or a layout",
            ),
            (target.clone(), ArrayPlacement::in_place().with_layout(layout), reduction_error),
            (target, ArrayPlacement::from(Memory::Host { pinned: true }), reduction_error),
        ];
        for (target, source, expected_error) in cases {
            let options = PlacementOptions::new(
                Some(target.clone()),
                Some(source.clone()),
                Option::<bool>::None,
                Option::<Option<bool>>::None,
            );
            let eager = session.place(input.clone(), options).err().unwrap().to_string();
            let operation = PlaceOperation::new(
                target.into_request(&session, 1).unwrap(),
                source.into_request(&session, 1).unwrap(),
            );
            let inferred = operation.infer_output_types(&[input.r#type().into_owned()], &[]).err().unwrap();
            let inferred = Error::from(inferred).to_string();
            assert_eq!((eager, inferred), (expected_error.to_string(), expected_error.to_string()));
        }

        // A host value is uploaded as the placement of a value without a tracked sharding in device memory, so its
        // destination is rejected with the first error of the type rule for such a value, as the destination of an
        // upload of the same value without a mesh is, and as the destination of a traced value of that type is.
        let manual_mesh = two_device_mesh(client, MeshAxisType::Manual);
        let host = CoreArray::from_elements(ArrayType::new_static(DataType::F32, [2]), &[1f32, 2.]).unwrap();
        let replicated = || Sharding::replicated(mesh.logical_mesh().clone(), 1);
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let cases = [
            (
                ArrayPlacement::mesh(mesh.clone(), replicated().with_unreduced_axes(["x"]).unwrap()).unwrap(),
                reduction_error.to_string(),
            ),
            (
                ArrayPlacement::mesh(mesh.clone(), replicated().with_reduced_axes(["x"]).unwrap()).unwrap(),
                reduction_error.to_string(),
            ),
            (
                ArrayPlacement::mesh(
                    manual_mesh.clone(),
                    Sharding::replicated(manual_mesh.logical_mesh().clone(), 1)
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap(),
                "`place` cannot move values that vary along manual mesh axes or carry reduction state over them"
                    .to_string(),
            ),
            (
                ArrayPlacement::device(mesh.devices()[1]).with_layout(rank_2.clone()),
                "`place` target layout is invalid: tiled layout rank 2 does not match array rank 1".to_string(),
            ),
            (
                ArrayPlacement::mesh(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2)).unwrap(),
                "`place` target sharding rank (2) does not match the input rank (1)".to_string(),
            ),
        ];
        for (target, expected_error) in cases {
            let options = || PlacementOptions::to(target.clone());
            let uploaded = Array::from_host_buffer(
                &session,
                ArrayType::new_static(DataType::F32, [2]),
                None,
                None,
                values_to_bytes(&[1f32, 2.]),
            )
            .unwrap();
            let host = session.place(host.clone(), options()).err().unwrap().to_string();
            let array = session.place(uploaded, options()).err().unwrap().to_string();
            let error = RefCell::new(None);
            let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| {
                    session.place(x.clone(), options()).unwrap_or_else(|placement_error| {
                        error.replace(Some(placement_error.to_string()));
                        x
                    })
                },
                ArrayType::new_static(DataType::F32, [2]),
            );
            let traced = error.take().unwrap();
            assert_eq!((host, array, traced), (expected_error.clone(), expected_error.clone(), expected_error));
        }
    }

    #[test]
    fn test_session_place_agrees_across_eager_staged_compiled_and_transformed_paths() {
        // Every observable fact of one placed array, so that the arrays that different paths produce can be compared.
        #[derive(Clone, Debug, PartialEq)]
        struct Observed {
            devices: Vec<Device>,
            logical_mesh: LogicalMesh,
            sharding: Sharding,
            r#type: ArrayType,
            memory_kinds: Vec<String>,
            values: Vec<f32>,
            committed: bool,
        }

        // Records the error of a placement inside a transformed function, whose error type is not the error type of
        // `Session::place`, so that the path reports that error rather than the one that wraps it.
        fn record<T>(error: &RefCell<Option<String>>, result: Result<T, Error>) -> Result<T, ProgramError> {
            result.map_err(|placement_error| {
                error.replace(Some(placement_error.to_string()));
                ProgramError::InvalidArgument { message: placement_error.to_string() }
            })
        }

        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let pinned = Memory::Host { pinned: true };
        let single_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap();
        let double_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let observe = |array: &Array| Observed {
            devices: array.mesh().devices().to_vec(),
            logical_mesh: array.mesh().logical_mesh().clone(),
            sharding: array.sharding().clone(),
            r#type: array.r#type().into_owned(),
            memory_kinds: array
                .addressable_shards()
                .map(|shard| shard.buffer().unwrap().memory().unwrap().kind().unwrap().into_owned())
                .collect(),
            values: array.copy_to_host().unwrap().elements::<f32>().unwrap(),
            committed: array.is_committed(),
        };
        // Uploads `values` with the shape `shape`, the sharding `sharding` over the logical mesh of `mesh`, and in
        // `memory`, which a placement onto that memory selects after the upload. Without a sharding, the values are
        // uploaded without a mesh, so the array is uncommitted on the canonical mesh of the first device.
        let upload =
            |shape: &[usize], sharding: Option<&Sharding>, mesh: &DeviceMesh, memory: Memory, values: &[f32]| {
                let r#type = ArrayType::new_static(DataType::F32, shape.to_vec());
                let array = match sharding {
                    Some(sharding) => {
                        let r#type = r#type.with_sharding(sharding.clone()).unwrap();
                        Array::from_host_buffer(&session, r#type, mesh.clone(), None, values_to_bytes(values)).unwrap()
                    }
                    None => Array::from_host_buffer(&session, r#type, None, None, values_to_bytes(values)).unwrap(),
                };
                match memory {
                    Memory::Device => array,
                    memory => session.place(array, PlacementOptions::to(memory)).unwrap(),
                }
            };

        // The input states, each with the meshes of its own devices and of the devices that the other paths use.
        let single = (DeviceMesh::new(single_mesh.clone(), vec![devices[0]]).unwrap(), single_mesh.clone());
        let double = (DeviceMesh::new(double_mesh.clone(), devices[..2].to_vec()).unwrap(), double_mesh.clone());
        let other_mesh = |mesh: &DeviceMesh| {
            let other_devices = devices[2..2 + mesh.devices().len()].to_vec();
            DeviceMesh::new(mesh.logical_mesh().clone(), other_devices).unwrap()
        };
        let replicated = |mesh: &LogicalMesh| Sharding::replicated(mesh.clone(), 1);
        let unconstrained =
            |mesh: &LogicalMesh| Sharding::new(mesh.clone(), vec![ShardingDimension::Unconstrained]).unwrap();
        let canonical = DeviceMesh::new(LogicalMesh::single_device(), vec![devices[0]]).unwrap();
        let inputs = [
            ("single device", single.0.clone(), Some(replicated(&single.1)), Memory::Device),
            ("uncommitted on the canonical mesh", canonical, None, Memory::Device),
            (
                "sharded over two devices",
                double.0.clone(),
                Some(Sharding::new(double.1.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
                Memory::Device,
            ),
            ("replicated over two devices", double.0.clone(), Some(replicated(&double.1)), Memory::Device),
            (
                "reduced over an axis of size two",
                double.0.clone(),
                Some(replicated(&double.1).with_reduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            (
                "unreduced over an axis of size one",
                single.0.clone(),
                Some(replicated(&single.1).with_unreduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            (
                "unreduced over an axis of size two",
                double.0.clone(),
                Some(replicated(&double.1).with_unreduced_axes(["x"]).unwrap()),
                Memory::Device,
            ),
            ("pinned host memory", single.0.clone(), Some(replicated(&single.1)), pinned),
        ];
        let layout = Layout::Tiled(TiledLayout::new(vec![0], vec![]));

        let mut disagreements = Vec::new();
        for (input_name, mesh, sharding, memory) in inputs {
            let input = upload(&[4], sharding.as_ref(), &mesh, memory, &[1., 2., 3., 4.]);
            assert_eq!(input.mesh(), &mesh);
            let other_memory = if memory == Memory::Device { pinned } else { Memory::Device };
            // A sharding target resolves over the default mesh of the session handle that places the value. A target on
            // another logical mesh over the devices of the value (i.e., its mesh with another axis `'y'=1:explicit`,
            // which restates its reduction state) moves the value onto a mesh of another topology where it lives (e.g.,
            // an uncommitted value from the canonical mesh of its device onto `['device'=1:auto, 'y'=1:explicit]`). A
            // device target on the device of the value is the sole device of the mesh of a value on one device.
            let meshed_session = session.clone().with_device_mesh(other_mesh(&mesh));
            let input_sharding = input.sharding();
            let mut renamed_axes = input_sharding.mesh().axes().to_vec();
            renamed_axes.push(MeshAxis::new("y", 1, MeshAxisType::Explicit).unwrap());
            let renamed_mesh = LogicalMesh::new(renamed_axes).unwrap();
            let renamed = DeviceMesh::new(renamed_mesh.clone(), mesh.devices().to_vec()).unwrap();
            let renamed_sharding = Sharding::new(renamed_mesh, input_sharding.dimensions().to_vec())
                .and_then(|sharding| sharding.with_unreduced_axes(input_sharding.unreduced_axes().clone()))
                .and_then(|sharding| sharding.with_reduced_axes(input_sharding.reduced_axes().clone()))
                .unwrap();
            // A placement never changes reduction state on any path, so a target that does not restate the reduction
            // state of the value (i.e., the mesh of the value replicated, or a mesh that does not define the axis of
            // that state) is rejected on every path, as a device target is for a value that is unreduced over a larger
            // axis.
            let undefined_mesh =
                LogicalMesh::new(vec![MeshAxis::new("y", mesh.devices().len(), MeshAxisType::Explicit).unwrap()])
                    .unwrap();
            let undefined = DeviceMesh::new(undefined_mesh.clone(), other_mesh(&mesh).devices().to_vec()).unwrap();
            let targets = [
                ("mesh", Some(ArrayPlacement::mesh(other_mesh(&mesh), input.sharding().clone()).unwrap())),
                ("replicated", Some(ArrayPlacement::mesh(mesh.clone(), replicated(mesh.logical_mesh())).unwrap())),
                ("undefined axis", Some(ArrayPlacement::mesh(undefined, replicated(&undefined_mesh)).unwrap())),
                ("sharding", Some(ArrayPlacement::sharding(input.sharding().clone()))),
                ("other logical mesh", Some(ArrayPlacement::mesh(renamed, renamed_sharding).unwrap())),
                (
                    "unconstrained",
                    Some(ArrayPlacement::mesh(mesh.clone(), unconstrained(mesh.logical_mesh())).unwrap()),
                ),
                ("device", Some(ArrayPlacement::device(devices[2]))),
                ("own device", Some(ArrayPlacement::device(mesh.devices()[0]))),
                ("memory", Some(ArrayPlacement::from(other_memory))),
                ("none", None),
            ];
            // A device source states that the value lives on that device alone, so it is rejected on every path for a
            // value that is tracked over two devices. The devices of a source are informative, so a source on other
            // devices than the value is accepted on every path but a compiled function, whose program must contain
            // every device that a `place` names, and reverse mode sends the cotangent there. A source layout is checked
            // against the type of the value only, which declares none here, so it is accepted on every path, including
            // the batched one, where it describes a batch item.
            let resolved = ArrayPlacement::mesh(input.mesh().clone(), input.sharding().clone()).unwrap();
            let sources = [
                ("none", None),
                ("inferred", Some(ArrayPlacement::from(input.placement().unwrap()))),
                ("resolved", Some(resolved.clone())),
                ("other devices", Some(ArrayPlacement::mesh(other_mesh(&mesh), input.sharding().clone()).unwrap())),
                ("memory", Some(ArrayPlacement::from(memory))),
                ("device", Some(ArrayPlacement::device(mesh.devices()[0]))),
                ("layout", Some(ArrayPlacement::in_place().with_layout(layout.clone()))),
                ("resolved with layout", Some(resolved.with_layout(layout.clone()))),
                ("device with layout", Some(ArrayPlacement::device(mesh.devices()[0]).with_layout(layout.clone()))),
            ];
            for (target_name, target) in &targets {
                let session = if *target_name == "sharding" { &meshed_session } else { &session };
                for (source_name, source) in &sources {
                    let cell = format!("{input_name}, target {target_name}, source {source_name}");
                    let options = || {
                        PlacementOptions::new(
                            target.clone(),
                            source.clone(),
                            Option::<bool>::None,
                            Option::<Option<bool>>::None,
                        )
                    };
                    let error = RefCell::new(None::<String>);
                    let failure = |other: String| error.take().unwrap_or(other);

                    // The eager placement is what every other path must agree with.
                    let eager = session.place(input.clone(), options()).map(|output| observe(&output));
                    let eager = eager.map_err(|error| error.to_string());

                    // A value that is unreduced over an axis of size two holds a contribution on each device, which no
                    // placement combines, so the targets that do not restate that state are rejected with the error of
                    // the type rule, whatever the source, and every other path must report the same error.
                    let rejection = match (input_name, *target_name) {
                        ("unreduced over an axis of size two", "replicated") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the \
                             corresponding collective) instead",
                        )),
                        ("unreduced over an axis of size two", "undefined axis") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot redefine preserved explicit mesh axis `'x'=2:explicit` as \
                             an undefined axis",
                        )),
                        ("unreduced over an axis of size two", "device" | "own device") => Some(format!(
                            "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger \
                             than one (i.e., `x`) onto a single device; use a reduction (or the corresponding \
                             collective) first",
                        )),
                        _ => None,
                    };
                    if let Some(rejection) = rejection
                        && eager != Err(rejection.clone())
                    {
                        disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                    }

                    // The core `Place` capability places the array with the requests that the frontend stages for a
                    // traced value, through the same type rule and resolver.
                    let request = |placement: &Option<ArrayPlacement>| {
                        placement.clone().map(|placement| placement.into_request(session, 1)).transpose()
                    };
                    let core = match (request(target), request(source)) {
                        (Ok(target), Ok(source)) => input
                            .place(&target, &source)
                            .map(|output| observe(&output))
                            .map_err(|error| Error::from(error).to_string()),
                        (Err(error), _) | (_, Err(error)) => Err(error.to_string()),
                    };
                    if core != eager {
                        disagreements.push(format!("{cell}, core: {core:?} != eager: {eager:?}"));
                    }

                    // The type rule owns the type of a placed value, so the array that the eager placement produces has
                    // the type that the type rule infers for the request as it is given.
                    if let (Ok(eager), Ok(target), Ok(source)) = (&eager, request(target), request(source)) {
                        let r#type = input.r#type();
                        let sharding = Some(input.sharding());
                        let inferred =
                            PlaceOperation::infer_output_type_allowing_uneven_splits(&r#type, sharding, target.as_ref(), source.as_ref());
                        if inferred.as_ref() != Ok(&eager.r#type) {
                            disagreements.push(format!("{cell}, type rule: {inferred:?} != eager: {:?}", eager.r#type));
                        }
                    }

                    // A traced function interpreted on the session (i.e., the replay of a staged program).
                    let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                        |x: ArrayTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
                        input.r#type().into_owned(),
                    );
                    let staged = match (error.take(), traced) {
                        (Some(message), _) => Err(message),
                        (None, Err(error)) => Err(error.to_string()),
                        (None, Ok(traced)) => {
                            match traced.program().interpret_in_context(session, Value::Array(input.clone())) {
                                Ok(Value::Array(output)) => Ok(observe(&output)),
                                Ok(_) => Err("placing an array must produce an array".to_string()),
                                Err(error) => Err(error.to_string()),
                            }
                        }
                    };
                    if staged != eager {
                        disagreements.push(format!("{cell}, staged: {staged:?} != eager: {eager:?}"));
                    }

                    // A function compiled for the mesh of the input, where the decisions allow the target (i.e., one
                    // that moves no value to other devices, or a device target on the sole device of that mesh). Every
                    // device that a `place` names inside a compiled function must belong to the program, so a source on
                    // other devices, to which transposition would return the cotangent, is rejected there, after the
                    // type rule.
                    let is_sole_device = *target_name == "own device" && mesh.devices().len() == 1;
                    let expected_compiled = match (*source_name, &eager) {
                        ("other devices", Ok(_)) => {
                            let device_ids =
                                |mesh: &DeviceMesh| mesh.devices().iter().map(|device| device.id()).collect::<Vec<_>>();
                            Err(format!(
                                "cannot lower `{PLACE_OPERATION_NAME}`: its source names the devices {:?} of mesh \
                                 `{}`, which are not the devices {:?} of the mesh `{}` that the program is compiled \
                                 for, and transposition would return the cotangent to them",
                                device_ids(&other_mesh(&mesh)),
                                mesh.logical_mesh(),
                                device_ids(&mesh),
                                mesh.logical_mesh(),
                            ))
                        }
                        _ => eager.clone(),
                    };
                    if matches!(*target_name, "memory" | "none" | "unconstrained") || is_sole_device {
                        let compiled: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                            |x: CompileTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
                            input.r#type().into_owned(),
                            &session,
                            input.mesh().clone(),
                        );
                        let compiled = match (error.take(), compiled) {
                            (Some(message), _) => Err(message),
                            (None, Err(error)) => Err(error.to_string()),
                            (None, Ok(compiled)) => compiled
                                .executable()
                                .execute(&session, input.clone())
                                .map(|output| observe(&output))
                                .map_err(|error| error.to_string()),
                        };
                        if compiled != expected_compiled {
                            disagreements.push(format!("{cell}, compiled: {compiled:?} != {expected_compiled:?}"));
                        }
                    }

                    // Eager forward mode differentiation places the primal and the tangent alike.
                    let jvp = differentiate_at(input.clone())
                        .jvp(input.clone(), |x| record(&error, session.place(x, options())))
                        .map_err(|error| failure(error.to_string()));
                    let (primal, tangent) = match jvp {
                        Ok((primal, tangent)) => (Ok(observe(&primal)), Ok(observe(&tangent))),
                        Err(error) => (Err(error.clone()), Err(error)),
                    };
                    if primal != eager {
                        disagreements.push(format!("{cell}, jvp primal: {primal:?} != eager: {eager:?}"));
                    }
                    if tangent != eager {
                        disagreements.push(format!("{cell}, jvp tangent: {tangent:?} != eager: {eager:?}"));
                    }

                    // Reverse mode rejects one combination by design. The cotangent of a device target onto a value
                    // that is reduced over a larger axis is unreduced over it, which a placement cannot produce. A
                    // memory source of a concrete value is completed with where the value lives, so the cotangent of
                    // a value that a device target gathers returns to its devices with every source.
                    let is_gathered = mesh.devices().len() > 1 && matches!(*target_name, "device" | "own device");
                    let rejected_in_reverse_mode = match () {
                        _ if eager.is_err() || !is_gathered => None,
                        _ if input_name == "reduced over an axis of size two" => Some(format!(
                            "cannot transpose `{PLACE_OPERATION_NAME}` onto a single device for a value that is \
                             reduced over mesh axes larger than one (i.e., `x`), because its cotangent is unreduced \
                             over them",
                        )),
                        _ => None,
                    };
                    let rejected_in_batched_reverse_mode = rejected_in_reverse_mode.clone();

                    // Returns the disagreement of the result of a reverse mode path with the rejection `expected`, if
                    // there is one. A result that is rejected as expected agrees.
                    let check_rejection =
                        |path: &str, expected: &Option<String>, result: &Result<(), String>| match expected {
                            Some(expected)
                                if !result.as_ref().is_err_and(|message| message.contains(expected.as_str())) =>
                            {
                                Some(format!("{cell}, {path}: {result:?} does not report `{expected}`"))
                            }
                            _ => None,
                        };

                    // Returns a seed for the cotangent of `output`, with the values `values` and the shape `shape` of
                    // `output`, which lives on other devices than `output` in the memory of `output`.
                    let seed = |output: &Observed, shape: &[usize], sharding: &Sharding, values: &[f32]| {
                        let seed_devices = devices
                            .iter()
                            .copied()
                            .filter(|device| !output.devices.contains(device))
                            .take(output.devices.len())
                            .collect::<Vec<_>>();
                        let seed_mesh = DeviceMesh::new(output.logical_mesh.clone(), seed_devices).unwrap();
                        upload(shape, Some(&sharding.cotangent().unwrap()), &seed_mesh, output.r#type.memory(), values)
                    };

                    // Returns the disagreement of a cotangent with the cotangent of `primal` that its pullback must
                    // return for a seed whose values are `values`. It returns to the memory of the primal
                    // and to the devices that the source names, which are those of the primal, except for a source
                    // on other devices. A memory source names no devices, and it is completed with the devices of the
                    // concrete primal, so the cotangent returns to them, also from a seed on other devices.
                    let check_cotangent = |path: &str, cotangent: &Array, primal: &Array, values: Vec<f32>| {
                        let cotangent = observe(cotangent);
                        let devices = match *source_name {
                            "other devices" => other_mesh(&mesh).devices().to_vec(),
                            _ => primal.mesh().devices().to_vec(),
                        };
                        let expected =
                            (devices, observe(primal).memory_kinds, primal.r#type().cotangent().unwrap(), values);
                        let actual = (cotangent.devices, cotangent.memory_kinds, cotangent.r#type, cotangent.values);
                        (actual != expected).then(|| format!("{cell}, {path} cotangent: {actual:?} != {expected:?}"))
                    };

                    // Eager reverse mode differentiation, whose pullback is applied to a seed that lives on other
                    // devices than the output, returns the cotangent where `check_cotangent` says.
                    let vjp = differentiate_at(input.clone())
                        .vjp(|x| record(&error, session.place(x, options())))
                        .map_err(|error| failure(error.to_string()));
                    match vjp {
                        Ok((value, pullback)) => {
                            let value = observe(&value);
                            if Ok(&value) != eager.as_ref() {
                                disagreements.push(format!("{cell}, vjp primal: {value:?} != eager: {eager:?}"));
                            }
                            let seed = seed(&value, &[4], value.r#type.sharding().unwrap(), &[5., 6., 7., 8.]);
                            let cotangent =
                                pullback.apply_materialized(seed.clone()).map_err(|error| error.to_string());
                            let result = cotangent.as_ref().map(|_| ()).map_err(Clone::clone);
                            match (check_rejection("vjp", &rejected_in_reverse_mode, &result), cotangent) {
                                (Some(disagreement), _) => disagreements.push(disagreement),
                                (None, _) if rejected_in_reverse_mode.is_some() => {}
                                (None, Ok(cotangent)) => {
                                    let values = observe(&seed).values;
                                    disagreements.extend(check_cotangent("vjp", &cotangent, &input, values));
                                }
                                (None, Err(error)) => disagreements.push(format!("{cell}, vjp cotangent: {error}")),
                            }
                        }
                        Err(error) => {
                            if Err(&error) != eager.as_ref().map(|_| ()) {
                                disagreements.push(format!("{cell}, vjp: {error} != eager: {eager:?}"));
                            }
                        }
                    }

                    // Batched reverse mode differentiation places every batch item where the eager placement places the
                    // input, and its pullback returns the cotangent of every batch item where `check_cotangent` says.
                    // The seed is mapped when the output lives on the logical mesh of the input, and it is the same for
                    // every batch item otherwise, because the batches of all mapped inputs live on one logical mesh.
                    let Ok(eager) = &eager else { continue };
                    let packed_sharding =
                        sharding.as_ref().map(|sharding| sharding.batched(0, ShardingDimension::Replicated).unwrap());
                    let packed = upload(
                        &[2, 4],
                        packed_sharding.as_ref(),
                        input.mesh(),
                        memory,
                        &[1., 2., 3., 4., 5., 6., 7., 8.],
                    );
                    let eager_sharding = eager.r#type.sharding().unwrap();
                    let (packed_seed, seed_axis, seed_repeats) = if &eager.logical_mesh == input.mesh().logical_mesh() {
                        let sharding = eager_sharding.batched(0, ShardingDimension::Replicated).unwrap();
                        let values = [9., 10., 11., 12., 13., 14., 15., 16.];
                        (seed(eager, &[2, 4], &sharding, &values), BatchAxis::new(0), 1)
                    } else {
                        (seed(eager, &[4], eager_sharding, &[9., 10., 11., 12.]), BatchAxis::replicated(), 2)
                    };
                    let batched: Result<(Array, Array), _> = batch(
                        |(x, seed)| {
                            let (value, pullback) =
                                differentiate_at(x).vjp(|x| record(&error, session.place(x, options())))?;
                            Ok((value, pullback.apply_materialized(seed)?))
                        },
                        (packed.clone(), packed_seed.clone()),
                        (BatchAxis::new(0), seed_axis),
                        (BatchAxis::new(0), BatchAxis::new(0)),
                        None,
                    );
                    let batched = batched.map_err(|error| failure(error.to_string()));
                    let result = batched.as_ref().map(|_| ()).map_err(Clone::clone);
                    match (check_rejection("batched vjp", &rejected_in_batched_reverse_mode, &result), batched) {
                        (Some(disagreement), _) => disagreements.push(disagreement),
                        (None, _) if rejected_in_batched_reverse_mode.is_some() => {}
                        (None, Ok((value, cotangent))) => {
                            let value = observe(&value);
                            let replicated = ShardingDimension::Replicated;
                            let expected = Observed {
                                devices: eager.devices.clone(),
                                logical_mesh: eager.logical_mesh.clone(),
                                sharding: eager.sharding.batched(0, replicated.clone()).unwrap(),
                                r#type: eager.r#type.batched(0, Dimension::Static(2), replicated).unwrap(),
                                memory_kinds: eager.memory_kinds.clone(),
                                values: observe(&packed).values,
                                committed: eager.committed,
                            };
                            let actual = &value;
                            if actual != &expected {
                                disagreements.push(format!("{cell}, batched vjp value: {actual:?} != {expected:?}"));
                            }
                            let values = observe(&packed_seed).values.repeat(seed_repeats);
                            disagreements.extend(check_cotangent("batched vjp", &cotangent, &packed, values));
                        }
                        (None, Err(error)) => disagreements.push(format!("{cell}, batched vjp: {error}")),
                    }
                }
            }
        }
        // A host value is uploaded where a destination says, which is where the same destination places an upload of
        // that value without a mesh, so both kinds of leaves agree (a host value ignores sources, so none is given).
        let host = CoreArray::from_elements(ArrayType::new_static(DataType::F32, [4]), &[1f32, 2., 3., 4.]).unwrap();
        let meshed_session = session.clone().with_device_mesh(other_mesh(&single.0));
        for (target_name, target) in [
            ("mesh", Some(ArrayPlacement::mesh(other_mesh(&single.0), replicated(&single.1)).unwrap())),
            ("sharding", Some(ArrayPlacement::sharding(replicated(&single.1)))),
            ("device", Some(ArrayPlacement::device(devices[2]))),
            ("memory", Some(ArrayPlacement::from(pinned))),
            ("none", None),
        ] {
            let session = if target_name == "sharding" { &meshed_session } else { &session };
            let options = || {
                PlacementOptions::new(
                    target.clone(),
                    Option::<ArrayPlacement>::None,
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                )
            };
            let uploaded = Array::from_host_buffer(
                session,
                ArrayType::new_static(DataType::F32, [4]),
                None,
                None,
                values_to_bytes(&[1f32, 2., 3., 4.]),
            )
            .unwrap();
            let host = session.place(host.clone(), options()).map(|output| observe(&output));
            let array = session.place(uploaded, options()).map(|output| observe(&output));
            let (host, array) = (host.map_err(|error| error.to_string()), array.map_err(|error| error.to_string()));
            if host != array {
                disagreements.push(format!("host value, target {target_name}: {host:?} != array: {array:?}"));
            }
        }

        // The type of an array omits its reduction state over auto mesh axes, which its runtime sharding records (e.g.,
        // a scalar that is imported as the contributions 1 and 2 of a pending sum over `'x'=2:auto`, or that is reduced
        // over that axis), so the eager placement, the core `Place` capability, and forward mode validate a concrete
        // array against its runtime sharding. A target that restates that state (i.e., the placement of the array, or
        // the same state over other devices) is accepted and keeps the value, whereas a target that drops it and a
        // device target of a pending sum over a larger axis are rejected, as for the same state over an explicit axis.
        // The type rule with the runtime sharding of the array agrees with the eager placement on every cell. A staged
        // value has only its type, so the replay of a staged program rejects the targets that restate a state over an
        // auto axis when it is traced, and it agrees with the eager placement on every other target.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto = DeviceMesh::new(auto_mesh.clone(), devices[..2].to_vec()).unwrap();
        let other_auto = DeviceMesh::new(auto_mesh.clone(), devices[2..].to_vec()).unwrap();
        let auto_replicated = Sharding::replicated(auto_mesh.clone(), 0);
        let scalar = ArrayType::scalar(DataType::F32);
        let changed = format!(
            "`{PLACE_OPERATION_NAME}` cannot change reduction state; use a reduction (or the corresponding collective) \
             instead",
        );
        let unreduced_auto = auto_replicated.clone().with_unreduced_axes(["x"]).unwrap();
        let reduced_auto = auto_replicated.clone().with_reduced_axes(["x"]).unwrap();
        let auto_inputs = [
            (
                "unreduced over an auto axis",
                Array::from_host_shards(
                    &session,
                    scalar.clone(),
                    auto.clone(),
                    Some(unreduced_auto.clone()),
                    [
                        (devices[0].id(), values_to_bytes(&[1f32]).as_slice()),
                        (devices[1].id(), values_to_bytes(&[2f32]).as_slice()),
                    ],
                )
                .unwrap(),
            ),
            (
                "reduced over an auto axis",
                Array::from_host_buffer(
                    &session,
                    scalar.clone(),
                    auto.clone(),
                    Some(reduced_auto),
                    values_to_bytes(&[3f32]),
                )
                .unwrap(),
            ),
        ];
        for (input_name, input) in auto_inputs {
            assert_eq!(
                (observe(&input).values, input.r#type().into_owned()),
                (vec![3.], scalar.clone().with_sharding(auto_replicated.clone()).unwrap())
            );
            let state = input.sharding().clone();
            let is_unreduced = !state.unreduced_axes().is_empty();
            let targets = [
                ("identity", Some(ArrayPlacement::mesh(auto.clone(), state.clone()).unwrap()), None),
                ("other devices", Some(ArrayPlacement::mesh(other_auto.clone(), state.clone()).unwrap()), None),
                (
                    "replicated",
                    Some(ArrayPlacement::mesh(auto.clone(), auto_replicated.clone()).unwrap()),
                    Some(changed.clone()),
                ),
                (
                    "device",
                    Some(ArrayPlacement::device(devices[2])),
                    is_unreduced.then(|| {
                        format!(
                            "`{PLACE_OPERATION_NAME}` cannot move a value with reduction state over mesh axes larger \
                             than one (i.e., `x`) onto a single device; use a reduction (or the corresponding \
                             collective) first",
                        )
                    }),
                ),
                ("memory", Some(ArrayPlacement::from(pinned)), None),
                ("none", None, None),
            ];
            for (target_name, target, rejection) in targets {
                let cell = format!("{input_name}, target {target_name}");
                let options =
                    || PlacementOptions::new(target.clone(), None::<ArrayPlacement>, None::<bool>, None::<bool>);
                let error = RefCell::new(None::<String>);
                let eager = session.place(input.clone(), options()).map(|output| observe(&output));
                let eager = eager.map_err(|error| error.to_string());
                match (&rejection, &eager) {
                    (Some(rejection), _) if eager.as_ref().err() != Some(rejection) => {
                        disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                    }
                    (None, Err(_)) => disagreements.push(format!("{cell}, eager: {eager:?}")),
                    (None, Ok(output)) if output.values != vec![3.] => {
                        disagreements.push(format!("{cell}, eager values: {:?} != [3.0]", output.values));
                    }
                    (None, Ok(output)) if target_name == "identity" && output.sharding != state => {
                        disagreements.push(format!("{cell}, eager sharding: {} != {state}", output.sharding));
                    }
                    _ => {}
                }

                let request = target.clone().map(|target| target.into_request(&session, 0)).transpose().unwrap();
                let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(&input.r#type(), Some(&state), request.as_ref(), None);
                let inferred = inferred.map_err(|error| Error::from(error).to_string());
                if inferred != eager.clone().map(|output| output.r#type) {
                    disagreements.push(format!("{cell}, type rule: {inferred:?} != eager: {eager:?}"));
                }

                let core = input.place(&request, None).map(|output| observe(&output));
                let core = core.map_err(|error| Error::from(error).to_string());
                let jvp = differentiate_at(input.clone())
                    .jvp(input.clone(), |x| record(&error, session.place(x, options())))
                    .map(|(primal, _)| observe(&primal))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                    |x: ArrayTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
                    input.r#type().into_owned(),
                );
                let staged = match (error.take(), traced) {
                    (Some(message), _) => Err(message),
                    (None, Err(error)) => Err(error.to_string()),
                    (None, Ok(traced)) => {
                        match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                            Ok(Value::Array(output)) => Ok(observe(&output)),
                            Ok(_) => Err("placing an array must produce an array".to_string()),
                            Err(error) => Err(error.to_string()),
                        }
                    }
                };
                let expected_staged = match target_name {
                    "identity" | "other devices" => Err(changed.clone()),
                    _ => eager.clone(),
                };
                for (path, result, expected) in
                    [("core", core, &eager), ("jvp", jvp, &eager), ("staged", staged, &expected_staged)]
                {
                    if &result != expected {
                        disagreements.push(format!("{cell}, {path}: {result:?} != {expected:?}"));
                    }
                }
            }
        }

        // The runtime sharding of an array replicated along an auto axis determines its shards, which its type does
        // not, so an expected source layout that overlaps them is rejected by the eager placement as forward mode
        // rejects it after it completes the source with that runtime placement, with the error of the type rule.
        let auto_matrix = Array::from_host_buffer(
            &session,
            ArrayType::new_static(DataType::F32, [4, 4]),
            auto.clone(),
            Some(Sharding::replicated(auto_mesh.clone(), 2)),
            values_to_bytes(&[42f32; 16]),
        )
        .unwrap();
        let source = ArrayPlacement::in_place().with_layout(Layout::Strided(StridedLayout::new(vec![8, 4])));
        let options = || PlacementOptions::defaults().with_source(source.clone());
        let source_request = source.clone().into_request(&session, 2).unwrap();
        let error = RefCell::new(None::<String>);
        let invalid = Err(format!(
            "`{PLACE_OPERATION_NAME}` source layout is invalid: strided layout stride 8 on axis 0 is smaller than the \
             16-byte span occupied by more minor axes and may alias array elements",
        ));
        let results = [
            ("eager", session.place(auto_matrix.clone(), options()).map(|_| ()).map_err(|error| error.to_string())),
            (
                "type rule",
                PlaceOperation::infer_output_type_allowing_uneven_splits(
                    &auto_matrix.r#type(),
                    Some(auto_matrix.sharding()),
                    None,
                    Some(&source_request),
                )
                .map(|_| ())
                .map_err(|error| Error::from(error).to_string()),
            ),
            (
                "core",
                auto_matrix
                    .place(&None::<PlacementRequest>, &Some(source_request.clone()))
                    .map(|_| ())
                    .map_err(|error| Error::from(error).to_string()),
            ),
            (
                "jvp",
                differentiate_at(auto_matrix.clone())
                    .jvp(auto_matrix.clone(), |x| record(&error, session.place(x, options())))
                    .map(|_| ())
                    .map_err(|other| error.take().unwrap_or(other.to_string())),
            ),
        ];
        for (path, result) in results {
            if result != invalid {
                disagreements.push(format!("auto source layout, {path}: {result:?} != {invalid:?}"));
            }
        }

        // A layout describes each shard, so the byte strides `[8, 4]` of a dense `f32[4, 2]` column shard of an
        // `f32[4, 4]` value that is split into two column shards pass the type rule (also with the runtime sharding of
        // the array), although they would overlap the rows of the whole value. No backend here can execute the accepted
        // type, and each says why: the CPU plugin cannot create strided buffers, which every path that places the array
        // reports alike, and the reference backend stores a whole value in one buffer, which it reports as a shard
        // layout that it cannot materialize.
        let columns =
            Sharding::new(double_mesh.clone(), vec![ShardingDimension::Replicated, ShardingDimension::sharded(["x"])])
                .unwrap();
        let input = upload(
            &[4, 4],
            Some(&columns),
            &double.0,
            Memory::Device,
            &(1..=16).map(|value| value as f32).collect::<Vec<_>>(),
        );
        let layout = Layout::Strided(StridedLayout::new(vec![8, 4]));
        let options = || PlacementOptions::to(ArrayPlacement::in_place().with_layout(layout.clone()));
        let eager = session
            .place(input.clone(), options())
            .map(|output| observe(&output))
            .map_err(|error| error.to_string());
        let error = RefCell::new(None::<String>);
        let traced: TracedProgram<ArrayType, ArrayType> = trace(
            |x: ArrayTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
            input.r#type().into_owned(),
        )
        .unwrap();
        let staged = match error.take() {
            Some(message) => Err(message),
            None => match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                Ok(Value::Array(output)) => Ok(observe(&output)),
                Ok(_) => Err("placing an array must produce an array".to_string()),
                Err(error) => Err(error.to_string()),
            },
        };
        let jvp = differentiate_at(input.clone())
            .jvp(input.clone(), |x| record(&error, session.place(x, options())))
            .map(|(primal, _)| observe(&primal))
            .map_err(|other| error.take().unwrap_or(other.to_string()));
        let request = ArrayPlacement::in_place().with_layout(layout.clone()).into_request(&session, 2).unwrap();
        let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(&input.r#type(), Some(input.sharding()), Some(&request), None);
        let output_type = input.r#type().into_owned().with_layout(layout.clone());
        if inferred.as_ref() != Ok(&output_type) {
            disagreements.push(format!("per-shard layout, type rule: {inferred:?} != {output_type:?}"));
        }
        let unsupported =
            Err("PJRT_Buffer_MemoryLayout_Type_Strides is not supported to be converted to a xla::Layout.".to_string());
        for (path, result) in [("eager", &eager), ("staged", &staged), ("jvp", &jvp)] {
            if result != &unsupported {
                disagreements.push(format!("per-shard layout, {path}: {result:?} != {unsupported:?}"));
            }
        }
        let values = (1..=16).map(|value| value as f32).collect::<Vec<_>>();
        let reference = CoreArray::from_elements(input.r#type().into_owned(), &values).unwrap();
        let reference = reference.place(&request, None).map_err(|error| error.to_string());
        let whole_value = Err(format!(
            "the reference `Array` backend stores a whole value in one buffer, so it cannot materialize the shard \
             layout of `{output_type}` for it: strided layout stride 8 on axis 0 is smaller than the 16-byte span occupied by \
             more minor axes and may alias array elements",
        ));
        if reference != whole_value {
            disagreements.push(format!("per-shard layout, reference: {reference:?} != {whole_value:?}"));
        }

        // Without a layout, a destination keeps the layout of a value while it still describes each destination shard,
        // and the type rule says the same. The CPU plugin cannot create the strided layouts above, so the case of a
        // layout that does not describe the gathered shard is covered by the type rule and the reference backend, and
        // here a tiled layout, which describes every shard, is gathered from the column shards onto one device.
        let column_major = Layout::Tiled(TiledLayout::new(vec![0, 1], vec![]));
        let tiled = PlacementOptions::to(ArrayPlacement::in_place().with_layout(column_major));
        let tiled = session.place(input.clone(), tiled).unwrap();
        let gather = || PlacementOptions::to(ArrayPlacement::device(devices[2]));
        let inferred = PlaceOperation::new(devices[2], None).infer_output_types(&[tiled.r#type().into_owned()], &[]);
        let inferred = inferred.map(|mut types| types.remove(0)).map_err(|error| error.to_string());
        let eager = session.place(tiled.clone(), gather()).map(|output| observe(&output));
        let eager = eager.map_err(|error| error.to_string());
        let core = tiled.place(&devices[2], None).map(|output| observe(&output));
        let core = core.map_err(|error| Error::from(error).to_string());
        let traced: TracedProgram<ArrayType, ArrayType> =
            trace(|x: ArrayTracer| session.place(x, gather()).unwrap(), tiled.r#type().into_owned()).unwrap();
        let staged = match traced.program().interpret_in_context(&session, Value::Array(tiled.clone())) {
            Ok(Value::Array(output)) => Ok(observe(&output)),
            Ok(_) => Err("placing an array must produce an array".to_string()),
            Err(error) => Err(error.to_string()),
        };
        for (path, result) in [("core", &core), ("staged", &staged)] {
            if result != &eager {
                disagreements.push(format!("gathered layout, {path}: {result:?} != eager: {eager:?}"));
            }
        }
        if eager.as_ref().map(|output| &output.r#type) != inferred.as_ref() {
            disagreements.push(format!("gathered layout, eager: {eager:?} != type rule: {inferred:?}"));
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }

    #[test]
    fn test_session_place_agrees_across_paths_for_manual_regions_and_host_values() {
        // The matrix of `test_session_place_agrees_across_eager_staged_compiled_and_transformed_paths` covers values
        // outside manual regions on every path, including reverse mode and batching. The values of this matrix (i.e.,
        // values inside a manual region, values that are moved into one, host values, and the values of a `shard_map`
        // body) are rejected wherever they would move, and reverse mode and batching of an eager value do not apply to
        // them, so they are compared on the paths that do (i.e., eager, core, staged, forward mode, and compiled).
        fn record<T>(error: &RefCell<Option<String>>, result: Result<T, Error>) -> Result<T, ProgramError> {
            result.map_err(|placement_error| {
                error.replace(Some(placement_error.to_string()));
                ProgramError::InvalidArgument { message: placement_error.to_string() }
            })
        }

        let client = cpu_client(4);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let pinned = Memory::Host { pinned: true };
        let observe =
            |array: &Array| (array.mesh().devices().to_vec(), array.r#type().into_owned(), shard_values(array));
        let mut disagreements = Vec::new();

        // Values inside a manual region (i.e., that vary along the manual axis `m`, or that are reduced over it) can
        // only stay on their devices, and nothing can be moved into such a region, so every target that moves a value
        // is rejected on every path with the error of the type rule, even a target that is the placement of the value.
        let manual_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let manual = DeviceMesh::new(manual_mesh.clone(), devices[..2].to_vec()).unwrap();
        let other_manual = DeviceMesh::new(manual_mesh.clone(), devices[2..].to_vec()).unwrap();
        let varying = Sharding::replicated(manual_mesh.clone(), 1).with_varying_manual_axes(["m"]).unwrap();
        let reduced = Sharding::replicated(manual_mesh.clone(), 1).with_reduced_axes(["m"]).unwrap();
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let varying_input = Array::from_host_shards(
            &session,
            vector.clone(),
            manual.clone(),
            Some(varying.clone()),
            [
                (devices[0].id(), values_to_bytes(&[1f32, 2.]).as_slice()),
                (devices[1].id(), values_to_bytes(&[10f32, 20.]).as_slice()),
            ],
        )
        .unwrap();
        let reduced_type = vector.clone().with_sharding(reduced.clone()).unwrap();
        let reduced_input =
            Array::from_host_buffer(&session, reduced_type, manual.clone(), None, values_to_bytes(&[1f32, 2.]))
                .unwrap();
        let plain =
            Array::from_host_buffer(&session, vector.clone(), None, None, values_to_bytes(&[1f32, 2.])).unwrap();
        let manual_error =
            "`place` cannot move values that vary along manual mesh axes or carry reduction state over them"
                .to_string();
        let region_targets = |sharding: &Sharding| {
            vec![
                ("device", Some(ArrayPlacement::device(devices[2])), Some(manual_error.clone())),
                ("own device", Some(ArrayPlacement::device(devices[0])), Some(manual_error.clone())),
                (
                    "other manual mesh",
                    Some(ArrayPlacement::mesh(other_manual.clone(), sharding.clone()).unwrap()),
                    Some(manual_error.clone()),
                ),
                (
                    "own placement",
                    Some(ArrayPlacement::mesh(manual.clone(), sharding.clone()).unwrap()),
                    Some(manual_error.clone()),
                ),
                ("memory", Some(ArrayPlacement::from(pinned)), None),
                ("none", None, None),
            ]
        };
        let inputs = [
            ("varying along a manual axis", varying_input, region_targets(&varying)),
            ("reduced over a manual axis", reduced_input, region_targets(&reduced)),
            (
                "outside any manual region",
                plain,
                vec![(
                    "varying along a manual axis",
                    Some(ArrayPlacement::mesh(manual.clone(), varying.clone()).unwrap()),
                    Some(manual_error.clone()),
                )],
            ),
        ];
        for (input_name, input, targets) in inputs {
            for (target_name, target, rejection) in targets {
                let cell = format!("{input_name}, target {target_name}");
                let options =
                    || PlacementOptions::new(target.clone(), None::<ArrayPlacement>, None::<bool>, None::<bool>);
                let error = RefCell::new(None::<String>);
                let eager = session.place(input.clone(), options()).map(|output| observe(&output));
                let eager = eager.map_err(|error| error.to_string());
                if let Some(rejection) = &rejection
                    && eager.as_ref().err() != Some(rejection)
                {
                    disagreements.push(format!("{cell}, eager: {eager:?} != rejection: {rejection:?}"));
                }
                if rejection.is_none() && eager.is_err() {
                    disagreements.push(format!("{cell}, eager: {eager:?}"));
                }
                let request = target.clone().map(|target| target.into_request(&session, 1)).transpose().unwrap();
                let core = input.place(&request, None).map(|output| observe(&output));
                let core = core.map_err(|error| Error::from(error).to_string());
                let traced: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                    |x: ArrayTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
                    input.r#type().into_owned(),
                );
                let staged = match (error.take(), traced) {
                    (Some(message), _) => Err(message),
                    (None, Err(error)) => Err(error.to_string()),
                    (None, Ok(traced)) => {
                        match traced.program().interpret_in_context(&session, Value::Array(input.clone())) {
                            Ok(Value::Array(output)) => Ok(observe(&output)),
                            Ok(_) => Err("placing an array must produce an array".to_string()),
                            Err(error) => Err(error.to_string()),
                        }
                    }
                };
                let jvp = differentiate_at(input.clone())
                    .jvp(input.clone(), |x| record(&error, session.place(x, options())))
                    .map(|(primal, _)| observe(&primal))
                    .map_err(|other| error.take().unwrap_or(other.to_string()));
                for (path, result) in [("core", core), ("staged", staged), ("jvp", jvp)] {
                    if result != eager {
                        disagreements.push(format!("{cell}, {path}: {result:?} != eager: {eager:?}"));
                    }
                }
            }
        }

        // A host value is uploaded onto a destination as the placement of a value without a tracked sharding in device
        // memory, so it is rejected or placed as an upload of the same value without a mesh is (a host value ignores
        // sources, so none is given), with the first error of the type rule for such a value, except that both split a
        // dimension unevenly. A traced value of that type follows the type rule with the even split included.
        let explicit = two_device_mesh(client, MeshAxisType::Explicit);
        let explicit_sharding =
            |dimensions: Vec<ShardingDimension>| Sharding::new(explicit.logical_mesh().clone(), dimensions).unwrap();
        let rank_2 = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let host_targets = [
            (
                "unreduced",
                ArrayPlacement::mesh(
                    explicit.clone(),
                    explicit_sharding(vec![ShardingDimension::Replicated]).with_unreduced_axes(["x"]).unwrap(),
                )
                .unwrap(),
            ),
            (
                "reduced",
                ArrayPlacement::mesh(
                    explicit.clone(),
                    explicit_sharding(vec![ShardingDimension::Replicated]).with_reduced_axes(["x"]).unwrap(),
                )
                .unwrap(),
            ),
            ("varying along a manual axis", ArrayPlacement::mesh(manual.clone(), varying.clone()).unwrap()),
            ("device with a layout of rank 2", ArrayPlacement::device(devices[1]).with_layout(rank_2.clone())),
            (
                "rank 2",
                ArrayPlacement::mesh(explicit.clone(), explicit_sharding(vec![ShardingDimension::Replicated; 2]))
                    .unwrap(),
            ),
            (
                "unconstrained",
                ArrayPlacement::mesh(explicit.clone(), explicit_sharding(vec![ShardingDimension::Unconstrained]))
                    .unwrap(),
            ),
            (
                "uneven split",
                ArrayPlacement::mesh(explicit.clone(), explicit_sharding(vec![ShardingDimension::sharded(["x"])]))
                    .unwrap(),
            ),
            ("device", ArrayPlacement::device(devices[1])),
            ("memory", ArrayPlacement::from(pinned)),
        ];
        let host_type = ArrayType::new_static(DataType::F32, [3]);
        let host = CoreArray::from_elements(host_type.clone(), &[1f32, 2., 3.]).unwrap();
        for (target_name, target) in host_targets {
            let cell = format!("host value, target {target_name}");
            let options = || PlacementOptions::to(target.clone());
            let uploaded =
                Array::from_host_buffer(&session, host_type.clone(), None, None, values_to_bytes(&[1f32, 2., 3.]))
                    .unwrap();
            let host = session.place(host.clone(), options()).map(|output| (observe(&output), output.is_committed()));
            let host = host.map_err(|error| error.to_string());
            let array = session.place(uploaded, options()).map(|output| (observe(&output), output.is_committed()));
            let array = array.map_err(|error| error.to_string());
            if host != array {
                disagreements.push(format!("{cell}: {host:?} != array: {array:?}"));
            }
            let request = target.clone().into_request(&session, 1).unwrap();
            let inferred = PlaceOperation::infer_output_type_allowing_uneven_splits(&host_type, None, Some(&request), None);
            let inferred = inferred.map_err(|error| Error::from(error).to_string());
            if let (Err(host), Err(inferred)) = (&host, &inferred)
                && host != inferred
            {
                disagreements.push(format!("{cell}: {host:?} != type rule: {inferred:?}"));
            }
            if host.is_ok() != inferred.is_ok() {
                disagreements.push(format!("{cell}: {host:?} != type rule: {inferred:?}"));
            }
            let error = RefCell::new(None::<String>);
            let _: Result<TracedProgram<ArrayType, ArrayType>, _> = trace(
                |x: ArrayTracer| record(&error, session.place(x.clone(), options())).unwrap_or(x),
                host_type.clone(),
            );
            let traced = error.take();
            let expected = match (target_name, &host) {
                ("uneven split", _) => Some(
                    "`place` target sharding splits dimension #0 with extent 3 into 2 shards, which do not divide it \
                     evenly"
                        .to_string(),
                ),
                (_, Err(error)) => Some(error.clone()),
                (_, Ok(_)) => None,
            };
            if traced != expected {
                disagreements.push(format!("{cell}, traced: {traced:?} != {expected:?}"));
            }
        }

        // Inside a `shard_map` body, a source that names devices (i.e., the placement of the local value over the
        // devices of the region, or the sole device of a region of one device) would become the target of the
        // transposed placement of an invariant local value (i.e., one that does not vary along the manual axis), which
        // would move its cotangent, so it is rejected when the function is compiled forward or applied eagerly, and the
        // program of reverse mode, which keeps only the transposed placement, is rejected for its target. The cotangent
        // of a varying local value stays where it lives (refer to `PlaceOperation::transposes_in_place`), so every
        // source compiles and executes on every path for it, as does every other source for an invariant value.
        let pinned_seed = |r#type: &ArrayType| r#type.clone().with_memory(pinned);
        let source_error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: the source of a value inside a manual region cannot name devices \
             unless the placement moves nothing and the value varies along the manual axes of the region or carries \
             reduction state over them, because transposition would move the cotangent of that value to them",
        );
        let moved_error = format!(
            "cannot lower `{PLACE_OPERATION_NAME}`: a value inside a manual region cannot be moved to another \
             placement",
        );
        for count in [1usize, 2] {
            let program_logical_mesh =
                LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Explicit).unwrap()]).unwrap();
            let program_mesh = DeviceMesh::new(program_logical_mesh.clone(), devices[..count].to_vec()).unwrap();
            let region_mesh = LogicalMesh::new(vec![MeshAxis::new("x", count, MeshAxisType::Manual).unwrap()]).unwrap();
            let region_devices = DeviceMesh::new(region_mesh.clone(), devices[..count].to_vec()).unwrap();
            let r#type = ArrayType::new_static(DataType::F32, [4])
                .with_sharding(Sharding::replicated(program_logical_mesh.clone(), 1))
                .unwrap();
            for is_varying in [false, true] {
                // A specification that splits the value along the manual axis makes the local value vary along it.
                let (specification, local_sharding, output_type) = if is_varying {
                    let split =
                        |mesh: &LogicalMesh| Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]);
                    let varying = Sharding::replicated(region_mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap();
                    let output_type = r#type.clone().with_sharding(split(&program_logical_mesh).unwrap()).unwrap();
                    (split(&region_mesh).unwrap(), varying, output_type)
                } else {
                    let specification = Sharding::replicated(region_mesh.clone(), 1);
                    (specification.clone(), specification, r#type.clone())
                };
                let mut sources = vec![
                    ("none", None, false),
                    ("memory", Some(ArrayPlacement::from(Memory::Device)), false),
                    (
                        "layout",
                        Some(ArrayPlacement::in_place().with_layout(Layout::Tiled(TiledLayout::new(vec![0], vec![])))),
                        false,
                    ),
                    (
                        "resolved",
                        Some(ArrayPlacement::mesh(region_devices.clone(), local_sharding.clone()).unwrap()),
                        !is_varying,
                    ),
                ];
                if count == 1 {
                    sources.push(("device", Some(ArrayPlacement::device(devices[0])), !is_varying));
                }
                for (source_name, source, is_rejected) in sources {
                    for (target_name, target) in [("memory", Some(ArrayPlacement::from(pinned))), ("none", None)] {
                        let input_name = if is_varying { "varying" } else { "invariant" };
                        let cell = format!(
                            "`shard_map` over {count} devices, {input_name} value, target {target_name}, source \
                             {source_name}",
                        );
                        let options =
                            || PlacementOptions::new(target.clone(), source.clone(), None::<bool>, None::<bool>);
                        let body = |local| record(&RefCell::new(None), session.place(local, options()));
                        let forward: Result<CompiledFunction<ArrayType, ArrayType>, _> = compile(
                            |x: CompileTracer| {
                                let options = ShardMapOptions::new();
                                shard_map(
                                    body,
                                    x,
                                    region_mesh.clone(),
                                    specification.clone(),
                                    specification.clone(),
                                    options,
                                )
                                .unwrap()
                            },
                            r#type.clone(),
                            &session,
                            program_mesh.clone(),
                        );
                        let seed_type = if target.is_some() { pinned_seed(&output_type) } else { output_type.clone() };
                        let reverse: Result<CompiledFunction<(ArrayType, ArrayType), ArrayType>, _> = compile(
                            |(x, seed): (CompileTracer, CompileTracer)| {
                                let (_, pullback) = differentiate_at(x.into_value())
                                    .vjp(|x| {
                                        let x = ValueProjection::<ArrayType>::into_projected(x)?;
                                        let output = shard_map(
                                            |local| record(&RefCell::new(None), session.place(local, options())),
                                            x,
                                            region_mesh.clone(),
                                            specification.clone(),
                                            specification.clone(),
                                            ShardMapOptions::new(),
                                        )
                                        .map_err(|error| {
                                            ProgramError::InvalidArgument { message: error.to_string() }
                                        })?;
                                        Ok(output.into_value())
                                    })
                                    .unwrap();
                                let cotangent = pullback.apply_materialized(seed.into_value()).unwrap();
                                ValueProjection::<ArrayType>::into_projected(cotangent).unwrap()
                            },
                            (r#type.clone(), seed_type),
                            &session,
                            program_mesh.clone(),
                        );
                        let input = Array::from_host_buffer(
                            &session,
                            r#type.clone(),
                            program_mesh.clone(),
                            None,
                            values_to_bytes(&[1f32, 2., 3., 4.]),
                        )
                        .unwrap();
                        let eager = shard_map(
                            |local| record(&RefCell::new(None), session.place(local, options())),
                            input,
                            region_mesh.clone(),
                            specification.clone(),
                            specification.clone(),
                            ShardMapOptions::new(),
                        );
                        let results = [
                            ("forward", forward.map(|_| ()).map_err(|error| error.to_string())),
                            ("reverse", reverse.map(|_| ()).map_err(|error| error.to_string())),
                            ("eager", eager.map(|_| ()).map_err(|error| error.to_string())),
                        ];
                        for (path, result) in results {
                            let expected = match (is_rejected, path) {
                                (false, _) => Ok(()),
                                (true, "reverse") => Err(moved_error.clone()),
                                (true, _) => Err(source_error.clone()),
                            };
                            if result != expected {
                                disagreements.push(format!("{cell}, {path}: {result:?} != {expected:?}"));
                            }
                        }
                    }
                }
            }
        }

        assert!(disagreements.is_empty(), "{} disagreements:\n{}", disagreements.len(), disagreements.join("\n"));
    }
}
