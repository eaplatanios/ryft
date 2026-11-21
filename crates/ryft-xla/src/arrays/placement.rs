use std::sync::{Arc, Weak};

use half::{bf16, f16};

use ryft_core::{
    Array as CoreArray, ArrayType, BatchingPolicy, BatchingTracer, Complex, Context, Device, DeviceMesh,
    DifferentiationPolicy, DifferentiationTracer, Dimension, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
    Parameter, Parameterized, ParameterizedFamily, PartialTracer, PlaceDispatch, Placed, Placement, ProjectedValue,
    Shape, Sharding, ShardingDimension, Tracer, Typed, Value as CoreValue, check_sharding, f4e2m1fn, f6e2m3fn,
    f6e3m2fn, f8e3m4, f8e4m3, f8e4m3b11fnuz, f8e4m3fn, f8e4m3fnuz, f8e5m2, f8e5m2fnuz, f8e8m0fnu, i1, i2, i4, u1, u2,
    u4,
};
use ryft_macros::Parameter;
use ryft_pjrt::{Buffer, Error as PjrtError};

use crate::arrays::shards::ArrayShard;
use crate::arrays::{Array, ArrayError};
use crate::{Error, FromPjrt, Session, Value};

#[cfg(doc)]
use ryft_core::Place;

// TODO(eaplatanios): Review from here onwards.

/// Placement of an [`Array`] on devices. It specifies a destination for [`Session::place`],
/// [`Array::to_placement`], and [`Array::into_placement`], or an expected source in [`PlacementOptions`]. Its
/// [`ArrayPlacementTarget`] selects the devices and the sharding, and its optional memory and layout select the
/// storage of each shard. Destination and source trees in [`PlacementOptions`] can use `Some(placement)` and
/// `None` together. A `None` leaf omits the option for its branch.
///
/// An [`ArrayPlacement`] is a request that may depend on a [`Session`] (e.g., on its default mesh) and on the array
/// that it applies to. It resolves against them to a [`Placement`], which names the concrete devices, the sharding
/// over them, and the selected memory and layout, and which is what a [`PlaceOperation`](ryft_core::PlaceOperation)
/// records.
///
/// Use [`Self::with_layout`] and [`Self::with_memory`] to select the storage of each shard. Host uploads use
/// device memory and the backend's default layout when these options are omitted. A destination that is a bare
/// device without a memory selects the default memory of that device, as JAX's `device_put` does, whereas existing
/// arrays keep their memory under a mesh or sharding destination without a memory. An explicit layout applies to the
/// destination shard buffers.
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

    /// Sets the physical layout of each shard, or the expected declared layout when used as a source.
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
    /// [`ArrayPlacementTarget::Device`] target resolves to a mesh with one auto axis named `device` of size 1 and to
    /// the replicated sharding over that mesh. The resolved placement keeps the [batch record](Self::batch_shardings)
    /// of this placement, so a placement that was converted from a resolved [`Placement`] resolves back to it.
    ///
    /// # Errors
    ///
    /// Returns [`SessionError::MissingMesh`](crate::SessionError::MissingMesh) when an
    /// [`ArrayPlacementTarget::Sharding`] target needs a default mesh and the session has none, or
    /// [`ShardingError::MeshMismatch`](ryft_core::ShardingError::MeshMismatch) when the sharding and the concrete mesh
    /// have different logical topologies. The batch record was validated when it was recorded, so it adds no errors.
    pub(crate) fn resolve(self, session: &Session, rank: usize) -> Result<Placement, Error> {
        let (mesh, sharding) = match self.target {
            ArrayPlacementTarget::Device(device) => {
                let logical_mesh = LogicalMesh::new(vec![MeshAxis::new("device", 1, MeshAxisType::Auto)?])?;
                let mesh = DeviceMesh::new(logical_mesh, vec![device])?;
                let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
                (mesh, sharding)
            }
            ArrayPlacementTarget::Mesh { mesh, sharding } => (mesh, sharding),
            ArrayPlacementTarget::Sharding(sharding) => (session.device_mesh()?.clone(), sharding),
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

    /// Resolves this placement against `session` to the [`Placement`] that it selects as the destination of `array`,
    /// as [`Self::resolve`] does for a value of its rank. An [`ArrayPlacementTarget::Device`] target names a device and
    /// not a mesh, so an array that already lives on exactly that device keeps its own mesh, and like every other
    /// device target, it selects the replicated sharding over that mesh (i.e., an ordinary array on that device, which
    /// completes any pending reduction along the axes of that mesh). A device target without a memory selects the
    /// default memory of its device (refer to [`Self::into_destination`]).
    pub(crate) fn resolve_for(self, session: &Session, array: &Array) -> Result<Placement, Error> {
        let sharding = Sharding::replicated(array.mesh().logical_mesh().clone(), array.sharding().rank());
        self.into_destination().resolve_from(session, array.mesh(), &sharding)
    }

    /// Returns this placement as a destination. An [`ArrayPlacementTarget::Device`] destination without a memory
    /// selects the [default memory](Memory::Device) of its device, as JAX's `device_put` does for a bare device, so
    /// that a value moves out of any other memory (e.g., pinned host memory). Every other placement is returned as
    /// it is, so a mesh or sharding destination without a memory keeps the memory of the value. A source is never
    /// converted, because it only constrains the memory of a value when it names one.
    fn into_destination(self) -> Self {
        match (&self.target, self.memory) {
            (ArrayPlacementTarget::Device(_), None) => self.with_memory(Memory::default()),
            _ => self,
        }
    }

    /// Resolves this placement against `session` to the [`Placement`] that it selects for a value that lives on
    /// `mesh` with `sharding`. This is [`Self::resolve`], except that an [`ArrayPlacementTarget::Device`] target names
    /// a device and not a mesh, so when `mesh` consists of exactly that device, it resolves to `mesh` with `sharding`
    /// (e.g., to the current placement of a value, which a device source must accept, or which a device destination
    /// of a traced value keeps, because a staged placement cannot change the reduction state of its input).
    fn resolve_from(self, session: &Session, mesh: &DeviceMesh, sharding: &Sharding) -> Result<Placement, Error> {
        if let ArrayPlacementTarget::Device(device) = &self.target
            && mesh.devices() == [*device]
        {
            let target = ArrayPlacementTarget::Mesh { mesh: mesh.clone(), sharding: sharding.clone() };
            return Self { target, ..self }.resolve(session, sharding.rank());
        }
        self.resolve(session, sharding.rank())
    }

    /// Resolves this placement against `session` to the [`Placement`] that it selects for a value of rank `rank` that
    /// has no concrete placement (e.g., a traced value inside a compiled function, whose devices are not known while
    /// it is traced). This is [`Self::resolve`], except that an [`ArrayPlacementTarget::Device`] target that is the
    /// only device of the [default mesh](Session::device_mesh) of `session` resolves to that mesh with replicated
    /// sharding, as [`Self::resolve_from`] does for a value that lives on that mesh. A function compiled for that
    /// device uses that mesh, whose axis can have any name, whereas the mesh that [`Self::resolve`] creates for a
    /// device has an axis named `device`, which the compiler would reject as a mesh other than that of the function.
    fn resolve_unplaced(self, session: &Session, rank: usize) -> Result<Placement, Error> {
        // The only error of `Session::device_mesh` reports that the session has no default mesh, which leaves nothing
        // to resolve the device against.
        if let ArrayPlacementTarget::Device(device) = &self.target
            && let Some(mesh) = session.device_mesh().ok()
            && mesh.devices() == [*device]
        {
            let sharding = Sharding::replicated(mesh.logical_mesh().clone(), rank);
            return self.resolve_from(session, mesh, &sharding);
        }
        self.resolve(session, rank)
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

impl From<Placement> for ArrayPlacement {
    #[inline]
    fn from(value: Placement) -> Self {
        let batch_shardings = value.batch_shardings().to_vec();
        let (mesh, sharding, layout, memory) = value.into_parts();
        Self { target: ArrayPlacementTarget::Mesh { mesh, sharding }, layout, memory, batch_shardings }
    }
}

/// Devices and sharding that an [`ArrayPlacement`] selects.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ArrayPlacementTarget {
    /// Commit the value to one concrete device with replicated sharding. As a destination without a memory, it also
    /// selects the default memory of that device (e.g., it moves a value out of pinned host memory), as JAX's
    /// `device_put` does. As a source without a memory, it accepts a value in any memory of that device.
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
}

/// A destination or source leaf in the tree prefixes of [`PlacementOptions`]. Resolved [`Placement`]s,
/// [`ArrayPlacement`] requests, and bare [`Device`]s and [`Sharding`]s (which convert into [`ArrayPlacement`]
/// requests) are accepted, as are their [`Option`]s. A [`None`] leaf omits the option for its branch of the input
/// tree.
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

/// Options of [`Session::place`].
///
/// Each field follows JAX's tree prefix semantics, so when a field is present, its structure is broadcast over the
/// input tree and applied leafwise. Destination and source prefixes accept [`PlaceableParameter`]s, including
/// [`Placement`], [`ArrayPlacement`], and their optional forms. Each prefix has one leaf type. To mix resolved
/// placements and requests within the same prefix, convert the resolved leaves with [`ArrayPlacement::from`].
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

    /// Aliasing tree prefix. With `Some(false)`, a resulting array shares no buffer with any other array, which
    /// copies the buffers that it would otherwise share. Concrete [`Array`] leaves apply it when they are placed, and
    /// traced leaves stage their placements with a forced copy, which applies it when those placements execute on
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
    pub(crate) fn into_parts(self) -> (Option<Target>, Option<Source>, Option<Donate>, Option<MayAlias>) {
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
    /// Creates [`Session::place`] options with only the destination tree prefix `target` (e.g., a [`Device`], an
    /// [`ArrayPlacement`], or a tuple of optional placements). The other options are unset, and their types are fixed
    /// as in [`Self::defaults`], so that this needs no generic type annotations.
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
/// [`PlaceOperation`](ryft_core::PlaceOperation) whose copy policy follows the `may_alias` option of the leaf), and
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

    /// Places `value` using the provided leafwise placement options. [`Session::place`] validates the options of
    /// each leaf before it places any leaf, so implementations can assume that `donate` is `false` whenever
    /// `may_alias` is `Some(true)`.
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
    /// concrete leaf that does not live where `source` says (e.g., [`ArrayError::SourcePlacementMismatch`]), and a
    /// placement that fails to execute or to stage (e.g., a failed transfer, or a target that the rules of
    /// [`PlaceOperation`](ryft_core::PlaceOperation) reject for a traced leaf). The leaf is consumed either way, and
    /// [`Session::place`] returns the first error of any leaf as its own.
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
        let (element_type, bytes) = (host.r#type().data_type(), host.logical_bytes());
        let committed = target.is_some();
        let placement = match target {
            Some(target) => target.into_destination().resolve(session, shape.len())?,
            None => {
                let device = client
                    .addressable_devices()?
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::from(ArrayError::MissingDefaultDevice))?;
                ArrayPlacement::device(Device::from_pjrt(device)?).resolve(session, shape.len())?
            }
        };
        let r#type = ArrayType::new(element_type, Shape::new(shape.iter().copied().map(Dimension::Static).collect()))
            .with_memory(placement.memory().unwrap_or_default())
            .with_layout(placement.layout().cloned());
        let (mesh, sharding) = (placement.mesh().clone(), placement.sharding().clone());
        Ok(Array::from_host_buffer(session, r#type, mesh, Some(sharding), bytes.as_slice())?.with_committed(committed))
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
        if let Some(source) = source {
            let expected = source.resolve_from(session, value.mesh(), value.sharding())?;
            if expected.mesh() != value.mesh() || expected.sharding() != value.sharding() {
                return Err(Error::from(ArrayError::SourcePlacementMismatch {
                    expected_mesh: expected.mesh().clone(),
                    expected_sharding: expected.sharding().clone(),
                    actual_mesh: value.mesh().clone(),
                    actual_sharding: value.sharding().clone(),
                }));
            }
            let actual_memory = value.r#type().memory();
            if let Some(expected) = expected.memory().filter(|&memory| memory != actual_memory) {
                return Err(ArrayError::SourceMemoryMismatch { expected, actual: actual_memory }.into());
            }
            if let Some(expected) = expected.layout().filter(|&layout| value.layout() != Some(layout)) {
                let expected = expected.clone();
                return Err(ArrayError::SourceLayoutMismatch { expected, actual: value.layout().cloned() }.into());
            }
        }

        // A forced copy must not return a buffer of this leaf, even one that only the result holds once the placement
        // consumes the leaf. Weak handles of those buffers keep a donation of the leaf possible.
        let buffers: Vec<_> = match may_alias {
            Some(false) => value.addressable_shards().filter_map(ArrayShard::buffer).map(Arc::downgrade).collect(),
            _ => Vec::new(),
        };
        let output = match target {
            Some(target) => {
                let target = target.resolve_for(session, &value)?;
                value.run_placement_dispatch(session, &target, donate)?
            }
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
    /// A tree has one leaf type, so a tree that mixes host values and [`Array`]s holds them as [`HostOrArray`]s. An
    /// input without leaves (e.g., an empty [`Vec`]) is returned as it is, whatever options are provided, because the
    /// leaf paths of a tree describe no subtree without leaves, so they cannot tell a valid prefix of such an input
    /// from an invalid one. For the same reason, a prefix leaf for a subtree without leaves of an input with other
    /// leaves (e.g., one of two destinations for an empty [`Vec`] and a scalar) is reported as unused.
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
    /// An [`ArrayPlacementTarget::Device`] destination makes an [`Array`] leaf an ordinary replicated array on that
    /// device, also when the leaf already lives there, in which case it keeps its mesh but loses any pending reduction
    /// along the axes of that mesh, as in JAX. Without a memory, such a destination also selects the default memory of
    /// its device for every leaf (e.g., it moves an array out of pinned host memory), as JAX's `device_put` does for a
    /// device, whereas a mesh or sharding destination without a memory keeps the memory of the leaf.
    ///
    /// Inside a batched function, the placement of a batch item (refer to [`Placed`]) records how its batch is
    /// distributed. A destination or source that is such a [`Placement`] keeps that record (refer to
    /// [`ArrayPlacement::batch_shardings`]), so that placing a batch item on its own placement leaves every batch item
    /// on the devices that hold it.
    ///
    /// Traced leaves (e.g., the inputs of a function that is compiled, differentiated, or batched) stay traced, and
    /// their placements resolve against this session and are recorded through the [`PlaceDispatch`] primitive of the
    /// [`Place`] capability, so the same call works inside those transformations. A traced leaf follows the rules of
    /// [`PlaceOperation`](ryft_core::PlaceOperation), so its placement cannot change its reduction state, whereas
    /// placing an [`Array`] leaf combines the values along unreduced axes when the destination requires it.
    ///
    /// A traced leaf without a destination stages a placement without a target, which leaves the value where it is
    /// (i.e., its devices, sharding, memory, and layout are unchanged), as JAX's `device_put` does without a device.
    /// That placement records the provided source. Without one, it records where the value concretely lives wherever
    /// it executes on a value that lives somewhere (e.g., when a differentiated function is evaluated on concrete
    /// arrays, refer to [`Placed`]). Reverse mode differentiation moves the cotangent back to that source. Such a
    /// placement is staged for every traced leaf, including one without a concrete placement and without a source, so
    /// that its copy policy applies wherever the staged program executes on concrete arrays.
    ///
    /// A placement inside a compiled function must target the device mesh that the function is compiled for (refer
    /// to [`LoweringError::PlacementOutsideProgramMesh`](crate::LoweringError::PlacementOutsideProgramMesh)), so a
    /// compiled function cannot move a value to other devices. The devices of a traced leaf are not known there, so
    /// an [`ArrayPlacementTarget::Device`] destination or source resolves to the default mesh of this session when
    /// that device is the only device of that mesh, which lets a function compiled for that mesh place values on its
    /// sole device. Any other device resolves to a mesh of its own with a single axis named `device`.
    ///
    /// The donation option is only applied to concrete [`Array`] leaves, and it is ignored for host leaves and traced
    /// leaves. The aliasing option is applied to concrete [`Array`] leaves when they are placed. With `Some(false)`,
    /// the result holds no buffer of the leaf, even when the leaf was the only handle of its buffers, and the buffers
    /// that the placement creates are not copied again. A traced leaf whose `may_alias` option is `Some(false)` stages
    /// its placement with a forced copy (refer to
    /// [`PlaceOperation::with_may_alias`](ryft_core::PlaceOperation::with_may_alias)), so that when a batched or
    /// differentiated function is evaluated on concrete arrays, its result shares no buffer with any other array, as
    /// in JAX. A compiled function has no observable buffer identity, so the option has no effect inside one. Host
    /// leaves ignore the option, because their uploads never share buffers. A leaf cannot be donated while its
    /// `may_alias` option is `Some(true)`, which this function reports with [`ArrayError::DonatedArrayMayAlias`] before
    /// it places any leaf, as JAX does.
    ///
    /// The source option is validated against the current placement, memory, and layout of an [`Array`] leaf, and
    /// the call fails when they differ. For a traced leaf, it states where the value is expected to live, which
    /// transposition uses to send cotangents back, and nothing compares it with the devices that the value actually
    /// lives on when the placement executes.
    ///
    /// # Parameters
    ///
    ///   - `x`: Leaf or tree of leaves to place.
    ///   - `options`: Destination, expected source, donation, and aliasing options for the leaves of `x`.
    pub fn place<
        P: Placeable,
        Input: Parameterized<P>,
        TargetPlacement: PlaceableParameter,
        Target: Parameterized<TargetPlacement>,
        SourcePlacement: PlaceableParameter,
        Source: Parameterized<SourcePlacement>,
        Donate: Parameterized<bool>,
        MayAlias: Parameterized<Option<bool>>,
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
    {
        let structure = x.parameter_structure();
        let leaf_count = structure.parameter_count();

        // The options apply to no leaf of an input without leaves, so they are not broadcast (which would reject every
        // option prefix that has a leaf), and the input structure is returned as it is.
        if leaf_count == 0 {
            return Input::To::<P::Output>::from_parameters(structure, Vec::new()).map_err(Into::into);
        }

        let (target, source, donate, may_alias) = options.into_parts();

        let target_values = match target {
            Some(target) => Input::To::<Option<ArrayPlacement>>::from_broadcasted_named_parameters(
                structure.clone(),
                target.into_named_parameters().map(|(path, placement)| (path, placement.into_optional_placement())),
            )?
            .into_parameters()
            .collect::<Vec<_>>(),
            None => vec![None; leaf_count],
        };
        let source_values = match source {
            Some(source) => Input::To::<Option<ArrayPlacement>>::from_broadcasted_named_parameters(
                structure.clone(),
                source.into_named_parameters().map(|(path, placement)| (path, placement.into_optional_placement())),
            )?
            .into_parameters()
            .collect::<Vec<_>>(),
            None => vec![None; leaf_count],
        };
        let donate_values = match donate {
            Some(donate) => {
                Input::To::<bool>::from_broadcasted_named_parameters(structure.clone(), donate.into_named_parameters())?
                    .into_parameters()
                    .collect::<Vec<_>>()
            }
            None => vec![false; leaf_count],
        };
        let may_alias_values = match may_alias {
            Some(may_alias) => Input::To::<Option<bool>>::from_broadcasted_named_parameters(
                structure.clone(),
                may_alias.into_named_parameters(),
            )?
            .into_parameters()
            .collect::<Vec<_>>(),
            None => vec![None; leaf_count],
        };

        // Validate the options of every leaf before placing any leaf, so that an invalid request has no effects.
        if let Some(index) = donate_values
            .iter()
            .zip(&may_alias_values)
            .position(|(&donate, &may_alias)| donate && may_alias == Some(true))
        {
            let path = x.parameter_paths().nth(index).unwrap();
            return Err(ArrayError::DonatedArrayMayAlias { path }.into());
        }

        let mut output_parameters = Vec::with_capacity(leaf_count);
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

/// Places the traced `value` through its [`PlaceDispatch`] primitive, with `target` and `source` resolved against
/// `session`. This implements [`Placeable::place`] for every traced value type, so that they all handle the placement
/// options identically (refer to [`Session::place`] for the resulting behavior).
///
/// A traced value that wraps a concrete array lives where that array lives (refer to [`Placed`]), so its placements
/// resolve for that placement (refer to [`ArrayPlacement::resolve_from`]), and without a source, it records that
/// placement as its source. The devices of any other traced value are not known, so its placements resolve for the
/// rank of its type (which is the rank of each batch item for a batched value) with
/// [`ArrayPlacement::resolve_unplaced`]. Without a target, the value is placed without one, which leaves it where it
/// is, with the resolved source if one is provided. Otherwise, the rules of the staged placement infer its source
/// wherever they place a value that concretely lives somewhere (e.g., when a batched or differentiated function is
/// evaluated on concrete arrays), so that reverse mode differentiation moves the cotangent back to that source.
///
/// With `may_alias` set to `Some(false)`, the placement forces a copy (refer to
/// [`PlaceOperation::with_may_alias`](ryft_core::PlaceOperation::with_may_alias)), also without a target and a source,
/// so that its result shares no buffer with any other array once it executes on concrete arrays. Any other `may_alias`
/// stages the placement with the default copy policy. The donation option only applies to concrete [`Array`] leaves,
/// so it is ignored here.
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
    let current = value.placement();
    let rank = value.r#type().rank();
    let resolve = |placement: ArrayPlacement| match &current {
        Some(current) => placement.resolve_from(session, current.mesh(), current.sharding()),
        None => placement.resolve_unplaced(session, rank),
    };
    let source = source.map(resolve).transpose()?;
    let (target, source) = match target {
        // Without a source, the value records where it concretely lives as its source, if anywhere, as with `Place`.
        Some(target) => (Some(resolve(target.into_destination())?), source.or(current)),
        None => (None, source),
    };
    Ok(V::Dispatch::place(&value, target.as_ref(), source, may_alias != Some(false))?)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        BatchAxis, DataType, DeviceMesh, Error as CoreError, Exp, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        Sharding, ShardingDimension, ShardingError, TiledLayout, Typed, batch, differentiate_at,
    };
    use ryft_pjrt::BufferType;

    use crate::sessions::tests::{domain_mesh, two_device_mesh};
    use crate::tests::{cpu_client, device_mesh_2x2, other_cpu_client, values_from_bytes, values_to_bytes};
    use crate::{
        ArrayTracer, CompileTracer, CompiledFunction, SessionError, ToPjrt, TracedProgram, Value, compile, trace,
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
        // names its axis differently from the mesh that a device target resolves to, and it replicates the array
        // over that mesh, also when the array is unreduced along its axis.
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

        // Any other device target resolves like it does for a value of the rank of the array.
        let other = ArrayPlacement::device(devices[1]).with_memory(memory);
        assert_eq!(other.clone().resolve_for(&session, &array).unwrap(), other.resolve(&session, 1).unwrap());

        // A device target without a memory selects the default memory of its device, as JAX's `device_put` does,
        // whereas a sharding target without a memory selects none, so the array keeps its own memory.
        assert_eq!(
            ArrayPlacement::device(devices[0]).resolve_for(&session, &array).unwrap(),
            replicated.clone().with_memory(Memory::Device),
        );
        assert_eq!(
            ArrayPlacement::device(devices[1]).resolve_for(&session, &array).unwrap(),
            ArrayPlacement::device(devices[1]).with_memory(Memory::Device).resolve(&session, 1).unwrap(),
        );
        let configured = session.with_device_mesh(mesh.clone());
        let sharding = Sharding::replicated(mesh.logical_mesh().clone(), 1);
        assert_eq!(ArrayPlacement::sharding(sharding).resolve_for(&configured, &array).unwrap(), replicated);
    }

    #[test]
    fn test_array_placement_resolve_unplaced() {
        let client = cpu_client(2);
        let session = Session::builder().client(client).build().unwrap();
        let devices = client
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| Device::from_pjrt(device).unwrap())
            .collect::<Vec<_>>();
        let device = ArrayPlacement::device(devices[0]).with_memory(Memory::Device);

        // Without a default mesh, a device resolves to a mesh of its own.
        assert_eq!(device.clone().resolve_unplaced(&session, 2).unwrap(), device.clone().resolve(&session, 2).unwrap());

        // The sole device of the default mesh resolves to that mesh, whatever its axis is named.
        let mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Explicit).unwrap()]).unwrap(),
            vec![devices[0]],
        )
        .unwrap();
        let configured = session.with_device_mesh(mesh.clone());
        assert_eq!(
            device.clone().resolve_unplaced(&configured, 2).unwrap(),
            Placement::new(mesh.clone(), Sharding::replicated(mesh.logical_mesh().clone(), 2))
                .unwrap()
                .with_memory(Memory::Device),
        );

        // Other devices and other placements resolve as they do for any value of that rank.
        let other = ArrayPlacement::device(devices[1]);
        assert_eq!(other.clone().resolve_unplaced(&configured, 2).unwrap(), other.resolve(&configured, 2).unwrap());
        let sharding = ArrayPlacement::sharding(Sharding::replicated(mesh.logical_mesh().clone(), 2));
        assert_eq!(
            sharding.clone().resolve_unplaced(&configured, 2).unwrap(),
            sharding.resolve(&configured, 2).unwrap(),
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
        let options = |source: Device| {
            PlacementOptions::<ArrayPlacement, ArrayPlacement>::new(
                None,
                Some(ArrayPlacement::device(source)),
                None,
                None,
            )
        };
        let (scalar, kept) =
            session.place((HostOrArray::from(1u8), HostOrArray::from(upload())), options(devices[0])).unwrap();
        assert_eq!(scalar.mesh().devices(), [devices[0]]);
        assert_eq!(scalar.copy_to_host_bytes().unwrap(), vec![1u8]);
        assert_eq!(kept.mesh().devices(), [devices[0]]);
        let error = session
            .place((HostOrArray::from(1u8), HostOrArray::from(upload())), options(devices[1]))
            .unwrap_err();
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(), ArrayError::SourcePlacementMismatch { .. }),
        ));
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
        let held = Arc::clone(array.shards()[0].buffer().unwrap());
        let unaliased = array.into_unaliased(&session).unwrap();
        assert!(!Arc::ptr_eq(unaliased.shards()[0].buffer().unwrap(), &held));
        assert_eq!(buffer_addresses(&unaliased)[1], addresses[1]);
        assert_eq!(shard_values(&unaliased), vec![vec![1., 2.]; 2]);

        // A clone shares every buffer, so all of them are copied.
        let array = upload();
        let unaliased = array.clone().into_unaliased(&session).unwrap();
        for (unaliased_shard, shard) in unaliased.shards().iter().zip(array.shards()) {
            assert!(!Arc::ptr_eq(unaliased_shard.buffer().unwrap(), shard.buffer().unwrap()));
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
    fn test_session_place_returns_empty_tree_for_input_without_leaves() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = ArrayPlacement::device(Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap());

        // Every option prefix has a leaf that no input leaf uses, which is not an error for an input without leaves.
        let options = PlacementOptions::new(Some(device.clone()), Some(device), Some(true), Some(Some(false)));
        let placed = session.place(Vec::<Array>::new(), options).unwrap();
        assert!(placed.is_empty());
        let placed = session.place(Vec::<f32>::new(), PlacementOptions::defaults()).unwrap();
        assert!(placed.is_empty());
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

        // The options of every leaf are validated before any leaf is placed, and the error names the first leaf that
        // is donated while it may alias.
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
    fn test_session_place_rejects_mismatched_source_for_array_leaf() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let source_mesh = domain_mesh(client, "x", 1);
        let source_device = source_mesh.devices()[0];
        let source_sharding = Sharding::replicated(source_mesh.logical_mesh().clone(), 1);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(source_sharding.clone())
            .unwrap();
        let source_array = Array::from_host_buffer(
            &session,
            source_type,
            source_mesh.clone(),
            None,
            values_to_bytes::<f32>(&[0.0, 1.0]).as_slice(),
        )
        .unwrap();
        let expected_placement = ArrayPlacement::device(Device::new(source_device.id() + 1, 0));
        let expected_source = expected_placement.clone().resolve(&session, 1).unwrap();
        let actual_source = Placement::new(source_mesh, source_sharding).unwrap();

        let error = session
            .place(
                (source_array.clone(), source_array),
                PlacementOptions::new(
                    Option::<ArrayPlacement>::None,
                    Some((None, Some(expected_placement))),
                    Option::<bool>::None,
                    Option::<Option<bool>>::None,
                ),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            format!(
                "expected source mesh `{:?}` with sharding `{:?}` does not match the array's current mesh `{:?}` \
                 with sharding `{:?}`",
                expected_source.mesh(),
                expected_source.sharding(),
                actual_source.mesh(),
                actual_source.sharding(),
            ),
        );
        assert!(matches!(
            &error,
            Error::Array(error) if matches!(error.as_ref(),
                ArrayError::SourcePlacementMismatch {
                    expected_mesh: reported_expected_mesh,
                    expected_sharding: reported_expected_sharding,
                    actual_mesh: reported_actual_mesh,
                    actual_sharding: reported_actual_sharding,
                } if reported_expected_mesh == expected_source.mesh()
                    && reported_expected_sharding == expected_source.sharding()
                    && reported_actual_mesh == actual_source.mesh()
                    && reported_actual_sharding == actual_source.sharding(),
            ),
        ));
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

        let memory = Memory::Host { pinned: true };
        let error = place(source.clone().with_memory(memory)).unwrap_err();
        assert!(matches!(
            error,
            Error::Array(error) if matches!(
                *error, ArrayError::SourceMemoryMismatch { expected, actual: Memory::Device } if expected == memory
            ),
        ));

        let other_layout = Layout::Tiled(TiledLayout::new(vec![1, 0], vec![]));
        let error = place(source.with_layout(other_layout.clone())).unwrap_err();
        assert!(matches!(
            error,
            Error::Array(error) if matches!(
                error.as_ref(),
                ArrayError::SourceLayoutMismatch { expected, actual: Some(actual) }
                    if expected == &other_layout && actual == &layout,
            ),
        ));
    }

    #[test]
    fn test_session_place_accepts_device_source_for_array_on_that_device() {
        let client = cpu_client(1);
        let session = Session::builder().client(client).build().unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();

        // An eager upload without a mesh names its single mesh axis differently from a device placement.
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
    fn test_session_place_places_traced_leaves_on_sole_device_of_compiled_mesh() {
        let client = cpu_client(1);
        let mesh = domain_mesh(client, "x", 1);
        let device = mesh.devices()[0];
        let session = Session::builder().client(client).build().unwrap().with_device_mesh(mesh.clone());
        let input_type = ArrayType::new_static(DataType::F32, [2]);

        // A device names the sole device of the mesh that the function is compiled for, whose axis is not named
        // `device`, so the placement stays on that mesh instead of moving the value to a mesh of its own.
        let target = ArrayPlacement::device(device);
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
            Array::from_host_buffer(&session, input_type, mesh.clone(), None, values_to_bytes(&[1f32, 2.])).unwrap();
        let output = compiled.executable().execute(&session, input).unwrap();
        assert_eq!(output.mesh(), &mesh);
        assert_eq!(shard_values(&output), vec![vec![1., 2.]]);
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

        // The session has no default mesh, so the source resolves to a mesh of its own, whose axis is named `device`
        // and not `x` like the axis of the mesh that the function is compiled for. The source is truthful, and the
        // placement has no target, so it compiles and leaves the value where it is.
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
}
