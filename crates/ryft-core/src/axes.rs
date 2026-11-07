//! Defines positional array axes and dynamically scoped named axes used by array operations and program transforms.
//!
//! Positional [`Axis`] values identify dimensions within one concrete array rank. Named axes instead identify
//! logical transform dimensions—such as a vectorized batch or device-mesh axis—through the active [`Context`] stack.
//! The two forms meet inside operation-owned rules: a named binding supplies value-free scope metadata, while the
//! rule supplies the physical dimension of each participating value when one exists. Refer to [`NamedAxes`] for a
//! diagram of named-axis lookup and consumption.
//!
//! # Positional Axes
//!
//! [`Axis`] stores a signed index and delays normalization until an array rank is known. Nonnegative indices count
//! from the leading dimension. Negative indices count backward, so `-1` denotes the trailing dimension. Normalization
//! accepts exactly `[-rank, rank)` and returns a non-negative position. [`Axes`] preserves an ordered collection of
//! these values and rejects duplicates after normalization, including aliases such as `0` and `-rank`.
//!
//! Positional axes are array-boundary descriptors. They differ from [`BatchAxis`](crate::BatchAxis), which additionally
//! represents replication and records whether one physical dimension of a packed value carries the mapped batch.
//!
//! # Named Axes and Dynamic Scope
//!
//! [`NamedAxis`] records the kind of logical binding and any statically known size. [`NamedAxes`] resolves names
//! innermost-first through the context stack: a batching level may bind its mapped axis, a tracing context may be
//! seeded with device-mesh axes, and nested tracing may introduce nearer bindings that shadow outer ones. Projection,
//! partial evaluation, differentiation, and other transparent wrappers delegate unresolved names to their parent.
//!
//! A binding deliberately does not identify a dimension of every value. A replicated input has no mapped dimension
//! even when a collective over the enclosing logical axis is meaningful. The operation rule that consumes the name
//! combines the binding with its transform-specific per-value metadata.
//!
//! # Axis Values
//!
//! [`NamedAxes`] answers whether a name is in scope and what it denotes; it does not produce a runtime value.
//! Its value-producing counterpart is [`AxisIndex`](crate::AxisIndex), which validates the binding and then returns
//! a `u64` scalar containing the current batch-item index or mesh-coordinate index according to the binder kind. The
//! resulting [`AxisIndexOperation`](crate::AxisIndexOperation) remains an ordinary operation and therefore composes
//! with interpretation, tracing, batching, differentiation, and partial evaluation through their normal rule contracts.
//!
//! # Errors and Extension Points
//!
//! [`AxisError`] distinguishes an out-of-range positional axis, a duplicate normalized position, and an unbound name.
//! New context wrappers that introduce a named axis should resolve their local binding first and delegate every other
//! name to the parent. Transparent wrappers should delegate all names unchanged. New named-axis operations should use
//! [`NamedAxes`] for scope validation and leave per-value dimension handling to the transform that owns that metadata.

use std::cell::RefCell;
use std::fmt::Display;
use std::ops::Deref;
use std::rc::Rc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::arrays::LogicalMesh;
use crate::batching::{BatchingContext, RecursiveBatchingPolicy};
use crate::contexts::{Context, DomainProjection, EagerContext, ProjectedContext};
use crate::differentiation::{DifferentiationContext, DifferentiationPolicy};
use crate::parameters::Parameter;
use crate::partial::PartialEvaluationContext;
use crate::programs::{Operation, Type, Value};
use crate::tracing::{NestedTracingContext, TracingContext};

/// Represents axis-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisError {
    #[error("axis {axis} is out of bounds for rank {rank}")]
    OutOfBounds { axis: Axis, rank: usize },

    #[error("axes contain duplicate axis {axis}")]
    DuplicateAxis { axis: usize },

    #[error("axis name `{name}` is not bound by any enclosing transform")]
    UnboundAxisName { name: String },
}

/// Positional array axis. Negative values index from the final axis, so `-1` denotes the trailing axis. [`Axis`]
/// converts from signed and unsigned integer types and defers normalization until the rank of the indexed array
/// is known.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Parameter)]
pub struct Axis(i128);

impl Axis {
    /// Returns the signed positional index represented by this [`Axis`].
    #[inline]
    pub fn value(self) -> i128 {
        self.0
    }

    /// Normalizes this [`Axis`] against `rank`, returning its non-negative position. Valid axes lie in `[-rank, rank)`.
    #[inline]
    pub fn normalize(self, rank: usize) -> Result<usize, AxisError> {
        let position = if self.0 >= 0 {
            usize::try_from(self.0).ok().filter(|&axis| axis < rank)
        } else {
            usize::try_from(self.0.unsigned_abs()).ok().and_then(|distance| rank.checked_sub(distance))
        };
        position.ok_or(AxisError::OutOfBounds { axis: self, rank })
    }
}

impl Display for Axis {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.0)
    }
}

/// Zero or more positional array [`Axis`] values. Scalar conversions produce a one-element axis list, while vectors,
/// arrays, and borrowed slices preserve every provided axis.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Parameter)]
pub struct Axes(Vec<Axis>);

impl Axes {
    /// Returns the axes as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[Axis] {
        self.0.as_slice()
    }

    /// Returns the number of axes in this collection.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if this collection contains no axes.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Normalizes every [`Axis`] in this collection against `rank`, preserving order and rejecting duplicates
    /// after negative axes are resolved.
    pub fn normalize(&self, rank: usize) -> Result<Vec<usize>, AxisError> {
        let mut normalized_axes = Vec::with_capacity(self.len());
        let mut seen = vec![false; rank];
        for axis in self.iter() {
            let normalized_axis = axis.normalize(rank)?;
            if seen[normalized_axis] {
                return Err(AxisError::DuplicateAxis { axis: normalized_axis });
            }
            seen[normalized_axis] = true;
            normalized_axes.push(normalized_axis);
        }
        Ok(normalized_axes)
    }
}

impl Deref for Axes {
    type Target = [Axis];

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl AsRef<[Axis]> for Axes {
    #[inline]
    fn as_ref(&self) -> &[Axis] {
        self.as_slice()
    }
}

impl From<Axis> for Axes {
    #[inline]
    fn from(axis: Axis) -> Self {
        Self(vec![axis])
    }
}

impl From<&Axes> for Axes {
    #[inline]
    fn from(axes: &Axes) -> Self {
        axes.clone()
    }
}

impl<A: Into<Axis>> From<Vec<A>> for Axes {
    #[inline]
    fn from(axes: Vec<A>) -> Self {
        Self(axes.into_iter().map(Into::into).collect())
    }
}

impl<A: Copy + Into<Axis>> From<&Vec<A>> for Axes {
    #[inline]
    fn from(axes: &Vec<A>) -> Self {
        Self::from(axes.as_slice())
    }
}

impl<A: Copy + Into<Axis>> From<&[A]> for Axes {
    #[inline]
    fn from(axes: &[A]) -> Self {
        Self(axes.iter().copied().map(Into::into).collect())
    }
}

impl<A: Into<Axis>, const N: usize> From<[A; N]> for Axes {
    #[inline]
    fn from(axes: [A; N]) -> Self {
        Self(axes.into_iter().map(Into::into).collect())
    }
}

impl<A: Copy + Into<Axis>, const N: usize> From<&[A; N]> for Axes {
    #[inline]
    fn from(axes: &[A; N]) -> Self {
        Self::from(axes.as_slice())
    }
}

macro_rules! impl_axis_conversions {
    ($integer:ty) => {
        impl From<$integer> for Axis {
            #[inline]
            fn from(axis: $integer) -> Self {
                Self(axis as i128)
            }
        }

        impl From<$integer> for Axes {
            #[inline]
            fn from(axis: $integer) -> Self {
                Axis::from(axis).into()
            }
        }
    };
}

impl_axis_conversions!(i8);
impl_axis_conversions!(i16);
impl_axis_conversions!(i32);
impl_axis_conversions!(i64);
impl_axis_conversions!(i128);
impl_axis_conversions!(isize);
impl_axis_conversions!(u8);
impl_axis_conversions!(u16);
impl_axis_conversions!(u32);
impl_axis_conversions!(u64);
impl_axis_conversions!(usize);

/// A named axis resolved by a [`NamedAxes`] context specifying what an axis name is currently bound to, and by which
/// kind of transform, at a given trace level. This carries only the *value-free* facts about a binding (i.e., its kind
/// and any statically known size or owning mesh), not which dimension of any particular value carries the axis. That
/// per-value mapping is partial (a replicated input has no such dimension even though a collective over it is still
/// meaningful) and is supplied at consumption time by the owning transform's rule dispatch (e.g., an
/// [`ArrayBatch`](crate::ArrayBatch)'s [`batch_axis`](crate::ArrayBatch::batch_axis)).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum NamedAxis {
    /// Axis bound by an enclosing batching (i.e., vectorization) level.
    Batched {
        /// Number of batch items along this axis when statically known, or `None` when its extent is dynamic
        /// (i.e., not known statically at tracing time).
        size: Option<usize>,
    },

    /// Axis bound to a device mesh axis by an enclosing manual sharding region.
    Mesh {
        /// [`LogicalMesh`] that owns the binding, including the axis kinds used for value metadata.
        mesh: LogicalMesh,

        /// Index of the mesh axis this name resolves to.
        axis: usize,

        /// Number of shards along this mesh axis.
        size: usize,

        /// Whether this binding checks manual variation. With `false`, collective operations retain their physical
        /// communication semantics without adding variation metadata to local values.
        checks_manual_variation: bool,
    },
}

impl NamedAxis {
    /// Returns the statically known number of batch items or device shards along this [`NamedAxis`], or [`None`] for a
    /// batched axis whose extent is dynamic. Mesh axes always have a static size.
    #[inline]
    pub fn size(&self) -> Option<usize> {
        match self {
            Self::Batched { size } => *size,
            Self::Mesh { size, .. } => Some(*size),
        }
    }

    /// Returns the logical mesh that owns this binding, for both checked and unchecked mesh axes. Batched axes have
    /// no mesh.
    #[inline]
    pub fn mesh(&self) -> Option<&LogicalMesh> {
        match self {
            Self::Mesh { mesh, .. } => Some(mesh),
            Self::Batched { .. } => None,
        }
    }

    /// Returns whether this binding enables manual-variation checks. Batched axes and unchecked mesh axes return
    /// `false`, so their collective operations use ordinary array types.
    #[inline]
    pub fn checks_manual_variation(&self) -> bool {
        match self {
            Self::Mesh { checks_manual_variation, .. } => *checks_manual_variation,
            Self::Batched { .. } => false,
        }
    }
}

/// Capability for resolving named axes visible at one context-stack level. Named axes are dynamically scoped binders
/// introduced by transforms and manual sharding regions, then consumed by named-axis operations such as collectives.
/// Resolution is innermost-first, so a nearer binder shadows a farther one. The returned [`NamedAxis`] carries only
/// value-free kind, size, and mesh facts; the owning operation rule remains responsible for how a use consumes that
/// logical axis and which physical dimension of each value carries it.
///
/// # Dynamic-Scope Lookup
///
/// ```text
/// ┌───────────────────────────────────────┐
/// │ Operation Requests an Axis Name       │
/// └───────────────────┬───────────────────┘
///                     ▼
/// ┌───────────────────────────────────────┐
/// │ Current Context                       │
/// └───────────────────┬───────────────────┘
///                     ▼
/// ┌───────────────────────────────────────┐ not bound locally ┌───────────────────────────────┐
/// │ Check Local Named-Axis Bindings       ├──────────────────▶│ Delegate to Parent Context    │
/// └───────────────────┬───────────────────┘                   └───┬───────────────────────────┘
///                     │ nearest local binding                     │
///                     ▼                                           ▼
/// ┌───────────────────────────────────────┐   binding found   ┌───────────────────────────────┐
/// │ Named Axis: Batched or Mesh           │◀──────────────────┤ Repeat Innermost-First Lookup │
/// └───────────────────┬───────────────────┘                   └───┬───────────────────────────┘
///                     │                                           │ no enclosing binding
///                     ▼                                           ▼
/// ┌───────────────────────────────────────┐                   ┌───────────────────────────────┐
/// │ Value-Free Kind, Optional Static Size ├─────────┐         │ Unbound Axis Error            │
/// └───────────────────┬───────────────────┘         │         └───────────────────────────────┘
///                     ▼                             │
/// ┌───────────────────────────────────────┐         │         ┌───────────────────────────────┐
/// │ Operation-Owned Rule                  │         └────────▶│ `AxisIndex` Capability        │
/// └───────────────────────────────────────┘                   └───┬───────────────────────────┘
///                     ▲                                           ▼
/// ┌───────────────────┴───────────────────┐                   ┌───────────────────────────────┐
/// │ Per-Value Mapped-Axis Metadata        │                   │ Current Index Value           │
/// └───────────────────────────────────────┘                   └───────────────────────────────┘
/// ```
///
/// Lookup (i.e., [`NamedAxes::named_axis`]) checks local bindings before delegating outward unless the context is a
/// leaf. Enumeration (i.e., [`NamedAxes::named_axes`]) includes the enclosing bindings with the same shadowing rules.
pub trait NamedAxes: Context {
    /// Resolves `name` against this context, returning the [`NamedAxis`] it is bound to,
    /// or `None` when no enclosing binder binds it.
    fn named_axis(&self, name: &str) -> Option<NamedAxis>;

    /// Returns the named axes in scope at this context, innermost first, with each name once (i.e., without the
    /// bindings that nearer binders shadow), so that [`Self::named_axis`] resolves a name to the binding that this
    /// function returns for it. Functions that trace user code in a fresh trace instead of in the context that they
    /// are called in (e.g., [`Composite::call`](crate::Composite::call) for its primal and rules) seed that trace
    /// with these bindings, so that the code resolves the same names as it would in the calling context.
    fn named_axes(&self) -> Vec<(String, NamedAxis)>;
}

impl<V: Value, O: Operation<Type = V::Type>> NamedAxes for EagerContext<V, O>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
        // An eager context binds no named axes as it is a leaf of the resolution stack. So every lookup returns `None`.
        None
    }

    #[inline]
    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        Vec::new()
    }
}

impl<T: Type, C: NamedAxes + DomainProjection<T>> NamedAxes for ProjectedContext<C, T>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // Projection changes only the visible type/value/operation member. Named-axis scope belongs to the parent
        // context stack and therefore passes through unchanged.
        self.parent().named_axis(name)
    }

    #[inline]
    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        self.parent().named_axes()
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Capture> NamedAxes for TracingContext<V, O, Capture>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A `TracingContext` is a leaf of the resolution stack and it resolves only the named axes it was seeded with
        // (e.g., a `shard_map` body's device mesh axes) and reports every other name unbound. Ordinary traces are
        // seeded with no axes. Named-axis binders such as `BatchingContext` wrap a base trace and resolve against it.
        self.local_named_axes()
            .iter()
            .find(|(axis_name, _)| axis_name == name)
            .map(|(_, axis)| self.local_mesh_retypings().retype_named_axis(axis.clone()))
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        // The first of several seeded bindings of one name is the one that `named_axis` resolves.
        let mut axes = Vec::<(String, NamedAxis)>::new();
        for (name, axis) in self.local_named_axes() {
            if !axes.iter().any(|(visible, _)| visible == name) {
                axes.push((name.clone(), self.local_mesh_retypings().retype_named_axis(axis.clone())));
            }
        }
        axes
    }
}

impl<C: NamedAxes> NamedAxes for NestedTracingContext<C>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A lookup resolves against the axes this nested trace was seeded with first, and otherwise delegates to the
        // parent context it is nested into, because named axes are dynamically scoped: a seeded binding shadows an
        // enclosing one, while a collective staged inside an unseeded nested tracing context still resolves an axis
        // bound by an enclosing transform.
        self.local_named_axes()
            .iter()
            .find(|(axis_name, _)| axis_name == name)
            .map(|(_, axis)| axis.clone())
            .or_else(|| self.parent().named_axis(name))
            .map(|axis| self.local_mesh_retypings().retype_named_axis(axis))
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        // Seeded bindings shadow the parent's bindings of the same names, and the first of several seeded bindings
        // of one name is the one that `named_axis` resolves.
        let mut axes = Vec::<(String, NamedAxis)>::new();
        for (name, axis) in self.local_named_axes().iter().cloned().chain(self.parent().named_axes()) {
            if !axes.iter().any(|(visible, _)| visible == &name) {
                axes.push((name, self.local_mesh_retypings().retype_named_axis(axis)));
            }
        }
        axes
    }
}

impl<C: NamedAxes> NamedAxes for PartialEvaluationContext<C>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A partial-evaluation context resolves named axes against its known-side inner context, so collectives
        // inside a partially evaluated closure resolve against the enclosing batching levels and mesh regions.
        self.parent().named_axis(name)
    }

    #[inline]
    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        self.parent().named_axes()
    }
}

impl<C: NamedAxes, P: RecursiveBatchingPolicy<C>> NamedAxes for BatchingContext<C, P>
where
    Self: Context,
{
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A batching level binds the axis it introduces: a lookup for this level's `axis_name` resolves to
        // `NamedAxis::Batched` with whatever static size the policy can report for its extent (a host `usize` extent
        // is always known, while a first-class dimension extent is known only once it resolves to a constant), and any
        // other name delegates to the parent context. Because nested batching composes by context wrapping, the
        // delegation chain naturally shadows outer bindings with inner ones.
        if self.axis_name() == Some(name) {
            Some(NamedAxis::Batched { size: P::static_batch_axis_extent(self) })
        } else {
            self.parent().named_axis(name)
        }
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        // This level's axis shadows the parent's binding of the same name.
        let mut axes = self.parent().named_axes();
        if let Some(name) = self.axis_name() {
            axes.retain(|(visible, _)| visible != name);
            axes.insert(0, (name.to_string(), NamedAxis::Batched { size: P::static_batch_axis_extent(self) }));
        }
        axes
    }
}

impl<C: NamedAxes, P: DifferentiationPolicy<C>> NamedAxes for DifferentiationContext<C, P>
where
    Self: Context,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A `DifferentiationContext` binds no named axes of its own: axis-name resolution passes through to the inner
        // context, so collectives inside a differentiated closure resolve against the enclosing batching levels and
        // mesh regions.
        self.primal().named_axis(name)
    }

    #[inline]
    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        self.primal().named_axes()
    }
}

/// Capability for resolving the default [`LogicalMesh`] of a context. Functions that need a mesh but are not
/// given one (e.g., [`shard_map`](fn@crate::shard_map) called with [`None`] for its mesh) use it.
///
/// The default mesh is owned explicitly by the contexts that introduce it rather than derived from their
/// [named axes](NamedAxes::named_axes). Root traces are seeded with it (e.g., a `shard_map` body is seeded with
/// the mesh of its map), nested traces without a mesh of their own and transform contexts (e.g., batching and
/// differentiation) delegate to their parent, and eager contexts have none. A derived default mesh would be wrong
/// in two ways. A named batching level that shadows a manual mesh axis hides the binding of that axis although the
/// computation is still inside the manual region of its mesh (and with several enclosing meshes, the shadowing may
/// expose an outer mesh instead), and a context may have a default mesh without binding any manual axes. Concrete
/// device meshes stay with backend domains and compilation options.
pub trait MeshContext: Context {
    /// Returns the default [`LogicalMesh`] of this context, or [`None`] when it has none.
    fn default_mesh(&self) -> Option<LogicalMesh>;

    /// Runs `function` with `mesh` retyped to `retyped_mesh` in this context, which must have the topology of `mesh`,
    /// and returns its result. While `function` runs, this context (and every clone of it, including those that its
    /// values hold) reports every named-axis binding over `mesh` (refer to [`NamedAxes`]) and a default mesh equal to
    /// `mesh` with `retyped_mesh` instead, so that the values that operations create from those bindings (e.g.,
    /// [`axis_index`](crate::AxisIndex::axis_index) and the payload meshes of collectives) are typed over
    /// `retyped_mesh`. Retypings nest, and each one ends when `function` returns or unwinds. This is how
    /// [`auto_axes`](crate::auto_axes) and [`explicit_axes`](crate::explicit_axes) retype the mesh of
    /// the context that their function runs in, as JAX's `use_abstract_mesh` retypes its context mesh.
    ///
    /// The retyping is owned by the contexts that own named-axis bindings or default meshes (i.e., tracing contexts),
    /// and wrapper contexts (e.g., batching, differentiation, partial evaluation, and projection) forward it to their
    /// parents. The default implementation runs `function` unchanged, which is correct for contexts that own neither
    /// (e.g., eager contexts).
    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        let _ = (mesh, retyped_mesh);
        function()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> MeshContext for EagerContext<V, O>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        None
    }
}

impl<T: Type, C: MeshContext + DomainProjection<T>> MeshContext for ProjectedContext<C, T>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        self.parent().default_mesh()
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.parent().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, Capture> MeshContext for TracingContext<V, O, Capture>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        self.local_default_mesh().cloned().map(|mesh| self.local_mesh_retypings().retype_mesh(mesh))
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.local_mesh_retypings().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

impl<C: MeshContext> MeshContext for NestedTracingContext<C>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        self.local_default_mesh()
            .cloned()
            .or_else(|| self.parent().default_mesh())
            .map(|mesh| self.local_mesh_retypings().retype_mesh(mesh))
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.local_mesh_retypings().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

impl<C: MeshContext> MeshContext for PartialEvaluationContext<C>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        self.parent().default_mesh()
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.parent().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

impl<C: MeshContext, P: RecursiveBatchingPolicy<C>> MeshContext for BatchingContext<C, P>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        // A batching level binds a named axis but never a mesh, so it does not change the default mesh even when its
        // axis shadows a manual mesh axis.
        self.parent().default_mesh()
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.parent().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

impl<C: MeshContext, P: DifferentiationPolicy<C>> MeshContext for DifferentiationContext<C, P>
where
    Self: Context,
{
    #[inline]
    fn default_mesh(&self) -> Option<LogicalMesh> {
        self.primal().default_mesh()
    }

    #[inline]
    fn with_retyped_mesh<R, F: FnOnce() -> R>(&self, mesh: &LogicalMesh, retyped_mesh: &LogicalMesh, function: F) -> R {
        self.primal().with_retyped_mesh(mesh, retyped_mesh, function)
    }
}

/// Stack of the mesh retypings that are active in a tracing context (refer to [`MeshContext::with_retyped_mesh`]),
/// innermost last. It is shared by every clone of the context that owns it, like that context's program builder, so
/// that the values of the context observe a retyping while it is active.
#[derive(Clone, Debug, Default)]
pub(crate) struct MeshRetypings {
    /// Active retypings, each from its original mesh to its retyped mesh.
    retypings: Rc<RefCell<Vec<(LogicalMesh, LogicalMesh)>>>,
}

impl MeshRetypings {
    /// Runs `function` with the retyping from `mesh` to `retyped_mesh` active, removing it again once `function`
    /// returns or unwinds.
    pub(crate) fn with_retyped_mesh<R, F: FnOnce() -> R>(
        &self,
        mesh: &LogicalMesh,
        retyped_mesh: &LogicalMesh,
        function: F,
    ) -> R {
        /// Removes the innermost retyping when dropped.
        struct Guard<'r>(&'r RefCell<Vec<(LogicalMesh, LogicalMesh)>>);

        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.borrow_mut().pop();
            }
        }

        self.retypings.borrow_mut().push((mesh.clone(), retyped_mesh.clone()));
        let _guard = Guard(&self.retypings);
        function()
    }

    /// Returns `mesh` with the active retypings applied, outermost first, so that a retyping registered inside another
    /// one applies to the mesh that the outer one produced.
    pub(crate) fn retype_mesh(&self, mut mesh: LogicalMesh) -> LogicalMesh {
        for (original, retyped) in self.retypings.borrow().iter() {
            if &mesh == original {
                mesh = retyped.clone();
            }
        }
        mesh
    }

    /// Returns `axis` with the active retypings applied to the mesh of a [`NamedAxis::Mesh`] binding.
    pub(crate) fn retype_named_axis(&self, axis: NamedAxis) -> NamedAxis {
        match axis {
            NamedAxis::Mesh { mesh, axis, size, checks_manual_variation } => {
                NamedAxis::Mesh { mesh: self.retype_mesh(mesh), axis, size, checks_manual_variation }
            }
            axis => axis,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{ArrayType, DataType, MeshAxis, MeshAxisType};
    use crate::programs::ProgramError;
    use crate::tests::{TestArrayContext, TestArrayTracingContext};
    use crate::tracing::{Tracer, TracingScope};

    use super::*;

    /// Runs context assertions in a seeded trace and checks that resolving mesh metadata stages no work.
    fn with_mesh_context(
        mesh: Option<LogicalMesh>,
        named_axes: Vec<(String, NamedAxis)>,
        check: impl FnOnce(&TestArrayTracingContext),
    ) {
        let (_, program) = TestArrayTracingContext::trace_with_context_in_scope(
            |context, ()| {
                check(context);
                Ok(())
            },
            (),
            TracingScope::new(named_axes, mesh),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                in ()
            "}
            .trim_end(),
        );
    }

    /// Meshes that retain the manual axis `x` while switching the free axis `y` from explicit to auto.
    fn retyping_meshes() -> (LogicalMesh, LogicalMesh) {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 3, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let retyped = mesh.with_axis_types(["y"], MeshAxisType::Auto).unwrap();
        (mesh, retyped)
    }

    #[test]
    fn test_axis_error_renders_unbound_axis_name() {
        let error = AxisError::UnboundAxisName { name: "batch".to_string() };
        assert_eq!(error.to_string(), "axis name `batch` is not bound by any enclosing transform");
        assert_eq!(format!("{error:?}"), "UnboundAxisName { name: \"batch\" }");
        assert_eq!(error, AxisError::UnboundAxisName { name: "batch".to_string() });
        assert_ne!(error, AxisError::UnboundAxisName { name: "device".to_string() });
    }

    #[test]
    fn test_axis() {
        assert_eq!(Axis::from(0).normalize(3), Ok(0));
        assert_eq!(Axis::from(2usize).normalize(3), Ok(2));
        assert_eq!(Axis::from(-1).normalize(3), Ok(2));
        assert_eq!(Axis::from(-3).normalize(3), Ok(0));
        assert_eq!(Axis::from(usize::MAX).value(), i128::try_from(usize::MAX).unwrap());
        assert_eq!(Axis::from(3).normalize(3), Err(AxisError::OutOfBounds { axis: Axis::from(3), rank: 3 }));
        assert_eq!(Axis::from(-4).normalize(3), Err(AxisError::OutOfBounds { axis: Axis::from(-4), rank: 3 }));
        assert_eq!(
            Axis::from(i128::MIN).normalize(usize::MAX),
            Err(AxisError::OutOfBounds { axis: Axis::from(i128::MIN), rank: usize::MAX }),
        );
        assert_eq!(Axis::from(-1).to_string(), "-1");
    }

    #[test]
    fn test_axes() {
        let axes = Axes::from([0, -1, 1]);
        assert_eq!(axes.as_slice(), &[Axis::from(0), Axis::from(-1), Axis::from(1)]);
        assert_eq!(axes.normalize(3), Ok(vec![0, 2, 1]));
        assert_eq!(Axes::from([0, -3]).normalize(3), Err(AxisError::DuplicateAxis { axis: 0 }));
        assert_eq!(Axes::from(Axis::from(1)).as_slice(), &[Axis::from(1)]);
        assert_eq!(Axes::from(&axes), axes);
        assert_eq!(Axes::default().normalize(0), Ok(Vec::new()));
    }

    #[test]
    fn test_named_axis_equality_and_hashing() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        assert_eq!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: Some(3) });
        assert_ne!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: Some(4) });
        assert_ne!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: None });
        assert_eq!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2, checks_manual_variation: true },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2, checks_manual_variation: true }
        );
        assert_ne!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2, checks_manual_variation: true }
        );

        // A batched axis never equals a mesh axis, even when their sizes match.
        assert_ne!(
            NamedAxis::Batched { size: Some(2) },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true }
        );
        assert_ne!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: false },
        );

        let axes = HashSet::from([
            NamedAxis::Batched { size: Some(3) },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2, checks_manual_variation: true },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2, checks_manual_variation: false },
        ]);
        assert!(axes.contains(&NamedAxis::Batched { size: Some(3) }));
        assert!(axes.contains(&NamedAxis::Mesh {
            mesh: mesh.clone(),
            axis: 1,
            size: 2,
            checks_manual_variation: true
        }));
        assert!(axes.contains(&NamedAxis::Mesh {
            mesh: mesh.clone(),
            axis: 1,
            size: 2,
            checks_manual_variation: false
        }));
        assert!(!axes.contains(&NamedAxis::Batched { size: Some(2) }));
    }

    #[test]
    fn test_named_axis_size() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(NamedAxis::Batched { size: Some(3) }.size(), Some(3));
        assert_eq!(NamedAxis::Batched { size: None }.size(), None);
        assert_eq!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true }.size(),
            Some(2)
        );
        assert_eq!(NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: false }.size(), Some(2));
    }

    #[test]
    fn test_named_axis_mesh() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(NamedAxis::Batched { size: Some(2) }.mesh(), None);
        assert_eq!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true }.mesh(),
            Some(&mesh)
        );
        assert_eq!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: false }.mesh(),
            Some(&mesh)
        );
    }

    #[test]
    fn test_named_axis_checks_manual_variation() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert!(!NamedAxis::Batched { size: Some(2) }.checks_manual_variation());
        assert!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true }
                .checks_manual_variation()
        );
        assert!(!NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: false }.checks_manual_variation());
    }

    #[test]
    fn test_mesh_context_default_mesh() {
        assert_eq!(TestArrayContext::new().default_mesh(), None);
    }

    #[test]
    fn test_mesh_context_default_mesh_projected_context() {
        assert_eq!(ProjectedContext::<_, ArrayType>::new(TestArrayContext::new()).default_mesh(), None);

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            let projected = ProjectedContext::<_, ArrayType>::new(context.clone());
            assert_eq!(projected.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_default_mesh_tracing_context() {
        assert_eq!(TestArrayTracingContext::new().default_mesh(), None);

        // Named mesh axes do not supply a default mesh, and an explicit default needs no named axis bindings.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let binding = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        with_mesh_context(None, vec![("x".to_string(), binding)], |context| {
            assert_eq!(context.default_mesh(), None);
        });
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_default_mesh_nested_tracing_context() {
        assert_eq!(NestedTracingContext::new(TestArrayContext::new()).default_mesh(), None);

        let outer_mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let inner_mesh = LogicalMesh::new(vec![MeshAxis::new("y", 3, MeshAxisType::Manual).unwrap()]).unwrap();
        with_mesh_context(Some(outer_mesh.clone()), Vec::new(), |parent| {
            let inherited = NestedTracingContext::new(parent.clone());
            assert_eq!(inherited.default_mesh(), Some(outer_mesh.clone()));

            let (_, program) = NestedTracingContext::trace_in_scope(
                inherited,
                |inputs: Vec<Tracer<_>>| {
                    let context = inputs[0].context();
                    assert_eq!(context.default_mesh(), Some(inner_mesh.clone()));
                    assert_eq!(NestedTracingContext::new(context.clone()).default_mesh(), Some(inner_mesh.clone()));
                    Ok(inputs)
                },
                vec![ArrayType::scalar(DataType::F64)],
                TracingScope::new(Vec::new(), Some(inner_mesh.clone())),
            )
            .unwrap();
            assert_eq!(
                program.to_string(),
                indoc! {"
                    lambda %0:f64[] .
                    in (%0)
                "}
                .trim_end(),
            );
            assert_eq!(parent.default_mesh(), Some(outer_mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_default_mesh_partial_evaluation_context() {
        assert_eq!(PartialEvaluationContext::new(TestArrayContext::new()).default_mesh(), None);

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            assert_eq!(PartialEvaluationContext::new(context.clone()).default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_default_mesh_batching_context() {
        assert_eq!(BatchingContext::new(TestArrayContext::new(), 3).default_mesh(), None);

        // Shadowing the manual axis with a batching axis must not hide the enclosing default mesh.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let binding = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        with_mesh_context(Some(mesh.clone()), vec![("x".to_string(), binding)], |context| {
            let batched = BatchingContext::new(context.clone(), 3).with_axis_name(Some("x".to_string()));
            assert_eq!(batched.named_axis("x"), Some(NamedAxis::Batched { size: Some(3) }));
            assert_eq!(batched.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_default_mesh_differentiation_context() {
        assert_eq!(DifferentiationContext::fused(TestArrayContext::new()).default_mesh(), None);

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            assert_eq!(DifferentiationContext::fused(context.clone()).default_mesh(), Some(mesh.clone()));
            let primal = PartialEvaluationContext::new(context.clone());
            assert_eq!(DifferentiationContext::partitioned(primal).default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh() {
        let (mesh, retyped) = retyping_meshes();
        let context = TestArrayContext::new();
        assert_eq!(
            context.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(context.default_mesh(), None);
                assert_eq!(context.named_axes(), Vec::new());
                7
            }),
            7,
        );
        assert_eq!(context.default_mesh(), None);
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_projected_context() {
        let (mesh, retyped) = retyping_meshes();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            let projected = ProjectedContext::<_, ArrayType>::new(context.clone());
            projected.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(projected.default_mesh(), Some(retyped.clone()));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
            });
            assert_eq!(projected.default_mesh(), Some(mesh.clone()));
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_tracing_context() {
        let (mesh, retyped) = retyping_meshes();
        let original_axis = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        let retyped_axis = NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: true };
        let axes = vec![("x".to_string(), original_axis.clone())];
        with_mesh_context(Some(mesh.clone()), axes.clone(), |context| {
            let cloned = context.clone();
            // Independent traces stay on their own mesh even when they were seeded with the same bindings.
            with_mesh_context(Some(mesh.clone()), axes.clone(), |independent| {
                assert_eq!(
                    context.with_retyped_mesh(&mesh, &retyped, || {
                        assert_eq!(context.default_mesh(), Some(retyped.clone()));
                        assert_eq!(context.named_axis("x"), Some(retyped_axis.clone()));
                        assert_eq!(context.named_axis("missing"), None);
                        assert_eq!(context.named_axes(), vec![("x".to_string(), retyped_axis.clone())]);
                        assert_eq!(cloned.default_mesh(), Some(retyped.clone()));
                        assert_eq!(cloned.named_axis("x"), Some(retyped_axis.clone()));
                        assert_eq!(independent.default_mesh(), Some(mesh.clone()));
                        assert_eq!(independent.named_axes(), axes);
                        11
                    }),
                    11,
                );
                assert_eq!(context.default_mesh(), Some(mesh.clone()));
                assert_eq!(cloned.default_mesh(), Some(mesh.clone()));
                assert_eq!(cloned.named_axis("x"), Some(original_axis.clone()));
                assert_eq!(context.named_axes(), axes);
            });
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_restores_after_error() {
        let (mesh, retyped) = retyping_meshes();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            let result = context.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
                Err::<(), _>(ProgramError::InvalidArgument { message: "mesh retyping failure".to_string() })
            });
            assert_eq!(result, Err(ProgramError::InvalidArgument { message: "mesh retyping failure".to_string() }));
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_restores_after_unwind() {
        let (mesh, retyped) = retyping_meshes();
        let inner_mesh = retyped.with_axis_types(["y"], MeshAxisType::Manual).unwrap();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            context.with_retyped_mesh(&mesh, &retyped, || {
                let panic = catch_unwind(AssertUnwindSafe(|| {
                    context.with_retyped_mesh(&retyped, &inner_mesh, || {
                        assert_eq!(context.default_mesh(), Some(inner_mesh.clone()));
                        panic!("mesh retyping panic");
                    });
                }))
                .unwrap_err();
                assert_eq!(panic.downcast_ref::<&str>(), Some(&"mesh retyping panic"));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
            });
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_nested_tracing_context() {
        let (mesh, retyped) = retyping_meshes();
        let original_axis = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        let retyped_axis = NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: true };
        with_mesh_context(Some(mesh.clone()), vec![("x".to_string(), original_axis.clone())], |parent| {
            let nested = NestedTracingContext::new(parent.clone());
            let cloned = nested.clone();
            nested.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(nested.default_mesh(), Some(retyped.clone()));
                assert_eq!(nested.named_axis("x"), Some(retyped_axis.clone()));
                assert_eq!(nested.named_axes(), vec![("x".to_string(), retyped_axis.clone())]);
                assert_eq!(cloned.default_mesh(), Some(retyped.clone()));
                assert_eq!(parent.default_mesh(), Some(mesh.clone()));
                assert_eq!(parent.named_axis("x"), Some(original_axis.clone()));
            });
            assert_eq!(nested.default_mesh(), Some(mesh.clone()));
            assert_eq!(cloned.named_axis("x"), Some(original_axis.clone()));

            // A retyping owned by the parent remains visible through an existing nested trace.
            parent.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(nested.default_mesh(), Some(retyped.clone()));
                assert_eq!(nested.named_axis("x"), Some(retyped_axis.clone()));
            });
            assert_eq!(nested.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_nested_tracing_context_local_bindings() {
        let (mesh, retyped) = retyping_meshes();
        let parent_axis = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        let local_axis = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: false };
        let retyped_axis = NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: false };
        let rows = NamedAxis::Batched { size: Some(5) };
        with_mesh_context(
            Some(mesh.clone()),
            vec![("x".to_string(), parent_axis.clone()), ("rows".to_string(), rows.clone())],
            |parent| {
                let (_, program) = NestedTracingContext::trace_in_scope(
                    parent.clone(),
                    |inputs: Vec<Tracer<_>>| {
                        let context = inputs[0].context();
                        context.with_retyped_mesh(&mesh, &retyped, || {
                            assert_eq!(context.default_mesh(), Some(retyped.clone()));
                            assert_eq!(context.named_axis("x"), Some(retyped_axis.clone()));
                            assert_eq!(context.named_axis("rows"), Some(rows.clone()));
                            assert_eq!(
                                context.named_axes(),
                                vec![("x".to_string(), retyped_axis.clone()), ("rows".to_string(), rows.clone())],
                            );
                            assert_eq!(parent.default_mesh(), Some(mesh.clone()));
                            assert_eq!(parent.named_axis("x"), Some(parent_axis.clone()));
                        });
                        assert_eq!(context.default_mesh(), Some(mesh.clone()));
                        assert_eq!(context.named_axis("x"), Some(local_axis.clone()));
                        Ok(inputs)
                    },
                    vec![ArrayType::scalar(DataType::F64)],
                    TracingScope::new(
                        vec![("x".to_string(), local_axis.clone()), ("x".to_string(), parent_axis.clone())],
                        Some(mesh.clone()),
                    ),
                )
                .unwrap();
                assert_eq!(
                    program.to_string(),
                    indoc! {"
                        lambda %0:f64[] .
                        in (%0)"
                    },
                );
            },
        );
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_partial_evaluation_context() {
        let (mesh, retyped) = retyping_meshes();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            let partial = PartialEvaluationContext::new(context.clone());
            partial.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(partial.default_mesh(), Some(retyped.clone()));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
            });
            assert_eq!(partial.default_mesh(), Some(mesh.clone()));
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_batching_context() {
        let (mesh, retyped) = retyping_meshes();
        let binding = NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true };
        with_mesh_context(Some(mesh.clone()), vec![("x".to_string(), binding.clone())], |context| {
            let batched = BatchingContext::new(context.clone(), 3).with_axis_name(Some("x".to_string()));
            batched.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(batched.default_mesh(), Some(retyped.clone()));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
                assert_eq!(batched.named_axis("x"), Some(NamedAxis::Batched { size: Some(3) }));
                assert_eq!(batched.named_axes(), vec![("x".to_string(), NamedAxis::Batched { size: Some(3) })]);
                assert_eq!(
                    context.named_axis("x"),
                    Some(NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: true }),
                );
            });
            assert_eq!(batched.default_mesh(), Some(mesh.clone()));
            assert_eq!(context.named_axis("x"), Some(binding.clone()));
        });
    }

    #[test]
    fn test_mesh_context_with_retyped_mesh_differentiation_context() {
        let (mesh, retyped) = retyping_meshes();
        with_mesh_context(Some(mesh.clone()), Vec::new(), |context| {
            let fused = DifferentiationContext::fused(context.clone());
            fused.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(fused.default_mesh(), Some(retyped.clone()));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
            });
            assert_eq!(fused.default_mesh(), Some(mesh.clone()));
            assert_eq!(context.default_mesh(), Some(mesh.clone()));

            let partitioned = DifferentiationContext::partitioned(PartialEvaluationContext::new(context.clone()));
            partitioned.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(partitioned.default_mesh(), Some(retyped.clone()));
                assert_eq!(context.default_mesh(), Some(retyped.clone()));
            });
            assert_eq!(partitioned.default_mesh(), Some(mesh.clone()));
            assert_eq!(context.default_mesh(), Some(mesh.clone()));
        });
    }

    #[test]
    fn test_mesh_retypings_with_retyped_mesh() {
        let (mesh, retyped) = retyping_meshes();
        let retypings = MeshRetypings::default();
        let cloned = retypings.clone();
        let independent = MeshRetypings::default();
        assert_eq!(
            retypings.with_retyped_mesh(&mesh, &retyped, || {
                assert_eq!(cloned.retype_mesh(mesh.clone()), retyped);
                assert_eq!(independent.retype_mesh(mesh.clone()), mesh);
                13
            }),
            13,
        );
        assert_eq!(cloned.retype_mesh(mesh.clone()), mesh);
    }

    #[test]
    fn test_mesh_retypings_retype_mesh() {
        let (mesh, retyped) = retyping_meshes();
        let inner_mesh = retyped.with_axis_types(["y"], MeshAxisType::Manual).unwrap();
        let unrelated = mesh.with_axis_types(["x"], MeshAxisType::Explicit).unwrap();
        let retypings = MeshRetypings::default();
        assert_eq!(retypings.retype_mesh(mesh.clone()), mesh);
        retypings.with_retyped_mesh(&mesh, &retyped, || {
            assert_eq!(retypings.retype_mesh(mesh.clone()), retyped);
            assert_eq!(retypings.retype_mesh(unrelated.clone()), unrelated);
            retypings.with_retyped_mesh(&retyped, &inner_mesh, || {
                assert_eq!(retypings.retype_mesh(mesh.clone()), inner_mesh);
                assert_eq!(retypings.retype_mesh(retyped.clone()), inner_mesh);
                assert_eq!(retypings.retype_mesh(unrelated.clone()), unrelated);
            });
            assert_eq!(retypings.retype_mesh(mesh.clone()), retyped);
        });
        assert_eq!(retypings.retype_mesh(mesh.clone()), mesh);
    }

    #[test]
    fn test_mesh_retypings_retype_named_axis() {
        let (mesh, retyped) = retyping_meshes();
        let unrelated = LogicalMesh::new(vec![MeshAxis::new("z", 5, MeshAxisType::Manual).unwrap()]).unwrap();
        let unrelated_axis = NamedAxis::Mesh { mesh: unrelated, axis: 0, size: 5, checks_manual_variation: false };
        let retypings = MeshRetypings::default();
        retypings.with_retyped_mesh(&mesh, &retyped, || {
            assert_eq!(
                retypings.retype_named_axis(NamedAxis::Mesh {
                    mesh: mesh.clone(),
                    axis: 0,
                    size: 2,
                    checks_manual_variation: true,
                }),
                NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: true },
            );
            assert_eq!(
                retypings.retype_named_axis(NamedAxis::Mesh {
                    mesh: mesh.clone(),
                    axis: 0,
                    size: 2,
                    checks_manual_variation: false,
                }),
                NamedAxis::Mesh { mesh: retyped.clone(), axis: 0, size: 2, checks_manual_variation: false },
            );
            assert_eq!(retypings.retype_named_axis(unrelated_axis.clone()), unrelated_axis);
            assert_eq!(
                retypings.retype_named_axis(NamedAxis::Batched { size: Some(3) }),
                NamedAxis::Batched { size: Some(3) },
            );
            assert_eq!(
                retypings.retype_named_axis(NamedAxis::Batched { size: None }),
                NamedAxis::Batched { size: None },
            );
        });
    }
}
