//! Defines positional array axes and dynamically scoped named axes used by array operations and program transforms.
//!
//! Positional [`Axis`] values identify dimensions within one concrete array rank. Named axes instead identify
//! logical transform dimensions—such as a vectorized batch or device-mesh axis—through the active [`Context`] stack.
//! The two forms meet inside operation-owned rules: a named binding supplies value-free scope metadata, while the
//! rule supplies the physical dimension of each participating value when one exists. Refer to [`NamedAxes`] for a
//! rendered diagram of named-axis lookup and consumption.
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

use std::fmt::Display;
use std::ops::Deref;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::arrays::LogicalMesh;
use crate::batching::{BatchableOperation, BatchingContext, RecursiveBatchingPolicy};
use crate::contexts::{Context, DomainProjection, EagerContext, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, DifferentiationPolicy, ResidualZeroProvider,
};
use crate::interpretation::InterpretableOperation;
use crate::parameters::Parameter;
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
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
}

/// Capability for resolving named axes visible at one context-stack level. Named axes are dynamically scoped binders
/// introduced by transforms and manual sharding regions, then consumed by named-axis operations such as collectives.
/// Resolution is innermost-first, so a nearer binder shadows a farther one. The returned [`NamedAxis`] carries only
/// value-free kind, size, and mesh facts; the owning operation rule remains responsible for how a use consumes that
/// logical axis and which physical dimension of each value carries it.
///
/// # Dynamic-Scope Lookup
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   request["Operation Requests an Axis Name"] --> current["Current Context"]
///   current --> local["Check Local Named-Axis Bindings"]
///   local -->|"nearest local binding"| binding["Named Axis: Batched or Mesh"]
///   local -->|"not bound locally"| parent["Delegate to Parent Context"]
///   parent --> lookup["Repeat Innermost-First Lookup"]
///   lookup -->|"binding found"| binding
///   lookup -->|"no enclosing binding"| unbound["Unbound Axis Error"]
///   binding --> facts["Value-Free Kind and Optional Static Size"]
///   facts --> rule["Operation-Owned Rule"]
///   per_value["Per-Value Mapped-Axis Metadata"] --> rule
///   facts --> axis_index["&lt;code&gt;AxisIndex&lt;/code&gt; Capability"]
///   axis_index --> value["Current Index Value"]
/// ```
///
/// Lookup (i.e., [`NamedAxes::named_axis`]) checks local bindings before delegating outward unless the context is a
/// leaf. Enumeration (i.e., [`NamedAxes::named_axes`]) includes the enclosing bindings with the same shadowing rules.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait NamedAxes: Context {
    /// Resolves `name` against this context, returning the [`NamedAxis`] it is bound to,
    /// or `None` when no enclosing binder binds it.
    fn named_axis(&self, name: &str) -> Option<NamedAxis>;

    /// Returns the named axes in scope at this context, innermost first, with each name once (i.e., without the
    /// bindings that nearer binders shadow), so that [`Self::named_axis`] resolves a name to the binding that this
    /// function returns for it. Functions that trace user code in a fresh trace instead of in the context that they
    /// are called in (e.g., [`CustomFunction::call`](crate::CustomFunction::call) for its primal and rules) seed that
    /// trace with these bindings, so that the code resolves the same names as it would in the calling context.
    fn named_axes(&self) -> Vec<(String, NamedAxis)>;
}

impl<V: Value, O: Operation<Type = V::Type> + InterpretableOperation<EagerContext<V, O>>> NamedAxes
    for EagerContext<V, O>
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

impl<T: Type, C: NamedAxes + DomainProjection<T>> NamedAxes for ProjectedContext<C, T> {
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

impl<V: Value, O: Operation<Type = V::Type>, C> NamedAxes for TracingContext<V, O, C> {
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A `TracingContext` is a leaf of the resolution stack and it resolves only the named axes it was seeded with
        // (e.g., a `shard_map` body's device mesh axes) and reports every other name unbound. Ordinary traces are
        // seeded with no axes. Named-axis binders such as `BatchingContext` wrap a base trace and resolve against it.
        self.local_named_axes()
            .iter()
            .find(|(axis_name, _)| axis_name == name)
            .map(|(_, axis)| axis.clone())
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        // The first of several seeded bindings of one name is the one that `named_axis` resolves.
        let mut axes = Vec::<(String, NamedAxis)>::new();
        for (name, axis) in self.local_named_axes() {
            if !axes.iter().any(|(visible, _)| visible == name) {
                axes.push((name.clone(), axis.clone()));
            }
        }
        axes
    }
}

impl<C: NamedAxes> NamedAxes for NestedTracingContext<C> {
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
    }

    fn named_axes(&self) -> Vec<(String, NamedAxis)> {
        // Seeded bindings shadow the parent's bindings of the same names, and the first of several seeded bindings
        // of one name is the one that `named_axis` resolves.
        let mut axes = Vec::<(String, NamedAxis)>::new();
        for (name, axis) in self.local_named_axes().iter().cloned().chain(self.parent().named_axes()) {
            if !axes.iter().any(|(visible, _)| visible == &name) {
                axes.push((name, axis));
            }
        }
        axes
    }
}

impl<C: NamedAxes> NamedAxes for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
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

impl<C: NamedAxes<Operation: BatchableOperation<C, P>>, P: RecursiveBatchingPolicy<C>> NamedAxes
    for BatchingContext<C, P>
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
    C::Type: DifferentiableType,
    C::Operation: DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>,
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use pretty_assertions::assert_eq;

    use crate::arrays::{MeshAxis, MeshAxisType};

    use super::*;

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
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 }
        );
        assert_ne!(
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 }
        );

        // A batched axis never equals a mesh axis, even when their sizes match.
        assert_ne!(NamedAxis::Batched { size: Some(2) }, NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2 });

        let axes = HashSet::from([
            NamedAxis::Batched { size: Some(3) },
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 },
        ]);
        assert!(axes.contains(&NamedAxis::Batched { size: Some(3) }));
        assert!(axes.contains(&NamedAxis::Mesh { mesh: mesh.clone(), axis: 1, size: 2 }));
        assert!(!axes.contains(&NamedAxis::Batched { size: Some(2) }));
    }

    #[test]
    fn test_named_axis_size() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        assert_eq!(NamedAxis::Batched { size: Some(3) }.size(), Some(3));
        assert_eq!(NamedAxis::Batched { size: None }.size(), None);
        assert_eq!(NamedAxis::Mesh { mesh, axis: 0, size: 2 }.size(), Some(2));
    }
}
