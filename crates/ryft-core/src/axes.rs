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
//! accepts exactly `[-rank, rank)` and returns a nonnegative position. [`Axes`] preserves an ordered collection of
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
//! A binding deliberately does not identify a dimension of every value. A replicated operand has no mapped dimension
//! even when a collective over the enclosing logical axis is meaningful. The operation rule that consumes the name
//! combines the binding with its transform-specific per-value metadata.
//!
//! # Axis Values
//!
//! [`NamedAxes`] answers whether a name is in scope and what it denotes; it does not produce a runtime value.
//! [`AxisIndex`] is the value-producing counterpart. It validates the binding, then returns a `u64` scalar containing
//! the current batch-item index or mesh-coordinate index according to the binder kind. The resulting
//! [`AxisIndexOperation`] remains an ordinary operation and therefore composes with interpretation, tracing, batching,
//! differentiation, and partial evaluation through their normal rule contracts.
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

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, DataType, Dimension, Shape};
use crate::batching::{BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError};
use crate::contexts::{Context, Domain, EagerContext, ProjectedContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationContext, ResidualZeroProvider,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_non_differentiable_operation, impl_nullary_transposable_operation};
use crate::operations::{BroadcastOperation, IotaOperation, TransposeOperation};
use crate::parameters::Parameter;
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationDriver, PartialEvaluationValue, PartiallyEvaluatableOperation,
};
use crate::programs::{
    Operation, OperationFormatter, OperationProjection, ProgramError, RegionInterface, Type, TypeError, Value,
    ValueProjection,
};
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

    /// Normalizes this [`Axis`] against `rank`, returning its nonnegative position. Valid axes lie in `[-rank, rank)`.
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
/// and any statically known size), not which dimension of any particular value carries the axis. That per-value mapping
/// is partial (a replicated operand has no such dimension even though a collective over it is still meaningful) and is
/// supplied at consumption time by the owning transform's rule dispatch (e.g., for batching, an [`ArrayBatch`]'s
/// [`batch_axis`](ArrayBatch::batch_axis)).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NamedAxis {
    /// Axis bound by an enclosing batching (i.e., vectorization) level.
    Batched {
        /// Number of batch items along this axis when statically known, or `None` when its extent is dynamic
        /// (i.e., not known statically at tracing time).
        size: Option<usize>,
    },

    /// Axis bound to a device mesh axis by an enclosing manual sharding region.
    Mesh {
        /// Index of the mesh axis this name resolves to.
        axis: usize,

        /// Number of shards along this mesh axis.
        size: usize,
    },
}

/// Capability for resolving named axes visible at one context-stack level. Named axes are dynamically scoped binders
/// introduced by transforms and manual sharding regions, then consumed by named-axis operations such as collectives.
/// Resolution is innermost-first, so a nearer binder shadows a farther one. The returned [`NamedAxis`] carries only
/// value-free kind and size facts; the owning operation rule remains responsible for how a use consumes that logical
/// axis and which physical dimension of each value carries it.
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
/// Implementations introduce only their local bindings. Every miss delegates outward unless the context is a leaf.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait NamedAxes: Context {
    /// Resolves `name` against this context, returning the [`NamedAxis`] it is bound to,
    /// or `None` when no enclosing binder binds it.
    fn named_axis(&self, name: &str) -> Option<NamedAxis>;
}

impl<V: Value, O: Operation<Type = V::Type> + InterpretableOperation<EagerContext<V, O>>> NamedAxes
    for EagerContext<V, O>
{
    #[inline]
    fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
        // An eager context binds no named axes as it is a leaf of the resolution stack. So every lookup returns `None`.
        None
    }
}

impl<C: NamedAxes, T: Type> NamedAxes for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // Projection changes only the visible type/value/operation member. Named-axis scope belongs to the parent
        // context stack and therefore passes through unchanged.
        self.parent().named_axis(name)
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> NamedAxes for TracingContext<V, O, C> {
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A `TracingContext` is a leaf of the resolution stack and it resolves only the named axes it was seeded with
        // (e.g., a `shard_map` body's device mesh axes) and reports every other name unbound. Ordinary traces are
        // seeded with no axes. Named-axis binders such as `BatchingContext` wrap a base trace and resolve against it.
        self.named_axes().iter().find(|(axis_name, _)| axis_name == name).map(|(_, axis)| *axis)
    }
}

impl<C: NamedAxes> NamedAxes for NestedTracingContext<C> {
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A lookup resolves against the axes this nested trace was seeded with first, and otherwise delegates to the
        // parent context it is nested into, because named axes are dynamically scoped: a seeded binding shadows an
        // enclosing one, while a collective staged inside an unseeded nested tracing context still resolves an axis
        // bound by an enclosing transform.
        self.named_axes()
            .iter()
            .find(|(axis_name, _)| axis_name == name)
            .map(|(_, axis)| *axis)
            .or_else(|| self.parent().named_axis(name))
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
}

impl<C: NamedAxes<Type = ArrayType>> NamedAxes for BatchingContext<C, ArrayBatching>
where
    C::Operation: BatchableOperation<C, ArrayBatching>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>, ArrayBatching>
        + From<TransposeOperation>
        + From<BroadcastOperation>,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A batching level binds the axis it introduces: a lookup for this level's `axis_name` resolves to
        // `NamedAxis::Batched` with this level's batch size, and any other name delegates to the parent context.
        // Because nested batching composes by context wrapping, the delegation chain naturally shadows outer
        // bindings with inner ones.
        if self.axis_name() == Some(name) {
            Some(NamedAxis::Batched { size: Some(*self.axis_extent()) })
        } else {
            self.parent().named_axis(name)
        }
    }
}

impl<C: NamedAxes> NamedAxes for DifferentiationContext<C>
where
    C::Type: DifferentiableType,
    C::Operation: DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type>,
{
    #[inline]
    fn named_axis(&self, name: &str) -> Option<NamedAxis> {
        // A `DifferentiationContext` binds no named axes of its own: axis-name resolution passes through to the inner
        // context, so collectives inside a differentiated closure resolve against the enclosing batching levels and
        // mesh regions.
        self.parent().named_axis(name)
    }
}

/// Capability to read the index of the current element along a named axis. This is the value-producing counterpart of
/// [`NamedAxes`]. `NamedAxes` answers whether a name is in scope, while [`AxisIndex`] reads out the position along it.
/// Resolution is validated against the active [`NamedAxes`] environment.
pub trait AxisIndex: Context {
    /// Returns a [`DataType::U64`] scalar giving the current element's position along `name`. What that position counts
    /// follows the kind of binder that introduced the axis (refer to the documentation of [`NamedAxis`] for more
    /// information): a batching axis of size `N` yields the current element's position in `0..N`, and a device mesh
    /// axis yields the current shard's coordinate along that mesh axis. `U64` matches the `usize` axis sizes the
    /// indices are drawn from and cannot be negative. A name that no enclosing binder binds will result in
    /// [`AxisError::UnboundAxisName`].
    fn axis_index(&self, name: &str) -> Result<Self::Value, ProgramError>;
}

impl<C: Context<Operation: From<AxisIndexOperation>> + NamedAxes> AxisIndex for C {
    fn axis_index(&self, name: &str) -> Result<Self::Value, ProgramError> {
        // Every context reads an axis index the same way. It validates `name` against the active `NamedAxes`
        // environment and then binds a nullary `AxisIndexOperation`, so the caller needs no knowledge of whether `name`
        // is a batching or mesh axis. That operation carries the per-axis-kind resolution as it flows outward: the
        // batching level that bound `name` consumes it (its batching rule materializes the per-element index), an inner
        // batching level re-binds it into its parent, and a mesh axis survives into the base program to lower during
        // sharded execution (refer to the documentation of `AxisIndexOperation`). Because resolution happens as the
        // operation is consumed, a batched axis reached across a non-batching wrapper that *interprets* a nested
        // program (e.g., an outer batch addressed from inside a `jvp` trace, whose primal program is spliced by
        // interpretation) is not supported. The operation is interpreted before any batching rule can consume it
        // and reports `ProgramError::UnsupportedOperation`. Mesh axes are unaffected, as they are meant to survive
        // interpretation.
        if self.named_axis(name).is_none() {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: name.to_string() }).into());
        }
        let mut outputs = self.bind(AxisIndexOperation::new(name.to_string()), Vec::new(), &[])?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Canonical operation name for [`AxisIndexOperation`].
pub const AXIS_INDEX_OPERATION_NAME: &str = "axis_index";

/// Nullary primitive [`Operation`] that produces the current batch item's or device shard's index along a
/// [`NamedAxis`] as a scalar [`DataType::U64`] value. This is the Ryft analogue of JAX's
/// [`axis_index`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.axis_index.html). [`AxisIndex::axis_index`]
/// stages this operation uniformly for every axis kind and resolution depends on the enclosing binder. A *batched* axis
/// is consumed by this operation's staged batching rule at the `batch` level that binds it, which materializes the
/// per-item index as an [`iota`](crate::operations::constants::Iota) over the known batch size. This means that an
/// `AxisIndexOperation` for a batched axis never survives into a staged body. A *device mesh* axis has no such
/// trace-time binder. Its per-device coordinate is known only at execution time, and so the operation stays in the
/// staged body and lowers inside a `shard_map` manual region to `partition_id`-based coordinate arithmetic. Only mesh
/// uses therefore reach interpretation, which is why this operation is *not* eagerly interpretable and, having no
/// operands, is [partially evaluated](PartiallyEvaluatableOperation) by residualizing rather than folding (that is
/// because folding a nullary operation would result in trying to interpret it).
///
/// # Examples
///
/// The following batches a function over a named `items` axis and returns each item's index along that axis:
///
/// ```rust
/// # use ryft_core::{Array, AxisIndex, BatchAxis, BatchAxisSpecification, batch};
/// #
/// let indices: Array = batch(
///     |item| item.context().axis_index("items"),
///     Array::vector(vec![10.0, 20.0, 30.0]),
///     BatchAxis::new(0),
///     BatchAxis::new(0),
///     BatchAxisSpecification::named("items"),
/// )
/// .unwrap();
/// assert_eq!(indices.to_f64s(), vec![0.0, 1.0, 2.0]);
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct AxisIndexOperation {
    /// Name of the device mesh axis whose per-shard index this [`AxisIndexOperation`] produces.
    axis_name: String,
}

impl AxisIndexOperation {
    /// Creates a new [`AxisIndexOperation`] referencing the mesh axis `axis_name`.
    #[inline]
    pub fn new(axis_name: String) -> Self {
        Self { axis_name }
    }

    /// Returns the mesh axis name referenced by this [`AxisIndexOperation`].
    #[inline]
    pub fn axis_name(&self) -> &str {
        &self.axis_name
    }
}

impl Display for AxisIndexOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for AxisIndexOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        AXIS_INDEX_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 0, TypeError);
        Ok(vec![ArrayType::scalar(DataType::U64)])
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, AXIS_INDEX_OPERATION_NAME)?
            .bracketed(|operation| operation.field("axis_name", format_args!("{:?}", self.axis_name)))
    }
}

impl<C: Domain<Type = ArrayType>> InterpretableOperation<C> for AxisIndexOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        // A mesh axis index is a per-device coordinate that only exists during sharded execution. There is no eager
        // value to produce. It is lowered inside a `shard_map` manual region and never interpreted directly.
        check_count!("input", inputs, 0, ProgramError);
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "`{}` for the device mesh axis `{}` has no eager value; \
                it is only defined inside a `shard_map` manual region",
                AXIS_INDEX_OPERATION_NAME, self.axis_name,
            ),
        })
    }
}

impl<C: Context<Type = ArrayType, Operation: From<AxisIndexOperation>>> PartiallyEvaluatableOperation<C>
    for AxisIndexOperation
{
    #[inline]
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        _driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Partial evaluation always residualizes an `AxisIndexOperation`. Its value depends on the executing device,
        // and so it is never a foldable constant even though it has no (known) inputs.
        context.residualize(self.clone(), Vec::new(), inputs)
    }
}

impl_non_differentiable_operation!(AxisIndexOperation);
impl_nullary_transposable_operation!(AxisIndexOperation);

impl<
    C: Context<Type = ArrayType, Operation: From<IotaOperation<ArrayType>> + From<AxisIndexOperation>>,
    P: ArrayBatchingPolicy<C>,
> BatchableOperation<C, ArrayBatching<P>> for AxisIndexOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        _inputs: &[ArrayBatch<<C as Domain>::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        if context.axis_name() == Some(self.axis_name.as_str()) {
            // This level binds the axis. The per-item index is the length-`size` `iota(0)`, bound into the parent and
            // mapped on this level's batch axis (position 0). The mapped packed `[size]` dimension is then stripped
            // back to the per-item scalar `u64`.
            let size = P::axis_size(context)?;
            let r#type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(size)]));
            let operation = IotaOperation::new(r#type.clone(), 0)?;
            let mut index = context.parent().bind(operation, Vec::new(), &[])?;
            check_count!("output", index, 1, ProgramError);
            Ok(vec![ArrayBatch::new(index.remove(0), Some(0))?].into())
        } else {
            // The axis is bound by an outer `batch` level or a device mesh. Re-bind into the parent, which repeats the
            // resolution and present the forwarded index as replicated across this level.
            let operation = AxisIndexOperation::new(self.axis_name.clone());
            let mut index = context.parent().bind(operation, Vec::new(), &[])?;
            check_count!("output", index, 1, ProgramError);
            Ok(vec![ArrayBatch::replicated(index.remove(0))].into())
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType, Dimension, Shape};
    use crate::batching::{Batch, BatchAxis, BatchAxisSpecification, BatchingError, batch};
    use crate::contexts::EagerContext;
    use crate::programs::Typed;
    use crate::tracing::DomainTracingContext;

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
        assert_eq!(Axis::from(3).normalize(3), Err(AxisError::OutOfBounds { axis: Axis::from(3), rank: 3 }),);
        assert_eq!(Axis::from(-4).normalize(3), Err(AxisError::OutOfBounds { axis: Axis::from(-4), rank: 3 }),);
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
        assert_eq!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: Some(3) });
        assert_ne!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: Some(4) });
        assert_ne!(NamedAxis::Batched { size: Some(3) }, NamedAxis::Batched { size: None });
        assert_eq!(NamedAxis::Mesh { axis: 1, size: 2 }, NamedAxis::Mesh { axis: 1, size: 2 });
        assert_ne!(NamedAxis::Mesh { axis: 0, size: 2 }, NamedAxis::Mesh { axis: 1, size: 2 });

        // A batched axis never equals a mesh axis, even when their sizes match.
        assert_ne!(NamedAxis::Batched { size: Some(2) }, NamedAxis::Mesh { axis: 0, size: 2 });

        let axes = HashSet::from([NamedAxis::Batched { size: Some(3) }, NamedAxis::Mesh { axis: 1, size: 2 }]);
        assert!(axes.contains(&NamedAxis::Batched { size: Some(3) }));
        assert!(axes.contains(&NamedAxis::Mesh { axis: 1, size: 2 }));
        assert!(!axes.contains(&NamedAxis::Batched { size: Some(2) }));
    }

    #[test]
    fn test_axis_index_stages_a_nullary_operation_for_a_bound_axis() {
        // Validate `name` against the seeded `NamedAxes` environment and stage a nullary `AxisIndexOperation`
        // producing a scalar `u64`, regardless of whether the axis is batch- or mesh-bound.
        let (output_type, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |input| input.context().axis_index("device"),
                ArrayType::scalar(DataType::F64),
                vec![("device".to_string(), NamedAxis::Mesh { axis: 0, size: 4 })],
            )
            .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::U64));
        assert_eq!(
            program.to_string(),
            indoc! {r#"
                lambda %0:f64[] .
                let %1:u64[] = axis_index [axis_name="device"]
                in (%1)"#},
        );
    }

    #[test]
    fn test_axis_index_rejects_an_unbound_axis() {
        // A name that no enclosing binder binds fails fast at the reader, before any operation is staged, surfacing
        // `AxisError::UnboundAxisName` through the `BatchingError::Axis` channel riding `ProgramError`.
        let error = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
            |input| input.context().axis_index("missing"),
            ArrayType::scalar(DataType::F64),
            vec![("device".to_string(), NamedAxis::Mesh { axis: 0, size: 4 })],
        )
        .unwrap_err();
        assert!(matches!(
            error.downcast_custom::<BatchingError>(),
            Some(BatchingError::Axis(AxisError::UnboundAxisName { name })) if name == "missing",
        ));
    }

    #[test]
    fn test_batch_axis_index_produces_per_item_indices() {
        // `axis_index("i")` gives each batch item its own position along the mapped axis `"i"` (size 3), so the
        // batched result is the `u64` index vector `[0, 1, 2]` regardless of the operand values.
        let output: Array = batch(
            |item| item.context().axis_index("i"),
            Array::vector(vec![10.0, 20.0, 30.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        )
        .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(3)])));
        assert_eq!(output.to_f64s(), vec![0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_nested_batch_axis_index_forwards_outer_axis_through_inner_level() {
        // Outer `batch` over axis 0 (size 2, named "o") of a [2, 3] matrix; inner `batch` over axis 0 (size 3, named
        // "i") of each row. The inner body asks for `axis_index("o")`, which the inner level does not bind, so it is
        // forwarded to the outer level and re-wrapped as replicated across the inner axis (the outer index does not
        // vary over inner items). The inner output is therefore declared replicated, and the outer level stacks the
        // per-row outer index, giving the `u64` vector `[0, 1]`.
        let x = Array::matrix(2, 3, vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |row| {
                    let context = row.context().clone();
                    Ok(Batch::batch(
                        &context,
                        |scalar| scalar.context().axis_index("o"),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::replicated(),
                        BatchAxisSpecification::named("i"),
                    )?)
                },
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::named("o"),
            )
            .unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(2)])));
        assert_eq!(output.to_f64s(), vec![0.0, 1.0]);
    }

    #[test]
    fn test_batch_axis_index_rejects_unbound_axis() {
        // `axis_index` over a name no enclosing batch binds fails fast, mirroring the collective readers.
        let result: Result<Array, BatchingError> = EagerContext::<Array, ArrayOperation<Array>>::new().batch(
            |item| item.context().axis_index("j"),
            Array::vector(vec![10.0, 20.0, 30.0]),
            BatchAxis::new(0),
            BatchAxis::new(0),
            BatchAxisSpecification::named("i"),
        );
        assert_eq!(result.unwrap_err(), BatchingError::Axis(AxisError::UnboundAxisName { name: "j".to_string() }));
    }
}
