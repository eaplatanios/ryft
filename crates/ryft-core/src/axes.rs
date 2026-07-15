use thiserror::Error;

use crate::batching::{BatchableOperation, BatchingContext, BatchingError};
use crate::contexts::{Context, EagerContext};
use crate::differentiation::{DifferentiableOperation, DifferentiationContext};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::constants::ZeroOperation;
use crate::operations::manipulation::{BroadcastOperation, TransposeOperation};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, Value};
use crate::tracing::{NestedTracingContext, TracingContext};
// TODO(eaplatanios): Should we move `AxisIndexOperation` to this module?
use crate::tracing_v2::operations::collective::AxisIndexOperation;
use crate::types::ArrayType;

/// Represents axis-related errors.
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum AxisError {
    #[error("axis name '{name}' is not bound by any enclosing transform")]
    UnboundAxisName { name: String },
}

/// A named axis resolved by a [`NamedAxes`] context specifying what an axis name is currently bound to, and by which
/// kind of transform, at a given trace level. This carries only the *value-free* facts about a binding (i.e., its kind
/// and size) not which physical dimension of any particular value carries the axis. That per--alue mapping is partial
/// (a replicated operand has no such dimension even though a collective over it is still meaningful) and is supplied at
/// consumption time by the owning transform's rule dispatch (e.g., for batching, an [`ArrayBatch`](crate::ArrayBatch)'s
/// [`batch_axis`](crate::ArrayBatch::batch_axis)).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NamedAxis {
    /// Axis bound by an enclosing batching (i.e., vectorization) level.
    Batched {
        /// Number of batch items along this axis.
        size: usize,
    },

    /// Axis bound to a device mesh axis by an enclosing manual sharding region.
    Mesh {
        /// Index of the mesh axis this name resolves to.
        axis: usize,

        /// Number of shards along this mesh axis.
        size: usize,
    },
}

/// Capability for resolving named axes visible at a trace level. Named axes are dynamically scoped binders introduced
/// by transforms (e.g., a batching level names the axis it introduces, and a manual sharding region names its mesh
/// axes) and read by named-axis primitives such as collective operations. Resolution walks the enclosing context stack
/// innermost-first, and so a nearer binder shadows a farther one. This capability answers only whether a name is
/// currently bound and, if so, what kind of axis it is and how large: the "is this name in scope" lookup used for
/// bind-time validation and scalar queries. It is deliberately *not* the evaluator. Specifying how a use site consumes
/// the axis (and against which physical dimension of a given operand) is the owning transform's per-operation rule's
/// responsibility, which already receives that per-value position at dispatch time.
pub trait NamedAxes: Context {
    /// Resolves `name` against this context, returning the [`NamedAxis`] it is bound to,
    /// or `None` when no enclosing binder binds it.
    fn named_axis(&self, name: &str) -> Option<NamedAxis>;
}

impl<V: Value, O: Operation<V::Type> + InterpretableOperation<EagerContext<V, O>>> NamedAxes for EagerContext<V, O> {
    #[inline]
    fn named_axis(&self, _name: &str) -> Option<NamedAxis> {
        // An eager context binds no named axes as it is a leaf of the resolution stack. So every lookup returns `None`.
        None
    }
}

impl<V: Value, O: Operation<V::Type>, C> NamedAxes for TracingContext<V, O, C> {
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

impl<C: NamedAxes<Type = ArrayType>> NamedAxes for BatchingContext<C>
where
    C::Operation: BatchableOperation<C>
        + BatchableOperation<TracingContext<C::Constant, C::Operation>>
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
            Some(NamedAxis::Batched { size: self.axis_size() })
        } else {
            self.parent().named_axis(name)
        }
    }
}

impl<C: NamedAxes> NamedAxes for DifferentiationContext<C>
where
    C::Operation: DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + From<ZeroOperation<C::Type>>,
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
    /// Returns a [`DataType::U64`](crate::DataType::U64) scalar giving the current element's position along `name`.
    /// What that position counts follows the kind of binder that introduced the axis (refer to the documentation of
    /// [`NamedAxis`] for more information): a batching axis of size `N` yields the current element's position in
    /// `0..N`, and a device mesh axis yields the current shard's coordinate along that mesh axis. `U64` matches the
    /// `usize` axis sizes the indices are drawn from and cannot be negative. A name that no enclosing binder binds will
    /// result in [`AxisError::UnboundAxisName`].
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

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::batching::BatchingError;
    use crate::contexts::EagerContext;
    use crate::tests::TestArray;
    use crate::tracing::DomainTracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{ArrayType, DataType};

    use super::*;

    #[test]
    fn test_axis_error_renders_unbound_axis_name() {
        let error = AxisError::UnboundAxisName { name: "batch".to_string() };
        assert_eq!(error.to_string(), "axis name 'batch' is not bound by any enclosing transform");
        assert_eq!(format!("{error:?}"), "UnboundAxisName { name: \"batch\" }");
        assert_eq!(error, AxisError::UnboundAxisName { name: "batch".to_string() });
        assert_ne!(error, AxisError::UnboundAxisName { name: "device".to_string() });
    }

    #[test]
    fn test_named_axis_equality_and_hashing() {
        assert_eq!(NamedAxis::Batched { size: 3 }, NamedAxis::Batched { size: 3 });
        assert_ne!(NamedAxis::Batched { size: 3 }, NamedAxis::Batched { size: 4 });
        assert_eq!(NamedAxis::Mesh { axis: 1, size: 2 }, NamedAxis::Mesh { axis: 1, size: 2 });
        assert_ne!(NamedAxis::Mesh { axis: 0, size: 2 }, NamedAxis::Mesh { axis: 1, size: 2 });

        // A batched axis never equals a mesh axis, even when their sizes match.
        assert_ne!(NamedAxis::Batched { size: 2 }, NamedAxis::Mesh { axis: 0, size: 2 });

        let axes = HashSet::from([NamedAxis::Batched { size: 3 }, NamedAxis::Mesh { axis: 1, size: 2 }]);
        assert!(axes.contains(&NamedAxis::Batched { size: 3 }));
        assert!(axes.contains(&NamedAxis::Mesh { axis: 1, size: 2 }));
        assert!(!axes.contains(&NamedAxis::Batched { size: 2 }));
    }

    #[test]
    fn test_axis_index_stages_a_nullary_operation_for_a_bound_axis() {
        // Validate `name` against the seeded `NamedAxes` environment and stage a nullary `AxisIndexOperation`
        // producing a scalar `u64`, regardless of whether the axis is batch- or mesh-bound.
        let (output_type, program) =
            DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::trace_with_named_axes(
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
        let error = DomainTracingContext::<EagerContext<TestArray, ArrayOperation<TestArray>>>::trace_with_named_axes(
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
}
