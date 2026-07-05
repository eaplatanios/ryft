use thiserror::Error;

use crate::batching::BatchingError;
use crate::contexts::{Context, StagingContext};
use crate::macros::check_count;
use crate::programs::ProgramError;
use crate::tracing_v2::operations::collective::AxisIndexOperation;

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

impl<C: StagingContext<Operation: From<AxisIndexOperation>> + NamedAxes> AxisIndex for C {
    fn axis_index(&self, name: &str) -> Result<Self::Value, ProgramError> {
        // Every staging context reads an axis index the same way: validate `name` against the active `NamedAxes`
        // environment and then stage a nullary `AxisIndexOperation`, so the reader needs no knowledge of whether `name`
        // is a batching or mesh axis. That staged operation carries the per-axis-kind resolution as it flows outward
        // through `stage_operation`: the batching level that bound `name` consumes it (its batching rule materializes
        // the per-element index) an inner batching level re-stages it into its parent, and a mesh axis survives into
        // the base program to lower during sharded execution (refer to the documentation of `AxisIndexOperation`).
        // Because resolution happens as the operation is consumed, a batched axis reached across a non-batching wrapper
        // that *interprets* a nested program (e.g., an outer batch addressed from inside a `jvp` trace, whose primal
        // program is spliced by interpretation) is not supported; the operation is interpreted before any batching rule
        // can consume it and reports `ProgramError::UnsupportedOperation`. Mesh axes are unaffected, as they are meant
        // to survive interpretation.
        if self.named_axis(name).is_none() {
            return Err(BatchingError::Axis(AxisError::UnboundAxisName { name: name.to_string() }).into());
        }
        let mut outputs = self.stage_nullary_operation(AxisIndexOperation::new(name.to_string()))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}
