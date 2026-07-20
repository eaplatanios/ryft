use thiserror::Error;

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
