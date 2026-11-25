//! Combinators that compose optimizers or apply them to subsets of the leaves, in the style of the
//! [Optax combinators](https://optax.readthedocs.io/en/latest/api/combining_optimizers.html).
//!
//!   - **Sequences:** [`Chain`] in [`chain`] applies two optimizers in order, and longer chains nest pairs.
//!   - **Subsets of leaves:** [`Masked`] in [`masking`] applies an optimizer to the leaves that a predicate on their
//!     [`ParameterPath`](ryft_core::ParameterPath)s selects and passes the other updates through unchanged, which with
//!     [`SetToZero`](crate::SetToZero) freezes the selected leaves (refer to [`Freeze`]). [`Partition`] in
//!     [`partition`] labels every leaf by its path and applies one optimizer per label, which with a frozen label
//!     gives [`SelectiveTransform`].
//!
//! The optimizers that [`Masked`] and [`Partition`] apply see only the selected leaves, as a [`SelectedLeaves`] map
//! from the paths of those leaves to the leaves, so the leaves of their states are named by the paths of the
//! parameters that they track. Combinators pass the same [`UpdateArguments`](crate::UpdateArguments) to every optimizer
//! that they apply, restricted to the selected leaves where they apply an optimizer to a subset.

pub mod chain;
pub mod masking;
pub mod partition;

pub use chain::Chain;
pub use masking::{Freeze, LeafPath, Masked, MaskedState, SelectedLeaves};
pub use partition::{Partition, PartitionState, SelectiveTransform};

// TODO(eaplatanios): Review this module.
