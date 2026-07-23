#[cfg(feature = "benchmarking")]
/// Internal benchmark-case definitions that stay within the plain `tracing_v2` staged IR.
pub(crate) mod benchmark_support;
#[cfg(feature = "benchmarking")]
/// IR benchmarking utilities that emit raw artifacts and normalized summaries for comparison.
pub mod benchmarking;
/// Semantic operation traits and built-in operation enums.
///
/// Per-op staging stays on small operation-local capability traits rather than on catch-all
/// `Supports*` bundles.
pub mod operations;
pub mod rematerialization;

#[cfg(test)]
pub(crate) mod test_util;

pub use crate::operations::math::{Cos, Sin};
pub use crate::operations::tag::{TAG_OPERATION_NAME, Tag, TagOperation};
pub use crate::tracing::NestedTracer;
pub use operations::custom_derivatives::transpose_primal_custom_vjp;
pub use rematerialization::{
    DotsSaveable, DotsWithNoBatchDimsSaveable, EitherStorage, EverythingSaveable, MemoryTransferStorage, NoStorage,
    NothingSaveable, OffloadDotsWithNoBatchDims, PolicyFn, RematerializationCandidate, RematerializationDecision,
    RematerializationError, RematerializationPolicy, RematerializationProducer, RematerializationRejection,
    RematerializationRejectionKind, Rematerialize, RematerializeOperation, ResidualStorage,
    SaveAndOffloadOnlyTheseNames, SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies,
    SaveOnlyTheseNames, rematerialize,
};
