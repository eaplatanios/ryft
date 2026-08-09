/// Higher-order custom-derivative operations (`custom_jvp` / `custom_vjp`).
pub mod custom_derivatives;
mod operands;
pub mod rematerialization;

pub use crate::operations::{Cos, Sin, TAG_OPERATION_NAME, Tag, TagOperation};
pub use crate::tracing::NestedTracer;
pub use custom_derivatives::{CustomJvp, CustomVjp, custom_jvp, custom_vjp};
pub use rematerialization::{
    DotsSaveable, DotsWithNoBatchDimsSaveable, EitherStorage, EverythingSaveable, MemoryTransferStorage, NoStorage,
    NothingSaveable, OffloadDotsWithNoBatchDims, PolicyFn, RematerializationCandidate, RematerializationDecision,
    RematerializationError, RematerializationPolicy, RematerializationProducer, RematerializationRejection,
    RematerializationRejectionKind, Rematerialize, RematerializeOperation, ResidualStorage,
    SaveAndOffloadOnlyTheseNames, SaveAnyNamesButThese, SaveAnythingExceptTheseNames, SaveFromBothPolicies,
    SaveOnlyTheseNames, rematerialize,
};
