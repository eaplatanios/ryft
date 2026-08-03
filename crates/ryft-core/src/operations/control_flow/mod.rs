pub mod condition;
pub mod scan;
pub mod select;
pub mod r#while;

use crate::programs::{Operation, Type, TypeError};

pub use condition::{CONDITION_OPERATION_NAME, ConditionOperation, transpose_primal_condition};
pub use scan::{SCAN_OPERATION_NAME, ScanOperation, transpose_primal_scan};
pub use select::{SELECT_OPERATION_NAME, Select, SelectOperation};
pub use r#while::{WHILE_OPERATION_NAME, WhileOperation, WhilePredicate, WhileTypeSemantics};
pub(crate) use r#while::{WhileResidualStackOperation, WhileResidualStackType, jvp_array_program_while};

/// Type-family storage policy for values that must cross an iteration boundary as stacked residuals.
///
/// Array residuals store themselves directly. A composite backend can instead assign a checked array-backed storage
/// type to metadata values such as first-class dimensions, keeping the temporal representation explicit in SSA.
pub(crate) trait TemporalResidualType: Type {
    /// Returns the per-iteration array-backed storage type for this residual.
    fn temporal_storage_type(&self) -> Result<Self, TypeError>;
}

/// Operation-family conversions paired with [`TemporalResidualType`].
///
/// Returning `None` means the residual already uses its storage representation. Returning an operation makes the
/// conversion visible in the generated program before stacking or after slicing one iteration's stored value.
pub(crate) trait TemporalResidualOperation<T: TemporalResidualType>: Operation<Type = T> {
    /// Returns the operation that converts a residual to temporal storage, if conversion is required.
    fn residual_to_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;

    /// Returns the operation that restores a residual from temporal storage, if conversion is required.
    fn residual_from_storage(residual_type: &T) -> Result<Option<Self>, TypeError>;
}
