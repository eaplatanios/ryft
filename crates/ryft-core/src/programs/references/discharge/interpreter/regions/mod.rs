mod analysis;
mod boundaries;

pub use analysis::ReferenceDischargeRegionSummary;
pub(in crate::programs::references::discharge) use analysis::{summarize_region_closure, validate_region_accesses};
pub use boundaries::ReferenceDischargeStateWidening;
