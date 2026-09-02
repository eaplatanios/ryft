mod regions;

pub use regions::{ReferenceDischargeRegionSummary, ReferenceDischargeStateWidening};
pub(in crate::programs::references::discharge) use regions::{summarize_region_closure, validate_region_accesses};
