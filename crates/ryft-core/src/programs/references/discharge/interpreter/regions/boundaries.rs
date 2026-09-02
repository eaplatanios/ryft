use std::collections::BTreeSet;

use crate::programs::references::discharge::transform::ReferenceDischargeAllocationId;

// TODO(eaplatanios): Review this module.

/// Symmetric widening facts one structured rule derives from a region summary through
/// [`state_widening`](crate::programs::references::ReferenceDischargeContext::state_widening).
///
/// The three sets state one algorithm every symmetric structured rewrite shares: the *threaded* allocations are the
/// discharged references crossing as immutable state, the *entering* allocations gain added positions because no
/// declared position already carries them, and the *published* allocations are the discharged references whose final
/// states the rebuilt regions must return. An entering preserved reference crosses in its added position as a
/// reference, so it belongs to neither the threaded nor the published set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceStateWidening {
    /// Every discharged reference the region closures reach, in canonical allocation order.
    pub(super) threaded: BTreeSet<ReferenceDischargeAllocationId>,

    /// Reached allocations gaining added boundary positions, in canonical allocation order. Discharged allocations
    /// cross as state and preserved allocations cross as references.
    pub(super) entering: Vec<ReferenceDischargeAllocationId>,

    /// Threaded allocations some closure mutates, in canonical allocation order.
    pub(super) published: Vec<ReferenceDischargeAllocationId>,
}

impl ReferenceStateWidening {
    /// Returns every discharged reference the region closures reach, in canonical allocation order.
    #[inline]
    pub fn threaded(&self) -> &BTreeSet<ReferenceDischargeAllocationId> {
        &self.threaded
    }

    /// Returns the reached allocations gaining added boundary positions, in canonical allocation order.
    #[inline]
    pub fn entering(&self) -> &[ReferenceDischargeAllocationId] {
        self.entering.as_slice()
    }

    /// Returns the threaded allocations some closure mutates, in canonical allocation order.
    #[inline]
    pub fn published(&self) -> &[ReferenceDischargeAllocationId] {
        self.published.as_slice()
    }
}
