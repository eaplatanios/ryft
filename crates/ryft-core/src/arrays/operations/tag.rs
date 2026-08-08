//! Reference [`Array`] answer to the value-tagging operation family contract.
//!
//! Tagging is metadata for staged programs only, so a concrete array carries itself through unchanged.

use crate::arrays::arrays::Array;
use crate::operations::Tag;

// TODO(eaplatanios): Review this.

impl Tag for Array {
    #[inline]
    fn tag(self, _key: &str) -> Self {
        self
    }
}
