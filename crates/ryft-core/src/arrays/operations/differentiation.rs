//! Reference [`Array`] answer to the differentiation operation family contracts.
//!
//! Stopping the gradient is a value-level identity: the barrier lives in the transform rules of the corresponding
//! operation, so a concrete array simply carries itself through.

use crate::arrays::arrays::Array;
use crate::operations::StopGradient;

// TODO(eaplatanios): Review this.

impl StopGradient for Array {
    #[inline]
    fn stop_gradient(&self) -> Self {
        self.clone()
    }
}
