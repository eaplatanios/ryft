//! Reference [`Array`] answer to the custom-call operation family contract.
//!
//! The reference backend has no foreign-kernel registry, so it reports every custom call as an unsupported
//! operation instead of silently producing a value.

use crate::arrays::arrays::Array;
use crate::arrays::types::arrays::ArrayType;
use crate::operations::custom_call::{CustomCall, CustomCallOperation};
use crate::programs::ProgramError;

// TODO(eaplatanios): Review this.

impl CustomCall for Array {
    /// The reference array backend has no foreign-kernel registry, so custom calls always report an
    /// [`UnsupportedOperation`](ProgramError::UnsupportedOperation) error.
    fn custom_call<'a, I: IntoIterator<Item = &'a Self>>(
        operation: &CustomCallOperation<ArrayType>,
        _inputs: I,
    ) -> Result<Vec<Self>, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "the reference array backend cannot execute the foreign kernel '{}'",
                operation.target_name(),
            ),
        })
    }
}
