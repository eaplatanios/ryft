//! Reference [`Array`] answer to the memory operation family contract.
//!
//! The reference payload is host-resident regardless of the requested memory, so a transfer updates only the
//! [`Memory`](crate::arrays::Memory) carried by the array's type.

use crate::arrays::arrays::Array;
use crate::operations::TransferToMemory;

// TODO(eaplatanios): Review this.

impl TransferToMemory for Array {
    /// Re-places this [`Array`] in `destination` by updating the [`Memory`](crate::arrays::Memory) carried by its
    /// type. The payload is host-resident either way, but the carried type must reflect the transfer so that staged
    /// programs whose declared types park values in other memories (e.g., offloaded residuals) accept the
    /// interpreted value.
    #[inline]
    fn transfer_to_memory(&self, destination: crate::arrays::Memory) -> Self {
        Self { r#type: self.r#type.clone().with_memory(destination), bytes: self.bytes.clone() }
    }
}
