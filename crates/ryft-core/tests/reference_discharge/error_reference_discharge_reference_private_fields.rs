// A downstream rule may hold a reference handle and read it through its accessors, but the fields behind those
// accessors are not part of its surface, so the allocation ID a handle carries stays checked rather than
// borrowable.
//
// Building a handle and fabricating an allocation ID are rejected in separate fixtures so each privacy contract
// produces its own compiler diagnostic.

use ryft_core::{Domain, ReferenceDischargeAllocationId, ReferenceDischargePolicy, ReferenceDischargeReference};

fn read_private_field<C: Domain, P: ReferenceDischargePolicy<C>>(
    reference: &ReferenceDischargeReference<C, P>,
) -> ReferenceDischargeAllocationId {
    reference.allocation_id
}

fn main() {}
