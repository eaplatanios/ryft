// A downstream rule may hold a reference handle and read it through its accessors, but the fields behind those
// accessors are not part of its surface, so the allocation identity a handle carries stays checked rather than
// borrowable.
//
// Building a handle is rejected separately, in `error_reference_handle_fabrication.rs`, because the two rejections
// belong to different compiler passes and only the earlier one is reported when they share a file.

use ryft_core::{Domain, ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceAllocationHandle};

fn read_private_field<C: Domain, P: ReferenceDischargePolicy<C>>(
    reference: &ReferenceDischargeReference<C, P>,
) -> ReferenceAllocationHandle {
    reference.allocation
}

fn main() {}
