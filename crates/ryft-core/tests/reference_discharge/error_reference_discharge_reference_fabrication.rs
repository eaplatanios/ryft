// Only the discharge context mints a reference handle, so a downstream rule cannot build one: doing so would provide
// an allocation ID, an alias, and a derived reference type that no environment ever checked.
//
// Constructing the allocation ID and reading a reference handle's fields are rejected in separate fixtures so each
// privacy contract produces its own compiler diagnostic.

use ryft_core::{
    Domain, ReferenceDischargeAllocationId, ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceType,
};

fn construct_reference<C: Domain, P: ReferenceDischargePolicy<C>>(
    allocation_id: ReferenceDischargeAllocationId,
    alias: P::Alias,
    r#type: ReferenceType<P::Referent>,
) -> ReferenceDischargeReference<C, P> {
    ReferenceDischargeReference { allocation_id, alias, r#type }
}

fn main() {}
