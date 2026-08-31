// An allocation ID is meaningful only in the discharge environment that minted it, so downstream rules cannot
// fabricate one from arbitrary coordinates. Its representation remains private even though the ID itself is public.

use ryft_core::ReferenceDischargeAllocationId;

// The diverging field initializer avoids needing to name the private environment-identity type while still asking the
// compiler to validate construction through the ID's private representation.
#[allow(unreachable_code)]
fn construct_allocation_id() -> ReferenceDischargeAllocationId {
    ReferenceDischargeAllocationId { index: 0, environment: unreachable!() }
}

fn main() {}
