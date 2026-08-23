// Only the discharge context mints a reference handle, so a downstream rule cannot build one: doing so would install
// a root identity, an alias, and a derived reference type that no environment ever checked. Nor can it mint the root
// identity itself, whose own environment field is not even nameable downstream, which is what keeps a handle
// addressable only in the environment that produced it.
//
// Reading a handle's fields is rejected separately, in `error_reference_handle_private_fields.rs`, because the two
// rejections belong to different compiler passes and only the earlier one is reported when they share a file.

use ryft_core::{Domain, ReferenceDischargePolicy, ReferenceDischargeReference, ReferenceRootHandle, ReferenceType};

fn construct_handle<C: Domain, P: ReferenceDischargePolicy<C>>(
    root: ReferenceRootHandle,
    alias: P::Alias,
    r#type: ReferenceType<P::Referent>,
) -> ReferenceDischargeReference<C, P> {
    ReferenceDischargeReference { root, alias, r#type, preserved: None }
}

// The diverging field initializer keeps the snapshot to the privacy rejections themselves: the environment identity
// has no downstream-nameable value to supply, which is the point.
#[allow(unreachable_code)]
fn construct_root_handle() -> ReferenceRootHandle {
    ReferenceRootHandle { index: 0, environment: unreachable!() }
}

fn main() {}
