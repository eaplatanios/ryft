use ryft_xla_sys::mlir::dialects::triton::ttg::{
    mlirTritonTtgBarrierAddressSpacesAttrGet, mlirTritonTtgBarrierScopeAttrGet,
};

use crate::{
    Attribute, AttributeRef, DetachedOp, DialectHandle, Error, Location, Operation, OperationBuilder, mlir_op,
    mlir_op_trait,
};

use super::attributes::{BarrierAddressSpaces, BarrierScope};

/// Name of the barrier address-space mask attribute.
pub const ADDRESS_SPACES_ATTRIBUTE: &str = "addrSpace";

/// Name of the barrier scope attribute.
pub const SCOPE_ATTRIBUTE: &str = "scope";

/// TritonGPU barrier. All threads in its scope must reach the barrier. On return, earlier accesses in its selected
/// address spaces are visible within that scope.
///
/// # Example
///
/// ```mlir
/// ttg.barrier global_read|global_write
/// ```
///
/// Refer to the [official Triton documentation](https://triton-lang.org/main/dialects/TritonGPUOps.html#ttg-barrier)
/// for supported scopes and address spaces.
pub trait BarrierOperation<'o, 'c: 'o, 't: 'c>: Operation<'o, 'c, 't> {
    /// Returns the memory accesses ordered by this barrier.
    fn address_spaces(&self) -> Result<BarrierAddressSpaces, Error> {
        let value = self.integer_attribute(ADDRESS_SPACES_ATTRIBUTE)?.signless_value();
        u32::try_from(value)
            .ok()
            .and_then(BarrierAddressSpaces::from_bits)
            .ok_or_else(|| Error::invalid_argument("invalid address spaces in `ttg.barrier`"))
    }

    /// Returns the threads that participate in this barrier.
    fn scope(&self) -> Result<BarrierScope, Error> {
        if !self.has_attribute(SCOPE_ATTRIBUTE) {
            return Ok(BarrierScope::Cta);
        }
        match self.integer_attribute(SCOPE_ATTRIBUTE)?.signless_value() {
            0 => Ok(BarrierScope::Cta),
            1 => Ok(BarrierScope::Warp),
            _ => Err(Error::invalid_argument("invalid scope in `ttg.barrier`")),
        }
    }
}

mlir_op!(Barrier);
mlir_op_trait!(Barrier, ZeroOperands);
mlir_op_trait!(Barrier, ZeroRegions);
mlir_op_trait!(Barrier, ZeroSuccessors);

/// Constructs a detached [`BarrierOperation`] with the selected memory ordering and thread scope.
pub fn barrier<'c, 't: 'c, L: Location<'c, 't>>(
    address_spaces: BarrierAddressSpaces,
    scope: BarrierScope,
    location: L,
) -> Result<DetachedBarrierOperation<'c, 't>, Error> {
    if scope == BarrierScope::Warp && address_spaces != BarrierAddressSpaces::LOCAL {
        return Err(Error::invalid_argument("a warp-scoped `ttg.barrier` requires the local address space"));
    }
    let context = location.context();
    context.load_dialect(DialectHandle::triton_ttg()?)?;
    let (spaces, scope) = unsafe {
        let context_handle = context.handle.borrow_mut();
        (
            AttributeRef::from_c_api(
                mlirTritonTtgBarrierAddressSpacesAttrGet(*context_handle, address_spaces.bits()),
                context,
            )?,
            AttributeRef::from_c_api(mlirTritonTtgBarrierScopeAttrGet(*context_handle, scope as u32), context)?,
        )
    };
    OperationBuilder::new("ttg.barrier", location)
        .add_attribute(ADDRESS_SPACES_ATTRIBUTE, spaces)?
        .add_attribute(SCOPE_ATTRIBUTE, scope)?
        .build()
        .and_then(|operation| unsafe {
            operation.cast().ok_or_else(|| Error::invalid_argument("invalid arguments to `ttg.barrier`"))
        })
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle};

    use super::*;

    #[test]
    fn test_barrier() {
        let Ok(handle) = DialectHandle::triton_ttg() else { return };
        assert_eq!(handle.namespace().unwrap(), "ttg");
        let context = Context::new();
        let location = context.unknown_location();
        let spaces = BarrierAddressSpaces::GLOBAL_READ | BarrierAddressSpaces::GLOBAL_WRITE;
        let operation = barrier(spaces, BarrierScope::Cta, location).unwrap();
        assert_eq!(operation.address_spaces().unwrap(), spaces);
        assert_eq!(operation.scope().unwrap(), BarrierScope::Cta);
        assert_eq!(operation.to_string().trim(), "ttg.barrier global_read|global_write");
    }

    #[test]
    fn test_barrier_rejects_warp_scope_with_global_ordering() {
        let context = Context::new();
        assert!(matches!(
            barrier(BarrierAddressSpaces::GLOBAL_WRITE, BarrierScope::Warp, context.unknown_location()),
            Err(Error::InvalidArgument { message, .. })
                if message == "a warp-scoped `ttg.barrier` requires the local address space"
        ));
    }
}
