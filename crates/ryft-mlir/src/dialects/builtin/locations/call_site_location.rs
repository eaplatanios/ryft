use ryft_xla_sys::bindings::{
    MlirLocation, mlirLocationCallSiteGet, mlirLocationCallSiteGetCallee, mlirLocationCallSiteGetCaller,
    mlirLocationCallSiteGetTypeID,
};

use crate::{Context, Location, LocationRef, TypeId, mlir_subtype_trait_impls};

/// [`Location`] that represents a specific call chain. For example, think about inlining. In this case, an operation
/// originates inside a callee, but also at a specific call site in the caller. A [`CallSiteLocationRef`] contains both
/// the [`CallSiteLocationRef::callee`] location and the [`CallSiteLocationRef::caller`] location.
/// Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#callsiteloc)
/// for more information.
#[derive(Copy, Clone)]
pub struct CallSiteLocationRef<'c, 't> {
    /// Handle that represents this [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns this [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> CallSiteLocationRef<'c, 't> {
    /// Returns the [`TypeId`] that corresponds to [`CallSiteLocationRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLocationCallSiteGetTypeID()).unwrap() }
    }

    /// Returns the callee [`Location`] of this [`CallSiteLocationRef`] (i.e., the location where the underlying thing was
    /// originally defined; e.g., in a function body).
    pub fn callee(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirLocationCallSiteGetCallee(self.handle), self.context()).unwrap() }
    }

    /// Returns the caller [`Location`] of this [`CallSiteLocationRef`] (i.e., the location where the call happened,
    /// which could itself be another call site location, chaining up the stack).
    pub fn caller(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirLocationCallSiteGetCaller(self.handle), self.context()).unwrap() }
    }
}

mlir_subtype_trait_impls!(CallSiteLocationRef<'c, 't> as Location, mlir_type = Location, mlir_subtype = CallSite);

impl<'t> Context<'t> {
    /// Creates a new [`CallSiteLocationRef`] with the specified callee and caller [`Location`]s,
    /// owned by this [`Context`].
    pub fn call_site_location<'c, CalleeLocation: Location<'c, 't>, CallerLocation: Location<'c, 't>>(
        &'c self,
        callee: CalleeLocation,
        caller: CallerLocation,
    ) -> CallSiteLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        let _guard = self.borrow();
        unsafe {
            CallSiteLocationRef::from_c_api(mlirLocationCallSiteGet(callee.to_c_api(), caller.to_c_api()), &self)
                .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::locations::tests::{test_location_casting, test_location_display_and_debug};

    use super::*;

    #[test]
    fn test_call_site_location_type_id() {
        let call_site_location_type_id = CallSiteLocationRef::type_id();
        assert_eq!(CallSiteLocationRef::type_id(), CallSiteLocationRef::type_id());
        assert_eq!(call_site_location_type_id, CallSiteLocationRef::type_id());
    }

    #[test]
    fn test_call_site_location() {
        let context = Context::new();

        // Create a simple call site location.
        let callee = context.file_location("callee.rs", 10, 5);
        let caller = context.file_location("caller.rs", 42, 10);
        let location = context.call_site_location(callee, caller);
        assert_eq!(&context, location.context());
        assert_eq!(location.callee(), callee);
        assert_eq!(location.caller(), caller);

        // Create a call site location with nested calls.
        let inner_callee = context.file_location("inner.rs", 5, 1);
        let middle = context.file_location("middle.rs", 20, 3);
        let outer_caller = context.file_location("outer.rs", 100, 50);
        let inner_call_site = context.call_site_location(inner_callee, middle);
        let outer_call_site = context.call_site_location(inner_call_site, outer_caller);
        assert_eq!(&context, outer_call_site.context());
        assert_eq!(outer_call_site.callee(), inner_call_site);
        assert_eq!(outer_call_site.caller(), outer_caller);
    }

    #[test]
    fn test_call_site_location_equality() {
        let context = Context::new();
        let callee = context.file_location("callee.rs", 10, 5);
        let caller = context.file_location("caller.rs", 42, 10);

        // Same locations from the same context must be equal because they are "uniqued".
        let location_1 = context.call_site_location(callee, caller);
        let location_2 = context.call_site_location(callee, caller);
        assert_eq!(location_1, location_2);

        // Different locations from the same context must not be equal.
        let other_caller = context.file_location("other.rs", 1, 1);
        let location_2 = context.call_site_location(callee, other_caller);
        assert_ne!(location_1, location_2);

        // Same locations from different contexts must not be equal.
        let context = Context::new();
        let other_callee = context.file_location("callee.rs", 10, 5);
        let other_caller = context.file_location("caller.rs", 42, 10);
        let location_2 = context.call_site_location(other_callee, other_caller);
        assert_ne!(location_1, location_2);
    }

    #[test]
    fn test_call_site_location_display_and_debug() {
        let context = Context::new();
        let callee = context.file_location("callee.rs", 10, 5);
        let caller = context.file_location("caller.rs", 42, 10);
        let location = context.call_site_location(callee, caller);
        test_location_display_and_debug(location, "loc(callsite(\"callee.rs\":10:5 at \"caller.rs\":42:10))");
    }

    #[test]
    fn test_call_site_location_casting() {
        let context = Context::new();
        let callee = context.file_location("callee.rs", 10, 5);
        let caller = context.file_location("caller.rs", 42, 10);
        let location = context.call_site_location(callee, caller);
        test_location_casting(location);
    }
}
