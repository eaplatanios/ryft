use ryft_xla_sys::bindings::{
    MlirLocation, mlirLocationIsACallSite, mlirLocationIsAFileLineColRange, mlirLocationIsAFused, mlirLocationIsAName,
    mlirLocationUnknownGet,
};

use crate::{Context, Location, mlir_subtype_trait_impls};

/// [`UnknownLocationRef`]s represent unknown [`Location`]s (either because they were specified as such or because we
/// were unable to parse them from the MLIR native library). They exist because locations are an important concept in
/// MLIR must always be provided for all operations in an MLIR program, and so we need a way to specify truly unknown
/// locations. Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#unknownloc)
/// for more information.
#[derive(Copy, Clone)]
pub struct UnknownLocationRef<'c, 't> {
    /// Handle that represents this [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns this [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> Location<'c, 't> for UnknownLocationRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirLocation, context: &'c Context<'t>) -> Option<Self> {
        // Unfortunately, there is no `mlirLocationIsAUnknown` or `mlirLocationUnknownGetTypeID` function in the MLIR
        // C API and so we just check that this handle does not correspond to any of the other known location types.
        if !handle.ptr.is_null()
            && unsafe {
                !mlirLocationIsACallSite(handle)
                    && !mlirLocationIsAFileLineColRange(handle)
                    && !mlirLocationIsAFused(handle)
                    && !mlirLocationIsAName(handle)
            }
        {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirLocation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(UnknownLocationRef<'c, 't> as Location, mlir_type = Location);

impl<'t> Context<'t> {
    /// Creates a new [`UnknownLocationRef`] owned by this [`Context`].
    pub fn unknown_location<'c>(&'c self) -> UnknownLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe { UnknownLocationRef::from_c_api(mlirLocationUnknownGet(*self.handle.borrow()), self).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::locations::tests::test_location_display_and_debug;

    use super::*;

    #[test]
    fn test_unknown_location() {
        let context = Context::new();
        let location = context.unknown_location();
        assert_eq!(&context, location.context());
    }

    #[test]
    fn test_unknown_location_equality() {
        let context = Context::new();

        // Same locations from the same context must be equal because they are "uniqued".
        let location_1 = context.unknown_location();
        let location_2 = context.unknown_location();
        assert_eq!(location_1, location_2);

        // Same locations from different contexts must not be equal.
        let context = Context::new();
        let location_2 = context.unknown_location();
        assert_ne!(location_1, location_2);
    }

    #[test]
    fn test_unknown_location_display_and_debug() {
        let context = Context::new();
        let location = context.unknown_location();
        test_location_display_and_debug(location, "loc(unknown)");
    }

    #[test]
    fn test_unknown_location_casting() {
        let context = Context::new();
        let location = context.unknown_location();
        let rendered_location = location.to_string();

        // Test upcasting.
        let location = location.as_ref();
        assert!(location.is::<UnknownLocationRef>());
        assert_eq!(location.to_string(), rendered_location);

        // Test downcasting.
        let location = location.cast::<UnknownLocationRef>().unwrap();
        assert!(location.is::<UnknownLocationRef>());
        assert_eq!(location.to_string(), rendered_location);

        // Invalid cast from specific location.
        let location = context.file_location("test.rs", 1, 1);
        assert!(!location.is::<UnknownLocationRef>());
        assert_eq!(location.cast::<UnknownLocationRef>(), None);

        // Invalid cast from a generic location reference.
        let location = location.as_ref();
        assert!(!location.is::<UnknownLocationRef>());
        assert_eq!(location.cast::<UnknownLocationRef>(), None);
    }
}
