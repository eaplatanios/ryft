use ryft_xla_sys::bindings::{
    MlirLocation, mlirLocationNameGet, mlirLocationNameGetChildLoc, mlirLocationNameGetName, mlirLocationNameGetTypeID,
};

use crate::{Context, Identifier, Location, LocationRef, StringRef, TypeId, mlir_subtype_trait_impls};

/// [`Location`] that consists of a name and an optional associated child [`Location`]. A [`NamedLocationRef`] is a kind
/// of location that attaches a symbolic name string (plus optionally another underlying location). It is useful when
/// you want to associate an operation with a semantic name rather than (or in addition to) just a file, line, column,
/// etc. Refer to the [MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#nameloc) for more information.
#[derive(Copy, Clone)]
pub struct NamedLocationRef<'c, 't> {
    /// Handle that represents this [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns this [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> NamedLocationRef<'c, 't> {
    /// Returns the [`TypeId`] that corresponds to [`NamedLocationRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirLocationNameGetTypeID()).unwrap() }
    }

    /// Returns the name of this [`NamedLocationRef`].
    pub fn name(&self) -> Identifier<'c, 't> {
        unsafe { Identifier::from_c_api(mlirLocationNameGetName(self.handle)) }
    }

    /// Returns the child [`Location`] of this [`NamedLocationRef`]. If this location was created with no children,
    /// then this function will return an [`UnknownLocationRef`](crate::UnknownLocationRef).
    pub fn child(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(mlirLocationNameGetChildLoc(self.handle), self.context).unwrap() }
    }
}

mlir_subtype_trait_impls!(NamedLocationRef<'c, 't> as Location, mlir_type = Location, mlir_subtype = Name);

impl<'t> Context<'t> {
    /// Creates a new [`NamedLocationRef`] with the specified name and optional child `[Location`].
    pub fn named_location<'c, S: AsRef<str>, L: Location<'c, 't>>(
        &'c self,
        name: S,
        child: Option<L>,
    ) -> NamedLocationRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let child = child.map(|location| location.to_c_api()).unwrap_or(MlirLocation { ptr: std::ptr::null_mut() });
            NamedLocationRef::from_c_api(
                mlirLocationNameGet(*self.handle.borrow(), StringRef::from(name.as_ref()).to_c_api(), child),
                self,
            )
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
    fn test_named_location_type_id() {
        let named_location_type_id = NamedLocationRef::type_id();
        assert_eq!(NamedLocationRef::type_id(), NamedLocationRef::type_id());
        assert_eq!(named_location_type_id, NamedLocationRef::type_id());
    }

    #[test]
    fn test_named_location() {
        let context = Context::new();

        // Test named location without child (i.e., the child defaults to unknown location).
        let location = context.named_location::<_, LocationRef>("test_name", None);
        assert_eq!(&context, location.context());
        assert_eq!(location.name().as_str().unwrap(), "test_name");
        assert_eq!(location.child(), context.unknown_location());

        // Test named location with a child.
        let child = context.file_location("test.rs", 10, 5);
        let location = context.named_location("parent_name", Some(child));
        assert_eq!(&context, location.context());
        assert_eq!(location.name().as_str().unwrap(), "parent_name");
        assert_eq!(location.child(), child);
    }

    #[test]
    fn test_named_location_equality() {
        let context = Context::new();

        // Same locations from the same context must be equal because they are "uniqued".
        let location_1 = context.named_location::<_, LocationRef>("test", None);
        let location_2 = context.named_location::<_, LocationRef>("test", None);
        assert_eq!(location_1, location_2);

        // Different locations from the same context must not be equal.
        let location_2 = context.named_location::<_, LocationRef>("other", None);
        assert_ne!(location_1, location_2);

        // Same locations from different contexts must not be equal.
        let context = Context::new();
        let location_2 = context.named_location::<_, LocationRef>("test", None);
        assert_ne!(location_1, location_2);
    }

    #[test]
    fn test_named_location_display_and_debug() {
        let context = Context::new();
        let location = context.named_location::<_, LocationRef>("my_name", None);
        test_location_display_and_debug(location, "loc(\"my_name\")");
    }

    #[test]
    fn test_named_location_with_child_display_and_debug() {
        let context = Context::new();
        let child = context.file_location("test.rs", 42, 10);
        let location = context.named_location("my_name", Some(child));
        test_location_display_and_debug(location, "loc(\"my_name\"(\"test.rs\":42:10))");
    }

    #[test]
    fn test_named_location_casting() {
        let context = Context::new();
        let location = context.named_location::<_, LocationRef>("test", None);
        test_location_casting(location);
    }
}
