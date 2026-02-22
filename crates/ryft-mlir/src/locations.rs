use std::fmt::{Debug, Display};

use ryft_xla_sys::bindings::{MlirLocation, mlirEmitError};

use crate::{Context, mlir_subtype_trait_impls};

/// MLIR location that is used for describing where some operation, identifier, etc., is defined in the source code.
/// In MLIR, locations are effectively metadata that indicate where an operation or value originated from (e.g., a
/// file, line, column, etc. in the source program). They are mainly used for debugging, error reporting, and mapping
/// back to the original source code.
///
/// Note that there are multiple types of locations in MLIR and in order to maintain safety in our Rust bindings, we
/// represent them using separate types that share some common functionality via the [`Location`] trait, and
/// with the ability to cast [`Location`]s to more specific location types to access their unique properties using
/// the [`Location::cast`] method.
///
/// This `struct` acts effectively as the super-type of all MLIR [`Location`]s and can be checked and specialized using
/// the [`Location::is`] and [`Location::cast`] functions.
///
/// Note that MLIR locations are technically defined as part of the built-in MLIR dialect, but they are used throughout
/// the system to describe the origin of operations and values and that is why they are not under the
/// [`dialects`](crate::dialects) module.
///
/// For more information on MLIR locations refer to the
/// [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#location-attributes).
pub trait Location<'c, 't: 'c>: Sized + Copy + Clone + PartialEq + Eq + Display + Debug {
    /// Constructs a new [`Location`] of this type from the provided handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirLocation, context: &'c Context<'t>) -> Option<Self>;

    /// Returns the [`MlirLocation`] that corresponds to this [`Location`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirLocation;

    /// Returns a reference to the [`Context`] that owns this [`Location`].
    fn context(&self) -> &'c Context<'t>;

    /// Returns `true` if this [`Location`] is an instance of `L`.
    fn is<L: Location<'c, 't>>(&self) -> bool {
        Self::cast::<L>(&self).is_some()
    }

    /// Tries to cast this [`Location`] to an instance of `L` (e.g., an instance of
    /// [`FileLocationRef`](crate::FileLocationRef)). If this is not an instance of the specified location type,
    /// this function will return [`None`].
    fn cast<L: Location<'c, 't>>(&self) -> Option<L> {
        unsafe { L::from_c_api(self.to_c_api(), self.context()) }
    }

    /// Up-casts this [`Location`] to an instance of [`LocationRef`].
    fn as_location_ref(&self) -> LocationRef<'c, 't> {
        unsafe { LocationRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Emits an error message at this [`Location`] through the diagnostics engine.
    /// This function is mainly used for testing purposes.
    fn emit_error<Message: AsRef<str>>(&self, message: Message) {
        unsafe {
            let message = std::ffi::CString::new(message.as_ref()).unwrap();
            mlirEmitError(self.to_c_api(), message.as_c_str().as_ptr());
        }
    }
}

/// Reference to an MLIR [`Location`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct LocationRef<'c, 't> {
    /// Handle that represents the underlying [`Location`] in the MLIR C API.
    handle: MlirLocation,

    /// [`Context`] that owns the underlying [`Location`].
    context: &'c Context<'t>,
}

impl<'c, 't> Location<'c, 't> for LocationRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirLocation, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    unsafe fn to_c_api(&self) -> MlirLocation {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(LocationRef<'c, 't> as Location, mlir_type = Location);

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Helper for testing [`Location`] [`Display`] and [`Debug`] implementations.
    pub(crate) fn test_location_display_and_debug<'c, 't: 'c, L: Location<'c, 't>>(
        location: L,
        expected: &'static str,
    ) {
        assert_eq!(format!("{}", location), expected);

        // Extract the type name for `L` to check the [`Debug`] implementation.
        let type_name = std::any::type_name::<L>().rsplit("::").next().unwrap_or("").split("<").next().unwrap_or("");
        assert_eq!(format!("{:?}", location), format!("{type_name}[{expected}]"));
    }

    /// Helper for testing [`Location`] casting.
    pub(crate) fn test_location_casting<'c, 't: 'c, L: Location<'c, 't>>(location: L) {
        let rendered_location = location.to_string();

        // Test upcasting.
        let location = location.as_location_ref();
        assert!(location.is::<L>());
        assert_eq!(location.to_string(), rendered_location);

        // Test downcasting.
        let location = location.cast::<L>().unwrap();
        assert!(location.is::<L>());
        assert_eq!(location.to_string(), rendered_location);

        // Invalid cast from specific location.
        let location = location.context().unknown_location();
        assert!(!location.is::<L>());
        assert_eq!(location.cast::<L>(), None);

        // Invalid cast from a generic location reference.
        let location = location.as_location_ref();
        assert!(!location.is::<L>());
        assert_eq!(location.cast::<L>(), None);
    }

    #[test]
    fn test_location() {
        let context = Context::new();
        let location_0 = context.unknown_location();
        let location_1 = context.unknown_location();
        let location_2 = context.file_location("foo", 1, 1);
        assert_eq!(location_0.context(), &context);
        assert_eq!(location_0, location_1);
        assert_ne!(location_0.as_location_ref(), location_2.as_location_ref());
        assert_ne!(location_1.as_location_ref(), location_2);
        assert_eq!(format!("{location_0}").as_str(), "loc(unknown)");
        assert_eq!(format!("{location_0:?}").as_str(), "UnknownLocationRef[loc(unknown)]");
        location_0.emit_error("dummy error");
    }
}
