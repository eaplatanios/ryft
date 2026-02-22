use std::fmt::{Debug, Display};
use std::marker::PhantomData;

use ryft_xla_sys::bindings::{
    MlirIdentifier, mlirIdentifierEqual, mlirIdentifierGet, mlirIdentifierGetContext, mlirIdentifierStr,
};

use crate::support::StringRef;
use crate::{Context, ContextRef};

/// MLIR identifier (used e.g., for the names of attributes, operations, etc.).
#[derive(Copy, Clone)]
pub struct Identifier<'c, 't> {
    /// Handle that represents this [`Identifier`] in the MLIR C API.
    handle: MlirIdentifier,

    /// [`PhantomData`] used to track the lifetime of the [`Context`] that owns this [`Identifier`].
    owner: PhantomData<&'c Context<'t>>,
}

impl<'c, 't> Identifier<'c, 't> {
    /// Constructs a new [`Identifier`] from the provided [`MlirIdentifier`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirIdentifier) -> Self {
        Self { handle, owner: PhantomData }
    }

    /// Returns the [`MlirIdentifier`] that corresponds to this [`Identifier`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirIdentifier {
        self.handle
    }

    /// Returns a [`ContextRef`] for the [`Context`] that owns this [`Identifier`].
    pub fn context(&self) -> ContextRef<'c, 't> {
        unsafe { ContextRef::from_c_api(mlirIdentifierGetContext(self.handle)) }
    }

    /// Returns a [`StringRef`] referencing the underlying string.
    pub fn as_ref(&self) -> StringRef<'c> {
        self.into()
    }

    /// Returns an [`str`] slice representation of the underlying string.
    pub fn as_str(&self) -> Result<&str, std::str::Utf8Error> {
        self.try_into()
    }
}

impl PartialEq for Identifier<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirIdentifierEqual(self.handle, other.handle) }
    }
}

impl Eq for Identifier<'_, '_> {}

impl Display for Identifier<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.as_str().unwrap_or("<failed to decode as a UTF-8 string>"))
    }
}

impl Debug for Identifier<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Identifier[{self}]")
    }
}

impl<'c> From<&Identifier<'c, '_>> for StringRef<'c> {
    fn from(value: &Identifier<'c, '_>) -> Self {
        unsafe { StringRef::from_c_api(mlirIdentifierStr(value.handle)) }
    }
}

impl<'c> TryFrom<&Identifier<'c, '_>> for &'c str {
    type Error = std::str::Utf8Error;

    fn try_from(value: &Identifier<'c, '_>) -> Result<Self, Self::Error> {
        value.as_ref().try_into()
    }
}

impl<'t> Context<'t> {
    /// Creates an [`Identifier`] with the provided name in this [`Context`].
    pub fn identifier<'c, 's, S: Into<StringRef<'s>>>(&'c self, name: S) -> Identifier<'c, 't> {
        unsafe {
            Identifier::from_c_api(mlirIdentifierGet(
                // While this operation can mutate the context (in that it might add an entry to its corresponding
                // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
                // function quite inconvenient/annoying in practice. This should have no negative consequences in
                // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
                // should be no possibility for this function to cause problems with an immutable borrow.
                *self.handle.borrow(),
                name.into().to_c_api(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identifier_display() {
        let context = Context::new();
        let foo = context.identifier("foo");
        let bar = context.identifier("bar");
        assert_eq!(format!("{foo}"), "foo".to_string());
        assert_eq!(format!("{bar}"), "bar".to_string());
    }

    #[test]
    fn test_identifier_debug() {
        let context = Context::new();
        let foo = context.identifier("foo");
        let bar = context.identifier("bar");
        assert_eq!(format!("{foo:?}"), "Identifier[foo]".to_string());
        assert_eq!(format!("{bar:?}"), "Identifier[bar]".to_string());
    }

    #[test]
    fn test_identifier_comparison() {
        let context_1 = Context::new();
        let context_2 = Context::new();
        let foo_1_1 = context_1.identifier("foo");
        let foo_1_2 = context_1.identifier("foo");
        let foo_2_1 = context_2.identifier("foo");
        let bar_1_1 = context_1.identifier("bar");
        let bar_1_2 = context_1.identifier("bar");
        let bar_2_1 = context_2.identifier("bar");
        assert_eq!(foo_1_1, foo_1_1);
        assert_eq!(foo_1_1, foo_1_2);
        assert_eq!(foo_1_2, foo_1_1);
        assert_eq!(foo_1_2, foo_1_2);
        assert_eq!(bar_1_1, bar_1_1);
        assert_eq!(bar_1_1, bar_1_2);
        assert_eq!(bar_1_2, bar_1_1);
        assert_eq!(bar_1_2, bar_1_2);
        assert_ne!(foo_1_1, bar_1_1);
        assert_ne!(foo_1_1, bar_1_2);
        assert_ne!(bar_1_1, foo_1_1);
        assert_ne!(bar_1_1, foo_1_2);
        assert_ne!(bar_1_2, foo_1_1);
        assert_ne!(bar_1_2, foo_1_2);
        assert_ne!(foo_1_1, foo_2_1);
        assert_ne!(bar_1_1, bar_2_1);
    }

    #[test]
    fn test_identifier_context() {
        let context_1 = Context::new();
        let context_2 = Context::new();
        let foo_1 = context_1.identifier("foo");
        let foo_2 = context_2.identifier("foo");
        let bar_1 = context_1.identifier("bar");
        let bar_2 = context_2.identifier("bar");
        assert_ne!(context_1, context_2);
        assert_eq!(foo_1.context(), context_1);
        assert_eq!(foo_2.context(), context_2);
        assert_eq!(bar_1.context(), context_1);
        assert_eq!(bar_2.context(), context_2);
        assert_eq!(context_1, foo_1.context());
        assert_eq!(context_2, foo_2.context());
        assert_eq!(context_1, bar_1.context());
        assert_eq!(context_2, bar_2.context());
        assert_eq!(foo_1.context(), bar_1.context());
        assert_eq!(foo_2.context(), bar_2.context());
        assert_eq!(bar_1.context(), foo_1.context());
        assert_eq!(bar_2.context(), foo_2.context());
        assert_ne!(foo_1.context(), foo_2.context());
        assert_ne!(foo_2.context(), foo_1.context());
        assert_ne!(bar_1.context(), bar_2.context());
        assert_ne!(bar_2.context(), bar_1.context());
    }
}
