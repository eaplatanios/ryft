use std::marker::PhantomData;

use ryft_xla_sys::bindings::{MlirDialect, mlirDialectEqual, mlirDialectGetContext, mlirDialectGetNamespace};

use crate::{Context, ContextRef, StringRef};

/// [`Dialect`]s are the mechanism by which the MLIR ecosystem achieves great extensibility. They enable the definition
/// of new/custom [`Operation`](crate::Operation)s, [`Attribute`](crate::Attribute)s, and [`Type`](crate::Type)s. Each
/// dialect is given a unique namespace that is prefixed to each of its custom [`Operation`](crate::Operation)s,
/// [`Attribute`](crate::Attribute)s, and [`Type`](crate::Type)s. For example, the `affine` dialect defines the
/// namespace `"affine"`. MLIR allows for multiple dialects, even those outside of the main MLIR tree,
/// to co-exist together within one [`Module`](crate::Module). Dialects are produced and consumed by certain
/// [`Pass`](crate::Pass)es. MLIR provides a framework to convert between, and within, different dialects.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#dialects) for more information.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Dialect<'c, 't> {
    /// Handle that represents this [`Dialect`] in the MLIR C API.
    handle: MlirDialect,

    /// [`PhantomData`] used to track the lifetime of the [`Context`] that owns this [`Dialect`].
    owner: PhantomData<&'c Context<'t>>,
}

impl<'c, 't> Dialect<'c, 't> {
    /// Constructs a new [`Dialect`] from the provided [`MlirDialect`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirDialect) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, owner: PhantomData }) }
    }

    /// Returns the [`MlirDialect`] that corresponds to this [`Dialect`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirDialect {
        self.handle
    }

    /// Returns the [`Context`] that owns this [`Dialect`].
    pub fn context(&self) -> ContextRef<'c, 't> {
        unsafe { ContextRef::from_c_api(mlirDialectGetContext(self.handle)) }
    }

    /// Returns the namespace of this [`Dialect`]. The returned string is owned by the underlying [`Context`]
    /// which also owns this [`Dialect`].
    pub fn namespace(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirDialectGetNamespace(self.handle)).as_str() }
    }
}

impl PartialEq for Dialect<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlirDialectEqual(self.handle, other.handle) }
    }
}

impl Eq for Dialect<'_, '_> {}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::DialectHandle;

    use super::*;

    #[test]
    fn test_dialect_context() {
        let context = Context::new();

        // Load the `gpu` dialect and verify we can get its context.
        context.register_dialect(DialectHandle::gpu());
        let dialect = context.load_dialect(DialectHandle::gpu()).unwrap();
        let dialect_context = dialect.context();
        assert_eq!(context, dialect_context);

        // Verify that the context reference works by checking dialect counts.
        assert!(dialect_context.registered_dialect_count() > 0);
        assert!(dialect_context.loaded_dialect_count() > 0);
    }

    #[test]
    fn test_dialect_namespace() {
        let context = Context::new();

        // Load the `gpu` dialect and check its namespace.
        context.register_dialect(DialectHandle::gpu());
        let dialect = context.load_dialect(DialectHandle::gpu());
        assert!(dialect.is_some());
        assert_eq!(dialect.unwrap().namespace().unwrap(), "gpu");

        // Load the `linalg` dialect and check its namespace.
        context.register_dialect(DialectHandle::linalg());
        let dialect = context.load_dialect(DialectHandle::linalg());
        assert!(dialect.is_some());
        assert_eq!(dialect.unwrap().namespace().unwrap(), "linalg");

        // Load the `sparse_tensor` dialect and check its namespace.
        context.register_dialect(DialectHandle::sparse_tensor());
        let dialect = context.load_dialect(DialectHandle::sparse_tensor());
        assert!(dialect.is_some());
        assert_eq!(dialect.unwrap().namespace().unwrap(), "sparse_tensor");
    }

    #[test]
    fn test_dialect_equality() {
        let context = Context::new();

        // Load the `gpu` dialect twice and verify that the two loaded dialects are equal.
        context.register_dialect(DialectHandle::gpu());
        let dialect_0 = context.load_dialect(DialectHandle::gpu()).unwrap();
        let dialect_1 = context.load_dialect(DialectHandle::gpu()).unwrap();
        assert_eq!(dialect_0, dialect_1);

        // Load the `linalg` dialect and make sure that it is not equal to the loaded `gpu` dialect.
        context.register_dialect(DialectHandle::linalg());
        let dialect_2 = context.load_dialect(DialectHandle::linalg()).unwrap();
        assert_ne!(dialect_0, dialect_2);
    }

    #[test]
    fn test_dialect_c_api() {
        let context = Context::new();
        context.register_dialect(DialectHandle::gpu());
        let dialect = context.load_dialect(DialectHandle::gpu());
        assert!(dialect.is_some());
        assert_eq!(unsafe { Dialect::from_c_api(dialect.unwrap().to_c_api()) }, dialect);
    }
}
