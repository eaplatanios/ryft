use std::marker::PhantomData;

use ryft_xla_sys::bindings::{MlirDialectHandle, mlirDialectHandleGetNamespace};

use crate::{Context, StringRef};

/// [`DialectHandle`]s are the means by which dialects can be referred to and registered in MLIR.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct DialectHandle<'c, 't> {
    /// Handle that represents this [`DialectHandle`] in the MLIR C API.
    handle: MlirDialectHandle,

    /// [`PhantomData`] used to track the lifetime of the [`Context`] that owns this [`DialectHandle`].
    owner: PhantomData<&'c Context<'t>>,
}

impl<'c, 't> DialectHandle<'c, 't> {
    /// Constructs a new [`DialectHandle`] from the provided [`MlirDialectHandle`] handle
    /// that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirDialectHandle) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, owner: PhantomData }) }
    }

    /// Returns the [`MlirDialectHandle`] that corresponds to this [`DialectHandle`]
    /// and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirDialectHandle {
        self.handle
    }

    /// Returns the namespace associated with this [`DialectHandle`].
    /// The returned string is owned by the underlying [`Context`] which also owns this [`DialectHandle`].
    pub fn namespace(&'c self) -> Result<&'c str, std::str::Utf8Error> {
        unsafe { StringRef::from_c_api(mlirDialectHandleGetNamespace(self.handle)).as_str() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;

    #[test]
    fn test_dialect_handle() {
        assert_eq!(DialectHandle::gpu().namespace().unwrap(), "gpu");
        assert_eq!(DialectHandle::linalg().namespace().unwrap(), "linalg");
        assert_eq!(DialectHandle::sparse_tensor().namespace().unwrap(), "sparse_tensor");
        assert_eq!(DialectHandle::func().namespace().unwrap(), "func");
        assert_eq!(DialectHandle::r#async().namespace().unwrap(), "async");

        // Check that we can construct multiple handles for the same dialect.
        let handle_1 = DialectHandle::gpu();
        let handle_2 = DialectHandle::gpu();
        assert_eq!(handle_1.namespace().unwrap(), handle_2.namespace().unwrap());

        // Check that dialect handles integrate properly with the MLIR C API.
        let handle = unsafe { DialectHandle::from_c_api(DialectHandle::gpu().to_c_api()) };
        assert!(handle.is_some());
        assert_eq!(handle.unwrap().namespace().unwrap(), "gpu");
    }
}
