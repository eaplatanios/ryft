//! The `tensor` dialect is intended to hold core tensor creation and manipulation [`Operation`](crate::Operation)s,
//! which are not strongly associated with any particular other dialect or domain abstraction. The aim for operations
//! in this dialect is that they make sense for any tensor element type. When this is not the case, the operation is
//! left to live in other dialects. Examples of element types that could be supported by the tensor dialect include:
//!
//!   - representing large, dense aggregations of primitive types, suitable for high-performance numerical computing,
//!   - representing shapes in the shape dialect, which consist of small 1D tensors of index data type,
//!   - representing aggregations of strings or "variant" types, and
//!   - representing large, sparse aggregations of primitive types, suitable for high-performance numerical computing.
//!
//! Because of this broad element type support and because of the existence of more dedicated dialects, such as the
//! [`sparse_tensor`](crate::dialects::sparse_tensor) and [`linalg`](crate::dialects::linalg) dialects, we prefer for
//! now to keep the tensor dialect as small as possible. The expectation is that at some point in the future, the
//! tensor dialect's scope may be broadened through a careful discussion of the tradeoffs.
//!
//! On the `tensor` type itself, note that it is actually a builtin type (i.e., it lives in the
//! [`builtin`](crate::dialects::builtin) dialect), and does not live in this dialect. Furthermore, a tensor is an
//! immutable object. For example, this means that a copy will always be made of the tensor object when it is passed
//! to the destination operand used by some operations in this dialect. As an optimization, an implementation can
//! eliminate these copies during lowering when they are redundant and perform in-place mutation. Refer to the
//! [Destination-Passing Style documentation](https://mlir.llvm.org/docs/Bufferization/#destination-passing-style)
//! for more information.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/TensorOps/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__tensor__;

use crate::DialectHandle;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `tensor` [`Dialect`](crate::Dialect).
    pub fn tensor() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__tensor__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_tensor_dialect() {
        let handle = DialectHandle::tensor();
        assert_eq!(handle.namespace().unwrap(), "tensor");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "tensor");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::tensor());
        assert_eq!(dialect_1, dialect_2);
    }
}
