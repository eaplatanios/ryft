//! The `index` dialect contains [`Operation`](crate::Operation)s for manipulating values of the builtin index type.
//! The index type models target-specific values of pointer width, like `intptr_t`. Index values are typically used as
//! loop bounds, array subscripts, tensor dimensions, etc.
//!
//! The operations in this dialect operate exclusively on scalar index types. The dialect and its operations treat the
//! index type as signless and contains signed and unsigned versions of certain operations where the distinction is
//! meaningful. In particular, the operations and transformations are careful to be aware of the target-independent-ness
//! of the index type, such as when folding.
//!
//! The folding semantics of the `index` dialect operations ensure that folding produces the same results irrespective
//! of the eventual target pointer width. All index constants are stored in `APInt`s of maximum index bitwidth (`64`).
//! Operations are folded using 64-bit integer arithmetic.
//!
//! For operations where the values of the upper 32 bits do not impact the values of the lower 32 bits, no additional
//! handling is required because if the target is 32-bit, the truncated folded result will be the same as if the
//! operation were computed with 32-bit arithmetic, and if the target is 64-bit, the fold result is valid by default.
//!
//! Consider addition: an overflow in 32-bit is the same as truncating the result computed in 64-bit. For example,
//! `add(0x800000008, 0x800000008)` is `0x1000000010` in 64-bit, which truncates to `0x10`, the same result as
//! truncating the operands first: `add(0x08, 0x08)`. Specifically, an operation `f` can always be folded if it
//! satisfies the following for all 64-bit values of `a` and `b`: `trunc(f(a, b)) = f(trunc(a), trunc(b))`. When
//! materializing target-specific code, constants just need to be truncated as appropriate.
//!
//! Operations where the values of the upper 32 bits do impact the values of the lower 32 bits are not folded if the
//! results would be different in 32-bit. These are operations that right shift – division, remainder, etc. These
//! operations are only folded for subsets of `a` and `b` for which the above property is satisfied. This is checked
//! per fold attempt.
//!
//! Consider division: the 32-bit computation will differ from 64-bit if the latter results in a high bit shifted into
//! the lower 32 bits. For example, `div(0x100000002, 2)` is `0x80000001` in 64-bit but `0x01` in 32-bit; it cannot be
//! folded. However, `div(0x200000002, 2)` can be folded. The 64-bit result is `0x100000001`, which truncated to 32 bits
//! is `0x01`. The 32-bit result of the operation with truncated operands `div(0x02, 2)` which is `0x01`, the same as
//! truncating the 64-bit result.
//!
//! Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/IndexOps/) for more information.

use ryft_xla_sys::bindings::mlirGetDialectHandle__index__;

use crate::DialectHandle;

mod operations;

pub use operations::*;

impl DialectHandle<'_, '_> {
    /// Returns a [`DialectHandle`] for the `index` [`Dialect`](crate::Dialect).
    pub fn index() -> Self {
        unsafe { Self::from_c_api(mlirGetDialectHandle__index__()).unwrap() }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::{Context, DialectHandle, DialectRegistry};

    #[test]
    fn test_index_dialect() {
        let handle = DialectHandle::index();
        assert_eq!(handle.namespace().unwrap(), "index");

        // Check that registration works (both in the context and in a registry).
        let context = Context::new();
        let registry = DialectRegistry::new();
        registry.insert(handle);
        context.register_dialect(handle);

        // Check that loading works.
        let context = Context::new();
        let dialect_1 = context.load_dialect(handle);
        assert!(dialect_1.is_some());
        assert_eq!(dialect_1.unwrap().namespace().unwrap(), "index");

        // Check that comparison works.
        let dialect_2 = context.load_dialect(DialectHandle::index());
        assert_eq!(dialect_1, dialect_2);
    }
}
