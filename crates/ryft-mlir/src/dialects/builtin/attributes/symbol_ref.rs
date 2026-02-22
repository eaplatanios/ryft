use ryft_xla_sys::bindings::{
    MlirAttribute, mlirFlatSymbolRefAttrGet, mlirFlatSymbolRefAttrGetValue, mlirSymbolRefAttrGet,
    mlirSymbolRefAttrGetLeafReference, mlirSymbolRefAttrGetNestedReference, mlirSymbolRefAttrGetNumNestedReferences,
    mlirSymbolRefAttrGetRootReference, mlirSymbolRefAttrGetTypeID,
};

use crate::{Attribute, AttributeRef, Context, FromWithContext, StringRef, TypeId, mlir_subtype_trait_impls};

/// Built-in MLIR [`Attribute`] that stores a symbolic reference to an operation. A symbol reference attribute is a
/// literal attribute that represents a named reference to an operation that is nested within an operation with the
/// MLIR `OpTrait::SymbolTable` trait. As such, this reference is given meaning by the nearest parent operation
/// containing the `OpTrait::SymbolTable` trait. It may optionally contain a set of nested references that further
/// resolve to a symbol nested within a different symbol table.
///
/// The rationale for this kind of attribute is that identifying accesses to global data is critical to enabling
/// efficient multithreaded compilation. Restricting global data access to occur through symbols and limiting the
/// places that can legally hold a symbol reference simplifies reasoning about such data accesses. Refer to the
/// [symbols & tables section of the official MLIR documentation](https://mlir.llvm.org/docs/SymbolsAndSymbolTables/)
/// for more information.
///
/// # Examples
///
/// The following are examples of [`SymbolRefAttributeRef`]s represented using their
/// [`Display`](std::fmt::Display) rendering:
///
/// ```text
/// @flat_reference
/// @parent_reference::@nested_reference
/// ```
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Builtin/#symbolrefattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct SymbolRefAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> SymbolRefAttributeRef<'c, 't> {
    /// Gets the [`TypeId`] that corresponds to [`SymbolRefAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirSymbolRefAttrGetTypeID()).unwrap() }
    }

    /// Returns a string reference to the root referenced symbol of this [`SymbolRefAttributeRef`].
    pub fn root_reference(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirSymbolRefAttrGetRootReference(self.handle)) }
    }

    /// Returns a string reference to the leaf referenced symbol of this [`SymbolRefAttributeRef`].
    pub fn leaf_reference(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirSymbolRefAttrGetLeafReference(self.handle)) }
    }

    /// Returns the number of references nested in this [`SymbolRefAttributeRef`].
    pub fn nested_reference_count(&self) -> usize {
        unsafe { mlirSymbolRefAttrGetNumNestedReferences(self.handle).cast_unsigned() }
    }

    /// Returns the references nested in this [`SymbolRefAttributeRef`].
    pub fn nested_references(&self) -> impl Iterator<Item = AttributeRef<'c, 't>> {
        (0..self.nested_reference_count()).map(|index| self.nested_reference(index).unwrap())
    }

    /// Returns the `index`-th reference nested in this [`SymbolRefAttributeRef`]
    /// and [`None`] if the provided index is out of bounds.
    pub fn nested_reference(&self, index: usize) -> Option<AttributeRef<'c, 't>> {
        unsafe {
            AttributeRef::from_c_api(
                mlirSymbolRefAttrGetNestedReference(self.handle, index.cast_signed()),
                self.context,
            )
        }
    }
}

mlir_subtype_trait_impls!(SymbolRefAttributeRef<'c, 't> as Attribute, mlir_type = Attribute, mlir_subtype = SymbolRef);

impl<'t> Context<'t> {
    /// Creates a new [`SymbolRefAttributeRef`] owned by this [`Context`], referencing the symbol identified
    /// by `symbol`, inside the provided list of references (which must not themselves be nested).
    pub fn symbol_ref_attribute<'c, 's, A: Attribute<'c, 't>>(
        &'c self,
        symbol: StringRef<'s>,
        references: &[A],
    ) -> SymbolRefAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            let references = references.iter().map(|reference| reference.to_c_api()).collect::<Vec<_>>();
            SymbolRefAttributeRef::from_c_api(
                mlirSymbolRefAttrGet(
                    *self.handle.borrow(),
                    symbol.to_c_api(),
                    references.len().cast_signed(),
                    references.as_ptr() as *const _,
                ),
                self,
            )
            .unwrap()
        }
    }
}

/// Built-in MLIR [`Attribute`] that stores a flat symbolic reference to an operation. The difference between this
/// attribute and [`SymbolRefAttributeRef`] is that this attribute does not contain any nested references.
#[derive(Copy, Clone)]
pub struct FlatSymbolRefAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> FlatSymbolRefAttributeRef<'c, 't> {
    /// Returns a string reference to the referenced symbol of this [`SymbolRefAttributeRef`].
    pub fn reference(&self) -> StringRef<'c> {
        unsafe { StringRef::from_c_api(mlirFlatSymbolRefAttrGetValue(self.handle)) }
    }
}

mlir_subtype_trait_impls!(
    FlatSymbolRefAttributeRef<'c, 't> as Attribute,
    mlir_type = Attribute,
    mlir_subtype = FlatSymbolRef,
);

impl<'c, 't, 's, S: Into<StringRef<'s>>> FromWithContext<'c, 't, S> for FlatSymbolRefAttributeRef<'c, 't> {
    fn from_with_context(value: S, context: &'c Context<'t>) -> Self {
        context.flat_symbol_ref_attribute(value)
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`FlatSymbolRefAttributeRef`] owned by this [`Context`], referencing the symbol identified
    /// by `symbol`.
    pub fn flat_symbol_ref_attribute<'c, 's, S: Into<StringRef<'s>>>(
        &'c self,
        symbol: S,
    ) -> FlatSymbolRefAttributeRef<'c, 't> {
        // While this operation can mutate the context (in that it might add an entry to its corresponding
        // uniquing table), we use an immutable borrow here as a mutable borrow would make using this
        // function quite inconvenient/annoying in practice. This should have no negative consequences in
        // terms of safety since MLIR contexts are not thread-safe and in a single-threaded context there
        // should be no possibility for this function to cause problems with an immutable borrow.
        unsafe {
            FlatSymbolRefAttributeRef::from_c_api(
                mlirFlatSymbolRefAttrGet(*self.handle.borrow(), symbol.into().to_c_api()),
                self,
            )
            .unwrap()
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_flat_symbol_ref_attribute() {
        let context = Context::new();

        // Test simple flat symbol reference.
        let attribute = context.flat_symbol_ref_attribute("foo");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.reference().as_str().unwrap(), "foo");

        // Test another flat symbol reference.
        let attribute = context.flat_symbol_ref_attribute("my_symbol");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.reference().as_str().unwrap(), "my_symbol");
    }

    #[test]
    fn test_flat_symbol_ref_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.flat_symbol_ref_attribute("foo");
        let attribute_2 = context.flat_symbol_ref_attribute("foo");
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.flat_symbol_ref_attribute("bar");
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.flat_symbol_ref_attribute("foo");
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_flat_symbol_ref_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.flat_symbol_ref_attribute("my_symbol");
        test_attribute_display_and_debug(attribute, "@my_symbol");
    }

    #[test]
    fn test_flat_symbol_ref_attribute_parsing() {
        let context = Context::new();
        let attribute = context.flat_symbol_ref_attribute("my_symbol");
        let parsed = context.parse_attribute("@my_symbol").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_flat_symbol_ref_attribute_casting() {
        let context = Context::new();
        let attribute = context.flat_symbol_ref_attribute("foo");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_symbol_ref_attribute_type_id() {
        let context = Context::new();
        let symbol_ref_attribute_id = SymbolRefAttributeRef::type_id();
        let symbol_ref_attribute_1 = context.symbol_ref_attribute::<AttributeRef>("test_symbol".into(), &[]);
        let symbol_ref_attribute_2 = context.symbol_ref_attribute::<AttributeRef>("test_symbol".into(), &[]);
        assert_eq!(symbol_ref_attribute_1.type_id(), symbol_ref_attribute_2.type_id());
        assert_eq!(symbol_ref_attribute_id, symbol_ref_attribute_1.type_id());
    }

    #[test]
    fn test_symbol_ref_attribute() {
        let context = Context::new();

        // Test flat symbol reference (no nested references).
        let attribute = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.root_reference().as_str().unwrap(), "foo");
        assert_eq!(attribute.leaf_reference().as_str().unwrap(), "foo");
        assert_eq!(attribute.nested_reference_count(), 0);

        // Test nested symbol reference.
        let nested_1 = context.flat_symbol_ref_attribute("nested1");
        let nested_2 = context.flat_symbol_ref_attribute("nested2");
        let attribute = context.symbol_ref_attribute("root".into(), &[nested_1, nested_2]);
        assert_eq!(attribute.root_reference().as_str().unwrap(), "root");
        assert_eq!(attribute.leaf_reference().as_str().unwrap(), "nested2");
        assert_eq!(attribute.nested_reference_count(), 2);
        assert_eq!(attribute.nested_references().collect::<Vec<_>>(), vec![nested_1, nested_2]);
    }

    #[test]
    fn test_symbol_ref_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        let attribute_2 = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.symbol_ref_attribute::<AttributeRef>("bar".into(), &[]);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_symbol_ref_attribute_display_and_debug() {
        let context = Context::new();

        // Flat reference.
        let attribute = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        test_attribute_display_and_debug(attribute, "@foo");

        // Nested references.
        let nested_1 = context.flat_symbol_ref_attribute("nested1");
        let nested_2 = context.flat_symbol_ref_attribute("nested2");
        let attribute = context.symbol_ref_attribute("root".into(), &[nested_1, nested_2]);
        test_attribute_display_and_debug(attribute, "@root::@nested1::@nested2");
    }

    #[test]
    fn test_symbol_ref_attribute_parsing() {
        let context = Context::new();

        // Test parsing flat reference.
        let attribute = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        let parsed = context.parse_attribute("@foo").unwrap();
        assert_eq!(parsed, attribute);

        // Test parsing nested references.
        let nested_1 = context.flat_symbol_ref_attribute("nested1");
        let nested_2 = context.flat_symbol_ref_attribute("nested2");
        let attribute = context.symbol_ref_attribute("root".into(), &[nested_1, nested_2]);
        let parsed = context.parse_attribute("@root::@nested1::@nested2").unwrap();
        assert_eq!(parsed, attribute);
    }

    #[test]
    fn test_symbol_ref_attribute_casting() {
        let context = Context::new();
        let attribute = context.symbol_ref_attribute::<AttributeRef>("foo".into(), &[]);
        test_attribute_casting(attribute);
    }
}
