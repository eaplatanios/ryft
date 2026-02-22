use std::fmt::{Debug, Display};

use ryft_xla_sys::bindings::{
    MlirAttribute, MlirNamedAttribute, mlirAttributeDump, mlirAttributeGetDialect, mlirAttributeGetNull,
    mlirAttributeGetType, mlirAttributeGetTypeID, mlirAttributeParseGet, mlirNamedAttributeGet,
};

use crate::{Context, Dialect, Identifier, StringRef, Type, TypeId, TypeRef, mlir_subtype_trait_impls};

/// MLIR attributes are the mechanism for specifying constant data on operations in places where a variable is never
/// allowed (e.g., the comparison predicate of a `arith.cmpi` operation). Each operation has an attribute dictionary,
/// which associates a set of attribute names to attribute values. MLIR's built-in dialect provides a rich set of
/// built-in attributes out of the box (e.g., arrays, dictionaries, strings, etc.). Additionally, dialects can define
/// their own dialect-specific attributes.
///
/// For dialects which have not adopted properties yet, the top-level attribute dictionary attached to an operation has
/// special semantics. The attribute entries are considered to be of two different kinds based on whether their
/// dictionary key has a dialect prefix:
///
///   - _inherent attributes_ are inherent to the definition of an operation's semantics. The operation itself is
///     expected to verify the consistency of these attributes. An example is the predicate attribute of the
///     `arith.cmpi` operation. These attributes must have names that do not start with a dialect prefix.
///   - _discardable attributes_ have semantics defined externally to the operation itself, but must be compatible with
///     the operation's semantics. These attributes must have names that start with a dialect prefix. The dialect
///     indicated by the dialect prefix is expected to verify these attributes. An example is the
///     `gpu.container_module` attribute.
///
/// Note that attribute values are allowed to themselves be dictionary attributes, but only the top-level dictionary
/// attribute attached to the operation is subject to the classification above. When properties are adopted, only
/// discardable attributes are stored in the top-level dictionary, while inherent attributes are stored in the
/// properties storage.
///
/// This `struct` acts effectively as the super-type of all MLIR [`Attribute`]s and can be checked and specialized
/// using the [`Attribute::is`] and [`Attribute::cast`] functions.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#attributes) for more information.
pub trait Attribute<'c, 't: 'c>: Sized + Copy + Clone + PartialEq + Eq + Display + Debug {
    /// Constructs a new attribute of this type from the provided handle that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self>;

    /// Returns the [`MlirAttribute`] that corresponds to this attribute and which can be passed to functions
    /// in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirAttribute;

    /// Returns a reference to the [`Context`] that owns this attribute.
    fn context(&self) -> &'c Context<'t>;

    /// Returns `true` if this attribute is an instance of `A`.
    fn is<A: Attribute<'c, 't>>(&self) -> bool {
        Self::cast::<A>(&self).is_some()
    }

    /// Tries to cast this attribute to an instance of `A` (e.g., an instance of
    /// [`TypeAttributeRef`](crate::TypeAttributeRef)). If this is not an instance of the specified type,
    /// this function will return [`None`].
    fn cast<A: Attribute<'c, 't>>(&self) -> Option<A> {
        unsafe { A::from_c_api(self.to_c_api(), self.context()) }
    }

    /// Up-casts this attribute to an instance of [`Attribute`].
    fn as_ref(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Gets the [`TypeId`] of this attribute. Note that this function may return the same [`TypeId`] for different
    /// instances of the same attribute with potentially different properties and nested attributes. That is because
    /// a [`TypeId`] is a unique identifier of the corresponding MLIR C++ type for [`Attribute`] and not for a specific
    /// instance of [`Attribute`].
    fn type_id(&self) -> TypeId<'c> {
        unsafe { TypeId::from_c_api(mlirAttributeGetTypeID(self.to_c_api())).unwrap() }
    }

    /// Returns the [`Type`] of this attribute.
    fn r#type(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(mlirAttributeGetType(self.to_c_api()), self.context()).unwrap() }
    }

    /// Returns the [`Dialect`] that this attribute belongs to.
    fn dialect(&self) -> Dialect<'c, 't> {
        unsafe { Dialect::from_c_api(mlirAttributeGetDialect(self.to_c_api())).unwrap() }
    }

    /// Dumps this attribute to the standard error stream.
    fn dump(&self) {
        unsafe { mlirAttributeDump(self.to_c_api()) }
    }
}

/// Reference to an MLIR [`Attribute`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct AttributeRef<'c, 't> {
    /// Handle that represents the underlying [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns the underlying [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> Attribute<'c, 't> for AttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(AttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Returns a null (i.e., empty) [`Attribute`].
    pub fn null_attribute<'c>(&'c self) -> AttributeRef<'c, 't> {
        AttributeRef { handle: unsafe { mlirAttributeGetNull() }, context: &self }
    }

    /// Parses an [`Attribute`] from the provided string representation. Returns [`None`] if MLIR fails to parse
    /// the provided string into an [`Attribute`]. The resulting [`Attribute`] is owned by this [`Context`].
    pub fn parse_attribute<'c>(&'c self, source: &str) -> Option<AttributeRef<'c, 't>> {
        unsafe {
            let handle = mlirAttributeParseGet(*self.handle.borrow_mut(), StringRef::from(source).to_c_api());
            if handle.ptr.is_null() { None } else { AttributeRef::from_c_api(handle, &self) }
        }
    }
}

/// [`Attribute`] paired with a name that acts as an attribute value alias. MLIR supports defining named aliases for
/// attribute values. An attribute alias is an [`Identifier`] that can be used in the place of the attribute that it
/// defines. These aliases must be defined before their uses. Alias names may not contain `.` characters, since such
/// names are reserved for dialect attributes.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#attribute-value-aliases)
/// for more information.
#[derive(Copy, Clone)]
pub struct NamedAttributeRef<'c, 't> {
    /// Handle that represents the underlying named [`Attribute`] in the MLIR C API.
    handle: MlirNamedAttribute,

    /// [`Context`] that owns the underlying named [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> NamedAttributeRef<'c, 't> {
    /// Constructs a new [`NamedAttributeRef`] from the provided handle that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn from_c_api(handle: MlirNamedAttribute, context: &'c Context<'t>) -> Self {
        Self { handle, context }
    }

    /// Returns the [`MlirNamedAttribute`] that corresponds to this [`NamedAttributeRef`] and which can be passed to
    /// functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    pub unsafe fn to_c_api(&self) -> MlirNamedAttribute {
        self.handle
    }

    /// Returns the name of this [`NamedAttributeRef`].
    pub fn name(&self) -> Identifier<'c, 't> {
        unsafe { Identifier::from_c_api(self.handle.name) }
    }

    /// Returns the underlying [`Attribute`] of this [`NamedAttributeRef`].
    pub fn attribute(&self) -> AttributeRef<'c, 't> {
        unsafe { AttributeRef::from_c_api(self.handle.attribute, self.context).unwrap() }
    }
}

impl PartialEq for NamedAttributeRef<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        self.name() == other.name() && self.attribute() == other.attribute()
    }
}

impl Eq for NamedAttributeRef<'_, '_> {}

impl Display for NamedAttributeRef<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}: {}", self.name().to_string(), self.attribute().to_string())
    }
}

impl Debug for NamedAttributeRef<'_, '_> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "NamedAttributeRef[{}]", self.to_string())
    }
}

impl<'t> Context<'t> {
    /// Creates a new [`NamedAttributeRef`] owned by this [`Context`].
    pub fn named_attribute<'c, A: Attribute<'c, 't>>(
        &'c self,
        name: Identifier<'c, 't>,
        attribute: A,
    ) -> NamedAttributeRef<'c, 't> {
        unsafe { NamedAttributeRef::from_c_api(mlirNamedAttributeGet(name.to_c_api(), attribute.to_c_api()), &self) }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use pretty_assertions::assert_eq;

    use crate::{IntegerAttributeRef, UnitAttributeRef};

    use super::*;

    /// Helper for testing [`Attribute`] [`Display`] and [`Debug`] implementations.
    pub(crate) fn test_attribute_display_and_debug<'c, 't: 'c, A: Attribute<'c, 't>>(
        attribute: A,
        expected: &'static str,
    ) {
        assert_eq!(format!("{}", attribute), expected);

        // Extract the type name for `A` to check the [`Debug`] implementation.
        let type_name = std::any::type_name::<A>().rsplit("::").next().unwrap_or("").split("<").next().unwrap_or("");
        assert_eq!(format!("{:?}", attribute), format!("{type_name}[{expected}]"));
    }

    /// Helper for testing [`Attribute`] casting.
    pub(crate) fn test_attribute_casting<'c, 't: 'c, A: Attribute<'c, 't>>(attribute: A) {
        let rendered_attribute = attribute.to_string();

        // Test upcasting.
        let attribute = attribute.as_ref();
        assert!(attribute.is::<A>());
        assert_eq!(attribute.to_string(), rendered_attribute);

        // Test downcasting.
        let attribute = attribute.cast::<A>().unwrap();
        assert!(attribute.is::<A>());
        assert_eq!(attribute.to_string(), rendered_attribute);

        // Invalid cast from specific attribute.
        let attribute = attribute.context().unit_attribute();
        assert!(!attribute.is::<A>());
        assert_eq!(attribute.cast::<A>(), None);

        // Invalid cast from a generic attribute reference.
        let attribute = attribute.as_ref();
        assert!(!attribute.is::<A>());
        assert_eq!(attribute.cast::<A>(), None);
    }

    #[test]
    fn test_attribute_ref_type_id() {
        let context = Context::new();
        let unit_attribute_1 = context.unit_attribute();
        let unit_attribute_2 = context.unit_attribute();
        assert_eq!(unit_attribute_1.type_id(), unit_attribute_2.type_id());
    }

    #[test]
    fn test_attribute_ref() {
        let context = Context::new();
        let attribute = context.unit_attribute();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.r#type(), context.none_type());
        assert_eq!(attribute.dialect().namespace().unwrap(), "builtin");
    }

    #[test]
    fn test_attribute_ref_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal.
        let attribute_1 = context.unit_attribute();
        let attribute_2 = context.unit_attribute();
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.integer_attribute(context.signless_integer_type(32), 42);
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.unit_attribute();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_attribute_ref_display_and_debug() {
        let context = Context::new();
        let attribute = context.unit_attribute();
        test_attribute_display_and_debug(attribute, "unit");
    }

    #[test]
    fn test_attribute_ref_dump() {
        let context = Context::new();
        let attribute = context.unit_attribute();

        // We are just checking that [`AttributeRef::dump`] runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        attribute.dump();
    }

    #[test]
    fn test_attribute_ref_is_and_cast() {
        let context = Context::new();
        let attribute = context.unit_attribute();

        // Test `is` method.
        assert!(attribute.is::<UnitAttributeRef>());
        assert!(!attribute.is::<IntegerAttributeRef>());

        // Test `cast` method.
        assert!(attribute.cast::<UnitAttributeRef>().is_some());
        assert!(attribute.cast::<IntegerAttributeRef>().is_none());
    }

    #[test]
    fn test_null_attribute() {
        let context = Context::new();
        let attribute = context.null_attribute();
        assert_eq!(&context, attribute.context());
    }

    #[test]
    fn test_parse_attribute() {
        let context = Context::new();

        let attribute = context.parse_attribute("unit").unwrap();
        assert_eq!(attribute, context.unit_attribute());

        let attribute = context.parse_attribute("42 : i32").unwrap();
        assert_eq!(attribute, context.integer_attribute(context.signless_integer_type(32), 42));

        let attribute = context.parse_attribute("\"hello\"").unwrap();
        assert_eq!(attribute, context.string_attribute("hello"));

        let attribute = context.parse_attribute("[]").unwrap();
        assert_eq!(attribute, context.array_attribute::<AttributeRef>(&[]));

        assert!(context.parse_attribute("invalid_attribute_syntax!@#").is_none());
    }

    #[test]
    fn test_named_attribute() {
        let context = Context::new();
        let name = context.identifier("test_name");
        let attribute = context.unit_attribute();
        let named_attribute = context.named_attribute(name, attribute);
        assert_eq!(named_attribute.name(), name);
        assert_eq!(named_attribute.attribute(), attribute);
        assert_eq!(format!("{}", named_attribute), "test_name: unit");
        assert_eq!(format!("{:?}", named_attribute), "NamedAttributeRef[test_name: unit]");
    }

    #[test]
    fn test_named_attribute_ref_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal.
        let attribute_1 = context.named_attribute(context.identifier("name"), context.unit_attribute());
        let attribute_2 = context.named_attribute(context.identifier("name"), context.unit_attribute());

        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = context.named_attribute(
            context.identifier("name"),
            context.integer_attribute(context.signless_integer_type(32), 42),
        );
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = context.named_attribute(context.identifier("name"), context.unit_attribute());
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_named_attribute_with_integer_attribute() {
        let context = Context::new();
        let name = context.identifier("count");
        let attribute = context.integer_attribute(context.signless_integer_type(64), 42);
        let named_attribute = context.named_attribute(name, attribute);
        assert_eq!(named_attribute.name(), name);
        assert_eq!(named_attribute.attribute(), attribute);
    }

    #[test]
    fn test_named_attribute_with_string_attribute() {
        let context = Context::new();
        let name = context.identifier("message");
        let attribute = context.string_attribute("hello world");
        let named_attribute = context.named_attribute(name, attribute);
        assert_eq!(named_attribute.name(), name);
        assert_eq!(named_attribute.attribute(), attribute);
        assert_eq!(named_attribute.attribute().to_string(), "\"hello world\"");
    }
}
