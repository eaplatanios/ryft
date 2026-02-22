use std::collections::HashMap;
use std::fmt::{Debug, Display};

use ryft_xla_sys::bindings::{MlirType, mlirTypeDump, mlirTypeGetDialect, mlirTypeGetTypeID, mlirTypeParseGet};

use crate::{AttributeRef, Context, Dialect, StringRef, TypeId, mlir_subtype_trait_impls};

/// Each value in MLIR has a [`Type`] defined by the MLIR type system. MLIR has an open type system (i.e., there is no
/// fixed set of types), and types may have application-specific semantics.MLIR dialects may define any number of
/// types with no restrictions on the abstractions they represent.
///
/// This trait acts effectively as the super-type of all MLIR [`Type`]s and can be checked and specialized using the
/// [`Type::is`] and [`Type::cast`] functions.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/LangRef/#type-system) for more information.
pub trait Type<'c, 't: 'c>: Sized + Copy + Clone + PartialEq + Eq + Display + Debug {
    /// Constructs a new type of this type from the provided handle that came from a function in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self>;

    /// Returns the [`MlirType`] that corresponds to this type and which can be passed to functions in the MLIR C API.
    ///
    /// This function is marked as unsafe because handling the MLIR C API representations in Rust is generally not
    /// safe and should not be necessary outside of this library. However, it is still supported via making functions
    /// like this one public so that users of this library can extend it with yet unsupported features that the
    /// underlying MLIR C API supports.
    unsafe fn to_c_api(&self) -> MlirType;

    /// Returns a reference to the [`Context`] that owns this type.
    fn context(&self) -> &'c Context<'t>;

    /// Returns `true` if this type is an instance of `T`.
    fn is<T: Type<'c, 't>>(&self) -> bool {
        Self::cast::<T>(&self).is_some()
    }

    /// Tries to cast this type to an instance of `T` (e.g., an instance of [`IntegerTypeRef`](crate::IntegerTypeRef)).
    /// If this is not an instance of the specified type, this function will return [`None`].
    fn cast<T: Type<'c, 't>>(&self) -> Option<T> {
        unsafe { T::from_c_api(self.to_c_api(), self.context()) }
    }

    /// Up-casts this type to an instance of [`Type`].
    fn as_ref(&self) -> TypeRef<'c, 't> {
        unsafe { TypeRef::from_c_api(self.to_c_api(), self.context()).unwrap() }
    }

    /// Gets the [`TypeId`] of this type. Note that this function may return the same [`TypeId`] for different
    /// instances of the same type with potentially different attributes. That is because a [`TypeId`] is a unique
    /// identifier of the corresponding MLIR C++ type for [`Type`] and not for a specific instance of [`Type`].
    fn type_id(&self) -> TypeId<'c> {
        unsafe { TypeId::from_c_api(mlirTypeGetTypeID(self.to_c_api())).unwrap() }
    }

    /// Returns the [`Dialect`] that this type belongs to.
    fn dialect(&self) -> Dialect<'c, 't> {
        unsafe { Dialect::from_c_api(mlirTypeGetDialect(self.to_c_api())).unwrap() }
    }

    /// Dumps this type to the standard error stream.
    fn dump(&self) {
        unsafe { mlirTypeDump(self.to_c_api()) }
    }
}

/// Reference to an MLIR [`Type`] that is owned by a [`Context`].
#[derive(Copy, Clone)]
pub struct TypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> Type<'c, 't> for TypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() { None } else { Some(Self { handle, context }) }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(TypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Parses a [`Type`] from the provided string representation. Returns [`None`] if MLIR fails to parse
    /// the provided string into a [`Type`] (this function will also emit diagnostics if that happens).
    /// The resulting [`Type`] is owned by this [`Context`].
    pub fn parse_type<'c, S: AsRef<str>>(&'c self, source: S) -> Option<TypeRef<'c, 't>> {
        unsafe {
            TypeRef::from_c_api(
                mlirTypeParseGet(*self.handle.borrow_mut(), StringRef::from(source.as_ref()).to_c_api()),
                &self,
            )
        }
    }
}

/// A [`TypeRef`] paired with optional named [`Attribute`](crate::Attribute)s. This is typically used to represent the
/// type and attributes of function arguments and results.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TypeAndAttributes<'c, 't, 's> {
    /// Reference to a [`Type`].
    pub r#type: TypeRef<'c, 't>,

    /// Optional [`HashMap`] from attribute names to [`Attribute`](crate::Attribute)s.
    pub attributes: Option<HashMap<StringRef<'s>, AttributeRef<'c, 't>>>,
}

impl<'c, 't, T: Type<'c, 't>> From<T> for TypeAndAttributes<'c, 't, '_> {
    fn from(value: T) -> Self {
        Self { r#type: value.as_ref(), attributes: None }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use crate::Attribute;

    use super::*;

    /// Helper for testing [`Type`] [`Display`] and [`Debug`] implementations.
    pub(crate) fn test_type_display_and_debug<'c, 't: 'c, T: Type<'c, 't>>(r#type: T, expected: &'static str) {
        assert_eq!(format!("{}", r#type), expected);

        // Extract the type name for `L` to check the [`Debug`] implementation.
        let type_name = std::any::type_name::<T>().rsplit("::").next().unwrap_or("").split("<").next().unwrap_or("");
        assert_eq!(format!("{:?}", r#type), format!("{type_name}[{expected}]"));
    }

    /// Helper for testing [`Type`] casting.
    pub(crate) fn test_type_casting<'c, 't: 'c, T: Type<'c, 't>>(r#type: T) {
        let rendered_type = r#type.to_string();

        // Test upcasting.
        let r#type = r#type.as_ref();
        assert!(r#type.is::<T>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Test downcasting.
        let r#type = r#type.cast::<T>().unwrap();
        assert!(r#type.is::<T>());
        assert_eq!(r#type.to_string(), rendered_type);

        // Invalid cast from specific type.
        let r#type = r#type.context().none_type();
        assert!(!r#type.is::<T>());
        assert_eq!(r#type.cast::<T>(), None);

        // Invalid cast from a generic type reference.
        let r#type = r#type.as_ref();
        assert!(!r#type.is::<T>());
        assert_eq!(r#type.cast::<T>(), None);
    }

    #[test]
    fn test_type() {
        let context = Context::new();
        let index_type = context.index_type();
        assert_eq!(index_type.context(), &context);
        assert_eq!(index_type.type_id(), context.index_type().type_id());
        assert_ne!(index_type.type_id(), context.signed_integer_type(32).type_id());
        assert_eq!(index_type.clone().dialect().namespace().unwrap(), "builtin");
        test_type_display_and_debug(index_type, "index");
        test_type_casting(index_type);

        // Test C API integration.
        let type_ref = unsafe { TypeRef::from_c_api(index_type.to_c_api(), &context).unwrap() };
        assert_eq!(type_ref.clone().to_string(), "index");
        assert_eq!(type_ref.context(), &context);

        // Test null pointer edge case.
        let bad_handle = MlirType { ptr: std::ptr::null_mut() };
        let type_ref = unsafe { TypeRef::from_c_api(bad_handle, &context) };
        assert!(type_ref.is_none());
    }

    #[test]
    fn test_type_dump() {
        let context = Context::new();
        let index_type = context.index_type();

        // We are just checking that [`Type::dump`] runs successfully without crashing.
        // Ideally, we would want a way to capture the standard error stream and verify that it printed the right thing.
        index_type.dump();
    }

    #[test]
    fn test_type_equality() {
        let context = Context::new();
        let index_type_0 = context.index_type();
        let index_type_1 = context.index_type();
        let i32_type = context.signed_integer_type(32);
        assert_eq!(index_type_0, index_type_0);
        assert_eq!(index_type_0, index_type_0.as_ref());
        assert_eq!(index_type_0.as_ref(), index_type_0);
        assert_eq!(index_type_0.as_ref(), index_type_0.as_ref());
        assert_eq!(index_type_0, index_type_1.as_ref());
        assert_eq!(index_type_1, index_type_1);
        assert_ne!(index_type_0, i32_type);
        assert_ne!(i32_type, index_type_0);
        assert_ne!(index_type_1, i32_type);
        assert_eq!(i32_type.as_ref(), i32_type);
    }

    #[test]
    fn test_type_parsing() {
        let context = Context::new();
        context.allow_unregistered_dialects();
        assert_eq!(context.parse_type("index"), Some(context.index_type().as_ref()));
        assert_eq!(context.parse_type("i32"), Some(context.signless_integer_type(32).as_ref()));
        assert_eq!(context.parse_type("f64"), Some(context.float64_type().as_ref()));
        assert!(context.parse_type("tensor<3x4xf32>").is_some());
        assert!(context.parse_type("!llvm.ptr").is_some());
        assert!(context.parse_type("invalid_type_xyz").is_none());
    }

    #[test]
    fn test_type_and_attributes() {
        let context = Context::new();
        let index_type = context.index_type();
        let type_and_attributes = TypeAndAttributes::from(index_type);
        assert_eq!(type_and_attributes.r#type.to_string(), "index");
        assert!(type_and_attributes.attributes.is_none());
        let type_and_attributes = TypeAndAttributes {
            r#type: index_type.as_ref(),
            attributes: Some(HashMap::from([("test_attr".into(), context.unit_attribute().as_ref())])),
        };
        assert_eq!(type_and_attributes.clone().r#type.to_string(), "index");
        assert!(type_and_attributes.attributes.is_some());
        assert_eq!(type_and_attributes.attributes.unwrap().len(), 1);
    }
}
