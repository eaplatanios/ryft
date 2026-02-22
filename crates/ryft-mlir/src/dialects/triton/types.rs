// TODO(eaplatanios): Clean this up and make sure it is correct.

use ryft_xla_sys::bindings::MlirType;

use crate::{mlir_subtype_trait_impls, Context, OpaqueTypeRef, Type, TypeRef};

use super::TritonDialect;

/// Triton TTIR [`Type`] wrapper for types in the Triton dialect namespace (`tt`).
///
/// Triton type support in `ryft-mlir` currently relies on generic MLIR type parsing/casting paths rather than
/// dedicated Triton C API constructors. This wrapper provides a typed entry point for those types while preserving
/// standard [`Type`] casting semantics.
///
/// # Examples
///
/// The following is an example of a [`TritonTypeRef`] rendered using [`Display`]:
///
/// ```text
/// !tt.ptr<f32>
/// ```
///
/// Refer to the [official Triton dialect documentation](https://triton-lang.org/main/dialects/TritonDialect.html)
/// for more information.
#[derive(Copy, Clone)]
pub struct TritonTypeRef<'c, 't> {
    /// Handle that represents this [`Type`] in the MLIR C API.
    handle: MlirType,

    /// [`Context`] that owns this [`Type`].
    context: &'c Context<'t>,
}

impl<'c, 't> TritonTypeRef<'c, 't> {
    /// Returns the Triton dialect that this type belongs to.
    pub fn triton_dialect(&self) -> Option<TritonDialect> {
        type_triton_dialect(*self)
    }

    /// Returns the type mnemonic if it can be derived from its textual form.
    pub fn mnemonic(&self) -> Option<String> {
        type_mnemonic(*self)
    }
}

impl<'c, 't> Type<'c, 't> for TritonTypeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
        let type_ref = unsafe { TypeRef::from_c_api(handle, context) }?;
        if type_triton_dialect(type_ref).is_some() {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirType {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(TritonTypeRef<'c, 't> as Type, mlir_type = Type);

impl<'t> Context<'t> {
    /// Parses a Triton TTIR [`Type`] from the provided source string.
    pub fn parse_triton_type<'c, S: AsRef<str>>(&'c self, source: S) -> Option<TritonTypeRef<'c, 't>> {
        let _dialect = self.load_triton_dialect(TritonDialect::Triton);
        self.parse_type(source.as_ref()).and_then(|r#type| r#type.cast::<TritonTypeRef>())
    }

    /// Creates a Triton TTIR opaque [`Type`] in the provided Triton dialect namespace, owned by this [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `dialect`: Triton dialect namespace in which to create the type.
    ///   - `data`: Raw type payload without the `!<dialect>.` prefix.
    pub fn triton_opaque_type<'c, S: AsRef<str>>(&'c self, dialect: TritonDialect, data: S) -> TritonTypeRef<'c, 't> {
        let _dialect = self.load_triton_dialect(dialect);
        self.opaque_type(dialect.namespace(), data).cast::<TritonTypeRef>().unwrap()
    }
}

macro_rules! triton_type_ref {
    ($(#[$doc:meta])* $type_ref:ident, $mnemonic:literal, $constructor:ident, $parser:ident) => {
        $(#[$doc])*
        #[derive(Copy, Clone)]
        pub struct $type_ref<'c, 't> {
            /// Handle that represents this [`Type`] in the MLIR C API.
            handle: MlirType,

            /// [`Context`] that owns this [`Type`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> Type<'c, 't> for $type_ref<'c, 't> {
            unsafe fn from_c_api(handle: MlirType, context: &'c Context<'t>) -> Option<Self> {
                let type_ref = unsafe { TritonTypeRef::from_c_api(handle, context) }?;
                if type_has_triton_mnemonic(type_ref, $mnemonic) { Some(Self { handle, context }) } else { None }
            }

            unsafe fn to_c_api(&self) -> MlirType {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($type_ref<'c, 't> as Type, mlir_type = Type);

        impl<'t> Context<'t> {
            #[doc = concat!("Creates a new `tt.", $mnemonic, "` type owned by this [`Context`].")]
            ///
            /// # Parameters
            ///
            ///   - `parameters`: Textual payload to include inside `tt.<mnemonic><...>`.
            pub fn $constructor<'c, S: AsRef<str>>(&'c self, parameters: S) -> $type_ref<'c, 't> {
                self.triton_opaque_type(
                    TritonDialect::Triton,
                    format!(concat!($mnemonic, "<{}>"), parameters.as_ref()),
                )
                .cast::<$type_ref>()
                .unwrap()
            }

            #[doc = concat!("Parses a `tt.", $mnemonic, "` type from the provided source string.")]
            pub fn $parser<'c, S: AsRef<str>>(&'c self, source: S) -> Option<$type_ref<'c, 't>> {
                self.parse_triton_type(source).and_then(|r#type| r#type.cast::<$type_ref>())
            }
        }
    };
}

triton_type_ref!(
    /// Triton TTIR `tt.ptr` [`Type`].
    ///
    /// Refer to the upstream Triton ODS type definitions in
    /// [`TritonTypes.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonTypes.td)
    /// for more information.
    PointerTypeRef,
    "ptr",
    triton_pointer_type,
    parse_triton_pointer_type
);

triton_type_ref!(
    /// Triton TTIR `tt.tensordesc` [`Type`].
    ///
    /// Refer to the upstream Triton ODS type definitions in
    /// [`TritonTypes.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonTypes.td)
    /// for more information.
    TensorDescriptorTypeRef,
    "tensordesc",
    triton_tensor_descriptor_type,
    parse_triton_tensor_descriptor_type
);

fn type_triton_dialect<'c, 't: 'c, T: Type<'c, 't>>(r#type: T) -> Option<TritonDialect> {
    r#type.dialect().namespace().ok().and_then(TritonDialect::from_namespace).or_else(|| {
        r#type
            .cast::<OpaqueTypeRef>()
            .and_then(|opaque_type| opaque_type.dialect_namespace().ok().and_then(TritonDialect::from_namespace))
    })
}

fn type_has_triton_mnemonic<'c, 't: 'c, T: Type<'c, 't>, M: AsRef<str>>(r#type: T, mnemonic: M) -> bool {
    type_mnemonic(r#type).as_deref() == Some(mnemonic.as_ref())
}

fn type_mnemonic<'c, 't: 'c, T: Type<'c, 't>>(r#type: T) -> Option<String> {
    let dialect = type_triton_dialect(r#type)?;
    let rendered = r#type.to_string();
    let prefix = format!("!{}.", dialect.namespace());
    if !rendered.starts_with(prefix.as_str()) {
        return None;
    }

    let suffix = &rendered[prefix.len()..];
    let end = suffix
        .char_indices()
        .find_map(|(index, character)| {
            if character == '<' || character == ' ' || character == '\t' || character == '\n' || character == ':' {
                Some(index)
            } else {
                None
            }
        })
        .unwrap_or(suffix.len());

    Some(suffix[..end].to_owned())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::types::tests::{test_type_casting, test_type_display_and_debug};

    use super::*;

    fn create_triton_type<'c, 't>(context: &'c Context<'t>, data: &str) -> TritonTypeRef<'c, 't> {
        context.triton_opaque_type(TritonDialect::Triton, data)
    }

    #[test]
    fn test_triton_type() {
        let context = Context::new();

        let r#type = create_triton_type(&context, "ptr<f32>");
        assert_eq!(&context, r#type.context());
        assert_eq!(r#type.triton_dialect(), Some(TritonDialect::Triton));
        assert_eq!(r#type.mnemonic().as_deref(), Some("ptr"));
    }

    #[test]
    fn test_triton_type_equality() {
        let context = Context::new();

        // Same types from the same context must be equal because they are "uniqued".
        let type_1 = create_triton_type(&context, "ptr<f32>");
        let type_2 = create_triton_type(&context, "ptr<f32>");
        assert_eq!(type_1, type_2);

        // Different types from the same context must not be equal.
        let type_2 = create_triton_type(&context, "ptr<i32>");
        assert_ne!(type_1, type_2);

        // Same types from different contexts must not be equal.
        let context = Context::new();
        let type_2 = create_triton_type(&context, "ptr<f32>");
        assert_ne!(type_1, type_2);
    }

    #[test]
    fn test_triton_type_display_and_debug() {
        let context = Context::new();
        let r#type = create_triton_type(&context, "ptr<f32>");
        test_type_display_and_debug(r#type, "!tt.ptr<f32>");
    }

    #[test]
    fn test_triton_type_casting() {
        let context = Context::new();
        let r#type = create_triton_type(&context, "ptr<f32>");
        test_type_casting(r#type);
    }

    #[test]
    fn test_triton_type_from_opaque_type() {
        let context = Context::new();

        let pointer = context.opaque_type("tt", "ptr<f32>");
        let pointer = pointer.cast::<PointerTypeRef>();
        assert!(pointer.is_some());
        assert_eq!(pointer.unwrap().cast::<TritonTypeRef>().unwrap().mnemonic().as_deref(), Some("ptr"));

        let tensor_desc = context.opaque_type("tt", "tensordesc<tensor<4xf16>>");
        let tensor_desc = tensor_desc.cast::<TensorDescriptorTypeRef>();
        assert!(tensor_desc.is_some());

        let not_frontend = context.opaque_type("ttg", "memdesc<16xf32, #ttg.blocked>");
        assert!(not_frontend.cast::<TritonTypeRef>().is_none());
    }

    #[test]
    fn test_triton_pointer_type() {
        let context = Context::new();
        let pointer = context.triton_pointer_type("f32");
        assert_eq!(&context, pointer.context());
        assert_eq!(pointer.cast::<TritonTypeRef>().unwrap().mnemonic().as_deref(), Some("ptr"));
        test_type_display_and_debug(pointer, "!tt.ptr<f32>");
        test_type_casting(pointer);
    }

    #[test]
    fn test_triton_tensor_descriptor_type() {
        let context = Context::new();
        let tensor_descriptor = context.triton_tensor_descriptor_type("tensor<4xf16>");
        assert_eq!(&context, tensor_descriptor.context());
        assert_eq!(tensor_descriptor.cast::<TritonTypeRef>().unwrap().mnemonic().as_deref(), Some("tensordesc"));
        test_type_display_and_debug(tensor_descriptor, "!tt.tensordesc<tensor<4xf16>>");
        test_type_casting(tensor_descriptor);
    }

    #[test]
    fn test_parse_triton_type() {
        let context = Context::new();
        context.allow_unregistered_dialects();

        assert_eq!(context.parse_triton_type("!tt.ptr<f32>").unwrap(), create_triton_type(&context, "ptr<f32>"),);
        assert_eq!(context.parse_triton_type("!ttg.memdesc<16xf32, #ttg.blocked>"), None);

        let pointer = context.parse_triton_pointer_type("!tt.ptr<f32>");
        assert!(pointer.is_some());

        let tensor_descriptor = context.parse_triton_tensor_descriptor_type("!tt.tensordesc<tensor<4xf16>>");
        assert!(tensor_descriptor.is_some());
    }
}
