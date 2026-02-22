// TODO(eaplatanios): Clean this up and make sure it is correct.

use ryft_xla_sys::bindings::MlirAttribute;

use crate::{mlir_subtype_trait_impls, Attribute, AttributeRef, Context, OpaqueAttributeRef, Type};

use super::TritonDialect;

/// Triton TTIR [`Attribute`] wrapper for attributes in the Triton dialect namespace (`tt`).
///
/// Triton attribute support in `ryft-mlir` currently relies on generic MLIR attribute parsing/casting paths rather
/// than dedicated Triton C API constructors. This wrapper provides a typed entry point for those attributes while
/// preserving standard [`Attribute`] casting semantics.
///
/// # Examples
///
/// The following is an example of a [`TritonAttributeRef`] rendered using [`Display`]:
///
/// ```text
/// #tt.cache_modifier<none>
/// ```
///
/// Refer to the [official Triton dialect documentation](https://triton-lang.org/main/dialects/TritonDialect.html)
/// for more information.
#[derive(Copy, Clone)]
pub struct TritonAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> TritonAttributeRef<'c, 't> {
    /// Returns the Triton dialect that this attribute belongs to.
    pub fn triton_dialect(&self) -> Option<TritonDialect> {
        attribute_triton_dialect(*self)
    }

    /// Returns the attribute mnemonic if it can be derived from its textual form.
    pub fn mnemonic(&self) -> Option<String> {
        attribute_mnemonic(*self)
    }
}

impl<'c, 't> Attribute<'c, 't> for TritonAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
        if attribute_triton_dialect(attribute).is_some() {
            Some(Self { handle, context })
        } else {
            None
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        &self.context
    }
}

mlir_subtype_trait_impls!(TritonAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Parses a Triton TTIR [`Attribute`] from the provided source string.
    pub fn parse_triton_attribute<'c, S: AsRef<str>>(&'c self, source: S) -> Option<TritonAttributeRef<'c, 't>> {
        let _dialect = self.load_triton_dialect(TritonDialect::Triton);
        self.parse_attribute(source.as_ref()).and_then(|attribute| attribute.cast::<TritonAttributeRef>())
    }

    /// Creates a Triton TTIR opaque [`Attribute`] in the provided Triton dialect namespace, owned by this
    /// [`Context`].
    ///
    /// # Parameters
    ///
    ///   - `dialect`: Triton dialect namespace in which to create the attribute.
    ///   - `data`: Raw attribute payload without the `#<dialect>.` prefix.
    ///   - `r#type`: MLIR type attached to the resulting opaque attribute.
    pub fn triton_opaque_attribute<'c, S: AsRef<str>, T: Type<'c, 't>>(
        &'c self,
        dialect: TritonDialect,
        data: S,
        r#type: T,
    ) -> TritonAttributeRef<'c, 't> {
        let _dialect = self.load_triton_dialect(dialect);
        self.opaque_attribute(dialect.namespace(), data, r#type).cast::<TritonAttributeRef>().unwrap()
    }
}

macro_rules! triton_enum_attribute_ref {
    ($(#[$doc:meta])* $attribute_ref:ident, $mnemonic:literal, $constructor:ident) => {
        $(#[$doc])*
        #[derive(Copy, Clone)]
        pub struct $attribute_ref<'c, 't> {
            /// Handle that represents this [`Attribute`] in the MLIR C API.
            handle: MlirAttribute,

            /// [`Context`] that owns this [`Attribute`].
            context: &'c Context<'t>,
        }

        impl<'c, 't> $attribute_ref<'c, 't> {
            /// Returns the enum token value stored in this attribute.
            pub fn value(&self) -> String {
                let rendered = self.to_string();
                parse_triton_enum_attribute_value(rendered.clone(), $mnemonic)
                    .unwrap_or_else(|| panic!("invalid tt.{} attribute: '{}'", $mnemonic, rendered))
            }
        }

        impl<'c, 't> Attribute<'c, 't> for $attribute_ref<'c, 't> {
            unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
                let attribute = unsafe { AttributeRef::from_c_api(handle, context) }?;
                if attribute_has_triton_mnemonic(attribute, $mnemonic) { Some(Self { handle, context }) } else { None }
            }

            unsafe fn to_c_api(&self) -> MlirAttribute {
                self.handle
            }

            fn context(&self) -> &'c Context<'t> {
                self.context
            }
        }

        mlir_subtype_trait_impls!($attribute_ref<'c, 't> as Attribute, mlir_type = Attribute);

        impl<'t> Context<'t> {
            #[doc = concat!("Creates a new `tt.", $mnemonic, "` attribute owned by this [`Context`].")]
            #[doc = ""]
            #[doc = "# Parameters"]
            #[doc = ""]
            #[doc = "   - `value`: Raw enum token value to place in the attribute payload."]
            pub fn $constructor<'c, S: AsRef<str>>(&'c self, value: S) -> $attribute_ref<'c, 't> {
                self.triton_opaque_attribute(
                    TritonDialect::Triton,
                    format!(concat!($mnemonic, "<{}>"), value.as_ref()),
                    self.none_type(),
                )
                .cast::<$attribute_ref>()
                .unwrap()
            }
        }
    };
}

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.cache_modifier` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    CacheModifierAttributeRef,
    "cache_modifier",
    triton_cache_modifier_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.mem_semantic` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    MemSemanticAttributeRef,
    "mem_semantic",
    triton_mem_semantic_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.eviction_policy` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    EvictionPolicyAttributeRef,
    "eviction_policy",
    triton_eviction_policy_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.padding_option` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    PaddingOptionAttributeRef,
    "padding_option",
    triton_padding_option_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.atomic_rmw` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    AtomicRmwAttributeRef,
    "atomic_rmw",
    triton_atomic_rmw_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.descriptor_reduce_kind` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    DescriptorReduceKindAttributeRef,
    "descriptor_reduce_kind",
    triton_descriptor_reduce_kind_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.mem_sync_scope` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    MemSyncScopeAttributeRef,
    "mem_sync_scope",
    triton_mem_sync_scope_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.program_dim` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    ProgramDimAttributeRef,
    "program_dim",
    triton_program_dim_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.rounding_mode` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    RoundingModeAttributeRef,
    "rounding_mode",
    triton_rounding_mode_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.propagate_nan` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    PropagateNanAttributeRef,
    "propagate_nan",
    triton_propagate_nan_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.input_precision` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    InputPrecisionAttributeRef,
    "input_precision",
    triton_input_precision_attribute
);

triton_enum_attribute_ref!(
    /// Triton TTIR `tt.scale_dot_elem_type` enum [`Attribute`].
    ///
    /// Refer to the upstream Triton ODS definitions in
    /// [`TritonAttrDefs.td`](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/Triton/IR/TritonAttrDefs.td)
    /// for more information.
    ScaleDotElemTypeAttributeRef,
    "scale_dot_elem_type",
    triton_scale_dot_elem_type_attribute
);

fn attribute_triton_dialect<'c, 't: 'c, A: Attribute<'c, 't>>(attribute: A) -> Option<TritonDialect> {
    attribute.dialect().namespace().ok().and_then(TritonDialect::from_namespace).or_else(|| {
        attribute.cast::<OpaqueAttributeRef>().and_then(|opaque_attribute| {
            opaque_attribute.dialect_namespace().ok().and_then(TritonDialect::from_namespace)
        })
    })
}

fn attribute_has_triton_mnemonic<'c, 't: 'c, A: Attribute<'c, 't>, M: AsRef<str>>(attribute: A, mnemonic: M) -> bool {
    attribute_mnemonic(attribute).as_deref() == Some(mnemonic.as_ref())
}

fn attribute_mnemonic<'c, 't: 'c, A: Attribute<'c, 't>>(attribute: A) -> Option<String> {
    let dialect = attribute_triton_dialect(attribute)?;
    let rendered = attribute.to_string();
    let prefix = format!("#{}.", dialect.namespace());
    if !rendered.starts_with(prefix.as_str()) {
        return None;
    }

    let suffix = &rendered[prefix.len()..];
    let end = suffix
        .char_indices()
        .find_map(|(index, character)| {
            if character == ':'
                || character == '>'
                || character == '<'
                || character == ' '
                || character == '\t'
                || character == '\n'
            {
                Some(index)
            } else {
                None
            }
        })
        .unwrap_or(suffix.len());

    Some(suffix[..end].to_owned())
}

fn parse_triton_enum_attribute_value(rendered: String, mnemonic: &str) -> Option<String> {
    let rendered = rendered.trim().to_owned();
    let prefix = format!("#tt.{mnemonic}<");
    let suffix = rendered.strip_prefix(prefix.as_str())?;
    let end = suffix.find('>')?;
    Some(suffix[..end].to_owned())
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    fn create_triton_attribute<'c, 't>(context: &'c Context<'t>, data: &str) -> TritonAttributeRef<'c, 't> {
        context.triton_opaque_attribute(TritonDialect::Triton, data, context.none_type())
    }

    macro_rules! test_ttir_enum_attribute {
        ($test_name:ident, $constructor:ident, $attribute_ref:ident, $mnemonic:literal, $value:literal) => {
            #[test]
            fn $test_name() {
                let context = Context::new();
                let attribute = context.$constructor($value);
                assert_eq!(&context, attribute.context());
                assert_eq!(attribute.value(), $value);
                assert_eq!(attribute.cast::<TritonAttributeRef>().unwrap().mnemonic().as_deref(), Some($mnemonic),);
                test_attribute_display_and_debug(attribute, concat!("#tt.", $mnemonic, "<", $value, ">"));
                test_attribute_casting(attribute);

                let cast = context
                    .opaque_attribute("tt", concat!($mnemonic, "<", $value, ">"), context.none_type())
                    .cast::<$attribute_ref>();
                assert!(cast.is_some());
                assert_eq!(cast.unwrap(), attribute);
            }
        };
    }

    #[test]
    fn test_triton_attribute() {
        let context = Context::new();

        let attribute = create_triton_attribute(&context, "cache_modifier<none>");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.triton_dialect(), Some(TritonDialect::Triton));
        assert_eq!(attribute.mnemonic().as_deref(), Some("cache_modifier"));
    }

    #[test]
    fn test_triton_attribute_equality() {
        let context = Context::new();

        // Same attributes from the same context must be equal because they are "uniqued".
        let attribute_1 = create_triton_attribute(&context, "cache_modifier<none>");
        let attribute_2 = create_triton_attribute(&context, "cache_modifier<none>");
        assert_eq!(attribute_1, attribute_2);

        // Different attributes from the same context must not be equal.
        let attribute_2 = create_triton_attribute(&context, "cache_modifier<ca>");
        assert_ne!(attribute_1, attribute_2);

        // Same attributes from different contexts must not be equal.
        let context = Context::new();
        let attribute_2 = create_triton_attribute(&context, "cache_modifier<none>");
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_triton_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = create_triton_attribute(&context, "cache_modifier<none>");
        test_attribute_display_and_debug(attribute, "#tt.cache_modifier<none>");
    }

    #[test]
    fn test_triton_attribute_casting() {
        let context = Context::new();
        let attribute = create_triton_attribute(&context, "cache_modifier<none>");
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_triton_attribute_from_opaque_attribute() {
        let context = Context::new();

        let attribute = context.opaque_attribute("tt", "cache_modifier<none>", context.none_type());
        let attribute = attribute.cast::<TritonAttributeRef>();
        assert!(attribute.is_some());
        assert_eq!(attribute.unwrap().triton_dialect(), Some(TritonDialect::Triton));

        let not_frontend = context.opaque_attribute("ttg", "blocked<{}>", context.none_type());
        assert!(not_frontend.cast::<TritonAttributeRef>().is_none());
    }

    test_ttir_enum_attribute!(
        test_cache_modifier_attribute,
        triton_cache_modifier_attribute,
        CacheModifierAttributeRef,
        "cache_modifier",
        "none"
    );
    test_ttir_enum_attribute!(
        test_mem_semantic_attribute,
        triton_mem_semantic_attribute,
        MemSemanticAttributeRef,
        "mem_semantic",
        "relaxed"
    );
    test_ttir_enum_attribute!(
        test_eviction_policy_attribute,
        triton_eviction_policy_attribute,
        EvictionPolicyAttributeRef,
        "eviction_policy",
        "normal"
    );
    test_ttir_enum_attribute!(
        test_padding_option_attribute,
        triton_padding_option_attribute,
        PaddingOptionAttributeRef,
        "padding_option",
        "zero"
    );
    test_ttir_enum_attribute!(
        test_atomic_rmw_attribute,
        triton_atomic_rmw_attribute,
        AtomicRmwAttributeRef,
        "atomic_rmw",
        "add"
    );
    test_ttir_enum_attribute!(
        test_descriptor_reduce_kind_attribute,
        triton_descriptor_reduce_kind_attribute,
        DescriptorReduceKindAttributeRef,
        "descriptor_reduce_kind",
        "add"
    );
    test_ttir_enum_attribute!(
        test_mem_sync_scope_attribute,
        triton_mem_sync_scope_attribute,
        MemSyncScopeAttributeRef,
        "mem_sync_scope",
        "gpu"
    );
    test_ttir_enum_attribute!(
        test_program_dim_attribute,
        triton_program_dim_attribute,
        ProgramDimAttributeRef,
        "program_dim",
        "x"
    );
    test_ttir_enum_attribute!(
        test_rounding_mode_attribute,
        triton_rounding_mode_attribute,
        RoundingModeAttributeRef,
        "rounding_mode",
        "rtne"
    );
    test_ttir_enum_attribute!(
        test_propagate_nan_attribute,
        triton_propagate_nan_attribute,
        PropagateNanAttributeRef,
        "propagate_nan",
        "none"
    );
    test_ttir_enum_attribute!(
        test_input_precision_attribute,
        triton_input_precision_attribute,
        InputPrecisionAttributeRef,
        "input_precision",
        "ieee"
    );
    test_ttir_enum_attribute!(
        test_scale_dot_elem_type_attribute,
        triton_scale_dot_elem_type_attribute,
        ScaleDotElemTypeAttributeRef,
        "scale_dot_elem_type",
        "fp16"
    );

    #[test]
    fn test_parse_triton_attribute() {
        let context = Context::new();
        context.allow_unregistered_dialects();

        assert_eq!(
            context.parse_triton_attribute("#tt.cache_modifier<none>").unwrap(),
            create_triton_attribute(&context, "cache_modifier<none>"),
        );
        assert_eq!(context.parse_triton_attribute("#ttg.blocked<{}>"), None);
    }
}
