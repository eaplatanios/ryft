use ryft_xla_sys::bindings::{
    MlirAttribute, MlirEmitCCmpPredicate, MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_EQ,
    MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GE, MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GT,
    MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LE, MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LT,
    MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_NE, MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_THREE_WAY,
    mlirAttributeIsAEmitCCmpPredicate, mlirAttributeIsAEmitCOpaque, mlirEmitCCmpPredicateAttrGet,
    mlirEmitCCmpPredicateAttrGetTypeID, mlirEmitCCmpPredicateAttrGetValue, mlirEmitCOpaqueAttrGet,
    mlirEmitCOpaqueAttrGetTypeID, mlirEmitCOpaqueAttrGetValue,
};

use crate::{Attribute, Context, DialectHandle, StringRef, TypeId, mlir_subtype_trait_impls};

/// Emit-C comparison predicate.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u64)]
pub enum CmpPredicate {
    /// Equal comparison, spelled `eq` in MLIR.
    Equal = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_EQ,

    /// Not-equal comparison, spelled `ne` in MLIR.
    NotEqual = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_NE,

    /// Less-than comparison, spelled `lt` in MLIR.
    LessThan = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LT,

    /// Less-than-or-equal comparison, spelled `le` in MLIR.
    LessThanOrEqual = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LE,

    /// Greater-than comparison, spelled `gt` in MLIR.
    GreaterThan = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GT,

    /// Greater-than-or-equal comparison, spelled `ge` in MLIR.
    GreaterThanOrEqual = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GE,

    /// Three-way comparison, spelled `three_way` in MLIR.
    ThreeWay = MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_THREE_WAY,
}

impl CmpPredicate {
    /// Returns the MLIR spelling of this predicate.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::LessThan => "lt",
            Self::LessThanOrEqual => "le",
            Self::GreaterThan => "gt",
            Self::GreaterThanOrEqual => "ge",
            Self::ThreeWay => "three_way",
        }
    }

    /// Constructs a [`CmpPredicate`] from its MLIR C API representation.
    pub fn from_c_api(value: MlirEmitCCmpPredicate) -> Option<Self> {
        if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_EQ {
            Some(Self::Equal)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_NE {
            Some(Self::NotEqual)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LT {
            Some(Self::LessThan)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_LE {
            Some(Self::LessThanOrEqual)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GT {
            Some(Self::GreaterThan)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_GE {
            Some(Self::GreaterThanOrEqual)
        } else if value == MlirEmitCCmpPredicate_MLIR_EMITC_CMP_PREDICATE_THREE_WAY {
            Some(Self::ThreeWay)
        } else {
            None
        }
    }

    /// Returns the MLIR C API representation of this predicate.
    pub fn to_c_api(&self) -> MlirEmitCCmpPredicate {
        *self as MlirEmitCCmpPredicate
    }
}

/// Emit-C comparison predicate [`Attribute`].
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#attributes)
/// for more information.
#[derive(Copy, Clone)]
pub struct CmpPredicateAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl CmpPredicateAttributeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`CmpPredicateAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCCmpPredicateAttrGetTypeID()).unwrap() }
    }

    /// Returns the comparison predicate stored by this attribute.
    pub fn value(&self) -> CmpPredicate {
        CmpPredicate::from_c_api(unsafe { mlirEmitCCmpPredicateAttrGetValue(self.handle) })
            .expect("invalid EmitC comparison predicate")
    }
}

impl<'c, 't> Attribute<'c, 't> for CmpPredicateAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirAttributeIsAEmitCCmpPredicate(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(CmpPredicateAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Emit-C opaque [`Attribute`] containing a C/C++ source fragment that is emitted as-is.
///
/// Refer to the [official MLIR Emit-C dialect documentation](https://mlir.llvm.org/docs/Dialects/emitc/#attributes)
/// for more information.
#[derive(Copy, Clone)]
pub struct OpaqueAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl OpaqueAttributeRef<'_, '_> {
    /// Gets the [`TypeId`] that corresponds to [`OpaqueAttributeRef`].
    pub fn type_id() -> TypeId<'static> {
        unsafe { TypeId::from_c_api(mlirEmitCOpaqueAttrGetTypeID()).unwrap() }
    }

    /// Returns the opaque value stored by this attribute.
    pub fn value(&self) -> StringRef<'_> {
        unsafe { StringRef::from_c_api(mlirEmitCOpaqueAttrGetValue(self.handle)) }
    }
}

impl<'c, 't> Attribute<'c, 't> for OpaqueAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Option<Self> {
        if handle.ptr.is_null() {
            return None;
        }
        if unsafe { mlirAttributeIsAEmitCOpaque(handle) } { Some(Self { handle, context }) } else { None }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(OpaqueAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new Emit-C [`CmpPredicateAttributeRef`] owned by this [`Context`].
    pub fn emit_c_cmp_predicate_attribute<'c>(&'c self, predicate: CmpPredicate) -> CmpPredicateAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            CmpPredicateAttributeRef::from_c_api(
                mlirEmitCCmpPredicateAttrGet(*self.handle.borrow(), predicate.to_c_api()),
                self,
            )
            .expect("invalid EmitC comparison predicate attribute")
        }
    }

    /// Creates a new Emit-C [`OpaqueAttributeRef`] owned by this [`Context`].
    pub fn emit_c_opaque_attribute<'c, S: AsRef<str>>(&'c self, value: S) -> OpaqueAttributeRef<'c, 't> {
        self.load_dialect(DialectHandle::emit_c());
        unsafe {
            OpaqueAttributeRef::from_c_api(
                mlirEmitCOpaqueAttrGet(*self.handle.borrow(), StringRef::from(value.as_ref()).to_c_api()),
                self,
            )
            .expect("invalid EmitC opaque attribute")
        }
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_cmp_predicate() {
        assert_eq!(CmpPredicate::Equal.as_str(), "eq");
        assert_eq!(CmpPredicate::NotEqual.as_str(), "ne");
        assert_eq!(CmpPredicate::LessThan.as_str(), "lt");
        assert_eq!(CmpPredicate::LessThanOrEqual.as_str(), "le");
        assert_eq!(CmpPredicate::GreaterThan.as_str(), "gt");
        assert_eq!(CmpPredicate::GreaterThanOrEqual.as_str(), "ge");
        assert_eq!(CmpPredicate::ThreeWay.as_str(), "three_way");
        assert_eq!(CmpPredicate::from_c_api(CmpPredicate::Equal.to_c_api()), Some(CmpPredicate::Equal));
        assert_eq!(CmpPredicate::from_c_api(u64::MAX), None);
    }

    #[test]
    fn test_cmp_predicate_attribute() {
        let context = Context::new();
        let attribute = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "builtin");
        assert_eq!(attribute.value(), CmpPredicate::LessThan);
        assert_eq!(attribute.type_id(), CmpPredicateAttributeRef::type_id());
    }

    #[test]
    fn test_cmp_predicate_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        let attribute_2 = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.emit_c_cmp_predicate_attribute(CmpPredicate::GreaterThan);
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_cmp_predicate_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        test_attribute_display_and_debug(attribute, "2 : i64");
    }

    #[test]
    fn test_cmp_predicate_attribute_casting() {
        let context = Context::new();
        let attribute = context.emit_c_cmp_predicate_attribute(CmpPredicate::LessThan);
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_opaque_attribute() {
        let context = Context::new();
        let attribute = context.emit_c_opaque_attribute("NULL");
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.dialect().namespace().unwrap(), "emitc");
        assert_eq!(attribute.value().as_str().unwrap(), "NULL");
        assert_eq!(attribute.type_id(), OpaqueAttributeRef::type_id());
    }

    #[test]
    fn test_opaque_attribute_equality() {
        let context = Context::new();
        let attribute_1 = context.emit_c_opaque_attribute("NULL");
        let attribute_2 = context.emit_c_opaque_attribute("NULL");
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.emit_c_opaque_attribute("nullptr");
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 = context.emit_c_opaque_attribute("NULL");
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_opaque_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.emit_c_opaque_attribute("NULL");
        test_attribute_display_and_debug(attribute, "#emitc.opaque<\"NULL\">");
    }

    #[test]
    fn test_opaque_attribute_casting() {
        let context = Context::new();
        let attribute = context.emit_c_opaque_attribute("NULL");
        test_attribute_casting(attribute);
    }
}
