use ryft_xla_sys::bindings::MlirAttribute;
use ryft_xla_sys::mlir::dialects::transform::{
    MlirTransformEnumAttribute, mlirAttributeIsATransformEnumAttr, mlirAttributeIsATransformParamOperandAttr,
    mlirTransformEnumAttrGet, mlirTransformEnumAttrGetValue, mlirTransformParamOperandAttrGet,
    mlirTransformParamOperandAttrGetIndex,
};

use crate::{Attribute, Context, DialectHandle, Error, IntegerAttributeRef, mlir_subtype_trait_impls};

/// Policy used by Transform dialect container operations when nested operations produce silenceable failures.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum FailurePropagationMode {
    /// Propagates silenceable failures from nested operations to the parent operation.
    Propagate,

    /// Suppresses silenceable failures from nested operations after the parent operation handles them.
    Suppress,
}

impl FailurePropagationMode {
    /// Returns the integer value used by the MLIR enum attribute.
    pub fn value(&self) -> u32 {
        match self {
            Self::Propagate => 1,
            Self::Suppress => 2,
        }
    }

    /// Returns the textual spelling used in MLIR assembly.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Propagate => "propagate",
            Self::Suppress => "suppress",
        }
    }

    /// Returns the enum value corresponding to the provided MLIR integer value.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            1 => Some(Self::Propagate),
            2 => Some(Self::Suppress),
            _ => None,
        }
    }
}

/// Transform dialect [`Attribute`] that stores a [`FailurePropagationMode`].
#[derive(Copy, Clone)]
pub struct FailurePropagationModeAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl FailurePropagationModeAttributeRef<'_, '_> {
    /// Returns the enum value stored in this attribute.
    pub fn value(&self) -> Result<FailurePropagationMode, Error> {
        let value = unsafe {
            mlirTransformEnumAttrGetValue(
                self.handle,
                MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE,
            )
        };
        FailurePropagationMode::from_value(value)
            .ok_or_else(|| Error::invalid_argument("invalid Transform failure propagation mode attribute"))
    }
}

impl<'c, 't> Attribute<'c, 't> for FailurePropagationModeAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsATransformEnumAttr(
                    handle,
                    MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE,
                )
            }
        {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(FailurePropagationModeAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Signed integer comparison predicate used by `transform.match.param.cmpi`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MatchCmpIPredicate {
    /// Tests whether the left parameter is equal to the right parameter.
    Equal,

    /// Tests whether the left parameter is not equal to the right parameter.
    NotEqual,

    /// Tests whether the left parameter is less than the right parameter.
    LessThan,

    /// Tests whether the left parameter is less than or equal to the right parameter.
    LessThanOrEqual,

    /// Tests whether the left parameter is greater than the right parameter.
    GreaterThan,

    /// Tests whether the left parameter is greater than or equal to the right parameter.
    GreaterThanOrEqual,
}

impl MatchCmpIPredicate {
    /// Returns the integer value used by the MLIR enum attribute.
    pub fn value(&self) -> u32 {
        match self {
            Self::Equal => 0,
            Self::NotEqual => 1,
            Self::LessThan => 2,
            Self::LessThanOrEqual => 3,
            Self::GreaterThan => 4,
            Self::GreaterThanOrEqual => 5,
        }
    }

    /// Returns the textual spelling used in MLIR assembly.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Equal => "eq",
            Self::NotEqual => "ne",
            Self::LessThan => "lt",
            Self::LessThanOrEqual => "le",
            Self::GreaterThan => "gt",
            Self::GreaterThanOrEqual => "ge",
        }
    }

    /// Returns the enum value corresponding to the provided MLIR integer value.
    pub fn from_value(value: u32) -> Option<Self> {
        match value {
            0 => Some(Self::Equal),
            1 => Some(Self::NotEqual),
            2 => Some(Self::LessThan),
            3 => Some(Self::LessThanOrEqual),
            4 => Some(Self::GreaterThan),
            5 => Some(Self::GreaterThanOrEqual),
            _ => None,
        }
    }
}

/// Transform dialect [`Attribute`] that stores a [`MatchCmpIPredicate`].
#[derive(Copy, Clone)]
pub struct MatchCmpIPredicateAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl MatchCmpIPredicateAttributeRef<'_, '_> {
    /// Returns the enum value stored in this attribute.
    pub fn value(&self) -> Result<MatchCmpIPredicate, Error> {
        let value = unsafe {
            mlirTransformEnumAttrGetValue(
                self.handle,
                MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE,
            )
        };
        MatchCmpIPredicate::from_value(value)
            .ok_or_else(|| Error::invalid_argument("invalid Transform match comparison predicate attribute"))
    }
}

impl<'c, 't> Attribute<'c, 't> for MatchCmpIPredicateAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null()
            && unsafe {
                mlirAttributeIsATransformEnumAttr(
                    handle,
                    MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE,
                )
            }
        {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(MatchCmpIPredicateAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

/// Transform dialect [`Attribute`] that refers to a specific parameter operand by index.
///
/// Refer to the [official MLIR documentation](https://mlir.llvm.org/docs/Dialects/Transform/#paramoperandattr)
/// for more information.
#[derive(Copy, Clone)]
pub struct ParamOperandAttributeRef<'c, 't> {
    /// Handle that represents this [`Attribute`] in the MLIR C API.
    handle: MlirAttribute,

    /// [`Context`] that owns this [`Attribute`].
    context: &'c Context<'t>,
}

impl<'c, 't> ParamOperandAttributeRef<'c, 't> {
    /// Returns the integer attribute that stores the referenced parameter operand index.
    pub fn index(&self) -> Result<IntegerAttributeRef<'c, 't>, Error> {
        unsafe {
            IntegerAttributeRef::from_c_api(mlirTransformParamOperandAttrGetIndex(self.handle), self.context)
                .map_err(|_| Error::invalid_argument("invalid `#transform.param_operand` index"))
        }
    }
}

impl<'c, 't> Attribute<'c, 't> for ParamOperandAttributeRef<'c, 't> {
    unsafe fn from_c_api(handle: MlirAttribute, context: &'c Context<'t>) -> Result<Self, Error> {
        if !handle.ptr.is_null() && unsafe { mlirAttributeIsATransformParamOperandAttr(handle) } {
            Ok(Self { handle, context })
        } else {
            Err(Error::invalid_argument("expected MLIR attribute handle"))
        }
    }

    unsafe fn to_c_api(&self) -> MlirAttribute {
        self.handle
    }

    fn context(&self) -> &'c Context<'t> {
        self.context
    }
}

mlir_subtype_trait_impls!(ParamOperandAttributeRef<'c, 't> as Attribute, mlir_type = Attribute);

impl<'t> Context<'t> {
    /// Creates a new [`FailurePropagationModeAttributeRef`] owned by this [`Context`].
    pub fn transform_failure_propagation_mode_attribute<'c>(
        &'c self,
        value: FailurePropagationMode,
    ) -> Result<FailurePropagationModeAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::transform()?)?;
        Ok(unsafe {
            FailurePropagationModeAttributeRef {
                handle: mlirTransformEnumAttrGet(
                    *self.handle.borrow_mut(),
                    MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_FAILURE_PROPAGATION_MODE,
                    value.value(),
                ),
                context: self,
            }
        })
    }

    /// Creates a new [`MatchCmpIPredicateAttributeRef`] owned by this [`Context`].
    pub fn transform_match_cmp_i_predicate_attribute<'c>(
        &'c self,
        value: MatchCmpIPredicate,
    ) -> Result<MatchCmpIPredicateAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::transform()?)?;
        Ok(unsafe {
            MatchCmpIPredicateAttributeRef {
                handle: mlirTransformEnumAttrGet(
                    *self.handle.borrow_mut(),
                    MlirTransformEnumAttribute::MLIR_TRANSFORM_ENUM_ATTRIBUTE_MATCH_CMP_I_PREDICATE,
                    value.value(),
                ),
                context: self,
            }
        })
    }

    /// Creates a new [`ParamOperandAttributeRef`] owned by this [`Context`].
    pub fn transform_param_operand_attribute<'c>(
        &'c self,
        index: IntegerAttributeRef<'c, 't>,
    ) -> Result<ParamOperandAttributeRef<'c, 't>, Error> {
        self.load_dialect(DialectHandle::transform()?)?;
        Ok(unsafe {
            ParamOperandAttributeRef {
                handle: mlirTransformParamOperandAttrGet(*self.handle.borrow_mut(), index.to_c_api()),
                context: self,
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::Attribute;
    use crate::attributes::tests::{test_attribute_casting, test_attribute_display_and_debug};

    use super::*;

    #[test]
    fn test_failure_propagation_mode_attribute() {
        let context = Context::new();
        let attribute =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value().unwrap(), FailurePropagationMode::Propagate);
        assert_eq!(FailurePropagationMode::Propagate.value(), 1);
        assert_eq!(FailurePropagationMode::Propagate.as_str(), "propagate");
        assert_eq!(FailurePropagationMode::from_value(1), Some(FailurePropagationMode::Propagate));
    }

    #[test]
    fn test_failure_propagation_mode_attribute_equality() {
        let context = Context::new();
        let attribute_1 =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        let attribute_2 =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Suppress).unwrap();
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_failure_propagation_mode_attribute_display_and_debug() {
        let context = Context::new();
        let attribute =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        test_attribute_display_and_debug(attribute, "1 : i32");
    }

    #[test]
    fn test_failure_propagation_mode_attribute_parsing() {
        let context = Context::new();
        let attribute =
            context.parse_attribute("1 : i32").unwrap().cast::<FailurePropagationModeAttributeRef>().unwrap();
        assert_eq!(attribute.value().unwrap(), FailurePropagationMode::Propagate);

        let attribute = context.parse_attribute("0 : i32").unwrap();
        assert_eq!(attribute.cast::<FailurePropagationModeAttributeRef>(), None);
    }

    #[test]
    fn test_failure_propagation_mode_attribute_casting() {
        let context = Context::new();
        let attribute =
            context.transform_failure_propagation_mode_attribute(FailurePropagationMode::Propagate).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.value().unwrap(), MatchCmpIPredicate::LessThanOrEqual);
        assert_eq!(MatchCmpIPredicate::LessThanOrEqual.value(), 3);
        assert_eq!(MatchCmpIPredicate::LessThanOrEqual.as_str(), "le");
        assert_eq!(MatchCmpIPredicate::from_value(3), Some(MatchCmpIPredicate::LessThanOrEqual));
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_equality() {
        let context = Context::new();
        let attribute_1 =
            context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        let attribute_2 =
            context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        assert_eq!(attribute_1, attribute_2);

        let attribute_2 = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::GreaterThan).unwrap();
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let attribute_2 =
            context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_display_and_debug() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        test_attribute_display_and_debug(attribute, "3 : i32");
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_parsing() {
        let context = Context::new();
        let attribute = context.parse_attribute("3 : i32").unwrap().cast::<MatchCmpIPredicateAttributeRef>().unwrap();
        assert_eq!(attribute.value().unwrap(), MatchCmpIPredicate::LessThanOrEqual);

        let attribute = context.parse_attribute("6 : i32").unwrap();
        assert_eq!(attribute.cast::<MatchCmpIPredicateAttributeRef>(), None);
    }

    #[test]
    fn test_match_cmp_i_predicate_attribute_casting() {
        let context = Context::new();
        let attribute = context.transform_match_cmp_i_predicate_attribute(MatchCmpIPredicate::LessThanOrEqual).unwrap();
        test_attribute_casting(attribute);
    }

    #[test]
    fn test_param_operand_attribute() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index).unwrap();
        assert_eq!(&context, attribute.context());
        assert_eq!(attribute.index().unwrap(), index);
    }

    #[test]
    fn test_param_operand_attribute_equality() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute_1 = context.transform_param_operand_attribute(index).unwrap();
        let attribute_2 = context.transform_param_operand_attribute(index).unwrap();
        assert_eq!(attribute_1, attribute_2);

        let index = context.integer_attribute(context.signless_integer_type(64), 8);
        let attribute_2 = context.transform_param_operand_attribute(index).unwrap();
        assert_ne!(attribute_1, attribute_2);

        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute_2 = context.transform_param_operand_attribute(index).unwrap();
        assert_ne!(attribute_1, attribute_2);
    }

    #[test]
    fn test_param_operand_attribute_display_and_debug() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index).unwrap();
        test_attribute_display_and_debug(attribute, "#transform.param_operand<index = 7 : i64>");
    }

    #[test]
    fn test_param_operand_attribute_parsing() {
        let context = Context::new();
        context.load_dialect(DialectHandle::transform().unwrap()).unwrap();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index).unwrap();
        assert_eq!(
            context
                .parse_attribute("#transform.param_operand<index = 7 : i64>")
                .unwrap()
                .cast::<ParamOperandAttributeRef>()
                .unwrap(),
            attribute
        );
    }

    #[test]
    fn test_param_operand_attribute_casting() {
        let context = Context::new();
        let index = context.integer_attribute(context.signless_integer_type(64), 7);
        let attribute = context.transform_param_operand_attribute(index).unwrap();
        test_attribute_casting(attribute);
    }
}
