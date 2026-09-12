//! Constant values and type-driven array constructors.
//!
//! [`Zero`], [`One`], [`Iota`], and [`Fill`] construct values from types with statically known extents. Their
//! dynamic counterparts accept explicit first-class dimension operands, allowing the same constructors to be used
//! with runtime shapes. [`Constant`] carries an existing value, while [`ZeroLike`] and [`OneLike`] obtain their
//! geometry from an exemplar and therefore need no separate dynamic capability.

use crate::arrays::{ArrayIrType, ArrayType, Dimension, DimensionType};
use crate::programs::{ProgramError, RegionInterface, Type, TypeError, TypeIdentityPosition, Typed};

/// Implements the mixed [`ArrayIrType`] [`MemberOperation`](crate::MemberOperation) boundary for an array constant
/// constructor. The generated implementation treats the operation's stored [`ArrayType`] as the complete output type
/// and requires one first-class [`DimensionType`] input per dynamic axis, in axis order. Static axes remain stored
/// metadata and consume no inputs. This is specialized to constant constructors because they have no data inputs
/// or regions and derive their complete result solely from stored type metadata plus dynamic extent inputs.
macro_rules! impl_member_operation_for_array_ir_constant_operation {
    ($operation:ty $(, $validate:ident)?) => {
        impl $crate::programs::MemberOperation<$crate::arrays::ArrayIrType> for $operation {
            #[inline]
            fn infer_parent_region_input_types(
                &self,
                _input_types: &[$crate::arrays::ArrayIrType],
                region_interfaces: &[$crate::programs::RegionInterface<$crate::arrays::ArrayIrType>],
            ) -> Result<Vec<Option<Vec<$crate::arrays::ArrayIrType>>>, $crate::programs::TypeError> {
                Ok(vec![None; region_interfaces.len()])
            }

            #[inline]
            fn infer_parent_output_types(
                &self,
                input_types: &[$crate::arrays::ArrayIrType],
                region_interfaces: &[$crate::programs::RegionInterface<$crate::arrays::ArrayIrType>],
            ) -> Result<Vec<$crate::arrays::ArrayIrType>, $crate::programs::TypeError> {
                $(self.r#type().$validate()?;)?
                $crate::operations::constants::infer_array_ir_constant_constructor_output_types(
                    self.name(),
                    self.r#type(),
                    input_types,
                    region_interfaces,
                )
            }

            #[inline]
            fn rename_parent_type_identities(
                &self,
                renaming: &$crate::programs::TypeIdentityRenaming<$crate::arrays::DimensionVariable>,
            ) -> Result<Self, $crate::programs::TypeError> {
                self.rename_type_identities(renaming)
            }
        }
    };
}

/// Implements mixed [`MemberInterpretableOperation`](crate::MemberInterpretableOperation) semantics for an array
/// constant constructor. The generated implementation projects each explicit first-class dimension input to its
/// runtime extent, validates that extent against the corresponding dynamic axis's declared bounds, and replaces the
/// dynamic axes with static extents before invoking the constructor's eager capability. This runtime concretization is
/// required because replay deliberately does not refine types stored in operation payloads; concrete shape information
/// reaches mixed constructors only through their Static Single Assignment (SSA) dimension inputs. The final argument
/// names the generated eager context, concrete output type, and operation payload for use in an operation-specific
/// expression that returns the constructed projected value.
macro_rules! impl_member_interpretable_operation_for_array_ir_constant_operation {
    // Implements one array-IR constant constructor from its capability and operation-specific eager expression.
    (
        $operation:ty,
        $capability:ident,
        |$context:ident, $output_type:ident, $operation_value:ident| $interpretation:expr $(,)?
    ) => {
        impl<C> $crate::interpretation::MemberInterpretableOperation<C> for $operation
        where
            C: $crate::contexts::Domain<
                    Type = $crate::arrays::ArrayIrType,
                    Value: $crate::programs::ValueProjection<
                        $crate::arrays::ArrayType,
                        Projected: $crate::programs::Value<Type = $crate::arrays::ArrayType>,
                    > + $crate::programs::ValueProjection<
                        $crate::arrays::DimensionType,
                        Projected = $crate::arrays::DimensionValue,
                    >,
                >,
            $crate::contexts::EagerContext<
                <C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::Projected,
                $crate::arrays::ArrayOperation<
                    <C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::Projected,
                >,
            >: $capability<<C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::Projected>,
        {
            fn interpret_in_parent<D: $crate::interpretation::InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                inputs: &[C::Value],
            ) -> Result<Vec<C::Value>, $crate::programs::ProgramError> {
                let expected = self
                    .r#type()
                    .shape()
                    .dimensions()
                    .iter()
                    .filter(|dimension| matches!(dimension, $crate::arrays::Dimension::Dynamic(_)))
                    .count();
                let mut extents = inputs.iter();
                let mut refinements = $crate::arrays::ArrayTypeRefinements::default();
                let dimensions = self
                    .r#type()
                    .shape()
                    .dimensions()
                    .iter()
                    .map(|dimension| match dimension {
                        $crate::arrays::Dimension::Static(extent) => Ok($crate::arrays::Dimension::Static(*extent)),
                        $crate::arrays::Dimension::Dynamic(variable) => {
                            let extent = extents.next().ok_or($crate::programs::ProgramError::InvalidInputCount {
                                expected,
                                actual: inputs.len(),
                            })?;

                            let extent = <C::Value as $crate::programs::ValueProjection<
                                $crate::arrays::DimensionType,
                            >>::into_projected(extent.clone())?;

                            // Eager binds skip inference and intermediate results skip boundary refinement checks, so
                            // validate each runtime extent against the stored output axis before allocating. The input
                            // may use the equivalent dimension identity supplied by the calling program, so its
                            // identity does not need to exactly match the one stored on the operation.
                            if !variable.bounds().contains(extent.extent()) {
                                return Err($crate::arrays::DimensionError::BindingOutOfBounds {
                                    variable: variable.to_string(),
                                    value: extent.extent(),
                                    bounds: variable.bounds(),
                                }
                                .into());
                            }

                            // Repeated stored identities must denote one extent even when replay renames inputs.
                            refinements.bind(variable, extent.extent())?;
                            Ok($crate::arrays::Dimension::Static(extent.extent()))
                        }
                    })
                    .collect::<Result<Vec<_>, $crate::programs::ProgramError>>()?;
                if extents.next().is_some() {
                    return Err($crate::programs::ProgramError::InvalidInputCount { expected, actual: inputs.len() });
                }

                let $output_type = self.r#type().clone().with_shape($crate::arrays::Shape::new(dimensions));
                let $context = $crate::contexts::EagerContext::<
                    <C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::Projected,
                    $crate::arrays::ArrayOperation<
                        <C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::Projected,
                    >,
                >::new();
                let $operation_value = self;
                let output = $interpretation?;
                Ok(vec![<C::Value as $crate::programs::ValueProjection<$crate::arrays::ArrayType>>::from_projected(
                    output,
                )])
            }
        }
    };
}

/// Rejects a nullary constructor output [`Type`] that carries an ungrounded [`TypeIdentity`](crate::TypeIdentity)
/// reference. A reference-position identity in a constructed-from-nothing type names a runtime quantity that no input
/// supplies. Such outputs must use a mixed constructor that consumes explicit dimension inputs. Definition-position
/// identities remain valid because the constructed value establishes them itself.
pub(crate) fn check_constructor_type_has_no_identity_references<T: Type>(
    name: &str,
    r#type: &T,
) -> Result<(), TypeError> {
    match r#type.identities().find(|(position, _)| *position == TypeIdentityPosition::Reference) {
        Some((_, reference)) => Err(TypeError::invalid(format!(
            "`{}` cannot construct type {} without operands because it references identity {}",
            name, r#type, reference,
        ))),
        None => Ok(()),
    }
}

/// Infers the output type of one mixed [`ArrayIrType`] constant constructor whose stored [`ArrayType`] is the complete
/// output type. The constructor consumes one first-class dimension input per _dynamic_ dimension of its stored shape,
/// in axis order, and each input's [`DimensionType`] must define exactly the
/// [`DimensionVariable`](crate::DimensionVariable) named by the corresponding output axis. Static axes remain ordinary
/// stored type metadata and consume no inputs. This is deliberately narrower than the mixed reshape/broadcast
/// contract, which derives *every* output axis from an input (including exact constants): a constructor's static axes
/// have no input geometry to relate to, so passing them as inputs would only grow the interpreted representation. A
/// stored type with no dynamic axes is valid with no inputs, although canonical operation-family lifts prefer the
/// equivalent homogeneous nullary constructor inside the array member family. Dynamic axes retain their declared
/// identities, including singleton-bounded axes; the explicit dimension inputs bind those identities in the result.
pub(crate) fn infer_array_ir_constant_constructor_output_types(
    name: &str,
    r#type: &ArrayType,
    input_types: &[ArrayIrType],
    region_interfaces: &[RegionInterface<ArrayIrType>],
) -> Result<Vec<ArrayIrType>, TypeError> {
    if !region_interfaces.is_empty() {
        return Err(TypeError::invalid(format!("`{}` expects no regions but got {}", name, region_interfaces.len())));
    }
    let variables = r#type.shape().dimensions().iter().filter_map(Dimension::variable).collect::<Vec<_>>();
    if input_types.len() != variables.len() {
        return Err(TypeError::invalid(format!(
            "`{}` expects one dimension operand per dynamic output dimension ({}) but got {} operands",
            name,
            variables.len(),
            input_types.len(),
        )));
    }
    for (index, (input_type, variable)) in input_types.iter().zip(variables).enumerate() {
        let dimension_type = <&DimensionType>::try_from(input_type).map_err(|_| {
            TypeError::invalid(format!("`{name}` operand {index} must be a dimension but has type {input_type}"))
        })?;
        if dimension_type.variable() != variable {
            let required_type = DimensionType::new(variable.clone());
            return Err(TypeError::invalid(format!(
                "`{name}` operand {index} has type {dimension_type} but the output shape requires {required_type}",
            )));
        }
    }

    // Exact input-variable validation above establishes the declared identity, even when its bounds imply one size.
    Ok(vec![ArrayIrType::Array(r#type.clone())])
}

/// Checks that the provided dimension inputs agree with the symbolic shape declared by `type` before a dynamic
/// constructor binds any operations. An [`ArrayType`] can describe a shape without knowing all its extents as it stores
/// static sizes directly and represents each dynamic axis by a [`DimensionVariable`](crate::DimensionVariable),
/// including its identity and bounds. The inputs supply values for those dynamic axes at runtime.
///
/// This function checks that there is exactly one input per dynamic axis, that every input has a [`DimensionType`],
/// and that its dimension variable matches the corresponding output axis. Static axes consume no inputs. It inspects
/// input types only. It neither reads runtime extents nor constructs an array. Concrete values and their extent bounds
/// are checked when created or interpreted. This also gives eager capability calls the same shape checks as staged
/// constructors, without relying on a staging context to run type inference.
///
/// # Example
///
/// For an `F32` output with shape `[N, 2, M]`, `dimensions` must contain two values with dimension types `N` and `M`,
/// in that order. If their runtime extents are `3` and `4`, the constructor produces an array of shape `[3, 2, 4]`.
/// An array input, a missing input, or inputs ordered as `[M, N]` fail validation. A shape `[N, N]` requires two inputs
/// of type `N` and thus callers must supply the same extent for both occurrences. A fully static shape `[3, 2]`
/// requires an empty input slice.
///
/// # Parameters
///
///   - `name`: Operation name included in validation errors.
///   - `type`: Declared output type, including static extents and dynamic dimension variables.
///   - `dimensions`: One dimension input per dynamic output axis, in shape order.
pub(crate) fn validate_dynamic_constant_dimensions<V: Typed<Type = ArrayIrType>>(
    name: &str,
    r#type: &ArrayType,
    dimensions: &[V],
) -> Result<(), ProgramError> {
    let input_types = dimensions.iter().map(|dimension| dimension.r#type().into_owned()).collect::<Vec<_>>();
    infer_array_ir_constant_constructor_output_types(name, r#type, &input_types, &[])?;
    Ok(())
}

pub mod constant;
pub mod fill;
pub mod iota;
pub mod one;
pub mod one_like;
pub mod zero;
pub mod zero_like;

pub use constant::{CONSTANT_OPERATION_NAME, Constant, ConstantOperation, DimensionConstant};
pub use fill::{DynamicFill, Fill};
pub use iota::{DynamicIota, IOTA_OPERATION_NAME, Iota, IotaOperation};
pub use one::{DynamicOne, ONE_OPERATION_NAME, One, OneOperation};
pub use one_like::{ONE_LIKE_OPERATION_NAME, OneLike, OneLikeOperation};
pub use zero::{DynamicZero, ZERO_OPERATION_NAME, Zero, ZeroOperation};
pub use zero_like::{ZERO_LIKE_OPERATION_NAME, ZeroLike, ZeroLikeOperation};

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, DataType, DimensionBounds, DimensionError, DimensionValue,
        DimensionVariable, Shape,
    };
    use crate::contexts::EagerContext;
    use crate::interpretation::MemberInterpretableOperation;
    use crate::programs::{EmptyRegionDriver, TypeError};

    use super::*;

    #[test]
    fn test_array_ir_constant_operation_interpretation_repeated_dimensions() {
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let variable = DimensionVariable::new("size", DimensionBounds::unbounded());
        let operation = ZeroOperation::new(ArrayType::new(
            DataType::F32,
            Shape::new(vec![variable.clone().into(), variable.clone().into()]),
        ));

        // Replay may rename an input identity; consistency is enforced against the stored output identity.
        let renamed = DimensionType::new(DimensionVariable::new("renamed", DimensionBounds::unbounded()));
        let two = ArrayIrValue::Dimension(DimensionValue::new(renamed.clone(), 2).unwrap());
        let three = ArrayIrValue::Dimension(DimensionValue::new(renamed, 3).unwrap());
        assert_eq!(
            operation.interpret_in_parent(&context, &EmptyRegionDriver, &[two.clone(), two.clone()]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[0.0f32; 4]).unwrap()
            )]),
        );
        assert_eq!(
            operation.interpret_in_parent(&context, &EmptyRegionDriver, &[two.clone(), three.clone()]),
            Err(ProgramError::Type(
                DimensionError::InputDimensionMismatch { dimension: "size".to_owned(), expected: 2, actual: 3 }.into()
            )),
        );

        // Diagnostic names do not establish identity: separately created variables may have different sizes.
        let other = DimensionVariable::new("size", DimensionBounds::unbounded());
        let operation =
            ZeroOperation::new(ArrayType::new(DataType::F32, Shape::new(vec![variable.into(), other.into()])));
        assert_eq!(
            operation.interpret_in_parent(&context, &EmptyRegionDriver, &[two, three]),
            Ok(vec![ArrayIrValue::Array(
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 3]), &[0.0f32; 6]).unwrap()
            )]),
        );
    }

    #[test]
    fn test_infer_array_ir_constant_constructor_output_types() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]));

        // One identity-validated dimension operand per dynamic axis, in axis order; static axes consume no operands.
        assert_eq!(
            infer_array_ir_constant_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Dimension(DimensionType::new(rows.clone()))],
                &[],
            ),
            Ok(vec![ArrayIrType::Array(dynamic_type.clone())]),
        );
        assert_eq!(
            infer_array_ir_constant_constructor_output_types("zero", &dynamic_type, &[], &[]),
            Err(TypeError::invalid(
                "`zero` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        let other = DimensionVariable::new("other", DimensionBounds::non_negative(Some(8)).unwrap());
        assert_eq!(
            infer_array_ir_constant_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Dimension(DimensionType::new(other))],
                &[],
            ),
            Err(TypeError::invalid(
                "`zero` operand 0 has type dimension<other ∈ [0, 8)> but the output shape requires \
                 dimension<rows ∈ [0, 8)>",
            )),
        );
        assert_eq!(
            infer_array_ir_constant_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Array(ArrayType::scalar(DataType::I64))],
                &[],
            ),
            Err(TypeError::invalid("`zero` operand 0 must be a dimension but has type i64[]")),
        );

        // Static mixed construction is valid with no operands, while unexpected operands remain invalid.
        let static_type = ArrayType::scalar(DataType::F32);
        assert_eq!(
            infer_array_ir_constant_constructor_output_types("zero", &static_type, &[], &[]),
            Ok(vec![ArrayIrType::Array(static_type.clone())]),
        );
        assert_eq!(
            infer_array_ir_constant_constructor_output_types(
                "zero",
                &static_type,
                &[ArrayIrType::Array(ArrayType::scalar(DataType::I64))],
                &[],
            ),
            Err(TypeError::invalid(
                "`zero` expects one dimension operand per dynamic output dimension (0) but got 1 operands",
            )),
        );
    }

    #[test]
    fn test_infer_array_ir_constant_constructor_output_types_singleton_dimensions() {
        let dimension = DimensionType::new(DimensionVariable::new("size", DimensionBounds::new(3, Some(4)).unwrap()));
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![dimension.variable().clone().into(), 2.into()]));
        assert_eq!(
            infer_array_ir_constant_constructor_output_types("zero", &r#type, &[dimension.into()], &[]),
            Ok(vec![ArrayIrType::Array(r#type.clone())]),
        );

        // The declared singleton identity still requires its explicit dimension input.
        assert_eq!(
            infer_array_ir_constant_constructor_output_types("zero", &r#type, &[], &[]),
            Err(TypeError::invalid(
                "`zero` expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
    }

    #[test]
    fn test_validate_dynamic_constant_dimensions() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(8)).unwrap());
        let output_type =
            ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), 2.into(), columns.clone().into()]));
        let row_extent =
            ArrayIrValue::<Array>::Dimension(DimensionValue::new(DimensionType::new(rows.clone()), 3).unwrap());
        let column_extent =
            ArrayIrValue::<Array>::Dimension(DimensionValue::new(DimensionType::new(columns), 4).unwrap());

        // Operands correspond only to dynamic axes and must follow their order in the stored shape.
        assert_eq!(
            validate_dynamic_constant_dimensions("fill", &output_type, &[row_extent.clone(), column_extent.clone()]),
            Ok(()),
        );
        assert_eq!(
            validate_dynamic_constant_dimensions("fill", &output_type, std::slice::from_ref(&row_extent)),
            Err(ProgramError::Type(TypeError::invalid(
                "`fill` expects one dimension operand per dynamic output dimension (2) but got 1 operands",
            ))),
        );
        assert_eq!(
            validate_dynamic_constant_dimensions("fill", &output_type, &[column_extent.clone(), row_extent.clone()]),
            Err(ProgramError::Type(TypeError::invalid(
                "`fill` operand 0 has type dimension<columns ∈ [0, 8)> but the output shape requires \
                 dimension<rows ∈ [0, 8)>",
            ))),
        );
        assert_eq!(
            validate_dynamic_constant_dimensions(
                "fill",
                &output_type,
                &[ArrayIrValue::Array(Array::scalar(3i64).unwrap()), column_extent],
            ),
            Err(ProgramError::Type(TypeError::invalid("`fill` operand 0 must be a dimension but has type i64[]"))),
        );

        // Repeated dynamic axes consume repeated operands; fully static shapes consume none.
        let repeated_type = ArrayType::new(DataType::F32, Shape::new(vec![rows.clone().into(), rows.into()]));
        assert_eq!(
            validate_dynamic_constant_dimensions("fill", &repeated_type, &[row_extent.clone(), row_extent.clone()]),
            Ok(()),
        );
        let static_type = ArrayType::new_static(DataType::F32, [3, 2]);
        assert_eq!(validate_dynamic_constant_dimensions::<ArrayIrValue<Array>>("fill", &static_type, &[]), Ok(()));
        assert_eq!(
            validate_dynamic_constant_dimensions("fill", &static_type, &[row_extent]),
            Err(ProgramError::Type(TypeError::invalid(
                "`fill` expects one dimension operand per dynamic output dimension (0) but got 1 operands",
            ))),
        );
    }
}
