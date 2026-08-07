pub mod constant;
pub mod fill;
pub mod iota;
pub mod one;
pub mod one_like;
pub mod zero;
pub mod zero_like;

pub use constant::{CONSTANT_OPERATION_NAME, Constant, ConstantOperation};
pub use fill::Fill;
pub use iota::{IOTA_OPERATION_NAME, Iota, IotaOperation};
pub use one::{ONE_OPERATION_NAME, One, OneOperation};
pub use one_like::{ONE_LIKE_OPERATION_NAME, OneLike, OneLikeOperation};
pub use zero::{ZERO_OPERATION_NAME, Zero, ZeroOperation, ZeroOperationProvider};
pub use zero_like::{ZERO_LIKE_OPERATION_NAME, ZeroLike, ZeroLikeOperation};

use crate::arrays::{ArrayIrType, ArrayType, Dimension, DimensionType};
use crate::programs::{RegionInterface, Type, TypeError, TypeIdentityPosition};

/// Rejects a nullary constructor output [`Type`] that carries an ungrounded [`TypeIdentity`](crate::TypeIdentity)
/// reference. A reference-position identity in a constructed-from-nothing type names a runtime quantity that no operand
/// supplies. Such outputs must use a mixed constructor that consumes explicit dimension operands. Definition-position
/// identities remain valid because the constructed value establishes them itself.
pub(crate) fn check_constructor_type_has_no_identity_references<T: Type>(
    name: &str,
    r#type: &T,
) -> Result<(), TypeError> {
    match r#type.identities().find(|(position, _)| *position == TypeIdentityPosition::Reference) {
        Some((_, reference)) => Err(TypeError::invalid(format!(
            "'{}' cannot construct type {} without operands because it references identity {}",
            name, r#type, reference,
        ))),
        None => Ok(()),
    }
}

/// Infers the output type of one mixed [`ArrayIrType`] constructor whose stored [`ArrayType`] is the
/// complete output type. The constructor consumes one first-class dimension operand per *dynamic* dimension
/// of its stored shape, in axis order, and each operand's [`DimensionType`] must define exactly the
/// [`DimensionVariable`](crate::DimensionVariable) named by the corresponding output axis. Static axes remain ordinary
/// stored type metadata and consume no operands, This is deliberately narrower than the mixed reshape/broadcast
/// contract, which derives *every* output axis from an operand (including exact constants): a constructor's static axes
/// have no input geometry to relate to, so passing them as operands would only grow the interpreted representation. A
/// stored type with no dynamic axes is rejected here because identity-free construction has one canonical encoding: the
/// homogeneous nullary constructor inside the array member family.
pub(crate) fn infer_dynamic_constructor_output_types(
    name: &str,
    r#type: &ArrayType,
    input_types: &[ArrayIrType],
    region_interfaces: &[RegionInterface<ArrayIrType>],
) -> Result<Vec<ArrayIrType>, TypeError> {
    if !region_interfaces.is_empty() {
        return Err(TypeError::invalid(format!("'{}' expects no regions but got {}", name, region_interfaces.len())));
    }
    let variables = r#type.shape().dimensions().iter().filter_map(Dimension::variable).collect::<Vec<_>>();
    if variables.is_empty() {
        return Err(TypeError::invalid(format!(
            "'{}' with static output type {} has no dynamic dimensions; use the homogeneous nullary \
            constructor instead",
            name, r#type,
        )));
    }
    if input_types.len() != variables.len() {
        return Err(TypeError::invalid(format!(
            "'{}' expects one dimension operand per dynamic output dimension ({}) but got {} operands",
            name,
            variables.len(),
            input_types.len(),
        )));
    }
    for (index, (input_type, variable)) in input_types.iter().zip(variables).enumerate() {
        let dimension_type = <&DimensionType>::try_from(input_type).map_err(|_| {
            TypeError::invalid(format!("'{name}' operand {index} must be a dimension but has type {input_type}"))
        })?;
        if dimension_type.variable() != variable {
            let required_type = DimensionType::new(variable.clone());
            return Err(TypeError::invalid(format!(
                "'{name}' operand {index} has type {dimension_type} but the output shape requires {required_type}",
            )));
        }
    }
    Ok(vec![ArrayIrType::Array(r#type.clone())])
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DataType, DimensionBounds, DimensionVariable, Shape};
    use crate::programs::TypeError;

    use super::*;

    #[test]
    fn test_infer_dynamic_constructor_output_types() {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(8)).unwrap());
        let dynamic_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(3)]));

        // One identity-validated dimension operand per dynamic axis, in axis order; static axes consume no operands.
        assert_eq!(
            infer_dynamic_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Dimension(DimensionType::new(rows.clone()))],
                &[],
            ),
            Ok(vec![ArrayIrType::Array(dynamic_type.clone())]),
        );
        assert_eq!(
            infer_dynamic_constructor_output_types("zero", &dynamic_type, &[], &[]),
            Err(TypeError::invalid(
                "'zero' expects one dimension operand per dynamic output dimension (1) but got 0 operands",
            )),
        );
        let other = DimensionVariable::new("other", DimensionBounds::non_negative(Some(8)).unwrap());
        assert_eq!(
            infer_dynamic_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Dimension(DimensionType::new(other))],
                &[],
            ),
            Err(TypeError::invalid(
                "'zero' operand 0 has type dimension<other ∈ [0, 8)> but the output shape requires \
                 dimension<rows ∈ [0, 8)>",
            )),
        );
        assert_eq!(
            infer_dynamic_constructor_output_types(
                "zero",
                &dynamic_type,
                &[ArrayIrType::Array(ArrayType::scalar(DataType::I64))],
                &[],
            ),
            Err(TypeError::invalid("'zero' operand 0 must be a dimension but has type i64[]")),
        );

        // Reference-free static construction has one canonical encoding: the homogeneous nullary constructor.
        assert_eq!(
            infer_dynamic_constructor_output_types("zero", &ArrayType::scalar(DataType::F32), &[], &[]),
            Err(TypeError::invalid(
                "'zero' with static output type f32[] has no dynamic dimensions; use the homogeneous nullary \
                 constructor instead",
            )),
        );
    }
}
