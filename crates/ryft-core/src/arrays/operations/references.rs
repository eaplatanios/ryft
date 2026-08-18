//! Eager [`ArrayIrValue`] implementations of the reference capability traits, delegating type checks to the
//! canonical operation inference rules before touching any holder.

use crate::arrays::ir::ArrayIrValue;
use crate::arrays::types::arrays::ArrayType;
use crate::operations::{
    Add, FreezeReference, FreezeReferenceOperation, NewReference, NewReferenceOperation, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceRead, ReferenceReadOperation, ReferenceSwap, ReferenceSwapOperation,
};
use crate::programs::{Operation, ProgramError, Reference, ReferenceType, Typed, Value, ValueProjection};

// TODO(eaplatanios): Review this module.

impl<A: Value<Type = ArrayType>> NewReference for ArrayIrValue<A> {
    fn new_reference(&self) -> Result<Self, ProgramError> {
        NewReferenceOperation.infer_output_types(std::slice::from_ref(&self.r#type().into_owned()), &[])?;
        let value = <Self as ValueProjection<ArrayType>>::projected(self)?.clone();
        Ok(Self::Reference(Reference::new(value)))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceRead for ArrayIrValue<A> {
    fn read(&self) -> Result<Self, ProgramError> {
        ReferenceReadOperation.infer_output_types(std::slice::from_ref(&self.r#type().into_owned()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.read().map_err(ProgramError::custom)?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceSwap for ArrayIrValue<A> {
    fn swap(&self, replacement: &Self) -> Result<Self, ProgramError> {
        ReferenceSwapOperation
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        Ok(Self::Array(reference.swap(replacement).map_err(ProgramError::custom)?))
    }
}

impl<A: Value<Type = ArrayType> + Add> ReferenceAddUpdate for ArrayIrValue<A> {
    fn add_update(&self, update: &Self) -> Result<(), ProgramError> {
        ReferenceAddUpdateOperation
            .infer_output_types(&[self.r#type().into_owned(), update.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        reference.update_with(|current| current.add(update))
    }
}

impl<A: Value<Type = ArrayType>> FreezeReference for ArrayIrValue<A> {
    fn freeze(&self) -> Result<Self, ProgramError> {
        FreezeReferenceOperation.infer_output_types(std::slice::from_ref(&self.r#type().into_owned()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.freeze().map_err(ProgramError::custom)?))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::captures::CaptureReference;
    use crate::programs::{ReferenceError, TypeError};

    use super::*;

    #[test]
    fn test_eager_reference_allocation_and_read_roundtrip() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert!(matches!(reference, ArrayIrValue::Reference(_)));
        assert_eq!(ReferenceRead::read(&reference).unwrap(), initial);
    }

    #[test]
    fn test_eager_reference_operations_reject_mismatched_member_kinds() {
        let array = ArrayIrValue::<Array>::Array(Array::scalar(1.0_f32));
        assert_eq!(
            ReferenceRead::read(&array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        let reference = array.new_reference().unwrap();
        assert_eq!(
            reference.new_reference(),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(
            ReferenceSwap::swap(&array, &array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            ReferenceSwap::swap(&reference, &reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(
            ReferenceAddUpdate::add_update(&array, &array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            ReferenceAddUpdate::add_update(&reference, &reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(
            FreezeReference::freeze(&array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
    }

    #[test]
    fn test_eager_reference_updates_enforce_exact_storage_and_preserve_rejected_state() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();

        let error =
            ReferenceSwap::swap(&reference, &ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(ReferenceRead::read(&reference), Ok(initial.clone()));

        let error = ReferenceAddUpdate::add_update(&reference, &ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0])))
            .unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_add_update` addition result type `f64[2]` must exactly match reference referent \
                 type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(ReferenceRead::read(&reference), Ok(initial));

        // Broadcasting is valid only because the computed result preserves the exact stored type.
        assert_eq!(ReferenceAddUpdate::add_update(&reference, &ArrayIrValue::Array(Array::scalar(1.0_f32))), Ok(()),);
        assert_eq!(ReferenceRead::read(&reference), Ok(ArrayIrValue::Array(Array::vector(vec![2.0_f32, 3.0]))),);
    }

    #[test]
    fn test_eager_reference_freeze_invalidates_composite_aliases() {
        let reference = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).new_reference().unwrap();
        let alias = reference.clone();
        assert_eq!(FreezeReference::freeze(&reference), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))),);

        let error = ReferenceRead::read(&alias).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = ReferenceSwap::swap(&alias, &ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = ReferenceAddUpdate::add_update(&alias, &ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = FreezeReference::freeze(&alias).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_reject_dynamic_referents() {
        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let dynamic_value = ArrayIrValue::Array(CaptureReference::new(0, dynamic_type.clone()));
        assert_eq!(
            dynamic_value.new_reference(),
            Err(TypeError::invalid(
                "`new_reference` does not support dynamically shaped reference referent type `f32[length]`",
            )
            .into()),
        );

        let dynamic_reference = ArrayIrValue::Reference(Reference::new(CaptureReference::new(0, dynamic_type.clone())));
        let replacement = ArrayIrValue::Array(CaptureReference::new(1, dynamic_type.clone()));
        assert_eq!(
            ReferenceRead::read(&dynamic_reference),
            Err(TypeError::invalid(
                "`reference_read` does not support dynamically shaped reference referent type `f32[length]`",
            )
            .into()),
        );
        assert_eq!(
            ReferenceSwap::swap(&dynamic_reference, &replacement),
            Err(TypeError::invalid(
                "`reference_swap` does not support dynamically shaped reference referent type `f32[length]`",
            )
            .into()),
        );
        assert_eq!(
            FreezeReference::freeze(&dynamic_reference),
            Err(TypeError::invalid(
                "`freeze_reference` does not support dynamically shaped reference referent type `f32[length]`",
            )
            .into()),
        );
    }
}
