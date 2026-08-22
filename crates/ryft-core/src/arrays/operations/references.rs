//! Eager [`ArrayIrValue`] implementations of the reference capability traits, delegating type checks to the
//! canonical operation inference rules before touching any holder.

// TODO(eaplatanios): Review this module.

use crate::arrays::addressing::ArraySliceAxis;
use crate::arrays::ir::ArrayIrValue;
use crate::arrays::reference_views::{ArrayReference, ArrayReferenceViewTransform};
use crate::arrays::types::arrays::ArrayType;
use crate::operations::{
    Add, FreezeReference, FreezeReferenceOperation, NewReference, NewReferenceOperation, ReferenceAddUpdate,
    ReferenceAddUpdateOperation, ReferenceIndex, ReferenceIndexOperation, ReferenceRead, ReferenceReadOperation,
    ReferenceSlice, ReferenceSliceOperation, ReferenceSwap, ReferenceSwapOperation, Reshape, Slice, UpdateSlice,
};
use crate::programs::{Operation, ProgramError, ReferenceType, Typed, Value, ValueProjection};

impl<A: Value<Type = ArrayType>> NewReference for ArrayIrValue<A> {
    fn new_reference(&self) -> Result<Self, ProgramError> {
        NewReferenceOperation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let value = <Self as ValueProjection<ArrayType>>::projected(self)?.clone();
        Ok(Self::Reference(ArrayReference::new(value)))
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice> ReferenceRead for ArrayIrValue<A> {
    fn read(&self) -> Result<Self, ProgramError> {
        ReferenceReadOperation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.read_view()?))
    }
}

impl<A: Value<Type = ArrayType> + Reshape + Slice + UpdateSlice> ReferenceSwap for ArrayIrValue<A> {
    fn swap(&self, replacement: &Self) -> Result<Self, ProgramError> {
        ReferenceSwapOperation
            .infer_output_types(&[self.r#type().into_owned(), replacement.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let replacement = <Self as ValueProjection<ArrayType>>::projected(replacement)?.clone();
        Ok(Self::Array(reference.swap(replacement)?))
    }
}

impl<A: Value<Type = ArrayType> + Add + Reshape + Slice + UpdateSlice> ReferenceAddUpdate for ArrayIrValue<A> {
    fn add_update(&self, update: &Self) -> Result<(), ProgramError> {
        ReferenceAddUpdateOperation
            .infer_output_types(&[self.r#type().into_owned(), update.r#type().into_owned()], &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        let update = <Self as ValueProjection<ArrayType>>::projected(update)?;
        reference.add_update(update)
    }
}

impl<A: Value<Type = ArrayType>> FreezeReference for ArrayIrValue<A> {
    fn freeze(&self) -> Result<Self, ProgramError> {
        FreezeReferenceOperation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Array(reference.freeze()?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceIndex for ArrayIrValue<A> {
    fn reference_index(&self, axis: usize, index: usize) -> Result<Self, ProgramError> {
        let operation = ReferenceIndexOperation::new(axis, index);
        operation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceViewTransform::Index { axis, index })?))
    }
}

impl<A: Value<Type = ArrayType>> ReferenceSlice for ArrayIrValue<A> {
    fn reference_slice(&self, axes: &[ArraySliceAxis]) -> Result<Self, ProgramError> {
        let operation = ReferenceSliceOperation::new(axes.to_vec());
        operation.infer_output_types(std::slice::from_ref(self.r#type().as_ref()), &[])?;
        let reference = <Self as ValueProjection<ReferenceType<ArrayType>>>::projected(self)?;
        Ok(Self::Reference(reference.with_transform(ArrayReferenceViewTransform::Slice { axes: axes.to_vec() })?))
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::reference_views::ArrayReferenceViewError;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::programs::{ReferenceError, TypeError};

    use super::*;

    #[test]
    fn test_eager_reference_allocation_and_read_roundtrip() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();
        assert!(matches!(reference, ArrayIrValue::Reference(_)));
        assert_eq!(reference.read().unwrap(), initial);
    }

    #[test]
    fn test_eager_reference_index_slice_and_composition() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let root = initial.new_reference().unwrap();
        let row = root.reference_index(0, 1).unwrap();
        assert_eq!(row.read(), Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))));

        let slice = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1), ArraySliceAxis::new(1, 2, 1)]).unwrap();
        assert_eq!(
            slice.read(),
            Ok(ArrayIrValue::Array(Array::from_f64s(
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]),),
                vec![2.0, 3.0, 5.0, 6.0],
            ))),
        );
        let composed = slice.reference_index(0, 1).unwrap();
        assert_eq!(composed.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
    }

    #[test]
    fn test_eager_reference_indexed_mutation_reconstructs_removed_axis() {
        let matrix_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let initial = ArrayIrValue::Array(Array::from_f64s(matrix_type.clone(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let root = initial.new_reference().unwrap();
        let row = root.reference_index(0, 1).unwrap();

        assert_eq!(
            row.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0, 6.0]))),
        );
        row.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(
            root.read(),
            Ok(ArrayIrValue::Array(Array::from_f64s(matrix_type, vec![1.0, 2.0, 3.0, 11.0, 22.0, 33.0],))),
        );
    }

    #[test]
    fn test_eager_reference_views_share_overlapping_root_state() {
        let root = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0, 4.0])).new_reference().unwrap();
        let left = root.reference_slice(&[ArraySliceAxis::new(0, 3, 1)]).unwrap();
        let right = root.reference_slice(&[ArraySliceAxis::new(1, 3, 1)]).unwrap();

        assert_eq!(
            left.swap(&ArrayIrValue::Array(Array::vector(vec![10.0_f32, 20.0, 30.0]))),
            Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
        );
        assert_eq!(right.read(), Ok(ArrayIrValue::Array(Array::vector(vec![20.0_f32, 30.0, 4.0]))));
        right.add_update(&ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))).unwrap();
        assert_eq!(root.read(), Ok(ArrayIrValue::Array(Array::vector(vec![10.0_f32, 21.0, 32.0, 7.0]))));
    }

    #[test]
    fn test_eager_reference_view_validation_and_freeze_invalidation() {
        let root = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])).new_reference().unwrap();
        assert_eq!(
            root.reference_index(1, 0),
            Err(TypeError::invalid("reference index axis 1 is out of bounds for rank 1").into()),
        );
        assert_eq!(
            root.reference_index(0, 3),
            Err(TypeError::invalid("reference index 3 on axis 0 is out of bounds for size 3").into()),
        );
        assert_eq!(
            root.reference_slice(&[ArraySliceAxis::new(2, 2, 1)]),
            Err(TypeError::invalid("reference slice on axis 0 with start 2 and size 2 exceeds input size 3",).into()),
        );
        assert_eq!(
            root.reference_slice(&[ArraySliceAxis::new(0, 2, 2)]),
            Err(TypeError::invalid(
                "reference slice axis 0 stride must be 1 until scatter-backed strided updates are supported",
            )
            .into()),
        );

        let view = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        let same_view = root.reference_slice(&[ArraySliceAxis::new(0, 2, 1)]).unwrap();
        assert_eq!(view, same_view);
        assert_ne!(view, root);
        let different_view = root.reference_slice(&[ArraySliceAxis::new(1, 2, 1)]).unwrap();
        let view_handle = <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&view)
            .unwrap()
            .clone();
        let same_view_handle =
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&same_view)
                .unwrap()
                .clone();
        let different_view_handle =
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&different_view)
                .unwrap()
                .clone();
        let root_handle = <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&root)
            .unwrap()
            .clone();
        assert!(root_handle.is_runtime_root_handle());
        assert!(!view_handle.is_runtime_root_handle());
        let Err(error) = view_handle.lock_root() else {
            panic!("reference view must not expose a root transaction guard")
        };
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::InvalidRuntimeRoot),
        );
        let mut views = HashMap::new();
        views.insert(view_handle, 7);
        assert_eq!(views.get(&same_view_handle), Some(&7));
        assert_eq!(views.get(&different_view_handle), None);
        assert_eq!(views.get(&root_handle), None);
        let error = view.freeze().unwrap_err();
        assert_eq!(
            error.downcast_custom::<ArrayReferenceViewError>(),
            Some(&ArrayReferenceViewError::CannotFreezeView)
        );
        assert_eq!(root.read(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));

        assert_eq!(root.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0]))));
        let error = view.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_reject_mismatched_member_kinds() {
        let array = ArrayIrValue::<Array>::Array(Array::scalar(1.0_f32));
        assert_eq!(array.read(), Err(TypeError::invalid("expected reference type but got array type").into()));
        let reference = array.new_reference().unwrap();
        assert_eq!(
            reference.new_reference(),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(array.swap(&array), Err(TypeError::invalid("expected reference type but got array type").into()));
        assert_eq!(
            reference.swap(&reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(
            array.add_update(&array),
            Err(TypeError::invalid("expected reference type but got array type").into()),
        );
        assert_eq!(
            reference.add_update(&reference),
            Err(TypeError::invalid("expected array type but got reference type").into()),
        );
        assert_eq!(array.freeze(), Err(TypeError::invalid("expected reference type but got array type").into()));
    }

    #[test]
    fn test_eager_reference_updates_enforce_exact_storage_and_preserve_rejected_state() {
        let initial = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]));
        let reference = initial.new_reference().unwrap();

        let error = reference.swap(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0, 5.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_swap` replacement type `f32[3]` must exactly match reference referent type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(reference.read(), Ok(initial.clone()));

        let error = reference.add_update(&ArrayIrValue::Array(Array::vector(vec![3.0_f64, 4.0]))).unwrap_err();
        assert_eq!(
            error,
            TypeError::invalid(
                "`reference_add_update` addition result type `f64[2]` must exactly match reference referent \
                 type `f32[2]`",
            )
            .into(),
        );
        assert_eq!(reference.read(), Ok(initial));

        let replacement = ArrayIrValue::Array(Array::vector(vec![4.0_f32, 5.0]));
        assert_eq!(reference.swap(&replacement), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))),);
        assert_eq!(reference.read(), Ok(replacement));

        // Broadcasting is valid only because the computed result preserves the exact stored type.
        assert_eq!(reference.add_update(&ArrayIrValue::Array(Array::scalar(1.0_f32))), Ok(()));
        assert_eq!(reference.read(), Ok(ArrayIrValue::Array(Array::vector(vec![5.0_f32, 6.0]))));
    }

    #[test]
    fn test_eager_reference_freeze_invalidates_composite_aliases() {
        let reference = ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0])).new_reference().unwrap();
        let alias = reference.clone();
        assert_eq!(reference.freeze(), Ok(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0]))));

        let error = alias.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.swap(&ArrayIrValue::Array(Array::vector(vec![3.0_f32, 4.0]))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.add_update(&ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        let error = alias.freeze().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_eager_reference_operations_preserve_dynamic_referents() {
        // Equality over a dynamically typed `Array` cannot address its elements, so every referent this test observes
        // is unwrapped here and then compared by its declared type and its exact physical storage.
        fn referent(value: ArrayIrValue<Array>) -> Array {
            <ArrayIrValue<Array> as ValueProjection<ArrayType>>::into_projected(value).unwrap()
        }

        let dynamic_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()))]),
        );
        let initial_bytes = 1.0_f32.to_le_bytes().to_vec();
        let replacement_bytes = 2.0_f32.to_le_bytes().to_vec();

        // `Array`'s checked constructors reject dynamically shaped types, so both referents come from the test-only
        // unchecked hatch. They declare exactly the same dynamic type, which is what makes each holder transition
        // observable only through the payload it returns.
        let initial = Array::with_unchecked_type(dynamic_type.clone(), initial_bytes.clone());
        let replacement = Array::with_unchecked_type(dynamic_type.clone(), replacement_bytes.clone());
        let reference = ArrayIrValue::Array(initial).new_reference().unwrap();

        let read = referent(reference.read().unwrap());
        assert_eq!(read.r#type().into_owned(), dynamic_type);
        assert_eq!(read.storage_bytes(), initial_bytes.as_slice());

        // Swapping installs the replacement and hands back exactly the previous payload, so the later freeze consumes
        // the installed replacement rather than the original value.
        let old = referent(reference.swap(&ArrayIrValue::Array(replacement)).unwrap());
        assert_eq!(old.r#type().into_owned(), dynamic_type);
        assert_eq!(old.storage_bytes(), initial_bytes.as_slice());
        let frozen = referent(reference.freeze().unwrap());
        assert_eq!(frozen.r#type().into_owned(), dynamic_type);
        assert_eq!(frozen.storage_bytes(), replacement_bytes.as_slice());
        let error = reference.read().unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }
}
