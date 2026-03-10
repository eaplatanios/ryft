use thiserror::Error;

use crate::differentiation::JvpTracer;
use crate::types::array_type::{ArrayType, ArrayTypeBroadcastingError};

/// Represents a (possibly) nested data structure over [`ArrayType`]s. Specifically, this represents types
/// of nested structures over [`ArrayType`]d values. For example, a `JvpTracer<f32, f32>` has type
/// `JvpTracer<ArrayType, ArrayType>` and this type itself is an [`ArrayStructureType`], as it holds
/// two nested [`ArrayType`]s.
pub trait ArrayStructureType: Sized {
    /// Returns an [`Iterator`] over the nested [`ArrayType`]s.
    fn types(&self) -> impl Iterator<Item = &ArrayType>;

    /// Constructs a new instance of this [`ArrayStructureType`] from the provided [`ArrayType`]s. In some ways,
    /// this can be thought of as the inverse of [`ArrayStructureType::types`]. Note that this function will go
    /// over as many [`ArrayType`]s from the provided [`Iterator`] as it needs, but it will not return an error
    /// if the iterator is not exhausted. [`ArrayStructureType::from_types`] must be used if you want to ensure
    /// that the provided iterator is exhausted.
    fn from_types_with_remainder<I: Iterator<Item = ArrayType>>(types: &mut I)
    -> Result<Self, ArrayStructureTypeError>;

    /// Constructs a new instance of this [`ArrayStructureType`] from the provided [`ArrayType`]s. In some ways,
    /// this can be thought of as the inverse of [`ArrayStructureType::types`]. Note that this function will make
    /// sure to exhaust the provided [`Iterator`]. If the iterator contains more types than are needed to construct
    /// an instance of this type, then a [`ArrayStructureTypeError::UnusedArrayTypes`] will be returned. If you do
    /// not want to necessarily exhaust the provided iterator, then you must use
    /// [`ArrayStructureType::from_types_with_remainder`] instead.
    fn from_types<I: IntoIterator<Item = ArrayType>>(types: I) -> Result<Self, ArrayStructureTypeError> {
        let mut array_types = types.into_iter();
        let array_structure_type = Self::from_types_with_remainder(&mut array_types)?;
        array_types
            .next()
            .map(|_| Err(ArrayStructureTypeError::UnusedArrayTypes))
            .unwrap_or_else(|| Ok(array_structure_type))
    }

    /// Creates a broadcast instance of this [`ArrayStructureType`] from the provided [`ArrayStructureType`]s.
    /// Broadcasting is performed separately for each nested [`ArrayType`] using [`ArrayType::broadcast`].
    /// Refer to that function for information on the broadcasting rules.
    fn broadcast(types: &[&Self]) -> Result<Self, ArrayStructureTypeBroadcastingError> {
        if types.is_empty() {
            return Err(ArrayStructureTypeBroadcastingError::Empty);
        }

        // Collect all [`ArrayType`]s from each input structure into parallel vectors.
        let array_types_per_structure = types.iter().map(|tpe| tpe.types().collect::<Vec<_>>()).collect::<Vec<_>>();

        // Determine the number of [`ArrayType`]s that we need to broadcast. All structures should have the same
        // number of array types for proper broadcasting. We check for that condition and return an error if it
        // is violated.
        let array_types_count = array_types_per_structure[0].len();
        for array_types in array_types_per_structure.iter() {
            if array_types.len() != array_types_count {
                return Err(ArrayStructureTypeBroadcastingError::IncompatibleSizes {
                    lhs_size: array_types_count,
                    rhs_size: array_types.len(),
                });
            }
        }

        // Broadcast the [`ArrayType`]s for each position in `array_types_per_structure` in parallel.
        let mut broadcast_array_types = Vec::with_capacity(array_types_count);
        for i in 0..array_types_count {
            let structure_array_types = array_types_per_structure.iter().map(|types| types[i]).collect::<Vec<_>>();
            broadcast_array_types.push(ArrayType::broadcast(structure_array_types.as_slice())?);
        }

        // Reconstruct the [`ArrayStructureType`] from the broadcast [`ArrayType`]s.
        Ok(Self::from_types(broadcast_array_types.into_iter())?)
    }
}

// [`ArrayType`]s are always trivially [`ArrayStructureType`]s that contain a single [`ArrayType`] (i.e., themselves).
impl ArrayStructureType for ArrayType {
    fn types(&self) -> impl Iterator<Item = &ArrayType> {
        std::iter::once(self)
    }

    fn from_types_with_remainder<I: Iterator<Item = ArrayType>>(
        types: &mut I,
    ) -> Result<Self, ArrayStructureTypeError> {
        types.next().ok_or_else(|| ArrayStructureTypeError::InsufficientArrayTypes { expected_count: 1 })
    }
}

// The [`Type`]s of [`JvpTracer`]s are defined as [`JvpTracer`]s holding the underlying value and tangent types.
// Therefore, we need to also implement [`ArrayStructureType`] for [`JvpTracer`]s.
impl<V: ArrayStructureType, T: ArrayStructureType> ArrayStructureType for JvpTracer<V, T> {
    fn types(&self) -> impl Iterator<Item = &ArrayType> {
        self.value.types().chain(self.tangent.types())
    }

    fn from_types_with_remainder<I: Iterator<Item = ArrayType>>(
        types: &mut I,
    ) -> Result<Self, ArrayStructureTypeError> {
        Ok(Self { value: V::from_types_with_remainder(types)?, tangent: T::from_types_with_remainder(types)? })
    }
}

/// Represents an [`Error`] that can be returned by [`ArrayStructureType::from_types_with_remainder`]
/// or [`ArrayStructureType::from_types`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArrayStructureTypeError {
    /// Error returned when [`ArrayStructureType::from_types`] fails to exhaust the provided iterator.
    #[error("Got more array types than expected.")]
    UnusedArrayTypes,

    /// Error returned when [`ArrayStructureType::from_types_with_remainder`] or [`ArrayStructureType::from_types`]
    /// fails to construct an instance of the corresponding [`ArrayStructureType`] due to the provided iterator not
    /// containing enough [`ArrayType`]s (e.g., trying to construct a `JvpTracer<ArrayType, ArrayType>` from an
    /// iterator that only contains a single [`ArrayType`]).
    #[error("Expected at least {expected_count} array types but got fewer.")]
    InsufficientArrayTypes { expected_count: usize },
}

/// Represents an [`Error`] related to [`ArrayStructureType`] broadcasting. For more information on broadcasting,
/// refer to [`ArrayStructureType::broadcast`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ArrayStructureTypeBroadcastingError {
    /// Error returned when attempting to compute a broadcast [`ArrayStructureType`] for an empty collection of
    /// [`ArrayStructureType`]s (i.e., using [`ArrayStructureType::broadcast`]).
    #[error("Cannot construct a broadcast array structure type from an empty collection of array structure types.")]
    Empty,

    /// Error returned when attempting to compute a broadcast [`ArrayStructureType`] from a collection of
    /// [`ArrayStructureType`]s that contain a different number of nested [`ArrayType`]s.
    #[error(
        "Array type structures with a different number of nested array types ({lhs_size} vs {rhs_size}) cannot be broadcast."
    )]
    IncompatibleSizes { lhs_size: usize, rhs_size: usize },

    /// Error returned when attempting to compute a broadcast [`ArrayStructureType`] due to an underlying
    /// [`ArrayTypeBroadcastingError`].
    #[error("{0}")]
    ArrayTypeBroadcastingError(#[from] ArrayTypeBroadcastingError),

    /// Error returned when attempting to compute a broadcast [`ArrayStructureType`] due to an underlying
    /// [`ArrayStructureTypeError`].
    #[error("{0}")]
    ArrayStructureTypeError(#[from] ArrayStructureTypeError),
}

#[cfg(test)]
mod tests {
    use crate::types::array_type::DataType::*;
    use crate::types::array_type::{Shape, Size};

    use super::*;

    #[test]
    fn test_array_type_array_structure_type_types_and_from_types() {
        let t0 = ArrayType::new(Float32, Shape::new(vec![42.into(), 4.into()]));
        let t1 = ArrayType::new(Float32, Shape::new(vec![1.into(), 4.into()]));
        let t2 = ArrayType::scalar(Boolean);

        assert_eq!(t0.types().collect::<Vec<_>>(), vec![&t0]);
        assert_eq!(t1.types().collect::<Vec<_>>(), vec![&t1]);
        assert_eq!(t2.types().collect::<Vec<_>>(), vec![&t2]);

        assert_eq!(ArrayType::from_types(vec![t0.clone()]), Ok(t0.clone()));
        assert_eq!(ArrayType::from_types(vec![t1.clone()]), Ok(t1.clone()));
        assert_eq!(ArrayType::from_types(vec![t2.clone()]), Ok(t2.clone()));

        assert!(ArrayType::from_types(vec![]).is_err());
        assert!(ArrayType::from_types(vec![t0.clone(), t1.clone()]).is_err());
    }

    #[test]
    fn test_array_type_array_structure_type_broadcast() {
        let t0 = ArrayType::new(Float32, Shape::new(vec![42.into(), 4.into()]));
        let t1 = ArrayType::new(Float32, Shape::new(vec![1.into(), 4.into()]));
        let t2 = ArrayType::scalar(Boolean);
        let t3 = ArrayType::new(Float32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(ArrayType::broadcast(&[&t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcast(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcast(&[&t2, &t1]), Ok(t1.clone()));

        assert!(ArrayType::broadcast(&[]).is_err());
        assert!(ArrayType::broadcast(&[&t0, &t3]).is_err());
    }

    #[test]
    fn test_jvp_tracer_array_structure_type_types_and_from_types() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, Shape::new(vec![10.into(), 5.into()]));
        let t2 = ArrayType::new(Float32, Shape::new(vec![1.into(), 5.into()]));
        let t3 = ArrayType::new(Float32, Shape::new(vec![10.into(), 1.into()]));

        let j0 = JvpTracer { value: t1.clone(), tangent: t2.clone() };
        let j1 = JvpTracer { value: t0.clone(), tangent: t3.clone() };

        assert_eq!(j0.types().collect::<Vec<_>>(), vec![&j0.value, &j0.tangent]);
        assert_eq!(j1.types().collect::<Vec<_>>(), vec![&j1.value, &j1.tangent]);

        assert_eq!(JvpTracer::<ArrayType, ArrayType>::from_types(vec![t1.clone(), t2.clone()]), Ok(j0.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::from_types(vec![t0.clone(), t3.clone()]), Ok(j1.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::from_types(j0.types().map(|tpe| tpe.clone())), Ok(j0.clone()));

        assert!(JvpTracer::<ArrayType, ArrayType>::from_types(vec![]).is_err());
        assert!(JvpTracer::<ArrayType, ArrayType>::from_types(vec![t1.clone()]).is_err());
        assert!(JvpTracer::<ArrayType, ArrayType>::from_types(vec![t1.clone(), t2.clone(), t0.clone()]).is_err());
    }

    #[test]
    fn test_jvp_tracer_array_structure_type_broadcast() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, Shape::new(vec![10.into(), 5.into()]));
        let t2 = ArrayType::new(Float32, Shape::new(vec![1.into(), 5.into()]));
        let t3 = ArrayType::new(Float32, Shape::new(vec![10.into(), 1.into()]));
        let t4 = ArrayType::new(Float32, Shape::new(vec![10.into(), 5.into()]));
        let t5 = ArrayType::new(Float32, Shape::new(vec![5.into(), 3.into()]));
        let t6 = ArrayType::new(Float32, Shape::new(vec![4.into(), 2.into()]));

        let j0 = JvpTracer { value: t1.clone(), tangent: t2.clone() };
        let j1 = JvpTracer { value: t0.clone(), tangent: t3.clone() };
        let j2 = JvpTracer { value: t1.clone(), tangent: t4.clone() };
        let j3 = JvpTracer { value: t5.clone(), tangent: t6.clone() };

        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0]), Ok(j0.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0, &j1]), Ok(j2.clone()),);

        assert!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[]).is_err());
        assert!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0, &j3]).is_err());
    }

    #[test]
    fn test_nested_jvp_tracer_array_structure_type_types_and_from_types() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, Shape::new(vec![42.into(), 4.into(), 2.into()]));
        let t2 = ArrayType::new(BFloat16, Shape::new(vec![4.into(), 1.into()]));
        let t3 = ArrayType::new(Float16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t4 = ArrayType::new(Complex64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t5 = ArrayType::new(Float8E4M3FN, Shape::new(vec![42.into(), Size::Dynamic(None)]));

        let j0 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t0.clone(), tangent: t1.clone() }, tangent: t2.clone() },
            tangent: JvpTracer { value: t3.clone(), tangent: JvpTracer { value: t4.clone(), tangent: t5.clone() } },
        };

        type J = JvpTracer<
            JvpTracer<JvpTracer<ArrayType, ArrayType>, ArrayType>,
            JvpTracer<ArrayType, JvpTracer<ArrayType, ArrayType>>,
        >;

        assert_eq!(j0.types().collect::<Vec<_>>(), vec![&t0, &t1, &t2, &t3, &t4, &t5]);
        assert_eq!(J::from_types(j0.types().map(|tpe| tpe.clone())), Ok(j0.clone()),);
        assert!(J::from_types(j0.types().take(2).map(|tpe| tpe.clone())).is_err());
    }

    #[test]
    fn test_nested_jvp_tracer_array_structure_type_broadcast() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(Float32, Shape::new(vec![42.into(), 2.into(), 4.into()]));
        let t2 = ArrayType::new(BFloat16, Shape::new(vec![4.into(), 1.into()]));
        let t3 = ArrayType::new(Float16, Shape::new(vec![4.into(), 1.into()]));
        let t4 = ArrayType::new(Complex64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t5 = ArrayType::new(Float8E4M3FN, Shape::new(vec![42.into(), Size::Dynamic(None)]));
        let t6 = ArrayType::new(Float32, Shape::new(vec![1.into()]));
        let t7 = ArrayType::new(Float32, Shape::new(vec![4.into()]));
        let t8 = ArrayType::new(Float32, Shape::new(vec![1.into(), 4.into()]));
        let t9 = ArrayType::new(Float32, Shape::new(vec![42.into(), 1.into()]));
        let t10 = ArrayType::new(Float32, Shape::new(vec![42.into(), 4.into()]));
        let t11 = ArrayType::new(Float32, Shape::new(vec![5.into(), 3.into()]));
        let t12 = ArrayType::new(Float32, Shape::new(vec![4.into(), 2.into()]));
        let t13 = ArrayType::new(Float32, Shape::new(vec![4.into(), 4.into()]));
        let t14 = ArrayType::new(Float32, Shape::new(vec![42.into(), Size::Dynamic(None)]));

        let j0 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t0.clone(), tangent: t1.clone() }, tangent: t2.clone() },
            tangent: JvpTracer { value: t3.clone(), tangent: JvpTracer { value: t4.clone(), tangent: t5.clone() } },
        };

        let j1 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t6.clone(), tangent: t8.clone() }, tangent: t0.clone() },
            tangent: JvpTracer { value: t7.clone(), tangent: JvpTracer { value: t6.clone(), tangent: t9.clone() } },
        };

        let j2 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t0.clone(), tangent: t6.clone() }, tangent: t7.clone() },
            tangent: JvpTracer { value: t8.clone(), tangent: JvpTracer { value: t9.clone(), tangent: t10.clone() } },
        };

        let j3 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t0.clone(), tangent: t7.clone() }, tangent: t6.clone() },
            tangent: JvpTracer { value: t9.clone(), tangent: JvpTracer { value: t8.clone(), tangent: t10.clone() } },
        };

        let j4 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t11.clone(), tangent: t12.clone() }, tangent: t7.clone() },
            tangent: JvpTracer { value: t8.clone(), tangent: JvpTracer { value: t9.clone(), tangent: t10.clone() } },
        };

        let j5 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t6.clone(), tangent: t1.clone() }, tangent: t2.clone() },
            tangent: JvpTracer { value: t13.clone(), tangent: JvpTracer { value: t4.clone(), tangent: t14.clone() } },
        };

        let j6 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t6.clone(), tangent: t8.clone() }, tangent: t7.clone() },
            tangent: JvpTracer { value: t8.clone(), tangent: JvpTracer { value: t9.clone(), tangent: t10.clone() } },
        };

        let j7 = JvpTracer {
            value: JvpTracer { value: JvpTracer { value: t6.clone(), tangent: t8.clone() }, tangent: t7.clone() },
            tangent: JvpTracer { value: t10.clone(), tangent: JvpTracer { value: t10.clone(), tangent: t10.clone() } },
        };

        type J = JvpTracer<
            JvpTracer<JvpTracer<ArrayType, ArrayType>, ArrayType>,
            JvpTracer<ArrayType, JvpTracer<ArrayType, ArrayType>>,
        >;

        assert_eq!(J::broadcast(&[&j0]), Ok(j0.clone()));
        assert_eq!(J::broadcast(&[&j1, &j0]), Ok(j5.clone()));
        assert_eq!(J::broadcast(&[&j1, &j2]), Ok(j6.clone()),);
        assert_eq!(J::broadcast(&[&j1, &j2, &j3]), Ok(j7.clone()),);
        assert_eq!(J::broadcast(&[&j3, &j1, &j2]), Ok(j7.clone()),);

        assert!(J::broadcast(&[]).is_err());
        assert!(J::broadcast(&[&j1, &j4]).is_err());
    }
}
