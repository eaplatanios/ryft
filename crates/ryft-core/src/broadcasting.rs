use thiserror::Error;

use crate::{
    parameters::{ParameterError, Parameterized},
    types::DataType,
    types::data_type::DataTypeError,
    types_v0::array_type::{ArrayType, Shape, Size},
};

/// Error returned by the operations supported by [`Broadcastable`] types.
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BroadcastingError {
    #[error("cannot broadcast an empty collection of types")]
    EmptyBroadcastingInput,

    #[error("failed to broadcast due to incompatible data types; {0}")]
    IncompatibleDataTypes(#[from] DataTypeError),

    #[error("failed to broadcast shape `{lhs}` to shape `{rhs}`")]
    IncompatibleShapes { lhs: Shape, rhs: Shape },

    #[error("failed to reconstruct the parameterized structure after broadcasting; {0}")]
    ParameterError(#[from] ParameterError),
}

// TODO(eaplatanios): Expand the documentation by explaining how `Parameterized` data structures are handled and also
//  explaining the NumPy broadcasting semantics based on reading the NumPy docs using our documentation conventions.
/// Represents [`Type`]s or values that can be broadcast together using a combination of
/// [NumPy](https://numpy.org/doc/stable/user/basics.broadcasting.html) broadcasting semantics
/// and recursion into [`Parameterized`] data structures.
pub trait Broadcastable: Sized {
    /// Broadcasts this value with `other` and returns the least common result that both values can broadcast to.
    /// This operation is _symmetric_. For example, with [`Shape`] values this returns the smallest shape that both
    /// operands can broadcast to, and with [`ArrayType`] values it combines [`DataType`] promotion with [`Shape`]
    /// broadcasting.
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts this value to the provided `other` value. Unlike [`Broadcastable::broadcast`], this operation
    /// is _not symmetric_: `x.broadcast_to(y)` and `y.broadcast_to(x)` may differ and one of them may even fail.
    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError>;

    /// Broadcasts the provided values into a single value by folding over [`Broadcastable::broadcast`] from left to
    /// right. Returns [`BroadcastingError::EmptyBroadcastingInput`] if no values are provided.
    fn broadcasted(values: &[&Self]) -> Result<Self, BroadcastingError>
    where
        Self: Clone,
    {
        let (head, tail) = values.split_first().ok_or(BroadcastingError::EmptyBroadcastingInput)?;
        tail.iter().try_fold((*head).clone(), |accumulator, value| accumulator.broadcast(*value))
    }

    /// Returns `true` if this value can be broadcast to `other`, and `false` otherwise.
    fn is_broadcastable_to(&self, other: &Self) -> bool {
        self.broadcast_to(other).is_ok()
    }
}

impl Broadcastable for DataType {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(DataType::promoted(&[*self, *other])?)
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        Ok(self.promote_to(*other)?)
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        self.is_promotable_to(*other)
    }
}

impl Broadcastable for Shape {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        // Handle differing array ranks by (conceptually) padding the shorter shape with ones on the left
        // (i.e., as a prefix), up to the rank of the longer shape.
        let broadcasted_rank = self.rank().max(other.rank());
        let self_offset = broadcasted_rank - self.rank();
        let other_offset = broadcasted_rank - other.rank();
        let mut broadcasted_dimensions = Vec::with_capacity(broadcasted_rank);
        for i in 0..broadcasted_rank {
            let self_size = if i < self_offset { Size::Static(1) } else { self.dimensions[i - self_offset] };
            let other_size = if i < other_offset { Size::Static(1) } else { other.dimensions[i - other_offset] };
            let broadcasted_size = match (&self_size, &other_size) {
                (_, Size::Static(1)) => self_size,
                (Size::Static(1), _) => other_size,
                (Size::Static(x), Size::Static(y)) if x == y => Size::Static(*x),
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => Size::Dynamic(*x),
                _ => {
                    return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
                }
            };
            broadcasted_dimensions.push(broadcasted_size);
        }

        Ok(Shape::new(broadcasted_dimensions))
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        if self.rank() > other.rank() {
            return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
        }

        // Handle differing array ranks by (conceptually) padding the dimension sizes of the left shape with
        // ones on the left (i.e., as a prefix), up to the rank of the right shape.
        let broadcasted_rank = other.rank();
        let offset = broadcasted_rank - self.rank();
        let mut broadcasted_shape = Vec::with_capacity(broadcasted_rank);
        for i in 0..broadcasted_rank {
            let self_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let other_size = other.dimensions[i];
            let broadcasted_size = match (&self_size, &other_size) {
                (Size::Static(1), _) => other_size,
                (Size::Static(x), Size::Static(y)) if x == y => Size::Static(*y),
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => Size::Dynamic(*y),
                _ => {
                    return Err(BroadcastingError::IncompatibleShapes { lhs: self.clone(), rhs: other.clone() });
                }
            };
            broadcasted_shape.push(broadcasted_size);
        }

        Ok(Shape::new(broadcasted_shape))
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        if self.rank() > other.rank() {
            return false;
        }

        let broadcasted_rank = other.rank();
        let offset = broadcasted_rank - self.rank();
        for i in 0..broadcasted_rank {
            let self_size = if i < offset { Size::Static(1) } else { self.dimensions[i - offset] };
            let other_size = other.dimensions[i];
            match (&self_size, &other_size) {
                (Size::Static(1), _) => continue,
                (Size::Static(x), Size::Static(y)) if x == y => continue,
                (Size::Dynamic(x), Size::Dynamic(y)) if x == y => continue,
                _ => return false,
            };
        }

        true
    }
}

impl<T: Parameterized<ArrayType, ParameterStructure: Clone>> Broadcastable for T {
    fn broadcast(&self, other: &Self) -> Result<Self, BroadcastingError> {
        let broadcast_to = |lhs: &Self, rhs: &Self| -> Result<Self, BroadcastingError> {
            let structure = rhs.parameter_structure();
            let broadcasted_array_types = lhs
                .broadcast_to_parameter_structure::<T>(structure.clone())?
                .parameters()
                .zip(rhs.parameters())
                .map(|(lhs, rhs)| {
                    let broadcasted_data_type = DataType::promoted(&[lhs.data_type, rhs.data_type])?;
                    let broadcasted_shape = lhs.shape.broadcast(&rhs.shape)?;
                    Ok(ArrayType::new(broadcasted_data_type, broadcasted_shape))
                })
                .collect::<Result<Vec<_>, BroadcastingError>>()?;
            Ok(Self::from_parameters(structure, broadcasted_array_types)?)
        };

        match broadcast_to(self, other) {
            Ok(broadcasted) => Ok(broadcasted),
            Err(_) => broadcast_to(other, self),
        }
    }

    fn broadcast_to(&self, other: &Self) -> Result<Self, BroadcastingError> {
        let structure = other.parameter_structure();
        let broadcasted_array_types = self
            .broadcast_to_parameter_structure::<T>(structure.clone())?
            .parameters()
            .zip(other.parameters())
            .map(|(lhs, rhs)| {
                let broadcasted_data_type = lhs.data_type.promote_to(rhs.data_type)?;
                let broadcasted_shape = lhs.shape.broadcast_to(&rhs.shape)?;
                Ok(ArrayType::new(broadcasted_data_type, broadcasted_shape))
            })
            .collect::<Result<Vec<_>, BroadcastingError>>()?;
        Ok(Self::from_parameters(structure, broadcasted_array_types)?)
    }

    fn is_broadcastable_to(&self, other: &Self) -> bool {
        let Ok(broadcasted_self) = self.broadcast_to_parameter_structure::<T>(other.parameter_structure()) else {
            return false;
        };
        broadcasted_self.parameters().zip(other.parameters()).all(|(lhs, rhs)| {
            lhs.data_type.is_promotable_to(rhs.data_type) && lhs.shape.is_broadcastable_to(&rhs.shape)
        })
    }
}

#[cfg(test)]
mod tests {
    use ryft_macros::Parameterized;

    use crate::parameters::{Parameter, ParameterError};
    use crate::types::data_type::DataType::*;
    use crate::types_v0::array_type::Shape;

    use super::*;

    #[test]
    fn test_data_type_broadcasting() {
        assert_eq!(Boolean.broadcast(&U16), Ok(U16));
        assert_eq!(U16.broadcast(&Boolean), Ok(U16));
        assert!(matches!(F8E3M4.broadcast(&F32), Err(BroadcastingError::IncompatibleDataTypes(_))));

        assert_eq!(Boolean.broadcast_to(&U16), Ok(U16));
        assert!(matches!(F64.broadcast_to(&I32), Err(BroadcastingError::IncompatibleDataTypes(_))));

        assert!(Boolean.is_broadcastable_to(&U16));
        assert!(!F64.is_broadcastable_to(&I32));

        assert_eq!(DataType::broadcasted(&[&Boolean]), Ok(Boolean));
        assert_eq!(DataType::broadcasted(&[&Boolean, &U16]), Ok(U16));
        assert!(matches!(DataType::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(DataType::broadcasted(&[&F8E3M4, &F32]), Err(BroadcastingError::IncompatibleDataTypes(_))));
    }

    #[test]
    fn test_shape_broadcasting() {
        let s0 = Shape::new(vec![42.into(), 4.into()]);
        let s1 = Shape::new(vec![1.into(), 4.into()]);
        let s2 = Shape::scalar();
        let s3 = Shape::new(vec![5.into(), 3.into()]);

        assert_eq!(s1.broadcast(&s2), Ok(s1.clone()));
        assert_eq!(s2.broadcast(&s1), Ok(s1.clone()));
        assert!(matches!(s0.broadcast(&s3), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(s2.broadcast_to(&s1), Ok(s1.clone()));
        assert!(matches!(s1.broadcast_to(&s2), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(Shape::broadcasted(&[&s0]), Ok(s0.clone()));
        assert_eq!(Shape::broadcasted(&[&s1, &s2]), Ok(s1.clone()));
        assert_eq!(Shape::broadcasted(&[&s2, &s1]), Ok(s1.clone()));
        assert!(matches!(Shape::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(Shape::broadcasted(&[&s0, &s3]), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert!(s2.is_broadcastable_to(&s1));
        assert!(!s0.is_broadcastable_to(&s3));
    }

    #[test]
    fn test_array_type_broadcasting() {
        let t0 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let t1 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let t2 = ArrayType::scalar(Boolean);
        let t3 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(t1.broadcast(&t2), Ok(t1.clone()));
        assert_eq!(t2.broadcast(&t1), Ok(t1.clone()));
        assert!(matches!(t0.broadcast(&t3), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(t2.broadcast_to(&t1), Ok(t1.clone()));
        assert!(matches!(t0.broadcast_to(&t3), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert_eq!(ArrayType::broadcasted(&[&t0]), Ok(t0.clone()));
        assert_eq!(ArrayType::broadcasted(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(ArrayType::broadcasted(&[&t2, &t1]), Ok(t1.clone()));
        assert!(matches!(ArrayType::broadcasted(&[]), Err(BroadcastingError::EmptyBroadcastingInput)));
        assert!(matches!(ArrayType::broadcasted(&[&t0, &t3]), Err(BroadcastingError::IncompatibleShapes { .. })));

        assert!(t2.is_broadcastable_to(&t1));
        assert!(!t0.is_broadcastable_to(&t3));
    }

    #[test]
    fn test_parameterized_array_type_broadcastable() {
        #[derive(Parameterized, Clone, Debug, Eq, PartialEq)]
        #[ryft(crate = "crate::parameters")]
        enum TestEnum<P: Parameter> {
            Wrapped { inner: P },
            Pair { left: P, right: P },
        }

        let t0 = TestEnum::Pair {
            left: ArrayType::scalar(F32),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()])),
        };

        let t1 = TestEnum::Pair {
            left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
        };

        let t2 = TestEnum::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };

        let t3 = TestEnum::Wrapped { inner: ArrayType::scalar(F32) };

        assert_eq!(t0.broadcast(&t1), Ok(t1.clone()));
        assert!(matches!(
            t3.broadcast(&t2),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert_eq!(t0.broadcast_to(&t1), Ok(t1.clone()));
        assert!(matches!(
            t3.broadcast_to(&t2),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert_eq!(TestEnum::broadcasted(&[&t0]), Ok(t0.clone()));
        assert_eq!(
            TestEnum::broadcasted(&[&t0, &t1]),
            Ok(TestEnum::Pair {
                left: ArrayType::new(F64, Shape::new(vec![2.into(), 1.into()])),
                right: ArrayType::new(F64, Shape::new(vec![3.into(), 4.into()])),
            }),
        );
        assert!(matches!(
            TestEnum::broadcasted(&[&t3, &t2]),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));

        assert!(t0.is_broadcastable_to(&t1));
        assert!(!t3.is_broadcastable_to(&t2));
    }
}
