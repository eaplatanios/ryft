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

// TODO(eaplatanios): Review these tests.
#[cfg(test)]
mod tests {
    use std::vec::IntoIter;

    use crate::differentiation::JvpTracer;
    use crate::parameters::{
        Parameter, ParameterError, ParameterPath, Parameterized, ParameterizedFamily, Placeholder,
    };
    use crate::types::data_type::DataType::*;
    use crate::types_v0::array_type::{Shape, Size};

    use super::*;

    // TODO(eaplatanios): Use `#[derive(Parameterized)]` like in `parameters.rs`.
    /// Test-only parameterized type whose leaf variant uses the root parameter path.
    #[derive(Clone, Debug, Eq, PartialEq)]
    enum PrefixBroadcast<P: Parameter> {
        Leaf(P),
        Wrapped { inner: P },
        Pair { left: P, right: P },
    }

    /// [`ParameterizedFamily`] for [`PrefixBroadcast`].
    #[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
    struct PrefixBroadcastFamily;

    impl<P: Parameter> ParameterizedFamily<P> for PrefixBroadcastFamily {
        type To = PrefixBroadcast<P>;
    }

    impl<P: Parameter> Parameterized<P> for PrefixBroadcast<P> {
        type Family = PrefixBroadcastFamily;
        type To<T: Parameter> = PrefixBroadcast<T>;
        type ParameterStructure = PrefixBroadcast<Placeholder>;
        type ParameterIterator<'t, T: 't + Parameter>
            = IntoIter<&'t T>
        where
            Self: 't;
        type ParameterIteratorMut<'t, T: 't + Parameter>
            = IntoIter<&'t mut T>
        where
            Self: 't;
        type ParameterIntoIterator<T: Parameter> = IntoIter<T>;
        type NamedParameterIterator<'t, T: 't + Parameter>
            = IntoIter<(ParameterPath, &'t T)>
        where
            Self: 't;
        type NamedParameterIteratorMut<'t, T: 't + Parameter>
            = IntoIter<(ParameterPath, &'t mut T)>
        where
            Self: 't;
        type NamedParameterIntoIterator<T: Parameter> = IntoIter<(ParameterPath, T)>;

        fn parameter_count(&self) -> usize {
            match self {
                Self::Leaf(_) | Self::Wrapped { .. } => 1,
                Self::Pair { .. } => 2,
            }
        }

        fn parameter_structure(&self) -> Self::ParameterStructure {
            match self {
                Self::Leaf(_) => PrefixBroadcast::Leaf(Placeholder),
                Self::Wrapped { .. } => PrefixBroadcast::Wrapped { inner: Placeholder },
                Self::Pair { .. } => PrefixBroadcast::Pair { left: Placeholder, right: Placeholder },
            }
        }

        fn parameters(&self) -> Self::ParameterIterator<'_, P> {
            match self {
                Self::Leaf(parameter) => vec![parameter].into_iter(),
                Self::Wrapped { inner } => vec![inner].into_iter(),
                Self::Pair { left, right } => vec![left, right].into_iter(),
            }
        }

        fn parameters_mut(&mut self) -> Self::ParameterIteratorMut<'_, P> {
            match self {
                Self::Leaf(parameter) => vec![parameter].into_iter(),
                Self::Wrapped { inner } => vec![inner].into_iter(),
                Self::Pair { left, right } => vec![left, right].into_iter(),
            }
        }

        fn into_parameters(self) -> Self::ParameterIntoIterator<P> {
            match self {
                Self::Leaf(parameter) => vec![parameter].into_iter(),
                Self::Wrapped { inner } => vec![inner].into_iter(),
                Self::Pair { left, right } => vec![left, right].into_iter(),
            }
        }

        fn named_parameters(&self) -> Self::NamedParameterIterator<'_, P> {
            match self {
                Self::Leaf(parameter) => vec![(ParameterPath::root(), parameter)].into_iter(),
                Self::Wrapped { inner } => vec![(ParameterPath::root().field("inner"), inner)].into_iter(),
                Self::Pair { left, right } => {
                    vec![(ParameterPath::root().field("left"), left), (ParameterPath::root().field("right"), right)]
                        .into_iter()
                }
            }
        }

        fn named_parameters_mut(&mut self) -> Self::NamedParameterIteratorMut<'_, P> {
            match self {
                Self::Leaf(parameter) => vec![(ParameterPath::root(), parameter)].into_iter(),
                Self::Wrapped { inner } => vec![(ParameterPath::root().field("inner"), inner)].into_iter(),
                Self::Pair { left, right } => {
                    vec![(ParameterPath::root().field("left"), left), (ParameterPath::root().field("right"), right)]
                        .into_iter()
                }
            }
        }

        fn into_named_parameters(self) -> Self::NamedParameterIntoIterator<P> {
            match self {
                Self::Leaf(parameter) => vec![(ParameterPath::root(), parameter)].into_iter(),
                Self::Wrapped { inner } => vec![(ParameterPath::root().field("inner"), inner)].into_iter(),
                Self::Pair { left, right } => {
                    vec![(ParameterPath::root().field("left"), left), (ParameterPath::root().field("right"), right)]
                        .into_iter()
                }
            }
        }

        fn from_parameters_with_remainder<I: Iterator<Item = P>>(
            structure: Self::ParameterStructure,
            parameters: &mut I,
        ) -> Result<Self, ParameterError> {
            match structure {
                PrefixBroadcast::Leaf(_) => {
                    parameters.next().map(Self::Leaf).ok_or_else(|| ParameterError::MissingParameters {
                        expected_count: 1,
                        paths: Some(vec![ParameterPath::root().to_string()]),
                    })
                }
                PrefixBroadcast::Wrapped { .. } => {
                    parameters.next().map(|inner| Self::Wrapped { inner }).ok_or_else(|| {
                        ParameterError::MissingParameters {
                            expected_count: 1,
                            paths: Some(vec![ParameterPath::root().field("inner").to_string()]),
                        }
                    })
                }
                PrefixBroadcast::Pair { .. } => {
                    let left = parameters.next();
                    let right = parameters.next();
                    match (left, right) {
                        (Some(left), Some(right)) => Ok(Self::Pair { left, right }),
                        (left, right) => {
                            let mut missing_paths = Vec::new();
                            if left.is_none() {
                                missing_paths.push(ParameterPath::root().field("left").to_string());
                            }
                            if right.is_none() {
                                missing_paths.push(ParameterPath::root().field("right").to_string());
                            }
                            Err(ParameterError::MissingParameters { expected_count: 2, paths: Some(missing_paths) })
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_array_type_broadcastable_broadcast() {
        let t0 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let t1 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let t2 = ArrayType::scalar(Boolean);
        let t3 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(<ArrayType as Broadcastable>::broadcasted(&[&t0]), Ok(t0.clone()));
        assert_eq!(<ArrayType as Broadcastable>::broadcasted(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(<ArrayType as Broadcastable>::broadcasted(&[&t2, &t1]), Ok(t1.clone()));

        assert!(matches!(
            <ArrayType as Broadcastable>::broadcasted(&[]),
            Err(BroadcastingError::EmptyBroadcastingInput)
        ));
        assert!(matches!(
            <ArrayType as Broadcastable>::broadcasted(&[&t0, &t3]),
            Err(BroadcastingError::IncompatibleShapes { .. }),
        ));
    }

    #[test]
    fn test_jvp_tracer_broadcastable_broadcast() {
        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, Shape::new(vec![10.into(), 5.into()]));
        let t2 = ArrayType::new(F32, Shape::new(vec![1.into(), 5.into()]));
        let t3 = ArrayType::new(F32, Shape::new(vec![10.into(), 1.into()]));
        let t4 = ArrayType::new(F32, Shape::new(vec![10.into(), 5.into()]));
        let t5 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));
        let t6 = ArrayType::new(F32, Shape::new(vec![4.into(), 2.into()]));

        let j0 = JvpTracer { value: t1.clone(), tangent: t2.clone() };
        let j1 = JvpTracer { value: t0.clone(), tangent: t3.clone() };
        let j2 = JvpTracer { value: t1.clone(), tangent: t4.clone() };
        let j3 = JvpTracer { value: t5.clone(), tangent: t6.clone() };

        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0]), Ok(j0.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0, &j1]), Ok(j2));

        assert!(matches!(
            JvpTracer::<ArrayType, ArrayType>::broadcasted(&[]),
            Err(BroadcastingError::EmptyBroadcastingInput)
        ));
        assert!(matches!(
            JvpTracer::<ArrayType, ArrayType>::broadcasted(&[&j0, &j3]),
            Err(BroadcastingError::IncompatibleShapes { .. }),
        ));
    }

    #[test]
    fn test_nested_jvp_tracer_broadcastable_broadcast() {
        type NestedJvpTracer =
            JvpTracer<JvpTracer<ArrayType, ArrayType>, JvpTracer<ArrayType, JvpTracer<ArrayType, ArrayType>>>;

        let t0 = ArrayType::scalar(Boolean);
        let t1 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into(), 2.into()]));
        let t2 = ArrayType::new(BF16, Shape::new(vec![4.into(), 1.into()]));
        let t3 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t4 = ArrayType::new(C64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t5 = ArrayType::new(BF16, Shape::new(vec![42.into(), Size::Dynamic(None)]));
        let t6 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into(), 2.into()]));
        let t7 = ArrayType::new(BF16, Shape::new(vec![1.into(), 1.into()]));
        let t8 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t9 = ArrayType::new(C64, Shape::new(vec![1.into(), 42.into(), 1.into()]));
        let t11 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into(), 2.into()]));
        let t12 = ArrayType::new(BF16, Shape::new(vec![4.into(), 1.into()]));
        let t13 = ArrayType::new(F16, Shape::new(vec![4.into(), Size::Dynamic(Some(1))]));
        let t14 = ArrayType::new(C64, Shape::new(vec![Size::Dynamic(None), 42.into(), Size::Dynamic(None)]));
        let t15 = ArrayType::new(BF16, Shape::new(vec![42.into(), Size::Dynamic(None)]));
        let t17 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        let j0 = NestedJvpTracer {
            value: JvpTracer { value: t0.clone(), tangent: t1.clone() },
            tangent: JvpTracer { value: t2.clone(), tangent: JvpTracer { value: t3.clone(), tangent: t4.clone() } },
        };
        let j1 = NestedJvpTracer {
            value: JvpTracer { value: t5.clone(), tangent: t6.clone() },
            tangent: JvpTracer { value: t7.clone(), tangent: JvpTracer { value: t8.clone(), tangent: t9.clone() } },
        };
        let j2 = NestedJvpTracer {
            value: JvpTracer { value: t15.clone(), tangent: t11.clone() },
            tangent: JvpTracer { value: t12.clone(), tangent: JvpTracer { value: t13.clone(), tangent: t14.clone() } },
        };
        let j3 = NestedJvpTracer {
            value: JvpTracer { value: t15.clone(), tangent: t17.clone() },
            tangent: JvpTracer { value: t12.clone(), tangent: JvpTracer { value: t13.clone(), tangent: t14.clone() } },
        };

        assert_eq!(NestedJvpTracer::broadcasted(&[&j0]), Ok(j0.clone()));
        assert_eq!(NestedJvpTracer::broadcasted(&[&j0, &j1]), Ok(j2));
        assert!(
            matches!(NestedJvpTracer::broadcasted(&[&j0, &j3]), Err(BroadcastingError::IncompatibleShapes { .. }),)
        );
    }

    #[test]
    fn test_prefix_broadcastable_parameter_structures() {
        let leaf = PrefixBroadcast::Leaf(ArrayType::scalar(F32));
        let pair = PrefixBroadcast::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };

        assert_eq!(leaf.broadcast_to(&pair), Ok(pair.clone()));
        assert_eq!(leaf.broadcast(&pair), Ok(pair.clone()));
        assert!(leaf.is_broadcastable_to(&pair));
    }

    #[test]
    fn test_incompatible_prefix_broadcastable_parameter_structures() {
        let wrapped = PrefixBroadcast::Wrapped { inner: ArrayType::scalar(F32) };
        let pair = PrefixBroadcast::Pair {
            left: ArrayType::new(F32, Shape::new(vec![2.into(), 1.into()])),
            right: ArrayType::new(F32, Shape::new(vec![1.into(), 3.into()])),
        };

        assert!(matches!(
            wrapped.broadcast_to(&pair),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));
        assert!(matches!(
            wrapped.broadcast(&pair),
            Err(BroadcastingError::ParameterError(ParameterError::MissingParameters { .. })),
        ));
        assert!(!wrapped.is_broadcastable_to(&pair));
    }
}
