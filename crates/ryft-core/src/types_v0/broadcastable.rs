//! Broadcasting support for type descriptors with nested [`ArrayType`] leaves.

use thiserror::Error;

use crate::parameters::{ParameterError, Parameterized};
use crate::types::Type;

use super::array_type::{ArrayType, ArrayTypeBroadcastingError};

/// Type descriptors that can be broadcast together by broadcasting each nested [`ArrayType`] leaf independently.
///
/// This trait replaces the old `ArrayStructureType` abstraction. Implementations are defined in terms of
/// [`Parameterized<ArrayType>`]: values are flattened into their nested [`ArrayType`] leaves, those leaves are
/// broadcast position-wise with [`ArrayType::broadcast`], and the result is reconstructed using the original
/// parameter structure.
pub trait Broadcastable: Sized {
    /// Broadcasts the provided values into a single compatible value.
    ///
    /// All inputs must share the same [`Parameterized`] structure. Once that structural check passes, broadcasting is
    /// performed independently for each nested [`ArrayType`] leaf using [`ArrayType::broadcast`].
    fn broadcast(values: &[&Self]) -> Result<Self, BroadcastingError>;
}

impl<T> Broadcastable for T
where
    T: Clone + Parameterized<ArrayType>,
    T::ParameterStructure: Clone + PartialEq,
{
    fn broadcast(values: &[&Self]) -> Result<Self, BroadcastingError> {
        if values.is_empty() {
            return Err(BroadcastingError::Empty);
        }

        let structure = values[0].parameter_structure();
        let parameter_count = structure.parameter_count();
        let array_types_per_value = values
            .iter()
            .map(|value| {
                if value.parameter_structure() != structure {
                    return Err(BroadcastingError::IncompatibleStructures);
                }

                Ok(value.parameters().collect::<Vec<_>>())
            })
            .collect::<Result<Vec<_>, BroadcastingError>>()?;

        let mut broadcast_array_types = Vec::with_capacity(parameter_count);
        for index in 0..parameter_count {
            let array_types = array_types_per_value.iter().map(|types| types[index]).collect::<Vec<_>>();
            broadcast_array_types.push(ArrayType::broadcast(array_types.as_slice())?);
        }

        Ok(Self::from_parameters(structure, broadcast_array_types)?)
    }
}

/// Error returned by [`Broadcastable::broadcast`].
#[derive(Error, Clone, Debug, Eq, PartialEq, Hash)]
pub enum BroadcastingError {
    /// Error returned when attempting to broadcast an empty collection of values.
    #[error("cannot broadcast an empty collection of values")]
    Empty,

    /// Error returned when the provided values do not share the same [`Parameterized`] structure.
    #[error("broadcast requires all inputs to share the same parameter structure")]
    IncompatibleStructures,

    /// Error returned when broadcasting one of the nested [`ArrayType`] leaves fails.
    #[error("{0}")]
    ArrayTypeBroadcastingError(#[from] ArrayTypeBroadcastingError),

    /// Error returned when reconstruction of the broadcast result fails.
    #[error("{0}")]
    ParameterError(#[from] ParameterError),
}

#[cfg(test)]
mod tests {
    use crate::differentiation::JvpTracer;
    use crate::types::data_type::DataType::*;
    use crate::types_v0::array_type::{Shape, Size};

    use super::*;

    #[test]
    fn test_array_type_broadcastable_broadcast() {
        let t0 = ArrayType::new(F32, Shape::new(vec![42.into(), 4.into()]));
        let t1 = ArrayType::new(F32, Shape::new(vec![1.into(), 4.into()]));
        let t2 = ArrayType::scalar(Boolean);
        let t3 = ArrayType::new(F32, Shape::new(vec![5.into(), 3.into()]));

        assert_eq!(<ArrayType as Broadcastable>::broadcast(&[&t0]), Ok(t0.clone()));
        assert_eq!(<ArrayType as Broadcastable>::broadcast(&[&t1, &t2]), Ok(t1.clone()));
        assert_eq!(<ArrayType as Broadcastable>::broadcast(&[&t2, &t1]), Ok(t1.clone()));

        assert!(matches!(<ArrayType as Broadcastable>::broadcast(&[]), Err(BroadcastingError::Empty)));
        assert!(matches!(
            <ArrayType as Broadcastable>::broadcast(&[&t0, &t3]),
            Err(BroadcastingError::ArrayTypeBroadcastingError(_)),
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

        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0]), Ok(j0.clone()));
        assert_eq!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0, &j1]), Ok(j2));

        assert!(matches!(JvpTracer::<ArrayType, ArrayType>::broadcast(&[]), Err(BroadcastingError::Empty)));
        assert!(matches!(
            JvpTracer::<ArrayType, ArrayType>::broadcast(&[&j0, &j3]),
            Err(BroadcastingError::ArrayTypeBroadcastingError(_)),
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

        assert_eq!(NestedJvpTracer::broadcast(&[&j0]), Ok(j0.clone()));
        assert_eq!(NestedJvpTracer::broadcast(&[&j0, &j1]), Ok(j2));
        assert!(matches!(
            NestedJvpTracer::broadcast(&[&j0, &j3]),
            Err(BroadcastingError::ArrayTypeBroadcastingError(_)),
        ));
    }
}
