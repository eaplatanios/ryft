use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::arrays::dimensions::DimensionValue;
use crate::arrays::operations::ArrayIrOperation;
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::dimensions::DimensionType;
use crate::arrays::types::ir::ArrayIrType;
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::{
    Concretizable, ProgramError, Type, TypeError, TypeIdentityRenaming, Typed, Value, ValueProjection,
};

/// [`Value`]-level counterpart to [`ArrayIrType`] that is used by [`Program`](crate::Program)s that may contain
/// both [`ArrayType`]-typed [`Value`]s and [`DimensionValue`]. `A` is the concrete array representation selected by the
/// owning backend. Dimensions use the common [`DimensionValue`] which is a checked host representation, so that eager
/// dimension arithmetic remains host integer work and does not allocate arrays or dispatch to device backends.
///
/// This type allows arrays and checked host-side dimensions to share one storage universe, while [`ValueProjection`]
/// lets homogeneous [`Operation`](crate::Operation) machinery borrow or consume only the member it understands.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayIrValue<A: Value<Type = ArrayType>> {
    /// Ordinary backend [`ArrayType`]-typed [`Value`].
    Array(A),

    /// Checked host-side runtime [`DimensionValue`].
    Dimension(DimensionValue),
}

impl<A: Value<Type = ArrayType>> Display for ArrayIrValue<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
        }
    }
}

impl<A: Value<Type = ArrayType>> Typed for ArrayIrValue<A> {
    type Type = ArrayIrType;

    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayIrType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayIrType::Dimension(value.r#type().into_owned()),
        })
    }
}

impl<A: Value<Type = ArrayType>> Value for ArrayIrValue<A> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self, ArrayIrOperation<A>>;

    #[inline]
    fn dispatch_domain(&self) -> Self::DispatchDomain {
        EagerContext::new()
    }

    #[inline]
    fn execution_domain(&self) -> Self::ExecutionDomain {
        EagerContext::new()
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<ArrayType> for ArrayIrValue<A> {
    type Projected = A;
    type ProjectedRef<'v>
        = &'v A
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: A) -> Self {
        Self::Array(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v A, TypeError>
    where
        ArrayType: 'v,
    {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<A, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            Self::Dimension(_) => Err(TypeError::invalid("expected array type but got dimension type")),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<DimensionType> for ArrayIrValue<A> {
    type Projected = DimensionValue;
    type ProjectedRef<'v>
        = &'v DimensionValue
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v DimensionValue, TypeError>
    where
        DimensionType: 'v,
    {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Array(_) => Err(TypeError::invalid("expected dimension type but got array type")),
            Self::Dimension(value) => Ok(value),
        }
    }
}

impl<A: Value<Type = ArrayType>> From<A> for ArrayIrValue<A> {
    #[inline]
    fn from(value: A) -> Self {
        Self::Array(value)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionValue> for ArrayIrValue<A> {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

impl<A: Concretizable<bool> + Value<Type = ArrayType>> Concretizable<bool> for ArrayIrValue<A> {
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(value) => value.concretize(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a first-class dimension `{value}`"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, RandomState};

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionVariable};
    use crate::contexts::StagingContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::ConstantOperation;
    use crate::partial::PartialTracer;
    use crate::tracing::{Tracer, TracingContext};

    use super::*;

    #[test]
    fn test_array_ir_dimension_values_share_one_abstract_type() {
        // The retained-JIT dispatch key is built from `Typed::r#type` of each input, so dimension values with different
        // runtime extents must report one identical abstract type: a `DimensionType` is strictly identity plus bounds,
        // and concrete extents never participate in structural type equality, hashing, or display. Otherwise, every
        // concrete extent would acquire its own compiled specialization, turning a runtime dynamic dimension back into
        // a static specialization parameter.
        let extent_type =
            DimensionType::new(DimensionVariable::new("extent", DimensionBounds::new(1, Some(9)).unwrap()));
        let three = ArrayIrValue::<Array>::Dimension(DimensionValue::new(extent_type.clone(), 3).unwrap());
        let four = ArrayIrValue::<Array>::Dimension(DimensionValue::new(extent_type.clone(), 4).unwrap());
        assert_eq!(three.r#type().into_owned(), ArrayIrType::Dimension(extent_type));
        assert_eq!(three.r#type().into_owned(), four.r#type().into_owned());
        assert_eq!(three.r#type().to_string(), four.r#type().to_string());
        let hasher = RandomState::new();
        assert_eq!(hasher.hash_one(three.r#type().as_ref()), hasher.hash_one(four.r#type().as_ref()));
    }

    #[test]
    fn test_array_ir_value_projection() {
        let array = Array::vector((0..4096).map(|value| value as f32).collect());
        let payload = array.storage_bytes().as_ptr();
        let stored = ArrayIrValue::Array(array);
        let projected = <ArrayIrValue<Array> as ValueProjection<ArrayType>>::projected(&stored).unwrap();
        assert_eq!(projected.storage_bytes().as_ptr(), payload);
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<DimensionType>>::projected(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
        let projected = <ArrayIrValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap();
        assert_eq!(projected.storage_bytes().as_ptr(), payload);
    }

    #[test]
    fn test_array_ir_dimension_projection() {
        let variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let dimension = DimensionValue::new(DimensionType::new(variable), 4).unwrap();
        let stored = ArrayIrValue::<Array>::Dimension(dimension.clone());
        assert_eq!(<ArrayIrValue<Array> as ValueProjection<DimensionType>>::projected(&stored), Ok(&dimension),);
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ArrayType>>::projected(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(<ArrayIrValue<Array> as ValueProjection<DimensionType>>::into_projected(stored), Ok(dimension),);
    }

    #[test]
    fn test_array_ir_tracer_projection_preserves_ssa_identity() {
        type TestContext = TracingContext<ArrayIrValue<Array>, ConstantOperation<ArrayIrValue<Array>>>;

        let context = TestContext::new();
        let tracer = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let atom = tracer.atom_id().unwrap();
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::projected(&tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        assert_eq!(<Tracer<TestContext> as ValueProjection<ArrayType>>::from_projected(projected).atom_id(), Ok(atom),);

        fn assert_projection<V: ValueProjection<ArrayType>>() {}
        assert_projection::<PartialTracer<TestContext>>();
        assert_projection::<DifferentiationTracer<TestContext>>();
    }
}
