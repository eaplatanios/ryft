//! Contains heterogeneous value storage for array programs with first-class runtime dimensions.
//!
//! [`ArrayProgramValue`] is the value-level counterpart of [`ArrayProgramType`](crate::ArrayProgramType). It keeps
//! arrays and checked host-side dimensions in one storage universe while [`ValueProjection`] lets homogeneous
//! operation machinery borrow or consume only the member it understands.

use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;

use crate::backends::dimensions::DimensionValue;
use crate::contexts::EagerContext;
use crate::parameters::Parameter;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::types::{ArrayProgramType, ArrayType, DimensionType};

/// Value stored by an array program whose graph may contain arrays and first-class runtime dimensions.
///
/// `A` is the concrete array representation selected by the owning backend. Runtime dimensions use the common checked
/// host representation, so eager dimension arithmetic remains host integer work and does not allocate an array or
/// dispatch to a device backend.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayProgramValue<A: Value<Type = ArrayType>> {
    /// An ordinary backend array value.
    Array(A),

    /// A checked host-side runtime dimension value.
    Dimension(DimensionValue),
}

impl<A: Value<Type = ArrayType>> Display for ArrayProgramValue<A> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
        }
    }
}

impl<A: Value<Type = ArrayType>> Typed for ArrayProgramValue<A> {
    type Type = ArrayProgramType;

    fn r#type(&self) -> Cow<'_, ArrayProgramType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayProgramType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayProgramType::Dimension(value.r#type().clone()),
        })
    }
}

impl<A: Value<Type = ArrayType>> Value for ArrayProgramValue<A> {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

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

impl<A: Value<Type = ArrayType>> ValueProjection<ArrayType> for ArrayProgramValue<A> {
    type Projected = A;
    type ProjectedRef<'a>
        = &'a A
    where
        Self: 'a;

    #[inline]
    fn project_ref(&self) -> Result<&A, TypeError> {
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

    #[inline]
    fn lift(value: A) -> Self {
        Self::Array(value)
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<DimensionType> for ArrayProgramValue<A> {
    type Projected = DimensionValue;
    type ProjectedRef<'a>
        = &'a DimensionValue
    where
        Self: 'a;

    #[inline]
    fn project_ref(&self) -> Result<&DimensionValue, TypeError> {
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

    #[inline]
    fn lift(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

impl<A: Value<Type = ArrayType>> From<A> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: A) -> Self {
        Self::Array(value)
    }
}

impl<A: Value<Type = ArrayType>> From<DimensionValue> for ArrayProgramValue<A> {
    #[inline]
    fn from(value: DimensionValue) -> Self {
        Self::Dimension(value)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::Array;
    use crate::contexts::StagingContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::constants::ConstantOperation;
    use crate::partial::PartialTracer;
    use crate::programs::types::TypeProjection;
    use crate::tracing::{Tracer, TracingContext};
    use crate::types::{DataType, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    #[test]
    fn test_array_program_value_projection() {
        let array = Array::vector((0..4096).map(|value| value as f32).collect());
        let payload = array.values().as_ptr();
        let stored = ArrayProgramValue::Array(array);

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::project_ref(&stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<DimensionType>>::project_ref(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );

        let projected = <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::into_projected(stored).unwrap();
        assert_eq!(projected.values().as_ptr(), payload);
    }

    #[test]
    fn test_array_program_dimension_projection() {
        let variable = DimensionVariable::new("extent", DimensionBounds::positive(Some(9)).unwrap());
        let dimension = DimensionValue::new(DimensionType::new(variable), 4).unwrap();
        let stored = ArrayProgramValue::<Array>::Dimension(dimension.clone());

        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::project_ref(&stored), Ok(&dimension),);
        assert_eq!(
            <ArrayProgramValue<Array> as ValueProjection<ArrayType>>::project_ref(&stored),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(<ArrayProgramValue<Array> as ValueProjection<DimensionType>>::into_projected(stored), Ok(dimension),);
    }

    #[test]
    fn test_array_program_type_projection() {
        let array = ArrayType::new(DataType::F32, Shape::scalar());
        let stored = ArrayProgramType::Array(array.clone());
        assert_eq!(<ArrayProgramType as TypeProjection<ArrayType>>::project(&stored), Ok(&array));
        assert_eq!(
            <ArrayProgramType as TypeProjection<DimensionType>>::project(&stored),
            Err(TypeError::invalid("expected dimension type but got array type")),
        );
    }

    #[test]
    fn test_symbolic_value_projection_preserves_ssa_identity() {
        type TestContext = TracingContext<ArrayProgramValue<Array>, ConstantOperation<ArrayProgramValue<Array>>>;

        let context = TestContext::new();
        let tracer = context.input(ArrayProgramType::Array(ArrayType::scalar(DataType::F32)));
        let atom = tracer.atom_id().unwrap();
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::project_ref(&tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        let projected = <Tracer<TestContext> as ValueProjection<ArrayType>>::into_projected(tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        assert_eq!(<Tracer<TestContext> as ValueProjection<ArrayType>>::lift(projected).atom_id(), Ok(atom),);

        fn assert_projection<V: ValueProjection<ArrayType>>() {}
        assert_projection::<PartialTracer<TestContext>>();
        assert_projection::<DifferentiationTracer<TestContext>>();
    }
}
