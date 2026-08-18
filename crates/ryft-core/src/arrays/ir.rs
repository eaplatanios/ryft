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
use crate::programs::values::rename_type_identities_by_rejection;
use crate::programs::{
    Concretizable, Operation, ProgramError, Reference, ReferenceType, RegionRef, Type, TypeError, TypeIdentityRenaming,
    Typed, Value, ValueProjection,
};

/// [`Value`]-level counterpart to [`ArrayIrType`] that is used by [`Program`](crate::Program)s that may contain
/// [`ArrayType`]-typed [`Value`]s, [`DimensionValue`]s, and identity-bearing array [`Reference`]s. `A` is the concrete
/// array representation selected by the owning backend. Dimensions use the common [`DimensionValue`] which is a checked
/// host representation, so that eager dimension arithmetic remains host integer work and does not allocate arrays or
/// dispatch to device backends.
///
/// This type lets arrays, checked host-side dimensions, and identity-bearing array references share one storage
/// universe, while [`ValueProjection`] lets homogeneous [`Operation`] machinery borrow or consume only the member it
/// understands. Reference operations remain composite-native because their signatures cross member kinds. Ordinary
/// numeric operations still project only the array member.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum ArrayIrValue<A: Value<Type = ArrayType>> {
    /// Ordinary backend [`ArrayType`]-typed [`Value`].
    Array(A),

    /// Checked host-side runtime [`DimensionValue`].
    Dimension(DimensionValue),

    /// Identity-bearing [`Reference`] to an ordinary backend [`ArrayType`]-typed [`Value`].
    Reference(Reference<A>),
}

impl<A: Value<Type = ArrayType>> ArrayIrValue<A> {
    /// Returns this composite member's diagnostic kind name.
    fn kind_name(&self) -> &'static str {
        match self {
            Self::Array(_) => "array",
            Self::Dimension(_) => "dimension",
            Self::Reference(_) => "reference",
        }
    }
}

impl<A: Value<Type = ArrayType>> Display for ArrayIrValue<A> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Array(value) => Display::fmt(value, formatter),
            Self::Dimension(value) => Display::fmt(value, formatter),
            Self::Reference(value) => Display::fmt(value, formatter),
        }
    }
}

impl<A: Value<Type = ArrayType>> Typed for ArrayIrValue<A> {
    type Type = ArrayIrType;

    fn r#type(&self) -> Cow<'_, ArrayIrType> {
        Cow::Owned(match self {
            Self::Array(value) => ArrayIrType::Array(value.r#type().into_owned()),
            Self::Dimension(value) => ArrayIrType::Dimension(value.r#type().into_owned()),
            Self::Reference(value) => ArrayIrType::Reference(value.r#type().into_owned()),
        })
    }
}

impl<A: Value<Type = ArrayType>> Value for ArrayIrValue<A> {
    const VALIDATES_EAGER_REPLAY: bool = true;

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

    fn validate_eager_replay<V: Value<Type = Self::Type>, O: Operation<Type = Self::Type>>(
        region: RegionRef<'_, V, O>,
    ) -> Result<(), ProgramError> {
        // TODO(eaplatanios): Phase 4 capture lifting must thread the lifted-capture count here so that external-root
        //  diagnostics name captures instead of public inputs.
        let analysis = region.analyze_references(0)?;
        if let Some(external) = analysis.external_roots().first() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "program replay of external reference {} is not supported before external holder runtime \
                     integration",
                    external.source(),
                ),
            });
        }
        Ok(())
    }

    fn rename_type_identities(
        &self,
        renaming: &TypeIdentityRenaming<<Self::Type as Type>::Identity>,
    ) -> Result<Self, TypeError> {
        match self {
            Self::Array(value) => Ok(Self::Array(value.rename_type_identities(renaming)?)),
            Self::Dimension(value) => Ok(Self::Dimension(value.rename_type_identities(renaming)?)),
            Self::Reference(value) => {
                // TODO(eaplatanios): Add a handle-local identity/view mapping that can reconstruct values crossing the
                //  boundary between this handle's type and the root-shared stored value. The holder and handle metadata
                //  are already separate, but changing only the handle type would make reads claim a renamed type while
                //  returning the root's original value. Once this mapping exists, inline
                //  `rename_type_identities_by_rejection` back into `Value::rename_type_identities` and remove it.
                rename_type_identities_by_rejection(value, renaming).map(Self::Reference)
            }
        }
    }

    fn validate_as_constant(&self) -> Result<(), TypeError> {
        match self {
            Self::Array(_) | Self::Dimension(_) => Ok(()),
            Self::Reference(_) => {
                // A reference holder's runtime identity is process-local and deliberately absent from its deterministic
                // rendering, so storing one as a program constant would let two programs over distinct holders render
                // (and therefore fingerprint) identically. External references enter programs through inputs or
                // captures instead.
                Err(TypeError::invalid(
                    "reference values cannot be stored as program constants; pass external references through program \
                     inputs or captures instead",
                ))
            }
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
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind_name()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<A, TypeError> {
        match self {
            Self::Array(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected array type but got {} type", other.kind_name()))),
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
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind_name()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<DimensionValue, TypeError> {
        match self {
            Self::Dimension(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected dimension type but got {} type", other.kind_name()))),
        }
    }
}

impl<A: Value<Type = ArrayType>> ValueProjection<ReferenceType<ArrayType>> for ArrayIrValue<A> {
    type Projected = Reference<A>;
    type ProjectedRef<'v>
        = &'v Reference<A>
    where
        Self: 'v;

    #[inline]
    fn from_projected(value: Reference<A>) -> Self {
        Self::Reference(value)
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<&'v Reference<A>, TypeError>
    where
        ReferenceType<ArrayType>: 'v,
    {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind_name()))),
        }
    }

    #[inline]
    fn into_projected(self) -> Result<Reference<A>, TypeError> {
        match self {
            Self::Reference(value) => Ok(value),
            other => Err(TypeError::invalid(format!("expected reference type but got {} type", other.kind_name()))),
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

impl<A: Value<Type = ArrayType>> From<Reference<A>> for ArrayIrValue<A> {
    #[inline]
    fn from(value: Reference<A>) -> Self {
        Self::Reference(value)
    }
}

impl<A: Concretizable<bool> + Value<Type = ArrayType>> Concretizable<bool> for ArrayIrValue<A> {
    fn concretize(&self) -> Result<bool, ProgramError> {
        match self {
            Self::Array(value) => value.concretize(),
            Self::Dimension(value) => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from a first-class dimension `{value}`"),
            }),
            Self::Reference(value) => Err(ProgramError::Concretization {
                message: format!("cannot extract a concrete boolean from reference `{value}`"),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::hash::{BuildHasher, RandomState};

    use pretty_assertions::assert_eq;

    use crate::arrays::arrays::Array;
    use crate::arrays::operations::{ArrayIrOperation, ArrayOperation};
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, DimensionBounds, DimensionVariable, Shape};
    use crate::contexts::StagingContext;
    use crate::differentiation::DifferentiationTracer;
    use crate::operations::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialTracer;
    use crate::programs::{Atom, Program, ProgramBuilder, Region, RegionArena, RegionId};
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
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&stored),
            Err(TypeError::invalid("expected reference type but got array type")),
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
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&stored),
            Err(TypeError::invalid("expected reference type but got dimension type")),
        );
        assert_eq!(<ArrayIrValue<Array> as ValueProjection<DimensionType>>::into_projected(stored), Ok(dimension),);
    }

    #[test]
    fn test_array_ir_reference_projection_preserves_holder_identity() {
        let reference = Reference::new(Array::vector(vec![1.0_f32, 2.0]));
        let stored = ArrayIrValue::Reference(reference.clone());
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::projected(&stored),
            Ok(&reference),
        );
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ArrayType>>::projected(&stored),
            Err(TypeError::invalid("expected array type but got reference type")),
        );
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<DimensionType>>::projected(&stored),
            Err(TypeError::invalid("expected dimension type but got reference type")),
        );
        assert_eq!(
            <ArrayIrValue<Array> as ValueProjection<ReferenceType<ArrayType>>>::into_projected(stored),
            Ok(reference),
        );
    }

    #[test]
    fn test_array_ir_reference_values_reject_program_constant_storage() {
        // A reference holder's identity is deliberately absent from its deterministic rendering, so storing one as
        // a program constant would let two programs over distinct holders render (and fingerprint) identically. The
        // rejection lives at region sealing (i.e., the one boundary every construction path crosses) which this test
        // exercises through the builder path, direct `RegionArena::from_regions` construction, and public
        // `Program::new`. Sealing validates every region in the arena through the same path, so nested regions
        // are covered by the same mechanism.
        let expected_error = Err(TypeError::invalid(
            "reference values cannot be stored as program constants; pass external references through program \
            inputs or captures instead",
        )
        .into());

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let constant = builder.add_constant(ArrayIrValue::Reference(Reference::new(Array::scalar(1.0_f32))));
        assert_eq!(
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    vec![constant],
                    Vec::new(),
                    vec![Placeholder],
                )
                .map(|_| ()),
            expected_error,
        );

        let reference_region = || {
            Region::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new(
                vec![Atom::Constant(ArrayIrValue::Reference(Reference::new(Array::scalar(1.0_f32))))],
                Vec::new(),
                Vec::new(),
                Vec::new(),
            )
        };
        assert_eq!(RegionArena::from_regions(vec![reference_region()]).map(|_| ()), expected_error);
        type TestValue = ArrayIrValue<Array>;
        type TestProgram = Program<TestValue, ArrayIrOperation<Array>, Vec<TestValue>, Vec<TestValue>>;
        assert_eq!(
            TestProgram::new(Vec::new(), Vec::new(), vec![reference_region()], RegionId::new(0)).map(|_| ()),
            expected_error,
        );
    }

    #[test]
    fn test_array_ir_reference_value_identity_renaming() {
        type TestContext = TracingContext<Array, ArrayOperation<Array>>;

        let bounds = DimensionBounds::positive(Some(9)).unwrap();
        let source = DimensionVariable::new("source", bounds);
        let target = DimensionVariable::new("target", bounds);
        let source_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(source.clone())]));
        let context = TestContext::new();
        let reference = Reference::new(context.input(source_type));
        let value = ArrayIrValue::Reference(reference.clone());

        // An identity renaming preserves the concrete holder and therefore every alias of it.
        let unchanged = value.rename_type_identities(&TypeIdentityRenaming::new()).unwrap();
        assert_eq!(
            <ArrayIrValue<Tracer<TestContext>> as ValueProjection<ReferenceType<ArrayType>>>::projected(&unchanged),
            Ok(&reference),
        );

        // Renaming handle-local type metadata is deliberately deferred until the view-aware carrier lands.
        // It must not mutate every alias or mint a new resource identity in the meantime.
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(source, target).unwrap();
        let reference_type = reference.r#type();
        assert_eq!(
            value.rename_type_identities(&renaming),
            Err(TypeError::invalid(format!(
                "cannot rename type identities in value of type {} without a value-specific reconstruction \
                 implementation",
                reference_type.as_ref(),
            ))),
        );
        assert_eq!(
            <ArrayIrValue<Tracer<TestContext>> as ValueProjection<ReferenceType<ArrayType>>>::projected(&value),
            Ok(&reference),
        );
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

        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let tracer = context.input(ArrayIrType::Reference(reference_type));
        let atom = tracer.atom_id().unwrap();
        let projected = <Tracer<TestContext> as ValueProjection<ReferenceType<ArrayType>>>::projected(&tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        let projected =
            <Tracer<TestContext> as ValueProjection<ReferenceType<ArrayType>>>::into_projected(tracer).unwrap();
        assert_eq!(projected.value().atom_id(), Ok(atom));
        assert_eq!(
            <Tracer<TestContext> as ValueProjection<ReferenceType<ArrayType>>>::from_projected(projected).atom_id(),
            Ok(atom),
        );

        fn assert_projection<V: ValueProjection<ArrayType>>() {}
        fn assert_reference_projection<V: ValueProjection<ReferenceType<ArrayType>>>() {}
        assert_projection::<PartialTracer<TestContext>>();
        assert_projection::<DifferentiationTracer<TestContext>>();
        assert_reference_projection::<PartialTracer<TestContext>>();
        assert_reference_projection::<DifferentiationTracer<TestContext>>();
    }
}
