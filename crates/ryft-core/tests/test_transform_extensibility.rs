use std::borrow::Cow;
use std::convert::Infallible;
use std::fmt::{Display, Formatter};
use std::sync::Arc;

use ryft_core::programs::transforms::{Transform, TransformArtifact};
use ryft_core::{
    Accuracy, Array, ArrayOperation, ArrayType, Context, DataType, EagerContext, LogicalMesh, ManualVariationAlignment,
    Operation, ParallelVary, Parameter, Placeholder, Program, ProgramBuilder, ProgramError, Region, Sin, SinOperation,
    Typed, Value, ValueDirectDispatch,
};

/// External transform arguments selecting one independently retained identity artifact.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExternalArguments {
    /// User-defined semantic variant of the transform.
    variant: usize,
}

/// External transform marker defined entirely outside `ryft-core`'s library target.
struct ExternalIdentityTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for ExternalIdentityTransform {
    type Arguments = ExternalArguments;
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 2;
}

/// A second external marker proving that marker identity namespaces otherwise identical keys and artifacts.
struct OtherExternalIdentityTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for OtherExternalIdentityTransform {
    type Arguments = ExternalArguments;
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 1;
}

/// Builds the source program used by the downstream extension test.
fn identity_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
    let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
    let input = builder.add_input(ArrayType::scalar(DataType::F64));
    builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
}

/// Derives an external identity transform while preserving its argument as metadata.
fn derive_identity(
    region: ryft_core::RegionRef<'_, Array, ArrayOperation<Array>>,
    arguments: &ExternalArguments,
) -> Result<TransformArtifact<Array, ArrayOperation<Array>, usize>, Infallible> {
    Ok(TransformArtifact::new(vec![Arc::new(region.to_program())], arguments.variant))
}

#[test]
fn test_external_region_transform_uses_public_cache_extension_point() {
    let program = identity_program();
    let first = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    let repeated = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(Arc::ptr_eq(&first.programs()[0], &repeated.programs()[0]));

    let distinct_arguments = program
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 1 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &distinct_arguments.programs()[0]));
    assert_eq!(distinct_arguments.metadata(), &1);

    let other_marker = program
        .entry_region_ref()
        .transform::<OtherExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &other_marker.programs()[0]));

    let cloned = program.clone();
    let from_clone = cloned
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(Arc::ptr_eq(&first.programs()[0], &from_clone.programs()[0]));

    let independent = identity_program();
    let from_independent = independent
        .entry_region_ref()
        .transform::<ExternalIdentityTransform, _, Infallible>(ExternalArguments { variant: 0 }, derive_identity)
        .unwrap();
    assert!(!Arc::ptr_eq(&first.programs()[0], &from_independent.programs()[0]));
}

/// Downstream value that implements capabilities directly even though its domain can bind the operations that the
/// domain-binding blanket capability implementations would bind, which only the [`ValueDirectDispatch`] marker keeps
/// disjoint from those blanket implementations.
#[derive(Clone, Debug, PartialEq)]
struct DirectArray {
    /// Host array that this value wraps.
    array: Array,
}

impl Display for DirectArray {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.array, formatter)
    }
}

impl Parameter for DirectArray {}

impl Typed for DirectArray {
    type Type = ArrayType;

    fn r#type(&self) -> Cow<'_, ArrayType> {
        self.array.r#type()
    }
}

impl Value for DirectArray {
    type Dispatch = ValueDirectDispatch;
    type Domain = EagerContext<Self, SinOperation<ArrayType>>;

    fn domain(&self) -> Self::Domain {
        EagerContext::new()
    }
}

// The domain's operation family converts from `SinOperation`, so the blanket `Sin` implementation would also apply to
// this value (i.e., a conflicting implementation, E0119) if the value did not use `ValueDirectDispatch`.
impl Sin for DirectArray {
    fn sin_with_accuracy(&self, accuracy: Accuracy) -> Result<Self, ProgramError> {
        Ok(Self { array: self.array.sin_with_accuracy(accuracy)? })
    }
}

// Composition blanket implementations, such as `ManualVariationAlignment`, still apply to direct values, and this one
// requires a direct `ParallelVary` implementation, because the domain-binding `ParallelVary` blanket does not apply.
impl ParallelVary for DirectArray {
    fn parallel_vary(&self, axis_name: &str) -> Result<Self, ProgramError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!("`parallel_vary` requires an active manual mesh axis `{axis_name}`"),
        })
    }

    fn parallel_vary_on_mesh(&self, axis_name: &str, _mesh: &LogicalMesh) -> Result<Self, ProgramError> {
        self.parallel_vary(axis_name)
    }
}

#[test]
fn test_direct_dispatch_value_implements_domain_binding_capabilities_directly() {
    let value = DirectArray { array: Array::scalar(0.5f64).unwrap() };
    let expected = DirectArray { array: Array::scalar(0.5f64.sin()).unwrap() };

    // Capability calls use the direct implementation.
    assert_eq!(value.sin(), Ok(expected.clone()));

    // The domain binds the same operation, and its interpretation bottoms out in the direct implementation.
    assert_eq!(
        value.domain().bind(SinOperation::<ArrayType>::new(), Vec::new(), std::slice::from_ref(&value)),
        Ok(vec![expected]),
    );

    // Composition blanket implementations still apply, and values without manual variation are already aligned.
    let other = DirectArray { array: Array::scalar(2.0f64).unwrap() };
    assert_eq!(
        ManualVariationAlignment::align_manual_variation(&[value.clone(), other.clone()]),
        Ok(vec![value, other]),
    );
}
