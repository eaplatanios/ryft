use std::convert::Infallible;
use std::sync::Arc;

use ryft_core::programs::transforms::{Transform, TransformArtifact};
use ryft_core::{
    Array, ArrayOperation, ArrayType, DataType, Operation, Placeholder, Program, ProgramBuilder, Region, Value,
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
