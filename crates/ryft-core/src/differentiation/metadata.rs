use std::ops::Range;

use crate::contexts::Context;
use crate::programs::{Operation, ProgramError, Type, Value};

/// Selects the uses for which a [`DifferentialMetadataProvider`] captures runtime metadata. Validation-only callers
/// do not retain values needed solely to reconstruct a differential later.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentialMetadataPurpose {
    /// Checks supplied or returned differentials against their primal's runtime metadata.
    Validation,

    /// Supplies runtime information needed to reconstruct differentials, including concrete structural zeros.
    Reconstruction,

    /// Shares one capture between runtime compatibility checks and later reconstruction.
    ValidationAndReconstruction,
}

impl DifferentialMetadataPurpose {
    /// Returns whether this [`DifferentialMetadataPurpose`] includes differential validation.
    #[inline]
    pub fn validates(self) -> bool {
        matches!(self, Self::Validation | Self::ValidationAndReconstruction)
    }

    /// Returns whether this [`DifferentialMetadataPurpose`] includes differential reconstruction.
    #[inline]
    pub fn reconstructs(self) -> bool {
        matches!(self, Self::Reconstruction | Self::ValidationAndReconstruction)
    }
}

/// Captures and validates runtime metadata that relate differential values to their primals at one linearization point
/// in type universe `T`. A differential's type can omit runtime information needed to validate or reconstruct it.
/// Metadata supplies that information as ordinary family values, which may be eager values or tracers, with zero, one,
/// or several values per primal leaf. Implement this capability on the enclosing operation family, whose operations
/// read metadata and establish ordered checks.
///
/// [`DifferentiableType`](crate::DifferentiableType) owns tangent and cotangent type mappings, while
/// [`ResidualZeroProvider`](crate::ResidualZeroProvider) owns the residual protocol that constructs concrete zeros.
/// Families whose types completely determine compatibility need no validation metadata; reconstruction may still need
/// runtime sources consumed by that zero residual protocol.
///
/// Capture must preserve every occurrence and coordinate relationship required by the requested differential type,
/// without identifying independent payloads solely by type identity. Retain numerical primal storage only when the
/// family's reconstruction protocol requires an exemplar and provides no metadata-only representation. For example,
/// indexed array extents use a carrier without numerical primal storage; other type universes choose their own
/// representations. Array families implement this contract in their differentiation module; generic transforms do not
/// inspect array representations. Implementations must agree on capture order and validation semantics. There is no
/// blanket opt-in for arbitrary operation families because an empty capture would silently omit checks for types with
/// runtime metadata.
pub trait DifferentialMetadataProvider<T: Type>: Operation {
    /// Captures the metadata needed for `purpose` while `primal` is available. The requested `differential_type` may
    /// have a different element representation from the primal, while retaining its relevant runtime structure.
    /// Reconstruction captures must supply the sources needed by the family's zero residual protocol as well as any
    /// metadata needed to reconstruct live differentials. Validation captures must include zero-space values that a
    /// compact derivative program subsequently omits. An empty capture means this purpose needs no runtime data.
    ///
    /// Internal derivative rules transfer captured metadata into the tangent context only when reconstruction consumes
    /// it as a program residual.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which metadata reads are executed or staged.
    ///   - `primal`: Value whose runtime structure determines the differential's metadata.
    ///   - `differential_type`: Type of the differential to validate or reconstruct.
    ///   - `purpose`: Required uses of the captured metadata.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if the primal cannot supply the requested metadata or a metadata read fails.
    fn capture_differential_metadata<C: Context<Type = T, Operation = Self>>(
        context: &C,
        primal: &C::Value,
        differential_type: &T,
        purpose: DifferentialMetadataPurpose,
    ) -> Result<Vec<C::Value>, ProgramError>
    where
        Self: Operation<Type = T>;

    /// Checks `differential` against `metadata` captured at its primal's linearization point. The returned value must
    /// retain exactly the original differential type and depend on every staged check, including when later operations
    /// immediately project its numerical storage. Ignore inactive metadata padding where the representation defines it.
    /// Callers establish the required tangent or cotangent type before invoking this function; this function checks
    /// the remaining runtime relationships. The default preserves the value for families whose compatibility follows
    /// from the differential type alone.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which runtime checks are executed or staged.
    ///   - `differential`: Value to validate before use, compact-slot omission, or return to the caller.
    ///   - `primal_type`: Cached type of the corresponding primal boundary leaf.
    ///   - `metadata`: Captured values for that leaf, in the family's capture order.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] for incompatible runtime metadata or a failed check. Staged checks can instead fail
    /// when the resulting program executes.
    #[inline]
    fn validate_differential<C: Context<Type = T, Operation = Self>>(
        context: &C,
        differential: C::Value,
        primal_type: &T,
        metadata: &[C::Value],
    ) -> Result<C::Value, ProgramError>
    where
        Self: Operation<Type = T>,
    {
        let _ = context;
        let _ = primal_type;
        let _ = metadata;
        Ok(differential)
    }
}

/// Runtime metadata retained by reusable derivative callables beside their compact executable programs. Metadata tracks
/// complete public boundary positions, so zero-space leaves still take part in validation when their executable slots
/// are omitted. Empty captures allocate nothing; non-empty captures share one flat value vector across sparse leaf
/// ranges and both validation and reconstruction.
///
/// [`Pushforward::into_parts`](crate::Pushforward::into_parts) returns these collections so callers can preserve
/// runtime validation and reconstruction sources when interpreting the compact program themselves. The operation family
/// owns the captured values' representation; callers use [`validate`](Self::validate) for compatibility checks and
/// [`sources`](Self::sources) to obtain reconstruction sources for one public boundary leaf.
///
/// # Example
///
/// Consider a pushforward whose flattened primal outputs contain two static scalars and two indexed arrays. During
/// linearization, outputs are visited in public boundary order. The array-IR family captures one shape carrier for
/// each indexed array, preserving its runtime extents without numerical primal storage:
///
///   | Public Leaf | Primal Output         | Captured Values   |
///   |-------------|-----------------------|-------------------|
///   | 0           | Static Float Scalar   | `[]`              |
///   | 1           | Indexed Integer Array | `[integer_shape]` |
///   | 2           | Static Float Scalar   | `[]`              |
///   | 3           | Indexed Float Array   | `[float_shape]`   |
///
/// All four leaves need validation. Leaf 1 also needs reconstruction because its integer primal has a zero-space
/// tangent. Capturing leaf 1 with [`DifferentialMetadataPurpose::ValidationAndReconstruction`] and the other leaves
/// with [`DifferentialMetadataPurpose::Validation`] produces the following field contents, shown schematically:
///
/// ```text
/// values = [integer_shape, float_shape]
/// leaves = [
///     (1, 0..1, ValidationAndReconstruction),
///     (3, 1..2, Validation),
/// ]
/// ```
///
/// The ranges index metadata values, so leaf 3 uses `values[1..2]`. Leaves 0 and 2 have no entries because their
/// captures are empty. [`validate`](Self::validate) still calls the family's validator for those leaves with an empty
/// slice. For leaf 1, [`sources`](Self::sources) returns `[integer_shape]` to reconstruct its omitted zero tangent.
/// For leaf 3, `sources` returns an empty slice because its capture is for validation only; `validate` uses
/// `[float_shape]`. Leaf 3 keeps public index 3 even though it occupies output slot 2 in the compact program,
/// which omits leaf 1.
pub struct DifferentialBoundaryMetadata<V: Value> {
    /// Metadata values from non-empty leaf captures, concatenated in public boundary order. Each leaf's values retain
    /// the operation family's capture order and occupy one contiguous range.
    values: Vec<V>,

    /// Sparse entries `(public_boundary_index, value_range, purpose)`, ordered by increasing boundary index. Each range
    /// indexes [`values`](Self::values); leaves with empty captures have no entry.
    leaves: Vec<(usize, Range<usize>, DifferentialMetadataPurpose)>,
}

impl<V: Value> DifferentialBoundaryMetadata<V> {
    /// Returns the captured reconstruction sources for the public boundary leaf at `index`, in the operation family's
    /// capture order. Returns an empty slice if the leaf has no captured values or was captured only for validation.
    /// The index refers to the complete public boundary, including zero-space leaves omitted from the compact program.
    pub fn sources(&self, index: usize) -> &[V] {
        match self.leaves.binary_search_by_key(&index, |(index, _, _)| *index) {
            Ok(leaf) if self.leaves[leaf].2.reconstructs() => &self.values[self.leaves[leaf].1.clone()],
            _ => &[],
        }
    }

    /// Captures one public leaf, keeping non-empty entries in increasing boundary order.
    pub(crate) fn capture<C: Context<Value = V, Type = V::Type, Operation: DifferentialMetadataProvider<V::Type>>>(
        &mut self,
        context: &C,
        index: usize,
        primal: &V,
        differential_type: &V::Type,
        purpose: DifferentialMetadataPurpose,
    ) -> Result<(), ProgramError> {
        if self.leaves.last().is_some_and(|(previous, _, _)| *previous >= index) {
            return Err(ProgramError::MalformedProgram(
                "differential metadata leaves are not in boundary order".into(),
            ));
        }
        let values = C::Operation::capture_differential_metadata(context, primal, differential_type, purpose)?;
        if !values.is_empty() {
            let start = self.values.len();
            if self.values.is_empty() {
                self.values = values;
            } else {
                self.values.extend(values);
            }
            self.leaves.push((index, start..self.values.len(), purpose));
        }
        Ok(())
    }

    /// Validates `value` against the metadata captured for the public boundary leaf at `index`, returning the checked
    /// value. Calls the operation family's validator even when that leaf has no captured values. Callers must first
    /// establish the expected tangent or cotangent type from `primal_type`; this function checks runtime compatibility.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which runtime checks execute or are staged, using the family that captured
    ///     the metadata.
    ///   - `index`: Position in the complete public boundary, including omitted zero-space leaves.
    ///   - `value`: Differential value to validate before compact-slot omission, use, or return to the caller.
    ///   - `primal_type`: Type of the primal leaf for which this metadata was captured.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if runtime metadata is incompatible or a validation operation fails. Staged checks
    /// can instead fail when the resulting program executes.
    pub fn validate<C: Context<Value = V, Type = V::Type, Operation: DifferentialMetadataProvider<V::Type>>>(
        &self,
        context: &C,
        index: usize,
        value: V,
        primal_type: &V::Type,
    ) -> Result<V, ProgramError> {
        let metadata = match self.leaves.binary_search_by_key(&index, |(index, _, _)| *index) {
            Ok(leaf) => &self.values[self.leaves[leaf].1.clone()],
            Err(_) => &[],
        };
        C::Operation::validate_differential(context, value, primal_type, metadata)
    }
}

impl<V: Value> Default for DifferentialBoundaryMetadata<V> {
    #[inline]
    fn default() -> Self {
        Self { values: Vec::new(), leaves: Vec::new() }
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayType, DataType};
    use crate::contexts::{Domain, EagerContext, StagingContext};
    use crate::differentiation::types::DifferentiableType;
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::ConstantOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{RegionInterface, TypeError, Typed};
    use crate::tracing::TracingContext;

    use super::*;

    /// Captures distinguishable metadata values and validates their forwarding through the boundary collection.
    #[derive(Clone, Debug, ryft_macros::Operation)]
    #[ryft(type(ArrayType), constant(Array))]
    enum MetadataProbeOperation {
        Constant(ConstantOperation<Array>),
        Validate(MetadataValidationProbeOperation),
    }

    /// Requires the original differential, its primal type, and an ordered prefix of the probe's metadata markers.
    #[derive(Clone, Debug)]
    struct MetadataValidationProbeOperation {
        /// Primal type forwarded by the provider; its tangent type must match the differential's type.
        primal_type: ArrayType,
    }

    impl Operation for MetadataValidationProbeOperation {
        type Type = ArrayType;

        fn name(&self) -> &'static str {
            "metadata_validation_probe"
        }

        fn infer_output_types(
            &self,
            inputs: &[ArrayType],
            _regions: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            let differential =
                inputs.first().ok_or_else(|| TypeError::invalid("metadata probe needs a differential"))?;
            Ok(vec![differential.clone()])
        }
    }

    impl<C: Domain<Type = ArrayType, Value = Array>> InterpretableOperation<C> for MetadataValidationProbeOperation {
        fn interpret<D: InterpretationDriver<C>>(
            &self,
            _context: &C,
            _driver: &D,
            inputs: &[Array],
        ) -> Result<Vec<Array>, ProgramError> {
            let (differential, metadata) = inputs.split_first().unwrap();
            if metadata.is_empty() {
                return Err(ProgramError::InvalidArgument {
                    message: "metadata probe received no captured values".into(),
                });
            }
            assert_eq!(&self.primal_type, metadata[0].r#type().as_ref());
            assert_eq!(&self.primal_type.tangent().unwrap(), differential.r#type().as_ref());
            let expected_primal = if differential.r#type().data_type() == DataType::Zero {
                metadata[0].clone()
            } else {
                differential.clone()
            };
            let expected = [expected_primal, Array::scalar(7f32).unwrap(), Array::scalar(11f32).unwrap()];
            if metadata.len() > expected.len() || metadata != &expected[..metadata.len()] {
                return Err(ProgramError::InvalidArgument {
                    message: "metadata probe received incompatible values".into(),
                });
            }
            Ok(vec![differential.clone()])
        }
    }

    impl DifferentialMetadataProvider<ArrayType> for MetadataProbeOperation {
        fn capture_differential_metadata<C: Context<Type = ArrayType, Operation = Self>>(
            context: &C,
            primal: &C::Value,
            differential_type: &ArrayType,
            purpose: DifferentialMetadataPurpose,
        ) -> Result<Vec<C::Value>, ProgramError> {
            match differential_type.data_type() {
                DataType::Zero => return Ok(Vec::new()),
                DataType::F64 => {
                    return Err(ProgramError::InvalidArgument { message: "metadata capture probe failed".into() });
                }
                _ => {}
            }
            let markers: &[f32] = match purpose {
                DifferentialMetadataPurpose::Validation => &[],
                DifferentialMetadataPurpose::Reconstruction => &[7f32],
                DifferentialMetadataPurpose::ValidationAndReconstruction => &[7f32, 11f32],
            };
            let mut values = vec![primal.clone()];
            for marker in markers {
                values.push(
                    context.bind(ConstantOperation::new(Array::scalar(*marker).unwrap()), Vec::new(), &[])?.remove(0),
                );
            }
            Ok(values)
        }

        fn validate_differential<C: Context<Type = ArrayType, Operation = Self>>(
            context: &C,
            differential: C::Value,
            primal_type: &ArrayType,
            metadata: &[C::Value],
        ) -> Result<C::Value, ProgramError> {
            let inputs = std::iter::once(differential).chain(metadata.iter().cloned()).collect::<Vec<_>>();
            Ok(context
                .bind(MetadataValidationProbeOperation { primal_type: primal_type.clone() }, Vec::new(), &inputs)?
                .remove(0))
        }
    }

    #[test]
    fn test_differential_metadata_purpose_validates() {
        assert!(DifferentialMetadataPurpose::Validation.validates());
        assert!(!DifferentialMetadataPurpose::Reconstruction.validates());
        assert!(DifferentialMetadataPurpose::ValidationAndReconstruction.validates());
    }

    #[test]
    fn test_differential_metadata_purpose_reconstructs() {
        assert!(!DifferentialMetadataPurpose::Validation.reconstructs());
        assert!(DifferentialMetadataPurpose::Reconstruction.reconstructs());
        assert!(DifferentialMetadataPurpose::ValidationAndReconstruction.reconstructs());
    }

    #[test]
    fn test_differential_metadata_provider_validate_differential() {
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(type(ArrayType), constant(Array))]
        enum TypeOnlyOperation {
            Constant(ConstantOperation<Array>),
        }

        impl DifferentialMetadataProvider<ArrayType> for TypeOnlyOperation {
            fn capture_differential_metadata<C: Context<Type = ArrayType, Operation = Self>>(
                _context: &C,
                _primal: &C::Value,
                _differential_type: &ArrayType,
                _purpose: DifferentialMetadataPurpose,
            ) -> Result<Vec<C::Value>, ProgramError> {
                Ok(Vec::new())
            }
        }

        let context = EagerContext::<Array, TypeOnlyOperation>::new();
        let differential = Array::scalar(2f32).unwrap();
        assert_eq!(
            TypeOnlyOperation::validate_differential(
                &context,
                differential.clone(),
                differential.r#type().as_ref(),
                &[],
            ),
            Ok(differential),
        );
    }

    #[test]
    fn test_differential_boundary_metadata_sources() {
        let first = Array::scalar(2f32).unwrap();
        let second = Array::scalar(3f32).unwrap();
        let third = Array::scalar(5f32).unwrap();
        let marker = Array::scalar(7f32).unwrap();
        let other_marker = Array::scalar(11f32).unwrap();
        let metadata = DifferentialBoundaryMetadata {
            values: vec![first, second.clone(), marker.clone(), third.clone(), marker.clone(), other_marker.clone()],
            leaves: vec![
                (1, 0..1, DifferentialMetadataPurpose::Validation),
                (4, 1..3, DifferentialMetadataPurpose::Reconstruction),
                (7, 3..6, DifferentialMetadataPurpose::ValidationAndReconstruction),
            ],
        };

        // Sparse lookups use public positions; validation-only values remain stored but are not reconstruction sources.
        assert_eq!(metadata.sources(0), &[]);
        assert_eq!(metadata.sources(1), &[]);
        assert_eq!(metadata.sources(2), &[]);
        assert_eq!(metadata.sources(4), &[second, marker.clone()]);
        assert_eq!(metadata.sources(5), &[]);
        assert_eq!(metadata.sources(7), &[third, marker, other_marker]);
        assert_eq!(metadata.sources(8), &[]);
    }

    #[test]
    fn test_differential_boundary_metadata_capture() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let first = Array::scalar(2f32).unwrap();
        let second = Array::scalar(3f32).unwrap();
        let third = Array::scalar(5f32).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        assert_eq!(
            metadata.capture(&context, 1, &first, &differential_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(
            metadata.capture(&context, 4, &second, &differential_type, DifferentialMetadataPurpose::Reconstruction),
            Ok(()),
        );
        assert_eq!(
            metadata.capture(
                &context,
                7,
                &third,
                &differential_type,
                DifferentialMetadataPurpose::ValidationAndReconstruction,
            ),
            Ok(()),
        );

        // Each purpose produces a different number of values, making both purpose forwarding and range offsets visible.
        assert_eq!(
            metadata.values,
            vec![
                first,
                second,
                Array::scalar(7f32).unwrap(),
                third,
                Array::scalar(7f32).unwrap(),
                Array::scalar(11f32).unwrap()
            ],
        );
        assert_eq!(
            metadata.leaves,
            vec![
                (1, 0..1, DifferentialMetadataPurpose::Validation),
                (4, 1..3, DifferentialMetadataPurpose::Reconstruction),
                (7, 3..6, DifferentialMetadataPurpose::ValidationAndReconstruction),
            ],
        );
    }

    #[test]
    fn test_differential_boundary_metadata_capture_empty() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let integer = Array::scalar(1i32).unwrap();
        let zero_type = ArrayType::scalar(DataType::Zero);
        let first = Array::scalar(2f32).unwrap();
        let second = Array::scalar(3f32).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        assert_eq!(
            metadata.capture(&context, 0, &integer, &zero_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(metadata.values.capacity(), 0);
        assert_eq!(metadata.leaves.capacity(), 0);
        assert_eq!(
            metadata.capture(&context, 2, &first, &differential_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(
            metadata.capture(&context, 3, &integer, &zero_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(
            metadata.capture(&context, 5, &second, &differential_type, DifferentialMetadataPurpose::Reconstruction),
            Ok(()),
        );

        // Omitted captures neither allocate an entry nor shift later public indices or metadata ranges.
        assert_eq!(metadata.values, vec![first, second, Array::scalar(7f32).unwrap()]);
        assert_eq!(
            metadata.leaves,
            vec![
                (2, 0..1, DifferentialMetadataPurpose::Validation),
                (5, 1..3, DifferentialMetadataPurpose::Reconstruction),
            ],
        );
    }

    #[test]
    fn test_differential_boundary_metadata_capture_rejects_unordered_indices() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let primal = Array::scalar(2f32).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        assert_eq!(
            metadata.capture(&context, 4, &primal, &differential_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );

        // Both a duplicate and an earlier position must fail before changing the existing capture.
        for index in [4, 2] {
            assert_eq!(
                metadata.capture(&context, index, &primal, &differential_type, DifferentialMetadataPurpose::Validation),
                Err(ProgramError::MalformedProgram("differential metadata leaves are not in boundary order".into())),
            );
            assert_eq!(metadata.values, vec![primal.clone()]);
            assert_eq!(metadata.leaves, vec![(4, 0..1, DifferentialMetadataPurpose::Validation)]);
        }
    }

    #[test]
    fn test_differential_boundary_metadata_capture_propagates_error() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let primal = Array::scalar(2f32).unwrap();
        let unsupported = Array::scalar(3f64).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        assert_eq!(
            metadata.capture(&context, 1, &primal, &differential_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(
            metadata.capture(
                &context,
                4,
                &unsupported,
                unsupported.r#type().as_ref(),
                DifferentialMetadataPurpose::Validation,
            ),
            Err(ProgramError::InvalidArgument { message: "metadata capture probe failed".into() }),
        );
        assert_eq!(metadata.values, vec![primal.clone()]);
        assert_eq!(metadata.leaves, vec![(1, 0..1, DifferentialMetadataPurpose::Validation)]);

        // Failure must not reserve the failed leaf's position or leave partial metadata behind.
        assert_eq!(
            metadata.capture(&context, 4, &primal, &differential_type, DifferentialMetadataPurpose::Validation),
            Ok(()),
        );
        assert_eq!(metadata.values, vec![primal.clone(), primal]);
        assert_eq!(
            metadata.leaves,
            vec![
                (1, 0..1, DifferentialMetadataPurpose::Validation),
                (4, 1..2, DifferentialMetadataPurpose::Validation),
            ],
        );
    }

    #[test]
    fn test_differential_boundary_metadata_validate() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let first = Array::scalar(2f32).unwrap();
        let second = Array::vector(vec![3f32, 4f32]).unwrap();
        let integer = Array::scalar(9i32).unwrap();
        let zero = Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap();
        let marker = Array::scalar(7f32).unwrap();
        let other_marker = Array::scalar(11f32).unwrap();
        let metadata = DifferentialBoundaryMetadata {
            values: vec![first.clone(), second.clone(), marker, other_marker, integer.clone()],
            leaves: vec![
                (1, 0..1, DifferentialMetadataPurpose::Validation),
                (4, 1..4, DifferentialMetadataPurpose::ValidationAndReconstruction),
                (6, 4..5, DifferentialMetadataPurpose::Validation),
            ],
        };
        assert_eq!(metadata.validate(&context, 1, first.clone(), first.r#type().as_ref()), Ok(first));
        assert_eq!(metadata.validate(&context, 4, second.clone(), second.r#type().as_ref()), Ok(second));

        // An integer primal and its zero-space differential have different types; the cached primal type must survive.
        assert_eq!(metadata.validate(&context, 6, zero.clone(), integer.r#type().as_ref()), Ok(zero));
    }

    #[test]
    fn test_differential_boundary_metadata_validate_staged() {
        let context = TracingContext::<Array, MetadataProbeOperation>::new();
        let primal_type = ArrayType::scalar(DataType::F32);
        let differential = context.input(primal_type.clone());
        let captured = context.input(primal_type.clone());
        let metadata = DifferentialBoundaryMetadata {
            values: vec![captured],
            leaves: vec![(1, 0..1, DifferentialMetadataPurpose::Validation)],
        };
        let checked = metadata.validate(&context, 1, differential, &primal_type).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![checked.atom_id().unwrap()], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The result must depend on the validation operation, preserving its checks when later operations consume it.
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:f32[] .
                let %2:f32[] = metadata_validation_probe %0 %1
                in (%2)"},
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(2f32).unwrap(), Array::scalar(2f32).unwrap()]),
            Ok(vec![Array::scalar(2f32).unwrap()]),
        );
    }

    #[test]
    fn test_differential_boundary_metadata_validate_empty() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let differential = Array::scalar(2f32).unwrap();
        let differential_type = differential.r#type().into_owned();
        let metadata = DifferentialBoundaryMetadata {
            values: vec![differential.clone(), differential.clone()],
            leaves: vec![
                (2, 0..1, DifferentialMetadataPurpose::Validation),
                (4, 1..2, DifferentialMetadataPurpose::Validation),
            ],
        };

        // Missing entries still dispatch validation with an empty slice, before, between, or after stored leaves.
        for index in [0, 3, 5] {
            assert_eq!(
                metadata.validate(&context, index, differential.clone(), &differential_type),
                Err(ProgramError::InvalidArgument { message: "metadata probe received no captured values".into() }),
            );
        }
        let empty = DifferentialBoundaryMetadata::default();
        assert_eq!(
            empty.validate(&context, 0, differential, &differential_type),
            Err(ProgramError::InvalidArgument { message: "metadata probe received no captured values".into() }),
        );
    }

    #[test]
    fn test_differential_boundary_metadata_validate_propagates_error() {
        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let primal = Array::scalar(2f32).unwrap();
        let differential = Array::scalar(3f32).unwrap();
        let metadata = DifferentialBoundaryMetadata {
            values: vec![primal],
            leaves: vec![(1, 0..1, DifferentialMetadataPurpose::Validation)],
        };
        assert_eq!(
            metadata.validate(&context, 1, differential.clone(), differential.r#type().as_ref()),
            Err(ProgramError::InvalidArgument { message: "metadata probe received incompatible values".into() }),
        );
    }

    #[test]
    fn test_differential_boundary_metadata_default() {
        let metadata = DifferentialBoundaryMetadata::<Array>::default();
        assert_eq!(metadata.values, Vec::<Array>::new());
        assert_eq!(metadata.leaves, Vec::new());
        assert_eq!(metadata.sources(0), &[]);
    }
}
