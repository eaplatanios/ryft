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
/// # Example
///
/// Consider a pushforward whose flattened primal outputs contain two static scalars and two indexed arrays. During
/// linearization, [`capture`](Self::capture) visits each output in public boundary order. The array-IR family captures
/// one shape carrier for each indexed array, preserving its runtime extents without numerical primal storage:
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
pub(crate) struct DifferentialBoundaryMetadata<V: Value> {
    /// Metadata values from non-empty leaf captures, concatenated in public boundary order. Each leaf's values retain
    /// the operation family's capture order and occupy one contiguous range.
    values: Vec<V>,

    /// Sparse entries `(public_boundary_index, value_range, purpose)`, ordered by increasing boundary index. Each range
    /// indexes [`values`](Self::values); leaves with empty captures have no entry.
    leaves: Vec<(usize, Range<usize>, DifferentialMetadataPurpose)>,
}

impl<V: Value> DifferentialBoundaryMetadata<V> {
    /// Returns the captured reconstruction sources for a public boundary leaf.
    pub(crate) fn sources(&self, index: usize) -> &[V] {
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

    /// Returns the checked differential at a public position captured for validation, including empty metadata.
    pub(crate) fn validate<C: Context<Value = V, Type = V::Type, Operation: DifferentialMetadataProvider<V::Type>>>(
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
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::contexts::EagerContext;
    use crate::operations::ConstantOperation;

    use super::*;

    #[test]
    fn test_differential_boundary_metadata() {
        #[derive(Clone, Debug, ryft_macros::Operation)]
        #[ryft(type(ArrayType), constant(Array))]
        enum MetadataProbeOperation {
            Constant(ConstantOperation<Array>),
        }

        impl DifferentialMetadataProvider<ArrayType> for MetadataProbeOperation {
            fn capture_differential_metadata<C: Context<Type = ArrayType, Operation = Self>>(
                _context: &C,
                primal: &C::Value,
                _differential_type: &ArrayType,
                purpose: DifferentialMetadataPurpose,
            ) -> Result<Vec<C::Value>, ProgramError> {
                Ok(if purpose == DifferentialMetadataPurpose::Validation {
                    Vec::new()
                } else {
                    vec![primal.clone(), primal.clone()]
                })
            }

            fn validate_differential<C: Context<Type = ArrayType, Operation = Self>>(
                _context: &C,
                _differential: C::Value,
                _primal_type: &ArrayType,
                _metadata: &[C::Value],
            ) -> Result<C::Value, ProgramError> {
                Err(ProgramError::InvalidArgument { message: "metadata validation probe".into() })
            }
        }

        let context = EagerContext::<Array, MetadataProbeOperation>::new();
        let first = Array::scalar(2f32).unwrap();
        let second = Array::scalar(3f32).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        metadata
            .capture(&context, 1, &first, &differential_type, DifferentialMetadataPurpose::Reconstruction)
            .unwrap();
        metadata
            .capture(&context, 4, &second, &differential_type, DifferentialMetadataPurpose::ValidationAndReconstruction)
            .unwrap();
        assert_eq!(metadata.sources(0), &[]);
        assert_eq!(metadata.sources(1), &[first.clone(), first.clone()]);
        assert_eq!(metadata.sources(4), &[second.clone(), second.clone()]);
        assert_eq!(metadata.values, vec![first.clone(), first.clone(), second.clone(), second.clone()]);
        assert_eq!(
            metadata.validate(&context, 4, second, &differential_type),
            Err(ProgramError::InvalidArgument { message: "metadata validation probe".into() }),
        );
        let third = Array::scalar(4f32).unwrap();
        metadata
            .capture(&context, 7, &third, &differential_type, DifferentialMetadataPurpose::Validation)
            .unwrap();
        assert_eq!(metadata.sources(7), &[]);
        assert_eq!(
            metadata.validate(&context, 7, third, &differential_type),
            Err(ProgramError::InvalidArgument { message: "metadata validation probe".into() }),
        );
        assert_eq!(
            metadata.capture(
                &context,
                3,
                &Array::scalar(1f32).unwrap(),
                &differential_type,
                DifferentialMetadataPurpose::Validation
            ),
            Err(ProgramError::MalformedProgram("differential metadata leaves are not in boundary order".into())),
        );
    }

    #[test]
    fn test_differential_boundary_metadata_static() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::scalar(2f32).unwrap();
        let differential_type = ArrayType::scalar(DataType::F32);
        let mut metadata = DifferentialBoundaryMetadata::default();
        metadata
            .capture(&context, 0, &primal, &differential_type, DifferentialMetadataPurpose::ValidationAndReconstruction)
            .unwrap();
        assert_eq!(metadata.values, Vec::<Array>::new());
        assert_eq!(metadata.leaves, Vec::new());
        assert_eq!(metadata.sources(0), &[]);
        assert_eq!(metadata.validate(&context, 0, primal.clone(), &differential_type), Ok(primal));
    }
}
