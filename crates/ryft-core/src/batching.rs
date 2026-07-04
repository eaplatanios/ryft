use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::operations::Operation;
use crate::parameters::{Parameter, ParameterError};
use crate::programs::{Program, ProgramError, Value};
use crate::types::{ArrayType, TypeError, Typed};

/// Represents batching-related errors.
///
/// [`BatchingError`] and [`ProgramError`] deliberately form a conversion cycle in which each type can carry the
/// other. Batching rules get executed by binding operations (i.e., via [`Context::bind`](crate::Context::bind) and
/// [`StagingContext::stage_operation`](crate::StagingContext::stage_operation)), which can result in [`ProgramError`]s.
/// So, [`BatchingError`]s travel up a trace, type-erased, inside [`ProgramError::Custom`] payloads. In the other
/// direction, the public [`Batch::batch`](crate::Batch::batch) entry point is typed to [`BatchingError`], and a
/// batching trace can also fail for reasons that are not batching-related. Those program errors surface through the
/// [`BatchingError::Program`] variant. The paired [`From`] implementations keep this cycle normalized instead of
/// letting the two types nest: converting to [`ProgramError`] unwraps a [`BatchingError::Program`] back into the
/// program error that it carries and wraps every other variant in [`ProgramError::Custom`], while converting to
/// [`BatchingError`] unwraps a [`ProgramError::Custom`] payload holding a [`BatchingError`] and wraps every other
/// program error in [`BatchingError::Program`]. Round trips therefore never nest one error type inside the other,
/// and `?` re-types errors correctly at both boundaries. Outside of these conversions, a [`BatchingError`] carried by
/// a [`ProgramError`] can be recovered using [`ProgramError::downcast_custom`].
#[derive(Error, Clone, Debug, PartialEq, Eq, Hash)]
pub enum BatchingError {
    #[error("encountered an empty batch")]
    EmptyBatch,

    #[error("mismatched batch sizes across batched leaves; expected size {expected} but got {actual}")]
    MismatchedBatchSizes { expected: usize, actual: usize },

    #[error("{message}")]
    MisalignedBatchAxes { message: String },

    #[error("batch axis {axis} of array type {type} has dynamic size")]
    DynamicBatchAxis { r#type: ArrayType, axis: usize },

    #[error("batch axis {axis} is out of bounds for array type {type}")]
    BatchAxisOutOfBounds { r#type: ArrayType, axis: usize },

    #[error("{message}")]
    UnsupportedOperation { message: String },

    #[error("mismatched batch output axes; expected {expected:?} but got {actual:?}")]
    MismatchedOutputAxes { expected: Option<usize>, actual: Option<usize> },

    #[error(transparent)]
    Parameter(#[from] ParameterError),

    #[error(transparent)]
    Type(#[from] TypeError),

    #[error(transparent)]
    Program(ProgramError),
}

impl From<ProgramError> for BatchingError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(batching) = error.downcast_custom::<BatchingError>() {
            batching.clone()
        } else {
            BatchingError::Program(error)
        }
    }
}

impl From<BatchingError> for ProgramError {
    #[inline]
    fn from(error: BatchingError) -> Self {
        match error {
            BatchingError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// A batched value's mapped batch axis. [`BatchAxis::mapped`]`(k)` means that the value's batch dimension sits at
/// physical axis `k`. [`BatchAxis::replicated`] (the [`Default`]) means that the value is *replicated* (i.e., it
/// carries no physical dimension for the batch and is interpreted as the same value for every batch item). For example,
/// a traced constant in `batch(|x| x + 1)` is replicated, while `x` carries the mapped input axis. Runtime control flow
/// predicates may also be replicated, because a single predicate may select one branch for the whole batch while a
/// batch-varying predicate would need a dedicated batching rule. Note that replication is not limited to rank-0 (i.e.,
/// scalar) values. Any shaped constant or operand is replicated when none of its physical dimensions indexes the batch.
///
/// This is the batch axis carried by an [`ArrayBatch`] and, during the batching transform, by the
/// [`Tracer`](crate::Tracer) metadata. Carrying it on the value itself lets the per-operation batching rules route the
/// mapped batch axis straight from the value in hand.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Parameter)]
pub struct BatchAxis(Option<usize>);

impl BatchAxis {
    /// Creates a mapped [`BatchAxis`] at physical position `axis`.
    #[inline]
    pub fn mapped(axis: usize) -> Self {
        Self(Some(axis))
    }

    /// Creates a replicated [`BatchAxis`] (i.e., the batched value is shared unchanged across every batch item).
    /// This is equivalent to [`BatchAxis::default`].
    #[inline]
    pub fn replicated() -> Self {
        Self(None)
    }

    /// Returns the mapped batch axis position, or `None` if this [`BatchAxis`] is replicated.
    #[inline]
    pub fn axis(&self) -> Option<usize> {
        self.0
    }

    /// Returns `true` if this [`BatchAxis`] is replicated (i.e., if it carries no mapped batch axis).
    #[inline]
    pub fn is_replicated(&self) -> bool {
        self.0.is_none()
    }
}

impl From<Option<usize>> for BatchAxis {
    #[inline]
    fn from(axis: Option<usize>) -> Self {
        Self(axis)
    }
}

impl From<usize> for BatchAxis {
    #[inline]
    fn from(axis: usize) -> Self {
        Self(Some(axis))
    }
}

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform. It pairs a physical array value with a [`BatchAxis`]
/// that marks which of its dimensions indexes the batch items. A value is either *batched* (i.e., its physical type
/// carries the batch dimension) or *replicated*, meaning that it is shared unchanged across every batch item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Physical array type of `value`. When the value is batched this type includes the mapped batch dimension at
    /// `batch_axis`. The unbatched (i.e., per-item) [`ArrayType`] is recovered by removing that dimension and can be
    /// obtained using [`Self::unbatched_type`].
    r#type: ArrayType,

    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: BatchAxis,
}

impl<V: Typed<ArrayType>> ArrayBatch<V> {
    /// Creates a new [`ArrayBatch`].
    #[inline]
    pub fn new<A: Into<BatchAxis>>(r#type: ArrayType, value: V, batch_axis: A) -> Result<Self, BatchingError> {
        let batch_axis = batch_axis.into();
        if let Some(axis) = batch_axis.axis()
            && axis >= r#type.rank()
        {
            return Err(BatchingError::BatchAxisOutOfBounds { r#type, axis }.into());
        }
        Ok(Self { r#type, value, batch_axis })
    }

    /// Creates a new [`ArrayBatch`] that replicates the provided value across the batch.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: BatchAxis::replicated() }
    }

    /// Returns the [`BatchAxis`] marking which dimension of [`value`](Self::value) indexes the batch items.
    #[inline]
    pub fn batch_axis(&self) -> BatchAxis {
        self.batch_axis
    }

    /// Returns the packed array value.
    #[inline]
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Consumes `self` and returns the packed array value.
    #[inline]
    pub fn into_value(self) -> V {
        self.value
    }

    /// Returns the batch size of this [`ArrayBatch`] (i.e., the number of items that are batched together),
    /// or `None` if it is replicated (i.e., shared as-is across the whole batch).
    #[inline]
    pub fn batch_size(&self) -> Result<Option<usize>, BatchingError> {
        let Some(axis) = self.batch_axis.axis() else {
            return Ok(None);
        };
        let size = self
            .r#type
            .dimension(axis as isize)
            .value()
            .ok_or_else(|| BatchingError::DynamicBatchAxis { r#type: self.r#type.clone(), axis })?;
        Ok(Some(size))
    }

    /// Returns the [`ArrayType`] of each item in the batch (i.e., with the batch axis removed, if any).
    #[inline]
    pub fn unbatched_type(&self) -> Result<ArrayType, BatchingError> {
        let Some(axis) = self.batch_axis.axis() else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }

    /// Computes and validates the common batch size across `inputs`, returning `None` when no input is batched.
    /// Returns [`BatchingError::MismatchedBatchSizes`] when two batched inputs disagree on their batch size and
    /// [`BatchingError::DynamicBatchAxis`] when any batched input's mapped axis has a non-static size.
    pub fn common_batch_size(inputs: &[Self]) -> Result<Option<usize>, BatchingError> {
        inputs.iter().try_fold(None, |common_size, input| match (common_size, input.batch_size()?) {
            (Some(common_size), Some(size)) if common_size != size => {
                Err(BatchingError::MismatchedBatchSizes { expected: common_size, actual: size })
            }
            (None, Some(size)) => Ok(Some(size)),
            (common_size, _) => Ok(common_size),
        })
    }
}

impl<V: Display> Display for ArrayBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis.axis() {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, replicated]({})", self.r#type, self.value),
        }
    }
}

impl<V: Typed<ArrayType>> Typed<ArrayType> for ArrayBatch<V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, ArrayType> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<V: Value<ArrayType>> Value<ArrayType> for ArrayBatch<V> {}

/// Represents [`Operation`]s that can be batched (i.e., vectorized).
pub trait BatchableOperation<V: Value<ArrayType>, C>: Operation<ArrayType> {
    /// Applies this operation to packed batched inputs, returning batched outputs with the resulting batch axes,
    /// using `context` for rules that need active transform state.
    ///
    /// # Contract
    ///
    ///   - **Axis Alignment:** If two or more inputs carry a mapped axis (i.e., `batch_axis.is_some()`), elementwise
    ///     operations require them to agree on the axis position. When they disagree, this function returns
    ///     [`BatchingError::MisalignedBatchAxes`] with an error message that names the misaligned axes and suggests the
    ///     user repositions one of them with [`Transpose`](crate::Transpose) (i.e., the N-D axis permutation primitive)
    ///     before invoking the operation. Operations with explicit axis arguments (e.g., `Dot`, `Transpose`, `Reshape`,
    ///     etc.) rewrite those arguments to thread the mapped axis through correctly.
    ///   - **Output Axes:** For elementwise operations, the output [`ArrayBatch::batch_axis`] matches the common input
    ///     batch axis. For operations with explicit axis arguments, the output axis follows from the lifted axis
    ///     arguments.
    ///   - **Zero Propagation:** Linear batching rules preserve zero tangent payloads through their operation-specific
    ///     semantics. Canonical staged zeros are handled before batching reaches concrete value-level interpretation.
    ///
    /// Note that in order to be able to provide [`BatchableOperation`] implementations for operation families that are
    /// derived using our `#[derive(BatchableOperation)]` macro, it is a common convention for operations that can be
    /// part of such operation families to implement this trait even if they do not support batching and to have this
    /// function simply return a [`BatchingError::UnsupportedOperation`] error.
    fn batch(&self, context: &C, inputs: &[ArrayBatch<V>]) -> Result<Vec<ArrayBatch<V>>, BatchingError>;
}

/// Policy for choosing a batched [`Program`]'s output axes. Program batching always replays the program over physical
/// values whose mapped batch axes are specified by the caller. This policy controls how the replayed output tracers are
/// packaged into the resulting program.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ProgramBatchingOutputAxesPolicy {
    /// Keep the output axes naturally produced by the per-operation batching rules. Replicated outputs remain
    /// replicated and are reported as `None`.
    Natural,

    /// Align/normalize every output to the specified mapped axis, moving already-batched outputs with
    /// [`Transpose`](crate::Transpose) and broadcasting replicated outputs across the batch.
    AlignAllTo(usize),
}

/// Represents closed [`Operation`] families whose captured flat [`Program`]s can be batched into standalone batched
/// programs. This is the batching analogue of [`InterpretableProgramOperation`](crate::InterpretableProgramOperation).
/// It names the recursive fixed point needed by higher-order batching helpers without requiring the full operation
/// enum's [`BatchableOperation`] implementation while proving that implementation. Operation families implement it by
/// replaying captured flat [`Program`]s through their operation-owned batching rules, via
/// [`batch_program`](crate::batch_program).
///
/// The re-wrapping batch rules of [`CustomJvpOperation`](crate::CustomJvpOperation) and
/// [`CustomVjpOperation`](crate::CustomVjpOperation) bound their captured-program operation type by this trait. Routing
/// program-level batching through this dedicated, lifetime-free witness keeps the trait solver's recursion finite: the
/// closed enum implementation discharges the derived batching-context obligations once, instead of every batching rule
/// re-deriving them with fresh higher-ranked lifetimes (which defeats the solver's cycle detection and overflows).
pub trait BatchableProgramOperation<V: Value<ArrayType>>: Operation<ArrayType> + Sized {
    /// Batches a captured [`Program`] into a standalone program over batch-carrying physical types. Refer to
    /// the documentation of [`batch_program`](crate::batch_program) for the input and output axis conventions.
    ///
    /// # Parameters
    ///
    ///   - `program`: Captured program to batch, over per-item input and output types.
    ///   - `batch_size`: Size of the batch axis (i.e., number of items being batched together).
    ///   - `input_batch_axes`: Mapped batch-axis position for each program input, or `None` for a replicated input.
    ///   - `output_axes_policy`: Policy for packaging the batched program's outputs.
    fn batch_program(
        program: &Program<ArrayType, V, Self, Vec<V>, Vec<V>>,
        batch_size: usize,
        input_batch_axes: &[Option<usize>],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Program<ArrayType, V, Self, Vec<V>, Vec<V>>, Vec<Option<usize>>), BatchingError>;
}

impl<V: Value<ArrayType>, O: BatchableProgramOperation<V>> Program<ArrayType, V, O, Vec<V>, Vec<V>> {
    #[inline]
    pub fn batched(
        &self,
        batch_size: usize,
        input_batch_axes: &[Option<usize>],
        output_axes_policy: ProgramBatchingOutputAxesPolicy,
    ) -> Result<(Self, Vec<Option<usize>>), BatchingError> {
        O::batch_program(self, batch_size, input_batch_axes, output_axes_policy)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::tests::TestArray;
    use crate::types::{ArrayType, DataType, Shape, Size, Typed};

    use super::*;

    #[test]
    fn test_array_batch() {
        let matrix = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();

        // `new` builds a batched value when the mapped axis is in bounds, and the accessors report the packed value,
        // its physical type, the batch size read off the mapped axis, and the per-item type with that axis removed.
        let batched = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(0)).unwrap();
        assert_eq!(batched.batch_axis(), BatchAxis::mapped(0));
        assert_eq!(batched.value(), &matrix);
        assert_eq!(*batched.r#type(), matrix_type);
        assert_eq!(batched.batch_size(), Ok(Some(2)));
        assert_eq!(batched.unbatched_type(), Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]))));
        assert_eq!(batched.to_string(), "batch[f64[2, 3], axis=0]([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])");
        assert_eq!(batched.into_value(), matrix);

        // A different mapped axis reads the batch size and per-item type from that axis instead.
        let batched_axis_one = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(1)).unwrap();
        assert_eq!(batched_axis_one.batch_size(), Ok(Some(3)));
        assert_eq!(
            batched_axis_one.unbatched_type(),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2)]))),
        );

        // `new` rejects an out-of-bounds mapped axis.
        assert_eq!(
            ArrayBatch::new(matrix_type.clone(), matrix, Some(2)),
            Err(BatchingError::BatchAxisOutOfBounds { r#type: matrix_type, axis: 2 }),
        );

        // `replicated` shares the value unchanged across the batch: no mapped axis, no batch size, and the per-item
        // type is the whole physical type.
        let vector = TestArray::vector(vec![1.0, 2.0, 3.0]);
        let vector_type = vector.r#type().into_owned();
        let replicated = ArrayBatch::replicated(vector.clone());
        assert_eq!(replicated.batch_axis(), BatchAxis::replicated());
        assert_eq!(*replicated.r#type(), vector_type);
        assert_eq!(replicated.batch_size(), Ok(None));
        assert_eq!(replicated.unbatched_type(), Ok(vector_type));
        assert_eq!(replicated.to_string(), "batch[f64[3], replicated]([1.0, 2.0, 3.0])");
        assert_eq!(replicated.into_value(), vector);
    }

    #[test]
    fn test_array_batch_common_batch_size() {
        let matrix = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_type = matrix.r#type().into_owned();
        let vector = TestArray::vector(vec![7.0, 8.0]);

        // All-replicated inputs pin no batch size.
        let replicated = ArrayBatch::replicated(matrix.clone());
        assert_eq!(ArrayBatch::common_batch_size(&[replicated.clone()]), Ok(None));

        // A single batched input pins its own batch size, and a replicated input alongside it is ignored.
        let batched_axis_zero = ArrayBatch::new(matrix_type.clone(), matrix.clone(), Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(&[batched_axis_zero.clone()]), Ok(Some(2)));
        assert_eq!(ArrayBatch::common_batch_size(&[replicated, batched_axis_zero.clone()]), Ok(Some(2)));

        // Two batched inputs that agree on their batch size share it, even across different mapped axes.
        let batched_vector = ArrayBatch::new(vector.r#type().into_owned(), vector, Some(0)).unwrap();
        assert_eq!(ArrayBatch::common_batch_size(&[batched_axis_zero.clone(), batched_vector]), Ok(Some(2)));

        // Two batched inputs that disagree on their batch size are rejected.
        let batched_axis_one = ArrayBatch::new(matrix_type, matrix, Some(1)).unwrap();
        assert_eq!(
            ArrayBatch::common_batch_size(&[batched_axis_zero, batched_axis_one]),
            Err(BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 }),
        );
    }
}
