use std::borrow::Cow;
use std::fmt::Display;

use ryft_macros::Parameter;
use thiserror::Error;

use crate::parameters::{Parameter, ParameterError};
use crate::programs::{ProgramError, Value};
use crate::types::{ArrayType, Typed};

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

/// Value with [`ArrayType`] type that represents a _packed_ batch of arrays. [`ArrayBatch`] is the batching
/// representation for Ryft's batching/vectorization transform. It pairs a physical array value with a batch axis that
/// marks which of its dimensions indexes the batch items. A value is either *batched* (i.e., its physical type carries
/// the batch dimension) or *replicated*, meaning that it is shared unchanged across every batch item.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Physical array type of `value`. When the value is batched this type includes the mapped batch dimension at
    /// `batch_axis`. The unbatched (i.e., per-item) [`ArrayType`] is recovered by removing that dimension and can be
    /// obtained using [`Self::unbatched_type`].
    r#type: ArrayType,

    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: Option<usize>,
}

impl<V: Typed<ArrayType>> ArrayBatch<V> {
    /// Creates a new [`ArrayBatch`].
    #[inline]
    pub fn new(r#type: ArrayType, value: V, batch_axis: Option<usize>) -> Result<Self, ProgramError> {
        if let Some(axis) = batch_axis
            && axis >= r#type.rank()
        {
            return Err(BatchingError::BatchAxisOutOfBounds { r#type, axis }.into());
        }
        Ok(Self { r#type, value, batch_axis })
    }

    /// Creates a new [`ArrayBatch`] that replicates the provided value across the batch.
    #[inline]
    pub fn replicated(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }

    /// Returns the axis in [`r#type`](Self::type) and [`value`](Self::value) that indexes the batch items, or `None`
    /// when `value` is *replicated* (i.e., it carries no physical dimension for the batch and is interpreted as the
    /// same value for every batch item). For example, a traced constant in `batch(|x| x + 1)` has a `None` batch axis,
    /// while `x` carries the mapped input axis. Runtime control flow predicates may also require `None`, because a
    /// single predicate may select one branch for the whole batch while a batch-varying predicate would need a
    /// dedicated batching rule. Note that `None` is not limited to rank-0 (i.e., scalar) values. Any shaped constant
    /// or operand is replicated when none of its physical dimensions indexes the batch.
    #[inline]
    pub fn batch_axis(&self) -> Option<usize> {
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
    pub fn batch_size(&self) -> Result<Option<usize>, ProgramError> {
        let Some(axis) = self.batch_axis else {
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
    pub fn unbatched_type(&self) -> Result<ArrayType, ProgramError> {
        let Some(axis) = self.batch_axis else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }
}

impl<V: Display> Display for ArrayBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis {
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
