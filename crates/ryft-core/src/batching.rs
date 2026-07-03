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
/// marks which of its dimensions indexes the current batch lanes. A value is either *batched* (i.e., its physical type
/// carries the lane dimension) or *lane-uniform*, meaning that it is shared unchanged across every batch lane.
#[derive(Clone, Debug, PartialEq, Parameter)]
pub struct ArrayBatch<V> {
    /// Physical array type of `value`. When the value is batched this type includes the mapped lane dimension at
    /// `batch_axis`. The logical per-lane [`ArrayType`] is recovered by removing that dimension and can be obtained
    /// using [`Self::logical_type`].
    r#type: ArrayType,

    /// Refer to the documentation of [`value`](Self::value) for more information.
    value: V,

    /// Refer to the documentation of [`batch_axis`](Self::batch_axis) for more information.
    batch_axis: Option<usize>,
}

impl<V: Typed<ArrayType>> ArrayBatch<V> {
    /// Creates a new [`ArrayBatch`].
    pub fn new(r#type: ArrayType, value: V, batch_axis: Option<usize>) -> Result<Self, ProgramError> {
        if let Some(axis) = batch_axis
            && axis >= r#type.rank()
        {
            return Err(BatchingError::BatchAxisOutOfBounds { r#type, axis }.into());
        }
        Ok(Self { r#type, value, batch_axis })
    }

    /// Returns the axis in [`r#type`](Self::type) and [`value`](Self::value) that indexes the current batch lanes,
    /// or `None` when `value` is *lane-uniform* (i.e., it carries no physical dimension for the current lanes and is
    /// interpreted as the same value for every lane). For example, a traced constant in `batch(|x| x + 1)` has a `None`
    /// batch axis, while `x` carries the mapped input axis. Runtime control flow predicates may also require `None`,
    /// because a single predicate may select one branch for all lanes while a lane-varying predicate would need a
    /// dedicated batching rule. Note that `None` is not limited to rank-0 (i.e., scalar) values. Any shaped constant
    /// or operand is lane-uniform when none of its physical dimensions indexes the current lanes.
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

    // TODO(eaplatanios): Review from here onwards.
    
    /// Returns the static mapped axis size, if this value is batched.
    pub fn axis_size(&self) -> Result<Option<usize>, ProgramError> {
        let Some(axis) = self.batch_axis else {
            return Ok(None);
        };
        let Some(size) = self.r#type.dimension(axis as isize).value() else {
            return Err(BatchingError::DynamicBatchAxis { r#type: self.r#type.clone(), axis }.into());
        };
        Ok(Some(size))
    }

    /// Returns the scalar-body type obtained by removing the mapped axis.
    pub fn logical_type(&self) -> Result<ArrayType, ProgramError> {
        let Some(axis) = self.batch_axis else {
            return Ok(self.r#type.clone());
        };
        Ok(self.r#type.without_dimension(axis)?.0)
    }

    /// Wraps a value that already contains a mapped axis.
    ///
    /// # Parameters
    ///
    ///   - `value`: Packed array value.
    ///   - `batch_axis`: Mapped axis in `value`.
    pub fn mapped(value: V, batch_axis: usize) -> Result<Self, ProgramError> {
        Self::new(value.r#type().into_owned(), value, Some(batch_axis))
    }

    /// Wraps a value that is uniform across the current batch lanes.
    pub fn unbatched(value: V) -> Self {
        Self { r#type: value.r#type().into_owned(), value, batch_axis: None }
    }
}

impl<V: Display> Display for ArrayBatch<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.batch_axis {
            Some(axis) => write!(formatter, "batch[{}, axis={axis}]({})", self.r#type, self.value),
            None => write!(formatter, "batch[{}, lane-uniform]({})", self.r#type, self.value),
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
