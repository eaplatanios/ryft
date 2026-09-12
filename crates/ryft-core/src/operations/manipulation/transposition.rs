use std::fmt::{Debug, Display};
use std::ops::Deref;
use std::sync::Arc;

use crate::arrays::{
    Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType, Shape, Sharding,
};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

/// [`Axis`] permutation used by [`TransposeOperation`] and the [`Transpose`] capability. For each output axis `i`,
/// `permutation[i]` identifies the input axis routed to it. Negative axes count from the end of the input, so
/// `[-1, -2]` exchanges the axes of a matrix. Signed and unsigned vectors, arrays, and borrowed slices convert
/// into a permutation, as do [`Axes`] collections. Use [`Self::default`] for an empty permutation.
///
/// Construction preserves the supplied indices without validating them. [`Self::normalize`] resolves negative axes
/// against the input rank and checks that every input axis appears exactly once. The resulting nonnegative positions
/// are suitable for indexing and backend lowering. Equality and hashing compare the supplied notation, so `[-1, -2]`
/// and `[1, 0]` are distinct values even though they describe the same matrix transpose.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct Permutation(Vec<Axis>);

impl Permutation {
    /// Returns the permutation axes, where element `i` identifies the input axis routed to output axis `i`.
    #[inline]
    pub fn as_slice(&self) -> &[Axis] {
        self.0.as_slice()
    }

    /// Returns the inverse [`Permutation`] using non-negative axes. Transposing by a permutation and then by its
    /// inverse restores the original axis order. Negative axes are resolved against this permutation's length;
    /// invalid or repeated axes return a [`TypeError`].
    #[inline]
    pub fn inverse(&self) -> Result<Permutation, TypeError> {
        let axes = self.normalize(self.len())?;
        let mut inverse = vec![0usize; axes.len()];
        for (position, axis) in axes.into_iter().enumerate() {
            inverse[axis] = position;
        }
        Ok(inverse.into())
    }

    /// Resolves signed axes against `rank` and returns their nonnegative positions in output-axis order.
    /// The [`Permutation`] must contain exactly `rank` axes, with no duplicates after normalization.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{Permutation, TypeError};
    /// assert_eq!(Permutation::from([-1, 0, 1]).normalize(3)?, vec![2, 0, 1]);
    /// # Ok::<(), TypeError>(())
    /// ```
    pub fn normalize(&self, rank: usize) -> Result<Vec<usize>, TypeError> {
        if self.len() != rank {
            return Err(TypeError::invalid(format!(
                "permutation has length {} but input has rank {}",
                self.len(),
                rank,
            )));
        }
        let mut seen = vec![false; rank];
        let mut axes = Vec::with_capacity(rank);
        for axis in self.iter() {
            let position = axis
                .normalize(rank)
                .map_err(|_| TypeError::invalid(format!("permutation axis {axis} is out of bounds")))?;
            if seen[position] {
                return Err(TypeError::invalid(format!("permutation contains duplicate axis {position}")));
            }
            seen[position] = true;
            axes.push(position);
        }
        Ok(axes)
    }
}

impl Debug for Permutation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_tuple("Permutation")
            .field(&self.iter().map(|axis| axis.value()).collect::<Vec<_>>())
            .finish()
    }
}

impl Deref for Permutation {
    type Target = [Axis];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[Axis]> for Permutation {
    #[inline]
    fn as_ref(&self) -> &[Axis] {
        &self.0
    }
}

impl<A: Into<Axis>> From<Vec<A>> for Permutation {
    #[inline]
    fn from(axes: Vec<A>) -> Self {
        Self(axes.into_iter().map(Into::into).collect())
    }
}

impl<A: Copy + Into<Axis>> From<&Vec<A>> for Permutation {
    #[inline]
    fn from(axes: &Vec<A>) -> Self {
        Self::from(axes.as_slice())
    }
}

impl<A: Copy + Into<Axis>> From<&[A]> for Permutation {
    #[inline]
    fn from(axes: &[A]) -> Self {
        Self(axes.iter().copied().map(Into::into).collect())
    }
}

impl<A: Into<Axis>, const N: usize> From<[A; N]> for Permutation {
    #[inline]
    fn from(axes: [A; N]) -> Self {
        Self(axes.into_iter().map(Into::into).collect())
    }
}

impl<A: Copy + Into<Axis>, const N: usize> From<&[A; N]> for Permutation {
    #[inline]
    fn from(axes: &[A; N]) -> Self {
        Self::from(axes.as_slice())
    }
}

impl From<Axes> for Permutation {
    #[inline]
    fn from(axes: Axes) -> Self {
        Self::from(axes.as_slice())
    }
}

impl From<&Axes> for Permutation {
    #[inline]
    fn from(axes: &Axes) -> Self {
        Self::from(axes.as_slice())
    }
}

impl From<&Permutation> for Permutation {
    #[inline]
    fn from(permutation: &Permutation) -> Self {
        permutation.clone()
    }
}

/// Canonical operation name for [`TransposeOperation`].
pub const TRANSPOSE_OPERATION_NAME: &str = "transpose";

/// [`Operation`] that reorders the axes of its input array according to a static permutation.
/// Refer to the documentation of [`Transpose`] for more information.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TransposeOperation {
    /// Axis [`Permutation`] of this [`TransposeOperation`].
    permutation: Permutation,
}

impl TransposeOperation {
    /// Creates a new [`TransposeOperation`] with the provided axis permutation.
    #[inline]
    pub fn new<P: Into<Permutation>>(permutation: P) -> Self {
        Self { permutation: permutation.into() }
    }

    /// Returns the axis [`Permutation`] of this [`TransposeOperation`].
    #[inline]
    pub fn permutation(&self) -> &Permutation {
        &self.permutation
    }
}

impl Display for TransposeOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for TransposeOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        TRANSPOSE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].transpose(&self.permutation) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field(
                "permutation",
                format_args!("{:?}", self.permutation.iter().map(|axis| axis.value()).collect::<Vec<_>>()),
            )
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Transpose>> InterpretableOperation<C> for TransposeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].transpose(&self.permutation)?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<TransposeOperation>>> PartiallyEvaluatableOperation<C>
    for TransposeOperation
{
}

impl<C: Context<Type = ArrayType, Value: Transpose>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for TransposeOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        // Validate logical axes before shifting them around the mapped axis. Invalid indices must not overflow
        // or wrap into a valid physical permutation during lifting.
        let permutation = self.permutation.normalize(inputs[0].unbatched_type().rank()).map_err(ProgramError::from)?;
        let (lifted_permutation, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let mut lifted_permutation = Vec::with_capacity(permutation.len() + 1);
                for output_axis in 0..=permutation.len() {
                    if output_axis == batch_axis {
                        lifted_permutation.push(batch_axis);
                    } else {
                        let original_output_axis = if output_axis < batch_axis { output_axis } else { output_axis - 1 };
                        let input_axis = permutation[original_output_axis];
                        lifted_permutation.push(if input_axis >= batch_axis { input_axis + 1 } else { input_axis });
                    }
                }
                (lifted_permutation, Some(batch_axis))
            }
            None => (permutation, None),
        };
        let lifted_operation = TransposeOperation::new(lifted_permutation);
        let mut outputs = lifted_operation.interpret_with_batch_axes(
            context,
            inputs,
            &[BatchAxis::from_optional_position(output_axis)],
        )?;
        let output_axes = lifted_operation
            .permutation()
            .inverse()?
            .normalize(lifted_operation.permutation().len())?
            .into_iter()
            .map(Some)
            .collect::<Vec<_>>();
        let ragged_axes = inputs[0]
            .ragged_axes()
            .iter()
            .cloned()
            .map(|ragged_axis| {
                // `output_axes` is the total inverse of a validated permutation, so every stored input axis survives.
                ragged_axis.relocated(output_axes.as_slice()).unwrap()
            })
            .collect();
        let output = outputs.remove(0).with_ragged_axes(ragged_axes)?;
        Ok(vec![output].into())
    }
}

impl_differentiable_operation! {
    TransposeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Transpose,
    {
        |operation, _context, _driver, inputs| {
            // A live tangent follows the primal permutation. Structural zero tangents remain symbolic, including
            // when this rule is invoked directly instead of through a driver's zero-tangent fast path.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().transpose(operation.permutation())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.transpose(operation.permutation())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<TransposeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Transpose,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            let inverse = operation.permutation().inverse()?;
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let contribution = MaybeZero::Value(
                        cotangent.transpose(inverse)?.unalign_cotangent(&inputs[0].r#type().cotangent()?)?,
                    );
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Reorders the axes of an array according to a [`Permutation`]. Output axis `i` receives input axis `permutation[i]`,
/// so the permutation must contain every input axis exactly once. An identity permutation passes the input through
/// unchanged. Every other transposition preserves the element type, memory space, and reduction state, permutes shape
/// and per-dimension sharding in the same way as the data, and clears an explicit physical layout because a logical
/// axis permutation does not determine a unique output storage layout.
///
/// [`Transpose`] fills the same role for [`TransposeOperation`] that [`std::ops::Add`] and [`std::ops::Neg`] fill for
/// their corresponding arithmetic [`Operation`]s.
///
/// # Examples
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Transpose};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let output = input.transpose([1, 0])?;
/// assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Transpose: Sized {
    /// Reorders the axes of `self` according to the provided [`Permutation`], validating that the permutation is a
    /// bijection of the input axes. Negative indices count from the end. Use [`Self::transpose_reversed`] to
    /// reverse every axis, or [`Self::matrix_transpose`] to exchange only the last two axes.
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError>;

    /// Reverses the order of all axes. Scalars and vectors remain unchanged while matrices exchange their two axes.
    /// Higher-rank arrays reverse every axis, whereas [`Self::matrix_transpose`] exchanges only the last two.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{ArrayType, DataType, ProgramError, Transpose};
    /// let input = ArrayType::new_static(DataType::F32, [2, 3, 4]);
    /// assert_eq!(input.transpose_reversed()?, ArrayType::new_static(DataType::F32, [4, 3, 2]));
    /// # Ok::<(), ProgramError>(())
    /// ```
    #[inline]
    fn transpose_reversed(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        self.transpose((0..self.r#type().rank()).rev().collect::<Vec<_>>())
    }

    /// Exchanges the last two axes while retaining all leading batch axes. This is an ordinary transpose, without
    /// complex conjugation. Inputs must have rank at least two (scalars and vectors return a [`TypeError`]).
    ///
    /// # Example
    ///
    /// ```rust
    /// # use ryft_core::{ArrayType, DataType, ProgramError, Transpose};
    /// let input = ArrayType::new_static(DataType::F32, [2, 3, 4]);
    /// assert_eq!(input.matrix_transpose()?, ArrayType::new_static(DataType::F32, [2, 4, 3]));
    /// # Ok::<(), ProgramError>(())
    /// ```
    #[inline]
    fn matrix_transpose(&self) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        if rank < 2 {
            return Err(TypeError::invalid(format!(
                "matrix transpose requires rank at least 2 but input has rank {rank}",
            ))
            .into());
        }
        self.swap_axes(-2, -1)
    }

    /// Moves each `source` axis to its corresponding `destination`, shifting the other axes to preserve their relative
    /// order. Scalar axes move one axis, while arrays, vectors, and slices move several axes at once. Negative axes
    /// index from the end. This is the analogue of NumPy's
    /// [`moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html).
    /// An out-of-bounds or duplicate axis, or mismatched source and destination lengths, yields a [`TypeError`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{ArrayType, DataType, ProgramError, Transpose};
    /// let input = ArrayType::new_static(DataType::F32, [2, 3, 4]);
    /// assert_eq!(input.move_axis(0, -1)?, ArrayType::new_static(DataType::F32, [3, 4, 2]));
    /// assert_eq!(input.move_axis([0, 2], [2, 0])?, ArrayType::new_static(DataType::F32, [4, 3, 2]));
    /// # Ok::<(), ProgramError>(())
    /// ```
    ///
    /// # Parameters
    ///
    ///   - `source`: Input axes to move. Each normalized axis must appear at most once.
    ///   - `destination`: Final positions for the corresponding source axes, after all moves have taken place.
    ///     Each normalized position must appear at most once, and the number of positions must match `source`.
    #[inline]
    fn move_axis<S: Into<Axes>, D: Into<Axes>>(&self, source: S, destination: D) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        let source = source.into();
        let destination = destination.into();
        if source.len() != destination.len() {
            return Err(TypeError::invalid(format!(
                "`{}` move source has length {} but destination has length {}",
                TRANSPOSE_OPERATION_NAME,
                source.len(),
                destination.len(),
            ))
            .into());
        }
        let source = source
            .normalize(rank)
            .map_err(|error| TypeError::invalid(format!("`{TRANSPOSE_OPERATION_NAME}` move source {error}")))?;
        let destination = destination
            .normalize(rank)
            .map_err(|error| TypeError::invalid(format!("`{TRANSPOSE_OPERATION_NAME}` move destination {error}")))?;
        let mut permutation = (0..rank).filter(|axis| !source.contains(axis)).collect::<Vec<_>>();
        let mut moves = destination.into_iter().zip(source).collect::<Vec<_>>();
        moves.sort_by_key(|(destination, _)| *destination);
        for (destination, source) in moves {
            permutation.insert(destination, source);
        }
        self.transpose(permutation)
    }

    /// Swaps axes `i` and `j`, leaving every other axis in place. Negative axes index from the end. This is the
    /// analogue of NumPy's [`swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html). Returns
    /// `self` unchanged when both axes normalize to the same valid index. An out-of-bounds axis yields a [`TypeError`].
    ///
    /// # Examples
    ///
    /// ```rust
    /// # use ryft_core::{Array, ProgramError, Transpose};
    /// let input = Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap();
    /// assert_eq!(input.swap_axes(-2, -1)?, Array::matrix(3, 2, vec![1_i32, 4, 2, 5, 3, 6]).unwrap());
    /// # Ok::<(), ProgramError>(())
    /// ```
    #[inline]
    fn swap_axes<I: Into<Axis>, J: Into<Axis>>(&self, i: I, j: J) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        let i = i.into();
        let i = i.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("`{TRANSPOSE_OPERATION_NAME}` swap axis {i} is out of bounds for rank {rank}"))
        })?;
        let j = j.into();
        let j = j.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("`{TRANSPOSE_OPERATION_NAME}` swap axis {j} is out of bounds for rank {rank}"))
        })?;
        let mut permutation = (0..rank).collect::<Vec<_>>();
        permutation.swap(i, j);
        self.transpose(permutation)
    }
}

impl Transpose for Sharding {
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Reorder the per-dimension `ShardingDimension` entries so that output dimension `i` carries the entry of input
        // dimension `permutation[i]`, while leaving the reduction-state and manual-axis sets unchanged. This is the
        // sharding-level analogue of an array axis permutation. `permutation` must be a permutation of `0..rank`
        // matching this sharding's rank. Otherwise, a type error describing the offending dimension is returned.
        let permutation = permutation.into().normalize(self.rank())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let dimensions = permutation.iter().map(|axis| self.dimensions()[*axis].clone()).collect();
        Sharding::new(self.mesh().clone(), dimensions)
            .and_then(|sharding| sharding.with_unreduced_axes(self.unreduced_axes().clone()))
            .and_then(|sharding| sharding.with_reduced_axes(self.reduced_axes().clone()))
            .and_then(|sharding| sharding.with_varying_manual_axes(self.varying_manual_axes().clone()))
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl Transpose for ArrayType {
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate that `permutation` has length equal to the input rank and is a permutation of `0..rank` (i.e.,
        // every axis in range with no duplicates), and then return the input unchanged for the identity permutation
        // or permute its shape and output sharding, otherwise. Output axis `i` carries input axis `permutation[i]`.
        let permutation = permutation.into().normalize(self.rank())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let permuted = permutation.iter().map(|axis| self.dimension(*axis)).collect::<Vec<_>>();

        // The output sharding permutes its dimension entries the same way as the array axes (i.e., the reduction-state
        // and manual-axis sets are unchanged for every mesh axis type).
        let sharding = self.sharding().map(|sharding| sharding.transpose(permutation)).transpose()?;

        ArrayType::new(self.data_type(), Shape::new(permuted))
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl Transpose for Array {
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        // Validate the permutation and compute the output type (including sharding) via the type-level rule,
        // so that an out-of-range or duplicated axis is a clean error rather than an out-of-bounds panic.
        let permutation = permutation.into().normalize(self.r#type().rank())?;
        let output_type = self.r#type().transpose(permutation.clone())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let rank = self.r#type().rank();
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];

        // Structural-zero elements have no storage even when their logical shape is enormous.
        if bytes.is_empty() {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }
        let mut output_index = vec![0usize; rank];
        let mut input_index = vec![0usize; rank];
        for output_flat in 0..output_addressing.element_count() {
            for (position, &input_axis) in permutation.iter().enumerate() {
                input_index[input_axis] = output_index[position];
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            output_addressing.advance_index(&mut output_index);
        }
        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<TransposeOperation>>>>
    Transpose for V
{
    #[inline]
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        let permutation = permutation.into().normalize(self.r#type().rank())?;
        self.r#type().transpose(permutation.clone())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            TransposeOperation::new(permutation),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, DataType, Dimension, DimensionBounds, DimensionVariable, Layout, LogicalMesh, Memory,
        MeshAxis, MeshAxisType, RaggedAxis, Sharding, ShardingDimension, StridedLayout, Tile, TileDimension,
        TiledLayout, f8e8m0fnu,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, TransposableOperation, TranspositionContext,
    };
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::PartialValue;
    use crate::programs::{
        BindingRegionDriver, EmptyRegionDriver, ProgramBuilder, ProgramError, Provenance, ProvenanceScope, Typed,
    };

    use super::*;

    #[test]
    fn test_permutation_as_slice() {
        let permutation = Permutation::from([2, 0, 1]);
        assert_eq!(permutation.as_slice(), &[Axis::from(2), Axis::from(0), Axis::from(1)]);
        assert_eq!(permutation.as_ref(), &[Axis::from(2), Axis::from(0), Axis::from(1)]);
        assert_eq!(&*permutation, &[Axis::from(2), Axis::from(0), Axis::from(1)]);
    }

    #[test]
    fn test_permutation_inverse() {
        // Empty and identity permutations are their own inverses.
        assert_eq!(Permutation::default().inverse(), Ok(Permutation::default()));
        assert_eq!(Permutation::from(vec![0, 1, 2]).inverse(), Ok(Permutation::from(vec![0, 1, 2])));

        // A swap is its own inverse, while a cycle inverts to the reverse cycle.
        assert_eq!(Permutation::from(vec![1, 0]).inverse(), Ok(Permutation::from(vec![1, 0])));
        assert_eq!(Permutation::from(vec![2, 0, 1]).inverse(), Ok(Permutation::from(vec![1, 2, 0])));

        assert_eq!(Permutation::from([-1, 0, 1]).inverse(), Ok(Permutation::from([1, 2, 0])));

        // Invalid wrappers report the same precise validation errors as the type-level transpose contract.
        assert_eq!(
            Permutation::from(vec![2, 0]).inverse(),
            Err(TypeError::invalid("permutation axis 2 is out of bounds")),
        );
        assert_eq!(
            Permutation::from(vec![0, 0]).inverse(),
            Err(TypeError::invalid("permutation contains duplicate axis 0")),
        );

        // Inverting twice recovers the original permutation, and applying the inverse after the permutation restores
        // the identity ordering.
        let permutation = Permutation::from(vec![3, 0, 2, 1]);
        let inverse = permutation.inverse().unwrap();
        assert_eq!(inverse.inverse(), Ok(permutation.clone()));
        let axes = permutation.normalize(4).unwrap();
        let composed = inverse.normalize(4).unwrap().iter().map(|axis| axes[*axis]).collect::<Vec<_>>();
        assert_eq!(composed, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_permutation_normalize() {
        assert_eq!(Permutation::default().normalize(0), Ok(vec![]));
        assert_eq!(Permutation::from([-1, 0, 1]).normalize(3), Ok(vec![2, 0, 1]));
        assert_eq!(Permutation::from([2_usize, 0, 1]).normalize(3), Ok(vec![2, 0, 1]));
        assert_eq!(
            Permutation::from([0, -2]).normalize(2),
            Err(TypeError::invalid("permutation contains duplicate axis 0")),
        );
        assert_eq!(
            Permutation::from([-3, 0]).normalize(2),
            Err(TypeError::invalid("permutation axis -3 is out of bounds")),
        );
        assert_eq!(
            Permutation::from([i128::MIN]).normalize(1),
            Err(TypeError::invalid(format!("permutation axis {} is out of bounds", i128::MIN))),
        );
        assert_eq!(
            Permutation::from([0]).normalize(2),
            Err(TypeError::invalid("permutation has length 1 but input has rank 2")),
        );
        assert_eq!(
            Permutation::from([0, 0]).normalize(2),
            Err(TypeError::invalid("permutation contains duplicate axis 0")),
        );
        assert_eq!(
            Permutation::from([0, 2]).normalize(2),
            Err(TypeError::invalid("permutation axis 2 is out of bounds")),
        );
    }

    #[test]
    fn test_permutation_from() {
        // Common owned and borrowed axis collections convert to the canonical permutation representation.
        let axes = vec![2, 0, 1];
        assert_eq!(Permutation::from([2, 0, 1]), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(&[2, 0, 1]), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(axes.as_slice()), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(&axes), Permutation::from(axes));

        // Signed, explicitly typed unsigned, and `Axis` collections retain ergonomic conversions.
        let axes = [-1, 0, 1];
        let permutation = Permutation::from(axes);
        assert_eq!(Permutation::from(&axes), permutation);
        assert_eq!(Permutation::from(axes.as_slice()), permutation);
        assert_eq!(Permutation::from(axes.to_vec()), permutation);
        assert_eq!(Permutation::from(Axes::from(axes)), permutation);
        assert_eq!(Permutation::from(&Axes::from(axes)), permutation);
        assert_eq!(Permutation::from([Axis::from(-1), Axis::from(0), Axis::from(1)]), permutation);
        assert_eq!(Permutation::from([2_usize, 0, 1]), Permutation::from([2, 0, 1]));
        assert_eq!(format!("{permutation:?}"), "Permutation([-1, 0, 1])");
    }

    #[test]
    fn test_transpose() {
        let operation = TransposeOperation::new(vec![1, 0]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), TRANSPOSE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "TransposeOperation { permutation: Permutation([1, 0]) }");
        assert_eq!(format!("{operation}"), "transpose [permutation=[1, 0]]");
        assert_eq!(operation.permutation().as_slice(), &[Axis::from(1), Axis::from(0)]);

        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        // Program rendering uses the canonical operation name and includes the captured permutation.
        let mut builder = ProgramBuilder::<Array, TransposeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[3, 2] = transpose [permutation=[1, 0]] %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_transpose_type_inference() {
        let operation = TransposeOperation::new(vec![1, 0]);
        // Type inference permutes the input shape, including dynamic dimension sizes.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let placed_input_type = input_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type = output_type.clone().with_memory(Memory::Host { pinned: true });
        let rows = DimensionVariable::new("rows", DimensionBounds::unbounded());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![
                            Dimension::Dynamic(rows.clone()),
                            Dimension::Dynamic(columns.clone()),
                        ]),
                    )],
                    output_types = [ArrayType::new(
                        DataType::F64,
                        Shape::new(vec![Dimension::Dynamic(columns), Dimension::Dynamic(rows)]),
                    )],
                },
                {
                    input_types = [placed_input_type.clone()],
                    output_types = [placed_output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2)]))],
                    error = "permutation has length 2 but input has rank 1",
                },
            ],
        );
    }

    #[test]
    fn test_transpose_type_inference_sharding() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("u", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("v", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();

        // The first input dimension is sharded over `x`; reduction and manual-axis state rides along untouched.
        let input_sharding = Sharding::new(
            mesh.clone(),
            vec![
                ShardingDimension::sharded(["x"]),
                ShardingDimension::unconstrained(),
                ShardingDimension::replicated(),
            ],
        )
        .unwrap()
        .with_reduced_axes(["r"])
        .unwrap()
        .with_unreduced_axes(["u"])
        .unwrap()
        .with_varying_manual_axes(["v"])
        .unwrap();
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        )
        .with_sharding(input_sharding)
        .unwrap();

        // Permutation [2, 0, 1] makes output dimension i carry input dimension permutation[i].
        let operation = TransposeOperation::new(vec![2, 0, 1]);
        let expected = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]),
        )
        .with_sharding(
            Sharding::new(
                mesh,
                vec![
                    ShardingDimension::replicated(),
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::unconstrained(),
                ],
            )
            .unwrap()
            .with_reduced_axes(["r"])
            .unwrap()
            .with_unreduced_axes(["u"])
            .unwrap()
            .with_varying_manual_axes(["v"])
            .unwrap(),
        )
        .unwrap();
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [{ input_types = [input_type], output_types = [expected] }],
        );

        // An input without a sharding yields an output without one.
        let unsharded = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&unsharded), &[]).unwrap()[0].sharding(), None);
    }

    #[test]
    fn test_transpose_type_inference_invalid_permutation() {
        check_operation_type_inference!(
            operation = TransposeOperation::new([0, 2]),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F64, [2, 3])],
                error = "permutation axis 2 is out of bounds",
            }],
        );
        check_operation_type_inference!(
            operation = TransposeOperation::new([0, 0]),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F64, [2, 3])],
                error = "permutation contains duplicate axis 0",
            }],
        );
    }

    #[test]
    fn test_transpose_interpretation() {
        let operation = TransposeOperation::new(vec![1, 0]);
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        // Interpretation reorders the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = operation
            .clone()
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        assert_eq!(
            operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_transpose_interpretation_invalid_permutation() {
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            TransposeOperation::new([0]).interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[input]),
            Err(ProgramError::Type(TypeError::invalid("permutation has length 1 but input has rank 2"))),
        );
    }

    #[test]
    fn test_transpose_interpretation_signed_axes() {
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 3]), &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.transpose([-1, -2]),
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [3, 2]), &[1, 4, 2, 5, 3, 6]).unwrap()),
        );
        assert_eq!(input.transpose([0, -1]), Ok(input.clone()));
        assert_eq!(input.transpose([0, -2]), Err(TypeError::invalid("permutation contains duplicate axis 0").into()));
        assert_eq!(input.transpose([-3, 0]), Err(TypeError::invalid("permutation axis -3 is out of bounds").into()));
        assert_eq!(
            input.transpose([0]),
            Err(TypeError::invalid("permutation has length 1 but input has rank 2").into()),
        );
        let scalar = ArrayType::scalar(DataType::F32);
        assert_eq!(scalar.transpose(Axes::default()), Ok(scalar));
    }

    #[test]
    fn test_transpose_partial_evaluation() {
        // Check standard partial evaluation with known and residual operands.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let expected = Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.clone().r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_transpose_batching() {
        // Check that batching lifts the per-item permutation while leaving the mapped axis in place.
        let batched_input = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()])),
            &(0..24).map(|value| value as f64).collect::<Vec<_>>(),
        )
        .unwrap();
        let batched_output = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 4.into(), 3.into()])),
            &[
                0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 12.0, 16.0, 20.0, 13.0, 17.0, 21.0, 14.0,
                18.0, 22.0, 15.0, 19.0, 23.0,
            ],
        )
        .unwrap();
        // Signed axes resolve against the logical rank before the mapped axis is inserted.
        for permutation in [Permutation::from([1, 0]), Permutation::from([-1, -2])] {
            check_operation_batching!(
                @exact,
                operation = TransposeOperation::new(permutation),
                axis_size = 2,
                cases = [{
                    inputs = [(@mapped(axis = 0), batched_input.clone())],
                    outputs = [(@mapped(axis = 0), batched_output.clone())],
                }],
            );
        }
    }

    #[test]
    fn test_transpose_batching_replicated() {
        check_operation_batching!(
            @exact,
            operation = TransposeOperation::new([1, 0]),
            axis_size = 2,
            cases = [{
                inputs = [(@replicated, Array::matrix(2, 3, vec![1_i32, 2, 3, 4, 5, 6]).unwrap())],
                outputs = [(@replicated, Array::matrix(3, 2, vec![1_i32, 4, 2, 5, 3, 6]).unwrap())],
            }],
        );
    }

    #[test]
    fn test_transpose_batching_invalid_permutation() {
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        let input = ArrayBatch::new(
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 3]), &(0..12).collect::<Vec<i32>>())
                .unwrap(),
            BatchAxis::new(1),
        )
        .unwrap();

        // Validate logical axes before lifting them around the mapped axis; even usize::MAX must report an error.
        assert!(matches!(
            TransposeOperation::new([usize::MAX, 1]).batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == format!("permutation axis {} is out of bounds", usize::MAX),
        ));
        assert!(matches!(
            TransposeOperation::new([0, 0]).batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "permutation contains duplicate axis 0",
        ));
        assert!(matches!(
            TransposeOperation::new([0, 2]).batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "permutation axis 2 is out of bounds",
        ));
    }

    #[test]
    fn test_transpose_batching_nonleading_axis() {
        // The batch axis may occupy any physical position. The lifted permutation leaves it in that position while
        // applying the logical rank-3 cycle around it.
        let middle_axis_input = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into(), 4.into()])),
            &(0..48).map(f64::from).collect::<Vec<_>>(),
        )
        .unwrap();
        let middle_axis_output = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 2.into(), 3.into()])),
            &[
                0.0, 4.0, 8.0, 24.0, 28.0, 32.0, 12.0, 16.0, 20.0, 36.0, 40.0, 44.0, 1.0, 5.0, 9.0, 25.0, 29.0, 33.0,
                13.0, 17.0, 21.0, 37.0, 41.0, 45.0, 2.0, 6.0, 10.0, 26.0, 30.0, 34.0, 14.0, 18.0, 22.0, 38.0, 42.0,
                46.0, 3.0, 7.0, 11.0, 27.0, 31.0, 35.0, 15.0, 19.0, 23.0, 39.0, 43.0, 47.0,
            ],
        )
        .unwrap();
        let trailing_axis_input = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into(), 2.into()])),
            &(0..48).map(f64::from).collect::<Vec<_>>(),
        )
        .unwrap();
        let trailing_axis_output = Array::from_elements::<f64>(
            ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 3.into(), 2.into()])),
            &[
                0.0, 1.0, 8.0, 9.0, 16.0, 17.0, 24.0, 25.0, 32.0, 33.0, 40.0, 41.0, 2.0, 3.0, 10.0, 11.0, 18.0, 19.0,
                26.0, 27.0, 34.0, 35.0, 42.0, 43.0, 4.0, 5.0, 12.0, 13.0, 20.0, 21.0, 28.0, 29.0, 36.0, 37.0, 44.0,
                45.0, 6.0, 7.0, 14.0, 15.0, 22.0, 23.0, 30.0, 31.0, 38.0, 39.0, 46.0, 47.0,
            ],
        )
        .unwrap();
        check_operation_batching!(
            @exact,
            operation = TransposeOperation::new(vec![2, 0, 1]),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 1), middle_axis_input)],
                    outputs = [(@mapped(axis = 1), middle_axis_output)],
                },
                {
                    inputs = [(@mapped(axis = 3), trailing_axis_input)],
                    outputs = [(@mapped(axis = 3), trailing_axis_output)],
                },
            ],
        );
    }

    #[test]
    fn test_transpose_batching_ragged_axes() {
        // Ragged metadata names physical packed axes, so the lifted transpose must apply its inverse axis map to the
        // ragged dimension and every extent axis while preserving the mapped batch axis.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(5)).unwrap());
        let extents = Array::matrix(2, 2, vec![1_i32, 4, 2, 3]).unwrap();
        let packed = Array::from_elements::<f64>(
            ArrayType::new_static(DataType::F64, [2, 2, 3, 4]),
            &(0..48).map(f64::from).collect::<Vec<_>>(),
        )
        .unwrap();
        let expected_value = packed.transpose([3, 1, 0, 2]).unwrap();
        let input = ArrayBatch::new(packed, BatchAxis::new(1))
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(3, extents.clone(), length.clone(), vec![0, 1])])
            .unwrap();
        assert!(matches!(
            TransposeOperation::new([0, 1]).batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &EmptyRegionDriver,
                std::slice::from_ref(&input),
            ),
            Err(BatchingError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "permutation has length 2 but input has rank 3",
        ));
        let output = TransposeOperation::new([2, 0, 1])
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &EmptyRegionDriver, &[input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(output.value(), &expected_value);
        assert_eq!(output.batch_axis(), BatchAxis::new(1));
        assert_eq!(output.ragged_axes(), &[RaggedAxis::new(0, extents, length.clone(), vec![2, 1])]);
        assert_eq!(
            output.unbatched_type(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(length), Dimension::Static(2), Dimension::Static(3)]),
            ),
        );
    }

    #[test]
    fn test_transpose_batching_symbolic_axis() {
        // Transpose only reorders axes, so its batching rule also works when the mapped dimension is symbolic. It
        // stages the physical permutation [3, 1, 0, 2] without demanding a concrete batch size from the input type.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let symbolic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![2.into(), Dimension::Dynamic(batch.clone()), 3.into(), 4.into()]),
        );
        let symbolic_input = context.input(symbolic_input_type.clone());
        let symbolic_input = ArrayBatch::new(symbolic_input, BatchAxis::new(1)).unwrap();
        let symbolic_output = TransposeOperation::new([2, 0, 1])
            .batch(&BatchingContext::new(context.clone(), 2), &EmptyRegionDriver, &[symbolic_input])
            .unwrap()
            .into_parts()
            .0
            .remove(0);
        assert_eq!(symbolic_output.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            symbolic_output.r#type().as_ref(),
            &ArrayType::new(DataType::F64, Shape::new(vec![4.into(), Dimension::Dynamic(batch), 2.into(), 3.into()])),
        );
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        assert_eq!(
            format!("{}", context.builder().borrow().instructions()[0].operation()),
            "transpose [permutation=[3, 1, 0, 2]]",
        );
    }

    #[test]
    fn test_transpose_differentiation() {
        // Transpose is structural-linear: its JVP applies the same permutation and its pullback applies the inverse.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                primals = [Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap()],
                tangents = [Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0]).unwrap()],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0]).unwrap()],
                tangent_outputs = [Array::matrix(2, 2, vec![5.0, 7.0, 6.0, 8.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_transpose_differentiation_inverse_cycle() {
        // A non-self-inverse cycle exercises the distinction between the forward rule and inverse pullback rule.
        let cycle_input_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()]));
        let cycle_output_type = ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 3.into()]));
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = TransposeOperation::new(vec![2, 0, 1]),
            cases = [{
                primals = [Array::from_elements::<f64>(
                    cycle_input_type.clone(),
                    &(0..24).map(f64::from).collect::<Vec<_>>(),
                ).unwrap()],
                tangents = [Array::from_elements::<f64>(
                    cycle_input_type.clone(),
                    &(24..48).map(f64::from).collect::<Vec<_>>(),
                ).unwrap()],
                primal_outputs = [Array::from_elements::<f64>(cycle_output_type.clone(), &[
                        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0,
                        14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
                    ]).unwrap()],
                tangent_outputs = [Array::from_elements::<f64>(cycle_output_type.clone(), &[
                        24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 26.0, 30.0,
                        34.0, 38.0, 42.0, 46.0, 27.0, 31.0, 35.0, 39.0, 43.0, 47.0,
                    ]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_transpose_differentiation_structural_zero() {
        let cycle_input_type = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        let cycle_output_type = ArrayType::new_static(DataType::F64, [4, 2, 3]);
        // A structural zero tangent does not stage a tangent transpose.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(cycle_input_type.clone());
        let duals = TransposeOperation::new([2, 0, 1])
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(primal).unwrap()],
            )
            .unwrap();
        assert!(duals[0].tangent().is_zero());
        assert_eq!(duals[0].tangent().r#type().as_ref(), &cycle_output_type.tangent().unwrap());
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_transpose_transposition() {
        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))))],
                output_cotangents = [Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap()],
                input_cotangents = [Array::matrix(2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_transpose_transposition_inverse_cycle() {
        let cycle_input_type = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        let cycle_output_type = ArrayType::new_static(DataType::F64, [4, 2, 3]);
        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![2, 0, 1]),
            cases = [{
                inputs = [(@linear(type = cycle_input_type.clone()))],
                output_cotangents = [Array::from_elements::<f64>(
                    cycle_output_type.clone(),
                    &(0..24).map(f64::from).collect::<Vec<_>>(),
                ).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(cycle_input_type.clone(), &[
                        0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0,
                        21.0, 4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0,
                    ]).unwrap()],
            }],
        );
    }

    #[test]
    fn test_transpose_transposition_structural_zero() {
        let cycle_input_type = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        let cycle_output_type = ArrayType::new_static(DataType::F64, [4, 2, 3]);
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = {
            let mut rule_context = TranspositionContext::new(context.clone());
            let rule_inputs = &[PartialValue::Unknown(cycle_input_type.clone())];
            let accumulators = rule_context.cotangent_accumulators(rule_inputs, &[]).unwrap();
            TransposeOperation::new([2, 0, 1])
                .transpose(
                    &mut rule_context,
                    &EmptyRegionDriver,
                    rule_inputs,
                    &[MaybeZero::Zero(cycle_output_type.cotangent().unwrap())],
                    &accumulators,
                )
                .unwrap();
            rule_context.take_cotangents(&accumulators).unwrap()
        };
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &cycle_input_type.cotangent().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_transpose_transposition_layout() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]));
        let placed_input_type = input_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type = output_type.clone().with_memory(Memory::Host { pinned: true });
        // The inverse permutation restores the complete input cotangent type, including placement metadata that the
        // forward transpose intentionally clears because it cannot infer a new physical layout.
        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                inputs = [(@linear(type = placed_input_type.clone()))],
                output_cotangents = [Array::from_elements::<f64>(
                    placed_output_type,
                    &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                ).unwrap()],
                input_cotangents = [Array::from_elements::<f64>(
                    placed_input_type,
                    &[1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
                ).unwrap()],
            }],
        );
    }

    #[test]
    fn test_transpose_transpose() {
        let placed_input_type = ArrayType::new_static(DataType::F64, [2, 3])
            .with_layout(Layout::Strided(StridedLayout::new(vec![24, 8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(placed_input_type.transpose([0, 1]), Ok(placed_input_type.clone()));
        // Identity transpose preserves the exact tracer and placement metadata without staging an instruction.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(placed_input_type.clone());
        let output = input.transpose([0, 1]).unwrap();
        assert_eq!(output.atom_id(), input.atom_id());
        assert_eq!(output.r#type(), input.r#type());
        assert!(context.builder().borrow().instructions().is_empty());
        let transposed = input.transpose([1, 0]).unwrap();
        assert_eq!(
            transposed.r#type().as_ref(),
            &ArrayType::new_static(DataType::F64, [3, 2]).with_memory(Memory::Host { pinned: true }),
        );
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_transpose_transpose_invalid_output_count() {
        /// Array wrapper that dispatches through a deliberately malformed context.
        #[derive(Clone, Debug)]
        struct DispatchArray {
            array: Array,
            output_count: usize,
        }

        impl Display for DispatchArray {
            fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(formatter, "{}", self.array)
            }
        }

        impl Parameter for DispatchArray {}

        impl Typed for DispatchArray {
            type Type = ArrayType;

            fn r#type(&self) -> Cow<'_, ArrayType> {
                self.array.r#type()
            }
        }

        impl Value for DispatchArray {
            type DispatchDomain = InvalidOutputContext;
            type ExecutionDomain = InvalidOutputContext;

            fn dispatch_domain(&self) -> Self::DispatchDomain {
                InvalidOutputContext(self.output_count)
            }

            fn execution_domain(&self) -> Self::ExecutionDomain {
                self.dispatch_domain()
            }
        }

        /// Context that violates the transpose output arity while accepting otherwise valid inputs.
        #[derive(Clone)]
        struct InvalidOutputContext(usize);

        impl Domain for InvalidOutputContext {
            type Type = ArrayType;
            type Value = DispatchArray;
            type Constant = Array;
            type Operation = TransposeOperation;
        }

        impl Context for InvalidOutputContext {
            fn lift(&self, array: Array) -> Result<DispatchArray, ProgramError> {
                Ok(DispatchArray { array, output_count: self.0 })
            }

            fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
                &self,
                _operation: O,
                _driver: D,
                inputs: &[DispatchArray],
            ) -> Result<Vec<DispatchArray>, ProgramError> {
                Ok(vec![inputs[0].clone(); self.0])
            }

            fn is_eager(&self) -> bool {
                true
            }

            fn provenance(&self) -> Provenance {
                Provenance::unknown()
            }

            fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, _origin: Provenance, function: F) -> R {
                function()
            }

            fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, _scope: ProvenanceScope, function: F) -> R {
                function()
            }
        }

        let array = Array::from_elements(ArrayType::new_static(DataType::I32, [1, 2]), &[3_i32, 7]).unwrap();
        let input = InvalidOutputContext(0).lift(array.clone()).unwrap();
        assert!(matches!(input.transpose([1, 0]), Err(ProgramError::InvalidOutputCount { expected: 1, actual: 0 })));
        let input = InvalidOutputContext(2).lift(array).unwrap();
        assert!(matches!(input.transpose([1, 0]), Err(ProgramError::InvalidOutputCount { expected: 1, actual: 2 })));
    }

    #[test]
    fn test_transpose_transpose_reversed() {
        let input = Array::from_elements(ArrayType::new_static(DataType::I32, [2, 1, 3]), &[1, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(
            input.transpose_reversed(),
            Ok(Array::from_elements(ArrayType::new_static(DataType::I32, [3, 1, 2]), &[1, 4, 2, 5, 3, 6]).unwrap()),
        );
        let scalar = ArrayType::scalar(DataType::F32);
        assert_eq!(scalar.transpose_reversed(), Ok(scalar));
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let vector = context.input(ArrayType::new_static(DataType::F32, [3]));
        assert_eq!(vector.transpose_reversed().unwrap().atom_id(), vector.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_transpose_matrix_transpose() {
        let input = Array::from_elements(
            ArrayType::new_static(DataType::C64, [2, 1, 2]),
            &[
                ComplexNumber::new(1f32, 2.),
                ComplexNumber::new(3., 4.),
                ComplexNumber::new(5., 6.),
                ComplexNumber::new(7., 8.),
            ],
        )
        .unwrap();
        // Leading batch axes stay fixed and imaginary components retain their signs.
        assert_eq!(
            input.matrix_transpose(),
            Ok(Array::from_elements(
                ArrayType::new_static(DataType::C64, [2, 2, 1]),
                &[
                    ComplexNumber::new(1f32, 2.),
                    ComplexNumber::new(3., 4.),
                    ComplexNumber::new(5., 6.),
                    ComplexNumber::new(7., 8.),
                ],
            )
            .unwrap()),
        );
        let empty = ArrayType::new_static(DataType::F32, [5, 0, 2]);
        assert_eq!(empty.matrix_transpose(), Ok(ArrayType::new_static(DataType::F32, [5, 2, 0])));
        assert_eq!(
            ArrayType::scalar(DataType::F32).matrix_transpose(),
            Err(TypeError::invalid("matrix transpose requires rank at least 2 but input has rank 0").into()),
        );
        assert_eq!(
            ArrayType::new_static(DataType::F32, [2]).matrix_transpose(),
            Err(TypeError::invalid("matrix transpose requires rank at least 2 but input has rank 1").into()),
        );
    }

    #[test]
    fn test_transpose_move_axis() {
        // `move_axis` shifts intervening dimensions while preserving their relative order.
        // On a matrix, moving axis 0 to position 1 is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let output = matrix.move_axis(0, 1).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // On a rank-3 array, moving axis 0 to the last position shifts the other axes left to preserve their relative
        // order, so [2, 3, 4] becomes [3, 4, 2] (the permutation [1, 2, 0]).
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_elements::<f64>(input_type, &values).unwrap().move_axis(0, 2).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)]),
            ),
        );

        // `from == to` leaves the array unchanged.
        assert_eq!(matrix.move_axis(1, 1).unwrap(), matrix);

        // An out-of-bounds source axis is a clean error rather than an out-of-bounds panic, since the built
        // permutation is validated by the type-level transpose.
        assert_eq!(
            matrix.move_axis(2, 0),
            Err(ProgramError::Type(TypeError::invalid("`transpose` move source axis 2 is out of bounds for rank 2"))),
        );
        assert_eq!(
            matrix.move_axis(0, 2),
            Err(TypeError::invalid("`transpose` move destination axis 2 is out of bounds for rank 2").into()),
        );
    }

    #[test]
    fn test_transpose_move_axis_multiple() {
        // Negative indices are normalized before multiple axes are moved to their paired destinations.
        let rank_four_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into(), 5.into()]));
        assert_eq!(
            rank_four_type.move_axis(1, -1).unwrap().shape(),
            &Shape::new(vec![2.into(), 4.into(), 5.into(), 3.into()]),
        );
        assert_eq!(
            rank_four_type.move_axis(-1, 1).unwrap().shape(),
            &Shape::new(vec![2.into(), 5.into(), 3.into(), 4.into()]),
        );
        assert_eq!(
            rank_four_type.move_axis([0, 1], [-1, -2]).unwrap().shape(),
            &Shape::new(vec![4.into(), 5.into(), 3.into(), 2.into()]),
        );
        assert_eq!(
            rank_four_type.move_axis([0, 2], [2, 0]).unwrap().shape(),
            &Shape::new(vec![4.into(), 3.into(), 2.into(), 5.into()]),
        );
        assert_eq!(rank_four_type.move_axis(-1, 3), Ok(rank_four_type.clone()));
        assert_eq!(rank_four_type.move_axis(Axes::default(), Axes::default()), Ok(rank_four_type.clone()));
        assert_eq!(
            rank_four_type.move_axis([0, 1], [2]),
            Err(ProgramError::Type(TypeError::invalid(
                "`transpose` move source has length 2 but destination has length 1",
            ))),
        );
        assert_eq!(
            rank_four_type.move_axis([0, -4], [1, 2]),
            Err(ProgramError::Type(TypeError::invalid("`transpose` move source axes contain duplicate axis 0"))),
        );
        assert_eq!(
            rank_four_type.move_axis([0, 1], [0, -4]),
            Err(ProgramError::Type(TypeError::invalid("`transpose` move destination axes contain duplicate axis 0"))),
        );
        assert_eq!(
            rank_four_type.move_axis(-5, 0),
            Err(ProgramError::Type(TypeError::invalid("`transpose` move source axis -5 is out of bounds for rank 4"))),
        );
        assert_eq!(
            ArrayType::scalar(DataType::F64).move_axis(0, 0),
            Err(ProgramError::Type(TypeError::invalid("`transpose` move source axis 0 is out of bounds for rank 0"))),
        );
        // Moving multiple axes permutes the values as well as their shape, with each destination paired to its source.
        let input =
            Array::from_elements(ArrayType::new_static(DataType::I32, [2, 2, 3]), &(0..12).collect::<Vec<i32>>())
                .unwrap();
        let output = input.move_axis([0, 2], [2, 0]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::I32, [3, 2, 2]));
        assert_eq!(output.elements::<i32>(), Ok(vec![0, 6, 3, 9, 1, 7, 4, 10, 2, 8, 5, 11]));
    }

    #[test]
    fn test_transpose_swap_axes() {
        // `swap_axes` exchanges exactly two dimensions and validates both indices.
        // Swapping axes 0 and 1 of a matrix is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let swapped = matrix.swap_axes(0, 1).unwrap();
        assert_eq!(
            swapped.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(swapped.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Swapping is symmetric in its axis arguments.
        assert_eq!(matrix.swap_axes(1, 0).unwrap(), swapped);

        // Swapping the first two axes of a rank-3 array leaves the untouched trailing axis in place, so [2, 3, 4]
        // becomes [3, 2, 4] (the permutation [1, 0, 2]).
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_elements::<f64>(input_type, &values).unwrap().swap_axes(0, 1).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(3), Dimension::Static(2), Dimension::Static(4)]),
            ),
        );

        // `i == j` leaves the array unchanged.
        assert_eq!(matrix.swap_axes(1, 1).unwrap(), matrix);

        assert_eq!(matrix.swap_axes(-1, 1), Ok(matrix.clone()));
        assert_eq!(matrix.swap_axes(-1, 0), Ok(swapped));

        // An out-of-bounds axis is a clean error rather than an out-of-bounds panic.
        assert_eq!(
            matrix.swap_axes(2, 0),
            Err(ProgramError::Type(TypeError::invalid("`transpose` swap axis 2 is out of bounds for rank 2"))),
        );
        assert!(matches!(
            matrix.swap_axes(0, 2),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`transpose` swap axis 2 is out of bounds for rank 2",
        ));
        assert!(matches!(
            Array::scalar(1_i32).unwrap().swap_axes(0, 0),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`transpose` swap axis 0 is out of bounds for rank 0",
        ));
    }

    #[test]
    fn test_sharding_transpose() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let sharding = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::Unconstrained, ShardingDimension::Replicated],
        )
        .unwrap();
        assert_eq!(
            sharding.transpose([2, 0, 1]),
            Ok(Sharding::new(
                mesh,
                vec![
                    ShardingDimension::Replicated,
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::Unconstrained,
                ],
            )
            .unwrap()),
        );
        // Direct sharding transposition validates the complete permutation even when duplicate axes refer to
        // replicated or unconstrained dimensions.
        assert_eq!(
            sharding.transpose([1, 1, 0]),
            Err(ProgramError::Type(TypeError::invalid("permutation contains duplicate axis 1"))),
        );
        assert_eq!(
            sharding.transpose([0, 1]),
            Err(ProgramError::Type(TypeError::invalid("permutation has length 2 but input has rank 3"))),
        );
        assert_eq!(
            sharding.transpose([0, 1, 3]),
            Err(ProgramError::Type(TypeError::invalid("permutation axis 3 is out of bounds"))),
        );
    }

    #[test]
    fn test_array_type_transpose() {
        assert_eq!(
            ArrayType::new_static(DataType::F32, [2, 3]).transpose([1, 0]),
            Ok(ArrayType::new_static(DataType::F32, [3, 2])),
        );
        // Transpose is independent of element representation, including non-differentiable, complex, structural-zero,
        // and low-precision element types.
        for data_type in [
            DataType::Boolean,
            DataType::I32,
            DataType::U64,
            DataType::F8E8M0FNU,
            DataType::F32,
            DataType::F64,
            DataType::C64,
            DataType::C128,
            DataType::Zero,
        ] {
            let input = ArrayType::new(data_type, Shape::new(vec![2.into(), 3.into(), 4.into()]));
            let expected = ArrayType::new(data_type, Shape::new(vec![4.into(), 2.into(), 3.into()]));
            assert_eq!(input.transpose([2, 0, 1]), Ok(expected));
        }
    }

    #[test]
    fn test_array_transpose() {
        // Rank-2 swap of a row-major 2x3 payload.
        let output = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap().transpose(vec![1, 0]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)])),
        );
        assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // The eager kernel reorders exact typed payloads without changing their element representation.
        let input = Array::matrix(2, 3, vec![false, true, true, false, false, true]).unwrap();
        assert_eq!(
            input.transpose([1, 0]),
            Ok(Array::matrix(3, 2, vec![false, false, true, false, true, true]).unwrap())
        );
        let input = Array::matrix(2, 3, (0..6).collect::<Vec<i32>>()).unwrap();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, vec![0, 3, 1, 4, 2, 5]).unwrap()));
        let input = Array::matrix(2, 3, (1..=6).map(f8e8m0fnu::from_bits).collect()).unwrap();
        let expected = [1, 4, 2, 5, 3, 6].map(f8e8m0fnu::from_bits).to_vec();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, expected).unwrap()));
        let input =
            Array::matrix(2, 3, (0..6).map(|value| ComplexNumber::new(f64::from(value), -f64::from(value))).collect())
                .unwrap();
        let expected = [0, 3, 1, 4, 2, 5].map(|value| ComplexNumber::new(f64::from(value), -f64::from(value))).to_vec();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, expected).unwrap()));
    }

    #[test]
    fn test_array_transpose_layout() {
        // Transposition reads the input through its physical layout and produces the canonical layout-free output.
        let input_type =
            ArrayType::new_static(DataType::U16, [2, 3]).with_layout(Layout::Strided(StridedLayout::new(vec![8, 2])));
        let input = Array::from_elements(input_type, &[1u16, 2, 3, 4, 5, 6]).unwrap();
        let output = input.transpose([1, 0]).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::new_static(DataType::U16, [3, 2]));
        assert_eq!(output.elements::<u16>(), Ok(vec![1, 4, 2, 5, 3, 6]));
        assert_eq!(output.storage_bytes(), [1, 0, 4, 0, 2, 0, 5, 0, 3, 0, 6, 0]);

        // Negative strides reverse physical traversal without changing the requested logical transpose.
        let input_type =
            ArrayType::new_static(DataType::I32, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8, 4])));
        let input = Array::from_elements(input_type, &[1_i32, 2, 3, 4]).unwrap();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(2, 2, vec![1_i32, 3, 2, 4]).unwrap()));

        // Tiled storage is decoded before permuting, including the padding in a partial tile.
        let input_type = ArrayType::new_static(DataType::I32, [2, 3])
            .with_layout(Layout::Tiled(TiledLayout::new(vec![1, 0], vec![Tile::new(vec![TileDimension::Sized(2)])])));
        let input = Array::from_elements(input_type, &[1_i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, vec![1_i32, 4, 2, 5, 3, 6]).unwrap()));
    }

    #[test]
    fn test_array_transpose_cycle() {
        // Rank-3 permutation moving the last axis to the front.
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_elements::<f64>(input_type, &values).unwrap().transpose(vec![2, 0, 1]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)]),
            ),
        );
        assert_eq!(
            output.to_f64s(),
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0,
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
            ],
        );
    }

    #[test]
    fn test_array_transpose_empty() {
        // Rank-0 and empty payloads pass through unchanged.
        let output = Array::scalar(42.0).unwrap().transpose(Vec::<usize>::new()).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output.to_f64s(), vec![42.0]);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let output = Array::from_elements::<f64>(input_type, &[]).unwrap().transpose(vec![1, 0]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(0)])),
        );
        assert_eq!(output.to_f64s(), Vec::<f64>::new());

        // Empty arrays do not calculate strides for irrelevant dimensions, whose products need not fit in `usize`.
        let input = Array::new(ArrayType::new_static(DataType::F64, [0, usize::MAX, usize::MAX]), Vec::new()).unwrap();
        assert_eq!(
            input.transpose([1, 2, 0]),
            Ok(Array::new(ArrayType::new_static(DataType::F64, [usize::MAX, usize::MAX, 0]), Vec::new()).unwrap()),
        );
    }

    #[test]
    fn test_array_transpose_structural_zero() {
        // Structural zeros have no physical elements to move, even when their logical element count is enormous.
        let input = Array::new(ArrayType::new_static(DataType::Zero, [usize::MAX, 1]), Vec::new()).unwrap();
        let output = input.transpose([1, 0]).unwrap();
        assert_eq!(output.r#type().as_ref(), &ArrayType::new_static(DataType::Zero, [1, usize::MAX]));
        assert!(output.storage_bytes().is_empty());
    }

    #[test]
    fn test_array_transpose_identity() {
        let input_type =
            ArrayType::new_static(DataType::F32, [2]).with_layout(Layout::Strided(StridedLayout::new(vec![-8])));
        let input = Array::from_elements(input_type, &[f32::from_bits(0x7fc01234), -0.0_f32]).unwrap();
        let output = input.transpose([0]).unwrap();
        assert_eq!(output.r#type(), input.r#type());
        assert_eq!(output.storage_bytes(), input.storage_bytes());
        assert!(Arc::ptr_eq(output.shared_storage(), input.shared_storage()));
    }

    #[test]
    fn test_array_transpose_invalid_permutation() {
        // An invalid permutation is a clean error rather than an out-of-bounds panic, since the value-level transpose
        // validates the permutation through the type-level rule before indexing.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            matrix.transpose(vec![1]),
            Err(ProgramError::Type(TypeError::invalid("permutation has length 1 but input has rank 2"))),
        );
        assert_eq!(
            matrix.transpose(vec![0, 2]),
            Err(ProgramError::Type(TypeError::invalid("permutation axis 2 is out of bounds"))),
        );
        assert_eq!(
            matrix.transpose(vec![0, 0]),
            Err(ProgramError::Type(TypeError::invalid("permutation contains duplicate axis 0"))),
        );
    }
}
