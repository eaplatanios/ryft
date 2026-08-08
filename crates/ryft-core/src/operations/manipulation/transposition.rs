use std::fmt::{Debug, Display};
use std::ops::Deref;

use crate::arrays::{ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, Shape, Sharding};
use crate::axes::{Axes, Axis};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError, InterpretableBatchableOperation,
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

/// Canonical operation name for [`TransposeOperation`].
pub const TRANSPOSE_OPERATION_NAME: &str = "transpose";

/// Axis permutation used by [`TransposeOperation`] and the [`Transpose`] capability. For each output axis `i`,
/// `permutation[i]` is the input axis routed to it. [`Permutation`] is a thin wrapper over the axis vector. It
/// [`Deref`]s to `[usize]` and implements [`AsRef<[usize]>`](AsRef), and so it composes with everything that accepts an
/// axis slice. It supports [`From`] conversion from owned vectors and arrays as well as borrowed permutations, vectors,
/// arrays, and slices. Validity (i.e., being a bijection of `0..len`) is not enforced at construction time. It is
/// validated against a concrete input rank by the type-level [`Transpose`] rule.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct Permutation(Vec<usize>);

impl Permutation {
    /// Returns the permutation axes as a slice, where element `i` is the input axis routed to output axis `i`.
    #[inline]
    pub fn as_slice(&self) -> &[usize] {
        self.0.as_slice()
    }

    /// Returns the inverse [`Permutation`] of this one (i.e., the one that _undoes_ it). Transposing by a permutation
    /// and then by its inverse restores the original axis order. Returns a [`TypeError`] if this value is not a
    /// bijection over `0..self.len()`.
    #[inline]
    pub fn inverse(&self) -> Result<Permutation, TypeError> {
        self.validate(self.len())?;
        let mut inverse = vec![0usize; self.0.len()];
        for (position, axis) in self.0.iter().enumerate() {
            inverse[*axis] = position;
        }
        Ok(Permutation(inverse))
    }

    /// Validates that this [`Permutation`] is a bijection over the axes of an input with the provided rank.
    fn validate(&self, rank: usize) -> Result<(), TypeError> {
        if self.len() != rank {
            return Err(TypeError::invalid(format!(
                "'{}' permutation has length {} but input has rank {}",
                TRANSPOSE_OPERATION_NAME,
                self.len(),
                rank,
            )));
        }
        let mut seen = vec![false; rank];
        for axis in self.iter() {
            if *axis >= rank {
                return Err(TypeError::invalid(format!(
                    "'{TRANSPOSE_OPERATION_NAME}' permutation axis {axis} is out of bounds",
                )));
            }
            if seen[*axis] {
                return Err(TypeError::invalid(format!(
                    "'{TRANSPOSE_OPERATION_NAME}' permutation contains duplicate axis {axis}",
                )));
            }
            seen[*axis] = true;
        }
        Ok(())
    }
}

impl Deref for Permutation {
    type Target = [usize];

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl AsRef<[usize]> for Permutation {
    #[inline]
    fn as_ref(&self) -> &[usize] {
        &self.0
    }
}

impl From<Vec<usize>> for Permutation {
    #[inline]
    fn from(axes: Vec<usize>) -> Self {
        Self(axes)
    }
}

impl From<&Vec<usize>> for Permutation {
    #[inline]
    fn from(axes: &Vec<usize>) -> Self {
        Self(axes.clone())
    }
}

impl From<&[usize]> for Permutation {
    #[inline]
    fn from(axes: &[usize]) -> Self {
        Self(axes.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Permutation {
    #[inline]
    fn from(axes: [usize; N]) -> Self {
        Self(axes.into())
    }
}

impl<const N: usize> From<&[usize; N]> for Permutation {
    #[inline]
    fn from(axes: &[usize; N]) -> Self {
        Self(axes.to_vec())
    }
}

impl From<&Permutation> for Permutation {
    #[inline]
    fn from(permutation: &Permutation) -> Self {
        permutation.clone()
    }
}

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
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("permutation", format_args!("{:?}", self.permutation.as_slice())))
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

impl_differentiable_operation! {
    TransposeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<TransposeOperation>,
        C::Value: Transpose,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `TransposeOperation`. `transpose` is structural-linear, and so the
            // tangent is the same transpose applied to the operand tangent. The shared all-zero fast path handles a
            // zero operand tangent before this rule is consulted, so the operand tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().transpose(operation.permutation())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
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
        |operation, _context, _driver, inputs, outputs| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            let inverse = operation.permutation().inverse()?;
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let cotangent = cotangent.transpose(inverse)?;
                    Ok(vec![MaybeZero::Value(
                        cotangent.unalign_cotangent(&inputs[0].r#type().cotangent())?,
                    )])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            }
        }
    },
}

impl<C: Context<Type = ArrayType>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for TransposeOperation
where
    TransposeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let (lifted_permutation, output_axis) = match inputs[0].batch_axis_position() {
            Some(batch_axis) => {
                let permutation = self.permutation();
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
            None => (self.permutation().to_vec(), None),
        };
        let lifted_operation = TransposeOperation::new(lifted_permutation);
        lifted_operation.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(output_axis)])
    }
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
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::operations::manipulation::Transpose;
/// # use ryft_core::programs::ProgramError;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let output = input.transpose([1, 0])?;
/// assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Transpose: Sized {
    /// Reorders the axes of `self` according to the provided [`Permutation`], validating that the permutation is a
    /// bijection of the input axes.
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError>;

    /// Moves each `source` axis to its corresponding `destination`, shifting the other axes to preserve their relative
    /// order. Scalar axes move one axis, while arrays, vectors, and slices move several axes at once. Negative axes
    /// index from the end. This is the analogue of NumPy's
    /// [`moveaxis`](https://numpy.org/doc/stable/reference/generated/numpy.moveaxis.html).
    /// An out-of-bounds or duplicate axis, or mismatched source and destination lengths, yields a [`TypeError`].
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
                "'{}' move source has length {} but destination has length {}",
                TRANSPOSE_OPERATION_NAME,
                source.len(),
                destination.len(),
            ))
            .into());
        }
        let source = source
            .normalize(rank)
            .map_err(|error| TypeError::invalid(format!("'{TRANSPOSE_OPERATION_NAME}' move source {error}")))?;
        let destination = destination
            .normalize(rank)
            .map_err(|error| TypeError::invalid(format!("'{TRANSPOSE_OPERATION_NAME}' move destination {error}")))?;
        let mut permutation = (0..rank).filter(|axis| !source.contains(axis)).collect::<Vec<_>>();
        let mut moves = destination.into_iter().zip(source).collect::<Vec<_>>();
        moves.sort_by_key(|(destination, _)| *destination);
        for (destination, source) in moves {
            permutation.insert(destination, source);
        }
        self.transpose(permutation)
    }

    /// Swaps axes `i` and `j`, leaving every other axis in place. This is the analogue of NumPy's
    /// [`swapaxes`](https://numpy.org/doc/stable/reference/generated/numpy.swapaxes.html). Returns `self`
    /// unchanged when `i == j`. An out-of-bounds axis yields a [`TypeError`] rather than a panic.
    #[inline]
    fn swap_axes<I: Into<Axis>, J: Into<Axis>>(&self, i: I, j: J) -> Result<Self, ProgramError>
    where
        Self: Typed<Type = ArrayType>,
    {
        let rank = self.r#type().rank();
        let i = i.into();
        let i = i.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("'{TRANSPOSE_OPERATION_NAME}' swap axis {i} is out of bounds for rank {rank}"))
        })?;
        let j = j.into();
        let j = j.normalize(rank).map_err(|_| {
            TypeError::invalid(format!("'{TRANSPOSE_OPERATION_NAME}' swap axis {j} is out of bounds for rank {rank}"))
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
        let permutation = permutation.into();
        permutation.validate(self.rank())?;
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
        let permutation = permutation.into();
        let input = self;
        let rank = input.rank();
        permutation.validate(rank)?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(input.clone());
        }
        let permuted = permutation.iter().map(|axis| input.dimension(*axis)).collect::<Vec<_>>();

        // The output sharding permutes its dimension entries the same way as the array axes: the reduction-state and
        // manual-axis sets are unchanged. This mirrors JAX's `_transpose_sharding_rule` and is correct for every mesh
        // axis type (it is a pure reordering, not explicit-mode reasoning).
        let sharding = input.sharding().map(|sharding| sharding.transpose(permutation)).transpose()?;

        ArrayType::new(input.data_type(), Shape::new(permuted))
            .with_memory(input.memory())
            .with_sharding(sharding)
            .map_err(|error| TypeError::invalid(error.to_string()).into())
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<TransposeOperation>>>>
    Transpose for V
{
    #[inline]
    fn transpose<P: Into<Permutation>>(&self, permutation: P) -> Result<Self, ProgramError> {
        let permutation = permutation.into();
        self.r#type().transpose(permutation.clone())?;
        if permutation.iter().enumerate().all(|(index, axis)| index == *axis) {
            return Ok(self.clone());
        }
        Ok(self
            .dispatch_domain()
            .bind(TransposeOperation::new(permutation), Vec::new(), std::slice::from_ref(self))?
            .remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        DataType, Dimension, DimensionBounds, DimensionVariable, Layout, LogicalMesh, Memory, MeshAxis, MeshAxisType,
        Sharding, ShardingDimension, StridedLayout, f8e8m0fnu,
    };
    use crate::backends::{Array, ArrayOperation};
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{DifferentiableOperation, TransposableOperation};
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{EmptyRegionDriver, ProgramBuilder, ProgramError, Typed};

    use super::*;

    #[test]
    fn test_permutation() {
        // Common owned and borrowed axis collections convert to the canonical permutation representation.
        let axes = vec![2, 0, 1];
        assert_eq!(Permutation::from([2, 0, 1]), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(&[2, 0, 1]), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(axes.as_slice()), Permutation::from(axes.clone()));
        assert_eq!(Permutation::from(&axes), Permutation::from(axes));

        // Empty and identity permutations are their own inverses.
        assert_eq!(Permutation::from(vec![]).inverse(), Ok(Permutation::from(vec![])));
        assert_eq!(Permutation::from(vec![0, 1, 2]).inverse(), Ok(Permutation::from(vec![0, 1, 2])));

        // A swap is its own inverse, while a cycle inverts to the reverse cycle.
        assert_eq!(Permutation::from(vec![1, 0]).inverse(), Ok(Permutation::from(vec![1, 0])));
        assert_eq!(Permutation::from(vec![2, 0, 1]).inverse(), Ok(Permutation::from(vec![1, 2, 0])));

        // Invalid wrappers report the same precise validation errors as the type-level transpose contract.
        assert_eq!(
            Permutation::from(vec![2, 0]).inverse(),
            Err(TypeError::invalid("'transpose' permutation axis 2 is out of bounds".to_string())),
        );
        assert_eq!(
            Permutation::from(vec![0, 0]).inverse(),
            Err(TypeError::invalid("'transpose' permutation contains duplicate axis 0".to_string())),
        );

        // Inverting twice recovers the original permutation, and applying the inverse after the permutation restores
        // the identity ordering.
        let permutation = Permutation::from(vec![3, 0, 2, 1]);
        let inverse = permutation.inverse().unwrap();
        assert_eq!(inverse.inverse(), Ok(permutation.clone()));
        let composed = inverse.iter().map(|axis| permutation[*axis]).collect::<Vec<_>>();
        assert_eq!(composed, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_transpose() {
        let operation = TransposeOperation::new(vec![1, 0]);

        // Operation identity and accessors.
        assert_eq!(operation.name(), TRANSPOSE_OPERATION_NAME);
        assert_eq!(format!("{operation:?}"), "TransposeOperation { permutation: Permutation([1, 0]) }");
        assert_eq!(format!("{operation}"), "transpose [permutation=[1, 0]]");
        assert_eq!(operation.permutation().as_slice(), &[1, 0]);

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
                    error = "'transpose' permutation has length 2 but input has rank 1",
                },
            ],
        );

        // Interpretation reorders the row-major payload.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .clone()
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Invalid permutations and interpreter arity report precise errors.
        assert_eq!(
            TransposeOperation::new(vec![0, 2]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError::invalid("'transpose' permutation axis 2 is out of bounds".to_string())),
        );
        assert_eq!(
            TransposeOperation::new(vec![0, 0]).infer_output_types(std::slice::from_ref(&input_type), &[]),
            Err(TypeError::invalid("'transpose' permutation contains duplicate axis 0".to_string())),
        );
        assert_eq!(
            input.transpose([0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation has length 1 but input has rank 2".to_string(),
            ))),
        );
        assert_eq!(placed_input_type.transpose([0, 1]), Ok(placed_input_type.clone()));
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[],
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured permutation.
        let mut builder = ProgramBuilder::<Array, TransposeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
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

        // Check standard partial evaluation with known and residual operands.
        let input = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = Array::matrix(3, 2, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
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

        // Check that batching lifts the per-item permutation while leaving the mapped axis in place.
        let batched_input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()])),
            (0..24).map(|value| value as f64).collect(),
        );
        let batched_output = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 4.into(), 3.into()])),
            vec![
                0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0, 12.0, 16.0, 20.0, 13.0, 17.0, 21.0, 14.0,
                18.0, 22.0, 15.0, 19.0, 23.0,
            ],
        );
        check_operation_batching!(
            @exact,
            operation = TransposeOperation::new(vec![1, 0]),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), batched_input)],
                outputs = [(@mapped(axis = 0), batched_output)],
            }],
        );

        // The batch axis may occupy any physical position. The lifted permutation leaves it in that position while
        // applying the logical rank-3 cycle around it.
        let middle_axis_input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into(), 4.into()])),
            (0..48).map(f64::from).collect(),
        );
        let middle_axis_output = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 2.into(), 3.into()])),
            vec![
                0.0, 4.0, 8.0, 24.0, 28.0, 32.0, 12.0, 16.0, 20.0, 36.0, 40.0, 44.0, 1.0, 5.0, 9.0, 25.0, 29.0, 33.0,
                13.0, 17.0, 21.0, 37.0, 41.0, 45.0, 2.0, 6.0, 10.0, 26.0, 30.0, 34.0, 14.0, 18.0, 22.0, 38.0, 42.0,
                46.0, 3.0, 7.0, 11.0, 27.0, 31.0, 35.0, 15.0, 19.0, 23.0, 39.0, 43.0, 47.0,
            ],
        );
        let trailing_axis_input = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into(), 2.into()])),
            (0..48).map(f64::from).collect(),
        );
        let trailing_axis_output = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 3.into(), 2.into()])),
            vec![
                0.0, 1.0, 8.0, 9.0, 16.0, 17.0, 24.0, 25.0, 32.0, 33.0, 40.0, 41.0, 2.0, 3.0, 10.0, 11.0, 18.0, 19.0,
                26.0, 27.0, 34.0, 35.0, 42.0, 43.0, 4.0, 5.0, 12.0, 13.0, 20.0, 21.0, 28.0, 29.0, 36.0, 37.0, 44.0,
                45.0, 6.0, 7.0, 14.0, 15.0, 22.0, 23.0, 30.0, 31.0, 38.0, 39.0, 46.0, 47.0,
            ],
        );
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

        // Transpose only reorders axes, so its batching rule also works when the mapped dimension is symbolic. It
        // stages the physical permutation [3, 1, 0, 2] without demanding a concrete batch size from the input type.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let symbolic_input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![2.into(), Dimension::Dynamic(batch.clone()), 3.into(), 4.into()]),
        );
        let symbolic_input = context.input(symbolic_input_type.clone());
        let symbolic_input = ArrayBatch::new(symbolic_input_type, symbolic_input, BatchAxis::new(1)).unwrap();
        let symbolic_output = TransposeOperation::new([2, 0, 1])
            .batch(&BatchingContext::new(context.clone(), 2), &EmptyRegionDriver, &[symbolic_input])
            .unwrap()
            .remove(0);
        assert_eq!(symbolic_output.batch_axis(), BatchAxis::new(1));
        assert_eq!(
            symbolic_output.r#type().as_ref(),
            &ArrayType::new(DataType::F64, Shape::new(vec![4.into(), Dimension::Dynamic(batch), 2.into(), 3.into()]),),
        );
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        assert_eq!(
            format!("{}", context.builder().borrow().instructions()[0].operation()),
            "transpose [permutation=[3, 1, 0, 2]]"
        );

        // Transpose is structural-linear: its JVP applies the same permutation and its pullback applies the inverse.
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                primals = [Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])],
                tangents = [Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0])],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 3.0, 2.0, 4.0])],
                tangent_outputs = [Array::matrix(2, 2, vec![5.0, 7.0, 6.0, 8.0])],
            }],
        );

        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))))],
                output_cotangents = [Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::matrix(2, 3, vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0])],
            }],
        );

        // A non-self-inverse cycle exercises the distinction between the forward rule and inverse pullback rule.
        let cycle_input_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 4.into()]));
        let cycle_output_type = ArrayType::new(DataType::F64, Shape::new(vec![4.into(), 2.into(), 3.into()]));
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = TransposeOperation::new(vec![2, 0, 1]),
            cases = [{
                primals = [Array::from_f64s(cycle_input_type.clone(), (0..24).map(f64::from).collect())],
                tangents = [Array::from_f64s(cycle_input_type.clone(), (24..48).map(f64::from).collect())],
                primal_outputs = [Array::from_f64s(
                    cycle_output_type.clone(),
                    vec![
                        0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0,
                        14.0, 18.0, 22.0, 3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
                    ],
                )],
                tangent_outputs = [Array::from_f64s(
                    cycle_output_type.clone(),
                    vec![
                        24.0, 28.0, 32.0, 36.0, 40.0, 44.0, 25.0, 29.0, 33.0, 37.0, 41.0, 45.0, 26.0, 30.0,
                        34.0, 38.0, 42.0, 46.0, 27.0, 31.0, 35.0, 39.0, 43.0, 47.0,
                    ],
                )],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![2, 0, 1]),
            cases = [{
                inputs = [(@linear(type = cycle_input_type.clone()))],
                output_cotangents = [Array::from_f64s(
                    cycle_output_type.clone(),
                    (0..24).map(f64::from).collect(),
                )],
                input_cotangents = [Array::from_f64s(
                    cycle_input_type.clone(),
                    vec![
                        0.0, 6.0, 12.0, 18.0, 1.0, 7.0, 13.0, 19.0, 2.0, 8.0, 14.0, 20.0, 3.0, 9.0, 15.0,
                        21.0, 4.0, 10.0, 16.0, 22.0, 5.0, 11.0, 17.0, 23.0,
                    ],
                )],
            }],
        );

        // Structural zeros remain symbolic in both differentiation directions and do not stage needless tangent or
        // cotangent transposes.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = context.input(cycle_input_type.clone());
        let duals = TransposeOperation::new([2, 0, 1])
            .jvp(&context, &EmptyRegionDriver, &[DifferentiationDual::new_with_zero_tangent(primal)])
            .unwrap();
        assert!(duals[0].tangent().is_zero());
        assert_eq!(duals[0].tangent().r#type().as_ref(), &cycle_output_type.tangent());
        assert_eq!(context.builder().borrow().instructions().len(), 1);

        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let contributions = TransposeOperation::new([2, 0, 1])
            .transpose(
                &mut context,
                &EmptyRegionDriver,
                &[PartialValue::Unknown(cycle_input_type.clone())],
                &[MaybeZero::Zero(cycle_output_type.cotangent())],
            )
            .unwrap();
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &cycle_input_type.cotangent());
        assert!(context.builder().borrow().instructions().is_empty());

        // Identity transpose preserves the exact tracer and placement metadata without staging an instruction.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(placed_input_type.clone());
        let output = input.transpose([0, 1]).unwrap();
        assert_eq!(output.atom_id(), input.atom_id());
        assert_eq!(output.r#type(), input.r#type());
        assert!(context.builder().borrow().instructions().is_empty());

        // The inverse permutation restores the complete input cotangent type, including placement metadata that the
        // forward transpose intentionally clears because it cannot infer a new physical layout.
        check_operation_transposition!(
            @exact,
            operation = TransposeOperation::new(vec![1, 0]),
            cases = [{
                inputs = [(@linear(type = placed_input_type.clone()))],
                output_cotangents = [Array::from_f64s(
                    placed_output_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )],
                input_cotangents = [Array::from_f64s(
                    placed_input_type,
                    vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0],
                )],
            }],
        );
    }

    #[test]
    fn test_array_type_transpose() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("u", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("v", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        // Input dimensions are sharded over `x` and `y`; reduction and manual-axis state rides along untouched.
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
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&input_type), &[]), Ok(vec![expected]));

        // Direct sharding transposition validates the complete permutation even when duplicate axes refer to
        // replicated or unconstrained dimensions.
        let sharding = input_type.sharding().unwrap();
        assert_eq!(
            sharding.transpose([1, 1, 0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation contains duplicate axis 1".to_string(),
            ))),
        );
        assert_eq!(
            sharding.transpose([0, 1]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation has length 2 but input has rank 3".to_string(),
            ))),
        );
        assert_eq!(
            sharding.transpose([0, 1, 3]),
            Err(ProgramError::Type(TypeError::invalid("'transpose' permutation axis 3 is out of bounds".to_string()))),
        );

        // An input without a sharding yields an output without one.
        let unsharded = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        assert_eq!(operation.infer_output_types(std::slice::from_ref(&unsharded), &[]).unwrap()[0].sharding(), None);

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
        let output = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).transpose(vec![1, 0]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))
        );
        assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // The eager kernel reorders exact typed payloads without changing their element representation.
        let input = Array::matrix(2, 3, vec![false, true, true, false, false, true]);
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, vec![false, false, true, false, true, true])));
        let input = Array::matrix(2, 3, (0..6).collect::<Vec<i32>>());
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, vec![0, 3, 1, 4, 2, 5])));
        let input = Array::matrix(2, 3, (1..=6).map(f8e8m0fnu::from_bits).collect());
        let expected = [1, 4, 2, 5, 3, 6].map(f8e8m0fnu::from_bits).to_vec();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, expected)));
        let input =
            Array::matrix(2, 3, (0..6).map(|value| ComplexNumber::new(f64::from(value), -f64::from(value))).collect());
        let expected = [0, 3, 1, 4, 2, 5].map(|value| ComplexNumber::new(f64::from(value), -f64::from(value))).to_vec();
        assert_eq!(input.transpose([1, 0]), Ok(Array::matrix(3, 2, expected)));

        // Rank-3 permutation moving the last axis to the front.
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_f64s(input_type, values).transpose(vec![2, 0, 1]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(4), Dimension::Static(2), Dimension::Static(3)])
            ),
        );
        assert_eq!(
            output.to_f64s(),
            vec![
                0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 1.0, 5.0, 9.0, 13.0, 17.0, 21.0, 2.0, 6.0, 10.0, 14.0, 18.0, 22.0,
                3.0, 7.0, 11.0, 15.0, 19.0, 23.0,
            ],
        );

        // Rank-0 and empty payloads pass through unchanged.
        let output = Array::scalar(42.0).transpose(vec![]).unwrap();
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(output.to_f64s(), vec![42.0]);
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(0), Dimension::Static(2)]));
        let output = Array::from_f64s(input_type, Vec::new()).transpose(vec![1, 0]).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(0)]))
        );
        assert_eq!(output.to_f64s(), Vec::<f64>::new());

        // An invalid permutation is a clean error rather than an out-of-bounds panic, since the value-level transpose
        // validates the permutation through the type-level rule before indexing.
        let matrix = || Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(
            matrix().transpose(vec![1]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation has length 1 but input has rank 2".to_string(),
            ))),
        );
        assert_eq!(
            matrix().transpose(vec![0, 2]),
            Err(ProgramError::Type(TypeError::invalid("'transpose' permutation axis 2 is out of bounds".to_string()))),
        );
        assert_eq!(
            matrix().transpose(vec![0, 0]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' permutation contains duplicate axis 0".to_string(),
            ))),
        );

        // `move_axis` shifts intervening dimensions while preserving their relative order.
        // On a matrix, moving axis 0 to position 1 is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = matrix.move_axis(0, 1).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))
        );
        assert_eq!(output.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // On a rank-3 array, moving axis 0 to the last position shifts the other axes left to preserve their relative
        // order, so [2, 3, 4] becomes [3, 4, 2] (the permutation [1, 2, 0]).
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_f64s(input_type, values).move_axis(0, 2).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(3), Dimension::Static(4), Dimension::Static(2)])
            ),
        );

        // `from == to` leaves the array unchanged.
        assert_eq!(matrix.move_axis(1, 1).unwrap(), matrix);

        // An out-of-bounds source axis is a clean error rather than an out-of-bounds panic, since the built
        // permutation is validated by the type-level transpose.
        assert_eq!(
            matrix.move_axis(2, 0),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move source axis 2 is out of bounds for rank 2".to_string(),
            ))),
        );
        assert_eq!(
            matrix.move_axis(0, 2),
            Err(TypeError::invalid("'transpose' move destination axis 2 is out of bounds for rank 2".to_string())
                .into()),
        );

        // Negative and multiple axes use NumPy/JAX normalization and paired sorted-insertion semantics.
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
        assert_eq!(rank_four_type.move_axis(Axes::default(), Axes::default()), Ok(rank_four_type.clone()),);
        assert_eq!(
            rank_four_type.move_axis([0, 1], [2]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move source has length 2 but destination has length 1".to_string(),
            ))),
        );
        assert_eq!(
            rank_four_type.move_axis([0, -4], [1, 2]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move source axes contain duplicate axis 0".to_string(),
            ))),
        );
        assert_eq!(
            rank_four_type.move_axis([0, 1], [0, -4]),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move destination axes contain duplicate axis 0".to_string(),
            ))),
        );
        assert_eq!(
            rank_four_type.move_axis(-5, 0),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move source axis -5 is out of bounds for rank 4".to_string(),
            ))),
        );
        assert_eq!(
            ArrayType::scalar(DataType::F64).move_axis(0, 0),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' move source axis 0 is out of bounds for rank 0".to_string(),
            ))),
        );

        // `swap_axes` exchanges exactly two dimensions and validates both indices.
        // Swapping axes 0 and 1 of a matrix is a plain transpose: the [2, 3] payload becomes [3, 2].
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let swapped = matrix.swap_axes(0, 1).unwrap();
        assert_eq!(
            swapped.r#type().into_owned(),
            ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3), Dimension::Static(2)]))
        );
        assert_eq!(swapped.to_f64s(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

        // Swapping is symmetric in its axis arguments.
        assert_eq!(matrix.swap_axes(1, 0).unwrap(), swapped);

        // Swapping the outer two axes of a rank-3 array leaves the untouched trailing axis in place, so [2, 3, 4]
        // becomes [3, 2, 4] (the permutation [1, 0, 2]).
        let input_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(3), Dimension::Static(4)]),
        );
        let values = (0..24).map(|value| value as f64).collect::<Vec<_>>();
        let output = Array::from_f64s(input_type, values).swap_axes(0, 1).unwrap();
        assert_eq!(
            output.r#type().into_owned(),
            ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(3), Dimension::Static(2), Dimension::Static(4)])
            ),
        );

        // `i == j` leaves the array unchanged.
        assert_eq!(matrix.swap_axes(1, 1).unwrap(), matrix);

        assert_eq!(matrix.swap_axes(-1, 1), Ok(matrix.clone()));
        assert_eq!(matrix.swap_axes(-1, 0), Ok(swapped));

        // An out-of-bounds axis is a clean error rather than an out-of-bounds panic.
        assert_eq!(
            matrix.swap_axes(2, 0),
            Err(ProgramError::Type(TypeError::invalid(
                "'transpose' swap axis 2 is out of bounds for rank 2".to_string(),
            ))),
        );
    }
}
