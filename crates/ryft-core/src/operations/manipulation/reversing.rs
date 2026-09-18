use std::fmt::{Debug, Display};
use std::sync::Arc;

use crate::arrays::{Array, ArrayAddressing, ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType};
use crate::axes::Axes;
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation, impl_reference_dischargeable_operation};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

/// Canonical operation name for [`ReverseOperation`].
pub const REVERSE_OPERATION_NAME: &str = "reverse";

/// [`Operation`] that reverses element order along the selected axes of its input array.
/// Refer to the documentation of [`Reverse`] for more information.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ReverseOperation {
    /// Refer to the documentation of [`axes`](Self::axes) for more information.
    axes: Axes,
}

impl ReverseOperation {
    /// Creates a new [`ReverseOperation`] with the provided axes.
    #[inline]
    pub fn new<A: Into<Axes>>(axes: A) -> Self {
        Self { axes: axes.into() }
    }

    /// Returns the axes along which this operation reverses element order.
    #[inline]
    pub fn axes(&self) -> &Axes {
        &self.axes
    }
}

impl Display for ReverseOperation {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation for ReverseOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        REVERSE_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].reverse(&self.axes) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError::invalid(error.to_string())),
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("axes", format_args!("{:?}", self.axes.iter().map(|axis| axis.value()).collect::<Vec<_>>()))
        })
    }
}

impl_reference_dischargeable_operation!(@reference_free ReverseOperation);

impl<C: Domain<Type = ArrayType, Value: Reverse>> InterpretableOperation<C> for ReverseOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reverse(&self.axes)?])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<ReverseOperation>>> PartiallyEvaluatableOperation<C>
    for ReverseOperation
{
}

impl<C: Context<Type = ArrayType, Value: Reverse>, P: ArrayExtentBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatchingPolicy<P>> for ReverseOperation
{
    fn batch<D: BatchingDriver<C, ArrayBatchingPolicy<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatchingPolicy<P>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let axes = self
            .axes
            .normalize(inputs[0].unbatched_type().rank())
            .map_err(|error| TypeError::invalid(error.to_string()))?;

        // Identity reversal preserves the packed value and all ragged geometry without reindexing storage.
        if axes.is_empty() {
            return Ok(vec![inputs[0].clone()].into());
        }

        if !inputs[0].ragged_axes().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{REVERSE_OPERATION_NAME}` does not support bounded ragged array inputs"),
            });
        }

        // The mapped axis indexes independent arrays and must never be reversed with their logical axes.
        let batch_axis = inputs[0].batch_axis_position();
        let axes = axes
            .into_iter()
            .map(|axis| if batch_axis.is_some_and(|batch_axis| axis >= batch_axis) { axis + 1 } else { axis })
            .collect::<Vec<_>>();
        Ok(ReverseOperation::new(axes)
            .interpret_with_batch_axes(context, inputs, &[BatchAxis::from_optional_position(batch_axis)])?
            .into())
    }
}

impl_differentiable_operation! {
    ReverseOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Value: Reverse,
    {
        |operation, _context, _driver, inputs| {
            // A live tangent follows the primal reversal. Structural zero tangents remain symbolic, including
            // when this rule is invoked directly instead of through a driver's zero-tangent fast path.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reverse(operation.axes())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()?),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reverse(operation.axes())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<Type = ArrayType> + From<ReverseOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Reverse,
    {
        |operation, context, _driver, inputs, outputs, accumulators| {
            // Reversal is its own inverse and no primal values or runtime extents need to be retained.
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            check_count!("accumulator", accumulators, 1, DifferentiationError);
            match &outputs[0] {
                MaybeZero::Value(_) if !accumulators[0].is_needed() => Ok(()),
                MaybeZero::Value(cotangent) => {
                    let contribution = MaybeZero::Value(
                        cotangent.reverse(operation.axes())?.unalign_cotangent(&inputs[0].r#type().cotangent()?)?,
                    );
                    accumulators[0].accumulate(context, contribution)?;
                    Ok(())
                }
                MaybeZero::Zero(_) => Ok(()),
            }
        }
    },
}

/// Reverses element order along selected axes, retaining shape, element type, sharding, and memory placement.
/// Negative axes count from the end, and duplicate or out-of-range axes are errors. Reversal preserves element bytes
/// exactly, including NaN payloads. An empty axis list returns the input unchanged. A non-empty reversal clears the
/// physical layout, since its result may use different storage. This operation supports symbolic extents and is
/// linear and self-adjoint. Batching preserves the mapped axis and reverses only the selected per-item axes.
/// Non-empty reversals of bounded ragged batches are rejected because reversing their padded storage would move
/// padding into valid data. Empty axis lists preserve the ragged batch unchanged.
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, ProgramError, Reverse};
/// let input = Array::vector(vec![1.0, 2.0, 3.0])?;
/// assert_eq!(input.reverse([0])?.to_f64s(), vec![3.0, 2.0, 1.0]);
/// # Ok::<(), ProgramError>(())
/// ```
pub trait Reverse: Sized {
    /// Reverses the elements of `self` along `axes`, resolving signed axes against the input rank.
    fn reverse<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError>;
}

impl Reverse for ArrayType {
    fn reverse<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError> {
        let axes = axes.into().normalize(self.rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        if axes.is_empty() {
            return Ok(self.clone());
        }
        Ok(self.clone().with_layout(None))
    }
}

impl Reverse for Array {
    fn reverse<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError> {
        let axes =
            axes.into().normalize(self.r#type().rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        let output_type = self.r#type().reverse(axes.clone())?;
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let input_addressing = ArrayAddressing::new(self.r#type().into_owned())?;
        let output_addressing = ArrayAddressing::new(output_type.clone())?;
        let mut bytes = vec![0; output_addressing.storage_byte_len()];

        // Empty arrays and structural-zero arrays have no bytes to copy, even for enormous logical shapes.
        if bytes.is_empty() {
            return Ok(Self::new_unchecked(output_type, Arc::new(bytes)));
        }

        let mut output_index = vec![0; output_type.rank()];
        let mut input_index = output_index.clone();
        for output_flat in 0..output_addressing.element_count() {
            input_index.copy_from_slice(&output_index);
            for &axis in &axes {
                // Non-empty storage guarantees a positive extent on each axis. Addressing maps logical coordinates
                // to physical bytes, so reversal also works for noncontiguous input layouts.
                input_index[axis] = output_type.dimension(axis).value().unwrap() - 1 - output_index[axis];
            }
            bytes[output_addressing.byte_range_for_flat_index(output_flat)]
                .copy_from_slice(&self.storage_bytes()[input_addressing.byte_range_unchecked(&input_index)]);
            output_addressing.advance_index(&mut output_index);
        }

        Ok(Self::new_unchecked(output_type, Arc::new(bytes)))
    }
}

impl<V: Value<Type = ArrayType, DispatchDomain: Context<Type = ArrayType, Operation: From<ReverseOperation>>>> Reverse
    for V
{
    fn reverse<A: Into<Axes>>(&self, axes: A) -> Result<Self, ProgramError> {
        let axes =
            axes.into().normalize(self.r#type().rank()).map_err(|error| TypeError::invalid(error.to_string()))?;
        self.r#type().reverse(axes.clone())?;
        if axes.is_empty() {
            return Ok(self.clone());
        }
        let mut outputs =
            self.dispatch_domain().bind(ReverseOperation::new(axes), Vec::new(), std::slice::from_ref(self))?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayOperation, DataType, Dimension, DimensionBounds, DimensionVariable, Layout, RaggedAxis, Shape,
        StridedLayout,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::{
        DifferentiableOperation, DifferentiationContext, TransposableOperation, TranspositionContext,
    };
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::partial::PartialValue;
    use crate::programs::EmptyRegionDriver;

    use super::*;

    #[test]
    fn test_reverse() {
        let operation = ReverseOperation::new([1]);
        assert_eq!(operation.name(), REVERSE_OPERATION_NAME);
        assert_eq!(operation.axes().normalize(2), Ok(vec![1]));
        assert_eq!(operation.to_string(), "reverse [axes=[1]]");
    }

    #[test]
    fn test_reverse_type_inference() {
        let dimension = Dimension::Dynamic(DimensionVariable::new("length", DimensionBounds::unbounded()));
        let input = ArrayType::new(DataType::F32, Shape::new(vec![dimension]));
        check_operation_type_inference!(
            operation = ReverseOperation::new([-1]),
            cases = [{ input_types = [input.clone()], output_types = [input] }],
        );
        check_operation_type_inference!(
            operation = ReverseOperation::new([0, 0]),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                error = "axes contain duplicate axis 0",
            }],
        );
        check_operation_type_inference!(
            operation = ReverseOperation::new([1]),
            cases = [{
                input_types = [ArrayType::new_static(DataType::F32, [2])],
                error = "axis 1 is out of bounds for rank 1",
            }],
        );
    }

    #[test]
    fn test_reverse_interpretation() {
        let input = Array::matrix(2, 3, vec![1i32, 2, 3, 4, 5, 6]).unwrap();
        assert_eq!(input.reverse([-1]).unwrap(), Array::matrix(2, 3, vec![3i32, 2, 1, 6, 5, 4]).unwrap());
        assert_eq!(input.reverse([0, 1]).unwrap(), Array::matrix(2, 3, vec![6i32, 5, 4, 3, 2, 1]).unwrap());
        assert_eq!(input.reverse(Vec::<usize>::new()).unwrap(), input);
        assert_eq!(
            Array::vector(Vec::<i32>::new()).unwrap().reverse([0]).unwrap(),
            Array::vector(Vec::<i32>::new()).unwrap(),
        );

        // Logical addressing preserves noncontiguous storage and exact floating-point encodings.
        let values = [f32::from_bits(0x7fc00123), -0.0, 1.0, 2.0];
        let input = Array::from_elements(
            ArrayType::new_static(DataType::F32, [2, 2]).with_layout(Layout::Strided(StridedLayout::new(vec![4, 8]))),
            &values,
        )
        .unwrap();
        assert_eq!(
            input.reverse([1]).unwrap().storage_bytes(),
            Array::matrix(2, 2, vec![values[1], values[0], values[3], values[2]]).unwrap().storage_bytes(),
        );
    }

    #[test]
    fn test_reverse_partial_evaluation() {
        let input = Array::vector(vec![1.0, 2.0, 3.0]).unwrap();
        let expected = Array::vector(vec![3.0, 2.0, 1.0]).unwrap();
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = ReverseOperation::new([0]),
            cases = [
                {
                    inputs = [(@known, input.clone())],
                    outputs = [(@known, expected.clone())],
                    residual_instructions = 0,
                },
                {
                    inputs = [(@unknown(type = input.r#type().into_owned(), replay = input.clone()))],
                    outputs = [(@residual, expected)],
                    residual_instructions = 1,
                },
            ],
        );
    }

    #[test]
    fn test_reverse_batching() {
        check_operation_batching!(
            @exact,
            operation = ReverseOperation::new([-1]),
            axis_size = 2,
            cases = [
                {
                    inputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                    outputs = [(@mapped(axis = 0), Array::matrix(2, 3, vec![3.0, 2.0, 1.0, 6.0, 5.0, 4.0]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap())],
                    outputs = [(@mapped(axis = 1), Array::matrix(3, 2, vec![5.0, 6.0, 3.0, 4.0, 1.0, 2.0]).unwrap())],
                },
                {
                    inputs = [(@mapped(axis = 1), Array::from_elements(
                        ArrayType::new_static(DataType::F64, [1, 2, 3]),
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    ).unwrap())],
                    outputs = [(@mapped(axis = 1), Array::from_elements(
                        ArrayType::new_static(DataType::F64, [1, 2, 3]),
                        &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0],
                    ).unwrap())],
                },
                {
                    inputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]).unwrap())],
                    outputs = [(@replicated, Array::vector(vec![3.0, 2.0, 1.0]).unwrap())],
                },
            ],
        );

        // Only identity reversal can preserve ragged geometry without moving padded elements into valid data.
        let length = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let input =
            ArrayBatch::new(Array::matrix(2, 3, vec![1.0, 2.0, 0.0, 3.0, 0.0, 0.0]).unwrap(), BatchAxis::new(0))
                .unwrap()
                .with_ragged_axes(vec![RaggedAxis::new(1, Array::vector(vec![2_i32, 1]).unwrap(), length, vec![0])])
                .unwrap();
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);
        assert!(matches!(
            ReverseOperation::new([0]).batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input)),
            Err(BatchingError::UnsupportedOperation { message })
                if message == format!("`{REVERSE_OPERATION_NAME}` does not support bounded ragged array inputs"),
        ));
        let outputs = ReverseOperation::new(Vec::<usize>::new())
            .batch(&context, &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs, vec![input]);
    }

    #[test]
    fn test_reverse_differentiation() {
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = ReverseOperation::new([0]),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                tangents = [Array::vector(vec![4.0, 5.0, 6.0]).unwrap()],
                primal_outputs = [Array::vector(vec![3.0, 2.0, 1.0]).unwrap()],
                tangent_outputs = [Array::vector(vec![6.0, 5.0, 4.0]).unwrap()],
            }],
        );

        // Invoke the rule directly so the driver's zero-tangent shortcut cannot hide a materialized zero.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let dimension = DimensionVariable::new("length", DimensionBounds::unbounded());
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![dimension.into()]));
        let input = context.input(input_type.clone());
        let outputs = ReverseOperation::new([0])
            .jvp(
                &DifferentiationContext::fused(context.clone()),
                &EmptyRegionDriver,
                &[DifferentiationDual::new_with_zero_tangent(input).unwrap()],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].tangent().is_zero());
        assert_eq!(outputs[0].tangent().r#type().as_ref(), &input_type.tangent().unwrap());
        assert_eq!(context.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_reverse_transposition() {
        check_operation_transposition!(
            @exact,
            operation = ReverseOperation::new([0]),
            cases = [{
                inputs = [(@linear(type = ArrayType::new_static(DataType::F64, [3])))],
                output_cotangents = [Array::vector(vec![1.0, 2.0, 3.0]).unwrap()],
                input_cotangents = [Array::vector(vec![3.0, 2.0, 1.0]).unwrap()],
            }],
        );

        // Structural zeros and unrequested cotangents must not stage reversal or alignment instructions.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input_type = ArrayType::new_static(DataType::F64, [3]);
        let inputs = [PartialValue::Unknown(input_type.clone())];
        let mut rule_context = TranspositionContext::new(context.clone());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[]).unwrap();
        ReverseOperation::new([0])
            .transpose(
                &mut rule_context,
                &EmptyRegionDriver,
                &inputs,
                &[MaybeZero::Zero(input_type.cotangent().unwrap())],
                &accumulators,
            )
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type.cotangent().unwrap());
        let accumulators = rule_context.cotangent_accumulators(&inputs, &[false]).unwrap();
        let cotangent = context.input(input_type.cotangent().unwrap());
        ReverseOperation::new([0])
            .transpose(&mut rule_context, &EmptyRegionDriver, &inputs, &[MaybeZero::Value(cotangent)], &accumulators)
            .unwrap();
        let contributions = rule_context.take_cotangents(&accumulators).unwrap();
        assert_eq!(contributions.len(), 1);
        assert!(contributions[0].is_zero());
        assert_eq!(contributions[0].r#type().as_ref(), &input_type.cotangent().unwrap());
        assert!(context.builder().borrow().instructions().is_empty());
    }
}
