use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, DifferentiationDual, ElementwiseDerivativeAlignment};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, impl_differentiable_operation};
use crate::operations::manipulation::{Broadcast, Transpose};
use crate::partial::PartiallyEvaluatableOperation;
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::sharding::{Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Shape, Size};

// TODO(eaplatanios): Review this.

/// Canonical operation name for [`ReshapeOperation`].
pub const RESHAPE_OPERATION_NAME: &str = "reshape";

/// [`Operation`] that reshapes its input array to a target [`Shape`]. The input shape is not part of the operation
/// payload; it is recoverable from the staged input types wherever a rule needs it. Refer to the documentation of
/// [`Reshape`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReshapeOperation {
    /// Output [`Shape`] of this [`ReshapeOperation`].
    shape: Shape,
}

impl ReshapeOperation {
    /// Creates a new [`ReshapeOperation`] with the provided output [`Shape`].
    #[inline]
    pub fn new(shape: Shape) -> Self {
        Self { shape }
    }

    /// Returns the output shape of this [`ReshapeOperation`].
    #[inline]
    pub fn output_shape(&self) -> &Shape {
        &self.shape
    }
}

impl Display for ReshapeOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for ReshapeOperation {
    #[inline]
    fn name(&self) -> &'static str {
        RESHAPE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        match input_types[0].reshape(self.shape.clone()) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("shape", &self.shape))
    }
}

impl<C: Domain<Type = ArrayType, Value: Reshape>> InterpretableOperation<C> for ReshapeOperation {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].reshape(self.shape.clone())?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ReshapeOperation where
    C::Operation: From<ReshapeOperation>
{
}

impl_differentiable_operation! {
    ReshapeOperation,
    jvp<C>
    where
        C: Context<Type = ArrayType>,
        C::Operation: From<ReshapeOperation>,
        C::Value: Reshape,
    {
        |operation, _context, _driver, inputs| {
            // Forward-mode differentiation rule for `ReshapeOperation`. `reshape` is structural-linear, and so the
            // tangent is the same reshape applied to the operand tangent. The shared all-zero fast path handles a zero
            // operand tangent before this rule is consulted, so the operand tangent reaching here is always live.
            check_count!("input", inputs, 1, ProgramError);
            let primal = inputs[0].primal().reshape(operation.output_shape().clone())?;
            let tangent = match inputs[0].tangent() {
                MaybeZero::Zero(_) => MaybeZero::Zero(primal.r#type().tangent()),
                MaybeZero::Value(tangent) => MaybeZero::Value(tangent.reshape(operation.output_shape().clone())?),
            };
            Ok(vec![DifferentiationDual::new(primal, tangent)?])
        }
    },
    transpose<V, O>
    where
        V: Value<Type = ArrayType>,
        O: Operation<ArrayType> + From<ReshapeOperation>,
        Tracer<TracingContext<V, O>>: ElementwiseDerivativeAlignment<ArrayType> + Reshape,
    {
        |_operation, _context, _driver, inputs, outputs| {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            match &outputs[0] {
                MaybeZero::Value(cotangent) => {
                    let cotangent = cotangent.reshape(inputs[0].r#type().shape().clone())?;
                    Ok(vec![MaybeZero::Value(
                        cotangent.unalign_cotangent(&inputs[0].r#type().cotangent())?,
                    )])
                }
                MaybeZero::Zero(_) => Ok(vec![MaybeZero::Zero(inputs[0].r#type().cotangent())]),
            }
        }
    },
}

/// Lifts a reshape's per-item `input_shape` / `output_shape` pair through one batching level by
/// inserting a new dimension of size `axis_size` at the supplied input position and finding the
/// matching output position.
///
/// The lifted reshape preserves per-item semantics in row-major order, which requires that the
/// element count to the left of the batch dimension is the same on both sides:
/// `product(input_shape[..k_in]) == product(output_shape[..k_out])`. When such a `k_out` exists,
/// the helper inserts `axis_size` at position `k_in` in the input shape and at position `k_out`
/// in the output shape, and returns `Some((lifted_input_shape, lifted_output_shape, k_out))`. If
/// no matching position can be found (for example, the batch axis falls in the middle of a
/// reshape that mixes dimensions on both sides), the helper returns `None` and the caller should
/// surface a [`BatchingError::UnsupportedOperation`]
/// pointing at a future fix that emits an explicit transpose before the reshape.
///
/// Dynamic dimensions in `input_shape[..k_in]` or in any candidate `output_shape[..k_out]` are
/// rejected (they make the prefix product undefined).
///
/// # Parameters
///
///   - `input_shape`: Per-item shape of the reshape's input.
///   - `output_shape`: Per-item shape produced by [`ReshapeOperation::output_shape`].
///   - `k_in`: Position of the batched axis in the parent-physical input.
///   - `axis_size`: Size of the batched item this level introduces.
pub fn lift_reshape_shapes(
    input_shape: &Shape,
    output_shape: &Shape,
    k_in: usize,
    axis_size: usize,
) -> Option<(Shape, Shape, usize)> {
    if k_in > input_shape.rank() {
        return None;
    }
    let mut prefix_product = 1usize;
    for dim in &input_shape.dimensions()[..k_in] {
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        prefix_product = prefix_product.checked_mul(value)?;
    }

    let target_prefix_product = prefix_product;
    let mut output_prefix_product = 1usize;
    let mut k_out = None;
    for (index, dim) in output_shape.dimensions().iter().enumerate() {
        if output_prefix_product == target_prefix_product {
            k_out = Some(index);
            break;
        }
        let value = match dim {
            Size::Static(value) => *value,
            Size::Dynamic(_) => return None,
        };
        output_prefix_product = output_prefix_product.checked_mul(value)?;
    }
    if k_out.is_none() && output_prefix_product == target_prefix_product {
        k_out = Some(output_shape.rank());
    }
    let k_out = k_out?;

    let mut lifted_input_dimensions = input_shape.dimensions().to_vec();
    lifted_input_dimensions.insert(k_in, Size::Static(axis_size));
    let mut lifted_output_dimensions = output_shape.dimensions().to_vec();
    lifted_output_dimensions.insert(k_out, Size::Static(axis_size));

    Some((Shape::new(lifted_input_dimensions), Shape::new(lifted_output_dimensions), k_out))
}

impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ReshapeOperation
where
    C::Value: Broadcast + Transpose,
    ReshapeOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 1, ProgramError);
        let Some(k_in) = inputs[0].batch_axis_position() else {
            // Replicated input: there is no batch axis to thread through the reshape, so interpret it as given and
            // report the output replicated.
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let input_shape = inputs[0].unbatched_type().shape().clone();
        let Some((_, lifted_output_shape, k_out)) =
            lift_reshape_shapes(&input_shape, self.output_shape(), k_in, axis_size)
        else {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "missing batching rule for ReshapeOperation with batch axis {k_in} crossing reshape group \
                    boundaries in {input_shape} -> {}",
                    self.output_shape(),
                ),
            });
        };
        let lifted_op = ReshapeOperation::new(lifted_output_shape);
        lifted_op.interpret_with_batch_axes(context, inputs, &[BatchAxis::from_position(k_out)])
    }
}

/// Represents the ability to reshape an array to a target [`Shape`] without changing its element count or row-major
/// element order.
///
/// `t.reshape(target_shape)` reinterprets `t`'s payload under the specified target [`Shape`]. The input and target
/// shapes must have equal element counts. When the input carries a [`Sharding`], it is propagated using singleton
/// stripping and contiguous split/merge grouping: dimensions that map one-to-one keep their sharding, while dimensions
/// that split or merge must be replicated. A non-identity reshape preserves the input memory space and clears explicit
/// physical layout metadata because the logical shape change does not determine a unique output storage layout.
///
/// # Examples
///
/// The following example shows how to use [`Reshape`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Reshape;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// # use ryft_core::types::{Shape, Size};
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Reshape a length-6 vector to a `[2, 3]` matrix while keeping the row-major payload unchanged.
/// let x = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let y = x.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)]))?;
/// assert_eq!(y.to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// # Ok(())
/// # }
/// ```
pub trait Reshape: Sized {
    /// Reshapes `self` to `shape`. Refer to the documentation of this trait for more information on what this
    /// operation does.
    fn reshape(&self, shape: Shape) -> Result<Self, ProgramError>;
}

impl Reshape for ArrayType {
    fn reshape(&self, shape: Shape) -> Result<ArrayType, ProgramError> {
        if *self.shape() == shape {
            return Ok(self.clone());
        }

        let Some(input_elements) = self.element_count().map_err(|error| TypeError { message: error.to_string() })?
        else {
            return Err(
                TypeError { message: "'reshape' requires statically known input element counts".to_string() }.into()
            );
        };
        let Some(output_elements) = shape.element_count().map_err(|error| TypeError { message: error.to_string() })?
        else {
            return Err(
                TypeError { message: "'reshape' requires statically known output element counts".to_string() }.into()
            );
        };
        if input_elements != output_elements {
            return Err(TypeError { message: "'reshape' changes the number of elements".to_string() }.into());
        }

        // Propagate the input sharding (when present) to the target shape using JAX-style singleton stripping and
        // contiguous split/merge grouping.
        let sharding = if let Some(sharding) = self.sharding() {
            let alignment_error =
                || TypeError { message: "'reshape' could not align static reshape dimension groups".to_string() };

            // Strip singleton and dynamic dimensions on both sides. Only the remaining static dimensions take part
            // in the split/merge analysis, so shardings move freely across inserted or removed size-1 axes.
            let input_dimensions = self
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(index, size)| match size {
                    Size::Static(1) => None,
                    Size::Static(value) => Some((index, *value)),
                    Size::Dynamic(_) => None,
                })
                .collect::<Vec<_>>();
            let output_dimensions = shape
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(index, size)| match size {
                    Size::Static(1) => None,
                    Size::Static(value) => Some((index, *value)),
                    Size::Dynamic(_) => None,
                })
                .collect::<Vec<_>>();

            // Partition the two stripped shapes into corresponding contiguous groups with matching element counts.
            // Each group pairs the input dimensions that the reshape merges or splits into the paired output
            // dimensions. Starting a group with one dimension from each side, the side with the smaller running element
            // product absorbs its next dimension until the two products match. When one side runs out of dimensions
            // (or a product overflows) before the products match, the reshape mixes dimensions in a way that sharding
            // propagation cannot describe.
            let mut input_start_index = 0usize;
            let mut output_start_index = 0usize;
            let mut groups = Vec::new();
            while input_start_index < input_dimensions.len() || output_start_index < output_dimensions.len() {
                if input_start_index == input_dimensions.len() || output_start_index == output_dimensions.len() {
                    return Err(alignment_error().into());
                }
                let input_group_start_index = input_start_index;
                let output_group_start_index = output_start_index;
                let mut input_group_product = input_dimensions[input_start_index].1;
                let mut output_group_product = output_dimensions[output_start_index].1;
                input_start_index += 1;
                output_start_index += 1;
                while input_group_product != output_group_product {
                    if input_group_product < output_group_product {
                        if input_start_index == input_dimensions.len() {
                            return Err(alignment_error().into());
                        }
                        input_group_product = input_group_product
                            .checked_mul(input_dimensions[input_start_index].1)
                            .ok_or_else(alignment_error)?;
                        input_start_index += 1;
                    } else {
                        if output_start_index == output_dimensions.len() {
                            return Err(alignment_error().into());
                        }
                        output_group_product = output_group_product
                            .checked_mul(output_dimensions[output_start_index].1)
                            .ok_or_else(alignment_error)?;
                        output_start_index += 1;
                    }
                }
                groups.push((input_group_start_index, input_start_index, output_group_start_index, output_start_index));
            }

            // Distribute the input dimension shardings over the target dimensions. Output dimensions start out
            // replicated, which already covers the singleton axes stripped above and every split/merge group.
            // One-to-one groups then carry their input dimension's sharding over to the paired output dimension.
            // Dimensions that take part in an actual split or merge must be replicated on the input side, because
            // the reshape redistributes their elements across mesh shards.
            let mut output_sharding_dimensions =
                std::iter::repeat_n(ShardingDimension::replicated(), shape.rank()).collect::<Vec<_>>();
            for (input_group_start_index, input_group_end_index, output_group_start_index, output_group_end_index) in
                groups
            {
                let input_group_length = input_group_end_index - input_group_start_index;
                let output_group_length = output_group_end_index - output_group_start_index;
                if input_group_length == 1 && output_group_length == 1 {
                    let input_dimension_index = input_dimensions[input_group_start_index].0;
                    let output_dimension_index = output_dimensions[output_group_start_index].0;
                    output_sharding_dimensions[output_dimension_index] =
                        sharding.dimensions()[input_dimension_index].clone();
                    continue;
                }
                if !input_dimensions[input_group_start_index..input_group_end_index]
                    .iter()
                    .all(|(index, _)| matches!(sharding.dimensions()[*index], ShardingDimension::Replicated))
                {
                    return Err(TypeError {
                        message: "'reshape' cannot preserve sharding across the requested reshape".to_string(),
                    }
                    .into());
                }
            }

            // Rebuild the sharding over the target rank. The unreduced/reduced and manual-axis sets describe pending
            // cross-device reductions over mesh axes, which are orthogonal to how the array's ranked dimensions are
            // regrouped, so they pass through unchanged while only the per-dimension placement is recomputed (JAX's
            // `_reshape_unreduced_rule` / `_reshape_reduced_rule` likewise propagate them as-is).
            Some(
                Sharding::new(sharding.mesh().clone(), output_sharding_dimensions)
                    .and_then(|output| output.with_unreduced_axes(sharding.unreduced_axes().clone()))
                    .and_then(|output| output.with_reduced_axes(sharding.reduced_axes().clone()))
                    .and_then(|output| output.with_varying_manual_axes(sharding.varying_manual_axes().clone()))
                    .map(|sharding| sharding.without_auto_axes())
                    .map_err(|_| TypeError { message: "'reshape' produced an invalid output sharding".to_string() })?,
            )
        } else {
            None
        };

        ArrayType::new(self.data_type(), shape)
            .with_memory(self.memory())
            .with_sharding(sharding)
            .map_err(|_| TypeError { message: "'reshape' produced an invalid output type".to_string() }.into())
    }
}

/// Any context-carrying value reshapes by binding a [`ReshapeOperation`] through its own context. The
/// `From<ReshapeOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Reshape for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ReshapeOperation>,
{
    #[inline]
    fn reshape(&self, shape: Shape) -> Result<Self, ProgramError> {
        let input_type = self.r#type().into_owned();
        let output_type = input_type.reshape(shape)?;
        if input_type == output_type {
            return Ok(self.clone());
        }
        let mut outputs = self.dispatch_domain().bind(
            ReshapeOperation::new(output_type.shape().clone()),
            Vec::new(),
            std::slice::from_ref(self),
        )?;
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::contexts::EagerContext;
    use crate::macros::{
        check_operation_batching, check_operation_differentiation, check_operation_partial_evaluation,
        check_operation_transposition, check_operation_type_inference,
    };
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding};
    use crate::types::{DataType, Layout, Memory, StridedLayout};

    use super::*;

    #[test]
    fn test_reshape() {
        let shape = Shape::new(vec![Size::Static(2), Size::Static(3)]);
        let operation = ReshapeOperation::new(shape.clone());

        // Operation identity and accessors.
        assert_eq!(operation.name(), RESHAPE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "reshape [shape=[2, 3]]");
        assert_eq!(*operation.output_shape(), shape);

        // Type inference validates the element count and returns the target shape.
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        let output_type = ArrayType::new(DataType::F64, shape.clone());
        check_operation_type_inference!(
            operation = operation.clone(),
            cases = [
                {
                    input_types = [input_type.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [],
                    error = "expected 1 input but got 0",
                },
                {
                    input_types = [ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(5)]))],
                    error = "'reshape' changes the number of elements",
                },
            ],
        );

        // Type-level (abstract) reshaping validates the target shape and returns the output type without consuming
        // the borrowed input type.
        assert_eq!(input_type.reshape(shape.clone()), Ok(output_type.clone()));

        // Interpretation reinterprets the row-major payload under the target shape.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let output = operation
            .interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Invalid interpreter arity reports the exact program error.
        assert_eq!(
            InterpretableOperation::<EagerContext<Array>>::interpret(
                &operation,
                &EagerContext::<Array>::new(),
                &EmptyRegionDriver,
                &[]
            ),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );

        // Program rendering uses the canonical operation name and includes the captured output shape.
        let mut builder = ProgramBuilder::<Array, ReshapeOperation>::new();
        let program_input = builder.add_input(input_type);
        let program_output = builder.add_instruction(operation, Vec::new(), vec![program_input]).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![program_output], Placeholder, Placeholder).unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[6] .
                let %1:f64[2, 3] = reshape [shape=[2, 3]] %0
                in (%1)
            "}
            .trim_end(),
        );

        // Check the standard partial-evaluation contract for both known and residual inputs.
        let input = Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let expected = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        check_operation_partial_evaluation!(
            backend = (Array, ArrayOperation<Array>),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
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

        // Check batching, forward differentiation, and the inverse-reshape pullback.
        let batched_input = Array::matrix(2, 6, (0..12).map(|value| value as f64).collect());
        let batched_output = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 2.into(), 3.into()])),
            (0..12).map(|value| value as f64).collect(),
        );
        check_operation_batching!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            axis_size = 2,
            cases = [{
                inputs = [(@mapped(axis = 0), batched_input)],
                outputs = [(@mapped(axis = 0), batched_output)],
            }],
        );
        check_operation_differentiation!(
            @approx(step = 0.125, epsilon = 1e-9),
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 2.into()])),
            cases = [{
                primals = [Array::vector(vec![1.0, 2.0, 3.0, 4.0])],
                tangents = [Array::vector(vec![5.0, 6.0, 7.0, 8.0])],
                primal_outputs = [Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0])],
                tangent_outputs = [Array::matrix(2, 2, vec![5.0, 6.0, 7.0, 8.0])],
            }],
        );
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))))],
                output_cotangents = [Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
                input_cotangents = [Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])],
            }],
        );

        // Reshaping back to the input shape restores its complete cotangent type after the forward reshape has
        // intentionally cleared physical layout metadata.
        let layout = Layout::Strided(StridedLayout::new(vec![8]));
        let placed_input_type = ArrayType::new(DataType::F64, Shape::new(vec![6.into()]))
            .with_layout(layout)
            .with_memory(Memory::Host { pinned: true });
        let placed_output_type = ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into()]))
            .with_memory(Memory::Host { pinned: true });
        check_operation_transposition!(
            @exact,
            operation = ReshapeOperation::new(Shape::new(vec![2.into(), 3.into()])),
            cases = [{
                inputs = [(@linear(type = placed_input_type.clone()))],
                output_cotangents = [Array::from_f64s(
                    placed_output_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )],
                input_cotangents = [Array::from_f64s(
                    placed_input_type,
                    vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                )],
            }],
        );
    }

    #[test]
    fn test_array_type_reshape() {
        // Reshaping requires statically known element counts on both sides, so dynamic input and target shapes are
        // rejected with precise errors at the type level, through operation inference, and through value kernels.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(6)]));
        let dynamic_shape = Shape::new(vec![Size::Dynamic(None), Size::Static(3)]);
        let dynamic_type = ArrayType::new(DataType::F64, dynamic_shape.clone());
        assert_eq!(
            dynamic_type.reshape(Shape::new(vec![Size::Static(6)])),
            Err(ProgramError::Type(TypeError {
                message: "'reshape' requires statically known input element counts".to_string(),
            })),
        );
        assert_eq!(
            static_type.reshape(dynamic_shape.clone()),
            Err(ProgramError::Type(TypeError {
                message: "'reshape' requires statically known output element counts".to_string(),
            })),
        );
        assert_eq!(
            ReshapeOperation::new(dynamic_shape.clone()).infer_output_types(std::slice::from_ref(&static_type), &[]),
            Err(TypeError { message: "'reshape' requires statically known output element counts".to_string() }),
        );
        assert_eq!(
            Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).reshape(dynamic_shape),
            Err(ProgramError::Type(TypeError {
                message: "'reshape' requires statically known output element counts".to_string(),
            })),
        );

        // Reshaping a dynamically sized type to its own shape short-circuits as the identity.
        assert_eq!(dynamic_type.reshape(dynamic_type.shape().clone()), Ok(dynamic_type.clone()));

        // A non-identity reshape preserves memory placement but clears a layout whose output strides cannot be
        // inferred from the logical target shape alone.
        let placed_type = static_type
            .clone()
            .with_layout(Layout::Strided(StridedLayout::new(vec![8])))
            .with_memory(Memory::Host { pinned: true });
        assert_eq!(
            placed_type.reshape(Shape::new(vec![Size::Static(2), Size::Static(3)])),
            Ok(ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]))
                .with_memory(Memory::Host { pinned: true })),
        );

        // Singleton insertion preserves the corresponding non-singleton dimension placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(1), Size::Static(8), Size::Static(1)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::replicated(),
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap())
        );

        // Merging replicated axes preserves an independent unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type =
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8), Size::Static(6)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                        .unwrap(),
                )
                .unwrap())
        );

        // Splitting a replicated axis likewise preserves an unchanged sharded dimension.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap(),
                )
                .unwrap())
        );

        // Reshape regroups ranked dimensions but leaves the reduction-state (unreduced/reduced) and varying-manual
        // axis sets untouched, since those describe mesh axes that do not correspond to ranked array dimensions.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("r", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8), Size::Static(2), Size::Static(3)]))
                .with_sharding(
                    Sharding::new(
                        mesh,
                        vec![
                            ShardingDimension::sharded(["x"]),
                            ShardingDimension::replicated(),
                            ShardingDimension::replicated(),
                        ],
                    )
                    .unwrap()
                    .with_reduced_axes(["r"])
                    .unwrap(),
                )
                .unwrap())
        );

        // A genuinely split sharded dimension cannot preserve its placement.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(8)]))
            .with_sharding(Sharding::new(mesh, vec![ShardingDimension::sharded(["x"])]).unwrap())
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(2), Size::Static(4)])),
            Err(ProgramError::Type(TypeError {
                message: "'reshape' cannot preserve sharding across the requested reshape".to_string(),
            })),
        );

        // A genuinely merged sharded dimension cannot preserve its placement either.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])]).unwrap(),
            )
            .unwrap();
        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(8)])),
            Err(ProgramError::Type(TypeError {
                message: "'reshape' cannot preserve sharding across the requested reshape".to_string(),
            })),
        );

        // Many-to-many regrouping is supported when every participating dimension is replicated.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(6)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            input_type.reshape(Shape::new(vec![Size::Static(3), Size::Static(4)])),
            Ok(ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(4)]))
                .with_sharding(
                    Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
                        .unwrap()
                        .with_varying_manual_axes(["x"])
                        .unwrap(),
                )
                .unwrap())
        );
    }
}
