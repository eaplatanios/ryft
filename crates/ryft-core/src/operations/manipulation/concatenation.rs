use std::collections::BTreeSet;
use std::fmt::Display;

use crate::batching::{
    ArrayBatch, BatchAxis, BatchableOperation, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::Zero;
use crate::operations::manipulation::{Broadcast, SliceOperation, Transpose};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::values::Value;
use crate::programs::{MaybeZero, ProgramError};
use crate::sharding::Sharding;
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, Shape, Size};

// TODO(eaplatanios): Review from here onwards.

/// Canonical operation name for [`ConcatenateOperation`].
pub const CONCATENATE_OPERATION_NAME: &str = "concatenate";

/// [`Operation`] that joins two or more input arrays end to end along one axis. Refer to the documentation of
/// [`Concatenate`] for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ConcatenateOperation {
    /// Axis along which the operands are joined.
    axis: usize,
}

impl ConcatenateOperation {
    /// Creates a new [`ConcatenateOperation`] that joins its operands along `axis`.
    #[inline]
    pub fn new(axis: usize) -> Self {
        Self { axis }
    }

    /// Returns the axis along which this [`ConcatenateOperation`] joins its operands.
    #[inline]
    pub fn axis(&self) -> usize {
        self.axis
    }
}

impl Display for ConcatenateOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for ConcatenateOperation {
    #[inline]
    fn name(&self) -> &'static str {
        CONCATENATE_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        match ArrayType::concatenate(input_types, self.axis) {
            Ok(output_type) => Ok(vec![output_type]),
            Err(ProgramError::Type(error)) => Err(error),
            Err(error) => Err(TypeError { message: error.to_string() }),
        }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("axis", self.axis))
    }
}

impl<C: Domain<Type = ArrayType, Value: Concatenate>> InterpretableOperation<C> for ConcatenateOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(vec![Concatenate::concatenate(inputs, self.axis)?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ConcatenateOperation where
    C::Operation: From<ConcatenateOperation>
{
}

/// Forward-mode rule for [`ConcatenateOperation`]: `concatenate` is linear in every operand, so the tangent
/// concatenates the operand tangents along the same axis.
impl<C: Context<Type = ArrayType> + Zero<C::Value>> DifferentiableOperation<C> for ConcatenateOperation
where
    C::Operation: From<ConcatenateOperation>,
    C::Value: Concatenate,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let primals = inputs.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
        // The concatenation needs every operand tangent as a real value, so materialize the structurally zero ones
        // (the shared all-zero fast path already handled the case where every operand tangent is zero).
        let tangents = inputs
            .iter()
            .map(|dual| dual.tangent().clone().materialize(context))
            .collect::<Result<Vec<_>, _>>()?;
        let primal = Concatenate::concatenate(&primals, self.axis())?;
        let tangent = Concatenate::concatenate(&tangents, self.axis())?;
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Transpose (vector-Jacobian product) for a [`ConcatenateOperation`].
///
/// The forward map `(t_0, ..., t_n) ↦ concatenate([t_0, ..., t_n], axis)` lays the operands end to end along `axis`,
/// so its pullback splits the output cotangent back into the per-operand pieces by slicing the cotangent at the
/// cumulative operand offsets along `axis`: operand `i` receives `slice(cotangent, start, limit, unit strides)` with
/// `start[axis]` and `limit[axis]` set to that operand's `[offset, offset + operand_axis_size)` window and the full
/// extent on every other axis. The operands must have a static size along `axis` so the offsets are known.
/// Symbolic-zero cotangents propagate unchanged to every operand.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ConcatenateOperation
where
    O: Operation<ArrayType> + From<SliceOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("output", outputs, 1, ProgramError);
        if inputs.is_empty() {
            return Err(TypeError {
                message: "'concatenate' transpose expects at least one operand but got none".to_string(),
            }
            .into());
        }
        let axis = self.axis();
        match &outputs[0] {
            MaybeZero::Zero(_) => Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect()),
            MaybeZero::Value(cotangent) => {
                let rank = inputs[0].r#type().rank();
                let mut offset = 0usize;
                let mut input_cotangents = Vec::with_capacity(inputs.len());
                for (index, input) in inputs.iter().enumerate() {
                    let input_type = input.r#type();
                    let dimension = input_type.dimension(axis as isize);
                    let Size::Static(operand_axis_size) = dimension else {
                        return Err(TypeError {
                            message: format!(
                                "'concatenate' transpose requires a static size along the concatenated axis {axis} \
                                but operand {index} has size {dimension}",
                            ),
                        }
                        .into());
                    };
                    let mut start_indices = vec![0usize; rank];
                    let mut limit_indices = input_type
                        .shape()
                        .dimensions()
                        .iter()
                        .enumerate()
                        .map(|(other_axis, size)| {
                            size.value().ok_or_else(|| {
                                TypeError {
                                    message: format!(
                                        "'concatenate' transpose requires a static operand shape but operand {index} \
                                        has size {size} on axis {other_axis}",
                                    ),
                                }
                                .into()
                            })
                        })
                        .collect::<Result<Vec<usize>, ProgramError>>()?;
                    start_indices[axis] = offset;
                    limit_indices[axis] = offset + operand_axis_size;
                    let strides = vec![1; rank];
                    let outputs = context.stage_operation(
                        SliceOperation::new(start_indices, limit_indices).with_strides(strides)?,
                        Vec::new(),
                        std::slice::from_ref(cotangent),
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    input_cotangents.push(MaybeZero::Value(outputs.into_iter().next().unwrap()));
                    offset += operand_axis_size;
                }
                Ok(input_cotangents)
            }
        }
    }
}

/// Batching rule for [`ConcatenateOperation`].
///
/// All operands are aligned on one physical batch axis (replicated operands are broadcast to gain it via
/// [`ArrayBatch::match_axis`](crate::batching::ArrayBatch::match_axis), so each batch item
/// concatenates its own operands), and the concatenated axis is
/// shifted past the inserted batch axis when the batch axis sits at or before it. When no operand is batched, the
/// operation passes through unchanged.
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ConcatenateOperation
where
    C::Value: Broadcast + Transpose,
    ConcatenateOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        if inputs.is_empty() {
            return Err(
                TypeError { message: "'concatenate' expects at least one operand but got none".to_string() }.into()
            );
        }
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        let Some(batch_axis) = batch_axes.iter().copied().flatten().next() else {
            return self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()]);
        };
        let axis_size = ArrayBatch::common_batch_size(inputs)?.expect("a mapped input pins the batch size");
        let materialized = inputs
            .iter()
            .map(|input| input.match_axis(batch_axis as isize, axis_size, context.axis_sharding().clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let lifted_axis = if batch_axis <= self.axis() { self.axis() + 1 } else { self.axis() };
        ConcatenateOperation::new(lifted_axis).interpret_with_batch_axes(
            context,
            materialized.as_slice(),
            &[BatchAxis::from_position(batch_axis)],
        )
    }
}

/// Represents the ability to join two or more arrays end to end along one axis. This is the direct analogue of the
/// StableHLO [`concatenate`](https://openxla.org/stablehlo/spec#concatenate) operation and JAX's
/// [`lax.concatenate`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html).
///
/// `Concatenate::concatenate(operands, axis)` returns an array whose elements along `axis` are the elements of the
/// first operand followed by the elements of the second operand, and so on. There must be at least one operand, all
/// operands must share one data type and one rank, and every axis other than `axis` must have the same [`Size`]
/// across all operands. The output keeps the shared non-concatenated dimensions and has size along `axis` equal to
/// the sum of the operand sizes along `axis`.
///
/// # Dynamic Dimensions
///
/// The concatenated axis may be dynamic: if any operand's `axis` dimension is [`Size::Dynamic`], the output `axis`
/// dimension is also [`Size::Dynamic`], with an upper bound equal to the sum of the operand upper bounds when every
/// operand is bounded and an unbounded `Size::Dynamic(None)` otherwise. So concatenating a dynamic stack with a
/// fixed slice along the dynamic axis grows the stack while keeping its type dynamic — `[?, d] ++ [1, d] = [?, d]`
/// along axis `0`. This is the typing that lets a dynamically-sized residual stack accumulate one iteration per loop
/// iteration. The non-concatenated axes must still match per [`Size`] equality (a [`Size::Dynamic`] non-concatenated
/// axis is allowed only when every operand carries the same [`Size`] there) and propagate unchanged.
///
/// # Example
///
/// The following example shows how to use [`Concatenate`] in practice:
///
/// ```rust
/// # use ryft_core::operations::manipulation::Concatenate;
/// # use ryft_core::programs::ProgramError;
/// # use ryft_core::backends::arrays::Array;
/// #
/// # fn main() -> Result<(), ProgramError> {
/// // Join two 1x2 matrices along axis 0 into a 2x2 matrix. This is equivalent to
/// // `jax.lax.concatenate([x, y], dimension=0)` in JAX.
/// let x = Array::matrix(1, 2, vec![1.0, 2.0]);
/// let y = Array::matrix(1, 2, vec![3.0, 4.0]);
/// let z = Concatenate::concatenate(&[x, y], 0)?;
/// // `z` has shape [2, 2] with values [[1.0, 2.0], [3.0, 4.0]].
/// assert_eq!(z.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);
///
/// // Joining the same operands along axis 1 produces a 1x4 matrix instead.
/// let x = Array::matrix(1, 2, vec![1.0, 2.0]);
/// let y = Array::matrix(1, 2, vec![3.0, 4.0]);
/// let z = Concatenate::concatenate(&[x, y], 1)?;
/// assert_eq!(z.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);
/// # Ok(())
/// # }
/// ```
pub trait Concatenate: Sized {
    /// Joins `operands` end to end along `axis`. Refer to the documentation of this trait for more information on
    /// what this operation does.
    ///
    /// # Parameters
    ///
    ///   - `operands`: Arrays to join, in order. There must be at least one operand, and all operands must share one
    ///     data type and rank and agree on every axis other than `axis`.
    ///   - `axis`: Axis along which the operands are joined.
    fn concatenate(operands: &[Self], axis: usize) -> Result<Self, ProgramError>;
}

impl Concatenate for ArrayType {
    fn concatenate(operands: &[Self], axis: usize) -> Result<Self, ProgramError> {
        let Some(first) = operands.first() else {
            return Err(
                TypeError { message: "'concatenate' expects at least one operand but got none".to_string() }.into()
            );
        };
        let rank = first.rank();
        if axis >= rank {
            return Err(TypeError {
                message: format!("'concatenate' axis {axis} is out of bounds for operands of rank {rank}"),
            }
            .into());
        }
        // The concatenated dimension accumulates as we scan operands: a static run sums sizes, while any dynamic
        // operand size forces a dynamic output dimension whose upper bound stays known only if every operand is
        // bounded.
        let mut concatenated_static = 0usize;
        let mut concatenated_dynamic = false;
        let mut concatenated_upper_bound = Some(0usize);
        for (index, operand) in operands.iter().enumerate() {
            if operand.data_type() != first.data_type() {
                return Err(TypeError {
                    message: format!(
                        "'concatenate' operands must share one data type but operand {index} has data type {} and \
                        operand 0 has data type {}",
                        operand.data_type(),
                        first.data_type(),
                    ),
                }
                .into());
            }
            if operand.rank() != rank {
                return Err(TypeError {
                    message: format!(
                        "'concatenate' operands must share one rank but operand {index} has rank {} and operand 0 has \
                        rank {rank}",
                        operand.rank(),
                    ),
                }
                .into());
            }
            for other_axis in 0..rank {
                if other_axis == axis {
                    continue;
                }
                let dimension = operand.dimension(other_axis as isize);
                let first_dimension = first.dimension(other_axis as isize);
                if dimension != first_dimension {
                    return Err(TypeError {
                        message: format!(
                            "'concatenate' operands must agree on every axis other than {axis} but operand {index} has \
                            size {dimension} on axis {other_axis} and operand 0 has size {first_dimension}",
                        ),
                    }
                    .into());
                }
            }
            let dimension = operand.dimension(axis as isize);
            match dimension {
                Size::Static(size) => concatenated_static += size,
                Size::Dynamic(_) => concatenated_dynamic = true,
            }
            concatenated_upper_bound = match (concatenated_upper_bound, dimension.upper_bound()) {
                (Some(accumulated), Some(bound)) => Some(accumulated + bound),
                _ => None,
            };
        }
        let concatenated = if concatenated_dynamic {
            // When every operand is bounded the summed upper bound includes one slack unit per operand (each
            // `Size::upper_bound` is exclusive), which still soundly over-approximates the concatenated extent.
            Size::Dynamic(concatenated_upper_bound)
        } else {
            Size::Static(concatenated_static)
        };
        let mut dimensions = first.shape().dimensions().to_vec();
        dimensions[axis] = concatenated;

        // Output sharding (JAX's `_concatenate_sharding_rule` + `_concatenate_(un)reduced_rule`): every sharded
        // operand must agree, since concatenation only interleaves elements and each element keeps its operand's
        // placement and pending-reduction status. Operands without a sharding impose no constraint, and their
        // varying-manual axes are unioned. A disagreement is an error only when it involves an Explicit axis (see
        // `Sharding::conflicts_on_explicit_axes_with`); a Manual/Auto-only disagreement is tolerated and the first
        // sharded operand's placement is kept, so a `shard_map` manual body concatenating local shards is not
        // rejected. The output adopts the common sharding (with the unioned varying-manual axes) or stays unsharded.
        let mut output_sharding: Option<Sharding> = None;
        let mut varying_manual_axes = BTreeSet::new();
        for operand in operands {
            let Some(sharding) = operand.sharding() else {
                continue;
            };
            varying_manual_axes.extend(sharding.varying_manual_axes().iter().cloned());
            match &output_sharding {
                None => output_sharding = Some(sharding.clone()),
                Some(reference) => {
                    if reference.mesh() != sharding.mesh() {
                        return Err(
                            TypeError { message: "'concatenate' operands must use the same mesh".to_string() }.into()
                        );
                    }
                    if reference.conflicts_on_explicit_axes_with(sharding) {
                        return Err(TypeError {
                            message: format!(
                                "'concatenate' operands must be sharded identically, but got {reference} and {sharding}"
                            ),
                        }
                        .into());
                    }
                }
            }
        }
        let output_sharding = output_sharding
            .map(|sharding| sharding.with_varying_manual_axes(varying_manual_axes))
            .transpose()
            .map_err(|error| TypeError { message: error.to_string() })?;
        ArrayType::new(first.data_type(), Shape::new(dimensions))
            .with_sharding(output_sharding)
            .map_err(|error| TypeError { message: error.to_string() }.into())
    }
}

/// Any context-carrying value concatenates by binding a [`ConcatenateOperation`] through its own context. The
/// `From<ConcatenateOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Concatenate for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<ConcatenateOperation>,
{
    fn concatenate(operands: &[Self], axis: usize) -> Result<Self, ProgramError> {
        let Some(first) = operands.first() else {
            return Err(
                TypeError { message: "'concatenate' expects at least one operand but got none".to_string() }.into()
            );
        };
        let mut outputs = first.dispatch_domain().bind(ConcatenateOperation::new(axis), Vec::new(), operands)?;
        crate::macros::check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, BatchingContext};
    use crate::contexts::EagerContext;
    use crate::differentiation::reverse::ReverseModeDifferentiate;
    use crate::macros::check_operation;
    use crate::operations::math::{Reduce, ReductionKind};
    use crate::parameters::Placeholder;
    use crate::programs::ProgramError;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::regions::EmptyRegionDriver;
    use crate::programs::types::Typed;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing_v2::linear::DenseDifferentiate;
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_concatenate() {
        let operation = ConcatenateOperation::new(0);

        // Operation identity and accessors.
        assert_eq!(operation.name(), CONCATENATE_OPERATION_NAME);
        assert_eq!(format!("{operation}"), "concatenate [axis=0]");
        assert_eq!(operation.axis(), 0);

        // Type inference sums the concatenated axis and keeps the shared axes, and the type-level (abstract)
        // capability backs it without consuming the borrowed input types.
        let first_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        let second_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(2)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(4), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[first_type.clone(), second_type.clone()], &[]),
            Ok(vec![output_type.clone()]),
        );
        assert_eq!(ArrayType::concatenate(&[first_type.clone(), second_type.clone()], 0), Ok(output_type.clone()),);

        // A single operand passes through unchanged.
        assert_eq!(ArrayType::concatenate(std::slice::from_ref(&first_type), 0), Ok(first_type.clone()));

        // Interpretation joins the row-major payloads along axis 0.
        let first = Array::matrix(1, 2, vec![1.0, 2.0]);
        let second = Array::matrix(3, 2, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let output = operation.interpret(&EagerContext::<Array>::new(), &EmptyRegionDriver, &[first, second]).unwrap();
        assert_eq!(*output[0].r#type(), output_type);
        assert_eq!(output[0].to_f64s(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);

        // Concatenating along a middle axis keeps the leading and trailing axes and sums the middle one.
        let middle = ConcatenateOperation::new(1);
        let left = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(1)]));
        let right = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        assert_eq!(
            middle.infer_output_types(&[left.clone(), right.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(4)]))]),
        );

        // Dynamic-axis propagation: concatenating along a dynamic axis yields a dynamic axis. With one operand
        // unbounded, the output upper bound is unknown; with all bounded, the output upper bound sums the operand
        // bounds. The non-concatenated axes still propagate their (possibly equal dynamic) sizes.
        let dynamic_stack = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)]));
        let fixed_slice = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[dynamic_stack.clone(), fixed_slice.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(2)]))]),
        );
        let bounded_stack = ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(4)), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[bounded_stack.clone(), fixed_slice.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(Some(6)), Size::Static(2)]))]),
        );
        let dynamic_non_axis = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Dynamic(None)]));
        assert_eq!(
            operation.infer_output_types(&[dynamic_non_axis.clone(), dynamic_non_axis.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Dynamic(None)]))]),
        );

        // Invalid inputs report precise operation and capability errors.
        assert_eq!(
            operation.infer_output_types(&[], &[]),
            Err(TypeError { message: "'concatenate' expects at least one operand but got none".to_string() }),
        );
        assert_eq!(
            ConcatenateOperation::new(2).infer_output_types(&[first_type.clone(), second_type.clone()], &[]),
            Err(TypeError { message: "'concatenate' axis 2 is out of bounds for operands of rank 2".to_string() }),
        );
        assert_eq!(
            operation.infer_output_types(&[first_type.clone(), ArrayType::scalar(DataType::F64)], &[]),
            Err(TypeError {
                message: "'concatenate' operands must share one rank but operand 1 has rank 0 and operand 0 has rank 2"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    first_type.clone(),
                    ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(3), Size::Static(2)])),
                ],
                &[]
            ),
            Err(TypeError {
                message: "'concatenate' operands must share one data type but operand 1 has data type f32 and operand \
                    0 has data type f64"
                    .to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    first_type.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3), Size::Static(5)])),
                ],
                &[]
            ),
            Err(TypeError {
                message: "'concatenate' operands must agree on every axis other than 0 but operand 1 has size 5 on \
                    axis 1 and operand 0 has size 2"
                    .to_string(),
            }),
        );
        // A non-concatenated dynamic axis must match per `Size` equality across operands.
        assert_eq!(
            operation.infer_output_types(
                &[
                    dynamic_non_axis.clone(),
                    ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1), Size::Dynamic(Some(3))])),
                ],
                &[]
            ),
            Err(TypeError {
                message: "'concatenate' operands must agree on every axis other than 0 but operand 1 has size <3 on \
                    axis 1 and operand 0 has size *"
                    .to_string(),
            }),
        );

        // Program rendering uses the canonical operation name and includes the captured axis.
        let mut builder = ProgramBuilder::<Array, ConcatenateOperation>::new();
        let program_first = builder.add_input(first_type);
        let program_second = builder.add_input(second_type);
        let program_output =
            builder.add_instruction(operation, Vec::new(), vec![program_first, program_second]).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![program_output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[1, 2], %1:f64[3, 2] .
                let %2:f64[4, 2] = concatenate [axis=0] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_concatenate_propagates_operand_sharding() {
        use std::collections::BTreeSet;

        use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};

        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let operation = ConcatenateOperation::new(0);
        let sharded = || {
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap()
        };
        let row = |sharding: Sharding| {
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]))
                .with_sharding(sharding)
                .unwrap()
        };

        // Operands sharded identically: the output (size summed on axis 0) inherits the common sharding.
        let output = operation.infer_output_types(&[row(sharded()), row(sharded())], &[]).unwrap();
        assert_eq!(output[0].sharding(), Some(&sharded()));
        assert_eq!(output[0].dimension(0), Size::Static(8));

        // Operands that differ only by varying-manual axes are tolerated and the axes are unioned on the output.
        let varying = |axis: &str| {
            ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]))
                .with_sharding(
                    Sharding::new(
                        mesh.clone(),
                        vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
                    )
                    .unwrap()
                    .with_varying_manual_axes([axis])
                    .unwrap(),
                )
                .unwrap()
        };
        let output = operation.infer_output_types(&[varying("m"), row(sharded())], &[]).unwrap();
        assert_eq!(output[0].sharding().unwrap().varying_manual_axes(), &BTreeSet::from(["m".to_string()]));

        // A conflicting Explicit-axis placement is an error.
        let replicated =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap();
        assert!(operation.infer_output_types(&[row(sharded()), row(replicated)], &[]).is_err());

        // Operands without a sharding leave the output unsharded.
        let plain = ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(4), Size::Static(2)]));
        assert_eq!(operation.infer_output_types(&[plain.clone(), plain], &[]).unwrap()[0].sharding(), None);
    }

    #[test]
    fn test_concatenate_array_kernel() {
        // A rank-3 concatenation along a middle axis exercises the row-major odometer: the two operands interleave
        // their middle-axis blocks while keeping the leading and trailing axes intact.
        let first = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(1), Size::Static(2)])),
            vec![1.0, 2.0, 3.0, 4.0],
        );
        let second = Array::from_f64s(
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)])),
            vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        );
        let output = Concatenate::concatenate(&[first, second], 1).unwrap();
        assert_eq!(
            *output.r#type(),
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3), Size::Static(2)])),
        );
        // Each leading slice gets the first operand's slice followed by the second operand's two slices.
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 3.0, 4.0, 9.0, 10.0, 11.0, 12.0]);

        // Three operands joined along axis 0 stack in order.
        let output = Concatenate::concatenate(
            &[Array::vector(vec![1.0]), Array::vector(vec![2.0, 3.0]), Array::vector(vec![4.0])],
            0,
        )
        .unwrap();
        assert_eq!(output.to_f64s(), vec![1.0, 2.0, 3.0, 4.0]);

        // The kernel validates its operand shapes eagerly through the type-level capability.
        assert_eq!(
            Concatenate::concatenate(&[Array::vector(vec![1.0]), Array::scalar(2.0)], 0),
            Err(ProgramError::Type(TypeError {
                message: "'concatenate' operands must share one rank but operand 1 has rank 0 and operand 0 has rank 1"
                    .to_string(),
            })),
        );
    }

    #[test]
    fn test_concatenate_value_and_grad_routes_cotangent_per_operand() {
        // f(x, y) = sum(concatenate([x, y], 0) * w) with w = [1, 2, 3, 4, 5]: the joined output is [x0, x1, y0, y1,
        // y2], so f = x0 + 2*x1 + 3*y0 + 4*y1 + 5*y2. The pullback slices the weighted cotangent [1, 2, 3, 4, 5] into
        // the first two entries for x and the last three for y.
        let (value, (x_gradient, y_gradient)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .value_and_gradient(
                |(x, y)| {
                    let weights = x.context().lift(Array::vector(vec![1.0, 2.0, 3.0, 4.0, 5.0])).unwrap();
                    (Concatenate::concatenate(&[x, y], 0).unwrap() * weights).reduce(&[0], ReductionKind::Sum)
                },
                (Array::vector(vec![1.0, 2.0]), Array::vector(vec![3.0, 4.0, 5.0])),
            )
            .unwrap();
        // f = 1 + 4 + 9 + 16 + 25 = 55.
        assert_abs_diff_eq!(value.to_f64s()[0], 55.0, epsilon = 1e-9);
        assert_eq!(x_gradient.to_f64s(), vec![1.0, 2.0]);
        assert_eq!(y_gradient.to_f64s(), vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_concatenate_jacfwd_stacks_operand_coordinates() {
        // Forward mode through `f(x, y) = concatenate([x, y], 0)` over `x = [a, b]` and `y = [c]` produces one
        // selection Jacobian block per operand: `x` maps to the first two output rows and `y` to the last.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacfwd(
                |(x, y)| Concatenate::concatenate(&[x, y], 0),
                (Array::vector(vec![1.0, 2.0]), Array::vector(vec![3.0])),
            )
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [x_block, y_block] = blocks.as_slice() else { unreachable!() };
        assert_eq!(x_block.output_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(x_block.input_type().static_shape().unwrap().as_slice(), &[2]);
        // d(output)/d(x): output rows 0 and 1 are x0 and x1; row 2 (from y) is unaffected by x.
        assert_eq!(x_block.value().values(), &[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]);
        assert_eq!(y_block.output_type().static_shape().unwrap().as_slice(), &[3]);
        assert_eq!(y_block.input_type().static_shape().unwrap().as_slice(), &[1]);
        // d(output)/d(y): only output row 2 (from y0) depends on y.
        assert_eq!(y_block.value().values(), &[0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_concatenate_batching_lifts_batch_axis() {
        check_operation!(
            @batching @exact,
            operation = ConcatenateOperation::new(0),
            axis_size = 2,
            cases = [
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0])),
                    ],
                    outputs = [(@mapped(
                        axis = 0
                    ), Array::matrix(2, 4, vec![0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0]))],
                },
                {
                    inputs = [
                        (@mapped(axis = 0), Array::matrix(2, 2, vec![0.0, 1.0, 2.0, 3.0])),
                        (@replicated, Array::vector(vec![8.0, 9.0])),
                    ],
                    outputs = [(@mapped(
                        axis = 0
                    ), Array::matrix(2, 4, vec![0.0, 1.0, 8.0, 9.0, 2.0, 3.0, 8.0, 9.0]))],
                },
                {
                    inputs = [
                        (@replicated, Array::vector(vec![1.0, 2.0])),
                        (@replicated, Array::vector(vec![3.0])),
                    ],
                    outputs = [(@replicated, Array::vector(vec![1.0, 2.0, 3.0]))],
                },
            ],
        );
    }

    #[test]
    fn test_concatenate_batching_preserves_materialized_batch_placement() {
        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let physical_sharding =
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                    .unwrap()
                    .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
                    .unwrap();
            let physical_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2)]))
                .with_sharding(physical_sharding.clone())
                .unwrap();
            let mapped = ArrayBatch::new(
                physical_type.clone(),
                Array::from_f64s(physical_type, vec![1.0, 2.0, 3.0, 4.0]),
                BatchAxis::new(0),
            )
            .unwrap();
            let replicated_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(1)]))
                .with_sharding(Sharding::replicated(mesh, 1))
                .unwrap();
            let replicated = ArrayBatch::replicated(Array::from_f64s(replicated_type, vec![5.0]));
            let context = BatchingContext::new(EagerContext::<Array>::new(), 2)
                .with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = ConcatenateOperation::new(0)
                .batch(&context, &crate::EmptyRegionDriver, &[mapped, replicated])
                .unwrap();

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            assert_eq!(outputs[0].r#type().sharding().unwrap().dimensions(), physical_sharding.dimensions(),);
            assert_eq!(outputs[0].value().to_f64s(), vec![1.0, 2.0, 5.0, 3.0, 4.0, 5.0]);
        }
    }
}
