use ryft_macros::capability;

use crate::arrays::{Array, ArrayIrType, ArrayType};
use crate::axes::AxisError;
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::collectives::axis_index::AxisIndex;
use crate::operations::collectives::parallel_reduce::ParallelReduce;
use crate::operations::comparisons::Compare;
use crate::operations::constants::constant::ConstantOperation;
use crate::operations::control_flow::select::Select;
use crate::operations::manipulation::memory::TransferToMemory;
use crate::operations::reductions::ReductionKind;
use crate::programs::{ProgramError, Value, ValueDomainDispatch};

/// Represents the ability to find, for every element, the position along a named axis of the participant that holds
/// its maximum or minimum. Every participant (i.e., every batch item of a named `batch` level or every device shard of
/// a manual mesh axis) receives the same positions, so the result is invariant over a manual mesh axis. These are the
/// Ryft analogues of JAX's
/// [`pargmax` and `pargmin`](https://github.com/jax-ml/jax/blob/8e6230833ef1/jax/_src/lax/parallel.py#L319-L337).
/// The extremum is reduced over the axis with [`parallel_reduce`](ParallelReduce::parallel_reduce), every
/// participant compares it with its own value, and participants whose value equals the extremum contribute their
/// [`axis_index`](AxisIndex::axis_index) while the others contribute [`u64::MAX`], and the minimum of those
/// contributions is reduced over the axis. Ties therefore resolve to the lowest position. A NaN extremum equals no
/// value and yields [`u64::MAX`] (JAX yields the maximum `int32` instead, because its positions are `int32` values
/// while Ryft's are `u64` values). Whether a NaN wins a cross-device extremum depends on the backend. Refer to
/// [`ParallelReduce`] for more information. The output preserves the input's shape and memory space. Only one named
/// axis is supported, without participant groups, as in JAX.
///
/// Inputs must not carry unreduced axes, and the selected manual axis must not be reduced. A named batch axis must
/// have a statically known extent. An empty batch has no winning participant and returns [`u64::MAX`].
///
/// Positions have no tangent components. The composition still requires the derivative of its maximum or minimum
/// reduction when its input has a live tangent, which [`ParallelReduce`] does not support for a manual mesh axis.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// The batch items of a named `batch` level are the participants of its axis, so positions can be found eagerly.
///
/// ```rust
/// # use ryft_core::{
/// #     Array, ArrayBatchingPolicy, ArrayOperation, BatchAxis, BatchAxisSpecification, BatchingTracer, EagerContext,
/// #     ParallelArgExtrema, batch,
/// # };
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let rows = Array::matrix(3, 2, vec![1.0f32, 9.0, 7.0, 2.0, 7.0, 4.0])?;
/// let positions = batch(
///     |row: BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>| {
///         row.parallel_argmax("rows")
///     },
///     rows,
///     BatchAxis::new(0),
///     BatchAxis::replicated(),
///     BatchAxisSpecification::named("rows"),
/// )?;
/// assert_eq!(positions, Array::vector(vec![1u64, 0])?);
/// # Ok(())
/// # }
/// ```
#[capability(projection(ArrayIrType => ArrayType))]
pub trait ParallelArgExtrema<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns, for every element, the lowest position along the named axis `axis_name` of a participant whose value
    /// is the maximum of the values of all participants, as a `u64` value with the shape of this value.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is unbound, if the mesh or reduction state is incompatible with the
    /// composition, or if [`ReductionKind::Max`] does not support this value's data type. Delegated operations may
    /// also reject an unsupported execution or transformation context.
    fn parallel_argmax(&self, axis_name: &str) -> Result<Self, ProgramError>;

    /// Returns, for every element, the lowest position along the named axis `axis_name` of a participant whose value
    /// is the minimum of the values of all participants, as a `u64` value with the shape of this value.
    ///
    /// # Parameters
    ///
    ///   - `axis_name`: Name of an axis bound by an enclosing `batch` level or manual region.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if `axis_name` is unbound, if the mesh or reduction state is incompatible with the
    /// composition, or if [`ReductionKind::Min`] does not support this value's data type. Delegated operations may
    /// also reject an unsupported execution or transformation context.
    fn parallel_argmin(&self, axis_name: &str) -> Result<Self, ProgramError>;
}

impl ParallelArgExtrema for Array {
    // A concrete `Array` never executes inside an axis binder, because the values under a `batch` level or inside
    // a manual region are tracers, so every axis name is unbound for it, exactly as for `parallel_reduce`.

    #[inline]
    fn parallel_argmax(&self, axis_name: &str) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }

    #[inline]
    fn parallel_argmin(&self, axis_name: &str) -> Result<Self, ProgramError> {
        Err(AxisError::UnboundAxisName { name: axis_name.to_string() }.into())
    }
}

impl<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayType,
            Domain: Context<Operation: From<ConstantOperation<Array>>> + AxisIndex,
        > + ParallelReduce
        + Compare
        + Select
        + TransferToMemory,
> ParallelArgExtrema<ArrayType> for V
{
    #[inline]
    fn parallel_argmax(&self, axis_name: &str) -> Result<Self, ProgramError> {
        parallel_arg_extremum(self, axis_name, ReductionKind::Max)
    }

    #[inline]
    fn parallel_argmin(&self, axis_name: &str) -> Result<Self, ProgramError> {
        parallel_arg_extremum(self, axis_name, ReductionKind::Min)
    }
}

/// Returns the lowest position along `axis_name` of a participant attaining the `kind` extremum of `value`.
fn parallel_arg_extremum<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayType,
            Domain: Context<Operation: From<ConstantOperation<Array>>> + AxisIndex,
        > + ParallelReduce
        + Compare
        + Select
        + TransferToMemory,
>(
    value: &V,
    axis_name: &str,
    kind: ReductionKind,
) -> Result<V, ProgramError> {
    let context = value.domain();
    let extremum = value.parallel_reduce(kind, axis_name)?;
    let attains_extremum = extremum.equal(value)?;
    let position = context.axis_index(axis_name)?;
    let memory = value.r#type().memory();

    // Axis indices default to device memory, but both candidates must share the predicate's memory space.
    let position = if position.r#type().memory() == memory { position } else { position.transfer_to_memory(memory)? };
    let sentinel = Array::scalar(u64::MAX)?.transfer_to_memory(memory)?;

    let mut sentinel = context.bind(ConstantOperation::new(sentinel), Vec::new(), &[])?;
    check_count!("output", sentinel, 1, ProgramError);
    let candidates = attains_extremum.select(&position, &sentinel.remove(0))?;
    candidates.parallel_reduce(ReductionKind::Min, axis_name)
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        ArrayBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, Complex, DataType, DimensionValue,
        LogicalMesh, Memory, MeshAxis, MeshAxisType, Sharding, UnreducedKind,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingContext, BatchingError, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::DifferentiationError;
    use crate::operations::collectives::parallel_reduce::PARALLEL_REDUCE_OPERATION_NAME;
    use crate::operations::comparisons::COMPARE_OPERATION_NAME;
    use crate::partial::{PartialEvaluationInput, PartialEvaluationOutput, PartialValue};
    use crate::programs::TypeError;
    use crate::tracing::TracingContext;

    use super::*;

    /// Batch item of an eager named batch over arrays.
    type TestTracer = BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>;

    /// Applies a collective to rows under a batch level that binds `"rows"`.
    fn batch_rows(
        function: impl Fn(TestTracer) -> Result<TestTracer, ProgramError>,
        rows: Array,
    ) -> Result<Array, BatchingError> {
        batch(function, rows, BatchAxis::new(0), BatchAxis::replicated(), BatchAxisSpecification::named("rows"))
    }

    /// Checked manual binding and a vector varying over its axis.
    fn manual_axis_fixture() -> (LogicalMesh, ArrayType, Vec<(String, NamedAxis)>) {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_type = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let axes = vec![(
            "x".to_string(),
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
        )];
        (mesh, input_type, axes)
    }

    /// Complete checked manual argmax composition, also expected after residualization and inactive differentiation.
    fn manual_argmax_rendering() -> &'static str {
        indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
            let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                    parallel_reduce [kind=max, axis_name=\"x\", mesh=['x'=2:manual]] %0
                %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %1
                %3:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    compare [direction=Equal] %2 %0
                %4:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                    axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                %5:u64[] = constant [value=18446744073709551615]
                %6:u64[][sharding={mesh<['x'=2:manual]>, []}] = \
                    broadcast [output_type=u64[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %5
                %7:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %6
                %8:u64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %4 %7
                %9:u64[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                    parallel_reduce [kind=min, axis_name=\"x\", mesh=['x'=2:manual]] %8
            in (%9)"
        }
    }

    /// Residual program retains the device reductions and index while hoisting the known sentinel.
    fn residual_argmax_rendering() -> &'static str {
        indoc! {"
            lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}], \
            %1:u64[][sharding={mesh<['x'=2:manual]>, []}] .
            let %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                    parallel_reduce [kind=max, axis_name=\"x\", mesh=['x'=2:manual]] %0
                %3:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    parallel_vary [axis_name=\"x\"] %2
                %4:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                    compare [direction=Equal] %3 %0
                %5:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                    axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                %6:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = parallel_vary [axis_name=\"x\"] %1
                %7:u64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %4 %5 %6
                %8:u64[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                    parallel_reduce [kind=min, axis_name=\"x\", mesh=['x'=2:manual]] %7
            in (%8)"
        }
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax() {
        // Equal maxima use the lowest position. A NaN extremum has no equal participant.
        let rows = Array::matrix(3, 3, vec![3f32, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0, 5.0, f32::NAN]).unwrap();
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("rows"), rows),
            Ok(Array::vector(vec![0u64, 1, u64::MAX]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_unbound_axis() {
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("other"), Array::vector(vec![1f32, 2.0]).unwrap()),
            Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "other".to_string() })),
        );
        assert_eq!(
            Array::scalar(1f32).unwrap().parallel_argmax("rows"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "rows".to_string() })),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_boolean() {
        let rows = Array::matrix(2, 2, vec![false, true, true, false]).unwrap();
        assert_eq!(batch_rows(|row| row.parallel_argmax("rows"), rows), Ok(Array::vector(vec![1u64, 0]).unwrap()));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_exact_integers() {
        // These adjacent values must remain distinct above the exact integer range of f64.
        let rows = Array::matrix(2, 2, vec![u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX - 1]).unwrap();
        assert_eq!(batch_rows(|row| row.parallel_argmax("rows"), rows), Ok(Array::vector(vec![1u64, 0]).unwrap()));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_infinities_and_signed_zero() {
        let rows = Array::matrix(2, 3, vec![f32::NEG_INFINITY, -0.0, f32::INFINITY, f32::INFINITY, 0.0, f32::INFINITY])
            .unwrap();
        assert_eq!(batch_rows(|row| row.parallel_argmax("rows"), rows), Ok(Array::vector(vec![1u64, 0, 0]).unwrap()));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_degenerate_batches() {
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("rows"), Array::vector(vec![3f32]).unwrap()),
            Ok(Array::scalar(0u64).unwrap()),
        );
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("rows"), Array::matrix(2, 0, Vec::<f32>::new()).unwrap()),
            Ok(Array::vector(Vec::<u64>::new()).unwrap()),
        );
        // With no participants, the minimum of the empty candidate set is the sentinel.
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("rows"), Array::vector(Vec::<f32>::new()).unwrap()),
            Ok(Array::scalar(u64::MAX).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_host_memory() {
        let memory = Memory::Host { pinned: true };
        let rows = Array::matrix(2, 2, vec![1f32, 4.0, 2.0, 3.0]).unwrap().transfer_to_memory(memory).unwrap();
        assert_eq!(
            batch_rows(|row| row.parallel_argmax("rows"), rows),
            Ok(Array::vector(vec![1u64, 0]).unwrap().transfer_to_memory(memory).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_replicated_input() {
        // Replicated values tie on every participant, so their winning position is zero.
        assert_eq!(
            batch(
                |row: TestTracer| row.parallel_argmax("rows"),
                Array::vector(vec![3f32, 5.0]).unwrap(),
                BatchAxis::replicated(),
                BatchAxis::replicated(),
                BatchAxisSpecification::new(3, "rows"),
            ),
            Ok(Array::vector(vec![0u64, 0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_nested_batching() {
        // The inner column batch forwards the outer row name without consuming its own axis.
        let rows = Array::matrix(3, 2, vec![1f32, 9.0, 7.0, 2.0, 7.0, 4.0]).unwrap();
        assert_eq!(
            batch_rows(
                |row| {
                    Ok(batch(
                        |element: BatchingTracer<
                            BatchingContext<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>,
                            ArrayBatchingPolicy,
                        >| element.parallel_argmax("rows"),
                        row,
                        BatchAxis::new(0),
                        BatchAxis::new(0),
                        BatchAxisSpecification::named("columns"),
                    )?)
                },
                rows,
            ),
            Ok(Array::vector(vec![1u64, 0]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_manual_mesh_axis() {
        let (mesh, input_type, axes) = manual_axis_fixture();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            input_type,
            axes,
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayType::new_static(DataType::U64, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
        );
        assert_eq!(program.to_string(), manual_argmax_rendering());
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_manual_mesh_axis_host_memory() {
        let (mesh, input_type, axes) = manual_axis_fixture();
        let memory = Memory::Host { pinned: true };
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            input_type.with_memory(memory),
            axes,
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayType::new_static(DataType::U64, [2])
                .with_memory(memory)
                .with_sharding(Sharding::replicated(mesh, 1))
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]@Host[Pinned] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}]@Host[Pinned] = \
                        parallel_reduce [kind=max, axis_name=\"x\", mesh=['x'=2:manual]] %0
                    %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]@Host[Pinned] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]@Host[Pinned] = \
                        compare [direction=Equal] %2 %0
                    %4:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                        axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                    %5:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]@Host[Pinned] = \
                        transfer_to_memory [destination=Host[Pinned]] %4
                    %6:u64[]@Host[Pinned] = constant [value=18446744073709551615]
                    %7:u64[][sharding={mesh<['x'=2:manual]>, []}]@Host[Pinned] = broadcast [
                        output_type=u64[][sharding={mesh<['x'=2:manual]>, []}]@Host[Pinned],
                        output_axes=[],
                    ] %6
                    %8:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}]@Host[Pinned] = \
                        parallel_vary [axis_name=\"x\"] %7
                    %9:u64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}]@Host[Pinned] = \
                            select %3 %5 %8
                    %10:u64[2][sharding={mesh<['x'=2:manual]>, [{}]}]@Host[Pinned] = \
                        parallel_reduce [kind=min, axis_name=\"x\", mesh=['x'=2:manual]] %9
                in (%10)"
            },
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_unchecked_manual_mesh_axis() {
        let (mesh, _, _) = manual_axis_fixture();
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            ArrayType::scalar(DataType::F32),
            vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: false })],
        )
        .unwrap();
        assert_eq!(output, ArrayType::scalar(DataType::U64));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = parallel_reduce [kind=max, axis_name=\"x\"] %0
                    %2:bool[] = compare [direction=Equal] %1 %0
                    %3:u64[] = axis_index [axis_name=\"x\"]
                    %4:u64[] = constant [value=18446744073709551615]
                    %5:u64[] = select %2 %3 %4
                    %6:u64[] = parallel_reduce [kind=min, axis_name=\"x\"] %5
                in (%6)"
            },
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_invalid_reduction_state() {
        let (mesh, input_type, axes) = manual_axis_fixture();
        let reduced = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_reduced_axes(["x"]).unwrap())
            .unwrap();
        assert!(matches!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_argmax("x"), reduced, axes.clone(),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{PARALLEL_REDUCE_OPERATION_NAME}` axis `x` must not carry reduced state"),
        ));
        let unreduced = input_type
            .with_sharding(
                Sharding::replicated(mesh, 1)
                    .with_unreduced_axes(["x"])
                    .unwrap()
                    .with_unreduced_kind(UnreducedKind::Max)
                    .unwrap(),
            )
            .unwrap();
        // Completing the maximum is insufficient because comparing it with a local partial value is invalid.
        assert!(matches!(
            TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
                |input| input.parallel_argmax("x"), unreduced, axes,
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == format!("`{COMPARE_OPERATION_NAME}` does not support unreduced inputs"),
        ));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_partial_evaluation() {
        let (mesh, input_type, axes) = manual_axis_fixture();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            input_type.clone(),
            axes,
        )
        .unwrap();
        assert_eq!(program.to_string(), manual_argmax_rendering());
        let sentinel = Array::from_elements(
            ArrayType::scalar(DataType::U64).with_sharding(Sharding::replicated(mesh, 0)).unwrap(),
            &[u64::MAX],
        )
        .unwrap();
        let evaluation =
            program.to_flat_program().partially_evaluate(&[PartialValue::Unknown(input_type.clone())]).unwrap();
        assert_eq!(
            evaluation.inputs(),
            &[PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(sentinel.clone())],
        );
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(evaluation.program().to_string(), residual_argmax_rendering());

        // Concrete local data still cannot determine values owned by the other devices.
        let input = Array::from_elements(input_type, &[3f32, 1.0]).unwrap();
        let evaluation = program.to_flat_program().partially_evaluate(&[PartialValue::Known(input.clone())]).unwrap();
        assert_eq!(
            evaluation.inputs(),
            &[PartialEvaluationInput::Known(input), PartialEvaluationInput::Known(sentinel)],
        );
        assert_eq!(evaluation.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(evaluation.program().to_string(), residual_argmax_rendering());
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_differentiation() {
        let (_, input_type, axes) = manual_axis_fixture();
        let (_, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            input_type,
            axes,
        )
        .unwrap();
        assert_eq!(program.to_string(), manual_argmax_rendering());
        assert!(matches!(
            program.to_flat_program().jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == format!("`{PARALLEL_REDUCE_OPERATION_NAME}` with kind `max` is not differentiable"),
        ));
        assert_eq!(program.to_flat_program().jvp_with_respect_to(&[]).unwrap().to_string(), manual_argmax_rendering());
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin() {
        let rows = Array::matrix(3, 3, vec![3f32, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0, 5.0, f32::NAN]).unwrap();
        assert_eq!(
            batch_rows(|row| row.parallel_argmin("rows"), rows),
            Ok(Array::vector(vec![2u64, 0, u64::MAX]).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_unbound_axis() {
        assert_eq!(
            batch_rows(|row| row.parallel_argmin("other"), Array::vector(vec![1f32, 2.0]).unwrap()),
            Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "other".to_string() })),
        );
        assert_eq!(
            Array::scalar(1f32).unwrap().parallel_argmin("rows"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "rows".to_string() })),
        );
        assert_eq!(
            ArrayIrValue::Array(Array::scalar(1f32).unwrap()).parallel_argmin("rows"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "rows".to_string() })),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_exact_integers() {
        let rows = Array::matrix(2, 2, vec![i64::MIN + 1, i64::MIN, i64::MIN, i64::MIN + 1]).unwrap();
        assert_eq!(batch_rows(|row| row.parallel_argmin("rows"), rows), Ok(Array::vector(vec![1u64, 0]).unwrap()));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_complex() {
        let rows = Array::matrix(
            2,
            2,
            vec![Complex::new(1f32, 2.0), Complex::new(2.0, -9.0), Complex::new(1.0, 1.0), Complex::new(1.0, 9.0)],
        )
        .unwrap();
        assert_eq!(batch_rows(|row| row.parallel_argmin("rows"), rows), Ok(Array::vector(vec![1u64, 1]).unwrap()));
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_array_ir_non_array() {
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.parallel_argmin("rows"),
            Err(TypeError::invalid("expected array type but got dimension type").into()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_host_memory() {
        let memory = Memory::Host { pinned: false };
        let rows = Array::matrix(2, 2, vec![1f32, 4.0, 2.0, 3.0]).unwrap().transfer_to_memory(memory).unwrap();
        assert_eq!(
            batch_rows(|row| row.parallel_argmin("rows"), rows),
            Ok(Array::vector(vec![0u64, 1]).unwrap().transfer_to_memory(memory).unwrap()),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin_array_ir_manual_mesh_axis() {
        let (mesh, input_type, axes) = manual_axis_fixture();
        let (output, program) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmin("x"),
            ArrayIrType::from(input_type),
            axes,
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayIrType::from(
                ArrayType::new_static(DataType::U64, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
            ),
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] .
                let %1:f32[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                        parallel_reduce [kind=min, axis_name=\"x\", mesh=['x'=2:manual]] %0
                    %2:f32[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %1
                    %3:bool[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = \
                        compare [direction=Equal] %2 %0
                    %4:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                        axis_index [axis_name=\"x\", mesh=['x'=2:manual]]
                    %5:u64[] = constant [value=18446744073709551615]
                    %6:u64[][sharding={mesh<['x'=2:manual]>, []}] = \
                        broadcast [output_type=u64[][sharding={mesh<['x'=2:manual]>, []}], output_axes=[]] %5
                    %7:u64[][sharding={mesh<['x'=2:manual]>, [], varying_manual={'x'}}] = \
                        parallel_vary [axis_name=\"x\"] %6
                    %8:u64[2][sharding={mesh<['x'=2:manual]>, [{}], varying_manual={'x'}}] = select %3 %4 %7
                    %9:u64[2][sharding={mesh<['x'=2:manual]>, [{}]}] = \
                        parallel_reduce [kind=min, axis_name=\"x\", mesh=['x'=2:manual]] %8
                in (%9)"
            },
        );
    }
}
