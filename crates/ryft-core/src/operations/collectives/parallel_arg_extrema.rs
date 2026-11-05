use ryft_macros::capability;

use crate::arrays::{Array, ArrayIrType, ArrayType};
use crate::axes::AxisError;
use crate::contexts::Context;
use crate::macros::check_count;
use crate::operations::Capability;
use crate::operations::collectives::axis_index::{AxisIndex, AxisIndexOperation};
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
/// Ryft analogues of JAX's `pargmax` and `pargmin`, and they are composed exactly as JAX composes them: the extremum
/// is reduced over the axis with [`parallel_reduce`](ParallelReduce::parallel_reduce), every participant compares it
/// with its own value, the participants whose value equals the extremum contribute their
/// [`axis_index`](AxisIndex::axis_index) while the others contribute [`u64::MAX`], and the minimum of those
/// contributions is reduced over the axis. Ties therefore resolve to the lowest position. A NaN extremum equals no
/// value and yields [`u64::MAX`] (JAX yields the maximum `int32` instead, because its positions are `int32` values
/// while Ryft's are `u64` values), but whether a NaN wins a cross-device extremum is backend-defined (e.g., the CPU
/// all-reduce of XLA drops it; refer to [`ParallelReduce`] for more information). The positions are not differentiable,
/// and neither are their components.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// The batch items of a named `batch` level are the participants of its axis, so positions are found eagerly:
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
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing binder, or if
    /// [`ReductionKind::Max`] does not support the element data type of this value.
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
    /// Returns a [`ProgramError`] if `axis_name` is not bound by an enclosing binder, or if
    /// [`ReductionKind::Min`] does not support the element data type of this value.
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
            Domain: Context<Operation: From<ConstantOperation<Array>> + From<AxisIndexOperation>> + AxisIndex,
        > + ParallelReduce
        + Compare
        + Select,
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

/// Returns the lowest position along `axis_name` of a participant whose value is the `kind` extremum of `value`, as
/// JAX's `_axis_index_of_val` composes it. Refer to [`ParallelArgExtrema`] for more information.
fn parallel_arg_extremum<
    V: Value<
            Dispatch = ValueDomainDispatch,
            Type = ArrayType,
            Domain: Context<Operation: From<ConstantOperation<Array>> + From<AxisIndexOperation>> + AxisIndex,
        > + ParallelReduce
        + Compare
        + Select,
>(
    value: &V,
    axis_name: &str,
    kind: ReductionKind,
) -> Result<V, ProgramError> {
    let context = value.domain();
    let extremum = value.parallel_reduce(kind, axis_name)?;
    let attains_extremum = extremum.equal(value)?;
    let position = context.axis_index(axis_name)?;
    let sentinel = Array::scalar(u64::MAX)?.transfer_to_memory(value.r#type().memory())?;
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
        ArrayBatchingPolicy, ArrayIrOperation, ArrayIrValue, ArrayOperation, DataType, LogicalMesh, MeshAxis,
        MeshAxisType, Sharding,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, BatchAxisSpecification, BatchingError, BatchingTracer, batch};
    use crate::contexts::EagerContext;
    use crate::tracing::TracingContext;

    use super::*;

    /// Batch item of an eager `batch` level over [`Array`]s.
    type Row = BatchingTracer<EagerContext<Array, ArrayOperation<Array>>, ArrayBatchingPolicy>;

    /// Applies `function` to the rows of `rows` under a `batch` level that binds `"rows"`.
    fn over_rows(
        function: impl Fn(&Row, &str) -> Result<Row, ProgramError>,
        rows: Array,
        axis_name: &str,
    ) -> Result<Array, BatchingError> {
        batch(
            |row: Row| function(&row, axis_name),
            rows,
            BatchAxis::new(0),
            BatchAxis::replicated(),
            BatchAxisSpecification::named("rows"),
        )
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax() {
        // Ties resolve to the lowest position, and a NaN maximum equals no value, so it yields `u64::MAX`.
        let rows = Array::matrix(3, 3, vec![3.0f32, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0, 5.0, f32::NAN]).unwrap();
        assert_eq!(
            over_rows(|row, axis_name| row.parallel_argmax(axis_name), rows.clone(), "rows"),
            Ok(Array::vector(vec![0u64, 1, u64::MAX]).unwrap()),
        );
        assert_eq!(
            over_rows(|row, axis_name| row.parallel_argmax(axis_name), rows, "other"),
            Err(BatchingError::Axis(AxisError::UnboundAxisName { name: "other".to_string() })),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().parallel_argmax("rows"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "rows".to_string() })),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmin() {
        let rows = Array::matrix(3, 3, vec![3.0f32, 1.0, 2.0, 3.0, 5.0, 0.0, 1.0, 5.0, f32::NAN]).unwrap();
        assert_eq!(
            over_rows(|row, axis_name| row.parallel_argmin(axis_name), rows, "rows"),
            Ok(Array::vector(vec![2u64, 0, u64::MAX]).unwrap()),
        );
        assert_eq!(
            Array::scalar(1.0f32).unwrap().parallel_argmin("rows"),
            Err(ProgramError::Axis(AxisError::UnboundAxisName { name: "rows".to_string() })),
        );
    }

    #[test]
    fn test_parallel_arg_extrema_parallel_argmax_manual_mesh_axis() {
        // Over a checked manual mesh axis, the composition stages the mesh-form reductions and the device position, and
        // the result is invariant over the axis.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = ArrayType::new_static(DataType::F32, [2])
            .with_sharding(Sharding::replicated(mesh.clone(), 1).with_varying_manual_axes(["x"]).unwrap())
            .unwrap();
        let axes = vec![(
            "x".to_string(),
            NamedAxis::Mesh { mesh: mesh.clone(), axis: 0, size: 2, checks_manual_variation: true },
        )];
        let (output, program) = TracingContext::<Array, ArrayOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmax("x"),
            varying.clone(),
            axes.clone(),
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayType::new_static(DataType::U64, [2])
                .with_sharding(Sharding::replicated(mesh.clone(), 1))
                .unwrap(),
        );
        assert_eq!(
            program.to_string(),
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
            },
        );

        // Array IR values find positions through the projection of the capability.
        let (output, _) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace_with_named_axes(
            |input| input.parallel_argmin("x"),
            ArrayIrType::from(varying),
            axes,
        )
        .unwrap();
        assert_eq!(
            output,
            ArrayIrType::from(
                ArrayType::new_static(DataType::U64, [2]).with_sharding(Sharding::replicated(mesh, 1)).unwrap(),
            ),
        );
    }
}
