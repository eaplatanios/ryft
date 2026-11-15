use ryft_macros::capability;

use crate::arrays::{ArrayIrType, ArrayType, DataType, Dimension, Shape, Sharding};
use crate::operations::Capability;
use crate::operations::constants::iota::{DynamicIota, Iota};
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::manipulation::gathering::{Gather, GatherDimensionNumbers, GatherOptions};
use crate::operations::manipulation::reshaping::Reshape;
use crate::operations::manipulation::slicing::Slice;
use crate::programs::{ProgramError, TypeError, Value, ValueProjection};

/// Value-level capability for taking a diagonal of an array along two of its axes. For a non-negative `offset`,
/// element `k` of the diagonal is the element at index `k` along the first axis and index `k + offset` along the
/// second axis. For a negative `offset`, it is the element at index `k - offset` along the first axis and index `k`
/// along the second axis. The diagonal has as many elements as there are such index pairs, which may be none. It takes
/// the place of the first axis in the result and the second axis is removed, so every other axis keeps its position
/// relative to the diagonal. NumPy's [`diagonal`](https://numpy.org/doc/stable/reference/generated/numpy.diagonal.html)
/// selects the same elements but moves the diagonal to the last axis, which is one [`Transpose`](crate::Transpose)
/// of this result.
///
/// [`Diagonal`] is not a primitive operation. [`Self::diagonal`] first slices the two axes to the extent of the
/// diagonal, unless both already have that extent, and then reads the diagonal with one [`Gather`] whose indices are
/// an [`iota`](crate::IotaOperation) along the diagonal. Every program transform therefore supports it through the
/// rules of [`SliceOperation`](crate::SliceOperation) and [`GatherOperation`](crate::GatherOperation), and elements
/// are copied exactly (e.g., signed zeros and `NaN`s are preserved). The gather reads one element for every element of
/// the result, unlike a mask over the whole input followed by a reduction. A diagonal that is statically known to be
/// empty reads nothing, so it is the empty slice reshaped to the shape of the result.
///
/// Array IR values read their dynamic extents from the value itself, so one staged diagonal serves every extent.
/// The two diagonal axes may be dynamic only when they share one extent and the offset is zero, because every other
/// diagonal needs a slice with a statically known extent, and an empty diagonal needs every extent to be static.
/// Homogeneous array values construct their indices from host extents, so all of their extents must be static.
///
/// The universe parameter `T` defaults to the [`Capability`] universe of the implementor, so that homogeneous array
/// values implement this capability for [`ArrayType`] and array IR values implement it for [`ArrayIrType`].
///
/// # Example
///
/// ```rust
/// # use ryft_core::{Array, Diagonal, ProgramError};
/// # fn main() -> Result<(), ProgramError> {
/// let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0])?;
/// assert_eq!(matrix.diagonal(0, 0, 1)?, Array::vector(vec![1.0, 5.0])?);
/// assert_eq!(matrix.diagonal(1, 0, 1)?, Array::vector(vec![2.0, 6.0])?);
/// assert_eq!(matrix.diagonal(-1, 0, 1)?, Array::vector(vec![4.0])?);
/// # Ok(())
/// # }
/// ```
#[capability]
pub trait Diagonal<T = <Self as Capability>::Universe>: Capability + Sized {
    /// Returns the diagonal of `self` along `axis1` and `axis2` with the provided `offset`. The result has the shape
    /// of `self` without `axis2` and with the extent of the diagonal along `axis1`, so `axis1` moves one position
    /// earlier when it follows `axis2`. Refer to the documentation of this trait for more information.
    ///
    /// # Parameters
    ///
    ///   - `offset`: Offset of the diagonal from the main diagonal. A positive offset selects a diagonal with larger
    ///     indices along `axis2`, and a negative offset selects one with larger indices along `axis1`.
    ///   - `axis1`: Axis that the diagonal takes the place of.
    ///   - `axis2`: Axis that is removed.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when either axis is out of range, when the two axes are the same, when an extent
    /// that must be static is dynamic, or when executing or staging the composition fails.
    fn diagonal(&self, offset: isize, axis1: usize, axis2: usize) -> Result<Self, ProgramError>;
}

impl<V: Value<Type = ArrayType, Domain: Iota<V>> + Gather + Slice + Reshape> Diagonal<ArrayType> for V {
    fn diagonal(&self, offset: isize, axis1: usize, axis2: usize) -> Result<Self, ProgramError> {
        // Homogeneous array contexts construct indices from host extents only, so every extent must be static. The
        // indices are built through the `Iota` capability of the domain of the value, which materializes them for
        // eager arrays and stages an `IotaOperation` for values that carry a context.
        let r#type = self.r#type();
        let layout = diagonal_layout(r#type.as_ref(), offset, axis1, axis2)?;
        if layout.indices_type.shape().dimensions().iter().any(|dimension| dimension.value().is_none()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot take the diagonal of a value of type `{type}` with dynamic extents \
                     in a homogeneous array context",
                ),
            });
        }
        let value = match &layout.slice {
            Some((start_indices, limits)) => self.slice(start_indices, limits, &vec![1; limits.len()])?,
            None => self.clone(),
        };
        if let Some(empty_shape) = layout.empty_shape {
            return value.reshape(empty_shape);
        }
        let indices = self.domain().iota(&layout.indices_type, layout.iota_dimension)?;
        value.gather(&indices, &layout.dimensions, &layout.slice_sizes, &GatherOptions::new())
    }
}

impl<V: Value<Type = ArrayIrType, Domain: DynamicIota<V>> + DimensionSize<Extent = V> + ValueProjection<ArrayType>>
    Diagonal<ArrayIrType> for V
where
    V::Projected: Gather + Slice + Reshape,
{
    fn diagonal(&self, offset: isize, axis1: usize, axis2: usize) -> Result<Self, ProgramError> {
        let r#type = self.r#type();
        let r#type = <&ArrayType>::try_from(r#type.as_ref())?;
        let layout = diagonal_layout(r#type, offset, axis1, axis2)?;
        let value = <V as ValueProjection<ArrayType>>::into_projected(self.clone())?;
        let value = match &layout.slice {
            Some((start_indices, limits)) => value.slice(start_indices, limits, &vec![1; limits.len()])?,
            None => value,
        };
        if let Some(empty_shape) = layout.empty_shape {
            return Ok(<V as ValueProjection<ArrayType>>::from_projected(value.reshape(empty_shape)?));
        }

        // Dynamic extents are read from `self` itself, so the indices share their dimension identities with the
        // paired input axes. Static extents consume no dimension values. A slice only changes the two diagonal axes,
        // which are then static, so the extents of every other axis are those of `self`.
        let mut extents = Vec::new();
        for axis in (0..r#type.rank()).filter(|axis| *axis != axis2) {
            if r#type.dimension(axis).value().is_none() {
                extents.push(self.dimension_size(axis)?);
            }
        }
        let indices = self.domain().dynamic_iota(&layout.indices_type, layout.iota_dimension, &extents)?;
        let indices = <V as ValueProjection<ArrayType>>::into_projected(indices)?;
        let diagonal = value.gather(&indices, &layout.dimensions, &layout.slice_sizes, &GatherOptions::new())?;
        Ok(<V as ValueProjection<ArrayType>>::from_projected(diagonal))
    }
}

/// Slice and paired gather that take one diagonal of an array. Refer to [`diagonal_layout`] for more information.
struct DiagonalLayout {
    /// Start indices and limits of the slice that restricts the two diagonal axes to the extent of the diagonal, or
    /// [`None`] when both axes already have that extent.
    slice: Option<(Vec<usize>, Vec<Dimension>)>,

    /// Shape of the result when the diagonal is statically known to be empty, in which case the result is the sliced
    /// array reshaped to this shape and the gather is not used. A gather cannot read an empty diagonal, because its
    /// window along the removed axis must be one element.
    empty_shape: Option<Shape>,

    /// Type of the indices of the gather.
    indices_type: ArrayType,

    /// Axis of the indices along which they count.
    iota_dimension: usize,

    /// Dimension numbers of the gather.
    dimensions: GatherDimensionNumbers,

    /// Window sizes of the gather.
    slice_sizes: Vec<usize>,
}

/// Returns the [`DiagonalLayout`] that takes the diagonal of an array of type `type` along `axis1` and `axis2` with
/// the provided `offset` (refer to [`Diagonal::diagonal`]). The slice makes both axes start at the first element of
/// the diagonal and gives both the extent of the diagonal, so the diagonal of the sliced array is its main diagonal.
/// The indices of the gather then enumerate every coordinate of the sliced array except those of `axis2`, followed by
/// a one-component index vector, and count along the coordinates of `axis1`. Pairing every enumerated axis with the
/// axis that it enumerates (i.e., as gather batching dimensions) reads, for each coordinate, the element whose `axis2`
/// index equals its `axis1` index, with no window over a possibly dynamic extent.
fn diagonal_layout(
    r#type: &ArrayType,
    offset: isize,
    axis1: usize,
    axis2: usize,
) -> Result<DiagonalLayout, ProgramError> {
    let rank = r#type.rank();
    if axis1 >= rank || axis2 >= rank || axis1 == axis2 {
        return Err(TypeError::invalid(format!(
            "cannot take the diagonal of a value of type `{type}` along axes {axis1} and {axis2}, which must be two \
             distinct axes of its rank {rank}",
        ))
        .into());
    }

    // Two axes that share one extent already have the extent of their main diagonal, including a dynamic extent.
    // Every other diagonal is shorter than at least one of its axes, so both axes are sliced to it, which needs
    // their extents to be static.
    let mut dimensions = r#type.shape().dimensions().to_vec();
    let slice = if offset == 0 && dimensions[axis1] == dimensions[axis2] {
        None
    } else {
        let (Some(extent1), Some(extent2)) = (dimensions[axis1].value(), dimensions[axis2].value()) else {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot take the diagonal of a value of type `{type}` along axes {axis1} and {axis2} with offset \
                     {offset}, because dynamic extents require a zero offset and two axes that share one extent",
                ),
            });
        };
        let start1 = if offset < 0 { offset.unsigned_abs().min(extent1) } else { 0 };
        let start2 = if offset > 0 { offset.unsigned_abs().min(extent2) } else { 0 };
        let extent = (extent1 - start1).min(extent2 - start2);
        let mut start_indices = vec![0; rank];
        start_indices[axis1] = start1;
        start_indices[axis2] = start2;
        let mut limits = dimensions.clone();
        limits[axis1] = Dimension::Static(start1 + extent);
        limits[axis2] = Dimension::Static(start2 + extent);
        dimensions[axis1] = Dimension::Static(extent);
        dimensions[axis2] = Dimension::Static(extent);
        Some((start_indices, limits))
    };

    let paired_axes = (0..rank).filter(|axis| *axis != axis2).collect::<Vec<_>>();
    let mut indices_shape = paired_axes.iter().map(|axis| dimensions[*axis].clone()).collect::<Vec<_>>();
    indices_shape.push(Dimension::Static(1));
    let indices_type = ArrayType::new(DataType::I64, Shape::new(indices_shape));

    // The indices share the varying manual axes of the input. An iota holds the same elements on every device,
    // so typing it as varying is always valid, and an integer iota is a constant that has no cotangent for that
    // transition to affect. It lets values that insert no variation transitions (e.g., eager arrays) gather with it.
    let indices_type = match r#type.sharding() {
        Some(sharding) if !sharding.varying_manual_axes().is_empty() => {
            let indices_sharding = Sharding::replicated(sharding.mesh().clone(), rank)
                .with_varying_manual_axes(sharding.varying_manual_axes().clone())
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            indices_type
                .with_sharding(indices_sharding)
                .map_err(|error| TypeError::invalid(error.to_string()))?
        }
        _ => indices_type,
    };
    let iota_dimension = if axis2 < axis1 { axis1 - 1 } else { axis1 };
    let gather_dimensions = GatherDimensionNumbers::new(Vec::new(), vec![axis2], vec![axis2])
        .with_batching_dimensions(paired_axes.iter().enumerate().map(|(index, axis)| (*axis, index)).collect());

    // Paired axes use a zero window when they may be empty, exactly as `DynamicGather::dynamic_gather_axis` does.
    let slice_sizes = (0..rank)
        .map(|axis| if axis == axis2 { 1 } else { dimensions[axis].bounds().lower().min(1) })
        .collect();

    // An empty diagonal is reshaped instead of gathered, which needs the shape of the result to be static.
    let empty_shape = if dimensions[axis1].value() == Some(0) {
        if paired_axes.iter().any(|axis| dimensions[*axis].value().is_none()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "cannot take the diagonal of a value of type `{type}` along axes {axis1} and {axis2} with offset \
                     {offset}, because an empty diagonal requires every extent to be static",
                ),
            });
        }
        Some(Shape::new(paired_axes.iter().map(|axis| dimensions[*axis].clone()).collect()))
    } else {
        None
    };
    Ok(DiagonalLayout { slice, empty_shape, indices_type, iota_dimension, dimensions: gather_dimensions, slice_sizes })
}

#[cfg(test)]
mod tests {
    use indoc::{formatdoc, indoc};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrValue, ArrayOperation, Complex, DimensionBounds, DimensionValue,
        DimensionVariable, LogicalMesh, MeshAxis, MeshAxisType,
    };
    use crate::axes::NamedAxis;
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::macros::check_gradient;
    use crate::operations::arithmetic::Mul;
    use crate::operations::reductions::Reduce;
    use crate::tracing::{DomainTracer, DomainTracingContext, Trace};

    use super::*;

    type ArrayIrEagerContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    #[test]
    fn test_diagonal() {
        // The main diagonal of a square matrix is the same along either axis order.
        let matrix = Array::matrix(3, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();
        assert_eq!(matrix.diagonal(0, 0, 1), Ok(Array::vector(vec![1.0, 5.0, 9.0]).unwrap()));
        assert_eq!(matrix.diagonal(0, 1, 0), Ok(Array::vector(vec![1.0, 5.0, 9.0]).unwrap()));

        // A positive offset selects larger indices along the second axis and a negative offset selects larger
        // indices along the first axis, so swapping the axes mirrors the offset. A diagonal past the last column or
        // row is empty.
        assert_eq!(matrix.diagonal(1, 0, 1), Ok(Array::vector(vec![2.0, 6.0]).unwrap()));
        assert_eq!(matrix.diagonal(2, 0, 1), Ok(Array::vector(vec![3.0]).unwrap()));
        assert_eq!(matrix.diagonal(-1, 0, 1), Ok(Array::vector(vec![4.0, 8.0]).unwrap()));
        assert_eq!(matrix.diagonal(1, 1, 0), Ok(Array::vector(vec![4.0, 8.0]).unwrap()));
        assert_eq!(matrix.diagonal(3, 0, 1), Ok(Array::vector(Vec::<f64>::new()).unwrap()));
        assert_eq!(matrix.diagonal(-7, 0, 1), Ok(Array::vector(Vec::<f64>::new()).unwrap()));

        // The diagonal of an empty matrix is empty, and an empty diagonal keeps the other axes of its input.
        let empty = Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [0, 0]), &[]).unwrap();
        assert_eq!(empty.diagonal(0, 0, 1), Ok(Array::vector(Vec::<f64>::new()).unwrap()));
        let values = (0..12).map(f64::from).collect::<Vec<_>>();
        let cube = Array::from_elements(ArrayType::new_static(DataType::F64, [2, 3, 2]), &values).unwrap();
        assert_eq!(
            cube.diagonal(2, 2, 0),
            Ok(Array::from_elements::<f64>(ArrayType::new_static(DataType::F64, [3, 0]), &[]).unwrap()),
        );

        // Axes with different extents have a diagonal as long as the index pairs that exist.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(matrix.diagonal(0, 0, 1), Ok(Array::vector(vec![1.0, 5.0]).unwrap()));
        assert_eq!(matrix.diagonal(1, 0, 1), Ok(Array::vector(vec![2.0, 6.0]).unwrap()));
        assert_eq!(matrix.diagonal(2, 0, 1), Ok(Array::vector(vec![3.0]).unwrap()));
        assert_eq!(matrix.diagonal(-1, 0, 1), Ok(Array::vector(vec![4.0]).unwrap()));
        assert_eq!(matrix.diagonal(0, 1, 0), Ok(Array::vector(vec![1.0, 5.0]).unwrap()));

        // The diagonal takes the place of the first axis and the other axes keep their positions, so the first axis
        // moves one position earlier when it follows the second axis. Element `(i, j, k)` of the input is
        // `6i + 3j + k`, so element `(i, j)` of the first result and element `(j, i)` of the second are `7i + 3j`,
        // and element `(i, j)` of the third is `7i + 3j + 1`.
        let values = (0..18).map(f64::from).collect::<Vec<_>>();
        let cube = Array::from_elements(ArrayType::new_static(DataType::F64, [3, 2, 3]), &values).unwrap();
        assert_eq!(cube.diagonal(0, 0, 2), Ok(Array::matrix(3, 2, vec![0.0, 3.0, 7.0, 10.0, 14.0, 17.0]).unwrap()));
        assert_eq!(cube.diagonal(0, 2, 0), Ok(Array::matrix(2, 3, vec![0.0, 7.0, 14.0, 3.0, 10.0, 17.0]).unwrap()));
        assert_eq!(cube.diagonal(1, 0, 2), Ok(Array::matrix(2, 2, vec![1.0, 4.0, 8.0, 11.0]).unwrap()));

        // Elements are copied exactly, including signed zeros, `NaN`s, and complex values.
        let special = Array::matrix(2, 2, vec![-0.0, 1.0, 2.0, f64::NAN]).unwrap().diagonal(0, 0, 1).unwrap();
        let special = special.elements::<f64>().unwrap();
        assert!(special[0] == 0.0 && special[0].is_sign_negative() && special[1].is_nan());
        let complex = Array::matrix(
            2,
            2,
            vec![Complex::new(1.0f64, -1.0), Complex::new(2.0, 0.0), Complex::new(3.0, 0.0), Complex::new(4.0, 4.0)],
        )
        .unwrap();
        assert_eq!(
            complex.diagonal(0, 0, 1),
            Ok(Array::vector(vec![Complex::new(1.0f64, -1.0), Complex::new(4.0, 4.0)]).unwrap()),
        );

        // The axes must be two distinct axes within the rank of the input.
        assert_eq!(
            matrix.diagonal(0, 1, 1),
            Err(ProgramError::Type(TypeError::invalid(
                "cannot take the diagonal of a value of type `f64[2, 3]` along axes 1 and 1, which must be two \
                 distinct axes of its rank 2",
            ))),
        );
        assert_eq!(
            matrix.diagonal(1, 0, 2),
            Err(ProgramError::Type(TypeError::invalid(
                "cannot take the diagonal of a value of type `f64[2, 3]` along axes 0 and 2, which must be two \
                 distinct axes of its rank 2",
            ))),
        );

        // An array type is its own value where only types are computed, so its diagonal is the type of the result.
        let r#type = ArrayType::new_static(DataType::F64, [2, 3, 4]);
        assert_eq!(r#type.diagonal(0, 1, 2), Ok(ArrayType::new_static(DataType::F64, [2, 3])));
        assert_eq!(r#type.diagonal(2, 2, 1), Ok(ArrayType::new_static(DataType::F64, [2, 1])));
        assert_eq!(r#type.diagonal(-1, 0, 2), Ok(ArrayType::new_static(DataType::F64, [1, 3])));
    }

    #[test]
    fn test_diagonal_staging() {
        // The main diagonal of two axes that share one extent stages one gather whose indices count along the
        // diagonal.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| x.diagonal(0, 0, 1),
            ArrayType::new_static(DataType::F64, [3, 3]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f64[3, 3] .
                let %1:i64[3, 1] = iota [type=i64[3, 1], dimension=0]
                    %2:f64[3] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 1],
                    ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Every other diagonal first slices both axes to its extent, starting at its first element.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| x.diagonal(1, 0, 1),
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[2, 2] = slice [start_indices=[0, 1], limits=[2, 3]] %0
                    %2:i64[2, 1] = iota [type=i64[2, 1], dimension=0]
                    %3:f64[2] = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 1],
                    ] %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // An empty diagonal reads nothing, so it stages the empty slice reshaped to the shape of the result.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| x.diagonal(-2, 0, 1),
            ArrayType::new_static(DataType::F64, [2, 3]),
        )
        .unwrap();
        assert_eq!(
            program.to_flat_program().to_string(),
            indoc! {"
                lambda %0:f64[2, 3] .
                let %1:f64[0, 0] = slice [start_indices=[2, 0], limits=[2, 0]] %0
                    %2:f64[0] = reshape [shape=[0]] %1
                in (%2)
            "}
            .trim_end(),
        );

        // A homogeneous context constructs its indices from host extents, so it rejects dynamic extents.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let result = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x: DomainTracer<EagerContext<Array, ArrayOperation<Array>>>| x.diagonal(0, 0, 1),
            ArrayType::new(DataType::F64, Shape::new(vec![extent.clone(), extent])),
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot take the diagonal of a value of type `f64[n, n]` with dynamic extents in a \
                          homogeneous array context"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_diagonal_manual_variation() {
        // The indices share the varying manual axes of the input, so eager arrays (which insert no variation
        // transitions) take diagonals of varying values, and traced values stage no `parallel_vary` for the indices.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let varying = |rank: usize| Sharding::replicated(mesh.clone(), rank).with_varying_manual_axes(["x"]).unwrap();
        let input_type = ArrayType::new_static(DataType::F64, [2, 2]).with_sharding(varying(2)).unwrap();
        let output_type = ArrayType::new_static(DataType::F64, [2]).with_sharding(varying(1)).unwrap();
        let indices_type = ArrayType::new_static(DataType::I64, [2, 1]).with_sharding(varying(2)).unwrap();
        assert_eq!(
            Array::from_elements(input_type.clone(), &[1.0, 2.0, 3.0, 4.0]).unwrap().diagonal(0, 0, 1),
            Ok(Array::from_elements(output_type.clone(), &[1.0, 4.0]).unwrap()),
        );
        let (outputs, program) =
            DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::trace_with_named_axes(
                |inputs| Ok(vec![inputs[0].diagonal(0, 0, 1)?]),
                vec![input_type.clone()],
                vec![("x".to_string(), NamedAxis::Mesh { mesh, axis: 0, size: 2, checks_manual_variation: true })],
            )
            .unwrap();
        assert_eq!(outputs, vec![output_type.clone()]);
        assert_eq!(
            program.to_string(),
            formatdoc! {"
                lambda %0:{input_type} .
                let %1:{indices_type} = iota [
                    type={indices_type},
                    dimension=0,
                ]
                    %2:{output_type} = gather [
                        dimensions=(offset=[], collapsed_slice=[1], start_index_map=[1], batching=[(0, 0)]),
                        slice_sizes=[1, 1],
                    ] %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_diagonal_batching() {
        // Each batch item takes its own diagonal, with the axes of the item shifted past the mapped axis. The two
        // batch items are `[[1, 2], [3, 4]]` and `[[5, 6], [7, 8]]`.
        let items = Array::from_elements(
            ArrayType::new_static(DataType::F64, [2, 2, 2]),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        )
        .unwrap();
        assert_eq!(
            batch(|item| item.diagonal(0, 0, 1), items.clone(), BatchAxis::new(0), BatchAxis::new(0), None),
            Ok(Array::matrix(2, 2, vec![1.0, 4.0, 5.0, 8.0]).unwrap()),
        );
        assert_eq!(
            batch(|item| item.diagonal(1, 0, 1), items.clone(), BatchAxis::new(0), BatchAxis::new(0), None),
            Ok(Array::matrix(2, 1, vec![2.0, 6.0]).unwrap()),
        );

        // Mapping the last axis instead makes the two batch items `[[1, 3], [5, 7]]` and `[[2, 4], [6, 8]]`.
        assert_eq!(
            batch(|item| item.diagonal(0, 0, 1), items, BatchAxis::new(2), BatchAxis::new(0), None),
            Ok(Array::matrix(2, 2, vec![1.0, 7.0, 2.0, 8.0]).unwrap()),
        );
    }

    #[test]
    fn test_diagonal_differentiation() {
        // A diagonal is linear, so its tangent is the diagonal of the input tangent.
        let primal = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let tangent = Array::matrix(2, 3, vec![0.5, -1.0, 2.0, 3.0, -4.0, 0.25]).unwrap();
        assert_eq!(
            differentiate_at(primal.clone()).jvp(tangent, |input| input.diagonal(1, 0, 1)),
            Ok((Array::vector(vec![2.0, 6.0]).unwrap(), Array::vector(vec![-1.0, 0.25]).unwrap())),
        );

        // Reverse mode scatters each output cotangent to the input element that it was read from. The weighted sum
        // `10 · x[0, 1] + 100 · x[1, 2]` of the diagonal above has those two weights as its only nonzero partials.
        check_gradient!(
            |input| {
                let weights = input.domain().lift(Array::vector(vec![10.0, 100.0])?)?;
                input.diagonal(1, 0, 1)?.mul(&weights)?.reduce_sum(&[0], None)
            },
            at = primal,
            step = 1e-5,
            tolerance = 1e-7,
        );
    }

    #[test]
    fn test_diagonal_array_ir() {
        // Dynamic extents are read from the value, so one staged diagonal serves every extent, and static extents
        // consume no dimension values.
        let extent = Dimension::Dynamic(DimensionVariable::new("n", DimensionBounds::new(1, Some(8)).unwrap()));
        let r#type = ArrayType::new(DataType::F32, Shape::new(vec![extent.clone(), Dimension::Static(2), extent]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |value| value.diagonal(0, 2, 0),
            ArrayIrType::from(r#type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[n, 2, n] .
                let %1:dimension<n ∈ [1, 8)> = dimension_size [axis=2] %0
                    %2:i64[2, n, 1] = iota [type=i64[2, n, 1], dimension=1] %1
                    %3:f32[2, n] = gather [
                        dimensions=(offset=[], collapsed_slice=[0], start_index_map=[0], batching=[(1, 0), (2, 1)]),
                        slice_sizes=[1, 1, 1],
                    ] %0 %2
                in (%3)
            "}
            .trim_end(),
        );
        let values = (0..18).map(|value| value as f32).collect::<Vec<_>>();
        let cube = Array::from_elements(ArrayType::new_static(DataType::F32, [3, 2, 3]), &values).unwrap();
        assert_eq!(
            program.interpret(ArrayIrValue::Array(cube.clone())),
            Ok(ArrayIrValue::Array(Array::matrix(2, 3, vec![0.0f32, 7.0, 14.0, 3.0, 10.0, 17.0]).unwrap())),
        );

        // A diagonal that needs a slice keeps the dynamic extents of the other axes, which the slice leaves whole.
        let other = Dimension::Dynamic(DimensionVariable::new("m", DimensionBounds::new(1, Some(8)).unwrap()));
        let sliced_type =
            ArrayType::new(DataType::F32, Shape::new(vec![other, Dimension::Static(2), Dimension::Static(3)]));
        let (_, program) = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |value| value.diagonal(1, 1, 2),
            ArrayIrType::from(sliced_type.clone()),
        )
        .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f32[m, 2, 3] .
                let %1:f32[m, 2, 2] = slice [start_indices=[0, 0, 1], limits=[m, 2, 3]] %0
                    %2:dimension<m ∈ [1, 8)> = dimension_size [axis=0] %0
                    %3:i64[m, 2, 1] = iota [type=i64[m, 2, 1], dimension=1] %2
                    %4:f32[m, 2] = gather [
                        dimensions=(offset=[], collapsed_slice=[2], start_index_map=[2], batching=[(0, 0), (1, 1)]),
                        slice_sizes=[1, 1, 1],
                    ] %1 %3
                in (%4)
            "}
            .trim_end(),
        );
        assert_eq!(
            program.interpret(ArrayIrValue::Array(cube.clone())),
            Ok(ArrayIrValue::Array(Array::matrix(3, 2, vec![1.0f32, 5.0, 7.0, 11.0, 13.0, 17.0]).unwrap())),
        );

        // An empty diagonal is reshaped, so it needs every extent to be static.
        let result = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |value| value.diagonal(3, 1, 2),
            ArrayIrType::from(sliced_type),
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot take the diagonal of a value of type `f32[m, 2, 3]` along axes 1 and 2 with offset \
                          3, because an empty diagonal requires every extent to be static"
                    .to_string(),
            }),
        );

        // The two diagonal axes may be dynamic only for the main diagonal of two axes that share one extent.
        let result = DomainTracingContext::<ArrayIrEagerContext>::trace(
            |value| value.diagonal(1, 2, 0),
            ArrayIrType::from(r#type),
        );
        assert_eq!(
            result.map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "cannot take the diagonal of a value of type `f32[n, 2, n]` along axes 2 and 0 with offset \
                          1, because dynamic extents require a zero offset and two axes that share one extent"
                    .to_string(),
            }),
        );

        // Concrete array IR values take the diagonals of their array members, while first-class dimension members
        // are rejected.
        assert_eq!(
            ArrayIrValue::Array(cube).diagonal(1, 0, 2),
            Ok(ArrayIrValue::Array(Array::matrix(2, 2, vec![1.0f32, 4.0, 8.0, 11.0]).unwrap())),
        );
        let singleton = ArrayIrValue::Array(Array::matrix(1, 1, vec![3.0f32]).unwrap());
        assert_eq!(singleton.diagonal(0, 0, 1), Ok(ArrayIrValue::Array(Array::vector(vec![3.0f32]).unwrap())));
        let dimension = ArrayIrValue::<Array>::Dimension(DimensionValue::constant(2).unwrap());
        assert_eq!(
            dimension.diagonal(0, 0, 1),
            Err(ProgramError::Type(TypeError::invalid("expected array type but got dimension type"))),
        );
    }
}
