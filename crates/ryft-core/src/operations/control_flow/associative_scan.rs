//! Parallel-prefix scans of arbitrary associative operators, built out of ordinary manipulation primitives. Refer to
//! [`associative_scan`] for the construction and its semantics.
//!
//! Unlike the control-flow [`ScanOperation`](crate::operations::control_flow::ScanOperation), which stages one
//! region-carrying instruction that runs its body sequentially, [`associative_scan`] is a function rather than an
//! operation. It stages the logarithmic-depth construction directly, so every transformation (e.g., batching,
//! differentiation, or partial evaluation) applies the rules of the primitives it stages.

// TODO(eaplatanios): Review this module.

use crate::arrays::{ArrayType, DataType};
use crate::axes::Axis;
use crate::contexts::Context;
use crate::operations::arithmetic::Add;
use crate::operations::constants::zero::Zero;
use crate::operations::logical::Or;
use crate::operations::manipulation::concatenation::Concatenate;
use crate::operations::manipulation::padding::Pad;
use crate::operations::manipulation::slicing::Slice;
use crate::parameters::Parameterized;
use crate::programs::{ProgramError, ProvenanceScope, TypeError, Typed, Value};

/// Returns the inclusive prefix scans of the arrays in `values` along `axis` under the associative operator `combine`,
/// built out of ordinary manipulation primitives instead of out of one
/// [`CumulativeOperation`](crate::operations::cumulative::CumulativeOperation).
///
/// This is Ryft's port of the log-depth Blelloch construction that JAX's
/// [`lax.associative_scan`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html) implements
/// (`jax/_src/lax/control_flow/loops.py`), and it exists here for the reason it is reached for there: a cumulative
/// operation whose combining operator is nonlinear has no closed-form primitive derivative, so the nonlinear
/// [`CumulativeKind`](crate::operations::cumulative::CumulativeKind)s define their forward mode by differentiating
/// *through* this decomposition rather than by carrying a bespoke gradient formula (JAX's `_cumulative_jvp_rule`). It
/// is also useful on its own for combining operators that no cumulative kind covers.
///
/// For an operator that a cumulative kind does cover (e.g., a running sum or maximum), prefer the
/// [`Cumulative`](crate::operations::cumulative::Cumulative) capability. It stages a single cumulative instruction
/// instead of this construction, which keeps programs small and leaves each backend free to choose its own lowering
/// (e.g., `ryft-xla` lowers forward cumulative sums to `chlo.scan` on GPUs). The two can also round floating-point
/// results differently in their last bits, because they associate the combinations differently.
///
/// Like JAX's, the scan runs over a whole structure of arrays at once: `values` is any [`Parameterized`] structure of
/// arrays (e.g., a single array, a tuple, a vector, or a derived structure), and `combine` receives and returns
/// structures shaped like it. This lets one scan carry several arrays that combine jointly, such as a running maximum
/// together with the position at which it is attained. The arrays are sliced and interleaved along `axis` in lockstep,
/// so they must all have the same extent along it, while their other dimensions and their data types can differ.
///
/// The recursion combines adjacent pairs along `axis`, scans the halved sequence recursively, combines the scanned
/// halves back against the elements the pairing skipped, and interleaves the two halves into the result. `combine`
/// always receives its operands in scan order (the accumulated prefix first), so the construction stays correct for
/// associative operators that are not commutative. A `reverse` scan mirrors the same recursion around the end of the
/// axis (the pairing simply starts one element in when the extent is odd) instead of reversing the operands before and
/// after a forward scan, which saves two array reversals per array and scan. Boolean arrays are interleaved with a
/// disjunction rather than an addition, because Booleans have no addition.
///
/// The scanned axis of every array must have a static extent, because the construction slices it at staging-time
/// positions, while every other axis can be dynamic (it is kept whole). A scanned axis shorter than two elements leaves
/// the arrays unchanged, and so does a structure that holds no arrays. A negative
/// `axis` counts from the end of the shape of the first array, and the resulting position is scanned in every array.
///
/// # Parameters
///
///   - `values`: [`Parameterized`] structure of the scanned arrays.
///   - `axis`: Scanned [`Axis`] of every array, normalized against the rank of the first array.
///   - `reverse`: Whether to accumulate from the end of the scanned axis toward its start.
///   - `combine`: Associative binary operator over structures shaped like `values`, receiving the accumulated prefix
///     and the next elements in scan order. It must return as many arrays as `values` holds.
///
/// # Errors
///
/// Returns a [`ProgramError`] if `axis` is out of bounds for any array, if the scanned extent of any array is not
/// static, if
/// the arrays have different extents along `axis`, if `combine` returns a different number of arrays, or if staging
/// any of the primitives of the construction (including those that `combine` stages) fails.
pub fn associative_scan<V, Values, A: Into<Axis>, F>(
    values: &Values,
    axis: A,
    reverse: bool,
    combine: &F,
) -> Result<Values, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Or + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    Values: Parameterized<V>,
    F: Fn(&Values, &Values) -> Result<Values, ProgramError>,
{
    let structure = values.parameter_structure();
    let arrays = values.parameters().cloned().collect::<Vec<_>>();
    let Some(first) = arrays.first() else {
        return Ok(Values::from_parameters(structure, arrays)?);
    };
    let axis = axis
        .into()
        .normalize(first.r#type().rank())
        .map_err(|error| TypeError::invalid(format!("`associative_scan` {error}")))?;
    let extents = arrays
        .iter()
        .map(|array| {
            let array_type = array.r#type();
            let rank = array_type.rank();
            if axis >= rank {
                return Err(TypeError::invalid(format!(
                    "`associative_scan` axis {axis} is out of bounds for rank {rank}",
                )));
            }
            array_type.dimension(axis).value().ok_or_else(|| {
                TypeError::invalid(format!(
                    "`associative_scan` requires a static extent along the scanned axis {axis} but got `{array_type}`",
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let extent = extents[0];
    if let Some(other) = extents.iter().find(|other| **other != extent) {
        return Err(TypeError::invalid(format!(
            "`associative_scan` requires operands with equal extents along axis {axis} but got {extent} and {other}",
        ))
        .into());
    }

    // The recursion runs over the flat arrays, so the combining operator is wrapped to rebuild its structured operands
    // on the way in and to flatten its structured result on the way out.
    let array_count = arrays.len();
    let flat_combine = |left: &[V], right: &[V]| -> Result<Vec<V>, ProgramError> {
        let left = Values::from_parameters(structure.clone(), left.iter().cloned())?;
        let right = Values::from_parameters(structure.clone(), right.iter().cloned())?;
        let combined = combine(&left, &right)?;
        let combined_count = combined.parameter_count();
        if combined_count != array_count {
            return Err(TypeError::invalid(format!(
                "`associative_scan` combining operator must return {array_count} arrays but returned {combined_count}",
            ))
            .into());
        }
        Ok(combined.into_parameters().collect())
    };

    // The scopes below are purely diagnostic: they attribute every instruction the decomposition stages, and they are a
    // no-op under an eager context, which records no instructions at all.
    let domain = first.dispatch_domain();
    let scanned = domain.invoke_with_provenance_scope(ProvenanceScope::new("ryft"), || {
        domain.invoke_with_provenance_scope(ProvenanceScope::new("associative_scan"), || {
            associative_scan_recursively(&arrays, extent, axis, reverse, &flat_combine)
        })
    })?;
    Ok(Values::from_parameters(structure, scanned)?)
}

/// Recursive half of [`associative_scan`], operating on the flat arrays of the scanned structure, whose (static) extent
/// along `axis` is `extent`.
fn associative_scan_recursively<V, F>(
    values: &[V],
    extent: usize,
    axis: usize,
    reverse: bool,
    combine: &F,
) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Concatenate + Or + Pad + Slice,
    V::DispatchDomain: Context<Type = ArrayType> + Zero<V>,
    F: Fn(&[V], &[V]) -> Result<Vec<V>, ProgramError>,
{
    if extent < 2 {
        return Ok(values.to_vec());
    }
    let half = extent / 2;

    // Pair adjacent elements. A forward scan pairs from the start of the axis and a reverse scan pairs from its end,
    // which is the one place the two directions differ: an odd extent leaves the first element unpaired going forward
    // and the *last* one unpaired going backward, so the pairing starts one element in.
    let pair_offset = match reverse {
        true => extent % 2,
        false => 0,
    };
    let earlier = scan_slice(values, axis, pair_offset, extent - 1, 2)?;
    let later = scan_slice(values, axis, pair_offset + 1, extent, 2)?;
    let reduced = match reverse {
        true => combine(&later, &earlier)?,
        false => combine(&earlier, &later)?,
    };

    // Scanning the pairwise reductions yields every other output element: the odd positions of a forward scan, and the
    // positions congruent to `pair_offset` of a reverse one.
    let aligned = associative_scan_recursively(&reduced, half, axis, reverse, combine)?;

    // Each complementary position extends the aligned result before it by the one element that separates them, except
    // for the position at the scan's own start, which is just the operand element there. An even extent has one fewer
    // complementary combination than there are aligned results, so the aligned side is trimmed; an extent of exactly
    // two has none at all, and its complementary half is that lone start element.
    let complement_count = match extent % 2 {
        0 => half - 1,
        _ => half,
    };
    let (complement, aligned_leads) = match reverse {
        true => {
            let last = scan_slice(values, axis, extent - 1, extent, 1)?;
            let complement = match complement_count {
                0 => last,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, axis, 1, half, 1)?,
                        _ => aligned.clone(),
                    };
                    let start = (pair_offset + 1) % 2;
                    let operands = scan_slice(values, axis, start, start + 2 * complement_count, 2)?;
                    combine(&trimmed, &operands)?
                        .iter()
                        .zip(&last)
                        .map(|(combined, last)| V::concatenate([combined, last], axis))
                        .collect::<Result<Vec<_>, _>>()?
                }
            };
            (complement, extent % 2 == 0)
        }
        false => {
            let first = scan_slice(values, axis, 0, 1, 1)?;
            let complement = match complement_count {
                0 => first,
                _ => {
                    let trimmed = match extent % 2 {
                        0 => scan_slice(&aligned, axis, 0, half - 1, 1)?,
                        _ => aligned.clone(),
                    };
                    let operands = scan_slice(values, axis, 2, (2 + 2 * complement_count).min(extent), 2)?;
                    first
                        .iter()
                        .zip(&combine(&trimmed, &operands)?)
                        .map(|(first, combined)| V::concatenate([first, combined], axis))
                        .collect::<Result<Vec<_>, _>>()?
                }
            };
            (complement, false)
        }
    };

    match aligned_leads {
        true => scan_interleave(&aligned, &complement, axis, half, extent - half),
        false => scan_interleave(&complement, &aligned, axis, extent - half, half),
    }
}

/// Returns the elements of each array in `values` at positions `start`, `start + stride`, ... below `limit` along
/// `axis`, keeping every other axis whole (including a dynamic one) through [`Slice::slice_axis`].
fn scan_slice<V: Slice + Typed<Type = ArrayType>>(
    values: &[V],
    axis: usize,
    start: usize,
    limit: usize,
    stride: usize,
) -> Result<Vec<V>, ProgramError> {
    values.iter().map(|value| value.slice_axis(axis, start, limit, stride)).collect()
}

/// Returns each array of `left` interleaved along `axis` with the corresponding array of `right`, starting with the
/// `left` one. Each `left` array must hold either as many elements along `axis` as its `right` counterpart or exactly
/// one more.
///
/// Both operands are dilated into the output extent with interior padding (writing zeros into the positions that the
/// other operand occupies) and then combined with an addition, or with a disjunction for Boolean operands, which have
/// no addition. The combination is exact because the two dilated operands have disjoint support and zero (i.e.,
/// `false`) is the identity of both combiners.
fn scan_interleave<V>(
    left: &[V],
    right: &[V],
    axis: usize,
    left_count: usize,
    right_count: usize,
) -> Result<Vec<V>, ProgramError>
where
    V: Value<Type = ArrayType> + Add + Or + Pad,
    V::DispatchDomain: Zero<V>,
{
    if left_count != right_count && left_count != right_count + 1 {
        return Err(TypeError::invalid(format!(
            "`associative_scan` cannot interleave {left_count} elements with {right_count} elements"
        ))
        .into());
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let rank = left.r#type().rank();
            let padding_value = left.dispatch_domain().zero(&left.r#type().scalar_like()?)?;
            let mut edge_padding_low = vec![0; rank];
            let mut edge_padding_high = vec![0; rank];
            let mut interior_padding = vec![0; rank];
            interior_padding[axis] = 1;
            edge_padding_high[axis] = i64::from(left_count == right_count);
            let dilated_left = left.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
            edge_padding_low[axis] = 1;
            edge_padding_high[axis] = i64::from(left_count != right_count);
            let dilated_right = right.pad(&padding_value, &edge_padding_low, &edge_padding_high, &interior_padding)?;
            match left.r#type().data_type() {
                DataType::Boolean => dilated_left.or(&dilated_right),
                _ => dilated_left.add(&dilated_right),
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{Array, ArrayOperation, Dimension, DimensionBounds, DimensionVariable, Shape, StaticShape};
    use crate::contexts::StagingContext;
    use crate::operations::comparisons::{Compare, ComparisonDirection};
    use crate::operations::control_flow::select::Select;
    use crate::operations::cumulative::cumulative_evaluate;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramRenderingMode;
    use crate::tracing::TracingContext;

    use super::*;

    #[test]
    fn test_associative_scan() {
        // The decomposition is checked against the sequential scan of the same combiner, over both parities of the
        // scanned extent and in both directions. Summation pins the positions each output accumulates over, and the
        // left projection (which is associative but not commutative) additionally pins the operand order that the
        // construction passes to the combiner: its forward scan is the first element repeated and its reverse scan the
        // last.
        let add = |left: &Array, right: &Array| left.add(right);
        let first = |left: &Array, _right: &Array| Ok(left.clone());
        for extent in 0..=9usize {
            let values = (1..=extent).map(|value| value as f64).collect::<Vec<_>>();
            let input = Array::vector(values.clone()).unwrap();
            let shape = StaticShape::new(vec![extent]);
            for reverse in [false, true] {
                assert_eq!(
                    associative_scan(&input, 0, reverse, &add).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, right| Ok(left + right)),
                    "summation over extent {extent}, reverse {reverse}",
                );
                assert_eq!(
                    associative_scan(&input, 0, reverse, &first).map(|output| output.to_f64s()),
                    cumulative_evaluate(values.as_slice(), &shape, 0, reverse, |left, _right| Ok(left)),
                    "left projection over extent {extent}, reverse {reverse}",
                );
            }
        }

        // Boolean operands are interleaved with a disjunction, because Booleans have no addition.
        let or = |left: &Array, right: &Array| left.or(right);
        let booleans = Array::vector(vec![false, false, true, false, false]).unwrap();
        assert_eq!(
            associative_scan(&booleans, 0, false, &or),
            Ok(Array::vector(vec![false, false, true, true, true]).unwrap()),
        );
        assert_eq!(
            associative_scan(&booleans, 0, true, &or),
            Ok(Array::vector(vec![true, true, true, false, false]).unwrap()),
        );

        // The construction scans one axis of a higher-rank operand independently per row.
        let matrix = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert_eq!(
            associative_scan(&matrix, 1, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]).unwrap()),
        );
        assert_eq!(
            associative_scan(&matrix, 1, true, &add),
            Ok(Array::matrix(2, 3, vec![6.0, 5.0, 3.0, 15.0, 11.0, 6.0]).unwrap()),
        );
        assert_eq!(
            associative_scan(&matrix, 0, false, &add),
            Ok(Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 5.0, 7.0, 9.0]).unwrap()),
        );

        // Negative axes count from the end of the shape.
        assert_eq!(associative_scan(&matrix, -1, false, &add), associative_scan(&matrix, 1, false, &add));
        assert_eq!(associative_scan(&matrix, -2, true, &add), associative_scan(&matrix, 0, true, &add));

        // The construction slices at staging-time positions, so it needs an in-bounds axis.
        assert_eq!(
            associative_scan(&matrix, 2, false, &add),
            Err(ProgramError::Type(TypeError::invalid("`associative_scan` axis 2 is out of bounds for rank 2"))),
        );
        assert_eq!(
            associative_scan(&matrix, -3, false, &add),
            Err(ProgramError::Type(TypeError::invalid("`associative_scan` axis -3 is out of bounds for rank 2"))),
        );
    }

    #[test]
    fn test_associative_scan_dynamic_unscanned_axes() {
        // Only the scanned axis needs a static extent: every other axis is sliced whole, so a dynamic one keeps its
        // dynamic extent through the construction.
        let batch = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2)]),
        ));
        let output = associative_scan(&input, 1, false, &|left, right| left.add(right)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[batch, 2] .
                let %1:f64[batch, 1] = slice [start_indices=[0, 0], limits=[batch, 1], strides=[1, 2]] %0
                    %2:f64[batch, 1] = slice [start_indices=[0, 1], limits=[batch, 2], strides=[1, 2]] %0
                    %3:f64[batch, 1] = add %1 %2
                    %4:f64[batch, 1] = slice [start_indices=[0, 0], limits=[batch, 1]] %0
                    %5:f64[] = zero [type=f64[]]
                    %6:f64[batch, 2] = pad [edge_padding_low=[0, 0], edge_padding_high=[0, 1], interior_padding=[0, 1]] %4 %5
                    %7:f64[batch, 2] = pad [edge_padding_low=[0, 1], edge_padding_high=[0, 0], interior_padding=[0, 1]] %3 %5
                    %8:f64[batch, 2] = add %6 %7
                in (%8)"
            },
        );

        // The scanned axis itself must still be static, since the construction slices it at staging-time positions.
        assert_eq!(
            associative_scan(&input, 0, false, &|left, right| left.add(right)),
            Err(ProgramError::Type(TypeError::invalid(
                "`associative_scan` requires a static extent along the scanned axis 0 but got `f64[batch, 2]`",
            ))),
        );
    }

    #[test]
    fn test_associative_scan_structures() {
        // A structure of arrays is scanned in lockstep under one combining operator over whole structures, which lets
        // arrays of different data types and ranks combine jointly. Here, a running maximum carries the position at
        // which it is attained (keeping the accumulated position on ties) alongside a running sum over a matrix.
        let running_maximum = |left: &(Array, Array, Array), right: &(Array, Array, Array)| {
            let (left_maximum, left_position, left_sum) = left;
            let (right_maximum, right_position, right_sum) = right;
            let greater = right_maximum.compare(left_maximum, ComparisonDirection::GreaterThan)?;
            Ok((
                Array::select(&greater, right_maximum, left_maximum)?,
                Array::select(&greater, right_position, left_position)?,
                left_sum.add(right_sum)?,
            ))
        };
        let values = (
            Array::vector(vec![3.0, 1.0, 4.0, 1.0, 5.0]).unwrap(),
            Array::vector(vec![0i32, 1, 2, 3, 4]).unwrap(),
            Array::matrix(5, 2, vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).unwrap(),
        );
        assert_eq!(
            associative_scan(&values, 0, false, &running_maximum),
            Ok((
                Array::vector(vec![3.0, 3.0, 4.0, 4.0, 5.0]).unwrap(),
                Array::vector(vec![0i32, 0, 2, 2, 4]).unwrap(),
                Array::matrix(5, 2, vec![1.0f32, 2.0, 4.0, 6.0, 9.0, 12.0, 16.0, 20.0, 25.0, 30.0]).unwrap(),
            )),
        );
        assert_eq!(
            associative_scan(&values, 0, true, &running_maximum),
            Ok((
                Array::vector(vec![5.0; 5]).unwrap(),
                Array::vector(vec![4i32; 5]).unwrap(),
                Array::matrix(5, 2, vec![25.0f32, 30.0, 24.0, 28.0, 21.0, 24.0, 16.0, 18.0, 9.0, 10.0]).unwrap(),
            )),
        );

        // A structure that holds no arrays has nothing to scan.
        let add_all = |left: &Vec<Array>, right: &Vec<Array>| {
            left.iter().zip(right).map(|(left, right)| left.add(right)).collect::<Result<Vec<_>, _>>()
        };
        assert_eq!(associative_scan(&Vec::<Array>::new(), 0, false, &add_all), Ok(Vec::new()));

        // A negative axis is normalized against the rank of the first array, and that position is scanned in every
        // array, so every array must have it.
        let mixed_ranks =
            vec![Array::matrix(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap(), Array::vector(vec![5.0, 6.0]).unwrap()];
        assert_eq!(
            associative_scan(&mixed_ranks, -2, false, &add_all),
            Ok(vec![Array::matrix(2, 2, vec![1.0, 2.0, 4.0, 6.0]).unwrap(), Array::vector(vec![5.0, 11.0]).unwrap()]),
        );
        assert_eq!(
            associative_scan(&mixed_ranks, -1, false, &add_all),
            Err(ProgramError::Type(TypeError::invalid("`associative_scan` axis 1 is out of bounds for rank 1"))),
        );

        // The arrays are sliced in lockstep, so they must agree on the scanned extent, and the combining operator must
        // return as many arrays as it receives.
        let mismatched = vec![Array::vector(vec![1.0; 3]).unwrap(), Array::vector(vec![1.0; 2]).unwrap()];
        assert_eq!(
            associative_scan(&mismatched, 0, false, &add_all),
            Err(ProgramError::Type(TypeError::invalid(
                "`associative_scan` requires operands with equal extents along axis 0 but got 3 and 2",
            ))),
        );
        let pair = vec![Array::vector(vec![1.0; 3]).unwrap(), Array::vector(vec![2.0; 3]).unwrap()];
        assert_eq!(
            associative_scan(&pair, 0, false, &|left: &Vec<Array>, right: &Vec<Array>| {
                Ok(vec![left[0].add(&right[0])?])
            }),
            Err(ProgramError::Type(TypeError::invalid(
                "`associative_scan` combining operator must return 2 arrays but returned 1",
            ))),
        );
    }

    #[test]
    fn test_associative_scan_provenance() {
        // Every instruction that the decomposition stages carries the nested framework scopes, which attribute it to
        // the associative-scan decomposition in renderings that include provenance.
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let input = context.input(ArrayType::new_static(DataType::F64, [2]));
        let output = associative_scan(&input, 0, false, &|left, right| left.add(right)).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(vec![output.atom_id().unwrap()], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            std::fmt::from_fn(|formatter| program.render(formatter, 0, ProgramRenderingMode::WithProvenance))
                .to_string(),
            indoc! {"
                lambda %0:f64[2] .
                let %1:f64[1] = slice [start_indices=[0], limits=[1], strides=[2]] %0 ; provenance=ryft::associative_scan
                    %2:f64[1] = slice [start_indices=[1], limits=[2], strides=[2]] %0 ; provenance=ryft::associative_scan
                    %3:f64[1] = add %1 %2 ; provenance=ryft::associative_scan
                    %4:f64[1] = slice [start_indices=[0], limits=[1]] %0 ; provenance=ryft::associative_scan
                    %5:f64[] = zero [type=f64[]] ; provenance=ryft::associative_scan
                    %6:f64[2] = pad [edge_padding_low=[0], edge_padding_high=[1], interior_padding=[1]] %4 %5 ; provenance=ryft::associative_scan
                    %7:f64[2] = pad [edge_padding_low=[1], edge_padding_high=[0], interior_padding=[1]] %3 %5 ; provenance=ryft::associative_scan
                    %8:f64[2] = add %6 %7 ; provenance=ryft::associative_scan
                in (%8)"
            },
        );
    }
}
