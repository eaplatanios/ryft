//! Contains the named-axis collective operations, which exchange or reduce values across a named axis, together with
//! their interpretation, partial-evaluation, batching, forward-mode differentiation, and transposition rules. These
//! are the analogues of [JAX's parallel operators](https://docs.jax.dev/en/latest/jax.lax.html#parallel-operators).
//!
//! This module owns the vocabulary that every collective shares (i.e., [`CollectiveMode`], [`CollectiveOptions`],
//! named axis resolution, and [`forward_collective_to_parent`]), while each operation family lives in its own
//! submodule: [`parallel_reduce`], [`parallel_vary`], [`all_gather`], [`parallel_sum_scatter`], [`parallel_permute`],
//! [`all_to_all`], and [`ragged_all_to_all`]. Two private submodules hold the machinery shared across families:
//! `linear` implements the common structure of the single-operand linear collectives (`parallel_permute`,
//! `all_gather`, `parallel_sum_scatter`, and `all_to_all`) through its operation-generating macro, and
//! `shape_changing` implements the batching, first-class extent, and explicit array IR rules of the collectives that
//! resize an array axis (`all_gather`, `parallel_sum_scatter`, and `all_to_all`, whose batching policy
//! `ragged_all_to_all` also reuses).
//!
//! Collectives reference an enclosing named-axis binder by name, validated against the active
//! [`NamedAxes`] environment at staging time. A name bound by an enclosing `batch` level is
//! resolved at trace time by the operations' batching rules, which collapse or materialize the mapped batch axis at
//! the binding level, while a name bound to a device mesh axis by a `shard_map` manual region stays in the staged
//! body and lowers to cross-device collectives over that mesh axis.

// TODO(eaplatanios): Review this module.

use std::fmt::Debug;

use crate::arrays::{ArrayBatch, ArrayBatchingPolicy, ArrayExtentBatchingPolicy, ArrayType};
use crate::axes::{AxisError, NamedAxes, NamedAxis};
use crate::batching::{BatchingContext, BatchingError};
use crate::contexts::{Context, Domain};
use crate::macros::check_count;
use crate::programs::{Operation, ProgramError, TypeError, Value};

pub mod all_gather;
pub mod all_to_all;
mod linear;
pub mod parallel_permute;
pub mod parallel_reduce;
pub mod parallel_sum_scatter;
pub mod parallel_vary;
pub mod ragged_all_to_all;
mod shape_changing;

pub use all_gather::{ALL_GATHER_OPERATION_NAME, AllGather, AllGatherOperation, AllGatherOutputVariance};
pub use all_to_all::{ALL_TO_ALL_OPERATION_NAME, AllToAll, AllToAllOperation, ParallelSwapAxes};
pub use parallel_permute::{
    PARALLEL_PERMUTE_OPERATION_NAME, ParallelPermute, ParallelPermuteOperation, ParallelShuffle,
};
pub use parallel_reduce::{ParallelReduce, ParallelReduceOperation, ParallelReductionKind};
pub use parallel_sum_scatter::{PARALLEL_SUM_SCATTER_OPERATION_NAME, ParallelSumScatter, ParallelSumScatterOperation};
pub use parallel_vary::{ManualVariationAlignment, PARALLEL_VARY_OPERATION_NAME, ParallelVary, ParallelVaryOperation};
pub use ragged_all_to_all::{RAGGED_ALL_TO_ALL_OPERATION_NAME, RaggedAllToAll, RaggedAllToAllOperation};

/// Shape semantics used by collectives that can either materialize a named axis or tile an existing array axis.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub enum CollectiveMode {
    /// Materializes the named axis as a new ranked array dimension, or consumes one ranked dimension when scattering.
    #[default]
    Untiled,

    /// Preserves array rank by multiplying or dividing an existing ranked array dimension.
    Tiled,
}

/// Shared shape and grouping options for all-gather, sum-scatter, and all-to-all.
#[derive(Clone, Default, PartialEq, Eq, Hash)]
pub struct CollectiveOptions {
    /// Rank-changing or rank-preserving shape semantics.
    mode: CollectiveMode,

    /// Optional ordered partition of logical participant indices.
    axis_index_groups: Option<Vec<Vec<usize>>>,
}

impl Debug for CollectiveOptions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.axis_index_groups {
            None => Debug::fmt(&self.mode, formatter),
            Some(axis_index_groups) => formatter
                .debug_struct("CollectiveOptions")
                .field("mode", &self.mode)
                .field("axis_index_groups", axis_index_groups)
                .finish(),
        }
    }
}

impl CollectiveOptions {
    /// Creates collective options for `mode` with no participant subgroups.
    #[inline]
    pub fn new(mode: CollectiveMode) -> Self {
        Self { mode, axis_index_groups: None }
    }

    /// Creates rank-preserving tiled collective options with no participant subgroups.
    #[inline]
    pub fn tiled() -> Self {
        Self::new(CollectiveMode::Tiled)
    }

    /// Returns these options with the provided ordered participant groups.
    #[inline]
    pub fn with_axis_index_groups(mut self, axis_index_groups: Vec<Vec<usize>>) -> Self {
        self.axis_index_groups = Some(axis_index_groups);
        self
    }

    /// Returns the selected shape mode.
    #[inline]
    pub fn mode(&self) -> CollectiveMode {
        self.mode
    }

    /// Returns the ordered participant groups, if any.
    #[inline]
    pub fn axis_index_groups(&self) -> Option<&[Vec<usize>]> {
        self.axis_index_groups.as_deref()
    }

    /// Validates these options against the full named-axis size and returns the effective group size used for shape
    /// arithmetic.
    pub(super) fn effective_axis_size(&self, operation_name: &str, axis_size: usize) -> Result<usize, TypeError> {
        effective_collective_axis_size(operation_name, axis_size, self.axis_index_groups())
    }
}

/// Validates an optional ordered participant partition and returns its effective group size without copying it.
pub(super) fn effective_collective_axis_size(
    operation_name: &str,
    axis_size: usize,
    groups: Option<&[Vec<usize>]>,
) -> Result<usize, TypeError> {
    validate_collective_axis_size(operation_name, axis_size)?;
    let Some(groups) = groups else {
        return Ok(axis_size);
    };
    let Some(first_group) = groups.first() else {
        return Err(TypeError::invalid(format!("`{operation_name}` axis index groups must not be empty")));
    };
    if first_group.is_empty() {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups must contain at least one participant",
        )));
    }
    let group_size = first_group.len();
    let mut seen = vec![false; axis_size];
    for (group_index, group) in groups.iter().enumerate() {
        if group.len() != group_size {
            return Err(TypeError::invalid(format!(
                "`{operation_name}` axis index group {group_index} has size {} but every group must have size \
                     {group_size}",
                group.len(),
            )));
        }
        for &participant in group {
            let Some(participant_seen) = seen.get_mut(participant) else {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index {participant} is out of bounds for axis size {axis_size}",
                )));
            };
            if *participant_seen {
                return Err(TypeError::invalid(format!(
                    "`{operation_name}` axis index groups contain participant {participant} more than once",
                )));
            }
            *participant_seen = true;
        }
    }
    if let Some(missing) = seen.iter().position(|seen| !seen) {
        return Err(TypeError::invalid(format!(
            "`{operation_name}` axis index groups do not contain participant {missing}",
        )));
    }
    Ok(group_size)
}

/// Rejects ragged collective operands before any parent binding can stage or execute collective work.
pub(super) fn reject_ragged_collective_inputs<V: Value<Type = ArrayType>>(
    operation_name: &str,
    inputs: &[ArrayBatch<V>],
) -> Result<(), BatchingError> {
    if let Some((index, ragged_axis)) = inputs
        .iter()
        .enumerate()
        .find_map(|(index, input)| input.ragged_axes().first().map(|ragged_axis| (index, ragged_axis)))
    {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{}` does not support bounded ragged dimension `{}` on operand {}",
                operation_name,
                ragged_axis.dimension(),
                index,
            ),
        });
    }
    Ok(())
}

/// Re-stages a collective that targets a different (outer) named axis into the batching context's parent.
///
/// Under nested `batch` levels, a collective is consumed by the level whose
/// [`axis_name`](crate::batching::BatchingContext::axis_name) matches its axis name and must pass through
/// every inner level untouched: each inner batch item participates in the outer collective independently, so the
/// operands' mapped axes are preserved as-is on the forwarded outputs. The parent may itself be another
/// [`BatchingContext`] — whose own rule dispatch repeats this name
/// resolution at the next level — or an ordinary tracing context. Batching rules for custom collective-like
/// operations should use this helper for their "not my axis" arm.
pub fn forward_collective_to_parent<C, P: ArrayExtentBatchingPolicy<C>>(
    context: &BatchingContext<C, ArrayBatchingPolicy<P>>,
    parent_operation: C::Operation,
    inputs: &[ArrayBatch<<C as Domain>::Value>],
) -> Result<Vec<ArrayBatch<<C as Domain>::Value>>, BatchingError>
where
    C: Context<Type = ArrayType>,
{
    reject_ragged_collective_inputs(parent_operation.name(), inputs)?;
    let parent_input_values: Vec<<C as Domain>::Value> = inputs.iter().map(|batch| batch.value().clone()).collect();
    let parent_outputs = context.parent().bind(parent_operation, Vec::new(), &parent_input_values)?;
    check_count!("output", parent_outputs, inputs.len(), ProgramError);
    parent_outputs
        .into_iter()
        .zip(inputs.iter())
        .map(|(parent_value, input_batch)| ArrayBatch::new(parent_value, input_batch.batch_axis()))
        .collect()
}

/// Resolves the size of the named axis bound by the active [`NamedAxes`] environment, failing fast with
/// [`AxisError::UnboundAxisName`] when no enclosing binder binds `axis_name`. The collective capabilities bake the
/// resolved size into their operation payloads at staging time, because their output shapes and payload validation
/// depend on it while [`Operation::infer_output_types`] only sees input types.
pub(super) fn resolve_named_axis_size<C: NamedAxes>(context: &C, axis_name: &str) -> Result<usize, ProgramError> {
    match context.named_axis(axis_name) {
        Some(NamedAxis::Batched { size: Some(size) } | NamedAxis::Mesh { size, .. }) if size > 0 => Ok(size),
        Some(NamedAxis::Batched { size: Some(_) } | NamedAxis::Mesh { .. }) => {
            Err(TypeError::invalid(format!("collective axis `{axis_name}` must contain at least one participant",))
                .into())
        }
        Some(NamedAxis::Batched { size: None }) => Err(BatchingError::UnsupportedOperation {
            message: format!(
                "collective axis `{axis_name}` has a dynamic extent that must remain a first-class operand"
            ),
        }
        .into()),
        None => Err(BatchingError::Axis(AxisError::UnboundAxisName { name: axis_name.to_string() }).into()),
    }
}

/// Rejects an invalid zero-participant collective before any multiplication, division, or remainder operation.
pub(crate) fn validate_collective_axis_size(operation_name: &str, axis_size: usize) -> Result<(), TypeError> {
    if axis_size == 0 {
        Err(TypeError::invalid(format!("`{operation_name}` axis size must be greater than zero")))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{DataType, Dimension, Shape};

    use super::*;

    /// Returns the static `f32` vector type of the provided length shared by the collective tests.
    pub(super) fn f32_vector(length: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(length)]))
    }

    #[test]
    fn test_collective_options_validate_axis_index_groups() {
        let options = CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
        assert_eq!(options.mode(), CollectiveMode::Tiled);
        assert_eq!(options.axis_index_groups(), Some([vec![0, 2], vec![3, 1]].as_slice()));
        assert_eq!(options.effective_axis_size("all_gather", 4), Ok(2));

        assert_eq!(
            CollectiveOptions::default().with_axis_index_groups(Vec::new()).effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups must not be empty")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index group 1 has size 1 but every group must have size 2",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![1, 2]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index groups contain participant 1 more than once",)),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1], vec![2, 4]])
                .effective_axis_size("all_gather", 4),
            Err(TypeError::invalid("`all_gather` axis index 4 is out of bounds for axis size 4")),
        );
        assert_eq!(
            CollectiveOptions::default()
                .with_axis_index_groups(vec![vec![0, 1]])
                .effective_axis_size("all_gather", 3),
            Err(TypeError::invalid("`all_gather` axis index groups do not contain participant 2")),
        );
    }
}
