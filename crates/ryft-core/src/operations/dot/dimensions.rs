use super::*;

/// Specification of contracting and batching dimensions for a generalized dot product.
///
/// Mirrors StableHLO's `dot_general` operand: the contracting dimensions index axes that are
/// summed over (the "K" axes in matrix multiplication), and the batching dimensions index axes
/// that are aligned 1:1 between the two operands and preserved in the output (the leading "B"
/// axes in batched matrix multiplication).
///
/// Both `lhs_contracting_dimensions` and `rhs_contracting_dimensions` must have the same length
/// and their corresponding dimensions in the two operands must match in size. The same applies
/// to `lhs_batching_dimensions` / `rhs_batching_dimensions`.
///
/// The output shape is `[batching..., lhs_result..., rhs_result...]`, where the result
/// dimensions are the remaining (non-contracting, non-batching) dimensions of each operand, in
/// their original order.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotDimensionNumbers {
    /// Axes on the LHS operand that contract with `rhs_contracting_dimensions` on the RHS.
    lhs_contracting_dimensions: Vec<usize>,

    /// Axes on the RHS operand that contract with `lhs_contracting_dimensions` on the LHS.
    rhs_contracting_dimensions: Vec<usize>,

    /// Axes on the LHS operand that are aligned 1:1 with `rhs_batching_dimensions` on the RHS
    /// and that are preserved in the output.
    lhs_batching_dimensions: Vec<usize>,

    /// Axes on the RHS operand that are aligned 1:1 with `lhs_batching_dimensions` on the LHS
    /// and that are preserved in the output.
    rhs_batching_dimensions: Vec<usize>,
}

impl DotDimensionNumbers {
    /// Creates dimension numbers from explicit contracting and batching dimensions.
    #[inline]
    pub fn new(
        lhs_contracting_dimensions: Vec<usize>,
        rhs_contracting_dimensions: Vec<usize>,
        lhs_batching_dimensions: Vec<usize>,
        rhs_batching_dimensions: Vec<usize>,
    ) -> Self {
        Self {
            lhs_contracting_dimensions,
            rhs_contracting_dimensions,
            lhs_batching_dimensions,
            rhs_batching_dimensions,
        }
    }

    /// Dimension numbers for a standard rank-2 matrix multiplication:
    /// `[M, K] @ [K, N] -> [M, N]`. Contracting dimension is the last axis of the LHS and the
    /// first axis of the RHS; there are no batching dimensions.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(vec![1], vec![0], Vec::new(), Vec::new())
    }

    /// Dimension numbers for a rank-1 inner product: `[K] · [K] -> []`. The single dimension of
    /// each operand contracts.
    #[inline]
    pub fn inner_product() -> Self {
        Self::new(vec![0], vec![0], Vec::new(), Vec::new())
    }

    /// Returns the contracting dimensions of the LHS operand.
    #[inline]
    pub fn lhs_contracting_dimensions(&self) -> &[usize] {
        &self.lhs_contracting_dimensions
    }

    /// Returns the contracting dimensions of the RHS operand.
    #[inline]
    pub fn rhs_contracting_dimensions(&self) -> &[usize] {
        &self.rhs_contracting_dimensions
    }

    /// Returns the batching dimensions of the LHS operand.
    #[inline]
    pub fn lhs_batching_dimensions(&self) -> &[usize] {
        &self.lhs_batching_dimensions
    }

    /// Returns the batching dimensions of the RHS operand.
    #[inline]
    pub fn rhs_batching_dimensions(&self) -> &[usize] {
        &self.rhs_batching_dimensions
    }
}

impl Display for DotDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(lhs_contracting={:?}, rhs_contracting={:?}, lhs_batching={:?}, rhs_batching={:?})",
            self.lhs_contracting_dimensions,
            self.rhs_contracting_dimensions,
            self.lhs_batching_dimensions,
            self.rhs_batching_dimensions,
        )
    }
}

/// Placement mode of the single ragged LHS dimension in a [`RaggedDotDimensionNumbers`] specification.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RaggedDotMode {
    /// The ragged dimension is a non-contracting result dimension.
    NonContracting,

    /// The ragged dimension is contracted with the RHS operand.
    Contracting,

    /// The ragged dimension is one of the paired batching dimensions.
    Batch,
}

impl Display for RaggedDotMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::NonContracting => "non-contracting",
            Self::Contracting => "contracting",
            Self::Batch => "batch",
        })
    }
}

/// Dimension-number specification for a grouped generalized dot product.
///
/// The ordinary [`DotDimensionNumbers`] describe the contraction and batching axes. Exactly one LHS axis is marked
/// ragged, and RHS group dimensions identify the explicit group axis used by non-contracting ragged dots.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RaggedDotDimensionNumbers {
    /// Underlying generalized-dot dimension numbers.
    dot_dimensions: DotDimensionNumbers,

    /// LHS axes marked ragged. Type inference requires exactly one entry.
    lhs_ragged_dimensions: Vec<usize>,

    /// RHS axes indexing groups. Non-contracting mode requires exactly one; the other modes require none.
    rhs_group_dimensions: Vec<usize>,
}

impl RaggedDotDimensionNumbers {
    /// Creates a grouped-dot dimension-number specification.
    #[inline]
    pub fn new(
        dot_dimensions: DotDimensionNumbers,
        lhs_ragged_dimensions: Vec<usize>,
        rhs_group_dimensions: Vec<usize>,
    ) -> Self {
        Self { dot_dimensions, lhs_ragged_dimensions, rhs_group_dimensions }
    }

    /// Creates the basic non-contracting matrix form `[M, K] × [G, K, N] → [M, N]`.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()), vec![0], vec![0])
    }

    /// Returns the underlying generalized-dot dimensions.
    #[inline]
    pub fn dot_dimensions(&self) -> &DotDimensionNumbers {
        &self.dot_dimensions
    }

    /// Returns the LHS ragged dimensions.
    #[inline]
    pub fn lhs_ragged_dimensions(&self) -> &[usize] {
        &self.lhs_ragged_dimensions
    }

    /// Returns the RHS group dimensions.
    #[inline]
    pub fn rhs_group_dimensions(&self) -> &[usize] {
        &self.rhs_group_dimensions
    }

    /// Classifies this specification according to the role of its single LHS ragged dimension.
    pub fn mode(&self, lhs_rank: usize) -> Result<RaggedDotMode, TypeError> {
        if self.lhs_ragged_dimensions.len() != 1 {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` expects exactly one LHS ragged dimension, but got {}",
                self.lhs_ragged_dimensions.len(),
            )));
        }
        let axis = self.lhs_ragged_dimensions[0];
        if axis >= lhs_rank {
            return Err(TypeError::invalid(format!(
                "`{RAGGED_DOT_OPERATION_NAME}` LHS ragged dimension {axis} is out of bounds for rank {lhs_rank}",
            )));
        }
        if self.dot_dimensions.lhs_contracting_dimensions().contains(&axis) {
            Ok(RaggedDotMode::Contracting)
        } else if self.dot_dimensions.lhs_batching_dimensions().contains(&axis) {
            Ok(RaggedDotMode::Batch)
        } else {
            Ok(RaggedDotMode::NonContracting)
        }
    }

    /// Returns the input-prefix axes that index a prefix-shaped `group_sizes` operand.
    pub fn group_sizes_prefix_dimensions(&self, lhs_rank: usize) -> Result<Vec<usize>, TypeError> {
        let ragged_axis = *self.lhs_ragged_dimensions.first().ok_or_else(|| {
            TypeError::invalid(format!("`{RAGGED_DOT_OPERATION_NAME}` expects exactly one LHS ragged dimension",))
        })?;
        let dimensions = self.dot_dimensions();
        Ok(match self.mode(lhs_rank)? {
            RaggedDotMode::NonContracting => {
                let result_axes = lhs_result_axes(dimensions, lhs_rank);
                let position = result_axes.iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions
                    .lhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(result_axes[..position].iter().copied())
                    .collect()
            }
            RaggedDotMode::Contracting => {
                let position =
                    dimensions.lhs_contracting_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions
                    .lhs_batching_dimensions()
                    .iter()
                    .copied()
                    .chain(dimensions.lhs_contracting_dimensions()[..position].iter().copied())
                    .collect()
            }
            RaggedDotMode::Batch => {
                let position =
                    dimensions.lhs_batching_dimensions().iter().position(|axis| *axis == ragged_axis).unwrap();
                dimensions.lhs_batching_dimensions()[..position].to_vec()
            }
        })
    }
}

impl Display for RaggedDotDimensionNumbers {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "(dot={}, lhs_ragged={:?}, rhs_group={:?})",
            self.dot_dimensions, self.lhs_ragged_dimensions, self.rhs_group_dimensions,
        )
    }
}

/// Returns the lhs result axes of `dimensions` for an LHS of the supplied rank.
pub fn lhs_result_axes(dimensions: &DotDimensionNumbers, lhs_rank: usize) -> Vec<usize> {
    (0..lhs_rank)
        .filter(|axis| {
            !dimensions.lhs_batching_dimensions.contains(axis) && !dimensions.lhs_contracting_dimensions.contains(axis)
        })
        .collect()
}

/// Returns the rhs result axes of `dimensions` for an RHS of the supplied rank.
pub fn rhs_result_axes(dimensions: &DotDimensionNumbers, rhs_rank: usize) -> Vec<usize> {
    (0..rhs_rank)
        .filter(|axis| {
            !dimensions.rhs_batching_dimensions.contains(axis) && !dimensions.rhs_contracting_dimensions.contains(axis)
        })
        .collect()
}

/// Lifts a [`DotDimensionNumbers`] through one batching level.
///
/// Given the unbatched dimension numbers and the batch-axis positions of the two operands (each
/// optional — `None` indicates a replicated operand), returns the dimension numbers that
/// describe the same contraction over the parent-physical (batched) operands. The mapping:
///
/// - When both operands are batched at positions `(k_lhs, k_rhs)`, the lifted op gains one new
///   batching dimension pair `(k_lhs, k_rhs)` at the front of the batching lists, and every
///   existing contracting / batching index `i` is shifted to `i + 1` if `i >= k_{lhs|rhs}`. The
///   new batch axis ends up at position `0` of the output (since batching dims are output-first).
/// - When neither operand is batched, the dimension numbers are unchanged.
/// - Mixed cases (exactly one operand batched) are not yet supported and return `Ok(None)` so
///   the caller can surface `UnsupportedOperation`.
pub fn lift_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    lhs_batch_axis: Option<usize>,
    rhs_batch_axis: Option<usize>,
) -> Option<(DotDimensionNumbers, Option<usize>)> {
    let shift = |axes: &[usize], k: Option<usize>| -> Vec<usize> {
        match k {
            Some(k) => axes.iter().map(|i| if *i >= k { *i + 1 } else { *i }).collect(),
            None => axes.to_vec(),
        }
    };
    match (lhs_batch_axis, rhs_batch_axis) {
        (Some(k_lhs), Some(k_rhs)) => {
            let mut lhs_batching = vec![k_lhs];
            lhs_batching.extend(shift(&dimensions.lhs_batching_dimensions, Some(k_lhs)));
            let mut rhs_batching = vec![k_rhs];
            rhs_batching.extend(shift(&dimensions.rhs_batching_dimensions, Some(k_rhs)));
            Some((
                DotDimensionNumbers {
                    lhs_contracting_dimensions: shift(&dimensions.lhs_contracting_dimensions, Some(k_lhs)),
                    rhs_contracting_dimensions: shift(&dimensions.rhs_contracting_dimensions, Some(k_rhs)),
                    lhs_batching_dimensions: lhs_batching,
                    rhs_batching_dimensions: rhs_batching,
                },
                Some(0),
            ))
        }
        (None, None) => Some((dimensions.clone(), None)),
        _ => None,
    }
}

/// Lifts an optional requested output sharding through one batching level by inserting `axis_sharding` at the new
/// output batch axis. `axis_sharding` is the [`ShardingDimension`] derived from the batched inputs' mapped axis
/// (see [`ArrayBatch::sharding_for_inputs`](crate::ArrayBatch::sharding_for_inputs)), so the batched
/// dimension carries the same sharding as the operands' mapped axis, mirroring JAX's `get_sharding_for_vmap`.
pub(super) fn lift_output_sharding(
    output_sharding: Option<&Sharding>,
    output_axis: Option<usize>,
    axis_sharding: ShardingDimension,
) -> Result<Option<Sharding>, ProgramError> {
    match (output_sharding, output_axis) {
        (Some(output_sharding), Some(axis)) => output_sharding
            .with_inserted_dimension(axis, axis_sharding)
            .map(Some)
            .map_err(|error| BatchingError::MisalignedBatchAxes { message: error.to_string() }.into()),
        (Some(output_sharding), None) => Ok(Some(output_sharding.clone())),
        (None, _) => Ok(None),
    }
}

/// Computes the dimension numbers for the adjoint of the left-factor linear map `t ↦ dot(factor, t)`: maps
/// `t ↦ dot(factor, t; dimensions)`'s output cotangent back to a cotangent for `t`.
pub fn adjoint_dimensions_for_left_dot(dimensions: &DotDimensionNumbers, factor_rank: usize) -> DotDimensionNumbers {
    let n_batching = dimensions.lhs_batching_dimensions.len();
    let factor_result = lhs_result_axes(dimensions, factor_rank);
    let n_factor_result = factor_result.len();
    DotDimensionNumbers {
        lhs_batching_dimensions: dimensions.lhs_batching_dimensions.clone(),
        rhs_batching_dimensions: (0..n_batching).collect(),
        lhs_contracting_dimensions: factor_result,
        rhs_contracting_dimensions: (n_batching..(n_batching + n_factor_result)).collect(),
    }
}

/// Computes the dimension numbers for the adjoint of the right-factor linear map `t ↦ dot(t, factor)`: maps
/// `t ↦ dot(t, factor; dimensions)`'s output cotangent back to a cotangent for `t`.
pub fn adjoint_dimensions_for_right_dot(
    dimensions: &DotDimensionNumbers,
    factor_rank: usize,
    t_rank: usize,
) -> DotDimensionNumbers {
    let n_batching = dimensions.rhs_batching_dimensions.len();
    let factor_result = rhs_result_axes(dimensions, factor_rank);
    let t_result_count =
        t_rank - dimensions.rhs_batching_dimensions.len() - dimensions.rhs_contracting_dimensions.len();
    DotDimensionNumbers {
        lhs_batching_dimensions: (0..n_batching).collect(),
        rhs_batching_dimensions: dimensions.rhs_batching_dimensions.clone(),
        lhs_contracting_dimensions: ((n_batching + t_result_count)
            ..(n_batching + t_result_count + factor_result.len()))
            .collect(),
        rhs_contracting_dimensions: factor_result,
    }
}

/// Computes the grouped-dot dimensions and unpermuted output-axis order for the LHS adjoint of a non-contracting
/// [`RaggedDotDimensionNumbers`] specification.
pub(super) fn adjoint_ragged_dimensions_for_lhs(
    dimensions: &RaggedDotDimensionNumbers,
    lhs_rank: usize,
    rhs_rank: usize,
) -> (RaggedDotDimensionNumbers, Vec<usize>) {
    let dot = dimensions.dot_dimensions();
    let lhs_kept = lhs_result_axes(dot, lhs_rank);
    let rhs_kept = rhs_result_axes(dot, rhs_rank)
        .into_iter()
        .filter(|axis| !dimensions.rhs_group_dimensions().contains(axis))
        .collect::<Vec<_>>();
    let output_batch = 0..dot.lhs_batching_dimensions().len();
    let output_rhs = (output_batch.end + lhs_kept.len())..(output_batch.end + lhs_kept.len() + rhs_kept.len());
    let lhs_ragged_axis = dimensions.lhs_ragged_dimensions()[0];
    let ragged_output_axis =
        dot.lhs_batching_dimensions().len() + lhs_kept.iter().position(|axis| *axis == lhs_ragged_axis).unwrap();
    let adjoint = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(
            output_rhs.collect(),
            rhs_kept,
            output_batch.collect(),
            dot.rhs_batching_dimensions().to_vec(),
        ),
        vec![ragged_output_axis],
        dimensions.rhs_group_dimensions().to_vec(),
    );
    let mut sorted_contracting = dot
        .lhs_contracting_dimensions()
        .iter()
        .copied()
        .zip(dot.rhs_contracting_dimensions().iter().copied())
        .collect::<Vec<_>>();
    sorted_contracting.sort_by_key(|(_, rhs_axis)| *rhs_axis);
    let output_axes = dot
        .lhs_batching_dimensions()
        .iter()
        .copied()
        .chain(lhs_kept)
        .chain(sorted_contracting.into_iter().map(|(lhs_axis, _)| lhs_axis))
        .collect();
    (adjoint, output_axes)
}

/// Computes the grouped-dot dimensions and unpermuted output-axis order for the RHS adjoint of a non-contracting
/// [`RaggedDotDimensionNumbers`] specification.
pub(super) fn adjoint_ragged_dimensions_for_rhs(
    dimensions: &RaggedDotDimensionNumbers,
    lhs_rank: usize,
    rhs_rank: usize,
) -> (RaggedDotDimensionNumbers, Vec<usize>) {
    let dot = dimensions.dot_dimensions();
    let lhs_kept = lhs_result_axes(dot, lhs_rank);
    let rhs_kept = rhs_result_axes(dot, rhs_rank)
        .into_iter()
        .filter(|axis| !dimensions.rhs_group_dimensions().contains(axis))
        .collect::<Vec<_>>();
    let output_batch = 0..dot.lhs_batching_dimensions().len();
    let output_lhs = output_batch.end..(output_batch.end + lhs_kept.len());
    let adjoint = RaggedDotDimensionNumbers::new(
        DotDimensionNumbers::new(
            lhs_kept,
            output_lhs.collect(),
            dot.lhs_batching_dimensions().to_vec(),
            output_batch.collect(),
        ),
        dimensions.lhs_ragged_dimensions().to_vec(),
        Vec::new(),
    );
    let mut sorted_contracting = dot
        .rhs_contracting_dimensions()
        .iter()
        .copied()
        .zip(dot.lhs_contracting_dimensions().iter().copied())
        .collect::<Vec<_>>();
    sorted_contracting.sort_by_key(|(_, lhs_axis)| *lhs_axis);
    let output_axes = dimensions
        .rhs_group_dimensions()
        .iter()
        .copied()
        .chain(dot.rhs_batching_dimensions().iter().copied())
        .chain(sorted_contracting.into_iter().map(|(rhs_axis, _)| rhs_axis))
        .chain(rhs_kept)
        .collect();
    (adjoint, output_axes)
}
