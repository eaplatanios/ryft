use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

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
use crate::macros::{check_count, impl_non_transposable_operation};
use crate::operations::manipulation::{Broadcast, ConvertElementType, Reshape, Transpose};
use crate::operations::math::Mul;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::operations::{Operation, OperationFormatter};
use crate::programs::regions::RegionInterface;
use crate::programs::types::{TypeError, Typed};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{ArrayType, DataType, Shape, Size, StaticShape};

// TODO(eaplatanios): Review this module.

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

/// Returns whether `dimension` is sharded over at least one [`MeshAxisType::Explicit`] mesh axis of `mesh`.
///
/// The explicit-mode dot sharding rules — contracting-dimension ambiguity, batch-dimension consistency, and the
/// unreduced-operand rejection — apply only to Explicit axes. [`MeshAxisType::Manual`] axes are managed by the user
/// inside `shard_map` (a local per-shard dot over a manual-sharded contracting dimension is ordinary, not
/// ambiguous), and [`MeshAxisType::Auto`] axes are left to the compiler's propagation. A dimension sharded only over
/// those axis types therefore passes through these checks, mirroring how JAX gates its trace-time sharding rules to
/// Explicit mesh axes.
fn dimension_has_explicit_axis(mesh: &LogicalMesh, dimension: &ShardingDimension) -> bool {
    match dimension {
        ShardingDimension::Sharded(axis_names) => {
            axis_names.iter().any(|axis_name| mesh.axis_type(axis_name) == Some(MeshAxisType::Explicit))
        }
        ShardingDimension::Replicated | ShardingDimension::Unconstrained => false,
    }
}

/// Merges the [`ShardingDimension`]s of one aligned batch dimension pair, preferring the more informative entry
/// (`Sharded` over `Replicated` over `Unconstrained`). Returns [`None`] when the two entries are sharded over
/// different mesh axes, which the caller reports as an inconsistent-sharding error. Note that preferring a one-sided
/// `Sharded` entry over the other operand's entry intentionally diverges from JAX, which reads output batch
/// dimension specs from the LHS operand only.
fn merge_batch_sharding_dimensions(lhs: &ShardingDimension, rhs: &ShardingDimension) -> Option<ShardingDimension> {
    match (lhs, rhs) {
        (ShardingDimension::Sharded(left), ShardingDimension::Sharded(right)) if left == right => Some(lhs.clone()),
        (ShardingDimension::Sharded(_), ShardingDimension::Sharded(_)) => None,
        (ShardingDimension::Sharded(_), _) => Some(lhs.clone()),
        (_, ShardingDimension::Sharded(_)) => Some(rhs.clone()),
        (ShardingDimension::Replicated, _) | (_, ShardingDimension::Replicated) => Some(ShardingDimension::Replicated),
        (ShardingDimension::Unconstrained, ShardingDimension::Unconstrained) => Some(ShardingDimension::Unconstrained),
    }
}

/// Canonical operation name for [`DotOperation`].
pub const DOT_OPERATION_NAME: &str = "dot";

/// Computes the abstract output type of one generalized dot product.
///
/// The result shape is `[batching..., lhs_result..., rhs_result...]`, where the result dimensions are the operand
/// axes that are neither batching nor contracting, in their original order. The output element type is the LHS
/// element type (after a compatibility check with the RHS element type).
///
/// The output [`Sharding`] follows JAX's `dot_general` sharding rule (`_dot_general_sharding_rule` in
/// `jax/_src/lax/lax.py`); refer to the
/// [StableHLO `dot_general` specification](https://openxla.org/stablehlo/spec#dot_general) for the underlying
/// operation semantics. Concretely:
///
///   - When `output_sharding` is provided, it is validated (rank, mesh, no auto axes, and the unreduced-output rule
///     requiring identically sharded contracting dimensions whose sharding axes equal the requested unreduced set)
///     and returned directly, bypassing the consistency checks below.
///   - Operands must not be unreduced. Reduced operands are legal, and reduced and varying manual axes are unioned
///     across the operands into the output sharding.
///   - When neither operand carries a sharding, the output carries none. When exactly one does, the rule runs with
///     the absent side treated as fully replicated on the present operand's mesh. Operand meshes must match.
///   - Batch dimension entries are merged preferring the more informative entry (`Sharded` over `Replicated` over
///     `Unconstrained`); two different `Sharded` entries are an error. This intentionally diverges from JAX, which
///     reads batch dimension specs from the LHS operand only.
///   - Contracting dimensions sharded on both operands are an error (identically sharded ones make the output
///     sharding ambiguous and require an explicit output sharding); a contracting dimension sharded on only one
///     operand is allowed and its sharding is dropped from the output.
///   - Result dimension entries are copied from the owning operand, and auto mesh axes are stripped from the final
///     sharding.
/// Returns whether operands of `operand` element type may accumulate at `accumulation`: the identical type is
/// always valid, floating-point operands of any precision (including the sub-byte and 8-bit types that live outside
/// the standard promotion lattice) accumulate at `f32` or `f64` (the accumulator widths XLA's low-precision matrix
/// units expose), and integer operands accumulate at a same-signedness integer type at least as wide.
fn accumulation_type_is_compatible(operand: DataType, accumulation: DataType) -> bool {
    /// Returns the signedness and bit width of an integer data type, or `None` for any other type.
    fn integer_parts(data_type: DataType) -> Option<(bool, usize)> {
        Some(match data_type {
            DataType::I1 => (true, 1),
            DataType::I2 => (true, 2),
            DataType::I4 => (true, 4),
            DataType::I8 => (true, 8),
            DataType::I16 => (true, 16),
            DataType::I32 => (true, 32),
            DataType::I64 => (true, 64),
            DataType::U1 => (false, 1),
            DataType::U2 => (false, 2),
            DataType::U4 => (false, 4),
            DataType::U8 => (false, 8),
            DataType::U16 => (false, 16),
            DataType::U32 => (false, 32),
            DataType::U64 => (false, 64),
            _ => return None,
        })
    }

    if operand == accumulation {
        return true;
    }
    let operand_is_float = matches!(
        operand,
        DataType::F4E2M1FN
            | DataType::F6E2M3FN
            | DataType::F6E3M2FN
            | DataType::F8E3M4
            | DataType::F8E4M3
            | DataType::F8E4M3FN
            | DataType::F8E4M3FNUZ
            | DataType::F8E4M3B11FNUZ
            | DataType::F8E5M2
            | DataType::F8E5M2FNUZ
            | DataType::F8E8M0FNU
            | DataType::BF16
            | DataType::F16
            | DataType::F32
            | DataType::F64
    );
    if operand_is_float {
        return matches!(accumulation, DataType::F32 | DataType::F64);
    }
    match (integer_parts(operand), integer_parts(accumulation)) {
        (Some((operand_signed, operand_width)), Some((accumulation_signed, accumulation_width))) => {
            operand_signed == accumulation_signed && accumulation_width >= operand_width
        }
        _ => false,
    }
}

fn dot_abstract(
    lhs: &ArrayType,
    rhs: &ArrayType,
    dimensions: &DotDimensionNumbers,
    accumulation_type: Option<DataType>,
    output_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    if lhs.data_type() != rhs.data_type() {
        return Err(TypeError { message: format!("'{DOT_OPERATION_NAME}' input element types are incompatible") });
    }
    if let Some(accumulation_type) = accumulation_type {
        if output_sharding.is_some() {
            return Err(TypeError {
                message: format!(
                    "'{DOT_OPERATION_NAME}' does not support combining an accumulation type with a requested \
                     output sharding yet"
                ),
            });
        }
        if !accumulation_type_is_compatible(lhs.data_type(), accumulation_type) {
            return Err(TypeError {
                message: format!(
                    "'{DOT_OPERATION_NAME}' operand data type {} cannot accumulate at data type {accumulation_type}",
                    lhs.data_type(),
                ),
            });
        }
    }
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();

    if lhs_batching.len() != rhs_batching.len() {
        return Err(TypeError {
            message: format!("'{DOT_OPERATION_NAME}' batching dimensions have different lengths on the two operands"),
        });
    }
    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(TypeError {
            message: format!(
                "'{DOT_OPERATION_NAME}' contracting dimensions have different lengths on the two operands"
            ),
        });
    }
    if lhs_batching.iter().any(|axis| *axis >= lhs_rank) || lhs_contracting.iter().any(|axis| *axis >= lhs_rank) {
        return Err(TypeError { message: format!("'{DOT_OPERATION_NAME}' LHS dimension index out of bounds") });
    }
    if rhs_batching.iter().any(|axis| *axis >= rhs_rank) || rhs_contracting.iter().any(|axis| *axis >= rhs_rank) {
        return Err(TypeError { message: format!("'{DOT_OPERATION_NAME}' RHS dimension index out of bounds") });
    }

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "'{DOT_OPERATION_NAME}' batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
                ),
            });
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "'{DOT_OPERATION_NAME}' contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
                ),
            });
        }
    }

    let lhs_result = lhs_result_axes(dimensions, lhs_rank);
    let rhs_result = rhs_result_axes(dimensions, rhs_rank);

    let output_dimensions: Vec<Size> = lhs_batching
        .iter()
        .map(|axis| lhs.dimension(*axis as isize))
        .chain(lhs_result.iter().map(|axis| lhs.dimension(*axis as isize)))
        .chain(rhs_result.iter().map(|axis| rhs.dimension(*axis as isize)))
        .collect();

    // Output sharding (JAX's `_dot_general_sharding_rule`; see the rule summary above). When `output_sharding` is
    // provided it is validated and returned directly, bypassing the batch and contracting dimension consistency
    // checks (matching JAX's `out_sharding` behavior).
    let lhs_sharding = lhs.sharding();
    let rhs_sharding = rhs.sharding();

    // Operands unreduced over an Explicit axis are rejected on every path, including the explicit `output_sharding`
    // bypass: a pending cross-device reduction must be discharged (e.g., through a sharding constraint) before the
    // value is contracted. The check is gated to Explicit axes — a `shard_map` value unreduced over a Manual axis is
    // the user's to manage. Reduced operands are always legal; this is what lets adjoint dots consume reduced
    // cotangents.
    for sharding in [lhs_sharding, rhs_sharding].into_iter().flatten() {
        if sharding
            .unreduced_axes()
            .iter()
            .any(|axis_name| sharding.mesh().axis_type(axis_name) == Some(MeshAxisType::Explicit))
        {
            return Err(TypeError { message: format!("'{DOT_OPERATION_NAME}' operands cannot be unreduced") });
        }
    }

    let mesh = match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) if left.mesh() != right.mesh() => {
            return Err(TypeError {
                message: format!("'{DOT_OPERATION_NAME}' operand shardings must use the same mesh"),
            });
        }
        (Some(left), _) => Some(left.mesh()),
        (_, Some(right)) => Some(right.mesh()),
        (None, None) => None,
    };

    // A missing operand sharding is treated as fully replicated so that one-sided shardings still propagate.
    let dimension_of = |sharding: Option<&Sharding>, axis: usize| -> ShardingDimension {
        sharding.map_or(ShardingDimension::Replicated, |sharding| sharding.dimensions()[axis].clone())
    };

    let sharding = if let Some(output_sharding) = output_sharding {
        let output_rank = dimensions.lhs_batching_dimensions().len() + lhs_result.len() + rhs_result.len();
        if output_sharding.rank() != output_rank {
            return Err(TypeError {
                message: format!(
                    "'{DOT_OPERATION_NAME}' output sharding rank ({}) does not match the output rank ({output_rank})",
                    output_sharding.rank(),
                ),
            });
        }
        if let Some(mesh) = mesh
            && output_sharding.mesh() != mesh
        {
            return Err(TypeError {
                message: format!("'{DOT_OPERATION_NAME}' output sharding must use the same mesh as the operands"),
            });
        }
        let mut referenced_axes: Vec<&String> = output_sharding.unreduced_axes().iter().collect();
        referenced_axes.extend(output_sharding.reduced_axes());
        for dimension in output_sharding.dimensions() {
            if let ShardingDimension::Sharded(axis_names) = dimension {
                referenced_axes.extend(axis_names);
            }
        }
        if referenced_axes
            .iter()
            .any(|name| output_sharding.mesh().axis_type(name) == Some(MeshAxisType::Auto))
        {
            return Err(TypeError {
                message: format!("'{DOT_OPERATION_NAME}' output sharding cannot reference auto mesh axes"),
            });
        }
        // A requested unreduced output is a deferred all-reduce: contracting over a dimension sharded across mesh axes
        // leaves each device holding only a partial product-sum over its shard of the contraction, whose true value is
        // the cross-device sum. Marking the output unreduced over exactly those axes defers that sum to a later op
        // instead of materializing it here, so the request is valid only when it names precisely the axes that shard
        // the (identically sharded) contracting dimensions (JAX's `_dot_general_unreduced_rule`).
        if !output_sharding.unreduced_axes().is_empty() {
            let lhs_contracting_spec: Vec<ShardingDimension> = dimensions
                .lhs_contracting_dimensions()
                .iter()
                .map(|axis| dimension_of(lhs_sharding, *axis))
                .collect();
            let rhs_contracting_spec: Vec<ShardingDimension> = dimensions
                .rhs_contracting_dimensions()
                .iter()
                .map(|axis| dimension_of(rhs_sharding, *axis))
                .collect();
            if lhs_contracting_spec != rhs_contracting_spec {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' contracting dimensions must be sharded identically when the output sharding is unreduced"
                    ),
                });
            }
            let mut contracting_axes = BTreeSet::new();
            for dimension in &lhs_contracting_spec {
                if let ShardingDimension::Sharded(axis_names) = dimension {
                    contracting_axes.extend(axis_names.iter().cloned());
                }
            }
            if output_sharding.unreduced_axes() != &contracting_axes {
                return Err(TypeError {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                    ),
                });
            }
        }
        Some(output_sharding.clone())
    } else if let Some(mesh) = mesh {
        for (lhs_axis, rhs_axis) in
            dimensions.lhs_contracting_dimensions().iter().zip(dimensions.rhs_contracting_dimensions())
        {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);
            // Only Explicit-axis sharding of both contracting operands triggers the ambiguity/consistency errors;
            // Manual/Auto contracting shardings fall through (handled by `shard_map` / the compiler).
            let both_explicitly_sharded =
                dimension_has_explicit_axis(mesh, &left) && dimension_has_explicit_axis(mesh, &right);
            if both_explicitly_sharded {
                let (ShardingDimension::Sharded(left_axes), ShardingDimension::Sharded(right_axes)) = (&left, &right)
                else {
                    unreachable!("dimension_has_explicit_axis only returns true for sharded dimensions")
                };
                if left_axes != right_axes {
                    return Err(TypeError {
                        message: format!(
                            "'{DOT_OPERATION_NAME}' contracting dimensions must have consistent shardings, but got {left} and {right}"
                        ),
                    });
                }
                return Err(TypeError {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' contracting dimensions are sharded, making the output sharding ambiguous; request an \
                         explicit output sharding (e.g., one with unreduced axes) to resolve it"
                    ),
                });
            }
            // A contracting dimension sharded on only one operand (or only over Manual/Auto axes) is allowed and its
            // sharding is dropped from the output, matching JAX (the partitioner inserts the necessary communication).
        }

        let mut placement =
            Vec::with_capacity(dimensions.lhs_batching_dimensions().len() + lhs_result.len() + rhs_result.len());
        for (lhs_axis, rhs_axis) in
            dimensions.lhs_batching_dimensions().iter().zip(dimensions.rhs_batching_dimensions())
        {
            let left = dimension_of(lhs_sharding, *lhs_axis);
            let right = dimension_of(rhs_sharding, *rhs_axis);
            let merged = match merge_batch_sharding_dimensions(&left, &right) {
                Some(merged) => merged,
                // A batch-dimension conflict is an error only when an Explicit axis is involved; a conflict purely over
                // Manual/Auto axes drops to `Replicated` and is left to `shard_map` / the compiler.
                None if dimension_has_explicit_axis(mesh, &left) || dimension_has_explicit_axis(mesh, &right) => {
                    return Err(TypeError {
                        message: format!(
                            "'{DOT_OPERATION_NAME}' batching dimensions must have consistent shardings, but got {left} and {right}"
                        ),
                    });
                }
                None => ShardingDimension::Replicated,
            };
            placement.push(merged);
        }
        placement.extend(lhs_result.iter().map(|axis| dimension_of(lhs_sharding, *axis)));
        placement.extend(rhs_result.iter().map(|axis| dimension_of(rhs_sharding, *axis)));

        let merged_axes = |select: fn(&Sharding) -> &BTreeSet<String>| -> BTreeSet<String> {
            match (lhs_sharding, rhs_sharding) {
                (Some(left), Some(right)) => select(left).union(select(right)).cloned().collect(),
                (Some(left), None) => select(left).clone(),
                (None, Some(right)) => select(right).clone(),
                (None, None) => BTreeSet::new(),
            }
        };
        let reduced_axes = merged_axes(Sharding::reduced_axes);
        let varying_manual_axes = merged_axes(Sharding::varying_manual_axes);
        let sharding = Sharding::new(mesh.clone(), placement)
            .and_then(|output| output.with_reduced_axes(reduced_axes))
            .and_then(|output| output.with_varying_manual_axes(varying_manual_axes))
            .map_err(|error| TypeError {
                message: format!("'{DOT_OPERATION_NAME}' output sharding construction failed: {error}"),
            })?;
        Some(sharding.without_auto_axes())
    } else {
        None
    };

    ArrayType::new(accumulation_type.unwrap_or(lhs.data_type()), Shape::new(output_dimensions))
        .with_sharding(sharding)
        .map_err(|error| TypeError { message: error.to_string() })
}

/// Primitive representing a generalized dot (tensor contraction).
///
/// [`DotOperation`] is the unified primitive for matrix multiplication, batched matrix
/// multiplication, vector inner products, and arbitrary tensor contractions. It lowers to
/// StableHLO's `dot_general` op in the XLA backend.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct DotOperation {
    /// Contracting and batching dimension specification.
    dimensions: DotDimensionNumbers,

    /// Optional accumulation data type. Refer to the documentation of [`Self::with_accumulation_type`].
    accumulation_type: Option<DataType>,

    /// Optional requested output [`Sharding`]. Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl DotOperation {
    /// Creates a new [`DotOperation`] with the supplied dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions, accumulation_type: None, output_sharding: None }
    }

    /// Returns a [`DotOperation`] configured for standard rank-2 matrix multiplication.
    #[inline]
    pub fn matmul() -> Self {
        Self::new(DotDimensionNumbers::matmul())
    }

    /// Attaches a requested output [`Sharding`] to this operation, mirroring the `out_sharding` parameter of JAX's
    /// `dot_general`. When set, type inference validates the requested sharding (rank, mesh, no auto axes, and the
    /// unreduced-output rule) and uses it for the output instead of the inferred sharding, bypassing the batch and
    /// contracting dimension consistency checks. This is the only way to produce an output with unreduced axes
    /// (i.e., per-device partial results whose cross-device reduction is delayed).
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns a copy of this [`DotOperation`] with the provided accumulation data type. The operand element types
    /// must still match each other and must promote to the accumulation type, which becomes the output element
    /// type: the backend upcasts the operands and accumulates the contraction at the wider type (XLA's
    /// `preferred_element_type` contract, which is what its low-precision matrix units implement natively — e.g.,
    /// `f8 × f8 → f32` and `bf16 × bf16 → f32`). Accumulation-typed dots reject differentiation (differentiate a
    /// full-precision dot instead) and cannot yet be combined with a requested output sharding.
    #[inline]
    pub fn with_accumulation_type(mut self, accumulation_type: impl Into<Option<DataType>>) -> Self {
        self.accumulation_type = accumulation_type.into();
        self
    }

    /// Returns the optional accumulation data type. Refer to the documentation of
    /// [`Self::with_accumulation_type`].
    #[inline]
    pub fn accumulation_type(&self) -> Option<DataType> {
        self.accumulation_type
    }

    /// Returns the contracting and batching dimension specification.
    #[inline]
    pub fn dimensions(&self) -> &DotDimensionNumbers {
        &self.dimensions
    }

    /// Returns the requested output sharding, if any.
    #[inline]
    pub fn output_sharding(&self) -> Option<&Sharding> {
        self.output_sharding.as_ref()
    }
}

impl Display for DotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl Operation<ArrayType> for DotOperation {
    #[inline]
    fn name(&self) -> &'static str {
        DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![dot_abstract(
            &input_types[0],
            &input_types[1],
            &self.dimensions,
            self.accumulation_type,
            self.output_sharding.as_ref(),
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            if let Some(accumulation_type) = self.accumulation_type {
                operation.field("accumulation_type", &accumulation_type)?;
            }
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: Dot>> InterpretableOperation<C> for DotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        // The requested output sharding and accumulation type flow through the capability methods so that
        // interpretation over staging values (e.g., during program batching) preserves them; concrete values
        // ignore the sharding and upcast for the accumulation type. Type inference rejects combining the two.
        Ok(vec![match (&self.accumulation_type, &self.output_sharding) {
            (Some(accumulation_type), _) => {
                inputs[0].dot_with_accumulation_type(&inputs[1], &self.dimensions, *accumulation_type)
            }
            (None, Some(output_sharding)) => {
                inputs[0].dot_with_output_sharding(&inputs[1], &self.dimensions, output_sharding)
            }
            (None, None) => inputs[0].dot(&inputs[1], &self.dimensions),
        }])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

impl<C: Context<Type = ArrayType, Value: Broadcast>> crate::batching::BatchableOperation<C> for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C>>(
        &self,
        context: &BatchingContext<C>,
        _driver: &D,
        inputs: &[crate::batching::ArrayBatch<C::Value>],
    ) -> Result<Vec<crate::batching::ArrayBatch<C::Value>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        // Validate the common batch size across both operands (catching mismatched batched operands) before the
        // mixed arms consult it; a mixed operand pair always has at least one mapped operand.
        let axis_size = crate::batching::ArrayBatch::common_batch_size(inputs)?;
        // Mixed batched/unbatched: broadcast the replicated operand to gain a singleton batch
        // axis at position 0 (JAX's `matchaxis(0)` convention), then fall through to the
        // both-batched arm of `lift_dot_dimensions`.
        let mixed_axis_size = || axis_size.expect("a mapped input pins the batch size");
        let aligned_inputs: Vec<crate::batching::ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => {
                vec![inputs[0].clone(), inputs[1].broadcast(0, mixed_axis_size(), context.axis_sharding().clone())?]
            }
            (None, Some(_)) => {
                vec![inputs[0].broadcast(0, mixed_axis_size(), context.axis_sharding().clone())?, inputs[1].clone()]
            }
        };
        let aligned_axes: Vec<Option<usize>> = aligned_inputs.iter().map(|input| input.batch_axis_position()).collect();
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(&self.dimensions, aligned_axes[0], aligned_axes[1])
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: "'dot' batching failed to lift its dimension numbers for the aligned batch axes".to_string(),
            })?;
        let axis_sharding = crate::batching::ArrayBatch::sharding_for_inputs(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions)
            .with_accumulation_type(self.accumulation_type)
            .with_output_sharding(lift_output_sharding(self.output_sharding.as_ref(), output_axis, axis_sharding)?);
        lifted_op.interpret_with_batch_axes(context, &aligned_inputs, &[BatchAxis::from_optional_position(output_axis)])
    }
}

/// Forward-mode rule for [`DotOperation`]: the product rule for the contraction
/// `d(dot(a, b)) = dot(da, b) + dot(a, db)`. Each term holds the corresponding primal operand fixed on its original
/// contracting side, staged as an ordinary `Dot` whose dimension numbers and requested output sharding match the
/// primal, so the tangent dots match the primal dot exactly and stay capture-free.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DotOperation
where
    C::Operation: From<DotOperation>,
    C::Value: ConvertElementType + Dot + std::ops::Add<Output = C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        if self.accumulation_type.is_some() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "'{DOT_OPERATION_NAME}' with an accumulation type does not support differentiation; \
                     differentiate a full-precision dot instead"
                ),
            }
            .into());
        }
        let left = &inputs[0];
        let right = &inputs[1];
        let stage_dot = |left: &C::Value, right: &C::Value| match self.output_sharding() {
            Some(output_sharding) => left.dot_with_output_sharding(right, self.dimensions(), output_sharding),
            None => left.dot(right, self.dimensions()),
        };
        let primal = stage_dot(left.primal(), right.primal());
        let tangent_type = primal.r#type().tangent();
        let convert_to_tangent_type = |value: &C::Value| {
            if value.r#type().data_type() == tangent_type.data_type() {
                Ok(value.clone())
            } else {
                value.convert_element_type(tangent_type.data_type()).map_err(DifferentiationError::from)
            }
        };
        let left_term = left
            .tangent()
            .as_value()
            .map(|tangent| -> Result<_, DifferentiationError> {
                Ok(stage_dot(&convert_to_tangent_type(tangent)?, &convert_to_tangent_type(right.primal())?))
            })
            .transpose()?;
        let right_term = right
            .tangent()
            .as_value()
            .map(|tangent| -> Result<_, DifferentiationError> {
                Ok(stage_dot(&convert_to_tangent_type(left.primal())?, &convert_to_tangent_type(tangent)?))
            })
            .transpose()?;
        // Combine the surviving terms, falling back to a structural zero of the primal's type when both were dropped.
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(tangent_type), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for the primal [`DotOperation`]. A generalized dot is bilinear: it is linear in
/// each operand separately but not in both jointly, so in a valid pushforward exactly one operand is linear and the
/// other is a known runtime value. The known operand selects which adjoint the linear operand receives, reproducing
/// captured-factor transpose rules without first folding the known
/// operand into a captured factor: the known operand's value is read from the pullback through `operand_values` and
/// fed back into a primal `dot` with the adjoint dimension numbers.
///
///   - When the RHS operand is known, the forward map is `t ↦ dot(t, rhs; dimensions)` — the linear form modeled by
///     a right-factor dot — whose adjoint maps the output cotangent to `dot(cotangent, rhs; adjoint)` with
///     `adjoint = adjoint_dimensions_for_right_dot(dimensions, rhs_rank, lhs_rank)`. The LHS (linear) operand receives
///     that contribution and the RHS (known) operand receives a structural zero.
///   - When the LHS operand is known, the forward map is `t ↦ dot(lhs, t; dimensions)` — the linear form modeled by
///     a left-factor dot — whose adjoint maps the output cotangent to `dot(lhs, cotangent; adjoint)` with
///     `adjoint = adjoint_dimensions_for_left_dot(dimensions, lhs_rank)`. The RHS (linear) operand receives that
///     contribution and the LHS (known) operand receives a structural zero.
///
/// The adjoint dot's output sharding is pinned to the cotangent dual of the linear operand's sharding, matching the
/// captured-factor rules: the produced value *is* that operand's cotangent, so its sharding swaps the operand's
/// unreduced and reduced axes instead of being re-derived. A zero output cotangent stays a structural zero, and two
/// linear operands (a bilinear product that is not a linear map jointly) are rejected as unsupported.
impl<
    V: Value<Type = ArrayType>,
    O: Operation<ArrayType> + From<crate::operations::manipulation::ConvertElementTypeOperation> + From<DotOperation>,
> TransposableOperation<V, O> for DotOperation
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        check_count!("input", inputs, 2, ProgramError);
        if self.accumulation_type.is_some() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "'{DOT_OPERATION_NAME}' with an accumulation type does not support differentiation; \
                     differentiate a full-precision dot instead"
                ),
            }
            .into());
        }
        check_count!("output", outputs, 1, ProgramError);
        match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
            // Both operands linear is a bilinear product, which is not a linear map in both operands jointly and so
            // never appears in a valid pushforward.
            (true, true) => Err(ProgramError::UnsupportedOperation {
                message: "bilinear `dot` with two linear operands cannot be transposed".to_string(),
            }
            .into()),
            // Exactly one operand is linear: stage the adjoint dot reading the known operand's value, and emit a
            // structural zero for the known operand. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let linear_cotangent_type = inputs[linear_index].r#type().cotangent();
                let contribution = match &outputs[0] {
                    MaybeZero::Zero(_) => MaybeZero::Zero(linear_cotangent_type),
                    MaybeZero::Value(output_cotangent) => {
                        // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                        let known_value = inputs[known_index]
                            .as_known()
                            .expect("dispatch guarantees a known operand carries its pullback value");
                        let known_value = if known_value.r#type().data_type() == output_cotangent.r#type().data_type() {
                            known_value.clone()
                        } else {
                            known_value.convert_element_type(output_cotangent.r#type().data_type())?
                        };
                        let left_rank = inputs[0].r#type().rank();
                        let right_rank = inputs[1].r#type().rank();
                        let adjoint_output_sharding = inputs[linear_index].r#type().sharding().map(Sharding::cotangent);
                        let adjoint = if left_is_linear {
                            // Known RHS: linear LHS cotangent is `dot(cotangent, rhs; adjoint_right)`.
                            let dimensions = adjoint_dimensions_for_right_dot(&self.dimensions, right_rank, left_rank);
                            DotOperation::new(dimensions).with_output_sharding(adjoint_output_sharding)
                        } else {
                            // Known LHS: linear RHS cotangent is `dot(lhs, cotangent; adjoint_left)`.
                            let dimensions = adjoint_dimensions_for_left_dot(&self.dimensions, left_rank);
                            DotOperation::new(dimensions).with_output_sharding(adjoint_output_sharding)
                        };
                        let operands = if left_is_linear {
                            [output_cotangent.clone(), known_value]
                        } else {
                            [known_value, output_cotangent.clone()]
                        };
                        let mut outputs = context.stage_operation(adjoint, Vec::new(), &operands)?;
                        check_count!("output", outputs, 1, ProgramError);
                        MaybeZero::Value(outputs.remove(0))
                    }
                };
                let mut contributions = inputs
                    .iter()
                    .map(|input| {
                        let input_type = input.r#type();
                        MaybeZero::Zero(input_type.cotangent())
                    })
                    .collect::<Vec<_>>();
                contributions[linear_index] = contribution;
                Ok(contributions)
            }
        }
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
fn lift_output_sharding(
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

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It
/// performs the contraction described by `dimensions`, supporting standard matrix
/// multiplication, batched matrix multiplication, vector inner products, and arbitrary tensor
/// contractions.

/// Canonical operation name for [`ScaledDotOperation`].
pub const SCALED_DOT_OPERATION_NAME: &str = "scaled_dot";

/// Primitive representing a block-scaled ("microscaling") matrix product — the analogue of
/// [JAX's `jax.nn.scaled_matmul`](https://docs.jax.dev/en/latest/_autosummary/jax.nn.scaled_matmul.html). Each
/// operand pairs a narrow element tensor with a tensor of per-block scales along the contracting dimension,
/// covering the [OCP MX formats](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
/// (e.g., MXFP8: `f8e4m3fn` elements with `f8e8m0fnu` scales over blocks of 32) and NVIDIA's NVFP4 (`f4e2m1fn`
/// elements with `f8e4m3fn` scales over blocks of 16; fold any additional per-tensor scale into one scale tensor
/// or the result). The contracting dimension is the last dimension of *both* element operands, matching the
/// hardware block-scaled gemm convention: the operands are `lhs [m, k]` with scales `[m, k / block_size]` and
/// `rhs [n, k]` with scales `[n, k / block_size]`, and the result is `[m, n]` at the accumulation type.
///
/// Semantically the operation dequantizes both operands (upcasting elements and scales to the accumulation type
/// and expanding each scale across its block) and contracts them — which is exactly how the reference array
/// backend and the portable XLA lowering evaluate it (see [`scaled_dot_composition`]). On CUDA targets, the XLA
/// lowering instead emits the `__op$block_scaled_dot` custom call for the standard MXFP8/NVFP4 format and block
/// combinations, which XLA's GPU block-scaling rewriter lowers to cuDNN's native block-scaled tensor-core dot
/// (cuDNN 9.10+) or to expanded reference HLO.
///
/// Because quantized operands are inference-oriented, the operation rejects differentiation and batching with
/// errors directing users to differentiate or batch an explicit dequantization composition instead.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct ScaledDotOperation {
    /// Number of contracting-dimension elements sharing one scale.
    block_size: usize,

    /// Element type at which the operands are dequantized and the contraction accumulates.
    accumulation_type: DataType,
}

impl ScaledDotOperation {
    /// Creates a new [`ScaledDotOperation`] with the provided block size and accumulation type.
    #[inline]
    pub fn new(block_size: usize, accumulation_type: DataType) -> Self {
        Self { block_size, accumulation_type }
    }

    /// Returns the number of contracting-dimension elements sharing one scale.
    #[inline]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the element type at which the operands are dequantized and the contraction accumulates.
    #[inline]
    pub fn accumulation_type(&self) -> DataType {
        self.accumulation_type
    }
}

impl Display for ScaledDotOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Operation::<ArrayType>::render(self, formatter, 0)
    }
}

impl Operation<ArrayType> for ScaledDotOperation {
    #[inline]
    fn name(&self) -> &'static str {
        SCALED_DOT_OPERATION_NAME
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCALED_DOT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("block_size", &self.block_size)?;
            operation.field("accumulation_type", &self.accumulation_type)
        })
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 4, TypeError);
        let lhs = static_rank_2_dimensions("'scaled_dot' left operand", &input_types[0])?;
        let lhs_scales = static_rank_2_dimensions("'scaled_dot' left scales", &input_types[1])?;
        let rhs = static_rank_2_dimensions("'scaled_dot' right operand", &input_types[2])?;
        let rhs_scales = static_rank_2_dimensions("'scaled_dot' right scales", &input_types[3])?;
        if lhs[1] != rhs[1] {
            return Err(TypeError {
                message: format!(
                    "'scaled_dot' contracting dimension sizes do not match: {} versus {}",
                    lhs[1], rhs[1],
                ),
            });
        }
        if self.block_size == 0 || lhs[1] % self.block_size != 0 {
            return Err(TypeError {
                message: format!(
                    "'scaled_dot' contracting dimension size {} is not divisible by block size {}",
                    lhs[1], self.block_size,
                ),
            });
        }
        for (descriptor, elements, scales) in
            [("left", lhs, lhs_scales), ("right", rhs, rhs_scales)]
        {
            if scales != [elements[0], elements[1] / self.block_size] {
                return Err(TypeError {
                    message: format!(
                        "'scaled_dot' {descriptor} scales must have shape [{}, {}] but got [{}, {}]",
                        elements[0],
                        elements[1] / self.block_size,
                        scales[0],
                        scales[1],
                    ),
                });
            }
        }
        for input_type in input_types {
            if !accumulation_type_is_compatible(input_type.data_type(), self.accumulation_type) {
                return Err(TypeError {
                    message: format!(
                        "'scaled_dot' operand data type {} cannot accumulate at data type {}",
                        input_type.data_type(),
                        self.accumulation_type,
                    ),
                });
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError { message: "'scaled_dot' does not support unreduced operands".to_string() });
            }
        }
        Ok(vec![ArrayType::new(
            self.accumulation_type,
            Shape::new(vec![Size::Static(lhs[0]), Size::Static(rhs[0])]),
        )])
    }
}

impl<C: Domain<Type = ArrayType, Value: ScaledDot>> InterpretableOperation<C> for ScaledDotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        check_count!("input", inputs, 4, ProgramError);
        Ok(vec![inputs[0].scaled_dot(
            &inputs[1],
            &inputs[2],
            &inputs[3],
            self.block_size,
            self.accumulation_type,
        )?])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScaledDotOperation where
    C::Operation: From<ScaledDotOperation>
{
}

/// Quantized operands are inference-oriented, so there is no differentiation rule: differentiating reports an
/// error directing users to differentiate an explicit dequantization composition instead.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ScaledDotOperation
where
    C::Operation: From<ScaledDotOperation>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::UnsupportedOperation {
            message: format!(
                "'{SCALED_DOT_OPERATION_NAME}' does not support differentiation; differentiate an explicit \
                 dequantization composition instead"
            ),
        }
        .into())
    }
}

impl_non_transposable_operation!(ScaledDotOperation);

/// Quantized operands are inference-oriented, so there is no batching rule: batching reports an error directing
/// users to batch an explicit dequantization composition instead.
impl<C: Context<Type = ArrayType>> BatchableOperation<C> for ScaledDotOperation {
    fn batch<D: BatchingDriver<C>>(
        &self,
        _context: &BatchingContext<C>,
        _driver: &D,
        _inputs: &[ArrayBatch<C::Value>],
    ) -> Result<Vec<ArrayBatch<C::Value>>, BatchingError> {
        Err(BatchingError::UnsupportedOperation {
            message: format!(
                "'{SCALED_DOT_OPERATION_NAME}' does not support batching; batch an explicit dequantization \
                 composition instead"
            ),
        })
    }
}

/// Value-level block-scaled ("microscaling") dot capability. Refer to the documentation of [`ScaledDotOperation`]
/// for the operand convention (the contracting dimension is last on *both* element operands), the supported
/// formats, and the transform rules.
pub trait ScaledDot: Sized {
    /// Computes the block-scaled matrix product of `self` (shape `[m, k]`, scaled by `lhs_scales` of shape
    /// `[m, k / block_size]`) and `rhs` (shape `[n, k]`, scaled by `rhs_scales` of shape `[n, k / block_size]`),
    /// dequantizing both operands to `accumulation_type` and returning the `[m, n]` product at that type, and a
    /// [`ProgramError`] if something goes wrong.
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError>;
}

/// Any context-carrying value computes a block-scaled dot by binding a [`ScaledDotOperation`] through its own
/// context. The `From<ScaledDotOperation>` bound makes this disjoint from the eager reference value types (whose
/// context operation is [`ConstantOperation`](crate::operations::constants::ConstantOperation)), so it covers the
/// transform tracers and backend-owned values without conflicting with concrete implementations.
impl<V: Value<Type = ArrayType>> ScaledDot for V
where
    V::DispatchDomain: Context<Operation: From<ScaledDotOperation>>,
{
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ScaledDotOperation::new(block_size, accumulation_type),
            Vec::new(),
            &[self.clone(), lhs_scales.clone(), rhs.clone(), rhs_scales.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Evaluates a block-scaled dot as the portable dequantization composition: both operands (whose contracting
/// dimension is last) are upcast to the accumulation type, their scales are expanded across the blocks (a
/// broadcast inserting the block axis, merged back by a reshape), multiplied in, and contracted over the last
/// dimension of both sides. This is the shared semantics behind the concrete [`ScaledDot`] implementations and
/// the portable XLA lowering.
pub(crate) fn scaled_dot_composition<V>(
    lhs: &V,
    lhs_scales: &V,
    rhs: &V,
    rhs_scales: &V,
    block_size: usize,
    accumulation_type: DataType,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Dot + Mul + Reshape,
{
    let lhs = dequantize_block_scaled(lhs, lhs_scales, block_size, accumulation_type)?;
    let rhs = dequantize_block_scaled(rhs, rhs_scales, block_size, accumulation_type)?;
    Ok(lhs.dot(&rhs, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new())))
}

/// Dequantizes one block-scaled rank-2 operand whose contracting dimension is last: converts the elements and
/// scales to `accumulation_type`, expands each scale across its block of `block_size` contracting elements (a
/// broadcast appending the block axis, merged back by a reshape), and multiplies.
fn dequantize_block_scaled<V>(
    elements: &V,
    scales: &V,
    block_size: usize,
    accumulation_type: DataType,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Mul + Reshape,
{
    let element_type = elements.r#type().into_owned();
    let element_dimensions = static_rank_2_dimensions("'scaled_dot' operand", &element_type)?;
    let scale_dimensions = static_rank_2_dimensions("'scaled_dot' scales", &scales.r#type())?;
    if block_size == 0 || element_dimensions[1] % block_size != 0 {
        return Err(TypeError {
            message: format!(
                "'scaled_dot' contracting dimension size {} is not divisible by block size {block_size}",
                element_dimensions[1],
            ),
        }
        .into());
    }
    if scale_dimensions != [element_dimensions[0], element_dimensions[1] / block_size] {
        return Err(TypeError {
            message: format!(
                "'scaled_dot' scales must have shape [{}, {}] but got [{}, {}]",
                element_dimensions[0],
                element_dimensions[1] / block_size,
                scale_dimensions[0],
                scale_dimensions[1],
            ),
        }
        .into());
    }
    let expanded_type = ArrayType::new(
        accumulation_type,
        Shape::new(vec![
            Size::Static(scale_dimensions[0]),
            Size::Static(scale_dimensions[1]),
            Size::Static(block_size),
        ]),
    );
    let element_sizes = element_dimensions.iter().map(|&size| Size::Static(size)).collect::<Vec<_>>();
    let expanded_scales = scales
        .convert_element_type(accumulation_type)?
        .broadcast(expanded_type, &[0, 1])?
        .reshape(Shape::new(element_sizes))?;
    elements.convert_element_type(accumulation_type)?.mul(&expanded_scales)
}

/// Returns the static rank-2 dimensions of a [`ScaledDot`] operand type, rejecting other ranks and dynamic shapes.
fn static_rank_2_dimensions(descriptor: &str, value_type: &ArrayType) -> Result<[usize; 2], TypeError> {
    let Some(shape) = value_type.static_shape() else {
        return Err(TypeError { message: format!("{descriptor} must have a static shape") });
    };
    match shape.dimensions() {
        &[rows, columns] => Ok([rows, columns]),
        dimensions => Err(TypeError {
            message: format!("{descriptor} must have rank 2 but got rank {}", dimensions.len()),
        }),
    }
}

pub trait Dot<Rhs = Self>: Sized {
    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`.
    fn dot(&self, rhs: &Rhs, dimensions: &DotDimensionNumbers) -> Self;

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, requesting `output_sharding`
    /// for the result. The requested sharding overrides the inferred output sharding and is validated by the staged
    /// operation's type inference (refer to the documentation of [`DotOperation::with_output_sharding`]). The
    /// default implementation ignores the requested sharding and delegates to [`Self::dot`], which is correct for
    /// concrete (single-device) values, for which a sharding only describes distribution metadata; staging
    /// implementations override this method to attach the requested sharding to the staged operation.
    fn dot_with_output_sharding(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        let _ = output_sharding;
        self.dot(rhs, dimensions)
    }

    /// Computes the generalized dot product of `self` and `rhs` using `dimensions`, upcasting the operands to
    /// `accumulation_type` and accumulating the contraction there, so the result carries the accumulation type.
    /// Refer to the documentation of [`DotOperation::with_accumulation_type`] for the exact contract.
    fn dot_with_accumulation_type(
        &self,
        rhs: &Rhs,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self;
}

/// Any context-carrying value takes a dot product by binding a [`DotOperation`] through its own context. The
/// `From<DotOperation>` bound makes this disjoint from the eager value types (whose context operation is
/// `ConstantOperation`), so it covers the transform tracers without conflicting with the concrete implementations.
impl<V: Value<Type = ArrayType>> Dot for V
where
    V::DispatchDomain: Context<Type = ArrayType>,
    <V::DispatchDomain as Domain>::Operation: From<DotOperation>,
{
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        self.dispatch_domain()
            .bind(DotOperation::new(dimensions.clone()), Vec::new(), &[self.clone(), rhs.clone()])
            .expect("`dot` operation failed")
            .remove(0)
    }

    fn dot_with_accumulation_type(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        accumulation_type: DataType,
    ) -> Self {
        self.dispatch_domain()
            .bind(
                DotOperation::new(dimensions.clone()).with_accumulation_type(accumulation_type),
                Vec::new(),
                &[self.clone(), rhs.clone()],
            )
            .expect("`dot` operation failed")
            .remove(0)
    }

    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        self.dispatch_domain()
            .bind(
                DotOperation::new(dimensions.clone()).with_output_sharding(output_sharding.clone()),
                Vec::new(),
                &[self.clone(), rhs.clone()],
            )
            .expect("`dot` operation failed")
            .remove(0)
    }
}

/// Combined generalized dot product and transposition capability.
///
/// This convenience trait groups the value-level [`Dot`] and [`Transpose`] operations used by the unified
/// [`DotOperation`] and [`TransposeOperation`](crate::operations::manipulation::TransposeOperation) primitives.
pub trait DotOps: Dot + Transpose {}

impl<T: Dot + Transpose> DotOps for T {}

/// Generalized N-dimensional dot-product helper.
///
/// Implements StableHLO `dot_general` semantics over a flat row-major payload and an explicit
/// shape. Used by value-level [`Dot`] implementations for `Vec`-backed array types.
///
/// # Parameters
///
///   - `lhs`: Flat row-major payload of the left operand.
///   - `lhs_shape`: Shape of the left operand.
///   - `rhs`: Flat row-major payload of the right operand.
///   - `rhs_shape`: Shape of the right operand.
///   - `dimensions`: Contracting and batching dimension numbers.
///   - `accumulator_init`: Zero value of the accumulator type (called once per output element).
///   - `multiply_accumulate`: Accumulator update — `accumulator + lhs_value * rhs_value`.
pub fn dot_general_evaluate<T, FInit, FAcc>(
    lhs: &[T],
    lhs_shape: &StaticShape,
    rhs: &[T],
    rhs_shape: &StaticShape,
    dimensions: &DotDimensionNumbers,
    accumulator_init: FInit,
    multiply_accumulate: FAcc,
) -> (Vec<T>, StaticShape)
where
    T: Clone,
    FInit: Fn() -> T,
    FAcc: Fn(T, &T, &T) -> T,
{
    let lhs_batching = dimensions.lhs_batching_dimensions.as_slice();
    let rhs_batching = dimensions.rhs_batching_dimensions.as_slice();
    let lhs_contracting = dimensions.lhs_contracting_dimensions.as_slice();
    let rhs_contracting = dimensions.rhs_contracting_dimensions.as_slice();

    let lhs_result: Vec<usize> = (0..lhs_shape.rank())
        .filter(|axis| !lhs_batching.contains(axis) && !lhs_contracting.contains(axis))
        .collect();
    let rhs_result: Vec<usize> = (0..rhs_shape.rank())
        .filter(|axis| !rhs_batching.contains(axis) && !rhs_contracting.contains(axis))
        .collect();

    let batching_extents: Vec<usize> = lhs_batching.iter().map(|axis| lhs_shape[*axis]).collect();
    let lhs_result_extents: Vec<usize> = lhs_result.iter().map(|axis| lhs_shape[*axis]).collect();
    let rhs_result_extents: Vec<usize> = rhs_result.iter().map(|axis| rhs_shape[*axis]).collect();
    let contracting_extents: Vec<usize> = lhs_contracting.iter().map(|axis| lhs_shape[*axis]).collect();

    let output_shape = StaticShape::new(
        batching_extents
            .iter()
            .copied()
            .chain(lhs_result_extents.iter().copied())
            .chain(rhs_result_extents.iter().copied())
            .collect(),
    );
    let output_count: usize = output_shape.dimensions().iter().product();
    let mut output = Vec::with_capacity(output_count);
    if output_count == 0 {
        return (output, output_shape);
    }

    let lhs_strides = lhs_shape.row_major_strides();
    let rhs_strides = rhs_shape.row_major_strides();
    let mut lhs_index = vec![0usize; lhs_shape.rank()];
    let mut rhs_index = vec![0usize; rhs_shape.rank()];

    for_each_multi_index(batching_extents.as_slice(), |batching_index| {
        for (slot, axis) in lhs_batching.iter().enumerate() {
            lhs_index[*axis] = batching_index[slot];
        }
        for (slot, axis) in rhs_batching.iter().enumerate() {
            rhs_index[*axis] = batching_index[slot];
        }
        for_each_multi_index(lhs_result_extents.as_slice(), |lhs_result_index| {
            for (slot, axis) in lhs_result.iter().enumerate() {
                lhs_index[*axis] = lhs_result_index[slot];
            }
            for_each_multi_index(rhs_result_extents.as_slice(), |rhs_result_index| {
                for (slot, axis) in rhs_result.iter().enumerate() {
                    rhs_index[*axis] = rhs_result_index[slot];
                }
                let mut accumulator = accumulator_init();
                for_each_multi_index(contracting_extents.as_slice(), |contracting_index| {
                    for (slot, axis) in lhs_contracting.iter().enumerate() {
                        lhs_index[*axis] = contracting_index[slot];
                    }
                    for (slot, axis) in rhs_contracting.iter().enumerate() {
                        rhs_index[*axis] = contracting_index[slot];
                    }
                    let lhs_flat = flat_index(&lhs_index, &lhs_strides);
                    let rhs_flat = flat_index(&rhs_index, &rhs_strides);
                    accumulator = multiply_accumulate(accumulator.clone(), &lhs[lhs_flat], &rhs[rhs_flat]);
                });
                output.push(accumulator);
            });
        });
    });

    (output, output_shape)
}

fn flat_index(multi_index: &[usize], strides: &[usize]) -> usize {
    multi_index.iter().zip(strides.iter()).map(|(index, stride)| index * stride).sum()
}

fn for_each_multi_index(extents: &[usize], mut action: impl FnMut(&[usize])) {
    if extents.is_empty() {
        action(&[]);
        return;
    }
    let mut index = vec![0usize; extents.len()];
    loop {
        action(&index);
        let mut axis = extents.len();
        while axis > 0 {
            axis -= 1;
            index[axis] += 1;
            if index[axis] < extents[axis] {
                break;
            }
            index[axis] = 0;
            if axis == 0 {
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{Batch, BatchAxis};
    use crate::contexts::EagerContext;
    use crate::macros::check_operation_transposition;
    use crate::programs::operations::Operation;
    use crate::programs::types::TypeError;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("b", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("n", 2, MeshAxisType::Explicit).unwrap(),
            MeshAxis::new("k", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    fn plain_array(sizes: &[usize]) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(sizes.iter().map(|size| Size::Static(*size)).collect()))
    }

    fn sharded_array(mesh: &LogicalMesh, sizes: &[usize], dimensions: Vec<ShardingDimension>) -> ArrayType {
        plain_array(sizes).with_sharding(Sharding::new(mesh.clone(), dimensions).unwrap()).unwrap()
    }

    #[test]
    fn test_scaled_dot() {
        // NVFP4-flavored case (with a compact block size of 2 instead of 16): `f4e2m1fn` elements with `f8e4m3fn`
        // per-block scales along the trailing contracting dimension of BOTH operands (`lhs [m, k]`, `rhs [n, k]`).
        // Every element, scale, product, and partial sum below is exactly representable, so the `f32` result is
        // exact.
        let lhs_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Size::Static(2), Size::Static(4)]));
        let rhs_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Size::Static(2), Size::Static(4)]));
        let scale_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let lhs = Array::from_f64s(lhs_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let lhs_scales = Array::from_f64s(scale_type.clone(), vec![0.5, 2.0, 1.0, 0.5]);
        let rhs = Array::from_f64s(rhs_type.clone(), vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let rhs_scales = Array::from_f64s(scale_type.clone(), vec![2.0, 0.5, 1.0, 2.0]);
        let product = lhs.scaled_dot(&lhs_scales, &rhs, &rhs_scales, 2, DataType::F32).unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)])),
        );
        assert_eq!(product.to_f64s(), vec![6.75, 11.25, 10.375, 7.0]);

        // MXFP8-flavored case: `f8e4m3fn` elements with power-of-two `f8e8m0fnu` scales.
        let f8_lhs_type = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(4)]));
        let mx_scale_type = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let f8_lhs = Array::from_f64s(f8_lhs_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let f8_lhs_scales = Array::from_f64s(mx_scale_type.clone(), vec![0.5, 2.0, 1.0, 0.5]);
        let f8_rhs = Array::from_f64s(f8_lhs_type, vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let f8_rhs_scales = Array::from_f64s(mx_scale_type, vec![2.0, 0.5, 1.0, 2.0]);
        let product = f8_lhs.scaled_dot(&f8_lhs_scales, &f8_rhs, &f8_rhs_scales, 2, DataType::F32).unwrap();
        assert_eq!(product.to_f64s(), vec![6.75, 11.25, 10.375, 7.0]);

        // The staged operation renders its payload.
        let operation = ScaledDotOperation::new(2, DataType::F32);
        assert_eq!(operation.block_size(), 2);
        assert_eq!(operation.accumulation_type(), DataType::F32);
        assert_eq!(operation.name(), SCALED_DOT_OPERATION_NAME);
        assert_eq!(operation.to_string(), "scaled_dot [block_size=2, accumulation_type=f32]");
        assert_eq!(
            operation.infer_output_types(
                &[lhs_type.clone(), scale_type.clone(), rhs_type.clone(), scale_type.clone()],
                &[],
            ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]))]),
        );

        // Contract violations report clear errors through type inference.
        assert_eq!(
            ScaledDotOperation::new(3, DataType::F32).infer_output_types(
                &[lhs_type.clone(), scale_type.clone(), rhs_type.clone(), scale_type.clone()],
                &[],
            ),
            Err(TypeError {
                message: "'scaled_dot' contracting dimension size 4 is not divisible by block size 3".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs_type.clone(), rhs_type.clone(), rhs_type, scale_type.clone()], &[]),
            Err(TypeError {
                message: "'scaled_dot' left scales must have shape [2, 2] but got [2, 4]".to_string(),
            }),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    lhs_type.clone(),
                    scale_type.clone(),
                    ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Size::Static(4)])),
                    scale_type.clone(),
                ],
                &[],
            ),
            Err(TypeError {
                message: "'scaled_dot' right operand must have rank 2 but got rank 1".to_string(),
            }),
        );

        // Differentiation and batching are rejected with errors directing to explicit dequantization.
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(lhs_type),
            builder.add_input(scale_type.clone()),
            builder.add_input(ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Size::Static(2), Size::Static(4)]))),
            builder.add_input(scale_type),
        ];
        let output = builder.add_instruction(operation, Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![output],
                vec![crate::parameters::Placeholder; 4],
                vec![crate::parameters::Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error) if error.to_string().contains("'scaled_dot' does not support differentiation"),
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                crate::batching::ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(error) if error.to_string().contains("'scaled_dot' does not support batching"),
        ));
    }

    #[test]
    fn test_dot_accumulation_type() {
        // Type inference widens the output to the accumulation type for promotable operand types and rejects
        // non-promotable ones, combining with a requested output sharding, and differentiation.
        let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
        assert_eq!(operation.accumulation_type(), Some(DataType::F32));
        let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        let rhs = lhs.clone();
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]))]),
        );
        let bf16_operand = ArrayType::new(DataType::BF16, Shape::new(vec![Size::Static(2), Size::Static(2)]));
        assert_eq!(
            operation.infer_output_types(&[bf16_operand.clone(), bf16_operand.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]))]),
        );
        let narrowing = DotOperation::matmul().with_accumulation_type(DataType::F16);
        let f32_operand = plain_array(&[2, 2]);
        assert_eq!(
            narrowing.infer_output_types(&[f32_operand.clone(), f32_operand.clone()], &[]),
            Err(TypeError { message: "'dot' operand data type f32 cannot accumulate at data type f16".to_string() }),
        );
        let mesh = test_mesh();
        let sharded = DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]).unwrap(),
        );
        assert_eq!(
            sharded.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Err(TypeError {
                message: "'dot' does not support combining an accumulation type with a requested output sharding \
                          yet"
                .to_string(),
            }),
        );

        // The eager reference backend upcasts the operands and accumulates at the accumulation type: every value
        // below is exactly representable in `f8e4m3fn`, so the `f32` results are exact.
        let lhs_values = Array::from_f64s(lhs.clone(), vec![0.5, 1.0, 1.5, 2.0]);
        let rhs_values = Array::from_f64s(rhs.clone(), vec![1.0, 0.5, 0.5, 1.0]);
        let product = lhs_values.dot_with_accumulation_type(&rhs_values, &DotDimensionNumbers::matmul(), DataType::F32);
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Size::Static(2), Size::Static(2)]))
        );
        assert_eq!(product.to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);

        // Differentiating an accumulation-typed dot is rejected with a message directing to full precision.
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs);
        let rhs_input = builder.add_input(rhs);
        let output = builder
            .add_instruction(
                DotOperation::matmul().with_accumulation_type(DataType::F32),
                Vec::new(),
                vec![lhs_input, rhs_input],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![output],
                vec![crate::parameters::Placeholder; 2],
                vec![crate::parameters::Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.jvp(),
            Err(error) if error.to_string().contains("does not support differentiation"),
        ));

        // Batching lifts the dimension numbers while carrying the accumulation type, so per-item products still
        // accumulate at the widened type.
        let lifted = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                crate::batching::ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .0;
        let batched_lhs_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)]));
        let batched_lhs = Array::from_f64s(batched_lhs_type.clone(), vec![0.5, 1.0, 1.5, 2.0, 0.5, 1.0, 1.5, 2.0]);
        let batched_rhs = Array::from_f64s(batched_lhs_type, vec![1.0, 0.5, 0.5, 1.0, 1.0, 0.5, 0.5, 1.0]);
        let outputs = lifted.interpret(vec![batched_lhs, batched_rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().data_type(), DataType::F32);
        // Both batch items repeat the unbatched case, whose exact product is [[1, 1.25], [2.5, 2.75]].
        assert_eq!(outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75, 1.0, 1.25, 2.5, 2.75]);
    }

    #[test]
    fn test_dot_inference_with_dynamic_dimensions() {
        // Batched matrix multiplication contracting axis 2 of the LHS with axis 1 of the RHS over batching axis 0.
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));

        // Dynamic dimension sizes that compare equal flow through inference into the output type: the dynamic
        // batching dimension is preserved and the equal bounded dynamic contracting dimensions are dropped.
        let lhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(None), Size::Static(2), Size::Dynamic(Some(4))]),
        );
        let rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(None), Size::Dynamic(Some(4)), Size::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Ok(vec![ArrayType::new(
                DataType::F64,
                Shape::new(vec![Size::Dynamic(None), Size::Static(2), Size::Static(3)]),
            )]),
        );

        // Static-vs-dynamic and unequal dynamic dimension pairs keep erroring under the strict size equality used
        // for batching and contracting dimensions.
        let static_rhs =
            ArrayType::new(DataType::F64, Shape::new(vec![Size::Dynamic(None), Size::Static(4), Size::Static(3)]));
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), static_rhs], &[]),
            Err(TypeError {
                message: "'dot' contracting dimension sizes do not match (LHS axis 2, RHS axis 1)".to_string(),
            }),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(Some(8)), Size::Dynamic(Some(4)), Size::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, mismatched_batch_rhs], &[]),
            Err(TypeError {
                message: "'dot' batching dimension sizes do not match (LHS axis 0, RHS axis 0)".to_string(),
            }),
        );
    }

    #[test]
    fn test_dot_inference_batched_sharding_propagation() {
        let mesh = test_mesh();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));
        // The batch dimension merges the more informative entry (LHS `b` over RHS replicated), and the result
        // dimensions are copied from their owning operands.
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::replicated(), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[2, 4, 16],
                vec![
                    ShardingDimension::sharded(["b"]),
                    ShardingDimension::sharded(["m"]),
                    ShardingDimension::sharded(["n"]),
                ],
            )]),
        );
    }

    #[test]
    fn test_dot_inference_matmul_sharding_propagation() {
        let mesh = test_mesh();
        let operation = DotOperation::matmul();
        // Fully replicated operands stay fully replicated.
        let replicated_lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[replicated_lhs, replicated_rhs], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            )]),
        );

        // `[M@m, K] · [K, N@n] -> [M@m, N@n]`.
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["n"])]);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])],
            )]),
        );
    }

    #[test]
    fn test_dot_inference_one_sided_sharding_propagation() {
        let mesh = test_mesh();
        let operation = DotOperation::matmul();
        // A missing operand sharding is treated as fully replicated on the present operand's mesh.
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs, plain_array(&[8, 16])], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
            )]),
        );
        // Without any operand shardings, the output carries none.
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])], &[]),
            Ok(vec![plain_array(&[4, 16])]),
        );
    }

    #[test]
    fn test_dot_inference_batch_sharding_conflict() {
        let mesh = test_mesh();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]));
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Err(TypeError {
                message: "'dot' batching dimensions must have consistent shardings, but got {'b'} and {'m'}"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_dot_inference_contracting_sharding_errors() {
        let mesh = test_mesh();
        let operation = DotOperation::matmul();
        // Identically sharded contracting dimensions make the output sharding ambiguous.
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs], &[]),
            Err(TypeError {
                message: "'dot' contracting dimensions are sharded, making the output sharding ambiguous; request an \
                          explicit output sharding (e.g., one with unreduced axes) to resolve it"
                    .to_string(),
            }),
        );
        // Differently sharded contracting dimensions are inconsistent.
        let mismatched_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), mismatched_rhs], &[]),
            Err(TypeError {
                message: "'dot' contracting dimensions must have consistent shardings, but got {'k'} and {'m'}"
                    .to_string(),
            }),
        );
        // A contracting dimension sharded on only one operand is allowed, and its sharding is dropped.
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs, replicated_rhs], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            )]),
        );
    }

    #[test]
    fn test_dot_inference_mesh_mismatch() {
        let mesh = test_mesh();
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let operation = DotOperation::matmul();
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        let rhs = sharded_array(
            &other_mesh,
            &[8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Err(TypeError { message: "'dot' operand shardings must use the same mesh".to_string() }),
        );
    }

    #[test]
    fn test_dot_inference_unreduced_and_reduced_operands() {
        let mesh = test_mesh();
        let operation = DotOperation::matmul();
        // Unreduced operands are rejected: the pending reduction must be discharged before the contraction.
        let unreduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["k"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[unreduced_lhs, plain_array(&[8, 16])], &[]),
            Err(TypeError { message: "'dot' operands cannot be unreduced".to_string() }),
        );

        // Reduced operands are legal (this is what lets adjoint dots consume reduced cotangents), and their reduced
        // axes are unioned into the output sharding.
        let reduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["k"])
                    .unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[reduced_lhs, plain_array(&[8, 16])], &[]),
            Ok(vec![
                plain_array(&[4, 16])
                    .with_sharding(
                        Sharding::new(mesh, vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
                            .unwrap()
                            .with_reduced_axes(["k"])
                            .unwrap(),
                    )
                    .unwrap()
            ]),
        );
    }

    #[test]
    fn test_dot_inference_strips_auto_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("m", 2, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap();
        let operation = DotOperation::matmul();
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])]);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::replicated(), ShardingDimension::sharded(["m"])],
            )]),
        );
    }

    #[test]
    fn test_dot_inference_output_sharding_bypass_and_validation() {
        let mesh = test_mesh();
        // The requested output sharding bypasses the batch consistency checks (here, conflicting batch shardings).
        let lhs = sharded_array(
            &mesh,
            &[2, 4, 8],
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        let rhs = sharded_array(
            &mesh,
            &[2, 8, 16],
            vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated(), ShardingDimension::replicated()],
        );
        let requested = Sharding::new(
            mesh.clone(),
            vec![ShardingDimension::sharded(["b"]), ShardingDimension::replicated(), ShardingDimension::sharded(["n"])],
        )
        .unwrap();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(requested.clone());
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Ok(vec![plain_array(&[2, 4, 16]).with_sharding(requested).unwrap()]),
        );

        // Rank validation.
        let rank_mismatched = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(rank_mismatched);
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Err(TypeError { message: "'dot' output sharding rank (1) does not match the output rank (3)".to_string() }),
        );

        // Mesh validation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(other_mesh_sharding);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Err(TypeError { message: "'dot' output sharding must use the same mesh as the operands".to_string() }),
        );

        // Auto mesh axes cannot be requested explicitly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_sharding =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(auto_sharding);
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])], &[]),
            Err(TypeError { message: "'dot' output sharding cannot reference auto mesh axes".to_string() }),
        );
    }

    #[test]
    fn test_dot_inference_unreduced_output_sharding() {
        let mesh = test_mesh();
        let lhs =
            sharded_array(&mesh, &[4, 8], vec![ShardingDimension::replicated(), ShardingDimension::sharded(["k"])]);
        let rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["k"]), ShardingDimension::replicated()]);
        // Identically sharded contracting dimensions plus a matching unreduced set produce an unreduced output.
        let unreduced =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["k"])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Ok(vec![plain_array(&[4, 16]).with_sharding(unreduced.clone()).unwrap()]),
        );

        // The contracting dimensions must be sharded identically.
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), replicated_rhs.clone()], &[]),
            Err(TypeError {
                message:
                    "'dot' contracting dimensions must be sharded identically when the output sharding is unreduced"
                        .to_string(),
            }),
        );

        // The unreduced set must equal the axes that shard the contracting dimensions.
        let mismatched =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::replicated()])
                .unwrap()
                .with_unreduced_axes(["n"])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(mismatched);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Err(TypeError {
                message:
                    "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                        .to_string(),
            }),
        );

        // Unsharded contracting dimensions cannot produce an unreduced output.
        let operation = DotOperation::matmul().with_output_sharding(unreduced);
        assert_eq!(
            operation.infer_output_types(
                &[
                    replicated_rhs.clone(),
                    sharded_array(
                        &mesh,
                        &[16, 4],
                        vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    )
                ],
                &[]
            ),
            Err(TypeError {
                message:
                    "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_dot_operation_output_sharding_builder_and_render() {
        let mesh = test_mesh();
        let sharding =
            Sharding::new(mesh, vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(sharding.clone());
        assert_eq!(operation.output_sharding(), Some(&sharding));
        assert_eq!(DotOperation::matmul().output_sharding(), None);
        // The output sharding is rendered only when present.
        assert!(!DotOperation::matmul().to_string().contains("output_sharding="));
        assert!(operation.to_string().contains(&format!("output_sharding={sharding}")));
    }

    #[test]
    fn test_dot_batching_stages_the_lifted_output_sharding() {
        use std::rc::Rc;

        use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext};
        use crate::parameters::Placeholder;
        use crate::tracing::TracingContext;

        let mesh = test_mesh();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(output_sharding.clone());

        // Batch the operation over tracer inputs, which is how program batching applies lifted operations: the
        // staged batched dot must carry the lifted output sharding instead of dropping it.
        let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
        let builder = context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 4, 8]));
        let rhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 8, 16]));
        let batching_context = BatchingContext::new(context.clone(), 2);
        let lhs = {
            let value = context.tracer(lhs_atom, None);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let rhs = {
            let value = context.tracer(rhs_atom, None);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0))
        }
        .unwrap();
        let outputs = operation.batch(&batching_context, &crate::EmptyRegionDriver, &[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        let output_atom = outputs[0].value().atom_id().unwrap();
        drop(outputs);
        drop(batching_context);
        drop(context);

        let builder = Rc::try_unwrap(builder).expect("batching should not hold on to the builder").into_inner();
        let program = builder
            .build::<Vec<ArrayType>, Vec<ArrayType>>(vec![output_atom], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let lifted_sharding = output_sharding.with_inserted_dimension(0, ShardingDimension::replicated()).unwrap();
        assert!(program.to_string().contains(&format!("output_sharding={lifted_sharding}")));
    }

    #[test]
    fn test_dot_batching_preserves_materialized_batch_placement() {
        use std::rc::Rc;

        use crate::backends::arrays::{Array, ArrayOperation};
        use crate::batching::{ArrayBatch, BatchAxis, BatchableOperation, BatchingContext};
        use crate::parameters::Placeholder;
        use crate::tracing::TracingContext;

        for axis_type in [MeshAxisType::Explicit, MeshAxisType::Manual] {
            let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, axis_type).unwrap()]).unwrap();
            let lhs_sharding = Sharding::new(
                mesh.clone(),
                vec![
                    ShardingDimension::sharded(["x"]),
                    ShardingDimension::replicated(),
                    ShardingDimension::replicated(),
                ],
            )
            .unwrap()
            .with_varying_manual_axes((axis_type == MeshAxisType::Manual).then_some("x"))
            .unwrap();
            let lhs_type =
                ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(2), Size::Static(2)]))
                    .with_sharding(lhs_sharding)
                    .unwrap();
            let rhs_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(1)]))
                .with_sharding(Sharding::replicated(mesh, 2))
                .unwrap();
            let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
            let builder = parent.builder().clone();
            let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
            let rhs_atom = builder.borrow_mut().add_input(rhs_type);
            let lhs = ArrayBatch::new(lhs_type, parent.tracer(lhs_atom, None), BatchAxis::new(0)).unwrap();
            let rhs = ArrayBatch::replicated(parent.tracer(rhs_atom, None));
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = DotOperation::matmul().batch(&context, &crate::EmptyRegionDriver, &[lhs, rhs]).unwrap();

            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
            let output_atom = outputs[0].value().atom_id().unwrap();
            drop(outputs);
            drop(context);
            drop(parent);

            let builder = Rc::try_unwrap(builder).expect("batching should not retain the tracing builder").into_inner();
            let program = builder
                .build::<Vec<Array>, Vec<Array>>(vec![output_atom], vec![Placeholder, Placeholder], vec![Placeholder])
                .unwrap();
            assert_eq!(
                program.output_types()[0].sharding().unwrap().dimensions(),
                &[ShardingDimension::sharded(["x"]), ShardingDimension::replicated(), ShardingDimension::replicated(),],
            );
        }
    }

    #[test]
    fn test_dot_batching_lifts_dimension_numbers() {
        // x has shape [3, 4]; outer batch over axis 0 produces per-item rank-1 vectors. Inside,
        // we want every per-item vector dotted with itself, giving a per-item scalar; batch
        // over the leading axis then yields a length-3 vector of dot products.
        let x_data: Vec<f64> = (1..=12).map(|value| value as f64).collect();
        let x = Array::matrix(3, 4, x_data);

        let output: Array = EagerContext::<Array, ArrayOperation<Array>>::new()
            .batch(
                |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())),
                x,
                BatchAxis::new(0),
                BatchAxis::new(0),
                None,
            )
            .unwrap();

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)])),);
        // Batch item 0: [1,2,3,4]·[1,2,3,4] = 30. Batch item 1: [5,6,7,8]·[5,6,7,8] = 174. Batch item 2: 446.
        for (actual, expected) in output.to_f64s().iter().zip([30.0_f64, 174.0, 446.0].iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_dot_partitioned_transpose_computes_operand_adjoints() {
        let matmul = DotDimensionNumbers::matmul();
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let cotangent = Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]);
        check_operation_transposition!(
            @exact,
            operation = DotOperation::new(matmul),
            cases = [
                {
                    inputs = [
                        (@known, left),
                        (@linear(type = right.r#type().into_owned())),
                    ],
                    output_cotangents = [cotangent.clone()],
                    input_cotangents = [Array::matrix(3, 2, vec![3.0, 10.0, 4.5, 11.0, 6.0, 12.0])],
                },
                {
                    inputs = [
                        (@linear(type = ArrayType::new(
                            DataType::F64,
                            Shape::new(vec![Size::Static(2), Size::Static(3)]),
                        ))),
                        (@known, right),
                    ],
                    output_cotangents = [cotangent],
                    input_cotangents = [Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5])],
                },
            ],
        );
    }
}
