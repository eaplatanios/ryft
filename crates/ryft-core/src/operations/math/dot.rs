use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayType, DataType, Dimension, LogicalMesh, MeshAxisType, Shape,
    Sharding, ShardingDimension,
};
use crate::batching::{
    BatchAxis, BatchableOperation, BatchedOutputs, BatchingContext, BatchingDriver, BatchingError,
    InterpretableBatchableOperation,
};
use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
    TransposableOperation, TranspositionDriver,
};
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::check_count;
use crate::operations::constants::fill::Fill;
use crate::operations::manipulation::broadcasting::{Broadcast, BroadcastOperation};
use crate::operations::manipulation::conversion::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::reshaping::{Reshape, ReshapeOperation};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::abs::Abs;
use crate::operations::math::clamp::Clamp;
use crate::operations::math::div::Div;
use crate::operations::math::exp::Exp;
use crate::operations::math::floor::Floor;
use crate::operations::math::log::Log;
use crate::operations::math::max::Max;
use crate::operations::math::mul::{Mul, MulOperation};
use crate::operations::math::reduce::{Reduce, ReductionKind};
use crate::operations::math::sub::Sub;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value,
};
use crate::tracing::{Tracer, TracingContext};

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
    if operand.is_floating_point() {
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
        return Err(TypeError::invalid(format!("'{DOT_OPERATION_NAME}' input element types are incompatible")));
    }
    if let Some(accumulation_type) = accumulation_type {
        if output_sharding.is_some() {
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' does not support combining an accumulation type with a requested \
                     output sharding yet",
            )));
        }
        if !accumulation_type_is_compatible(lhs.data_type(), accumulation_type) {
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' operand data type {} cannot accumulate at data type {accumulation_type}",
                lhs.data_type(),
            )));
        }
    }
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();

    if lhs_batching.len() != rhs_batching.len() {
        return Err(TypeError::invalid(format!(
            "'{DOT_OPERATION_NAME}' batching dimensions have different lengths on the two operands"
        )));
    }
    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(TypeError::invalid(format!(
            "'{DOT_OPERATION_NAME}' contracting dimensions have different lengths on the two operands"
        )));
    }
    if lhs_batching.iter().any(|axis| *axis >= lhs_rank) || lhs_contracting.iter().any(|axis| *axis >= lhs_rank) {
        return Err(TypeError::invalid(format!("'{DOT_OPERATION_NAME}' LHS dimension index out of bounds")));
    }
    if rhs_batching.iter().any(|axis| *axis >= rhs_rank) || rhs_contracting.iter().any(|axis| *axis >= rhs_rank) {
        return Err(TypeError::invalid(format!("'{DOT_OPERATION_NAME}' RHS dimension index out of bounds")));
    }

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching.iter()) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
            )));
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting.iter()) {
        if lhs.dimension(*lhs_axis) != rhs.dimension(*rhs_axis) {
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
            )));
        }
    }

    let lhs_result = lhs_result_axes(dimensions, lhs_rank);
    let rhs_result = rhs_result_axes(dimensions, rhs_rank);

    let output_dimensions: Vec<Dimension> = lhs_batching
        .iter()
        .map(|axis| lhs.dimension(*axis))
        .chain(lhs_result.iter().map(|axis| lhs.dimension(*axis)))
        .chain(rhs_result.iter().map(|axis| rhs.dimension(*axis)))
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
            return Err(TypeError::invalid(format!("'{DOT_OPERATION_NAME}' operands cannot be unreduced")));
        }
    }

    let mesh = match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) if left.mesh() != right.mesh() => {
            return Err(TypeError::invalid(format!("'{DOT_OPERATION_NAME}' operand shardings must use the same mesh")));
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
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' output sharding rank ({}) does not match the output rank ({output_rank})",
                output_sharding.rank(),
            )));
        }
        if let Some(mesh) = mesh
            && output_sharding.mesh() != mesh
        {
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' output sharding must use the same mesh as the operands"
            )));
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
            return Err(TypeError::invalid(format!(
                "'{DOT_OPERATION_NAME}' output sharding cannot reference auto mesh axes"
            )));
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
                return Err(TypeError::invalid(format!(
                    "'{DOT_OPERATION_NAME}' contracting dimensions must be sharded identically when the output sharding is unreduced"
                )));
            }
            let mut contracting_axes = BTreeSet::new();
            for dimension in &lhs_contracting_spec {
                if let ShardingDimension::Sharded(axis_names) = dimension {
                    contracting_axes.extend(axis_names.iter().cloned());
                }
            }
            if output_sharding.unreduced_axes() != &contracting_axes {
                return Err(TypeError::invalid(format!(
                    "'{DOT_OPERATION_NAME}' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                )));
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
                    return Err(TypeError::invalid(format!(
                        "'{DOT_OPERATION_NAME}' contracting dimensions must have consistent shardings, but got {left} and {right}"
                    )));
                }
                return Err(TypeError::invalid(format!(
                    "'{DOT_OPERATION_NAME}' contracting dimensions are sharded, making the output sharding ambiguous; request an \
                         explicit output sharding (e.g., one with unreduced axes) to resolve it"
                )));
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
                    return Err(TypeError::invalid(format!(
                        "'{DOT_OPERATION_NAME}' batching dimensions must have consistent shardings, but got {left} and {right}"
                    )));
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
            .map_err(|error| {
                TypeError::invalid(format!("'{DOT_OPERATION_NAME}' output sharding construction failed: {error}"))
            })?;
        Some(sharding.without_auto_axes())
    } else {
        None
    };

    ArrayType::new(accumulation_type.unwrap_or(lhs.data_type()), Shape::new(output_dimensions))
        .with_sharding(sharding)
        .map_err(|error| TypeError::invalid(error.to_string()))
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
    /// `f8 × f8 → f32` and `bf16 × bf16 → f32`). Accumulation-typed dots differentiate like ordinary dots, with
    /// tangents and cotangents carried at the accumulation type (refer to the forward-mode and transpose rule
    /// documentation on this operation), and cannot yet be combined with a requested output sharding.
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

impl Operation for DotOperation {
    type Type = ArrayType;

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

impl<C: Context<Type = ArrayType, Value: Broadcast>, P: ArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
    for DotOperation
where
    DotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis_position()).collect();
        // Validate the common batch size across both operands (catching mismatched batched operands) before the
        // mixed arms consult it; a mixed operand pair always has at least one mapped operand.
        let axis_size = ArrayBatch::common_batch_size(inputs)?;
        // Mixed batched/unbatched: broadcast the replicated operand to gain a singleton batch
        // axis at position 0 (JAX's `matchaxis(0)` convention), then fall through to the
        // both-batched arm of `lift_dot_dimensions`.
        let mixed_axis_size = || axis_size.expect("a mapped input pins the batch size");
        let aligned_inputs: Vec<ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
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
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions)
            .with_accumulation_type(self.accumulation_type)
            .with_output_sharding(lift_output_sharding(self.output_sharding.as_ref(), output_axis, axis_sharding)?);
        Ok(lifted_op
            .interpret_with_batch_axes(context, &aligned_inputs, &[BatchAxis::from_optional_position(output_axis)])?
            .into())
    }
}

/// Forward-mode rule for [`DotOperation`]: the product rule for the contraction
/// `d(dot(a, b)) = dot(da, b) + dot(a, db)`. Each term holds the corresponding primal operand fixed on its original
/// contracting side, staged as an ordinary `Dot` whose dimension numbers, accumulation type, and requested output
/// sharding match the primal, so the tangent dots match the primal dot exactly and stay capture-free. For an
/// accumulation-typed dot the tangent terms stay accumulation-typed dots over the operand-typed tangents whenever a
/// term's operand element types agree (the common case, because every low-precision floating-point type except
/// `f8e8m0fnu` is its own tangent representation), so the output tangent lives at the accumulation type exactly like
/// the primal output; when a tangent arrives at a widened representation instead, both term operands are converted
/// to the output tangent element type first.
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
        let left = &inputs[0];
        let right = &inputs[1];
        let stage_dot = |left: &C::Value, right: &C::Value| match (self.accumulation_type, self.output_sharding()) {
            (Some(accumulation_type), _) => {
                left.dot_with_accumulation_type(right, self.dimensions(), accumulation_type)
            }
            (None, Some(output_sharding)) => left.dot_with_output_sharding(right, self.dimensions(), output_sharding),
            (None, None) => left.dot(right, self.dimensions()),
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
        // Each term's dot needs equal operand element types: matching pairs (including operand-typed tangents of an
        // accumulation-typed dot) stage directly, while a widened tangent representation pulls both operands up to
        // the output tangent element type.
        let stage_tangent_dot = |left: &C::Value, right: &C::Value| -> Result<C::Value, DifferentiationError> {
            if left.r#type().data_type() == right.r#type().data_type() {
                Ok(stage_dot(left, right))
            } else {
                Ok(stage_dot(&convert_to_tangent_type(left)?, &convert_to_tangent_type(right)?))
            }
        };
        let left_term =
            left.tangent().as_value().map(|tangent| stage_tangent_dot(tangent, right.primal())).transpose()?;
        let right_term =
            right.tangent().as_value().map(|tangent| stage_tangent_dot(left.primal(), tangent)).transpose()?;
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
///
/// For an accumulation-typed dot the output cotangent arrives at the accumulation type's cotangent representation,
/// the known operand is converted up to it, and the adjoint contraction runs at that widened type; the result is
/// then converted to the linear operand's cotangent element type (e.g., back down to `f8e4m3fn` for an
/// `f8 × f8 → f32` dot) so the produced cotangent matches the operand's cotangent representation exactly.
impl<
    V: Value<Type = ArrayType>,
    O: Operation<Type = ArrayType> + From<ConvertElementTypeOperation<ArrayType>> + From<DotOperation>,
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
                        let adjoint_value = outputs.remove(0);
                        // An accumulation-typed primal contracts its adjoint at the widened cotangent type; convert
                        // the result back to the linear operand's cotangent element type when the two differ.
                        let adjoint_value = if adjoint_value.r#type().data_type() == linear_cotangent_type.data_type() {
                            adjoint_value
                        } else {
                            adjoint_value.convert_element_type(linear_cotangent_type.data_type())?
                        };
                        MaybeZero::Value(adjoint_value)
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
/// `rhs [n, k]` with scales `[n, k / block_size]`, and the result is `[m, n]` at the accumulation type. All four
/// operands may instead carry one shared leading batch dimension (`lhs [b, m, k]`, `rhs [b, n, k]`, scales
/// `[b, ·, k / block_size]`, result `[b, m, n]`), matching the custom call's 3D form; mixed ranks are rejected. An
/// optional fifth operand supplies a scalar global scale at the accumulation type that is multiplied into the
/// result (the custom call's per-tensor scale); its presence is inferred from the operand count.
///
/// Semantically the operation dequantizes both operands (upcasting elements and scales to the accumulation type
/// and expanding each scale across its block) and contracts them — which is exactly how the reference array
/// backend and the portable XLA lowering evaluate it (see [`scaled_dot_composition`]). On CUDA targets, the XLA
/// lowering instead emits the `__op$block_scaled_dot` custom call for the standard MXFP8/NVFP4 format and block
/// combinations, which XLA's GPU block-scaling rewriter lowers to cuDNN's native block-scaled tensor-core dot
/// (cuDNN 9.10+) or to expanded reference HLO.
///
/// The operation differentiates as a bilinear contraction of its element operands with the scales (and the optional
/// global scale) held fixed — refer to the forward-mode and transpose rule documentation on this operation.
/// Batching lifts the rank-2 form to the rank-3 batched form (preserving the native fast path under `vmap`); only
/// batching an already-batched rank-3 operation is rejected, because no rank-4 form exists — refer to the batching
/// rule documentation on this operation.
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
        self.render(formatter, 0)
    }
}

impl Operation for ScaledDotOperation {
    type Type = ArrayType;

    #[inline]
    fn name(&self) -> &'static str {
        SCALED_DOT_OPERATION_NAME
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        _region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 4 && input_types.len() != 5 {
            return Err(TypeError::invalid(format!(
                "'scaled_dot' expects 4 inputs plus an optional scalar global scale, but got {}",
                input_types.len(),
            )));
        }
        let lhs = scaled_dot_dimensions("'scaled_dot' left operand", &input_types[0])?;
        let lhs_scales = scaled_dot_dimensions("'scaled_dot' left scales", &input_types[1])?;
        let rhs = scaled_dot_dimensions("'scaled_dot' right operand", &input_types[2])?;
        let rhs_scales = scaled_dot_dimensions("'scaled_dot' right scales", &input_types[3])?;
        let rank = lhs.len();
        for (descriptor, dimensions) in
            [("left scales", &lhs_scales), ("right operand", &rhs), ("right scales", &rhs_scales)]
        {
            if dimensions.len() != rank {
                return Err(TypeError::invalid(format!(
                    "'scaled_dot' operands must share one rank, but got rank {rank} for the left operand and \
                         rank {} for the {descriptor}",
                    dimensions.len(),
                )));
            }
        }
        if rank == 3 && lhs[0] != rhs[0] {
            return Err(TypeError::invalid(format!(
                "'scaled_dot' batch dimension sizes do not match: {} versus {}",
                lhs[0], rhs[0]
            )));
        }
        let contracting = rank - 1;
        if lhs[contracting] != rhs[contracting] {
            return Err(TypeError::invalid(format!(
                "'scaled_dot' contracting dimension sizes do not match: {} versus {}",
                lhs[contracting], rhs[contracting],
            )));
        }
        let Dimension::Static(contracting_size) = &lhs[contracting] else {
            return Err(TypeError::invalid(format!(
                "'scaled_dot' contracting dimension must be static but got {}",
                lhs[contracting],
            )));
        };
        if self.block_size == 0 || contracting_size % self.block_size != 0 {
            return Err(TypeError::invalid(format!(
                "'scaled_dot' contracting dimension size {} is not divisible by block size {}",
                contracting_size, self.block_size,
            )));
        }
        for (descriptor, elements, scales) in [("left", &lhs, &lhs_scales), ("right", &rhs, &rhs_scales)] {
            let mut expected = elements.clone();
            expected[contracting] = Dimension::Static(contracting_size / self.block_size);
            if *scales != expected {
                return Err(TypeError::invalid(format!(
                    "'scaled_dot' {descriptor} scales must have shape {} but got {}",
                    Shape::new(expected),
                    Shape::new(scales.clone()),
                )));
            }
        }
        if let Some(global_scale) = input_types.get(4) {
            if global_scale.static_shape().is_none_or(|shape| shape.rank() != 0) {
                return Err(TypeError::invalid(format!(
                    "'scaled_dot' global scale must be a static scalar but got shape {}",
                    global_scale.shape(),
                )));
            }
            if global_scale.data_type() != self.accumulation_type {
                return Err(TypeError::invalid(format!(
                    "'scaled_dot' global scale data type {} must match the accumulation type {}",
                    global_scale.data_type(),
                    self.accumulation_type,
                )));
            }
        }
        for input_type in input_types {
            if !accumulation_type_is_compatible(input_type.data_type(), self.accumulation_type) {
                return Err(TypeError::invalid(format!(
                    "'scaled_dot' operand data type {} cannot accumulate at data type {}",
                    input_type.data_type(),
                    self.accumulation_type,
                )));
            }
            if !input_type.unreduced_axes().is_empty() {
                return Err(TypeError::invalid("'scaled_dot' does not support unreduced operands".to_string()));
            }
        }
        let mut output_dimensions = Vec::with_capacity(rank);
        if rank == 3 {
            output_dimensions.push(lhs[0].clone());
        }
        output_dimensions.push(lhs[rank - 2].clone());
        output_dimensions.push(rhs[rank - 2].clone());
        Ok(vec![ArrayType::new(self.accumulation_type, Shape::new(output_dimensions))])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, SCALED_DOT_OPERATION_NAME)?.bracketed(|operation| {
            operation.field("block_size", &self.block_size)?;
            operation.field("accumulation_type", &self.accumulation_type)
        })
    }
}

impl<C: Domain<Type = ArrayType, Value: ScaledDot>> InterpretableOperation<C> for ScaledDotOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        match inputs {
            [lhs, lhs_scales, rhs, rhs_scales] => {
                Ok(vec![lhs.scaled_dot(lhs_scales, rhs, rhs_scales, self.block_size, self.accumulation_type)?])
            }
            [lhs, lhs_scales, rhs, rhs_scales, global_scale] => Ok(vec![lhs.scaled_dot_with_global_scale(
                lhs_scales,
                rhs,
                rhs_scales,
                global_scale,
                self.block_size,
                self.accumulation_type,
            )?]),
            inputs => Err(TypeError::invalid(format!(
                "'scaled_dot' expects 4 inputs plus an optional scalar global scale, but got {}",
                inputs.len(),
            ))
            .into()),
        }
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for ScaledDotOperation where
    C::Operation: From<ScaledDotOperation>
{
}

/// Forward-mode rule for [`ScaledDotOperation`]: with the scales held fixed the operation is the bilinear
/// contraction of its dequantized element operands, so
/// `d(scaled_dot(lhs, rhs)) = scaled_dot(d_lhs, rhs) + scaled_dot(lhs, d_rhs)` with both tangent products reusing
/// the primal scales (and the primal global scale, when present). Each tangent term stays a `scaled_dot`, so the
/// output tangent lives at the accumulation type exactly like the primal output. Scale and global-scale
/// perturbations do not propagate: the scales are quantization parameters that gradients flow *through* rather than
/// *into* (the straight-through convention JAX's `scaled_matmul` uses), so their tangent inputs are ignored and
/// their transpose cotangents are structural zeros.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for ScaledDotOperation
where
    C::Operation: From<ScaledDotOperation>,
    C::Value: ScaledDot + std::ops::Add<Output = C::Value>,
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if inputs.len() != 4 && inputs.len() != 5 {
            return Err(ProgramError::from(TypeError::invalid(format!(
                "'scaled_dot' expects 4 inputs plus an optional scalar global scale, but got {}",
                inputs.len(),
            )))
            .into());
        }
        let lhs_scales = inputs[1].primal();
        let rhs_scales = inputs[3].primal();
        let global_scale = inputs.get(4).map(|dual| dual.primal());
        let stage = |lhs: &C::Value, rhs: &C::Value| -> Result<C::Value, DifferentiationError> {
            match global_scale {
                Some(global_scale) => lhs
                    .scaled_dot_with_global_scale(
                        lhs_scales,
                        rhs,
                        rhs_scales,
                        global_scale,
                        self.block_size,
                        self.accumulation_type,
                    )
                    .map_err(DifferentiationError::from),
                None => lhs
                    .scaled_dot(lhs_scales, rhs, rhs_scales, self.block_size, self.accumulation_type)
                    .map_err(DifferentiationError::from),
            }
        };
        let primal = stage(inputs[0].primal(), inputs[2].primal())?;
        let left_term = inputs[0].tangent().as_value().map(|tangent| stage(tangent, inputs[2].primal())).transpose()?;
        let right_term =
            inputs[2].tangent().as_value().map(|tangent| stage(inputs[0].primal(), tangent)).transpose()?;
        let tangent = left_term
            .into_iter()
            .chain(right_term)
            .reduce(|left_term, right_term| left_term + right_term)
            .map_or_else(|| MaybeZero::Zero(primal.r#type().tangent()), MaybeZero::Value);
        Ok(vec![DifferentiationDual::new(primal, tangent)?])
    }
}

/// Partition-aware transpose rule for [`ScaledDotOperation`]. With the scales (and the optional global scale) held
/// fixed as known values, the operation is the bilinear contraction of its two dequantized element operands, so in
/// a valid pushforward exactly one element operand is linear and the other is known. The linear operand's adjoint
/// stages the dequantization composition of the known element operand (see [`dequantize_block_scaled`]) and
/// contracts it with the output cotangent at the accumulation type:
///
///   - linear LHS: `dot(cotangent [b?, m, n], dequantize(rhs) [b?, n, k])` contracting `n` yields `[b?, m, k]`, and
///   - linear RHS: `dot(cotangent [b?, m, n], dequantize(lhs) [b?, m, k])` contracting `m` yields `[b?, n, k]`,
///
/// with a known global scale multiplied into the cotangent first (the forward map scales linearly by it) and a
/// final element-type conversion when the linear operand's cotangent representation differs from the accumulation
/// type (e.g., back down to `f4e2m1fn` elements). The scales are quantization parameters that gradients flow
/// *through* rather than *into*, so their cotangents are structural zeros and transposing with respect to a scale
/// (or the global scale) is rejected. A zero output cotangent stays a structural zero, and two linear element
/// operands are rejected exactly like the bilinear [`DotOperation`] case.
impl<V: Value<Type = ArrayType>, O> TransposableOperation<V, O> for ScaledDotOperation
where
    O: Operation<Type = ArrayType>
        + From<BroadcastOperation>
        + From<ConvertElementTypeOperation<ArrayType>>
        + From<DotOperation>
        + From<MulOperation<ArrayType>>
        + From<ReshapeOperation>,
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        _driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if inputs.len() != 4 && inputs.len() != 5 {
            return Err(ProgramError::from(TypeError::invalid(format!(
                "'scaled_dot' expects 4 inputs plus an optional scalar global scale, but got {}",
                inputs.len(),
            )))
            .into());
        }
        check_count!("output", outputs, 1, ProgramError);
        if inputs[1].is_unknown() || inputs[3].is_unknown() || inputs.get(4).is_some_and(PartialValue::is_unknown) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "'{SCALED_DOT_OPERATION_NAME}' scales are held fixed under differentiation and cannot be \
                     transposed with respect to; differentiate an explicit dequantization composition instead"
                ),
            }
            .into());
        }
        if inputs[0].is_unknown() && inputs[2].is_unknown() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "bilinear `{SCALED_DOT_OPERATION_NAME}` with two linear element operands cannot be transposed"
                ),
            }
            .into());
        }
        // Exactly one element operand is linear: dequantize the known one, contract it with the output cotangent
        // (scaled by a known global scale first), and emit structural zeros for every held-fixed operand.
        let left_is_linear = inputs[0].is_unknown();
        let (linear_index, known_index, known_scales_index) = if left_is_linear { (0, 2, 3) } else { (2, 0, 1) };
        let linear_cotangent_type = inputs[linear_index].r#type().cotangent();
        let contribution = match &outputs[0] {
            MaybeZero::Zero(_) => MaybeZero::Zero(linear_cotangent_type),
            MaybeZero::Value(output_cotangent) => {
                // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                let known_value = |index: usize| {
                    inputs[index].as_known().expect("dispatch guarantees a known operand carries its pullback value")
                };
                let dequantized = dequantize_block_scaled(
                    known_value(known_index),
                    known_value(known_scales_index),
                    self.block_size,
                    self.accumulation_type,
                )?;
                let cotangent = match inputs.get(4) {
                    Some(_) => {
                        let cotangent_type = output_cotangent.r#type().into_owned();
                        output_cotangent.mul(&known_value(4).broadcast(cotangent_type, &[])?)?
                    }
                    None => output_cotangent.clone(),
                };
                let rank = inputs[linear_index].r#type().rank();
                let dimensions = match (left_is_linear, rank) {
                    // Linear LHS: `cotangent [m, n] × dequantize(rhs) [n, k]` contracting `n`.
                    (true, 2) => DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()),
                    (true, _) => DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]),
                    // Linear RHS: `cotangent [m, n] × dequantize(lhs) [m, k]` contracting `m`.
                    (false, 2) => DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()),
                    (false, _) => DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]),
                };
                let operands = [cotangent, dequantized];
                let mut adjoint_outputs =
                    context.stage_operation(DotOperation::new(dimensions), Vec::new(), &operands)?;
                check_count!("output", adjoint_outputs, 1, ProgramError);
                let adjoint = adjoint_outputs.remove(0);
                // The adjoint contraction runs at the accumulation type; convert the result to the linear element
                // operand's cotangent representation when the two differ.
                let adjoint = if adjoint.r#type().data_type() == linear_cotangent_type.data_type() {
                    adjoint
                } else {
                    adjoint.convert_element_type(linear_cotangent_type.data_type())?
                };
                MaybeZero::Value(adjoint)
            }
        };
        let mut contributions =
            inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect::<Vec<_>>();
        contributions[linear_index] = contribution;
        Ok(contributions)
    }
}

/// Batching rule for [`ScaledDotOperation`]: one mapped batch level lifts the rank-2 form to the operation's own
/// rank-3 batched form. Every element and scale operand is aligned to a physical batch axis at position 0 (mapped
/// operands are realigned, replicated operands are broadcast into `axis_size` copies), and the lifted operation is
/// the same `scaled_dot`, whose type inference already accepts one shared leading batch dimension. A replicated
/// global scale stays on the lifted operation as its scalar fifth operand (so the native block-scaled fast path is
/// preserved under `vmap`), while a mapped global scale cannot ride the rank-3 form's scalar operand: it is dropped
/// from the lifted operation and multiplied into the `[b, m, n]` result per batch item instead. The rank-3 form has
/// no rank-4 analogue, so batching an already-batched `scaled_dot` reports an error directing users to batch an
/// explicit dequantization composition instead.
impl<C: Context<Type = ArrayType, Value: Broadcast + Mul + Transpose>, P: ArrayBatchingPolicy<C>>
    BatchableOperation<C, ArrayBatching<P>> for ScaledDotOperation
where
    ScaledDotOperation: InterpretableOperation<C>,
{
    fn batch<D: BatchingDriver<C, ArrayBatching<P>>>(
        &self,
        context: &BatchingContext<C, ArrayBatching<P>>,
        _driver: &D,
        inputs: &[ArrayBatch<C::Value>],
    ) -> Result<BatchedOutputs<C, ArrayBatching<P>>, BatchingError> {
        if inputs.len() != 4 && inputs.len() != 5 {
            return Err(ProgramError::InvalidInputCount { expected: 4, actual: inputs.len() }.into());
        }
        let Some(axis_size) = ArrayBatch::common_batch_size(inputs)? else {
            // Every operand is replicated: the lifted operation is the unbatched operation itself.
            return Ok(self.interpret_with_batch_axes(context, inputs, &[BatchAxis::replicated()])?.into());
        };
        let lhs_primal_rank = inputs[0].r#type().rank() - usize::from(inputs[0].batch_axis_position().is_some());
        if lhs_primal_rank != 2 {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "'{SCALED_DOT_OPERATION_NAME}' has no rank-4 block-scaled form, so a batched rank-{lhs_primal_rank} \
                     operation cannot be batched again; batch an explicit dequantization composition instead"
                ),
            });
        }
        let axis_sharding = ArrayBatch::sharding_for_inputs(inputs)?;
        let mut aligned_inputs = inputs[..4]
            .iter()
            .map(|input| input.match_axis(0, axis_size, axis_sharding.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mapped_global_scale = match inputs.get(4) {
            Some(global_scale) if global_scale.batch_axis().is_replicated() => {
                aligned_inputs.push(global_scale.clone());
                None
            }
            Some(global_scale) => Some(global_scale.move_axis(0)?),
            None => None,
        };
        let mut outputs = self.interpret_with_batch_axes(context, aligned_inputs.as_slice(), &[BatchAxis::new(0)])?;
        let Some(global_scale) = mapped_global_scale else {
            return Ok(outputs.into());
        };
        // A mapped global scale is a per-item `[b]` factor of the `[b, m, n]` result: broadcast it along the batch
        // axis and multiply it into the lifted output.
        let output = outputs.remove(0);
        let output_type = output.r#type().into_owned();
        let broadcast_global_scale = global_scale.value().broadcast(output_type.clone(), &[0])?;
        let scaled_value = output.value().mul(&broadcast_global_scale)?;
        Ok(vec![ArrayBatch::new(output_type, scaled_value, BatchAxis::new(0))?].into())
    }
}

/// Value-level block-scaled ("microscaling") dot capability. Refer to the documentation of [`ScaledDotOperation`]
/// for the operand convention (the contracting dimension is last on *both* element operands, with an optional
/// shared leading batch dimension), the supported formats, and the transform rules.
pub trait ScaledDot: Sized {
    /// Computes the block-scaled matrix product of `self` (shape `[b?, m, k]`, scaled by `lhs_scales` of shape
    /// `[b?, m, k / block_size]`) and `rhs` (shape `[b?, n, k]`, scaled by `rhs_scales` of shape
    /// `[b?, n, k / block_size]`), dequantizing both operands to `accumulation_type` and returning the `[b?, m, n]`
    /// product at that type, and a [`ProgramError`] if something goes wrong.
    fn scaled_dot(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError>;

    /// Computes the block-scaled matrix product exactly like [`Self::scaled_dot`] and multiplies the scalar
    /// `global_scale` (which must carry `accumulation_type`) into the result, matching the optional per-tensor
    /// scale operand of the `__op$block_scaled_dot` custom call. Returns a [`ProgramError`] if something goes
    /// wrong.
    fn scaled_dot_with_global_scale(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        global_scale: &Self,
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

    fn scaled_dot_with_global_scale(
        &self,
        lhs_scales: &Self,
        rhs: &Self,
        rhs_scales: &Self,
        global_scale: &Self,
        block_size: usize,
        accumulation_type: DataType,
    ) -> Result<Self, ProgramError> {
        let mut outputs = self.dispatch_domain().bind(
            ScaledDotOperation::new(block_size, accumulation_type),
            Vec::new(),
            &[self.clone(), lhs_scales.clone(), rhs.clone(), rhs_scales.clone(), global_scale.clone()],
        )?;
        check_count!("output", outputs, 1, ProgramError);
        Ok(outputs.remove(0))
    }
}

/// Evaluates a block-scaled dot as the portable dequantization composition: both operands (whose contracting
/// dimension is last) are upcast to the accumulation type, their scales are expanded across the blocks (a
/// broadcast inserting the block axis, merged back by a reshape), multiplied in, and contracted over the last
/// dimension of both sides — over the shared leading batch dimension for rank-3 operands. A present `global_scale`
/// scalar is broadcast and multiplied into the product. This is the shared semantics behind the concrete
/// [`ScaledDot`] implementations and the portable XLA lowering.
pub(crate) fn scaled_dot_composition<V>(
    lhs: &V,
    lhs_scales: &V,
    rhs: &V,
    rhs_scales: &V,
    global_scale: Option<&V>,
    block_size: usize,
    accumulation_type: DataType,
) -> Result<V, ProgramError>
where
    V: Value<Type = ArrayType> + Broadcast + ConvertElementType + Dot + Mul + Reshape,
{
    let rank = lhs.r#type().rank();
    let lhs = dequantize_block_scaled(lhs, lhs_scales, block_size, accumulation_type)?;
    let rhs = dequantize_block_scaled(rhs, rhs_scales, block_size, accumulation_type)?;
    let dimensions = match rank {
        3 => DotDimensionNumbers::new(vec![2], vec![2], vec![0], vec![0]),
        _ => DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()),
    };
    let product = lhs.dot(&rhs, &dimensions);
    match global_scale {
        Some(global_scale) => {
            let product_type = product.r#type().into_owned();
            product.mul(&global_scale.broadcast(product_type, &[])?)
        }
        None => Ok(product),
    }
}

/// Dequantizes one block-scaled operand whose contracting dimension is last: converts the elements and scales to
/// `accumulation_type`, expands each scale across its block of `block_size` contracting elements (a broadcast
/// appending the block axis, merged back by a reshape), and multiplies. Any static rank works, so the helper serves
/// both the rank-2 and rank-3 `scaled_dot` forms.
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
    let Some(element_shape) = element_type.static_shape() else {
        return Err(TypeError::invalid("'scaled_dot' operand must have a static shape".to_string()).into());
    };
    let element_dimensions = element_shape.dimensions().to_vec();
    let Some(scale_shape) = scales.r#type().static_shape() else {
        return Err(TypeError::invalid("'scaled_dot' scales must have a static shape".to_string()).into());
    };
    let scale_dimensions = scale_shape.dimensions().to_vec();
    let Some(&contracting_size) = element_dimensions.last() else {
        return Err(TypeError::invalid("'scaled_dot' operand must have rank at least 1".to_string()).into());
    };
    if block_size == 0 || contracting_size % block_size != 0 {
        return Err(TypeError::invalid(format!(
            "'scaled_dot' contracting dimension size {contracting_size} is not divisible by block size \
                 {block_size}"
        ))
        .into());
    }
    let mut expected_scale_dimensions = element_dimensions.clone();
    *expected_scale_dimensions.last_mut().unwrap() = contracting_size / block_size;
    if scale_dimensions != expected_scale_dimensions {
        return Err(TypeError::invalid(format!(
            "'scaled_dot' scales must have shape {expected_scale_dimensions:?} but got {scale_dimensions:?}"
        ))
        .into());
    }
    let expanded_type = ArrayType::new(
        accumulation_type,
        Shape::new(
            scale_dimensions
                .iter()
                .map(|&size| Dimension::Static(size))
                .chain(std::iter::once(Dimension::Static(block_size)))
                .collect(),
        ),
    );
    let scale_axes = (0..scale_dimensions.len()).collect::<Vec<_>>();
    let element_sizes = element_dimensions.iter().map(|&size| Dimension::Static(size)).collect::<Vec<_>>();
    let expanded_scales = scales
        .convert_element_type(accumulation_type)?
        .broadcast(expanded_type, scale_axes.as_slice())?
        .reshape(Shape::new(element_sizes))?;
    elements.convert_element_type(accumulation_type)?.mul(&expanded_scales)
}

/// Returns the dimensions of a [`ScaledDot`] operand type, rejecting any rank other than 2 or 3. The rank-3 form
/// carries one leading batch dimension shared by all operands.
fn scaled_dot_dimensions(descriptor: &str, value_type: &ArrayType) -> Result<Vec<Dimension>, TypeError> {
    match value_type.shape().dimensions() {
        dimensions @ (&[_, _] | &[_, _, _]) => Ok(dimensions.to_vec()),
        dimensions => Err(TypeError::invalid(format!(
            "{descriptor} must have rank 2 or rank 3 but got rank {}",
            dimensions.len()
        ))),
    }
}

/// Value-level block-quantization capability: splits a full-precision (`f32` or `f64`) tensor of rank 1 through 3
/// into a narrow element tensor and a tensor of per-block scales along the trailing dimension (which must be
/// divisible by the block size), producing operands for [`ScaledDot`]. This enables on-the-fly quantization (e.g.,
/// of a KV cache) without a dedicated primitive: the recipe is a pure composition of existing operations, so it
/// inherits its transform rules from them.
///
/// Two recipes cover the standard microscaling formats, selected by the scale type:
///
///   - **`f8e4m3fn` scales** (NVIDIA's NVFP4 recipe): each block's scale is `max_abs(block) / element_max`, where
///     `element_max` is the element type's maximum finite magnitude (`6.0` for `f4e2m1fn`), so the block's largest
///     element quantizes to the top of the element grid.
///   - **`f8e8m0fnu` scales** (the [OCP MX](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
///     recipe, e.g. MXFP8): each block's shared scale is the power of two `2^(floor(log2(max_abs(block))) - emax)`,
///     where `emax` is the element type's maximum exponent (`8` for `f8e4m3fn` and `15` for `f8e5m2`), so the
///     block's largest element lands in the element type's top binade. Elements just past the maximum finite
///     magnitude — up to `2^(emax + 1)` — are explicitly clamped before conversion, as the OCP MX specification
///     prescribes; this avoids relying on a floating-point format's overflow conversion policy.
///
/// In both recipes the elements are the input divided by its block's *stored* (already narrowed) scale and
/// converted to the element type, so dequantization (see [`ScaledDotOperation`]) reproduces the input up to element
/// quantization error. The `log2` in the MX recipe is composed as `log(x) / log(2)` plus a `1e-4` nudge before the
/// floor: the nudge keeps block maxima that are exact powers of two in their own binade despite floating-point
/// rounding in the quotient, and the subsequent conversion of `exp(exponent · log(2))` to `f8e8m0fnu` rounds to the
/// nearest power of two, absorbing the remaining approximation error entirely. All-zero (and denormal-tiny) blocks
/// clamp their scale up to a small representable positive value instead of producing a zero or infinite scale.
pub trait BlockQuantize: Sized {
    /// Quantizes `self` into `(elements, scales)` per block of `block_size` trailing-dimension values, where
    /// `elements` carries `element_type` with the shape of `self` and `scales` carries `scale_type` with the
    /// trailing dimension divided by `block_size`. Refer to the trait documentation for the exact recipes. Returns
    /// a [`ProgramError`] if something goes wrong.
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError>;
}

/// Every value with the elementwise, reduction, and reshaping capabilities used by the recipe (which covers both
/// the concrete reference [`Array`](crate::arrays::Array) backend and the transform tracers) quantizes
/// through the shared composition.
impl<V> BlockQuantize for V
where
    V: Value<Type = ArrayType>
        + Abs
        + Clamp
        + Broadcast
        + ConvertElementType
        + Div
        + Exp
        + Floor
        + Log
        + Max
        + Mul
        + Reduce
        + Reshape
        + Sub,
    V::DispatchDomain: Fill<f64, V>,
{
    fn block_quantize(
        &self,
        block_size: usize,
        element_type: DataType,
        scale_type: DataType,
    ) -> Result<(Self, Self), ProgramError> {
        // Max finite magnitude and maximum exponent of the supported microscaling element types.
        let (element_max, element_max_exponent) = match element_type {
            DataType::F4E2M1FN => (6.0, 2.0),
            DataType::F8E4M3FN => (448.0, 8.0),
            DataType::F8E5M2 => (57344.0, 15.0),
            element_type => {
                return Err(TypeError::invalid(format!(
                    "'block_quantize' does not support element data type {element_type}"
                ))
                .into());
            }
        };
        let input_type = self.r#type().into_owned();
        let compute_type = input_type.data_type();
        if !matches!(compute_type, DataType::F32 | DataType::F64) {
            return Err(TypeError::invalid(format!(
                "'block_quantize' expects an f32 or f64 input but got {compute_type}"
            ))
            .into());
        }
        let Some(shape) = input_type.static_shape() else {
            return Err(TypeError::invalid("'block_quantize' input must have a static shape".to_string()).into());
        };
        let dimensions = shape.dimensions().to_vec();
        if dimensions.is_empty() || dimensions.len() > 3 {
            return Err(TypeError::invalid(format!(
                "'block_quantize' input must have rank between 1 and 3 but got rank {}",
                dimensions.len(),
            ))
            .into());
        }
        let trailing_size = *dimensions.last().unwrap();
        if block_size == 0 || trailing_size % block_size != 0 {
            return Err(TypeError::invalid(format!(
                "'block_quantize' trailing dimension size {trailing_size} is not divisible by block size \
                     {block_size}"
            ))
            .into());
        }
        let mut scale_dimensions = dimensions.clone();
        *scale_dimensions.last_mut().unwrap() = trailing_size / block_size;
        let block_shape = Shape::new(
            scale_dimensions
                .iter()
                .map(|&size| Dimension::Static(size))
                .chain(std::iter::once(Dimension::Static(block_size)))
                .collect(),
        );
        let scale_value_type = ArrayType::new(
            compute_type,
            Shape::new(scale_dimensions.iter().map(|&size| Dimension::Static(size)).collect()),
        );
        let domain = self.dispatch_domain();
        let fill = |value: f64| domain.fill(&scale_value_type, value);

        // Per-block maximum magnitude along the trailing dimension.
        let block_max = self.reshape(block_shape.clone())?.abs()?.reduce(&[scale_dimensions.len()], ReductionKind::Max);
        let (scale, smallest_scale) = match scale_type {
            // NVFP4-style linear scaling: the block maximum maps to the element type's maximum magnitude. The clamp
            // floor is the scale type's smallest positive normal, `2^-6`.
            DataType::F8E4M3FN => (block_max.div(&fill(element_max)?)?, (-6.0f64).exp2()),
            // OCP MX power-of-two scaling: `2^(floor(log2(max_abs)) - emax)` with the boundary nudge documented on
            // the trait, folded into one subtraction because `floor(x + ε) - emax = floor(x + ε - emax)` for the
            // integer `emax`. The clamp floor is the scale type's smallest representable value, `2^-127` (which
            // also absorbs the `exp(-inf) = 0` produced by all-zero blocks).
            DataType::F8E8M0FNU => {
                let log_2 = fill(std::f64::consts::LN_2)?;
                let exponent = block_max.log()?.div(&log_2)?.sub(&fill(element_max_exponent - 1e-4)?)?.floor()?;
                (exponent.mul(&log_2)?.exp()?, (-127.0f64).exp2())
            }
            scale_type => {
                return Err(TypeError::invalid(format!(
                    "'block_quantize' does not support scale data type {scale_type}"
                ))
                .into());
            }
        };
        let scales = scale.max(&fill(smallest_scale)?)?.convert_element_type(scale_type)?;

        // Divide by the *stored* scale — exactly the value `scaled_dot` dequantizes with — and narrow the elements.
        let stored_scales = scales.convert_element_type(compute_type)?;
        let expanded_type = ArrayType::new(compute_type, block_shape);
        let scale_axes = (0..scale_dimensions.len()).collect::<Vec<_>>();
        let expanded_scales =
            stored_scales.broadcast(expanded_type, scale_axes.as_slice())?.reshape(input_type.shape().clone())?;
        let fill_scalar = |value: f64| domain.fill(&ArrayType::scalar(compute_type), value);
        let elements = self
            .div(&expanded_scales)?
            .clamp(&fill_scalar(-element_max)?, &fill_scalar(element_max)?)?
            .convert_element_type(element_type)?;
        Ok((elements, scales))
    }
}

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It performs the contraction
/// described by `dimensions`, supporting standard matrix multiplication, batched matrix multiplication, vector inner
/// products, and arbitrary tensor contractions.
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayBatch, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable,
        LogicalMesh, MeshAxis, MeshAxisType, Shape, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::{JacobianDifferentiate, jacobian_reverse};
    use crate::macros::{check_operation_transposition, check_operation_type_inference};
    use crate::programs::{Operation, TypeError};

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
        ArrayType::new(DataType::F32, Shape::new(sizes.iter().map(|size| Dimension::Static(*size)).collect()))
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
        let lhs_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let rhs_type = ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let lhs = Array::from_f64s(lhs_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let lhs_scales = Array::from_f64s(scale_type.clone(), vec![0.5, 2.0, 1.0, 0.5]);
        let rhs = Array::from_f64s(rhs_type.clone(), vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let rhs_scales = Array::from_f64s(scale_type.clone(), vec![2.0, 0.5, 1.0, 2.0]);
        let product = lhs.scaled_dot(&lhs_scales, &rhs, &rhs_scales, 2, DataType::F32).unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(product.to_f64s(), vec![6.75, 11.25, 10.375, 7.0]);

        // MXFP8-flavored case: `f8e4m3fn` elements with power-of-two `f8e8m0fnu` scales.
        let f8_lhs_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let mx_scale_type =
            ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
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
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))]),
        );

        // Bounded dynamic non-contracting dimensions propagate to the result and must be shared exactly by the
        // corresponding scale operands. The contracting dimension remains static because its scale relationship is
        // defined by the block size.
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(5)).unwrap());
        let columns = DimensionVariable::new("columns", DimensionBounds::non_negative(Some(7)).unwrap());
        let dynamic_lhs_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(4)]),
        );
        let dynamic_lhs_scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Static(2)]),
        );
        let dynamic_rhs_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Dynamic(columns.clone()), Dimension::Static(4)]),
        );
        let dynamic_rhs_scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Dynamic(columns.clone()), Dimension::Static(2)]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[dynamic_lhs_type.clone(), dynamic_lhs_scale_type, dynamic_rhs_type, dynamic_rhs_scale_type,],
                &[],
            ),
            Ok(vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(columns)]),
            )]),
        );
        let other_rows = DimensionVariable::new("other_rows", rows.bounds().clone());
        assert_eq!(
            operation.infer_output_types(
                &[
                    dynamic_lhs_type.clone(),
                    ArrayType::new(
                        DataType::F8E4M3FN,
                        Shape::new(vec![Dimension::Dynamic(other_rows), Dimension::Static(2)]),
                    ),
                    rhs_type.clone(),
                    scale_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' left scales must have shape [rows, 2] but got [other_rows, 2]".to_string(),
            )),
        );

        // Contract violations report clear errors through type inference.
        assert_eq!(
            ScaledDotOperation::new(3, DataType::F32).infer_output_types(
                &[lhs_type.clone(), scale_type.clone(), rhs_type.clone(), scale_type.clone()],
                &[],
            ),
            Err(TypeError::invalid("'scaled_dot' contracting dimension size 4 is not divisible by block size 3".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs_type.clone(), rhs_type.clone(), rhs_type, scale_type.clone()], &[]),
            Err(TypeError::invalid("'scaled_dot' left scales must have shape [2, 2] but got [2, 4]".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    lhs_type.clone(),
                    scale_type.clone(),
                    ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(4)])),
                    scale_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::invalid("'scaled_dot' right operand must have rank 2 or rank 3 but got rank 1".to_string())),
        );
        let contracting = DimensionVariable::new("contracting", DimensionBounds::non_negative(Some(5)).unwrap());
        assert_eq!(
            operation.infer_output_types(
                &[
                    ArrayType::new(
                        DataType::F4E2M1FN,
                        Shape::new(vec![Dimension::Dynamic(rows.clone()), Dimension::Dynamic(contracting.clone())]),
                    ),
                    ArrayType::new(
                        DataType::F8E4M3FN,
                        Shape::new(vec![Dimension::Dynamic(rows), Dimension::Static(2)]),
                    ),
                    ArrayType::new(
                        DataType::F4E2M1FN,
                        Shape::new(vec![Dimension::Static(2), Dimension::Dynamic(contracting)]),
                    ),
                    scale_type.clone(),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' contracting dimension must be static but got contracting".to_string(),
            )),
        );

        // Batching lifts the rank-2 form to the operation's own rank-3 batched form, so the batched program stays a
        // single `scaled_dot` instruction (preserving the native block-scaled fast path under `vmap`).
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(lhs_type),
            builder.add_input(scale_type.clone()),
            builder.add_input(ArrayType::new(
                DataType::F4E2M1FN,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]),
            )),
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
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0); 4],
                crate::batching::ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.to_string(),
            indoc! {"
                lambda %0:f4e2m1fn[2, 2, 4], %1:f8e4m3fn[2, 2, 2], %2:f4e2m1fn[2, 2, 4], %3:f8e4m3fn[2, 2, 2] .
                let %4:f32[2, 2, 2] = scaled_dot [block_size=2, accumulation_type=f32] %0 %1 %2 %3
                in (%4)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scaled_dot_rank_3() {
        // The rank-3 form carries one leading batch dimension shared by all four operands. Stacking the rank-2
        // fixture from `test_scaled_dot` twice must reproduce its exact result per batch item.
        let element_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(4)]),
        );
        let scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        );
        let item_lhs = vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5];
        let item_lhs_scales = vec![0.5, 2.0, 1.0, 0.5];
        let item_rhs = vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0];
        let item_rhs_scales = vec![2.0, 0.5, 1.0, 2.0];
        let stack = |item: &[f64]| item.iter().chain(item.iter()).copied().collect::<Vec<_>>();
        let lhs = Array::from_f64s(element_type.clone(), stack(&item_lhs));
        let lhs_scales = Array::from_f64s(scale_type.clone(), stack(&item_lhs_scales));
        let rhs = Array::from_f64s(element_type.clone(), stack(&item_rhs));
        let rhs_scales = Array::from_f64s(scale_type.clone(), stack(&item_rhs_scales));
        let product = lhs.scaled_dot(&lhs_scales, &rhs, &rhs_scales, 2, DataType::F32).unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)])
            ),
        );
        assert_eq!(product.to_f64s(), vec![6.75, 11.25, 10.375, 7.0, 6.75, 11.25, 10.375, 7.0]);

        // Type inference accepts the rank-3 form and rejects mixed ranks and mismatched batch dimensions.
        let operation = ScaledDotOperation::new(2, DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                &[element_type.clone(), scale_type.clone(), element_type.clone(), scale_type.clone()],
                &[],
            ),
            Ok(vec![ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            )]),
        );
        let rank_2_element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        assert_eq!(
            operation.infer_output_types(
                &[element_type.clone(), scale_type.clone(), rank_2_element_type, scale_type.clone()],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' operands must share one rank, but got rank 3 for the left operand and rank 2 \
                          for the right operand"
                    .to_string()
            )),
        );
        let mismatched_batch_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Static(3), Dimension::Static(2), Dimension::Static(4)]),
        );
        assert_eq!(
            operation.infer_output_types(&[element_type, scale_type.clone(), mismatched_batch_type, scale_type], &[],),
            Err(TypeError::invalid("'scaled_dot' batch dimension sizes do not match: 2 versus 3".to_string())),
        );
    }

    #[test]
    fn test_scaled_dot_global_scale() {
        // The optional fifth operand is a scalar at the accumulation type that is multiplied into the result, so
        // the `test_scaled_dot` fixture with a global scale of 2 exactly doubles.
        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let lhs = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let lhs_scales = Array::from_f64s(scale_type.clone(), vec![0.5, 2.0, 1.0, 0.5]);
        let rhs = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let rhs_scales = Array::from_f64s(scale_type.clone(), vec![2.0, 0.5, 1.0, 2.0]);
        let global_scale = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0]);
        let product = lhs
            .scaled_dot_with_global_scale(&lhs_scales, &rhs, &rhs_scales, &global_scale, 2, DataType::F32)
            .unwrap();
        assert_eq!(product.to_f64s(), vec![13.5, 22.5, 20.75, 14.0]);

        // Type inference validates the fifth operand: it must be a static scalar at the accumulation type.
        let operation = ScaledDotOperation::new(2, DataType::F32);
        assert_eq!(
            operation.infer_output_types(
                &[
                    element_type.clone(),
                    scale_type.clone(),
                    element_type.clone(),
                    scale_type.clone(),
                    ArrayType::scalar(DataType::F32),
                ],
                &[],
            ),
            Ok(vec![ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    element_type.clone(),
                    scale_type.clone(),
                    element_type.clone(),
                    scale_type.clone(),
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])),
                ],
                &[],
            ),
            Err(TypeError::invalid("'scaled_dot' global scale must be a static scalar but got shape [2]".to_string())),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    element_type.clone(),
                    scale_type.clone(),
                    element_type.clone(),
                    scale_type.clone(),
                    ArrayType::scalar(DataType::F64),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' global scale data type f64 must match the accumulation type f32".to_string()
            )),
        );
        assert_eq!(
            operation.infer_output_types(&[element_type, scale_type.clone(), scale_type], &[]),
            Err(TypeError::invalid(
                "'scaled_dot' expects 4 inputs plus an optional scalar global scale, but got 3".to_string()
            )),
        );
    }

    #[test]
    fn test_scaled_dot_batching() {
        use crate::batching::BatchingContext;
        use crate::programs::EmptyRegionDriver;

        // Two batch items built from the `test_scaled_dot` NVFP4 fixture: item 0 is the fixture itself and item 1
        // swaps its operand sides, so the per-item expectations come from unbatched `scaled_dot` calls.
        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let item_lhs = [1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5];
        let item_lhs_scales = [0.5, 2.0, 1.0, 0.5];
        let item_rhs = [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0];
        let item_rhs_scales = [2.0, 0.5, 1.0, 2.0];
        let element = |values: &[f64]| Array::from_f64s(element_type.clone(), values.to_vec());
        let scales = |values: &[f64]| Array::from_f64s(scale_type.clone(), values.to_vec());
        let expected_item_0 = element(&item_lhs)
            .scaled_dot(&scales(&item_lhs_scales), &element(&item_rhs), &scales(&item_rhs_scales), 2, DataType::F32)
            .unwrap();
        let expected_item_1 = element(&item_rhs)
            .scaled_dot(&scales(&item_rhs_scales), &element(&item_lhs), &scales(&item_lhs_scales), 2, DataType::F32)
            .unwrap();
        let expected: Vec<f64> = expected_item_0.to_f64s().into_iter().chain(expected_item_1.to_f64s()).collect();

        let stacked_element_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(4)]),
        );
        let stacked_scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        );
        let stack = |element_values: bool, first: &[f64], second: &[f64]| {
            let values = first.iter().chain(second.iter()).copied().collect::<Vec<_>>();
            let r#type = if element_values { stacked_element_type.clone() } else { stacked_scale_type.clone() };
            let value = Array::from_f64s(r#type, values);
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
        };
        let operation = ScaledDotOperation::new(2, DataType::F32);
        let context = BatchingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new(), 2);

        // All four operands mapped at axis 0: the lifted operation is the rank-3 form and the output is mapped.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(true, &item_lhs, &item_rhs),
                    stack(false, &item_lhs_scales, &item_rhs_scales),
                    stack(true, &item_rhs, &item_lhs),
                    stack(false, &item_rhs_scales, &item_lhs_scales),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected);

        // Mixed mapped/replicated operands: the replicated right-hand pair is broadcast into per-item copies, so
        // every batch item multiplies against the same right-hand side.
        let replicated = |value: Array| ArrayBatch::new(value.r#type().into_owned(), value, BatchAxis::replicated());
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(true, &item_lhs, &item_rhs),
                    stack(false, &item_lhs_scales, &item_rhs_scales),
                    replicated(element(&item_rhs)).unwrap(),
                    replicated(scales(&item_rhs_scales)).unwrap(),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        let expected_item_1_shared_rhs = element(&item_rhs)
            .scaled_dot(&scales(&item_rhs_scales), &element(&item_rhs), &scales(&item_rhs_scales), 2, DataType::F32)
            .unwrap();
        let expected_shared_rhs: Vec<f64> =
            expected_item_0.to_f64s().into_iter().chain(expected_item_1_shared_rhs.to_f64s()).collect();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected_shared_rhs);

        // A replicated global scale stays on the lifted operation as its scalar fifth operand.
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(true, &item_lhs, &item_rhs),
                    stack(false, &item_lhs_scales, &item_rhs_scales),
                    stack(true, &item_rhs, &item_lhs),
                    stack(false, &item_rhs_scales, &item_lhs_scales),
                    replicated(Array::from_f64s(ArrayType::scalar(DataType::F32), vec![2.0])).unwrap(),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected.iter().map(|value| value * 2.0).collect::<Vec<_>>());

        // A mapped global scale is multiplied into the result per batch item instead of riding the lifted
        // operation's scalar operand.
        let mapped_global_scales =
            Array::from_f64s(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)])), vec![2.0, 0.5]);
        let outputs = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stack(true, &item_lhs, &item_rhs),
                    stack(false, &item_lhs_scales, &item_rhs_scales),
                    stack(true, &item_rhs, &item_lhs),
                    stack(false, &item_rhs_scales, &item_lhs_scales),
                    ArrayBatch::new(mapped_global_scales.r#type().into_owned(), mapped_global_scales, Some(0)).unwrap(),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        let expected_per_item_scales: Vec<f64> = expected_item_0
            .to_f64s()
            .into_iter()
            .map(|value| value * 2.0)
            .chain(expected_item_1.to_f64s().into_iter().map(|value| value * 0.5))
            .collect();
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), expected_per_item_scales);

        // The rank-3 form has no rank-4 analogue: batching an already-batched operation is rejected.
        let stacked_rank_3 = |values: &[f64], r#type: &ArrayType| {
            let dimensions: Vec<Dimension> =
                std::iter::once(Dimension::Static(2)).chain(r#type.shape().dimensions().iter().cloned()).collect();
            let value = Array::from_f64s(
                ArrayType::new(r#type.data_type(), Shape::new(dimensions)),
                values.iter().chain(values.iter()).copied().collect(),
            );
            ArrayBatch::new(value.r#type().into_owned(), value, Some(0)).unwrap()
        };
        let rank_3_element_type = ArrayType::new(
            DataType::F4E2M1FN,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(4)]),
        );
        let rank_3_scale_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(1), Dimension::Static(2), Dimension::Static(2)]),
        );
        let error = operation
            .batch(
                &context,
                &EmptyRegionDriver,
                &[
                    stacked_rank_3(&item_lhs, &rank_3_element_type),
                    stacked_rank_3(&item_lhs_scales, &rank_3_scale_type),
                    stacked_rank_3(&item_rhs, &rank_3_element_type),
                    stacked_rank_3(&item_rhs_scales, &rank_3_scale_type),
                ],
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "'scaled_dot' has no rank-4 block-scaled form, so a batched rank-3 operation cannot be batched again; \
             batch an explicit dequantization composition instead",
        );
    }

    #[test]
    fn test_scaled_dot_differentiation() {
        // Forward mode: `scaled_dot` is linear in each element operand with the scales held fixed, so the tangent
        // is the sum of two `scaled_dot`s reusing the primal scales, and the (nonzero) scale tangents supplied
        // below are ignored by design (straight-through with respect to the elements). Every value is exactly
        // representable in its storage format, so the `f32` results are exact.
        let element_type =
            ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)]));
        let scale_type =
            ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
        ];
        let output = builder.add_instruction(ScaledDotOperation::new(2, DataType::F32), Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![output],
                vec![crate::parameters::Placeholder; 4],
                vec![crate::parameters::Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f4e2m1fn[2, 4], %1:f8e4m3fn[2, 2], %2:f4e2m1fn[2, 4], %3:f8e4m3fn[2, 2], %4:f4e2m1fn[2, 4], %5:f8e4m3fn[2, 2], %6:f4e2m1fn[2, 4], %7:f8e4m3fn[2, 2] .
                let %8:f32[2, 2] = scaled_dot [block_size=2, accumulation_type=f32] %0 %1 %2 %3
                    %9:f32[2, 2] = scaled_dot [block_size=2, accumulation_type=f32] %4 %1 %2 %3
                    %10:f32[2, 2] = scaled_dot [block_size=2, accumulation_type=f32] %0 %1 %6 %3
                    %11:f32[2, 2] = add %9 %10
                in (%8, %11)
            "}
            .trim_end(),
        );
        let lhs = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let lhs_scales = Array::from_f64s(scale_type.clone(), vec![0.5, 2.0, 1.0, 0.5]);
        let rhs = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let rhs_scales = Array::from_f64s(scale_type.clone(), vec![2.0, 0.5, 1.0, 2.0]);
        let lhs_tangent = Array::from_f64s(element_type.clone(), vec![1.0; 8]);
        let rhs_tangent = Array::from_f64s(element_type.clone(), vec![0.5; 8]);
        let scale_tangent = Array::from_f64s(scale_type.clone(), vec![1.0; 4]);
        let jvp_outputs = jvp
            .interpret(vec![
                lhs,
                lhs_scales.clone(),
                rhs,
                rhs_scales.clone(),
                lhs_tangent,
                scale_tangent.clone(),
                rhs_tangent,
                scale_tangent,
            ])
            .unwrap();
        assert_eq!(jvp_outputs[0].to_f64s(), vec![6.75, 11.25, 10.375, 7.0]);
        // Tangent = scaled_dot(d_lhs, rhs) + scaled_dot(lhs, d_rhs) with the primal scales held fixed.
        assert_eq!(jvp_outputs[1].to_f64s(), vec![7.0, 17.5, 10.6875, 7.75]);

        // Transposition stages the dequantization-composition adjoint of the known element operand and converts
        // the accumulation-typed contraction back to the linear operand's element type. Unit scales keep the
        // dequantized operands on the `f4e2m1fn` grid, so with an identity output cotangent the linear LHS adjoint
        // is exactly the dequantized RHS and the linear RHS adjoint is exactly the dequantized LHS.
        let unit_scales = Array::from_f64s(scale_type.clone(), vec![1.0; 4]);
        let lhs_values = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.5, 3.0, 1.0, 2.0, 0.5]);
        let rhs_values = Array::from_f64s(element_type.clone(), vec![1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0]);
        let identity_cotangent = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![1.0, 0.0, 0.0, 1.0],
        );
        check_operation_transposition!(
            @exact,
            operation = ScaledDotOperation::new(2, DataType::F32),
            cases = [
                {
                    inputs = [
                        (@linear(type = element_type.clone())),
                        (@known, unit_scales.clone()),
                        (@known, rhs_values.clone()),
                        (@known, unit_scales.clone()),
                    ],
                    output_cotangents = [identity_cotangent.clone()],
                    input_cotangents = [rhs_values.clone()],
                },
                {
                    inputs = [
                        (@known, lhs_values.clone()),
                        (@known, unit_scales.clone()),
                        (@linear(type = element_type.clone())),
                        (@known, unit_scales.clone()),
                    ],
                    output_cotangents = [identity_cotangent],
                    input_cotangents = [lhs_values],
                },
            ],
        );

        // Transposing with respect to a scale operand is rejected: the scales are held fixed under differentiation.
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let inputs = vec![
            builder.add_input(element_type.clone()),
            builder.add_input(scale_type.clone()),
            builder.add_input(element_type),
            builder.add_input(scale_type),
        ];
        let output = builder.add_instruction(ScaledDotOperation::new(2, DataType::F32), Vec::new(), inputs).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![output],
                vec![crate::parameters::Placeholder; 4],
                vec![crate::parameters::Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[1]),
            Err(error) if error.to_string().contains("'scaled_dot' scales are held fixed under differentiation"),
        ));
    }

    #[test]
    fn test_block_quantize_nvfp4() {
        // NVFP4 recipe: `f4e2m1fn` elements with `f8e4m3fn` scales, `scale = max_abs(block) / 6.0`. Every block
        // below is a scaled copy of `f4e2m1fn` grid points whose scale is exactly representable in `f8e4m3fn`, so
        // quantization is exact; the all-zero block exercises the clamp to the smallest normal scale, `2^-6`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)]));
        let input = Array::from_f64s(
            input_type.clone(),
            vec![
                3.0, 1.5, 0.5, 6.0, 0.5, 1.0, 0.25, 1.5, // Blocks with scales 1.0 and 0.25.
                -12.0, 6.0, 3.0, -1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale 2.0 and the clamp floor.
            ],
        );
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(scales.to_f64s(), vec![1.0, 0.25, 2.0, 0.015625]);
        assert_eq!(
            elements.to_f64s(),
            vec![3.0, 1.5, 0.5, 6.0, 2.0, 4.0, 1.0, 6.0, -6.0, 3.0, 1.5, -0.5, 0.0, 0.0, 0.0, 0.0],
        );

        // Round trip: the quantized operands contract through `scaled_dot` to the exact full-precision dot.
        let product = elements.scaled_dot(&scales, &elements, &scales, 4, DataType::F32).unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        assert_eq!(product.to_f64s(), expected.to_f64s());

        // Contract violations report clear errors.
        assert!(matches!(
            input.block_quantize(3, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("trailing dimension size 8 is not divisible by block size 3"),
        ));
        assert!(matches!(
            input.block_quantize(4, DataType::F16, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("'block_quantize' does not support element data type f16"),
        ));
        assert!(matches!(
            input.block_quantize(4, DataType::F4E2M1FN, DataType::F16),
            Err(error) if error.to_string().contains("'block_quantize' does not support scale data type f16"),
        ));
        let integer_input =
            Array::from_f64s(ArrayType::new(DataType::I32, Shape::new(vec![Dimension::Static(8)])), vec![1.0; 8]);
        assert!(matches!(
            integer_input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("'block_quantize' expects an f32 or f64 input but got i32"),
        ));
        let scalar_input = Array::from_f64s(ArrayType::scalar(DataType::F32), vec![1.0]);
        assert!(matches!(
            scalar_input.block_quantize(1, DataType::F4E2M1FN, DataType::F8E4M3FN),
            Err(error) if error.to_string().contains("must have rank between 1 and 3 but got rank 0"),
        ));
    }

    #[test]
    fn test_block_quantize_mxfp8() {
        // OCP MX recipe: `f8e4m3fn` elements with power-of-two `f8e8m0fnu` scales,
        // `scale = 2^(floor(log2(max_abs)) - 8)`. The block maxima below sit exactly on powers of two (exercising
        // the boundary nudge in the `log2` composition) and every quotient is exactly representable in `f8e4m3fn`,
        // so quantization is exact; the all-zero block clamps its scale to `2^-127`.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)]));
        let input = Array::from_f64s(
            input_type.clone(),
            vec![
                4.0, 2.0, 1.0, 0.5, 1.75, 0.5, -1.0, 0.25, // Blocks with scales 2^-6 and 2^-8.
                -8.0, 4.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, // Blocks with scale 2^-5 and the clamp floor.
            ],
        );
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
        assert_eq!(scales.to_f64s(), vec![(-6.0f64).exp2(), (-8.0f64).exp2(), (-5.0f64).exp2(), (-127.0f64).exp2()],);
        assert_eq!(
            elements.to_f64s(),
            vec![256.0, 128.0, 64.0, 32.0, 448.0, 128.0, -256.0, 64.0, -256.0, 128.0, 64.0, 32.0, 0.0, 0.0, 0.0, 0.0],
        );

        // Round trip: the quantized operands contract through `scaled_dot` to the exact full-precision dot.
        let product = elements.scaled_dot(&scales, &elements, &scales, 4, DataType::F32).unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        assert_eq!(product.to_f64s(), expected.to_f64s());
    }

    #[test]
    fn test_block_quantize_round_trip_tolerance() {
        // Values off the storage grids round-trip within the element type's quantization error: `f8e4m3fn` carries
        // three mantissa bits, so each dequantized element is within about 6% of its input (plus the OCP MX
        // saturation of a block maximum landing past the finite range) and the contraction of eight such products
        // stays within a proportional tolerance of the full-precision dot.
        let input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(1), Dimension::Static(8)]));
        let input = Array::from_f64s(input_type.clone(), vec![1.1, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9]);
        let (elements, scales) = input.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(elements.r#type().shape(), input_type.shape());
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(1), Dimension::Static(2)])),
        );
        let product = elements.scaled_dot(&scales, &elements, &scales, 4, DataType::F32).unwrap();
        let expected = input.dot(&input, &DotDimensionNumbers::new(vec![1], vec![1], Vec::new(), Vec::new()));
        let expected_value = expected.to_f64s()[0];
        let actual_value = product.to_f64s()[0];
        assert_abs_diff_eq!(actual_value, expected_value, epsilon = 0.05 * expected_value);

        // Rank-1 inputs quantize per block along their only dimension.
        let vector_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let vector = Array::from_f64s(vector_type.clone(), vec![1.1, -2.3, 0.7, 3.9, 0.013, -0.27, 5.4, 8.9]);
        let (vector_elements, vector_scales) =
            vector.block_quantize(4, DataType::F8E4M3FN, DataType::F8E8M0FNU).unwrap();
        assert_eq!(vector_elements.r#type().shape(), vector_type.shape());
        assert_eq!(
            vector_scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![Dimension::Static(2)])),
        );
        assert_eq!(vector_elements.to_f64s(), elements.to_f64s());
        assert_eq!(vector_scales.to_f64s(), scales.to_f64s());
    }

    #[test]
    fn test_block_quantize_stages_through_tracers() {
        // The composition also covers staging values: quantizing a tracer stages the recipe's operations and the
        // staged outputs carry the quantized element and scale types.
        use crate::tracing::TracingContext;

        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let builder = context.builder().clone();
        let input_atom = builder
            .borrow_mut()
            .add_input(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])));
        let input = context.tracer(input_atom, None);
        let (elements, scales) = input.block_quantize(4, DataType::F4E2M1FN, DataType::F8E4M3FN).unwrap();
        assert_eq!(
            elements.r#type().as_ref(),
            &ArrayType::new(DataType::F4E2M1FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(8)])),
        );
        assert_eq!(
            scales.r#type().as_ref(),
            &ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
        );
    }

    #[test]
    fn test_dot_accumulation_type() {
        // Type inference widens the output to the accumulation type for promotable operand types and rejects
        // non-promotable ones, combining with a requested output sharding, and differentiation.
        let operation = DotOperation::matmul().with_accumulation_type(DataType::F32);
        assert_eq!(operation.accumulation_type(), Some(DataType::F32));
        let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let rhs = lhs.clone();
        let bf16_operand = ArrayType::new(DataType::BF16, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        let output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]));
        check_operation_type_inference!(
            operation = operation,
            cases = [
                {
                    input_types = [lhs.clone(), rhs.clone()],
                    output_types = [output_type.clone()],
                },
                {
                    input_types = [bf16_operand.clone(), bf16_operand],
                    output_types = [output_type],
                },
            ],
        );
        let narrowing = DotOperation::matmul().with_accumulation_type(DataType::F16);
        let f32_operand = plain_array(&[2, 2]);
        check_operation_type_inference!(
            operation = narrowing,
            cases = [{
                input_types = [f32_operand.clone(), f32_operand],
                error = "'dot' operand data type f32 cannot accumulate at data type f16",
            }],
        );
        let mesh = test_mesh();
        let sharded = DotOperation::matmul().with_accumulation_type(DataType::F32).with_output_sharding(
            Sharding::new(mesh, vec![ShardingDimension::Replicated, ShardingDimension::Replicated]).unwrap(),
        );
        check_operation_type_inference!(
            operation = sharded,
            cases = [{
                input_types = [lhs.clone(), rhs.clone()],
                error = "'dot' does not support combining an accumulation type with a requested output sharding yet",
            }],
        );

        // The eager reference backend upcasts the operands and accumulates at the accumulation type: every value
        // below is exactly representable in `f8e4m3fn`, so the `f32` results are exact.
        let lhs_values = Array::from_f64s(lhs.clone(), vec![0.5, 1.0, 1.5, 2.0]);
        let rhs_values = Array::from_f64s(rhs.clone(), vec![1.0, 0.5, 0.5, 1.0]);
        let product = lhs_values.dot_with_accumulation_type(&rhs_values, &DotDimensionNumbers::matmul(), DataType::F32);
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)]))
        );
        assert_eq!(product.to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);

        // Forward-mode differentiation stages accumulation-typed tangent dots over the operand-typed tangents, so
        // the output tangent lives at the accumulation type exactly like the primal output. Every value below is
        // exactly representable in `f8e4m3fn` and every product sum is exact in `f32`.
        let mut builder = crate::programs::builders::ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let lhs_input = builder.add_input(lhs.clone());
        let rhs_input = builder.add_input(rhs.clone());
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
        let jvp = program.jvp().unwrap();
        assert_eq!(
            jvp.to_string(),
            indoc! {"
                lambda %0:f8e4m3fn[2, 2], %1:f8e4m3fn[2, 2], %2:f8e4m3fn[2, 2], %3:f8e4m3fn[2, 2] .
                let %4:f32[2, 2] = dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    accumulation_type=f32,
                ] %0 %1
                    %5:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %2 %1
                    %6:f32[2, 2] = dot [
                        dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                        accumulation_type=f32,
                    ] %0 %3
                    %7:f32[2, 2] = add %5 %6
                in (%4, %7)
            "}
            .trim_end(),
        );
        let jvp_outputs = jvp
            .interpret(vec![
                lhs_values.clone(),
                rhs_values.clone(),
                Array::from_f64s(lhs.clone(), vec![1.0, 1.0, 1.0, 1.0]),
                Array::from_f64s(rhs.clone(), vec![0.5, 0.5, 0.5, 0.5]),
            ])
            .unwrap();
        assert_eq!(jvp_outputs[0].to_f64s(), vec![1.0, 1.25, 2.5, 2.75]);
        assert_eq!(jvp_outputs[1].r#type().data_type(), DataType::F32);
        // Tangent = d_lhs · rhs + lhs · d_rhs = [[1.5, 1.5], [1.5, 1.5]] + [[0.75, 0.75], [1.75, 1.75]].
        assert_eq!(jvp_outputs[1].to_f64s(), vec![2.25, 2.25, 3.25, 3.25]);

        // The transpose rule contracts the adjoint at the accumulation type and converts the result back to the
        // linear operand's `f8e4m3fn` cotangent representation. With an identity output cotangent, the adjoint of
        // the linear RHS is exactly `lhsᵀ`.
        check_operation_transposition!(
            @exact,
            operation = DotOperation::matmul().with_accumulation_type(DataType::F32),
            cases = [{
                inputs = [
                    (@known, lhs_values),
                    (@linear(type = rhs.clone())),
                ],
                output_cotangents = [Array::from_f64s(
                    ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
                    vec![1.0, 0.0, 0.0, 1.0],
                )],
                input_cotangents = [Array::from_f64s(rhs, vec![0.5, 1.5, 1.0, 2.0])],
            }],
        );

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
            .into_parts()
            .0;
        let batched_lhs_type = ArrayType::new(
            DataType::F8E4M3FN,
            Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
        );
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
        let batch = DimensionVariable::new("batch", DimensionBounds::unbounded());
        let contracting = DimensionVariable::new("contracting", DimensionBounds::non_negative(Some(4)).unwrap());
        let lhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(batch.clone()),
                Dimension::Static(2),
                Dimension::Dynamic(contracting.clone()),
            ]),
        );
        let rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Dynamic(contracting), Dimension::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()], &[]),
            Ok(vec![ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Dynamic(batch.clone()), Dimension::Static(2), Dimension::Static(3)]),
            )]),
        );

        // Static-vs-dynamic and unequal dynamic dimension pairs keep erroring under the strict size equality used
        // for batching and contracting dimensions.
        let static_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(batch), Dimension::Static(4), Dimension::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), static_rhs], &[]),
            Err(TypeError::invalid(
                "'dot' contracting dimension sizes do not match (LHS axis 2, RHS axis 1)".to_string()
            )),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(8)).unwrap())),
                Dimension::Dynamic(DimensionVariable::new("dynamic", DimensionBounds::non_negative(Some(4)).unwrap())),
                Dimension::Static(3),
            ]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, mismatched_batch_rhs], &[]),
            Err(TypeError::invalid("'dot' batching dimension sizes do not match (LHS axis 0, RHS axis 0)".to_string())),
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
            Err(TypeError::invalid(
                "'dot' batching dimensions must have consistent shardings, but got {'b'} and {'m'}".to_string()
            )),
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
            Err(TypeError::invalid(
                "'dot' contracting dimensions are sharded, making the output sharding ambiguous; request an \
                          explicit output sharding (e.g., one with unreduced axes) to resolve it"
                    .to_string()
            )),
        );
        // Differently sharded contracting dimensions are inconsistent.
        let mismatched_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), mismatched_rhs], &[]),
            Err(TypeError::invalid(
                "'dot' contracting dimensions must have consistent shardings, but got {'k'} and {'m'}".to_string()
            )),
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
            Err(TypeError::invalid("'dot' operand shardings must use the same mesh".to_string())),
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
            Err(TypeError::invalid("'dot' operands cannot be unreduced".to_string())),
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
            Err(TypeError::invalid("'dot' output sharding rank (1) does not match the output rank (3)".to_string())),
        );

        // Mesh validation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(other_mesh_sharding);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs], &[]),
            Err(TypeError::invalid("'dot' output sharding must use the same mesh as the operands".to_string())),
        );

        // Auto mesh axes cannot be requested explicitly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_sharding =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(auto_sharding);
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])], &[]),
            Err(TypeError::invalid("'dot' output sharding cannot reference auto mesh axes".to_string())),
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
            Err(TypeError::invalid(
                "'dot' contracting dimensions must be sharded identically when the output sharding is unreduced"
                    .to_string()
            )),
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
            Err(TypeError::invalid(
                "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                    .to_string()
            )),
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
            Err(TypeError::invalid(
                "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                    .to_string()
            )),
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

        use crate::arrays::ArrayBatch;
        use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
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
        let outputs =
            operation.batch(&batching_context, &crate::EmptyRegionDriver, &[lhs, rhs]).unwrap().into_parts().0;
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

        use crate::arrays::{Array, ArrayBatch, ArrayOperation};
        use crate::batching::{BatchAxis, BatchableOperation, BatchingContext};
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
            let lhs_type = ArrayType::new(
                DataType::F64,
                Shape::new(vec![Dimension::Static(2), Dimension::Static(2), Dimension::Static(2)]),
            )
            .with_sharding(lhs_sharding)
            .unwrap();
            let rhs_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(2), Dimension::Static(1)]))
                .with_sharding(Sharding::replicated(mesh, 2))
                .unwrap();
            let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
            let builder = parent.builder().clone();
            let lhs_atom = builder.borrow_mut().add_input(lhs_type.clone());
            let rhs_atom = builder.borrow_mut().add_input(rhs_type);
            let lhs = ArrayBatch::new(lhs_type, parent.tracer(lhs_atom, None), BatchAxis::new(0)).unwrap();
            let rhs = ArrayBatch::replicated(parent.tracer(rhs_atom, None));
            let context = BatchingContext::new(parent.clone(), 2).with_axis_sharding(ShardingDimension::sharded(["x"]));

            let outputs = DotOperation::matmul()
                .batch(&context, &crate::EmptyRegionDriver, &[lhs, rhs])
                .unwrap()
                .into_parts()
                .0;

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

        let output: Array = batch(
            |row| Ok(row.dot(&row, &DotDimensionNumbers::inner_product())),
            x,
            BatchAxis::new(0),
            BatchAxis::new(0),
            None,
        )
        .unwrap();

        assert_eq!(output.r#type().into_owned(), ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)])),);
        // Batch item 0: [1,2,3,4]·[1,2,3,4] = 30. Batch item 1: [5,6,7,8]·[5,6,7,8] = 174. Batch item 2: 446.
        for (actual, expected) in output.to_f64s().iter().zip([30.0_f64, 174.0, 446.0].iter()) {
            assert_abs_diff_eq!(*actual, *expected, epsilon = 1e-9);
        }

        // A replicated operand is broadcast across the mapped operand's batch axis before the dot dimensions are
        // lifted. Each row therefore contracts against the same right-hand vector.
        let lhs = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let lhs = ArrayBatch::new(lhs.r#type().into_owned(), lhs, BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::replicated(Array::vector(vec![10.0, 100.0, 1000.0]));
        let outputs = DotOperation::new(DotDimensionNumbers::inner_product())
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &crate::EmptyRegionDriver, &[lhs, rhs])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![3210.0, 6540.0]);
    }

    #[test]
    fn test_dot_dense_jacobians() {
        let inputs = (Array::vector(vec![2.0, 3.0, 5.0]), Array::vector(vec![7.0, 11.0, 13.0]));

        // Reverse mode batches the pullback's adjoint dots over output-coordinate cotangents.
        let jacobian = jacobian_reverse(
            |(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())),
            inputs.clone(),
        )
        .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
        assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);

        // Forward mode batches input-coordinate basis tangents through the dot pushforward.
        let jacobian = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jacobian_forward(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())), inputs)
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
        assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);
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
                            Shape::new(vec![Dimension::Static(2), Dimension::Static(3)]),
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
