use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use crate::arrays::{
    ArrayBatch, ArrayBatching, ArrayBatchingPolicy, ArrayIrType, ArrayIrValue, ArrayType, DataType, Dimension,
    DimensionType, DimensionValue, LogicalMesh, MeshAxisType, RaggedArrayBatchingPolicy, Shape, Sharding,
    ShardingDimension,
};
use crate::axes::Axis;
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
use crate::operations::dimensions::dimension_requirement::DimensionRequirement;
use crate::operations::dimensions::dimension_size::DimensionSize;
use crate::operations::manipulation::broadcasting::{Broadcast, DynamicBroadcast};
use crate::operations::manipulation::conversion::{ConvertElementType, ConvertElementTypeOperation};
use crate::operations::manipulation::reshaping::{DynamicReshape, Reshape};
use crate::operations::manipulation::transposition::Transpose;
use crate::operations::math::div::Div;
use crate::operations::math::mul::Mul;
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    MaybeZero, Operation, OperationFormatter, ProgramError, RegionInterface, TypeError, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

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

/// Batching rule for [`DotOperation`]: the operands are aligned onto one common mapped axis, the dimension numbers are
/// lifted past it with [`lift_dot_dimensions`], and the lifted contraction is re-interpreted over the packed values.
///
/// Alignment is delegated to [`ArrayBatchingPolicy::match_axis`], so this rule never needs a statically known mapped
/// extent: under a dimension-valued policy the batch axis materialized on a replicated operand is grounded by the
/// transform's first-class extent value, and the staged contraction simply carries that possibly-dynamic mapped
/// [`Dimension`] on its batching dimension. Two mapped operands must still describe the same mapped extent, which for
/// dynamic extents means the same [`DimensionVariable`](crate::arrays::DimensionVariable).
///
/// A contraction is also a zero-padding-discipline consumer of bounded ragged axes. Every ragged axis of an operand is
/// either contracted or free:
///
///   - A **contracted** ragged axis is consumed. [`RaggedArrayBatchingPolicy::pad_contraction_input`] zeroes that
///     operand's padded elements first, which removes their products from the contraction's sums, and the rule reports
///     each consumed [`DimensionVariable`](crate::arrays::DimensionVariable) as its [`BatchedOutputs`] evidence so the
///     carrier-invariant validation boundary can tell a deliberate consumption apart from a silently dropped extent.
///     Each operand is zeroed along its own contracted ragged axes only, because zeroing either factor of a contracted
///     pair already neutralizes that product.
///   - A **free** ragged axis survives into the result and propagates onto the output carrier, relocated through the
///     dot's output layout (i.e., the batching dimensions, then the LHS free axes, then the RHS free axes).
///
/// A ragged axis on a *batching* dimension of the dot is rejected: the two operands would have to agree on per-item
/// extents along paired batch dimensions, which no dimension identity established here can guarantee. A ragged axis on
/// a *replicated* operand is rejected as well, because materializing a batch axis on it is a broadcast with no per-item
/// extents to relocate. Operands without any ragged axis take the dense path unchanged.
impl<C: Context<Type = ArrayType>, P: RaggedArrayBatchingPolicy<C>> BatchableOperation<C, ArrayBatching<P>>
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
        // A replicated ragged operand is rejected before alignment, because the broadcast that materializes its batch
        // axis carries no per-item extents and would drop the metadata that records them.
        for (input, batch_axis) in inputs.iter().zip(batch_axes.iter()) {
            if batch_axis.is_none()
                && let Some(ragged_axis) = input.ragged_axes().first()
            {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' does not support bounded ragged dimension `{}` on a replicated operand",
                        ragged_axis.dimension(),
                    ),
                });
            }
        }
        // Two mapped operands must describe the same mapped extent. Comparing the mapped dimensions validates static
        // extents exactly as `ArrayBatch::common_batch_size` does, and it additionally admits a dynamic mapped extent,
        // which two operands share exactly when it is the same dimension variable.
        let mapped_dimension = |index: usize| batch_axes[index].map(|axis| inputs[index].r#type().dimension(axis));
        if let (Some(left), Some(right)) = (mapped_dimension(0), mapped_dimension(1))
            && left != right
        {
            return Err(match (left.value(), right.value()) {
                (Some(expected), Some(actual)) => BatchingError::MismatchedBatchSizes { expected, actual },
                _ => BatchingError::MisalignedBatchAxes {
                    message: format!(
                        // TODO(eaplatanios): Are backticks conventional in Rust for these kinds of error messages?
                        //  If so, can we use them conssitently in the codebase (e.g., replacing single quotes where
                        //  this same convention would apply)?
                        "'{DOT_OPERATION_NAME}' operands map different batch extents `{left}` and `{right}`"
                    ),
                },
            });
        }
        // Mixed batched/unbatched: materialize a batch axis on the replicated operand at position 0 (JAX's
        // `matchaxis(0)` convention), then fall through to the both-batched arm of `lift_dot_dimensions`. The active
        // policy owns that materialization, so the mapped extent never has to be a statically known host size: under a
        // dimension-valued policy it stays a first-class extent value grounding the staged broadcast.
        let aligned_inputs: Vec<ArrayBatch<C::Value>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => vec![inputs[0].clone(), P::match_axis(context, &inputs[1], Axis::from(0))?],
            (None, Some(_)) => vec![P::match_axis(context, &inputs[0], Axis::from(0))?, inputs[1].clone()],
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
        let output_batch_axes = [BatchAxis::from_optional_position(output_axis)];
        if aligned_inputs.iter().all(|input| input.ragged_axes().is_empty()) {
            return Ok(lifted_op.interpret_with_batch_axes(context, &aligned_inputs, &output_batch_axes)?.into());
        }

        // A generalized dot lays its result out as the batching dimensions, then the LHS free axes, then the RHS free
        // axes, so each operand axis either lands at a known result axis or is contracted away.
        let dimensions = lifted_op.dimensions();
        let batching_count = dimensions.lhs_batching_dimensions().len();
        let lhs_result = lhs_result_axes(dimensions, aligned_inputs[0].r#type().rank());
        let rhs_result = rhs_result_axes(dimensions, aligned_inputs[1].r#type().rank());
        let operand_output_axes = |rank: usize, batching: &[usize], result: &[usize], offset: usize| {
            (0..rank)
                .map(|axis| {
                    batching.iter().position(|batching_axis| *batching_axis == axis).or_else(|| {
                        result
                            .iter()
                            .position(|result_axis| *result_axis == axis)
                            .map(|index| batching_count + offset + index)
                    })
                })
                .collect::<Vec<_>>()
        };
        let output_axes = [
            operand_output_axes(
                aligned_inputs[0].r#type().rank(),
                dimensions.lhs_batching_dimensions(),
                lhs_result.as_slice(),
                0,
            ),
            operand_output_axes(
                aligned_inputs[1].r#type().rank(),
                dimensions.rhs_batching_dimensions(),
                rhs_result.as_slice(),
                lhs_result.len(),
            ),
        ];

        let contracting_axes = [dimensions.lhs_contracting_dimensions(), dimensions.rhs_contracting_dimensions()];
        let batching_axes = [dimensions.lhs_batching_dimensions(), dimensions.rhs_batching_dimensions()];
        let mut contracted_dimensions = Vec::new();
        let mut output_ragged_axes = Vec::new();
        let mut contraction_inputs = Vec::with_capacity(aligned_inputs.len());
        for (index, input) in aligned_inputs.iter().enumerate() {
            if let Some(ragged_axis) =
                input.ragged_axes().iter().find(|ragged_axis| batching_axes[index].contains(&ragged_axis.axis()))
            {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "'{DOT_OPERATION_NAME}' does not support bounded ragged dimension `{}` on a batching dimension",
                        ragged_axis.dimension(),
                    ),
                });
            }
            let contracted = input
                .ragged_axes()
                .iter()
                .filter(|ragged_axis| contracting_axes[index].contains(&ragged_axis.axis()))
                .map(|ragged_axis| ragged_axis.dimension().clone())
                .collect::<Vec<_>>();
            for ragged_axis in input.ragged_axes() {
                // A contracted ragged axis is consumed by the zeroing below and reported as evidence; every other one
                // must reach the result together with the axes its per-item extents index.
                if output_axes[index][ragged_axis.axis()].is_none() {
                    continue;
                }
                output_ragged_axes.push(ragged_axis.clone().relocated(output_axes[index].as_slice()).ok_or_else(
                    || BatchingError::UnsupportedOperation {
                        message: format!(
                            "'{DOT_OPERATION_NAME}' contracts an axis carrying the per-item extents of bounded ragged \
                             dimension `{}`",
                            ragged_axis.dimension(),
                        ),
                    },
                )?);
            }
            contraction_inputs.push(if contracted.is_empty() {
                input.clone()
            } else {
                P::pad_contraction_input(context, input, contracting_axes[index])?
            });
            contracted_dimensions.extend(contracted);
        }

        let mut outputs = lifted_op.interpret_with_batch_axes(context, &contraction_inputs, &output_batch_axes)?;
        check_count!("output", outputs, 1, ProgramError);
        let output = ArrayBatch::new(outputs.remove(0).into_value(), output_batch_axes[0])?
            .with_ragged_axes(output_ragged_axes)?;
        Ok(BatchedOutputs::new(vec![output], contracted_dimensions))
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

mod scaled;

pub use scaled::*;

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
        LogicalMesh, MeshAxis, MeshAxisType, RaggedAxis, Shape, Sharding, ShardingDimension,
    };
    use crate::batching::{BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, batch};
    use crate::contexts::EagerContext;
    use crate::differentiation::differentiate_at;
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
    fn test_scaled_dot_jax_contract() {
        // This fixture uses JAX's rank-2 default convention: the left trailing axis contracts with the right leading
        // axis. The two sides infer independent block ratios of two from different scale-axis positions.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(4)])),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(3)])),
            (1..=12).map(|value| value as f64).collect(),
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(2)])),
            vec![1.0, 2.0, 0.5, 1.0],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
            vec![1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
        );
        let product = lhs.scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), None, Some(DataType::F32)).unwrap();
        assert_eq!(
            product.r#type().as_ref(),
            &ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(3)])),
        );
        assert_eq!(product.to_f64s(), vec![253.0, 284.0, 315.0, 272.5, 308.0, 343.5]);

        // Each scale is independently optional. Missing scales are the multiplicative identity, and omitting both
        // therefore reduces to a `bf16`-intermediate generalized dot with an `f32` result.
        assert_eq!(
            lhs.scaled_dot(&rhs, None, None, None, Some(DataType::F32)).unwrap().to_f64s(),
            vec![70.0, 80.0, 90.0, 158.0, 184.0, 210.0],
        );
        assert_eq!(
            lhs.scaled_dot(&rhs, Some(&lhs_scale), None, None, Some(DataType::F32)).unwrap().to_f64s(),
            vec![131.0, 148.0, 165.0, 143.5, 164.0, 184.5],
        );

        let dimensions = ScaledDotOperation::default_dimensions(2).unwrap();
        let operation = ScaledDotOperation::new(dimensions.clone(), DataType::F32, true, true);
        assert_eq!(operation.dimensions(), &dimensions);
        assert_eq!(operation.preferred_element_type(), DataType::F32);
        assert!(operation.has_lhs_scale());
        assert!(operation.has_rhs_scale());
        assert_eq!(
            operation.to_string(),
            indoc! {"
                scaled_dot [
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    preferred_element_type=f32,
                    lhs_scale=true,
                    rhs_scale=true,
                ]
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_scaled_dot_inference() {
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let operation = ScaledDotOperation::new(dimensions, DataType::BF16, true, true);
        let lhs = ArrayType::new(DataType::F8E4M3FN, Shape::new(vec![2.into(), 3.into(), 4.into(), 6.into()]));
        let rhs = ArrayType::new(DataType::F8E5M2, Shape::new(vec![2.into(), 4.into(), 6.into(), 5.into()]));
        let lhs_scale = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 3.into(), 2.into(), 2.into()]));
        let rhs_scale = ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 2.into(), 2.into(), 5.into()]));
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::BF16, Shape::new(vec![2.into(), 3.into(), 5.into()]))]),
        );
        assert_eq!(
            ScaledDotOperation::new(operation.dimensions().clone(), DataType::F64, true, true)
                .infer_output_types(&[lhs.clone(), rhs.clone(), lhs_scale.clone(), rhs_scale.clone()], &[]),
            Ok(vec![ArrayType::new(DataType::F64, Shape::new(vec![2.into(), 3.into(), 5.into()]))]),
        );
        assert_eq!(
            operation.infer_output_types(
                &[
                    lhs,
                    rhs,
                    lhs_scale.with_shape(Shape::new(vec![2.into(), 3.into(), 4.into(), 2.into()])),
                    ArrayType::new(DataType::F8E8M0FNU, Shape::new(vec![2.into(), 2.into(), 2.into(), 5.into()]),),
                ],
                &[],
            ),
            Err(TypeError::invalid(
                "'scaled_dot' left contracting axis 2 to scale ratio must be at least 2 but got 1".to_string(),
            )),
        );
    }

    #[test]
    fn test_scaled_dot_composition_supports_multiple_contracting_dimensions() {
        // Both contracting axes carry independent block ratios. Expanding all scale axes in one broadcast preserves
        // their original axis positions before the final reshape; every dequantized element is one in this fixture.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 4.into(), 6.into()])),
            vec![1.0; 48],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 4.into(), 6.into(), 3.into()])),
            vec![1.0; 72],
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into(), 2.into()])),
            vec![1.0; 8],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into(), 3.into()])),
            vec![1.0; 12],
        );
        let dimensions = DotDimensionNumbers::new(vec![2, 3], vec![1, 2], vec![0], vec![0]);
        let output = lhs
            .scaled_dot(&rhs, Some(&lhs_scale), Some(&rhs_scale), Some(&dimensions), Some(DataType::F32))
            .unwrap();
        assert_eq!(output.r#type().shape(), &Shape::new(vec![1.into(), 2.into(), 3.into()]));
        assert_eq!(output.to_f64s(), vec![24.0; 6]);

        // The ergonomic wrapper follows JAX's `[B, M, K] x [B, N, K]` convention and defaults to an `f32` result.
        let lhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 1.into(), 4.into()])),
            vec![1.0; 4],
        );
        let rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 4.into()])),
            vec![1.0; 8],
        );
        let lhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 1.into(), 2.into()])),
            vec![1.0; 2],
        );
        let rhs_scale = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 2.into()])),
            vec![1.0; 4],
        );
        let output = lhs.scaled_matmul(&rhs, &lhs_scale, &rhs_scale, None).unwrap();
        assert_eq!(output.r#type().data_type(), DataType::F32);
        assert_eq!(output.r#type().shape(), &Shape::new(vec![1.into(), 1.into(), 2.into()]));
        assert_eq!(output.to_f64s(), vec![4.0; 2]);

        // Ryft honors the wrapper's documented independent block-ratio semantics even though pinned JAX's wrapper
        // currently rejects unequal scale contracting dimensions before reaching `lax.scaled_dot`.
        let independently_scaled_rhs = Array::from_f64s(
            ArrayType::new(DataType::F32, Shape::new(vec![1.into(), 2.into(), 1.into()])),
            vec![1.0; 2],
        );
        let output = lhs.scaled_matmul(&rhs, &lhs_scale, &independently_scaled_rhs, None).unwrap();
        assert_eq!(output.to_f64s(), vec![4.0; 2]);
    }

    #[test]
    fn test_scaled_dot_batching() {
        // Batching moves each mapped axis to the front and lifts it into the generalized-dot batch dimensions. Scale
        // operands follow the same rule, so each example retains its own block scales.
        let elements = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 4.into()]));
        let scales = ArrayType::new(DataType::F32, Shape::new(vec![2.into(), 2.into()]));
        let lhs = ArrayBatch::new(Array::from_f64s(elements.clone(), vec![1.0; 8]), BatchAxis::new(0)).unwrap();
        let rhs = ArrayBatch::new(Array::from_f64s(elements, vec![1.0; 8]), BatchAxis::new(0)).unwrap();
        let lhs_scale = ArrayBatch::new(Array::from_f64s(scales.clone(), vec![1.0; 4]), BatchAxis::new(0)).unwrap();
        let rhs_scale = ArrayBatch::new(Array::from_f64s(scales, vec![1.0; 4]), BatchAxis::new(0)).unwrap();
        let operation = ScaledDotOperation::new(DotDimensionNumbers::inner_product(), DataType::F32, true, true);

        let outputs = operation
            .batch(
                &BatchingContext::new(EagerContext::<Array>::new(), 2),
                &crate::EmptyRegionDriver,
                &[lhs, rhs, lhs_scale, rhs_scale],
            )
            .unwrap()
            .into_parts()
            .0;

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().to_f64s(), vec![4.0, 4.0]);
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
            ArrayBatch::new(value, Some(0))
        }
        .unwrap();
        let rhs = {
            let value = context.tracer(rhs_atom, None);
            ArrayBatch::new(value, Some(0))
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
            let lhs = ArrayBatch::new(parent.tracer(lhs_atom, None), BatchAxis::new(0)).unwrap();
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
        let lhs = ArrayBatch::new(lhs, BatchAxis::new(0)).unwrap();
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
    fn test_dot_batching_validates_mapped_extents() {
        use crate::batching::BatchingContext;
        use crate::tracing::TracingContext;

        let context = TracingContext::<ArrayType, ArrayOperation<ArrayType>>::new();
        let batching_context = BatchingContext::new(context.clone(), 2);
        let input = |r#type: ArrayType| {
            let atom = context.builder().borrow_mut().add_input(r#type.clone());
            context.tracer(atom, Some(r#type))
        };
        let operation = DotOperation::new(DotDimensionNumbers::inner_product());
        let dynamic_rows = |variable: &DimensionVariable| {
            input(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(variable.clone()), Dimension::Static(3)]),
            ))
        };

        // Two mapped operands whose mapped axes carry different statically known extents cannot describe one batch.
        assert_eq!(
            operation
                .batch(
                    &batching_context,
                    &crate::EmptyRegionDriver,
                    &[
                        ArrayBatch::new(input(plain_array(&[2, 3])), BatchAxis::new(0)).unwrap(),
                        ArrayBatch::new(input(plain_array(&[3, 3])), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .map(|outputs| outputs.into_parts().0)
                .unwrap_err(),
            BatchingError::MismatchedBatchSizes { expected: 2, actual: 3 },
        );

        // Two mapped operands sharing one dynamic mapped extent describe the same batch, so the lifted contraction is
        // staged with that dimension on its batching dimension.
        let variable = DimensionVariable::new("batch", DimensionBounds::new(1, Some(5)).unwrap());
        let outputs = operation
            .batch(
                &batching_context,
                &crate::EmptyRegionDriver,
                &[
                    ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                    ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                ],
            )
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(
            outputs[0].r#type().into_owned(),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable.clone())])),
        );

        // Dynamic extents are compared by variable identity, so two independently declared variables are two
        // independent extents even when their bounds agree.
        let other = DimensionVariable::new("other", DimensionBounds::new(1, Some(5)).unwrap());
        assert_eq!(
            operation
                .batch(
                    &batching_context,
                    &crate::EmptyRegionDriver,
                    &[
                        ArrayBatch::new(dynamic_rows(&variable), BatchAxis::new(0)).unwrap(),
                        ArrayBatch::new(dynamic_rows(&other), BatchAxis::new(0)).unwrap(),
                    ],
                )
                .map(|outputs| outputs.into_parts().0)
                .unwrap_err(),
            BatchingError::MisalignedBatchAxes {
                message: "'dot' operands map different batch extents `batch` and `other`".to_string(),
            },
        );
    }

    #[test]
    fn test_dot_batching_propagates_free_ragged_axes() {
        // Per item, a ragged `[length, 2]` matrix contracts its dense trailing axis against a `[2]` vector, so the
        // ragged axis is a free axis of the contraction and survives into the `[length]` per-item result. The lifted
        // dot lays its result out as the batching dimension followed by the LHS free axes, so the ragged axis moves
        // from packed axis 1 to output axis 1 and the mapped axis its extents index stays at output axis 0.
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let extents = Array::vector(vec![1_i32, 3]);
        let lhs = ArrayBatch::new(
            Array::from_f64s(plain_array(&[2, 3, 2]), (1..=12).map(f64::from).collect()),
            BatchAxis::new(0),
        )
        .unwrap()
        .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
        .unwrap();
        let rhs = ArrayBatch::new(Array::matrix(2, 2, vec![1.0_f32, 10.0, 100.0, 1000.0]), BatchAxis::new(0)).unwrap();
        let (outputs, evidence) = DotOperation::new(DotDimensionNumbers::new(vec![1], vec![0], Vec::new(), Vec::new()))
            .batch(&BatchingContext::new(EagerContext::<Array>::new(), 2), &crate::EmptyRegionDriver, &[lhs, rhs])
            .unwrap()
            .into_parts();

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].r#type().into_owned(), plain_array(&[2, 3]));
        assert_eq!(outputs[0].ragged_axes(), &[RaggedAxis::new(1, extents, variable.clone(), vec![0])]);
        assert_eq!(
            outputs[0].unbatched_type(),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(variable)])),
        );
        // Nothing was contracted away, so the rule claims no consumption evidence and the padded rows keep whatever
        // the dense contraction produced for them.
        assert!(evidence.is_empty());
        assert_eq!(outputs[0].value().to_f64s(), vec![21.0, 43.0, 65.0, 8700.0, 10900.0, 13100.0]);
    }

    #[test]
    fn test_dot_batching_rejects_unsupported_ragged_configurations() {
        let variable = DimensionVariable::new("length", DimensionBounds::new(0, Some(3)).unwrap());
        let extents = Array::vector(vec![1_i32, 3]);
        let ragged_matrix = || {
            ArrayBatch::new(
                Array::from_f64s(plain_array(&[2, 3, 2]), (1..=12).map(f64::from).collect()),
                BatchAxis::new(0),
            )
            .unwrap()
            .with_ragged_axes(vec![RaggedAxis::new(1, extents.clone(), variable.clone(), vec![0])])
            .unwrap()
        };
        let context = BatchingContext::new(EagerContext::<Array>::new(), 2);

        // Contracting the ragged axis requires zeroing its padding, which static array batching cannot stage.
        assert_eq!(
            DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], Vec::new(), Vec::new()))
                .batch(&context, &crate::EmptyRegionDriver, &[ragged_matrix(), ragged_matrix()])
                .map(|outputs| outputs.into_parts().0)
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "static array batching cannot zero-pad bounded ragged axes".to_string(),
            },
        );

        // A ragged axis declared as a batching dimension of the dot itself would require both operands to agree on
        // per-item extents along paired batch dimensions, which nothing here establishes.
        assert_eq!(
            DotOperation::new(DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]))
                .batch(&context, &crate::EmptyRegionDriver, &[ragged_matrix(), ragged_matrix()])
                .map(|outputs| outputs.into_parts().0)
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "'dot' does not support bounded ragged dimension `length` on a batching dimension".to_string(),
            },
        );

        // A replicated ragged operand gains its batch axis through a broadcast that carries no per-item extents.
        let replicated =
            ArrayBatch::replicated(Array::from_f64s(plain_array(&[3, 2]), (1..=6).map(f64::from).collect()))
                .with_ragged_axes(vec![RaggedAxis::new(0, extents, variable, vec![])])
                .unwrap();
        assert_eq!(
            DotOperation::matmul()
                .batch(
                    &context,
                    &crate::EmptyRegionDriver,
                    &[
                        ArrayBatch::new(Array::matrix(2, 3, vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]), BatchAxis::new(0))
                            .unwrap(),
                        replicated,
                    ],
                )
                .map(|outputs| outputs.into_parts().0)
                .unwrap_err(),
            BatchingError::UnsupportedOperation {
                message: "'dot' does not support bounded ragged dimension `length` on a replicated operand".to_string(),
            },
        );
    }

    #[test]
    fn test_dot_dense_jacobians() {
        let inputs = (Array::vector(vec![2.0, 3.0, 5.0]), Array::vector(vec![7.0, 11.0, 13.0]));

        // Reverse mode batches the pullback's adjoint dots over output-coordinate cotangents.
        let jacobian = differentiate_at(inputs.clone())
            .jacobian_reverse(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())))
            .unwrap();
        let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
        let [left, right] = blocks.as_slice() else { unreachable!() };
        assert_eq!(left.value().to_f64s(), vec![7.0, 11.0, 13.0]);
        assert_eq!(right.value().to_f64s(), vec![2.0, 3.0, 5.0]);

        // Forward mode batches input-coordinate basis tangents through the dot pushforward.
        let jacobian = differentiate_at(inputs)
            .jacobian_forward(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::inner_product())))
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
