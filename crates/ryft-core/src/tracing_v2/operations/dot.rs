use std::collections::BTreeSet;
use std::fmt::{Debug, Display};

use crate::batching::BatchingError;
use crate::batching::InterpretableBatchableOperation;
use crate::contexts::Domain;
use crate::contexts::{Context, StagingContext};
use crate::differentiation::TransposableOperation;
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::manipulation::Transpose;
use crate::operations::{Operation, OperationFormatter};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{MaybeZero, ProgramError, Value};
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{Tracer, TracingContext};

use crate::tracing_v2::differentiation::{DifferentiableOperation, JvpTracer, combine_terms};
use crate::types::{ArrayType, Shape, Size, StaticShape, TypeError, Typed};

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

/// Query trait classifying operations as dot-like contractions. Backend-owned closed operation enums implement this
/// trait so that generic transform code — most notably the dot-based members of
/// [`RematerializationPolicy`](crate::tracing_v2::rematerialization::RematerializationPolicy) — can classify staged
/// instructions without knowing the concrete operation enum. Higher-order operations whose bodies may contain dots
/// (jit calls, custom-derivative calls) are not themselves dot-like, mirroring how JAX's `dots_saveable` rematerialization
/// policy matches only dot primitives.
pub trait MaybeDot {
    /// Returns the dot dimension numbers when this operation is a dot-like contraction, and [`None`] otherwise.
    fn dot_dimensions(&self) -> Option<&DotDimensionNumbers>;

    /// Returns whether this operation is a dot-like contraction.
    #[inline]
    fn is_dot(&self) -> bool {
        self.dot_dimensions().is_some()
    }
}

/// Value-level generalized dot capability.
///
/// [`Dot`] is the receiver-style entry point for staging or executing [`DotOperation`]. It
/// performs the contraction described by `dimensions`, supporting standard matrix
/// multiplication, batched matrix multiplication, vector inner products, and arbitrary tensor
/// contractions.
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
            .bind(DotOperation::new(dimensions.clone()), &[self.clone(), rhs.clone()])
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
                &[self.clone(), rhs.clone()],
            )
            .expect("`dot` operation failed")
            .remove(0)
    }
}

/// Generalized N-C dot and transpose capability.
///
/// This convenience trait groups the value-level [`Dot`] and [`Transpose`] operations used by the unified
/// [`DotOperation`] and [`TransposeOperation`](crate::operations::manipulation::TransposeOperation) primitives.
pub trait DotOps: Dot + Transpose {}

impl<T: Dot + Transpose> DotOps for T {}

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

/// Canonical operation name reported by the generalized dot product type-inference rule. The captured-factor linear
/// forms shared this rule and reported under the same name.
const DOT_OPERATION_NAME: &'static str = "dot";

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
fn dot_abstract(
    lhs: &ArrayType,
    rhs: &ArrayType,
    dimensions: &DotDimensionNumbers,
    output_sharding: Option<&Sharding>,
) -> Result<ArrayType, TypeError> {
    if lhs.data_type() != rhs.data_type() {
        return Err(TypeError { message: format!("'{DOT_OPERATION_NAME}' input element types are incompatible") });
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
        let sharding = Sharding::with_manual_axes(
            mesh.clone(),
            placement,
            Vec::<String>::new(),
            reduced_axes,
            varying_manual_axes,
        )
        .map_err(|error| TypeError {
            message: format!("'{DOT_OPERATION_NAME}' output sharding construction failed: {error}"),
        })?;
        Some(sharding.without_auto_axes())
    } else {
        None
    };

    ArrayType::new(lhs.data_type(), Shape::new(output_dimensions))
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

    /// Optional requested output [`Sharding`]. Refer to the documentation of [`Self::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl DotOperation {
    /// Creates a new [`DotOperation`] with the supplied dimension numbers.
    #[inline]
    pub fn new(dimensions: DotDimensionNumbers) -> Self {
        Self { dimensions, output_sharding: None }
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

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 2, TypeError);
        Ok(vec![dot_abstract(&input_types[0], &input_types[1], &self.dimensions, self.output_sharding.as_ref())?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("dimensions", &self.dimensions)?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<V: Value<Type = ArrayType> + Dot, C> InterpretableOperation<V, C> for DotOperation {
    fn interpret(&self, _context: &C, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        // The requested output sharding flows through the capability method so that interpretation over staging
        // values (e.g., during program batching) preserves it; concrete values ignore it.
        Ok(vec![match &self.output_sharding {
            Some(output_sharding) => inputs[0].dot_with_output_sharding(&inputs[1], &self.dimensions, output_sharding),
            None => inputs[0].dot(&inputs[1], &self.dimensions),
        }])
    }
}

/// Partial evaluation defers to the default fold-or-residualize behavior of
/// [`Program::partially_evaluate`](crate::Program::partially_evaluate).
impl<C: Context<Type = ArrayType>> PartiallyEvaluatableOperation<C> for DotOperation where
    C::Operation: From<DotOperation>
{
}

impl<V: Value<Type = ArrayType> + crate::operations::manipulation::Broadcast, C>
    crate::batching::BatchableOperation<V, C> for DotOperation
where
    DotOperation: InterpretableOperation<V, C>,
{
    fn batch(
        &self,
        context: &C,
        inputs: &[crate::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::batching::ArrayBatch<V>>, BatchingError> {
        check_count!("input", inputs, 2, ProgramError);
        let batch_axes: Vec<Option<usize>> = inputs.iter().map(|input| input.batch_axis().axis()).collect();
        // Validate the common batch size across both operands (catching mismatched batched operands) before the
        // mixed arms consult it; a mixed operand pair always has at least one mapped operand.
        let axis_size = crate::batching::ArrayBatch::common_batch_size(inputs)?;
        // Mixed batched/unbatched: broadcast the replicated operand to gain a singleton batch
        // axis at position 0 (JAX's `matchaxis(0)` convention), then fall through to the
        // both-batched arm of `lift_dot_dimensions`.
        let mixed_axis_size = || axis_size.expect("a mapped input pins the batch size");
        let aligned_inputs: Vec<crate::batching::ArrayBatch<V>> = match (batch_axes[0], batch_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => vec![inputs[0].clone(), inputs[1].broadcast(0, mixed_axis_size())?],
            (None, Some(_)) => vec![inputs[0].broadcast(0, mixed_axis_size())?, inputs[1].clone()],
        };
        let aligned_axes: Vec<Option<usize>> = aligned_inputs.iter().map(|input| input.batch_axis().axis()).collect();
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(&self.dimensions, aligned_axes[0], aligned_axes[1])
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: "'dot' batching failed to lift its dimension numbers for the aligned batch axes".to_string(),
            })?;
        let batch_dimension = crate::batching::ArrayBatch::sharding_for_inputs(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions).with_output_sharding(lift_output_sharding(
            self.output_sharding.as_ref(),
            output_axis,
            batch_dimension,
        )?);
        lifted_op.interpret_with_batch_axes(context, &aligned_inputs, &[output_axis.into()])
    }
}

/// Forward-mode rule for [`DotOperation`]: the product rule for the contraction
/// `d(dot(a, b)) = dot(da, b) + dot(a, db)`. Each term holds the corresponding primal operand fixed on its original
/// contracting side, staged as an ordinary `Dot` whose dimension numbers and requested output sharding match the
/// primal, so the tangent dots match the primal dot exactly and stay capture-free.
impl<C: Context<Type = ArrayType>> DifferentiableOperation<C> for DotOperation
where
    C::Operation: Clone + From<DotOperation>,
    C::Value: Dot + std::ops::Add<Output = C::Value>,
{
    fn jvp(&self, _context: &C, inputs: &[JvpTracer<C>]) -> Result<Vec<JvpTracer<C>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        let stage_dot = |left: &C::Value, right: &C::Value| match self.output_sharding() {
            Some(output_sharding) => left.dot_with_output_sharding(right, self.dimensions(), output_sharding),
            None => left.dot(right, self.dimensions()),
        };
        let primal = stage_dot(left.primal(), right.primal());
        let left_term = left.tangent().as_value().map(|tangent| stage_dot(tangent, right.primal()));
        let right_term = right.tangent().as_value().map(|tangent| stage_dot(left.primal(), tangent));
        let tangent = combine_terms(left_term, right_term, &primal);
        Ok(vec![JvpTracer::new(primal, tangent)])
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
impl<V: Value<Type = ArrayType>, O: Operation<ArrayType> + From<DotOperation>> TransposableOperation<V, O>
    for DotOperation
{
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        check_count!("output", outputs, 1, ProgramError);
        match (inputs[0].is_unknown(), inputs[1].is_unknown()) {
            // Both operands linear is a bilinear product, which is not a linear map in both operands jointly and so
            // never appears in a valid pushforward.
            (true, true) => Err(ProgramError::UnsupportedOperation {
                message: "bilinear `dot` with two linear operands cannot be transposed".to_string(),
            }),
            // Exactly one operand is linear: stage the adjoint dot reading the known operand's value, and emit a
            // structural zero for the known operand. A zero output cotangent stays a structural zero.
            (left_is_linear, _) => {
                let (linear_index, known_index) = if left_is_linear { (0, 1) } else { (1, 0) };
                let contribution = match &outputs[0] {
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.clone()),
                    MaybeZero::Value(output_cotangent) => {
                        // The dispatch guarantees a `Known` operand carries its pullback value, so read it directly.
                        let known_value = inputs[known_index]
                            .as_known()
                            .expect("dispatch guarantees a known operand carries its pullback value")
                            .clone();
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
                        let mut outputs = context.stage_operation(adjoint, &operands)?;
                        check_count!("output", outputs, 1, ProgramError);
                        MaybeZero::Value(outputs.remove(0))
                    }
                };
                let mut contributions =
                    inputs.iter().map(|input| MaybeZero::Zero(input.r#type().into_owned())).collect::<Vec<_>>();
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

/// Lifts an optional requested output sharding through one batching level by inserting `batch_dimension` at the new
/// output batch axis. `batch_dimension` is the [`ShardingDimension`] derived from the batched inputs' mapped axis
/// (see [`ArrayBatch::sharding_for_inputs`](crate::ArrayBatch::sharding_for_inputs)), so the batched
/// dimension carries the same sharding as the operands' mapped axis, mirroring JAX's `get_sharding_for_vmap`.
fn lift_output_sharding(
    output_sharding: Option<&Sharding>,
    output_axis: Option<usize>,
    batch_dimension: ShardingDimension,
) -> Result<Option<Sharding>, ProgramError> {
    match (output_sharding, output_axis) {
        (Some(output_sharding), Some(axis)) => output_sharding
            .with_inserted_dimension(axis, batch_dimension)
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
    use crate::operations::constants::ZeroOperation;
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
    use crate::operations::arithmetic::AddOperation;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

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
            operation.infer_output_types(&[lhs.clone(), rhs.clone()]),
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
            operation.infer_output_types(&[lhs.clone(), static_rhs]),
            Err(TypeError {
                message: "'dot' contracting dimension sizes do not match (LHS axis 2, RHS axis 1)".to_string(),
            }),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(Some(8)), Size::Dynamic(Some(4)), Size::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, mismatched_batch_rhs]),
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
            operation.infer_output_types(&[lhs, rhs]),
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
            operation.infer_output_types(&[replicated_lhs, replicated_rhs]),
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
            operation.infer_output_types(&[lhs, rhs]),
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
            operation.infer_output_types(&[lhs, plain_array(&[8, 16])]),
            Ok(vec![sharded_array(
                &mesh,
                &[4, 16],
                vec![ShardingDimension::sharded(["m"]), ShardingDimension::replicated()],
            )]),
        );
        // Without any operand shardings, the output carries none.
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])]),
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
            operation.infer_output_types(&[lhs, rhs]),
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
            operation.infer_output_types(&[lhs.clone(), rhs]),
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
            operation.infer_output_types(&[lhs.clone(), mismatched_rhs]),
            Err(TypeError {
                message: "'dot' contracting dimensions must have consistent shardings, but got {'k'} and {'m'}"
                    .to_string(),
            }),
        );
        // A contracting dimension sharded on only one operand is allowed, and its sharding is dropped.
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        assert_eq!(
            operation.infer_output_types(&[lhs, replicated_rhs]),
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
            operation.infer_output_types(&[lhs, rhs]),
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
                Sharding::with_unreduced_axes(
                    mesh.clone(),
                    vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    ["k"],
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[unreduced_lhs, plain_array(&[8, 16])]),
            Err(TypeError { message: "'dot' operands cannot be unreduced".to_string() }),
        );

        // Reduced operands are legal (this is what lets adjoint dots consume reduced cotangents), and their reduced
        // axes are unioned into the output sharding.
        let reduced_lhs = plain_array(&[4, 8])
            .with_sharding(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    ["k"],
                    Vec::<&str>::new(),
                )
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            operation.infer_output_types(&[reduced_lhs, plain_array(&[8, 16])]),
            Ok(vec![
                plain_array(&[4, 16])
                    .with_sharding(
                        Sharding::with_manual_axes(
                            mesh,
                            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
                            Vec::<&str>::new(),
                            ["k"],
                            Vec::<&str>::new(),
                        )
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
            operation.infer_output_types(&[lhs, rhs]),
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
            operation.infer_output_types(&[lhs.clone(), rhs.clone()]),
            Ok(vec![plain_array(&[2, 4, 16]).with_sharding(requested).unwrap()]),
        );

        // Rank validation.
        let rank_mismatched = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(rank_mismatched);
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()]),
            Err(TypeError { message: "'dot' output sharding rank (1) does not match the output rank (3)".to_string() }),
        );

        // Mesh validation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(other_mesh_sharding);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs]),
            Err(TypeError { message: "'dot' output sharding must use the same mesh as the operands".to_string() }),
        );

        // Auto mesh axes cannot be requested explicitly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_sharding =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(auto_sharding);
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])]),
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
        let unreduced = Sharding::with_unreduced_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            ["k"],
        )
        .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), rhs.clone()]),
            Ok(vec![plain_array(&[4, 16]).with_sharding(unreduced.clone()).unwrap()]),
        );

        // The contracting dimensions must be sharded identically.
        let replicated_rhs =
            sharded_array(&mesh, &[8, 16], vec![ShardingDimension::replicated(), ShardingDimension::replicated()]);
        let operation = DotOperation::matmul().with_output_sharding(unreduced.clone());
        assert_eq!(
            operation.infer_output_types(&[lhs.clone(), replicated_rhs.clone()]),
            Err(TypeError {
                message:
                    "'dot' contracting dimensions must be sharded identically when the output sharding is unreduced"
                        .to_string(),
            }),
        );

        // The unreduced set must equal the axes that shard the contracting dimensions.
        let mismatched = Sharding::with_unreduced_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            ["n"],
        )
        .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(mismatched);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs]),
            Err(TypeError {
                message:
                    "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                        .to_string(),
            }),
        );

        // Unsharded contracting dimensions cannot produce an unreduced output.
        let operation = DotOperation::matmul().with_output_sharding(unreduced);
        assert_eq!(
            operation.infer_output_types(&[
                replicated_rhs.clone(),
                sharded_array(&mesh, &[16, 4], vec![ShardingDimension::replicated(), ShardingDimension::replicated()],)
            ]),
            Err(TypeError {
                message:
                    "'dot' output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_dot_batching_stages_the_lifted_output_sharding() {
        use std::rc::Rc;

        use crate::batching::ArrayBatch;
        use crate::batching::BatchAxis;
        use crate::batching::BatchableOperation;
        use crate::batching::BatchingContext;
        use crate::parameters::Placeholder;
        use crate::tracing::TracingContext;
        use crate::tracing_v2::ArrayOperation;

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
        let batching_context = BatchingContext::new(context.clone(), 2, None);
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
        let outputs = operation.batch(batching_context.parent(), &[lhs, rhs]).unwrap();
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

    /// Minimal operation enum hosting the primal [`DotOperation`] (used for both the forward dot and its staged
    /// adjoint dot) plus the structural `zero` and `add` operations the transpose pass needs. It lets the
    /// partition-aware [`DotOperation`] transpose run on a program whose known operand is a program input. The
    /// `Constant` variant carries the value parameter `V` so the [`Operation`] derive can infer the primary type.
    #[derive(Clone, Debug, ryft_macros::Operation, ryft_macros::TransposableOperation)]
    enum TestDotOperation<V: Value<Type = ArrayType>> {
        Zero(ZeroOperation<ArrayType>),
        Constant(crate::operations::constants::ConstantOperation<V>),
        Add(AddOperation),
        Dot(DotOperation),
    }

    #[test]
    fn test_dot_partitioned_transpose_computes_operand_adjoints() {
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tests::TestArray;

        let matmul = DotDimensionNumbers::matmul();
        let left = TestArray::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = TestArray::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let cotangent = TestArray::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0]);
        let left_type = left.r#type().into_owned();
        let right_type = right.r#type().into_owned();

        // Known LEFT operand (linear RHS): the partition-aware transpose stages the adjoint of `t -> dot(left, t)`,
        // whose RHS cotangent is `dot(left^T, cotangent)`. Build `dot(left, right)` over the test enum, treat only the
        // RHS as linear, and interpret the pullback on `[cotangent, left]`.
        let mut builder = ProgramBuilder::<TestArray, TestDotOperation<TestArray>>::new();
        let left_input = builder.add_input(left_type.clone());
        let right_input = builder.add_input(right_type.clone());
        let product =
            builder.add_instruction(DotOperation::new(matmul.clone()), vec![left_input, right_input]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![product], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1]).unwrap();
        assert_eq!(pullback.output_ids().len(), 1, "the known left input must receive no cotangent output");
        let right_cotangents = pullback.interpret(vec![cotangent.clone(), left.clone()]).unwrap();
        assert_eq!(right_cotangents.len(), 1);
        assert_eq!(*right_cotangents[0].r#type(), right_type);
        // `left^T @ cotangent` with `left^T = [[1,4],[2,5],[3,6]]` and `cotangent = [[1,-2],[0.5,3]]`.
        assert_eq!(right_cotangents[0].values, vec![3.0, 10.0, 4.5, 11.0, 6.0, 12.0]);

        // Known RIGHT operand (linear LHS): the partition-aware transpose stages the adjoint of `t -> dot(t, right)`,
        // whose LHS cotangent is `dot(cotangent, right^T)`.
        let mut builder = ProgramBuilder::<TestArray, TestDotOperation<TestArray>>::new();
        let left_input = builder.add_input(left_type.clone());
        let right_input = builder.add_input(right_type.clone());
        let product =
            builder.add_instruction(DotOperation::new(matmul.clone()), vec![left_input, right_input]).unwrap()[0];
        let program = builder
            .build::<(TestArray, TestArray), TestArray>(vec![product], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0]).unwrap();
        assert_eq!(pullback.output_ids().len(), 1, "the known right input must receive no cotangent output");
        let left_cotangents = pullback.interpret(vec![cotangent, right]).unwrap();
        assert_eq!(left_cotangents.len(), 1);
        assert_eq!(*left_cotangents[0].r#type(), left_type);
        // `cotangent @ right^T` with `cotangent = [[1,-2],[0.5,3]]` and `right^T = [[7,9,11],[8,10,12]]`.
        assert_eq!(left_cotangents[0].values, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5]);
    }
}
