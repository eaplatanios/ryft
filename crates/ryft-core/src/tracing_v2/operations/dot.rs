use std::collections::BTreeSet;
use std::fmt::Display;

use half::{bf16, f16};

use crate::batching::BatchingError;
use crate::contexts::StagingContext;
use crate::differentiation::{Cotangent, Tangent, TransposableOperation};
use crate::macros::check_count;
use crate::operations::arithmetic::AddOperation;
use crate::operations::manipulation::Transpose;
use crate::operations::{InterpretableOperation, Operation, OperationFormatter};
use crate::programs::{ProgramError, Value};
use crate::sharding::{LogicalMesh, MeshAxisType, Sharding, ShardingDimension};
use crate::tracing::{AbstractTracingContext, Tracer};
use crate::tracing_v2::differentiation::{JvpTracer, LinearOperationOf, ResidualFactor, TangentContext};
use crate::tracing_v2::{DifferentiableOperation, DifferentiationContext};
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

impl<C> Dot for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    C::Operation: From<DotOperation>,
{
    #[inline]
    fn dot(&self, rhs: &Self, dimensions: &DotDimensionNumbers) -> Self {
        self.binary(rhs, DotOperation::new(dimensions.clone()))
    }

    #[inline]
    fn dot_with_output_sharding(
        &self,
        rhs: &Self,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        self.binary(rhs, DotOperation::new(dimensions.clone()).with_output_sharding(output_sharding.clone()))
    }
}

macro_rules! impl_dot_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Dot for $ty {
                #[inline]
                fn dot(&self, rhs: &Self, _dimensions: &DotDimensionNumbers) -> Self {
                    *self * *rhs
                }
            }
        )*
    };
}

impl_dot_for_scalar!(bf16, f16, f32, f64);

macro_rules! impl_left_right_dot_for_scalar {
    ($($ty:ty),* $(,)?) => {
        $(
            impl LeftDot for $ty {
                #[inline]
                fn left_dot(&self, factor: Self, _dimensions: &DotDimensionNumbers) -> Self {
                    factor * *self
                }
            }

            impl RightDot for $ty {
                #[inline]
                fn right_dot(&self, factor: Self, _dimensions: &DotDimensionNumbers) -> Self {
                    *self * factor
                }
            }
        )*
    };
}

impl_left_right_dot_for_scalar!(bf16, f16, f32, f64);

/// Generalized N-D dot and transpose capability.
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
/// forms ([`LeftDotOperation`], [`RightDotOperation`]) share this rule and report under the same name.
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
        return Err(TypeError { message: format!("{DOT_OPERATION_NAME} input element types are incompatible") });
    }
    let lhs_rank = lhs.rank();
    let rhs_rank = rhs.rank();
    let lhs_batching = dimensions.lhs_batching_dimensions();
    let rhs_batching = dimensions.rhs_batching_dimensions();
    let lhs_contracting = dimensions.lhs_contracting_dimensions();
    let rhs_contracting = dimensions.rhs_contracting_dimensions();

    if lhs_batching.len() != rhs_batching.len() {
        return Err(TypeError {
            message: format!("{DOT_OPERATION_NAME} batching dimensions have different lengths on the two operands"),
        });
    }
    if lhs_contracting.len() != rhs_contracting.len() {
        return Err(TypeError {
            message: format!("{DOT_OPERATION_NAME} contracting dimensions have different lengths on the two operands"),
        });
    }
    if lhs_batching.iter().any(|axis| *axis >= lhs_rank) || lhs_contracting.iter().any(|axis| *axis >= lhs_rank) {
        return Err(TypeError { message: format!("{DOT_OPERATION_NAME} LHS dimension index out of bounds") });
    }
    if rhs_batching.iter().any(|axis| *axis >= rhs_rank) || rhs_contracting.iter().any(|axis| *axis >= rhs_rank) {
        return Err(TypeError { message: format!("{DOT_OPERATION_NAME} RHS dimension index out of bounds") });
    }

    for (lhs_axis, rhs_axis) in lhs_batching.iter().zip(rhs_batching.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "{DOT_OPERATION_NAME} batching dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
                ),
            });
        }
    }
    for (lhs_axis, rhs_axis) in lhs_contracting.iter().zip(rhs_contracting.iter()) {
        if lhs.dimension(*lhs_axis as isize) != rhs.dimension(*rhs_axis as isize) {
            return Err(TypeError {
                message: format!(
                    "{DOT_OPERATION_NAME} contracting dimension sizes do not match (LHS axis {lhs_axis}, RHS axis {rhs_axis})"
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
            return Err(TypeError { message: format!("{DOT_OPERATION_NAME} operands cannot be unreduced") });
        }
    }

    let mesh = match (lhs_sharding, rhs_sharding) {
        (Some(left), Some(right)) if left.mesh() != right.mesh() => {
            return Err(TypeError {
                message: format!("{DOT_OPERATION_NAME} operand shardings must use the same mesh"),
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
                    "{DOT_OPERATION_NAME} output sharding rank ({}) does not match the output rank ({output_rank})",
                    output_sharding.rank(),
                ),
            });
        }
        if let Some(mesh) = mesh
            && output_sharding.mesh() != mesh
        {
            return Err(TypeError {
                message: format!("{DOT_OPERATION_NAME} output sharding must use the same mesh as the operands"),
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
                message: format!("{DOT_OPERATION_NAME} output sharding cannot reference auto mesh axes"),
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
                        "{DOT_OPERATION_NAME} contracting dimensions must be sharded identically when the output sharding is unreduced"
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
                        "{DOT_OPERATION_NAME} output sharding unreduced axes must equal the axes that shard the contracting dimensions"
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
                            "{DOT_OPERATION_NAME} contracting dimensions must have consistent shardings, but got {left} and {right}"
                        ),
                    });
                }
                return Err(TypeError {
                    message: format!(
                        "{DOT_OPERATION_NAME} contracting dimensions are sharded, making the output sharding ambiguous; request an \
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
                            "{DOT_OPERATION_NAME} batching dimensions must have consistent shardings, but got {left} and {right}"
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
            message: format!("{DOT_OPERATION_NAME} output sharding construction failed: {error}"),
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

impl<V: Value<ArrayType> + Dot> InterpretableOperation<ArrayType, V> for DotOperation {
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        // The requested output sharding flows through the capability method so that interpretation over staging
        // values (e.g., during program batching) preserves it; concrete values ignore it.
        Ok(vec![match &self.output_sharding {
            Some(output_sharding) => inputs[0].dot_with_output_sharding(&inputs[1], &self.dimensions, output_sharding),
            None => inputs[0].dot(&inputs[1], &self.dimensions),
        }])
    }
}

impl<V: Value<ArrayType> + crate::operations::manipulation::Broadcast>
    crate::tracing_v2::batching::BatchableOperation<V, V::InterpretationContext> for DotOperation
where
    DotOperation: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 2, ProgramError);
        let (_, input_axes, axis_size) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        // Mixed batched/unbatched: broadcast the lane-uniform operand to gain a singleton batch
        // axis at position 0 (JAX's `matchaxis(0)` convention), then fall through to the
        // both-batched arm of `lift_dot_dimensions`.
        let aligned_inputs: Vec<crate::tracing_v2::batching::ArrayBatch<V>> = match (input_axes[0], input_axes[1]) {
            (Some(_), Some(_)) | (None, None) => inputs.to_vec(),
            (Some(_), None) => {
                vec![inputs[0].clone(), crate::tracing_v2::batching::broadcast_to_batched(&inputs[1], 0, axis_size)?]
            }
            (None, Some(_)) => {
                vec![crate::tracing_v2::batching::broadcast_to_batched(&inputs[0], 0, axis_size)?, inputs[1].clone()]
            }
        };
        let (_, aligned_axes, _) = crate::tracing_v2::batching::batch_input_metadata(&aligned_inputs)?;
        let (lifted_dimensions, output_axis) = lift_dot_dimensions(&self.dimensions, aligned_axes[0], aligned_axes[1])
            .ok_or_else(|| BatchingError::MisalignedBatchAxes {
                message: "dot batching failed to lift its dimension numbers for the aligned batch axes".to_string(),
            })?;
        let batch_dimension = crate::tracing_v2::batching::batch_dimension_sharding(inputs)?;
        let lifted_op = DotOperation::new(lifted_dimensions).with_output_sharding(lift_output_sharding(
            self.output_sharding.as_ref(),
            output_axis,
            batch_dimension,
        )?);
        crate::tracing_v2::batching::apply_with_axes(context, &lifted_op, &aligned_inputs, &[output_axis])
    }
}

/// JVP rule for the generalized dot product.
///
/// The pushforward of `dot(A, B; D)` is `dot(ΔA, B; D) + dot(A, ΔB; D)`: each operand's
/// contribution is itself a dot with the same dimension numbers, holding the other operand
/// constant. The two contributions are staged through [`RightDotOperation`] (holding the
/// right primal `B` constant on the right) and [`LeftDotOperation`] (holding the left primal
/// `A` constant on the left), respectively. A requested output sharding is forwarded to both
/// tangent dots, since tangent shardings mirror their primals.
impl<D> DifferentiableOperation<D> for DotOperation
where
    D: DifferentiationContext<Type = ArrayType>,
    D::Value: Dot,
    LinearOperationOf<D>: From<AddOperation>
        + From<LeftDotOperation<ResidualFactor<ArrayType, D::Value>>>
        + From<RightDotOperation<ResidualFactor<ArrayType, D::Value>>>,
{
    fn jvp<'jvp>(
        &self,
        context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 2, ProgramError);
        let left = &inputs[0];
        let right = &inputs[1];
        // Compute the primal with the requested output sharding so staged primals (tracer-valued contexts) keep the
        // attribute; concrete values fall through to the plain kernel via the trait's default method.
        let primal = match &self.output_sharding {
            Some(output_sharding) => {
                left.primal().dot_with_output_sharding(right.primal(), &self.dimensions, output_sharding)
            }
            None => left.primal().dot(right.primal(), &self.dimensions),
        };
        let left_term = match left.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(match &self.output_sharding {
                Some(output_sharding) => {
                    tangent.right_dot_with_output_sharding(right.factor(context), &self.dimensions, output_sharding)
                }
                None => tangent.right_dot(right.factor(context), &self.dimensions),
            }),
        };
        let right_term = match right.tangent().clone() {
            Tangent::Zero(r#type) => Tangent::Zero(r#type),
            Tangent::Value(tangent) => Tangent::Value(match &self.output_sharding {
                Some(output_sharding) => {
                    tangent.left_dot_with_output_sharding(left.factor(context), &self.dimensions, output_sharding)
                }
                None => tangent.left_dot(left.factor(context), &self.dimensions),
            }),
        };
        let tangent = left_term + right_term;
        Ok(vec![JvpTracer::new(primal, tangent)])
    }
}

/// Value-level "factor-on-the-left" dot capability.
///
/// `t.left_dot(factor, dimensions)` computes `dot(factor, t; dimensions)`. This is the linear
/// map produced by [`DotOperation`]'s JVP when the LHS primal is held constant.
pub trait LeftDot<F = Self>: Sized {
    /// Computes `dot(factor, self; dimensions)`.
    fn left_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self;

    /// Computes `dot(factor, self; dimensions)`, requesting `output_sharding` for the result. Refer to the
    /// documentation of [`Dot::dot_with_output_sharding`] for the contract. The default implementation ignores the
    /// requested sharding, which is correct for concrete (single-device) values.
    fn left_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        let _ = output_sharding;
        self.left_dot(factor, dimensions)
    }
}

/// Value-level "factor-on-the-right" dot capability.
///
/// `t.right_dot(factor, dimensions)` computes `dot(t, factor; dimensions)`. This is the linear
/// map produced by [`DotOperation`]'s JVP when the RHS primal is held constant.
pub trait RightDot<F = Self>: Sized {
    /// Computes `dot(self, factor; dimensions)`.
    fn right_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self;

    /// Computes `dot(self, factor; dimensions)`, requesting `output_sharding` for the result. Refer to the
    /// documentation of [`Dot::dot_with_output_sharding`] for the contract. The default implementation ignores the
    /// requested sharding, which is correct for concrete (single-device) values.
    fn right_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        let _ = output_sharding;
        self.right_dot(factor, dimensions)
    }
}

impl<C, F> LeftDot<F> for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    F: Value<ArrayType>,
    C::Operation: From<LeftDotOperation<F>>,
{
    #[inline]
    fn left_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        self.unary(LeftDotOperation::new(factor, dimensions.clone()))
    }

    #[inline]
    fn left_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        self.unary(LeftDotOperation::new(factor, dimensions.clone()).with_output_sharding(output_sharding.clone()))
    }
}

impl<C, F> RightDot<F> for Tracer<C>
where
    C: StagingContext<Type = ArrayType>,
    F: Value<ArrayType>,
    C::Operation: From<RightDotOperation<F>>,
{
    #[inline]
    fn right_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        self.unary(RightDotOperation::new(factor, dimensions.clone()))
    }

    #[inline]
    fn right_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        self.unary(RightDotOperation::new(factor, dimensions.clone()).with_output_sharding(output_sharding.clone()))
    }
}

/// Symbolic-zero-aware tangent left dot. `Zero.left_dot(_, _) -> Zero`. The output-sharding variant is overridden so
/// that the requested sharding reaches the wrapped value instead of being dropped by the provided default.
impl<T, V, F> LeftDot<F> for crate::differentiation::Tangent<T, V>
where
    T: crate::types::Type,
    V: crate::programs::Value<T> + LeftDot<F>,
{
    #[inline]
    fn left_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type.clone()),
            Self::Value(value) => Self::Value(value.left_dot(factor, dimensions)),
        }
    }

    #[inline]
    fn left_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type.clone()),
            Self::Value(value) => Self::Value(value.left_dot_with_output_sharding(factor, dimensions, output_sharding)),
        }
    }
}

/// Symbolic-zero-aware tangent right dot. `Zero.right_dot(_, _) -> Zero`. The output-sharding variant is overridden
/// so that the requested sharding reaches the wrapped value instead of being dropped by the provided default.
impl<T, V, F> RightDot<F> for crate::differentiation::Tangent<T, V>
where
    T: crate::types::Type,
    V: crate::programs::Value<T> + RightDot<F>,
{
    #[inline]
    fn right_dot(&self, factor: F, dimensions: &DotDimensionNumbers) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type.clone()),
            Self::Value(value) => Self::Value(value.right_dot(factor, dimensions)),
        }
    }

    #[inline]
    fn right_dot_with_output_sharding(
        &self,
        factor: F,
        dimensions: &DotDimensionNumbers,
        output_sharding: &Sharding,
    ) -> Self {
        match self {
            Self::Zero(r#type) => Self::Zero(r#type.clone()),
            Self::Value(value) => {
                Self::Value(value.right_dot_with_output_sharding(factor, dimensions, output_sharding))
            }
        }
    }
}

/// Captured-factor "left dot" linear operation.
///
/// Represents the linear map `t ↦ dot(factor, t; dimensions)`. Emitted by [`DotOperation`]'s
/// JVP rule when the LHS primal is held constant, and by the transpose of
/// [`RightDotOperation`] (the adjoint of `t ↦ dot(t, factor; dimensions)`).
#[derive(Clone, Debug, PartialEq)]
pub struct LeftDotOperation<F> {
    /// Captured constant factor (the LHS of the underlying dot).
    factor: F,

    /// Dimension numbers of the underlying dot.
    dimensions: DotDimensionNumbers,

    /// Optional requested output [`Sharding`]. Refer to the documentation of
    /// [`DotOperation::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl<F: Value<ArrayType>> LeftDotOperation<F> {
    /// Creates a new [`LeftDotOperation`].
    #[inline]
    pub fn new(factor: F, dimensions: DotDimensionNumbers) -> Self {
        Self { factor, dimensions, output_sharding: None }
    }

    /// Attaches a requested output [`Sharding`] to this operation. Refer to the documentation of
    /// [`DotOperation::with_output_sharding`] for the semantics.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the captured constant factor.
    #[inline]
    pub fn factor(&self) -> &F {
        &self.factor
    }

    /// Returns the dimension numbers of the underlying dot.
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

impl<F: Value<ArrayType>> Display for LeftDotOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for LeftDotOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "left_dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![dot_abstract(
            self.factor.r#type().as_ref(),
            &input_types[0],
            &self.dimensions,
            self.output_sharding.as_ref(),
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("factor", &self.factor)?;
            operation.field("dimensions", &self.dimensions)?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<F, V> InterpretableOperation<ArrayType, V> for LeftDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType> + LeftDot<F>,
{
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // dot(factor, input; dimensions): factor is on the left. The requested output sharding flows through the
        // capability method so that interpretation over staging values preserves it.
        Ok(vec![match &self.output_sharding {
            Some(output_sharding) => {
                inputs[0]
                    .clone()
                    .left_dot_with_output_sharding(self.factor.clone(), &self.dimensions, output_sharding)
            }
            None => inputs[0].clone().left_dot(self.factor.clone(), &self.dimensions),
        }])
    }
}

/// Captured-factor "right dot" linear operation.
///
/// Represents the linear map `t ↦ dot(t, factor; dimensions)`. Emitted by [`DotOperation`]'s
/// JVP rule when the RHS primal is held constant, and by the transpose of
/// [`LeftDotOperation`] (the adjoint of `t ↦ dot(factor, t; dimensions)`).
#[derive(Clone, Debug, PartialEq)]
pub struct RightDotOperation<F> {
    /// Captured constant factor (the RHS of the underlying dot).
    factor: F,

    /// Dimension numbers of the underlying dot.
    dimensions: DotDimensionNumbers,

    /// Optional requested output [`Sharding`]. Refer to the documentation of
    /// [`DotOperation::with_output_sharding`].
    output_sharding: Option<Sharding>,
}

impl<F: Value<ArrayType>> RightDotOperation<F> {
    /// Creates a new [`RightDotOperation`].
    #[inline]
    pub fn new(factor: F, dimensions: DotDimensionNumbers) -> Self {
        Self { factor, dimensions, output_sharding: None }
    }

    /// Attaches a requested output [`Sharding`] to this operation. Refer to the documentation of
    /// [`DotOperation::with_output_sharding`] for the semantics.
    #[inline]
    pub fn with_output_sharding(mut self, output_sharding: impl Into<Option<Sharding>>) -> Self {
        self.output_sharding = output_sharding.into();
        self
    }

    /// Returns the captured constant factor.
    #[inline]
    pub fn factor(&self) -> &F {
        &self.factor
    }

    /// Returns the dimension numbers of the underlying dot.
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

impl<F: Value<ArrayType>> Display for RightDotOperation<F> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<F: Value<ArrayType>> Operation<ArrayType> for RightDotOperation<F> {
    #[inline]
    fn name(&self) -> &'static str {
        "right_dot"
    }

    fn infer_output_types(&self, input_types: &[ArrayType]) -> Result<Vec<ArrayType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![dot_abstract(
            &input_types[0],
            self.factor.r#type().as_ref(),
            &self.dimensions,
            self.output_sharding.as_ref(),
        )?])
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?.bracketed(|operation| {
            operation.field("factor", &self.factor)?;
            operation.field("dimensions", &self.dimensions)?;
            if let Some(output_sharding) = &self.output_sharding {
                operation.field("output_sharding", output_sharding)?;
            }
            Ok(())
        })
    }
}

impl<F, V> InterpretableOperation<ArrayType, V> for RightDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType> + RightDot<F>,
{
    fn interpret(
        &self,
        _context: &<V as Value<ArrayType>>::InterpretationContext,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        // dot(input, factor; dimensions): factor is on the right. The requested output sharding flows through the
        // capability method so that interpretation over staging values preserves it.
        Ok(vec![match &self.output_sharding {
            Some(output_sharding) => {
                inputs[0]
                    .clone()
                    .right_dot_with_output_sharding(self.factor.clone(), &self.dimensions, output_sharding)
            }
            None => inputs[0].clone().right_dot(self.factor.clone(), &self.dimensions),
        }])
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
/// Given the per-lane dimension numbers and the batch-axis positions of the two operands (each
/// optional — `None` indicates a lane-uniform operand), returns the dimension numbers that
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

/// Lifts the dimension numbers of a captured-factor [`LeftDotOperation`] through one batching
/// level applied to its single (non-factor) input.
///
/// The factor stays lane-uniform; only the RHS operand of the underlying dot gains a new batch
/// dimension at position `k = t_batch_axis`. Existing RHS contracting / batching indices `i`
/// are shifted to `i + 1` when `i >= k`. The new axis at `k` becomes an RHS result axis (it
/// has no counterpart on the factor, so it can't be batching, and it isn't contracting).
///
/// Returns the lifted dimension numbers plus the output axis position. The output structure is
/// `[lhs_batching..., lhs_result..., rhs_result...]`, and the new batch axis ends up at
/// `lhs_batching_count + lhs_result_count + k_position_in_rhs_result`.
pub fn lift_left_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    factor_rank: usize,
    t_batch_axis: Option<usize>,
) -> (DotDimensionNumbers, Option<usize>) {
    let Some(k) = t_batch_axis else {
        return (dimensions.clone(), None);
    };
    let shift = |axes: &[usize]| -> Vec<usize> { axes.iter().map(|i| if *i >= k { i + 1 } else { *i }).collect() };
    let lifted = DotDimensionNumbers {
        lhs_contracting_dimensions: dimensions.lhs_contracting_dimensions.clone(),
        rhs_contracting_dimensions: shift(&dimensions.rhs_contracting_dimensions),
        lhs_batching_dimensions: dimensions.lhs_batching_dimensions.clone(),
        rhs_batching_dimensions: shift(&dimensions.rhs_batching_dimensions),
    };
    let lhs_batching_count = dimensions.lhs_batching_dimensions.len();
    let lhs_result_count = factor_rank - lhs_batching_count - dimensions.lhs_contracting_dimensions.len();
    let rhs_non_result: std::collections::BTreeSet<usize> = dimensions
        .rhs_contracting_dimensions
        .iter()
        .copied()
        .chain(dimensions.rhs_batching_dimensions.iter().copied())
        .collect();
    let k_position_in_rhs_result = (0..k).filter(|i| !rhs_non_result.contains(i)).count();
    let output_axis = lhs_batching_count + lhs_result_count + k_position_in_rhs_result;
    (lifted, Some(output_axis))
}

/// Lifts the dimension numbers of a captured-factor [`RightDotOperation`] through one batching
/// level applied to its single (non-factor) input.
///
/// Symmetric to [`lift_left_dot_dimensions`]: the LHS operand of the underlying dot is the
/// non-factor input, so it gains the new batch dimension. The new axis at `k` becomes an LHS
/// result axis in the output.
pub fn lift_right_dot_dimensions(
    dimensions: &DotDimensionNumbers,
    t_batch_axis: Option<usize>,
) -> (DotDimensionNumbers, Option<usize>) {
    let Some(k) = t_batch_axis else {
        return (dimensions.clone(), None);
    };
    let shift = |axes: &[usize]| -> Vec<usize> { axes.iter().map(|i| if *i >= k { i + 1 } else { *i }).collect() };
    let lifted = DotDimensionNumbers {
        lhs_contracting_dimensions: shift(&dimensions.lhs_contracting_dimensions),
        rhs_contracting_dimensions: dimensions.rhs_contracting_dimensions.clone(),
        lhs_batching_dimensions: shift(&dimensions.lhs_batching_dimensions),
        rhs_batching_dimensions: dimensions.rhs_batching_dimensions.clone(),
    };
    let lhs_batching_count = dimensions.lhs_batching_dimensions.len();
    let lhs_non_result: std::collections::BTreeSet<usize> = dimensions
        .lhs_contracting_dimensions
        .iter()
        .copied()
        .chain(dimensions.lhs_batching_dimensions.iter().copied())
        .collect();
    let k_position_in_lhs_result = (0..k).filter(|i| !lhs_non_result.contains(i)).count();
    let output_axis = lhs_batching_count + k_position_in_lhs_result;
    (lifted, Some(output_axis))
}

/// Lifts an optional requested output sharding through one batching level by inserting `batch_dimension` at the new
/// output batch axis. `batch_dimension` is the [`ShardingDimension`] derived from the batched inputs' mapped axis
/// (see [`batch_dimension_sharding`](crate::tracing_v2::batching::batch_dimension_sharding)), so the batched
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

/// Computes the dimension numbers for the adjoint of [`LeftDotOperation`]: maps the linear map
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

/// Computes the dimension numbers for the adjoint of [`RightDotOperation`]: maps the linear map
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

/// Transpose rule for [`LeftDotOperation`]. The adjoint is itself a left dot with the adjoint dimension numbers, and
/// its output sharding is pinned to the cotangent dual of the transposed input's sharding (the ryft analogue of JAX
/// reading the spec off the cotangent-typed accumulator): the produced value *is* that input's cotangent, so its
/// sharding swaps the input's unreduced and reduced axes instead of being re-derived. The cotangent of an
/// unreduced-output dot arrives typed reduced, which is a legal dot operand (only unreduced operands are rejected).
impl<V: Value<ArrayType> + Dot, F: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for LeftDotOperation<F>
where
    O: Operation<ArrayType> + From<LeftDotOperation<F>>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        let factor_rank = self.factor.r#type().as_ref().rank();
        let adjoint_dims = adjoint_dimensions_for_left_dot(&self.dimensions, factor_rank);
        let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                let contribution = match &adjoint_output_sharding {
                    Some(output_sharding) => {
                        cotangent.left_dot_with_output_sharding(self.factor.clone(), &adjoint_dims, output_sharding)
                    }
                    None => cotangent.left_dot(self.factor.clone(), &adjoint_dims),
                };
                Ok(vec![Cotangent::Staged(contribution)])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

/// Transpose rule for [`RightDotOperation`]. The adjoint is itself a right dot with the adjoint dimension numbers,
/// and its output sharding is pinned to the cotangent dual of the transposed input's sharding. Refer to the
/// documentation of the [`LeftDotOperation`] transpose rule for the rationale.
impl<V: Value<ArrayType> + Dot, F: Value<ArrayType>, O> TransposableOperation<ArrayType, V, O> for RightDotOperation<F>
where
    O: Operation<ArrayType> + From<RightDotOperation<F>>,
{
    fn transpose<'transpose>(
        &self,
        _context: &mut AbstractTracingContext<'transpose, ArrayType, V, O>,
        input_types: &[&ArrayType],
        output_cotangents: &[Cotangent<'transpose, ArrayType, V, O>],
    ) -> Result<Vec<Cotangent<'transpose, ArrayType, V, O>>, ProgramError> {
        check_count!("input", input_types, 1, ProgramError);
        check_count!("output", output_cotangents, 1, ProgramError);
        let factor_rank = self.factor.r#type().as_ref().rank();
        let cotangent_rank = match &output_cotangents[0] {
            Cotangent::Staged(value) => value.r#type().as_ref().rank(),
            Cotangent::Zero => {
                return Ok(vec![Cotangent::Zero]);
            }
        };
        // t_rank = (batching + lhs_result) + lhs_contracting
        //        = (cotangent_rank - rhs_result_count) + lhs_contracting_count
        //        = cotangent_rank + 2 * rhs_contracting_count + rhs_batching_count - factor_rank.
        let t_rank = cotangent_rank
            + 2 * self.dimensions.rhs_contracting_dimensions.len()
            + self.dimensions.rhs_batching_dimensions.len()
            - factor_rank;
        let adjoint_dims = adjoint_dimensions_for_right_dot(&self.dimensions, factor_rank, t_rank);
        let adjoint_output_sharding = input_types[0].sharding().map(Sharding::cotangent);
        match &output_cotangents[0] {
            Cotangent::Staged(cotangent) => {
                let contribution = match &adjoint_output_sharding {
                    Some(output_sharding) => {
                        cotangent.right_dot_with_output_sharding(self.factor.clone(), &adjoint_dims, output_sharding)
                    }
                    None => cotangent.right_dot(self.factor.clone(), &adjoint_dims),
                };
                Ok(vec![Cotangent::Staged(contribution)])
            }
            Cotangent::Zero => Ok(vec![Cotangent::Zero]),
        }
    }
}

impl<F, V> crate::tracing_v2::batching::BatchableOperation<V, V::InterpretationContext> for LeftDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>,
    LeftDotOperation<F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let factor_rank = self.factor.r#type().as_ref().rank();
        let (lifted_dimensions, output_axis) = lift_left_dot_dimensions(&self.dimensions, factor_rank, input_axes[0]);
        let batch_dimension = crate::tracing_v2::batching::batch_dimension_sharding(inputs)?;
        let lifted_op = LeftDotOperation::new(self.factor.clone(), lifted_dimensions)
            .with_output_sharding(lift_output_sharding(self.output_sharding.as_ref(), output_axis, batch_dimension)?);
        crate::tracing_v2::batching::apply_with_axes(context, &lifted_op, inputs, &[output_axis])
    }
}

impl<F, V> crate::tracing_v2::batching::BatchableOperation<V, V::InterpretationContext> for RightDotOperation<F>
where
    F: Value<ArrayType>,
    V: Value<ArrayType>,
    RightDotOperation<F>: InterpretableOperation<ArrayType, V>,
{
    fn batch(
        &self,
        context: &V::InterpretationContext,
        inputs: &[crate::tracing_v2::batching::ArrayBatch<V>],
    ) -> Result<Vec<crate::tracing_v2::batching::ArrayBatch<V>>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        let (_, input_axes, _) = crate::tracing_v2::batching::batch_input_metadata(inputs)?;
        let (lifted_dimensions, output_axis) = lift_right_dot_dimensions(&self.dimensions, input_axes[0]);
        let batch_dimension = crate::tracing_v2::batching::batch_dimension_sharding(inputs)?;
        let lifted_op = RightDotOperation::new(self.factor.clone(), lifted_dimensions)
            .with_output_sharding(lift_output_sharding(self.output_sharding.as_ref(), output_axis, batch_dimension)?);
        crate::tracing_v2::batching::apply_with_axes(context, &lifted_op, inputs, &[output_axis])
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
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
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
                message: "dot contracting dimension sizes do not match (LHS axis 2, RHS axis 1)".to_string(),
            }),
        );
        let mismatched_batch_rhs = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Size::Dynamic(Some(8)), Size::Dynamic(Some(4)), Size::Static(3)]),
        );
        assert_eq!(
            operation.infer_output_types(&[lhs, mismatched_batch_rhs]),
            Err(TypeError {
                message: "dot batching dimension sizes do not match (LHS axis 0, RHS axis 0)".to_string(),
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
                message: "dot batching dimensions must have consistent shardings, but got {'b'} and {'m'}".to_string(),
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
                message: "dot contracting dimensions are sharded, making the output sharding ambiguous; request an \
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
                message: "dot contracting dimensions must have consistent shardings, but got {'k'} and {'m'}"
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
            Err(TypeError { message: "dot operand shardings must use the same mesh".to_string() }),
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
            Err(TypeError { message: "dot operands cannot be unreduced".to_string() }),
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
            Err(TypeError { message: "dot output sharding rank (1) does not match the output rank (3)".to_string() }),
        );

        // Mesh validation.
        let other_mesh = LogicalMesh::new(vec![MeshAxis::new("m", 4, MeshAxisType::Explicit).unwrap()]).unwrap();
        let other_mesh_sharding = Sharding::replicated(other_mesh, 3);
        let operation = DotOperation::new(DotDimensionNumbers::new(vec![2], vec![1], vec![0], vec![0]))
            .with_output_sharding(other_mesh_sharding);
        assert_eq!(
            operation.infer_output_types(&[lhs, rhs]),
            Err(TypeError { message: "dot output sharding must use the same mesh as the operands".to_string() }),
        );

        // Auto mesh axes cannot be requested explicitly.
        let auto_mesh = LogicalMesh::new(vec![MeshAxis::new("a", 2, MeshAxisType::Auto).unwrap()]).unwrap();
        let auto_sharding =
            Sharding::new(auto_mesh, vec![ShardingDimension::sharded(["a"]), ShardingDimension::replicated()]).unwrap();
        let operation = DotOperation::matmul().with_output_sharding(auto_sharding);
        assert_eq!(
            operation.infer_output_types(&[plain_array(&[4, 8]), plain_array(&[8, 16])]),
            Err(TypeError { message: "dot output sharding cannot reference auto mesh axes".to_string() }),
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
                message: "dot contracting dimensions must be sharded identically when the output sharding is unreduced"
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
                message: "dot output sharding unreduced axes must equal the axes that shard the contracting dimensions"
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
                message: "dot output sharding unreduced axes must equal the axes that shard the contracting dimensions"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_dot_batching_stages_the_lifted_output_sharding() {
        use std::cell::RefCell;
        use std::rc::Rc;

        use crate::domains::AbstractDomain;
        use crate::parameters::Placeholder;
        use crate::programs::ProgramBuilder;
        use crate::tracing::AbstractTracingContext;
        use crate::tracing_v2::ArrayOperation;
        use crate::tracing_v2::batching::{ArrayBatch, BatchableOperation, BatchingContext};

        let mesh = test_mesh();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["m"]), ShardingDimension::sharded(["n"])])
                .unwrap();
        let operation = DotOperation::matmul().with_output_sharding(output_sharding.clone());

        // Batch the operation over tracer inputs, which is how program batching applies lifted operations: the
        // staged batched dot must carry the lifted output sharding instead of dropping it.
        let builder =
            Rc::new(RefCell::new(ProgramBuilder::<ArrayType, ArrayType, ArrayOperation<ArrayType, ArrayType>>::new()));
        let lhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 4, 8]));
        let rhs_atom = builder.borrow_mut().add_input(plain_array(&[2, 8, 16]));
        let domain = AbstractDomain::new();
        let context = AbstractTracingContext::new(&domain, builder.clone());
        let batching_context = BatchingContext::new(context.clone(), 2);
        let lhs = ArrayBatch::mapped(context.tracer(lhs_atom, None), 0).unwrap();
        let rhs = ArrayBatch::mapped(context.tracer(rhs_atom, None), 0).unwrap();
        let outputs = operation.batch(batching_context.parent_context(), &[lhs, rhs]).unwrap();
        assert_eq!(outputs[0].batch_axis(), Some(0));
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
    fn test_dot_linearization_and_transposition_thread_output_shardings() {
        use indoc::indoc;

        use crate::tests::{TestArray, TestArrayDomain};
        use crate::tracing_v2::DifferentiationContext;

        // End-to-end unreduced lifecycle: a forward dot with contracting dimensions sharded along `x` requests an
        // unreduced output. The JVP must forward that output sharding to both tangent dots, and the transposed
        // pullback must (a) type the output cotangent input with the reduced dual, and (b) pin each adjoint dot's
        // output sharding to the cotangent dual of the corresponding input's sharding.
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let lhs_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let rhs_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let output_sharding = Sharding::with_unreduced_axes(
            mesh.clone(),
            vec![ShardingDimension::replicated(), ShardingDimension::replicated()],
            ["x"],
        )
        .unwrap();
        let lhs = TestArray::new(plain_array(&[2, 2]).with_sharding(lhs_sharding).unwrap(), vec![1.0; 4]);
        let rhs = TestArray::new(plain_array(&[2, 2]).with_sharding(rhs_sharding).unwrap(), vec![1.0; 4]);

        let dimensions = DotDimensionNumbers::matmul();
        let (primal, pushforward) = TestArrayDomain
            .linearize(
                |inputs| Ok(inputs.0.dot_with_output_sharding(&inputs.1, &dimensions, &output_sharding)),
                (lhs, rhs),
            )
            .unwrap();
        assert_eq!(primal.values(), &[2.0; 4]);

        let pushforward = pushforward.instantiate_program().unwrap();
        assert_eq!(
            pushforward.to_string(),
            indoc! {"
            lambda %0:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {'x'}]}], %1:f32[2, 2][sharding={mesh<['x'=2]>, [{'x'}, {}]}] .
            let %2:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {}], unreduced={'x'}}] = right_dot [
                factor=[1.0, 1.0, 1.0, 1.0],
                dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                output_sharding={mesh<['x'=2]>, [{}, {}], unreduced={'x'}},
            ] %0
                %3:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {}], unreduced={'x'}}] = left_dot [
                    factor=[1.0, 1.0, 1.0, 1.0],
                    dimensions=(lhs_contracting=[1], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    output_sharding={mesh<['x'=2]>, [{}, {}], unreduced={'x'}},
                ] %1
                %4:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {}], unreduced={'x'}}] = add %2 %3
            in (%4)
            "}
            .trim_end(),
        );

        // The output cotangent input is typed with the reduced dual of the unreduced output, the adjoint dots
        // legally consume it, and each adjoint's output sharding is pinned to the cotangent dual of the
        // corresponding primal input's sharding, yielding sharded gradients with no communication.
        let pullback = pushforward.transpose().unwrap();
        assert_eq!(
            pullback.to_string(),
            indoc! {"
            lambda %0:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {}], reduced={'x'}}] .
            let %1:f32[2, 2][sharding={mesh<['x'=2]>, [{'x'}, {}]}] = left_dot [
                factor=[1.0, 1.0, 1.0, 1.0],
                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                output_sharding={mesh<['x'=2]>, [{'x'}, {}]},
            ] %0
                %2:f32[2, 2][sharding={mesh<['x'=2]>, [{}, {'x'}]}] = right_dot [
                    factor=[1.0, 1.0, 1.0, 1.0],
                    dimensions=(lhs_contracting=[1], rhs_contracting=[1], lhs_batching=[], rhs_batching=[]),
                    output_sharding={mesh<['x'=2]>, [{}, {'x'}]},
                ] %0
            in (%2, %1)
            "}
            .trim_end(),
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
}
