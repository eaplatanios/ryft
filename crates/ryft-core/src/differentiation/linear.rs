use std::fmt::Display;
use std::ops::Range;

use crate::axes::Axis;
use crate::backends::array_programs::batching::ArrayProgramBatching;
use crate::batching::{
    ArrayBatching, ArrayBatchingPolicy, BatchAxis, BatchableOperation, BatchedProgram, BatchingContext, BatchingDriver,
    BatchingError, BatchingPolicy, ProgramBatchingOutputAxesPolicy,
};
use crate::contexts::{Context, Domain};
use crate::differentiation::DifferentiationError;
use crate::differentiation::forward::{DifferentiableOperation, DifferentiationDriver, DifferentiationDual};
use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
use crate::differentiation::types::DifferentiableType;
use crate::interpretation::{InterpretableOperation, InterpretationDriver};
use crate::macros::{check_count, check_types};
use crate::operations::constants::{Zero, ZeroOperation, ZeroOperationProvider};
use crate::operations::math::{Reduce, ReduceOperation, ReductionKind};
use crate::partial::{PartialValue, PartiallyEvaluatableOperation};
use crate::programs::ProgramError;
use crate::programs::atoms::{AtomId, MaybeZero};
use crate::programs::builders::ProgramBuilder;
use crate::programs::identities::TypeIdentityRenaming;
use crate::programs::operations::{Operation, OperationFormatter, OperationProjection};
use crate::programs::regions::{OutputRegionProvenance, RegionInterface, RegionSlot};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{Value, ValueProjection};
use crate::tracing::{NestedTracingContext, Tracer, TracingContext};
use crate::types::{ArrayProgramType, ArrayType};

/// Differentiation-owned protocol through which an operation family materializes zeros whose runtime geometry must
/// be supplied by explicitly captured _residual_ values, because it is not derivable from the zero's [`Type`] alone.
///
/// # Why this Protocol Exists
///
/// Differentiation is the one transform that must synthesize values with no data edge to derive them from. For example,
/// transposition is *defined* to return a cotangent for every differentiated input, including inputs that are
/// disconnected from every output, and the mathematically determined value for such an input is a zero of its cotangent
/// type. For a static type this is easy as [`ZeroOperationProvider::zero_operation`] constructs the zero from the type,
/// with no operands. For a type with dynamic axes it is impossible as a [`Type`] carries only dimension _identities_
/// and bounds, never defining values, and so the zero operation needs one explicit dimension operand per dynamic axis.
/// Also, the value that could supply those operands (i.e., the primal input the cotangent corresponds to) is _not an
/// input of the pullback program_ where the zero must be staged. The only moment both the need and the geometry are in
/// scope is during linearization, and so the required extents must be captured then and threaded to transposition as
/// ordinary residuals. This trait is that capture/spend contract, expressed once per operation *family*. It is a set of
/// associated functions with no receiver (i.e., `self` argument), because it is invoked precisely when no [`Operation`]
/// instance exists.
///
/// # How to Use It
///
/// The protocol has three steps, executed by the differentiation machinery rather than by implementors:
///
///   1. **Declare:** When linearization finds a differentiated input whose cotangent will be disconnected, it calls
///      [`Self::zero_residual_types`] with the zero's type to learn which residual values the eventual zero needs
///      (e.g., one first-class dimension per *distinct* dynamic identity for a composite array type, and nothing for
///      a static type).
///   2. **Capture:** While the primal value is still in scope, linearization records those residuals from it.
///      [`Self::capture_zero_residuals`] stages the reads into the program being built (i.e., the program-level
///      [`Program::linearize`](crate::Program::linearize) path), and [`Self::capture_zero_residual_values`] is its
///      value-level counterpart for reusable derivative callables that close over concrete or tracer values. Program
///      transposition appends captured residuals to its ordinary trailing residual suffix. Reusable callables retain
///      boundary-reconstruction residuals beside that executable program.
///   3. **Spend:** [`Self::zero_operation_with_residuals`] assembles the zero operation and its operands from the
///      captured residuals. Callers then stage that operation inside a pullback program or bind it in the originating
///      [`Context`] of a reusable value-level derivative callable.
///
/// The three steps must agree on residual count and order. Every mismatch is a loud typed error (the capture sites
/// validate against the declared types, and the spend site validates the residual count), never a silently wrong-shaped
/// zero.
///
/// # Who Implements It
///
/// Almost nobody needs to implement this trait, by design. Every operation family with an input-free zero (i.e., every
/// family with a `From<ZeroOperation<T>>` conversion) receives the whole protocol through a blanket implementation
/// that declares nothing, captures nothing, and spends by constructing the type-only zero (i.e., the fail-loud default
/// rejects unexpected residuals rather than ignoring them, so a mismatched linearize/transpose pairing cannot be
/// silently accepted). Only families whose zero genuinely consumes runtime-geometry operands (e.g., the composite
/// program family and its XLA counterpart) override the declaration, capture, and operation-assembly functions. Every
/// spending path reuses that shared assembly.
///
/// [`LinearCallOperation`] below is this protocol's sibling. It retains residual geometry for the transpose of a
/// *non-trivial* residual-parameterized linear map by attaching explicit forward/transpose regions to an instruction,
/// while this trait retains it for the degenerate zero map, which has no instruction to attach anything to. Both exist
/// for the same reason (i.e., reverse mode needs geometry at a moment when its defining values would otherwise be out
/// of scope) and both keep residual selection and threading owned by the differentiation transform rather than leaking
/// into primal operation payloads.
pub trait ResidualZeroProvider<T: Type>: ZeroOperationProvider<T> {
    /// Returns the types of the residual values that a zero of `r#type` needs, in the exact order in which
    /// [`Self::capture_zero_residuals`] captures them and [`Self::zero_operation_with_residuals`] consumes them.
    /// Input-free [`Operation`] families use the empty default. The array-dimension composite family returns one
    /// dimension type per _distinct_ dynamic identity of `r#type`, in first-occurrence order, so repeated axes share
    /// one residual.
    #[inline]
    fn zero_residual_types(_type: &T) -> Vec<T> {
        Vec::new()
    }

    /// Stages instructions into `builder` that read the residual values declared by [`Self::zero_residual_types`] from
    /// the primal value `source` (e.g., one `dimension_size` read per declared residual), returning the new atoms in
    /// declaration order. Linearization calls this while `source` is still in scope of the program being built. The
    /// returned atoms are then threaded to transposition as ordinary residuals. Input-free [`Operation`] families
    /// capture nothing.
    #[inline]
    fn capture_zero_residuals<V: Value<Type = T>>(
        _builder: &mut ProgramBuilder<V, Self>,
        _source: AtomId,
        _type: &T,
    ) -> Result<Vec<AtomId>, ProgramError> {
        Ok(Vec::new())
    }

    /// Captures the residual values declared by [`Self::zero_residual_types`] from the primal value `source`
    /// in a live `context`, returning them in declaration order. This is the value-level counterpart of
    /// [`Self::capture_zero_residuals`] used by reusable pullback callables, whose captured residuals are
    /// concrete values or tracers closed over by the callable rather than atoms of a program under construction.
    #[inline]
    fn capture_zero_residual_values<C: Context<Type = T, Operation = Self>>(
        _context: &C,
        _source: &C::Value,
        _type: &T,
    ) -> Result<Vec<C::Value>, ProgramError> {
        Ok(Vec::new())
    }

    /// Returns the canonical zero operation for `r#type` and expands `residuals` into its operand order. The default
    /// represents an input-free zero operation. Families whose zero consumes runtime geometry override this function
    /// so that value-level binding, residualization, and builder-level staging share one operation assembly.
    #[inline]
    fn zero_operation_with_residuals<R: Clone>(r#type: T, residuals: &[R]) -> Result<(Self, Vec<R>), ProgramError>
    where
        Self: Operation<Type = T>,
    {
        if !residuals.is_empty() {
            return Err(ProgramError::InvalidArgument {
                message: format!("input-free zero expected 0 residuals but got {}", residuals.len()),
            });
        }
        Ok((Self::zero_operation(r#type)?, Vec::new()))
    }
}

// Every operation family that absorbs a type-only `ZeroOperation` has an input-free zero, and so the defaulted
// residual protocol applies verbatim. Composite families without that conversion implement the protocol directly.
impl<T: Type, O: Operation<Type = T> + From<ZeroOperation<T>>> ResidualZeroProvider<T> for O {}

/// Captures the runtime values needed to materialize a zero of `r#type` and validates the operation family's residual
/// protocol. [`ResidualZeroProvider::zero_residual_types`] declares the residual signature, while
/// [`ResidualZeroProvider::capture_zero_residual_values`] performs the operation-family-specific reads from `source`.
/// This helper calls both and verifies that capture returns exactly the declared number and types of values, in the
/// declared order. A disagreement is a malformed provider implementation and is reported as a
/// [`ProgramError::MalformedProgram`] before the residuals can construct a zero with incorrect runtime geometry.
///
/// For example, suppose `r#type` is `zero[n, n, m]` and `source` is the corresponding primal array. A composite array
/// provider declares one dimension residual per distinct dynamic identity, `[dimension(n), dimension(m)]`, and captures
/// the runtime extents `[n, m]` from the first source axis carrying each identity. This helper validates those two
/// captured values. Later, [`ResidualZeroProvider::zero_operation_with_residuals`] expands them into the per-axis
/// operand order `[n, n, m]`. A static zero declares and captures no residuals.
///
/// This function neither chooses which geometry to retain nor constructs the zero; those responsibilities belong to
/// the operation family. It only enforces the declaration/capture contract for concrete or tracer [`Value`]s. The
/// program-level [`AtomId`] capture path performs the corresponding validation where its atoms are staged.
///
/// # Parameters
///
///   - `context`: Context in which the provider reads residual values from `source`.
///   - `source`: Primal value whose runtime geometry determines the zero.
///   - `r#type`: Type of the zero that will eventually consume the captured residuals.
///   - `site`: Description of the capture site included in malformed-provider diagnostics.
pub(crate) fn capture_and_validate_zero_residual_values<C: Context<Operation: ResidualZeroProvider<C::Type>>>(
    context: &C,
    source: &C::Value,
    r#type: &C::Type,
    site: &str,
) -> Result<Vec<C::Value>, ProgramError> {
    let expected_types = C::Operation::zero_residual_types(r#type);
    let residuals = C::Operation::capture_zero_residual_values(context, source, r#type)?;
    if residuals.len() != expected_types.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "{} captured {} zero residuals but declared {}",
            site,
            residuals.len(),
            expected_types.len(),
        )));
    }
    for (index, (residual, expected_type)) in residuals.iter().zip(expected_types).enumerate() {
        if residual.r#type().as_ref() != &expected_type {
            return Err(ProgramError::MalformedProgram(format!(
                "{} zero residual {} has type {} but expected {}",
                site,
                index,
                residual.r#type(),
                expected_type,
            )));
        }
    }
    Ok(residuals)
}

/// Role of differential boundary whose [`ZeroSpaceBoundaryLeaf`]s are reconstructed
/// from [`ZeroSpaceBoundaryReconstruction`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ZeroSpaceBoundaryRole {
    /// Cotangents of the primal input boundary, returned by a _pullback_ function.
    InputCotangent,

    /// Tangents of the primal output boundary, returned by a _pushforward_ function.
    OutputTangent,
}

impl ZeroSpaceBoundaryRole {
    /// Returns the differential [`Type`] represented by this [`ZeroSpaceBoundaryRole`] for `primal_type`.
    #[inline]
    fn differential_type<T: DifferentiableType>(self, primal_type: &T) -> T {
        match self {
            Self::InputCotangent => primal_type.cotangent(),
            Self::OutputTangent => primal_type.tangent(),
        }
    }

    /// Returns the concise description of this [`ZeroSpaceBoundaryRole`] to be used in diagnostics.
    #[inline]
    const fn as_str(self) -> &'static str {
        match self {
            Self::InputCotangent => "input cotangent boundary",
            Self::OutputTangent => "output tangent boundary",
        }
    }
}

impl Display for ZeroSpaceBoundaryRole {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Reconstruction metadata for one zero-space leaf omitted from a compact derivative boundary.
/// This is only used as part of [`ZeroSpaceBoundaryReconstruction`].
pub(crate) struct ZeroSpaceBoundaryLeaf<T: Type> {
    /// Position of the omitted zero in the complete public differential boundary.
    index: usize,

    /// Tangent or cotangent type of the omitted zero.
    r#type: T,

    /// Range of captured residuals consumed to materialize the omitted zero.
    residual_range: Range<usize>,
}

/// Runtime-geometry residuals used to reconstruct the zero-space leaves omitted from one compact derivative boundary.
///
/// Let a flattened primal boundary have leaf types `T₁, …, Tₙ`, and let `D(Tᵢ)` denote the corresponding tangent or
/// cotangent type. When `D(Tᵢ)` is a _zero space_, it contains exactly one value, `0ᵢ`, so the executable derivative
/// [`Program`](crate::Program) omits that leaf entirely. That is because carrying a Single Static Assignment (SSA)
/// input or output for a value that cannot vary would add IR and ABI overhead without conveying information. A public
/// [`Pushforward`](crate::Pushforward) or [`Pullback`](crate::Pullback) must still reconstruct `0ᵢ` when it rebuilds
/// the complete user-facing boundary.
///
/// Static zeros can be constructed from their types alone. Dynamic zeros cannot. For example, the cotangent of a primal
/// `u64[n]` array has type `zero[n]` which records the identity and bounds of `n`, but not its runtime extent. While
/// the primal value is available, linearization therefore captures the minimal runtime geometry declared by
/// [`ResidualZeroProvider::zero_residual_types`] (e.g., the concrete value of `n`). This type stores the flattened
/// concatenation of those captured values in primal-leaf order together with a sparse reconstruction plan for the
/// zero-space leaves. [`Self::rebuild`] replays that plan to materialize each omitted `0ᵢ` and interleave it with the
/// live derivative values produced by the compact program.
///
/// These are **boundary reconstruction residuals**, not ordinary executable-program residuals. They are retained beside
/// the reusable derivative callable and are consumed only while restoring its public boundary. They never become
/// otherwise-unused inputs of the derivative program. The wrapper retains the boundary size, the position and type of
/// every omitted zero, and the range of residuals that reconstructs it. The tangent/cotangent mapping and primal
/// boundary are therefore consumed exactly once during capture and cannot be changed later during reconstruction.
pub struct ZeroSpaceBoundaryReconstruction<V: Value> {
    /// Semantic role of the differential boundary reconstructed by this instance.
    role: ZeroSpaceBoundaryRole,

    /// Flattened runtime-geometry residuals captured in zero-leaf and provider-declaration order.
    residuals: Vec<V>,

    /// Number of leaves in the complete public differential boundary.
    boundary_size: usize,

    /// Sparse reconstruction plan containing one entry for every zero-space boundary leaf, in boundary order.
    zero_leaves: Vec<ZeroSpaceBoundaryLeaf<V::Type>>,
}

impl<V: Value<Type: DifferentiableType>> ZeroSpaceBoundaryReconstruction<V> {
    /// Captures the runtime geometry and reconstruction plan for every zero-space leaf of one differential boundary.
    ///
    /// For each primal leaf type `Tᵢ`, `role` determines the boundary type `D(Tᵢ)`. A nonzero-space `D(Tᵢ)` remains a
    /// live input or output of the compact derivative program and needs no entry in the stored plan. For a zero-space
    /// `D(Tᵢ)`, this function captures and validates the runtime values declared by
    /// [`ResidualZeroProvider::zero_residual_types`] from the corresponding primal value, then records the leaf index,
    /// differential type, and captured residual range. Reconstruction therefore never needs the primal boundary or
    /// differential mapping again.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which the operation family reads runtime geometry from the primal values.
    ///   - `primal_values`: Complete flattened primal boundary values in leaf order.
    ///   - `primal_types`: Complete flattened primal boundary types in the same order as `primal_values`.
    ///   - `role`: Semantic boundary role that selects the tangent/cotangent mapping and identifies diagnostics.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] if the primal value/type counts differ or if an operation family
    /// captures residual values whose count or types disagree with its declaration.
    pub fn capture<C: Context<Value = V, Type = V::Type, Operation: ResidualZeroProvider<C::Type>>>(
        context: &C,
        primal_values: &[C::Value],
        primal_types: &[C::Type],
        role: ZeroSpaceBoundaryRole,
    ) -> Result<Self, ProgramError> {
        if primal_values.len() != primal_types.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "{} has {} primal values but {} primal types",
                role,
                primal_values.len(),
                primal_types.len(),
            )));
        }
        let mut residuals = Vec::new();
        let mut zero_leaves = Vec::new();
        for (index, (value, primal_type)) in primal_values.iter().zip(primal_types).enumerate() {
            let differential_type = role.differential_type(primal_type);
            if differential_type.is_zero_space() {
                let residual_start = residuals.len();
                residuals.extend(capture_and_validate_zero_residual_values(
                    context,
                    value,
                    &differential_type,
                    role.as_str(),
                )?);
                zero_leaves.push(ZeroSpaceBoundaryLeaf {
                    index,
                    r#type: differential_type,
                    residual_range: residual_start..residuals.len(),
                });
            }
        }
        Ok(Self { role, residuals, boundary_size: primal_types.len(), zero_leaves })
    }

    /// Rebuilds the complete differential boundary described by this instance, interleaving materialized zero-space
    /// leaves with the compact derivative program's `live_values`. The boundary size, zero-leaf positions, differential
    /// types, and residual partitions were fixed and validated by [`Self::capture`]. Reconstruction consequently needs
    /// only the context in which to bind each zero and the live values produced by the compact derivative program.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which residual-backed zero operations are bound.
    ///   - `live_values`: Differential values for every nonzero-space boundary leaf, in boundary order.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] if a zero cannot be materialized or if `live_values` does not contain exactly one
    /// value for every nonzero-space boundary leaf.
    pub fn rebuild<
        C: Context<Value = V, Type = V::Type, Operation: ResidualZeroProvider<C::Type>>,
        I: IntoIterator<Item = C::Value>,
    >(
        &self,
        context: &C,
        live_values: I,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut live_values = live_values.into_iter();
        let mut zero_leaves = self.zero_leaves.iter().peekable();
        let mut values = Vec::with_capacity(self.boundary_size);
        for index in 0..self.boundary_size {
            if zero_leaves.peek().is_some_and(|leaf| leaf.index == index) {
                let zero_leaf = zero_leaves.next().unwrap();
                let residuals = self.residuals.get(zero_leaf.residual_range.clone()).unwrap();
                let (operation, operands) =
                    C::Operation::zero_operation_with_residuals(zero_leaf.r#type.clone(), residuals)?;
                let mut outputs = context.bind(operation, Vec::new(), operands.as_slice())?;
                check_count!("output", outputs, 1, ProgramError);
                values.push(outputs.remove(0));
            } else {
                values.push(live_values.next().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!("{} omitted a nonzero differential value", self.role))
                })?);
            }
        }
        if live_values.next().is_some() {
            return Err(ProgramError::MalformedProgram(format!(
                "{} produced too many nonzero differential values",
                self.role,
            )));
        }
        Ok(values)
    }
}

/// Interface form implemented by a [`LinearCallOperation`].
#[derive(Clone, Debug, PartialEq)]
enum LinearCallInterface<T: DifferentiableType> {
    /// Executable linear map. Both the map and its transpose have attached regions (i.e., `forward` followed by
    /// `transpose`), and the complete operation interface is derived from the `forward` [`Region`](crate::Region)'s
    /// boundary, so no [`Type`](crate::Type)s are stored.
    ForwardAndTranspose,

    /// Reverse-only linear map that supplies a transpose program but no executable forward program, so exactly one
    /// region is attached, named `transpose`. The forward map `u ↦ Lᵣ(u)` therefore exists mathematically but has
    /// no region boundary to derive the operation interface from, and its input and output types are stored here
    /// explicitly. Interpreting this form is deliberately an error (i.e., the canonical reverse-only diagnostic).
    /// The call exists in a linearized program only so that reverse mode can transpose it by replaying the attached
    /// [`Region`](crate::Region). For example, [`CustomVjpOperation`](crate::CustomVjpOperation) stages this form
    /// because `custom_vjp` supplies a user-written backward program without a tangent program. Refer to the
    /// documentation of [`LinearCallOperation`] for how this form relates to [`Self::ForwardAndTranspose`].
    TransposeOnly {
        /// Input [`Type`](crate::Type)s of the unavailable forward map (one per linear operand).
        input_types: Vec<T>,

        /// Output [`Type`](crate::Type)s of the unavailable forward map.
        output_types: Vec<T>,
    },
}

/// [`Operation`] that represents a call to a residual-parameterized linear map together with its transpose. For
/// fixed residual values `r`, this operation represents a linear map `Lᵣ : U → V` and its transpose `Lᵣᵀ : V* → U*`.
/// Linearity applies only to `u ∈ U`: for scalars `a` and `b`, `Lᵣ(a · u₁ + b · u₂) = a · Lᵣ(u₁) + b · Lᵣ(u₂)`. The
/// residuals `r` are fixed primal values that parameterize the map, so transposition produces cotangents for `u` but
/// neither differentiates nor accumulates cotangents for `r`. Its two region interfaces are deliberately symmetric:
///
///   - `forward`: `(r, u) ↦ v = Lᵣ(u)`, and
///   - `transpose`: `(r, v̄) ↦ ū = Lᵣᵀ(v̄)`.
///
/// Operation operands are ordered as `[residuals..., linear_inputs...]`, matching both region interfaces, and
/// [`residual_count`](Self::residual_count) separates the two operand roles. That symmetry makes transposition a
/// *swap*: the transpose of `Lᵣ` staged with regions `(forward, transpose)` is the same operation staged with the
/// regions `(transpose, forward)` over `[residuals..., output_cotangents...]`, so transposing twice restores the
/// original call and a pullback program retains the linear-call boundary (and its explicit residual edges) instead
/// of dissolving it. This mirrors the transpose rule of [JAX's
/// `jax.custom_derivatives.linear_call`](https://github.com/jax-ml/jax/blob/main/jax/_src/custom_derivatives.py),
/// which has no rendered documentation page and is therefore linked at its source. The swap re-derives each side's
/// expected interface with the cotangent type mapping, which requires `cotangent(cotangent(u)) = u` for the linear
/// types. Tangent types (the only types linearization rules stage as linear operands) satisfy this even where primal
/// storage types do not (e.g., `f8e8m0fnu`, whose tangent and cotangent representations are both `f32`).
///
/// Every residual is an ordinary typed Single Static Assignment (SSA) edge rather than differentiation-only payload
/// metadata. Partial evaluation can lift those values into the enclosing [`Linearization`](crate::Linearization)
/// residual environment, and partition-aware transposition receives the same values as known operands in
/// deterministic order.
///
/// An explicit operation boundary is necessary because a tangent program must retain more than the computation of
/// `Lᵣ(u)`. After linearization, that program may be cloned, imported, simplified, differentiated again, or transposed
/// independently of the primal program. Merely inlining the forward computation would lose its association with both
/// the residual values and the program that implements `Lᵣᵀ`. Attaching both regions to this operation makes that
/// association survive through the ordinary [`Program`](crate::Program) region, operand, identity-renaming, and import
/// machinery, without an ambient value lookup or side table.
///
/// The dynamic reshape operation illustrates the need for this representation. Its tangent map reshapes an input
/// tangent using the output extents, whereas its transpose reshapes an output cotangent using the original input
/// extents. Those input extents cannot always be recovered from the output shape (e.g., `[n, 4] → [2, 2·n]`). The
/// reshape rule therefore carries both sets of extents as explicit residual operands and attaches forward and inverse
/// reshape regions to one [`LinearCallOperation`]. Other shape-dependent linear rules can use the same mechanism; the
/// operation itself has no array- or dimension-specific semantics.
///
/// The _forward-and-transpose_ form of this operation derives its complete interface from attached `forward` and
/// `transpose` [`Region`](crate::Region)s and can be interpreted, lowered, transposed, and differentiated again. The
/// _transpose-only_ form is the deliberate exception for linear maps that supply only a reverse rule: it stores the
/// unavailable forward input/output types and attaches only the transpose region, so attempting to execute or
/// differentiate it in forward mode is an error (e.g., [`CustomVjpOperation`](crate::CustomVjpOperation) stages
/// this form because it supplies a backward program without a tangent program).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearCallOperation<T: DifferentiableType> {
    /// Number of leading residual operands.
    residual_count: usize,

    /// Specifies whether this call has an executable forward program or is a reverse-only transpose-only map.
    interface: LinearCallInterface<T>,
}

impl<T: DifferentiableType> LinearCallOperation<T> {
    /// Creates a _forward-and-transpose_ [`LinearCallOperation`] with two attached regions,
    /// `forward` and `transpose`.
    #[inline]
    pub fn new(residual_count: usize) -> Self {
        Self { residual_count, interface: LinearCallInterface::ForwardAndTranspose }
    }

    /// Creates a reverse-only _transpose-only_ [`LinearCallOperation`] for a linear map that supplies a transpose
    /// program but no executable forward program, stating the unavailable forward map's interface explicitly.
    #[inline]
    pub fn transpose_only(residual_count: usize, input_types: Vec<T>, output_types: Vec<T>) -> Self {
        Self { residual_count, interface: LinearCallInterface::TransposeOnly { input_types, output_types } }
    }

    /// Returns the number of leading residual operands.
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Returns `true` if this call has the _transpose-only_ form, which attaches no executable forward
    /// [`Region`](crate::Region).
    #[inline]
    pub fn is_transpose_only(&self) -> bool {
        matches!(self.interface, LinearCallInterface::TransposeOnly { .. })
    }

    /// Splits the provided `input_types` into its leading residual types and trailing linear types.
    fn split_inputs<'a>(&self, input_types: &'a [T]) -> Result<(&'a [T], &'a [T]), TypeError> {
        if self.residual_count > input_types.len() {
            return Err(TypeError::invalid(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                input_types.len(),
            )));
        }
        Ok(input_types.split_at(self.residual_count))
    }

    /// Stages a [`LinearCallInterface::ForwardAndTranspose`] residual-parameterized [`LinearCallOperation`].
    /// It traces its two attached [`Region`](crate::Region)s and binds the resulting _forward-and-transpose_
    /// [`LinearCallOperation`] in `context`. This is a canonical constructor rather than a separate binding path as the
    /// final step is still an ordinary [`Context::bind`], and everything before it only constructs the operation's two
    /// region programs. It exists because every producer of an [`LinearCallInterface::ForwardAndTranspose`] linear call
    /// must uphold the same boundary convention, which is easy to get subtly wrong at any one of many call sites:
    ///
    ///   - the operands are ordered as `[residuals..., linear_inputs...]`,
    ///   - the `forward` region receives the same values in the same order,
    ///   - the `transpose` region receives the same residuals followed by one cotangent-typed input
    ///     per traced forward output, and
    ///   - the operation carries its regions in `[forward, transpose]` order, with
    ///     [`residual_count`](Self::residual_count) separating the two operand roles.
    ///
    /// Callers therefore provide only the operation-specific mathematics of the two maps, as closures over tracers that
    /// are already split into their residual and linear/cotangent groups.
    ///
    /// The _transpose-only_ form (i.e., [`LinearCallInterface::TransposeOnly`]) deliberately has no staging counterpart
    /// (e.g., an `Option`al `transpose_fn` or a `stage_transpose_only` sibling) as it shares _none_ of this function's
    /// mechanics, because there is no forward map to trace and therefore no traced outputs to derive the transpose
    /// boundary from (its interface types are stated explicitly instead).
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] in which the linear call is staged.
    ///   - `residuals`: Fixed primal values parameterizing both regions.
    ///   - `linear_inputs`: Tangent values to which the forward linear map is applied.
    ///   - `forward`: Function that builds the forward map `v = Lᵣ(u)` from `(residuals, linear_inputs)`.
    ///   - `transpose`: Function that builds the transpose map `ū = Lᵣᵀ(v̄)` from `(residuals, output_cotangents)`.
    pub(crate) fn stage<
        C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
        ForwardFn: FnOnce(
            &[Tracer<NestedTracingContext<C>>],
            &[Tracer<NestedTracingContext<C>>],
        ) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
        TransposeFn: FnOnce(
            &[Tracer<NestedTracingContext<C>>],
            &[Tracer<NestedTracingContext<C>>],
        ) -> Result<Vec<Tracer<NestedTracingContext<C>>>, ProgramError>,
    >(
        context: &C,
        residuals: Vec<C::Value>,
        linear_inputs: Vec<C::Value>,
        forward_fn: ForwardFn,
        transpose_fn: TransposeFn,
    ) -> Result<Vec<C::Value>, ProgramError> {
        let residual_count = residuals.len();
        let forward_input_types =
            residuals.iter().chain(&linear_inputs).map(|value| value.r#type().into_owned()).collect();
        let (_, forward) = NestedTracingContext::trace(
            context.clone(),
            move |inputs| {
                let (residuals, linear_inputs) = inputs.split_at(residual_count);
                forward_fn(residuals, linear_inputs)
            },
            forward_input_types,
        )?;
        let transpose_input_types = residuals
            .iter()
            .map(|value| value.r#type().into_owned())
            .chain(forward.outputs().map(|output| output.r#type().into_owned().cotangent()))
            .collect();
        let (_, transpose) = NestedTracingContext::trace(
            context.clone(),
            move |inputs| {
                let (residuals, output_cotangents) = inputs.split_at(residual_count);
                transpose_fn(residuals, output_cotangents)
            },
            transpose_input_types,
        )?;
        let mut inputs = residuals;
        inputs.extend(linear_inputs);
        context.bind(Self::new(residual_count), vec![forward, transpose], inputs.as_slice())
    }

    /// Batches an invocation to this [`LinearCallOperation`] by structurally batching its two attached
    /// [`Region`](crate::Region)s. Both [`LinearCallInterface`] forms preserve a completely replicated call unchanged.
    ///
    /// A mapped [`LinearCallInterface::ForwardAndTranspose`] call batches its forward region to discover the batched
    /// output axes, batches its transpose region under those axes, and aligns the transpose outputs with the original
    /// linear-input axes (i.e., a cotangent for a replicated linear input is summed across the batch by `collapse_fn`,
    /// while cotangents for mapped linear inputs retain their packed axes).
    ///
    /// A mapped [`LinearCallInterface::TransposeOnly`] call is rejected because its unavailable forward program cannot
    /// determine the batched output axes.
    ///
    /// The batching policy owns the boundary shape of its structurally batched programs.
    /// [`BatchingPolicy::adapt_batched_program`] adapts each batched region to the plain two-region linear-call
    /// boundary, and any [`BatchingPolicy::boundary_operands`] (e.g., a composite program's first-class mapped extent)
    /// become additional leading residuals of the batched call.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`BatchingContext`] for the transform level being applied.
    ///   - `driver`: [`BatchingDriver`] exposing this call's attached regions.
    ///   - `inputs`: Batched operands, ordered as `[residuals..., linear_inputs...]`.
    ///   - `input_axes`: Batch axis of each operand in `inputs`.
    pub(crate) fn batch_regions<
        C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
        P: LinearCallBatchingPolicy<C>,
        D: BatchingDriver<C, P>,
    >(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
        input_axes: Vec<BatchAxis>,
    ) -> Result<Vec<P::Batch>, BatchingError> {
        if self.residual_count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
            .into());
        }
        check_count!("input", input_axes, inputs.len(), ProgramError);
        let input_values = inputs.iter().map(P::value).cloned().collect::<Vec<_>>();

        // A completely replicated call needs no structural region rewrite. Keeping the original call also avoids
        // manufacturing a batch axis that neither region observes.
        if input_axes.iter().all(BatchAxis::is_replicated) {
            let outputs = context.parent().bind(
                self.clone(),
                driver.regions().map(|region| region.to_program()).collect::<Vec<_>>(),
                input_values.as_slice(),
            )?;
            return Ok(outputs.into_iter().map(P::replicated).collect());
        }

        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {}
            LinearCallInterface::TransposeOnly { .. } => {
                return Err(BatchingError::UnsupportedOperation {
                    message: "a transpose-only linear call cannot be batched with mapped inputs because its \
                             unavailable forward program does not determine output batch axes"
                        .to_string(),
                });
            }
        }

        let (residual_axes, linear_axes) = input_axes.split_at(self.residual_count);
        let (forward, output_axes) = P::adapt_batched_program(
            driver.batch_program(
                context,
                driver.region(0)?,
                input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::Natural,
            )?,
            None,
            P::sum_mapped_cotangents,
        )?
        .into_parts();

        // Batch the supplied transpose under the forward result axes. Cotangents for replicated linear inputs are
        // summed during adaptation; mapped linear inputs instead retain their packed axes.
        let transpose_input_axes = residual_axes.iter().copied().chain(output_axes.iter().copied()).collect::<Vec<_>>();
        let (transpose, transpose_output_axes) = P::adapt_batched_program(
            driver.batch_program(
                context,
                driver.region(1)?,
                transpose_input_axes.as_slice(),
                ProgramBatchingOutputAxesPolicy::AlignEachTo(linear_axes.to_vec()),
            )?,
            Some(linear_axes),
            P::sum_mapped_cotangents,
        )?
        .into_parts();
        check_count!("output", transpose_output_axes, linear_axes.len(), ProgramError);
        check_count!("output", transpose.output_ids(), linear_axes.len(), ProgramError);

        let boundary_operands = P::boundary_operands(context.axis_extent());
        let mut packed_inputs = Vec::with_capacity(boundary_operands.len() + input_values.len());
        let boundary_operand_count = boundary_operands.len();
        packed_inputs.extend(boundary_operands);
        packed_inputs.extend(input_values);
        context
            .parent()
            .bind(
                LinearCallOperation::new(self.residual_count + boundary_operand_count),
                vec![forward, transpose],
                packed_inputs.as_slice(),
            )?
            .into_iter()
            .zip(output_axes)
            .map(|(output, axis)| P::batch(output, axis))
            .collect()
    }
}

impl<T: DifferentiableType> Display for LinearCallOperation<T> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.render(formatter, 0)
    }
}

impl<T: DifferentiableType> Operation for LinearCallOperation<T> {
    type Type = T;

    #[inline]
    fn name(&self) -> &'static str {
        // The two forms render under distinct names so rendered programs and the diagnostics built from this name
        // distinguish a reverse-only call from an executable one without inspecting region counts.
        if self.is_transpose_only() { "transpose_only_linear_call" } else { "linear_call" }
    }

    #[inline]
    fn region_slots(&self) -> &'static [RegionSlot] {
        if self.is_transpose_only() {
            const { &[RegionSlot::rule("transpose")] }
        } else {
            const { &[RegionSlot::computation("forward"), RegionSlot::rule("transpose")] }
        }
    }

    fn infer_region_input_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<Option<Vec<T>>>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let renaming = T::derive_identity_renaming(forward.input_types(), input_types)?;
                let output_types = forward
                    .output_types()
                    .iter()
                    .map(|r#type| r#type.rename_identities(&renaming))
                    .collect::<Result<Vec<_>, _>>()?;
                let (residual_types, _) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect();
                Ok(vec![Some(input_types.to_vec()), Some(transpose_input_types)])
            }
            LinearCallInterface::TransposeOnly { output_types, .. } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let (residual_types, _) = self.split_inputs(input_types)?;
                Ok(vec![Some(
                    residual_types
                        .iter()
                        .cloned()
                        .chain(output_types.iter().map(DifferentiableType::cotangent))
                        .collect(),
                )])
            }
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[T],
        region_interfaces: &[RegionInterface<T>],
    ) -> Result<Vec<T>, TypeError> {
        match &self.interface {
            LinearCallInterface::ForwardAndTranspose => {
                check_count!("region", region_interfaces, 2, TypeError);
                let forward = &region_interfaces[0];
                let transpose = &region_interfaces[1];
                check_types!(@same, "linear call forward input", [input_types, forward.input_types()]);
                let (residual_types, linear_types) = self.split_inputs(input_types)?;
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(forward.output_types().iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(forward.output_types().to_vec())
            }
            LinearCallInterface::TransposeOnly { input_types: linear_types, output_types } => {
                check_count!("region", region_interfaces, 1, TypeError);
                let transpose = &region_interfaces[0];
                let (residual_types, actual_linear_types) = self.split_inputs(input_types)?;
                check_types!(@same, "transpose-only linear call input", [linear_types, actual_linear_types]);
                let transpose_input_types = residual_types
                    .iter()
                    .cloned()
                    .chain(output_types.iter().map(DifferentiableType::cotangent))
                    .collect::<Vec<_>>();
                let transpose_output_types = linear_types.iter().map(DifferentiableType::cotangent).collect::<Vec<_>>();
                check_types!(@same, "transpose-only linear call transpose input", [
                    &transpose_input_types,
                    transpose.input_types(),
                ]);
                check_types!(@same, "transpose-only linear call transpose output", [
                    &transpose_output_types,
                    transpose.output_types(),
                ]);
                Ok(output_types.clone())
            }
        }
    }

    #[inline]
    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        // The forward-and-transpose form's outputs are exactly its forward region's outputs.
        // The transpose-only form has no forward region and therefore no region provenance.
        if self.is_transpose_only() {
            Vec::new()
        } else {
            vec![OutputRegionProvenance { region_index: 0, output_index }]
        }
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("residual_count", self.residual_count))
    }

    fn rename_type_identities(&self, renaming: &TypeIdentityRenaming<T::Identity>) -> Result<Self, TypeError> {
        Ok(Self {
            residual_count: self.residual_count,
            interface: match &self.interface {
                LinearCallInterface::ForwardAndTranspose => LinearCallInterface::ForwardAndTranspose,
                LinearCallInterface::TransposeOnly { input_types, output_types } => {
                    LinearCallInterface::TransposeOnly {
                        input_types: input_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                        output_types: output_types
                            .iter()
                            .map(|r#type| r#type.rename_identities(renaming))
                            .collect::<Result<Vec<_>, _>>()?,
                    }
                }
            },
        })
    }
}

impl<C: Domain<Type: DifferentiableType>> InterpretableOperation<C> for LinearCallOperation<C::Type> {
    #[inline]
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward program to execute; it supports only \
                          reverse-mode differentiation (e.g., 'vjp', 'value_and_gradient', or 'jacobian_reverse')"
                    .to_string(),
            });
        }
        driver.interpret_region(context, 0, inputs.to_vec())
    }
}

impl<C: Context<Type: DifferentiableType, Operation: From<LinearCallOperation<C::Type>>>>
    PartiallyEvaluatableOperation<C> for LinearCallOperation<C::Type>
{
}

impl<
    T: DifferentiableType,
    C: Context<Type = T, Operation: From<LinearCallOperation<T>>>,
    P: LinearCallBatchingPolicy<C>,
> BatchableOperation<C, P> for LinearCallOperation<T>
{
    fn batch<D: BatchingDriver<C, P>>(
        &self,
        context: &BatchingContext<C, P>,
        driver: &D,
        inputs: &[P::Batch],
    ) -> Result<Vec<P::Batch>, BatchingError> {
        let input_axes = inputs.iter().map(P::batch_axis).collect::<Vec<_>>();
        self.batch_regions(context, driver, inputs, input_axes)
    }
}

impl<C: Context<Type: DifferentiableType> + Zero<C::Value>> DifferentiableOperation<C>
    for LinearCallOperation<C::Type>
{
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if self.is_transpose_only() {
            return Err(ProgramError::UnsupportedOperation {
                message: "a transpose-only linear call has no forward-mode (JVP) rule; it supports only \
                          reverse-mode differentiation"
                    .to_string(),
            }
            .into());
        }

        inputs.len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
        })?;

        let forward = driver.region(0)?;
        let primals = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        if inputs.iter().all(|input| input.tangent().is_zero()) {
            let primal_outputs = forward.interpret_in_context(context, primals)?;
            return Ok(primal_outputs.into_iter().map(DifferentiationDual::new_with_zero_tangent).collect());
        }

        // Higher-order differentiation must include the dependence of the linear map on its residual parameters.
        // Replay the ordinary fused JVP of the attached forward region instead of assuming residual tangents are zero.
        let jvp = driver.jvp_program(forward)?;
        let mut jvp_inputs = primals;
        for input in inputs {
            if !input.tangent().r#type().is_zero_space() {
                jvp_inputs.push(input.tangent().clone().materialize(context)?);
            }
        }

        let mut outputs = jvp.interpret_in_context(context, jvp_inputs)?;
        let output_types = forward.output_types();
        let tangent_outputs = outputs.split_off(output_types.len());
        let mut tangent_outputs = tangent_outputs.into_iter();
        outputs
            .into_iter()
            .zip(output_types)
            .map(|(primal, output_type)| {
                if output_type.tangent().is_zero_space() {
                    Ok(DifferentiationDual::new_with_zero_tangent(primal))
                } else {
                    DifferentiationDual::new(primal, tangent_outputs.next().unwrap())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }
}

impl<
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type> + ZeroOperationProvider<V::Type> + From<LinearCallOperation<V::Type>>,
> TransposableOperation<V, O> for LinearCallOperation<V::Type>
{
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TracingContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        if self.residual_count > inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linear call residual count {} exceeds input count {}",
                self.residual_count,
                inputs.len(),
            ))
            .into());
        }
        let (residual_inputs, linear_inputs) = inputs.split_at(self.residual_count);
        let residuals = residual_inputs
            .iter()
            .enumerate()
            .map(|(index, input)| {
                input.as_known().cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "linear call residual operand {index} is not known during transposition",
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        if outputs.iter().all(MaybeZero::is_zero) {
            return Ok(inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect());
        }

        // Classify each transpose-region output as structurally zero by inspecting its producing instruction in the
        // source region, so zero cotangents stay symbolic instead of accumulating. Outputs with no producing
        // instruction (forwarded region inputs and constants) conservatively classify as nonzero. The transpose
        // region follows the forward-and-transpose form's leading forward region (index 1) and is the transpose-only
        // form's only region (index 0).
        let transpose = driver.region(if self.is_transpose_only() { 0 } else { 1 })?;
        let output_is_zero = transpose
            .output_ids()
            .iter()
            .map(|output| {
                transpose
                    .instructions()
                    .iter()
                    .find_map(|instruction| {
                        instruction
                            .outputs()
                            .iter()
                            .position(|candidate| candidate == output)
                            .map(|output_index| instruction.operation().is_zero(output_index))
                    })
                    .unwrap_or(false)
            })
            .collect::<Vec<_>>();

        let mut transpose_inputs = residuals;
        transpose_inputs
            .extend(outputs.iter().cloned().map(|output| output.materialize(context)).collect::<Result<Vec<_>, _>>()?);
        let input_cotangents = if self.is_transpose_only() {
            // The transpose-only form's region is a user-supplied backward program with no linearity contract of its
            // own, so it cannot be re-transposed and is replayed inline into the pullback.
            transpose.interpret_in_context(context, transpose_inputs)?
        } else {
            // The forward-and-transpose form transposes by *swapping* its regions: the pullback stages the same
            // operation with the transpose region leading, over the same residuals followed by the output
            // cotangents. Transposing twice therefore restores the original call, and the pullback retains the
            // linear-call boundary together with its explicit residual edges.
            let swapped = LinearCallOperation::new(self.residual_count);
            let transpose_program = transpose.to_program();
            let forward_program = driver.region(0)?.to_program();
            context.bind(swapped, vec![transpose_program, forward_program], &transpose_inputs)?
        };
        check_count!("output", input_cotangents, linear_inputs.len(), ProgramError);

        let mut cotangents =
            residual_inputs.iter().map(|input| MaybeZero::Zero(input.r#type().cotangent())).collect::<Vec<_>>();
        cotangents.extend(linear_inputs.iter().zip(input_cotangents.into_iter().zip(output_is_zero)).map(
            |(input, (cotangent, is_zero))| {
                if input.is_unknown() {
                    if is_zero { MaybeZero::Zero(cotangent.r#type().into_owned()) } else { MaybeZero::Value(cotangent) }
                } else {
                    MaybeZero::Zero(input.r#type().cotangent())
                }
            },
        ));
        Ok(cotangents)
    }
}

/// [`BatchingPolicy`] that collapses a mapped cotangent when batching a [`LinearCallOperation`].
///
/// Batching a residual-parameterized linear map may replicate one of its linear inputs across the mapped axis. If the
/// batched transpose subsequently produces one cotangent `ūᵢ` for each batch item, the transpose of that replication
/// is summation, so the single cotangent for the original replicated input is:
///
/// ```text
/// ū = Σᵢ ūᵢ.
/// ```
///
/// [`LinearCallOperation::batch_regions`] owns every universe-independent part of this transformation: structurally
/// batching the forward and transpose regions, aligning their boundaries, threading policy-owned bookkeeping values,
/// and rebuilding the linear call. The representation of `ūᵢ` is the one step it cannot determine generically. An
/// ordinary array policy reduces the cotangent directly along its mapped axis, while a composite policy may first need
/// to project the cotangent to its differentiable member, perform that member's reduction, and lift the result back.
/// This capability supplies exactly that representation-dependent step and lets one generic [`BatchableOperation`]
/// implementation retain the complete linear-call algorithm.
///
/// Implement this trait for a [`BatchingPolicy`] only when its program universe supports batching executable linear
/// calls. An implementation must return a value owned by `context`, of the same program type as `cotangent`, with
/// `axis` removed and all batch-item cotangents combined by addition. Policies that do not support linear calls
/// should omit the implementation. This is deliberately an operation-specific opt-in rather than a method on
/// [`BatchingPolicy`] and ordinary batching policies need not provide differentiation semantics, and other operation
/// families must not acquire parallel policy traits unless they expose an independently irreducible universe-specific
/// step of their own.
///
/// # Parameters
///
///   - `context`: [`TracingContext`] that owns the structurally batched transpose program being adapted.
///   - `cotangent`: Mapped cotangent produced by that transpose program.
///   - `axis`: Physical axis containing the packed family of per-item cotangents.
pub trait LinearCallBatchingPolicy<C: Context<Type: DifferentiableType>>: BatchingPolicy<C> {
    /// Sums the per-item cotangents packed along `axis`.
    fn sum_mapped_cotangents(
        context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>;
}

impl<C: Context<Type = ArrayType, Operation: From<ReduceOperation>>, P: ArrayBatchingPolicy<C>>
    LinearCallBatchingPolicy<C> for ArrayBatching<P>
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        let axis = axis.normalize(cotangent.r#type().rank()).map_err(|_| BatchingError::BatchAxisOutOfBounds {
            r#type: Box::new(cotangent.r#type().into_owned()),
            axis,
        })?;
        Ok(cotangent.reduce(&[axis], ReductionKind::Sum))
    }
}

impl<
    C: Context<
            Type = ArrayProgramType,
            Constant: ValueProjection<ArrayType, Projected: Value<Type = ArrayType>>,
            Operation: OperationProjection<ArrayType, Projected: From<ReduceOperation>>,
        >,
> LinearCallBatchingPolicy<C> for ArrayProgramBatching
{
    fn sum_mapped_cotangents(
        _context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError> {
        // Projecting the replayed array cotangent gives it the ordinary `Reduce` capability,
        // whose staged operation lifts back through the composite operation family.
        let cotangent = ValueProjection::<ArrayType>::into_projected(cotangent)?;
        let axis = axis.normalize(cotangent.r#type().rank()).map_err(|_| BatchingError::BatchAxisOutOfBounds {
            r#type: Box::new(cotangent.r#type().into_owned()),
            axis,
        })?;
        Ok(ValueProjection::from_projected(cotangent.reduce(&[axis], ReductionKind::Sum)))
    }
}

#[cfg(test)]
mod tests {
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::array_programs::{ArrayProgramOperation, ArrayProgramValue};
    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::batching::{BatchAxis, ProgramBatchingOutputAxesPolicy};
    use crate::contexts::tests::{
        ProjectedMemberType, ProjectedMemberValue, ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::DifferentiationError;
    use crate::differentiation::reverse::{TransposableOperation, TranspositionDriver};
    use crate::operations::constants::zero_like::ZeroLikeOperation;
    use crate::operations::math::{AddOperation, MulOperation};
    use crate::parameters::Placeholder;
    use crate::partial::{PartialEvaluationOutput, PartialValue};
    use crate::programs::ProgramError;
    use crate::programs::atoms::MaybeZero;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::programs::Program;
    use crate::programs::regions::{RegionDriver, RegionRef, RegionSlot};
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::tracing::TracingContext;
    use crate::types::{ArrayProgramType, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape};

    use super::*;

    /// Builds the scalar program `(r, u) ↦ r · u` used as a residual-parameterized linear-map region.
    fn scalar_multiply_program() -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let r#type = ArrayType::scalar(DataType::F64);
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![linear, residual]).unwrap()[0];
        builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap()
    }

    /// Test-only transposition driver exposing one attached region, used to invoke the _transpose-only_ form's
    /// transposition rule directly on the transpose region it replays.
    struct TestTranspositionDriver<'r> {
        /// Transpose region exposed by this driver.
        region: RegionRef<'r, Array, ArrayOperation<Array>>,
    }

    impl RegionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
        where
            Array: 'r,
            ArrayOperation<Array>: 'r,
        {
            std::iter::once(self.region)
        }
    }

    impl TranspositionDriver<Array, ArrayOperation<Array>> for TestTranspositionDriver<'_> {
        fn transpose_program(
            &self,
            _region: RegionRef<'_, Array, ArrayOperation<Array>>,
            _input_linearity: &[bool],
        ) -> Result<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>, DifferentiationError> {
            Err(ProgramError::UnsupportedOperation {
                message: "test driver does not transpose nested regions".to_string(),
            }
            .into())
        }
    }

    #[test]
    fn test_residual_zero_provider_input_free_defaults() {
        let r#type = ArrayType::scalar(DataType::F64);

        // An input-free operation family (reached through the blanket implementation) declares no residuals, and
        // both capture hooks record nothing: the builder-level hook stages no instructions and the value-level hook
        // returns no values.
        assert_eq!(ArrayOperation::<Array>::zero_residual_types(&r#type), Vec::<ArrayType>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let source = builder.add_input(r#type.clone());
        assert_eq!(ArrayOperation::<Array>::capture_zero_residuals(&mut builder, source, &r#type), Ok(Vec::new()));
        assert!(builder.instructions().is_empty());
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            ArrayOperation::<Array>::capture_zero_residual_values(&context, &Array::scalar(3.0), &r#type),
            Ok(Vec::new()),
        );

        // Spending no residuals assembles the type-only zero, which the transposition path stages normally.
        let (operation, operands) =
            ArrayOperation::<Array>::zero_operation_with_residuals(r#type.clone(), &[] as &[AtomId]).unwrap();
        let zero = builder.add_instruction(operation, Vec::new(), operands).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(program.interpret(vec![Array::scalar(3.0)]), Ok(vec![Array::scalar(0.0)]));

        // The fail-loud default rejects unexpected residuals instead of ignoring them, so a mismatched
        // linearize/transpose pairing cannot be silently accepted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = builder.add_input(r#type.clone());
        assert_eq!(
            ArrayOperation::<Array>::zero_operation_with_residuals(r#type, &[residual]).map(|_| ()),
            Err(ProgramError::InvalidArgument {
                message: "input-free zero expected 0 residuals but got 1".to_string(),
            }),
        );
        assert!(builder.instructions().is_empty());
    }

    #[test]
    fn test_zero_space_boundary_reconstruction_reconstructs_dynamic_zero() {
        let extent = DimensionVariable::new("extent", DimensionBounds::positive(Some(8)).unwrap());
        let key_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Dynamic(extent)]));
        let accumulator_type = ArrayType::scalar(DataType::F64);
        let context = TracingContext::<ArrayProgramValue<Array>, ArrayProgramOperation<Array>>::new();
        let key = context.input(key_type.clone().into());
        let accumulator = context.input(accumulator_type.clone().into());
        let primal_values = vec![key, accumulator.clone()];
        let primal_types = vec![key_type.clone().into(), accumulator_type.into()];

        // Capture retains the key's dynamic extent and records that only the accumulator tangent remains live in the
        // compact output boundary. Rebuild must recover the omitted key tangent from that stored plan.
        let reconstruction = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            primal_values.as_slice(),
            primal_types.as_slice(),
            ZeroSpaceBoundaryRole::OutputTangent,
        )
        .unwrap();
        let outputs = reconstruction.rebuild(&context, [accumulator.clone()]).unwrap();
        let key_tangent_type = ArrayProgramType::Array(key_type.tangent());
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].r#type().as_ref(), &key_tangent_type);
        assert_eq!(outputs[1].atom_id(), accumulator.atom_id());

        // The captured dimension-size result is the sole operand of the dynamic zero constructor, proving that the
        // stored residual range—not a type-only zero—is used during reconstruction.
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 2);
        assert!(matches!(builder.instructions()[0].operation(), ArrayProgramOperation::DimensionSize(_)));
        assert!(matches!(builder.instructions()[1].operation(), ArrayProgramOperation::Zero(_)));
        assert_eq!(builder.instructions()[1].inputs(), builder.instructions()[0].outputs());
    }

    #[test]
    fn test_zero_space_boundary_reconstruction_reports_stored_boundary() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let primal = Array::scalar(3.0);
        let primal_type = ArrayType::scalar(DataType::F64);

        // The output-tangent role retained during capture identifies a missing compact-program result without a
        // caller-supplied diagnostic context.
        let output_tangent = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            std::slice::from_ref(&primal),
            std::slice::from_ref(&primal_type),
            ZeroSpaceBoundaryRole::OutputTangent,
        )
        .unwrap();
        assert_eq!(
            output_tangent.rebuild(&context, Vec::new()),
            Err(ProgramError::MalformedProgram(
                "output tangent boundary omitted a nonzero differential value".to_string(),
            )),
        );

        // The input-cotangent role independently identifies an excessive compact-program result.
        let input_cotangent = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            std::slice::from_ref(&primal),
            std::slice::from_ref(&primal_type),
            ZeroSpaceBoundaryRole::InputCotangent,
        )
        .unwrap();
        assert_eq!(
            input_cotangent.rebuild(&context, [Array::scalar(1.0), Array::scalar(2.0)]),
            Err(ProgramError::MalformedProgram(
                "input cotangent boundary produced too many nonzero differential values".to_string(),
            )),
        );
    }

    #[test]
    fn test_linear_call_operation() {
        let r#type = ArrayType::scalar(DataType::F64);

        // The forward-and-transpose form derives its interface from two attached regions.
        let operation = LinearCallOperation::<ArrayType>::new(1);
        assert_eq!(operation.residual_count(), 1);
        assert!(!operation.is_transpose_only());
        assert_eq!(operation.to_string(), "linear_call [residual_count=1]");
        assert_eq!(operation.region_slots(), &[RegionSlot::computation("forward"), RegionSlot::rule("transpose")]);
        assert_eq!(
            format!("{operation:?}"),
            "LinearCallOperation { residual_count: 1, interface: ForwardAndTranspose }",
        );

        // The transpose-only form stores the unavailable forward interface and renders under its distinct name.
        let operation = LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]);
        assert_eq!(operation.residual_count(), 1);
        assert!(operation.is_transpose_only());
        assert_eq!(operation.to_string(), "transpose_only_linear_call [residual_count=1]");
        assert_eq!(operation.region_slots(), &[RegionSlot::rule("transpose")]);

        // The transpose-only form derives its transpose region's expected interface from the stored forward interface
        // through the cotangent type mapping, so it infers the stored output types whenever the attached region agrees.
        // The linear types are differential types rather than primal storage types, which the mapping must respect.
        let residual_type = ArrayType::scalar(DataType::F64);
        let tangent_type = ArrayType::scalar(DataType::F32);
        let transpose_interface = RegionInterface::new(
            vec![residual_type.clone(), tangent_type.clone()],
            vec![tangent_type.clone()],
            Effects::PURE,
        );
        assert_eq!(
            LinearCallOperation::transpose_only(1, vec![tangent_type.clone()], vec![tangent_type.clone()])
                .infer_output_types(&[residual_type, tangent_type.clone()], std::slice::from_ref(&transpose_interface)),
            Ok(vec![tangent_type]),
        );
    }

    #[test]
    fn test_linear_call_operation_supports_a_third_composite_member() {
        // A third-member residual paired with a first-member linear value proves that the operation and its attached
        // regions use generic program storage without an array/dimension-specific projection.
        let linear_type = ProjectedProgramType::First(ProjectedMemberType);
        let residual_type = ProjectedProgramType::Third(ProjectedMemberType);
        let forward = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
            builder.add_input(residual_type.clone());
            let linear = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![linear],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let transpose = {
            let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
            builder.add_input(residual_type.clone());
            let cotangent = builder.add_input(linear_type.clone());
            builder
                .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                    vec![cotangent],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ProjectedProgramValue, LinearCallOperation<ProjectedProgramType>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(residual_type);
        let linear = builder.add_input(linear_type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<ProjectedProgramValue>, Vec<ProjectedProgramValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let linear = ProjectedProgramValue::First(ProjectedMemberValue(2));
        let residual = ProjectedProgramValue::Third(ProjectedMemberValue(7));
        assert_eq!(program.interpret(vec![residual, linear.clone()]), Ok(vec![linear]));
        assert_eq!(program.instructions()[0].regions().len(), 2);
    }

    #[test]
    fn test_linear_call_operation_transpose_only_validates_residual_count() {
        let r#type = ArrayType::scalar(DataType::F64);

        // A residual count larger than the operand count is a malformed call rather than a silently truncated split,
        // and inference rejects it before any region interface is consulted.
        let transpose_interface = RegionInterface::new(vec![r#type.clone()], vec![r#type.clone()], Effects::PURE);
        assert!(matches!(
            LinearCallOperation::transpose_only(2, Vec::new(), Vec::new())
                .infer_output_types(&[], std::slice::from_ref(&transpose_interface)),
            Err(TypeError::Invalid { message }) if message == "linear call residual count 2 exceeds input count 0",
        ));

        // Transposition enforces the same split independently, because a pullback may be built from an imported call
        // whose operands were pruned after inference ran.
        let transpose = scalar_multiply_program();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            LinearCallOperation::transpose_only(3, vec![r#type.clone()], vec![r#type.clone()]).transpose(
                &mut context,
                &driver,
                &[],
                &[],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "linear call residual count 3 exceeds input count 0",
        ));

        // Every residual must be known during transposition, because the replayed transpose region consumes the
        // residual values themselves rather than cotangents for them.
        let output_cotangent = context.input(r#type.clone());
        assert!(matches!(
            LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type.clone()]).transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type)],
                &[MaybeZero::Value(output_cotangent)],
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "linear call residual operand 0 is not known during transposition",
        ));
    }

    #[test]
    fn test_linear_call_operation_transpose_only_form_rejects_forward_execution() {
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert!(matches!(
            program.interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "a transpose-only linear call has no forward program to execute; it supports only \
                               reverse-mode differentiation (e.g., 'vjp', 'value_and_gradient', or \
                               'jacobian_reverse')",
        ));
        assert!(matches!(
            program.batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            ),
            Err(BatchingError::UnsupportedOperation { message })
                if message == "a transpose-only linear call cannot be batched with mapped inputs because its \
                               unavailable forward program does not determine output batch axes",
        ));
    }

    #[test]
    fn test_linear_call_operation_transpose_only_remains_opaque_to_partial_evaluation() {
        // A transpose-only carrier (the form `custom_vjp` stages into its tangent program) must survive partial
        // evaluation as one unknown-producing instruction. Splitting or folding it would separate the backward region
        // from the residuals that parameterize it, so the default opaque rule is the correct behavior here.
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                ArrayOperation::LinearCall(LinearCallOperation::transpose_only(
                    1,
                    vec![r#type.clone()],
                    vec![r#type.clone()],
                )),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(r#type.clone()), PartialValue::Unknown(r#type)])
            .unwrap();

        assert!(matches!(evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::LinearCall(_)));
    }

    #[test]
    fn test_linear_call_operation_batching_preserves_its_transpose() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A varying residual with a shared linear input produces mapped outputs. Its transpose must sum the per-item
        // cotangents back into the single shared input cotangent.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::scalar(4.0)]),
            Ok(vec![Array::vector(vec![8.0, 12.0])]),
        );

        let instruction = &batched.instructions()[0];
        let transpose = batched.region_ref(instruction.regions()[1]).unwrap().to_program();
        assert_eq!(
            transpose.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::vector(vec![5.0, 7.0])]),
            Ok(vec![Array::scalar(31.0)]),
        );

        // A replicated residual and mapped linear input preserve the mapped cotangent instead of reducing it.
        let (batched_linear, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            batched_linear.interpret(vec![Array::scalar(2.0), Array::vector(vec![3.0, 4.0])]),
            Ok(vec![Array::vector(vec![6.0, 8.0])]),
        );
        let instruction = &batched_linear.instructions()[0];
        let transpose = batched_linear.region_ref(instruction.regions()[1]).unwrap().to_program();
        assert_eq!(
            transpose.interpret(vec![Array::scalar(2.0), Array::vector(vec![5.0, 7.0])]),
            Ok(vec![Array::vector(vec![10.0, 14.0])]),
        );

        // Rebatching an executable linear call composes mapped axes without losing either attached region.
        let (nested, output_axes) = batched_linear
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::new(0), BatchAxis::new(0)],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::new(0)]);
        assert_eq!(
            nested.interpret(vec![Array::vector(vec![2.0, 3.0]), Array::matrix(2, 2, vec![4.0, 5.0, 6.0, 7.0]),]),
            Ok(vec![Array::matrix(2, 2, vec![8.0, 10.0, 18.0, 21.0])]),
        );
    }

    #[test]
    fn test_linear_call_operation_batching_preserves_a_replicated_transpose_only_call() {
        let r#type = ArrayType::scalar(DataType::F64);
        let transpose = scalar_multiply_program();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type.clone());
        let output = builder
            .add_instruction(
                LinearCallOperation::transpose_only(1, vec![r#type.clone()], vec![r#type]),
                vec![transpose],
                vec![residual, linear],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A completely replicated call needs no structural region rewrite, so batching is the one transform the
        // transpose-only form supports: the call rebinds itself over its untouched backward region.
        let (batched, output_axes) = program
            .batched(
                2,
                ShardingDimension::Replicated,
                &[BatchAxis::replicated(), BatchAxis::replicated()],
                ProgramBatchingOutputAxesPolicy::Natural,
            )
            .unwrap()
            .into_parts();
        assert_eq!(output_axes, vec![BatchAxis::replicated()]);
        let instruction = &batched.instructions()[0];
        assert_eq!(instruction.operation().name(), "transpose_only_linear_call");
        assert_eq!(instruction.regions().len(), 1);
    }

    #[test]
    fn test_linear_call_operation_nested_jvp_differentiates_residual_parameters() {
        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = forward.clone();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let forward = builder.import_region(forward.entry_region_ref());
        let transpose = builder.import_region(transpose.entry_region_ref());
        let residual = builder.add_input(r#type.clone());
        let linear = builder.add_input(r#type);
        let output = builder
            .add_instruction(LinearCallOperation::new(1), vec![forward, transpose], vec![residual, linear])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder, Placeholder], vec![Placeholder])
            .unwrap();

        // For `Lᵣ(u) = r · u`, the nested JVP is `(r, u, ṙ, u̇) ↦ (r · u, ṙ · u + r · u̇)`. At
        // `(r, u, ṙ, u̇) = (2, 3, 5, 7)`, the primal is `6` and the tangent is `5 · 3 + 2 · 7 = 29`.
        assert_eq!(
            program.jvp().unwrap().interpret(vec![
                Array::scalar(2.0),
                Array::scalar(3.0),
                Array::scalar(5.0),
                Array::scalar(7.0),
            ]),
            Ok(vec![Array::scalar(6.0), Array::scalar(29.0)]),
        );
    }

    #[test]
    fn test_linear_call_operation_transposition_swaps_the_attached_regions() {
        /// Exposes an executable linear call's two regions directly to its transposition rule.
        struct TwoRegionDriver<'r> {
            forward: RegionRef<'r, Array, ArrayOperation<Array>>,
            transpose: RegionRef<'r, Array, ArrayOperation<Array>>,
        }

        impl RegionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, Array, ArrayOperation<Array>>>
            where
                Array: 'r,
                ArrayOperation<Array>: 'r,
            {
                [self.forward, self.transpose].into_iter()
            }
        }

        impl TranspositionDriver<Array, ArrayOperation<Array>> for TwoRegionDriver<'_> {
            fn transpose_program(
                &self,
                _region: RegionRef<'_, Array, ArrayOperation<Array>>,
                _input_linearity: &[bool],
            ) -> Result<Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>, DifferentiationError>
            {
                unreachable!("linear call transposition swaps regions and never re-enters transposition")
            }
        }

        let r#type = ArrayType::scalar(DataType::F64);
        let forward = scalar_multiply_program();
        let transpose = {
            // Deliberately use a different body from `forward` so the rendered program proves the region order was
            // swapped instead of merely proving that two indistinguishable regions remain attached.
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let residual = builder.add_input(r#type.clone());
            let output_cotangent = builder.add_input(r#type.clone());
            let input_cotangent =
                builder.add_instruction(AddOperation::new(), Vec::new(), vec![output_cotangent, residual]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![input_cotangent], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };
        let driver = TwoRegionDriver { forward: forward.entry_region_ref(), transpose: transpose.entry_region_ref() };

        // A structural-zero output cotangent returns structural zeros for both the residual and the linear operand
        // without replaying either region or staging a materialized array zero.
        let mut zero_context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = zero_context.input(r#type.clone());
        let zero_cotangents = LinearCallOperation::new(1)
            .transpose(
                &mut zero_context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type.clone())],
                &[MaybeZero::Zero(r#type.clone())],
            )
            .unwrap();
        assert_eq!(zero_cotangents.len(), 2);
        assert!(zero_cotangents.iter().all(MaybeZero::is_zero));
        assert!(zero_context.builder().borrow().instructions().is_empty());

        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());

        // Transposition must assign a structural zero to the known residual and stage one swapped linear call for the
        // unknown linear input's cotangent.
        let cotangents = LinearCallOperation::new(1)
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type)],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert_eq!(cotangents.len(), 2);
        assert!(cotangents[0].is_zero());
        let output = match &cotangents[1] {
            MaybeZero::Value(value) => value.atom_id().unwrap(),
            MaybeZero::Zero(_) => panic!("expected a live linear-input cotangent"),
        };

        let builder = context.builder().clone();
        drop((context, cotangents));
        let builder = Rc::try_unwrap(builder).expect("the builder is uniquely owned once every tracer is dropped");
        let program = builder
            .into_inner()
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = linear_call [residual_count=1] %0 %1 [
                    forward={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = add %1 %0
                        in (%2)
                    },
                    transpose={
                        lambda %0:f64[], %1:f64[] .
                        let %2:f64[] = mul %1 %0
                        in (%2)
                    },
                ]
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_linear_call_operation_transposition_zeros_known_operand_cotangents() {
        // The transpose region of a transpose-only call returns one cotangent per linear operand, in operand order
        // after the leading residuals. Residual operands are fixed primal parameters rather than linear inputs, and
        // a *known* linear operand is not being differentiated, so both receive structural zeros while the replayed
        // region's cotangents are assigned only to the unknown linear operands.
        let r#type = ArrayType::scalar(DataType::F64);
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let residual = transpose_builder.add_input(r#type.clone());
        let output_cotangent = transpose_builder.add_input(r#type.clone());
        let first_input_cotangent = transpose_builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![residual, output_cotangent])
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_input_cotangent, output_cotangent],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let residual = context.input(r#type.clone());
        let known_linear = context.input(r#type.clone());
        let output_cotangent = context.input(r#type.clone());

        let linear_types = vec![r#type.clone(), r#type.clone()];
        let cotangents = LinearCallOperation::transpose_only(1, linear_types, vec![r#type.clone()])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Known(residual), PartialValue::Unknown(r#type), PartialValue::Known(known_linear)],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();

        assert_eq!(cotangents.len(), 3);
        assert!(cotangents[0].is_zero());
        assert!(matches!(cotangents[1], MaybeZero::Value(_)));
        assert!(cotangents[2].is_zero());
    }

    #[test]
    fn test_linear_call_operation_transposition_preserves_structural_zero_outputs() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Explicit).unwrap()]).unwrap();
        let primal_type = ArrayType::scalar(DataType::F64)
            .with_sharding(Sharding::new(mesh, Vec::new()).unwrap().with_unreduced_axes(["x"]).unwrap())
            .unwrap();
        let tangent_type = primal_type.tangent();
        let cotangent_type = primal_type.cotangent();
        assert_ne!(tangent_type, cotangent_type);

        // A canonical `zero` transpose-region output is already typed in the linear input's cotangent space.
        // Recovering its structural zero must retain that type instead of dualizing its sharding a second time.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        transpose_builder.add_input(cotangent_type.clone());
        let zero = transpose_builder
            .add_instruction(ZeroOperation::new(cotangent_type.clone()), Vec::new(), Vec::new())
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = LinearCallOperation::transpose_only(0, vec![tangent_type.clone()], vec![tangent_type.clone()])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(tangent_type.clone())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));

        // `zero_like` is equally structural even though it consumes an exemplar input. Opaque region replay must
        // recognize it instead of turning the result into a live zero-valued tracer.
        let mut transpose_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = transpose_builder.add_input(cotangent_type.clone());
        let zero_like = transpose_builder
            .add_instruction(ZeroLikeOperation::new(), Vec::new(), vec![output_cotangent])
            .unwrap()[0];
        let transpose = transpose_builder
            .build::<Vec<Array>, Vec<Array>>(vec![zero_like], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let driver = TestTranspositionDriver { region: transpose.entry_region_ref() };
        let mut context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let output_cotangent = context.input(cotangent_type.clone());
        let cotangents = LinearCallOperation::transpose_only(0, vec![tangent_type.clone()], vec![tangent_type])
            .transpose(
                &mut context,
                &driver,
                &[PartialValue::Unknown(primal_type.tangent())],
                &[MaybeZero::Value(output_cotangent)],
            )
            .unwrap();
        assert!(matches!(&cotangents[0], MaybeZero::Zero(r#type) if r#type == &cotangent_type));
    }
}
