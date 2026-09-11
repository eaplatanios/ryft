use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain, ProjectedContext, StagingContext, ValueResolution};
use crate::differentiation::reverse::TransposableOperation;
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::{
    ResidualZeroProvider, ZeroSpaceBoundaryReconstruction, ZeroSpaceBoundaryRole,
    capture_and_validate_zero_residual_values,
};
use crate::differentiation::{DifferentiationBoundaryPosition, DifferentiationError};
use crate::macros::check_count;
use crate::operations::{AddOperation, ReferenceAddUpdateOperation, ReferenceNewOperation};
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialTracer,
    PartialValue, PartiallyEvaluatableOperation, PartitionedProgram,
};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::{
    Atom, AtomId, BindingRegionDriver, EmptyRegionDriver, MaybeZero, Operation, OperationProjection, OperationProvider,
    Program, ProgramBuilder, ProgramError, ProjectedValue, Provenance, ProvenanceScope, ReferenceBoundary,
    ReferenceIdentity, ReferenceRoot, Region, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver, Type,
    TypeError, TypeIdentityPosition, Typed, Value, ValueId, ValueProjection,
};
use crate::tracing::{Tracer, TracerState, TracingContext};

/// Represents a differentiation _dual_ value which is a _primal_ value paired with a _tangent_ value. In the
/// context of differentiating a function `f(x)`, the value `y = f(x)` is the primal value and its tangent `ẏ` is
/// the directional derivative of `f` at `x` along an input tangent (i.e., perturbation direction) `ẋ` (i.e., the
/// Jacobian-vector product `ẏ = (∂f/∂x)(x) · ẋ`). Forward-mode differentiation propagates a dual `(x, ẋ)` at the
/// input to the dual `(y, ẏ) = (f(x), (∂f/∂x)(x) · ẋ)` at the output. This is the data that the per-operation
/// [`jvp`](DifferentiableOperation::jvp) rules consume and produce.
///
/// The tangent need not have the same type as the primal. Its type is determined by [`DifferentiableType::tangent`].
/// For example, an array stored using an unsigned low-precision floating-point representation may carry an `F32`
/// tangent.
#[derive(Clone, Debug)]
pub struct DifferentiationDual<V: Typed> {
    /// Primal value of this dual.
    primal: V,

    /// Tangent value of this dual. Note that this can be a [`MaybeZero::Zero`] enabling structural zero propagation.
    tangent: MaybeZero<V>,
}

impl<V: Value<Type: DifferentiableType>> DifferentiationDual<V> {
    /// Creates a new [`DifferentiationDual`], canonicalizing its tangent representation from the primal's tangent type.
    /// A live tangent remains live when the primal has a nontrivial tangent space, while structural zeros and all
    /// tangents of primals with a zero tangent space use a [`MaybeZero::Zero`] carrying the canonical tangent type.
    ///
    /// # Errors
    ///
    /// Returns a [`DifferentiationError`] if `primal` has no tangent representation or if a live `tangent` does not
    /// have the tangent type required by `primal`.
    #[inline]
    pub fn new<T: Into<MaybeZero<V>>>(primal: V, tangent: T) -> Result<Self, DifferentiationError> {
        let tangent_type = primal.r#type().tangent()?;
        let tangent = match tangent.into() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(tangent) => {
                if tangent.r#type().as_ref() != &tangent_type {
                    return Err(TypeError::invalid(format!(
                        "tangent type {} does not match type {} required by primal type {}",
                        tangent.r#type().as_ref(),
                        tangent_type,
                        primal.r#type().as_ref(),
                    ))
                    .into());
                }
                if tangent_type.is_zero_space() { MaybeZero::Zero(tangent_type) } else { MaybeZero::Value(tangent) }
            }
        };
        Ok(Self { primal, tangent })
    }

    /// Creates a new [`DifferentiationDual`] with a [`MaybeZero::Zero`] tangent carrying the primal's concrete
    /// tangent boundary [`Type`]. A primal with a zero tangent space uses its first-class zero-space [`Type`].
    ///
    /// # Errors
    ///
    /// Returns a [`DifferentiationError`] if `primal` has no tangent representation.
    #[inline]
    pub fn new_with_zero_tangent(primal: V) -> Result<Self, DifferentiationError> {
        let tangent = MaybeZero::Zero(primal.r#type().tangent()?);
        Ok(Self { primal, tangent })
    }

    /// Returns whether this [`DifferentiationDual`] has an _active tangent_ when handed to a differentiated child
    /// region (i.e., whether that region receives a live tangent input at the dual's position for this invocation).
    /// Callers use this classification to select input indices for [`DifferentiationDriver::jvp_program`]: a dual
    /// whose tangent is a live [`MaybeZero::Value`] is active. A dual whose tangent is a structural [`MaybeZero::Zero`]
    /// is active only when its type alone can supply a real tangent input at the child boundary. This excludes plumbing
    /// references, tangent types carrying a runtime (i.e., [`Reference`](TypeIdentityPosition::Reference)-position)
    /// identity, and zero differential spaces.
    #[inline]
    pub fn is_tangent_active(&self) -> bool {
        match &self.tangent {
            MaybeZero::Value(_) => true,
            MaybeZero::Zero(tangent_type) => {
                can_materialize_zero_tangent_from_type(self.primal.r#type().as_ref(), tangent_type)
                    && !tangent_type.is_zero_space()
            }
        }
    }
}

impl<V: Value> DifferentiationDual<V> {
    /// Returns the primal value of this [`DifferentiationDual`].
    #[inline]
    pub fn primal(&self) -> &V {
        &self.primal
    }

    /// Returns the tangent value of this [`DifferentiationDual`].
    #[inline]
    pub fn tangent(&self) -> &MaybeZero<V> {
        &self.tangent
    }

    /// Consumes this [`DifferentiationDual`] and returns its primal and tangent values.
    #[inline]
    pub fn into_parts(self) -> (V, MaybeZero<V>) {
        (self.primal, self.tangent)
    }
}

impl<V: Typed + Display> Display for DifferentiationDual<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.tangent {
            MaybeZero::Zero(_) => write!(formatter, "{} + 0ε", self.primal),
            MaybeZero::Value(tangent) => write!(formatter, "{} + {}ε", self.primal, tangent),
        }
    }
}

/// Returns whether the types permit materializing a structural zero tangent without runtime operands. A
/// reference primal requires its rule to allocate a tangent reference, and a tangent type carrying a runtime (i.e.,
/// [`Reference`](TypeIdentityPosition::Reference)-position) identity needs live operands to construct its zero. This
/// checks type requirements, not whether the operation family provides zero construction. It is shared by the all-zero
/// fast paths and [`DifferentiationDual::is_tangent_active`].
fn can_materialize_zero_tangent_from_type<T: Type>(primal_type: &T, tangent_type: &T) -> bool {
    !primal_type.is_reference()
        && !tangent_type.identities().any(|(position, _)| position == TypeIdentityPosition::Reference)
}

/// Linearization of a [`Program`] computing `y = f(x)`, split into a nonlinear primal sub-program and a linear tangent
/// sub-program that communicate through a residual environment. This is the result of the program linearization
/// transform (i.e., [`Program::linearize`]). Direct linearization differentiates the source program while partially
/// evaluating the differentiated values, producing these two communicating programs without first constructing a
/// fused JVP program:
///
///   - the [`primal`](Self::primal) sub-program `x ↦ (y, r)`, computing the primal outputs `y = f(x)` together with
///     the residuals `r` (i.e., the intermediate values of the derivative computation that depend only on `x`; e.g.,
///     `cos(x)` when `f` is `sin`), and
///   - the [`tangent`](Self::tangent) sub-program `(live(ẋ), r) ↦ live(ẏ)`, computing
///     `ẏ = (∂f/∂x)(x) · ẋ`. Here `live(ẋ)` contains one SSA input for each primal input whose tangent type is not a
///     zero differential space, and `live(ẏ)` similarly contains one Single Static Assignment (SSA) output for each
///     primal output whose tangent type is not a zero differential space. A tangent in a zero differential space can
///     only be zero, so it requires no SSA slot; the corresponding primal value remains in the primal program, and
///     structured callable APIs reconstruct the uniquely determined typed zero where their public result structure
///     requires it. The tangent program is linear in `ẋ`, with the linearization point `x` entering only through the
///     residuals `r`.
///
/// This is the domain-free, interpretation-free core shared by every linearization entry point. It carries only the
/// two sub-programs and the residual count that relates them, leaving the concrete primal outputs to be recovered by
/// callers that interpret [`primal`](Self::primal) under a value semantics of their choice.
///
/// # Reference Arguments
///
/// Executing these raw programs requires the [`ReferenceBoundary`] contract of [`Program::jvp`]: distinct primal
/// reference inputs and captured reference bindings must denote distinct allocations, and each supplied tangent
/// reference must be distinct from that entire primal boundary and from the other tangent references. Views count
/// as aliases of their corresponding allocations even when their accessed elements do not overlap. These are caller
/// obligations; constructing or interpreting the returned programs does not insert runtime alias validation.
///
/// Run the primal program once for a linearization point and pass its trailing residuals to the corresponding tangent
/// program in their original order. Reference residuals retain live allocation identity and may alias the primal
/// references they forward; they are not additional independent boundary arguments. Keep their state and lifetime
/// consistent with the generated accesses. Each tangent invocation must satisfy the boundary contract, including
/// against primal references that are absent from the residual list. [`ForwardModeDifferentiate::linearize`] returns
/// a [`Pushforward`] whose [`apply`](Pushforward::apply) function checks supplied tangent reference identities.
///
/// # Differentiation Pipeline
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   source["Closure or Immutable Program"] --> direct["&lt;code&gt;jvp&lt;/code&gt;: Primals plus Tangents"]
///   direct --> dual_outputs["Primal Outputs plus Output Tangents"]
///   direct --> forward_jacobian["Forward Jacobian via Batched Input Directions"]
///   source --> linearize["Linearize with Unknown Tangents"]
///   linearize --> primal["Primal Program: x to y plus Residuals"]
///   linearize --> tangent["Linear Tangent Program: dx plus Residuals to dy"]
///   primal --> residuals["Evaluate Once and Save Residual Values"]
///   tangent --> pushforward["Reusable &lt;code&gt;Pushforward&lt;/code&gt;"]
///   residuals --> pushforward
///   tangent --> transpose["Transpose in Reverse Dataflow Order"]
///   transpose --> pullback["Reusable &lt;code&gt;Pullback&lt;/code&gt;"]
///   residuals --> pullback
///   pullback --> reverse_jacobian["Reverse Jacobian via Batched Output Cotangents"]
///   pullback --> gradient["Scalar-Output Gradient by Seeding One"]
///   gradient --> hessian["Hessian by Differentiating the Gradient"]
/// ```
///
/// The diagram includes both direct forward mode and the reverse-mode path built from this structural split. Concrete
/// [`Pushforward`] and [`Pullback`](crate::Pullback) callables additionally retain residual values from one
/// linearization point; this [`Linearization`] itself does not.
#[cfg_attr(doc, aquamarine::aquamarine)]
#[derive(Clone, Debug)]
pub struct Linearization<V: Value, O: Operation<Type = V::Type>> {
    /// Nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal outputs
    /// `y = f(x)` followed by the residuals `r`, its trailing [`residual_count`](Self::residual_count) outputs, which
    /// form the residual environment consumed by the tangent sub-program. Shared ownership preserves the cached
    /// program identity when a linearization is cloned or its programs are attached as callees.
    primal: Arc<Program<V, O, Vec<V>, Vec<V>>>,

    /// Linear tangent sub-program `(live(ẋ), r) ↦ live(ẏ)`. It has one leading Single Static Assignment (SSA) input
    /// for each selected primal input, in selection order and omitting zero differential spaces. [`Self::new`] selects
    /// all inputs in source order. These tangent inputs are followed by the residuals `r`, and one SSA output for each
    /// primal output with a nonzero differential space and a live tangent root. Inactive reference outputs have no
    /// tangent slot. Shared ownership lets callers retain this program independently of the primal program.
    tangent: Arc<Program<V, O, Vec<V>, Vec<V>>>,

    /// Number of residuals `r` threaded from the primal sub-program into the tangent sub-program (i.e., the count of
    /// the trailing outputs of [`primal`](Self::primal) and of the trailing inputs of [`tangent`](Self::tangent)).
    residual_count: usize,
}

impl<V: Value, O: Operation<Type = V::Type>> Linearization<V, O> {
    /// Creates a new [`Linearization`] from its parts, validating the boundary contract documented on [`Linearization`]
    /// where `primal` produces its primal outputs followed by its trailing `residual_count` residuals, and `tangent`
    /// consumes one tangent input per non-zero differential input followed by those same residuals and produces one
    /// tangent output per non-zero differential output. Violations (e.g., too few primal outputs or tangent inputs to
    /// hold the residuals, sub-program boundary counts that disagree with each other, or a residual whose primal output
    /// type differs from its tangent input type) are reported as [`MalformedProgram`](ProgramError::MalformedProgram)
    /// errors. [`Program::linearize`] is the function that typically calls this function and constructs
    /// [`Linearization`]s.
    ///
    /// Note that the stability of the tangent program's boundary liveness is load-bearing beyond construction.
    /// Transposition recovers the residual partition and each disconnected input's residual mapping by *recomputing*
    /// which tangent inputs are live from the stored program, rather than storing that partition here. Any pass that
    /// rewrites the tangent program between linearization and transposition must therefore preserve its input liveness
    /// exactly, or the pairing degrades to the residual-count check and typed extent-mismatch errors.
    ///
    /// # Parameters
    ///
    ///   - `primal`: Primal sub-program `x ↦ (y, r)`.
    ///   - `tangent`: Tangent sub-program `(live(ẋ), r) ↦ live(ẏ)` whose leading tangent inputs correspond to the
    ///     active primal inputs.
    ///   - `residual_count`: Number of trailing primal outputs that are residuals consumed by `tangent`.
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residual_count: usize,
    ) -> Result<Self, ProgramError>
    where
        V::Type: DifferentiableType,
    {
        let input_indices = (0..primal.input_ids().len()).collect::<Vec<_>>();
        Self::new_with_respect_to(primal, tangent, residual_count, &input_indices)
    }

    /// Creates a new [`Linearization`] from its parts with respect to selected primal inputs. The tangent program
    /// consumes their tangents in `input_indices` order, omitting zero differential spaces, followed by the residuals.
    /// Primal inputs and outputs retain their original order. [`Self::new`] selects every input in source order.
    /// Reference outputs rooted in unselected inputs or captures have no tangent slot. All other boundary checks are
    /// the same as [`Self::new`].
    ///
    /// # Parameters
    ///
    ///   - `primal`: Primal sub-program `x ↦ (y, r)`.
    ///   - `tangent`: Tangent sub-program whose leading inputs follow the selected input order.
    ///   - `residual_count`: Number of trailing primal outputs that are residuals consumed by `tangent`.
    ///   - `input_indices`: Unique primal input indices in tangent-input order. Selected zero-space inputs have no
    ///     tangent slot and are omitted when validating that order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] for duplicate or out-of-range indices, including repeated zero-space
    /// inputs. Returns [`ProgramError::MalformedProgram`] for incompatible program boundaries and propagates
    /// differential-type and reference-analysis errors.
    pub fn new_with_respect_to(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residual_count: usize,
        input_indices: &[usize],
    ) -> Result<Self, ProgramError>
    where
        V::Type: DifferentiableType,
    {
        let arguments = JvpAndLinearizationTransformArguments::new(primal.entry_region_ref(), input_indices)?;
        let primal_output_count = primal.output_ids().len().checked_sub(residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linearization primal program produces {} outputs which is fewer than its {} residuals",
                primal.output_ids().len(),
                residual_count,
            ))
        })?;
        let tangent_input_count = tangent.input_ids().len().checked_sub(residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linearization tangent program consumes {} inputs which is fewer than its {} residuals",
                tangent.input_ids().len(),
                residual_count,
            ))
        })?;
        let differentiable_primal_inputs = arguments
            .input_indices
            .iter()
            .map(|&index| &primal.atoms()[primal.input_ids()[index].index()])
            .collect::<Vec<_>>();
        if tangent_input_count != differentiable_primal_inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization tangent program consumes {} tangent inputs \
                 while the primal program has {} active inputs",
                tangent_input_count,
                differentiable_primal_inputs.len(),
            )));
        }
        let output_activity = primal.entry_region_ref().tangent_output_mask(&arguments.input_indices)?;
        let differentiable_primal_outputs = primal
            .outputs()
            .take(primal_output_count)
            .zip(output_activity)
            .filter_map(|(output, active)| active.then_some(output))
            .collect::<Vec<_>>();
        if tangent.output_ids().len() != differentiable_primal_outputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization tangent program produces {} outputs \
                 while the primal program has {} nonzero differential outputs",
                tangent.output_ids().len(),
                differentiable_primal_outputs.len(),
            )));
        }
        for (index, (primal_input, tangent_input)) in
            differentiable_primal_inputs.into_iter().zip(tangent.inputs().take(tangent_input_count)).enumerate()
        {
            let primal_type = primal_input.r#type();
            let tangent_type = primal_type.tangent()?;
            if tangent_input.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "linearization tangent input {} has type {} but primal input type {} requires tangent type {}",
                    index,
                    tangent_input.r#type().as_ref(),
                    primal_type,
                    tangent_type,
                )));
            }
        }
        for (index, (primal_output, tangent_output)) in
            differentiable_primal_outputs.into_iter().zip(tangent.outputs()).enumerate()
        {
            let primal_type = primal_output.r#type();
            let tangent_type = primal_type.tangent()?;
            if tangent_output.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "linearization tangent output {} has type {} but primal output type {} requires tangent type {}",
                    index,
                    tangent_output.r#type().as_ref(),
                    primal_type,
                    tangent_type,
                )));
            }
        }
        let primal_residuals = primal.outputs().skip(primal_output_count);
        let tangent_residuals = tangent.inputs().skip(tangent_input_count);
        for (index, (residual, input)) in primal_residuals.zip(tangent_residuals).enumerate() {
            if residual.r#type().as_ref() != input.r#type().as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "linearization residual {index} has type {} in the primal program \
                     but type {} in the tangent program",
                    residual.r#type().as_ref(),
                    input.r#type().as_ref(),
                )));
            }
        }
        Ok(Self { primal: Arc::new(primal), tangent: Arc::new(tangent), residual_count })
    }

    /// Returns the nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal
    /// outputs `y = f(x)` followed by the residuals `r` (i.e., the intermediate values of the derivative computation
    /// that depend only on `x`) whose trailing [`residual_count`](Self::residual_count) outputs form the residual
    /// environment consumed by the [`tangent`](Self::tangent) sub-program. Callers executing it must satisfy the
    /// [reference arguments contract](Linearization#reference-arguments), which this raw program does not validate.
    #[inline]
    pub fn primal(&self) -> &Arc<Program<V, O, Vec<V>, Vec<V>>> {
        &self.primal
    }

    /// Returns the compact linear tangent sub-program `(live(ẋ), r) ↦ live(ẏ)`. The sub-program is linear in its
    /// tangent inputs, ordered by the selection passed to [`Self::new_with_respect_to`], with zero differential spaces
    /// omitted. The linearization point `x` enters only through the residuals `r`. Callers must supply the matching
    /// primal residuals and satisfy the [reference arguments contract](Linearization#reference-arguments) on every
    /// invocation, which this raw program does not validate.
    ///
    /// Inputs start with the selected input tangents, including tangent references for selected reference inputs,
    /// followed by the [`residual_count`](Self::residual_count) residuals returned by [`primal`](Self::primal).
    /// Numeric residuals carry values computed at the linearization point. Reference residuals preserve the primal
    /// reference's identity rather than snapshotting its contents, allowing the tangent program to access that same
    /// allocation. Reference-typed program constants remain constants in the tangent program and do not occupy
    /// residual input slots.
    #[inline]
    pub fn tangent(&self) -> &Arc<Program<V, O, Vec<V>, Vec<V>>> {
        &self.tangent
    }

    /// Returns the number of residuals `r` threaded from the primal sub-program into the tangent sub-program
    /// (i.e., the count of the trailing outputs of [`primal`](Self::primal) and of the trailing inputs of
    /// [`tangent`](Self::tangent)).
    #[inline]
    pub fn residual_count(&self) -> usize {
        self.residual_count
    }

    /// Consumes this [`Linearization`] and returns its [`primal`](Self::primal) sub-program, [`tangent`](Self::tangent)
    /// sub-program, and [`residual_count`](Self::residual_count), in that order, moving the shared handles without
    /// cloning the programs. Executing the returned programs retains the caller obligations in the
    /// [reference argument contract](Linearization#reference-arguments).
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn into_parts(self) -> (Arc<Program<V, O, Vec<V>, Vec<V>>>, Arc<Program<V, O, Vec<V>, Vec<V>>>, usize) {
        (self.primal, self.tangent, self.residual_count)
    }

    /// Returns the compact forward-mode pushforward program `(live(ẋ), r) ↦ live(ẏ)`. Because linearization already
    /// produces the pushforward as its unknown half, this clones the shared handle to [`tangent`](Self::tangent),
    /// without cloning its program (i.e., the identity counterpart of [`pullback`](Self::pullback), which derives its
    /// program by transposition). The returned program has the same unchecked
    /// [reference argument contract](Linearization#reference-arguments) as [`Self::tangent`];
    /// use [`Pushforward::apply`] for a callable that validates the reference arguments.
    #[inline]
    pub fn pushforward(&self) -> Arc<Program<V, O, Vec<V>, Vec<V>>> {
        self.tangent.clone()
    }

    /// Builds the compact reverse-mode pullback program `(live(ȳ), r) ↦ live(x̄)` by transposing the
    /// [`tangent`](Self::tangent) sub-program. Conceptually, it takes the output cotangents `ȳ` followed by the
    /// residuals `r` and produces the input cotangents `x̄ = (∂f/∂x)(x)ᵀ · ȳ`. It is the derived third member of this
    /// [`Linearization`]'s program family, alongside the stored [`primal`](Self::primal) and [`tangent`](Self::tangent)
    /// sub-programs. Rather than re-keying each bilinear operation of the tangent sub-program into a closed captured
    /// factor (e.g., folding a scalar `Mul` against a known operand into a multiply-by-a-captured-constant) by folding
    /// the consuming residual value, this function leaves the tangent sub-program in the primal operation family `O`
    /// and transposes it through [`RegionRef::transpose_shared`], including the saved dimension mappings needed for
    /// disconnected cotangent zeros. The tangent sub-program's inputs are `(ẋ, r)`, so it is transposed with respect
    /// to the leading tangent inputs `ẋ` while the trailing [`residual_count`](Self::residual_count) residual inputs
    /// are held as known parameters. Partition-aware transposition then threads each known residual through to the
    /// pullback as a pullback input (consumed by the adjoint operation that the bilinear operation's transpose rule
    /// stages), rather than folding it into a captured factor, so the returned pullback program stays over the primal
    /// operation family `O` and produces the cotangents of the linear tangent inputs only, in the selected input order
    /// used to construct this linearization.
    ///
    /// Transposition is served from the tangent region's retained transform cache. This function clones the cached
    /// program to return an owned value; repeated calls reuse the transposition but still clone its program.
    #[inline]
    pub fn pullback(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: TransposableOperation<V, O>
            + ResidualZeroProvider<V::Type, Operation = O>
            + OperationProvider<V::Type, ReferenceNewOperation<V::Type, V::Type>, Operation = O>
            + OperationProvider<V::Type, ReferenceAddUpdateOperation<V::Type, V::Type>, Operation = O>
            + From<AddOperation<V::Type>>,
    {
        // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
        // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are
        // not folded into captured factors here. The subtraction cannot underflow because `Self::new` validated that
        // the tangent program consumes at least `residual_count` inputs. The default cotangent destination kinds apply,
        // and so a reference-typed tangent input exposes a cotangent reference input in the pullback (refer to the
        // documentation of `Program::transpose_with_respect_to` for more information).
        Ok(self.tangent.transpose_with_trailing_residuals_shared(self.residual_count, &[])?.as_ref().clone())
    }
}

/// Pushforward of a function `f` at a linearization point `x` (i.e., the linear map `ẋ ↦ (∂f/∂x)(x) · ẋ`), packaged
/// as a reusable callable. This is what [`ForwardModeDifferentiate::linearize`] returns (i.e., the analogue of
/// [JAX's `linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)), and it is the forward-mode
/// dual of [`Pullback`](crate::Pullback), whose callable applies the transposed map `ȳ ↦ (∂f/∂x)(x)ᵀ · ȳ` instead.
/// It wraps the pushforward program `(ẋ, r) ↦ ẏ` accumulated while partially evaluating the differentiated closure,
/// closed over the residuals `r` recovered at the linearization point. [`apply`](Self::apply) computes
/// `ẏ = (∂f/∂x)(x) · ẋ` by appending the residuals to the flattened tangents `ẋ`, interpreting the pushforward program,
/// and reshaping the flat tangent outputs against the closure's output structure. It thus pushes any number of tangents
/// through the function's Jacobian without re-tracing or re-differentiating (e.g., replaying every coordinate basis
/// tangent to build a Jacobian), amortizing the cost of differentiating once over many tangent applications.
/// The stored program is compact as zero differential input and output leaves are absent. [`apply`](Self::apply)
/// filters those input leaves and restores typed zeros in the returned public structure.
///
/// The context `C` supplies the value semantics and operation family, `Input` is the closure's structured input type,
/// and `Output` is its structured output type, whose [`ParameterStructure`](Parameterized::ParameterStructure) is
/// retained so that the flat tangent outputs reshape back into `Output::To<C::Value>`. `Input` is carried as a type
/// parameter so that [`apply`](Self::apply) infers the tangent family from the pushforward itself rather than requiring
/// a turbofish.
pub struct Pushforward<C: Context, Input, Output: Parameterized<C::Value>> {
    /// [`Context`] that the pushforward was built in. [`apply`](Self::apply) replays the pushforward program in it,
    /// mirroring how [`Pullback`](crate::Pullback) replays its pullback program.
    context: C,

    /// Pushforward [`Program`] over the primal operation family in the context's staged [`Constant`](Domain::Constant)
    /// space, mapping `[tangents ++ residuals]` to the flat tangent outputs. Its literal constants are lifted through
    /// the context's [`lift`](Context::lift) when [`apply`](Self::apply) replays it.
    program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Linearization-point residuals consumed by [`program`](Self::program), appended after the tangents when
    /// interpreting it.
    residuals: Vec<C::Value>,

    /// Reconstruction plan used by [`apply`](Self::apply) to restore zero-space output tangents omitted from
    /// [`program`](Self::program).
    tangent_reconstruction: ZeroSpaceBoundaryReconstruction<C::Value>,

    /// Complete public primal input boundary. The executable pushforward omits inputs whose derived tangent type is a
    /// zero differential space.
    primal_input_types: Vec<C::Type>,

    /// Complete public primal output boundary. The executable pushforward omits outputs whose derived tangent type is
    /// a zero differential space.
    primal_output_types: Vec<C::Type>,

    /// Contains the canonical identities of every reference bound at the primal boundary (inputs and captures alike),
    /// etained so that [`apply`](Self::apply) can reject a tangent reference aliasing one of them, exactly as
    /// [`Pullback`](crate::Pullback) rejects an aliasing cotangent destination.
    primal_references: ReferenceBoundary<DifferentiationBoundaryPosition>,

    /// Parameter structure of the closure's output, used to reshape the flat tangent outputs.
    output_structure: Output::ParameterStructure,

    /// Encodes the closure's input family `Input` so that [`apply`](Self::apply) can flatten the tangents without a
    /// turbofish. No `Input::ParameterStructure` is stored alongside it because [`apply`](Self::apply) only _flattens_
    /// its structured tangent argument, which needs no stored structure, and rebuilds structure only on the
    /// tangent-output side through `output_structure`. [`Pullback`](crate::Pullback) mirrors this with a stored input
    /// structure and a phantom `Output`.
    marker: PhantomData<fn() -> Input>,
}

impl<
    C: Context<Type: DifferentiableType>,
    Input: Parameterized<C::Value>,
    Output: Parameterized<C::Value, Family: ParameterizedFamily<C::Value>>,
> Pushforward<C, Input, Output>
{
    /// Creates a [`Pushforward`] from a compact tangent [`Program`] and the complete primal boundary from which that
    /// program was derived. The program has the boundary `(live(ẋ), r) ↦ live(ẏ)`. It omits every tangent input and
    /// output whose differential space contains only zero. Its boundary therefore cannot recover the omitted leaves'
    /// positions or types, and the tangent-type mapping is not generally invertible. `primal_input_types` preserves
    /// the complete public input boundary so that [`apply`](Self::apply) can validate and filter its tangent arguments.
    /// `tangent_reconstruction` independently preserves the reconstruction plan for omitted output leaves, while
    /// `primal_output_types` validates the compact program boundary and remains available to reverse-mode conversion.
    ///
    /// This function validates the relationship among all three boundaries. In particular, the program must consume
    /// one leading input for every nonzero tangent in `primal_input_types`, followed by one input for every residual,
    /// and must produce one output for every nonzero tangent in `primal_output_types`. A mismatch is reported as a
    /// [`MalformedProgram`](ProgramError::MalformedProgram) error.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context in which [`apply`](Self::apply) interprets or stages the compact pushforward program.
    ///   - `program`: Compact pushforward program `(live(ẋ), r) ↦ live(ẏ)` whose trailing inputs consume the
    ///     residual values.
    ///   - `residuals`: Primal values `r` captured at the linearization point, in the same order as the program's
    ///     trailing inputs.
    ///   - `tangent_reconstruction`: Reconstruction plan for zero-space output tangents rebuilt by
    ///     [`apply`](Self::apply).
    ///   - `primal_input_types`: Complete flattened input-type boundary of the original primal function, including
    ///     leaves whose tangent spaces contain only zero and which are consequently absent from `program`.
    ///   - `primal_output_types`: Complete flattened output-type boundary of the original primal function, including
    ///     leaves whose tangent spaces contain only zero and which are consequently absent from `program`.
    ///   - `primal_references`: Contains the canonical identities of every reference bound at the primal boundary,
    ///     which the tangent references supplied to [`apply`](Self::apply) must not alias.
    ///   - `output_structure`: Parameter structure used to rebuild the complete public tangent output after typed
    ///     zeros have been inserted for the omitted leaves.
    pub fn new(
        context: C,
        program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        residuals: Vec<C::Value>,
        tangent_reconstruction: ZeroSpaceBoundaryReconstruction<C::Value>,
        primal_input_types: Vec<C::Type>,
        primal_output_types: Vec<C::Type>,
        primal_references: ReferenceBoundary<DifferentiationBoundaryPosition>,
        output_structure: Output::ParameterStructure,
    ) -> Result<Self, ProgramError> {
        let tangent_input_count = program.input_ids().len().checked_sub(residuals.len()).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "pushforward program consumes {} inputs which is fewer than its {} residuals",
                program.input_ids().len(),
                residuals.len(),
            ))
        })?;
        for (index, (input, residual)) in program.inputs().skip(tangent_input_count).zip(&residuals).enumerate() {
            if input.r#type().as_ref() != residual.r#type().as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward residual {} has type {} in the pushforward program \
                     but carries a value of type {}",
                    index,
                    input.r#type().as_ref(),
                    residual.r#type().as_ref(),
                )));
            }
        }
        let live_input_tangent_types = primal_input_types
            .iter()
            .map(DifferentiableType::tangent)
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter(|r#type| !r#type.is_zero_space())
            .collect::<Vec<_>>();
        if live_input_tangent_types.len() != tangent_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward program consumes {} tangent inputs but its public boundary has {} \
                 nonzero differential inputs",
                tangent_input_count,
                live_input_tangent_types.len(),
            )));
        }
        for (index, (input, tangent_type)) in program.inputs().zip(&live_input_tangent_types).enumerate() {
            if input.r#type().as_ref() != tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward program tangent input {} has type {} but its public boundary requires tangent type {}",
                    index,
                    input.r#type().as_ref(),
                    tangent_type,
                )));
            }
        }
        let live_output_tangent_types = primal_output_types
            .iter()
            .map(DifferentiableType::tangent)
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter(|r#type| !r#type.is_zero_space())
            .collect::<Vec<_>>();
        if live_output_tangent_types.len() != program.output_ids().len() {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward program produces {} tangent outputs but its public boundary has {} nonzero differential \
                 outputs",
                program.output_ids().len(),
                live_output_tangent_types.len(),
            )));
        }
        for (index, (output, tangent_type)) in program.outputs().zip(&live_output_tangent_types).enumerate() {
            if output.r#type().as_ref() != tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward program tangent output {} has type {} but its public boundary requires tangent\
                     type {}",
                    index,
                    output.r#type().as_ref(),
                    tangent_type,
                )));
            }
        }
        Ok(Self {
            context,
            program,
            residuals,
            tangent_reconstruction,
            primal_input_types,
            primal_output_types,
            primal_references,
            output_structure,
            marker: PhantomData,
        })
    }

    /// Returns the pushforward [`Program`] `(ẋ, r) ↦ ẏ` that this callable closes over. Its inputs are the flat
    /// tangents followed by the residuals carried by [`residuals`](Self::residuals).
    #[inline]
    pub fn program(&self) -> &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        &self.program
    }

    /// Returns the linearization-point residuals `r` that this callable closes over, aligned with the trailing inputs
    /// of [`program`](Self::program).
    #[inline]
    pub fn residuals(&self) -> &[C::Value] {
        &self.residuals
    }

    /// Consumes this [`Pushforward`] and returns its context, compact pushforward program, linearization-point
    /// residuals, primal input types, primal output types, and retained primal reference boundary, in that order.
    /// The program maps `(live(ẋ), r)` to `live(ẏ)`, with the residuals `r` aligned to its trailing inputs. The
    /// complete primal types and reference boundary retain information needed to validate subsequent derivative calls
    /// that cannot be recovered from the compact program alone.
    ///
    /// Unlike [`apply`](Self::apply), the returned program does not insert typed-zero values for public tangent leaves
    /// omitted from its Single Static Assignment (SSA) boundary because their differential spaces contain only zero.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        C,
        Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        Vec<C::Value>,
        Vec<C::Type>,
        Vec<C::Type>,
        ReferenceBoundary<DifferentiationBoundaryPosition>,
    ) {
        (
            self.context,
            self.program,
            self.residuals,
            self.primal_input_types,
            self.primal_output_types,
            self.primal_references,
        )
    }

    /// Pushes the structured tangents `tangents` through the linearized Jacobian, returning the tangent outputs. The
    /// tangents are flattened, the linearization-point residuals are appended, the pushforward program is interpreted
    /// at that vector in the context that this pushforward was built in (i.e., the single replay path for both context
    /// flavors: an eager context interprets the pushforward immediately, while a staging context stages it into the
    /// enclosing trace and returns tracers), and the flat tangent outputs are reshaped against the closure's output
    /// structure.
    ///
    /// A reference-typed primal leaf takes a concrete tangent reference, which the tangent program mutates in place.
    /// It must denote an allocation distinct from every reference bound at the primal boundary and from every other
    /// tangent reference, under the identity and alias checks of [`ReferenceBoundary`]. An aliasing tangent is
    /// rejected with an [`InvalidArgument`](ProgramError::InvalidArgument) error before anything is interpreted.
    #[inline]
    pub fn apply(&self, tangents: Input::To<C::Value>) -> Result<Output::To<C::Value>, ProgramError>
    where
        C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>,
    {
        // Flatten the caller's structured tangent tree and first validate it against the complete primal boundary,
        // including the leaves whose differential spaces contain only zero.
        let public_tangents = tangents.into_parameters().collect::<Vec<_>>();
        if public_tangents.len() != self.primal_input_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.primal_input_types.len(),
                actual: public_tangents.len(),
            });
        }

        // Every tangent reference is mutated independently of the primal references, so it must be a distinct
        // allocation. Identity is what is compared here and not reference generation: a primal reference that advanced
        // generations after the differentiated closure ran is still the same allocation and is thus still rejected.
        self.primal_references.validate_differentiation_arguments(
            &self.context,
            public_tangents
                .iter()
                .enumerate()
                .map(|(index, value)| (DifferentiationBoundaryPosition::Tangent(index), value)),
        )?;

        // Validate every public tangent against the tangent type derived from its primal leaf. Forward only the
        // information-carrying values because the compact program has no SSA input for a zero differential space.
        let mut program_inputs = Vec::new();
        for (index, (value, primal_type)) in public_tangents.into_iter().zip(&self.primal_input_types).enumerate() {
            let tangent_type = primal_type.tangent()?;
            if value.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward tangent {} has type {} but its primal boundary requires tangent type {}",
                    index,
                    value.r#type().as_ref(),
                    tangent_type,
                )));
            }
            if !tangent_type.is_zero_space() {
                program_inputs.push(value);
            }
        }

        // Close the compact tangent boundary over the primal residuals and replay it in the originating context.
        program_inputs.extend(self.residuals.iter().cloned());
        let tangent_outputs = self.program.interpret_in_context(&self.context, program_inputs)?.into_iter();

        // Reconstruct the complete flattened public output boundary. Consume one program result for each nonzero
        // tangent space and materialize the uniquely determined typed zero for every omitted zero-space leaf.
        let outputs = self.tangent_reconstruction.rebuild(&self.context, tangent_outputs)?;

        // Restore the closure's original structured output shape after rebuilding every flattened tangent leaf.
        Ok(Output::To::<C::Value>::from_parameters(self.output_structure.clone(), outputs)?)
    }
}

/// Specifies how a [`DifferentiationContext`] uses its underlying [`Context`]s to compute primal values and tangents.
/// The primal computation evaluates the original function. The tangent computation applies its derivative to input
/// tangents. A policy decides whether these computations share one context or use separate contexts, and how a primal
/// value becomes available to the tangent computation.
///
/// [`FusedDifferentiationPolicy`] uses one context for both computations, as in a Jacobian-Vector Product (JVP)
/// transform. [`PartitionedDifferentiationPolicy`] uses separate partial evaluation contexts to compute primal results
/// once and build a tangent program that a [`Pushforward`] can call repeatedly. Both use the same differentiation
/// rules which access [`DifferentiationContext::primal`] and [`DifferentiationContext::tangent`], and pass any primal
/// coefficients needed by tangent work through [`DifferentiationContext::primal_to_tangent`].
///
/// Policies are stateless types, like the structural policies used by the batching transform. When a rule runs through
/// a projected member context, [`ProjectedDifferentiationPolicy`] preserves the enclosing policy's context creation
/// and value transfer. Projection therefore changes the value family visible to the rule while retaining the choice
/// of fused or partitioned computation.
pub trait DifferentiationPolicy<C: Context>: Copy + Clone + Debug {
    /// Creates a separate context for tangent work, if the policy needs one. [`DifferentiationContext::new`]
    /// calls this function once during construction. Returning `None` makes [`DifferentiationContext::tangent`] return
    /// the exact same context instance as [`DifferentiationContext::primal`]. Returning `Some(context)` makes it return
    /// that separate context instead.
    ///
    /// Both contexts have type `C` and use the same value and operation families, but may own different staged
    /// [`Program`]s. A separate context must retain provenance tracking and support the value transfers defined
    /// by [`primal_to_tangent`](Self::primal_to_tangent). For example, the partitioned policy creates a sibling
    /// partial evaluation context whose effectful operations remain in the tangent program.
    ///
    /// # Parameters
    ///
    ///   - `primal`: [`Context`] used for the primal computation, based on which a separate tangent context
    ///     can be constructed.
    fn tangent_context(primal: &C) -> Option<C>;

    /// Makes a primal value usable by operations in the tangent [`Context`]. This function transfers the value itself;
    /// it does not differentiate it or compute its tangent. For example, the rule for `sin(x)` needs the primal
    /// coefficient `cos(x)` to compute `cos(x) * ẋ` in the tangent context.
    ///
    /// [`FusedDifferentiationPolicy`] returns the value unchanged because both computations share a context.
    /// [`PartitionedDifferentiationPolicy`] imports it as a known value into the tangent context, allowing pure
    /// computations on it to specialize without immediately creating a residual [`Program`] input. This matters for
    /// context-carrying values: arithmetic on the returned value must use the tangent context, even when the value
    /// was originally computed by the primal context.
    ///
    /// Implementations must preserve known value specialization, live reference identity, and any deferred errors.
    /// Transferring a reference does not snapshot its contents. When contexts own separate programs, an imported value
    /// must use the receiving context's bookkeeping for residual inputs and constants, rather than copying identifiers
    /// that belong to the source program.
    ///
    /// # Parameters
    ///
    ///   - `tangent`: Context returned by [`DifferentiationContext::tangent`], which will use the transferred value.
    ///   - `value`: Primal value needed by the tangent computation, such as a coefficient, predicate, or a value
    ///     supplying shape information.
    fn primal_to_tangent(tangent: &C, value: C::Value) -> Result<C::Value, DifferentiationError>;
}

/// [`DifferentiationPolicy`] that uses the same underlying [`Context`] for primal and tangent operations. This is the
/// policy selected by [`DifferentiationContext::fused`], used to compute a Jacobian-Vector Product (JVP) with its
/// primal result and tangent result together. An eager context executes the operations and a staging context records
/// them in the same program.
///
/// No separate tangent context is created, and transferring a primal value to tangent work returns it unchanged.
/// [`PartitionedDifferentiationPolicy`] instead separates the computations so the tangent program can be called
/// repeatedly after computing the primal results once.
#[derive(Copy, Clone, Debug)]
pub struct FusedDifferentiationPolicy;

impl<C: Context> DifferentiationPolicy<C> for FusedDifferentiationPolicy {
    #[inline]
    fn tangent_context(_primal: &C) -> Option<C> {
        None
    }

    #[inline]
    fn primal_to_tangent(_tangent: &C, value: C::Value) -> Result<C::Value, DifferentiationError> {
        Ok(value)
    }
}

/// [`DifferentiationPolicy`] used by [`LinearizationContext`] to compute primal results once and retain a tangent
/// program for repeated [`Pushforward::apply`] calls. Unlike [`FusedDifferentiationPolicy`], it creates a separate
/// tangent [`PartialEvaluationContext`] using [`deferred_sibling`](PartialEvaluationContext::deferred_sibling). The
/// two partial evaluation contexts share a parent context for computations whose inputs are known, but own separate
/// residual programs for work that must run later.
///
/// Primal values needed by tangent work are imported into the tangent context as known values. Pure computations on
/// known inputs can still run in the shared parent and supply saved coefficients to the pushforward. Tangent effects
/// remain in the tangent program even when their inputs are known, so each pushforward call performs its own mutations
/// and creates its own local references. For example, a zero-initialized tangent accumulator must be allocated afresh
/// on every call rather than allocated once during linearization and reused across calls.
#[derive(Copy, Clone, Debug)]
pub struct PartitionedDifferentiationPolicy;

impl<C: Context> DifferentiationPolicy<PartialEvaluationContext<C>> for PartitionedDifferentiationPolicy
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    #[inline]
    fn tangent_context(primal: &PartialEvaluationContext<C>) -> Option<PartialEvaluationContext<C>> {
        Some(primal.deferred_sibling())
    }

    #[inline]
    fn primal_to_tangent(
        tangent: &PartialEvaluationContext<C>,
        value: PartialTracer<C>,
    ) -> Result<PartialTracer<C>, DifferentiationError> {
        Ok(tangent.import_known(&value)?)
    }
}

/// Adapts an existing [`DifferentiationPolicy`] `P` to a [`ProjectedContext`] while preserving how it separates
/// primal and tangent work. This wraps a policy such as [`FusedDifferentiationPolicy`]
/// or [`PartitionedDifferentiationPolicy`], rather than another choice of how to execute differentiation.
///
/// [`DifferentiationContext::project`] preserves existing contexts. When used with [`DifferentiationContext::new`],
/// the wrapper asks `P` to create a tangent context for the enclosing context, then projects that
/// context into the member family. If `P` shares the primal context, the projected rule shares its primal context too.
/// To transfer a value, the wrapper lifts it into the enclosing value family, applies `P`'s transfer, then projects
/// the result back. For example, an array-only rule inside a composite context retains partitioned known value imports
/// even though it sees only array values. All of these calls use static dispatch through `P`.
#[derive(Copy, Clone, Debug)]
pub struct ProjectedDifferentiationPolicy<P>(PhantomData<P>);

impl<T: Type, C: Context, P: DifferentiationPolicy<C>> DifferentiationPolicy<ProjectedContext<C, T>>
    for ProjectedDifferentiationPolicy<P>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
{
    #[inline]
    fn tangent_context(primal: &ProjectedContext<C, T>) -> Option<ProjectedContext<C, T>> {
        P::tangent_context(primal.parent()).map(ProjectedContext::new)
    }

    #[inline]
    fn primal_to_tangent(
        tangent: &ProjectedContext<C, T>,
        value: <C::Value as ValueProjection<T>>::Projected,
    ) -> Result<<C::Value as ValueProjection<T>>::Projected, DifferentiationError> {
        let value = <C::Value as ValueProjection<T>>::from_projected(value);
        P::primal_to_tangent(tangent.parent(), value)?.into_projected().map_err(Into::into)
    }
}

/// Provides call-scoped access to the regions attached to the instruction being differentiated. Transform
/// dispatch constructs a driver for one operation application and passes it directly to that operation's
/// [`jvp`](DifferentiableOperation::jvp) rule. [`RegionDriver`] provides structural region access, while this
/// trait adds differentiation-specific recursion. Region-free applications receive a driver with no regions.
///
/// Structural transform requests accept borrowed [`RegionRef`]s directly, allowing the same request to serve both a
/// region selected from this driver and the entry region of a program rebuilt by an operation rule. Implementations
/// must recursively dispatch each nested instruction with the driver for that nested application.
pub trait DifferentiationDriver<C: Context>: RegionDriver<C::Constant, C::Operation> {
    /// Builds a fused forward mode differentiation program of `region` and returns a shared handle to it. The
    /// `input_indices` select which region inputs to differentiate with respect to and specify their tangent-input
    /// order. Unselected inputs receive structural-zero tangents. All primal inputs remain available to the program.
    /// Selected inputs whose differential spaces contain only zero do not contribute tangent arguments.
    ///
    /// For example, for ordinary differentiable inputs and `y = f(x₀, x₁, x₂)`, selecting `[2, 0]` gives:
    ///
    /// ```text
    ///   (x₀, x₁, x₂, ẋ₂, ẋ₀) → (y, ẏ)
    ///   ẏ = ∂f/∂x₀ · ẋ₀ + ∂f/∂x₂ · ẋ₂
    /// ```
    ///
    /// Here `x₁` still affects the primal result and may affect the derivative coefficients, but its own tangent
    /// contributes nothing. Refer to [`RegionRef::jvp`] for the complete boundary and reference rules.
    /// Callers restore output tangents omitted from the compact program as structural zeros.
    ///
    /// For a nested region, callers derive this selection from incoming duals through
    /// [`DifferentiationDual::is_tangent_active`]; it need not match the user's original input selection. Selection
    /// describes which tangent arguments the child receives, not whether their numerical values are nonzero. A live
    /// tangent can contain zeros, and some structural zeros are classified as active because they can be materialized
    /// as tangent arguments from their types alone.
    ///
    /// The result is shared rather than owned because rules commonly re-attach the derived program as a nested region.
    /// An [`Arc`] lets a caching driver serve one artifact for a region that several programs share instead of
    /// re-differentiating it per program, and it lets repeated attachments of one artifact intern by [`Arc`] identity.
    /// The built-in recursive driver therefore serves the region's retained fused program through the cached
    /// counterpart of [`RegionRef::jvp`], while a custom driver that retains nothing simply derives
    /// the program uncached and wraps it in [`Arc::new`].
    ///
    /// # Parameters
    ///
    ///   - `region`: Region to differentiate.
    ///   - `input_indices`: Unique region input indices in tangent-input order. Zero differential spaces contribute
    ///     no tangent argument. Duplicate or out-of-range indices are rejected; an empty slice selects no inputs.
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_indices: &[usize],
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError>;

    /// Linearizes `region` into separate primal and tangent programs. As in [`Self::jvp_program`], `input_indices`
    /// select which region inputs to differentiate with respect to. All primal inputs remain available to the primal
    /// program. The tangent program receives the selected input tangents in the supplied order, omitting zero
    /// differential spaces, followed by the residual values saved by the primal program.
    ///
    /// For example, for ordinary differentiable inputs and `y = f(x₀, x₁, x₂)`, selecting `[2, 0]` gives:
    ///
    /// ```text
    ///   primal:  (x₀, x₁, x₂) → (y, r)
    ///   tangent: (ẋ₂, ẋ₀, r)  → ẏ
    ///   ẏ = ∂f/∂x₀ · ẋ₀ + ∂f/∂x₂ · ẋ₂
    /// ```
    ///
    /// Here `r` denotes the residuals needed to evaluate the derivative at the chosen primal inputs. They can depend
    /// on `x₁` even though its tangent is zero. The primal program can run once and its residuals can be reused for
    /// different tangent arguments. See [`RegionRef::linearize`] for the complete boundary and
    /// reference rules.
    ///
    /// For nested regions, callers derive the selected indices from incoming duals through
    /// [`DifferentiationDual::is_tangent_active`], rather than copying the user's original input selection. As with
    /// [`Self::jvp_program`], a selected tangent argument may contain zeros; selection records whether an argument is
    /// supplied, not whether its value is numerically nonzero.
    ///
    /// Unlike [`Self::jvp_program`], this hands back an owned [`Linearization`] because its consumers restructure the
    /// component programs instead of attaching them unchanged. The bounded `while` rule, for example, consumes the
    /// primal and tangent halves and rebuilds them into the residual-stacking forward loop and the reversed tangent
    /// scan, so there is no artifact left to share by identity.
    ///
    /// # Parameters
    ///
    ///   - `region`: Region to linearize.
    ///   - `input_indices`: Unique region input indices in tangent-input order. Zero differential spaces contribute
    ///     no tangent argument. Duplicate or out-of-range indices are rejected; an empty slice selects no inputs.
    fn linearize_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_indices: &[usize],
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError>;

    /// Partitions an existing fused Jacobian-Vector Product (JVP) [`Region`] into a known primal program and a residual
    /// tangent program. The region already computes primal and tangent results; this function separates their
    /// computations without differentiating the region again. Custom JVP rules and reconstructed control-flow JVPs
    /// use this when primal and tangent work must run separately.
    ///
    /// The partition supports one primal invocation followed by repeated tangent invocations using its residuals.
    /// Work whose inputs are known may run in the primal program, subject to effect ordering constraints. Local
    /// allocations needed only by tangent work remain in the tangent program so each invocation gets fresh state.
    /// `required_known_outputs` identifies results that must remain available from the primal invocation; partitioning
    /// fails if that requirement cannot be satisfied safely. For example, a fused rule producing `(y, ẏ)` normally
    /// requires `y` to remain known while allowing `ẏ` to depend on each invocation's tangent inputs.
    ///
    /// Requesting partitioning through the driver keeps the operation family's partial-evaluation bounds out of
    /// individual differentiation rules, as with the recursive differentiation requests provided by this trait.
    ///
    /// # Parameters
    ///
    ///   - `region`: Fused Jacobian-Vector Product (JVP) region to partition.
    ///   - `input_known`: One entry per region input, in input order. `true` means its value is available to the primal
    ///     computation; `false` means it is supplied to the tangent computation.
    ///   - `required_known_outputs`: Region output indices that must be produced by the primal computation. An empty
    ///     slice imposes no output requirement.
    fn partition_jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
        required_known_outputs: &[usize],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, DifferentiationError>;

    /// Binds an operation to dual values using the differentiation context's binding semantics, including its
    /// structural zero checks. Rules can recursively differentiate newly constructed operations or replay operations
    /// from a region without wrapping each operand in a [`DifferentiationTracer`]. For example, an eager `while`
    /// operation, this function replays its body this way, while a masked `while` operation differentiates its
    /// rewritten operation and regions.
    ///
    /// The driver supplies the enclosing operation family's differentiation bounds. Requiring those bounds on a
    /// recursive operation rule instead creates a cycle (e.g., differentiating the family requires its `while` rule,
    /// which would then require differentiation of the family again). The built-in driver delegates to the same
    /// internal context function as [`Context::bind`], so recursion preserves its reference and runtime geometry
    /// checks.
    ///
    /// # Parameters
    ///
    ///   - `context`: Differentiation context that computes the operation's primal and tangent results.
    ///   - `operation`: Operation to differentiate.
    ///   - `programs`: Complete attached region programs, in operation-defined order.
    ///   - `inputs`: Operand duals, in operation-input order.
    fn bind_jvp_operation<P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        operation: &C::Operation,
        programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

impl<C: Context> DifferentiationDriver<C> for EmptyRegionDriver {
    fn jvp_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_indices: &[usize],
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot differentiate a program".to_string()).into())
    }

    fn linearize_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_indices: &[usize],
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot linearize a program".to_string()).into())
    }

    fn partition_jvp_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_known: &[bool],
        _required_known_outputs: &[usize],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partition a JVP program".to_string()).into())
    }

    fn bind_jvp_operation<P: DifferentiationPolicy<C>>(
        &self,
        _context: &DifferentiationContext<C, P>,
        _operation: &C::Operation,
        _programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot differentiate an operation".to_string()).into())
    }
}

/// [`DifferentiationDriver`] scoped to one [`Operation`] application. It borrows the application's complete region
/// driver, which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees
/// without materializing a combined region collection. Recursive requests are answered through
/// [`RegionRef::jvp_shared`] and [`RegionRef::linearize_shared`].
struct RecursiveDifferentiationDriver<'r, D> {
    /// Application-scoped region driver, in operation-defined order.
    driver: &'r D,
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveDifferentiationDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<C, D> DifferentiationDriver<C> for RecursiveDifferentiationDriver<'_, D>
where
    C: Context<Type: DifferentiableType>,
    D: RegionDriver<C::Constant, C::Operation>,
    C::Operation: DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>,
{
    #[inline]
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_indices: &[usize],
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError> {
        region.jvp_shared(input_indices)
    }

    #[inline]
    fn linearize_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_indices: &[usize],
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError> {
        region.linearize_shared(input_indices)
    }

    #[inline]
    fn partition_jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
        required_known_outputs: &[usize],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, DifferentiationError> {
        Ok(region.partition_with_configuration(input_known, true, true, Some(required_known_outputs))?.0)
    }

    #[inline]
    fn bind_jvp_operation<P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        operation: &C::Operation,
        programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        context.bind_duals(operation, programs, inputs)
    }
}

// TODO(eaplatanios): Restore the strict `Operation<Type = C::Type>` super-trait bound once the next-generation trait
//  solver stabilizes. The current solver cannot discharge this projection equality at implementation heads whose
//  context type is built from `Self` (E0284); the equality is enforced per method through `where` clauses instead.
/// Represents [`Operation`]s that support forward-mode differentiation (i.e., computing Jacobian-Vector Products
/// or JVPs). Reading an operation as a function `y = f(x₁, …, xₙ)` from its operands to its outputs, the
/// [`jvp`](Self::jvp) function propagates [`DifferentiationDual`]s through it. Each input dual `(xᵢ, ẋᵢ)` pairs an
/// operand with its tangent (i.e., perturbation direction), and the function returns one dual `(yⱼ, ẏⱼ)` per output,
/// where `y = f(x)` is the primal result and `ẏ = Σᵢ (∂f/∂xᵢ)(x) · ẋᵢ` is the directional derivative of `f` at `x`
/// along the input tangents. For example, the `jvp` implementation for the sine operation maps `(x, ẋ)` to `(sin x,
/// cos x · ẋ)`. Both halves use ordinary operations bound through [`DifferentiationContext`]. Shared destinations
/// execute or stage a fused JVP while linearization uses separate destinations so that primal effects run once and
/// tangent effects run with each pushforward. Users must transfer primal coefficients through
/// [`DifferentiationContext::primal_to_tangent`] before combining them with tangents.
/// Structural zero tangents flow between implementations as [`MaybeZero::Zero`]s and stage nothing.
///
/// ## Deriving Differentiable Operation Enums
///
/// `#[derive(Operation)]` generates a [`DifferentiableOperation`] dispatcher when the enum specifies
/// `#[ryft(dispatch(differentiation))]`. This selection enables forward-mode differentiation only. Enums that also
/// need reverse-mode differentiation independently select `transposition`, whose dispatcher reverse mode is built
/// on. It follows the operation derivation's enum-shape inference rules and generates:
///
///   - An `impl DifferentiableOperation<C> for Enum` that is generic over a [`StagingContext`] `C` pinned to the enum's
///     primary type, program constant type, and the enum itself as its operation family. Every variant forwards
///     [`jvp`](Self::jvp) to its payload's own rule, and so payloads without a forward-mode form must still implement
///     the trait with a rule that returns an [`UnsupportedOperation`](ProgramError::UnsupportedOperation).
///   - A `where` clause following the same shape as the generated interpretation and partial-evaluation
///     implementations: a per-variant `Payload: DifferentiableOperation<C>` predicate for every payload which
///     transports each rule's own capability requirements (e.g., `C::Value: Sin` for the sine rule) to the use site,
///     so that the enum does not spell them, plus a `Self: From<Payload>` conversion for every concrete payload (the
///     rules stage ordinary primal-enum operations for both the primal and the tangent side) and the direct
///     `Self: OperationProvider<T, ZeroOperation<T>, Operation = Self>` bound that the nested-region differentiation
///     drivers require. Higher-order payload rules request nested forward-mode and linearization work through their
///     instruction-scoped [`DifferentiationDriver`], whose concrete implementation establishes the finite program-level
///     bounds at its construction site. Output-level semantic queries such as [`Operation::is_zero`] are forwarded by
///     the base operation dispatcher and therefore introduce no additional witness bounds.
///
/// The super-trait is plain [`Operation`] rather than `Operation<Type = C::Type>` because the current trait solver
/// cannot discharge that projection equality at implementation heads whose differentiation context is itself built
/// from `Self`. The equality is instead required per method through `where Self: Operation<Type = C::Type>`, so a
/// payload whose [`Operation::Type`] disagrees with `C::Type` cannot be differentiated in `C`: the requirement is
/// restated by the derived dispatcher's per-payload predicates and by the composite dispatchers, and any mismatched
/// payload is rejected with a type-mismatch error at its use site.
pub trait DifferentiableOperation<C: Context>: Operation {
    /// Applies this operation's capture-free forward-mode rule, mapping the input duals `(xᵢ, ẋᵢ)` to the output duals
    /// `(y, ẏ) = (f(x), Σᵢ (∂f/∂xᵢ)(x) · ẋᵢ)` where `f` is the function this operation computes. The returned vector
    /// must be aligned with this operation's outputs, each element pairing a primal output value with its tangent, both
    /// bound through `context`.
    ///
    /// Rules must be deterministic structural functions of their inputs (i.e., of this operation, the input duals, and
    /// the attached [`Region`]s reachable through `driver`), because the programs derived from them may be retained and
    /// replayed by the per-region transform cache behind [`RegionRef::linearize_shared`]. When the `debug_assertions`
    /// feature is enabled, every cache hit checks this with a rendering-based diagnostic whose fidelity is bounded by
    /// [`Operation::render`] on operation metadata and by [`Display`] on constants.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`DifferentiationContext`] selecting where primal and tangent operations are bound and how primal
    ///     values become available to tangent construction.
    ///   - `driver`: [`DifferentiationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to
    ///     attached [`Region`]s.
    ///   - `inputs`: Input [`DifferentiationDual`]s aligned with this operation's inputs/operands.
    fn jvp<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        Self: Operation<Type = C::Type>;
}

/// Forward-mode rule for a homogeneous member [`Operation`] whose Jacobian-Vector Product (JVP) must execute in the
/// parent [`Context`] enclosing its projection. If this operation has member type `T` and the enclosing context has
/// type `U`, [`jvp_projected_operation`] converts every input from the enclosing value type to the member value type,
/// applies the ordinary member rule in [`ProjectedContext<C, T>`](ProjectedContext), and converts its outputs back. A
/// rule executed that way can work only with values of member type `T`. This trait instead gives the rule the original
/// values and the projected context's parent `C` itself, allowing the rule to use values from other members of `U`
/// when constructing the derivative.
///
/// This distinction matters whenever the primal operation belongs to one projected member but its linearization must
/// retain values belonging to another member of the enclosing type universe. Implementing [`DifferentiableOperation`]
/// directly cannot express that relationship because its [`jvp`](DifferentiableOperation::jvp) function deliberately
/// requires `Self::Type = C::Type`. This trait preserves that same-universe invariant while making member
/// differentiation in the parent universe explicit.
///
/// Implementations bound the parent context by the projection vocabulary they actually use (typically
/// [`ValueProjection<T>`](ValueProjection) for values and constants, [`OperationProjection<T>`](OperationProjection)
/// for the member operation family, and the member and mixed operations they stage) rather than this trait imposing
/// one fixed vocabulary on every implementation. Operation-family dispatchers should use this trait only for projected
/// members whose derivative requires parent-universe values. Members whose inputs, outputs, and derivative all remain
/// within `T` should continue using [`jvp_projected_operation`].
pub trait MemberDifferentiableOperation<C: Context>: Operation<Type: DifferentiableType> {
    /// Applies this projected member's Jacobian-Vector Product (JVP) rule (i.e., its [`DifferentiableOperation::jvp`])
    /// in the parent context enclosing the member's projection, using that parent's own values.
    ///
    /// # Parameters
    ///
    ///   - `context`: Parent [`Context`] through which the rule stages member and mixed operations.
    ///   - `driver`: Instruction-scoped [`DifferentiationDriver`] that exposes any attached [`Region`]s.
    ///   - `inputs`: Parent-universe primal/tangent pairs aligned with this operation's operands.
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

/// [`DifferentiationDual`] flowing through a forward-mode [`DifferentiationContext`]. The function being differentiated
/// operates on [`DifferentiationTracer`]s directly, so each operation the closure performs (e.g., `x + y`, `x.sin()`,
/// etc.) dispatches its [`jvp`](DifferentiableOperation::jvp) rule through [`Context::bind`] on the stamped
/// [`DifferentiationContext`]. This is forward mode's counterpart of [`BatchingTracer`](crate::BatchingTracer):
/// the [`DifferentiationDual`] carries the data the rules operate on, exactly as [`ArrayBatch`](crate::ArrayBatch)
/// does for batching, while this wrapper adds the flowing context so that the value-capability sugar can dispatch.
#[derive(Clone, Parameter)]
pub struct DifferentiationTracer<C: Context, P: DifferentiationPolicy<C> = FusedDifferentiationPolicy> {
    /// [`DifferentiationContext`] this dual flows through.
    context: DifferentiationContext<C, P>,

    /// [`DifferentiationDual`] carrying the primal value and its tangent.
    dual: DifferentiationDual<C::Value>,
}

impl<C: Context, P: DifferentiationPolicy<C>> DifferentiationTracer<C, P> {
    /// Creates a new [`DifferentiationTracer`].
    #[inline]
    pub fn new(dual: DifferentiationDual<C::Value>, context: DifferentiationContext<C, P>) -> Self {
        Self { context, dual }
    }

    /// Returns the [`DifferentiationContext`] this [`DifferentiationTracer`] flows through.
    #[inline]
    pub fn context(&self) -> &DifferentiationContext<C, P> {
        &self.context
    }

    /// Returns the primal value of this [`DifferentiationTracer`].
    #[inline]
    pub fn primal(&self) -> &C::Value {
        self.dual.primal()
    }

    /// Returns the tangent of this [`DifferentiationTracer`].
    #[inline]
    pub fn tangent(&self) -> &MaybeZero<C::Value> {
        self.dual.tangent()
    }

    /// Returns the [`DifferentiationDual`] that this [`DifferentiationTracer`] carries.
    #[inline]
    pub fn dual(&self) -> &DifferentiationDual<C::Value> {
        &self.dual
    }

    /// Consumes this tracer and returns the [`DifferentiationDual`] that it carries.
    #[inline]
    pub fn into_dual(self) -> DifferentiationDual<C::Value> {
        self.dual
    }
}

// A dual compares by its two halves (through the carried values' own `PartialEq`, which is identity-shaped for its
// tracer-valued halves), ignoring the stamped context: consumers such as the scan/while loop-invariance fixed points
// of partial evaluation compare flowing values across replay rounds to detect passthrough, and a dual passes through
// exactly when both its halves do.
impl<C: Context<Value: PartialEq>, P: DifferentiationPolicy<C>> PartialEq for DifferentiationTracer<C, P> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.primal() == other.primal()
            && match (self.tangent(), other.tangent()) {
                (MaybeZero::Value(left), MaybeZero::Value(right)) => left == right,
                (MaybeZero::Zero(left), MaybeZero::Zero(right)) => left == right,
                _ => false,
            }
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> Debug for DifferentiationTracer<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiationTracer").field("dual", &self.dual).finish()
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> Display for DifferentiationTracer<C, P> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.dual())
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> Typed for DifferentiationTracer<C, P> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> std::borrow::Cow<'_, C::Type> {
        self.primal().r#type()
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> Value for DifferentiationTracer<C, P> {
    type DispatchDomain = DifferentiationContext<C, P>;
    type ExecutionDomain = DifferentiationContext<C, P>;

    #[inline]
    fn dispatch_domain(&self) -> DifferentiationContext<C, P> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> DifferentiationContext<C, P> {
        self.context().clone()
    }
}

impl<C: Context, T: Type, P: DifferentiationPolicy<C>> ValueProjection<T> for DifferentiationTracer<C, P>
where
    for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
{
    type Projected = ProjectedValue<T, Self>;
    type ProjectedRef<'v>
        = ProjectedValue<T, &'v Self>
    where
        Self: 'v,
        T: 'v;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        value.into_value()
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        T: 'v,
    {
        Ok(ProjectedValue::new(self, <&T>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(self.r#type().as_ref())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

/// [`DifferentiationContext`] used to linearize a function in `C`. Its [`PartialEvaluationContext`]s execute primal
/// work once and retain tangent work for subsequent pushforward calls, preserving specialization of pure known values.
/// The values flowing through this context are [`LinearizationTracer<C>`].
pub type LinearizationContext<C> =
    DifferentiationContext<PartialEvaluationContext<C>, PartitionedDifferentiationPolicy>;

/// Value type flowing through the closures of the partial-evaluation-backed differentiation entry points
/// (i.e., [`DifferentiationBuilder::linearize`](crate::DifferentiationBuilder::linearize),
/// [`DifferentiationBuilder::vjp`](crate::DifferentiationBuilder::vjp),
/// [`DifferentiationBuilder::gradient`](crate::DifferentiationBuilder::gradient), and their derivatives). It is a
/// [`DifferentiationTracer`] dual over a [`PartialEvaluationContext`] wrapping the context `C` the transform runs in.
/// Its primal half is a *known* partial-evaluation value carrying a concrete value under an eager `C` (so that e.g.,
/// host control flow on primal values works as expected) and its tangent half is *unknown*, accumulating the
/// pushforward program.
pub type LinearizationTracer<C> = DifferentiationTracer<PartialEvaluationContext<C>, PartitionedDifferentiationPolicy>;

/// Forward mode differentiation [`Context`] whose values are [`DifferentiationTracer`]s. Rules receive this
/// concrete context and use its primal and tangent contexts. [`Self::fused`] shares one context for both halves.
/// [`Self::partitioned`] selects [`PartitionedDifferentiationPolicy`], whose sibling contexts execute primal work once
/// and retain tangent effects for each pushforward while specializing pure coefficients. Both policies use the same
/// rule definitions and value family. [`Self::new`] supports custom policies, and [`Self::project`] preserves the
/// current policy while exposing a member family.
///
/// Structural zero tangents remain symbolic. When every input tangent is zero, binding can skip derivative work if
/// the operation cannot introduce reference state and its output geometry can be reconstructed from the primals.
/// Otherwise, the ordinary rule runs, so a zero initialized tangent reference still receives its own allocation.
#[derive(Clone)]
pub struct DifferentiationContext<C: Context, P: DifferentiationPolicy<C> = FusedDifferentiationPolicy> {
    /// [`Context`] that computes primal values and also computes tangent values when using the
    /// [`FusedDifferentiationPolicy`] (i.e., when [`tangent`](Self::tangent) is [`None`]).
    primal: C,

    /// Optional tangent [`Context`].
    tangent: Option<C>,

    /// Phantom marker pinning the [`DifferentiationPolicy`] type.
    policy: PhantomData<P>,
}

impl<C: Context> DifferentiationContext<C> {
    /// Creates a new [`DifferentiationContext`] over the provided primal [`Context`]
    /// that uses the [`FusedDifferentiationPolicy`].
    #[inline]
    pub fn fused(primal: C) -> Self {
        Self::new(primal)
    }
}

impl<C: Context> DifferentiationContext<C, PartitionedDifferentiationPolicy>
where
    PartitionedDifferentiationPolicy: DifferentiationPolicy<C>,
{
    /// Creates a new [`DifferentiationContext`] over the provided primal [`PartialEvaluationContext`] that uses the
    /// [`PartitionedDifferentiationPolicy`] and thus separates primal work from tangent work for repeated pushforward
    /// calls. The primal context is a [`PartialEvaluationContext`] and its
    /// [deferred sibling](PartialEvaluationContext::deferred_sibling) retains tangent effects for each call.
    #[inline]
    pub fn partitioned(primal: C) -> Self {
        Self::new(primal)
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> DifferentiationContext<C, P> {
    /// Creates a new [`DifferentiationContext`] over the provided primal [`Context`] that uses policy `P`
    /// to choose its tangent context.
    #[inline]
    pub fn new(primal: C) -> Self {
        let tangent = P::tangent_context(&primal);
        Self { primal, tangent, policy: PhantomData }
    }

    /// Returns the [`Context`] that computes primal values and also computes tangent values when using the
    /// [`FusedDifferentiationPolicy`] (i.e., when [`tangent`](Self::tangent) is [`None`]).
    #[inline]
    pub fn primal(&self) -> &C {
        &self.primal
    }

    /// Returns the [`Context`] that computes tangent values. For certain policies (e.g., for
    /// [`FusedDifferentiationPolicy`]), this will be the same as the [primal context](Self::primal).
    #[inline]
    pub fn tangent(&self) -> &C {
        self.tangent.as_ref().unwrap_or(&self.primal)
    }

    /// Returns a projected view of this [`DifferentiationContext`] for a member type, preserving its primal/tangent
    /// separation and value transfer policy. The projected contexts share the existing contexts' state (i.e.,
    /// projection does not create a new tangent computation or discard values already transferred to it).
    ///
    /// When using the [`FusedDifferentiationPolicy`], [`primal`](Self::primal) and [`tangent`](Self::tangent) still
    /// return the same projected context. When using [`PartitionedDifferentiationPolicy`], rules append to the original
    /// primal and tangent programs through their projected contexts.
    pub fn project<T: Type>(&self) -> DifferentiationContext<ProjectedContext<C, T>, ProjectedDifferentiationPolicy<P>>
    where
        C::Value: ValueProjection<T, Projected: Value<Type = T>>,
        C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
        C::Operation: OperationProjection<T>,
    {
        // Reuse the existing contexts instead of invoking the policy's tangent context constructor again.
        DifferentiationContext {
            primal: ProjectedContext::new(self.primal().clone()),
            tangent: self.tangent.as_ref().map(|tangent| ProjectedContext::new(tangent.clone())),
            policy: PhantomData,
        }
    }

    /// Transfers the provided primal value to the tangent context without forcing residual materialization.
    /// The transfer preserves reference identity (i.e., it does not snapshot references).
    #[inline]
    pub fn primal_to_tangent(&self, value: C::Value) -> Result<C::Value, DifferentiationError> {
        P::primal_to_tangent(self.tangent(), value)
    }

    /// Transfers each dual's primal value to the tangent context while retaining its existing tangent value. Use this
    /// when a rule needs primal coefficients alongside tangents, such as the input shapes used to construct a tangent
    /// reshape operation. As with [`Self::primal_to_tangent`], the transfer preserves reference identity and does not
    /// force residual materialization. The returned [`DifferentiationDual`]s are operands for tangent work; their
    /// primals no longer belong to the original primal context when the policy uses separate contexts.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Duals whose primals belong to [`Self::primal`] and whose live tangents
    ///     belong to [`Self::tangent`].
    ///
    /// # Errors
    ///
    /// Propagates transfer errors and errors validating the resulting primal/tangent type pairs.
    #[inline]
    pub fn dual_primal_to_tangent(
        &self,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C::Type: DifferentiableType,
    {
        inputs
            .iter()
            .map(|input| {
                DifferentiationDual::new(self.primal_to_tangent(input.primal().clone())?, input.tangent().clone())
            })
            .collect()
    }

    /// Binds an operation to the provided dual inputs using the same dispatch as [`Context::bind`]. Ordinary binding
    /// unwraps tracers before calling this function. Recursive differentiation instead passes its duals directly.
    /// Keeping the structural zero checks here ensures that both paths run reference and runtime geometry rules when
    /// needed, while evaluating eligible primal operations only once.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation whose primal and tangent results are required.
    ///   - `driver`: Complete ordered regions attached to this application.
    ///   - `inputs`: Operand duals, in operation-input order.
    fn bind_duals<D: BindingRegionDriver<C::Constant, C::Operation>>(
        &self,
        operation: &C::Operation,
        driver: D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C::Type: DifferentiableType,
        C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
            + DifferentiableOperation<C>
            + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
            + ResidualZeroProvider<C::Type, Operation = C::Operation>,
    {
        operation.validate_region_count(driver.region_count())?;

        // All-zero fast path mirroring `Program::jvp`. When an operation consumes at least one input and every input
        // tangent is a structural zero, skip its rule only when each output tangent can later be materialized without
        // runtime identity operands. Zero-input operations remain excluded so their dedicated rules keep handling
        // primal synthesis and tangent typing. Reference operations are differentiated by their own rules: a reference
        // input whose tangent is a symbolic zero is plumbing, every rule decides what plumbing means for it, and a
        // reference output has no structural zero because its rule must allocate the tangent reference.
        let zero_input_tangents = !inputs.is_empty() && inputs.iter().all(|dual| dual.tangent().is_zero());

        // Region-carrying operations retain structural zero tangents because the transform boundary captures their
        // runtime geometry from the staged primal outputs, but only while no reference is involved. A reference operand
        // or an attached region that allocates or accesses references (e.g., a `condition` whose branches allocate and
        // return a reference) must reach the rule so that every reference result carries its tangent reference.
        // Region-free operations can instead inspect their outputs without reproducing the primal's region-identity
        // instantiation.
        let reusable_zero_outputs = if !zero_input_tangents {
            None
        } else if !operation.region_slots().is_empty() {
            let touches_references = inputs.iter().any(|input| input.primal().r#type().is_reference())
                || driver.regions().any(|region| region.contains_references_in_closure());
            (!touches_references).then_some(Vec::new())
        } else {
            let input_types = inputs.iter().map(|input| input.primal().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
            let mut reusable_zero_outputs = Vec::new();
            let mut can_materialize = true;
            for (output_index, output_type) in output_types.iter().enumerate() {
                let tangent_type = output_type.tangent()?;
                if operation.is_zero(output_index) && output_type == &tangent_type {
                    reusable_zero_outputs.push(output_index);
                } else if !can_materialize_zero_tangent_from_type(output_type, &tangent_type) {
                    can_materialize = false;
                    break;
                }
            }
            can_materialize.then_some(reusable_zero_outputs)
        };

        let outputs = if let Some(reusable_zero_outputs) = reusable_zero_outputs {
            let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
            self.primal
                .bind(operation.clone(), driver, &primal_inputs)?
                .into_iter()
                .enumerate()
                .map(|(output_index, primal)| {
                    // When the primal operation is itself known to produce zero and its output already has the required
                    // tangent type, the primal value is the canonical materialized tangent. Reusing it avoids inventing
                    // a nullary dynamic zero when a staged tangent is later materialized, exactly like the fused replay
                    // in `RegionRef::jvp`.
                    if self.tangent.is_none() && reusable_zero_outputs.contains(&output_index) {
                        DifferentiationDual::new(primal.clone(), self.primal_to_tangent(primal)?)
                    } else {
                        DifferentiationDual::new_with_zero_tangent(primal)
                    }
                })
                .collect::<Result<Vec<_>, DifferentiationError>>()?
        } else {
            // Borrow the complete region driver directly, preserving operation-defined ordering without collecting
            // it into temporary storage.
            let differentiation_driver = RecursiveDifferentiationDriver { driver: &driver };
            operation.jvp(self, &differentiation_driver, inputs)?
        };

        Ok(outputs)
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> Domain for DifferentiationContext<C, P> {
    type Type = C::Type;
    type Value = DifferentiationTracer<C, P>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context, P: DifferentiationPolicy<C>> Context for DifferentiationContext<C, P>
where
    C::Type: DifferentiableType,
    C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type, Operation = C::Operation>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<DifferentiationTracer<C, P>, ProgramError> {
        // Constants are independent of every differentiation input and so their tangents are structural zeros.
        let dual = DifferentiationDual::new_with_zero_tangent(self.primal.lift(constant)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }

    fn bind<O: Into<C::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[DifferentiationTracer<C, P>],
    ) -> Result<Vec<DifferentiationTracer<C, P>>, ProgramError> {
        let operation = operation.into();

        // Unwrap the input tracers into context-free duals, run the rule against those, and rewrap the produced duals
        // with this context, mirroring how `BatchingContext::bind` unwraps to `ArrayBatch`es and rewraps.
        let input_duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();
        let output_duals = self.bind_duals(&operation, driver, &input_duals)?;

        // Stamp this context onto every value handed back to the caller so its capability sugar dispatches through this
        // forward-mode context (the `jvp` rules build their outputs context-free via `DifferentiationDual::new`).
        Ok(output_duals.into_iter().map(|dual| DifferentiationTracer::new(dual, self.clone())).collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // A forward mode differentiation context is eager exactly when the primal context carrying its duals' values
        // is (i.e., never over a staging primal context and always over an eager one).
        self.primal.is_eager()
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // Forward mode differentiation uses the primal context's provenance state for rewritten primitive work.
        self.primal.provenance()
    }

    #[inline]
    fn resolve(&self, value: &DifferentiationTracer<C, P>) -> ValueResolution<C::Constant> {
        // A value is constant in the differentiated computation only when its primal resolves in the primal context
        // and its tangent is structurally zero. A live tangent makes the dual input-dependent even for a constant
        // primal.
        if value.tangent().is_zero() { self.primal.resolve(value.primal()) } else { ValueResolution::Opaque }
    }

    #[inline]
    fn reference_identity(
        &self,
        value: &DifferentiationTracer<C, P>,
    ) -> Result<Option<ReferenceIdentity>, ProgramError> {
        self.primal.reference_identity(value.primal())
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.primal.invoke_with_provenance_origin(origin, function)
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.primal.invoke_with_provenance_scope(scope, function)
    }
}

impl<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>> RegionRef<'_, V, O> {
    /// Returns a mask with one boolean value per primal output indicating whether the differentiated program includes
    /// its tangent. Numeric outputs retain tangent slots even for zero derivatives, unless their differential space is
    /// zero. Reference outputs retain tangent slots when rooted in a selected input or a local allocation; references
    /// rooted in unselected inputs or captures do not.
    ///
    /// The mask follows primal output order regardless of the order of `input_indices`. Input selection order
    /// determines tangent input order in the differentiated program, but does not reorder its tangent outputs.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Indices of inputs with respect to which the program is differentiated. Selected inputs
    ///     with zero differential spaces contribute no tangent slot.
    ///
    /// # Errors
    ///
    /// Returns an invalid argument error for duplicate or out-of-range indices, including duplicate zero space
    /// inputs. Propagates reference analysis errors and errors computing selected input or output tangent types.
    pub fn tangent_output_mask(&self, input_indices: &[usize]) -> Result<Vec<bool>, DifferentiationError> {
        // Only membership matters for reference outputs, and so we sort the validated selection so that each root
        // lookup uses binary search without allocating another mask covering every region input.
        let mut arguments = JvpAndLinearizationTransformArguments::new(*self, input_indices)?;
        arguments.input_indices.sort_unstable();
        let has_reference_outputs =
            self.output_ids().iter().any(|output| self.atoms()[output.index()].r#type().is_reference());
        let analysis = has_reference_outputs
            .then(|| self.reference_analysis_with_configuration(None, true, &[]))
            .transpose()
            .map_err(ProgramError::from)?;
        self.output_ids()
            .iter()
            .map(|output| &self.atoms()[output.index()])
            .enumerate()
            .map(|(index, output)| {
                if output.r#type().is_reference() {
                    Ok(match analysis.as_ref().unwrap().output_roots()[index] {
                        Some(ReferenceRoot::RegionInput { region, input_index }) if region == self.id() => {
                            arguments.input_indices.binary_search(&input_index).is_ok()
                        }
                        Some(ReferenceRoot::Allocation { .. }) => true,
                        _ => false,
                    })
                } else {
                    Ok(!output.r#type().tangent()?.is_zero_space())
                }
            })
            .collect()
    }
}

impl<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>> RegionRef<'_, V, O>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>,
{
    /// Builds the fused Jacobian-Vector Product (JVP) [`Program`] of this borrowed [`Region`] with respect to
    /// `input_indices`. The program receives all primal inputs in source order, followed by tangent inputs in the
    /// requested index order. For example, selecting `[2, 0]` gives inputs `[x₀, x₁, x₂, ẋ₂, ẋ₀]`. Primal outputs
    /// retain source order, followed by live tangent outputs in source output order.
    ///
    /// Unselected inputs carry structural [`MaybeZero::Zero`] tangents. For a numeric input this is the usual symbolic
    /// zero. For a reference input it means no tangent reference is supplied. Reads then have zero tangents, and writes
    /// of live tangents are rejected by the reference forward mode differentiation rules. A selected reference input
    /// instead receives a fresh tangent input of its reference tangent type.
    ///
    /// Selected inputs with zero differential spaces contribute no tangent slot. They are removed from the ordered
    /// selection before deriving the program or constructing its cache key. Duplicate indices are still rejected,
    /// including duplicates of zero space inputs. An empty selection supplies no tangent inputs.
    ///
    /// Non-zero space ordinary outputs retain tangent outputs even when they are zero. Reference outputs rooted in
    /// selected inputs or local allocations retain tangent reference outputs; those rooted in unselected inputs or
    /// captures do not. A reference output that is a derived view is rejected.
    ///
    /// Before executing the returned program, callers must ensure that reference arguments satisfy the
    /// [allocation independence requirements](Linearization#reference-arguments). The generated program does not
    /// perform these runtime checks. Input selection does not establish allocation independence, and raw execution does
    /// not validate it. Internal aliases retain their canonical roots; the contract concerns distinct caller bindings.
    ///
    /// This function builds an owned program without consulting or populating the transform cache. Use
    /// [`Self::jvp_shared`] to retain and reuse a shared artifact. [`Program::jvp`] selects all inputs for callers
    /// that do not need an explicit selection.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Unique region input indices in tangent input order. Zero differential spaces are omitted.
    ///     An empty slice selects no inputs.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] for duplicate or out-of-range indices and
    /// [`ProgramError::UnsupportedOperation`] for reference outputs that are derived views. Propagates errors from
    /// tangent-type derivation and the replayed forward-mode rules.
    #[inline]
    pub fn jvp(&self, input_indices: &[usize]) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.jvp_impl(&JvpAndLinearizationTransformArguments::new(*self, input_indices)?)
    }

    /// Builds the fused Jacobian-Vector Product (JVP) program for `input_indices` through this region's retained
    /// transform cache, returning a shared program. Selection, validation, and reference boundary requirements are the
    /// same as for [`jvp`](Self::jvp).
    ///
    /// Content-preserving copies of a sealed region share an artifact when their ordered selections agree after
    /// omitting zero differential spaces. Reordering live inputs changes the tangent-input boundary and produces a
    /// distinct artifact. This avoids differentiating shared callees repeatedly and preserves the [`Arc`] identity
    /// used to intern repeated attachments of a derived program. Use [`jvp`](Self::jvp) for an owned, uncached result.
    ///
    /// Recursive requests for a transform currently in flight on this thread use uncached construction, as they do
    /// through [`jvp`](Self::jvp). Cache lookup and publication remain managed by [`transform`](Self::transform).
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Unique input indices in tangent input order. Zero differential spaces are omitted.
    ///     An empty slice selects no inputs.
    pub fn jvp_shared(
        &self,
        input_indices: &[usize],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        let arguments = JvpAndLinearizationTransformArguments::new(*self, input_indices)?;
        let artifact = (*self).transform::<JvpTransform, _, DifferentiationError>(arguments, |region, arguments| {
            Ok(TransformArtifact::new(vec![Arc::new(region.jvp_impl(arguments)?)], ()))
        })?;
        let (programs, ()) = artifact.into_parts();
        let mut programs = programs.into_iter();
        let program = programs.next().unwrap();
        assert!(programs.next().is_none(), "fused JVP transform retained more than one program");
        Ok(program)
    }

    /// Builds the fused Jacobian-Vector Product (JVP) program from validated, normalized `arguments`. Both
    /// [`jvp`](Self::jvp) and [`jvp_shared`](Self::jvp_shared) use this implementation without repeating input
    /// selection validation.
    fn jvp_impl(
        &self,
        arguments: &JvpAndLinearizationTransformArguments,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.validate_reference_output_views()?;
        let primal_input_count = self.input_ids().len();
        let tangent_input_count = arguments.input_indices.len();

        // Hold a standalone `Rc` clone of the context's builder, and move the context itself into the block below, so
        // that scoping every tracer (and the context) inside that block makes the `Rc::try_unwrap` at the end a real
        // ownership check rather than depending on manual drops. Only raw output atom IDs escape the block.
        let context = TracingContext::<V, O>::new();
        let builder = context.builder().clone();
        let output_atoms = {
            // Rebinding a non-`Copy` value is a move. The context enters this block's scope and is dropped at its end
            // together with every tracer created from it below, leaving the standalone `builder` handle above as the
            // sole owner of the shared builder `Rc` for the `Rc::try_unwrap` that follows this block.
            let context = context;
            let differentiation_context = DifferentiationContext::fused(context.clone());

            // Track the primal tracer and symbolic tangent for each source atom. Tangents of atoms not connected to an
            // input tangent (i.e., constants and dead inputs) are derived lazily as structural zeros typed with the
            // atom's tangent boundary type.
            let mut primals: Vec<Option<Tracer<TracingContext<V, O>>>> = vec![None; self.atoms().len()];
            let mut tangents: Vec<Option<MaybeZero<Tracer<TracingContext<V, O>>>>> = vec![None; self.atoms().len()];

            // Preserve primal input order, then allocate tangent inputs in the requested selection order. The
            // normalized selection omits zero differential spaces. Unselected inputs retain structural zeros,
            // including reference inputs for which no tangent reference is supplied.
            for input_id in self.input_ids().iter().copied() {
                let primal_type = self.atoms()[input_id.index()].r#type().into_owned();
                let tangent_type = primal_type.tangent()?;
                primals[input_id.index()] = Some(context.input(primal_type));
                tangents[input_id.index()] = Some(MaybeZero::Zero(tangent_type));
            }
            for &index in &arguments.input_indices {
                let input_id = self.input_ids()[index];
                let tangent_type = self.atoms()[input_id.index()].r#type().tangent()?;
                tangents[input_id.index()] = Some(MaybeZero::Value(context.input(tangent_type)));
            }

            // Constants are lifted into the builder as primal constants. Their tangents are derived lazily as
            // structural zeros typed with the atom's tangent boundary type. The call is disambiguated to
            // the staging method because the `Constant` capability trait also provides a `constant` method.
            for (atom_index, atom) in self.atoms().iter().enumerate() {
                if let Atom::Constant(value) = atom {
                    primals[atom_index] = Some(StagingContext::constant(&context, value.clone()));
                }
            }

            // Replay each primal instruction in JVP form, staging both the primal result and the tangent operations
            // into the shared builder.
            let region_mappings = RegionReplayMappings::new();
            for instruction in self.instructions() {
                let input_duals = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input_atom| {
                        let primal = primals[input_atom.index()]
                            .clone()
                            .ok_or(ProgramError::UnboundAtomId { id: input_atom })?;
                        // Atoms not connected to an input tangent (i.e., constants and dead inputs)
                        // take a structural zero typed with the atom's tangent boundary type.
                        let tangent = match &tangents[input_atom.index()] {
                            Some(tangent) => tangent.clone(),
                            None => MaybeZero::Zero(primal.r#type().tangent()?),
                        };
                        Ok(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new(primal, tangent)?)
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // All-zero fast path: skip the operation's rule only when every input tangent is structural zero and
                // every output zero tangent can later be materialized without runtime identity operands (or by reusing
                // a zero-producing primal). Zero-input operations remain excluded so their dedicated rules keep
                // handling primal synthesis and tangent typing. Dynamic one already relies on this routing to stage
                // an explicit dynamic-zero tangent. Other dynamic output rules must retain any runtime extents needed
                // to materialize their structural zero tangents.
                let all_input_tangents_are_zero =
                    !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero());
                let can_materialize_output_tangents_without_rules = || -> Result<bool, DifferentiationError> {
                    for (output_index, output_atom) in instruction.outputs().iter().copied().enumerate() {
                        let output_type = self.atoms()[output_atom.index()].r#type();
                        let tangent_type = output_type.tangent()?;
                        let output_type = output_type.as_ref();
                        let can_materialize = can_materialize_zero_tangent_from_type(output_type, &tangent_type)
                            || (instruction.operation().is_zero(output_index) && output_type == &tangent_type);
                        if !can_materialize {
                            return Ok(false);
                        }
                    }
                    Ok(true)
                };
                let driver = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;

                // Both dispatch paths run inside the replayed instruction's recorded origin so that everything they
                // stage records where it came from: the one-to-one fast path preserves the source provenance exactly,
                // and a rule that stages several instructions attaches it to each of them.
                let use_zero_tangent_fast_path =
                    all_input_tangents_are_zero && can_materialize_output_tangents_without_rules()?;
                let output_duals = context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    if use_zero_tangent_fast_path {
                        let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                        context
                            .stage_operation(instruction.operation().clone(), driver, primal_inputs.as_slice())?
                            .into_iter()
                            .enumerate()
                            .map(|(output_index, primal)| {
                                // When the primal instruction is itself known to produce zero and its output already
                                // has the required tangent type, the primal Single Static Assignment (SSA) value is
                                // the canonical materialized tangent. Reusing it preserves source-relative geometry
                                // such as explicit shaped-constructor extents and avoids inventing a nullary dynamic
                                // zero at the fused Jacobian-Vector Product (JVP) boundary.
                                let primal_type = primal.r#type();
                                let tangent_type = primal_type.tangent()?;
                                if instruction.operation().is_zero(output_index)
                                    && primal_type.as_ref() == &tangent_type
                                {
                                    DifferentiationDual::new(primal.clone(), primal)
                                } else {
                                    DifferentiationDual::new_with_zero_tangent(primal)
                                }
                            })
                            .collect::<Result<Vec<_>, DifferentiationError>>()
                    } else {
                        let differentiation_driver = RecursiveDifferentiationDriver { driver: &driver };
                        instruction.operation().jvp(
                            &differentiation_context,
                            &differentiation_driver,
                            input_duals.as_slice(),
                        )
                    }
                })?;

                check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
                for (output_atom, dual) in instruction.outputs().iter().copied().zip(output_duals) {
                    let (primal, tangent) = dual.into_parts();
                    primals[output_atom.index()] = Some(primal);
                    tangents[output_atom.index()] = Some(tangent);
                }
            }

            // Collect the primal outputs followed by the live tangent outputs. Zero differential spaces remain
            // structural and therefore contribute no executable result slot.
            let primal_output_atoms = self
                .output_ids()
                .iter()
                .copied()
                .map(|output_atom| {
                    primals[output_atom.index()]
                        .as_ref()
                        .map(|primal| primal.atom_id())
                        .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                })
                .collect::<Result<Vec<_>, _>>()?;
            let tangent_output_atoms = self
                .output_ids()
                .iter()
                .copied()
                .map(|output_atom| {
                    // Atoms not connected to an input tangent (i.e., constants and dead inputs)
                    // take a structural zero typed with the atom's tangent boundary type.
                    let primal =
                        primals[output_atom.index()].as_ref().ok_or(ProgramError::UnboundAtomId { id: output_atom })?;
                    let tangent = match &tangents[output_atom.index()] {
                        Some(tangent) => tangent.clone(),
                        None => MaybeZero::Zero(primal.r#type().tangent()?),
                    };
                    if tangent.r#type().is_zero_space() || (primal.r#type().is_reference() && tangent.is_zero()) {
                        Ok(None)
                    } else {
                        // A structural zero tangent has to become a real boundary value here. Its type alone cannot
                        // construct it when it references runtime identities, so the operation family's residual
                        // protocol reads the missing extents from the output's own primal, which is a live value of
                        // the same shape. An identity-free tangent type declares no residuals and therefore stages
                        // exactly the nullary zero it staged before.
                        let tangent = match tangent {
                            MaybeZero::Value(tangent) => tangent,
                            MaybeZero::Zero(tangent_type) => {
                                let residuals = capture_and_validate_zero_residual_values(
                                    &context,
                                    primal,
                                    &tangent_type,
                                    "jvp output tangent",
                                )?;
                                let (operation, operands) =
                                    O::zero_operation_with_residuals(tangent_type, residuals.as_slice())?;
                                let mut outputs = context.stage_operation(operation, Vec::new(), &operands)?;
                                check_count!("output", outputs, 1, ProgramError);
                                outputs.remove(0)
                            }
                        };
                        Ok(Some(tangent.atom_id()?))
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?
                .into_iter()
                .flatten()
                .collect::<Vec<_>>();

            let mut output_atoms = primal_output_atoms;
            output_atoms.extend(tangent_output_atoms);
            output_atoms
        };

        // All tracing handles are dropped here, so the builder can be recovered and finalized.
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let input_count = primal_input_count + tangent_input_count;
        let output_count = output_atoms.len();
        builder
            .build::<Vec<V>, Vec<V>>(output_atoms, vec![Placeholder; input_count], vec![Placeholder; output_count])
            .map_err(DifferentiationError::from)
    }

    /// Linearizes this borrowed [`Region`] with respect to `input_indices`. Selection, validation, and reference output
    /// semantics follow [`jvp`](Self::jvp). The tangent program consumes selected input tangents in the requested
    /// order, omitting zero differential spaces, followed by residuals. Primal inputs and outputs retain source order.
    /// Refer to the documentation of [`Linearization`] for information on the linearization algorithm.
    ///
    /// Reference operations linearize directly. The primal program keeps the primal accesses (its known side is the
    /// forward pass at the linearization point, so it allocates, reads, and mutates the primal references exactly as
    /// forward mode does), while the tangent program keeps the tangent accesses over the tangent references, which are
    /// unknown tangent inputs. The placement preserves the effect ordering described in the documentation of
    /// [`PartialEvaluationContext`], and the resulting tangent input layout is described in the documentation of
    /// [`Linearization::tangent`]. Executing either returned program requires the unchecked
    /// [reference argument contract](Linearization#reference-arguments), including for inactive
    /// primal reference inputs and captured reference bindings.
    ///
    /// This function builds owned programs without consulting or populating the transform cache. Use
    /// [`linearize_shared`](Self::linearize_shared) to retain shared artifacts, or [`Program::linearize`]
    /// to select every input.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Unique region input indices in tangent-input order. Zero differential spaces are omitted.
    ///     An empty slice selects no inputs.
    #[inline]
    pub fn linearize(&self, input_indices: &[usize]) -> Result<Linearization<V, O>, DifferentiationError> {
        self.linearize_impl(&JvpAndLinearizationTransformArguments::new(*self, input_indices)?)
    }

    /// Linearizes this region for `input_indices` through its retained transform cache, returning a [`Linearization`]
    /// whose primal and tangent programs share the cached handles. Selection, validation, and reference boundary
    /// requirements are the same as for [`linearize`](Self::linearize).
    ///
    /// Content preserving copies of a sealed region share the derived programs when their ordered selections agree
    /// after omitting zero differential spaces, just as for [`jvp_shared`](Self::jvp_shared). This avoids repeatedly
    /// linearizing shared callees and preserves the program identities used to intern repeated attachments. Use
    /// [`linearize`](Self::linearize) for freshly constructed programs without cache lookup or retention.
    ///
    /// Recursive requests for a linearization currently in flight on this thread use uncached construction, as they
    /// do through [`linearize`](Self::linearize). Cache lookup and publication remain managed by
    /// [`transform`](Self::transform).
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Unique input indices in tangent input order. Zero differential spaces are omitted.
    ///     An empty slice selects no inputs.
    pub fn linearize_shared(&self, input_indices: &[usize]) -> Result<Linearization<V, O>, DifferentiationError> {
        let arguments = JvpAndLinearizationTransformArguments::new(*self, input_indices)?;
        let artifact =
            (*self).transform::<LinearizationTransform, _, DifferentiationError>(arguments, |region, arguments| {
                let (primal, tangent, residual_count) = region.linearize_impl(arguments)?.into_parts();
                Ok(TransformArtifact::new(vec![primal, tangent], residual_count))
            })?;
        let (programs, residual_count) = artifact.into_parts();
        let mut programs = programs.into_iter();
        let primal = programs.next().unwrap();
        let tangent = programs.next().unwrap();
        assert!(programs.next().is_none(), "linearization transform retained more than two programs");
        Ok(Linearization { primal, tangent, residual_count })
    }

    /// Linearizes this region from validated, normalized `arguments`. Both [`linearize`](Self::linearize) and
    /// [`linearize_shared`](Self::linearize_shared) use this implementation without repeating input selection
    /// validation.
    fn linearize_impl(
        &self,
        arguments: &JvpAndLinearizationTransformArguments,
    ) -> Result<Linearization<V, O>, DifferentiationError> {
        self.validate_reference_output_views()?;
        let primal_input_count = self.input_ids().len();
        let tangent_input_count = arguments.input_indices.len();

        // Keep one standalone handle to the primal builder. Every tracer and context clone is scoped below and must
        // be gone before this handle can be unwrapped at the trace boundary.
        let primal_context = TracingContext::<V, O>::new();
        let primal_builder = primal_context.builder().clone();
        let primal_evaluation_context = PartialEvaluationContext::new(primal_context.clone());
        let differentiation_context = DifferentiationContext::partitioned(primal_evaluation_context.clone());
        let evaluation_context = differentiation_context.tangent().clone();

        // Allocate unknown tangents in selection order before constructing input duals in primal order. Both the
        // residual builder input order and its recorded unknown ordinals must follow the tangent calling convention.
        let mut input_tangents = vec![None; primal_input_count];
        for (tangent_index, &input_index) in arguments.input_indices.iter().enumerate() {
            let tangent_type = self.atoms()[self.input_ids()[input_index].index()].r#type().tangent()?;
            let tangent = evaluation_context.unknown_input(tangent_type, tangent_index);
            input_tangents[input_index] = Some(PartialTracer::new(evaluation_context.clone(), tangent));
        }

        let mut primal_input_atoms = Vec::with_capacity(primal_input_count);
        let input_duals = self
            .input_ids()
            .iter()
            .copied()
            .zip(input_tangents)
            .map(|(input_atom, tangent)| {
                let primal_type = self.atoms()[input_atom.index()].r#type().into_owned();
                let tangent_type = primal_type.tangent()?;
                let primal = primal_context.input(primal_type);
                primal_input_atoms.push(primal.atom_id()?);
                let tangent = match tangent {
                    Some(tangent) => MaybeZero::Value(tangent),
                    None => MaybeZero::Zero(tangent_type),
                };
                Ok::<_, ProgramError>(DifferentiationTracer::new(
                    DifferentiationDual::new(
                        PartialTracer::new(
                            primal_evaluation_context.clone(),
                            PartialEvaluationValue::known_input(primal),
                        ),
                        tangent,
                    )?,
                    differentiation_context.clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Replay the source program once. Constants lift as known values with structural-zero tangents. Instruction
        // binds dispatch through differentiation-over-partial-evaluation, including its all-structural-zero fast path.
        let region_mappings = RegionReplayMappings::new();
        let output_duals = self.interpret_with(
            input_duals,
            |_, constant| differentiation_context.lift(constant.clone()),
            |instruction, inputs| {
                // Bind inside the source instruction's recorded origin so linearization propagates provenance like
                // every other interpretation/replay boundary.
                let regions = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;
                differentiation_context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    differentiation_context.bind(instruction.operation().clone(), regions, inputs)
                })
            },
        )?;

        // Split the direct output duals. Primal halves must be known tracers in the primal builder. Structural-zero
        // tangent halves become residualized typed zeros so the tangent program preserves the source output arity. A
        // value tangent that folded to known is malformed: rules must preserve input-independent zeros structurally,
        // and accepting any other known value would turn the tangent program into an affine map. Do not bind through
        // `Zero` here as partial evaluation could classify that value as known and remove it from the tangent boundary.
        // Forced residualization keeps a structural zero as an unknown Single Static Assignment (SSA) output,
        // preserving the linear program's output arity without admitting an affine constant.
        let mut primal_output_atoms = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for dual in output_duals {
            let (primal, tangent) = dual.into_dual().into_parts();
            let primal = primal.into_value()?;
            let primal = match primal.value() {
                PartialValue::Known(value) => value,
                PartialValue::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced an unknown primal output but primal work depends only on the known \
                         primal inputs"
                            .to_string(),
                    )
                    .into());
                }
            };
            if !Rc::ptr_eq(primal.builder(), &primal_builder) {
                return Err(ProgramError::MalformedProgram(
                    "linearization produced a primal output owned by a foreign trace".to_string(),
                )
                .into());
            }
            primal_output_atoms.push(primal.atom_id()?);
            if tangent.r#type().is_zero_space() {
                continue;
            }

            // Inactive reference carries remain in the primal boundary and have no tangent allocation to return.
            if primal.r#type().is_reference() && tangent.is_zero() {
                continue;
            }

            let tangent = match tangent {
                MaybeZero::Value(tracer) => {
                    let value = tracer.into_value()?;
                    match value.value() {
                        PartialValue::Unknown(_) => value,
                        PartialValue::Known(_) => {
                            return Err(ProgramError::MalformedProgram(
                                "linearization produced a known tangent output; differentiation rules must represent \
                                 input-independent zero tangents structurally"
                                    .to_string(),
                            )
                            .into());
                        }
                    }
                }
                MaybeZero::Zero(r#type) => {
                    let residuals = capture_and_validate_zero_residual_atoms(
                        &mut primal_builder.borrow_mut(),
                        primal.atom_id()?,
                        &r#type,
                        "linearization output tangent",
                    )?;
                    let residuals = residuals
                        .into_iter()
                        .map(|residual| {
                            let residual_type = primal_builder
                                .borrow()
                                .atoms()
                                .get(residual.index())
                                .ok_or(ProgramError::UnboundAtomId { id: residual })?
                                .r#type()
                                .into_owned();
                            Ok(Tracer::new(primal_context.clone(), TracerState::Live(residual), residual_type))
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()?;
                    residualize_zero_from_residual_values(&evaluation_context, r#type, residuals)?
                }
            };
            tangent_outputs.push(tangent);
        }

        // Drop the differentiation context before finalizing partial evaluation as its primal context clone would
        // otherwise keep the residual builder alive and correctly trigger the escaped builder guard.
        drop(differentiation_context);

        let primal_evaluation = primal_evaluation_context.into_evaluation(Vec::new())?;
        if !primal_evaluation.program.instructions().is_empty() {
            return Err(ProgramError::MalformedProgram(
                "linearization deferred an operation bound to the primal destination".to_string(),
            )
            .into());
        }
        drop(primal_evaluation);

        let tangent_output_count = tangent_outputs.len();
        let evaluation = evaluation_context.into_evaluation(tangent_outputs)?;
        if evaluation.outputs.len() != tangent_output_count
            || evaluation.outputs.iter().enumerate().any(
                |(index, output)| !matches!(output, PartialEvaluationOutput::Unknown(ordinal) if *ordinal == index),
            )
        {
            return Err(ProgramError::MalformedProgram(
                "linearization produced a tangent output that did not residualize at its canonical output position"
                    .to_string(),
            )
            .into());
        }
        let mut tangent_program = evaluation.program;

        // Inputs are created as all tangent unknowns first, followed by lazily materialized residual feeders. The
        // residual program simplifier preserves public inputs, so this metadata must align one-for-one with its input
        // atoms. Collect the residual feeder atom IDs in precisely that trailing order for the primal boundary.
        if evaluation.inputs.len() != tangent_program.input_ids().len() {
            return Err(ProgramError::MalformedProgram(
                "linearization produced tangent input metadata that does not match its tangent program".to_string(),
            )
            .into());
        }

        let mut residual_output_atoms = Vec::with_capacity(evaluation.inputs.len().saturating_sub(tangent_input_count));
        for (index, input) in evaluation.inputs.into_iter().enumerate() {
            match input {
                PartialEvaluationInput::Unknown(ordinal) if index < tangent_input_count && ordinal == index => {}
                PartialEvaluationInput::Known(feeder) if index >= tangent_input_count => {
                    if !Rc::ptr_eq(feeder.builder(), &primal_builder) {
                        return Err(ProgramError::MalformedProgram(
                            "linearization produced a residual feeder owned by a foreign trace".to_string(),
                        )
                        .into());
                    }
                    residual_output_atoms.push(feeder.atom_id()?);
                }
                _ => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced a tangent program whose leading tangent inputs are not followed by \
                         its residuals"
                            .to_string(),
                    )
                    .into());
                }
            }
        }

        // Reference transposition can need the full allocation geometry even when a live tangent or its output
        // describes only a view. Retain dimensions of ordinary primal inputs without introducing reference reads.
        // Without reference state, only disconnected tangent inputs require additional zero-construction residuals.
        let has_references = tangent_program.entry_region_ref().contains_references_in_closure();
        let tangent_live_sets = tangent_program.live_sets();
        let differentiable_primal_inputs = arguments
            .input_indices
            .iter()
            .map(|&index| (&self.atoms()[self.input_ids()[index].index()], primal_input_atoms[index]));
        let mut zero_residual_types = Vec::new();
        for (((input, primal_input), tangent_input), tangent_input_atom) in differentiable_primal_inputs
            .zip(tangent_program.inputs().take(tangent_input_count))
            .zip(tangent_program.input_ids().iter().copied().take(tangent_input_count))
        {
            let tangent_type = tangent_input.r#type().into_owned();
            if tangent_type.is_reference() || (!has_references && tangent_live_sets.atoms()[tangent_input_atom.index()])
            {
                continue;
            }
            let expected_types = O::zero_residual_types(&tangent_type);
            let residuals = capture_and_validate_zero_residual_atoms(
                &mut primal_builder.borrow_mut(),
                primal_input,
                &tangent_type,
                &format!("transposition zero for input type {}", input.r#type()),
            )?;
            residual_output_atoms.extend(residuals);
            zero_residual_types.extend(expected_types);
        }

        if !zero_residual_types.is_empty() {
            // Program boundaries are immutable. Rebuild the tangent program with the old inputs as an unchanged prefix
            // and the zero-geometry residuals as a trailing suffix; the primal program emits residual values in this
            // same order below.
            let mut builder = ProgramBuilder::<V, O>::new();
            let old_input_count = tangent_program.input_ids().len();
            let inputs = tangent_program
                .input_types()
                .into_iter()
                .chain(zero_residual_types)
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let outputs = builder.splice_program(&tangent_program, &inputs[..old_input_count])?;
            let output_count = outputs.len();
            tangent_program =
                builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count])?;
        }

        let residual_count = residual_output_atoms.len();
        primal_output_atoms.extend(residual_output_atoms);

        // `evaluation.outputs` is deliberately dropped here. Every tangent output was forced unknown above and its
        // ordering is already represented by the residual program's output boundary.
        drop(evaluation.outputs);
        drop(primal_context);
        let primal_builder =
            Rc::try_unwrap(primal_builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let primal_output_count = primal_output_atoms.len();
        let primal_program = primal_builder
            .build::<Vec<V>, Vec<V>>(
                primal_output_atoms,
                vec![Placeholder; primal_input_count],
                vec![Placeholder; primal_output_count],
            )?
            .into_simplified()?;

        // Partial evaluation already gives the tangent program its flat vector boundary.
        // `Linearization::new_with_respect_to` is the sole cross-program contract validation.
        Linearization::new_with_respect_to(primal_program, tangent_program, residual_count, &arguments.input_indices)
            .map_err(DifferentiationError::from)
    }

    /// Rejects a reference-typed output of this [`Region`] that is a derived view of a reference (i.e., whose alias
    /// chain contains a [`ReferenceAliasKind::View`](crate::ReferenceAliasKind::View) edge), since supporting such an
    /// output would require applying the same view to the transformed root. Returned reference views are not currently
    /// supported. Their tangent outputs would need to preserve the view of the corresponding tangent reference. Return
    /// the underlying reference and apply the view outside the differentiated program instead. Refer to the
    /// documentation of [`ReferenceAnalysis::is_view`](crate::ReferenceAnalysis::is_view) for the analysis this
    /// consults.
    fn validate_reference_output_views(&self) -> Result<(), DifferentiationError> {
        // Local reference state does not require this output-boundary check unless a reference actually escapes.
        if !self.output_ids().iter().any(|output| self.atoms()[output.index()].r#type().is_reference()) {
            return Ok(());
        }
        let analysis = self.reference_analysis_with_configuration(None, true, &[]).map_err(ProgramError::from)?;
        for (output_index, output_atom) in self.output_ids().iter().copied().enumerate() {
            if analysis.is_view(ValueId::new(self.id(), output_atom)) {
                return Err(ProgramError::UnsupportedOperation {
                    message: format!(
                        "output {output_index} is a derived view of a reference and cannot be differentiated; \
                         return the viewed reference and apply the view outside the differentiated program"
                    ),
                }
                .into());
            }
        }
        Ok(())
    }
}

impl<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type, Operation = O>,
{
    /// Builds the _fused_ Jacobian-Vector Product (JVP) [`Program`] of this [`Program`]. Assume the input program
    /// represents a function `f` from its inputs to its outputs, `x ↦ y = f(x)`. This function returns the program that
    /// computes `f` together with its _pushforward_ (i.e., the forward-mode Jacobian-vector product): given an input
    /// tangent (i.e., perturbation direction) `ẋ`, the pushforward produces the output tangent `ẏ = (∂f/∂x)(x) · ẋ`,
    /// the directional derivative of `f` at `x` along `ẋ`. As a single map, the returned program computes
    /// `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ) = (y, ẏ)`. In terms of the program boundaries, if the input program has inputs
    /// `[x_1, …, x_n]` and outputs `[y_1, …, y_m]` (so that `y = f(x)`), the returned program has:
    ///
    ///   - inputs `[x_1, …, x_n, live(ẋ_1, …, ẋ_n)]`, which correspond to the primal inputs followed by one fresh
    ///     tangent input for each nonzero differential input, and
    ///   - outputs `[y_1, …, y_m, live(ẏ_1, …, ẏ_m)]`, which correspond to the primal outputs followed by the
    ///     tangents of nonzero differential outputs.
    ///
    /// More precisely, `live(ẋ_1, …, ẋ_n)` is the subsequence containing only tangents whose types are not zero
    /// differential spaces. A tangent in a zero differential space has exactly one possible value, so the transformed
    /// program allocates no Single Static Assignment (SSA) input or output for it. This omission applies only to
    /// tangent slots. All primal inputs and outputs remain present, and an ordinary primal residual remains present
    /// whenever derivative computation needs its value. Higher-level callable transforms retain their structured public
    /// boundaries and insert the uniquely determined typed zeros when rebuilding their results.
    ///
    /// The program is *not* split into separate primal and tangent sub-programs unlike [`Self::linearize`], which
    /// directly composes differentiation with partial evaluation. This un-split form remains exposed for fused
    /// higher-order JVP rules and direct forward-mode interpretation.
    ///
    /// Each primal instruction is replayed once through its [`DifferentiableOperation`] rule, which returns the dual
    /// (i.e., primal result plus tangent) for the instruction's outputs. Both are staged into the shared builder as
    /// ordinary primal operations, and so the result contains no symbolic captures.
    ///
    /// Atoms that are not reached by any input tangent are structurally zero. Their tangents stay symbolic as typed
    /// [`MaybeZero::Zero`]s and stage nothing. The shared all-zero fast path short-circuits operations whose every
    /// input tangent is structural zero only when each output zero tangent can be materialized without runtime identity
    /// operands (or by reusing a compatible zero-producing primal). It stages the primal directly and pairs each output
    /// with a typed structural zero tangent. Structural zeros are materialized as typed
    /// [`ZeroOperation`](crate::ZeroOperation) instructions only when a nonzero differential output requires a real
    /// value, preserving a compact `(primal_outputs ++ live_tangent_outputs)` program contract.
    ///
    /// Reference-typed inputs are differentiated directly, without discharging the program first. A reference type is
    /// never a zero differential space, so every reference-typed input receives a concrete tangent reference input of
    /// type `ref<tangent(T)>`, the reference operations' forward mode differentiation rules propagate tangents through
    /// the referenced state (allocating a tangent reference for every local allocation), and a reference-typed output
    /// has as its tangent output the tangent reference of the input root it forwards or of the local allocation that
    /// escapes through it. A reference output rooted in a captured (plumbing) reference and an output that is a derived
    /// view of a reference are rejected. Refer to the documentation of [`RegionRef::jvp`] for the selected-input form
    /// used by structured operation rules and for these boundary rules.
    ///
    /// # Reference Arguments
    ///
    /// When executing the returned program, distinct primal reference inputs and captured reference bindings must
    /// denote distinct allocations. Each tangent reference must denote an allocation distinct from every primal input
    /// or capture and from every other tangent reference. These requirements apply to allocation identity, including
    /// aliases and views, rather than to whether accessed elements overlap. Internally forwarded references retain
    /// their existing identity; forwarding does not create another independent caller binding.
    ///
    /// Note that this transform builds a raw program and does not insert runtime alias validation. Callers must
    /// establish these conditions before interpretation or compiled execution. [`ForwardModeDifferentiate::jvp`]
    /// validates explicit primal, tangent, and capture arguments using [`ReferenceBoundary`]. Raw program callers can
    /// use that validator directly when they have the concrete bindings and their owning context.
    #[inline]
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.entry_region_ref().jvp(&(0..self.input_ids().len()).collect::<Vec<_>>())
    }

    /// Builds the _fused_ Jacobian-Vector Product (JVP) [`Program`] of this [`Program`] with respect to the specified
    /// inputs. Primal inputs retain source order. Tangent inputs follow `input_indices`, omitting zero differential
    /// spaces. Refer to the documentation of [`Program::jvp`] and [`RegionRef::jvp`] for more information.
    #[inline]
    pub fn jvp_with_respect_to(
        &self,
        input_indices: &[usize],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.entry_region_ref().jvp(input_indices)
    }

    /// Linearizes this [`Program`] directly by replaying it once through a [`DifferentiationContext`] over a
    /// [`PartialEvaluationContext`] whose known-side parent is a fresh [`TracingContext`]. This context composition
    /// handles each source instruction once while simultaneously separating its two halves: primal-only work stages
    /// into the primal trace, and tangent-dependent work stages into the residual tangent program. An instruction
    /// normally dispatches its forward-mode rule once; the established nonempty all-structural-zero fast path instead
    /// binds only its primal operation and propagates typed structural zeros.
    ///
    /// The resulting [`Linearization`] has the boundary `x -> (y, r)` and `(live(dx), r) -> live(dy)`. Every source
    /// input is seeded eagerly as one known primal tracer and, for a nonzero differential space, one leading unknown
    /// tangent input. When tangent work first consumes a known primal value, partial evaluation materializes that value
    /// as a residual and its shared materialization slot deduplicates later uses; literal constants instead remain
    /// inline tangent-program constants. Residual feeder tracers are appended to the primal outputs in exactly the
    /// tangent program's trailing input order. Zero differential outputs remain structural and are omitted from the
    /// tangent program. A tangent that folds to a known value is rejected as a well-formed linear tangent map must
    /// represent an input-independent zero as [`MaybeZero::Zero`], while accepting an arbitrary known value would
    /// silently mask a nonlinear rule.
    ///
    /// Rules bind effects to their primal or tangent destination. Primal effects execute once at the linearization
    /// point, while tangent effects execute on each tangent invocation even when their operands are entirely known.
    /// Higher-order operations own their nested splitting through their existing differentiation and partial evaluation
    /// rules. The final pair's program interfaces are validated by [`Linearization::new`].
    ///
    /// Executing the resulting raw programs requires the caller to uphold the
    /// [reference argument contract](Linearization#reference-arguments). This transform does not insert runtime alias
    /// checks or retain concrete primal reference identities. Use [`ForwardModeDifferentiate::linearize`] and
    /// [`Pushforward::apply`] when the callable should validate its explicit reference arguments.
    #[inline]
    pub fn linearize(&self) -> Result<Linearization<V, O>, DifferentiationError> {
        self.entry_region_ref().linearize(&(0..self.input_ids().len()).collect::<Vec<_>>())
    }

    /// Linearizes this [`Program`] with respect to the specified inputs. Primal inputs retain source order. Tangent
    /// inputs follow `input_indices`, omitting zero differential spaces, and precede the residual inputs. Refer to the
    /// documentation of [`Program::linearize`] and [`RegionRef::linearize`] for more information.
    #[inline]
    pub fn linearize_with_respect_to(
        &self,
        input_indices: &[usize],
    ) -> Result<Linearization<V, O>, DifferentiationError> {
        self.entry_region_ref().linearize(input_indices)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> PartitionedProgram<V, O> {
    /// Interprets this partitioned fused Jacobian-Vector Product (JVP) program, computing known work in
    /// [`DifferentiationContext::primal`] and residual work in [`DifferentiationContext::tangent`]. Returns the
    /// original program's outputs in their original order. Specifically, the first `primal_output_count` outputs belong
    /// to the primal context and the remaining outputs belong to the tangent context, including tangent outputs that
    /// partial evaluation classified as known.
    ///
    /// Primal/tangent ownership differs from known/residual placement. For example, a constant tangent output can be
    /// known, but it must still be transferred to the tangent context. This partition records only the latter
    /// classification, so the primal-output prefix must be supplied separately. Inactive tangent outputs may have
    /// been omitted, so the prefix need not contain half of the outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Differentiation context in which to interpret the partition. The partition must preserve the
    ///     rule's effect ordering and reference lifetimes across calls in this context.
    ///   - `inputs`: Values in the original fused program's input order. Known inputs belong to the primal context;
    ///     unknown inputs belong to the tangent context. Unused inputs must be included as well as the partition
    ///     preserves their boundary positions even when no instruction consumes them.
    ///   - `primal_output_count`: Number of leading original outputs that must remain in the primal context. Every
    ///     output in this prefix must be known, and the count must not exceed the original output count.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidInputCount`] when the input count differs from the original boundary recorded
    /// by the partition. Returns [`ProgramError::InvalidArgument`] when the primal-output prefix is out-of-bounds or
    /// contains a residual output. These checks run before either program executes. Interpretation and value transfer
    /// errors are propagated when replaying the validated partition.
    pub fn interpret_in_context<
        C: Context<Type = V::Type, Constant = V, Operation = O>,
        P: DifferentiationPolicy<C>,
    >(
        &self,
        context: &DifferentiationContext<C, P>,
        inputs: &[C::Value],
        primal_output_count: usize,
    ) -> Result<Vec<C::Value>, DifferentiationError> {
        // Known inputs and unknown residual inputs together retain the original fused input boundary. Saved
        // residual values are not original inputs. Check this boundary before known work can execute effects.
        let input_count = self
            .known_input_indices()
            .iter()
            .copied()
            .chain(self.residual_inputs().iter().filter_map(|input| match input {
                PartialEvaluationInput::Unknown(index) => Some(*index),
                PartialEvaluationInput::Known(_) => None,
            }))
            .max()
            .map_or(0, |index| index + 1);
        check_count!("input", inputs, input_count, ProgramError);

        let primal_outputs =
            self.outputs().get(..primal_output_count).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "partitioned JVP declares {} primal outputs but has only {} outputs",
                    primal_output_count,
                    self.outputs().len(),
                ),
            })?;

        if let Some(index) = primal_outputs.iter().position(|output| !output.is_known()) {
            return Err(ProgramError::InvalidArgument {
                message: format!("partitioned JVP primal output {index} is residual; all primal outputs must be known"),
            }
            .into());
        }

        let known_inputs = self.known_input_indices().iter().map(|&index| inputs[index].clone()).collect();
        let known_outputs = self.known_program().interpret_in_context(context.primal(), known_inputs)?;

        // The known program returns known original outputs first, followed by saved values for residual inputs.
        // Feeder indices address only the latter group, so they need the known-output prefix offset.
        let known_count = self.outputs().iter().filter(|output| output.is_known()).count();
        let residual_inputs = self
            .residual_inputs()
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => Ok(inputs[*index].clone()),
                PartialEvaluationInput::Known(index) => {
                    context.primal_to_tangent(known_outputs[known_count + index].clone())
                }
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let residual_outputs = self.residual_program().interpret_in_context(context.tangent(), residual_inputs)?;

        self.outputs()
            .iter()
            .enumerate()
            .map(|(index, output)| match output {
                PartialEvaluationOutput::Known(position) if index < primal_output_count => {
                    Ok(known_outputs[*position].clone())
                }
                PartialEvaluationOutput::Known(position) => context.primal_to_tangent(known_outputs[*position].clone()),
                PartialEvaluationOutput::Unknown(position) => Ok(residual_outputs[*position].clone()),
            })
            .collect()
    }
}

/// Extension trait carrying the value-level *forward-mode* differentiation transforms on every [`Context`], mirroring
/// how [`Batch`](crate::Batch) carries batching. [`ReverseModeDifferentiate`](crate::ReverseModeDifferentiate) is its
/// sibling that builds reverse mode on top of it (i.e., `vjp = linearize + transpose`).
///
/// This trait is blanket-implemented for every [`Context`] whose type family is [`DifferentiableType`] and has no
/// items of its own to implement. Every entry point is a defaulted method whose `where` clause carries its remaining
/// requirements (e.g., the operation family's [`DifferentiableOperation`] rules), so whether a particular transform
/// is available on a particular context is decided per method at the call site, in exactly the same way as
/// [`Batch::batch`](crate::Batch::batch). Tangents are ordinary values of the same universe as the primals (i.e.,
/// [`Domain::Value`]) flowing through the same context. The type-level tangent structure, such as the cotangent types,
/// live on [`DifferentiableType`] instead. Operations that involve predicates such as `condition`, `while`, and
/// `select` impose their own [`Concretizable<bool>`](crate::Concretizable) bounds through their operation-family
/// implementations.
///
/// Whether a transform runs eagerly or stages a program is decided by the context's [`Value`](Domain::Value) (i.e.,
/// concrete vs [`Tracer`]), not by a separate trait. Captures follow the same operational validity rules as ordinary
/// values: an incompatible capture fails at the operation or execution boundary where it is used, while an unused
/// capture does not affect the transform.
pub trait ForwardModeDifferentiate: Context<Type: DifferentiableType> {
    /// Evaluates `function` on the primal `primal` and runtime `capture`, and propagates the tangent `tangent`
    /// forward only with respect to `primal`, with this [`Context`] executing or staging the differentiated
    /// operations. Refer to the documentation of [`DifferentiationBuilder::jvp`](crate::DifferentiationBuilder::jvp)
    /// for the forward-mode transform.
    fn jvp<
        F: FnOnce(
            Input::To<DifferentiationTracer<Self>>,
            Capture::To<DifferentiationTracer<Self>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<DifferentiationTracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Capture: Parameterized<Self::Value, Family: ParameterizedFamily<DifferentiationTracer<Self>>>,
        Output: Parameterized<DifferentiationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primal: Input,
        tangent: Input::To<Self::Value>,
        capture: Capture,
    ) -> Result<(Output::To<Self::Value>, Output::To<Self::Value>), DifferentiationError>
    where
        Self::Operation: DifferentiableOperation<Self> + ResidualZeroProvider<Self::Type, Operation = Self::Operation>,
    {
        if primal.parameters().next().is_none() {
            return Err(DifferentiationError::EmptyInput);
        }

        let primal_structure = primal.parameter_structure();
        let tangent_structure = tangent.parameter_structure();
        if tangent_structure != primal_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        // The tangents take part in the alias validation. A tangent reference is mutated by the reference forward mode
        // differentiation rules independently of every primal reference, and so it must be a distinct allocation.
        ReferenceBoundary::new_for_differentiation(
            self,
            primal.parameters(),
            tangent.parameters(),
            capture.parameters(),
        )?;

        // Active inputs receive the caller-provided tangents. Captures share the same transform context but receive
        // only structural zero tangents, so they affect primal evaluation without affecting differentiation.
        let context = DifferentiationContext::fused(self.clone());
        let input_duals = primal
            .into_parameters()
            .zip(tangent.into_parameters())
            .map(|(primal, tangent)| {
                Ok::<_, ProgramError>(DifferentiationTracer::new(
                    DifferentiationDual::new(primal, tangent)?,
                    context.clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = Input::To::<DifferentiationTracer<Self>>::from_parameters(primal_structure, input_duals)?;
        let capture_structure = capture.parameter_structure();
        let capture_duals = capture
            .into_parameters()
            .map(|primal| -> Result<_, DifferentiationError> {
                Ok(DifferentiationTracer::new(DifferentiationDual::new_with_zero_tangent(primal)?, context.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capture = Capture::To::<DifferentiationTracer<Self>>::from_parameters(capture_structure, capture_duals)?;
        let output = function(input, capture)?;

        // Split each output dual into its primal value and its materialized tangent. A structural zero derives its
        // runtime geometry from the corresponding primal result before the public boundary requires a concrete value.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for (output_index, output_dual) in output_duals.into_iter().enumerate() {
            let (primal, tangent) = output_dual.into_dual().into_parts();
            let tangent = match tangent {
                MaybeZero::Value(tangent) => tangent,
                MaybeZero::Zero(_) if primal.r#type().is_reference() => {
                    // A reference type is never a zero differential space and a zero reference tangent cannot be
                    // materialized, so a reference output rooted in a captured (i.e., plumbing) reference is rejected
                    // here with a boundary diagnostic instead of failing inside the zero materialization below.
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "output {output_index} is a reference rooted in a reference that carries no tangent \
                             (a captured or inactive reference); pass that reference as a differentiated input instead",
                        ),
                    }
                    .into());
                }
                MaybeZero::Zero(r#type) => {
                    let residuals =
                        capture_and_validate_zero_residual_values(self, &primal, &r#type, "jvp output tangent")?;
                    let (operation, operands) =
                        Self::Operation::zero_operation_with_residuals(r#type, residuals.as_slice())?;
                    let mut outputs = self.bind(operation, Vec::new(), operands.as_slice())?;
                    check_count!("output", outputs, 1, ProgramError);
                    outputs.remove(0)
                }
            };
            tangent_outputs.push(tangent);
            primal_outputs.push(primal);
        }
        let primal_output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;
        let tangent_output = Output::To::<Self::Value>::from_parameters(output_structure, tangent_outputs)?;
        Ok((primal_output, tangent_output))
    }

    /// Linearizes `function` at `primal`, treating `capture` as known nondifferentiated runtime inputs and returning
    /// the primal output and a reusable [`Pushforward`], with this [`Context`] executing or staging primal-side work.
    /// Refer to the documentation of [`DifferentiationBuilder::linearize`](crate::DifferentiationBuilder::linearize)
    /// for the forward-mode transform.
    fn linearize<
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
            Capture::To<LinearizationTracer<Self>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Capture: Parameterized<Self::Value, To<Self::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primal: Input,
        capture: Capture,
    ) -> Result<(Output::To<Self::Value>, Pushforward<Self, Input, Output::To<Self::Value>>), DifferentiationError>
    where
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + ResidualZeroProvider<Self::Type, Operation = Self::Operation>,
    {
        if primal.parameters().next().is_none() {
            return Err(DifferentiationError::EmptyInput);
        }

        let primal_references = ReferenceBoundary::new_for_differentiation(
            self,
            primal.parameters(),
            std::iter::empty(),
            capture.parameters(),
        )?;

        let input_structure = primal.parameter_structure();
        let input_values = primal.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();

        // Active primals receive unknown tangent inputs. Captures are known primal inputs paired with structural zeros,
        // so they can be residualized when needed without increasing the pushforward's tangent arity.
        let primal_evaluation_context = PartialEvaluationContext::new(self.clone());
        let differentiation_context = DifferentiationContext::partitioned(primal_evaluation_context.clone());
        let evaluation_context = differentiation_context.tangent().clone();
        let mut tangent_index = 0usize;

        // Retain the primals for the geometry capture below; only the values stamped into duals need cloning.
        let input_duals = input_values
            .iter()
            .cloned()
            .map(|value| {
                let primal_type = value.r#type().into_owned();
                let tangent_type = primal_type.tangent()?;
                let tangent = if !tangent_type.is_zero_space() {
                    let tangent = evaluation_context.unknown_input(tangent_type.clone(), tangent_index);
                    tangent_index += 1;
                    MaybeZero::Value(PartialTracer::new(evaluation_context.clone(), tangent))
                } else {
                    MaybeZero::Zero(tangent_type)
                };
                let dual = DifferentiationDual::new(
                    PartialTracer::new(primal_evaluation_context.clone(), PartialEvaluationValue::known_input(value)),
                    tangent,
                )?;
                Ok::<_, ProgramError>(DifferentiationTracer::new(dual, differentiation_context.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let tangent_input_count = tangent_index;
        let input = Input::To::<LinearizationTracer<Self>>::from_parameters(input_structure, input_duals)?;
        let capture_structure = capture.parameter_structure();
        let capture_duals = capture
            .into_parameters()
            .map(|value| -> Result<_, DifferentiationError> {
                let primal =
                    PartialTracer::new(primal_evaluation_context.clone(), PartialEvaluationValue::known_input(value));
                Ok(DifferentiationTracer::new(
                    DifferentiationDual::new_with_zero_tangent(primal)?,
                    differentiation_context.clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let capture = Capture::To::<LinearizationTracer<Self>>::from_parameters(capture_structure, capture_duals)?;
        let output = function(input, capture)?;

        // Split each output dual into its known primal value and its tangent. Primal work depends only on known primal
        // inputs, so every primal half must have folded to a known value.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();

        // Force structural zeros into the residual program. Calling `Zero` normally would allow partial evaluation to
        // keep the result known, but a reusable pushforward must expose every non-zero-space tangent output as a Single
        // Static Assignment (SSA) value.
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        let mut output_types = Vec::with_capacity(output_duals.len());
        for (output_index, output_dual) in output_duals.into_iter().enumerate() {
            let (primal, tangent) = output_dual.into_dual().into_parts();
            let primal = match primal.into_value()?.value() {
                PartialValue::Known(value) => value.clone(),
                PartialValue::Unknown(_) => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced an unknown primal output but primal work depends only on the known \
                         primal inputs"
                            .to_string(),
                    )
                    .into());
                }
            };
            let tangent_type = tangent.r#type().into_owned();
            output_types.push(primal.r#type().into_owned());
            if tangent_type.is_zero_space() {
                primal_outputs.push(primal);
                continue;
            }

            // A reference output rooted in a captured (i.e., plumbing) reference has no tangent reference to expose
            // and is rejected with a boundary diagnostic instead of failing inside the zero residualization below.
            if primal.r#type().is_reference() && tangent.is_zero() {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "output {output_index} is a reference rooted in a reference that carries no tangent \
                         (a captured or inactive reference); pass that reference as a differentiated input instead",
                    ),
                }
                .into());
            }

            let tangent = match tangent {
                MaybeZero::Value(tracer) => {
                    let value = tracer.into_value()?;
                    match value.value() {
                        PartialValue::Unknown(_) => value,
                        PartialValue::Known(_) => {
                            return Err(ProgramError::MalformedProgram(
                                "linearization produced a known tangent output; differentiation rules must represent \
                                 input-independent zero tangents structurally"
                                    .to_string(),
                            )
                            .into());
                        }
                    }
                }
                MaybeZero::Zero(r#type) => {
                    let residuals = capture_and_validate_zero_residual_values(
                        self,
                        &primal,
                        &r#type,
                        "pushforward output tangent",
                    )?;
                    residualize_zero_from_residual_values(&evaluation_context, r#type, residuals)?
                }
            };
            primal_outputs.push(primal);
            tangent_outputs.push(tangent);
        }
        let tangent_reconstruction = ZeroSpaceBoundaryReconstruction::capture(
            self,
            primal_outputs.as_slice(),
            output_types.as_slice(),
            ZeroSpaceBoundaryRole::OutputTangent,
        )?;
        let output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;

        // All tracer-stamped context clones are dropped here, so the accumulated pushforward program can be finalized.
        drop(differentiation_context);

        let primal_evaluation = primal_evaluation_context.into_evaluation(Vec::new())?;
        if !primal_evaluation.program.instructions().is_empty() {
            return Err(ProgramError::MalformedProgram(
                "linearization deferred an operation bound to the primal destination".to_string(),
            )
            .into());
        }
        drop(primal_evaluation);

        let evaluation = evaluation_context.into_evaluation(tangent_outputs)?;

        // The pushforward program's inputs are the leading active tangent unknowns
        // followed by captured residual values.
        let mut residuals = Vec::with_capacity(evaluation.inputs.len().saturating_sub(tangent_input_count));
        for (index, input) in evaluation.inputs.iter().enumerate() {
            match input {
                PartialEvaluationInput::Unknown(ordinal) if index < tangent_input_count && *ordinal == index => {}
                PartialEvaluationInput::Known(value) if index >= tangent_input_count => residuals.push(value.clone()),
                _ => {
                    return Err(ProgramError::MalformedProgram(
                        "linearization produced a pushforward program whose tangent inputs do not lead its residuals"
                            .to_string(),
                    )
                    .into());
                }
            }
        }

        // Preserve allocation geometry for reference adjoints, including live inputs whose outputs describe only a
        // view. Only ordinary inputs supply these extra dimensions as reading a primal reference solely for geometry
        // could move a synchronization or lifecycle error ahead of the original computation. For reference-free
        // programs, disconnected tangent inputs alone require extra zero geometry.
        let mut program = evaluation.program;
        let has_references = program.entry_region_ref().contains_references_in_closure();
        let live_sets = program.live_sets();
        let differentiable_primal_inputs = input_values
            .iter()
            .map(|value| Ok((value.r#type().tangent()?, value)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter_map(|(tangent_type, value)| (!tangent_type.is_zero_space()).then_some(value));
        let mut zero_residuals = Vec::new();
        for ((primal, tangent_input), tangent_input_atom) in differentiable_primal_inputs
            .zip(program.inputs().take(tangent_input_count))
            .zip(program.input_ids().iter().copied().take(tangent_input_count))
        {
            let tangent_type = tangent_input.r#type().into_owned();
            if tangent_type.is_reference() || (!has_references && live_sets.atoms()[tangent_input_atom.index()]) {
                continue;
            }
            let values = capture_and_validate_zero_residual_values(
                self,
                primal,
                &tangent_type,
                &format!("transposition zero for input type {}", primal.r#type()),
            )?;
            zero_residuals.extend(values);
        }

        if !zero_residuals.is_empty() {
            let mut builder = ProgramBuilder::<Self::Constant, Self::Operation>::new();
            let old_input_count = program.input_ids().len();
            let inputs = program
                .input_types()
                .into_iter()
                .chain(zero_residuals.iter().map(|value| value.r#type().into_owned()))
                .map(|r#type| builder.add_input(r#type))
                .collect::<Vec<_>>();
            let outputs = builder.splice_program(&program, &inputs[..old_input_count])?;
            let output_count = outputs.len();
            program = builder.build(outputs, vec![Placeholder; inputs.len()], vec![Placeholder; output_count])?;
            residuals.extend(zero_residuals);
        }

        let pushforward = Pushforward::new(
            self.clone(),
            program,
            residuals,
            tangent_reconstruction,
            input_types,
            output_types,
            primal_references,
            output_structure,
        )?;
        Ok((output, pushforward))
    }
}

impl<C: Context<Type: DifferentiableType>> ForwardModeDifferentiate for C {}

/// Applies a member operation's Jacobian-Vector Product (JVP) rule through a projected view of a composite
/// differentiation context. Use this function from a composite operation dispatcher when the operation is
/// [`Region`]-free and every operand and result belongs to the same projectable member type `T`. It projects primal
/// values and live tangent values into the member value family, carries structural-zero tangents as types without
/// materializing values, runs the member's existing [`DifferentiableOperation`] rule, and lifts the resulting duals
/// back into the composite value family.
///
/// Operations whose derivative crosses member types or whose rule needs attached regions require an explicit composite
/// Jacobian-Vector Product (JVP) rule instead. A member operation that declares [`RegionSlot`](crate::RegionSlot)s is
/// rejected with an exact diagnostic naming it, because projection reaches the member rule with no region access: the
/// attached regions are programs in the composite universe, and no projected driver can present them in the member
/// universe.
///
/// # Parameters
///
///   - `context`: Active composite [`Context`] through which the projected member rule stages its primal and tangent
///     operations.
///   - `operation`: Region-free operation expressed in the projected member operation family.
///   - `inputs`: Composite [`DifferentiationDual`]s corresponding to the operation's operands.
pub fn jvp_projected_operation<
    T: DifferentiableType,
    O: Operation<Type = T> + DifferentiableOperation<ProjectedContext<C, T>>,
    C: Context<
            Type: DifferentiableType + From<T>,
            Value: ValueProjection<T, Projected: Value<Type = T>>,
            Constant: ValueProjection<T, Projected: Value<Type = T>>,
            Operation: OperationProjection<T, Projected = O>,
        >,
    P: DifferentiationPolicy<C>,
>(
    context: &DifferentiationContext<C, P>,
    operation: &O,
    inputs: &[DifferentiationDual<C::Value>],
) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
    if !operation.region_slots().is_empty() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "projected operation `{}` carries regions and cannot be differentiated through its member family; \
                 differentiate it through a composite carrier for that operation instead",
                operation.name(),
            ),
        }
        .into());
    }
    let projected_inputs = inputs
        .iter()
        .map(|input| {
            let primal = <C::Value as ValueProjection<T>>::into_projected(input.primal().clone())?;
            match input.tangent() {
                MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal),
                MaybeZero::Value(value) => {
                    let tangent = <C::Value as ValueProjection<T>>::into_projected(value.clone())?;
                    DifferentiationDual::new(primal, tangent)
                }
            }
        })
        .collect::<Result<Vec<_>, DifferentiationError>>()?;
    operation
        .jvp(&context.project::<T>(), &EmptyRegionDriver, projected_inputs.as_slice())?
        .into_iter()
        .map(|output| {
            let (primal, tangent) = output.into_parts();
            let primal = <C::Value as ValueProjection<T>>::from_projected(primal);
            let tangent = match tangent {
                MaybeZero::Zero(r#type) => MaybeZero::Zero(C::Type::from(r#type)),
                MaybeZero::Value(value) => MaybeZero::Value(<C::Value as ValueProjection<T>>::from_projected(value)),
            };
            DifferentiationDual::new(primal, tangent)
        })
        .collect::<Result<Vec<_>, _>>()
}

/// Captures the program atoms needed to materialize a zero of `r#type` and verifies the provider's declaration.
/// [`ResidualZeroProvider::zero_residual_types`] declares the runtime values required to construct the zero, while
/// [`ResidualZeroProvider::capture_zero_residuals`] stages the operations that obtain those values from `source`. This
/// helper keeps the two methods consistent by checking both the captured atom count and every captured atom's type.
/// The returned atoms remain in provider-declaration order and can therefore be passed directly to the matching zero
/// operation.
///
/// # Parameters
///
///   - `builder`: Program builder in which residual-capture operations are staged.
///   - `source`: Primal program atom from which the provider obtains runtime geometry.
///   - `r#type`: Type of the zero that will later be materialized.
///   - `site`: Description of the capture site included in malformed-program diagnostics.
fn capture_and_validate_zero_residual_atoms<
    V: Value,
    O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type, Operation = O>,
>(
    builder: &mut ProgramBuilder<V, O>,
    source: AtomId,
    r#type: &V::Type,
    site: &str,
) -> Result<Vec<AtomId>, ProgramError> {
    let expected_types = O::zero_residual_types(r#type);
    let residuals = O::capture_zero_residuals(builder, source, r#type)?;
    if residuals.len() != expected_types.len() {
        return Err(ProgramError::MalformedProgram(format!(
            "{} captured {} zero residuals but declared {}",
            site,
            residuals.len(),
            expected_types.len(),
        )));
    }
    for (index, (residual, expected_type)) in residuals.iter().copied().zip(expected_types).enumerate() {
        let actual_type = builder.atoms().get(residual.index()).ok_or(ProgramError::UnboundAtomId { id: residual })?;
        if actual_type.r#type().as_ref() != &expected_type {
            return Err(ProgramError::MalformedProgram(format!(
                "{} zero residual {} has type {} but expected {}",
                site,
                index,
                actual_type.r#type().as_ref(),
                expected_type,
            )));
        }
    }
    Ok(residuals)
}

/// Stages a residual-backed zero as an unknown output of `context`'s residual program. Although `residual_values` are
/// known during partial evaluation, a reusable linear program must return its non-zero-space tangent outputs as Single
/// Static Assignment (SSA) values. Binding the zero normally could let partial evaluation fold it into a known value
/// and remove that output from the program boundary. This helper instead marks the residual values as known inputs to
/// the zero operation and explicitly residualizes that operation, preserving one unknown program output while retaining
/// the runtime geometry needed to materialize the zero.
///
/// # Parameters
///
///   - `context`: Partial-evaluation context whose residual program receives the zero operation.
///   - `r#type`: Type of the zero to materialize.
///   - `residual_values`: Runtime geometry values declared by [`ResidualZeroProvider::zero_residual_types`], in
///     provider-declaration order.
fn residualize_zero_from_residual_values<
    C: Context<Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>>,
>(
    context: &PartialEvaluationContext<C>,
    r#type: C::Type,
    residual_values: Vec<C::Value>,
) -> Result<PartialEvaluationValue<C::Value>, ProgramError> {
    let residual_values = residual_values.into_iter().map(PartialEvaluationValue::known_input).collect::<Vec<_>>();
    let (operation, operands) = C::Operation::zero_operation_with_residuals(r#type, residual_values.as_slice())?;
    let mut outputs = context.residualize(operation, Vec::new(), operands.as_slice())?;
    check_count!("output", outputs, 1, ProgramError);
    Ok(outputs.remove(0))
}

/// [`Region`] [`Transform`] marker for retained fused Jacobian-Vector Product (JVP) [`Program`]s.
struct JvpTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for JvpTransform {
    type Arguments = JvpAndLinearizationTransformArguments;
    type Artifact = TransformArtifact<V, O, ()>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// [`Region`] [`Transform`] marker for retained linearized [`Program`]s.
struct LinearizationTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for LinearizationTransform {
    type Arguments = JvpAndLinearizationTransformArguments;
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// Argument key for one retained [`JvpTransform`] or [`LinearizationTransform`] artifact. Selected inputs retain
/// their requested order, with zero differential spaces omitted. Reordering live inputs changes the tangent boundary
/// and therefore the cache key; adding or moving only zero space inputs does not.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct JvpAndLinearizationTransformArguments {
    /// Selected input indices in tangent input order, excluding zero differential spaces.
    input_indices: Vec<usize>,
}

impl JvpAndLinearizationTransformArguments {
    /// Creates a new [`JvpAndLinearizationTransformArguments`] instance after validating the provided input indices and
    /// removing zero space inputs while preserving the remaining order.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] for duplicate or out-of-range indices, including duplicates of
    /// zero space inputs. Propagates tangent-type errors for selected inputs.
    fn new<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        input_indices: &[usize],
    ) -> Result<Self, DifferentiationError> {
        let input_count = region.input_ids().len();
        let mut selected = vec![false; input_count];
        let mut indices = Vec::with_capacity(input_indices.len());
        for &index in input_indices {
            let input = *region.input_ids().get(index).ok_or_else(|| ProgramError::InvalidArgument {
                message: format!(
                    "differentiation input index {index} is out of range for a region with {input_count} inputs",
                ),
            })?;
            if std::mem::replace(&mut selected[index], true) {
                return Err(ProgramError::InvalidArgument {
                    message: format!("differentiation input index {index} is selected more than once"),
                }
                .into());
            }
            if !region.atoms()[input.index()].r#type().tangent()?.is_zero_space() {
                indices.push(index);
            }
        }
        Ok(Self { input_indices: indices })
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArraySliceAxis, ArrayType,
        DataType, Dimension, DimensionBounds, DimensionVariable, ReferenceSliceOperation, Shape,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::operations::differentiation::tests::custom_jvp_regions_with_reference_state;
    use crate::operations::{
        ConditionOperation, CosOperation, CustomJvpOperation, Dot, DotDimensionNumbers, MulOperation,
        ParallelReduceOperation, ParallelReductionKind, PrintOperation, ReferenceAddUpdate,
        ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceWriteOperation, Sin, SinOperation, StopGradient, StopGradientOperation,
        ZeroOperation,
    };
    use crate::parameters::{ParameterError, Placeholder};
    use crate::programs::{
        Concretizable, Operation, OperationProvider, ProgramBuilder, ReferenceError, ReferenceType, RegionId,
    };
    use crate::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedMemberValue, ProjectedProgramOperation,
        ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::tracing::{NestedTracingContext, Trace};

    #[cfg(debug_assertions)]
    use crate::operations::TagOperation;

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;

    // Index 3 is otherwise unused by the shared member fixtures. Its malformed provider tests that the callable
    // boundary validates residual capture before attempting to construct a zero.
    impl OperationProvider<ProjectedMemberType<3>, ZeroOperation<ProjectedMemberType<3>>> for ProjectedMemberOperation<3> {
        type Operation = Self;

        fn provide(
            _request: ZeroOperation<ProjectedMemberType<3>>,
            _input_types: &[&ProjectedMemberType<3>],
        ) -> Result<Self, ProgramError> {
            panic!("the invalid residual capture must be rejected before constructing a zero");
        }
    }

    impl ResidualZeroProvider<ProjectedMemberType<3>> for ProjectedMemberOperation<3> {
        fn zero_residual_types(r#type: &ProjectedMemberType<3>) -> Vec<ProjectedMemberType<3>> {
            vec![r#type.clone()]
        }

        fn capture_zero_residual_values<C: Context<Type = ProjectedMemberType<3>, Operation = Self>>(
            _context: &C,
            _source: &C::Value,
            _type: &ProjectedMemberType<3>,
        ) -> Result<Vec<C::Value>, ProgramError> {
            Ok(Vec::new())
        }
    }

    /// Builds a fused JVP with one primal output, one residual tangent, and one known zero tangent. Known work writes
    /// the primal result into the reference input, making execution before boundary validation observable.
    fn partitioned_jvp_with_known_effect() -> PartitionedProgram<TestValue, TestOperation> {
        let scalar: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let primal = builder.add_input(scalar.clone());
        let tangent = builder.add_input(scalar.clone());
        builder.add_input(scalar);
        let one = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32)));
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32)));
        let primal = builder.add_instruction(AddOperation::new(), Vec::new(), vec![primal, one], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, primal], None)
            .unwrap();
        let tangent = builder
            .add_instruction(ArrayOperation::from(MulOperation::new()), Vec::new(), vec![primal, tangent], None)
            .unwrap()[0];
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![primal, tangent, zero],
                vec![Placeholder; 4],
                vec![Placeholder; 3],
            )
            .unwrap()
            .partition(&[true, true, false, false])
            .unwrap()
    }

    /// Builds a program whose entry applies `operation` to its single scalar input.
    fn unary_program(
        operation: ArrayOperation<Array>,
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-instruction region tagging its input with `key`.
    #[cfg(debug_assertions)]
    fn tagged_program(key: &str) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TagOperation::new(key), Vec::new(), vec![input], None).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Builds a single-instruction region scaling its input by the constant `factor`.
    #[cfg(debug_assertions)]
    fn scaled_program(factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let constant = builder.add_constant(Array::scalar(factor));
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
        builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Adds the numeric input to a reference and returns its updated contents.
    fn add_to_reference<V: ReferenceAddUpdate + ReferenceRead>((reference, x): (V, V)) -> Result<V, ProgramError> {
        reference.add_update(&x)?;
        reference.read()
    }

    /// Adds one reference's contents to another and returns the updated contents.
    fn add_reference_contents<V: ReferenceAddUpdate + ReferenceRead>(
        (reference, other): (V, V),
    ) -> Result<V, ProgramError> {
        reference.add_update(&other.read()?)?;
        reference.read()
    }

    /// Builds a valid flat program with the requested input types and constant outputs. Constructor tests vary
    /// the relationship between two valid programs rather than corrupting either program's own structure.
    fn boundary_program(
        inputs: &[DataType],
        outputs: &[Array],
    ) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
        let mut builder = ProgramBuilder::new();
        for &data_type in inputs {
            builder.add_input(ArrayType::scalar(data_type));
        }
        let output_ids = outputs.iter().cloned().map(|value| builder.add_constant(value)).collect();
        builder
            .build(output_ids, vec![Placeholder; inputs.len()], vec![Placeholder; outputs.len()])
            .unwrap()
    }

    /// Constructs a pushforward with a reference-free public boundary and its matching reconstruction plan.
    /// Tests independently vary the compact program and residual values to exercise constructor validation.
    fn pushforward_with_boundary(
        program: Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>,
        residuals: Vec<Array>,
        inputs: &[Array],
        outputs: &[Array],
    ) -> Result<Pushforward<EagerContext<Array, ArrayOperation<Array>>, Vec<Array>, Vec<Array>>, ProgramError> {
        let context = EagerContext::new();
        let input_types = inputs.iter().map(|value| value.r#type().into_owned()).collect();
        let output_types = outputs.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let reconstruction = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            outputs,
            &output_types,
            ZeroSpaceBoundaryRole::OutputTangent,
        )?;
        let references = ReferenceBoundary::new_for_differentiation(&context, inputs, &[], &[])?;
        Pushforward::new(
            context,
            program,
            residuals,
            reconstruction,
            input_types,
            output_types,
            references,
            vec![Placeholder; outputs.len()],
        )
    }

    /// Renders the complete cache nondeterminism diagnostic expected from a deliberately corrupted artifact.
    #[cfg(debug_assertions)]
    fn transform_mismatch_message<T>(
        cached: &[&Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>],
        derived: &[&Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>],
        cached_metadata: &str,
        derived_metadata: &str,
    ) -> String {
        let mut programs = String::new();
        for index in 0..cached.len().max(derived.len()) {
            for (label, entries) in [("cached", cached), ("fresh", derived)] {
                let rendering = entries.get(index).map(|program| program.to_string());
                programs.push_str(&format!(
                    "--- {label} program {index} ---\n{}\n",
                    rendering.as_deref().unwrap_or("<absent>"),
                ));
            }
        }
        let arguments = JvpAndLinearizationTransformArguments { input_indices: vec![0] };
        format!(
            "nondeterministic transform rule detected for `{}` with arguments `{arguments:?}`: re-derivation \
             produced a different artifact than the region cache retained, but region transforms must be deterministic \
             structural functions of their complete reachable contents and arguments\n\n\
             cached metadata: {}\nderived metadata: {}\n\n{}",
            std::any::type_name::<T>(),
            cached_metadata,
            derived_metadata,
            programs,
        )
    }

    #[test]
    fn test_differentiation_dual_new() {
        let differentiable = DifferentiationDual::new(Array::scalar(2.0), Array::scalar(3.0)).unwrap();
        let (primal, tangent) = differentiable.into_parts();
        assert_eq!(primal, Array::scalar(2.0));
        assert!(matches!(tangent, MaybeZero::Value(value) if value == Array::scalar(3.0)));

        let differentiable_zero = DifferentiationDual::new(
            Array::scalar(2.0),
            MaybeZero::<Array>::Zero(ArrayType::scalar(DataType::Boolean)),
        )
        .unwrap();
        let (primal, tangent) = differentiable_zero.into_parts();
        assert_eq!(primal, Array::scalar(2.0));
        assert!(matches!(tangent, MaybeZero::Zero(r#type) if r#type == ArrayType::scalar(DataType::F64)));

        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let non_differentiable = DifferentiationDual::new(token.clone(), zero).unwrap();
        let (primal, tangent) = non_differentiable.into_parts();
        assert_eq!(primal, token.clone());
        assert!(matches!(tangent, MaybeZero::Zero(r#type) if r#type == ArrayType::scalar(DataType::Zero)));

        assert!(matches!(
            DifferentiationDual::new(token.clone(), token),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "tangent type token[] does not match type zero[] required by primal type token[]",
        ));
    }

    #[test]
    fn test_differentiation_dual_new_with_zero_tangent() {
        let dual = DifferentiationDual::new_with_zero_tangent(Array::scalar(2.0_f64)).unwrap();
        assert_eq!(dual.primal(), &Array::scalar(2.0_f64));
        assert!(matches!(dual.tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F64)));
        let zero_space = DifferentiationDual::new_with_zero_tangent(Array::scalar(true)).unwrap();
        assert!(
            matches!(zero_space.tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::Zero))
        );
    }

    #[test]
    fn test_differentiation_dual_is_tangent_active() {
        // A live tangent is always active. A structural zero is active only when it can be materialized from its type
        // alone: a numeric zero is, while a zero-space value, a plumbing reference, and a zero whose type carries a
        // runtime dimension identity are not.
        assert!(DifferentiationDual::new(Array::scalar(2.0), Array::scalar(3.0)).unwrap().is_tangent_active());
        assert!(DifferentiationDual::new_with_zero_tangent(Array::scalar(2.0)).unwrap().is_tangent_active());
        assert!(!DifferentiationDual::new_with_zero_tangent(Array::scalar(true)).unwrap().is_tangent_active());
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let tangent_reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0_f32)));
        assert!(!DifferentiationDual::new_with_zero_tangent(reference.clone()).unwrap().is_tangent_active());
        assert!(DifferentiationDual::new(reference, tangent_reference).unwrap().is_tangent_active());
        let context = TracingContext::<Array, ArrayOperation<Array>>::new();
        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let dynamic = context.input(ArrayType::new(
            DataType::F32,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("extent", bounds))]),
        ));
        assert!(!DifferentiationDual::new_with_zero_tangent(dynamic.clone()).unwrap().is_tangent_active());
        assert!(DifferentiationDual::new(dynamic.clone(), dynamic).unwrap().is_tangent_active());
    }

    #[test]
    fn test_linearization_new() {
        let primal = boundary_program(&[DataType::F64], &[Array::scalar(2.0_f64)]);
        let tangent = boundary_program(&[DataType::F64], &[Array::scalar(0.0_f64)]);
        let linearization = Linearization::new(primal.clone(), tangent.clone(), 0).unwrap();
        assert_eq!(linearization.primal().to_string(), primal.to_string());
        assert_eq!(linearization.tangent().to_string(), tangent.to_string());
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(linearization.pushforward().to_string(), tangent.to_string());
        let cloned = linearization.clone();
        assert!(Arc::ptr_eq(cloned.primal(), linearization.primal()));
        assert!(Arc::ptr_eq(cloned.tangent(), linearization.tangent()));
        assert!(Arc::ptr_eq(&linearization.pushforward(), linearization.tangent()));
        let (actual_primal, actual_tangent, residual_count) = linearization.into_parts();
        assert!(Arc::ptr_eq(&actual_primal, cloned.primal()));
        assert!(Arc::ptr_eq(&actual_tangent, cloned.tangent()));
        assert_eq!(actual_primal.to_string(), primal.to_string());
        assert_eq!(actual_tangent.to_string(), tangent.to_string());
        assert_eq!(residual_count, 0);
    }

    #[test]
    fn test_linearization_new_rejects_residual_counts() {
        assert!(matches!(
            Linearization::new(boundary_program(&[], &[]), boundary_program(&[], &[]), 1),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization primal program produces 0 \
                outputs which is fewer than its 1 residuals",
        ));
    }

    #[test]
    fn test_linearization_new_rejects_tangent_residual_count() {
        assert!(matches!(
            Linearization::new(boundary_program(&[], &[Array::scalar(1.0_f64)]), boundary_program(&[], &[]), 1),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization tangent program consumes 0 \
                inputs which is fewer than its 1 residuals",
        ));
    }

    #[test]
    fn test_linearization_new_rejects_input_count() {
        assert!(matches!(
            Linearization::new(boundary_program(&[DataType::F64], &[]), boundary_program(&[], &[]), 0),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization tangent program consumes 0 \
                tangent inputs while the primal program has 1 active inputs",
        ));
    }

    #[test]
    fn test_linearization_new_rejects_output_count() {
        assert!(matches!(
            Linearization::new(boundary_program(&[], &[Array::scalar(1.0_f64)]), boundary_program(&[], &[]), 0),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization tangent program produces 0 \
                outputs while the primal program has 1 nonzero differential outputs",
        ));
    }

    #[test]
    fn test_linearization_new_rejects_output_type() {
        assert!(matches!(
            Linearization::new(boundary_program(&[], &[Array::scalar(1.0_f64)]), boundary_program(&[], &[Array::scalar(1.0_f32)]), 0),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization tangent output 0 has type f32[] \
                but primal output type f64[] requires tangent type f64[]",
        ));
    }

    #[test]
    fn test_linearization_new_rejects_residual_type() {
        assert!(matches!(
            Linearization::new(boundary_program(&[], &[Array::scalar(1.0_f64)]), boundary_program(&[DataType::F32], &[]), 1),
            Err(ProgramError::MalformedProgram(message)) if message == "linearization residual 0 has type f64[] in \
                the primal program but type f32[] in the tangent program",
        ));
    }

    #[test]
    fn test_linearization_new_with_respect_to() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let last = builder.add_input(ArrayType::scalar(DataType::F32));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first, last], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.entry_region_ref().linearize(&[2, 0]).unwrap();
        let (primal, tangent, residual_count) = linearization.into_parts();
        let primal = Arc::unwrap_or_clone(primal);
        let tangent = Arc::unwrap_or_clone(tangent);
        // Constructor validation uses selected order and omits zero spaces just like the transform.
        let reconstructed =
            Linearization::new_with_respect_to(primal.clone(), tangent.clone(), residual_count, &[2, 1, 0]).unwrap();
        assert_eq!(reconstructed.primal().to_string(), primal.to_string());
        assert_eq!(reconstructed.tangent().to_string(), tangent.to_string());
        assert_eq!(reconstructed.residual_count(), residual_count);
        assert!(matches!(Linearization::new_with_respect_to(primal.clone(), tangent.clone(), residual_count, &[0, 2]),
            Err(ProgramError::MalformedProgram(message))
                if message == "linearization tangent input 0 has type f32[] but primal input type f64[] requires \
                    tangent type f64[]",
        ));
        assert!(matches!(Linearization::new_with_respect_to(primal.clone(), tangent.clone(), residual_count, &[1, 1]),
            Err(ProgramError::InvalidArgument { message })
                if message == "differentiation input index 1 is selected more than once",
        ));
        assert!(matches!(Linearization::new_with_respect_to(primal, tangent, residual_count, &[3]),
            Err(ProgramError::InvalidArgument { message })
                if message == "differentiation input index 3 is out of range for a region with 3 inputs",
        ));
    }

    #[test]
    fn test_pushforward_new() {
        let program = boundary_program(&[DataType::F64, DataType::F32], &[Array::scalar(0.0_f64)]);
        let pushforward = pushforward_with_boundary(
            program.clone(),
            vec![Array::scalar(7.0_f32)],
            &[Array::scalar(1.0_f64)],
            &[Array::scalar(2.0_f64)],
        )
        .unwrap();
        assert_eq!(pushforward.program().to_string(), program.to_string());
        assert_eq!(pushforward.residuals(), &[Array::scalar(7.0_f32)]);
        assert_eq!(pushforward.apply(vec![Array::scalar(5.0_f64)]), Ok(vec![Array::scalar(0.0_f64)]));
        let (_, actual_program, residuals, input_types, output_types, _) = pushforward.into_parts();
        assert_eq!(input_types, vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(output_types, vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(actual_program.to_string(), program.to_string());
        assert_eq!(residuals, vec![Array::scalar(7.0_f32)]);
    }

    #[test]
    fn test_pushforward_new_rejects_residual_count() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[], &[]), vec![Array::scalar(1.0_f64)], &[], &[]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward program consumes 0 inputs which \
                is fewer than its 1 residuals",
        ));
    }

    #[test]
    fn test_pushforward_new_rejects_residual_type() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[DataType::F32], &[]), vec![Array::scalar(1.0_f64)], &[], &[]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward residual 0 has type f32[] in the \
                pushforward program but carries a value of type f64[]",
        ));
    }

    #[test]
    fn test_pushforward_new_rejects_input_count() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[], &[]), vec![], &[Array::scalar(1.0_f64)], &[]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward program consumes 0 tangent inputs \
                but its public boundary has 1 nonzero differential inputs",
        ));
    }

    #[test]
    fn test_pushforward_new_rejects_input_type() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[DataType::F32], &[]), vec![], &[Array::scalar(1.0_f64)], &[]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward program tangent input 0 has type \
                f32[] but its public boundary requires tangent type f64[]",
        ));
    }

    #[test]
    fn test_pushforward_new_rejects_output_count() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[], &[]), vec![], &[], &[Array::scalar(1.0_f64)]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward program produces 0 tangent \
                outputs but its public boundary has 1 nonzero differential outputs",
        ));
    }

    #[test]
    fn test_pushforward_new_rejects_output_type() {
        assert!(matches!(
            pushforward_with_boundary(boundary_program(&[], &[Array::scalar(1.0_f32)]), vec![], &[], &[Array::scalar(1.0_f64)]),
            Err(ProgramError::MalformedProgram(message)) if message == "pushforward program tangent output 0 has type \
                f32[] but its public boundary requires tangenttype f64[]",
        ));
    }

    #[test]
    fn test_pushforward_apply() {
        // Structured inputs preserve their leaf order through a reusable multi-input pushforward.
        let function = |(a, b): (
            LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>,
            LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>,
        )| Ok(a.clone() * b + a.sin()?);
        let (_, pushforward) = differentiate_at((Array::scalar(0.5), Array::scalar(1.3))).linearize(function).unwrap();
        assert_abs_diff_eq!(
            pushforward.apply((Array::scalar(1.0), Array::scalar(0.0))).unwrap().to_f64s()[0],
            1.3 + 0.5f64.cos(),
            epsilon = 1e-9,
        );
        assert_abs_diff_eq!(
            pushforward.apply((Array::scalar(0.0), Array::scalar(1.0))).unwrap().to_f64s()[0],
            0.5,
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_pushforward_apply_rejects_aliased_tangent_references() {
        // `f(r, s) = { add_update(r, read(s)); read(r) }` over two live references.

        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let other = ArrayReference::new(Array::scalar(3.0_f32));
        let (value, pushforward) =
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Reference(other.clone())))
                .linearize(add_reference_contents)
                .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(4.0_f32)));

        // A tangent reference aliasing a reference bound at the primal boundary is rejected by identity even though
        // that reference advanced generations after linearization, and two tangents aliasing each other are rejected
        // as well. The rejections precede any interpretation, so the primal state is untouched.
        let tangent_reference = ArrayReference::new(Array::scalar(0.5_f32));
        assert!(matches!(
            pushforward.apply((
                ArrayIrValue::Reference(tangent_reference.clone()),
                ArrayIrValue::Reference(reference.clone()),
            )),
            Err(ProgramError::InvalidArgument { message })
                if message == "tangent 1 aliases a reference bound at the primal boundary of the differentiated \
                    function",
        ));
        assert!(matches!(
            pushforward.apply((
                ArrayIrValue::Reference(tangent_reference.clone()),
                ArrayIrValue::Reference(tangent_reference.clone()),
            )),
            Err(ProgramError::InvalidArgument { message })
                if message == "tangent 1 and tangent 0 bind the same reference allocation",
        ));
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
        assert_eq!(other.read(), Ok(Array::scalar(3.0_f32)));

        // Distinct tangent references push the tangent through: `ṫ = ṟ + ṡ = 0.5 + 2`.
        let other_tangent_reference = ArrayReference::new(Array::scalar(2.0_f32));
        assert_eq!(
            pushforward.apply((
                ArrayIrValue::Reference(tangent_reference.clone()),
                ArrayIrValue::Reference(other_tangent_reference),
            )),
            Ok(ArrayIrValue::Array(Array::scalar(2.5_f32))),
        );
        assert_eq!(tangent_reference.read(), Ok(Array::scalar(2.5_f32)));
    }

    #[test]
    fn test_recursive_differentiation_driver_jvp_program() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(ArrayType::scalar(DataType::F64));
        let second = builder.add_input(ArrayType::scalar(DataType::F32));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first, second], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();
        let driver = RecursiveDifferentiationDriver { driver: &EmptyRegionDriver };
        let differentiated =
            DifferentiationDriver::<EagerContext<Array, ArrayOperation<Array>>>::jvp_program(&driver, region, &[1, 0])
                .unwrap();

        // The driver preserves the requested order and shares the region's existing cached program.
        assert!(Arc::ptr_eq(&differentiated, &region.jvp_shared(&[1, 0]).unwrap()));
        assert_eq!(
            differentiated.input_types(),
            vec![
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::F32),
                ArrayType::scalar(DataType::F32),
                ArrayType::scalar(DataType::F64),
            ],
        );
        assert!(matches!(
            DifferentiationDriver::<EagerContext<Array, ArrayOperation<Array>>>::jvp_program(&driver, region, &[0, 0]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 0 is selected more than once",
        ));
    }

    #[test]
    fn test_recursive_differentiation_driver_linearize_program() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(ArrayType::scalar(DataType::F64));
        let second = builder.add_input(ArrayType::scalar(DataType::F32));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first, second], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();
        let driver = RecursiveDifferentiationDriver { driver: &EmptyRegionDriver };
        let linearization = DifferentiationDriver::<EagerContext<Array, ArrayOperation<Array>>>::linearize_program(
            &driver,
            region,
            &[1, 0],
        )
        .unwrap();
        assert_eq!(
            linearization.tangent().input_types(),
            vec![ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F64)],
        );
        assert_eq!(
            linearization.tangent().interpret(vec![Array::scalar(3.0_f32), Array::scalar(5.0_f64)]),
            Ok(vec![Array::scalar(5.0_f64), Array::scalar(3.0_f32)]),
        );
        assert!(matches!(
            DifferentiationDriver::<EagerContext<Array, ArrayOperation<Array>>>::linearize_program(&driver, region, &[2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 2 is out of range for a region with 2 inputs",
        ));
    }

    #[test]
    fn test_differentiation_tracer_new() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let primal = Array::scalar(2.0_f64);
        let tangent = Array::scalar(3.0_f64);
        let dual = DifferentiationDual::new(primal.clone(), tangent.clone()).unwrap();
        let tracer = DifferentiationTracer::new(dual, context);
        assert_eq!(tracer.primal(), &primal);
        assert_eq!(tracer.tangent().as_value(), Some(&tangent));
        assert_eq!(tracer.dual().primal(), &primal);
        assert!(tracer.context().is_eager());
        assert_eq!(tracer.r#type().as_ref(), primal.r#type().as_ref());
        assert_eq!(format!("{tracer}"), format!("{primal} + {tangent}ε"));
        assert_eq!(format!("{tracer:?}"), format!("DifferentiationTracer {{ dual: {:?} }}", tracer.dual()));
        let (actual_primal, actual_tangent) = tracer.into_dual().into_parts();
        assert_eq!(actual_primal, primal);
        assert_eq!(actual_tangent.as_value(), Some(&tangent));
    }

    #[test]
    fn test_differentiation_tracer_equality() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let first = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(2.0), Array::scalar(3.0)).unwrap(),
            context.clone(),
        );
        assert_eq!(first, first);
        assert_eq!(first, first.clone());
        // Context stamping does not change value equality; either half of the dual does.
        let other_context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(first, DifferentiationTracer::new(first.dual().clone(), other_context));
        assert_ne!(
            first,
            DifferentiationTracer::new(
                DifferentiationDual::new(Array::scalar(4.0), Array::scalar(3.0)).unwrap(),
                context.clone()
            )
        );
        assert_ne!(
            first,
            DifferentiationTracer::new(
                DifferentiationDual::new(Array::scalar(2.0), Array::scalar(4.0)).unwrap(),
                context.clone()
            )
        );
        let zero = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(Array::scalar(2.0)).unwrap(),
            context.clone(),
        );
        assert_eq!(zero, zero.clone());
        assert_ne!(first, zero);
        assert_ne!(
            zero,
            DifferentiationTracer::new(
                DifferentiationDual::new(Array::scalar(2.0), Array::scalar(0.0)).unwrap(),
                context
            )
        );
        assert_eq!(format!("{zero}"), format!("{} + 0ε", Array::scalar(2.0)));
    }

    #[test]
    fn test_differentiation_context_fused() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert!(std::ptr::eq(context.primal(), context.tangent()));
        assert_eq!(context.primal_to_tangent(Array::scalar(3.0_f32)), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_differentiation_context_partitioned() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            Array,
            ArrayOperation<Array>,
        >::new()));
        assert!(!std::ptr::eq(context.primal(), context.tangent()));
        let primal = context.primal().lift(Array::scalar(3.0_f32)).unwrap();
        let tangent = context.primal_to_tangent(primal.clone()).unwrap();
        assert_ne!(primal, tangent);
        assert_eq!(context.tangent().import_known(&tangent), Ok(tangent));
    }

    #[test]
    fn test_differentiation_context_new() {
        let context = DifferentiationContext::<_, FusedDifferentiationPolicy>::new(EagerContext::<
            Array,
            ArrayOperation<Array>,
        >::new());
        assert!(std::ptr::eq(context.primal(), context.tangent()));
    }

    #[test]
    fn test_differentiation_context_project() {
        let fused = DifferentiationContext::fused(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let projected = fused.project::<ArrayType>();
        assert!(std::ptr::eq(projected.primal(), projected.tangent()));
        assert_eq!(projected.primal_to_tangent(Array::scalar(3.0_f32)), Ok(Array::scalar(3.0_f32)));

        // Projection must reuse the existing tangent context, including its unknown inputs and earlier transfers.
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
        >::new()));
        let primal = context.primal().lift(ArrayIrValue::Array(Array::scalar(3.0_f32))).unwrap();
        let transferred = context.primal_to_tangent(primal.clone()).unwrap();
        let unknown = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        let projected = context.project::<ArrayType>();
        assert!(!std::ptr::eq(projected.primal(), projected.tangent()));
        assert_eq!(projected.tangent().parent().import_known(&unknown), Ok(unknown));
        assert_eq!(
            projected.primal_to_tangent(ValueProjection::<ArrayType>::into_projected(primal).unwrap()),
            Ok(ValueProjection::<ArrayType>::into_projected(transferred).unwrap()),
        );
    }

    #[test]
    fn test_differentiation_context_dual_primal_to_tangent() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            Array,
            ArrayOperation<Array>,
        >::new()));
        let primal = context.primal().lift(Array::scalar(3.0)).unwrap();
        let tangent = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F64), 0),
        );
        let inputs = [
            DifferentiationDual::new(primal.clone(), tangent.clone()).unwrap(),
            DifferentiationDual::new_with_zero_tangent(primal.clone()).unwrap(),
        ];
        let outputs = context.dual_primal_to_tangent(&inputs).unwrap();

        // Both primal occurrences share the transferred value, while live and structural tangents are unchanged.
        let transferred = context.primal_to_tangent(primal.clone()).unwrap();
        assert_eq!(outputs[0].primal(), &transferred);
        assert_eq!(outputs[1].primal(), &transferred);
        assert_ne!(outputs[0].primal(), &primal);
        assert_eq!(outputs[0].tangent().as_value(), Some(&tangent));
        assert!(matches!(outputs[1].tangent(), MaybeZero::Zero(r#type) if r#type == &ArrayType::scalar(DataType::F64)));
        assert_eq!(outputs[0].primal().context().import_known(&tangent), Ok(tangent));

        let fused = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = DifferentiationDual::new(Array::scalar(3.0), Array::scalar(2.0)).unwrap();
        let outputs = fused.dual_primal_to_tangent(std::slice::from_ref(&input)).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), input.primal());
        assert_eq!(outputs[0].tangent().as_value(), input.tangent().as_value());
    }

    #[test]
    fn test_differentiation_context_bind() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(0.0_f64), Array::scalar(3.0_f64)).unwrap(),
            context.clone(),
        );
        let outputs = context.bind(SinOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal(), &Array::scalar(0.0_f64));
        assert_eq!(outputs[0].tangent().as_value(), Some(&Array::scalar(3.0_f64)));
    }

    #[test]
    fn test_differentiation_context_bind_reference_outputs() {
        let context =
            DifferentiationContext::fused(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap(),
            context.clone(),
        );

        // A reference output has no structural zero tangent, so the all-zero fast path defers to the `reference_new`
        // rule, which allocates a tangent reference holding zero instead of skipping the rule.
        let outputs = context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].primal(), ArrayIrValue::Reference(reference)
            if reference.read().unwrap() == Array::scalar(1.0_f32)));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(ArrayIrValue::Reference(reference))
            if reference.read().unwrap() == Array::scalar(0.0_f32)));
    }

    #[test]
    fn test_differentiation_context_bind_region_reference_outputs() {
        // Both branches allocate a local reference from the numeric operand and return it: `f(p, x) = new(x)`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let branch = || {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let value = builder.add_input(scalar_type.clone().into());
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let predicate = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::scalar(true))).unwrap(),
            context.clone(),
        );
        let value = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(TestValue::Array(Array::scalar(2.0_f32))).unwrap(),
            context.clone(),
        );

        // Every operand tangent is a structural zero, but the branches allocate a reference, so the all-zero fast path
        // defers to the `condition` rule and the escaping allocation carries a tangent reference holding zero instead
        // of a symbolic zero that no later store could land in.
        let outputs = context.bind(ConditionOperation::new(), vec![branch(), branch()], &[predicate, value]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].primal(), TestValue::Reference(reference)
            if reference.read().unwrap() == Array::scalar(2.0_f32)));
        assert!(matches!(outputs[0].tangent(), MaybeZero::Value(TestValue::Reference(reference))
            if reference.read().unwrap() == Array::scalar(0.0_f32)));
    }

    #[test]
    fn test_differentiation_context_bind_symbolic_zero_tangents() {
        // `stop_gradient` severs the collective's tangent input. The differentiation context must therefore bind the
        // primal collective without consulting its absent JVP rule, while preserving the live tangent of the other
        // addition operand.
        let (primal, tangent) = differentiate_at(Array::scalar(2.0))
            .jvp(Array::scalar(1.0), |input| {
                let severed = input.stop_gradient();
                let mut outputs = severed.context().bind(
                    ParallelReduceOperation::new("batch".to_string(), ParallelReductionKind::Sum),
                    Vec::new(),
                    &[severed.clone()],
                )?;
                Ok(input + outputs.remove(0))
            })
            .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![1.0]);
    }

    #[test]
    fn test_differentiation_context_bind_custom_derivative_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_jvp_regions_with_reference_state(&scalar_type);

        // A custom derivative rule may allocate and use local reference state: the rule is replayed directly when it
        // consumes the active input, so its state executes like any other primitive operation of the identity rule.
        let result = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().jvp(
            {
                let regions = regions.clone();
                move |input: DifferentiationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>, ()| {
                    let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                    Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&input))?.remove(0))
                }
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert_eq!(
            result,
            Ok((ArrayIrValue::Array(Array::scalar(1.0_f32)), ArrayIrValue::Array(Array::scalar(1.0_f32)))),
        );

        // A lifted input has a structural zero tangent. The attached rule contains references, so binding still
        // invokes it; its identity tangent evaluates to zero. This case checks the result, not whether replay is skipped.
        let result = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new().jvp(
            move |input: DifferentiationTracer<EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>>, ()| {
                let lifted = input.context().lift(ArrayIrValue::Array(Array::scalar(1.0_f32)))?;
                let operation = ArrayIrOperation::CustomJvp(CustomJvpOperation::new());
                Ok(input.context().bind(operation, regions.clone(), std::slice::from_ref(&lifted))?.remove(0))
            },
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            ArrayIrValue::Array(Array::scalar(1.0_f32)),
            (),
        );
        assert_eq!(
            result,
            Ok((ArrayIrValue::Array(Array::scalar(1.0_f32)), ArrayIrValue::Array(Array::scalar(0.0_f32)))),
        );
    }

    #[test]
    fn test_differentiation_context_resolve() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let constant = context.lift(Array::scalar(2.0)).unwrap();
        assert!(matches!(
            context.resolve(&constant),
            ValueResolution::Constant(value) if value == Array::scalar(2.0)
        ));

        let live = DifferentiationTracer::new(
            DifferentiationDual::new(Array::scalar(2.0), Array::scalar(1.0)).unwrap(),
            context.clone(),
        );
        assert!(matches!(context.resolve(&live), ValueResolution::Opaque));

        let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
        let foreign = TracingContext::<Array, ArrayOperation<Array>>::new();
        let primal = foreign.input(ArrayType::scalar(DataType::F64));
        let context = DifferentiationContext::fused(parent);
        let opaque = DifferentiationTracer::new(
            DifferentiationDual::new(primal, MaybeZero::Zero(ArrayType::scalar(DataType::F64))).unwrap(),
            context.clone(),
        );
        assert!(matches!(context.resolve(&opaque), ValueResolution::Opaque));
    }

    #[test]
    fn test_region_tangent_output_mask() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let value = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let count = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let local = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![value, reference, local, count],
                vec![Placeholder; 3],
                vec![Placeholder; 4],
            )
            .unwrap();
        let region = program.entry_region_ref();
        assert_eq!(region.tangent_output_mask(&[]), Ok(vec![true, false, true, false]));

        // Selection order changes tangent inputs, but the output mask always follows the primal outputs.
        assert_eq!(region.tangent_output_mask(&[1, 0]), Ok(vec![true, true, true, false]));
        assert_eq!(region.tangent_output_mask(&[0, 1]), Ok(vec![true, true, true, false]));
        assert_eq!(region.tangent_output_mask(&[2]), Ok(vec![true, false, true, false]));

        // Validate all selected positions, including duplicate inputs whose differential space is zero.
        assert!(matches!(region.tangent_output_mask(&[1, 1]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 1 is selected more than once",
        ));
        assert!(matches!(region.tangent_output_mask(&[2, 2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 2 is selected more than once",
        ));
        assert!(matches!(region.tangent_output_mask(&[3]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 3 is out of range for a region with 3 inputs",
        ));
    }

    #[test]
    fn test_region_jvp() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let last = builder.add_input(ArrayType::scalar(DataType::F32));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![first, last], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();

        // Primals and outputs retain source order, while tangent inputs follow the selection and omit metadata.
        let jvp = region.jvp(&[2, 1, 0]).unwrap();
        assert_eq!(
            jvp.input_types(),
            vec![
                ArrayType::scalar(DataType::F64),
                ArrayType::scalar(DataType::Boolean),
                ArrayType::scalar(DataType::F32),
                ArrayType::scalar(DataType::F32),
                ArrayType::scalar(DataType::F64),
            ]
        );
        assert_eq!(
            jvp.interpret(vec![
                Array::scalar(2.0_f64),
                Array::scalar(true),
                Array::scalar(3.0_f32),
                Array::scalar(5.0_f32),
                Array::scalar(7.0_f64),
            ]),
            Ok(vec![Array::scalar(2.0_f64), Array::scalar(3.0_f32), Array::scalar(7.0_f64), Array::scalar(5.0_f32)])
        );

        // Selecting no inputs supplies zero output tangents and retains all primal inputs.
        assert_eq!(
            region.jvp(&[]).unwrap().interpret(vec![
                Array::scalar(2.0_f64),
                Array::scalar(true),
                Array::scalar(3.0_f32),
            ]),
            Ok(vec![Array::scalar(2.0_f64), Array::scalar(3.0_f32), Array::scalar(0.0_f64), Array::scalar(0.0_f32)])
        );
        assert!(matches!(region.jvp(&[0, 0]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 0 is selected more than once",
        ));
        assert!(matches!(region.jvp(&[1, 1]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 1 is selected more than once",
        ));
        assert!(matches!(region.jvp(&[3]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 3 is out of range for a region with 3 inputs",
        ));
    }

    #[test]
    fn test_region_jvp_reference_input_order() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let scalar = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(ReferenceType::new(scalar.clone()).into());
        let value = builder.add_input(scalar.into());
        let contents =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let product = builder
            .add_instruction(
                ArrayOperation::<Array>::from(MulOperation::new()),
                Vec::new(),
                vec![contents, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![product],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(4.0_f32)));
        let tangent_reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(6.0_f32)));
        let primals = vec![reference, Array::scalar(2.0_f32).into()];
        let tangents = vec![Array::scalar(5.0_f32).into(), tangent_reference];

        // The selected boundary puts the ordinary tangent before the reference tangent, despite primal order.
        let region = program.entry_region_ref();
        let jvp = region.jvp(&[1, 0]).unwrap();
        let mut inputs = primals.clone();
        inputs.extend(tangents.clone());
        assert_eq!(jvp.interpret(inputs), Ok(vec![Array::scalar(8.0_f32).into(), Array::scalar(32.0_f32).into()]));
    }

    #[test]
    fn test_region_jvp_omits_inactive_reference_output_tangents() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The compact region boundary retains the inactive primal reference and omits its tangent.
        let inactive_jvp = program.entry_region_ref().jvp(&[]).unwrap();
        assert!(inactive_jvp.instructions().is_empty());
        assert_eq!(inactive_jvp.input_ids().len(), 1);
        assert_eq!(inactive_jvp.output_ids(), inactive_jvp.input_ids());

        // A forwarded active reference has its tangent reference forwarded by identity, with no allocation: the fused
        // program stages nothing and returns its two inputs as its two outputs.
        let jvp = program.entry_region_ref().jvp(&[0]).unwrap();
        assert!(jvp.instructions().is_empty());
        assert_eq!(jvp.input_types().len(), 2);
        assert_eq!(jvp.output_ids(), jvp.input_ids());
    }

    #[test]
    fn test_region_jvp_rejects_reference_view_outputs() {
        let reference_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [4]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![view], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // A derived view output would need the same view applied to the tangent root, which is not supported.
        assert!(matches!(
            program.entry_region_ref().jvp(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "output 0 is a derived view of a reference and cannot be differentiated; return the \
                    viewed reference and apply the view outside the differentiated program",
        ));
    }

    #[test]
    fn test_region_jvp_shared() {
        // A shared region is differentiated once and reused by every copy of it, which is what removes the repeated
        // re-transformation that programs attaching one shared `condition` branch or `scan` body would otherwise pay.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        let retained = callee.entry_region_ref().jvp_shared(&[0]).unwrap();
        assert_eq!(retained.to_string(), callee.jvp().unwrap().to_string());
        assert!(Arc::ptr_eq(&callee.entry_region_ref().jvp_shared(&[0]).unwrap(), &retained));

        // Two independently built programs that intern the same callee share its retained program, because importing
        // a region copies its complete reachable contents and therefore carries its transforms along.
        let mut first_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first_region = first_builder.intern_callee(&callee, None).unwrap();
        let mut second_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let second_region = second_builder.intern_callee(&callee, None).unwrap();
        assert!(Arc::ptr_eq(
            &RegionRef::new(&first_builder.regions, first_region).unwrap().jvp_shared(&[0]).unwrap(),
            &retained,
        ));
        assert!(Arc::ptr_eq(
            &RegionRef::new(&second_builder.regions, second_region).unwrap().jvp_shared(&[0]).unwrap(),
            &retained,
        ));

        // The initial request produces the artifact; direct and interned-region requests reuse it three times.
        let statistics = callee.entry_region_ref().transform_statistics::<JvpTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 3));

        // A region whose contents are genuinely rewritten starts over with a freshly derived program.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.add_instruction(SinOperation::new(), Vec::new(), vec![output], None).unwrap();
        let with_dead_work =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let before = with_dead_work.entry_region_ref().jvp_shared(&[0]).unwrap();
        let simplified = with_dead_work.simplified().unwrap();
        let after = simplified.entry_region_ref().jvp_shared(&[0]).unwrap();
        assert!(!Arc::ptr_eq(&after, &before));
        assert_eq!(after.to_string(), retained.to_string());
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_jvp_shared_debug_recheck_detects_corrupted_cached_program() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // Publish an artifact that disagrees with what differentiating this region produces, which is exactly the
        // state a nondeterministic `jvp` rule would leave behind: a retained derivative of a program the region does
        // not compute.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(CosOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let unrelated = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        program.entry_region_ref().insert_transform_artifact_for_testing::<JvpTransform, _>(
            JvpAndLinearizationTransformArguments { input_indices: vec![0] },
            TransformArtifact::new(vec![unrelated.clone()], ()),
        );

        // The recheck runs on the hit and reports the contract violation rather than serving the wrong derivative.
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| program.entry_region_ref().jvp_shared(&[0])))
                .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert_eq!(
            message,
            &transform_mismatch_message::<JvpTransform>(&[&unrelated], &[&program.jvp().unwrap()], "()", "()")
        );
    }

    #[test]
    fn test_region_jvp_shared_keys_artifacts_by_effective_selection() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();

        // Distinct effective selections derive distinct programs, while repeated selections share an artifact.
        // Selecting every input in source order retains the complete tangent boundary.
        let partial = region.jvp_shared(&[0]).unwrap();
        let full = region.jvp_shared(&[0, 1]).unwrap();
        assert_eq!(partial.input_types().len(), 3);
        assert_eq!(full.input_types().len(), 4);
        assert!(!Arc::ptr_eq(&partial, &full));
        assert!(Arc::ptr_eq(&region.jvp_shared(&[0]).unwrap(), &partial));
        assert!(Arc::ptr_eq(&region.jvp_shared(&[0, 1]).unwrap(), &full));
        let reversed = region.jvp_shared(&[1, 0]).unwrap();
        assert!(!Arc::ptr_eq(&reversed, &full));
        assert!(Arc::ptr_eq(&region.jvp_shared(&[1, 0]).unwrap(), &reversed));
        assert_eq!(
            reversed.interpret(vec![
                Array::scalar(2.0_f64),
                Array::scalar(3.0_f64),
                Array::scalar(5.0_f64),
                Array::scalar(7.0_f64),
            ]),
            Ok(vec![Array::scalar(6.0_f64), Array::scalar(31.0_f64)])
        );
        assert_eq!(
            partial.interpret(vec![Array::scalar(2.0), Array::scalar(3.0), Array::scalar(1.0)]),
            Ok(vec![Array::scalar(6.0), Array::scalar(3.0)]),
        );
        assert!(matches!(
            region.jvp_shared(&[2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 2 is out of range for a region with 2 inputs",
        ));

        // Selecting a zero-space input adds no tangent slot and does not change the cached artifact.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let value = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();
        let requested = region.jvp_shared(&[0, 1]).unwrap();
        assert_eq!(requested.input_types().len(), 3);
        assert!(Arc::ptr_eq(&region.jvp_shared(&[0]).unwrap(), &requested));
    }

    #[test]
    fn test_region_linearize() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();
        let linearization = region.linearize(&[2, 1, 0]).unwrap();
        let primal_outputs = linearization
            .primal()
            .interpret(vec![Array::scalar(2.0_f64), Array::scalar(true), Array::scalar(3.0_f64)])
            .unwrap();
        assert_eq!(primal_outputs[0], Array::scalar(6.0_f64));
        let mut tangent_inputs = vec![Array::scalar(5.0_f64), Array::scalar(7.0_f64)];
        tangent_inputs.extend_from_slice(&primal_outputs[1..]);
        // Tangents are [dright, dleft], so dproduct = 2 * 5 + 3 * 7, not 2 * 7 + 3 * 5.
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![Array::scalar(31.0_f64)]));
        let mut pullback_inputs = vec![Array::scalar(1.0_f64)];
        pullback_inputs.extend_from_slice(&primal_outputs[1..]);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![Array::scalar(2.0_f64), Array::scalar(3.0_f64)])
        );

        let empty = region.linearize(&[]).unwrap();
        let outputs = empty
            .primal()
            .interpret(vec![Array::scalar(2.0_f64), Array::scalar(true), Array::scalar(3.0_f64)])
            .unwrap();
        assert_eq!(empty.tangent().interpret(outputs[1..].to_vec()), Ok(vec![Array::scalar(0.0_f64)]));
        assert!(matches!(region.linearize(&[2, 2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 2 is selected more than once",
        ));
        assert!(matches!(region.linearize(&[1, 1]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 1 is selected more than once",
        ));
        assert!(matches!(region.linearize(&[3]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "differentiation input index 3 is out of range for a region with 3 inputs",
        ));
    }

    #[test]
    fn test_region_linearize_omits_inactive_reference_output_tangents() {
        let reference_type = ReferenceType::new(ArrayType::scalar(DataType::F32));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The primal carries the inactive reference while the tangent program has no inputs or outputs.
        let inactive = program.entry_region_ref().linearize(&[]).unwrap();
        assert_eq!(inactive.residual_count(), 0);
        assert_eq!(inactive.primal().output_ids(), inactive.primal().input_ids());
        assert!(inactive.tangent().input_ids().is_empty());
        assert!(inactive.tangent().output_ids().is_empty());

        // A forwarded active reference has its tangent reference forwarded by identity through the tangent program.
        let linearization = program.entry_region_ref().linearize(&[0]).unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert!(linearization.tangent().instructions().is_empty());
        assert_eq!(linearization.tangent().output_ids(), linearization.tangent().input_ids());
    }

    #[test]
    fn test_region_linearize_rejects_reference_view_outputs() {
        let reference_type = ReferenceType::new(ArrayType::new_static(DataType::F32, [4]));
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.into());
        let view = builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![reference],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![view], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // The derived-view rejection applies to linearization exactly as it applies to the fused program.
        assert!(matches!(
            program.entry_region_ref().linearize(&[0]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "output 0 is a derived view of a reference and cannot be differentiated; return the \
                    viewed reference and apply the view outside the differentiated program",
        ));
    }

    #[test]
    fn test_region_linearize_retains_no_zero_residuals_for_write_only_tangent_references() {
        // `f(r, x) = { write(r, x); x }` stores into the reference without reading it back, so the tangent reference is
        // reached by no tangent output even though its state is live. A reference has no value cotangent to construct,
        // so the dead-tangent-input residual pass skips it and nothing crosses as a residual.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type.clone()).into());
        let value = builder.add_input(scalar_type.into());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let linearization = program.entry_region_ref().linearize(&[0, 1]).unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let reference_write %0 %1
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_region_linearize_omits_inactive_tangent_inputs() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();

        // The tangent half consumes one tangent input for the active input only, followed by the residuals, and
        // computes `ẏ = ẋ · right` without a term for the inactive input.
        let linearization = region.linearize(&[0]).unwrap();
        assert_eq!(linearization.tangent().input_ids().len(), 1 + linearization.residual_count());
        assert_eq!(linearization.tangent().output_ids().len(), 1);
        let primal_outputs = linearization.primal().interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        assert_eq!(primal_outputs[0], Array::scalar(6.0));
        let mut tangent_inputs = vec![Array::scalar(1.0)];
        tangent_inputs.extend_from_slice(&primal_outputs[1..]);
        assert_eq!(linearization.tangent().interpret(tangent_inputs), Ok(vec![Array::scalar(3.0)]));
    }

    #[test]
    fn test_region_linearize_reference_input_order() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let scalar = ArrayType::scalar(DataType::F32);
        let reference = builder.add_input(ReferenceType::new(scalar.clone()).into());
        let value = builder.add_input(scalar.into());
        let contents =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let product = builder
            .add_instruction(
                ArrayOperation::<Array>::from(MulOperation::new()),
                Vec::new(),
                vec![contents, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![product],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(4.0_f32)));
        let tangent_reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(6.0_f32)));
        let primals = vec![reference, Array::scalar(2.0_f32).into()];
        let tangents = vec![Array::scalar(5.0_f32).into(), tangent_reference];

        // The selected boundary puts the ordinary tangent before the reference tangent, despite primal order.
        let region = program.entry_region_ref();
        let linearization = region.linearize(&[1, 0]).unwrap();
        let outputs = linearization.primal().interpret(primals).unwrap();
        assert_eq!(outputs[0], Array::scalar(8.0_f32).into());
        let mut inputs = tangents;
        inputs.extend_from_slice(&outputs[1..]);
        assert_eq!(linearization.tangent().interpret(inputs), Ok(vec![Array::scalar(32.0_f32).into()]));
    }

    #[test]
    fn test_region_linearize_shared() {
        // A shared callee is linearized once and reused by every copy of its sealed region, which is what removes the
        // repeated re-transformation that outer programs interning one callee would otherwise pay.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        let (primal, tangent, residual_count) = callee.entry_region_ref().linearize_shared(&[0]).unwrap().into_parts();
        assert_eq!(residual_count, 1);
        assert_eq!(primal.to_string(), callee.linearize().unwrap().primal().to_string());
        assert_eq!(tangent.to_string(), callee.linearize().unwrap().tangent().to_string());

        // Two independently built programs that intern the same callee share its retained linearization, because
        // importing a region copies its complete reachable contents and therefore carries its transforms along.
        let mut first_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first_region = first_builder.intern_callee(&callee, None).unwrap();
        let mut second_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let second_region = second_builder.intern_callee(&callee, None).unwrap();
        let first = RegionRef::new(&first_builder.regions, first_region).unwrap().linearize_shared(&[0]).unwrap();
        let second = RegionRef::new(&second_builder.regions, second_region).unwrap().linearize_shared(&[0]).unwrap();
        assert!(Arc::ptr_eq(first.primal(), &primal));
        assert!(Arc::ptr_eq(first.tangent(), &tangent));
        assert!(Arc::ptr_eq(second.primal(), &primal));
        assert!(Arc::ptr_eq(second.tangent(), &tangent));

        // The original region produces the artifact, and both independently interned copies reuse it.
        let statistics = callee.entry_region_ref().transform_statistics::<LinearizationTransform>().unwrap();
        assert_eq!((statistics.productions, statistics.hits), (1, 2));

        // Simplification rebuilds every region, so it keeps the retained linearization only when the rebuild left the
        // region's contents untouched. A program carrying dead work is genuinely rewritten and must not reuse it.
        let simplified = callee.simplified().unwrap();
        assert!(Arc::ptr_eq(simplified.entry_region_ref().linearize_shared(&[0]).unwrap().primal(), &primal));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.add_instruction(SinOperation::new(), Vec::new(), vec![output], None).unwrap();
        let with_dead_work =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let before = with_dead_work.entry_region_ref().linearize_shared(&[0]).unwrap();
        let after = with_dead_work.simplified().unwrap();
        let after = after.entry_region_ref().linearize_shared(&[0]).unwrap();
        assert!(!Arc::ptr_eq(after.primal(), before.primal()));
        assert_eq!(after.primal().to_string(), primal.to_string());

        // Instantiating a program's type identities rewrites its types, so the rebuilt program starts over with no
        // retained transforms even though the source program keeps its own.
        let bounds = DimensionBounds::non_negative(Some(16)).unwrap();
        let dynamic_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("formal", bounds))]),
        );
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(dynamic_type);
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let dynamic_callee =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let formal = dynamic_callee.entry_region_ref().linearize_shared(&[0]).unwrap();
        let actual_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("actual", bounds))]),
        );
        let instantiated = dynamic_callee.with_instantiated_type_identities(&[actual_type]).unwrap().into_owned();
        let instantiated = instantiated.entry_region_ref().linearize_shared(&[0]).unwrap();
        assert!(!Arc::ptr_eq(instantiated.primal(), formal.primal()));
        assert!(Arc::ptr_eq(
            dynamic_callee.entry_region_ref().linearize_shared(&[0]).unwrap().primal(),
            formal.primal()
        ));
    }

    #[test]
    fn test_region_linearize_shared_invalidates_rebased_attached_regions() {
        // A region's retained transforms cover its complete reachable contents, but the identifiers it attaches its
        // descendants by are relative to the arena it is sealed in. Program `first` attaches the sine branch as both
        // branches of a condition, so its entry's linearization is the derivative of sine.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let sine = builder.import_region(unary_program(ArrayOperation::Sin(SinOperation::new())).entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![sine, sine], vec![predicate, input], None)
            .unwrap()[0];
        let first = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let retained = first.entry_region_ref().linearize_shared(&[0, 1]).unwrap();

        // Re-sealing a copy of that entry into an arena whose region 0 is the cosine branch changes what the copy
        // computes, so it must not be served the transforms derived from the sine branch.
        let rebased = Program::<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>::new(
            vec![Placeholder; 2],
            vec![Placeholder],
            vec![
                unary_program(ArrayOperation::Cos(CosOperation::new())).entry_region().clone(),
                first.entry_region().clone(),
            ],
            RegionId::new(1),
        )
        .unwrap();
        let derived = rebased.entry_region_ref().linearize_shared(&[0, 1]).unwrap();
        assert!(!Arc::ptr_eq(derived.primal(), retained.primal()));
        assert!(!Arc::ptr_eq(derived.tangent(), retained.tangent()));

        // The freshly derived tangent program differentiates the cosine branch the rebased arena actually attaches,
        // which is the wrong-derivative failure that serving the retained artifact would produce.
        assert_eq!(derived.tangent().to_string(), rebased.linearize().unwrap().tangent().to_string());
        assert_ne!(derived.tangent().to_string(), retained.tangent().to_string());

        // The source program keeps its own retained artifact, because only the re-sealed copy was rebased.
        assert!(Arc::ptr_eq(first.entry_region_ref().linearize_shared(&[0, 1]).unwrap().primal(), retained.primal()));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_linearize_shared_debug_recheck_detects_corrupted_cached_linearization() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // Publish an artifact that disagrees with what linearizing this region produces, which is exactly the state a
        // nondeterministic `jvp` rule would leave behind: a retained derivative of a program the region does not
        // compute.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(CosOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let unrelated = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        program.entry_region_ref().insert_transform_artifact_for_testing::<LinearizationTransform, _>(
            JvpAndLinearizationTransformArguments { input_indices: vec![0] },
            TransformArtifact::new(vec![unrelated.clone(), unrelated.clone()], 0),
        );

        // The recheck runs on the hit and reports the contract violation rather than serving the wrong derivative.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program.entry_region_ref().linearize_shared(&[0])
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        let fresh = program.linearize().unwrap();
        assert_eq!(
            message,
            &transform_mismatch_message::<LinearizationTransform>(
                &[&unrelated, &unrelated],
                &[fresh.primal(), fresh.tangent()],
                "0",
                &format!("{:?}", fresh.residual_count())
            )
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_linearize_shared_debug_recheck_detects_operation_metadata_changes() {
        // This test pins the metadata-fingerprint contract of `Operation::render`. The transform cache debugging
        // diagnostic compares programs purely by rendering, so an operation that fails to render its semantics-bearing
        // payload is an operation whose corruption the diagnostic cannot see. `tag` is the canonical example: its key
        // is invisible to types and structure, yet rematerialization policies classify residuals by it. The two regions
        // below differ only in that key, so this test panics exactly when `TagOperation::render` renders it; a
        // name-only rendering would make the corrupted artifact compare equal and serve silently.

        let program = tagged_program("saved");
        let other = tagged_program("recomputed");

        // The two programs differ only in operation metadata, so the corruption below is invisible to everything
        // except rendering that carries the key.
        assert_ne!(program.to_string(), other.to_string());

        // Publish the *other* region's genuine linearization against this region, which is the state a `jvp` rule that
        // is not a structural function of its operation would leave behind.
        let (primal, tangent, residual_count) = other.entry_region_ref().linearize_shared(&[0]).unwrap().into_parts();
        program.entry_region_ref().insert_transform_artifact_for_testing::<LinearizationTransform, _>(
            JvpAndLinearizationTransformArguments { input_indices: vec![0] },
            TransformArtifact::new(vec![primal.clone(), tangent.clone()], residual_count),
        );

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program.entry_region_ref().linearize_shared(&[0])
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        let fresh = program.linearize().unwrap();
        assert_eq!(
            message,
            &transform_mismatch_message::<LinearizationTransform>(
                &[&primal, &tangent],
                &[fresh.primal(), fresh.tangent()],
                &format!("{residual_count:?}"),
                &format!("{:?}", fresh.residual_count())
            )
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_linearize_shared_debug_recheck_detects_constant_payload_changes() {
        // This test pins the constant-payload part of the rendering contract at a cache hit. The two regions below
        // differ only in one `Atom::Constant` literal, so this test panics exactly when `Program::render` carries that
        // payload into the rendering compared by the recheck.

        let program = scaled_program(2.0);
        let other = scaled_program(3.0);

        // The ordinary rendering is semantically complete enough to distinguish the embedded literal.
        assert_ne!(program.to_string(), other.to_string());

        // Publish the *other* region's genuine linearization against this region, which is the state a `jvp` rule
        // that is not a structural function of the constants it embeds would leave behind.
        let (primal, tangent, residual_count) = other.entry_region_ref().linearize_shared(&[0]).unwrap().into_parts();
        program.entry_region_ref().insert_transform_artifact_for_testing::<LinearizationTransform, _>(
            JvpAndLinearizationTransformArguments { input_indices: vec![0] },
            TransformArtifact::new(vec![primal.clone(), tangent.clone()], residual_count),
        );

        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program.entry_region_ref().linearize_shared(&[0])
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        let fresh = program.linearize().unwrap();
        assert_eq!(
            message,
            &transform_mismatch_message::<LinearizationTransform>(
                &[&primal, &tangent],
                &[fresh.primal(), fresh.tangent()],
                &format!("{residual_count:?}"),
                &format!("{:?}", fresh.residual_count())
            )
        );
    }

    #[test]
    fn test_region_linearize_shared_does_not_retain_its_source_cache() {
        // `to_program` deliberately preserves the source entry region's cache. Linearizing that materialized copy is
        // therefore the strongest ownership-cycle attempt available through today's built-in transform API.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let source_cache = program.entry_region().transform_cache.downgrade();
        let materialized = program.entry_region_ref().to_program();
        assert!(program.entry_region().transform_cache.ptr_eq(&materialized.entry_region().transform_cache));
        let (primal, tangent, _) = materialized.entry_region_ref().linearize_shared(&[0]).unwrap().into_parts();

        drop(program);
        drop(materialized);
        assert!(!source_cache.is_alive());

        // The returned programs can outlive the source cache because built-in linearization constructs a fresh entry
        // and imports only strict descendants from the acyclic source arena; neither artifact embeds the source root.
        drop(primal);
        drop(tangent);
    }

    #[test]
    fn test_region_linearize_shared_normalizes_zero_spaces() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::Boolean));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();
        // Cache identity depends on live tangent order, but not on where zero-space indices appear.
        let reversed = region.linearize_shared(&[2, 1, 0]).unwrap();
        let equivalent = region.linearize_shared(&[1, 2, 0]).unwrap();
        let source_order = region.linearize_shared(&[0, 2]).unwrap();
        assert!(Arc::ptr_eq(reversed.primal(), equivalent.primal()));
        assert!(Arc::ptr_eq(reversed.tangent(), equivalent.tangent()));
        assert!(!Arc::ptr_eq(reversed.primal(), source_order.primal()));
        assert!(!Arc::ptr_eq(reversed.tangent(), source_order.tangent()));
    }

    #[test]
    fn test_region_linearize_shared_keys_artifacts_by_effective_selection() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();

        // The tangent half consumes one tangent input for the active input only, followed by the residuals, and
        // computes `ẏ = ẋ · right` without a term for the inactive input.
        // The retained linearization is keyed by the effective selection as well.
        let (partial_primal, ..) = region.linearize_shared(&[0]).unwrap().into_parts();
        let (full_primal, ..) = region.linearize_shared(&[0, 1]).unwrap().into_parts();
        assert!(!Arc::ptr_eq(&partial_primal, &full_primal));
        assert!(Arc::ptr_eq(region.linearize_shared(&[0]).unwrap().primal(), &partial_primal));
    }

    #[test]
    fn test_program_jvp() {
        // Test that the fused JVP program of `f(x) = sin(x)` presents the `[x, ẋ] ↦ [sin(x), cos(x) · ẋ]` boundary.
        // The primal input leads, one fresh tangent input follows, and the outputs are the primal output followed by
        // its tangent.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let fused = program.jvp().unwrap();
        assert_eq!(fused.input_ids().len(), 2);
        assert_eq!(fused.output_ids().len(), 2);
        assert_eq!(
            fused.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = sin %0
                    %3:f64[] = cos %0
                    %4:f64[] = mul %3 %1
                in (%2, %4)
            "}
            .trim_end(),
        );
        let outputs = fused.interpret(vec![Array::scalar(3.0), Array::scalar(1.0)]).unwrap();
        assert_eq!(outputs, vec![Array::scalar(3.0f64.sin()), Array::scalar(3.0f64.cos())]);
    }

    #[test]
    fn test_program_jvp_preserves_source_provenance() {
        // Every instruction the fused JVP program stages for one source instruction records that source instruction as
        // its origin, both for the primal replay and for the instructions the tangent rule contributes. The two source
        // instructions carry distinct scopes so per-instruction attribution is observable rather than incidental.
        let first = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let second = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let sine =
            builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], Some(first.clone())).unwrap()[0];
        let squared = builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![sine, sine], Some(second.clone()))
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let fused = program.jvp().unwrap();
        assert_eq!(
            fused
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>(),
            vec![
                ("sin", first.clone()),
                ("cos", first.clone()),
                ("mul", first),
                ("mul", second.clone()),
                ("mul", second.clone()),
                ("mul", second.clone()),
                ("add", second),
            ],
        );
    }

    #[test]
    fn test_program_jvp_with_local_references_matches_discharged_program() {
        // `f(x) = freeze(add_update(new(x), x)) = 2x` threads its state through a local reference. Forward mode
        // differentiates the reference operations directly: the fused program keeps the reference-typed atoms (with a
        // tangent reference allocated alongside the primal one) and agrees with the jvp of the discharged program.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let jvp = program.jvp().unwrap();
        assert!(jvp.entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
        assert_eq!(jvp.input_types().len(), 2);
        assert_eq!(jvp.output_types().len(), 2);

        let inputs = vec![ArrayIrValue::Array(Array::scalar(3.0_f32)), ArrayIrValue::Array(Array::scalar(2.0_f32))];
        let expected = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .jvp()
            .unwrap()
            .interpret(inputs.clone())
            .unwrap();
        assert_eq!(
            expected,
            vec![ArrayIrValue::Array(Array::scalar(6.0_f32)), ArrayIrValue::Array(Array::scalar(4.0_f32))],
        );
        assert_eq!(jvp.interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_program_jvp_replays_custom_derivative_regions_with_local_reference_state() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let wrapped = {
            let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
            let regions = custom_jvp_regions_with_reference_state(&scalar_type)
                .iter()
                .map(|region| builder.import_region(region.entry_region_ref()))
                .collect::<Vec<_>>();
            let input = builder.add_input(scalar_type.clone());
            let outputs = builder
                .add_instruction(ArrayIrOperation::CustomJvp(CustomJvpOperation::new()), regions, vec![input], None)
                .unwrap()
                .to_vec();
            builder
                .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                    outputs,
                    vec![Placeholder],
                    vec![Placeholder],
                )
                .unwrap()
        };

        // The entry region is pure because the state lives in the dormant rule region. Forward mode replays that rule
        // when it fires on the live tangent, so the fused program stages the rule's local allocation and read, which
        // execute like any other primitive operations of the identity rule.
        assert!(wrapped.effects().classes().is_empty());
        let jvp = wrapped.jvp().unwrap();
        assert!(jvp.entry_region_ref().contains_effect_in_closure(crate::programs::EffectClass::OrderedState));
        let inputs = vec![ArrayIrValue::Array(Array::scalar(1.0_f32)), ArrayIrValue::Array(Array::scalar(2.0_f32))];
        assert_eq!(jvp.interpret(inputs.clone()), Ok(inputs));
    }

    #[test]
    fn test_program_jvp_uses_the_primal_array_operation_family() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (_, program) = NestedTracingContext::trace(
            context,
            |inputs: Vec<_>| Ok(vec![inputs[0].dot(&inputs[0], &DotDimensionNumbers::inner_product())]),
            vec![Array::vector(vec![1.0, 2.0, 3.0]).r#type().into_owned()],
        )
        .unwrap();
        let program = program.into_simplified().unwrap().jvp().unwrap();

        // The fused JVP remains in the ordinary primal operation family instead of introducing a capture-keyed
        // linear operation family.
        let _: &Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> = &program;
        assert_eq!(program.input_ids().len(), 2);
        assert_eq!(program.output_ids().len(), 2);
        assert_eq!(
            program.interpret_in_context(
                &context,
                vec![Array::vector(vec![1.0, 2.0, 3.0]), Array::vector(vec![1.0, 1.0, 1.0])],
            ),
            Ok(vec![Array::scalar(14.0), Array::scalar(12.0)]),
        );
    }

    #[test]
    fn test_program_jvp_materializes_boundary_zeros() {
        // Test that structural zero tangents are materialized as typed `zero` instructions only at the output boundary,
        // preserving the `(primal_outputs ++ tangent_outputs)` contract. Both zero producers are covered: the
        // constant-valued output's tangent is a *derived* zero (a constant is connected to no input tangent) and the
        // `stop_gradient` output's tangent is a *rule-returned* zero, and exactly one `zero` instruction is staged per
        // zero tangent output with none staged mid-program.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let constant = builder.add_constant(Array::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
        let severed = builder.add_instruction(StopGradientOperation::new(), Vec::new(), vec![scaled], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![scaled, constant, severed], vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        let fused = program.jvp().unwrap();
        assert_eq!(fused.input_ids().len(), 2);
        assert_eq!(fused.output_ids().len(), 6);
        let zero_count =
            fused.instructions().iter().filter(|instruction| instruction.operation().name() == "zero").count();
        assert_eq!(zero_count, 2);
        assert_eq!(
            fused
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "zero")
                .flat_map(|instruction| instruction.outputs().iter().copied())
                .collect::<Vec<_>>(),
            fused.output_ids()[4..],
        );
        assert_eq!(
            fused
                .instructions()
                .iter()
                .rev()
                .take(2)
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero", "zero"],
        );
        let outputs = fused.interpret(vec![Array::scalar(3.0), Array::scalar(1.0)]).unwrap();
        assert_eq!(
            outputs,
            vec![
                Array::scalar(6.0),
                Array::scalar(2.0),
                Array::scalar(6.0),
                Array::scalar(2.0),
                Array::scalar(0.0),
                Array::scalar(0.0),
            ],
            "the fused outputs must be the primal outputs [3 * 2, 2, 3 * 2] followed by the tangents [2 * 1, 0, 0]",
        );
    }

    #[test]
    fn test_program_jvp_with_respect_to() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert_eq!(
            program.jvp_with_respect_to(&[]).unwrap().interpret(vec![Array::scalar(2.0_f64)]),
            Ok(vec![Array::scalar(2.0_f64), Array::scalar(0.0_f64)])
        );
    }

    #[test]
    fn test_program_linearize() {
        // Test that directly linearizing `f(x) = sin(x)` produces the primal sub-program `x ↦ (sin(x), cos(x))`,
        // whose trailing output is the `cos(x)` residual, and the linear tangent sub-program `(ẋ, r) ↦ r · ẋ`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = sin %0
                    %2:f64[] = cos %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let primal_outputs = linearization.primal().interpret(vec![Array::scalar(3.0)]).unwrap();
        assert_eq!(primal_outputs, vec![Array::scalar(3.0f64.sin()), Array::scalar(3.0f64.cos())]);
        let tangent_outputs =
            linearization.tangent().interpret(vec![Array::scalar(1.0), Array::scalar(3.0f64.cos())]).unwrap();
        assert_eq!(tangent_outputs, vec![Array::scalar(3.0f64.cos())]);

        // The ordinary direct-linearization boundary carries `cos(x)` as a residual, so transposing its tangent map
        // must not invoke known-intermediate replay. In particular, the primal-only `cos` chain must be absent from
        // the pullback: it runs once in the primal program and crosses the boundary as the pullback's residual input.
        let pullback = linearization.pullback().unwrap();
        assert!(
            pullback.instructions().iter().all(|instruction| instruction.operation().name() != "cos"),
            "ordinary linearize -> transpose unexpectedly replayed a primal-only producer:\n{pullback}",
        );
        assert_eq!(
            pullback.interpret(vec![Array::scalar(1.0), Array::scalar(3.0f64.cos())]).unwrap(),
            vec![Array::scalar(3.0f64.cos())],
        );
    }

    #[test]
    fn test_program_linearize_tangent_allocation_is_fresh_per_invocation() {
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let zero = builder.add_constant(Array::scalar(0.0_f32).into());
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program: Program<_, _, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>> =
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        let mut primal_outputs = linearization.primal().interpret(vec![Array::scalar(3.0_f32).into()]).unwrap();
        let residuals = primal_outputs.split_off(1);
        assert_eq!(primal_outputs, vec![Array::scalar(3.0_f32).into()]);
        let outputs = [2.0_f32, 5.0, 2.0].map(|tangent| {
            let mut inputs = vec![Array::scalar(tangent).into()];
            inputs.extend(residuals.clone());
            linearization.tangent().interpret(inputs).unwrap()
        });
        assert_eq!(
            outputs,
            [
                vec![Array::scalar(2.0_f32).into()],
                vec![Array::scalar(5.0_f32).into()],
                vec![Array::scalar(2.0_f32).into()],
            ],
        );
        assert_eq!(
            linearization
                .primal()
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "reference_new")
                .count(),
            1,
        );
        assert_eq!(
            linearization
                .tangent()
                .instructions()
                .iter()
                .filter(|instruction| instruction.operation().name() == "reference_new")
                .count(),
            1,
        );
    }

    #[test]
    fn test_program_linearize_with_local_references() {
        // `f(x) = freeze(add_update(new(x), x)) = 2x` reads, modifies, and writes a local reference. Linearization
        // keeps the primal accesses in the primal program and stages the tangent accesses, over a tangent reference
        // allocated from the tangent input, into the tangent program. Nothing crosses as a residual: the tangent side
        // never needs the primal reference, so the residual reference classification is empty.
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                    reference_add_update %1 %0
                    %2:f32[] = reference_freeze %1
                in (%2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:ref<f32[]> = reference_new %0
                    reference_add_update %1 %0
                    %2:f32[] = reference_freeze %1
                in (%2)
            "}
            .trim_end(),
        );

        // The two halves agree with the fused forward-mode program at the same point.
        let inputs = vec![ArrayIrValue::Array(Array::scalar(3.0_f32)), ArrayIrValue::Array(Array::scalar(2.0_f32))];
        let expected = program.jvp().unwrap().interpret(inputs.clone()).unwrap();
        assert_eq!(
            expected,
            vec![ArrayIrValue::Array(Array::scalar(6.0_f32)), ArrayIrValue::Array(Array::scalar(4.0_f32))],
        );
        assert_eq!(linearization.primal().interpret(vec![inputs[0].clone()]), Ok(vec![expected[0].clone()]));
        assert_eq!(linearization.tangent().interpret(vec![inputs[1].clone()]), Ok(vec![expected[1].clone()]));
    }

    #[test]
    fn test_program_linearize_dynamic_reference_preserves_access_order() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let reference_type =
            ReferenceType::new(ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)])));
        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = builder.add_input(reference_type.into());
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));

        // An unused reference can already be frozen. Saving dimensions must not introduce an access to it.
        let unused = builder
            .clone()
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![value],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let frozen = ArrayReference::new(Array::vector(vec![3.0_f32, 5.0, 7.0]));
        frozen.freeze().unwrap();
        assert_eq!(
            unused
                .linearize()
                .unwrap()
                .primal()
                .interpret(
                    vec![ArrayIrValue::Reference(frozen.clone()), ArrayIrValue::Array(Array::scalar(11.0_f32)),]
                ),
            Ok(vec![ArrayIrValue::Array(Array::scalar(11.0_f32))]),
        );

        // I/O preceding a failing original access must still happen first. No dimension read may move ahead of it.
        builder
            .add_instruction(ArrayOperation::from(PrintOperation::new("before_read")), Vec::new(), vec![value], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let context = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let mut events = Vec::new();
        let result = linearization.primal().interpret_with(
            vec![ArrayIrValue::Reference(frozen), ArrayIrValue::Array(Array::scalar(11.0_f32))],
            |_, value| Ok(value.clone()),
            |instruction, inputs| {
                events.push(instruction.operation().name());
                context.bind(instruction.operation().clone(), Vec::new(), inputs)
            },
        );
        assert_eq!(result.unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(events, vec!["print", "reference_read"]);
    }

    #[test]
    fn test_program_linearize_restores_pruned_tangent_inputs() {
        // `stop_gradient` disconnects `dy` from the tangent output while `y` remains live in the primal program.
        // The split must restore the canonical `dy` boundary slot after partial-evaluation liveness pruning.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (_, program) = NestedTracingContext::trace(
            context,
            |inputs| Ok(vec![inputs[0].sin()? + inputs[1].stop_gradient()]),
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let linearization = program.into_simplified().unwrap().linearize().unwrap();
        assert_eq!(linearization.primal().output_ids().len(), 1 + linearization.residual_count());
        assert_eq!(linearization.tangent().input_ids().len(), 2 + linearization.residual_count());

        let mut primal_outputs = linearization
            .primal()
            .interpret_in_context(&context, vec![Array::scalar(0.7), Array::scalar(1.3)])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![Array::scalar(1.0), Array::scalar(123.0)];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret_in_context(&context, tangent_inputs),
            Ok(vec![Array::scalar(0.7f64.cos())]),
        );
    }

    #[test]
    fn test_program_linearize_constant_output_tangent() {
        // A constant-valued output still has an ordinary tangent output whose value is zero. Preserve this output
        // alongside the live derivative, regardless of how its zero is represented internally.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let constant = builder.add_constant(Array::scalar(2.0));
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![scaled, constant], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().output_ids().len(), 2 + linearization.residual_count());
        assert_eq!(linearization.tangent().output_ids().len(), 2);
        let tangent_inputs =
            [vec![Array::scalar(1.0)], vec![Array::scalar(2.0); linearization.residual_count()]].concat();
        let tangent_outputs = linearization.tangent().interpret(tangent_inputs).unwrap();
        assert_eq!(
            tangent_outputs,
            vec![Array::scalar(2.0), Array::scalar(0.0)],
            "the tangent outputs must be [2 * ẋ] for the scaled output and a restored zero for the constant output",
        );
    }

    #[test]
    fn test_program_linearize_empty_boundaries() {
        // Boundary-degenerate programs retain their canonical signatures. A zero-input constant program has one
        // primal output and one zero tangent output, while a zero-output program still retains the dead tangent
        // input corresponding to its primal input.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let constant = builder.add_constant(Array::scalar(5.0));
        let program = builder.build::<Vec<Array>, Vec<Array>>(vec![constant], Vec::new(), vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(linearization.primal().interpret(Vec::new()).unwrap(), vec![Array::scalar(5.0)]);
        assert_eq!(linearization.tangent().interpret(Vec::new()).unwrap(), vec![Array::scalar(0.0)]);

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Vec<Array>, Vec<Array>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().input_ids().len(), 1);
        assert!(linearization.primal().output_ids().is_empty());
        assert_eq!(linearization.tangent().input_ids().len(), 1);
        assert!(linearization.tangent().output_ids().is_empty());
    }

    #[test]
    fn test_program_linearize_with_respect_to() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        let linearization = program.linearize_with_respect_to(&[]).unwrap();
        let outputs = linearization.primal().interpret(vec![Array::scalar(2.0_f64)]).unwrap();
        assert_eq!(outputs[0], Array::scalar(2.0_f64));
        assert_eq!(linearization.tangent().interpret(outputs[1..].to_vec()), Ok(vec![Array::scalar(0.0_f64)]));
    }

    #[test]
    fn test_partitioned_program_interpret_in_context() {
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let partition = partitioned_jvp_with_known_effect();
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let inputs = [
            TestValue::Reference(reference.clone()),
            TestValue::Array(Array::scalar(3.0_f32)),
            TestValue::Array(Array::scalar(2.0_f32)),
            TestValue::Array(Array::scalar(7.0_f32)),
        ];
        // The saved coefficient follows both known original outputs in the known program. Its residual feeder must
        // use that offset rather than mistake the known zero tangent for the coefficient.
        assert_eq!(
            partition.outputs(),
            &[
                PartialEvaluationOutput::Known(0),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(1)
            ],
        );
        assert_eq!(
            partition.interpret_in_context(&context, &inputs, 1),
            Ok(vec![
                TestValue::Array(Array::scalar(4.0_f32)),
                TestValue::Array(Array::scalar(8.0_f32)),
                TestValue::Array(Array::scalar(0.0_f32)),
            ]),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
    }

    #[test]
    fn test_partitioned_program_interpret_in_context_transfers_known_tangents() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            TestValue,
            TestOperation,
        >::new()));
        let partition = partitioned_jvp_with_known_effect();
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let tangent = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        let inputs = [
            context.primal().lift(TestValue::Reference(reference.clone())).unwrap(),
            context.primal().lift(TestValue::Array(Array::scalar(3.0_f32))).unwrap(),
            tangent.clone(),
            context.tangent().lift(TestValue::Array(Array::scalar(7.0_f32))).unwrap(),
        ];
        let outputs = partition.interpret_in_context(&context, &inputs, 1).unwrap();
        assert_eq!(outputs[0].value().unwrap().as_known(), Some(&TestValue::Array(Array::scalar(4.0_f32))));
        assert!(outputs[1].value().unwrap().is_unknown());
        assert_eq!(outputs[2].value().unwrap().as_known(), Some(&TestValue::Array(Array::scalar(0.0_f32))));
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));

        // Even the known zero tangent must share the tangent context: only that context can accept its existing
        // unknown input without treating it as a foreign unknown value.
        assert_eq!(outputs[1].context().import_known(&tangent), Ok(tangent.clone()));
        assert_eq!(outputs[2].context().import_known(&tangent), Ok(tangent.clone()));
        assert!(matches!(
            outputs[0].context().import_known(&tangent),
            Err(ProgramError::MalformedProgram(message))
                if message == "cannot import an unknown value from another partial-evaluation context",
        ));
    }

    #[test]
    fn test_partitioned_program_interpret_in_context_validates_before_effects() {
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let partition = partitioned_jvp_with_known_effect();
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let inputs = [
            TestValue::Reference(reference.clone()),
            TestValue::Array(Array::scalar(3.0_f32)),
            TestValue::Array(Array::scalar(2.0_f32)),
            TestValue::Array(Array::scalar(7.0_f32)),
        ];
        // Check missing known inputs, missing unknown inputs, and extra inputs before the known reference write.
        assert!(matches!(
            partition.interpret_in_context(&context, &[], 1),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 4, actual: 0 })),
        ));
        assert!(matches!(
            partition.interpret_in_context(&context, &inputs[..2], 1),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 4, actual: 2 })),
        ));
        // An unused unknown input still belongs to the original boundary.
        assert!(matches!(
            partition.interpret_in_context(&context, &inputs[..3], 1),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 4, actual: 3 })),
        ));
        let mut extra_inputs = inputs.to_vec();
        extra_inputs.push(TestValue::Array(Array::scalar(0.0_f32)));
        assert!(matches!(
            partition.interpret_in_context(&context, &extra_inputs, 1),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 4, actual: 5 })),
        ));
        assert!(matches!(
            partition.interpret_in_context(&context, &inputs, 4),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "partitioned JVP declares 4 primal outputs but has only 3 outputs",
        ));
        assert!(matches!(
            partition.interpret_in_context(&context, &inputs, 2),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "partitioned JVP primal output 1 is residual; all primal outputs must be known",
        ));
        assert_eq!(reference.read(), Ok(Array::scalar(1.0_f32)));
    }

    #[test]
    fn test_partitioned_program_interpret_in_context_empty_boundary() {
        let context = DifferentiationContext::fused(EagerContext::<Array, ArrayOperation<Array>>::new());
        let partition = ProgramBuilder::<Array, ArrayOperation<Array>>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap()
            .partition(&[])
            .unwrap();
        assert_eq!(partition.interpret_in_context(&context, &[], 0), Ok(Vec::new()));
    }

    #[test]
    fn test_forward_mode_differentiate_jvp() {
        // `ForwardModeDifferentiate::jvp` on an explicit context runs the closure directly on duals. For
        // `f(x) = sin(x)` at `x = 2` along the tangent `ẋ = 3`, the primal output is `sin(2)` and the tangent
        // output is `3 · cos(2)`.
        let (value, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(|x, ()| x.sin(), Array::scalar(2.0), Array::scalar(3.0), ())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        let (value, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(|input, scale| Ok(input * scale), Array::scalar(2.0), Array::scalar(1.0), Array::scalar(3.0))
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_validates_zero_residual_capture() {
        let context = TracingContext::<ProjectedMemberValue<3>, ProjectedMemberOperation<3>>::new();
        let primal = context.input(ProjectedMemberType::<3>);
        let tangent = context.input(ProjectedMemberType::<3>);
        assert!(matches!(
            context.jvp(
                |input, ()| {
                    Ok(DifferentiationTracer::new(
                        DifferentiationDual::new_with_zero_tangent(input.primal().clone())?,
                        input.context().clone(),
                    ))
                },
                primal,
                tangent,
                (),
            ),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "jvp output tangent captured 0 zero residuals but declared 1",
        ));
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_rejects_aliased_reference_inputs() {
        // The canonical boundary validator runs on the concrete inputs before any tracer exists, so the same allocation
        // at two input positions is rejected before the differentiation context could observe it.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let error = differentiate_at((reference.clone(), reference.clone()))
            .jvp((reference.clone(), reference), |(first, _): (_, _)| Ok(first))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "input 1 and input 0 bind the same reference allocation".to_string(),
            }),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_rejects_a_reference_that_is_both_captured_and_passed() {
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let error = differentiate_at(reference.clone())
            .with_captures(reference.clone())
            .jvp(reference, |input, _| Ok(input))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "capture 0 and input 0 bind the same reference allocation".to_string(),
            }),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_rejects_aliased_tangent_references() {
        // A tangent reference is mutated independently of every primal reference, so a tangent aliasing a primal input,
        // a capture, or another tangent is rejected before any rule runs and before any reference is touched.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let other = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0_f32)));
        let tangent = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0_f32)));
        let error = differentiate_at((reference.clone(), other.clone()))
            .jvp((reference.clone(), tangent.clone()), |(first, _): (_, _)| Ok(first))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "tangent 0 and input 0 bind the same reference allocation".to_string(),
            }),
        );
        let error = differentiate_at((reference.clone(), other.clone()))
            .jvp((tangent.clone(), tangent.clone()), |(first, _): (_, _)| Ok(first))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "tangent 1 and tangent 0 bind the same reference allocation".to_string(),
            }),
        );
        let error = differentiate_at(reference.clone())
            .with_captures(other.clone())
            .jvp(other, |input, _| Ok(input))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "tangent 0 and capture 0 bind the same reference allocation".to_string(),
            }),
        );

        // Distinct tangent references are accepted and forwarded by identity.
        assert_eq!(
            differentiate_at(reference.clone()).jvp(tangent.clone(), |input| Ok(input)),
            Ok((reference, tangent)),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_rejects_captured_reference_outputs() {
        // A captured reference is plumbing with a structural zero tangent, so returning it leaves a reference output
        // with no tangent to materialize; the boundary rejects it by output position.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let error = differentiate_at(ArrayIrValue::Array(Array::scalar(2.0_f32)))
            .with_captures(reference)
            .jvp(ArrayIrValue::Array(Array::scalar(1.0_f32)), |_, capture| Ok(capture))
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "output 0 is a reference rooted in a reference that carries no tangent (a captured or \
                          inactive reference); pass that reference as a differentiated input instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_staged_reference_boundaries() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (primal, tangent) =
                    differentiate_at(inputs[0].clone()).jvp(inputs[1].clone(), |reference| reference.read())?;
                let (_, pushforward) = differentiate_at(inputs[0].clone()).linearize(|reference| reference.read())?;
                let delayed = pushforward.apply(inputs[1].clone())?;
                assert!(matches!(
                    pushforward.apply(inputs[0].clone()),
                    Err(ProgramError::InvalidArgument { message })
                        if message == "tangent 0 aliases a reference bound at the primal boundary of the \
                            differentiated function",
                ));
                assert!(matches!(
                    differentiate_at(inputs[0].clone()).jvp(inputs[0].clone(), |reference| reference.read()),
                    Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                        if message == "tangent 0 and input 0 bind the same reference allocation",
                ));
                Ok(vec![primal, tangent, delayed])
            },
            vec![reference_type.clone(), reference_type],
        )
        .unwrap();
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(3.0_f32))),
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0_f32))),
            ]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(3.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
            ])
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_in_execution_domain() {
        // The builder's `jvp` terminal serves top-level concrete values through their `Value::ExecutionDomain`
        // declarations. A concrete array input recovers the eager array domain, so both dual halves are concrete.
        let (value, tangent) = differentiate_at(Array::scalar(2.0)).jvp(Array::scalar(3.0), |x| x.sin()).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_complex() {
        // Complex duals flow through the same rules. The jvp of z² pushes the tangent ż to `2z · ż`
        // at a genuinely complex point.
        let z = num_complex::Complex::new(0.7f64, -0.3f64);
        let tangent_seed = num_complex::Complex::new(1.0f64, 0.5f64);
        let (value, tangent) =
            differentiate_at(Array::scalar(z)).jvp(Array::scalar(tangent_seed), |x| Ok(x.clone() * x)).unwrap();
        assert_eq!(value.elements::<num_complex::Complex<f64>>().unwrap(), vec![z * z]);
        assert_eq!(tangent.elements::<num_complex::Complex<f64>>().unwrap(), vec![(z + z) * tangent_seed],);
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_host_control_flow() {
        // Eager JVP duals carry concrete primal halves, so ordinary host control flow can branch on a Boolean primal
        // without tracing the untaken branch. The Boolean has no tangent space and therefore receives a structural-zero
        // tangent alongside the live tangent of `x`.
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let (value, tangent) = differentiate_at((Array::scalar(true), Array::scalar(0.7)))
            .jvp((zero.clone(), Array::scalar(1.0)), |(predicate, x)| {
                Ok(if predicate.concretize()? { x.clone() * x.sin()? } else { -x })
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.7 * 0.7f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_zero_spaces() {
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        // Inputs without tangent spaces retain first-class zero-space boundary leaves. Their only valid tangent value
        // is a rank-zero structural-zero array, and output structural zeros materialize with the same type.
        let (value, tangent): ((Array, Array), (Array, Array)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(
                |inputs, ()| Ok(inputs),
                (Array::scalar(2.0f64), Array::scalar(3i32)),
                (Array::scalar(1.0f64), zero.clone()),
                (),
            )
            .unwrap();
        assert_eq!(value, (Array::scalar(2.0f64), Array::scalar(3i32)));
        assert_eq!(tangent, (Array::scalar(1.0f64), zero.clone()));

        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let (value, tangent) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .jvp(|token, ()| Ok(token), token.clone(), zero.clone(), ())
            .unwrap();
        assert_eq!(value, token.clone());
        assert_eq!(tangent, zero.clone());
        assert!(matches!(
            EagerContext::<Array, ArrayOperation<Array>>::new().jvp(
                |token, ()| Ok(token),
                token.clone(),
                token.clone(),
                (),
            ),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "tangent type token[] does not match type zero[] required by primal type token[]",
        ));
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_staged() {
        // Under an active trace, the builder's `jvp` terminal recovers the staging context from its tracer inputs, so
        // it composes inside traced code without threading a context. The closure stages the fused primal and tangent
        // operations into the enclosing trace.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, tangent) = differentiate_at(inputs[0].clone()).jvp(inputs[1].clone(), |x| x.sin())?;
                Ok(vec![value, tangent])
            },
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let outputs = program.interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1].to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_staged_zero_spaces() {
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        // The same composition preserves zero-space leaves for tokens instead of attempting to stage token
        // arithmetic while constructing the enclosing program.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, tangent) = differentiate_at(inputs[0].clone()).jvp(inputs[1].clone(), |token| Ok(token))?;
                Ok(vec![value, tangent])
            },
            vec![ArrayType::scalar(DataType::Token), ArrayType::scalar(DataType::Zero)],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![token.clone(), zero.clone()]), Ok(vec![token.clone(), zero.clone()]));
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_invalid_structure() {
        // Tangents pair with primals leaf-for-leaf and so a tangent structure that does not match the primal
        // structure is rejected.
        assert!(matches!(
            differentiate_at(vec![Array::scalar(1.0)])
                .jvp(vec![Array::scalar(1.0), Array::scalar(2.0)], |x| Ok(x))
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::Parameter(
                ParameterError::MismatchedParameterStructures { .. },
            )),
        ));
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_empty_input() {
        // With no leaf value to recover a context from, the builder's `jvp` terminal reports that differentiation
        // requires at least one input leaf.
        assert_eq!(
            differentiate_at(Vec::<Array>::new()).jvp(Vec::new(), |x| Ok(x)).unwrap_err(),
            DifferentiationError::EmptyInput,
        );
    }

    #[test]
    fn test_forward_mode_differentiate_jvp_low_precision() {
        // Rank-zero arrays support both half-precision variants through the ordinary array operations.
        assert_eq!(
            differentiate_at(Array::scalar(bf16::from_f32(3.0))).jvp(Array::scalar(bf16::ONE), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(bf16::from_f32(6.0)), Array::scalar(bf16::from_f32(2.0)))),
        );
        assert_eq!(
            differentiate_at(Array::scalar(f16::from_f32(3.0))).jvp(Array::scalar(f16::ONE), |x| Ok(x.clone() + x)),
            Ok((Array::scalar(f16::from_f32(6.0)), Array::scalar(f16::from_f32(2.0)))),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_linearize() {
        // `ForwardModeDifferentiate::linearize` on an explicit context runs the closure once at the primal point and
        // returns the primal output together with a reusable pushforward: applying it pushes any number of tangents
        // through the Jacobian at that point without re-tracing or re-differentiating.
        let (value, pushforward) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .linearize(|x, ()| x.sin(), Array::scalar(2.0), ())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pushforward.apply(Array::scalar(1.0)).unwrap().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            pushforward.apply(Array::scalar(3.0)).unwrap().to_f64s()[0],
            3.0 * 2.0f64.cos(),
            epsilon = 1e-9,
        );
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_rejects_aliased_reference_inputs() {
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let error = differentiate_at((reference.clone(), reference))
            .linearize(|(first, _): (_, _)| Ok(first))
            .err()
            .unwrap();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "input 1 and input 0 bind the same reference allocation".to_string(),
            }),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_with_local_tangent_accumulator() {
        let (primal, pushforward) = differentiate_at(ArrayIrValue::Array(Array::scalar(3.0_f32)))
            .linearize(|input: LinearizationTracer<EagerContext<TestValue, TestOperation>>| {
                let zero = input.dispatch_domain().lift(Array::scalar(0.0_f32).into())?;
                let reference = zero.reference_new()?;
                reference.add_update(&input)?;
                reference.read()
            })
            .unwrap();
        assert_eq!(primal, ArrayIrValue::Array(Array::scalar(3.0_f32)));
        assert!(pushforward.residuals().iter().all(|value| !value.r#type().is_reference()));
        assert_eq!(pushforward.apply(Array::scalar(2.0_f32).into()), Ok(Array::scalar(2.0_f32).into()));
        assert_eq!(pushforward.apply(Array::scalar(5.0_f32).into()), Ok(Array::scalar(5.0_f32).into()));
        assert_eq!(pushforward.apply(Array::scalar(2.0_f32).into()), Ok(Array::scalar(2.0_f32).into()));
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_with_reference_inputs() {
        // `f(r, x) = { add_update(r, x); read(r) }` over a live reference. The eager known side of the linearization
        // is the forward pass: the primal update mutates `r` at linearization time, exactly as forward mode does, while
        // the tangent accesses stage over the tangent reference the pushforward is later applied to.

        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let (value, pushforward) =
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Array(Array::scalar(3.0_f32))))
                .linearize(add_to_reference)
                .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::scalar(4.0_f32)));
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));

        // The pushforward is the tangent program over the tangent reference and the tangent of `x`, with no residuals:
        // it adds the tangent of `x` into the tangent reference and reads it back, binding the caller's tangent
        // reference by identity when applied.
        assert!(pushforward.residuals().is_empty());
        assert_eq!(
            pushforward.program().to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let reference_add_update %0 %1
                    %2:f32[] = reference_read %0
                in (%2)
            "}
            .trim_end(),
        );
        let tangent_reference = ArrayReference::new(Array::scalar(0.5_f32));
        assert_eq!(
            pushforward.apply((
                ArrayIrValue::Reference(tangent_reference.clone()),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
            )),
            Ok(ArrayIrValue::Array(Array::scalar(2.5_f32))),
        );
        assert_eq!(tangent_reference.read(), Ok(Array::scalar(2.5_f32)));

        // The fused forward-mode evaluation at the same point agrees on the primal output and on the final state of the
        // primal reference, and pushes the tangent through the tangent reference directly: `ṫ = ṟ + ẋ = 0.5 + 2`.
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let tangent_reference = ArrayReference::new(Array::scalar(0.5_f32));
        assert_eq!(
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Array(Array::scalar(3.0_f32))))
                .jvp(
                    (ArrayIrValue::Reference(tangent_reference.clone()), ArrayIrValue::Array(Array::scalar(2.0_f32))),
                    add_to_reference,
                ),
            Ok((ArrayIrValue::Array(Array::scalar(4.0_f32)), ArrayIrValue::Array(Array::scalar(2.5_f32)))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(4.0_f32)));
        assert_eq!(tangent_reference.read(), Ok(Array::scalar(2.5_f32)));
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_rejects_captured_reference_outputs() {
        // A captured reference is plumbing with a structural zero tangent, so returning it leaves a reference output
        // with no tangent reference to expose through the pushforward; the boundary rejects it by output position.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let error = differentiate_at(ArrayIrValue::Array(Array::scalar(2.0_f32)))
            .with_captures(reference)
            .linearize(|_, capture| Ok(capture))
            .err()
            .unwrap();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "output 0 is a reference rooted in a reference that carries no tangent (a captured or \
                          inactive reference); pass that reference as a differentiated input instead"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_in_execution_domain() {
        // The builder's `linearize` terminal serves top-level concrete values through their `Value::ExecutionDomain`
        // declarations. Primal work executes eagerly at the concrete linearization point while the pushforward program
        // accumulates.
        let (value, pushforward) = differentiate_at(Array::scalar(2.0)).linearize(|x| x.sin()).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pushforward.apply(Array::scalar(1.0)).unwrap().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-9);

        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let (value, pushforward) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .linearize(|token, ()| Ok(token), token.clone(), ())
            .unwrap();
        assert_eq!(value, token.clone());
        assert_eq!(pushforward.apply(zero.clone()), Ok(zero.clone()));
        assert!(matches!(
            pushforward.apply(token.clone()),
            Err(ProgramError::MalformedProgram(message))
                if message == "pushforward tangent 0 has type token[] but its primal boundary requires tangent type \
                    zero[]",
        ));
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_staged() {
        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        // Under an active trace, the builder's `linearize` terminal recovers the staging context from its tracer input,
        // so primal work stages into the enclosing trace and the pushforward replays there when applied.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, pushforward) = differentiate_at(inputs[0].clone()).linearize(|x| x.sin())?;
                let tangent = pushforward.apply(inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let outputs = program.interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1].to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, pushforward) = differentiate_at(inputs[0].clone()).linearize(|token| Ok(token))?;
                let tangent = pushforward.apply(inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![ArrayType::scalar(DataType::Token), ArrayType::scalar(DataType::Zero)],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![token.clone(), zero.clone()]), Ok(vec![token.clone(), zero]));
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_host_control_flow() {
        // The closure can branch on a Boolean *primal* with host control flow, because the duals' primal halves carry
        // concrete known values under an eager context. For a true predicate and `x = 3`, `f(x) = x * x` linearizes to
        // the pushforward `ẋ ↦ 2x · ẋ = 6ẋ`, and the untaken `sin(x)` branch is never traced at all. Neither `sin` nor
        // its `cos` derivative can appear in the pushforward program.
        let (value, pushforward) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .linearize(
                |(predicate, x), ()| Ok(if predicate.concretize().unwrap() { x.clone() * x } else { x.sin().unwrap() }),
                (Array::scalar(true), Array::scalar(3.0)),
                (),
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 9.0, epsilon = 1e-9);
        assert_eq!(
            pushforward
                .program()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["mul", "mul", "add"],
        );
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        assert_abs_diff_eq!(pushforward.apply((zero, Array::scalar(1.0))).unwrap().to_f64s()[0], 6.0, epsilon = 1e-9,);
    }

    #[test]
    fn test_forward_mode_differentiate_linearize_empty_input() {
        // With no leaf value to recover a context from, the builder's `linearize` terminal reports that differentiation
        // requires at least one input leaf.
        assert_eq!(
            differentiate_at(Vec::<Array>::new()).linearize(|x| Ok(x)).map(|(outputs, _)| outputs).unwrap_err(),
            DifferentiationError::EmptyInput
        );
    }

    #[test]
    fn test_jvp_projected_operation() {
        // The third fixture member is intentionally unrelated to arrays. Its identity JVP proves that the adapter
        // projects both halves of a live dual and lifts the resulting member values back into the composite family.
        let context =
            DifferentiationContext::fused(EagerContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3)),
        )
        .unwrap();
        let output = jvp_projected_operation(&context, &ProjectedMemberOperation::<2>::Identity, &[input])
            .unwrap()
            .remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)));
        assert!(matches!(tangent, MaybeZero::Value(ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3))),));

        // Structural zeros cross the same adapter as types and therefore do not stage or materialize member values.
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)),
            MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),
        )
        .unwrap();
        let output = jvp_projected_operation(&context, &ProjectedMemberOperation::<2>::Identity, &[input])
            .unwrap()
            .remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)));
        assert!(matches!(tangent, MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),));
    }

    #[test]
    fn test_jvp_projected_operation_partitioned() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            TestValue,
            TestOperation,
        >::new()));
        let primal = context.primal().lift(Array::scalar(0.0_f64).into()).unwrap();
        let tangent = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F64).into(), 0),
        );
        let input = DifferentiationDual::new(primal, tangent.clone()).unwrap();
        let outputs =
            jvp_projected_operation(&context, &ArrayOperation::<Array>::Sin(SinOperation::new()), &[input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal().value().unwrap().as_known(), Some(&Array::scalar(0.0_f64).into()));
        let tangent_output = outputs[0].tangent().as_value().unwrap();
        assert_eq!(tangent_output.context().import_known(&tangent), Ok(tangent.clone()));
        assert!(matches!(outputs[0].primal().context().import_known(&tangent),
            Err(ProgramError::MalformedProgram(message)) if message == "cannot import an unknown value from another \
                partial-evaluation context",
        ));
    }
}
