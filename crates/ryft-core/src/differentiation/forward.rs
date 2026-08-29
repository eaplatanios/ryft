use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain, ProjectedContext, StagingContext, ValueResolution};
use crate::differentiation::DifferentiationError;
use crate::differentiation::reverse::TransposableOperation;
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::{
    ResidualZeroProvider, ZeroSpaceBoundaryReconstruction, ZeroSpaceBoundaryRole,
    capture_and_validate_zero_residual_values,
};
use crate::macros::check_count;
use crate::operations::AddOperation;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialTracer,
    PartialValue, PartiallyEvaluatableOperation,
};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::{
    Atom, AtomId, BindingRegionDriver, Effect, EmptyRegionDriver, MaybeZero, Operation, OperationProjection, Program,
    ProgramBuilder, ProgramError, ProjectedValue, Provenance, ProvenanceScope, Region, RegionDriver, RegionRef,
    RegionReplayMappings, ReplayRegionDriver, Type, TypeError, TypeIdentityPosition, Typed, Value, ValueProjection,
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
pub struct Linearization<V: Value, O: Operation<Type = V::Type>> {
    /// Nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal outputs
    /// `y = f(x)` followed by the residuals `r`, its trailing [`residual_count`](Self::residual_count) outputs, which
    /// form the residual environment consumed by the tangent sub-program.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Linear tangent sub-program `(live(ẋ), r) ↦ live(ẏ)`. It has one leading Single Static Assignment (SSA) input
    /// for each primal input whose tangent type is not a zero differential space, followed by the residuals `r`, and
    /// one SSA output for each primal output whose tangent type is not a zero differential space.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

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
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residual_count: usize,
    ) -> Result<Self, ProgramError>
    where
        V::Type: DifferentiableType,
    {
        let primal_output_count = primal.output_ids().len().checked_sub(residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linearization primal program produces {} outputs which is fewer than its {residual_count} residuals",
                primal.output_ids().len(),
            ))
        })?;
        let tangent_input_count = tangent.input_ids().len().checked_sub(residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linearization tangent program consumes {} inputs which is fewer than its {residual_count} residuals",
                tangent.input_ids().len(),
            ))
        })?;
        let differentiable_primal_inputs = primal
            .inputs()
            .map(|input| Ok((input.r#type().tangent()?, input)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter_map(|(tangent_type, input)| (!tangent_type.is_zero_space()).then_some(input))
            .collect::<Vec<_>>();
        if tangent_input_count != differentiable_primal_inputs.len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization tangent program consumes {tangent_input_count} tangent inputs \
                 while the primal program has {} nonzero differential inputs",
                differentiable_primal_inputs.len(),
            )));
        }
        let differentiable_primal_outputs = primal
            .outputs()
            .take(primal_output_count)
            .map(|output| Ok((output.r#type().tangent()?, output)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter_map(|(tangent_type, output)| (!tangent_type.is_zero_space()).then_some(output))
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
        Ok(Self { primal, tangent, residual_count })
    }

    /// Returns the nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal
    /// outputs `y = f(x)` followed by the residuals `r` (i.e., the intermediate values of the derivative computation
    /// that depend only on `x`) whose trailing [`residual_count`](Self::residual_count) outputs form the residual
    /// environment consumed by the [`tangent`](Self::tangent) sub-program.
    #[inline]
    pub fn primal(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.primal
    }

    /// Returns the compact linear tangent sub-program `(live(ẋ), r) ↦ live(ẏ)`. The sub-program is linear in its
    /// tangent inputs, with the linearization point `x` entering only through the residuals `r`.
    #[inline]
    pub fn tangent(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
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
    /// sub-program, and [`residual_count`](Self::residual_count), in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, usize) {
        (self.primal, self.tangent, self.residual_count)
    }

    /// Returns the compact forward-mode pushforward program `(live(ẋ), r) ↦ live(ẏ)`. Because linearization already
    /// produces the pushforward as its unknown half, this is the [`tangent`](Self::tangent) sub-program itself, cloned
    /// (i.e., the identity counterpart of [`pullback`](Self::pullback), which derives its program by transposition).
    #[inline]
    pub fn pushforward(&self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.tangent.clone()
    }

    /// Builds the compact reverse-mode pullback program `(live(ȳ), r) ↦ live(x̄)` by transposing the
    /// [`tangent`](Self::tangent) sub-program. Conceptually, it takes the output cotangents `ȳ` followed by the
    /// residuals `r` and produces the input cotangents `x̄ = (∂f/∂x)(x)ᵀ · ȳ`. It is the derived third member of this
    /// [`Linearization`]'s program family, alongside the stored [`primal`](Self::primal) and [`tangent`](Self::tangent)
    /// sub-programs. Rather than re-keying each bilinear operation of the tangent sub-program into a closed captured
    /// factor (e.g., folding a scalar `Mul` against a known operand into a multiply-by-a-captured-constant) by folding
    /// the consuming residual value, this function leaves the tangent sub-program in the primal operation family `O`
    /// and transposes it through [`Program::transpose_with_respect_to`]. The tangent sub-program's inputs are `(ẋ, r)`,
    /// and so it is transposed with respect to the leading tangent inputs `ẋ` while the trailing
    /// [`residual_count`](Self::residual_count) residual inputs are held as known parameters. Partition-aware
    /// transposition then threads each known residual through to the pullback as a pullback input (consumed by the
    /// adjoint operation that the bilinear operation's transpose rule stages), rather than folding it into a captured
    /// factor, so the returned pullback program stays over the primal operation family `O` and produces the cotangents
    /// of the linear tangent inputs only.
    #[inline]
    pub fn pullback(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: TransposableOperation<V, O> + ResidualZeroProvider<V::Type> + From<AddOperation<V::Type>>,
    {
        // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
        // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are
        // not folded into captured factors here. The subtraction cannot underflow because `Self::new` validated that
        // the tangent program consumes at least `residual_count` inputs.
        self.tangent.transpose_with_trailing_residuals(self.residual_count)
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
    ///   - `output_structure`: Parameter structure used to rebuild the complete public tangent output after typed
    ///     zeros have been inserted for the omitted leaves.
    pub fn new(
        context: C,
        program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        residuals: Vec<C::Value>,
        tangent_reconstruction: ZeroSpaceBoundaryReconstruction<C::Value>,
        primal_input_types: Vec<C::Type>,
        primal_output_types: Vec<C::Type>,
        output_structure: Output::ParameterStructure,
    ) -> Result<Self, ProgramError>
    where
        C::Operation: ResidualZeroProvider<C::Type>,
    {
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
                    "pushforward residual {index} has type {} in the pushforward program \
                     but carries a value of type {}",
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
                    "pushforward program tangent output {} has type {} but its public boundary requires tangent \
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

    /// Returns the type of every flattened input leaf of the original primal function, including leaves whose tangent
    /// spaces contain only zero and which therefore have no corresponding input in [`program`](Self::program). This
    /// metadata lets reverse-mode construction derive the complete cotangent boundary before it consumes the compact
    /// [`Pushforward`].
    #[inline]
    pub(crate) fn primal_input_types(&self) -> &[C::Type] {
        &self.primal_input_types
    }

    /// Returns the type of every flattened output leaf of the original primal function, including leaves whose tangent
    /// spaces contain only zero and which therefore have no corresponding output in [`program`](Self::program). This
    /// metadata lets reverse-mode construction derive the complete cotangent boundary before it consumes the compact
    /// [`Pushforward`].
    #[inline]
    pub(crate) fn primal_output_types(&self) -> &[C::Type] {
        &self.primal_output_types
    }

    /// Consumes this [`Pushforward`] and returns its open parts: the compact pushforward program
    /// `(live(ẋ), r) ↦ live(ẏ)` and the linearization-point residuals `r` its trailing inputs consume, in that order.
    /// Unlike [`apply`](Self::apply), the returned program does not insert typed-zero values for public tangent leaves
    /// omitted from its SSA boundary because their differential spaces contain only zero.
    #[inline]
    pub fn into_parts(self) -> (Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<C::Value>) {
        (self.program, self.residuals)
    }

    /// Pushes the structured tangents `tangents` through the linearized Jacobian, returning the tangent outputs. The
    /// tangents are flattened, the linearization-point residuals are appended, the pushforward program is interpreted
    /// at that vector in the context that this pushforward was built in (i.e., the single replay path for both context
    /// flavors: an eager context interprets the pushforward immediately, while a staging context stages it into the
    /// enclosing trace and returns tracers), and the flat tangent outputs are reshaped against the closure's output
    /// structure.
    #[inline]
    pub fn apply(&self, tangents: Input::To<C::Value>) -> Result<Output::To<C::Value>, ProgramError>
    where
        C::Operation: ResidualZeroProvider<C::Type>,
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

        // Defensively verify that the filtered public boundary agrees with the compact program's leading tangent
        // boundary. `Pushforward::new` established the same contract when this callable was constructed.
        let tangent_input_count = self.program.input_ids().len() - self.residuals.len();
        if program_inputs.len() != tangent_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward received {} tangents but its program consumes {} tangent inputs",
                program_inputs.len(),
                tangent_input_count,
            )));
        }
        for (index, (input, expected)) in program_inputs.iter().zip(self.program.inputs()).enumerate() {
            if input.r#type().as_ref() != expected.r#type().as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward tangent {} has type {} but its program requires type {}",
                    index,
                    input.r#type().as_ref(),
                    expected.r#type().as_ref(),
                )));
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

/// Provides call-scoped access to the regions attached to the instruction being differentiated. Transform
/// dispatch constructs a driver for one operation application and passes it directly to that operation's
/// [`jvp`](DifferentiableOperation::jvp) rule. [`RegionDriver`] provides structural region access, while this
/// trait adds differentiation-specific recursion. Region-free applications receive a driver with no regions.
///
/// Structural transform requests accept borrowed [`RegionRef`]s directly, allowing the same request to serve both a
/// region selected from this driver and the entry region of a program rebuilt by an operation rule. Implementations
/// must recursively dispatch each nested instruction with the driver for that nested application.
pub trait DifferentiationDriver<C: Context>: RegionDriver<C::Constant, C::Operation> {
    /// Builds the compact fused forward-mode program of `region` and returns a shared handle to it. The returned
    /// program maps `[primals..., live(tangents)...]` to `[primal outputs..., live(tangent outputs)...]`, where
    /// `live(...)` omits every boundary leaf whose tangent type is a zero differential space (see
    /// [`DifferentiableType::is_zero_space`]). Callers must derive their tangent liveness masks from the same region
    /// boundary types this construction filters on and restore the omitted boundary leaves as structural zeros.
    ///
    /// The result is shared rather than owned because rules commonly re-attach the derived program as a nested region.
    /// An [`Arc`] lets a caching driver serve one artifact for a region that several programs share instead of
    /// re-differentiating it per program, and it lets repeated attachments of one artifact intern by [`Arc`] identity.
    /// The built-in recursive driver therefore serves the region's retained fused program through the cached
    /// counterpart of [`RegionRef::jvp`], while a custom driver that retains nothing simply derives the program
    /// uncached and wraps it in [`Arc::new`].
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError>;

    /// Linearizes `region` into its primal and tangent program halves, re-entering the active differentiation
    /// machinery.
    ///
    /// Unlike [`Self::jvp_program`], this hands back an owned [`Linearization`] because its consumers restructure the
    /// component programs instead of attaching them unchanged. The bounded `while` rule, for example, consumes the
    /// primal and tangent halves and rebuilds them into the residual-stacking forward loop and the reversed tangent
    /// scan, so there is no artifact left to share by identity.
    fn linearize_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError>;

    /// Applies `operation`'s forward-mode rule over the provided owned region programs (in region order),
    /// re-entering the active differentiation machinery. Rules that rewrite a region-carrying operation and
    /// differentiate the rewritten form recursively — for example the batched-predicate `while` rule, which rebuilds
    /// a masked condition and body — request that recursion here so they carry no operation-family semantic bounds
    /// of their own.
    fn jvp_operation(
        &self,
        operation: &C::Operation,
        programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;
}

impl<C: Context> DifferentiationDriver<C> for EmptyRegionDriver {
    fn jvp_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot differentiate a program".to_string()).into())
    }

    fn linearize_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot linearize a program".to_string()).into())
    }

    fn jvp_operation(
        &self,
        _operation: &C::Operation,
        _programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        _context: &C,
        _inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot differentiate an operation".to_string()).into())
    }
}

/// [`DifferentiationDriver`] scoped to one [`Operation`] application. It borrows the application's complete region
/// driver, which preserves the operation-defined ordering of owned regions, borrowed regions, and shared callees
/// without materializing a combined region collection. Recursive requests are answered through [`Program::jvp`] and
/// [`Program::linearize`].
struct RecursiveDifferentiationDriver<'r, D> {
    /// Application-scoped region driver, in operation-defined order.
    driver: &'r D,
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursiveDifferentiationDriver<'_, D>
{
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
        + ResidualZeroProvider<C::Type>,
{
    #[inline]
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError> {
        region.jvp_shared()
    }

    fn linearize_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Linearization<C::Constant, C::Operation>, DifferentiationError> {
        region.linearize()
    }

    fn jvp_operation(
        &self,
        operation: &C::Operation,
        programs: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        let driver = RecursiveDifferentiationDriver { driver: &programs };
        operation.jvp(context, &driver, inputs)
    }
}

// TODO(eaplatanios): Restore the strict `Operation<Type = C::Type>` super-trait bound once the next-generation trait
//  solver stabilizes. The current solver cannot discharge this projection equality at implementation heads whose
//  context type is built from `Self` (E0284); the equality is enforced per method through `where` clauses instead.
/// Represents [`Operation`]s that support forward-mode differentiation (i.e., computing Jacobian-Vector Products).
/// Reading an operation as a function `y = f(x₁, …, xₙ)` from its operands to its outputs, the [`jvp`](Self::jvp)
/// function propagates [`DifferentiationDual`]s through it. Each input dual `(xᵢ, ẋᵢ)` pairs an operand with its
/// tangent (i.e., perturbation direction), and the function returns one dual `(yⱼ, ẏⱼ)` per output, where `y = f(x)`
/// is the primal result and `ẏ = Σᵢ (∂f/∂xᵢ)(x) · ẋᵢ` is the directional derivative of `f` at `x` along the input
/// tangents. For example, the `jvp` implementation for the sine operation maps `(x, ẋ)` to `(sin x, cos x · ẋ)`. Both
/// halves are built from ordinary primal-family operations bound through [`Context::bind`], so no symbolic capture is
/// ever introduced: under a staging context the primal and tangent operations are staged into the one shared trace,
/// which is how [`Program::jvp`] builds the fused JVP program, while under an eager context both are computed
/// immediately. Structural zero tangents flow between implementations as [`MaybeZero::Zero`]s and stage nothing.
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
///     `Self: ZeroOperationProvider<T>` bound that the nested-region differentiation drivers require. Higher-order
///     payload rules request nested forward-mode and linearization work through their instruction-scoped
///     [`DifferentiationDriver`], whose concrete implementation establishes the finite program-level bounds at its
///     construction site. Output-level semantic queries such as [`Operation::is_zero`] are forwarded by the base
///     operation dispatcher and therefore introduce no additional witness bounds.
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
    ///   - `context`: [`Context`] through which the rule binds the primal and tangent [`Operation`]s it synthesizes.
    ///   - `driver`: [`DifferentiationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to
    ///     attached [`Region`]s.
    ///   - `inputs`: Input [`DifferentiationDual`]s aligned with this operation's inputs/operands.
    fn jvp<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
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
    fn jvp_in_parent<D: DifferentiationDriver<C>>(
        &self,
        context: &C,
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
pub struct DifferentiationTracer<C: Context> {
    /// [`DifferentiationContext`] this dual flows through.
    context: DifferentiationContext<C>,

    /// [`DifferentiationDual`] carrying the primal value and its tangent.
    dual: DifferentiationDual<C::Value>,
}

impl<C: Context> DifferentiationTracer<C> {
    /// Creates a new [`DifferentiationTracer`].
    #[inline]
    pub fn new(dual: DifferentiationDual<C::Value>, context: DifferentiationContext<C>) -> Self {
        Self { context, dual }
    }

    /// Returns the [`DifferentiationContext`] this [`DifferentiationTracer`] flows through.
    #[inline]
    pub fn context(&self) -> &DifferentiationContext<C> {
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
impl<C: Context<Value: PartialEq>> PartialEq for DifferentiationTracer<C> {
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

impl<C: Context> Debug for DifferentiationTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("DifferentiationTracer").field("dual", &self.dual).finish()
    }
}

impl<C: Context> Display for DifferentiationTracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.dual())
    }
}

impl<C: Context> Typed for DifferentiationTracer<C> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> std::borrow::Cow<'_, C::Type> {
        self.primal().r#type()
    }
}

impl<C: Context> Value for DifferentiationTracer<C> {
    type DispatchDomain = DifferentiationContext<C>;
    type ExecutionDomain = DifferentiationContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> DifferentiationContext<C> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> DifferentiationContext<C> {
        self.context().clone()
    }
}

impl<C: Context, T: Type> ValueProjection<T> for DifferentiationTracer<C>
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

/// Value type flowing through the closures of the partial-evaluation-backed differentiation entry points
/// (i.e., [`DifferentiationBuilder::linearize`](crate::DifferentiationBuilder::linearize),
/// [`DifferentiationBuilder::vjp`](crate::DifferentiationBuilder::vjp),
/// [`DifferentiationBuilder::gradient`](crate::DifferentiationBuilder::gradient), and their derivatives). It is a
/// [`DifferentiationTracer`] dual over a [`PartialEvaluationContext`] wrapping the context `C` the transform runs in.
/// Its primal half is a *known* partial-evaluation value carrying a concrete value under an eager `C` (so that e.g.,
/// host control flow on primal values works as expected) and its tangent half is *unknown*, accumulating the
/// pushforward program.
pub type LinearizationTracer<C> = DifferentiationTracer<PartialEvaluationContext<C>>;

/// Forward-mode differentiation [`Context`] that interleaves [`DifferentiableOperation`] implementations with an inner
/// [`Context`], without building a program. Its values are [`DifferentiationTracer`] duals over the inner context's
/// values, and binding an operation dispatches the operation's [`jvp`](DifferentiableOperation::jvp) rule against the
/// inner context directly. Over an eager inner context this computes primal and tangent values operation by operation
/// (i.e., it is the analogue of [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html) interpreter),
/// while over a staging inner context the rules stage the primal and tangent operations into the enclosing trace. This
/// is forward mode's counterpart of [`BatchingContext`](crate::BatchingContext): a transform context that wraps the
/// receiver and runs the user's closure directly on transform tracers (i.e., [`DifferentiationTracer`] duals here,
/// and [`BatchingTracer`](crate::BatchingTracer)s there), with eager-versus-staged behavior absorbed entirely by the
/// wrapped context. It is what makes [`ForwardModeDifferentiate::jvp`] the single forward-mode entry point. Structural
/// zero tangents stay symbolic [`MaybeZero::Zero`]s while they flow between rules. When every input tangent is a
/// structural zero, the [`bind`](Context::bind) fast path skips region-carrying operations because the transform
/// boundary can capture any required runtime geometry from their primal results through [`ResidualZeroProvider`]. It
/// skips a region-free operation only if each output zero tangent can be constructed without runtime identity operands
/// (or the zero-producing primal itself is reusable).
#[derive(Clone)]
pub struct DifferentiationContext<C: Context> {
    /// Parent [`Context`] that carries the primal and tangent values and executes (or stages) the operations
    /// that the forward-mode JVP rules bind.
    parent: C,
}

impl<C: Context> DifferentiationContext<C> {
    /// Creates a new [`DifferentiationContext`] over the provided parent [`Context`].
    #[inline]
    pub fn new(parent: C) -> Self {
        Self { parent }
    }

    /// Returns the parent [`Context`].
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }
}

impl<C: Context> Domain for DifferentiationContext<C> {
    type Type = C::Type;
    type Value = DifferentiationTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context> Context for DifferentiationContext<C>
where
    C::Type: DifferentiableType,
    C::Operation: PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<C>
        + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
        + ResidualZeroProvider<C::Type>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        // Constants are independent of every differentiation input and so their tangents are structural zeros.
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent.lift(constant)?)?;
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }

    fn bind<O: Into<C::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[DifferentiationTracer<C>],
    ) -> Result<Vec<DifferentiationTracer<C>>, ProgramError> {
        let operation = operation.into();
        operation.validate_region_count(driver.region_count())?;

        // The zero-tangent fast path below bypasses the operation's JVP rule, so we reject intrinsic state before that
        // shortcut just as we reject state hidden in attached regions. Operation-local differentiation rules remain a
        // defense in depth for callers that invoke them directly.
        if operation.effects().contains(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` must be discharged before differentiation", operation.name()),
            });
        }

        // Unwrap the input tracers into context-free duals, run the rule against those, and rewrap the produced duals
        // with this context, mirroring how `BatchingContext::bind` unwraps to `ArrayBatch`es and rewraps.
        let input_duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();

        // Attached regions can hide unresolved state (including dormant rule regions at any nesting depth under the
        // current conservative custom-derivative policy), and the all-zero fast path below binds the primal directly
        // without reaching any operation-local differentiation guard. Reject state centrally over the whole attached
        // closure so no differentiation path can execute it. The operation-local rule-region guards remain as defense
        // in depth.
        if driver.regions().any(|region| region.contains_effect_in_closure(Effect::OrderedState)) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` carries unresolved state in an attached region and must be discharged before \
                    differentiation",
                    operation.name(),
                ),
            });
        }

        // All-zero fast path mirroring `Program::jvp`. When an operation consumes at least one input and every input
        // tangent is a structural zero, skip its rule only when each output tangent can later be materialized without
        // runtime identity operands. Zero-input operations remain excluded so their dedicated rules keep handling
        // primal synthesis and tangent typing.
        let zero_input_tangents = !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero());

        // Only all-zero applications can skip their differentiation rule. Region-carrying operations retain structural
        // zero tangents because the transform boundary captures their runtime geometry from the staged primal outputs.
        // Region-free operations can instead inspect their outputs without reproducing the parent's region-identity
        // instantiation.
        let reusable_zero_outputs = if !zero_input_tangents {
            None
        } else if !operation.region_slots().is_empty() {
            Some(Vec::new())
        } else {
            let input_types = input_duals.iter().map(|dual| dual.primal().r#type().into_owned()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
            let mut reusable_zero_outputs = Vec::new();
            let mut can_materialize = true;
            for (output_index, output_type) in output_types.iter().enumerate() {
                let tangent_type = output_type.tangent()?;
                if operation.is_zero(output_index) && output_type == &tangent_type {
                    reusable_zero_outputs.push(output_index);
                } else if tangent_type.identities().any(|(position, _)| position == TypeIdentityPosition::Reference) {
                    can_materialize = false;
                    break;
                }
            }
            can_materialize.then_some(reusable_zero_outputs)
        };

        let output_duals = if let Some(reusable_zero_outputs) = reusable_zero_outputs {
            let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            self.parent
                .bind(operation, driver, &primal_inputs)?
                .into_iter()
                .enumerate()
                .map(|(output_index, primal)| {
                    // When the primal operation is itself known to produce zero and its output already has the required
                    // tangent type, the primal value is the canonical materialized tangent. Reusing it avoids inventing
                    // a nullary dynamic zero when a staged tangent is later materialized, exactly like the fused replay
                    // in `RegionRef::jvp`.
                    if reusable_zero_outputs.contains(&output_index) {
                        DifferentiationDual::new(primal.clone(), primal)
                    } else {
                        DifferentiationDual::new_with_zero_tangent(primal)
                    }
                })
                .collect::<Result<Vec<_>, DifferentiationError>>()?
        } else {
            // Borrow the complete region driver directly, preserving operation-defined ordering without collecting
            // it into temporary storage.
            let differentiation_driver = RecursiveDifferentiationDriver { driver: &driver };
            operation.jvp(&self.parent, &differentiation_driver, input_duals.as_slice())?
        };

        // Stamp this context onto every value handed back to the caller so its capability sugar dispatches through this
        // forward-mode context (the `jvp` rules build their outputs context-free via `DifferentiationDual::new`).
        Ok(output_duals.into_iter().map(|dual| DifferentiationTracer::new(dual, self.clone())).collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // A forward-mode context is eager exactly when the parent context carrying its duals' values is
        // (i.e., never over a staging parent context, always over an eager one).
        self.parent.is_eager()
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // Forward-mode differentiation stages rewritten primitive work through its parent, so that provenance state
        // lives with the parent.
        self.parent.provenance()
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.parent.invoke_with_provenance_origin(origin, function)
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.parent.invoke_with_provenance_scope(scope, function)
    }

    #[inline]
    fn resolve(&self, value: &DifferentiationTracer<C>) -> ValueResolution<C::Constant> {
        // A value is constant in the differentiated computation only when its primal resolves in the parent and its
        // tangent is structurally zero. A live tangent makes the dual input-dependent even if its primal is constant.
        if value.tangent().is_zero() { self.parent.resolve(value.primal()) } else { ValueResolution::Opaque }
    }
}

impl<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>> RegionRef<'_, V, O>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type>,
{
    /// Builds the _fused_ Jacobian-Vector Product (JVP) [`Program`] of this borrowed [`Region`].
    /// Refer to the documentation of [`Program::jvp`] for more information.
    ///
    /// This is the uncached half of a pair. Callers that publish their result, including the built-in
    /// [`DifferentiationDriver`], take it through [`RegionRef::jvp_shared`] counterpart instead, which serves
    /// the same program from the region's retained transform cache as a shared handle.
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        // This fused replay has its own all-zero shortcut that stages a primal instruction without consulting any
        // differentiation rule, so unresolved state anywhere in the attached closure (i.e., dormant rule regions
        // included, since differentiation is exactly what activates them) must be rejected up front. This guard covers
        // the public `Program::jvp` entry point, which builds the fused program through this function. Linearization
        // uses a separate replay through `RegionRef::linearize`, which carries its own matching guard.
        if self.contains_effect_in_closure(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: "program carries unresolved state and must be discharged before differentiation".to_string(),
            }
            .into());
        }

        let primal_input_count = self.input_ids().len();
        let tangent_input_count = self.input_ids().iter().try_fold(0usize, |count, input| {
            Ok::<_, DifferentiationError>(
                count + usize::from(!self.atoms()[input.index()].r#type().tangent()?.is_zero_space()),
            )
        })?;

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

            // Track the primal tracer and symbolic tangent for each source atom. Tangents of atoms not connected to an
            // input tangent (i.e., constants and dead inputs) are derived lazily as structural zeros typed with the
            // atom's tangent boundary type.
            let mut primals: Vec<Option<Tracer<TracingContext<V, O>>>> = vec![None; self.atoms().len()];
            let mut tangents: Vec<Option<MaybeZero<Tracer<TracingContext<V, O>>>>> = vec![None; self.atoms().len()];

            // Primal inputs become the leading inputs. One fresh tangent input is added afterward for each nonzero
            // differential space, so first-class metadata never acquires a fictitious tangent operand.
            for input_id in self.input_ids().iter().copied() {
                let r#type = self.atoms()[input_id.index()].r#type().into_owned();
                primals[input_id.index()] = Some(context.input(r#type));
            }
            for input_id in self.input_ids().iter().copied() {
                let primal_type = self.atoms()[input_id.index()].r#type();
                let tangent_type = primal_type.tangent()?;
                tangents[input_id.index()] = Some(if !tangent_type.is_zero_space() {
                    MaybeZero::Value(context.input(tangent_type.clone()))
                } else {
                    MaybeZero::Zero(tangent_type)
                });
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
                // an explicit dynamic-zero tangent. Other dynamic-output rules must retain any runtime extents needed
                // to materialize their structural-zero tangents.
                let all_input_tangents_are_zero =
                    !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero());
                let can_materialize_output_tangents_without_rules = || -> Result<bool, DifferentiationError> {
                    for (output_index, output_atom) in instruction.outputs().iter().copied().enumerate() {
                        let output_type = self.atoms()[output_atom.index()].r#type();
                        let tangent_type = output_type.tangent()?;
                        let can_materialize = tangent_type
                            .identities()
                            .all(|(position, _)| position != TypeIdentityPosition::Reference)
                            || (instruction.operation().is_zero(output_index) && output_type.as_ref() == &tangent_type);
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
                        instruction.operation().jvp(&context, &differentiation_driver, input_duals.as_slice())
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
                    if tangent.r#type().is_zero_space() {
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

    /// Builds the _fused_ Jacobian-Vector Product (JVP) [`Program`] of this borrowed [`Region`] through the region's
    /// retained transform cache, returning a shared handle to it.
    ///
    /// The fused forward-mode program is a pure function of the region's contents, so this returns exactly what
    /// [`RegionRef::jvp`] would produce, and every content-preserving copy of one sealed region shares one artifact.
    /// That is what keeps a shared region (e.g., a `jit_call` callee, a `condition` branch, or a `scan` body) from
    /// being differentiated once per program that attached it, and it additionally lets repeated binds of the derived
    /// program be interned by [`Arc`] identity by their consumers. Callers that want the owned [`Program`], or that
    /// must not publish their result, use [`Self::jvp`] instead.
    ///
    /// Recursive forward-mode construction of the region currently in flight on this thread is served without the
    /// cache, so a self-referential region behaves exactly as it does through [`Self::jvp`].
    pub fn jvp_shared(&self) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        let artifact = (*self).transform::<JvpTransform, _, DifferentiationError>((), |region, _| {
            Ok(TransformArtifact::new(vec![Arc::new(region.jvp()?)], ()))
        })?;
        let (programs, ()) = artifact.into_parts();
        let mut programs = programs.into_iter();
        let program = programs.next().unwrap();
        assert!(programs.next().is_none(), "fused JVP transform retained more than one program");
        Ok(program)
    }

    /// Linearizes this borrowed [`Region`] by replaying it once through a [`DifferentiationContext`] over a
    /// [`PartialEvaluationContext`] whose known-side parent is a fresh [`TracingContext`]. Refer to the documentation
    /// of [`Program::linearize`] for more information.
    ///
    /// This is the uncached half of a pair. Callers that publish their result take it through
    /// [`RegionRef::linearize_shared`] instead, which serves the same sub-programs from the region's
    /// retained transform cache as shared handles.
    pub fn linearize(&self) -> Result<Linearization<V, O>, DifferentiationError> {
        // This guard mirrors the fused `RegionRef::jvp` entry guard. Linearization replays through
        // `DifferentiationContext::bind` (whose own guard covers region-carrying instructions), but rejecting the whole
        // attached closure up front (dormant rule regions included) gives every structural differentiation entry point
        // one consistent, early diagnostic.
        if self.contains_effect_in_closure(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: "program carries unresolved state and must be discharged before differentiation".to_string(),
            }
            .into());
        }
        let primal_input_count = self.input_ids().len();
        let tangent_input_count = self.input_ids().iter().try_fold(0usize, |count, input| {
            Ok::<_, DifferentiationError>(
                count + usize::from(!self.atoms()[input.index()].r#type().tangent()?.is_zero_space()),
            )
        })?;

        // Keep one standalone handle to the primal builder. Every tracer and context clone is scoped below and must
        // be gone before this handle can be unwrapped at the trace boundary.
        let primal_context = TracingContext::<V, O>::new();
        let primal_builder = primal_context.builder().clone();
        let evaluation_context = PartialEvaluationContext::new(primal_context.clone());
        let differentiation_context = DifferentiationContext::new(evaluation_context.clone());

        // Seed the direct walk's boundary. Unknown tangent ordinals are already the canonical tangent input positions,
        // unlike the former fused program where they were offset by the primal-input count.
        let mut tangent_index = 0usize;
        let mut primal_input_atoms = Vec::with_capacity(primal_input_count);
        let input_duals = self
            .input_ids()
            .iter()
            .copied()
            .map(|input_atom| {
                let primal_type = self.atoms()[input_atom.index()].r#type().into_owned();
                let tangent_type = primal_type.tangent()?;
                let primal = primal_context.input(primal_type);
                primal_input_atoms.push(primal.atom_id()?);
                let tangent = if !tangent_type.is_zero_space() {
                    let tangent = evaluation_context.unknown_input(tangent_type.clone(), tangent_index);
                    tangent_index += 1;
                    MaybeZero::Value(PartialTracer::new(evaluation_context.clone(), tangent))
                } else {
                    MaybeZero::Zero(tangent_type)
                };
                Ok::<_, ProgramError>(DifferentiationTracer::new(
                    DifferentiationDual::new(
                        PartialTracer::new(evaluation_context.clone(), PartialEvaluationValue::known_input(primal)),
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

        // Drop the differentiation context before finalizing partial evaluation (its parent clone would otherwise
        // keep the residual builder alive and correctly trigger the escaped-builder guard).
        drop(differentiation_context);
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

        // Retain the explicit shape residuals needed to materialize a disconnected input cotangent. These residuals
        // are captured while the corresponding primal input is available and become ordinary trailing inputs of the
        // tangent program. Homogeneous operation families request no residuals. A live tangent input receives an
        // accumulated cotangent during transposition. Only a dead tangent input can require a newly constructed
        // disconnected zero, so only those inputs need their runtime geometry retained.
        let tangent_live_sets = tangent_program.live_sets();
        let differentiable_primal_inputs = self
            .input_ids()
            .iter()
            .map(|input| &self.atoms()[input.index()])
            .zip(primal_input_atoms)
            .map(|(input, primal)| Ok((input.r#type().tangent()?, input, primal)))
            .collect::<Result<Vec<_>, DifferentiationError>>()?
            .into_iter()
            .filter_map(|(tangent_type, input, primal)| (!tangent_type.is_zero_space()).then_some((input, primal)));
        let mut zero_residual_types = Vec::new();
        for (((input, primal_input), tangent_input), tangent_input_atom) in differentiable_primal_inputs
            .zip(tangent_program.inputs().take(tangent_input_count))
            .zip(tangent_program.input_ids().iter().copied().take(tangent_input_count))
        {
            if tangent_live_sets.atoms()[tangent_input_atom.index()] {
                continue;
            }
            let tangent_type = tangent_input.r#type().into_owned();
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
        // `Linearization::new` is the sole cross-program contract validation.
        Linearization::new(primal_program, tangent_program, residual_count).map_err(DifferentiationError::from)
    }

    /// Linearizes this borrowed [`Region`] through the region's retained transform cache, returning shared handles to
    /// the primal sub-program and the tangent sub-program together with the residual count relating them (i.e., the
    /// [`Linearization::into_parts`] triple behind [`Arc`]s).
    ///
    /// Linearization is a pure function of the region's contents, so this returns exactly what [`RegionRef::linearize`]
    /// would produce, and every content-preserving copy of one sealed region shares one artifact. That is what keeps a
    /// shared callee program from being linearized once per program that interned it, and it additionally lets repeated
    /// binds of the derived sub-programs be interned by [`Arc`] identity by their consumers. Callers that want the
    /// owned [`Linearization`], or that must not publish their result, use [`Self::linearize`] instead.
    ///
    /// Recursive linearization of the region currently being linearized on this thread is served without the cache,
    /// so a self-referential region behaves exactly as it does through [`Self::linearize`].
    pub fn linearize_shared(
        &self,
    ) -> Result<(Arc<Program<V, O, Vec<V>, Vec<V>>>, Arc<Program<V, O, Vec<V>, Vec<V>>>, usize), DifferentiationError>
    {
        let artifact = (*self).transform::<LinearizationTransform, _, DifferentiationError>((), |region, _| {
            let (primal, tangent, residual_count) = region.linearize()?.into_parts();
            Ok(TransformArtifact::new(vec![Arc::new(primal), Arc::new(tangent)], residual_count))
        })?;
        let (programs, residual_count) = artifact.into_parts();
        let mut programs = programs.into_iter();
        let primal = programs.next().unwrap();
        let tangent = programs.next().unwrap();
        assert!(programs.next().is_none(), "linearization transform retained more than two programs");
        Ok((primal, tangent, residual_count))
    }
}

impl<V: Value<Type: DifferentiableType>, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + ResidualZeroProvider<V::Type>,
{
    /// Builds the *fused* Jacobian-Vector Product (JVP) [`Program`] of this [`Program`]. Assume the input program
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
    #[inline]
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.entry_region_ref().jvp()
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
    /// Effect placement is inherited from [`PartialEvaluationContext`]. All-known effects stage once into the primal
    /// program, while tangent-dependent effects residualize once into the tangent program. Higher-order operations own
    /// their nested splitting through their existing differentiation and partial-evaluation rules. This function does
    /// not inspect or special-case their payloads. The final pair is validated only by [`Linearization::new`].
    #[inline]
    pub fn linearize(&self) -> Result<Linearization<V, O>, DifferentiationError> {
        self.entry_region_ref().linearize()
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
        Self::Operation: DifferentiableOperation<Self> + ResidualZeroProvider<Self::Type>,
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

        // Active inputs receive the caller-provided tangents. Captures share the same transform context but receive
        // only structural zero tangents, so they affect primal evaluation without affecting differentiation.
        let context = DifferentiationContext::new(self.clone());
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
        for output_dual in output_duals {
            let (primal, tangent) = output_dual.into_dual().into_parts();
            let tangent = match tangent {
                MaybeZero::Value(tangent) => tangent,
                MaybeZero::Zero(r#type) => {
                    let residuals = Self::Operation::capture_zero_residual_values(self, &primal, &r#type)?;
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
            + ResidualZeroProvider<Self::Type>,
    {
        if primal.parameters().next().is_none() {
            return Err(DifferentiationError::EmptyInput);
        }

        let input_structure = primal.parameter_structure();
        let input_values = primal.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();

        // The dual-seeding pass consumes `input_values` so we retain the active primals separately because dead tangent
        // inputs may later need their concrete runtime shapes captured as residuals before transposition.
        let primal_input_values = input_values.clone();
        let tangent_input_count = input_types.iter().try_fold(0usize, |count, r#type| {
            Ok::<_, DifferentiationError>(count + usize::from(!r#type.tangent()?.is_zero_space()))
        })?;

        // Active primals receive unknown tangent inputs. Captures are known primal inputs paired with structural zeros,
        // so they can be residualized when needed without increasing the pushforward's tangent arity.
        let evaluation_context = PartialEvaluationContext::new(self.clone());
        let differentiation_context = DifferentiationContext::new(evaluation_context.clone());
        let mut tangent_index = 0usize;
        let input_duals = input_values
            .into_iter()
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
                    PartialTracer::new(evaluation_context.clone(), PartialEvaluationValue::known_input(value)),
                    tangent,
                )?;
                Ok::<_, ProgramError>(DifferentiationTracer::new(dual, differentiation_context.clone()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = Input::To::<LinearizationTracer<Self>>::from_parameters(input_structure, input_duals)?;
        let capture_structure = capture.parameter_structure();
        let capture_duals = capture
            .into_parameters()
            .map(|value| -> Result<_, DifferentiationError> {
                let primal = PartialTracer::new(evaluation_context.clone(), PartialEvaluationValue::known_input(value));
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
        for output_dual in output_duals {
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

        // A dead active tangent input is the only one that can become a disconnected cotangent after transposition.
        let mut program = evaluation.program;
        let live_sets = program.live_sets();
        let differentiable_primal_inputs = primal_input_values
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
            if live_sets.atoms()[tangent_input_atom.index()] {
                continue;
            }
            let tangent_type = tangent_input.r#type().into_owned();
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
>(
    context: &C,
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
        .jvp(&ProjectedContext::new(context.clone()), &EmptyRegionDriver, projected_inputs.as_slice())?
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
        .map_err(Into::into)
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
fn capture_and_validate_zero_residual_atoms<V: Value, O: Operation<Type = V::Type> + ResidualZeroProvider<V::Type>>(
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
fn residualize_zero_from_residual_values<C: Context<Operation: ResidualZeroProvider<C::Type>>>(
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

///[`Region`] [`Transform`] marker for retained fused Jacobian-Vector Product (JVP) [`Program`]s.
pub(crate) struct JvpTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for JvpTransform {
    type Arguments = ();
    type Artifact = TransformArtifact<V, O, ()>;

    const DEFAULT_CACHE_CAPACITY: usize = 1;
}

/// [`Region`] [`Transform`] marker for retained linearized [`Program`]s.
pub(crate) struct LinearizationTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for LinearizationTransform {
    type Arguments = ();
    type Artifact = TransformArtifact<V, O, usize>;

    const DEFAULT_CACHE_CAPACITY: usize = 1;
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, Dimension,
        DimensionBounds, DimensionVariable, Shape,
    };
    use crate::contexts::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedMemberValue, ProjectedProgramOperation,
        ProjectedProgramType, ProjectedProgramValue,
    };
    use crate::contexts::{Context, EagerContext};
    use crate::differentiation::differentiate_at;
    use crate::differentiation::operations::tests::custom_jvp_regions_with_reference_state;
    use crate::differentiation::operations::{CustomJvpOperation, StopGradient, StopGradientOperation};
    use crate::operations::{
        ConditionOperation, CosOperation, Dot, DotDimensionNumbers, MulOperation, ParallelReduceOperation,
        ParallelReductionKind, Sin, SinOperation, WhileOperation,
    };
    use crate::parameters::{ParameterError, Placeholder};
    use crate::programs::{
        Concretizable, Operation, ProgramBuilder, ReferenceAddUpdateOperation, ReferenceFreezeOperation,
        ReferenceNewOperation, ReferenceReadOperation, ReferenceType, RegionId,
    };
    use crate::tests::test_condition_program;
    use crate::tracing::{NestedTracingContext, Trace};

    #[cfg(debug_assertions)]
    use crate::operations::TagOperation;

    use super::*;

    #[test]
    fn test_differentiation_dual_new_validates_and_canonicalizes_tangents() {
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
    fn test_differentiation_context_resolves_only_structural_zero_duals() {
        let context = DifferentiationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
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
        let context = DifferentiationContext::new(parent);
        let opaque = DifferentiationTracer::new(
            DifferentiationDual::new(primal, MaybeZero::Zero(ArrayType::scalar(DataType::F64))).unwrap(),
            context.clone(),
        );
        assert!(matches!(context.resolve(&opaque), ValueResolution::Opaque));
    }

    #[test]
    fn test_differentiation_context_rejects_state_before_symbolic_zero_fast_path() {
        let context = DifferentiationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let input = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Array(Array::scalar(1.0_f32))).unwrap(),
            context.clone(),
        );
        assert!(matches!(
            context.bind(ReferenceNewOperation::new(), Vec::new(), &[input]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_new` must be discharged before differentiation",
        ));
    }

    #[test]
    fn test_differentiation_context_skips_ruleless_operations_for_symbolic_zero_tangents() {
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
    fn test_differentiation_context_rejects_unresolved_references_in_custom_derivative_regions() {
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let regions = custom_jvp_regions_with_reference_state(&scalar_type);

        // A custom derivative cannot hide state in its rule regions when it consumes the active input directly.
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
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be discharged \
                    before differentiation",
        ));

        // Replacing the active input with a lifted value does not make the attached stateful rule dormant or valid.
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
        assert!(matches!(
            result,
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "`custom_jvp` carries unresolved state in an attached region and must be discharged \
                    before differentiation",
        ));
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
        assert_eq!(zero_count, 2, "expected exactly one boundary zero per zero tangent output, but got:\n{fused}");
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
    fn test_program_jvp_rejects_unresolved_references() {
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
        assert!(matches!(
            program.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "program carries unresolved state and must be discharged before differentiation",
        ));
    }

    #[test]
    fn test_program_jvp_rejects_unresolved_references_in_dormant_custom_derivative_regions() {
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

        // The entry region is pure, but whole-program validation must inspect the dormant custom rule closure.
        assert!(wrapped.effects().is_pure());
        assert!(matches!(
            wrapped.jvp(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "program carries unresolved state and must be discharged before differentiation",
        ));
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
    fn test_region_jvp_shared_reuses_only_identity_preserving_region_copies() {
        // A shared region is differentiated once and reused by every copy of it, which is what removes the repeated
        // re-transformation that programs attaching one shared `condition` branch or `scan` body would otherwise pay.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        let retained = callee.entry_region_ref().jvp_shared().unwrap();
        assert_eq!(retained.to_string(), callee.jvp().unwrap().to_string());
        assert!(Arc::ptr_eq(&callee.entry_region_ref().jvp_shared().unwrap(), &retained));

        // Two independently built programs that intern the same callee share its retained program, because importing
        // a region copies its complete reachable contents and therefore carries its transforms along.
        let mut first_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first_region = first_builder.intern_callee(&callee, None).unwrap();
        let mut second_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let second_region = second_builder.intern_callee(&callee, None).unwrap();
        assert!(Arc::ptr_eq(
            &RegionRef::new(&first_builder.regions, first_region).unwrap().jvp_shared().unwrap(),
            &retained,
        ));
        assert!(Arc::ptr_eq(
            &RegionRef::new(&second_builder.regions, second_region).unwrap().jvp_shared().unwrap(),
            &retained,
        ));

        // A region whose contents are genuinely rewritten starts over with a freshly derived program.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.add_instruction(SinOperation::new(), Vec::new(), vec![output], None).unwrap();
        let with_dead_work =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let before = with_dead_work.entry_region_ref().jvp_shared().unwrap();
        let simplified = with_dead_work.simplified().unwrap();
        let after = simplified.entry_region_ref().jvp_shared().unwrap();
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
        program
            .entry_region_ref()
            .insert_transform_artifact_for_testing::<JvpTransform, _>((), TransformArtifact::new(vec![unrelated], ()));

        // The recheck runs on the hit and reports the contract violation rather than serving the wrong derivative.
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| program.entry_region_ref().jvp_shared()))
                .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}",);
        assert!(message.contains("JvpTransform"), "{message}");
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

        // Test that a structurally zero tangent output (here, the tangent of a constant-valued program output) folds to
        // the known side during the split and is restored in the tangent sub-program as a staged `zero` instruction, so
        // the tangent sub-program keeps one tangent output per primal output.
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
    fn test_program_linearize_rejects_unresolved_references() {
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
        assert!(matches!(
            program.linearize(),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "program carries unresolved state and must be discharged before differentiation",
        ));
    }

    #[test]
    fn test_program_jvp_and_linearize_after_local_reference_discharge_across_condition() {
        let source = test_condition_program();

        // Forward mode, linearization, and transposition all consume the discharged program, so every derived
        // program must be pure and reference-free even though the source threads state through both branches.
        let jvp = source.clone().discharge_local_references(0, "differentiation").unwrap().jvp().unwrap();
        let linearization = source.discharge_local_references(0, "differentiation").unwrap().linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        for program in [&jvp, linearization.primal(), linearization.tangent(), &pullback] {
            assert!(!program.entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
            assert!(program.effects().is_pure());
        }

        // The true branch accumulates the input, so both public outputs remain differentiable.
        let predicate = ArrayIrValue::Array(Array::scalar(true));
        let initial = ArrayIrValue::Array(Array::scalar(4.0_f32));
        assert_eq!(
            jvp.interpret(vec![predicate.clone(), initial.clone(), ArrayIrValue::Array(Array::scalar(2.0_f32))]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(5.0_f32)),
                ArrayIrValue::Array(Array::scalar(5.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [ArrayIrValue::Array(Array::scalar(5.0_f32)), ArrayIrValue::Array(Array::scalar(5.0_f32))],
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::scalar(2.0_f32)), ArrayIrValue::Array(Array::scalar(3.0_f32))];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(5.0_f32))]));

        // The false branch replaces the state with a constant, so the frozen output has zero tangent and contributes
        // no cotangent to the input.
        let predicate = ArrayIrValue::Array(Array::scalar(false));
        let initial = ArrayIrValue::Array(Array::scalar(4.0_f32));
        assert_eq!(
            jvp.interpret(vec![predicate.clone(), initial.clone(), ArrayIrValue::Array(Array::scalar(2.0_f32))]),
            Ok(vec![
                ArrayIrValue::Array(Array::scalar(4.0_f32)),
                ArrayIrValue::Array(Array::scalar(9.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
                ArrayIrValue::Array(Array::scalar(0.0_f32)),
            ]),
        );
        let primal_outputs = linearization.primal().interpret(vec![predicate, initial]).unwrap();
        assert_eq!(
            primal_outputs[..2],
            [ArrayIrValue::Array(Array::scalar(4.0_f32)), ArrayIrValue::Array(Array::scalar(9.0_f32))],
        );
        let mut pullback_inputs =
            vec![ArrayIrValue::Array(Array::scalar(2.0_f32)), ArrayIrValue::Array(Array::scalar(3.0_f32))];
        pullback_inputs.extend_from_slice(&primal_outputs[2..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(2.0_f32))]));
    }

    #[test]
    fn test_program_jvp_and_linearize_after_local_reference_discharge_across_bounded_while() {
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut condition_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None)
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(true)));
        let condition = condition_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![predicate],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = body_builder.add_input(reference_type.into());
        let update = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0_f32)));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let initial = builder.add_input(ArrayIrType::Array(scalar_type));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let operation = WhileOperation::<ArrayIrType>::new().with_iteration_bound(2).unwrap();
        let reference = builder.add_instruction(operation, vec![condition, body], vec![reference], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The loop's mutated state becomes an ordinary carry, so the derived programs are pure and reference-free
        // and reverse mode remains available through the bounded loop.
        let jvp = source.clone().discharge_local_references(0, "differentiation").unwrap().jvp().unwrap();
        let linearization = source.discharge_local_references(0, "differentiation").unwrap().linearize().unwrap();
        let pullback = linearization.pullback().unwrap();
        for program in [&jvp, linearization.primal(), linearization.tangent(), &pullback] {
            assert!(!program.entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
            assert!(program.effects().is_pure());
        }

        // Two iterations accumulate the constant `1.0` into the state, so `f(x) = x + 2` and the tangent passes
        // through unscaled in both directions.
        assert_eq!(
            jvp.interpret(vec![
                ArrayIrValue::Array(Array::scalar(3.0_f32)),
                ArrayIrValue::Array(Array::scalar(2.0_f32)),
            ]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(5.0_f32)), ArrayIrValue::Array(Array::scalar(2.0_f32))]),
        );
        let primal_outputs =
            linearization.primal().interpret(vec![ArrayIrValue::Array(Array::scalar(3.0_f32))]).unwrap();
        assert_eq!(primal_outputs[0], ArrayIrValue::Array(Array::scalar(5.0_f32)));
        let mut pullback_inputs = vec![ArrayIrValue::Array(Array::scalar(4.0_f32))];
        pullback_inputs.extend_from_slice(&primal_outputs[1..]);
        assert_eq!(pullback.interpret(pullback_inputs), Ok(vec![ArrayIrValue::Array(Array::scalar(4.0_f32))]));
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
    fn test_region_linearize_shared_reuses_only_identity_preserving_region_copies() {
        // A shared callee is linearized once and reused by every copy of its sealed region, which is what removes the
        // repeated re-transformation that outer programs interning one callee would otherwise pay.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let callee = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        let (primal, tangent, residual_count) = callee.entry_region_ref().linearize_shared().unwrap();
        assert_eq!(residual_count, 1);
        assert_eq!(primal.to_string(), callee.linearize().unwrap().primal().to_string());
        assert_eq!(tangent.to_string(), callee.linearize().unwrap().tangent().to_string());

        // Two independently built programs that intern the same callee share its retained linearization, because
        // importing a region copies its complete reachable contents and therefore carries its transforms along.
        let mut first_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let first_region = first_builder.intern_callee(&callee, None).unwrap();
        let mut second_builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let second_region = second_builder.intern_callee(&callee, None).unwrap();
        let first = RegionRef::new(&first_builder.regions, first_region).unwrap().linearize_shared().unwrap();
        let second = RegionRef::new(&second_builder.regions, second_region).unwrap().linearize_shared().unwrap();
        assert!(Arc::ptr_eq(&first.0, &primal));
        assert!(Arc::ptr_eq(&first.1, &tangent));
        assert!(Arc::ptr_eq(&second.0, &primal));
        assert!(Arc::ptr_eq(&second.1, &tangent));

        // Simplification rebuilds every region, so it keeps the retained linearization only when the rebuild left the
        // region's contents untouched. A program carrying dead work is genuinely rewritten and must not reuse it.
        let simplified = callee.simplified().unwrap();
        assert!(Arc::ptr_eq(&simplified.entry_region_ref().linearize_shared().unwrap().0, &primal));

        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(SinOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        builder.add_instruction(SinOperation::new(), Vec::new(), vec![output], None).unwrap();
        let with_dead_work =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let before = with_dead_work.entry_region_ref().linearize_shared().unwrap();
        let after = with_dead_work.simplified().unwrap();
        let after = after.entry_region_ref().linearize_shared().unwrap();
        assert!(!Arc::ptr_eq(&after.0, &before.0));
        assert_eq!(after.0.to_string(), primal.to_string());

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
        let formal = dynamic_callee.entry_region_ref().linearize_shared().unwrap();
        let actual_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("actual", bounds))]),
        );
        let instantiated = dynamic_callee.with_instantiated_type_identities(&[actual_type]).unwrap().into_owned();
        let instantiated = instantiated.entry_region_ref().linearize_shared().unwrap();
        assert!(!Arc::ptr_eq(&instantiated.0, &formal.0));
        assert!(Arc::ptr_eq(&dynamic_callee.entry_region_ref().linearize_shared().unwrap().0, &formal.0));
    }

    #[test]
    fn test_region_linearize_shared_invalidates_rebased_attached_regions() {
        /// Builds a program whose entry applies `operation` to its single scalar input.
        fn branch(operation: ArrayOperation<Array>) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder.add_instruction(operation, Vec::new(), vec![input], None).unwrap()[0];
            builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        // A region's retained transforms cover its complete reachable contents, but the identifiers it attaches its
        // descendants by are relative to the arena it is sealed in. Program `first` attaches the sine branch as both
        // branches of a condition, so its entry's linearization is the derivative of sine.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let sine = builder.import_region(branch(ArrayOperation::Sin(SinOperation::new())).entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(ConditionOperation::new(), vec![sine, sine], vec![predicate, input], None)
            .unwrap()[0];
        let first = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let retained = first.entry_region_ref().linearize_shared().unwrap();

        // Re-sealing a copy of that entry into an arena whose region 0 is the cosine branch changes what the copy
        // computes, so it must not be served the transforms derived from the sine branch.
        let rebased = Program::<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>>::new(
            vec![Placeholder; 2],
            vec![Placeholder],
            vec![branch(ArrayOperation::Cos(CosOperation::new())).entry_region().clone(), first.entry_region().clone()],
            RegionId::new(1),
        )
        .unwrap();
        let derived = rebased.entry_region_ref().linearize_shared().unwrap();
        assert!(!Arc::ptr_eq(&derived.0, &retained.0));
        assert!(!Arc::ptr_eq(&derived.1, &retained.1));

        // The freshly derived tangent program differentiates the cosine branch the rebased arena actually attaches,
        // which is the wrong-derivative failure that serving the retained artifact would produce.
        assert_eq!(derived.1.to_string(), rebased.linearize().unwrap().tangent().to_string());
        assert_ne!(derived.1.to_string(), retained.1.to_string());

        // The source program keeps its own retained artifact, because only the re-sealed copy was rebased.
        assert!(Arc::ptr_eq(&first.entry_region_ref().linearize_shared().unwrap().0, &retained.0));
    }

    #[test]
    fn test_region_linearization_does_not_retain_its_source_cache() {
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
        let (primal, tangent, _) = materialized.entry_region_ref().linearize_shared().unwrap();

        drop(program);
        drop(materialized);
        assert!(!source_cache.is_alive());

        // The returned programs can outlive the source cache because built-in linearization constructs a fresh entry
        // and imports only strict descendants from the acyclic source arena; neither artifact embeds the source root.
        drop(primal);
        drop(tangent);
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
            (),
            TransformArtifact::new(vec![unrelated.clone(), unrelated], 0),
        );

        // The recheck runs on the hit and reports the contract violation rather than serving the wrong derivative.
        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| program.entry_region_ref().linearize_shared()))
                .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}",);
        assert!(message.contains("LinearizationTransform"), "{message}");
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

        // Builds a single-instruction region tagging its input with `key`.
        fn tagged_program(key: &str) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let output = builder.add_instruction(TagOperation::new(key), Vec::new(), vec![input], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        let program = tagged_program("saved");
        let other = tagged_program("recomputed");

        // The two programs differ only in operation metadata, so the corruption below is invisible to everything
        // except rendering that carries the key.
        assert_ne!(program.to_string(), other.to_string());

        // Publish the *other* region's genuine linearization against this region, which is the state a `jvp` rule that
        // is not a structural function of its operation would leave behind.
        let (primal, tangent, residual_count) = other.entry_region_ref().linearize_shared().unwrap();
        program.entry_region_ref().insert_transform_artifact_for_testing::<LinearizationTransform, _>(
            (),
            TransformArtifact::new(vec![primal, tangent], residual_count),
        );

        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| program.entry_region_ref().linearize_shared()))
                .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}",);
        assert!(message.contains("LinearizationTransform"), "{message}");
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_linearize_shared_debug_recheck_detects_constant_payload_changes() {
        // This test pins the constant-payload part of the rendering contract at a cache hit. The two regions below
        // differ only in one `Atom::Constant` literal, so this test panics exactly when `Program::render` carries that
        // payload into the rendering compared by the recheck.

        // Builds a single-instruction region scaling its input by the constant `factor`.
        fn scaled_program(factor: f64) -> Program<Array, ArrayOperation<Array>, Vec<Array>, Vec<Array>> {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            let constant = builder.add_constant(Array::scalar(factor));
            let output =
                builder.add_instruction(MulOperation::new(), Vec::new(), vec![input, constant], None).unwrap()[0];
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        let program = scaled_program(2.0);
        let other = scaled_program(3.0);

        // The ordinary rendering is semantically complete enough to distinguish the embedded literal.
        assert_ne!(program.to_string(), other.to_string());

        // Publish the *other* region's genuine linearization against this region, which is the state a `jvp` rule
        // that is not a structural function of the constants it embeds would leave behind.
        let (primal, tangent, residual_count) = other.entry_region_ref().linearize_shared().unwrap();
        program.entry_region_ref().insert_transform_artifact_for_testing::<LinearizationTransform, _>(
            (),
            TransformArtifact::new(vec![primal, tangent], residual_count),
        );

        let panicked =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| program.entry_region_ref().linearize_shared()))
                .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}",);
        assert!(message.contains("LinearizationTransform"), "{message}");
    }

    #[test]
    fn test_jvp() {
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

        // The builder's `jvp` terminal serves top-level concrete values through their `Value::ExecutionDomain`
        // declarations. A concrete array input recovers the eager array domain, so both dual halves are concrete.
        let (value, tangent) = differentiate_at(Array::scalar(2.0)).jvp(Array::scalar(3.0), |x| x.sin()).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent.to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // Complex duals flow through the same rules. The jvp of z² pushes the tangent ż to `2z · ż`
        // at a genuinely complex point.
        let z = num_complex::Complex::new(0.7f64, -0.3f64);
        let tangent_seed = num_complex::Complex::new(1.0f64, 0.5f64);
        let (value, tangent) =
            differentiate_at(Array::scalar(z)).jvp(Array::scalar(tangent_seed), |x| Ok(x.clone() * x)).unwrap();
        assert_eq!(value.elements::<num_complex::Complex<f64>>().unwrap(), vec![z * z]);
        assert_eq!(tangent.elements::<num_complex::Complex<f64>>().unwrap(), vec![(z + z) * tangent_seed],);

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

        // With no leaf value to recover a context from, the builder's `jvp` terminal reports that differentiation
        // requires at least one input leaf.
        assert_eq!(
            differentiate_at(Vec::<Array>::new()).jvp(Vec::new(), |x| Ok(x)).unwrap_err(),
            DifferentiationError::EmptyInput,
        );

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
    fn test_linearize() {
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
                if message == "pushforward tangent 0 has type token[] but its primal boundary requires tangent type zero[]",
        ));

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
        let program = pushforward.program().to_string();
        assert!(program.contains("mul"), "{program}");
        assert!(
            !program.contains("sin") && !program.contains("cos"),
            "the untaken branch must never be traced: {program}",
        );
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        assert_abs_diff_eq!(pushforward.apply((zero, Array::scalar(1.0))).unwrap().to_f64s()[0], 6.0, epsilon = 1e-9,);

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
        let context = EagerContext::<ProjectedProgramValue, ProjectedProgramOperation>::new();
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3)),
        )
        .unwrap();
        let output = jvp_projected_operation(&context, &ProjectedMemberOperation::<2>, &[input]).unwrap().remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)));
        assert!(matches!(tangent, MaybeZero::Value(ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3))),));

        // Structural zeros cross the same adapter as types and therefore do not stage or materialize member values.
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)),
            MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),
        )
        .unwrap();
        let output = jvp_projected_operation(&context, &ProjectedMemberOperation::<2>, &[input]).unwrap().remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)));
        assert!(matches!(tangent, MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),));
    }
}
