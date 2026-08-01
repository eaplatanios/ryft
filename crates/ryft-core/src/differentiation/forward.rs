use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, DifferentiationError, TransposableOperation};
use crate::macros::check_count;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::operations::math::AddOperation;
use crate::parameters::{Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{
    PartialEvaluationContext, PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, PartialTracer,
    PartialValue, PartiallyEvaluatableOperation,
};
use crate::programs::ProgramError;
use crate::programs::atoms::{Atom, MaybeZero};
use crate::programs::identities::TypeIdentityPosition;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::{
    BindingRegionDriver, EmptyRegionDriver, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver,
};
use crate::programs::types::{Type, TypeError, Typed};
use crate::programs::values::{ProjectedValue, Value, ValueProjection};
use crate::tracing::{Tracer, TracingContext};

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
    /// Returns a [`TypeError`] if a live `tangent` does not have the tangent type required by `primal`.
    #[inline]
    pub fn new<T: Into<MaybeZero<V>>>(primal: V, tangent: T) -> Result<Self, TypeError> {
        let tangent_type = primal.r#type().tangent();
        let tangent = match tangent.into() {
            MaybeZero::Zero(_) => MaybeZero::Zero(tangent_type),
            MaybeZero::Value(tangent) => {
                if tangent.r#type().as_ref() != &tangent_type {
                    return Err(TypeError::invalid(format!(
                        "tangent type {} does not match type {} required by primal type {}",
                        tangent.r#type(),
                        tangent_type,
                        primal.r#type(),
                    )));
                }
                if tangent_type.is_zero_space() { MaybeZero::Zero(tangent_type) } else { MaybeZero::Value(tangent) }
            }
        };
        Ok(Self { primal, tangent })
    }

    /// Creates a new [`DifferentiationDual`] with a [`MaybeZero::Zero`] tangent carrying the primal's concrete
    /// tangent boundary [`Type`]. A primal with no tangent space uses its first-class zero-space [`Type`].
    #[inline]
    pub fn new_with_zero_tangent(primal: V) -> Self {
        let tangent = MaybeZero::Zero(primal.r#type().tangent());
        Self::new(primal, tangent).unwrap()
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
///     `ẏ = (∂f/∂x)(x) · ẋ`. Here `live(ẋ)` contains one SSA input for each primal input whose tangent type is
///     not a zero differential space, and `live(ẏ)` similarly contains one Single Static Assignment (SSA) output for
///     each primal output whose tangent type is not a zero differential space. A tangent in a zero differential space
///     can only be zero, so it requires no SSA slot; the corresponding primal value remains in the primal program, and
///     structured callable APIs reconstruct the uniquely determined typed zero where their public result structure
///     requires it. The tangent program is linear in `ẋ`, with the linearization point `x` entering only through the
///     residuals `r`.
///
/// This is the domain-free, interpretation-free core shared by every linearization entry point. It carries only the
/// two sub-programs and the residual count that relates them, leaving the concrete primal outputs to be recovered by
/// callers that interpret [`primal`](Self::primal) under a value semantics of their choice.
pub struct Linearization<V: Value, O: Operation<V::Type>> {
    /// Nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal outputs
    /// `y = f(x)` followed by the residuals `r`, its trailing [`residual_count`](Self::residual_count) outputs, which
    /// form the residual environment consumed by the tangent sub-program.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Linear tangent sub-program `(live(ẋ), r) ↦ live(ẏ)`. It has one leading Single Static Assignment (SSA) input for
    /// each primal input whose tangent type is not a zero differential space, followed by the residuals `r`, and one
    /// SSA output for each primal output whose tangent type is not a zero differential space.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

    /// Number of residuals `r` threaded from the primal sub-program into the tangent sub-program (i.e., the count of
    /// the trailing outputs of [`primal`](Self::primal) and of the trailing inputs of [`tangent`](Self::tangent)).
    residual_count: usize,
}

impl<V: Value, O: Operation<V::Type>> Linearization<V, O> {
    /// Creates a new [`Linearization`] from its parts, validating the boundary contract documented on [`Linearization`]
    /// where `primal` produces its primal outputs followed by its trailing `residual_count` residuals, and `tangent`
    /// consumes one tangent input per non-zero differential input followed by those same residuals and produces one
    /// tangent output per non-zero differential output. Violations are reported as
    /// [`MalformedProgram`](ProgramError::MalformedProgram) errors: too few
    /// primal outputs or tangent inputs to hold the residuals, sub-program boundary counts that disagree with each
    /// other, or a residual whose primal output type differs from its tangent input type. [`Program::linearize`] is the
    /// function that typically calls this function and constructs [`Linearization`]s.
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
        let differentiable_primal_inputs =
            primal.inputs().filter(|input| !input.r#type().tangent().is_zero_space()).collect::<Vec<_>>();
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
            .filter(|output| !output.r#type().tangent().is_zero_space())
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
            let tangent_type = primal_type.tangent();
            if tangent_input.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "linearization tangent input {} has type {} but primal input type {} requires tangent type {}",
                    index,
                    tangent_input.r#type(),
                    primal_type,
                    tangent_type,
                )));
            }
        }
        for (index, (primal_output, tangent_output)) in
            differentiable_primal_outputs.into_iter().zip(tangent.outputs()).enumerate()
        {
            let primal_type = primal_output.r#type();
            let tangent_type = primal_type.tangent();
            if tangent_output.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "linearization tangent output {} has type {} but primal output type {} requires tangent type {}",
                    index,
                    tangent_output.r#type(),
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
                    residual.r#type(),
                    input.r#type(),
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
    /// the consuming residual value, this function leaves the tangent sub-program in the primal operation family `0`
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
        O: TransposableOperation<V, O> + From<ZeroOperation<V::Type>> + From<AddOperation>,
    {
        // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
        // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are
        // not folded into captured factors here. The subtraction cannot underflow because `Self::new` validated that
        // the tangent program consumes at least `residual_count` inputs.
        let tangent_input_count = self.tangent.input_ids().len() - self.residual_count;
        let with_respect_to = (0..tangent_input_count).collect::<Vec<_>>();
        self.tangent.transpose_with_respect_to(with_respect_to.as_slice())
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
    /// positions or types, and the tangent-type mapping is not generally invertible. `primal_input_types` and
    /// `primal_output_types` preserve that complete boundary so [`apply`](Self::apply) can validate all public tangent
    /// leaves, remove the zero-space inputs before replay, and insert typed zeros at the correct output positions
    /// afterward.
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
        primal_input_types: Vec<C::Type>,
        primal_output_types: Vec<C::Type>,
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
                    "pushforward residual {index} has type {} in the pushforward program \
                     but carries a value of type {}",
                    input.r#type(),
                    residual.r#type(),
                )));
            }
        }
        let live_input_count = primal_input_types.iter().filter(|r#type| !r#type.tangent().is_zero_space()).count();
        if live_input_count != tangent_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward program consumes {tangent_input_count} tangent inputs but its public boundary has {} \
                 nonzero differential inputs",
                live_input_count,
            )));
        }
        let live_output_count = primal_output_types.iter().filter(|r#type| !r#type.tangent().is_zero_space()).count();
        if live_output_count != program.output_ids().len() {
            return Err(ProgramError::MalformedProgram(format!(
                "pushforward program produces {} tangent outputs but its public boundary has {} nonzero differential \
                 outputs",
                program.output_ids().len(),
                live_output_count,
            )));
        }
        Ok(Self {
            context,
            program,
            residuals,
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
        C: Zero<C::Value>,
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
            let tangent_type = primal_type.tangent();
            if value.r#type().as_ref() != &tangent_type {
                return Err(ProgramError::MalformedProgram(format!(
                    "pushforward tangent {} has type {} but its primal boundary requires tangent type {}",
                    index,
                    value.r#type(),
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
                    input.r#type(),
                    expected.r#type(),
                )));
            }
        }

        // Close the compact tangent boundary over the primal residuals and replay it in the originating context.
        program_inputs.extend(self.residuals.iter().cloned());
        let mut tangent_outputs = self.program.interpret_in_context(&self.context, program_inputs)?.into_iter();

        // Reconstruct the complete flattened public output boundary. Consume one program result for each nonzero
        // tangent space and materialize the uniquely determined typed zero for every omitted zero-space leaf.
        let outputs = self
            .primal_output_types
            .iter()
            .map(|r#type| {
                let tangent_type = r#type.tangent();
                if tangent_type.is_zero_space() {
                    self.context.zero(&tangent_type)
                } else {
                    tangent_outputs.next().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "pushforward program omitted a nonzero differential output".to_string(),
                        )
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Construction validated the expected output count. Reject a malformed replay rather than silently dropping
        // an unexpected compact-program result.
        if tangent_outputs.next().is_some() {
            return Err(ProgramError::MalformedProgram(
                "pushforward program produced more nonzero differential outputs than its public boundary".to_string(),
            ));
        }

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
    /// Builds the compact fused forward-mode program of `region`. The returned program maps `[primals...,
    /// live(tangents)...]` to `[primal outputs..., live(tangent outputs)...]`, where `live(...)` omits every boundary
    /// leaf whose tangent type is a zero differential space (see [`DifferentiableType::is_zero_space`]). Callers must
    /// derive their tangent liveness masks from the same region boundary types this construction filters on and restore
    /// the omitted boundary leaves as structural zeros.
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, DifferentiationError>;

    /// Linearizes `region` into its primal and tangent program halves, re-entering the active differentiation
    /// machinery.
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
    ) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, DifferentiationError> {
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

impl<V: Value, O: Operation<V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
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
        + From<ZeroOperation<C::Type>>,
{
    fn jvp_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
    ) -> Result<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, DifferentiationError> {
        region.jvp()
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
///     `Self: From<ZeroOperation<T>>` bound that the nested-region differentiation drivers require. Higher-order
///     payload rules request nested forward-mode and linearization work through their instruction-scoped
///     [`DifferentiationDriver`], whose concrete implementation establishes the finite program-level bounds at its
///     construction site. Output-level semantic queries such as [`Operation::is_zero`] are forwarded by the base
///     operation dispatcher and therefore introduce no additional witness bounds.
pub trait DifferentiableOperation<C: Context>: Operation<C::Type> {
    /// Applies this operation's capture-free forward-mode rule, mapping the input duals `(xᵢ, ẋᵢ)` to the output duals
    /// `(y, ẏ) = (f(x), Σᵢ (∂f/∂xᵢ)(x) · ẋᵢ)` where `f` is the function this operation computes. The returned vector
    /// must be aligned with this operation's outputs, each element pairing a primal output value with its tangent, both
    /// bound through `context`.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] through which the rule binds the primal and tangent [`Operation`]s it synthesizes.
    ///   - `driver`: [`DifferentiationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to
    ///     attached [`Region`](crate::Region)s.
    ///   - `inputs`: Input [`DifferentiationDual`]s aligned with this operation's inputs/operands.
    fn jvp<D: DifferentiationDriver<C>>(
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

    /// Consumes this tracer and returns the[`DifferentiationDual`] that it carries.
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

/// Value type flowing through the closures of the partial-evaluation-backed differentiation entry points (i.e.,
/// [`ForwardModeDifferentiate::linearize`], [`ReverseModeDifferentiate::vjp`](crate::ReverseModeDifferentiate::vjp),
/// [`ReverseModeDifferentiate::gradient`](crate::ReverseModeDifferentiate::gradient), and their derivatives). It is a
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
/// structural zero, the [`bind`](Context::bind) fast path skips a region-free operation's rule only if each output
/// zero tangent can be constructed without runtime identity operands (or the zero-producing primal itself is reusable).
/// Eligible operations therefore construct no zero values until a boundary [`materialize`](MaybeZero::materialize)s
/// one through the inner context's [`Zero`] capability.
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
        + From<ZeroOperation<C::Type>>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        // Constants are independent of every differentiation input and so their tangents are structural zeros.
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent.lift(constant)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }

    fn bind<O: Into<C::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[DifferentiationTracer<C>],
    ) -> Result<Vec<DifferentiationTracer<C>>, ProgramError> {
        let operation = operation.into();

        // Unwrap the input tracers into context-free duals, run the rule against those, and rewrap the produced duals
        // with this context, mirroring how `BatchingContext::bind` unwraps to `ArrayBatch`es and rewraps.
        let input_duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();

        // All-zero fast path mirroring `Program::jvp`. When an operation consumes at least one input and every input
        // tangent is a structural zero, skip its rule only when each output tangent can later be materialized without
        // runtime identity operands. Zero-input operations remain excluded so their dedicated rules keep handling
        // primal synthesis and tangent typing.
        let zero_input_tangents = !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero());

        // Dynamic one already relies on this routing to stage an explicit dynamic-zero tangent. Other dynamic-output
        // rules that still return structural zeros need Phase 6 extent residuals, which must also revisit the
        // region-carrying escape below.
        let reusable_zero_outputs = if zero_input_tangents {
            // Region-free operations can be inspected without reproducing the parent's region-identity instantiation.
            // Region-carrying rules retain the established fast path here. Their explicit dynamic-output migration
            // owns the corresponding policy when those operation families move to first-class extent operands.
            if operation.region_slots().is_empty() {
                let input_types =
                    input_duals.iter().map(|dual| dual.primal().r#type().into_owned()).collect::<Vec<_>>();
                let output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
                let mut reusable_zero_outputs = Vec::new();
                let can_materialize = output_types.iter().enumerate().all(|(output_index, output_type)| {
                    let tangent_type = output_type.tangent();
                    if operation.is_zero(output_index) && output_type == &tangent_type {
                        reusable_zero_outputs.push(output_index);
                        true
                    } else {
                        tangent_type.identities().all(|(position, _)| position != TypeIdentityPosition::Reference)
                    }
                });
                can_materialize.then_some(reusable_zero_outputs)
            } else {
                Some(Vec::new())
            }
        } else {
            None
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
                        Ok(DifferentiationDual::new_with_zero_tangent(primal))
                    }
                })
                .collect::<Result<Vec<_>, TypeError>>()?
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
}

impl<V: Value<Type: DifferentiableType>, O: Operation<V::Type>> RegionRef<'_, V, O>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + From<ZeroOperation<V::Type>>,
{
    /// Builds the *fused* Jacobian-Vector Product (JVP) [`Program`] of this borrowed [`Region`](crate::Region).
    /// Refer to the documentation of [`Program::jvp`] for more information.
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let primal_input_count = self.input_ids().len();
        let tangent_input_count = self
            .input_ids()
            .iter()
            .filter(|input| !self.atoms()[input.index()].r#type().tangent().is_zero_space())
            .count();

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
                let tangent_type = primal_type.tangent();
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
                            None => MaybeZero::Zero(primal.r#type().tangent()),
                        };
                        Ok(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new(primal, tangent)?)
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // TODO(eaplatanios): Does this need to be updated in Phase 6?
                // All-zero fast path: skip the operation's rule only when every input tangent is structural zero and
                // every output zero tangent can later be materialized without runtime identity operands (or by reusing
                // a zero-producing primal). Zero-input operations remain excluded so their dedicated rules keep
                // handling primal synthesis and tangent typing. Dynamic one already relies on this routing to stage
                // an explicit dynamic-zero tangent. Other dynamic-output rules that still return structural zeros need
                // Phase 6 extent residuals.
                let all_input_tangents_are_zero =
                    !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero());
                let can_materialize_output_tangents_without_rules = || {
                    instruction.outputs().iter().copied().enumerate().all(|(output_index, output_atom)| {
                        let output_type = self.atoms()[output_atom.index()].r#type();
                        let tangent_type = output_type.tangent();
                        tangent_type.identities().all(|(position, _)| position != TypeIdentityPosition::Reference)
                            || (instruction.operation().is_zero(output_index) && output_type.as_ref() == &tangent_type)
                    })
                };
                let driver = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;
                let output_duals = if all_input_tangents_are_zero && can_materialize_output_tangents_without_rules() {
                    let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                    context
                        .stage_operation(instruction.operation().clone(), driver, primal_inputs.as_slice())?
                        .into_iter()
                        .enumerate()
                        .map(|(output_index, primal)| {
                            // When the primal instruction is itself known to produce zero and its output already
                            // has the required tangent type, the primal Single Static Assignment (SSA) value is the
                            // canonical materialized tangent. Reusing it preserves source-relative geometry such as
                            // explicit shaped-constructor extents and avoids inventing a nullary dynamic zero at the
                            // fused Jacobian-Vector Product (JVP) boundary.
                            let primal_type = primal.r#type();
                            if instruction.operation().is_zero(output_index)
                                && primal_type.as_ref() == &primal_type.tangent()
                            {
                                DifferentiationDual::new(primal.clone(), primal)
                            } else {
                                Ok(DifferentiationDual::new_with_zero_tangent(primal))
                            }
                        })
                        .collect::<Result<Vec<_>, TypeError>>()?
                } else {
                    let differentiation_driver = RecursiveDifferentiationDriver { driver: &driver };
                    instruction.operation().jvp(&context, &differentiation_driver, input_duals.as_slice())?
                };

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
                    let tangent = match &tangents[output_atom.index()] {
                        Some(tangent) => tangent.clone(),
                        None => MaybeZero::Zero(
                            primals[output_atom.index()]
                                .as_ref()
                                .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                                .r#type()
                                .tangent(),
                        ),
                    };
                    if tangent.r#type().is_zero_space() {
                        Ok(None)
                    } else {
                        Ok(Some(tangent.materialize(&context)?.atom_id()?))
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

    /// Linearizes this borrowed [`Region`](crate::Region) by replaying it once through a [`DifferentiationContext`]
    /// over a [`PartialEvaluationContext`] whose known-side parent is a fresh [`TracingContext`]. Refer to the
    /// documentation of [`Program::linearize`] for more information.
    pub fn linearize(&self) -> Result<Linearization<V, O>, DifferentiationError> {
        let primal_input_count = self.input_ids().len();
        let tangent_input_count = self
            .input_ids()
            .iter()
            .filter(|input| !self.atoms()[input.index()].r#type().tangent().is_zero_space())
            .count();

        // Keep one standalone handle to the primal builder. Every tracer and context clone is scoped below and must
        // be gone before this handle can be unwrapped at the trace boundary.
        let primal_context = TracingContext::<V, O>::new();
        let primal_builder = primal_context.builder().clone();
        let evaluation_context = PartialEvaluationContext::new(primal_context.clone());
        let differentiation_context = DifferentiationContext::new(evaluation_context.clone());

        // Seed the direct walk's boundary. Unknown tangent ordinals are already the canonical tangent input positions,
        // unlike the former fused program where they were offset by the primal-input count.
        let mut tangent_index = 0usize;
        let input_duals = self
            .input_ids()
            .iter()
            .copied()
            .map(|input_atom| {
                let primal_type = self.atoms()[input_atom.index()].r#type().into_owned();
                let tangent_type = primal_type.tangent();
                let primal = primal_context.input(primal_type);
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
                let regions = ReplayRegionDriver::new(*self, instruction.regions(), &region_mappings)?;
                differentiation_context.bind(instruction.operation().clone(), regions, inputs)
            },
        )?;

        // Split the direct output duals. Primal halves must be known tracers in the primal builder. Structural-zero
        // tangent halves become residualized typed zeros so the tangent program preserves the source output arity. A
        // value tangent that folded to known is malformed: rules must preserve input-independent zeros structurally,
        // and accepting any other known value would turn the tangent program into an affine map.
        let staged_zero = |r#type: V::Type| {
            let mut outputs = evaluation_context.residualize(ZeroOperation::new(r#type), Vec::new(), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok::<_, ProgramError>(outputs.remove(0))
        };
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
                MaybeZero::Zero(r#type) => staged_zero(r#type)?,
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
        let tangent_program = evaluation.program;

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
}

impl<V: Value<Type: DifferentiableType>, O: Operation<V::Type>> Program<V, O, Vec<V>, Vec<V>>
where
    O: PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + DifferentiableOperation<PartialEvaluationContext<TracingContext<V, O>>>
        + From<ZeroOperation<V::Type>>,
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
    /// with a typed structural zero tangent. Structural zeros are materialized as typed [`ZeroOperation`] instructions
    /// only when a nonzero differential output requires a real value, preserving a compact
    /// `(primal_outputs ++ live_tangent_outputs)` program contract.
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
/// concrete vs [`Tracer`]), not by a separate trait. Values from a *different* trace are detected lazily, like
/// everything else about staging: a foreign tracer fails the builder-identity check either when an operation binds it
/// (via [`StagingContext::stage_operation`]) or when it escapes through a trace boundary (i.e., the boundary output
/// checks), with [`ProgramError::MismatchedProgramBuilders`].
pub trait ForwardModeDifferentiate: Context<Type: DifferentiableType> {
    /// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward, with this
    /// [`Context`] executing (or staging) the differentiated operations. Refer to the documentation of the [`jvp`]
    /// function for information on the forward-mode transform and its arguments.
    fn jvp<
        F: FnOnce(Input::To<DifferentiationTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<DifferentiationTracer<Self>>,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<DifferentiationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primals: Input,
        tangents: Input::To<Self::Value>,
    ) -> Result<(Output::To<Self::Value>, Output::To<Self::Value>), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Operation: DifferentiableOperation<Self>,
    {
        if primals.parameters().next().is_none() {
            return Err(DifferentiationError::EmptyInput);
        }

        let primal_structure = primals.parameter_structure();
        let tangent_structure = tangents.parameter_structure();
        if tangent_structure != primal_structure {
            return Err(ParameterError::MismatchedParameterStructures {
                left_structure: format!("{primal_structure:?}"),
                right_structure: format!("{tangent_structure:?}"),
            }
            .into());
        }

        // Wrap each `(primal, tangent)` as a dual stamped with the forward-mode context so that the closure's value
        // sugar dispatches through it, then run the closure directly on those duals.
        let context = DifferentiationContext::new(self.clone());
        let input_duals = primals
            .into_parameters()
            .zip(tangents.into_parameters())
            .map(|(primal, tangent)| {
                Ok::<_, ProgramError>(DifferentiationTracer::new(
                    DifferentiationDual::new(primal, tangent)?,
                    context.clone(),
                ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let input = Input::To::<DifferentiationTracer<Self>>::from_parameters(primal_structure, input_duals)?;
        let output = function(input)?;

        // Split each output dual into its primal value and its materialized tangent.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let mut primal_outputs = Vec::with_capacity(output_duals.len());
        let mut tangent_outputs = Vec::with_capacity(output_duals.len());
        for output_dual in output_duals {
            let (primal, tangent) = output_dual.into_dual().into_parts();
            tangent_outputs.push(tangent.materialize(self)?);
            primal_outputs.push(primal);
        }
        let primal_output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;
        let tangent_output = Output::To::<Self::Value>::from_parameters(output_structure, tangent_outputs)?;
        Ok((primal_output, tangent_output))
    }

    /// Linearizes `function` at `primals`, returning the primal output and a reusable [`Pushforward`], with this
    /// [`Context`] executing (or staging) the primal-side operations. Refer to the documentation of the [`linearize`]
    /// function for information on the linearization transform and its arguments.
    fn linearize<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Output::To<Self::Value>, Pushforward<Self, Input, Output::To<Self::Value>>), DifferentiationError>
    where
        Self::Operation: PartiallyEvaluatableOperation<Self>
            + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
            + From<ZeroOperation<Self::Type>>,
    {
        if primals.parameters().next().is_none() {
            return Err(DifferentiationError::EmptyInput);
        }

        let input_structure = primals.parameter_structure();
        let input_values = primals.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let tangent_input_count = input_types.iter().filter(|r#type| !r#type.tangent().is_zero_space()).count();

        // Wrap each primal as a dual over a partial-evaluation context wrapping this context. The primal half is a
        // known value and the tangent half is an unknown seeded as a leading residual-program input, in primal-input
        // order.
        let evaluation_context = PartialEvaluationContext::new(self.clone());
        let differentiation_context = DifferentiationContext::new(evaluation_context.clone());
        let mut tangent_index = 0usize;
        let input_duals = input_values
            .into_iter()
            .map(|value| {
                let primal_type = value.r#type().into_owned();
                let tangent_type = primal_type.tangent();
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
        let output = function(input)?;

        // Split each output dual into its known primal value and its tangent. Primal work depends only on the known
        // primal inputs, so every primal half must have folded to a known value. Zero differential outputs remain
        // structural and are omitted from the executable pushforward boundary. A value tangent that folded to known is
        // malformed. A well-formed rule must preserve an input-independent zero as `MaybeZero::Zero`, while accepting
        // an arbitrary known value would silently turn the pushforward into an affine map.
        let output_structure = output.parameter_structure();
        let output_duals = output.into_parameters().collect::<Vec<_>>();
        let staged_zero = |r#type: Self::Type| {
            let mut outputs = evaluation_context.residualize(ZeroOperation::new(r#type), Vec::new(), &[])?;
            check_count!("output", outputs, 1, ProgramError);
            Ok::<_, ProgramError>(outputs.remove(0))
        };
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
                MaybeZero::Zero(r#type) => staged_zero(r#type)?,
            };
            primal_outputs.push(primal);
            tangent_outputs.push(tangent);
        }
        let output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), primal_outputs)?;

        // All tracer-stamped context clones are dropped here, so the accumulated pushforward program can be finalized.
        drop(differentiation_context);
        let evaluation = evaluation_context.into_evaluation(tangent_outputs)?;

        // The pushforward program's inputs are the leading tangent unknowns followed by the residuals materialized
        // during the trace. Collect the residual values in input order.
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

        // Close the pushforward program over the linearization-point residuals behind the reusable callable.
        let pushforward =
            Pushforward::new(self.clone(), evaluation.program, residuals, input_types, output_types, output_structure)?;
        Ok((output, pushforward))
    }
}

impl<C: Context<Type: DifferentiableType>> ForwardModeDifferentiate for C {}

/// Evaluates `function` on the primal `primals` and propagates the tangent `tangents` forward by running the closure
/// **directly on [`DifferentiationTracer`] duals** (i.e., the single forward-mode entry point, and the analogue of
/// [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)). For `f` computing `y = f(x)`, this
/// computes the dual `(y, ẏ) = (f(x), (∂f/∂x)(x) · ẋ)` — the primal output paired with the Jacobian-vector product
/// of the input tangent `ẋ`.
///
/// The transform recovers a [`Context`] from
/// the input's leaf values through [`Value::ExecutionDomain`], exactly like [`batch`](crate::batching::batch):
/// staged [`Tracer`]s recover their trace, transform tracers recover their transform level, and concrete values recover
/// the eager backend domain they name, so the transform composes uniformly across the whole stack. Each input is then
/// paired with its tangent as a dual over a [`DifferentiationContext`] wrapping the recovered context, and `function`
/// runs directly on those duals, with each operation the closure performs (e.g., `x.sin()`, `x * y`, etc.) dispatching
/// its [`jvp`](DifferentiableOperation::jvp) rule through [`Context::bind`]. Eager-versus-staged behavior is absorbed
/// entirely by that context:
///
///   - Over an **eager** context both dual halves are concrete, so the closure sees real primal values (i.e., it can
///     branch on them with `if x.concretize()? { … }`, print them, or otherwise use Rust control flow driven by the
///     primal) and a staged data-dependent `while` combinator differentiates by running directly at the concrete
///     primals, with no iteration bound needed.
///   - Over a **staging** context the same closure stages the primal and tangent operations into the enclosing trace
///     operation by operation (this is how a fused JVP computation is built under an outer trace), and branching on a
///     primal errors because it is a [`Tracer`] with no concrete payload.
///
/// The closure executes exactly as written: no dead code is trimmed, and observable effects fire as the closure runs.
/// Structural zero tangents stay symbolic between operations and are materialized through the recovered context's
/// [`Zero`] capability only at the output boundary. Transforms nest. Inside the closure, an inner transform invoked
/// on a dual's [`DifferentiationContext`] (a [`Context`] carrying these transforms itself) differentiates through the
/// duals, composing reverse-over-forward and higher-order forward modes.
///
/// Inputs with no leaf values are rejected with a [`DifferentiationError::EmptyInput`] error as there is nothing to
/// recover a context from, and differentiating a function of no inputs is degenerate anyway.
/// [`ForwardModeDifferentiate::jvp`] is the explicit-context method form behind this function.
#[inline]
pub fn jvp<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<Operation: DifferentiableOperation<V::ExecutionDomain>> + Zero<V>,
        >,
    F: FnOnce(Input::To<DifferentiationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<
            V,
            To<V> = Input,
            Family: ParameterizedFamily<DifferentiationTracer<V::ExecutionDomain>>,
            ParameterStructure: Debug + PartialEq,
        >,
    Output: Parameterized<DifferentiationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
>(
    function: F,
    primals: Input,
    tangents: Input::To<V>,
) -> Result<(Output::To<V>, Output::To<V>), DifferentiationError> {
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.jvp(function, primals, tangents)
}

/// Linearizes `function` at `primals`, returning the primal output and a reusable [`Pushforward`] (i.e., the
/// analogue of [JAX's `linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html)). For `f` computing
/// `y = f(x)`, this computes `y` together with the reusable linear pushforward map `ẋ ↦ ẏ = (∂f/∂x)(x) · ẋ` at the
/// linearization point `x`, so that differentiating once serves any number of tangents.
///
/// This is the partial-evaluation sibling of [`jvp`]. Where `jvp` runs the closure once per `(primal, tangent)` pair,
/// this function runs the closure once on [`DifferentiationTracer`] duals over a [`PartialEvaluationContext`] wrapping
/// the context recovered from the input's leaf values through [`Value::ExecutionDomain`] (exactly like [`jvp`] and
/// [`batch`](crate::batch)), with each dual's primal half *known* at its primal value and its tangent half *unknown*.
/// Primal-side operations are then all-known and fold through the recovered context itself (i.e., executing eagerly
/// under an eager context or staging into the enclosing trace under a staging one, so that linearization composes under
/// an outer trace), while tangent-side operations residualize into the accumulated pushforward program `(ẋ, r) ↦ ẏ`,
/// which is linear in `ẋ` with the linearization point entering only through the residuals `r` recovered along the way.
/// The returned [`Pushforward`] closes that program over those residuals, so that [`Pushforward::apply`] pushes any
/// number of tangents through the function's Jacobian at this point without re-tracing or re-differentiating.
///
/// Because the closure's primal halves carry concrete values under an eager context, host control flow on primals works
/// exactly as under [`jvp`]: the closure can branch on a primal (e.g., `if x.concretize()? { … }`), the untaken branch
/// is never traced at all, and a data-dependent `while` combinator differentiates by running directly at the concrete
/// primals. This matches JAX's `linearize`/`grad` tracing semantics, where the same JVP interpreter runs over a
/// partial-evaluation trace instead of the eval trace.
///
/// Inputs with no leaf values are rejected with a [`DifferentiationError::EmptyInput`] error: there is nothing to
/// recover a context from, and linearizing a function of no inputs is degenerate anyway.
/// [`ForwardModeDifferentiate::linearize`] is the explicit-context method form behind this function.
#[inline]
pub fn linearize<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + PartiallyEvaluatableOperation<
                    TracingContext<<V::ExecutionDomain as Domain>::Constant, <V::ExecutionDomain as Domain>::Operation>,
                > + From<ZeroOperation<V::Type>>,
            > + Zero<V>,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
>(
    function: F,
    primals: Input,
) -> Result<(Output::To<V>, Pushforward<V::ExecutionDomain, Input, Output::To<V>>), DifferentiationError> {
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.linearize(function, primals)
}

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use half::{bf16, f16};
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::backends::arrays::{Array, ArrayOperation};
    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::{Context, EagerContext};
    use crate::operations::collectives::{CollectiveKind, CollectiveOperation};
    use crate::operations::differentiation::{StopGradient, StopGradientOperation};
    use crate::operations::math::{Dot, DotDimensionNumbers, MulOperation, Sin, SinOperation};
    use crate::parameters::{ParameterError, Placeholder};
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::operations::Operation;
    use crate::programs::values::Concretizable;
    use crate::tracing::{NestedTracingContext, Trace};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_differentiation_dual_new_validates_and_canonicalizes_tangents() {
        let differentiable = DifferentiationDual::new(Scalar::F64(2.0), Scalar::F64(3.0)).unwrap();
        let (primal, tangent) = differentiable.into_parts();
        assert_eq!(primal, Scalar::F64(2.0));
        assert!(matches!(tangent, MaybeZero::Value(Scalar::F64(3.0))));

        let differentiable_zero =
            DifferentiationDual::new(Scalar::F64(2.0), MaybeZero::<Scalar>::Zero(DataType::Boolean)).unwrap();
        let (primal, tangent) = differentiable_zero.into_parts();
        assert_eq!(primal, Scalar::F64(2.0));
        assert!(matches!(tangent, MaybeZero::Zero(DataType::F64)));

        let non_differentiable = DifferentiationDual::new(Scalar::Token, Scalar::Zero).unwrap();
        let (primal, tangent) = non_differentiable.into_parts();
        assert_eq!(primal, Scalar::Token);
        assert!(matches!(tangent, MaybeZero::Zero(DataType::Zero)));

        assert!(matches!(
            DifferentiationDual::new(Scalar::Token, Scalar::Token),
            Err(TypeError::Invalid { message })
                if message == "tangent type token does not match type zero required by primal type token",
        ));
    }

    #[test]
    fn test_differentiation_context_skips_ruleless_operations_for_symbolic_zero_tangents() {
        // `stop_gradient` severs the collective's tangent input. The differentiation context must therefore bind the
        // primal collective without consulting its absent JVP rule, while preserving the live tangent of the other
        // addition operand.
        let (primal, tangent) = jvp(
            |input| {
                let severed = input.stop_gradient();
                let mut outputs = severed.context().bind(
                    CollectiveOperation::new("batch".to_string(), CollectiveKind::PSum),
                    Vec::new(),
                    &[severed.clone()],
                )?;
                Ok(input + outputs.remove(0))
            },
            Array::scalar(2.0),
            Array::scalar(1.0),
        )
        .unwrap();
        assert_eq!(primal.to_f64s(), vec![4.0]);
        assert_eq!(tangent.to_f64s(), vec![1.0]);
    }

    #[test]
    fn test_program_jvp() {
        // Test that the fused JVP program of `f(x) = sin(x)` presents the `[x, ẋ] ↦ [sin(x), cos(x) · ẋ]` boundary.
        // The primal input leads, one fresh tangent input follows, and the outputs are the primal output followed by
        // its tangent.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let fused = program.jvp().unwrap();
        assert_eq!(fused.input_ids().len(), 2);
        assert_eq!(fused.output_ids().len(), 2);
        assert_eq!(
            fused.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = sin %0
                    %3:f64 = cos %0
                    %4:f64 = mul %3 %1
                in (%2, %4)
            "}
            .trim_end(),
        );
        let outputs = fused.interpret(vec![Scalar::F64(3.0), Scalar::F64(1.0)]).unwrap();
        assert_eq!(outputs, vec![Scalar::F64(3.0f64.sin()), Scalar::F64(3.0f64.cos())]);

        // Test that structural zero tangents are materialized as typed `zero` instructions only at the output boundary,
        // preserving the `(primal_outputs ++ tangent_outputs)` contract. Both zero producers are covered: the
        // constant-valued output's tangent is a *derived* zero (a constant is connected to no input tangent) and the
        // `stop_gradient` output's tangent is a *rule-returned* zero, and exactly one `zero` instruction is staged per
        // zero tangent output with none staged mid-program.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::F64(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![input, constant]).unwrap()[0];
        let severed = builder.add_instruction(StopGradientOperation, Vec::new(), vec![scaled]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![scaled, constant, severed], vec![Placeholder], vec![Placeholder; 3])
            .unwrap();
        let fused = program.jvp().unwrap();
        assert_eq!(fused.input_ids().len(), 2);
        assert_eq!(fused.output_ids().len(), 6);
        let zero_count =
            fused.instructions().iter().filter(|instruction| instruction.operation().name() == "zero").count();
        assert_eq!(zero_count, 2, "expected exactly one boundary zero per zero tangent output, but got:\n{fused}");
        let outputs = fused.interpret(vec![Scalar::F64(3.0), Scalar::F64(1.0)]).unwrap();
        assert_eq!(
            outputs,
            vec![
                Scalar::F64(6.0),
                Scalar::F64(2.0),
                Scalar::F64(6.0),
                Scalar::F64(2.0),
                Scalar::F64(0.0),
                Scalar::F64(0.0),
            ],
            "the fused outputs must be the primal outputs [3 * 2, 2, 3 * 2] followed by the tangents [2 * 1, 0, 0]",
        );
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
    fn test_program_linearize() {
        // Test that directly linearizing `f(x) = sin(x)` produces the primal sub-program `x ↦ (sin(x), cos(x))`,
        // whose trailing output is the `cos(x)` residual, and the linear tangent sub-program `(ẋ, r) ↦ r · ẋ`.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, Vec::new(), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 1);
        assert_eq!(
            linearization.primal().to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = sin %0
                    %2:f64 = cos %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            linearization.tangent().to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
        let primal_outputs = linearization.primal().interpret(vec![Scalar::F64(3.0)]).unwrap();
        assert_eq!(primal_outputs, vec![Scalar::F64(3.0f64.sin()), Scalar::F64(3.0f64.cos())]);
        let tangent_outputs =
            linearization.tangent().interpret(vec![Scalar::F64(1.0), Scalar::F64(3.0f64.cos())]).unwrap();
        assert_eq!(tangent_outputs, vec![Scalar::F64(3.0f64.cos())]);

        // The ordinary direct-linearization boundary carries `cos(x)` as a residual, so transposing its tangent map
        // must not invoke known-intermediate replay. In particular, the primal-only `cos` chain must be absent from
        // the pullback: it runs once in the primal program and crosses the boundary as the pullback's residual input.
        let pullback = linearization.pullback().unwrap();
        assert!(
            pullback.instructions().iter().all(|instruction| instruction.operation().name() != "cos"),
            "ordinary linearize -> transpose unexpectedly replayed a primal-only producer:\n{pullback}",
        );
        assert_eq!(
            pullback.interpret(vec![Scalar::F64(1.0), Scalar::F64(3.0f64.cos())]).unwrap(),
            vec![Scalar::F64(3.0f64.cos())],
        );

        // Test that a structurally zero tangent output (here, the tangent of a constant-valued program output) folds to
        // the known side during the split and is restored in the tangent sub-program as a staged `zero` instruction, so
        // the tangent sub-program keeps one tangent output per primal output.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::F64(2.0));
        let scaled = builder.add_instruction(MulOperation, Vec::new(), vec![input, constant]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![scaled, constant], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().output_ids().len(), 2 + linearization.residual_count());
        assert_eq!(linearization.tangent().output_ids().len(), 2);
        let tangent_inputs = [vec![Scalar::F64(1.0)], vec![Scalar::F64(2.0); linearization.residual_count()]].concat();
        let tangent_outputs = linearization.tangent().interpret(tangent_inputs).unwrap();
        assert_eq!(
            tangent_outputs,
            vec![Scalar::F64(2.0), Scalar::F64(0.0)],
            "the tangent outputs must be [2 * ẋ] for the scaled output and a restored zero for the constant output",
        );

        // Boundary-degenerate programs retain their canonical signatures. A zero-input constant program has one
        // primal output and one zero tangent output, while a zero-output program still retains the dead tangent
        // input corresponding to its primal input.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let constant = builder.add_constant(Scalar::F64(5.0));
        let program = builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![constant], Vec::new(), vec![Placeholder]).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.residual_count(), 0);
        assert_eq!(linearization.primal().interpret(Vec::new()).unwrap(), vec![Scalar::F64(5.0)]);
        assert_eq!(linearization.tangent().interpret(Vec::new()).unwrap(), vec![Scalar::F64(0.0)]);

        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        builder.add_input(DataType::F64);
        let program = builder.build::<Vec<Scalar>, Vec<Scalar>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let linearization = program.linearize().unwrap();
        assert_eq!(linearization.primal().input_ids().len(), 1);
        assert!(linearization.primal().output_ids().is_empty());
        assert_eq!(linearization.tangent().input_ids().len(), 1);
        assert!(linearization.tangent().output_ids().is_empty());
    }

    #[test]
    fn test_program_linearize_restores_pruned_tangent_inputs() {
        // `stop_gradient` disconnects `dy` from the tangent output while `y` remains live in the primal program.
        // The split must restore the canonical `dy` boundary slot after partial-evaluation liveness pruning.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let (_, program) = NestedTracingContext::trace(
            context,
            |inputs| Ok(vec![inputs[0].sin()? + inputs[1].stop_gradient()]),
            vec![DataType::F64, DataType::F64],
        )
        .unwrap();
        let linearization = program.into_simplified().unwrap().linearize().unwrap();
        assert_eq!(linearization.primal().output_ids().len(), 1 + linearization.residual_count());
        assert_eq!(linearization.tangent().input_ids().len(), 2 + linearization.residual_count());

        let mut primal_outputs = linearization
            .primal()
            .interpret_in_context(&context, vec![Scalar::from(0.7), Scalar::from(1.3)])
            .unwrap();
        let residuals = primal_outputs.split_off(1);
        let mut tangent_inputs = vec![Scalar::from(1.0), Scalar::from(123.0)];
        tangent_inputs.extend(residuals);
        assert_eq!(
            linearization.tangent().interpret_in_context(&context, tangent_inputs),
            Ok(vec![Scalar::from(0.7f64.cos())]),
        );
    }

    #[test]
    fn test_jvp() {
        // `ForwardModeDifferentiate::jvp` on an explicit context runs the closure directly on duals. For
        // `f(x) = sin(x)` at `x = 2` along the tangent `ẋ = 3`, the primal output is `sin(2)` and the tangent
        // output is `3 · cos(2)`.
        let (value, tangent) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .jvp(|x| x.sin(), Scalar::from(2.0), Scalar::from(3.0))
            .unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The free `jvp` serves top-level concrete values through their `Value::ExecutionDomain` declarations.
        // A plain `Scalar` input recovers the eager scalar domain, so both dual halves are concrete.
        let (value, tangent): (Scalar, Scalar) = jvp(|x| x.sin(), Scalar::from(2.0), Scalar::from(3.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // Complex duals flow through the same rules. The jvp of z² pushes the tangent ż to `2z · ż`
        // at a genuinely complex point.
        let z = num_complex::Complex::new(0.7f64, -0.3f64);
        let tangent_seed = num_complex::Complex::new(1.0f64, 0.5f64);
        let (value, tangent) = jvp(|x| Ok(x.clone() * x), Scalar::from(z), Scalar::from(tangent_seed)).unwrap();
        assert_eq!(value, Scalar::from(z * z));
        assert_eq!(tangent, Scalar::from((z + z) * tangent_seed));

        // Eager JVP duals carry concrete primal halves, so ordinary host control flow can branch on a primal without
        // tracing the untaken branch.
        let (value, tangent) =
            jvp(|x| Ok(if x.concretize()? { x.clone() * x.sin()? } else { -x }), Scalar::from(0.7), Scalar::from(1.0))
                .unwrap();
        assert_abs_diff_eq!(value, 0.7 * 0.7f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);

        // Inputs without tangent spaces retain first-class zero-space boundary leaves. Their only valid tangent value
        // is `Scalar::Zero`, and output structural zeros materialize with the same type.
        let (value, tangent): ((Scalar, Scalar), (Scalar, Scalar)) =
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
                .jvp(
                    |inputs| Ok(inputs),
                    (Scalar::from(2.0f64), Scalar::from(3i32)),
                    (Scalar::from(1.0f64), Scalar::Zero),
                )
                .unwrap();
        assert_eq!(value, (Scalar::from(2.0f64), Scalar::from(3i32)));
        assert_eq!(tangent, (Scalar::from(1.0f64), Scalar::Zero));

        let (value, tangent) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .jvp(|token| Ok(token), Scalar::Token, Scalar::Zero)
            .unwrap();
        assert_eq!(value, Scalar::Token);
        assert_eq!(tangent, Scalar::Zero);
        assert!(matches!(
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
                .jvp(|token| Ok(token), Scalar::Token, Scalar::Token),
            Err(DifferentiationError::Program(ProgramError::Type(TypeError::Invalid { message })))
                if message == "tangent type token does not match type zero required by primal type token",
        ));

        // Under an active trace, the free `jvp` recovers the staging context from its tracer inputs instead, so it
        // composes inside traced code without threading a context. The closure stages the fused primal and tangent
        // operations into the enclosing trace.
        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, tangent) = jvp(|x| x.sin(), inputs[0].clone(), inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![DataType::F64, DataType::F64],
        )
        .unwrap();
        let outputs = program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The same composition preserves zero-space leaves for tokens instead of attempting to stage token
        // arithmetic while constructing the enclosing program.
        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, tangent) = jvp(|token| Ok(token), inputs[0].clone(), inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![DataType::Token, DataType::Zero],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![Scalar::Token, Scalar::Zero]), Ok(vec![Scalar::Token, Scalar::Zero]),);

        // Tangents pair with primals leaf-for-leaf and so a tangent structure that does not match the primal
        // structure is rejected.
        assert!(matches!(
            jvp(|x: Vec<_>| Ok(x), vec![Scalar::from(1.0)], vec![Scalar::from(1.0), Scalar::from(2.0)]).unwrap_err(),
            DifferentiationError::Program(ProgramError::Parameter(
                ParameterError::MismatchedParameterStructures { .. },
            )),
        ));

        // With no leaf value to recover a context from, the free `jvp` reports an invalid input count.
        assert_eq!(
            jvp(
                |x: Vec<DifferentiationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>| Ok(x),
                Vec::<Scalar>::new(),
                Vec::new(),
            )
            .unwrap_err(),
            DifferentiationError::EmptyInput,
        );

        // The unified scalar domain supports both half-precision variants through their ordinary scalar operations.
        assert_eq!(
            jvp(|x| Ok(x.clone() + x), Scalar::BF16(bf16::from_f32(3.0)), Scalar::BF16(bf16::ONE)),
            Ok((Scalar::BF16(bf16::from_f32(6.0)), Scalar::BF16(bf16::from_f32(2.0)))),
        );
        assert_eq!(
            jvp(|x| Ok(x.clone() + x), Scalar::F16(f16::from_f32(3.0)), Scalar::F16(f16::ONE)),
            Ok((Scalar::F16(f16::from_f32(6.0)), Scalar::F16(f16::from_f32(2.0)))),
        );
    }

    #[test]
    fn test_linearize() {
        // `ForwardModeDifferentiate::linearize` on an explicit context runs the closure once at the primal point and
        // returns the primal output together with a reusable pushforward: applying it pushes any number of tangents
        // through the Jacobian at that point without re-tracing or re-differentiating.
        let (value, pushforward) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .linearize(|x| x.sin(), Scalar::from(2.0))
            .unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pushforward.apply(Scalar::from(1.0)).unwrap(), 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(pushforward.apply(Scalar::from(3.0)).unwrap(), 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The free `linearize` serves top-level concrete values through their `Value::ExecutionDomain` declarations.
        // Primal work executes eagerly at the concrete linearization point while the pushforward program accumulates.
        let (value, pushforward) = linearize(|x| x.sin(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pushforward.apply(Scalar::from(1.0)).unwrap(), 2.0f64.cos(), epsilon = 1e-9);

        let (value, pushforward) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .linearize(|token| Ok(token), Scalar::Token)
            .unwrap();
        assert_eq!(value, Scalar::Token);
        assert_eq!(pushforward.apply(Scalar::Zero), Ok(Scalar::Zero));
        assert!(matches!(
            pushforward.apply(Scalar::Token),
            Err(ProgramError::MalformedProgram(message))
                if message == "pushforward tangent 0 has type token but its primal boundary requires tangent type zero",
        ));

        // Under an active trace, the free `linearize` recovers the staging context from its tracer input instead,
        // so primal work stages into the enclosing trace and the pushforward replays there when applied.
        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, pushforward) = linearize(|x| x.sin(), inputs[0].clone())?;
                let tangent = pushforward.apply(inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![DataType::F64, DataType::F64],
        )
        .unwrap();
        let outputs = program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, pushforward) = linearize(|token| Ok(token), inputs[0].clone())?;
                let tangent = pushforward.apply(inputs[1].clone())?;
                Ok(vec![value, tangent])
            },
            vec![DataType::Token, DataType::Zero],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![Scalar::Token, Scalar::Zero]), Ok(vec![Scalar::Token, Scalar::Zero]),);

        // The closure can branch on a *primal* with host control flow, because the duals' primal halves carry concrete
        // known values under an eager context. For `x = 3` the predicate is true, so `f(x) = x * x` linearizes to the
        // pushforward `ẋ ↦ 2x · ẋ = 6ẋ`, and the untaken `sin(x)` branch is never traced at all. Neither `sin` nor its
        // `cos` derivative can appear in the pushforward program.
        let (value, pushforward) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .linearize(
                |x| Ok(if x.concretize().unwrap() { x.clone() * x } else { x.sin().unwrap() }),
                Scalar::from(3.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value, 9.0, epsilon = 1e-9);
        let program = pushforward.program().to_string();
        assert!(program.contains("mul"), "{program}");
        assert!(
            !program.contains("sin") && !program.contains("cos"),
            "the untaken branch must never be traced: {program}",
        );
        assert_abs_diff_eq!(pushforward.apply(Scalar::from(1.0)).unwrap(), 6.0, epsilon = 1e-9);

        // Structured inputs preserve their leaf order through a reusable multi-input pushforward.
        let function = |(a, b): (
            LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
        )| Ok(a.clone() * b + a.sin()?);
        let (_, pushforward) = linearize(function, (Scalar::from(0.5), Scalar::from(1.3))).unwrap();
        assert_abs_diff_eq!(
            pushforward.apply((Scalar::from(1.0), Scalar::from(0.0))).unwrap(),
            1.3 + 0.5f64.cos(),
            epsilon = 1e-9,
        );
        assert_abs_diff_eq!(pushforward.apply((Scalar::from(0.0), Scalar::from(1.0))).unwrap(), 0.5, epsilon = 1e-9,);

        // With no leaf value to recover a context from, the free `linearize` reports an invalid input count.
        assert_eq!(
            linearize(
                |x: Vec<LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>| Ok(x),
                Vec::<Scalar>::new(),
            )
            .map(|(outputs, _)| outputs)
            .unwrap_err(),
            DifferentiationError::EmptyInput
        );
    }
}
