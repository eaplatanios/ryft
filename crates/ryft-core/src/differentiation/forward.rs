use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{DifferentiableType, TransposableOperation};
use crate::macros::check_count;
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{PartialEvaluationContext, PartialEvaluationInput, PartiallyEvaluatableOperation};
use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::Typed;

/// Represents a differentiation _dual_ value which is a _primal_ value paired with a _tangent_ value. In the
/// context of differentiating a function `f(x)`, the value `y = f(x)` is the primal value and its tangent `ẏ` is
/// the directional derivative of `f` at `x` along an input tangent (i.e., perturbation direction) `ẋ` (i.e., the
/// Jacobian-vector product `ẏ = (∂f/∂x)(x) · ẋ`). Forward-mode differentiation propagates a dual `(x, ẋ)` at the
/// input to the dual `(y, ẏ) = (f(x), (∂f/∂x)(x) · ẋ)` at the output. This is the data that the per-operation
/// [`jvp`](DifferentiableOperation::jvp) rules consume and produce.
#[derive(Clone, Debug)]
pub struct DifferentiationDual<V: Typed> {
    /// Primal value of this dual.
    primal: V,

    /// Tangent value of this dual. Note that this can be a [`MaybeZero::Zero`] enabling structural zero propagation.
    tangent: MaybeZero<V>,
}

impl<V: Value> DifferentiationDual<V> {
    /// Creates a new [`DifferentiationDual`].
    #[inline]
    pub fn new<Tangent: Into<MaybeZero<V>>>(primal: V, tangent: Tangent) -> Self {
        Self { primal, tangent: tangent.into() }
    }

    /// Creates a new [`DifferentiationDual`] with a [`MaybeZero::Zero`] tangent value.
    #[inline]
    pub fn new_with_zero_tangent(primal: V) -> Self {
        let tangent = MaybeZero::Zero(primal.r#type().into_owned());
        Self { primal, tangent }
    }

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
/// transform (i.e., [`Program::linearize`]). Linearization splits a program's fused JVP program
/// `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ)` by known-ness into:
///
///   - the [`primal`](Self::primal) sub-program `x ↦ (y, r)`, computing the primal outputs `y = f(x)` together with
///     the residuals `r` (i.e., the intermediate values of the derivative computation that depend only on `x`; e.g.,
///     `cos(x)` when `f` is `sin`), and
///   - the [`tangent`](Self::tangent) sub-program `(ẋ, r) ↦ ẏ`, computing the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`.
///     It is linear in `ẋ`, with the linearization point `x` entering only through the residuals `r`.
///
/// This is the domain-free, interpretation-free core shared by every linearization entry point. It carries only the
/// two sub-programs and the residual count that relates them, leaving the concrete primal outputs to be recovered by
/// callers that interpret [`primal`](Self::primal) under a value semantics of their choice.
pub struct Linearization<V: Value, O: Clone + Operation<V::Type>> {
    /// Nonlinear primal sub-program `x ↦ (y, r)`. It takes the primal inputs `x` and produces the primal outputs
    /// `y = f(x)` followed by the residuals `r`, its trailing [`residual_count`](Self::residual_count) outputs, which
    /// form the residual environment consumed by the tangent sub-program.
    primal: Program<V, O, Vec<V>, Vec<V>>,

    /// Linear tangent sub-program `(ẋ, r) ↦ ẏ`. It takes the tangent inputs `ẋ` followed by the residuals `r` and
    /// produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`.
    tangent: Program<V, O, Vec<V>, Vec<V>>,

    /// Number of residuals `r` threaded from the primal sub-program into the tangent sub-program (i.e., the count of
    /// the trailing outputs of [`primal`](Self::primal) and of the trailing inputs of [`tangent`](Self::tangent)).
    residual_count: usize,
}

impl<V: Value, O: Clone + Operation<V::Type>> Linearization<V, O> {
    /// Creates a new [`Linearization`] from its parts, validating the boundary contract documented on [`Linearization`]
    /// where `primal` produces its primal outputs followed by its trailing `residual_count` residuals, and `tangent`
    /// consumes one tangent input per primal input followed by those same residuals and produces one tangent output per
    /// primal output. Violations are reported as [`MalformedProgram`](ProgramError::MalformedProgram) errors: too few
    /// primal outputs or tangent inputs to hold the residuals, sub-program boundary counts that disagree with each
    /// other, or a residual whose primal output type differs from its tangent input type. [`Program::linearize`] is the
    /// function that typically calls this function and constructs [`Linearization`]s.
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residual_count: usize,
    ) -> Result<Self, ProgramError> {
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
        if tangent_input_count != primal.input_ids().len() {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization tangent program consumes {tangent_input_count} tangent inputs \
                 while the primal program consumes {} inputs",
                primal.input_ids().len(),
            )));
        }
        if tangent.output_ids().len() != primal_output_count {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization tangent program produces {} outputs \
                 while the primal program produces {primal_output_count} primal outputs",
                tangent.output_ids().len(),
            )));
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

    /// Returns the linear tangent sub-program `(ẋ, r) ↦ ẏ`. It takes the tangent inputs `ẋ` followed by the residuals
    /// `r` and produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`. The sub-program is linear in `ẋ`, with the
    /// linearization point `x` entering only through the residuals `r`.
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

    /// Returns the forward-mode pushforward program `(ẋ, r) ↦ ẏ` which takes the tangent inputs `ẋ` followed by the
    /// residuals `r` and produces the tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`. Because linearization already produces
    /// the pushforward as its unknown half, this is the [`tangent`](Self::tangent) sub-program itself, cloned (i.e.,
    /// the identity counterpart of [`pullback`](Self::pullback), which derives its program by transposition).
    #[inline]
    pub fn pushforward(&self) -> Program<V, O, Vec<V>, Vec<V>> {
        self.tangent.clone()
    }

    /// Builds the reverse-mode pullback program `(ȳ, r) ↦ x̄` by transposing the [`tangent`](Self::tangent) sub-program.
    /// It takes the output cotangents `ȳ` followed by the residuals `r` and produces the input cotangents
    /// `x̄ = (∂f/∂x)(x)ᵀ · ȳ`. It is the derived third member of this [`Linearization`]'s program family, alongside the
    /// stored [`primal`](Self::primal) and [`tangent`](Self::tangent) sub-programs. Rather than re-keying each bilinear
    /// operation of the tangent sub-program into a closed captured factor (e.g., folding a scalar `Mul` against a known
    /// operand into a multiply-by-a-captured-constant) by folding the consuming residual value, this function leaves
    /// the tangent sub-program in the primal operation family `O` and transposes it through
    /// [`Program::transpose_with_respect_to`]. The tangent sub-program's inputs are `(ẋ, r)`, and so it is transposed
    /// with respect to the leading tangent inputs `ẋ` while the trailing [`residual_count`](Self::residual_count)
    /// residual inputs are held as known parameters. Partition-aware transposition then threads each known residual
    /// through to the pullback as a pullback input (consumed by the adjoint operation that the bilinear operation's
    /// transpose rule stages), rather than folding it into a captured factor, so the returned pullback program stays
    /// over the primal operation family `O` and produces the cotangents of the linear tangent inputs only.
    #[inline]
    pub fn pullback(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError>
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
    C: Context<Operation: Clone>,
    Input: Parameterized<C::Value>,
    Output: Parameterized<C::Value, Family: ParameterizedFamily<C::Value>>,
> Pushforward<C, Input, Output>
{
    /// Creates a new [`Pushforward`] closing `program` over the linearization-point `residuals`, validating the
    /// contract documented on [`Pushforward`] where `program` consumes the flat tangents followed by `residuals`
    /// and produces the flat tangent outputs that `output_structure` reshapes. Violations are reported as
    /// [`MalformedProgram`](ProgramError::MalformedProgram) errors: too few program inputs to hold the residuals,
    /// or a trailing residual input whose type differs from the type of the residual value that feeds it.
    pub fn new(
        context: C,
        program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        residuals: Vec<C::Value>,
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
        Ok(Self { context, program, residuals, output_structure, marker: PhantomData })
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

    /// Consumes this [`Pushforward`] and returns its open parts: the pushforward program `(ẋ, r) ↦ ẏ` and the
    /// linearization-point residuals `r` its trailing inputs consume, in that order.
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
    pub fn apply(&self, tangents: Input::To<C::Value>) -> Result<Output::To<C::Value>, ProgramError> {
        let mut inputs = tangents.into_parameters().collect::<Vec<_>>();
        inputs.extend(self.residuals.iter().cloned());
        let tangent_outputs = self.program.interpret_in_context(&self.context, inputs)?;
        Ok(Output::To::<C::Value>::from_parameters(self.output_structure.clone(), tangent_outputs)?)
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
/// Ryft also provides a `#[derive(DifferentiableOperation)]` procedural macro for operation enums whose variants own
/// forward-mode (JVP) rules through this trait. The derivation enables forward-mode differentiation only. Enums that
/// also need reverse-mode differentiation additionally derive [`TransposableOperation`](crate::TransposableOperation),
/// whose transposition dispatchers reverse mode is built on. It follows the same enum-shape inference rules as
/// `#[derive(Operation)]` and generates:
///
///   - An `impl DifferentiableOperation<C> for Enum` that is generic over a [`StagingContext`](crate::StagingContext)
///     `C` pinned to the enum's primary type, program constant type, and the enum itself as its operation family. Every
///     variant forwards [`jvp`](Self::jvp) to its payload's own rule, and so payloads without a forward-mode form must
///     still implement the trait with a rule that returns an
///     [`UnsupportedOperation`](ProgramError::UnsupportedOperation).
///   - A `where` clause following the same shape as the generated interpretation and partial-evaluation
///     implementations: a per-variant `Payload: DifferentiableOperation<C>` predicate for every *non-recursive* payload
///     which transports each rule's own capability requirements (e.g., `C::Value: Sin` for the sine rule) to the use
///     site, so that the enum does not spell them, plus a `Self: From<Payload>` conversion for every concrete payload
///     (the rules stage ordinary primal-enum operations for both the primal and the tangent side) and the `Self:
///     MaybeZeroOperation<T> + From<ZeroOperation<T>> + DifferentiableProgramOperation<C::Constant, Self> +
///     LinearizableProgramOperation<C::Constant, Self>` fixed-point witnesses that higher-order payload rules like
///     those for `condition`, `while`, and `scan` use to forward-differentiate and linearize their nested programs.
///     *Recursive* payloads (i.e., those mentioning `Self`) are skipped (such a predicate would re-enter the enum's
///     own obligation and overflow the trait solver) and their rules are discharged as definition-time body obligations
///     against the witnesses instead. The enum must therefore supply its own [`DifferentiableProgramOperation`] and
///     [`LinearizableProgramOperation`] implementations, spelling only the leaf capabilities that [`Program::jvp`]
///     and [`Program::linearize`] need.
pub trait DifferentiableOperation<C: Context<Operation: Clone>>: Operation<C::Type> {
    /// Applies this operation's capture-free forward-mode rule, mapping the input duals `(xᵢ, ẋᵢ)` to the output duals
    /// `(y, ẏ) = (f(x), Σᵢ (∂f/∂xᵢ)(x) · ẋᵢ)` where `f` is the function this operation computes. The returned vector
    /// must be aligned with this operation's outputs, each element pairing a primal output value with its tangent, both
    /// bound through `context`.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] through which the rule binds the primal and tangent [`Operation`]s it synthesizes.
    ///   - `inputs`: Input [`DifferentiationDual`]s aligned with this operation's inputs/operands.
    fn jvp(
        &self,
        context: &C,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, ProgramError>;
}

/// Represents closed [`Operation`] families whose captured [`Program`]s can be built into *fused* jvp programs on
/// behalf of an enclosing forward-mode rule. Higher-order forward-mode rules, such as the control-flow rules, must
/// forward-differentiate captured branch or body programs whose operation family is the same closed enum currently
/// being proven [`DifferentiableOperation`]. Writing that need directly as a recursive [`DifferentiableOperation`]
/// bound at every recursive payload boundary makes Rust's trait solver re-enter the same enum and overflow.
/// [`DifferentiableProgramOperation`] names that recursive fixed point once. The value type `V` and operation
/// family `O` stay fixed across the recursion, and a closed operation enum implements this trait directly, calling
/// [`Program::jvp`] in the body while spelling only the *leaf* closure of capabilities that body needs in the
/// implementation's `where` clause, rather than the recursive `Self: DifferentiableOperation<…>` bound itself. That
/// recursive obligation is then discharged once, as a definition-time body check, which is what lets a higher-order
/// rule require `Self: DifferentiableProgramOperation<V, Self>` without sending the trait solver into an unbounded
/// recursion. Higher-order payloads depend on this semantic witness instead of reproducing the full forward-mode
/// obligation.
///
/// [`LinearizableProgramOperation`] is the sibling witness for the *split* linearization form. The two are separated so
/// a rule requires only the shape it actually stages: the fused forward-mode `scan`, `condition`, etc. rules need only
/// this trait, while the bounded `while` rule (which must stack per-iteration residuals) needs the split one.
///
/// This trait is intentionally about complete operation families rather than individual primitive payloads,
/// and is implemented explicitly per operation enum rather than through a blanket implementation as a blanket
/// `impl DifferentiableProgramOperation for O where O: DifferentiableOperation` implementation would reintroduce
/// exactly the kind of recursion that this trait exists to break.
pub trait DifferentiableProgramOperation<V: Value, O: Clone + Operation<V::Type> + From<ZeroOperation<V::Type>>>:
    Operation<V::Type> + Sized
{
    /// Builds the *fused* JVP [`Program`] of `program`. Interpreting the provided program as a function `x ↦ y = f(x)`
    /// over its flattened inputs and outputs, the returned program computes `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ) = (y, ẏ)`
    /// over the flat boundary `[x₁, …, xₙ, ẋ₁, …, ẋₙ] ↦ [y₁, …, yₘ, ẏ₁, …, ẏₘ]`, without splitting it into primal and
    /// tangent halves. Refer to [`Program::jvp`] for more information. This is what the fused higher-order JVP rules
    /// (e.g., `scan`, `condition`, etc.) stage as their nested JVP bodies. Keeping the body fused defers the
    /// primal/tangent separation to the partial-evaluation known-ness split that [`Program::linearize`] performs,
    /// and so pure forward mode stages no residual stacks and pays the cost of only a single pass.
    fn jvp_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

/// Represents closed [`Operation`] families whose captured [`Program`]s can be _linearized_ on behalf of an enclosing
/// rule. This is the split-form sibling of [`DifferentiableProgramOperation`]: where that witness builds the fused JVP
/// program `(x, ẋ) ↦ (y, ẏ)`, this one additionally splits it through the partial-evaluation known-ness split into a
/// [`Linearization`] holding the primal (i.e., known) sub-program `x ↦ (y, r)`, where the residuals `r` are the
/// intermediate values the derivative is evaluated at, and the tangent (i.e., unknown) sub-program
/// `(ẋ, r) ↦ ẏ = (∂f/∂x)(x) · ẋ`, which is linear in `ẋ`. Refer to [`Program::linearize`] for more information.
///
/// It breaks the same recursive fixed point the same way as [`DifferentiableProgramOperation`]. A closed operation
/// enum implements it directly, calling [`Program::linearize`] in the body while spelling only the *leaf* closure of
/// capabilities that body needs, so that a higher-order rule can require `Self: LinearizableProgramOperation<V, Self>`
/// without the trait solver re-entering the enum's own [`DifferentiableOperation`] obligation. For example, the bounded
/// `while` rule uses it because a loop must stack per-iteration residuals for its tangent map to replay. The fused
/// forward-mode rules that keep their bodies un-split depend on [`DifferentiableProgramOperation`] instead.
///
/// Like [`DifferentiableProgramOperation`], it is implemented explicitly per operation enum rather than through a
/// blanket implementation, which would reintroduce the recursion it exists to break.
pub trait LinearizableProgramOperation<V: Value, O: Clone + Operation<V::Type> + From<ZeroOperation<V::Type>>>:
    Clone + Operation<V::Type> + Sized
{
    /// Linearizes `program`, splitting its fused jvp form `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ)` into the primal sub-program
    /// `x ↦ (y, r)` and the linear tangent sub-program `(ẋ, r) ↦ ẏ`. Refer to [`Program::linearize`] for more
    /// information.
    fn linearize_program(program: &Program<V, Self, Vec<V>, Vec<V>>) -> Result<Linearization<V, Self>, ProgramError>;
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

/// Value type flowing through the closures of the partial-evaluation-backed differentiation entry points (i.e.,
/// [`ForwardModeDifferentiate::linearize`], [`ReverseModeDifferentiate::vjp`], [`ReverseModeDifferentiate::gradient`],
/// and their derivatives). It is a [`DifferentiationTracer`] dual over a [`PartialEvaluationContext`] wrapping the
/// context `C` the transform runs in. Its primal half is a *known* partial-evaluation value carrying a concrete value
/// under an eager `C` (so that e.g., host control flow on primal values works as expected) and its tangent half is
/// *unknown*, accumulating the pushforward program.
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
/// zero tangents stay symbolic [`MaybeZero::Zero`]s while they flow between rules. The [`bind`](Context::bind) fast
/// path skips an operation's rule entirely when every input tangent is a structural zero, exactly like the
/// program-level replay behind [`Program::linearize`], and so no zero values are constructed and no zero work
/// is performed until a boundary [`materialize`](MaybeZero::materialize)s one through the inner context's [`Zero`]
/// capability.
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

impl<C: Context<Operation: Clone + DifferentiableOperation<C>> + Zero<C::Value>> Context for DifferentiationContext<C> {
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<DifferentiationTracer<C>, ProgramError> {
        // Constants are independent of every differentiation input and so their tangents are structural zeros.
        let dual = DifferentiationDual::new_with_zero_tangent(self.parent.lift(constant)?);
        Ok(DifferentiationTracer::new(dual, self.clone()))
    }

    fn bind<O: Into<C::Operation>>(
        &self,
        operation: O,
        inputs: &[DifferentiationTracer<C>],
    ) -> Result<Vec<DifferentiationTracer<C>>, ProgramError> {
        let operation = operation.into();

        // Unwrap the input tracers into context-free duals, run the rule against those, and rewrap the produced duals
        // with this context, mirroring how `BatchingContext::bind` unwraps to `ArrayBatch`es and rewraps.
        let input_duals = inputs.iter().map(|input| input.dual().clone()).collect::<Vec<_>>();

        // All-zero fast path mirroring `Program::jvp`. When an operation consumes at least one input and every input
        // tangent is a structural zero, the operation's tangent is zero by the chain rule, and so the rule is skipped
        // and the primal operation binds directly. Zero-input operations are excluded so their dedicated rules keep
        // handling primal synthesis and tangent typing.
        let output_duals = if !input_duals.is_empty() && input_duals.iter().all(|dual| dual.tangent().is_zero()) {
            let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
            self.parent
                .bind(operation, &primal_inputs)?
                .into_iter()
                .map(DifferentiationDual::new_with_zero_tangent)
                .collect()
        } else {
            operation.jvp(&self.parent, input_duals.as_slice())?
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

impl<
    V: Value,
    O: Clone + Operation<V::Type> + DifferentiableOperation<TracingContext<V, O>> + From<ZeroOperation<V::Type>>,
> Program<V, O, Vec<V>, Vec<V>>
{
    /// Builds the *fused* Jacobian-Vector Product (JVP) [`Program`] of this [`Program`]. Assume the input program
    /// represents a function `f` from its inputs to its outputs, `x ↦ y = f(x)`. This function returns the program that
    /// computes `f` together with its _pushforward_ (i.e., the forward-mode Jacobian-vector product): given an input
    /// tangent (i.e., perturbation direction) `ẋ`, the pushforward produces the output tangent `ẏ = (∂f/∂x)(x) · ẋ`,
    /// the directional derivative of `f` at `x` along `ẋ`. As a single map, the returned program computes
    /// `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ) = (y, ẏ)`. In terms of the program boundaries, if the input program has inputs
    /// `[x_1, …, x_n]` and outputs `[y_1, …, y_m]` (so that `y = f(x)`), the returned program has:
    ///
    ///   - inputs `[x_1, …, x_n, ẋ_1, …, ẋ_n]`m which correspond to the `n` primal inputs followed by one fresh tangent
    ///     input `ẋ_i` per primal input `x_i`, of the same type, and
    ///   - outputs `[y_1, …, y_m, ẏ_1, …, ẏ_m]`, which correspond to the `m` primal outputs `y_j = f_j(x)` followed by
    ///     the `m` tangent outputs `ẏ = (∂f/∂x)(x) · ẋ`.
    ///
    /// The program is *not* split into separate primal and tangent sub-programs unlike [`Self::linearize`], whose
    /// partial-evaluation known-ness split consumes this fused program as its front half. This un-split form is exposed
    /// for fused higher-order JVP rules and direct forward-mode interpretation.
    ///
    /// Each primal instruction is replayed once through its [`DifferentiableOperation`] rule, which returns the dual
    /// (i.e., primal result plus tangent) for the instruction's outputs. Both are staged into the shared builder as
    /// ordinary primal operations, and so the result contains no symbolic captures.
    ///
    /// Atoms that are not reached by any input tangent are structurally zero. Their tangents stay symbolic as typed
    /// [`MaybeZero::Zero`]s and stage nothing. The shared all-zero fast path short-circuits the all-zero case (an
    /// operation consuming at least one input whose every input tangent is a structural zero) by staging the primal
    /// operation directly and pairing each primal output with a typed structural zero tangent, so that zero-ness
    /// propagates transitively without staging or scanning [`Instruction`]s. Structural zero tangents are materialized
    /// as typed [`ZeroOperation`] instructions only at the output boundary, preserving the
    /// `(primal_outputs ++ tangent_outputs)` program contract.
    pub fn jvp(&self) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
        let primal_input_count = self.input_ids().len();

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
            // atom's primal type.
            let mut primals: Vec<Option<Tracer<TracingContext<V, O>>>> = vec![None; self.atoms().len()];
            let mut tangents: Vec<Option<MaybeZero<Tracer<TracingContext<V, O>>>>> = vec![None; self.atoms().len()];

            // Primal inputs become the leading inputs. One fresh tangent input is added per primal input afterward
            // so that the input order is `(primals ++ tangents)`.
            for input_id in self.input_ids().iter().copied() {
                let r#type = self.atoms()[input_id.index()].r#type().into_owned();
                primals[input_id.index()] = Some(context.input(r#type));
            }
            for input_id in self.input_ids().iter().copied() {
                let r#type = self.atoms()[input_id.index()].r#type().into_owned();
                tangents[input_id.index()] = Some(MaybeZero::Value(context.input(r#type)));
            }

            // Constants are lifted into the builder as primal constants. Their tangents are derived lazily as
            // structural zeros typed with the atom's primal type. The call is disambiguated to the staging method
            // because the `Constant` capability trait also provides a `constant` method.
            for (atom_index, atom) in self.atoms().iter().enumerate() {
                if let Atom::Constant(value) = atom {
                    primals[atom_index] = Some(StagingContext::constant(&context, value.clone()));
                }
            }

            // Replay each primal instruction in JVP form, staging both the primal result and the tangent operations
            // into the shared builder.
            for instruction in self.instructions() {
                let input_duals = instruction
                    .inputs()
                    .iter()
                    .copied()
                    .map(|input_atom| {
                        let primal = primals[input_atom.index()]
                            .clone()
                            .ok_or(ProgramError::UnboundAtomId { id: input_atom })?;
                        // Atoms not connected to an input tangent (i.e., constants and dead inputs) take a structural
                        // zero typed with the atom's primal type.
                        let tangent = match &tangents[input_atom.index()] {
                            Some(tangent) => tangent.clone(),
                            None => MaybeZero::Zero(primal.r#type().into_owned()),
                        };
                        Ok(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new(primal, tangent))
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;

                // All-zero fast path: when an operation consumes at least one input and every input tangent is a
                // structural zero, the operation's tangent is zero by the chain rule, so the rule is skipped. The
                // primal outputs are staged directly and each output tangent is a typed structural zero. Zero-input
                // operations are excluded so their dedicated rules keep handling primal synthesis and tangent typing.
                let all_input_tangents_are_zero = input_duals.iter().all(|dual| dual.tangent().is_zero());
                let output_duals = if !input_duals.is_empty() && all_input_tangents_are_zero {
                    let primal_inputs = input_duals.iter().map(|dual| dual.primal().clone()).collect::<Vec<_>>();
                    context
                        .stage_operation(instruction.operation().clone(), primal_inputs.as_slice())?
                        .into_iter()
                        .map(DifferentiationDual::<Tracer<TracingContext<V, O>>>::new_with_zero_tangent)
                        .collect()
                } else {
                    instruction.operation().jvp(&context, input_duals.as_slice())?
                };

                check_count!("output", output_duals, instruction.outputs().len(), ProgramError);
                for (output_atom, dual) in instruction.outputs().iter().copied().zip(output_duals) {
                    let (primal, tangent) = dual.into_parts();
                    primals[output_atom.index()] = Some(primal);
                    tangents[output_atom.index()] = Some(tangent);
                }
            }

            // Collect the outputs: the primal outputs followed by the tangent outputs, in the original output order.
            // Structural zero tangents are materialized as typed `ZeroOperation` instructions here (the output boundary
            // is the only place the fused program requires a real atom for them).
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
                    // Atoms not connected to an input tangent (i.e., constants and dead inputs) take a structural zero
                    // typed with the atom's primal type.
                    let tangent = match &tangents[output_atom.index()] {
                        Some(tangent) => tangent.clone(),
                        None => MaybeZero::Zero(
                            primals[output_atom.index()]
                                .as_ref()
                                .ok_or(ProgramError::UnboundAtomId { id: output_atom })?
                                .r#type()
                                .into_owned(),
                        ),
                    };
                    tangent.materialize(&context)?.atom_id()
                })
                .collect::<Result<Vec<_>, _>>()?;

            let mut output_atoms = primal_output_atoms;
            output_atoms.extend(tangent_output_atoms);
            output_atoms
        };

        // All tracing handles are dropped here, so the builder can be recovered and finalized.
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let input_count = 2 * primal_input_count;
        let output_count = output_atoms.len();
        builder.build::<Vec<V>, Vec<V>>(output_atoms, vec![Placeholder; input_count], vec![Placeholder; output_count])
    }
}

impl<
    V: Value,
    O: Clone
        + Operation<V::Type>
        + PartiallyEvaluatableOperation<TracingContext<V, O>>
        + DifferentiableOperation<TracingContext<V, O>>
        + From<ZeroOperation<V::Type>>,
> Program<V, O, Vec<V>, Vec<V>>
{
    /// Linearizes this [`Program`] by fusing the forward-mode differentiation rules into one Jacobian-Vector Product
    /// (JVP) program and splitting it into its primal (i.e., known) and tangent (i.e., unknown) halves through the
    /// partial-evaluation known-ness split. This is the domain-free, interpretation-free generic core of the
    /// linearization transform, shared by every concrete entry point. It builds the fused JVP program, replaying each
    /// primal instruction once in JVP form so that the program stages both the primal computation and its pushforward
    /// over the primal operation family, and then partitions that program through
    /// [`Program::partition`](crate::Program::partition) with the leading primal inputs marked known and the trailing
    /// tangent inputs marked unknown. The split's fresh known-side staging trace becomes the primal program, and so
    /// *linearity separation is known-ness separation*: the per-operation partial-evaluation rules own the split,
    /// higher-order operations (e.g., `scan`, `condition`, etc.) separate through their known-ness splits instead of
    /// needing linearize-specific handling, and effectful primal work lands in the primal program per the effect
    /// placement contract of [`PartialEvaluationContext::fold_or_residualize`]. The known side computes the primal
    /// outputs followed by the residual edges and the residual side is the linear tangent map taking
    /// `(tangents ++ residuals)`. The tangent program's canonical input order is then rebuilt from the split's recorded
    /// per-input sources rather than assumed from the walk's input layout, so the tangent program always presents its
    /// full leading tangent inputs ahead of the residuals. No value semantics are applied: the returned
    /// [`Linearization`] carries only the two split sub-programs and the metadata needed to reassemble and transpose
    /// them, leaving interpretation of the primal side to callers.
    ///
    /// Note that linearization splits with the known-ness partial-evaluation rules rather than a value-free structural
    /// split. That is because [`Instruction`]-granular structural classification cannot separate a fused higher-order
    /// operation (e.g., a fused JVP `scan` mixes primal and tangent carries inside one instruction), while the
    /// known-ness rules split inside it.
    pub fn linearize(&self) -> Result<Linearization<V, O>, ProgramError> {
        let primal_input_count = self.input_ids().len();
        let primal_output_count = self.output_ids().len();

        // Build the fused jvp program over `[primals..., tangents...] -> [primal_outputs..., tangent_outputs...]`.
        let fused_jvp_program = self.jvp()?;

        // Split the fused program with the leading `primal_input_count` primal inputs known and the trailing tangent
        // inputs unknown. The split walks the fused program through the per-operation partial-evaluation rules against
        // a fresh known-side staging trace: known (i.e., primal) work folds by staging into that trace, and the
        // residual program that survives is the linear tangent map.
        let input_known = [vec![true; primal_input_count], vec![false; primal_input_count]].concat();
        let partitioned_jvp_program = fused_jvp_program.partition(input_known.as_slice())?;
        let residual_count = partitioned_jvp_program.residual_inputs().iter().filter(|input| input.is_known()).count();
        let known_output_indices = partitioned_jvp_program
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_known().then_some(index))
            .collect::<Vec<_>>();
        let residual_output_indices = partitioned_jvp_program
            .outputs()
            .iter()
            .enumerate()
            .filter_map(|(index, output)| output.is_unknown().then_some(index))
            .collect::<Vec<_>>();
        let (mut known_program, residual_program, _, residual_inputs, _) = partitioned_jvp_program.into_parts();

        // The known program's outputs are the fully known fused outputs followed by the residual edges. Every primal
        // output must be known (i.e., the primals are all known, and effectful primal work folds into the known trace).
        // Any *further* known outputs are structurally zero tangent outputs (e.g., the Boolean mask item of a batched
        // masked `while`, whose all-zero JVP fast path stages a fresh zero rather than threading the input tangent),
        // which belong to the tangent half and are restored there below.
        if known_output_indices.len() < primal_output_count
            || known_output_indices[..primal_output_count]
                .iter()
                .zip(0..primal_output_count)
                .any(|(&index, expected)| index != expected)
        {
            return Err(ProgramError::MalformedProgram(
                "a primal output did not fold to the known side during linearization".into(),
            ));
        }

        // Drop the stray tangent zeros from the known program's outputs so the primal program presents
        // `[primal_outputs..., residuals...]`. They occupy exactly the window between the primal outputs
        // and the residual edges.
        if known_output_indices.len() > primal_output_count {
            known_program.output_ids.drain(primal_output_count..known_output_indices.len());
            known_program.output_structure = vec![Placeholder; known_program.output_ids.len()];
        }

        // Restore the residual (i.e., tangent) program's canonical input order `[tangents..., residuals...]` from the
        // split's recorded per-input sources. Each tangent input's atom lands at its original tangent position, a
        // tangent position missing from the sources is restored as a fresh dead atom of its fused type, and each
        // residual edge lands after the tangents at its edge ordinal. Today's walk seeds every unknown input up front
        // in original order, appends residual edges in first-use order, and never prunes residual-program inputs, so
        // this rebuild is an identity and no tangent position is ever missing; it stays source-driven anyway because
        // that layout is an implementation detail of the walk rather than part of the partial-evaluation contract, and
        // a walk that materialized unknown inputs lazily or pruned dead ones (i.e., a structurally zero tangent whose
        // input reaches no tangent output) would invalidate a layout-based rebuild but not this one. The restored atoms
        // are fresh program inputs that no instruction references, so the direct program-field extensions preserve
        // every `Program` invariant a `ProgramBuilder` would have established.
        let mut tangent_program = residual_program;
        let surviving_input_ids = tangent_program.input_ids.split_off(0);
        let mut tangent_inputs: Vec<Option<AtomId>> = vec![None; primal_input_count];
        let mut edge_inputs: Vec<Option<AtomId>> = vec![None; residual_count];
        for (source, atom) in residual_inputs.iter().zip(surviving_input_ids) {
            match source {
                PartialEvaluationInput::Unknown(index) => {
                    let position = index.checked_sub(primal_input_count).ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "a known primal input survived as a residual-program input during linearization".into(),
                        )
                    })?;
                    tangent_inputs[position] = Some(atom);
                }
                PartialEvaluationInput::Known(ordinal) => edge_inputs[*ordinal] = Some(atom),
            }
        }
        for (position, atom) in tangent_inputs.into_iter().enumerate() {
            let restored = match atom {
                Some(atom) => atom,
                None => {
                    // The split recorded no source for this tangent position, so restore it as a fresh dead program
                    // input (i.e., referenced by no instruction) typed from the corresponding fused-program tangent
                    // input. The fused program's inputs are laid out as `[primals..., tangents...]`, and so the tangent
                    // for `position` lives at index `primal_input_count + position`.
                    let fused_input_index = primal_input_count + position;
                    let fused_input_id = fused_jvp_program.input_ids[fused_input_index].index();
                    let Atom::Variable(tangent_type) = &fused_jvp_program.atoms[fused_input_id] else {
                        return Err(ProgramError::MalformedProgram(format!(
                            "tangent input {fused_input_index} is not a variable",
                        )));
                    };
                    let restored = AtomId::new(tangent_program.atoms.len());
                    tangent_program.atoms.push(Atom::Variable(tangent_type.clone()));
                    restored
                }
            };
            tangent_program.input_ids.push(restored);
        }
        for atom in edge_inputs.into_iter() {
            tangent_program.input_ids.push(atom.ok_or_else(|| {
                ProgramError::MalformedProgram("a linearization residual edge has no residual-program input".into())
            })?);
        }
        tangent_program.input_structure = vec![Placeholder; tangent_program.input_ids.len()];

        // Restore the canonical tangent outputs. The residual program's outputs are the unknown fused outputs in
        // original order (all within the tangent half, since every primal output is known), and each structurally
        // zero tangent output that folded to the known side is restored as a fresh staged zero of its fused type.
        let surviving_outputs = tangent_program.output_ids.split_off(0);
        let mut survivors = residual_output_indices.into_iter().zip(surviving_outputs).peekable();
        for output in 0..primal_output_count {
            let fused_output_index = primal_output_count + output;
            match survivors.peek() {
                Some(&(index, atom)) if index == fused_output_index => {
                    survivors.next();
                    tangent_program.output_ids.push(atom);
                }
                _ => {
                    let zero_atom = fused_jvp_program.output_ids[fused_output_index];
                    let zero_type = fused_jvp_program.atoms[zero_atom.index()].r#type().into_owned();
                    let zero_output = AtomId::new(tangent_program.atoms.len());
                    tangent_program.atoms.push(Atom::Variable(zero_type.clone()));
                    tangent_program.instructions.push(Instruction::new(
                        O::from(ZeroOperation::new(zero_type)),
                        Vec::new(),
                        vec![zero_output],
                    ));
                    tangent_program.output_ids.push(zero_output);
                }
            }
        }
        tangent_program.output_structure = vec![Placeholder; tangent_program.output_ids.len()];

        Linearization::new(known_program, tangent_program, residual_count)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::operations::Operation;
    use crate::operations::arithmetic::MulOperation;
    use crate::operations::differentiation::StopGradientOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::ProgramBuilder;
    use crate::scalars::Scalar;
    use crate::types::DataType;

    #[test]
    fn test_program_jvp() {
        // Test that the fused JVP program of `f(x) = sin(x)` presents the `[x, ẋ] ↦ [sin(x), cos(x) · ẋ]` boundary.
        // The primal input leads, one fresh tangent input follows, and the outputs are the primal output followed by
        // its tangent.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
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
        let scaled = builder.add_instruction(MulOperation, vec![input, constant]).unwrap()[0];
        let severed = builder.add_instruction(StopGradientOperation, vec![scaled]).unwrap()[0];
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
    fn test_program_linearize() {
        // Test that linearizing `f(x) = sin(x)` splits its fused JVP program by known-ness into the primal sub-program
        // `x ↦ (sin(x), cos(x))`, whose trailing output is the `cos(x)` residual, and the linear tangent sub-program
        // `(ẋ, r) ↦ r · ẋ`.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(SinOperation, vec![input]).unwrap()[0];
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

        // Test that a structurally zero tangent output (here, the tangent of a constant-valued program output) folds
        // to the known side during the split and is restored in the tangent sub-program as a staged `zero` instruction,
        // so the tangent sub-program keeps one tangent output per primal output.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::F64(2.0));
        let scaled = builder.add_instruction(MulOperation, vec![input, constant]).unwrap()[0];
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
    }
}
