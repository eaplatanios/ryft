use std::fmt::{Debug, Display};

use ryft_macros::Parameter;

use crate::contexts::{Context, Domain};
use crate::differentiation::{DifferentiableType, TransposableOperation};
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{Zero, ZeroOperation};
use crate::parameters::Parameter;
use crate::partial::PartialEvaluationContext;
use crate::programs::{MaybeZero, Program, ProgramError, Value};
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
    /// Creates a new [`Linearization`] from its parts. The caller must uphold the boundary contract documented on
    /// [`Linearization`] where `primal` produces its outputs followed by its trailing `residual_count` residuals,
    /// and `tangent` consumes its tangent inputs followed by those same residuals. [`Program::linearize`] is the
    /// constructing entry point that establishes this contract.
    #[inline]
    pub fn new(
        primal: Program<V, O, Vec<V>, Vec<V>>,
        tangent: Program<V, O, Vec<V>, Vec<V>>,
        residual_count: usize,
    ) -> Self {
        Self { primal, tangent, residual_count }
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
        let tangent_input_count = self.tangent.input_ids().len().checked_sub(self.residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "tangent program has {} inputs which is fewer than its {} residuals",
                self.tangent.input_ids().len(),
                self.residual_count,
            ))
        })?;

        // Transpose with respect to the leading tangent inputs, holding the trailing residual inputs as known
        // parameters. Partial transposition exposes each known residual as a pullback input, so the residuals are
        // not folded into captured factors here.
        let with_respect_to = (0..tangent_input_count).collect::<Vec<_>>();
        self.tangent.transpose_with_respect_to(with_respect_to.as_slice())
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
/// [`Differentiate::linearize`], [`Differentiate::vjp`], [`Differentiate::gradient`], and their derivatives). It is a
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
/// wrapped context. It is what makes [`Differentiate::jvp`] the single forward-mode entry point. Structural zero
/// tangents stay symbolic [`MaybeZero::Zero`]s while they flow between rules. The [`bind`](Context::bind) fast path
/// skips an operation's rule entirely when every input tangent is a structural zero, exactly like the program-level
/// replay behind [`Program::linearize`], and so no zero values are constructed and no zero work is performed until a
/// boundary [`materialize`](MaybeZero::materialize)s one through the inner context's [`Zero`] capability.
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
