use crate::contexts::Context;
use crate::operations::Operation;
use crate::operations::constants::ZeroOperation;
use crate::programs::{MaybeZero, Program, ProgramError, Value};
use crate::tracing_v2::differentiation::Linearization;
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
    /// tangent halves. Refer to [`Program::jvp`] for the full contract. This is what the fused higher-order JVP rules
    /// (e.g., `scan`, `condition`, etc.) stage as their nested JVP bodies. Keeping the body fused defers the
    /// primal/tangent separation to the partial-evaluation known-ness split that [`Program::linearize`] performs,
    /// and so pure forward mode stages no residual stacks and pays the cost of only a single pass.
    fn jvp_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

// TODO(eaplatanios): Review from here onwards.

/// Represents closed [`Operation`] families whose captured flat programs can be linearized capture-free on behalf of
/// an enclosing rule.
///
/// This is the split-form sibling of [`DifferentiableProgramOperation`]: where that witness builds the fused jvp
/// program `(x, ẋ) ↦ (y, ẏ)`, this one additionally splits it through the partial-evaluation known-ness split into a
/// [`Linearization`] holding the primal (known) sub-program `x ↦ (y, r)` — where the residuals `r` are the
/// intermediate values the derivative is evaluated at — and the tangent (unknown) sub-program
/// `(ẋ, r) ↦ ẏ = (∂f/∂x)(x) · ẋ`, which is linear in `ẋ`. Refer to
/// [`Program::linearize`] for the full contract.
///
/// It breaks the same recursive fixed point the same way as [`DifferentiableProgramOperation`]: a closed operation
/// enum implements it directly, calling [`Program::linearize`] in the body while spelling
/// only the *leaf* closure of capabilities that body needs, so a higher-order rule can require
/// `Self: LinearizableProgramOperation<V, Self>` without the trait solver re-entering the enum's own
/// [`DifferentiableOperation`] obligation. The bounded `while` rule uses it because a loop must stack per-iteration
/// residuals for its tangent map to replay; the fused forward-mode rules that keep their bodies un-split depend on
/// [`DifferentiableProgramOperation`] instead.
///
/// Like [`DifferentiableProgramOperation`], it is implemented explicitly per operation enum rather than through a
/// blanket impl, which would reintroduce the recursion it exists to break. The value type `V` (whose carried type
/// descriptor types the programs) and operation family `O` match the primal program being linearized.
pub trait LinearizableProgramOperation<V: Value, O: Clone + Operation<V::Type> + From<ZeroOperation<V::Type>>>:
    Clone + Operation<V::Type> + Sized
{
    /// Linearizes `program` capture-free, splitting its fused jvp form `(x, ẋ) ↦ (f(x), (∂f/∂x)(x) · ẋ)` into the
    /// primal sub-program `x ↦ (y, r)` and the linear tangent sub-program `(ẋ, r) ↦ ẏ`; refer to
    /// [`Program::linearize`] for the returned packaging.
    ///
    /// # Parameters
    ///
    ///   - `program`: Already-traced flat sub-program over this operation family, with [`Vec`]-parameterized inputs
    ///     and outputs.
    fn linearize_program(program: &Program<V, Self, Vec<V>, Vec<V>>) -> Result<Linearization<V, Self>, ProgramError>;
}
