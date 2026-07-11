use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::contexts::{Context, Domain, StagingContext};
use crate::differentiation::{
    DifferentiableOperation, DifferentiableType, DifferentiationError, ForwardModeDifferentiate, LinearizationTracer,
};
use crate::errors::MaybeFallible;
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::{OneOperation, Zero, ZeroOperation};
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{Atom, AtomId, MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{Type, Typed};

/// Pullback of a function `f` at a linearization point `x` (i.e., the transposed linear map `ȳ ↦ x̄ = (∂f/∂x)(x)ᵀ · ȳ`),
/// packaged as a reusable callable. This is the reverse-mode dual of [`Pushforward`](crate::Pushforward), whose
/// callable applies the un-transposed map `ẋ ↦ (∂f/∂x)(x) · ẋ` instead. It wraps the pullback [`Program`] `(ȳ, r) ↦ x̄`
/// obtained by transposing the pushforward program of the differentiated closure, closed over the residuals `r`
/// recovered at the linearization point. [`apply`](Self::apply) computes `x̄ = (∂f/∂x)(x)ᵀ · ȳ` by appending the
/// residuals to the flattened output cotangents `ȳ`, interpreting the pullback program, and reshaping the flat input
/// cotangents against the closure's input structure. It thus pulls any number of cotangents back through the function's
/// transposed Jacobian without re-tracing or re-differentiating (e.g., replaying every coordinate basis cotangent to
/// build a Jacobian row by row), amortizing the cost of differentiating once over many cotangent applications.
///
/// The context `C` supplies the value semantics and operation family, `Input` is the closure's structured input type,
/// whose [`ParameterStructure`](Parameterized::ParameterStructure) is retained so that the flat input cotangents
/// reshape back into `Input::To<C::Value>`, and `Output` is its structured output type. `Output` is carried as a type
/// parameter so that [`apply`](Self::apply) infers the cotangent family from the pullback itself rather than requiring
/// a turbofish.
pub struct Pullback<C: Context, Input: Parameterized<C::Value>, Output> {
    /// [`Context`] that the pullback was built in. [`apply`](Self::apply) replays the pullback program in it,
    /// mirroring how [`Pushforward`](crate::Pushforward) replays its pushforward program.
    context: C,

    /// Pullback [`Program`] over the primal operation family in the context's staged
    /// [`Constant`](crate::Domain::Constant) space, mapping `[output_cotangents ++ residuals]` to the flat
    /// input cotangents. Its literal constants are lifted through the context's [`lift`](Context::lift) when
    /// [`apply`](Self::apply) replays it.
    program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Linearization-point residuals consumed by [`program`](Self::program), appended after the output cotangents
    /// when interpreting it.
    residuals: Vec<C::Value>,

    /// Parameter structure of the closure's input, used to reshape the flat input cotangents.
    input_structure: Input::ParameterStructure,

    /// Encodes the closure's output family `Output` so that [`apply`](Self::apply) can flatten the cotangents without
    /// a turbofish. No `Output::ParameterStructure` is stored alongside it because [`apply`](Self::apply) only
    /// _flattens_ its structured cotangent argument, which needs no stored structure, and rebuilds structure only on
    /// the input-cotangent side through `input_structure`. [`Pushforward`](crate::Pushforward) mirrors this with a
    /// stored output structure and a phantom `Input`.
    marker: PhantomData<fn() -> Output>,
}

impl<
    C: Context<Operation: Clone>,
    Input: Parameterized<C::Value, Family: ParameterizedFamily<C::Value>>,
    Output: Parameterized<C::Value>,
> Pullback<C, Input, Output>
{
    /// Creates a new [`Pullback`] closing `program` over the linearization-point `residuals`, validating the contract
    /// documented on [`Pullback`] where `program` consumes the flat output cotangents followed by `residuals` and
    /// produces the flat input cotangents that `input_structure` reshapes. Violations are reported as
    /// [`MalformedProgram`](ProgramError::MalformedProgram) errors: too few program inputs to hold the residuals,
    /// or a trailing residual input whose type differs from the type of the residual value that feeds it.
    pub fn new(
        context: C,
        program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        residuals: Vec<C::Value>,
        input_structure: Input::ParameterStructure,
    ) -> Result<Self, ProgramError> {
        let cotangent_input_count = program.input_ids().len().checked_sub(residuals.len()).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "pullback program consumes {} inputs which is fewer than its {} residuals",
                program.input_ids().len(),
                residuals.len(),
            ))
        })?;
        for (index, (input, residual)) in program.inputs().skip(cotangent_input_count).zip(&residuals).enumerate() {
            if input.r#type().as_ref() != residual.r#type().as_ref() {
                return Err(ProgramError::MalformedProgram(format!(
                    "pullback residual {index} has type {} in the pullback program but carries a value of type {}",
                    input.r#type(),
                    residual.r#type(),
                )));
            }
        }
        Ok(Self { context, program, residuals, input_structure, marker: PhantomData })
    }

    /// Returns the pullback [`Program`] `(ȳ, r) ↦ x̄` that this callable closes over. Its inputs are the flat output
    /// cotangents followed by the residuals carried by [`residuals`](Self::residuals).
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

    /// Consumes this [`Pullback`] and returns its open parts: the pullback program `(ȳ, r) ↦ x̄` and the
    /// linearization-point residuals `r` its trailing inputs consume, in that order.
    #[inline]
    pub fn into_parts(self) -> (Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<C::Value>) {
        (self.program, self.residuals)
    }

    /// Pulls the provided structured output cotangents `cotangents` back to the closure's input cotangents. The
    /// cotangents are flattened, the linearization-point residuals are appended, the pullback program is interpreted
    /// at that vector in the context that this pullback was built in (i.e., the single replay path for both context
    /// flavors: an eager context interprets the pullback immediately, while a staging context stages it into the
    /// enclosing trace and returns tracers), and the flat input cotangents are reshaped against the closure's input
    /// structure.
    #[inline]
    pub fn apply(&self, cotangents: Output::To<C::Value>) -> Result<Input::To<C::Value>, ProgramError> {
        let mut inputs = cotangents.into_parameters().collect::<Vec<_>>();
        inputs.extend(self.residuals.iter().cloned());
        let input_cotangents = self.program.interpret_in_context(&self.context, inputs)?;
        Ok(Input::To::<C::Value>::from_parameters(self.input_structure.clone(), input_cotangents)?)
    }
}

/// Represents [`Operation`]s that provide a transposition rule for linear [`Program`]s. Reading a linear
/// [`Instruction`](crate::Instruction) as a linear map `y = L(x)` (in differentiation, `L` is a piece of the tangent
/// map `(∂f/∂x)(x)` produced by linearization), the [`transpose`](Self::transpose) function computes the action of
/// the *transposed* (i.e., adjoint) map: given a cotangent `ȳ` for the output, it returns the cotangent contribution
/// `x̄ = Lᵀ(ȳ)` for each input, where `Lᵀ` is the unique linear map satisfying `⟨ȳ, L(x)⟩ = ⟨Lᵀ(ȳ), x⟩`. Applied
/// instruction by instruction in reverse program order, these rules compute the vector-Jacobian product
/// `x̄ = (∂f/∂x)(x)ᵀ · ȳ` that reverse-mode differentiation is built on. Cotangents flow symbolically as [`MaybeZero`]s:
/// rules may reuse existing cotangents, return [`MaybeZero::Zero`] for structural zeros, or stage additional linear
/// operations in the active [`TracingContext`]. The rule does not receive concrete primal values. Instead, it receives
/// each input's/operand's [`PartialValue`] knowledge (i.e., its [`Type`] when the operand is linear, or the staged
/// [`Tracer`] carrying its runtime value when the operand is a known factor) and any further metadata must be encoded
/// in the operation itself.
///
/// Refer to the documentation of [`Program::transpose`] for more information on what _transposition_ means here and
/// how it relates to the algebraic notion of transposition.
///
/// # Deriving Transposable Operation Enums
///
/// Ryft provides a `#[derive(TransposableOperation)]` procedural macro via the `ryft-macros` crate for
/// [`TransposableOperation`] sum types whose variants already own their transpose rules. Deriving it is what enables
/// *reverse-mode* differentiation for programs staged in the operation family. `#[derive(DifferentiableOperation)]`
/// adds forward-mode (JVP) support only, and the transposition dispatchers generated here are what
/// [`Program::transpose`] and the reverse-mode entry points build on, so enums that support both modes derive
/// both. The derivation is intentionally only a dispatcher: it matches on the enum variant and forwards
/// [`transpose`](Self::transpose) to the wrapped payload. Operation-specific transpose semantics still live on the
/// concrete payload types. The derived implementation follows the same enum-shape rules as `#[derive(Operation)]`:
///
///   - The derivation macro input must be an enum.
///   - Every variant must be a tuple variant with exactly one payload field.
///   - A payload may be stored directly as `Payload` or indirectly as `Box<Payload>`.
///   - The enum must implement [`Operation`]. In practice, transposable operation enums usually derive both
///     [`Operation`] and [`TransposableOperation`].
///
/// The generated implementation is:
///
///   - `impl TransposableOperation<V, Enum> for Enum`, where the transposition value type `V` carries the primary
///     type `T` selected using the same rules that are used for inferring `T` in the `#[derive(Operation)]` macro.
///   - For enums with one `Value<Type = T>` parameter and no recursive payloads (i.e., payloads that mention `Self`),
///     that parameter is treated as the operation family's stored constant type and the generated implementation is
///     generic over a separate transposition value type `V`. For enums with two or more `Value<Type = T>` parameters,
///     the first value parameter is treated as the tangent/cotangent value type and later value parameters are
///     constants or captured factors. Enums with recursive payloads likewise pin the transposition value to the
///     program constant type (i.e., no separate `V` is introduced). Recursive higher-order payload rules name the
///     operation-family fixed point through [`TransposableProgramOperation`] and pin their transposition value to the
///     program constant type, so the generated witness's [`Program::transpose_with_respect_to`] call is only provable
///     at that instantiation.
///   - Concrete payload variants forward directly to their payload implementations and receive a generated
///     `Payload: TransposableOperation<V, Enum>` `where` predicate. Payload-specific capability requirements should
///     live on the payload's own [`TransposableOperation`] implementation; the enum derivation carries them through
///     this generated payload bound.
///   - `impl TransposableProgramOperation<V> for Enum`, using the same value-type inference and all non-recursive
///     payload transposition bounds as the dispatcher implementation. The generated implementation is the standard
///     operation-family witness for nested linear programs: it calls [`Program::transpose_with_respect_to`] on the
///     provided program for [`transpose_program`](TransposableProgramOperation::transpose_program) (fully linear
///     callers pass an all-`true` linearity mask). If a higher-order payload contains the enum type being derived
///     (e.g., a branch or loop body whose operation family is `Self`), the macro does not restate that payload's direct
///     transposition bound on this witness. That recursive payload should instead depend on
///     [`TransposableProgramOperation`], which names the fixed point and keeps the trait solver finite.
///     The implementation is additionally constrained by the [`Zero`](crate::Zero)/[`Add`](std::ops::Add) capabilities
///     that [`Program::transpose`] requires.
///   - Bare generic payload variants such as `Extension(Extension)` receive the same generated
///     `Extension: TransposableOperation<V, Enum>` bound, because the macro cannot know which concrete extension
///     type will be substituted by the caller.
///
/// Recursive higher-order payloads that need to transpose captured linear programs should depend on
/// [`TransposableProgramOperation`] instead of restating a direct `Enum: TransposableOperation<V, Enum>` bound.
/// When those recursive payload rules need value capabilities, express those requirements on the enum's generic
/// parameters or on the payload implementations themselves, so the generated dispatcher and program-transposition
/// witness inherit them through normal Rust bounds.
///
/// The derivation macro also supports the same `#[ryft(crate = "...")]` attribute as the `#[derive(Operation)]` macro.
/// The default path is `ryft`, so downstream crates that depend on the `ryft` crate normally do not need this
/// attribute.
///
/// ## Example
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::{ArrayType, ConstantOperation, Value, ZeroOperation};
/// # use ryft_macros::{Operation, TransposableOperation};
///
/// #[derive(Clone, Debug, Operation, TransposableOperation)]
/// enum LinearOperation<V: Value<Type = ArrayType>> {
///     Zero(ZeroOperation<ArrayType>),
///     Constant(ConstantOperation<V>),
/// }
/// ```
pub trait TransposableOperation<V: Value, O: Operation<V::Type>>: Operation<V::Type> {
    /// Applies this operation's transpose rule to the provided symbolic output cotangents, computing `x̄ = Lᵀ(ȳ)` for
    /// the linear map `y = L(x)` this operation stages. The returned vector must contain one entry per operation input.
    /// Each [`MaybeZero::Value`] entry is a staged cotangent contribution in the active [`TracingContext`], and each
    /// [`MaybeZero::Zero`] means that the corresponding input receives a structural zero of the carried cotangent
    /// [`Type`] from this operation.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`TracingContext`] in which rules may stage additional linear operations.
    ///   - `inputs`: Per-input [`PartialValue`] knowledge, in operation-input order. A [`PartialValue::Unknown`]
    ///     entry marks an input that is linear in the transposed program and therefore receives a cotangent
    ///     contribution of that type. The type also recovers cotangent shapes that are not derivable from the operation
    ///     payload alone (e.g., a broadcast operation's pre-broadcast shape). A [`Known`](PartialValue::Known) entry
    ///     marks an input whose runtime value rides in the pullback function as a tracer: bilinear rules such as the
    ///     one for `Mul` read it directly to scale the output cotangent into the linear input's contribution, and the
    ///     rule emits a [`MaybeZero::Zero`] for that input. Each input's [`Type`] is available either way through the
    ///     [`Typed`] trait. Rules that transpose fully linear operations need only read the types, since every input
    ///     is then [`PartialValue::Unknown`].
    ///   - `outputs`: Symbolic cotangents for the instruction's outputs, in operation-output order.
    fn transpose(
        &self,
        context: &mut TracingContext<V, O>,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError>;
}

/// Represents closed [`Operation`] families whose linear [`Program`]s can be transposed as nested programs.
/// Higher-order transpose rules, such as the rules for linear condition branches and linear scan bodies, need
/// to transpose captured programs whose operation family is the same closed enum that is currently being proven
/// transposable. Writing that need directly as `O: TransposableOperation<V, O>` at every recursive payload boundary
/// can send Rust's trait solver through the enum's higher-order variants indefinitely. [`TransposableProgramOperation`]
/// names the recursive fixed point once. The closed operation enum implements this trait by calling
/// [`Program::transpose_with_respect_to`], while higher-order payloads depend on this semantic witness instead
/// of reproducing all variant-level transposition bounds.
///
/// The trait is intentionally about complete operation families, not individual primitive payloads. Implementations
/// that delegate to [`Program::transpose_with_respect_to`] add that function's [`Zero`](crate::Zero), etc. bounds
/// locally because those are requirements of the standard implementation strategy, not of this semantic witness itself.
pub trait TransposableProgramOperation<V: Value<Type: DifferentiableType>>: Operation<V::Type> + Sized {
    /// Transposes the provided [`Program`] in this operation family with respect to the inputs flagged in
    /// `input_linearity`. Refer to the documentation of [`Program::transpose_with_respect_to`] for the pullback's
    /// input and output layout. The operand-form higher-order transpose rules (e.g., for condition branches and scan
    /// bodies whose known residual factors ride as ordinary operands) pass their genuine linearity masks, while the
    /// fully linear captured-program rules pass an all-`true` mask. The flat-transposition case is exactly the partial
    /// one with every input linear, so this single function serves both.
    ///
    /// # Parameters
    ///
    ///   - `program`: [`Program`] whose inputs and outputs are flattened vectors of values.
    ///   - `input_linearity`: Per-input linearity flags, in program-input order.
    fn transpose_program(
        program: &Program<V, Self, Vec<V>, Vec<V>>,
        input_linearity: &[bool],
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, DifferentiationError>;
}

impl<
    T: DifferentiableType,
    V: Value<Type = T>,
    O: Clone + TransposableOperation<V, O> + From<ZeroOperation<T>> + From<AddOperation>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
> Program<V, O, Input, Output>
{
    /// Transposes this linear _pushforward_ [`Program`] into its reverse-mode _pullback_. This is the main entrypoint
    /// for transposing linear [`Program`]s. In the algebraic sense, _transposing_ a linear map `L: X -> Y` gives a map
    /// on _dual_ spaces `L^T: Y* -> X*`. In finite dimensions this is the same operation represented by a matrix
    /// transpose. Here the linear map is not stored as a matrix. It is a staged [`Program`] that maps input tangents
    /// to output tangents, and transposition builds the dual program that maps output cotangents back to input
    /// cotangents. Operationally, transposition creates cotangent inputs for this program's outputs, walks the
    /// instructions in reverse order, and applies each primitive operation's [`TransposableOperation::transpose`] rule
    /// to accumulate cotangent contributions for the original inputs. This is the same decomposition of reverse-mode
    /// automatic differentiation as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Over complex types, transposition is defined with respect to the **bilinear** (i.e., conjugation-free) pairing
    /// `⟨a, b⟩ = Real(a · b)`: the transpose of multiplying by a known complex factor multiplies by that same factor
    /// (never its conjugate), which keeps transposition an involution and keeps every bilinear transpose rule identical
    /// across real and complex types. Conjugation enters only through the transpose rules of the
    /// ℝ-linear-but-not-ℂ-linear primitives (i.e., `conjugate`, `real`, `imaginary`, and `complex`), whose adjoints
    /// under this pairing carry the conjugations and negations explicitly. The user-facing consequence is documented on
    /// the gradient entry points: the holomorphic ones return the complex derivative `∂f/∂z`, and the plain ones return
    /// `2 · ∂f/∂z̄` for ℂ → ℝ functions.
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`]s, which the value type's [`Zero`](crate::Zero)
    /// implementation evaluates at interpretation time. This applies uniformly to linear programs whose values are
    /// [`Tracer`]s from an outer trace. Interpreting such a pullback [`ZeroOperation`] over outer-trace [`Tracer`]s
    /// stages a typed zero into the surrounding tracing context, so that backends whose traced constants are abstract
    /// metadata do not need to materialize a runtime value just to transpose an enclosing traced program.
    ///
    /// This is the fully linear case of [`transpose_with_respect_to`](Self::transpose_with_respect_to). The program
    /// is transposed with respect to every input, so every reachable [`Atom`] is linear, each operation's transpose
    /// rule receives an all-`true` operand-linearity slice, and the pullback's inputs and outputs preserve this
    /// program's output and input structures respectively.
    #[inline]
    pub fn transpose(&self) -> Result<Program<V, O, Output, Input>, DifferentiationError> {
        // Every input is linear, so the pullback has one cotangent input per primal output and one cotangent output
        // per primal input. Recover the structured form by reattaching this program's output and input structures to
        // the flat pullback, keeping its atoms, instructions, and input/output `AtomId`s unchanged.
        let flat = self.transpose_with_respect_to(&(0..self.input_ids().len()).collect::<Vec<_>>())?;
        Ok(Program {
            atoms: flat.atoms,
            input_ids: flat.input_ids,
            output_ids: flat.output_ids,
            instructions: flat.instructions,
            input_structure: self.output_structure().clone(),
            output_structure: self.input_structure().clone(),
            marker: PhantomData,
        })
    }

    /// Transposes this linear _pushforward_ [`Program`] into its reverse-mode _pullback_ **with respect to** the inputs
    /// selected by `input_indices`, holding the remaining inputs as constant parameters of the linear map. The program
    /// must be linear in the selected inputs, but it can depend arbitrarily on the known ones. This is the partial
    /// entry point behind the fully linear [`transpose`](Self::transpose).
    ///
    /// Linearity is propagated forward from the program inputs: a program-input [`Atom`] is linear exactly when its
    /// index appears in `input_indices`, constant atoms are always known, and an operation result is linear when any
    /// of its operands is linear. Each operation's [`transpose`](TransposableOperation::transpose) rule receives the
    /// per-operand linearity knowledge derived from this propagation.
    ///
    /// The pullback's inputs are the cotangents of this program's outputs followed by the runtime values of the known
    /// inputs (in program-input order), and the pullback's outputs are the accumulated cotangents of the selected
    /// inputs, **in `input_indices` order**. Known inputs receive no cotangent output. Because this layout depends on
    /// `input_indices`, the pullback's inputs and outputs are returned as flat [`Vec`]s rather than reusing this
    /// program's structured input and output types. The fully linear [`transpose`](Self::transpose) recovers the
    /// structured form. Disconnected selected inputs are emitted as [`ZeroOperation`]s, exactly as in
    /// [`transpose`](Self::transpose).
    ///
    /// # Known Intermediates and Rematerialization
    ///
    /// The normal differentiation path linearizes and partially evaluates before transposition. Values computed only
    /// from primals then cross the linear boundary as residual inputs, and so transposing such a normalized pushforward
    /// does **not** rebuild their producer instructions in the pullback. This method nevertheless accepts a hand-built
    /// or otherwise unnormalized linear program whose live transpose rules read internal known values. For such a
    /// value, transposition lazily copies the demanded, pure known-producer ancestor subgraph into the generated
    /// pullback. This is _rematerialization_: the copied instructions execute every time the pullback is interpreted,
    /// trading saved residuals for recomputation. Only ancestors of a known value actually demanded by a live transpose
    /// rule are copied; dead known instructions and dead constants remain absent. Each source producer is copied at
    /// most once, all of its output atoms are memoized together, and every later consumer reuses those mapped outputs.
    /// The producer walk is iterative, so its call-stack usage does not grow with producer-chain depth. This behavior
    /// is a correctness fallback, and not an implicit recommendation to rematerialize hot or expensive primal work.
    /// Callers that want predictable pullback cost should partially evaluate and carry such values as residual inputs.
    /// Effectful known producers are never copied. Replaying one in the pullback could duplicate, omit, or reorder an
    /// effect that belongs on the primal side, so this method returns [`ProgramError::UnsupportedOperation`] and asks
    /// the caller to partial-evaluate that value into a residual input. Known program inputs are always exposed as
    /// pullback inputs, while literal constants are copied lazily under the same demand-driven policy.
    ///
    /// The pullback is staged into a fresh internal [`TracingContext`]: transposition records one cotangent input
    /// per program output, walks this program in reverse instruction order applying each [`Operation`]'s
    /// [`transpose`](TransposableOperation::transpose) rule, and accumulates the per-input cotangent contributions
    /// (summing repeated contributions with staged adds). A transpose rule that needs to transpose a nested subprogram
    /// (e.g., a captured control-flow branch) calls [`transpose`](Self::transpose) on it, which transposes it in its
    /// own fresh context.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Indices of the program inputs the program is transposed with respect to. Each index must
    ///     be in range and appear at most once, otherwise this returns [`ProgramError::InvalidArgument`]. The order of
    ///     the indices defines the order of the pullback's cotangent outputs.
    pub fn transpose_with_respect_to(
        &self,
        input_indices: &[usize],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        // Scatter the selected indices into the per-input linearity mask that seeds the forward propagation,
        // rejecting out-of-range and duplicate indices up front.
        let input_count = self.input_ids().len();
        let mut input_linearity = vec![false; input_count];
        for &index in input_indices {
            if index >= input_count {
                return Err(ProgramError::InvalidArgument {
                    message: format!(
                        "transposition input index {index} is out of range for a program with {input_count} input(s)",
                    ),
                }
                .into());
            }
            if input_linearity[index] {
                return Err(ProgramError::InvalidArgument {
                    message: format!("transposition input index {index} appears more than once"),
                }
                .into());
            }
            input_linearity[index] = true;
        }

        /// Accumulates one staged cotangent contribution for `atom` into the reverse-pass adjoint table. The first
        /// contribution is stored directly, while later contributions are summed by staging an add instruction in the
        /// transpose builder.
        ///
        /// # Parameters
        ///
        ///   - `builder`: Destination builder for the transposed program.
        ///   - `adjoints`: Per-primal-atom table storing the currently accumulated cotangent atom, if any.
        ///   - `atom`: Primal atom whose cotangent is being accumulated.
        ///   - `contribution`: Staged cotangent atom to add into `atom`'s adjoint slot.
        fn accumulate<V: Value, O: Operation<V::Type> + From<AddOperation>>(
            builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
            adjoints: &mut [Option<AtomId>],
            atom: AtomId,
            contribution: AtomId,
        ) -> Result<(), ProgramError> {
            // Contributions must already be atoms in the transpose builder. Otherwise the `AtomId` could alias an
            // unrelated atom index and corrupt the pullback graph.
            if builder.borrow().atoms().get(contribution.index()).is_none() {
                return Err(ProgramError::UnboundAtomId { id: contribution });
            }

            // Locate the primal atom's adjoint slot. An out-of-range slot means the input program is malformed.
            let adjoint = adjoints.get_mut(atom.index()).ok_or(ProgramError::UnboundAtomId { id: atom })?;

            // If this atom already has a cotangent, stage an add so both contributions flow into one accumulated
            // adjoint. Otherwise, keep the first contribution directly and avoid emitting an unnecessary add.
            *adjoint = Some(match *adjoint {
                Some(existing) => {
                    let mut builder_borrow = builder.borrow_mut();
                    let outputs = builder_borrow.add_instruction(AddOperation, vec![existing, contribution])?;
                    check_count!("output", outputs, 1, ProgramError);
                    outputs[0]
                }
                None => contribution,
            });
            Ok(())
        }

        /// Helper internal enum for the [`materialize_known`] implementation.
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum MaterializationState {
            Unseen,
            Visiting,
            Complete,
        }

        /// Helper internal enum for the [`materialize_known`] implementation.
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum MaterializationStep {
            Visit(AtomId),
            Replay(usize),
        }

        /// Replays the pure producer subgraph of one known atom into the pullback builder using an iterative postorder
        /// traversal. Program inputs are seeded in `known_map` by the caller, constants are copied directly, and all
        /// outputs of a replayed instruction are memoized together so shared producers and sibling results are emitted
        /// only once. `materialization_state` distinguishes scheduled producers from completed ones, both detecting a
        /// malformed cycle and keeping the traversal independent of the native call stack.
        ///
        /// # Parameters
        ///
        ///   - `program`: Source program containing the known atom and its producer subgraph.
        ///   - `instruction_by_output`: Source-instruction index for each produced atom, or `None` for atoms
        ///     without an instruction producer.
        ///   - `linear`: Per-source-atom mask indicating whether the atom depends on a selected linear input.
        ///   - `builder`: Destination pullback builder into which demanded pure producers are replayed.
        ///   - `known_map`: Per-source-atom mapping to an already materialized pullback atom.
        ///   - `materialization_state`: Per-source-instruction traversal state used for memoization
        ///      and cycle detection.
        ///   - `atom`: ID of the known source atom to materialize in the pullback builder.
        fn materialize_known<
            V: Value,
            O: Clone + Operation<V::Type>,
            Input: Parameterized<V>,
            Output: Parameterized<V>,
        >(
            program: &Program<V, O, Input, Output>,
            instruction_by_output: &[Option<usize>],
            linear: &[bool],
            builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
            known_map: &mut [Option<AtomId>],
            materialization_state: &mut [MaterializationState],
            atom: AtomId,
        ) -> Result<AtomId, ProgramError> {
            let mut steps = vec![MaterializationStep::Visit(atom)];
            while let Some(step) = steps.pop() {
                match step {
                    MaterializationStep::Visit(current) => {
                        if known_map.get(current.index()).copied().flatten().is_some() {
                            continue;
                        }
                        if *linear.get(current.index()).ok_or(ProgramError::UnboundAtomId { id: current })? {
                            return Err(ProgramError::MalformedProgram(
                                "a linear atom was requested as a known transpose operand".to_string(),
                            ));
                        }
                        let source =
                            program.atoms().get(current.index()).ok_or(ProgramError::UnboundAtomId { id: current })?;
                        if let Atom::Constant(value) = source {
                            let mapped = builder.borrow_mut().add_constant(value.clone());
                            known_map[current.index()] = Some(mapped);
                            continue;
                        }
                        let instruction_index =
                            instruction_by_output.get(current.index()).copied().flatten().ok_or_else(|| {
                                ProgramError::MalformedProgram("known variable atom has no owning instruction".into())
                            })?;
                        let instruction = program
                            .instructions()
                            .get(instruction_index)
                            .ok_or_else(|| ProgramError::MalformedProgram("known atom producer is missing".into()))?;
                        if !instruction.operation().effects().is_pure() {
                            return Err(ProgramError::UnsupportedOperation {
                                message: format!(
                                    "partition-aware transpose cannot replay effectful known intermediate producer \
                                     `{}`; partial-evaluate it into a residual input first",
                                    instruction.operation().name(),
                                ),
                            });
                        }
                        match materialization_state.get_mut(instruction_index).ok_or_else(|| {
                            ProgramError::MalformedProgram("known atom producer state is missing".into())
                        })? {
                            state @ MaterializationState::Unseen => *state = MaterializationState::Visiting,
                            MaterializationState::Visiting => {
                                return Err(ProgramError::MalformedProgram(
                                    "known intermediate producer graph contains a cycle".into(),
                                ));
                            }
                            MaterializationState::Complete => {
                                return Err(ProgramError::MalformedProgram(
                                    "materialized known producer output was not remapped".into(),
                                ));
                            }
                        }
                        steps.push(MaterializationStep::Replay(instruction_index));
                        steps.extend(instruction.inputs().iter().rev().copied().map(MaterializationStep::Visit));
                    }
                    MaterializationStep::Replay(instruction_index) => {
                        let instruction = program
                            .instructions()
                            .get(instruction_index)
                            .ok_or_else(|| ProgramError::MalformedProgram("known atom producer is missing".into()))?;
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .map(|input| {
                                known_map.get(input.index()).copied().flatten().ok_or_else(|| {
                                    ProgramError::MalformedProgram(
                                        "known producer input was not remapped before replay".into(),
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let outputs =
                            builder.borrow_mut().add_instruction(instruction.operation().clone(), inputs)?.to_vec();
                        check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                        for (source, mapped) in instruction.outputs().iter().copied().zip(outputs) {
                            known_map[source.index()] = Some(mapped);
                        }
                        *materialization_state.get_mut(instruction_index).ok_or_else(|| {
                            ProgramError::MalformedProgram("known atom producer state is missing".into())
                        })? = MaterializationState::Complete;
                    }
                }
            }
            known_map
                .get(atom.index())
                .copied()
                .flatten()
                .ok_or_else(|| ProgramError::MalformedProgram("known producer output was not remapped".into()))
        }

        // Propagate operand linearity forward over the primal atoms. A program-input atom takes its linearity from
        // `input_linearity`, a constant atom is always known (non-linear), and an instruction result is linear when
        // any of its operands is linear. Because instructions are stored in evaluation order, a single forward pass
        // suffices: every operand atom of an instruction is defined before that instruction. With an all-`true` mask
        // every reachable variable becomes linear, so each operation's transpose rule sees an all-`true` operand slice
        // and behaves exactly as it did before partition-aware transposition.
        let mut linear = vec![false; self.atoms().len()];
        for (input, &input_is_linear) in self.input_ids().iter().copied().zip(input_linearity.iter()) {
            *linear.get_mut(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? = input_is_linear;
        }
        for (index, atom) in self.atoms().iter().enumerate() {
            if matches!(atom, Atom::Constant(_)) {
                linear[index] = false;
            }
        }
        for instruction in self.instructions().iter() {
            let mut output_is_linear = false;
            for input in instruction.inputs().iter().copied() {
                if *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                    output_is_linear = true;
                    break;
                }
            }
            for output in instruction.outputs().iter().copied() {
                *linear.get_mut(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })? = output_is_linear;
            }
        }

        // Stage the pullback into a fresh tracing context's builder, and reserve the main structural vectors up front.
        // These are conservative lower bounds that cover cotangent inputs, one instruction per reversed primal
        // instruction, and possible zero outputs for disconnected primal inputs.
        let mut context = TracingContext::<V, O>::new();
        let builder = context.builder().clone();
        {
            let mut builder_borrow = builder.borrow_mut();
            builder_borrow.atoms.reserve(self.output_ids.len() + self.instructions.len() + self.input_ids.len());
            builder_borrow.input_ids.reserve(self.output_ids.len());
            builder_borrow.instructions.reserve(self.instructions.len() + self.input_ids.len());
        }

        // Seed the reverse pass with one cotangent input for each primal output, typed with that output's cotangent
        // slot type. A differentiable output's slot carries its cotangent dual (e.g., swapping unreduced and reduced
        // sharding axes for arrays). A non-differentiable output (i.e., the analogue to JAX's `float0`, such as a
        // Boolean or integer) has no cotangent space, so its slot carries only structural zeros typed by the output's
        // own primal type. The adjoint table is indexed by atoms from the original program, and each slot stores the
        // staged pullback atom that currently represents the accumulated cotangent for that primal atom.
        let mut adjoints = vec![None; self.atoms().len()];
        for output in self.output_ids().iter().copied() {
            let output_atom = self.atoms().get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
            let output_type = output_atom.r#type();
            let cotangent_type = output_type.cotangent().unwrap_or_else(|| output_type.into_owned());
            let cotangent_input = builder.borrow_mut().add_input(cotangent_type);
            if *linear.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })? {
                accumulate::<V, O>(&builder, adjoints.as_mut_slice(), output, cotangent_input)?;
            }
        }

        // Add a pullback input carrying the runtime value of each known program input, after the cotangent inputs so
        // the all-`true` mask leaves the pullback input numbering unchanged. Known inputs are exposed to transpose
        // rules as ordinary operand values, typed with the known input's own type (a runtime value, not a cotangent),
        // and recorded in `known_map` indexed by the primal atom so a rule can read the known operand's pullback atom.
        let mut known_map = vec![None; self.atoms().len()];
        for (input, &input_is_linear) in self.input_ids().iter().copied().zip(input_linearity.iter()) {
            if !input_is_linear {
                let input_atom = self.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                let known_input = builder.borrow_mut().add_input(input_atom.r#type().into_owned());
                known_map[input.index()] = Some(known_input);
            }
        }

        // Constants and pure known intermediates are materialized lazily below, only when a live transpose rule
        // needs them. This avoids copying dead constants and replaying dead known-side work into the pullback.
        let instruction_by_output = self.instruction_by_output();
        let mut materialization_state = vec![MaterializationState::Unseen; self.instructions().len()];

        // Walk the primal program backward, applying each operation's transpose rule only when at least one of its
        // outputs has a non-zero accumulated cotangent. The scratch vector avoids allocating a fresh cotangent vector
        // for every live instruction.
        let max_instruction_output_count =
            self.instructions().iter().map(|instruction| instruction.outputs().len()).max().unwrap_or(0);
        let mut instruction_output_cotangents = Vec::with_capacity(max_instruction_output_count);
        for instruction in self.instructions().iter().rev() {
            // Skip dead reverse edges early. If none of an instruction's outputs carries an adjoint, the instruction
            // cannot contribute to any input cotangent. This is the only operand-side guard. A live transpose rule may
            // read non-linear operands; pure known producer subgraphs are materialized lazily below, while effectful
            // known producers are rejected rather than duplicated or reordered in the pullback.
            let mut has_output_adjoint = false;
            for output in instruction.outputs().iter().copied() {
                if adjoints.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?.is_some() {
                    has_output_adjoint = true;
                    break;
                }
            }
            if !has_output_adjoint {
                continue;
            }

            // Materialize the instruction's output cotangents in operation-result order. Missing adjoint slots become
            // structural zeros so transpose rules can distinguish unused outputs without staging zero operations.
            // Structural zeros carry the output's cotangent slot type: a differentiable output's cotangent dual or,
            // for a non-differentiable output (i.e., the analogue to JAX's `float0`), the output's own primal type.
            // Accumulated adjoints are always live: rules communicate zero-ness symbolically through `MaybeZero`
            // (opaque program splices such as the custom-VJP backward replay recover it at their own boundary),
            // and so no staged canonical zero ever needs to be recognized here.
            instruction_output_cotangents.clear();
            for output in instruction.outputs().iter().copied() {
                let cotangent = adjoints.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
                instruction_output_cotangents.push(match cotangent {
                    Some(atom) => MaybeZero::Value(context.tracer(*atom, None)),
                    None => {
                        let output_type = self
                            .atoms()
                            .get(output.index())
                            .ok_or(ProgramError::UnboundAtomId { id: output })?
                            .r#type();
                        MaybeZero::Zero(output_type.cotangent().unwrap_or_else(|| output_type.into_owned()))
                    }
                });
            }

            // Apply the primitive transpose rule and require exactly one cotangent contribution per primal input. This
            // prevents malformed rules from silently dropping or inventing cotangents through iterator truncation. Each
            // input/operand becomes a self-describing `PartialValue`: a linear operand is `Unknown` of its type (the
            // rule produces a cotangent of that type), and a known operand is `Known` of the tracer reading its
            // pullback value atom from `known_map`. Known inputs are seeded above, constants are copied lazily, and
            // pure known intermediates iteratively replay their producer subgraphs exactly once before the rule runs.
            let inputs = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input| {
                    let r#type = self
                        .atoms()
                        .get(input.index())
                        .ok_or(ProgramError::UnboundAtomId { id: input })?
                        .r#type()
                        .into_owned();
                    if *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                        Ok(PartialValue::Unknown(r#type))
                    } else {
                        let atom = materialize_known(
                            self,
                            instruction_by_output.as_slice(),
                            linear.as_slice(),
                            &builder,
                            known_map.as_mut_slice(),
                            materialization_state.as_mut_slice(),
                            input,
                        )?;
                        Ok(PartialValue::Known(context.tracer(atom, Some(r#type))))
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let input_cotangents = instruction.operation().transpose(
                &mut context,
                inputs.as_slice(),
                instruction_output_cotangents.as_slice(),
            )?;
            check_count!("input", input_cotangents, instruction.inputs().len(), ProgramError);
            for (input, contribution) in instruction.inputs().iter().copied().zip(input_cotangents) {
                if !*linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                    continue;
                }
                if let Some(contribution) = contribution.as_value() {
                    // Staged contributions must belong to this builder before their atom IDs can be accumulated.
                    check_builders!(&builder, contribution.builder())?;
                    accumulate::<V, O>(&builder, adjoints.as_mut_slice(), input, contribution.atom_id()?)?;
                }
            }
        }
        instruction_output_cotangents.clear();

        // The pullback outputs are the accumulated cotangents of the selected inputs, emitted directly in
        // `input_indices` order. Known inputs receive no cotangent output. Disconnected selected inputs are emitted
        // as input-free `ZeroOperation` instructions, which the value type's `Zero` implementation evaluates at
        // interpretation time, typed with the input's cotangent slot type: a differentiable input's cotangent dual,
        // or, for a non-differentiable selected input (i.e., the analogue to JAX's `float0`), the input's own primal
        // type, whose cotangent slot carries only structural zeros.
        let outputs = input_indices
            .iter()
            .map(|&index| {
                let input = self.input_ids()[index];
                match adjoints.get(input.index()).copied().ok_or(ProgramError::UnboundAtomId { id: input })? {
                    Some(adjoint) => Ok::<AtomId, ProgramError>(adjoint),
                    None => {
                        let input_atom =
                            self.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                        let input_type = input_atom.r#type();
                        let cotangent_type = input_type.cotangent().unwrap_or_else(|| input_type.into_owned());
                        let mut builder_borrow = builder.borrow_mut();
                        let outputs = builder_borrow.add_instruction(ZeroOperation::new(cotangent_type), Vec::new())?;
                        check_count!("output", outputs, 1, ProgramError);
                        Ok(outputs[0])
                    }
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        // Drop the throwaway context so its builder reference is released; with every staged `Tracer` already dropped,
        // the cloned `builder` handle is now the sole owner and can be unwrapped to finalize the pullback.
        drop(context);

        // Build the pullback from the context's builder. The pullback inputs (i.e., cotangents per primal output, then
        // known-input values) and outputs (i.e., cotangents for the linear inputs) are flat, and so they are built with
        // flat `Vec` structures. The fully linear callers recover the structured form by reattaching the program's
        // input and output structures.
        let pullback_input_count = builder.borrow().input_ids().len();
        let pullback_output_count = outputs.len();
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(ProgramError::EscapedProgramBuilder.into()),
        };
        builder
            .build(outputs, vec![Placeholder; pullback_input_count], vec![Placeholder; pullback_output_count])
            .map_err(DifferentiationError::from)
    }
}

/// Extension trait carrying the value-level *reverse-mode* differentiation transforms on every [`Context`]. Reverse
/// mode differentiation is implemented as forward mode plus transposition. Like [`ForwardModeDifferentiate`], this
/// trait is blanket-implemented for all [`Context`]s and has no items of its own to implement: every entry point is a
/// defaulted function whose `where` clause carries its actual requirements. On top of the forward mode requirements,
/// reverse mode differentiation needs the operation family's [`TransposableOperation`] rules, and the gradient entry
/// points additionally need a [`DifferentiableType`] whose scalar outputs carry a cotangent space to seed. Cotangents,
/// like tangents, are ordinary values of the same universe as the primals (i.e., [`Domain::Value`]) flowing through
/// the same context.
pub trait ReverseModeDifferentiate: ForwardModeDifferentiate {
    /// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`],
    /// with this [`Context`] executing (or staging) the primal-side operations. Refer to the documentation of the
    /// [`vjp`] function for information on the reverse-mode differentiation transform and its arguments.
    fn vjp<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Output::To<Self::Value>, Pullback<Self, Input, Output::To<Self::Value>>), DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<AddOperation>,
    {
        let input_structure = primals.parameter_structure();
        let (output, pushforward) = self.linearize(function, primals)?;
        let (program, residuals) = pushforward.into_parts();
        // Transpose the pushforward program with respect to its leading tangent inputs, holding the trailing residual
        // inputs as known parameters. Partition-aware transposition threads each residual through to the pullback as a
        // pullback input rather than folding it into a captured factor, so the pullback maps
        // `(output_cotangents ++ residuals)` to the input cotangents.
        let with_respect_to = (0..program.input_ids().len() - residuals.len()).collect::<Vec<_>>();
        let program = program.transpose_with_respect_to(with_respect_to.as_slice())?;
        Ok((output, Pullback::new(self.clone(), program, residuals, input_structure)?))
    }

    /// Computes both the primal scalar output of `function` at `primals` and its reverse-mode gradient, with this
    /// [`Context`] executing (or staging) the primal-side operations and the pullback replay. Refer to the
    /// documentation of the [`value_and_gradient`] function for information on the transform and its arguments.
    fn value_and_gradient<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<LinearizationTracer<Self>, DifferentiationError>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input::To<Self::Value>), DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
    {
        // Seed the single output cotangent with the multiplicative identity of the scalar output and pull it back
        // to the input cotangents, reshaped against the closure's input structure.
        let (output, pullback) =
            self.vjp(|input| function(input).into_result().map_err(ProgramError::from), primals)?;
        let seed = self.gradient_seed(&output, false)?;
        let gradient = pullback.apply(seed)?;
        Ok((output, gradient))
    }

    /// Computes the reverse-mode gradient of `function` at `primals`, with this [`Context`] executing (or staging)
    /// the primal-side operations and the pullback replay. This is the gradient-only counterpart of
    /// [`value_and_gradient`](Self::value_and_gradient), discarding the primal output. Refer to the documentation of
    /// the [`gradient`] function for information on the transform and its arguments.
    #[inline]
    fn gradient<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<LinearizationTracer<Self>, DifferentiationError>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<Self::Value>, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
    {
        self.value_and_gradient(function, primals).map(|(_, gradient)| gradient)
    }

    /// Computes both the primal scalar output of `function` at `primals` and its holomorphic reverse-mode gradient,
    /// with this [`Context`] executing (or staging) the primal-side operations and the pullback replay. Refer to the
    /// documentation of the [`value_and_gradient_holomorphic`] function for information on the transform,
    /// its arguments, and the holomorphy promise it relies on.
    fn value_and_gradient_holomorphic<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<LinearizationTracer<Self>, DifferentiationError>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Self::Value, Input::To<Self::Value>), DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
    {
        // This function implementation is identical to `value_and_gradient` except that the seed is gated on
        // holomorphy. the output must be complex, and under the caller's holomorphy promise the single seed recovers
        // the complex derivative ∂f/∂z.
        let (output, pullback) =
            self.vjp(|input| function(input).into_result().map_err(ProgramError::from), primals)?;
        let seed = self.gradient_seed(&output, true)?;
        let gradient = pullback.apply(seed)?;
        Ok((output, gradient))
    }

    /// Computes the holomorphic reverse-mode gradient of `function` at `primals`, with this [`Context`] executing
    /// (or staging) the primal-side operations and the pullback replay. This is the gradient-only counterpart of
    /// [`value_and_gradient_holomorphic`](Self::value_and_gradient_holomorphic), discarding the primal output. Refer
    /// to the documentation of the [`gradient_holomorphic`] function for information on the transform, its arguments,
    /// and the holomorphy promise it relies on.
    #[inline]
    fn gradient_holomorphic<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<LinearizationTracer<Self>, DifferentiationError>,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<Input::To<Self::Value>, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
    {
        self.value_and_gradient_holomorphic(function, primals).map(|(_, gradient)| gradient)
    }

    /// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its reverse-mode gradient,
    /// with this [`Context`] executing (or staging) the primal-side operations and the pullback replay. Refer to the
    /// documentation of the [`value_and_gradient_with_aux`] function for information on the transform and its
    /// arguments.
    fn value_and_gradient_with_aux<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<(LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>), DifferentiationError>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<((Self::Value, Aux), Input::To<Self::Value>), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        let input_structure = primals.parameter_structure();
        let ((output, aux), pullback): ((Self::Value, Aux), _) =
            self.vjp(|input| function(input).into_result().map_err(ProgramError::from), primals)?;
        let (pullback, residuals) = pullback.into_parts();
        // The flat pullback consumes `[output_cotangents ++ residuals]`. The traced output flattens as the scalar
        // output leaf followed by the auxiliary leaves, so seed the output leaf with a one cotangent and every
        // auxiliary leaf with a zero cotangent, then append the linearization-point residuals. Both the seeds and the
        // replay go through this context itself. An eager context constructs and interprets concrete values, while a
        // staging context stages into its enclosing trace.
        let mut pullback_inputs = vec![self.gradient_seed(&output, false)?];
        for value in Parameterized::<Self::Value>::parameters(&aux) {
            pullback_inputs.push(self.zero(value.r#type().as_ref())?);
        }
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(self, pullback_inputs)?;
        let gradient =
            Input::To::<Self::Value>::from_parameters(input_structure, input_cotangents).map_err(ProgramError::from)?;
        Ok(((output, aux), gradient))
    }

    /// Computes the reverse-mode gradient of `function` at `primals` and its auxiliary outputs, with this [`Context`]
    /// executing (or staging) the primal-side operations and the pullback replay. This is the gradient-only counterpart
    /// of [`value_and_gradient_with_aux`](Self::value_and_gradient_with_aux), discarding the primal scalar output.
    /// Refer to the documentation of the [`gradient_with_aux`] function for information on the transform and its
    /// arguments.
    #[inline]
    fn gradient_with_aux<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<(LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>), DifferentiationError>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Input::To<Self::Value>, Aux), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        self.value_and_gradient_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
    }

    /// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its holomorphic reverse-mode
    /// gradient, with this [`Context`] executing (or staging) the primal-side operations and the pullback replay.
    /// Refer to the documentation of the [`value_and_gradient_holomorphic_with_aux`] function for information on
    /// the transform, its arguments, and the holomorphy promise it relies on.
    fn value_and_gradient_holomorphic_with_aux<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<(LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>), DifferentiationError>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<((Self::Value, Aux), Input::To<Self::Value>), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        // This function implementation is identical to `value_and_gradient_with_aux` except that the seed is gated on
        // holomorphy. The output must be complex, and under the caller's holomorphy promise the single seed recovers
        // the complex derivative ∂f/∂z.
        let input_structure = primals.parameter_structure();
        let ((output, aux), pullback): ((Self::Value, Aux), _) =
            self.vjp(|input| function(input).into_result().map_err(ProgramError::from), primals)?;
        let (pullback, residuals) = pullback.into_parts();
        let mut pullback_inputs = vec![self.gradient_seed(&output, true)?];
        for value in Parameterized::<Self::Value>::parameters(&aux) {
            pullback_inputs.push(self.zero(value.r#type().as_ref())?);
        }
        pullback_inputs.extend(residuals);
        let input_cotangents = pullback.interpret_in_context(self, pullback_inputs)?;
        let gradient =
            Input::To::<Self::Value>::from_parameters(input_structure, input_cotangents).map_err(ProgramError::from)?;
        Ok(((output, aux), gradient))
    }

    /// Computes the holomorphic reverse-mode gradient of `function` at `primals` and its auxiliary outputs, with this
    /// [`Context`] executing (or staging) the primal-side operations and the pullback replay. This is the gradient-only
    /// counterpart of [`value_and_gradient_holomorphic_with_aux`](Self::value_and_gradient_holomorphic_with_aux),
    /// discarding the primal scalar output. Refer to the documentation of the [`gradient_holomorphic_with_aux`]
    /// function for information on the transform, its arguments, and the holomorphy promise it relies on.
    #[inline]
    fn gradient_holomorphic_with_aux<
        F: FnOnce(Input::To<LinearizationTracer<Self>>) -> Output,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: MaybeFallible<(LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>), DifferentiationError>,
        Aux: Parameterized<
                Self::Value,
                To<Self::Value> = Aux,
                Family: ParameterizedFamily<LinearizationTracer<Self>, To = Aux::To<LinearizationTracer<Self>>>,
            >,
    >(
        &self,
        function: F,
        primals: Input,
    ) -> Result<(Input::To<Self::Value>, Aux), DifferentiationError>
    where
        Self: Zero<Self::Value>,
        Self::Type: DifferentiableType,
        Self::Operation: Clone
            + DifferentiableOperation<PartialEvaluationContext<Self>>
            + PartiallyEvaluatableOperation<Self>
            + TransposableOperation<Self::Constant, Self::Operation>
            + From<ZeroOperation<Self::Type>>
            + From<OneOperation<Self::Type>>
            + From<AddOperation>,
        (LinearizationTracer<Self>, Aux::To<LinearizationTracer<Self>>): Parameterized<
                LinearizationTracer<Self>,
                To<Self::Value> = (Self::Value, Aux),
                Family: ParameterizedFamily<Self::Value>,
            >,
    {
        self.value_and_gradient_holomorphic_with_aux(function, primals)
            .map(|((_, aux), gradient)| (gradient, aux))
    }

    /// Validates the scalar `output` of a gradient entry point and constructs its cotangent seed. The output must be
    /// a single rank-0 scalar with a cotangent space, and complex outputs additionally require `holomorphic`: a single
    /// reverse-mode seed recovers the derivative of a complex-output function only when the function is holomorphic, so
    /// without that promise a complex output is rejected with an error instead of silently computing a value that is
    /// not a derivative (i.e., `holomorphic` changes nothing for real outputs). The seed is the multiplicative identity
    /// typed with the output's cotangent type (e.g., swapping unreduced and reduced sharding axes for arrays) and bound
    /// through this context, so an eager context constructs a concrete value while a staging context stages into its
    /// enclosing trace.
    ///
    /// This is the shared seeding step behind [`value_and_gradient`](Self::value_and_gradient),
    /// [`value_and_gradient_holomorphic`](Self::value_and_gradient_holomorphic), and
    /// [`value_and_gradient_with_aux`](Self::value_and_gradient_with_aux), exposed so that custom gradient-style
    /// entry points built on top of [`vjp`](Self::vjp) can reuse the same validation and seeding contract.
    fn gradient_seed(&self, output: &Self::Value, holomorphic: bool) -> Result<Self::Value, DifferentiationError>
    where
        Self::Type: DifferentiableType,
        Self::Operation: From<OneOperation<Self::Type>>,
    {
        // Reverse mode only defines a gradient for scalar-output functions.
        let output_type = output.r#type();
        if !output_type.is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output_type.to_string() });
        }
        if !holomorphic && output_type.is_complex() {
            return Err(DifferentiationError::ComplexGradientOutput { output_type: output_type.to_string() });
        }
        // A non-differentiable scalar output carries no cotangent space and thus no "one" to seed, so reverse mode
        // is degenerate and is rejected up front.
        let output_cotangent_type = output_type.cotangent().ok_or_else(|| {
            DifferentiationError::NonDifferentiableGradientOutput { output_type: output_type.to_string() }
        })?;
        let mut seeds = self.bind(OneOperation::new(output_cotangent_type), &[])?;
        check_count!("output", seeds, 1, ProgramError);
        Ok(seeds.pop().unwrap())
    }
}

impl<C: Context> ReverseModeDifferentiate for C {}

/// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`]
/// (i.e., the analogue of [JAX's `vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html)). For `f` computing
/// `y = f(x)`, this computes `y` together with the reusable transposed linear map `ȳ ↦ x̄ = (∂f/∂x)(x)ᵀ · ȳ`, pulling
/// output cotangents back to input cotangents.
///
/// The returned [`Pullback`] closes the transposed program over the linearization-point residuals, so that
/// [`Pullback::apply`] maps output cotangents to input cotangents, appending the residuals, interpreting the program,
/// and reshaping the flat input cotangents against the closure's input structure, without the caller threading the
/// residuals by hand.
#[inline]
pub fn vjp<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<AddOperation>,
            >,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: Parameterized<LinearizationTracer<V::ExecutionDomain>, Family: ParameterizedFamily<V>>,
>(
    function: F,
    primals: Input,
) -> Result<(Output::To<V>, Pullback<V::ExecutionDomain, Input, Output::To<V>>), DifferentiationError> {
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.vjp(function, primals)
}

/// Computes both the primal scalar output of `function` at `primals` and its reverse-mode gradient. For `f` computing
/// a real scalar `y = f(x)`, this computes `(f(x), ∇f(x))`, where `∇f(x) = (∂f/∂x)(x)ᵀ · 1` is the pullback of the
/// multiplicative-identity seed. The provided `function` may return its traced output either directly or wrapped in a
/// [`Result`] whose error type converts into [`DifferentiationError`], so `?` can be used inside the closure. Refer to
/// [`MaybeFallible`] for the exact contract.
#[inline]
pub fn value_and_gradient<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            >,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<LinearizationTracer<V::ExecutionDomain>, DifferentiationError>,
>(
    function: F,
    primals: Input,
) -> Result<(V, Input::To<V>), DifferentiationError> {
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.value_and_gradient(function, primals)
}

/// Computes the reverse-mode gradient of `function` at `primals` (i.e., the analogue of
/// [JAX's `grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)). For `f` computing a real
/// scalar `y = f(x)`, this computes `∇f(x) = (∂f/∂x)(x)ᵀ · 1`. This is the gradient-only counterpart of
/// [`value_and_gradient`], discarding the primal output. The provided `function` may return its traced output either
/// directly or wrapped in a [`Result`] whose error type converts into [`DifferentiationError`], so `?` can be used
/// inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn gradient<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            >,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<LinearizationTracer<V::ExecutionDomain>, DifferentiationError>,
>(
    function: F,
    primals: Input,
) -> Result<Input::To<V>, DifferentiationError> {
    value_and_gradient(function, primals).map(|(_, gradient)| gradient)
}

/// Computes both the primal scalar output of `function` at `primals` and its holomorphic reverse-mode gradient (i.e.,
/// the analogue of [JAX's `grad(f, holomorphic=True)`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)).
/// For a holomorphic function `f` computing a complex scalar `y = f(z)`, this function computes `(f(z), ∂f/∂z)`
/// (i.e., the complex derivative itself rather than a conjugate steepest-ascent direction). The provided `function`
/// may return its traced output either directly or wrapped in a [`Result`] whose error type converts into
/// [`DifferentiationError`], so `?` can be used inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn value_and_gradient_holomorphic<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            >,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<LinearizationTracer<V::ExecutionDomain>, DifferentiationError>,
>(
    function: F,
    primals: Input,
) -> Result<(V, Input::To<V>), DifferentiationError> {
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.value_and_gradient_holomorphic(function, primals)
}

/// Computes the holomorphic reverse-mode gradient of `function` at `primals`. For a holomorphic function `f` computing
/// a complex scalar `y = f(z)`, this function computes `∂f/∂z`. This is the gradient-only counterpart of
/// [`value_and_gradient_holomorphic`], discarding the primal output. The provided `function` may return its traced
/// output either directly or wrapped in a [`Result`] whose error type converts into [`DifferentiationError`], so `?`
/// can be used inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn gradient_holomorphic<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            >,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<LinearizationTracer<V::ExecutionDomain>, DifferentiationError>,
>(
    function: F,
    primals: Input,
) -> Result<Input::To<V>, DifferentiationError> {
    value_and_gradient_holomorphic(function, primals).map(|(_, gradient)| gradient)
}

/// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its reverse-mode
/// gradient. For a function `f` computing `(y, aux) = f(x)` with a real scalar `y`, this function computes
/// `((y, aux), (∂y/∂x)(x)ᵀ · 1)` where only `y` is differentiated, while the auxiliary outputs ride
/// along as primal values with zero cotangent seeds. The provided `function` may return its traced output either
/// directly or wrapped in a [`Result`] whose error type converts into [`DifferentiationError`], so `?` can be used
/// inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn value_and_gradient_with_aux<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            > + Zero<V>,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<
            (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
            DifferentiationError,
        >,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
>(
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
where
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.value_and_gradient_with_aux(function, primals)
}

/// Computes the reverse-mode gradient of `function` at `primals` and its auxiliary outputs. For a function `f`
/// computing `(y, aux) = f(x)` with a real scalar `y`, this computes `((∂y/∂x)(x)ᵀ · 1, aux)`. This is the
/// gradient-only counterpart of [`value_and_gradient_with_aux`], discarding the primal scalar output. The provided
/// `function` may return its traced output either directly or wrapped in a [`Result`] whose error type converts into
/// [`DifferentiationError`], so `?` can be used inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn gradient_with_aux<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            > + Zero<V>,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<
            (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
            DifferentiationError,
        >,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
>(
    function: F,
    primals: Input,
) -> Result<(Input::To<V>, Aux), DifferentiationError>
where
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    value_and_gradient_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

/// Computes the scalar output of `function` at `primals`, its auxiliary outputs, and its holomorphic reverse-mode
/// gradient. For a holomorphic function `f` computing `(y, aux) = f(z)` with a complex scalar `y`, this function
/// computes `((y, aux), ∂y/∂z)`. This is [`value_and_gradient_with_aux`] with the complex-output guard lifted, exactly
/// as [`value_and_gradient_holomorphic`] lifts it for [`value_and_gradient`]. The provided `function` may return its
/// traced output either directly or wrapped in a [`Result`] whose error type converts into [`DifferentiationError`],
/// so `?` can be used inside the closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn value_and_gradient_holomorphic_with_aux<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            > + Zero<V>,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<
            (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
            DifferentiationError,
        >,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
>(
    function: F,
    primals: Input,
) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
where
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    let Some(context) = primals.parameters().next().map(Value::execution_domain) else {
        return Err(DifferentiationError::EmptyInput);
    };
    context.value_and_gradient_holomorphic_with_aux(function, primals)
}

/// Computes the holomorphic reverse-mode gradient of `function` at `primals` and its auxiliary outputs. For
/// a holomorphic function `f` computing `(y, aux) = f(z)` with a complex scalar `y`, this function computes
/// `(∂y/∂z, aux)`. This is the gradient-only counterpart of [`value_and_gradient_holomorphic_with_aux`],
/// discarding the primal scalar output. The provided `function` may return its traced output either directly or
/// wrapped in a [`Result`] whose error type converts into [`DifferentiationError`], so `?` can be used inside the
/// closure. Refer to [`MaybeFallible`] for the exact contract.
#[inline]
pub fn gradient_holomorphic_with_aux<
    V: Value<
            Type: DifferentiableType,
            ExecutionDomain: Context<
                Operation: Clone
                               + DifferentiableOperation<PartialEvaluationContext<V::ExecutionDomain>>
                               + PartiallyEvaluatableOperation<V::ExecutionDomain>
                               + TransposableOperation<
                    <V::ExecutionDomain as Domain>::Constant,
                    <V::ExecutionDomain as Domain>::Operation,
                > + From<ZeroOperation<V::Type>>
                               + From<OneOperation<V::Type>>
                               + From<AddOperation>,
            > + Zero<V>,
        >,
    F: FnOnce(Input::To<LinearizationTracer<V::ExecutionDomain>>) -> Output,
    Input: Parameterized<V, To<V> = Input, Family: ParameterizedFamily<LinearizationTracer<V::ExecutionDomain>>>,
    Output: MaybeFallible<
            (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>),
            DifferentiationError,
        >,
    Aux: Parameterized<
            V,
            To<V> = Aux,
            Family: ParameterizedFamily<
                LinearizationTracer<V::ExecutionDomain>,
                To = Aux::To<LinearizationTracer<V::ExecutionDomain>>,
            >,
        >,
>(
    function: F,
    primals: Input,
) -> Result<(Input::To<V>, Aux), DifferentiationError>
where
    (LinearizationTracer<V::ExecutionDomain>, Aux::To<LinearizationTracer<V::ExecutionDomain>>):
        Parameterized<LinearizationTracer<V::ExecutionDomain>, To<V> = (V, Aux), Family: ParameterizedFamily<V>>,
{
    value_and_gradient_holomorphic_with_aux(function, primals).map(|((_, aux), gradient)| (gradient, aux))
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use std::cell::Cell;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::differentiation::jvp;
    use crate::effects::{Effect, Effects};
    use crate::macros::check_count;
    use crate::operations::BooleanLike;
    use crate::operations::Operation;
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramBuilder, ProgramError, Value};
    use crate::tracing::{DomainTracer, DomainTracingContext, Trace, Tracer, TracingContext};
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    type TestTracingValue = DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>;

    /// Test-only linear operation type used to exercise transposition validation paths. Most variants model tiny scalar
    /// primitives so the generated programs stay readable. The sentinel variants intentionally violate transpose rule
    /// contracts or builder ownership rules. Built-in scalar operations cannot represent those failures because their
    /// transpose implementations are valid by construction.
    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        /// Single-input passthrough used when a test needs a live instruction whose transpose forwards its cotangent.
        Identity,

        /// Effectful passthrough used to verify that known-side producer replay never duplicates observable effects.
        EffectfulIdentity,

        /// Two-input addition used to verify cotangent accumulation through repeated primal inputs.
        Add,

        /// Single-input, two-output operation used to verify that unused operation results are passed to transpose
        /// rules as structural zero cotangents.
        TwoOutputs,

        /// Single-input operation whose transpose stages a [`ZeroOperation`] as the input cotangent contribution. The
        /// staged zero remains an input-free [`ZeroOperation`] instruction in the pullback and is materialized at
        /// interpretation time. Built-in scalar operations do not stage that exact structural-zero contribution, so
        /// this sentinel keeps that path directly covered.
        StagedZeroContribution,

        /// Single-input operation whose transpose deliberately returns no input cotangent contributions. This violates
        /// the transpose-rule arity contract and verifies that the transposition pass rejects malformed custom rules
        /// instead of silently dropping cotangents.
        BadArity,

        /// Single-input operation whose transpose deliberately returns a cotangent staged in another builder. This
        /// verifies that the transposition pass rejects contributions from foreign builders before their atom IDs can
        /// alias unrelated atoms in the destination pullback.
        ForeignContribution,

        /// Real zero operation wrapper used by the `From<ZeroOperation>`/`TryFrom<ZeroOperation>` conversions for this
        /// test operation enum.
        Zero(ZeroOperation<DataType>),
    }

    impl Operation<DataType> for TestLinearOperation {
        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Identity => "identity",
                Self::EffectfulIdentity => "effectful_identity",
                Self::Add => "add",
                Self::TwoOutputs => "two_outputs",
                Self::StagedZeroContribution => "staged_zero_contribution",
                Self::BadArity => "bad_arity",
                Self::ForeignContribution => "foreign_contribution",
                Self::Zero(_) => "zero",
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            match self {
                Self::Identity
                | Self::EffectfulIdentity
                | Self::StagedZeroContribution
                | Self::BadArity
                | Self::ForeignContribution => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Add => {
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone(), input_types[0].clone()])
                }
                Self::Zero(zero) => zero.infer_output_types(input_types),
            }
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Zero(zero) => zero.render(formatter, indentation),
                _ => formatter.write_str(self.name()),
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::EffectfulIdentity => Effects::single(Effect::OrderedIo),
                _ => Effects::PURE,
            }
        }
    }

    impl From<AddOperation> for TestLinearOperation {
        #[inline]
        fn from(_operation: AddOperation) -> Self {
            Self::Add
        }
    }

    impl From<ZeroOperation<DataType>> for TestLinearOperation {
        #[inline]
        fn from(operation: ZeroOperation<DataType>) -> Self {
            Self::Zero(operation)
        }
    }

    impl<'o> TryFrom<&'o TestLinearOperation> for &'o ZeroOperation<DataType> {
        type Error = ();

        #[inline]
        fn try_from(value: &'o TestLinearOperation) -> Result<Self, ()> {
            match value {
                TestLinearOperation::Zero(zero) => Ok(zero),
                _ => Err(()),
            }
        }
    }

    impl<V: Value<Type = DataType>> TransposableOperation<V, TestLinearOperation> for TestLinearOperation {
        fn transpose(
            &self,
            context: &mut TracingContext<V, TestLinearOperation>,
            _inputs: &[PartialValue<Tracer<TracingContext<V, TestLinearOperation>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, TestLinearOperation>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, TestLinearOperation>>>>, DifferentiationError> {
            match self {
                Self::Identity | Self::EffectfulIdentity => {
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(vec![outputs[0].clone()])
                }
                Self::Add => {
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(vec![outputs[0].clone(), outputs[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("output", outputs, 2, ProgramError);
                    assert!(outputs[1].is_zero());
                    Ok(vec![outputs[0].clone()])
                }
                Self::StagedZeroContribution => {
                    check_count!("output", outputs, 1, ProgramError);
                    let zero = {
                        let mut builder = context.builder().borrow_mut();
                        let outputs =
                            builder.add_instruction(Self::Zero(ZeroOperation::new(DataType::F64)), Vec::new())?;
                        check_count!("output", outputs, 1, ProgramError);
                        outputs[0]
                    };
                    Ok(vec![MaybeZero::Value(context.tracer(zero, None))])
                }
                Self::BadArity => {
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(Vec::new())
                }
                Self::ForeignContribution => {
                    check_count!("output", outputs, 1, ProgramError);
                    let foreign_context = TracingContext::<V, TestLinearOperation>::new();
                    Ok(vec![MaybeZero::Value(foreign_context.input(DataType::F64))])
                }
                Self::Zero(_) => {
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(Vec::new())
                }
            }
        }
    }

    #[test]
    fn test_program_transpose() {
        // Test that transposing an identity instruction forwards the output cotangent straight to the input.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Identity, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                in (%0)
            "}
            .trim_end(),
        );

        // Test that repeated uses of one input accumulate their cotangent contributions through a staged `add`.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Add, vec![input, input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(matches!(transposed.instructions()[0].operation(), TestLinearOperation::Add));
        assert_eq!(transposed.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(0)]);
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = add %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test that unused instruction outputs are passed to transpose rules as structural zero cotangents (the
        // `TwoOutputs` rule asserts that its second output cotangent is a structural zero).
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let outputs = builder.add_instruction(TestLinearOperation::TwoOutputs, vec![input]).unwrap().to_vec();
        let program = builder.build::<Scalar, Scalar>(vec![outputs[0]], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(outputs, &[AtomId::new(1), AtomId::new(2)]);
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                in (%0)
            "}
            .trim_end(),
        );

        // Test that a disconnected primal input's cotangent is emitted as an input-free `ZeroOperation` instruction,
        // which is materialized at interpretation time rather than at transpose time.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        builder.add_input(DataType::F64);
        let program = builder.build::<Scalar, ()>(Vec::new(), Placeholder, ()).unwrap();
        let transposed = program.transpose().unwrap();
        assert!(transposed.input_ids().is_empty());
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(0)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = zero [type=f64]
                in (%0)
            "}
            .trim_end(),
        );

        // Test that instructions whose outputs carry no adjoint are skipped in the reverse walk, with the dead input
        // still receiving a zero cotangent output.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let dead_input = builder.add_input(DataType::F64);
        let live_input = builder.add_input(DataType::F64);
        let dead_output = builder.add_instruction(TestLinearOperation::BadArity, vec![dead_input]).unwrap()[0];
        let output = builder.add_instruction(TestLinearOperation::Identity, vec![live_input]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(dead_output, AtomId::new(2));
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1), AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%1, %0)
            "}
            .trim_end(),
        );

        // Test that transposing a program whose values are tracers of an outer trace stays self-contained: the
        // disconnected input's zero is emitted as an instruction in the pullback and nothing is staged into the
        // outer tracing context.
        let tracing_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let outer_builder = tracing_context.builder().clone();
        let mut builder = ProgramBuilder::<TestTracingValue, TestLinearOperation>::new();
        let connected_input = builder.add_input(DataType::F64);
        let disconnected_input = builder.add_input(DataType::F64);
        let program = builder
            .build::<Vec<TestTracingValue>, TestTracingValue>(
                vec![connected_input],
                vec![Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        let pullback = program.transpose().unwrap();
        assert_eq!(disconnected_input, AtomId::new(1));
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(0), AtomId::new(1)]);
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%0, %1)
            "}
            .trim_end(),
        );
        assert!(outer_builder.borrow().atoms().is_empty());
        assert!(outer_builder.borrow().instructions().is_empty());

        // Test that a transpose-rule-staged structural zero contribution stays an input-free `ZeroOperation`
        // instruction in the pullback, again leaving the outer tracing context untouched.
        let tracing_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let outer_builder = tracing_context.builder().clone();
        let mut builder = ProgramBuilder::<TestTracingValue, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::StagedZeroContribution, vec![input]).unwrap()[0];
        let program =
            builder.build::<TestTracingValue, TestTracingValue>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose().unwrap();
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(1)]);
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &DataType::F64,
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero [type=f64]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(outer_builder.borrow().atoms().is_empty());
        assert!(outer_builder.borrow().instructions().is_empty());

        // Test that a transpose rule returning the wrong number of input cotangent contributions is rejected instead
        // of silently dropping cotangents.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::BadArity, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.transpose(),
            Err(DifferentiationError::Program(ProgramError::InvalidInputCount { expected: 1, actual: 0 })),
        ));

        // Test that a cotangent contribution staged in a foreign builder is rejected before its atom ID can alias an
        // unrelated atom in the destination pullback.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::ForeignContribution, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.transpose(),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));

        // Test that an unbound program input atom is reported.
        let input = AtomId::new(0);
        let program = Program::<Scalar, TestLinearOperation, Scalar, ()> {
            atoms: Vec::new(),
            input_ids: vec![input],
            output_ids: Vec::new(),
            instructions: Vec::new(),
            input_structure: Placeholder,
            output_structure: (),
            marker: PhantomData,
        };
        assert!(matches!(
            program.transpose(),
            Err(DifferentiationError::Program(ProgramError::UnboundAtomId { id })) if id == input,
        ));

        // Test that an unbound instruction output atom is reported.
        let input = AtomId::new(0);
        let missing_output = AtomId::new(1);
        let program = Program::<Scalar, TestLinearOperation, Scalar, Scalar> {
            atoms: vec![Atom::Variable(DataType::F64)],
            input_ids: vec![input],
            output_ids: vec![input],
            instructions: vec![Instruction::new(TestLinearOperation::Identity, vec![input], vec![missing_output])],
            input_structure: Placeholder,
            output_structure: Placeholder,
            marker: PhantomData,
        };
        assert!(matches!(
            program.transpose(),
            Err(DifferentiationError::Program(ProgramError::UnboundAtomId { id })) if id == missing_output,
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to() {
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Add, vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        // Test that the pullback's cotangent outputs follow the requested index order rather than program-input
        // order: both inputs of the `add` receive the seeded output cotangent, so the two orders are distinguishable
        // only through the output ordering.
        let forward = program.transpose_with_respect_to(&[0, 1]).unwrap();
        let reversed = program.transpose_with_respect_to(&[1, 0]).unwrap();
        assert_eq!(forward.output_ids().len(), 2);
        assert_eq!(reversed.output_ids().len(), 2);
        assert_eq!(
            reversed.output_ids(),
            &[forward.output_ids()[1], forward.output_ids()[0]],
            "requested index order must permute the pullback outputs",
        );

        // Test that out-of-range and duplicate input indices are rejected.
        assert!(matches!(
            program.transpose_with_respect_to(&[2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 2 is out of range for a program with 2 input(s)",
        ));
        assert!(matches!(
            program.transpose_with_respect_to(&[1, 1]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 1 appears more than once",
        ));

        // A live transpose rule may need a pure value produced entirely from known inputs. For `f(a, x) = (a², a²x)`,
        // transposing only with respect to `x` must replay `a²` in the pullback, ignore the cotangent supplied for the
        // non-linear `a²` output, and produce `d_x = d_product · a²`.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let known = builder.add_input(DataType::F64);
        let linear = builder.add_input(DataType::F64);
        let known_square = builder.add_instruction(MulOperation, vec![known, known]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![known_square, linear]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![known_square, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1]).unwrap();
        let outputs = pullback.interpret(vec![Scalar::F64(100.0), Scalar::F64(2.0), Scalar::F64(3.0)]).unwrap();
        assert_eq!(outputs, vec![Scalar::F64(18.0)]);

        // Two live transpose rules that demand the same pure known intermediate must share one rematerialized producer.
        // The pullback contains one `identity`, not one copy per `add` consumer, and both linear inputs still receive
        // their corresponding output cotangents.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let known = builder.add_input(DataType::F64);
        let first_linear = builder.add_input(DataType::F64);
        let second_linear = builder.add_input(DataType::F64);
        let known_intermediate = builder.add_instruction(TestLinearOperation::Identity, vec![known]).unwrap()[0];
        let first_output =
            builder.add_instruction(TestLinearOperation::Add, vec![known_intermediate, first_linear]).unwrap()[0];
        let second_output =
            builder.add_instruction(TestLinearOperation::Add, vec![known_intermediate, second_linear]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![first_output, second_output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1, 2]).unwrap();
        assert_eq!(
            pullback
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), TestLinearOperation::Identity))
                .count(),
            1,
            "a shared pure known producer must be replayed exactly once",
        );
        assert_eq!(
            pullback.output_ids(),
            &pullback.input_ids()[..2],
            "each linear input must receive its corresponding output cotangent",
        );

        // Replaying a known producer with observable effects in the pullback could duplicate or reorder that effect,
        // so the partition-aware transpose must require partial evaluation to residualize the value instead.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let known = builder.add_input(DataType::F64);
        let linear = builder.add_input(DataType::F64);
        let known_intermediate =
            builder.add_instruction(TestLinearOperation::EffectfulIdentity, vec![known]).unwrap()[0];
        let output = builder.add_instruction(TestLinearOperation::Add, vec![known_intermediate, linear]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Scalar>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[1]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message.contains("cannot replay effectful known intermediate producer `effectful_identity`"),
        ));
    }

    #[test]
    fn test_program_transpose_deep_known_chain_iteratively() {
        // This chain is intentionally much deeper than realistic scalar code. Materializing its tail exercises the
        // explicit postorder work stack and would make a recursive implementation consume one native stack frame per
        // producer. Keep the assertion structural so the test characterizes transformation behavior independently of
        // any interpretation backend.
        const CHAIN_LENGTH: usize = 10_000;

        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let known = builder.add_input(DataType::F64);
        let linear = builder.add_input(DataType::F64);
        let mut known_intermediate = known;
        for _ in 0..CHAIN_LENGTH {
            known_intermediate =
                builder.add_instruction(TestLinearOperation::Identity, vec![known_intermediate]).unwrap()[0];
        }
        let output = builder.add_instruction(TestLinearOperation::Add, vec![known_intermediate, linear]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Scalar>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1]).unwrap();
        assert_eq!(
            pullback
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), TestLinearOperation::Identity))
                .count(),
            CHAIN_LENGTH,
        );
    }

    #[test]
    fn test_vjp() {
        // `ReverseModeDifferentiate::vjp` on an explicit context linearizes and transposes: for `f(x) = sin(x)` at
        // `x = 2` the primal output is `sin(2)`, and the returned pullback maps any number of output cotangents back
        // through the transposed Jacobian without re-tracing or re-differentiating.
        let (value, pullback) =
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new().vjp(|x| x.sin(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pullback.apply(Scalar::from(1.0)).unwrap(), 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(pullback.apply(Scalar::from(3.0)).unwrap(), 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // The free `vjp` serves top-level concrete values through their `Value::ExecutionDomain` declarations: a
        // plain `Scalar` input recovers the eager scalar domain.
        let (value, pullback) = vjp(|x| x.sin(), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pullback.apply(Scalar::from(1.0)).unwrap(), 2.0f64.cos(), epsilon = 1e-9);

        // Under an active trace, the free `vjp` recovers the staging context from its tracer input instead, so the
        // primal work and the pullback replay both stage into the enclosing trace.
        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, pullback) = vjp(|x| x.sin(), inputs[0].clone())?;
                let cotangent = pullback.apply(inputs[1].clone())?;
                Ok(vec![value, cotangent])
            },
            vec![DataType::F64, DataType::F64],
        )
        .unwrap();
        let outputs = program.interpret(vec![Scalar::from(2.0), Scalar::from(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // With no leaf value to recover a context from, the free `vjp` reports an invalid input count.
        let error = vjp(
            |x: Vec<LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>| Ok(x),
            Vec::<Scalar>::new(),
        )
        .map(|(outputs, _)| outputs)
        .unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_value_and_gradient() {
        // `ReverseModeDifferentiate::value_and_gradient` on an explicit context: `f(x, y) = x * y + x` has value `8`
        // and gradient `(y + 1, x) = (4, 2)` at `(2, 3)`, reshaped into the closure's input structure.
        let (value, gradient): (Scalar, (Scalar, Scalar)) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .value_and_gradient(|(x, y)| x.clone() * y + x, (Scalar::from(2.0), Scalar::from(3.0)))
            .unwrap();
        assert_abs_diff_eq!(value, 8.0, epsilon = 1e-9);
        assert_eq!(gradient, (Scalar::from(4.0), Scalar::from(2.0)));

        // The free `value_and_gradient` recovers the eager domain from the concrete primals.
        let (value, gradient) = value_and_gradient(|x| x.clone() * x.sin().unwrap(), Scalar::from(0.7)).unwrap();
        assert_abs_diff_eq!(value, 0.7 * 0.7f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);

        // Under an active trace, the free `value_and_gradient` recovers the staging context from its tracer input
        // instead, so the primal work and the pullback replay both stage into the enclosing trace.
        let (_, program) = EagerContext::<Scalar, ScalarOperation<Scalar>>::trace(
            |inputs: Vec<_>| {
                let (value, gradient) = value_and_gradient(|x| x.sin().unwrap(), inputs[0].clone()).unwrap();
                Ok(vec![value, gradient])
            },
            vec![DataType::F64],
        )
        .unwrap();
        let outputs = program.interpret(vec![Scalar::from(2.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1], 2.0f64.cos(), epsilon = 1e-9);

        // JAX-parity marquee behavior: the closure can branch on a *primal* with host control flow, because the duals'
        // primal halves carry concrete known values under an eager context (exactly like branching on concrete primals
        // under JAX's `grad`). For `x = 3` the predicate is true, so `f(x) = x * x` with gradient `2x = 6`, and the
        // untaken `sin(x)` branch is never traced at all.
        let (value, gradient) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .value_and_gradient(
                |x| if x.boolean().unwrap() { x.clone() * x } else { x.sin().unwrap() },
                Scalar::from(3.0),
            )
            .unwrap();
        assert_abs_diff_eq!(value, 9.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 6.0, epsilon = 1e-9);

        // The closure is invoked exactly once: a single linearizing replay produces both the value and the gradient.
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::F64);
        let calls = Cell::new(0);
        let (_, gradient): (
            DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>,
            Vec<DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>,
        ) = context
            .value_and_gradient(
                |inputs| {
                    calls.set(calls.get() + 1);
                    inputs[0].clone() * inputs[0].clone()
                },
                vec![primal],
            )
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(gradient.len(), 1);

        // Mixing tracers of two different traces is rejected with `MismatchedProgramBuilders`. The closure runs on
        // differentiation duals whose operator sugar has no deferral point of its own, so the partial-evaluation
        // context defers the failed bind by poisoning its outputs, and the original error surfaces as a plain `Err`
        // at the evaluation boundary.
        let foreign_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::F64);
        let foreign_primal = foreign_context.input(DataType::F64);
        let result =
            context.value_and_gradient(|inputs| inputs[0].clone() + inputs[1].clone(), vec![primal, foreign_primal]);
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));

        // A complex scalar output is rejected toward the holomorphic entry points, and inputs with no leaf values
        // report an invalid input count.
        let z = Complex::new(0.7f64, -0.3f64);
        let error = value_and_gradient(|x| x.clone() * x, Scalar::from(z)).unwrap_err();
        assert!(matches!(error, DifferentiationError::ComplexGradientOutput { .. }));
        let error = value_and_gradient(
            |x: Vec<LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>| x.into_iter().next().unwrap(),
            Vec::<Scalar>::new(),
        )
        .unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_gradient() {
        // `ReverseModeDifferentiate::gradient` is the gradient-only counterpart of `value_and_gradient`.
        let method_gradient: (Scalar, Scalar) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .gradient(|(x, y)| x.clone() * y + x, (Scalar::from(2.0), Scalar::from(3.0)))
            .unwrap();
        assert_eq!(method_gradient, (Scalar::from(4.0), Scalar::from(2.0)));

        // The free `gradient` recovers the eager domain from the concrete primal and agrees with the value-carrying
        // form.
        let free_gradient = gradient(|x| x.clone() * x.sin().unwrap(), Scalar::from(0.7)).unwrap();
        assert_abs_diff_eq!(free_gradient, 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);

        // With no leaf value to recover a context from, the free `gradient` reports an invalid input count.
        let error = gradient(
            |x: Vec<LinearizationTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>>| x.into_iter().next().unwrap(),
            Vec::<Scalar>::new(),
        )
        .unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_value_and_gradient_holomorphic() {
        // `ReverseModeDifferentiate::value_and_gradient_holomorphic` on an explicit context recovers the complex
        // derivative under the holomorphy promise: `∂z²/∂z = 2z` at a genuinely complex point.
        let z = Complex::new(0.7f64, -0.3f64);
        let (value, gradient) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .value_and_gradient_holomorphic(|x| x.clone() * x, Scalar::from(z))
            .unwrap();
        assert_eq!(value, Scalar::from(z * z));
        assert_eq!(gradient, Scalar::from(z + z));

        // The free form recovers the eager domain from the concrete primal, and for real outputs the holomorphy
        // promise changes nothing.
        let (value, gradient) = value_and_gradient_holomorphic(|x| x.clone() * x, Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(value, 4.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient, 4.0, epsilon = 1e-9);

        // Under an active trace the guards run at the type level (the identity closure performs no complex arithmetic).
        // The plain entry point rejects a complex output toward the holomorphic one, which accepts it and seeds `one`
        // at the complex cotangent type, while a real output flows through the holomorphic entry point exactly like
        // the plain one.
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::C64);
        let result = context.value_and_gradient(|inputs: Vec<_>| inputs[0].clone(), vec![primal]);
        assert!(matches!(
            result,
            Err(DifferentiationError::ComplexGradientOutput { output_type }) if output_type == "c64",
        ));
        let primal = context.input(DataType::C64);
        let (value, gradient) =
            context.value_and_gradient_holomorphic(|inputs: Vec<_>| inputs[0].clone(), vec![primal]).unwrap();
        assert_eq!(*value.r#type(), DataType::C64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::C64);
        let primal = context.input(DataType::F64);
        let (value, gradient) =
            context.value_and_gradient_holomorphic(|inputs: Vec<_>| inputs[0].clone(), vec![primal]).unwrap();
        assert_eq!(*value.r#type(), DataType::F64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::F64);
    }

    #[test]
    fn test_gradient_holomorphic() {
        // `ReverseModeDifferentiate::gradient_holomorphic` is the gradient-only counterpart of
        // `value_and_gradient_holomorphic`: `∂sin(z)/∂z = cos(z)` at a genuinely complex point.
        let z = Complex::new(0.7f64, -0.3f64);
        let method_gradient = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .gradient_holomorphic(|x| x.sin().unwrap(), Scalar::from(z))
            .unwrap();
        assert_eq!(method_gradient, Scalar::from(z.cos()));

        // The free form recovers the eager domain from the concrete primal and agrees.
        let free_gradient = gradient_holomorphic(|x| x.sin().unwrap(), Scalar::from(z)).unwrap();
        assert_eq!(free_gradient, Scalar::from(z.cos()));
    }

    #[test]
    fn test_value_and_gradient_with_aux() {
        // `ReverseModeDifferentiate::value_and_gradient_with_aux` on an explicit context returns the auxiliary
        // outputs as ordinary primal values seeded with zero cotangents, so they do not contribute to the gradient.
        let ((value, aux), gradient): ((Scalar, Scalar), (Scalar, Scalar)) = EagerContext::<
            Scalar,
            ScalarOperation<Scalar>,
        >::new()
        .value_and_gradient_with_aux(|(x, y)| (x.clone() * y.clone(), x + y), (Scalar::from(2.0), Scalar::from(3.0)))
        .unwrap();
        assert_abs_diff_eq!(value, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(aux, 5.0, epsilon = 1e-9);
        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));

        // The free form recovers the eager domain from the concrete primals, and the auxiliary structure can carry
        // multiple leaves (each rides along as a primal value with a zero cotangent seed, so none contributes to the
        // gradient of `x * y`).
        let ((value, aux), gradient): ((Scalar, (Scalar, Scalar)), (Scalar, Scalar)) = value_and_gradient_with_aux(
            |(x, y)| {
                let value = x.clone() * y.clone();
                let aux = (x.clone() + y, x.clone() * x);
                (value, aux)
            },
            (Scalar::from(2.0), Scalar::from(3.0)),
        )
        .unwrap();
        assert_abs_diff_eq!(value, 6.0, epsilon = 1e-9);
        assert_eq!(aux, (Scalar::from(5.0), Scalar::from(4.0)));
        assert_eq!(gradient, (Scalar::from(3.0), Scalar::from(2.0)));
    }

    #[test]
    fn test_gradient_with_aux() {
        // `ReverseModeDifferentiate::gradient_with_aux` is the gradient-only counterpart
        // of `value_and_gradient_with_aux`, returning `(gradient, aux)`.
        let (method_gradient, aux): ((Scalar, Scalar), Scalar) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .gradient_with_aux(|(x, y)| (x.clone() * y.clone(), x + y), (Scalar::from(2.0), Scalar::from(3.0)))
            .unwrap();
        assert_eq!(method_gradient, (Scalar::from(3.0), Scalar::from(2.0)));
        assert_abs_diff_eq!(aux, 5.0, epsilon = 1e-9);

        // The free form recovers the eager domain from the concrete primals and agrees.
        let (free_gradient, aux): ((Scalar, Scalar), Scalar) =
            gradient_with_aux(|(x, y)| (x.clone() * y.clone(), x + y), (Scalar::from(2.0), Scalar::from(3.0))).unwrap();
        assert_eq!(free_gradient, (Scalar::from(3.0), Scalar::from(2.0)));
        assert_abs_diff_eq!(aux, 5.0, epsilon = 1e-9);
    }

    #[test]
    fn test_value_and_gradient_holomorphic_with_aux() {
        // `ReverseModeDifferentiate::value_and_gradient_holomorphic_with_aux` combines the holomorphy promise with
        // auxiliary outputs: the gradient is `∂z²/∂z = 2z` while the auxiliary value rides along with a zero
        // cotangent seed.
        let z = Complex::new(0.7f64, -0.3f64);
        let ((value, aux), gradient): ((Scalar, Scalar), Scalar) =
            EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
                .value_and_gradient_holomorphic_with_aux(|x| (x.clone() * x.clone(), x), Scalar::from(z))
                .unwrap();
        assert_eq!(value, Scalar::from(z * z));
        assert_eq!(aux, Scalar::from(z));
        assert_eq!(gradient, Scalar::from(z + z));

        // The free form recovers the eager domain from the concrete primal and agrees.
        let ((value, aux), gradient): ((Scalar, Scalar), Scalar) =
            value_and_gradient_holomorphic_with_aux(|x| (x.clone() * x.clone(), x), Scalar::from(z)).unwrap();
        assert_eq!(value, Scalar::from(z * z));
        assert_eq!(aux, Scalar::from(z));
        assert_eq!(gradient, Scalar::from(z + z));

        // The holomorphy gate also runs at the type level under an active trace. A complex output with
        // an auxiliary output is accepted end to end and seeds `one` at the complex cotangent type.
        type TestTracer = DomainTracer<EagerContext<Scalar, ScalarOperation<Scalar>>>;
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let primal = context.input(DataType::C64);
        let ((value, aux), gradient): ((TestTracer, TestTracer), Vec<TestTracer>) = context
            .value_and_gradient_holomorphic_with_aux(
                |inputs: Vec<_>| (inputs[0].clone(), inputs[0].clone()),
                vec![primal],
            )
            .unwrap();
        assert_eq!(*value.r#type(), DataType::C64);
        assert_eq!(*aux.r#type(), DataType::C64);
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), DataType::C64);
    }

    #[test]
    fn test_gradient_holomorphic_with_aux() {
        // `ReverseModeDifferentiate::gradient_holomorphic_with_aux` is the gradient-only counterpart of
        // `value_and_gradient_holomorphic_with_aux`, returning `(gradient, aux)`.
        let z = Complex::new(0.7f64, -0.3f64);
        let (method_gradient, aux): (Scalar, Scalar) = EagerContext::<Scalar, ScalarOperation<Scalar>>::new()
            .gradient_holomorphic_with_aux(|x| (x.clone() * x.clone(), x), Scalar::from(z))
            .unwrap();
        assert_eq!(method_gradient, Scalar::from(z + z));
        assert_eq!(aux, Scalar::from(z));

        // The free form recovers the eager domain from the concrete primal and agrees.
        let (free_gradient, aux): (Scalar, Scalar) =
            gradient_holomorphic_with_aux(|x| (x.clone() * x.clone(), x), Scalar::from(z)).unwrap();
        assert_eq!(free_gradient, Scalar::from(z + z));
        assert_eq!(aux, Scalar::from(z));
    }

    #[test]
    fn test_nested_differentiation() {
        // Every nesting shape differentiates `f(x) = sin(x²)` at `x = 0.7` through closure-level nesting. Inner
        // transforms run on the nested tracing context their tracers flow in, recovered either implicitly by the
        // free differentiation functions from their tracer inputs or explicitly through `x.context()`. Every closure
        // is fallible, propagating staging failures outward through `?` (or by returning the inner transform's
        // `Result` directly) instead of unwrapping.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let x: f64 = 0.7;

        // Reverse-over-reverse through the free functions alone: the outer value is `f'(x) = 2x cos(x²)` and the
        // outer gradient is the analytic second derivative `f''(x) = 2 cos(x²) - 4x² sin(x²)`.
        let (value, second_derivative) =
            value_and_gradient(|x| gradient(|y| (y.clone() * y).sin(), x), Scalar::from(x)).unwrap();
        assert_abs_diff_eq!(value, 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(second_derivative, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(), epsilon = 1e-9);

        // Three levels of nesting exercise the recursive `NestedTracingContext<NestedTracingContext<...>>` types
        // through the trait solver, with every inner transform run through an explicitly recovered context receiver.
        // The outer gradient is the analytic third derivative `f'''(x) = -12x sin(x²) - 8x³ cos(x²)`.
        let (value, third_derivative) = domain
            .value_and_gradient(
                |x| {
                    let context = x.context().clone();
                    context.gradient(
                        |y| {
                            let context = y.context().clone();
                            context.gradient(|z| (z.clone() * z).sin(), y)
                        },
                        x,
                    )
                },
                Scalar::from(x),
            )
            .unwrap();
        assert_abs_diff_eq!(value, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            third_derivative,
            -12.0 * x * (x * x).sin() - 8.0 * x * x * x * (x * x).cos(),
            epsilon = 1e-9,
        );

        // Forward-over-reverse through the free functions alone: pushing the tangent `v = 2` through the gradient
        // computes the Hessian-vector product `f''(x) · v` without materializing a dense Hessian, because the `jvp`
        // duals' stamped `DifferentiationContext` is itself a `ReverseModeDifferentiate` the inner transform nests
        // on.
        let (primal, tangent) =
            jvp(|x| Ok(gradient(|y| (y.clone() * y).sin(), x)?), Scalar::from(x), Scalar::from(2.0)).unwrap();
        assert_abs_diff_eq!(primal, 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(tangent, 2.0 * (2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin()), epsilon = 1e-9);
    }
}
