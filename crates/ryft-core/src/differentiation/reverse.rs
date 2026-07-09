use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::DifferentiableType;
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::ZeroOperation;
use crate::parameters::{Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::PartialValue;
use crate::programs::{Atom, AtomId, MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::Typed;

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
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, ProgramError>;
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
    ) -> Result<Program<V, Self, Vec<V>, Vec<V>>, ProgramError>;
}

impl<
    T: DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O> + From<ZeroOperation<T>> + From<AddOperation>,
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
    pub fn transpose(&self) -> Result<Program<V, O, Output, Input>, ProgramError> {
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
    /// Known operand values are program inputs and constants. Each known input is exposed as a pullback input and each
    /// constant atom as a pullback constant, so a bilinear rule can read either as the known operand's pullback value.
    /// If a transpose rule requests the value of a known *intermediate* (a known atom that is neither a program input
    /// nor a constant), this function returns [`ProgramError::UnsupportedOperation`]. This is not a limitation in
    /// practice because the partial-evaluation split that produces a partitioned tangent program prunes known
    /// intermediates into its known sub-program, leaving only known inputs and constants for an adjoint rule to read.
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
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
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
                });
            }
            if input_linearity[index] {
                return Err(ProgramError::InvalidArgument {
                    message: format!("transposition input index {index} appears more than once"),
                });
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
                return Err(ProgramError::UnboundAtomId { id: contribution }.into());
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
            accumulate::<V, O>(&builder, adjoints.as_mut_slice(), output, cotangent_input)?;
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

        // Constant atoms are also known operands, so expose each one to transpose rules as a pullback constant. A
        // bilinear operation such as `Mul` whose known operand is a constant (for example, the `3` in `3 * x`) reads
        // this pullback atom exactly as it reads a known input's value. This is the partition-aware analogue of folding
        // a rebuilt constant into a captured factor, and it keeps a constant-scaled tangent transposable directly
        // rather than reporting it as an unsupported known intermediate.
        for (index, atom) in self.atoms().iter().enumerate() {
            if let Some(value) = atom.as_constant() {
                known_map[index] = Some(builder.borrow_mut().add_constant(value.clone()));
            }
        }

        // Walk the primal program backward, applying each operation's transpose rule only when at least one of its
        // outputs has a non-zero accumulated cotangent. The scratch vector avoids allocating a fresh cotangent vector
        // for every live instruction.
        let max_instruction_output_count =
            self.instructions().iter().map(|instruction| instruction.outputs().len()).max().unwrap_or(0);
        let mut instruction_output_cotangents = Vec::with_capacity(max_instruction_output_count);
        for instruction in self.instructions().iter().rev() {
            // Skip dead reverse edges early. If none of an instruction's outputs carries an adjoint, the instruction
            // cannot contribute to any input cotangent. This is the only operand-side guard. A non-linear instruction
            // whose transpose rule would read a known operand is always safe because the partial-evaluation split that
            // produces partitioned tangent programs exposes every known operand's value (as a known input or constant),
            // so the known-intermediate guard below stays purely defensive. Note that a batched masked `while` threads
            // its structurally-zero Boolean-mask carry as a plain pushforward tangent rather than through an all-known
            // `select` operation over a restored zero, and so no such instruction reads an unexposed known
            // intermediate value.
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
            // pullback value atom from `known_map` (a known program input or a constant). A known operand with no
            // pullback value would be a known *intermediate* (i.e., a known atom that is neither a program input nor a
            // constant). The partial-evaluation split that produces partitioned tangent programs never leaves one, and
            // so guarding it here once lets every rule assume a `Known` operand carries its value.
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
                        match known_map.get(input.index()).copied().ok_or(ProgramError::UnboundAtomId { id: input })? {
                            Some(atom) => Ok(PartialValue::Known(context.tracer(atom, Some(r#type)))),
                            None => Err(ProgramError::UnsupportedOperation {
                                message: "partition-aware transpose of a known intermediate is not yet supported"
                                    .to_string(),
                            }),
                        }
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
            Err(_) => return Err(ProgramError::EscapedProgramBuilder),
        };
        builder.build(outputs, vec![Placeholder; pullback_input_count], vec![Placeholder; pullback_output_count])
    }
}

#[cfg(test)]
mod tests {
    use std::marker::PhantomData;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::EagerContext;
    use crate::contexts::StagingContext;
    use crate::macros::check_count;
    use crate::operations::Operation;
    use crate::operations::arithmetic::AddOperation;
    use crate::operations::constants::ZeroOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{Atom, AtomId, Instruction, MaybeZero, Program, ProgramBuilder, ProgramError, Value};
    use crate::scalars::Scalar;
    use crate::tracing::{DomainTracer, DomainTracingContext, Tracer, TracingContext};
    use crate::types::{DataType, TypeError};

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
                Self::Identity | Self::StagedZeroContribution | Self::BadArity | Self::ForeignContribution => {
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
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, TestLinearOperation>>>>, ProgramError> {
            match self {
                Self::Identity => {
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
        assert!(matches!(program.transpose(), Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),));

        // Test that a cotangent contribution staged in a foreign builder is rejected before its atom ID can alias an
        // unrelated atom in the destination pullback.
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::ForeignContribution, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(ProgramError::MismatchedProgramBuilders),));

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
        assert!(matches!(program.transpose(), Err(ProgramError::UnboundAtomId { id }) if id == input));

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
            Err(ProgramError::UnboundAtomId { id }) if id == missing_output,
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
            Err(ProgramError::InvalidArgument { message })
                if message == "transposition input index 2 is out of range for a program with 2 input(s)",
        ));
        assert!(matches!(
            program.transpose_with_respect_to(&[1, 1]),
            Err(ProgramError::InvalidArgument { message })
                if message == "transposition input index 1 appears more than once",
        ));
    }
}
