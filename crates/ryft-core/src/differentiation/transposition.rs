use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::contexts::StagingContext;
use crate::differentiation::DifferentiableType;
use crate::macros::{check_builders, check_count};
use crate::operations::Operation;
use crate::operations::arithmetic::AddOperation;
use crate::operations::constants::ZeroOperation;
use crate::parameters::{Parameterized, Placeholder};
use crate::partial::PartialValue;
use crate::programs::{Atom, AtomId, MaybeZero, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::{Tracer, TracingContext};
use crate::types::{Type, Typed};

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

// TODO(eaplatanios): Review from here onwards.

/// Represents closed [`Operation`] families whose flat linear [`Program`]s can be transposed as nested programs.
/// Higher-order transpose rules, such as the rules for linear condition branches and linear scan bodies, need to
/// transpose captured programs whose operation family is the same closed enum that is currently being proven
/// transposable. Writing that need directly as `O: TransposableOperation<V, O>` at every recursive payload boundary
/// can send Rust's trait solver through the enum's higher-order variants indefinitely. [`TransposableProgramOperation`]
/// names the recursive fixed point once: the closed operation enum implements this trait by calling
/// [`Program::transpose_with_respect_to`], while higher-order payloads depend on this semantic witness instead of
/// reproducing all variant-level transposition bounds.
///
/// The trait is intentionally about complete operation families, not individual primitive payloads. Implementations
/// that delegate to [`Program::transpose_with_respect_to`] add that method's [`Zero`](crate::Zero)/
/// [`Add`](std::ops::Add) bounds locally because those are requirements of the standard implementation strategy,
/// not of this semantic witness itself.
pub trait TransposableProgramOperation<V: Value>: Operation<V::Type> + Sized
where
    V::Type: DifferentiableType,
{
    /// Transposes the provided flat [`Program`] in this operation family with respect to the inputs flagged in
    /// `input_linearity`; refer to the documentation of [`Program::transpose_with_respect_to`] for the pullback's
    /// input and output layout. The operand-form higher-order transpose rules (condition branches and scan bodies
    /// whose known residual factors ride as ordinary operands) pass their genuine linearity masks, while the fully
    /// linear captured-program rules pass an all-`true` mask — the flat-transposition case is exactly the partial
    /// one with every input linear, so this single method serves both. The mask encoding (rather than the public
    /// entry point's index list) is kept here because it is the seed of the forward linearity propagation and the
    /// natural per-operand form for the higher-order rules.
    ///
    /// # Parameters
    ///
    ///   - `program`: Flat [`Program`] whose inputs and outputs are flattened vectors of values.
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
    /// Transposes this linear pushforward [`Program`] into its reverse-mode pullback. This is the main entrypoint for
    /// transposing linear [`Program`]s. In the algebraic sense, _transposing_ a linear map `L: X -> Y` gives a map on
    /// _dual_ spaces `L^T: Y* -> X*`. In finite dimensions this is the same operation represented by a matrix
    /// transpose. Here the linear map is not stored as a matrix. It is a staged [`Program`] that maps input tangents
    /// to output tangents, and transposition builds the dual program that maps output cotangents back to input
    /// cotangents. Operationally, transposition creates cotangent inputs for this program's outputs, walks the
    /// instructions in reverse order, and applies each primitive operation's [`TransposableOperation::transpose`] rule
    /// to accumulate cotangent contributions for the original inputs. This is the same decomposition of reverse-mode
    /// automatic differentiation as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`]s, which the value type's [`Zero`](crate::Zero)
    /// implementation evaluates at interpretation time. For linear programs whose values are [`Tracer`]s from an outer
    /// trace, use [`TracingContext::transpose_traced`] instead so that those disconnected-input zeros can be
    /// materialized in the surrounding tracing context.
    ///
    /// This is the fully linear case of [`transpose_with_respect_to`](Self::transpose_with_respect_to): the program
    /// is transposed with respect to every input, so every reachable [`Atom`] is linear, each operation's transpose
    /// rule receives an all-`true` operand-linearity slice, and the pullback's inputs and outputs preserve this
    /// program's output and input structures respectively.
    #[inline]
    pub fn transpose(&self) -> Result<Program<V, O, Output, Input>, ProgramError> {
        let flat = TracingContext::<V, O>::new().transpose(self, &vec![true; self.input_ids().len()])?;
        // Every input is linear, so the pullback has one cotangent input per primal output and one cotangent output per
        // primal input. Recover the structured form by reattaching this program's output and input structures to the
        // flat pullback, keeping its atoms, instructions, and input/output `AtomId`s unchanged.
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

    /// Transposes this linear pushforward [`Program`] into its reverse-mode pullback **with respect to** the inputs
    /// selected by `input_indices`, holding the remaining inputs as known (constant) parameters of the linear map.
    /// The program must be linear in the selected inputs; it may depend arbitrarily on the known ones. This is the
    /// partial entry point behind the fully linear [`transpose`](Self::transpose).
    ///
    /// Linearity is propagated forward from the program inputs: a program-input [`Atom`] is linear exactly when its
    /// index appears in `input_indices`, constant atoms are always known, and an operation result is linear when any
    /// of its operands is linear. Each operation's [`transpose`](TransposableOperation::transpose) rule receives the
    /// per-operand linearity knowledge derived from this propagation.
    ///
    /// The pullback's inputs are the cotangents of this program's outputs followed by the runtime values of the known
    /// inputs (in program-input order), and the pullback's outputs are the accumulated cotangents of the selected
    /// inputs, **in `input_indices` order**; known inputs receive no cotangent output. Because this layout depends on
    /// `input_indices`, the pullback's inputs and outputs are returned as flat [`Vec`]s rather than reusing this
    /// program's structured input and output types. The fully linear [`transpose`](Self::transpose) recovers the
    /// structured form. Disconnected selected inputs are emitted as [`ZeroOperation`]s, exactly as in
    /// [`transpose`](Self::transpose).
    ///
    /// Known operand values are program inputs and constants: each known input is exposed as a pullback input and each
    /// constant atom as a pullback constant, so a bilinear rule can read either as the known operand's pullback value.
    /// If a transpose rule requests the value of a known *intermediate* (a known atom that is neither a program input
    /// nor a constant), this returns [`ProgramError::UnsupportedOperation`]. This is not yet a limitation in practice
    /// because the partial-evaluation split that produces a partitioned tangent program prunes known intermediates into
    /// its known sub-program, leaving only known inputs and constants for an adjoint rule to read.
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
        let mut pullback = TracingContext::<V, O>::new().transpose(self, input_linearity.as_slice())?;

        // The reverse walk emits one cotangent output per selected input in program-input order; permute them so the
        // pullback's outputs follow the order of `input_indices` instead. The walk position of index `i` is the
        // number of selected indices smaller than `i`.
        let mut sorted_indices = input_indices.to_vec();
        sorted_indices.sort_unstable();
        pullback.output_ids = input_indices
            .iter()
            .map(|index| pullback.output_ids[sorted_indices.binary_search(index).unwrap()])
            .collect();
        Ok(pullback)
    }
}

impl<T: Type + DifferentiableType, V: Value<Type = T>, O: Operation<T>, Capture> TracingContext<V, O, Capture> {
    /// Transposes the provided traced linear [`Program`] whose values are [`Tracer`]s belonging to this outer
    /// [`TracingContext`]. This uses the same reverse-walk implementation as [`Program::transpose`] in a fresh
    /// [`TracingContext`].
    ///
    /// The transposed program's values are this context's own [`Tracer`]s, and its operation family `LinearOperation`
    /// is the linear operation family of the program being transposed (which need not equal this context's own
    /// operation family).
    ///
    /// Use this method when transposing a linear program inside an outer trace, such as when staging a traced
    /// reverse-mode pullback. Use [`Program::transpose`] for ordinary complete linear program transposition, and use
    /// [`TracingContext::transpose`] only when you already hold a [`TracingContext`] to consume and want to run the
    /// lower-level transposition algorithm directly.
    ///
    /// Disconnected primal inputs and transpose-rule-staged structural zeros are emitted as input-free
    /// [`ZeroOperation`] instructions in the pullback. These are materialized at interpretation time: a pullback
    /// [`ZeroOperation`] interpreted over outer-trace [`Tracer`]s stages a typed zero into the surrounding
    /// [`TracingContext`] through the threaded interpretation context, and so backends whose traced constants are
    /// abstract metadata do not need to materialize a runtime value just to transpose an enclosing traced program.
    #[inline]
    pub fn transpose_traced<
        Input: Parameterized<Tracer<Self>>,
        Output: Parameterized<Tracer<Self>>,
        LinearOperation: TransposableOperation<Tracer<Self>, LinearOperation> + From<ZeroOperation<T>> + From<AddOperation>,
    >(
        &self,
        program: &Program<Tracer<Self>, LinearOperation, Input, Output>,
    ) -> Result<Program<Tracer<Self>, LinearOperation, Output, Input>, ProgramError> {
        let flat = TracingContext::<Tracer<Self>, LinearOperation>::new()
            .transpose(program, &vec![true; program.input_ids().len()])?;
        // Every input is linear, so the flat pullback has one cotangent input per primal output and one cotangent
        // output per primal input. Recover the structured form by reattaching this program's output and input
        // structures to the flat pullback, keeping its atoms, instructions, and input/output `AtomId`s unchanged.
        Ok(Program {
            atoms: flat.atoms,
            input_ids: flat.input_ids,
            output_ids: flat.output_ids,
            instructions: flat.instructions,
            input_structure: program.output_structure().clone(),
            output_structure: program.input_structure().clone(),
            marker: PhantomData,
        })
    }
}

impl<
    T: Type + DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O> + From<ZeroOperation<T>> + From<AddOperation>,
> TracingContext<V, O>
{
    /// Transposes the provided linear [`Program`] using this [`TracingContext`]'s [`ProgramBuilder`]. This is the
    /// builder-level implementation behind [`Program::transpose`]. Refer to the documentation of [`Program::transpose`]
    /// for the conceptual relationship between program transposition, algebraic transposition, pushforward functions,
    /// and pullback functions. This function is for callers that hold a [`TracingContext`] to consume as the
    /// destination for the pullback.
    ///
    /// This function uses the context's [`builder`](Self::builder) as the destination for the transposed program,
    /// records cotangent inputs for the primal outputs, walks `program` in reverse instruction order, and transposes
    /// each [`Instruction`](crate::programs::Instruction) using [`TransposableOperation::transpose`]. It then consumes
    /// the context, taking sole ownership of its builder to build the pullback. A transpose rule that needs to
    /// transpose a nested subprogram (e.g., a captured control-flow branch) should instead call [`Program::transpose`],
    /// which transposes it in its own fresh context.
    ///
    /// `input_linearity` carries one linearity flag per program input. Linearity is propagated forward to every [`Atom`]
    /// so that each instruction's transpose rule receives a per-operand linearity slice. Callers that transpose a fully
    /// linear program pass an all-`true` mask, which keeps every reachable atom linear.
    ///
    /// The pullback's inputs are the cotangents of the primal outputs followed by the runtime values of the known
    /// (non-linear) inputs, and its outputs are the accumulated cotangents of the linear inputs only. Because this
    /// layout depends on `input_linearity`, the pullback is returned with flat [`Vec`] input and output structures;
    /// callers that know every input is linear recover the structured form by reattaching the program's input and
    /// output structures to the flat pullback. Known inputs are exposed as pullback inputs and constant atoms as
    /// pullback constants, so a transpose rule can read either as a known operand's value. A rule that requests the
    /// value of a known *intermediate* (a known atom that is neither a program input nor a constant) causes this to
    /// return [`ProgramError::UnsupportedOperation`].
    ///
    /// This shares the [`transpose`](Program::transpose) name with the [`Program`]-level entry point and the
    /// outer-trace [`transpose_traced`](Self::transpose_traced) variant on [`TracingContext`]: this builder-level
    /// method consumes a [`TracingContext`] and stages the pullback into its [`ProgramBuilder`] over the `(T, V, O)`
    /// universe, whereas [`transpose_traced`](Self::transpose_traced) transposes a program whose values are tracers of
    /// an enclosing trace.
    ///
    /// # Parameters
    ///
    ///   - `program`: Linear pushforward [`Program`] to transpose.
    ///   - `input_linearity`: Per-input linearity flags, in program-input order. Its length must equal the number of
    ///     program inputs, otherwise this returns [`ProgramError::InvalidInputCount`].
    #[inline]
    pub fn transpose<Input: Parameterized<V>, Output: Parameterized<V>>(
        mut self,
        program: &Program<V, O, Input, Output>,
        input_linearity: &[bool],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, ProgramError> {
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

        // Validate the per-input linearity mask before doing any work so a mismatched mask is reported up front.
        check_count!("input", input_linearity, program.input_ids().len(), ProgramError);

        // Propagate operand linearity forward over the primal atoms. A program-input atom takes its linearity from
        // `input_linearity`, a constant atom is always known (non-linear), and an instruction result is linear when any of
        // its operands is linear. Because instructions are stored in evaluation order, a single forward pass suffices:
        // every operand atom of an instruction is defined before that instruction. With an all-`true` mask every
        // reachable variable becomes linear, so each operation's transpose rule sees an all-`true` operand slice and
        // behaves exactly as it did before partition-aware transposition.
        let mut linear = vec![false; program.atoms().len()];
        for (input, &input_is_linear) in program.input_ids().iter().copied().zip(input_linearity) {
            *linear.get_mut(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? = input_is_linear;
        }
        for (index, atom) in program.atoms().iter().enumerate() {
            if matches!(atom, Atom::Constant(_)) {
                linear[index] = false;
            }
        }
        for instruction in program.instructions().iter() {
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

        // Reuse this context's current builder as the destination for the pullback program, and reserve the main
        // structural vectors up front. These are conservative lower bounds that cover cotangent inputs, one instruction
        // per reversed primal instruction, and possible zero outputs for disconnected primal inputs.
        let builder = self.builder().clone();
        {
            let mut builder_borrow = builder.borrow_mut();
            builder_borrow
                .atoms
                .reserve(program.output_ids.len() + program.instructions.len() + program.input_ids.len());
            builder_borrow.input_ids.reserve(program.output_ids.len());
            builder_borrow.instructions.reserve(program.instructions.len() + program.input_ids.len());
        }

        // Seed the reverse pass with one cotangent input for each primal output, typed with that output's cotangent
        // slot type. A differentiable output's slot carries its cotangent dual (e.g., swapping unreduced and reduced
        // sharding axes for arrays); a non-differentiable output (the `float0` analogue, such as a Boolean or integer)
        // has no cotangent space, so its slot carries only structural zeros typed by the output's own primal type. The
        // adjoint table is indexed by atoms from the original program, and each slot stores the staged pullback atom
        // that currently represents the accumulated cotangent for that primal atom.
        let mut adjoints = vec![None; program.atoms().len()];
        for output in program.output_ids().iter().copied() {
            let output_atom = program.atoms().get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
            let output_type = output_atom.r#type();
            let cotangent_type = output_type.cotangent().unwrap_or_else(|| output_type.into_owned());
            let cotangent_input = builder.borrow_mut().add_input(cotangent_type);
            accumulate::<V, O>(&builder, adjoints.as_mut_slice(), output, cotangent_input)?;
        }

        // Add a pullback input carrying the runtime value of each known program input, after the cotangent inputs so
        // the all-`true` mask leaves the pullback input numbering unchanged. Known inputs are exposed to transpose
        // rules as ordinary operand values, typed with the known input's own type (a runtime value, not a cotangent),
        // and recorded in `known_map` indexed by the primal atom so a rule can read the known operand's pullback atom.
        let mut known_map = vec![None; program.atoms().len()];
        for (input, &input_is_linear) in program.input_ids().iter().copied().zip(input_linearity) {
            if !input_is_linear {
                let input_atom = program.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                let known_input = builder.borrow_mut().add_input(input_atom.r#type().into_owned());
                known_map[input.index()] = Some(known_input);
            }
        }

        // Constant atoms are also known operands, so expose each one to transpose rules as a pullback constant. A
        // bilinear operation such as `Mul` whose known operand is a constant (for example, the `3` in `3 * x`) reads
        // this pullback atom exactly as it reads a known input's value. This is the partition-aware analogue of folding
        // a rebuilt constant into a captured factor, and it keeps a constant-scaled tangent transposable directly
        // rather than reporting it as an unsupported known intermediate.
        for (index, atom) in program.atoms().iter().enumerate() {
            if let Some(value) = atom.as_constant() {
                known_map[index] = Some(builder.borrow_mut().add_constant(value.clone()));
            }
        }

        // Walk the primal program backward, applying each operation's transpose rule only when at least one of its
        // outputs has a non-zero accumulated cotangent. The scratch vector avoids allocating a fresh cotangent vector
        // for every live instruction.
        let max_instruction_output_count =
            program.instructions().iter().map(|instruction| instruction.outputs().len()).max().unwrap_or(0);
        let mut instruction_output_cotangents = Vec::with_capacity(max_instruction_output_count);
        for instruction in program.instructions().iter().rev() {
            // Skip dead reverse edges early: if none of an instruction's outputs carries an adjoint, the instruction
            // cannot contribute to any input cotangent. This is the only operand-side guard: a non-linear instruction
            // whose transpose rule would read a known operand is always safe because the partial-evaluation split that
            // produces partitioned tangent programs exposes every known operand's value (as a known input or constant),
            // so the known-intermediate guard below stays purely defensive. (A vmapped masked `while` threads its
            // structurally-zero Boolean-mask carry as a plain pushforward tangent rather than through an all-known
            // `select` over a restored zero, so no such instruction reads an unexposed known intermediate.)
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
            // Structural zeros carry the output's cotangent slot type: a differentiable output's cotangent dual, or —
            // for a non-differentiable output (the `float0` analogue) — the output's own primal type. Accumulated
            // adjoints are always live: rules communicate zero-ness symbolically through [`MaybeZero`] (opaque
            // program splices such as the custom-VJP backward replay recover it at their own boundary), so no staged
            // canonical zero ever needs to be recognized here.
            instruction_output_cotangents.clear();
            for output in instruction.outputs().iter().copied() {
                let cotangent = adjoints.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
                instruction_output_cotangents.push(match cotangent {
                    Some(atom) => MaybeZero::Value(self.tracer(*atom, None)),
                    None => {
                        let output_type = program
                            .atoms()
                            .get(output.index())
                            .ok_or(ProgramError::UnboundAtomId { id: output })?
                            .r#type();
                        MaybeZero::Zero(output_type.cotangent().unwrap_or_else(|| output_type.into_owned()))
                    }
                });
            }

            // Apply the primitive transpose rule and require exactly one cotangent contribution per primal input. This
            // prevents malformed rules from silently dropping or inventing cotangents through iterator truncation.
            //
            // Each operand becomes a self-describing `PartialValue`: a linear operand is `Unknown` of its type (the
            // rule produces a cotangent of that type), and a known operand is `Known` of the tracer reading its
            // pullback value atom from `known_map` (a known program input or a constant). A known operand with no
            // pullback value would be a known *intermediate* (a known atom that is neither a program input nor a
            // constant). The partial-evaluation split that produces partitioned tangent programs never leaves one (see
            // this module's docs), so guarding it here once lets every rule assume a `Known` operand carries its value.
            let inputs = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input| {
                    let r#type = program
                        .atoms()
                        .get(input.index())
                        .ok_or(ProgramError::UnboundAtomId { id: input })?
                        .r#type()
                        .into_owned();
                    if *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                        Ok(PartialValue::Unknown(r#type))
                    } else {
                        match known_map.get(input.index()).copied().ok_or(ProgramError::UnboundAtomId { id: input })? {
                            Some(atom) => Ok(PartialValue::Known(self.tracer(atom, Some(r#type)))),
                            None => Err(ProgramError::UnsupportedOperation {
                                message: "partition-aware transpose of a known intermediate is not yet supported"
                                    .to_string(),
                            }),
                        }
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let input_cotangents = instruction.operation().transpose(
                &mut self,
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

        // The pullback outputs are the accumulated cotangents for the linear primal inputs only; known inputs receive
        // no cotangent output. Disconnected linear inputs are emitted as input-free [`ZeroOperation`] instructions,
        // which the value type's [`Zero`](crate::Zero) implementation evaluates at interpretation time, typed with the
        // input's cotangent slot type: a differentiable input's cotangent dual, or — for a non-differentiable linear
        // input (the `float0` analogue) — the input's own primal type, whose cotangent slot carries only structural
        // zeros. With an all-`true` mask every input is linear, so this keeps one cotangent output per primal input.
        let outputs = program
            .input_ids()
            .iter()
            .copied()
            .zip(input_linearity.iter().copied())
            .filter(|&(_, input_is_linear)| input_is_linear)
            .map(|(input, _)| {
                match adjoints.get(input.index()).copied().ok_or(ProgramError::UnboundAtomId { id: input })? {
                    Some(adjoint) => Ok::<AtomId, ProgramError>(adjoint),
                    None => {
                        let input_atom =
                            program.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
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

        // Build the pullback from this context's builder. The pullback inputs (cotangents per primal output, then
        // known-input values) and outputs (cotangents for the linear inputs) are flat, so they are built with flat
        // `Vec` structures; the fully linear callers recover the structured form by reattaching the program's input
        // and output structures.
        let pullback_input_count = builder.borrow().input_ids().len();
        let pullback_output_count = outputs.len();
        // Drop the throwaway context so its builder reference is released; with every staged `Tracer` already dropped,
        // the cloned `builder` handle is now the sole owner and can be unwrapped to finalize the pullback.
        drop(self);
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

    impl<'operation> TryFrom<&'operation TestLinearOperation> for &'operation ZeroOperation<DataType> {
        type Error = ();

        #[inline]
        fn try_from(value: &'operation TestLinearOperation) -> Result<Self, ()> {
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
    fn test_program_transpose_identity() {
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
    }

    #[test]
    fn test_program_transpose_accumulates_contributions_to_repeated_input() {
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
    }

    #[test]
    fn test_program_transpose_passes_zero_for_unused_instruction_output() {
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
    }

    #[test]
    fn test_program_transpose_rejects_invalid_rule_input_cotangent_count() {
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::BadArity, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),));
    }

    #[test]
    fn test_program_transpose_rejects_foreign_builder_contribution() {
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::ForeignContribution, vec![input]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(program.transpose(), Err(ProgramError::MismatchedProgramBuilders),));
    }

    #[test]
    fn test_program_transpose_with_respect_to_rejects_invalid_input_indices() {
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Add, vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
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

    #[test]
    fn test_program_transpose_with_respect_to_orders_outputs_by_the_requested_indices() {
        let mut builder = ProgramBuilder::<Scalar, TestLinearOperation>::new();
        let left = builder.add_input(DataType::F64);
        let right = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::Add, vec![left, right]).unwrap()[0];
        let program = builder
            .build::<(Scalar, Scalar), Scalar>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();

        // The pullback's cotangent outputs follow the requested index order, not program-input order: both inputs of
        // the `add` receive the seeded output cotangent, so the two orders are distinguishable only through the
        // output permutation.
        let forward = program.transpose_with_respect_to(&[0, 1]).unwrap();
        let reversed = program.transpose_with_respect_to(&[1, 0]).unwrap();
        assert_eq!(forward.output_ids().len(), 2);
        assert_eq!(reversed.output_ids().len(), 2);
        assert_eq!(
            reversed.output_ids(),
            &[forward.output_ids()[1], forward.output_ids()[0]],
            "requested index order must permute the pullback outputs",
        );
    }

    #[test]
    fn test_program_transpose_materializes_disconnected_input_zero() {
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
    }

    #[test]
    fn test_program_transpose_skips_dead_instruction() {
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
    }

    #[test]
    fn test_program_transpose_reports_unbound_input_atom() {
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
    }

    #[test]
    fn test_program_transpose_reports_unbound_instruction_output_atom() {
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
    fn test_tracing_context_transpose_materializes_disconnected_input_zero_as_zero_instruction() {
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
        let pullback = tracing_context.transpose_traced(&program).unwrap();
        assert_eq!(disconnected_input, AtomId::new(1));
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(0), AtomId::new(1)]);

        // The disconnected input's cotangent is emitted as an input-free `ZeroOperation` instruction in the pullback,
        // which is materialized at interpretation time rather than as a pullback constant staged at transpose time.
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

        // The outer tracing context is left untouched: transposition stages no zero into it.
        let outer_builder = outer_builder.borrow();
        assert!(outer_builder.atoms().is_empty());
        assert!(outer_builder.instructions().is_empty());
    }

    #[test]
    fn test_tracing_context_transpose_materializes_staged_zero_contribution_as_zero_instruction() {
        let tracing_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let outer_builder = tracing_context.builder().clone();
        let mut builder = ProgramBuilder::<TestTracingValue, TestLinearOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(TestLinearOperation::StagedZeroContribution, vec![input]).unwrap()[0];
        let program =
            builder.build::<TestTracingValue, TestTracingValue>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = tracing_context.transpose_traced(&program).unwrap();
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(1)]);

        // The transpose-rule-staged structural zero stays an input-free `ZeroOperation` instruction in the pullback,
        // materialized at interpretation time rather than as a pullback constant staged at transpose time.
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

        // The outer tracing context is left untouched: transposition stages no zero into it.
        let outer_builder = outer_builder.borrow();
        assert!(outer_builder.atoms().is_empty());
        assert!(outer_builder.instructions().is_empty());
    }
}
