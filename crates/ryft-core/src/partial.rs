use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt::Debug;

use crate::contexts::{Context, EagerContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::parameters::Placeholder;
use crate::programs::{Atom, AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::TracingContext;
use crate::types::{Type, Typed};

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial evaluator
/// interprets a [`Program`] over. Every [`Atom`] and every intermediate result is either [`Known`](Self::Known)
/// (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its [`Type`] is available until
/// the residual program runs). For more information on partial evaluation, refer to the documentation of
/// [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum PartialValue<T: Type, V: Value<T>> {
    /// [`Value`] that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// [`Value`] that is not known until the residual program runs and only its [`Type`] is known.
    Unknown(T),
}

impl<T: Type, V: Value<T>> PartialValue<T, V> {
    /// Returns `true` if this value is [`Known`](Self::Known).
    #[inline]
    pub fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this value is [`Unknown`](Self::Unknown).
    #[inline]
    pub fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }

    /// Returns the underlying concrete value when this is [`Known`](Self::Known) and [`None`] otherwise.
    #[inline]
    pub fn as_known(&self) -> Option<&V> {
        match self {
            Self::Known(value) => Some(value),
            Self::Unknown(_) => None,
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for PartialValue<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        match self {
            Self::Known(value) => value.r#type(),
            Self::Unknown(r#type) => Cow::Borrowed(r#type),
        }
    }
}

/// Represents the way in which a [`PartialEvaluationValue`] is represented when _residual_ work depends on it.
/// A [`PartialValue`] only records whether a value is known now or unknown until a residual [`Program`] runs.
/// [`PartialValueMaterialization`] records how that value is represented at the residual boundary. The optional
/// `source_atom` fields are source-program deduplication keys which let [`PartialEvaluator`]s reuse the same residual
/// [`PartialValueMaterialization::Input`] or [`PartialValueMaterialization::Constant`] when the same known source
/// value is materialized more than once in the current scope. They are `None` for known values synthesized by partial
/// evaluation rules or otherwise not tied to a stable [`Atom`] in the source program. By contrast, the `residual_atom`
/// field represents an atom _in_ the residual program and is required because [`PartialValueMaterialization::Variable`]
/// values have already been emitted there.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialValueMaterialization {
    /// Known value with no residual materialization decision yet. If residual work depends on it, the corresponding
    /// [`PartialEvaluator`] will materialize it as a fresh residual input.
    Undecided,

    /// Known value that should be materialized as a residual program input.
    Input {
        /// Source atom in the current program, if the value came from one. When present, repeated materialization
        /// in the same scope will reuse the residual input already created for this source atom. When absent,
        /// materialization creates a fresh residual input because the known value has no stable source program atom.
        source_atom: Option<AtomId>,
    },

    /// Known value that should be materialized as an inline residual program constant.
    Constant {
        /// Source atom in the current program, if the value came from one. When present, repeated materialization
        /// in the same scope will reuse the residual constant already created for this source atom. When absent,
        /// materialization creates a fresh residual constant because the known value has no stable source program atom.
        source_atom: Option<AtomId>,
    },

    /// Unknown value already represented as a residual program variable.
    Variable {
        /// Atom in the residual program that carries this value. This is not a source program atom (in contrast to the
        /// payloads of [`Self::Input`] and [`Self::Constant`]); residual operations consume it directly, and so it is
        /// not optional.
        residual_atom: AtomId,
    },
}

/// Represents the [`Value`] type used by [`PartialEvaluator`]s while partially evaluating [`Program`]s.
#[derive(Clone, Debug)]
pub struct PartialEvaluationValue<T: Type, V: Value<T>> {
    /// Underlying [`PartialValue`] that represents the abstract known/unknown classification of the value.
    value: PartialValue<T, V>,

    /// [`PartialValueMaterialization`] that describes how the underlying value is represented at the residual program
    /// boundary. This is deliberately separate from the underlying [`PartialValue`] because it answers a different
    /// question. A [`Known`](PartialValue::Known) value can still be consumed by residual work, materializing as a
    /// residual input or an inline residual constant according to its [`PartialValueMaterialization`], while an
    /// [`Unknown`](PartialValue::Unknown) value is always represented by a residual program variable that already
    /// exists.
    materialization: PartialValueMaterialization,
}

impl<T: Type, V: Value<T>> PartialEvaluationValue<T, V> {
    /// Creates a known [`PartialEvaluationValue`] with [`PartialValueMaterialization::Undecided`].
    #[inline]
    pub fn known(value: V) -> Self {
        Self { value: PartialValue::Known(value), materialization: PartialValueMaterialization::Undecided }
    }

    /// Creates a known [`PartialEvaluationValue`] with [`PartialValueMaterialization::Input`].
    #[inline]
    pub fn known_input(value: V, source_atom: Option<AtomId>) -> Self {
        Self { value: PartialValue::Known(value), materialization: PartialValueMaterialization::Input { source_atom } }
    }

    /// Creates a known [`PartialEvaluationValue`] with [`PartialValueMaterialization::Constant`].
    #[inline]
    pub fn known_constant(value: V, source_atom: Option<AtomId>) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: PartialValueMaterialization::Constant { source_atom },
        }
    }

    /// Creates an unknown [`PartialEvaluationValue`] with [`PartialValueMaterialization::Variable`].
    #[inline]
    pub fn variable(r#type: T, residual_atom: AtomId) -> Self {
        Self {
            value: PartialValue::Unknown(r#type),
            materialization: PartialValueMaterialization::Variable { residual_atom },
        }
    }

    /// Returns the underlying [`PartialValue`].
    #[inline]
    pub fn value(&self) -> &PartialValue<T, V> {
        &self.value
    }

    /// Returns the [`PartialValueMaterialization`] of this [`PartialEvaluationValue`].
    #[inline]
    pub fn materialization(&self) -> PartialValueMaterialization {
        self.materialization
    }

    /// Returns `true` if the underlying value is [`Known`](PartialValue::Known).
    #[inline]
    pub fn is_known(&self) -> bool {
        self.value.is_known()
    }

    /// Returns `true` if the underlying value is [`Unknown`](PartialValue::Unknown).
    #[inline]
    pub fn is_unknown(&self) -> bool {
        self.value.is_unknown()
    }

    /// Returns the underlying concrete value if this value is [`Known`](PartialValue::Known) and [`None`] otherwise.
    #[inline]
    pub fn as_known(&self) -> Option<&V> {
        self.value.as_known()
    }

    /// If this value is [`Known`](PartialValue::Known), this function marks it as being bound to the [`Atom`] that
    /// corresponds to the provided [`AtomId`], making that atom its residual input materialization deduplication key.
    #[inline]
    fn bind(self, atom: AtomId) -> Self {
        match self.value {
            PartialValue::Known(value) => Self::known_input(value, Some(atom)),
            PartialValue::Unknown(_) => self,
        }
    }
}

impl<T: Type, V: Value<T>> Typed<T> for PartialEvaluationValue<T, V> {
    #[inline]
    fn r#type(&self) -> Cow<'_, T> {
        self.value.r#type()
    }
}

/// Input of a partially evaluated (i.e., a _residual_) [`Program`] (i.e., an input of a [`PartialEvaluation`]).
/// The residual program's inputs are the original program's surviving unknown inputs followed by the known values
/// (i.e., the residuals) that its unknown subcomputation consumes.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug)]
pub enum PartialEvaluationInput<V> {
    /// Residual input fed by a value that partial evaluation folded to a concrete known residual value.
    Known(V),

    /// Residual input fed by an unknown input of the original program, identified by that input's index in the
    /// original program's inputs.
    Unknown(usize),
}

/// Output of a partially evaluated (i.e., a _residual_) [`Program`] (i.e., an input of a [`PartialEvaluation`]).
/// Partial evaluation splits the original outputs into those it could fold to a concrete value now and those that
/// remain computed by the residual program.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug)]
pub enum PartialEvaluationOutput<V> {
    /// Output that was folded to a concrete value during partial evaluation.
    Known(V),

    /// Output produced by the residual program, identified by its index into the residual program's outputs.
    Unknown(usize),
}

/// Result of partially evaluating a [`Program`] against a known-side [`Context`]. The residual program operates in
/// the *staged constant* space `C::Constant`, while the feeders that connect it to the known side flow as `C::Value`s.
/// Under an eager known-side context the two coincide and every [`PartialEvaluationInput::Known`] carries a concrete
/// folded value, while under a staging known-side context the feeders are [`Tracer`](crate::Tracer)s naming atoms of
/// the *outer* program that partial evaluation folded the known work into. To reconstruct the original program's
/// outputs, one must build the residual program's input vector by mapping each input from [`inputs`](Self::inputs)
/// to either a runtime unknown-input value or its carried known residual, replay [`program`](Self::program) in the
/// known-side context, and then read each output from [`outputs`](Self::outputs) as either its folded value or the
/// indexed residual program output.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
pub struct PartialEvaluation<C: Context> {
    /// Residual program over the surviving unknown inputs plus the known residuals, aligned with
    /// [`inputs`](Self::inputs) and producing the unknown outputs in their original order.
    pub program: Program<C::Type, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// [`PartialEvaluationInput`]s for [`program`](Self::program), in residual program input order.
    pub inputs: Vec<PartialEvaluationInput<C::Value>>,

    /// [`PartialEvaluationOutput`]s of [`program`](Self::program), in original output order.
    pub outputs: Vec<PartialEvaluationOutput<C::Value>>,
}

impl<C: Context<Operation: Debug>> Debug for PartialEvaluation<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PartialEvaluation")
            .field("program", &self.program)
            .field("inputs", &self.inputs)
            .field("outputs", &self.outputs)
            .finish()
    }
}

impl<C: Context<Operation: Clone>> PartialEvaluation<C> {
    /// Interprets the residual [`Program`] that this [`PartialEvaluation`] represents in the provided `context` and
    /// at the provided unknown input values, and reassembles the original program's outputs, in original output order.
    /// This is the single replay path for both known-side flavors: residual program constants are lifted through
    /// [`Context::lift`] and [`Instruction`](crate::Instruction)s are bound through [`Context::bind`], and so under
    /// an eager context the residual program is interpreted immediately, while under a [`StagingContext`] it is staged
    /// into the outer program that context is building. Each residual input is fed either by its carried known residual
    /// (i.e., a [`Known`](PartialEvaluationInput::Known) feeder) or by the next value of `inputs` (i.e., an
    /// [`Unknown`](PartialEvaluationInput::Unknown) feeder). Folded outputs are returned directly and the rest
    /// read the replayed residual program's outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Known-side context to interpret the residual program in.
    ///   - `inputs`: Values for the original program's surviving *unknown* inputs only, in their original relative
    ///     order. The known inputs are fed from the carried residual feeders, and so the size of `inputs` must equal
    ///     the number of [`Unknown`](PartialEvaluationInput::Unknown) feeders exactly (and not the original program's
    ///     number of inputs).
    pub fn interpret(&self, context: &C, inputs: &[C::Value]) -> Result<Vec<C::Value>, ProgramError> {
        let unknown_count = self.inputs.iter().filter(|i| matches!(i, PartialEvaluationInput::Unknown(_))).count();
        if inputs.len() != unknown_count {
            return Err(ProgramError::InvalidInputCount { expected: unknown_count, actual: inputs.len() });
        }
        let mut remaining_inputs = inputs.iter();
        let residual_inputs = self
            .inputs
            .iter()
            .map(|feeder| match feeder {
                PartialEvaluationInput::Known(value) => Ok(value.clone()),
                // The `.unwrap()` in the following line is safe because of the earlier check for `inputs.len()`.
                PartialEvaluationInput::Unknown(_) => Ok(remaining_inputs.next().cloned().unwrap()),
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let residual_outputs = self.program.interpret_with(
            residual_inputs,
            |_, constant| context.lift(constant.clone()),
            |instruction, inputs| context.bind(instruction.operation().clone(), inputs),
        )?;
        self.outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => Ok(value.clone()),
                PartialEvaluationOutput::Unknown(index) => residual_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "partial evaluation output references residual output {index} but the residual program \
                         produced {} output(s)",
                        residual_outputs.len(),
                    ))
                }),
            })
            .collect()
    }
}

// TODO(eaplatanios): Stick `PartitionedProgram` here.

/// [`Operation`] that supports partial evaluation via [`Program::partially_evaluate`]. This trait lets an individual
/// operation decide how partial evaluation treats it. It can be implemented with an empty implementation block,
/// deferring to [`PartialEvaluator::fold_or_residualize`], which is what most operations do, or its behavior can be
/// customized by overriding the [`PartiallyEvaluatableOperation::partially_evaluate`] function.
///
/// # Type Parameters
///
///   - `C`: Known-side [`Context`] that partial evaluation folds known work through. Its
///     [`Operation`](crate::Domain::Operation) is the operation family of the residual [`Program`] and of any inlined
///     nested programs (e.g., the enum this operation may belong to). Its [`Constant`](crate::Domain::Constant) is the
///     staged constant space those programs store. Finally, its [`Value`](crate::Domain::Value) is the space known
///     values flow in (i.e., concrete values under eager contexts and [`Tracer`](crate::Tracer)s into the outer program
///     under [`StagingContext`]s).
pub trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
    /// Partially evaluates this [`PartiallyEvaluatableOperation`] for the provided [`PartialEvaluationValue`]s. Unless
    /// overridden, this function will default to calling [`PartialEvaluator::fold_or_residualize`] which uses the
    /// following semantics:
    ///
    ///   - When *all* of the operation's inputs are [`Known`](PartialValue::Known), it **folds** the operation by
    ///     [`bind`](Context::bind)ing it in the known-side context, interpreting it immediately under an eager context,
    ///     and staging it into the outer program under a [`StagingContext`], so that the operation's outputs become
    ///     known values and the operation contributes nothing to the residual [`Program`].
    ///   - Otherwise, it **residualizes** the operation unchanged, meaning that it emits the operation into the
    ///     residual program over its inputs' residual program [`Atom`]s, materializing each known input as a residual
    ///     input for a known variable or as an inlined residual program constant for a literal, so that the operation
    ///     runs at residual program execution time.
    ///
    /// There are situations where overriding this function can result in improved performance and better partitioning
    /// of a computation into known and unknown parts. For example, a `condition` instruction whose predicate is
    /// [`Known`](PartialValue::Known) and concretizable may ask the evaluator to inline the selected branch and return
    /// that branch's output trace values, so that the condition disappears from the residual program and only the taken
    /// branch's work survives. Rules that inspect known *payloads* must gate that inspection on a
    /// [`Concrete`](ValueResolution::Concrete) [`Context::resolve`] resolution because a known value under a staging
    /// known-side context is a [`Tracer`](crate::Tracer) into the outer program rather than a concrete value, and
    /// partial evaluation should fall back to a conservative rewrite otherwise.
    ///
    /// # Parameters
    ///
    ///   - `evaluator`: [`PartialEvaluator`] that owns residual emission, inlining, and materialization.
    ///   - `inputs`: [`PartialEvaluationValue`] for each of this operation's inputs, in input order.
    #[inline]
    fn partially_evaluate(
        &self,
        evaluator: &mut PartialEvaluator<C>,
        inputs: &[PartialEvaluationValue<C::Type, C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError> {
        evaluator.fold_or_residualize(self.clone(), inputs)
    }
}

/// Represents closed [`Operation`] families whose nested flat [`Program`]s can be partially evaluated. This is the
/// partial evaluation analogue of [`InterpretableProgramOperation`](crate::InterpretableProgramOperation). It names the
/// recursive fixed point needed by higher-order partial evaluation helpers without requiring the full operation enum's
/// [`PartiallyEvaluatableOperation`] implementation while proving that implementation. Operation families implement it
/// by replaying nested flat [`Program`]s through their operation-owned partial evaluation rules.
///
/// Unlike the linearization and transposition witnesses, whose context and operation type parameters grow with each
/// recursion level and must therefore name a fixed point to stop the trait solver from diverging, this witness's
/// known-side context parameter `C` is fixed across recursion. The blanket implementation grounds it in
/// [`PartiallyEvaluatableOperation`], which a recursive operation enum's own generated implementation supplies, and so
/// proving it for a self-containing operation enum (i.e., one whose higher-order variants hold `Program`s of itself)
/// introduces no new recursive obligation.
pub trait PartiallyEvaluatableProgramOperation<C: Context<Operation = Self>>: Operation<C::Type> + Sized {
    /// Partially evaluates a nested flat [`Program`].
    ///
    /// # Parameters
    ///
    ///   - `context`: Known-side context used to fold known subcomputations.
    ///   - `program`: Nested [`Program`] to partially evaluate.
    ///   - `inputs`: Input [`PartialValue`]s to use for partially evaluating the provided [`Program`].
    fn partially_evaluate_program(
        context: &C,
        program: &Program<C::Type, C::Constant, Self, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: &[PartialValue<C::Type, C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>;
}

impl<C: Context<Operation: PartiallyEvaluatableOperation<C>>> PartiallyEvaluatableProgramOperation<C> for C::Operation {
    #[inline]
    fn partially_evaluate_program(
        context: &C,
        program: &Program<C::Type, C::Constant, Self, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: &[PartialValue<C::Type, C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        program.partially_evaluate_in_context(context, inputs)
    }
}

/// Driver of one [`Program::partially_evaluate`] walk. A [`PartialEvaluator`] carries out partial evaluation of a flat
/// [`Program`], accumulating the residual program as it goes and folding known subcomputations through the known-side
/// [`Context`]. It owns the [`ProgramBuilder`] that accumulates the residual program, the known-side [`Context`] used
/// to fold known subcomputations (where folding [`bind`](Context::bind)s the operation in that context, which
/// interprets it immediately under an eager context and stages it into the outer program under a staging context),
/// and the running list of residual inputs. The recursive inline walk handles a flat program (i.e., the top-level
/// program, or an inlined nested program) and returns the walk value of each of its outputs.
pub struct PartialEvaluator<C: Context> {
    /// Known-side [`Context`] used to fold [`Instruction`](crate::Instruction)s whose inputs are all known.
    context: C,

    /// [`ProgramBuilder`] accumulating the residual program's [`Atom`]s and [`Instruction`](crate::Instruction)s.
    builder: ProgramBuilder<C::Type, C::Constant, C::Operation>,

    /// [`PartialEvaluationInput`]s for the residual program, in residual program input order.
    inputs: Vec<PartialEvaluationInput<C::Value>>,

    /// Per-program materialization scopes deduplicating each known value's residual [`PartialValueMaterialization`]
    /// by its *source-program* [`Atom`], keyed structurally so one known variable consumed by several residualized
    /// [`Instruction`](crate::Instruction)s yields a single residual input (or inline constant) rather than one per
    /// consumer. When a residualized instruction consumes a known variable, that variable is materialized into the
    /// residual program (as a residual input or as an inline constant) and its source atom's slot records the
    /// resulting residual [`Atom`], so that later consumers of the same source atom can reuse it. This is a [`Vec`]
    /// of scopes rather than a single table because the walk recurses into inlined nested programs (e.g., a known
    /// predicate `condition` instruction inlining its branch through [`inline_program`](Self::inline_program)), and
    /// each program has its own [`AtomId`] space starting at zero. A single flat table would conflate atom `k` of an
    /// inlined nested program with atom `k` of the enclosing program and hand back the wrong residual atom, and so
    /// [`inline_program`](Self::inline_program) pushes a fresh scope sized to the nested program before walking it,
    /// and pops it afterward. The inlined source-atom lookup and recording in [`residualize`](Self::residualize) act on the innermost
    /// (`last`) scope.
    ///
    /// Note that this is the *primary*, always-active deduplication, and the only one available in two cases the
    /// walk-global [`staged_feeders`](Self::staged_feeders) cannot cover: under an eager known-side [`Context`], where
    /// known values resolve to [`Concrete`](ValueResolution::Concrete) instances and so carry no staged identity to key
    /// on, and for inline constants, whose materialization never consults a staged identity at all.
    materialization_scopes: Vec<Vec<Option<AtomId>>>,

    /// Walk-global deduplication of residual *input* feeders by the known value's *staged* identity
    /// (i.e., the outer program [`Atom`] a known value names when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in a staging known-side context), mapping the staged [`Atom`] to
    /// the residual input already created for it.
    ///
    /// This complements the per-program [`materialization_scopes`](Self::materialization_scopes) along the axis that
    /// scope-local, source-atom-keyed deduplication cannot reach. Staged atoms are stable identities across the whole
    /// walk (an outer-trace atom is the same value no matter which inlined program is being walked), and so two known
    /// feeders naming the same outer atom collapse to one residual input even when they arise in *different* inlined
    /// programs, and even for rule-produced knowns that carry no source atom to key the per-program table on. It holds
    /// only under a *staging* known-side context because an eager context resolves knowns as
    /// [`Concrete`](ValueResolution::Concrete) rather than [`Staged`](ValueResolution::Staged), and so nothing
    /// is ever recorded in that case, and inline constants are excluded because they carry no staged identity.
    staged_feeders: HashMap<AtomId, AtomId>,
}

impl<C: Context> PartialEvaluator<C> {
    /// Creates a fresh [`PartialEvaluator`] that folds known work through `context` and accumulates
    /// residual work in a new residual [`ProgramBuilder`].
    #[inline]
    pub fn new(context: C) -> Self {
        Self {
            context,
            builder: ProgramBuilder::new(),
            inputs: Vec::new(),
            materialization_scopes: Vec::new(),
            staged_feeders: HashMap::new(),
        }
    }

    /// Returns the known-side [`Context`] of this [`PartialEvaluator`] which is used to fold known subcomputations.
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Applies the default partial-evaluation policy to the provided `operation`. When all inputs are known, the
    /// operation is [`bind`](Context::bind)ed in the known-side [`Context`] (i.e., interpreting it under an eager
    /// context and staging it into the outer program under a [`StagingContext`]), and its outputs become known trace
    /// values. When any input is residual, all inputs are materialized into the residual program and the operation is
    /// emitted unchanged.
    ///
    /// # Effect Placement Contract
    ///
    /// Operations whose [`effects`](Operation::effects) are not [`Effects::PURE`](crate::Effects::PURE) follow the
    /// same known-ness placement: an all-known effectful operation folds into the known side, and a mixed-input one
    /// residualizes. This encodes the split's execution contract where all known work must run before residual work,
    /// so that an effect's side is determined by its input known-ness:
    ///
    ///   - Under an *eager* known-side context, folding executes the effect at partial-evaluation time, because the
    ///     known side of an eager split **is** executed at partial-evaluation time. This is also what makes
    ///     linearization of an effectful function fire its primal-side effects during the forward pass.
    ///   - Under a *staging* known-side context, folding stages the effect into the outer program, where it executes
    ///     in bind order with the rest of the known work.
    ///
    /// Ordered effects consequently keep their relative order *within* each side. Two ordered effects split across
    /// the known/residual boundary execute known-first regardless of their original instruction order, which is the
    /// documented reordering the split performs on *all* work.
    ///
    /// Some higher-order rules *analyze* a nested program by speculatively folding it (sometimes repeatedly, iterating
    /// to a fixed point) before deciding how to rewrite it (e.g., the `scan` and `while` loop-invariance fixed points).
    /// When such a rule folds through the **live** known-side context (rather than a throwaway context it builds and
    /// then discards) each speculative fold has a real consequence: it executes the effect when in an eager context or
    /// stages it into the outer program when in a [`StagingContext`]. A rule that iterates would then fire or stage the
    /// effect once per round instead of exactly once, and so it must skip effectful programs and residualize them
    /// unchanged.
    #[inline]
    pub fn fold_or_residualize<P: Into<C::Operation>>(
        &mut self,
        operation: P,
        inputs: &[PartialEvaluationValue<C::Type, C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError> {
        let operation = operation.into();
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
            Ok(self
                .context
                .bind(operation, known.as_slice())?
                .into_iter()
                .map(PartialEvaluationValue::known)
                .collect())
        } else {
            self.residualize(operation, inputs)
        }
    }

    /// _Residualizes_ the provided [`Operation`] into the residual [`Program`], materializing each known input into a
    /// residual program [`Atom`] according to its [`PartialValueMaterialization`], and returns the operation's outputs
    /// as [`PartialEvaluationValue`]s, in output order. Materializing a known value deduplicates it two ways so a value
    /// consumed by several residualized [`Instruction`](crate::Instruction)s yields one residual input (or inline
    /// constant): by its *source-program* atom within the current materialization scope, and, for inputs, by its
    /// *staged* identity across the whole walk when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in the known-side context. A
    /// [`Constant`](PartialValueMaterialization::Constant) materialization is only ever attached to values that
    /// originated as literals (i.e., walked-program constants lifted into the known-side context, or rule-produced
    /// [`known_constant`](PartialEvaluationValue::known_constant) values), and so recovering its payload through
    /// [`Context::resolve`] is expected to succeed. This is what keeps the residual program in the staged-constant
    /// space, since under a staging known-side context a known value is a [`Tracer`](crate::Tracer) that can never
    /// itself be a residual-program constant.
    pub fn residualize<P: Into<C::Operation>>(
        &mut self,
        operation: P,
        inputs: &[PartialEvaluationValue<C::Type, C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError> {
        // Materialize each known input into a residual-program atom. The deduplication fast-paths return early,
        // and a genuine error rides `?` out through the `collect` into `residualize`.
        let input_atoms = inputs
            .iter()
            .map(|input| -> Result<AtomId, ProgramError> {
                // A residual variable is already a residual atom. Every known value differs only in its source-atom
                // deduplication key and whether it materializes as an inline constant.
                let (source_atom, constant) = match input.materialization() {
                    PartialValueMaterialization::Variable { residual_atom } => return Ok(residual_atom),
                    PartialValueMaterialization::Undecided => (None, false),
                    PartialValueMaterialization::Input { source_atom } => (source_atom, false),
                    PartialValueMaterialization::Constant { source_atom } => (source_atom, true),
                };

                // Reuse the residual atom already created for this source atom in the current scope, if any. This
                // reads the scope separately from the recording at the tail. The fresh-atom creation between then
                // mutates the builder, so a single borrow of the scope cannot span both.
                if let Some(source_atom) = source_atom {
                    let scope = self.materialization_scopes.last().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "partial evaluation materialization has no active scope".to_string(),
                        )
                    })?;
                    let existing = scope.get(source_atom.index()).copied().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "residual materialization referenced source atom {source_atom} outside the active program",
                        ))
                    })?;
                    if let Some(atom) = existing {
                        return Ok(atom);
                    }
                }

                let known = input.as_known().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "residual materialization marked an unknown value as a known residual".to_string(),
                    )
                })?;

                // Inputs (but not inline constants) additionally deduplicate across the whole walk by the value's
                // staged identity in the known-side context; reuse the residual input already created for it, if any.
                let staged_atom = if constant {
                    None
                } else {
                    match self.context.resolve(known) {
                        ValueResolution::Staged(atom) => Some(atom),
                        _ => None,
                    }
                };

                // Reuse the residual input already registered for this staged identity, or create a fresh residual
                // constant (recovering the literal payload) or residual input and register it under that identity.
                let atom = match staged_atom.and_then(|staged_atom| self.staged_feeders.get(&staged_atom).copied()) {
                    Some(existing) => existing,
                    None => {
                        let atom = if constant {
                            let constant = self.context.resolve(known).into_concrete().ok_or_else(|| {
                                ProgramError::MalformedProgram(
                                    "residual materialization required a constant payload for a known value that is \
                                     not concretizable in the active known-side context"
                                        .to_string(),
                                )
                            })?;
                            self.builder.add_constant(constant)
                        } else {
                            let atom = self.builder.add_input(known.r#type().into_owned());
                            self.inputs.push(PartialEvaluationInput::Known(known.clone()));
                            atom
                        };
                        if let Some(staged_atom) = staged_atom {
                            self.staged_feeders.insert(staged_atom, atom);
                        }
                        atom
                    }
                };

                // Record the source-atom to residual-atom mapping for the current scope, obtaining the scope once
                // for whichever path produced `atom` above.
                if let Some(source_atom) = source_atom {
                    let scope = self.materialization_scopes.last_mut().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "partial evaluation materialization has no active scope".to_string(),
                        )
                    })?;
                    let slot = scope.get_mut(source_atom.index()).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "residual materialization referenced source atom {source_atom} outside the active program",
                        ))
                    })?;
                    *slot = Some(atom);
                }
                Ok(atom)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        Ok(self
            .builder
            .add_instruction(operation, input_atoms)?
            .to_vec()
            .iter()
            .copied()
            .map(|atom| {
                let r#type = self.builder.atoms()[atom.index()].r#type().into_owned();
                PartialEvaluationValue::variable(r#type, atom)
            })
            .collect())
    }

    /// Walks the provided [`Program`]'s instructions using the provided `inputs` bound to its input [`Atom`]s in
    /// input order, folds every all-known [`Instruction`](crate::Instruction) by [`bind`](Context::bind)ing it in the
    /// known-side [`Context`], dispatches each instruction to its [`PartiallyEvaluatableOperation::partially_evaluate`]
    /// implementation, and emits the residual work into this context's residual [`ProgramBuilder`]. Finally, it returns
    /// the walk value of each program output, in output order.
    ///
    /// [`Operation`]-specific rules can call this function to recursively walk nested programs over selected inputs,
    /// so that an operation can rewrite itself into transformed work. For example, a known-predicate `condition` can
    /// inline its selected branch. Program constants are [`lift`](Context::lift)ed into the known-side context and
    /// recorded as [`PartialEvaluationValue::known_constant`] on first use. When residual work consumes those
    /// constants, they are rebuilt inline in the residual program by recovering their original constant payload
    /// through [`Context::resolve`].
    pub fn inline_program(
        &mut self,
        program: &Program<C::Type, C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<PartialEvaluationValue<C::Type, C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<C>,
    {
        // A fresh materialization scope isolates this program's source-atom deduplication to its own atom space.
        // The walk runs inside a closure so the scope is popped on every exit path, including error paths, keeping
        // the scope stack balanced.
        self.materialization_scopes.push(vec![None; program.atoms.len()]);
        let result = (|| -> Result<Vec<PartialEvaluationValue<C::Type, C::Value>>, ProgramError> {
            // Walk-time value of each atom in `program`, populated as the forward pass reaches it.
            let mut values = vec![None; program.atoms.len()];
            for (input_id, value) in program.input_ids.iter().copied().zip(inputs) {
                values[input_id.index()] = Some(value.bind(input_id));
            }

            for instruction in program.instructions.iter() {
                // Resolve instruction inputs as walk values, lifting a program constant to `Known` on first use.
                let mut inputs = Vec::with_capacity(instruction.inputs().len());
                for input_id in instruction.inputs().iter().copied() {
                    let value = match values[input_id.index()].clone() {
                        Some(value) => value,
                        None => match &program.atoms[input_id.index()] {
                            Atom::Constant(constant) => {
                                let value = PartialEvaluationValue::known_constant(
                                    self.context.lift(constant.clone())?,
                                    Some(input_id),
                                );
                                values[input_id.index()] = Some(value.clone());
                                value
                            }
                            Atom::Variable(_) => return Err(ProgramError::UnboundAtomId { id: input_id }),
                        },
                    };
                    inputs.push(value);
                }

                let outputs = instruction.operation().partially_evaluate(self, inputs.as_slice())?;
                check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
                    values[output_id.index()] = Some(output.bind(output_id));
                }
            }

            program
                .output_ids
                .iter()
                .copied()
                .map(|output_id| match values[output_id.index()].clone() {
                    Some(value) => Ok(value),
                    None => match &program.atoms[output_id.index()] {
                        Atom::Constant(constant) => Ok(PartialEvaluationValue::known_constant(
                            self.context.lift(constant.clone())?,
                            Some(output_id),
                        )),
                        Atom::Variable(_) => Err(ProgramError::UnboundAtomId { id: output_id }),
                    },
                })
                .collect()
        })();
        self.materialization_scopes.pop();
        result
    }

    /// Recovers the staged-constant payload of the provided known value `value` through [`Context::resolve`], reporting
    /// a [`ProgramError`] when the known-side [`Context`] cannot prove that the provided value is a concrete constant.
    /// Higher-order rules use this when they must embed a known value *inside* a nested residual program (e.g., a
    /// folded loop-invariant carry spliced into a rebuilt `scan` body), where only a program constant can represent
    /// it (nested programs cannot reference atoms of the enclosing residual program or of the outer known-side
    /// program). Under an eager known-side context this always succeeds. Under a [`StagingContext`] it succeeds only
    /// for literal-backed values, and the caller must treat the error as "this rewrite is not available" and fall back
    /// to a conservative alternative.
    #[inline]
    pub fn known_constant(&self, value: &C::Value) -> Result<C::Constant, ProgramError> {
        self.context.resolve(value).into_concrete().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "a known value crossing into a nested residual program is not concretizable in the active \
                 known-side context"
                    .to_string(),
            )
        })
    }

    /// Returns `true` if every [`Known`](PartialEvaluationInput::Known) residual input and
    /// [`Known`](PartialEvaluationOutput::Known) output of the provided [`PartialEvaluation`] resolves to a concrete
    /// constant in the known-side [`Context`] of this [`PartialEvaluator`] (i.e., if a nested program rebuild that
    /// embeds those knowns as inline program constants through [`Self::known_constant`] can succeed). Under a staging
    /// known-side context, a probe's folds can produce known values that are genuine tracers into the live trace (e.g.,
    /// a constant-only chain staged by the fold). Rules that rebuild nested programs from a live context probe must
    /// check this and fall back to a conservative rewrite when it returns `false`.
    #[inline]
    pub fn all_knowns_are_concrete(&self, evaluation: &PartialEvaluation<C>) -> bool {
        evaluation.inputs.iter().all(|input| match input {
            PartialEvaluationInput::Known(value) => self.context.resolve(value).is_concrete(),
            PartialEvaluationInput::Unknown(_) => true,
        }) && evaluation.outputs.iter().all(|output| match output {
            PartialEvaluationOutput::Known(value) => self.context.resolve(value).is_concrete(),
            PartialEvaluationOutput::Unknown(_) => true,
        })
    }

    /// Returns `true` when any of the provided `inputs` is known but does not [`resolve`](Context::resolve) to a
    /// [`Concrete`](ValueResolution::Concrete) constant in the known-side [`Context`] of this [`PartialEvaluator`]
    /// (i.e., it is a genuine [`Tracer`](crate::Tracer) into a live outer trace). This is the signal online boundary
    /// rules split on: all-concrete knowledge keeps the default fold-or-residualize behavior.
    #[inline]
    pub fn any_known_is_symbolic(&self, inputs: &[PartialEvaluationValue<C::Type, C::Value>]) -> bool {
        inputs.iter().any(|input| match input.value() {
            PartialValue::Known(value) => !self.context.resolve(value).is_concrete(),
            PartialValue::Unknown(_) => false,
        })
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>> Program<T, V, O, Vec<V>, Vec<V>> {
    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work eagerly.
    /// This is the main partial evaluation entry point, instantiated at this program's own [`EagerContext`] so that
    /// known values are concrete values and folding interprets each all-known [`Instruction`](crate::Instruction)
    /// immediately. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the [`Context`]-taking
    /// core it delegates to and must be used instead with a [`StagingContext`] to fold known work into an enclosing
    /// trace.
    ///
    /// Partial evaluation classifies each [`Atom`] as *known* (i.e., computable _now_ from the provided values) or
    /// *unknown* (i.e., dependent on a runtime input), folds the known subcomputation away, and carves the remaining
    /// unknown subcomputation into a residual [`Program`] that consumes only the unknown inputs plus the known values
    /// it actually needs. During partial evaluation, each instruction is first offered to its own
    /// [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, which may override the default behavior.
    /// For example, a `condition` with a concretizable known predicate calls [`PartialEvaluator::inline_program`]
    /// to inline its selected branch in place of the operation, so that the condition disappears from the residual
    /// program. Building the residual program with a [`ProgramBuilder`] (rather than projecting the original) is what
    /// lets these rules emit *transformed* work; flat instructions with no override are emitted unchanged. The walk is
    /// flat per program but can recurse through operation rules into inlined nested programs, such as a selected
    /// `condition` branch; an instruction carrying a nested program that is *not* inlined is folded only when all of
    /// its inputs are known and is otherwise emitted unchanged.
    ///
    /// Each known *variable* a residualized instruction consumes, whether a program input or a folded intermediate,
    /// becomes a residual input of the residual program. Literal constants are rebuilt inline as residual-program
    /// constants (their staged payload is recovered through [`Context::resolve`]), so they are never residual inputs.
    /// The resulting [`PartialEvaluation`] carries everything a caller needs to reassemble the original outputs once
    /// the runtime (i.e., unknown) inputs are available.
    ///
    /// # Relationship to [`partially_evaluate_in_context`](Self::partially_evaluate_in_context)
    ///
    /// This function is the **eager** convenience form of partial evaluation: it evaluates known work under an
    /// [`EagerContext`], holding concrete known values and *folding the known subcomputation away* (through
    /// [`Context::bind`]) while applying per-operation rewrite rules, and yields a single residual [`Program`] with the
    /// folded output and residual-input *values*. Use it to **specialize or constant-fold** a program against inputs
    /// that are known. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the context-generic
    /// core behind it: passing a live [`StagingContext`] instead splits the program *online* against values known to an
    /// enclosing trace, staging the known work into the outer program rather than folding it to concrete values. The
    /// rewrite rules, residual construction, and output classification are identical across both; only the known-side
    /// [`Context`] differs.
    #[inline]
    pub fn partially_evaluate(
        &self,
        inputs: &[PartialValue<T, V>],
    ) -> Result<PartialEvaluation<EagerContext<T, V, O>>, ProgramError>
    where
        O: InterpretableOperation<T, V, EagerContext<T, V, O>> + PartiallyEvaluatableOperation<EagerContext<T, V, O>>,
    {
        self.partially_evaluate_in_context(&EagerContext::new(), inputs)
    }

    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work through
    /// the provided known-side [`Context`]. This is the context-taking core behind
    /// [`partially_evaluate`](Self::partially_evaluate).
    pub fn partially_evaluate_in_context<C: Context<Type = T, Constant = V, Operation = O>>(
        &self,
        context: &C,
        inputs: &[PartialValue<T, C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C>,
    {
        if inputs.len() != self.input_ids.len() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_ids.len(), actual: inputs.len() });
        }

        // Seed top-level inputs. Known inputs hold their value and unknown inputs lead the residual program's inputs.
        let mut residual = PartialEvaluator::new(context.clone());
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, (input_id, knowledge)) in self.input_ids.iter().copied().zip(inputs.iter()).enumerate() {
            match knowledge {
                PartialValue::Known(value) => {
                    seed.push(PartialEvaluationValue::known_input(value.clone(), Some(input_id)));
                }
                PartialValue::Unknown(r#type) => {
                    let atom = residual.builder.add_input(r#type.clone());
                    residual.inputs.push(PartialEvaluationInput::Unknown(index));
                    seed.push(PartialEvaluationValue::variable(r#type.clone(), atom));
                }
            }
        }

        // Assemble outputs. Folded values return directly and residual values index the residual program's outputs.
        let output_values = residual.inline_program(self, seed)?;
        let mut outputs = Vec::with_capacity(output_values.len());
        let mut residual_output_atoms: Vec<AtomId> = Vec::new();
        for value in output_values {
            match value.value {
                PartialValue::Known(value) => outputs.push(PartialEvaluationOutput::Known(value)),
                PartialValue::Unknown(_) => {
                    let PartialValueMaterialization::Variable { residual_atom } = value.materialization else {
                        return Err(ProgramError::MalformedProgram(
                            "partial evaluation produced an unknown output without a residual atom".to_string(),
                        ));
                    };
                    outputs.push(PartialEvaluationOutput::Unknown(residual_output_atoms.len()));
                    residual_output_atoms.push(residual_atom);
                }
            }
        }

        let output_count = residual_output_atoms.len();
        let residual_inputs = residual.inputs;
        let residual_program = residual
            .builder
            .build::<Vec<V>, Vec<V>>(
                residual_output_atoms,
                vec![Placeholder; residual_inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;

        Ok(PartialEvaluation { program: residual_program, inputs: residual_inputs, outputs })
    }

    // TODO(eaplatanios): Review from here onwards.
    
    /// Partitions this [`Program`] based on the provided per-input known-ness into a known-side program and a
    /// residual program joined by residual edges, packaged as a [`PartitionedProgram`]. This function invokes
    /// [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) with a **fresh** [`TracingContext`]
    /// whose inputs stand in for the known program inputs and so, instead of folding the known work into a
    /// caller-supplied context, the fresh trace reifies it as the known-side program. The same per-[`Operation`]
    /// rules drive both entry points, and they differ only in what happens to the known side.
    ///
    /// # Parameters
    ///
    ///   - `input_known`: Known-ness of each program input, in input order. The length of this slice must match the
    ///     number of inputs of this [`Program`].
    pub fn partition(&self, input_known: &[bool]) -> Result<PartitionedProgram<T, V, O>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<T, V, O>>,
    {
        let input_types = self.input_types();
        check_count!("input", input_known, input_types.len(), ProgramError);

        let context = TracingContext::<T, V, O>::new();
        let mut known_input_indices = Vec::new();
        let inputs = input_types
            .iter()
            .zip(input_known.iter())
            .enumerate()
            .map(|(index, (input_type, &known))| {
                if known {
                    known_input_indices.push(index);
                    PartialValue::Known(context.input(input_type.clone()))
                } else {
                    PartialValue::Unknown(input_type.clone())
                }
            })
            .collect::<Vec<_>>();
        let evaluation = self.partially_evaluate_in_context(&context, inputs.as_slice())?;

        let mut known_output_atoms = Vec::new();
        let mut outputs = Vec::with_capacity(evaluation.outputs.len());
        for output in evaluation.outputs.iter() {
            match output {
                PartialEvaluationOutput::Known(value) => {
                    outputs.push(PartialEvaluationOutput::Known(known_output_atoms.len()));
                    known_output_atoms.push(value.atom_id()?);
                }
                PartialEvaluationOutput::Unknown(ordinal) => {
                    outputs.push(PartialEvaluationOutput::Unknown(*ordinal));
                }
            }
        }

        let mut residual_inputs = Vec::with_capacity(evaluation.inputs.len());
        let mut residual_edge_count = 0;
        for input in evaluation.inputs.iter() {
            match input {
                PartialEvaluationInput::Unknown(index) => {
                    residual_inputs.push(PartialEvaluationInput::Unknown(*index));
                }
                PartialEvaluationInput::Known(value) => {
                    residual_inputs.push(PartialEvaluationInput::Known(residual_edge_count));
                    residual_edge_count += 1;
                    known_output_atoms.push(value.atom_id()?);
                }
            }
        }

        let known_program = if known_output_atoms.is_empty() {
            None
        } else {
            let known_output_count = known_output_atoms.len();
            Some(
                context
                    .builder()
                    .borrow()
                    .clone()
                    .build::<Vec<V>, Vec<V>>(
                        known_output_atoms,
                        vec![Placeholder; known_input_indices.len()],
                        vec![Placeholder; known_output_count],
                    )?
                    .into_simplified()?,
            )
        };
        Ok(PartitionedProgram {
            known_program,
            residual_program: evaluation.program,
            known_input_indices,
            residual_inputs,
            outputs,
        })
    }
}

/// Known-ness partition of a [`Program`] against per-input knowledge, produced by
/// [`Program::partition`] — the shared machinery behind *online* boundary partial-evaluation rules such as the XLA
/// `jit_call` and `shard_map` rules.
///
/// The program is partially evaluated through a **fresh** staging context whose inputs stand in for the known
/// boundary inputs, so no partition work can leak into the caller's live known-side context (the recursion contract
/// of [`PartiallyEvaluatableProgramOperation`]). The [`known_program`](Self::known_program) outputs the callee's fully
/// known outputs followed by the known→unknown *residual edges* — every known per-callee value the residual side
/// consumes; a rule wraps it in its boundary operation and binds it into the enclosing known-side context over the
/// original inputs named by [`known_input_indices`](Self::known_input_indices). The
/// [`residual_program`](Self::residual_program) is wrapped in the residual boundary operation and emitted over the
/// inputs assembled by [`residual_boundary_inputs`](Self::residual_boundary_inputs) — the surviving unknown boundary
/// inputs plus the known-side operation's residual-edge outputs — and [`assemble_outputs`](Self::assemble_outputs)
/// maps the two operations' outputs back to the original boundary output order. The
/// [`residual_inputs`](Self::residual_inputs) and [`outputs`](Self::outputs) reports reuse
/// [`PartialEvaluationInput`] and [`PartialEvaluationOutput`] with each value erased to a position, mirroring the
/// callee's [`PartialEvaluation`] one-to-one. All fields are public: a boundary rule keeps the partition whole —
/// taking the programs through [`Option::take`] on [`known_program`](Self::known_program) and through
/// [`take_residual_program`](Self::take_residual_program) — so every method up to and including
/// [`assemble_outputs`](Self::assemble_outputs) stays callable, while a consumer that wants full ownership of the
/// parts (such as linearization) destructures the partition instead.
pub struct PartitionedProgram<T: Type, V: Value<T>, O: Operation<T>> {
    /// Known-side callee program, or [`None`] when the partition has an empty known side: partial evaluation found
    /// no fully known callee output and no known→unknown residual edge — the known side computes nothing that
    /// anyone consumes (for example, when every known boundary input is dead in the callee) — so there is no
    /// known-side work to wrap in a boundary operation.
    pub known_program: Option<Program<T, V, O, Vec<V>, Vec<V>>>,

    /// Residual-side callee program: the callee's partial-evaluation residual program, whose inputs are described
    /// by [`residual_inputs`](Self::residual_inputs).
    pub residual_program: Program<T, V, O, Vec<V>, Vec<V>>,

    /// Indices of the original boundary inputs feeding the known-side operation, in order.
    pub known_input_indices: Vec<usize>,

    /// Source feeding each residual-program input, in residual-program input order. This is the callee's
    /// [`PartialEvaluation::inputs`] report with each feeder *value* erased to a position:
    /// [`Unknown`](PartialEvaluationInput::Unknown) entries keep their original boundary input index, and each
    /// [`Known`](PartialEvaluationInput::Known) feeder is erased to its residual-edge ordinal (which is also, offset
    /// by the fully known output count, the position of the edge among the known-side operation's outputs).
    pub residual_inputs: Vec<PartialEvaluationInput<usize>>,

    /// Source of each original callee output, in original output order. This is the callee's
    /// [`PartialEvaluation::outputs`] report with each folded *value* erased to a position:
    /// [`Known`](PartialEvaluationOutput::Known) entries carry the output's position among the known-side
    /// operation's outputs, and [`Unknown`](PartialEvaluationOutput::Unknown) entries keep their ordinal among the
    /// residual program's outputs.
    pub outputs: Vec<PartialEvaluationOutput<usize>>,
}

impl<T: Type, V: Value<T>, O: Operation<T>> PartitionedProgram<T, V, O> {
    /// Returns `true` when the partition's known side performs no computation: its known program (when present)
    /// contains no instructions, so every residual edge merely forwards a known boundary input. Boundary rules
    /// should fall back to the default fold-or-residualize behavior in that case — a forwarding-only known
    /// boundary operation adds a call layer without hoisting any work, while the default materializes the same
    /// known inputs directly as residual feeders.
    #[inline]
    pub fn is_trivial(&self) -> bool {
        self.known_program.as_ref().is_none_or(|program| program.instructions().is_empty())
    }

    /// Returns the type of each residual edge, in edge order, read off the residual program's input types at the
    /// [`Known`](PartialEvaluationInput::Known) source positions.
    pub fn residual_edge_types(&self) -> Vec<T> {
        self.residual_inputs
            .iter()
            .zip(self.residual_program.input_types())
            .filter_map(|(source, input_type)| matches!(source, PartialEvaluationInput::Known(_)).then_some(input_type))
            .collect()
    }

    /// Takes the residual program out of this partition, leaving an empty program in its place (the
    /// [`std::mem::take`] idiom), so the partition stays whole and its methods — most importantly
    /// [`assemble_outputs`](Self::assemble_outputs) — remain callable after the residual boundary operation has been
    /// built. Queries that read the residual program, most notably
    /// [`residual_edge_types`](Self::residual_edge_types), must run before this.
    pub fn take_residual_program(&mut self) -> Program<T, V, O, Vec<V>, Vec<V>> {
        // An empty program is trivially well-formed, so this construction can never fail.
        let empty = ProgramBuilder::new().build::<Vec<V>, Vec<V>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        std::mem::replace(&mut self.residual_program, empty)
    }

    /// Returns the original callee-output index of each fully known output, aligned with the known-side operation's
    /// leading outputs.
    pub fn known_output_indices(&self) -> Vec<usize> {
        self.outputs
            .iter()
            .enumerate()
            .filter_map(|(index, source)| matches!(source, PartialEvaluationOutput::Known(_)).then_some(index))
            .collect()
    }

    /// Returns the original callee-output index of each residual-owned output, aligned with the residual program's
    /// outputs.
    pub fn residual_output_indices(&self) -> Vec<usize> {
        self.outputs
            .iter()
            .enumerate()
            .filter_map(|(index, source)| matches!(source, PartialEvaluationOutput::Unknown(_)).then_some(index))
            .collect()
    }

    /// Returns `true` when the residual side produces at least one output, so a residual boundary operation must be
    /// emitted.
    #[inline]
    pub fn has_residual_outputs(&self) -> bool {
        self.outputs.iter().any(|source| matches!(source, PartialEvaluationOutput::Unknown(_)))
    }

    /// Returns the position of the residual edge with ordinal `edge` among the known-side operation's outputs,
    /// which list the fully known outputs first and the residual edges after them.
    fn edge_output_position(&self, edge: usize) -> usize {
        let known_output_count =
            self.outputs.iter().filter(|source| matches!(source, PartialEvaluationOutput::Known(_))).count();
        known_output_count + edge
    }

    /// Assembles the residual boundary operation's inputs: the original boundary input for each
    /// [`Unknown`](PartialEvaluationInput::Unknown) source and the known-side operation's matching residual-edge
    /// output for each [`Known`](PartialEvaluationInput::Known) source.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: Original boundary inputs, in boundary input order.
    ///   - `known_outputs`: Outputs of the known-side boundary operation bound in the enclosing known-side context.
    pub fn residual_boundary_inputs<Known: Value<T>>(
        &self,
        inputs: &[PartialEvaluationValue<T, Known>],
        known_outputs: &[PartialEvaluationValue<T, Known>],
    ) -> Result<Vec<PartialEvaluationValue<T, Known>>, ProgramError> {
        self.residual_inputs
            .iter()
            .map(|source| match source {
                PartialEvaluationInput::Unknown(index) => inputs
                    .get(*index)
                    .cloned()
                    .ok_or(ProgramError::InvalidInputCount { expected: *index + 1, actual: inputs.len() }),
                PartialEvaluationInput::Known(edge) => {
                    known_outputs.get(self.edge_output_position(*edge)).cloned().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "online boundary split known side produced no output for residual edge {edge}",
                        ))
                    })
                }
            })
            .collect()
    }

    /// Reassembles the original boundary outputs, in original output order, from the known-side and residual-side
    /// boundary operations' outputs: each [`Known`](PartialEvaluationOutput::Known) report picks the output at its
    /// position among `known_outputs` and each [`Unknown`](PartialEvaluationOutput::Unknown) report picks the
    /// output at its ordinal among `residual_outputs`.
    ///
    /// # Parameters
    ///
    ///   - `known_outputs`: Outputs of the known-side boundary operation bound in the enclosing known-side context.
    ///   - `residual_outputs`: Outputs of the residual boundary operation emitted into the enclosing residual
    ///     program.
    pub fn assemble_outputs<Known: Value<T>>(
        &self,
        known_outputs: &[PartialEvaluationValue<T, Known>],
        residual_outputs: &[PartialEvaluationValue<T, Known>],
    ) -> Result<Vec<PartialEvaluationValue<T, Known>>, ProgramError> {
        self.outputs
            .iter()
            .map(|source| match source {
                PartialEvaluationOutput::Known(index) => known_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "known program partition produced no output for known output {index}",
                    ))
                }),
                PartialEvaluationOutput::Unknown(index) => residual_outputs.get(*index).cloned().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "residual program partition produced no output for residual output {index}",
                    ))
                }),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::arithmetic::{AddOperation, MulOperation};
    use crate::operations::constants::ZeroOperation;
    use crate::operations::control_flow::ConditionOperation;
    use crate::operations::manipulation::BroadcastOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{Program, ProgramBuilder};
    use crate::scalars::Scalar;
    use crate::tests::TestArray;
    use crate::tracing::TracingContext;
    use crate::tracing_v2::ArrayOperation;
    use crate::types::{ArrayType, DataType, Shape, Size, TypeError};

    use super::*;

    type TestArrayProgram = Program<ArrayType, TestArray, ArrayOperation<TestArray>, Vec<TestArray>, Vec<TestArray>>;

    /// Eager known-side context the test operations' partial-evaluation rules are pinned to.
    type TestKnownContext = EagerContext<DataType, Scalar, TestPartialEvaluationOperation>;

    fn scalar_array_type() -> ArrayType {
        ArrayType::scalar(DataType::F64)
    }

    fn boolean_array(value: bool) -> TestArray {
        TestArray::new(ArrayType::scalar(DataType::Boolean), vec![f64::from(value as u8)])
    }

    fn scalar_branch(operation: ArrayOperation<TestArray>, factor: f64) -> TestArrayProgram {
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(scalar_array_type());
        let factor = builder.add_constant(TestArray::scalar(factor));
        let output = builder.add_instruction(operation, vec![input, factor]).unwrap()[0];
        builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    #[derive(Clone, Debug)]
    enum SymbolicOperation {
        Add,
        Mul,
    }

    impl Operation<DataType> for SymbolicOperation {
        fn name(&self) -> &'static str {
            match self {
                Self::Add => "symbolic_add",
                Self::Mul => "symbolic_mul",
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            check_count!("input", input_types, 2, TypeError);
            if input_types[0] != input_types[1] {
                return Err(TypeError {
                    message: format!(
                        "{} input types do not match: {} vs {}",
                        self.name(),
                        input_types[0],
                        input_types[1],
                    ),
                });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    #[derive(Clone, Debug)]
    enum TestPartialEvaluationOperation {
        Add(AddOperation),
        Mul(MulOperation),
        SplitConstantVariable(SplitConstantVariableOperation),
        MultiInstructionRewrite(MultiInstructionRewriteOperation),
        Inline(InlineProgramOperation),
        InvalidResidualOutput(InvalidResidualOutputOperation),
        TooManyOutputs(TooManyOutputsOperation),
    }

    impl From<AddOperation> for TestPartialEvaluationOperation {
        fn from(operation: AddOperation) -> Self {
            Self::Add(operation)
        }
    }

    impl From<MulOperation> for TestPartialEvaluationOperation {
        fn from(operation: MulOperation) -> Self {
            Self::Mul(operation)
        }
    }

    impl From<SplitConstantVariableOperation> for TestPartialEvaluationOperation {
        fn from(operation: SplitConstantVariableOperation) -> Self {
            Self::SplitConstantVariable(operation)
        }
    }

    impl From<MultiInstructionRewriteOperation> for TestPartialEvaluationOperation {
        fn from(operation: MultiInstructionRewriteOperation) -> Self {
            Self::MultiInstructionRewrite(operation)
        }
    }

    impl From<InlineProgramOperation> for TestPartialEvaluationOperation {
        fn from(operation: InlineProgramOperation) -> Self {
            Self::Inline(operation)
        }
    }

    impl From<InvalidResidualOutputOperation> for TestPartialEvaluationOperation {
        fn from(operation: InvalidResidualOutputOperation) -> Self {
            Self::InvalidResidualOutput(operation)
        }
    }

    impl From<TooManyOutputsOperation> for TestPartialEvaluationOperation {
        fn from(operation: TooManyOutputsOperation) -> Self {
            Self::TooManyOutputs(operation)
        }
    }

    impl Operation<DataType> for TestPartialEvaluationOperation {
        fn name(&self) -> &'static str {
            match self {
                Self::Add(operation) => <AddOperation as Operation<DataType>>::name(operation),
                Self::Mul(operation) => <MulOperation as Operation<DataType>>::name(operation),
                Self::SplitConstantVariable(operation) => operation.name(),
                Self::MultiInstructionRewrite(operation) => operation.name(),
                Self::Inline(operation) => operation.name(),
                Self::InvalidResidualOutput(operation) => operation.name(),
                Self::TooManyOutputs(operation) => operation.name(),
            }
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            match self {
                Self::Add(operation) => operation.infer_output_types(input_types),
                Self::Mul(operation) => operation.infer_output_types(input_types),
                Self::SplitConstantVariable(operation) => operation.infer_output_types(input_types),
                Self::MultiInstructionRewrite(operation) => operation.infer_output_types(input_types),
                Self::Inline(operation) => operation.infer_output_types(input_types),
                Self::InvalidResidualOutput(operation) => operation.infer_output_types(input_types),
                Self::TooManyOutputs(operation) => operation.infer_output_types(input_types),
            }
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for TestPartialEvaluationOperation {
        fn interpret(&self, context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            match self {
                Self::Add(operation) => operation.interpret(context, inputs),
                Self::Mul(operation) => operation.interpret(context, inputs),
                Self::SplitConstantVariable(operation) => operation.interpret(context, inputs),
                Self::MultiInstructionRewrite(operation) => operation.interpret(context, inputs),
                Self::Inline(operation) => operation.interpret(context, inputs),
                Self::InvalidResidualOutput(operation) => operation.interpret(context, inputs),
                Self::TooManyOutputs(operation) => operation.interpret(context, inputs),
            }
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for TestPartialEvaluationOperation {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<TestKnownContext>,
            inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            match self {
                Self::Add(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::Mul(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::SplitConstantVariable(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::MultiInstructionRewrite(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::Inline(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::InvalidResidualOutput(operation) => operation.partially_evaluate(evaluator, inputs),
                Self::TooManyOutputs(operation) => operation.partially_evaluate(evaluator, inputs),
            }
        }
    }

    #[derive(Clone, Debug)]
    struct SplitConstantVariableOperation;

    impl Operation<DataType> for SplitConstantVariableOperation {
        fn name(&self) -> &'static str {
            "split_constant_variable"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError { message: format!("expected 1 input but got {}", input_types.len()) });
            }
            Ok(vec![input_types[0].clone(), input_types[0].clone()])
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for SplitConstantVariableOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(vec![Scalar::from(7.0), inputs[0] + inputs[0]])
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for SplitConstantVariableOperation {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<TestKnownContext>,
            inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            let doubled = AddOperation.partially_evaluate(evaluator, &[inputs[0].clone(), inputs[0].clone()])?;
            Ok(vec![PartialEvaluationValue::known(Scalar::from(7.0)), doubled[0].clone()])
        }
    }

    #[derive(Clone, Debug)]
    struct MultiInstructionRewriteOperation;

    impl Operation<DataType> for MultiInstructionRewriteOperation {
        fn name(&self) -> &'static str {
            "multi_instruction_rewrite"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError { message: format!("expected 1 input but got {}", input_types.len()) });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for MultiInstructionRewriteOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(vec![inputs[0] * inputs[0] + inputs[0] + inputs[0]])
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for MultiInstructionRewriteOperation {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<TestKnownContext>,
            inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            let squared = MulOperation.partially_evaluate(evaluator, &[inputs[0].clone(), inputs[0].clone()])?;
            let doubled = AddOperation.partially_evaluate(evaluator, &[inputs[0].clone(), inputs[0].clone()])?;
            AddOperation.partially_evaluate(evaluator, &[squared[0].clone(), doubled[0].clone()])
        }
    }

    #[derive(Clone, Debug)]
    struct InlineProgramOperation {
        program: Program<DataType, Scalar, TestPartialEvaluationOperation, Vec<Scalar>, Vec<Scalar>>,
    }

    impl Operation<DataType> for InlineProgramOperation {
        fn name(&self) -> &'static str {
            "inline_program"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if self.program.input_types() != input_types {
                return Err(TypeError { message: "inline program input type signature mismatch".to_string() });
            }
            Ok(self.program.output_types())
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for InlineProgramOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            self.program.interpret(inputs.to_vec())
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for InlineProgramOperation {
        fn partially_evaluate(
            &self,
            evaluator: &mut PartialEvaluator<TestKnownContext>,
            inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            evaluator.inline_program(&self.program, inputs.to_vec())
        }
    }

    #[derive(Clone, Debug)]
    struct InvalidResidualOutputOperation;

    impl Operation<DataType> for InvalidResidualOutputOperation {
        fn name(&self) -> &'static str {
            "invalid_residual_output"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError { message: format!("expected 1 input but got {}", input_types.len()) });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for InvalidResidualOutputOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(vec![inputs[0]])
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for InvalidResidualOutputOperation {
        fn partially_evaluate(
            &self,
            _evaluator: &mut PartialEvaluator<TestKnownContext>,
            _inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            Ok(vec![PartialEvaluationValue::variable(DataType::F64, AtomId::new(999))])
        }
    }

    #[derive(Clone, Debug)]
    struct TooManyOutputsOperation;

    impl Operation<DataType> for TooManyOutputsOperation {
        fn name(&self) -> &'static str {
            "too_many_outputs"
        }

        fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
            if input_types.len() != 1 {
                return Err(TypeError { message: format!("expected 1 input but got {}", input_types.len()) });
            }
            Ok(vec![input_types[0].clone()])
        }
    }

    impl<C> InterpretableOperation<DataType, Scalar, C> for TooManyOutputsOperation {
        fn interpret(&self, _context: &C, inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
            check_count!("input", inputs, 1, ProgramError);
            Ok(vec![inputs[0]])
        }
    }

    impl PartiallyEvaluatableOperation<TestKnownContext> for TooManyOutputsOperation {
        fn partially_evaluate(
            &self,
            _evaluator: &mut PartialEvaluator<TestKnownContext>,
            _inputs: &[PartialEvaluationValue<DataType, Scalar>],
        ) -> Result<Vec<PartialEvaluationValue<DataType, Scalar>>, ProgramError> {
            Ok(vec![PartialEvaluationValue::known(Scalar::from(1.0)), PartialEvaluationValue::known(Scalar::from(2.0))])
        }
    }

    /// Builds `f(a, x) = (2*a*a, a*a*x, x + a)` over scalar `f64`, where `a*a` is a shared intermediate. With `a`
    /// known and `x` unknown: the first output folds to a constant, the second residualizes against the folded `a*a`
    /// (a known *intermediate*), and the third residualizes against `a` (a known *input*), exercising both kinds of
    /// residual boundary plus a fully folded output.
    #[test]
    fn test_partially_evaluate_folds_known_subcomputation_and_carves_residual() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let known_input = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let doubled_square = builder.add_instruction(AddOperation, vec![known_square, known_square]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![known_square, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![runtime_input, known_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![doubled_square, product, sum],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();

        let knowledge = vec![PartialValue::Known(Scalar::from(3.0)), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The fully known output is folded; the other two are produced by the residual program.
        match &evaluation.outputs[0] {
            PartialEvaluationOutput::Known(value) => assert_eq!(*value, 18.0),
            other => panic!("expected a folded known output but got {other:?}"),
        }
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.outputs[2], PartialEvaluationOutput::Unknown(1)));

        // The residual program drops the two folded instructions (`a*a` and `2*a*a`), keeping only the two unknown
        // ones, and takes the unknown input plus the two known residuals (the folded `a*a` and the input `a`).
        assert_eq!(program.instructions().len(), 4);
        assert_eq!(evaluation.program.instructions().len(), 2);
        assert_eq!(evaluation.inputs.len(), 3);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(value) if *value == 9.0));
        assert!(matches!(&evaluation.inputs[2], PartialEvaluationInput::Known(value) if *value == 3.0));

        // Reassembling the residual program's outputs with the folded outputs reproduces a full eager interpretation.
        let runtime_inputs = [Scalar::from(3.0), Scalar::from(5.0)];
        let residual_arguments = evaluation
            .inputs
            .iter()
            .map(|residual_input| match residual_input {
                PartialEvaluationInput::Known(value) => *value,
                PartialEvaluationInput::Unknown(original_input_index) => runtime_inputs[*original_input_index],
            })
            .collect::<Vec<_>>();
        let residual_outputs = evaluation.program.interpret(residual_arguments).unwrap();
        let reassembled = evaluation
            .outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => *value,
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index],
            })
            .collect::<Vec<_>>();

        assert_eq!(reassembled, program.interpret(runtime_inputs.to_vec()).unwrap());
        assert_eq!(reassembled, vec![18.0, 45.0, 8.0]);
    }

    /// With every input unknown, nothing folds: the residual program equals the original computation and there are no
    /// known residuals.
    #[test]
    fn test_partially_evaluate_with_all_unknown_inputs_residualizes_everything() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let left_input = builder.add_input(DataType::F64);
        let right_input = builder.add_input(DataType::F64);
        let product = builder.add_instruction(MulOperation, vec![left_input, right_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        let knowledge = vec![PartialValue::Unknown(DataType::F64), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(evaluation.inputs.iter().all(|input| matches!(input, PartialEvaluationInput::Unknown(_))));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(3.0), Scalar::from(5.0)]).unwrap(), vec![15.0]);
    }

    /// A program constant consumed by an unknown instruction must not be carried as a residual input: `filtered`
    /// rebuilds constants inline and rejects constant atoms as filter inputs. The residual program keeps the constant
    /// inside it and takes only the unknown input.
    #[test]
    fn test_partially_evaluate_keeps_program_constants_inline_in_the_residual() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let five = builder.add_constant(Scalar::from(5.0));
        let sum = builder.add_instruction(AddOperation, vec![input, five]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // Only the unknown input feeds the residual program; the constant stays inside it.
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.inputs.len(), 1);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(2.0)]).unwrap(), vec![7.0]);
    }

    /// A nullary `zero` has no inputs, so it folds to a concrete known value during partial evaluation and is dropped
    /// from the residual program. The symbolic-zero fact falls out of folding with no special handling.
    #[test]
    fn test_partially_evaluate_folds_nullary_zero_to_a_known_value() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let zero = builder.add_instruction(ZeroOperation::new(DataType::F64), vec![]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![zero, input], vec![Placeholder; 1], vec![Placeholder; 2])
            .unwrap();

        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        match &evaluation.outputs[0] {
            PartialEvaluationOutput::Known(value) => assert_eq!(*value, 0.0),
            other => panic!("expected the nullary zero to fold but got {other:?}"),
        }
        // The zero folded away; the residual program carries no instructions and just forwards the unknown input.
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 0);
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(5.0)]).unwrap(), vec![5.0]);
    }

    /// The builder forward pass emits every unknown instruction, then `into_simplified` prunes those that do not feed
    /// an output, so a dead unknown computation does not survive into the residual program.
    #[test]
    fn test_partially_evaluate_prunes_dead_unknown_instructions() {
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let one = builder.add_constant(Scalar::from(1.0));
        let two = builder.add_constant(Scalar::from(2.0));
        let used = builder.add_instruction(AddOperation, vec![input, one]).unwrap()[0];
        let _dead = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![used], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let knowledge = vec![PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // Only the live `x + 1` survives; the dead `x * 2` (and its constant) are pruned.
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(4.0)]).unwrap(), vec![5.0]);
    }

    /// An executable rule can return a folded known output and a residual output from the same source instruction.
    #[test]
    fn test_partially_evaluate_rule_returns_mixed_known_and_residual_outputs() {
        let mut builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
        let input = builder.add_input(DataType::F64);
        let outputs = builder.add_instruction(SplitConstantVariableOperation, vec![input]).unwrap().to_vec();
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(outputs, vec![Placeholder; 1], vec![Placeholder; 2])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(DataType::F64)]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Known(value) if *value == 7.0));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.inputs.len(), 1);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), TestPartialEvaluationOperation::Add(_)));

        let residual_outputs = evaluation.program.interpret(vec![Scalar::from(4.0)]).unwrap();
        let reassembled = evaluation
            .outputs
            .iter()
            .map(|output| match output {
                PartialEvaluationOutput::Known(value) => *value,
                PartialEvaluationOutput::Unknown(index) => residual_outputs[*index],
            })
            .collect::<Vec<_>>();

        assert_eq!(reassembled, program.interpret(vec![Scalar::from(4.0)]).unwrap());
        assert_eq!(reassembled, vec![Scalar::from(7.0), Scalar::from(8.0)]);
    }

    /// A rule-produced known value materializes as a residual input when a later residual instruction consumes it.
    #[test]
    fn test_partially_evaluate_materializes_rule_produced_known_values_as_residual_inputs() {
        let mut builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
        let input = builder.add_input(DataType::F64);
        let split_outputs = builder.add_instruction(SplitConstantVariableOperation, vec![input]).unwrap().to_vec();
        let output = builder.add_instruction(AddOperation, split_outputs).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(DataType::F64)]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(value) if *value == 7.0));
        assert_eq!(evaluation.program.instructions().len(), 2);
        assert_eq!(
            evaluation.program.interpret(vec![Scalar::from(4.0), Scalar::from(7.0)]).unwrap(),
            vec![Scalar::from(15.0)],
        );
    }

    /// An executable rule can rewrite one source instruction into several residual instructions by binding through
    /// the active partial evaluator.
    #[test]
    fn test_partially_evaluate_rule_emits_multiple_residual_operations() {
        let mut builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
        let input = builder.add_input(DataType::F64);
        let output = builder.add_instruction(MultiInstructionRewriteOperation, vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(DataType::F64)]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 3);
        assert!(matches!(evaluation.program.instructions()[0].operation(), TestPartialEvaluationOperation::Mul(_)));
        assert!(matches!(evaluation.program.instructions()[1].operation(), TestPartialEvaluationOperation::Add(_)));
        assert!(matches!(evaluation.program.instructions()[2].operation(), TestPartialEvaluationOperation::Add(_)));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(3.0)]).unwrap(), vec![Scalar::from(15.0)]);
        assert_eq!(
            evaluation.program.interpret(vec![Scalar::from(3.0)]).unwrap(),
            program.interpret(vec![Scalar::from(3.0)]).unwrap(),
        );
    }

    /// A rule can inline a nested program and return the inlined program's mixed known/residual output trace values.
    #[test]
    fn test_partially_evaluate_rule_inlines_nested_program_with_mixed_outputs() {
        let mut nested_builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
        let nested_input = nested_builder.add_input(DataType::F64);
        let nested_outputs =
            nested_builder.add_instruction(SplitConstantVariableOperation, vec![nested_input]).unwrap().to_vec();
        let nested_program = nested_builder
            .build::<Vec<Scalar>, Vec<Scalar>>(nested_outputs, vec![Placeholder; 1], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
        let input = builder.add_input(DataType::F64);
        let outputs = builder
            .add_instruction(InlineProgramOperation { program: nested_program }, vec![input])
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(outputs, vec![Placeholder; 1], vec![Placeholder; 2])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(DataType::F64)]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Known(value) if *value == 7.0));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), TestPartialEvaluationOperation::Add(_)));
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(5.0)]).unwrap(), vec![Scalar::from(10.0)]);
    }

    /// Broadcast intentionally uses the default primitive partial-evaluation policy. JAX has an explicit
    /// `broadcast_in_dim` partial-evaluation registration, but that rule delegates to the default primitive path; Ryft
    /// should therefore residualize an unknown broadcast unchanged instead of adding a redundant custom rule.
    #[test]
    fn test_partially_evaluate_broadcast_uses_default_primitive_policy() {
        let input_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(3)]));
        let output_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));

        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(input_type.clone());
        let output = builder.add_instruction(BroadcastOperation::new(output_type, vec![1]), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(input_type)]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Broadcast(_)));
        assert_eq!(
            evaluation.program.interpret(vec![TestArray::vector(vec![1.0, 2.0, 3.0])]).unwrap()[0].values,
            vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        );
    }

    /// Malformed executable rules are surfaced as [`ProgramError`]s by the central context walk.
    #[test]
    fn test_partially_evaluate_reports_malformed_rule_outputs() {
        let build = |operation: TestPartialEvaluationOperation| {
            let mut builder = ProgramBuilder::<DataType, Scalar, TestPartialEvaluationOperation>::new();
            let input = builder.add_input(DataType::F64);
            let output = builder.add_instruction(operation, vec![input]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![output], vec![Placeholder; 1], vec![Placeholder; 1])
                .unwrap()
        };

        let too_many_outputs = build(TestPartialEvaluationOperation::from(TooManyOutputsOperation))
            .partially_evaluate(&[PartialValue::Unknown(DataType::F64)])
            .unwrap_err();
        assert!(matches!(too_many_outputs, ProgramError::InvalidOutputCount { expected: 1, actual: 2 },));

        let invalid_residual_output = build(TestPartialEvaluationOperation::from(InvalidResidualOutputOperation))
            .partially_evaluate(&[PartialValue::Unknown(DataType::F64)])
            .unwrap_err();
        assert!(matches!(invalid_residual_output, ProgramError::UnboundAtomId { id } if id == AtomId::new(999)));
    }

    /// Stage 3 de-risking: the partial-evaluation witness must resolve for a *self-containing* operation enum.
    /// `ArrayOperation` holds `Scan`/`While`/`Condition` variants whose bodies are themselves
    /// `Program<..., ArrayOperation, ...>`, so satisfying the bound below is exactly the recursive case feared to overflow
    /// the trait solver. Because the witness's known-side context `C` is fixed across recursion and the blanket impl
    /// grounds it in the enum's own generated `PartiallyEvaluatableOperation` implementation, this compiles with no
    /// recursive obligation and no overflow — both at an eager known-side context and at a staging one.
    #[test]
    fn array_operation_satisfies_the_partial_evaluation_witness() {
        fn assert_partially_evaluatable<C: Context<Operation = O>, O: PartiallyEvaluatableProgramOperation<C>>() {}
        assert_partially_evaluatable::<
            EagerContext<ArrayType, TestArray, ArrayOperation<TestArray>>,
            ArrayOperation<TestArray>,
        >();
        assert_partially_evaluatable::<
            crate::tracing::TracingContext<ArrayType, TestArray, ArrayOperation<TestArray>>,
            ArrayOperation<TestArray>,
        >();
    }

    /// With a *known* predicate, a `condition` partially evaluates by inlining its selected branch: the condition
    /// disappears from the residual program, which then contains only the taken branch's work over the unknown
    /// input. Exercises both branches by partially evaluating the same program with a `true` and a `false`
    /// predicate.
    #[test]
    fn test_partially_evaluate_selects_branch_of_a_known_predicate_condition() {
        // `condition(p, x) = if p { x * 2 } else { x + 100 }`, staged into a flat program over `[predicate, x]`.
        let build_program = || -> TestArrayProgram {
            let condition = ConditionOperation::new(
                scalar_branch(ArrayOperation::Mul(MulOperation), 2.0),
                scalar_branch(ArrayOperation::Add(AddOperation), 100.0),
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(scalar_array_type());
            let output = builder
                .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // Known `true` predicate, unknown `x`: only the `x * 2` branch survives; the condition is gone.
        let program = build_program();
        let knowledge = vec![PartialValue::Known(boolean_array(true)), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Mul(_)));
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values, vec![8.0]);

        // Known `false` predicate, unknown `x`: only the `x + 100` branch survives.
        let program = build_program();
        let knowledge = vec![PartialValue::Known(boolean_array(false)), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Add(_)));
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(4.0)]).unwrap()[0].values, vec![104.0],);
    }

    /// With an *unknown* predicate, a `condition` cannot be inlined, so it survives and is shrunk: each branch
    /// is partially evaluated against the input knowledge and the two residual branches are reconciled into one
    /// rewritten `condition`. The branches are `if p { x * 2 } else { a * a + x * x }` over `[x, a]` with `a` known
    /// and `x` and `p` unknown, so the false branch folds `a * a` to a constant and shrinks from three to two.
    /// The rewritten condition is the only instruction in the residual program, and interpreting it for both
    /// predicates reproduces the original program.
    #[test]
    fn test_partially_evaluate_unknown_predicate_condition_shrinks_branches() {
        // True branch over `[x, a]`: `x * 2` (the `a` input is unused).
        let true_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let _known_input = builder.add_input(scalar_array_type());
            let two = builder.add_constant(TestArray::scalar(2.0));
            let output = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // False branch over `[x, a]`: `a * a + x * x`. With `a` known, `a * a` folds away during partial evaluation.
        let false_branch = || {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let input = builder.add_input(scalar_array_type());
            let known_input = builder.add_input(scalar_array_type());
            let known_square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
            let input_square = builder.add_instruction(MulOperation, vec![input, input]).unwrap()[0];
            let output = builder.add_instruction(AddOperation, vec![known_square, input_square]).unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap()
        };

        // `condition(p, x, a)` staged into a flat program over `[predicate, x, a]`.
        let build_program = || -> TestArrayProgram {
            let condition = ConditionOperation::new(true_branch(), false_branch()).unwrap();
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(scalar_array_type());
            let known_input = builder.add_input(scalar_array_type());
            let output = builder
                .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input, known_input])
                .unwrap()[0];
            builder
                .build::<Vec<TestArray>, Vec<TestArray>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };

        // Predicate and `x` unknown, `a` known: the condition survives but is rewritten over shrunk branches.
        let program = build_program();
        let knowledge = vec![
            PartialValue::Unknown(ArrayType::scalar(DataType::Boolean)),
            PartialValue::Unknown(scalar_array_type()),
            PartialValue::Known(TestArray::scalar(3.0)),
        ];
        let evaluation = program.partially_evaluate(knowledge.as_slice()).unwrap();

        // The output is produced by the residual program, whose only instruction is the rewritten condition.
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Condition(rewritten) = evaluation.program.instructions()[0].operation() else {
            panic!("expected the residual program to contain a rewritten condition");
        };

        // The false branch folded `a * a` away: it shrinks from three instructions to two, while the true branch is
        // unchanged at one. Neither reconciled branch is larger than its original.
        assert_eq!(rewritten.true_branch().instructions().len(), 1);
        assert_eq!(rewritten.false_branch().instructions().len(), 2);
        assert!(rewritten.true_branch().instructions().len() <= true_branch().instructions().len());
        assert!(rewritten.false_branch().instructions().len() <= false_branch().instructions().len());

        // Interpreting the residual program for both predicates reproduces the original program over the same inputs.
        let runtime = |predicate: bool, input: f64| -> Vec<f64> {
            let arguments = evaluation
                .inputs
                .iter()
                .map(|residual_input| match residual_input {
                    PartialEvaluationInput::Known(value) => value.clone(),
                    PartialEvaluationInput::Unknown(0) => boolean_array(predicate),
                    PartialEvaluationInput::Unknown(_) => TestArray::scalar(input),
                })
                .collect::<Vec<_>>();
            let residual_outputs = evaluation.program.interpret(arguments).unwrap();
            evaluation
                .outputs
                .iter()
                .map(|output| match output {
                    PartialEvaluationOutput::Known(value) => value.values[0],
                    PartialEvaluationOutput::Unknown(index) => residual_outputs[*index].values[0],
                })
                .collect()
        };
        let original = |predicate: bool, input: f64| {
            program
                .interpret(vec![boolean_array(predicate), TestArray::scalar(input), TestArray::scalar(3.0)])
                .unwrap()[0]
                .values
                .clone()
        };

        assert_eq!(runtime(true, 4.0), original(true, 4.0));
        assert_eq!(runtime(true, 4.0), vec![8.0]);
        assert_eq!(runtime(false, 4.0), original(false, 4.0));
        assert_eq!(runtime(false, 4.0), vec![25.0]);
    }

    /// Type alias for the staging known-side context the staging tests run under.
    type ScalarTracingContext = TracingContext<DataType, Scalar, ScalarOperation<Scalar>>;

    /// With `C` a live staging context, folding a known instruction *stages* it into the outer program instead of
    /// interpreting it, and a folded known value consumed by residual work becomes a known feeder naming the staged
    /// outer atom — the known→unknown residual edge.
    #[test]
    fn test_partially_evaluate_stages_known_work_into_a_live_outer_trace() {
        // `f(a, x) = (a * a) * x` with `a` known as an outer tracer and `x` unknown.
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let known_input = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![square, runtime_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1])
            .unwrap();

        let outer = ScalarTracingContext::new();
        let known = outer.input(DataType::F64);
        let knowledge = vec![PartialValue::Known(known), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known `a * a` landed in the outer program as a staged instruction over the outer input.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            assert_eq!(outer_builder.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(0)]);
        }

        // The residual program computes only the unknown product, over the unknown input plus the known edge, and the
        // known feeder is a live tracer naming the staged outer atom.
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        match &evaluation.inputs[1] {
            PartialEvaluationInput::Known(value) => assert_eq!(value.atom_id(), Ok(AtomId::new(1))),
            other => panic!("expected a known residual edge but got {other:?}"),
        }
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
    }

    /// A walked-program literal consumed by residual work is rebuilt inline as a residual-program constant carrying
    /// the original constant payload (recovered through the staging context's [`Concrete`](ValueResolution::Concrete)
    /// resolution), never a lifted tracer,
    /// and it never becomes a residual input.
    #[test]
    fn test_partially_evaluate_rebuilds_walked_literals_as_residual_constants_under_staging() {
        // `f(x) = x + 5` with `x` unknown: the literal `5` crosses into the residual program.
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let five = builder.add_constant(Scalar::from(5.0));
        let sum = builder.add_instruction(AddOperation, vec![input, five]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 1], vec![Placeholder; 1])
            .unwrap();

        let outer = ScalarTracingContext::new();
        let evaluation =
            program.partially_evaluate_in_context(&outer, &[PartialValue::Unknown(DataType::F64)]).unwrap();

        assert_eq!(evaluation.inputs.len(), 1);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(0)));
        // The residual program carries the constant inline with its original payload, so plain interpretation works.
        assert_eq!(evaluation.program.interpret(vec![Scalar::from(2.0)]).unwrap(), vec![7.0]);
        // Nothing folded, so the outer program stages no instructions. The walk does lift the literal into the outer
        // context on first reference, leaving one *dead* constant atom behind — the documented cost of eager lifting.
        // Deferring the lift to the first known fold would require a "known but unlifted" trace-value state, which
        // the pure known/unknown `PartialValue` contract deliberately rules out, and the standard `into_simplified`
        // passes prune the dead atom whenever the outer program is built.
        assert!(outer.builder().borrow().instructions().is_empty());
        assert_eq!(outer.builder().borrow().atoms().len(), 1);
    }

    /// An effectful operation is placed by its input known-ness like any other operation: all-known folds (under an
    /// eager known-side context this executes the effect at partial-evaluation time, which is the known side's
    /// execution time), and mixed-input residualizes with the known inputs materialized as feeders.
    #[test]
    fn test_partially_evaluate_places_effectful_operations_by_input_known_ness() {
        use crate::operations::debugging::PrintOperation;

        // `f(x) = print(x) * 2`.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(scalar_array_type());
        let two = builder.add_constant(TestArray::scalar(2.0));
        let printed = builder.add_instruction(PrintOperation::new("x"), vec![input]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![printed, two]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![product], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // All-known: the print folds (firing now) and the whole chain evaluates away.
        let evaluation = program.partially_evaluate(&[PartialValue::Known(TestArray::scalar(3.0))]).unwrap();
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Known(value) if value.values[0] == 6.0));
        assert!(evaluation.program.instructions().is_empty());
        assert!(evaluation.inputs.is_empty());

        // Unknown input: the print (and everything downstream) residualizes.
        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_array_type())]).unwrap();
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 2);
        assert_eq!(evaluation.program.interpret(vec![TestArray::scalar(3.0)]).unwrap()[0].values[0], 6.0);
    }

    /// A *dead* effectful operation over an unknown input — nothing consumes its output — still survives into the
    /// residual program (via the residual builder's effect keep-alive), alongside the residualized pure work.
    #[test]
    fn test_partially_evaluate_keeps_dead_effectful_operations_in_the_residual_program() {
        use crate::operations::debugging::PrintOperation;

        // `f(x) = x * 2`, plus a dead `print(x)`, with `x` unknown: both residualize, and only the effect keeps the
        // print alive through the residual program's simplification.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let input = builder.add_input(scalar_array_type());
        let two = builder.add_constant(TestArray::scalar(2.0));
        let product = builder.add_instruction(MulOperation, vec![input, two]).unwrap()[0];
        let _printed = builder.add_instruction(PrintOperation::new("x"), vec![input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(vec![product], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let evaluation = program.partially_evaluate(&[PartialValue::Unknown(scalar_array_type())]).unwrap();

        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(evaluation.program.instructions().len(), 2);
        assert!(evaluation.program.instructions().iter().any(|instruction| matches!(
            instruction.operation(),
            ArrayOperation::Print(operation) if operation.label() == "x",
        )));
    }

    /// Under a *staging* known-side context, an all-known effectful operation folds by staging into the live outer
    /// trace — the known side of the split, which executes first and in bind order — while a mixed-input effectful
    /// operation residualizes. Two ordered prints split across the boundary therefore execute known-first, the
    /// documented reordering the split performs on all work.
    #[test]
    fn test_partially_evaluate_stages_all_known_effectful_operations_into_a_live_outer_trace() {
        use crate::operations::debugging::PrintOperation;

        // `f(a, x) = (print(a), print(x))` with `a` known (an outer tracer) and `x` unknown.
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let known_input = builder.add_input(scalar_array_type());
        let runtime_input = builder.add_input(scalar_array_type());
        let known_print = builder.add_instruction(PrintOperation::new("a"), vec![known_input]).unwrap()[0];
        let runtime_print = builder.add_instruction(PrintOperation::new("x"), vec![runtime_input]).unwrap()[0];
        let program = builder
            .build::<Vec<TestArray>, Vec<TestArray>>(
                vec![known_print, runtime_print],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let outer = TracingContext::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let known = outer.input(scalar_array_type());
        let knowledge = vec![PartialValue::Known(known), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known print landed in the outer program; the unknown print stayed residual.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            assert!(matches!(
                outer_builder.instructions()[0].operation(),
                ArrayOperation::Print(operation) if operation.label() == "a",
            ));
        }
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(
            evaluation.program.instructions()[0].operation(),
            ArrayOperation::Print(operation) if operation.label() == "x",
        ));
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Known(value) if value.atom_id().is_ok()));
        assert!(matches!(&evaluation.outputs[1], PartialEvaluationOutput::Unknown(0)));
    }

    /// A folded known intermediate consumed by *several* residual instructions materializes as a single residual
    /// input, deduplicated by its source atom in the walked program — under a staging known-side context exactly as
    /// under an eager one.
    #[test]
    fn test_partially_evaluate_deduplicates_known_feeders_by_source_atom_under_staging() {
        // `f(a, x) = ((a * a) * x, (a * a) + x)` with `a` known (an outer tracer) and `x` unknown: the folded `a * a`
        // feeds both residual instructions and must become one residual input.
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let known_input = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![square, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![square, runtime_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product, sum], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let outer = ScalarTracingContext::new();
        let known = outer.input(DataType::F64);
        let knowledge = vec![PartialValue::Known(known), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        assert_eq!(evaluation.program.instructions().len(), 2);
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(_)));
    }

    /// Two known inputs fed by the *same* outer tracer deduplicate to a single residual input through the value's
    /// staged identity (its outer atom), even though their source atoms in the walked program differ — the
    /// walk-global counterpart of per-scope source-atom deduplication.
    #[test]
    fn test_partially_evaluate_deduplicates_known_feeders_by_staged_identity() {
        // `f(a, b, x) = (a * x, b + x)` with `a` and `b` both fed by the same outer tracer and `x` unknown.
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let first_known = builder.add_input(DataType::F64);
        let second_known = builder.add_input(DataType::F64);
        let runtime_input = builder.add_input(DataType::F64);
        let product = builder.add_instruction(MulOperation, vec![first_known, runtime_input]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![second_known, runtime_input]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product, sum], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();

        let outer = ScalarTracingContext::new();
        let known = outer.input(DataType::F64);
        let knowledge =
            vec![PartialValue::Known(known.clone()), PartialValue::Known(known), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // One unknown feeder plus exactly one known feeder for the shared outer tracer.
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(2)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(value) if value.atom_id().is_ok()));
        assert_eq!(evaluation.program.instructions().len(), 2);
    }

    /// [`PartialEvaluation::interpret`] is the single replay path for both known-side flavors: under an eager context
    /// it interprets the residual program immediately, and under a staging context it stages the residual work into
    /// the outer program and returns tracers.
    #[test]
    fn test_partial_evaluation_interpret_replays_under_eager_and_staging_contexts() {
        // `f(a, x) = (a * a) * x` with `a` known and `x` unknown.
        let build_program = || {
            let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
            let known_input = builder.add_input(DataType::F64);
            let runtime_input = builder.add_input(DataType::F64);
            let square = builder.add_instruction(MulOperation, vec![known_input, known_input]).unwrap()[0];
            let product = builder.add_instruction(MulOperation, vec![square, runtime_input]).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder; 1])
                .unwrap()
        };

        // Eager: the folded outputs and residual replay reproduce full interpretation.
        let program = build_program();
        let context = EagerContext::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let knowledge = vec![PartialValue::Known(Scalar::from(3.0)), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate_in_context(&context, knowledge.as_slice()).unwrap();
        assert_eq!(evaluation.interpret(&context, &[Scalar::from(5.0)]).unwrap(), vec![Scalar::from(45.0)]);

        // The arity check is strict in both directions: the input count must equal the number of unknown feeders,
        // so surplus values (e.g., passing the original program's full input vector) are rejected rather than
        // silently ignored.
        assert_eq!(
            evaluation.interpret(&context, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        );
        assert_eq!(
            evaluation.interpret(&context, &[Scalar::from(5.0), Scalar::from(7.0)]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        );

        // Staging: replaying stages the residual work into the outer program and returns live tracers.
        let program = build_program();
        let outer = ScalarTracingContext::new();
        let known = outer.input(DataType::F64);
        let knowledge = vec![PartialValue::Known(known), PartialValue::Unknown(DataType::F64)];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();
        let staged_before_replay = outer.builder().borrow().instructions().len();
        let tangent = outer.input(DataType::F64);
        let outputs = evaluation.interpret(&outer, &[tangent]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].atom_id().is_ok());
        assert_eq!(outer.builder().borrow().instructions().len(), staged_before_replay + 1);
    }

    /// A rule that inspects known payloads (the known-predicate `condition`) inlines only when the known predicate is
    /// concretizable (a literal-backed tracer), falling back to the rewritten condition with the predicate as a known
    /// feeder when it is a genuine outer tracer — the concretization gate.
    #[test]
    fn test_partially_evaluate_condition_predicate_concretization_gate_under_staging() {
        let build_program = || -> TestArrayProgram {
            let condition = ConditionOperation::new(
                scalar_branch(ArrayOperation::Mul(MulOperation), 2.0),
                scalar_branch(ArrayOperation::Add(AddOperation), 100.0),
            )
            .unwrap();
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
            let input = builder.add_input(scalar_array_type());
            let output = builder
                .add_instruction(ArrayOperation::Condition(Box::new(condition)), vec![predicate, input])
                .unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };

        type ArrayTracingContext = TracingContext<ArrayType, TestArray, ArrayOperation<TestArray>>;

        // A literal-backed known predicate concretizes, so the taken branch is inlined and the condition disappears.
        let program = build_program();
        let outer = ArrayTracingContext::new();
        let predicate = outer.constant(boolean_array(true));
        let knowledge = vec![PartialValue::Known(predicate), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Mul(_)));

        // A genuine outer-tracer predicate is known but not concretizable: with no known branch work to hoist, the
        // rule keeps the condition whole, with the predicate materialized as a known feeder.
        let program = build_program();
        let outer = ArrayTracingContext::new();
        let predicate = outer.input(ArrayType::scalar(DataType::Boolean));
        let knowledge = vec![PartialValue::Known(predicate), PartialValue::Unknown(scalar_array_type())];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();
        assert_eq!(evaluation.program.instructions().len(), 1);
        assert!(matches!(evaluation.program.instructions()[0].operation(), ArrayOperation::Condition(_)));
        // The unknown input's feeder is seeded before the walk, so the predicate's known feeder follows it.
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(_)));
    }

    /// A condition whose branches print inside their known chains keeps those effects *behind the outer
    /// conditional*: the per-branch fresh-context splits fold the all-known prints into the branch-local known
    /// programs, the composite split binds one known condition into the live outer trace, and the effect stays
    /// inside its own branch program — the known condition executes branch-selected, so an untaken branch's print
    /// can never fire speculatively (only its typed zero-padding for the peer's edge slots is staged). The residual
    /// condition stays pure.
    #[test]
    fn test_partially_evaluate_condition_keeps_effectful_known_work_behind_the_outer_conditional() {
        use crate::operations::debugging::PrintOperation;

        type ArrayTracingContext = TracingContext<ArrayType, TestArray, ArrayOperation<TestArray>>;

        // Branches over `[k, x]` whose known chains (`k + k` and `k * k`) print `k` first.
        let branch = |operation: ArrayOperation<TestArray>, combine: ArrayOperation<TestArray>| {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let k = builder.add_input(scalar_array_type());
            let x = builder.add_input(scalar_array_type());
            let printed = builder.add_instruction(PrintOperation::new("k"), vec![k]).unwrap()[0];
            let known = builder.add_instruction(operation, vec![printed, k]).unwrap()[0];
            let output = builder.add_instruction(combine, vec![known, x]).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };
        let condition = ConditionOperation::new(
            branch(ArrayOperation::Add(AddOperation), ArrayOperation::Mul(MulOperation)),
            branch(ArrayOperation::Mul(MulOperation), ArrayOperation::Add(AddOperation)),
        )
        .unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let known_input = builder.add_input(scalar_array_type());
        let runtime_input = builder.add_input(scalar_array_type());
        let outputs = builder
            .add_instruction(
                ArrayOperation::Condition(Box::new(condition)),
                vec![predicate, known_input, runtime_input],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 1]).unwrap();

        let outer = ArrayTracingContext::new();
        let predicate = outer.input(ArrayType::scalar(DataType::Boolean));
        let known = outer.input(scalar_array_type());
        let knowledge = vec![
            PartialValue::Known(predicate),
            PartialValue::Known(known),
            PartialValue::Unknown(scalar_array_type()),
        ];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known condition carries the branch prints (visible through the nested-program effects union) and each
        // print stays inside its own branch program; the residual condition is pure.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let ArrayOperation::Condition(known_condition) = outer_builder.instructions()[0].operation() else {
                panic!("expected the outer program to contain the known condition");
            };
            assert!(known_condition.true_branch().effects().is_ordered());
            assert!(known_condition.false_branch().effects().is_ordered());
        }
        assert!(evaluation.program.effects().is_pure());
        let residual_conditions = evaluation
            .program
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::Condition(_)))
            .count();
        assert_eq!(residual_conditions, 1);
    }

    /// With a known-but-symbolic predicate *and* known branch work to hoist, the condition splits into a composite:
    /// each branch's known work rides a known condition bound into the live outer trace — so it runs behind the
    /// *outer* conditional instead of being staged speculatively for both branches — while the unknown work stays
    /// behind a residual condition consuming the per-branch residual edges the known condition outputs.
    #[test]
    fn test_partially_evaluate_condition_composite_split_under_staging() {
        type ArrayTracingContext = TracingContext<ArrayType, TestArray, ArrayOperation<TestArray>>;

        // Branches over `[k, x]`: true computes `(k + k) * x`, false computes `(k * k) + x`, so each branch has one
        // known-only instruction (its residual edge) and one unknown instruction.
        let true_branch = {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let k = builder.add_input(scalar_array_type());
            let x = builder.add_input(scalar_array_type());
            let doubled = builder.add_instruction(AddOperation, vec![k, k]).unwrap()[0];
            let output = builder.add_instruction(MulOperation, vec![doubled, x]).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
            let k = builder.add_input(scalar_array_type());
            let x = builder.add_input(scalar_array_type());
            let squared = builder.add_instruction(MulOperation, vec![k, k]).unwrap()[0];
            let output = builder.add_instruction(AddOperation, vec![squared, x]).unwrap()[0];
            builder.build(vec![output], vec![Placeholder; 2], vec![Placeholder; 1]).unwrap()
        };
        let condition = ConditionOperation::new(true_branch, false_branch).unwrap();
        let mut builder = ProgramBuilder::<ArrayType, TestArray, ArrayOperation<TestArray>>::new();
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean));
        let known_input = builder.add_input(scalar_array_type());
        let runtime_input = builder.add_input(scalar_array_type());
        let outputs = builder
            .add_instruction(
                ArrayOperation::Condition(Box::new(condition)),
                vec![predicate, known_input, runtime_input],
            )
            .unwrap()
            .to_vec();
        let program = builder.build(outputs, vec![Placeholder; 3], vec![Placeholder; 1]).unwrap();

        let outer = ArrayTracingContext::new();
        let predicate = outer.input(ArrayType::scalar(DataType::Boolean));
        let known = outer.input(scalar_array_type());
        let knowledge = vec![
            PartialValue::Known(predicate),
            PartialValue::Known(known),
            PartialValue::Unknown(scalar_array_type()),
        ];
        let evaluation = program.partially_evaluate_in_context(&outer, knowledge.as_slice()).unwrap();

        // The known halves landed in the outer program as one known condition over `[predicate, k]`, each branch
        // producing its own residual edge plus a typed zero for the peer's edge slot.
        {
            let outer_builder = outer.builder().borrow();
            assert_eq!(outer_builder.instructions().len(), 1);
            let ArrayOperation::Condition(known_condition) = outer_builder.instructions()[0].operation() else {
                panic!("expected the outer program to contain the known condition");
            };
            assert_eq!(known_condition.true_branch().input_types().len(), 1);
            assert_eq!(known_condition.true_branch().output_types().len(), 2);
            assert_eq!(known_condition.true_branch().instructions().len(), 2);
            assert_eq!(known_condition.false_branch().output_types().len(), 2);
            assert_eq!(known_condition.false_branch().instructions().len(), 2);
        }

        // The unknown halves stayed behind one residual condition over `[predicate, x, true edge, false edge]`.
        assert_eq!(evaluation.program.instructions().len(), 1);
        let ArrayOperation::Condition(residual_condition) = evaluation.program.instructions()[0].operation() else {
            panic!("expected the residual program to contain the residual condition");
        };
        assert_eq!(residual_condition.true_branch().input_types().len(), 3);
        assert_eq!(residual_condition.true_branch().instructions().len(), 1);
        assert_eq!(residual_condition.false_branch().instructions().len(), 1);
        assert_eq!(evaluation.inputs.len(), 4);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(2)));
        assert!(matches!(&evaluation.inputs[1], PartialEvaluationInput::Known(_)));
        assert!(matches!(&evaluation.inputs[2], PartialEvaluationInput::Known(_)));
        assert!(matches!(&evaluation.inputs[3], PartialEvaluationInput::Known(_)));
        assert_eq!(evaluation.outputs.len(), 1);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
    }
}
