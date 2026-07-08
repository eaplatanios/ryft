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
use crate::types::Typed;

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial evaluator
/// interprets a [`Program`] over. Every [`Atom`] and every intermediate result is either [`Known`](Self::Known)
/// (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its [`Type`] is available until
/// the residual program runs). For more information on partial evaluation, refer to the documentation of
/// [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum PartialValue<V: Value> {
    /// [`Value`] that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// [`Value`] that is not known until the residual program runs and only its [`Type`] is known.
    Unknown(V::Type),
}

impl<V: Value> PartialValue<V> {
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

impl<V: Value> Typed for PartialValue<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
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
pub struct PartialEvaluationValue<V: Value> {
    /// Underlying [`PartialValue`] that represents the abstract known/unknown classification of the value.
    value: PartialValue<V>,

    /// [`PartialValueMaterialization`] that describes how the underlying value is represented at the residual program
    /// boundary. This is deliberately separate from the underlying [`PartialValue`] because it answers a different
    /// question. A [`Known`](PartialValue::Known) value can still be consumed by residual work, materializing as a
    /// residual input or an inline residual constant according to its [`PartialValueMaterialization`], while an
    /// [`Unknown`](PartialValue::Unknown) value is always represented by a residual program variable that already
    /// exists.
    materialization: PartialValueMaterialization,
}

impl<V: Value> PartialEvaluationValue<V> {
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
    pub fn variable(r#type: V::Type, residual_atom: AtomId) -> Self {
        Self {
            value: PartialValue::Unknown(r#type),
            materialization: PartialValueMaterialization::Variable { residual_atom },
        }
    }

    /// Returns the underlying [`PartialValue`].
    #[inline]
    pub fn value(&self) -> &PartialValue<V> {
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
}

impl<V: Value> Typed for PartialEvaluationValue<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, V::Type> {
        self.value.r#type()
    }
}

/// Input of a partially evaluated (i.e., a _residual_) [`Program`] (i.e., an input of a [`PartialEvaluation`]).
/// The residual program's inputs are the original program's surviving unknown inputs followed by the known values
/// (i.e., the residuals) that its unknown subcomputation consumes.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialEvaluationInput<V> {
    /// Residual input fed by a value that partial evaluation folded to a concrete known residual value.
    Known(V),

    /// Residual input fed by an unknown input of the original program, identified by that input's index in the
    /// original program's inputs.
    Unknown(usize),
}

impl<V> PartialEvaluationInput<V> {
    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Known`].
    #[inline]
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Unknown`].
    #[inline]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
}

/// Output of a partially evaluated (i.e., a _residual_) [`Program`] (i.e., an input of a [`PartialEvaluation`]).
/// Partial evaluation splits the original outputs into those it could fold to a concrete value now and those that
/// remain computed by the residual program.
///
/// For more information on partial evaluation, refer to the documentation of [`Program::partially_evaluate`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialEvaluationOutput<V> {
    /// Output that was folded to a concrete value during partial evaluation.
    Known(V),

    /// Output produced by the residual program, identified by its index into the residual program's outputs.
    Unknown(usize),
}

impl<V> PartialEvaluationOutput<V> {
    /// Returns `true` if this [`PartialEvaluationOutput`] is [`Self::Known`].
    #[inline]
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationOutput`] is [`Self::Unknown`].
    #[inline]
    pub const fn is_unknown(&self) -> bool {
        matches!(self, Self::Unknown(_))
    }
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
    /// Refer to the documentation of [`program`](Self::program) for more information.
    pub(crate) program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Refer to the documentation of [`inputs`](Self::inputs) for more information.
    pub(crate) inputs: Vec<PartialEvaluationInput<C::Value>>,

    /// Refer to the documentation of [`outputs`](Self::outputs) for more information.
    pub(crate) outputs: Vec<PartialEvaluationOutput<C::Value>>,
}

impl<C: Context> PartialEvaluation<C> {
    /// Returns the residual [`Program`] of this [`PartialEvaluation`], over the surviving unknown inputs plus the known
    /// residuals, aligned with [`inputs`](Self::inputs) and producing the unknown outputs in their original order.
    #[inline]
    pub fn program(&self) -> &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        &self.program
    }

    /// Returns the [`PartialEvaluationInput`]s of [`program`](Self::program), in residual program input order.
    #[inline]
    pub fn inputs(&self) -> &[PartialEvaluationInput<C::Value>] {
        &self.inputs
    }

    /// Returns the [`PartialEvaluationOutput`]s of [`program`](Self::program), in original output order.
    #[inline]
    pub fn outputs(&self) -> &[PartialEvaluationOutput<C::Value>] {
        &self.outputs
    }
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
        let residual_outputs = self.program.interpret_in_context(context, residual_inputs)?;
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

/// [`Program`] that has been partitioned into a _known_ program and a _residual_ program based on information about
/// which of its inputs are _known_. This is the result of calling [`Program::partition`]. This is typically passed to
/// [`PartialEvaluator::inline_partitioned_program`] to inline it as part of an ongoing partial evaluation transform.
pub struct PartitionedProgram<V: Value, O: Operation<V::Type>> {
    /// Refer to the documentation of [`known_program`](Self::known_program) for more information.
    pub(crate) known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`residual_program`](Self::residual_program) for more information.
    pub(crate) residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`known_input_indices`](Self::known_input_indices) for more information.
    pub(crate) known_input_indices: Vec<usize>,

    /// Refer to the documentation of [`residual_inputs`](Self::residual_inputs) for more information.
    pub(crate) residual_inputs: Vec<PartialEvaluationInput<usize>>,

    /// Refer to the documentation of [`outputs`](Self::outputs) for more information.
    pub(crate) outputs: Vec<PartialEvaluationOutput<usize>>,
}

impl<V: Value, O: Operation<V::Type>> PartitionedProgram<V, O> {
    /// Returns the known-side [`Program`] of this [`PartitionedProgram`], which represents the known work reified
    /// through a fresh trace, taking the original inputs identified by
    /// [`known_input_indices`](Self::known_input_indices) and producing the fully known outputs followed by the
    /// residual _edges_. When partial evaluation finds no fully known output and no known to unknown residual edge,
    /// this program has no outputs and (since simplification keeps only effectful dead work around) usually no
    /// instructions, in which case there is no known-side work worth wrapping in a boundary operation.
    #[inline]
    pub fn known_program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.known_program
    }

    /// Returns the residual-side [`Program`] of this [`PartitionedProgram`], which represents the callee's partial
    /// evaluation residual program, whose inputs are described by [`residual_inputs`](Self::residual_inputs).
    #[inline]
    pub fn residual_program(&self) -> &Program<V, O, Vec<V>, Vec<V>> {
        &self.residual_program
    }

    /// Returns the indices of the original program inputs feeding the known-side [`Program`]
    /// (i.e., [`known_program`](Self::known_program)), in order.
    #[inline]
    pub fn known_input_indices(&self) -> &[usize] {
        &self.known_input_indices
    }

    /// Returns the source feeding each residual [`Program`] (i.e., [`residual_program`](Self::residual_program))
    /// input, in residual program input order. This is the callee's [`PartialEvaluation::inputs`] with each feeder
    /// *value* erased to a position/index: [`Unknown`](PartialEvaluationInput::Unknown) entries keep their original
    /// boundary input index, and each [`Known`](PartialEvaluationInput::Known) feeder is erased to its residual edge
    /// ordinal which is also, offset by the fully known output count, the position of the edge among the known-side
    /// operation's outputs.
    #[inline]
    pub fn residual_inputs(&self) -> &[PartialEvaluationInput<usize>] {
        &self.residual_inputs
    }

    /// Returns the source of each original (i.e., pre-partitioning) [`Program`] output, in original output order.
    /// This is the callee's [`PartialEvaluation::outputs`] with each folded *value* erased to a position/index:
    /// [`Known`](PartialEvaluationOutput::Known) entries carry the output's position among the known-side operation's
    /// outputs, and [`Unknown`](PartialEvaluationOutput::Unknown) entries keep their ordinal among the residual
    /// program's outputs.
    #[inline]
    pub fn outputs(&self) -> &[PartialEvaluationOutput<usize>] {
        &self.outputs
    }
}

/// [`Operation`] that supports partial evaluation via [`Program::partially_evaluate`]. This trait lets an individual
/// operation decide how partial evaluation treats it. It can be implemented with an empty implementation block,
/// deferring to [`PartialEvaluator::fold_or_residualize`], which is what most operations do, or its behavior can be
/// customized by overriding the [`PartiallyEvaluatableOperation::partially_evaluate`] function.
///
/// # Type Parameters
///
///   - `C`: Known-side [`Context`] that partial evaluation folds known work through. Its
///     [`Operation`](crate::DispatchDomain::Operation) is the operation family of the residual [`Program`] and of any
///     inlined nested programs (e.g., the enum this operation may belong to). Its
///     [`Constant`](crate::DispatchDomain::Constant) is the staged constant space those programs store. Finally, its
///     [`Value`](crate::DispatchDomain::Value) is the space known values flow in (i.e., concrete values under eager
///     contexts and [`Tracer`](crate::Tracer)s into the outer program under [`StagingContext`]s).
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
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
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
        program: &Program<C::Constant, Self, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>;
}

impl<C: Context<Operation: PartiallyEvaluatableOperation<C>>> PartiallyEvaluatableProgramOperation<C> for C::Operation {
    #[inline]
    fn partially_evaluate_program(
        context: &C,
        program: &Program<C::Constant, Self, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: &[PartialValue<C::Value>],
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
    builder: ProgramBuilder<C::Constant, C::Operation>,

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
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let operation = operation.into();
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
            Ok(self.context.bind(operation, &known)?.into_iter().map(PartialEvaluationValue::known).collect())
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
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
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
        program: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<C>,
    {
        // A fresh materialization scope isolates this program's source-atom deduplication to its own atom space.
        // The walk runs inside a closure so the scope is popped on every exit path, including error paths, keeping
        // the scope stack balanced.
        self.materialization_scopes.push(vec![None; program.atoms.len()]);
        let result = (|| -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
            // Walk-time value of each atom in `program`, populated as the forward pass reaches it. Bind each
            // known input to its program atom, making that atom its materialization deduplication key.
            let mut values = vec![None; program.atoms.len()];
            for (input_id, input) in program.input_ids.iter().copied().zip(inputs) {
                values[input_id.index()] = Some(match input.value {
                    PartialValue::Known(known) => PartialEvaluationValue::known_input(known, Some(input_id)),
                    PartialValue::Unknown(_) => input,
                });
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

                // Bind each known output to its program atom, making that atom its materialization deduplication key.
                let outputs = instruction.operation().partially_evaluate(self, inputs.as_slice())?;
                check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                for (output_id, output) in instruction.outputs().iter().copied().zip(outputs) {
                    values[output_id.index()] = Some(match output.value {
                        PartialValue::Known(known) => PartialEvaluationValue::known_input(known, Some(output_id)),
                        PartialValue::Unknown(_) => output,
                    });
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

    /// Inlines a [`PartitionedProgram`] into this walk as two boundary operations, consuming the partitioned program
    /// and returning the reassembled original boundary outputs, in original output order. This is the shared emission
    /// protocol of online boundary partial-evaluation rules (i.e., the boundary-wise counterpart of the
    /// instruction-wise [`inline_program`](Self::inline_program)). The partitioned program's known [`Program`] is
    /// wrapped through `build_known_operation` and [folded-or-residualized](Self::fold_or_residualize) over the
    /// original known boundary inputs. The residual program is wrapped through `build_residual_operation` and
    /// [residualized](Self::residualize) over the surviving unknown boundary inputs plus the known-side operation's
    /// residual outputs. Each original output is picked from the known-side or residual-side operation's outputs per
    /// the partitioned program's [`outputs`](PartitionedProgram::outputs). Consuming the partitioned program in a
    /// single step keeps it whole until it is gone, so no partially-moved partition state can ever be observed.
    ///
    /// # Parameters
    ///
    ///   - `partition`: [`PartitionedProgram`] to inline, produced by [`Program::partition`].
    ///   - `inputs`: Input [`PartialEvaluationValue`]s in the order of the original program's input, pre-partitioning.
    ///   - `build_known_operation`: Wraps the provided known [`Program`] in the known-side boundary operation.
    ///   - `build_residual_operation`: Wraps the provided residual [`Program`] in the residual boundary operation.
    pub fn inline_partitioned_program<
        V: Value<Type = C::Type>,
        O: Operation<C::Type>,
        P: Into<C::Operation>,
        BuildKnownProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> P,
        BuildResidualProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> P,
    >(
        &mut self,
        program: PartitionedProgram<V, O>,
        inputs: &[PartialEvaluationValue<C::Value>],
        build_known_operation: BuildKnownProgramOperation,
        build_residual_operation: BuildResidualProgramOperation,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Bind the known-side operation into the known-side context over the original known inputs.
        let known_inputs = program
            .known_input_indices
            .iter()
            .map(|&index| {
                inputs
                    .get(index)
                    .cloned()
                    .ok_or(ProgramError::InvalidInputCount { expected: index + 1, actual: inputs.len() })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let known_program_operation = build_known_operation(program.known_program);
        let known_outputs = self.fold_or_residualize(known_program_operation, known_inputs.as_slice())?;

        // Emit the residual operation over the surviving unknown boundary inputs plus the residual edges, which trail
        // the fully known outputs among the known-side operation's outputs. The emission is unconditional: a residual
        // program without outputs can still carry effectful residual instructions whose effects must be preserved, and
        // an entirely empty residual program only yields a dead pure operation that the walk's final simplification
        // removes.
        let known_output_count = program.outputs.iter().filter(|output| output.is_known()).count();
        let residual_inputs = program
            .residual_inputs
            .iter()
            .map(|source| match source {
                PartialEvaluationInput::Unknown(index) => inputs
                    .get(*index)
                    .cloned()
                    .ok_or(ProgramError::InvalidInputCount { expected: *index + 1, actual: inputs.len() }),
                PartialEvaluationInput::Known(index) => {
                    known_outputs.get(known_output_count + index).cloned().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "known program partition produced no output for residual known input index {index}",
                        ))
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        let residual_program_operation = build_residual_operation(program.residual_program);
        let residual_outputs = self.residualize(residual_program_operation, residual_inputs.as_slice())?;

        // Reassemble the original outputs from the two operations' outputs.
        program
            .outputs
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
    pub fn any_known_is_symbolic(&self, inputs: &[PartialEvaluationValue<C::Value>]) -> bool {
        inputs.iter().any(|input| match input.value() {
            PartialValue::Known(value) => !self.context.resolve(value).is_concrete(),
            PartialValue::Unknown(_) => false,
        })
    }
}

impl<V: Value, O: Operation<V::Type>> Program<V, O, Vec<V>, Vec<V>> {
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
        inputs: &[PartialValue<V>],
    ) -> Result<PartialEvaluation<EagerContext<V, O>>, ProgramError>
    where
        O: InterpretableOperation<V, EagerContext<V, O>> + PartiallyEvaluatableOperation<EagerContext<V, O>>,
    {
        self.partially_evaluate_in_context(&EagerContext::new(), inputs)
    }

    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work through
    /// the provided known-side [`Context`]. This is the context-taking core behind
    /// [`partially_evaluate`](Self::partially_evaluate).
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        &self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
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
    pub fn partition(&self, input_known: &[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        let input_types = self.input_types();
        check_count!("input", input_known, input_types.len(), ProgramError);

        let context = TracingContext::<V, O>::new();
        let inputs = input_types
            .iter()
            .zip(input_known.iter())
            .map(|(input_type, &known)| match known {
                true => PartialValue::Known(context.input(input_type.clone())),
                false => PartialValue::Unknown(input_type.clone()),
            })
            .collect::<Vec<_>>();
        let evaluation = self.partially_evaluate_in_context(&context, inputs.as_slice())?;

        let known_input_indices = input_known
            .iter()
            .enumerate()
            .filter_map(|(index, &known)| known.then_some(index))
            .collect::<Vec<_>>();

        let known_output_atoms = evaluation
            .outputs
            .iter()
            .filter_map(|output| match output {
                PartialEvaluationOutput::Known(value) => Some(value.atom_id()),
                PartialEvaluationOutput::Unknown(_) => None,
            })
            .chain(evaluation.inputs.iter().filter_map(|input| match input {
                PartialEvaluationInput::Known(value) => Some(value.atom_id()),
                PartialEvaluationInput::Unknown(_) => None,
            }))
            .collect::<Result<Vec<_>, _>>()?;

        let residual_inputs = evaluation
            .inputs
            .iter()
            .scan(0, |known_count, input| {
                Some(match input {
                    PartialEvaluationInput::Unknown(index) => PartialEvaluationInput::Unknown(*index),
                    PartialEvaluationInput::Known(_) => {
                        let index = *known_count;
                        *known_count += 1;
                        PartialEvaluationInput::Known(index)
                    }
                })
            })
            .collect::<Vec<_>>();

        let outputs = evaluation
            .outputs
            .iter()
            .scan(0, |known_count, output| {
                Some(match output {
                    PartialEvaluationOutput::Known(_) => {
                        let index = *known_count;
                        *known_count += 1;
                        PartialEvaluationOutput::Known(index)
                    }
                    PartialEvaluationOutput::Unknown(ordinal) => PartialEvaluationOutput::Unknown(*ordinal),
                })
            })
            .collect::<Vec<_>>();

        let known_input_count = known_input_indices.len();
        let known_output_count = known_output_atoms.len();
        let known_program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<V>, Vec<V>>(
                known_output_atoms,
                vec![Placeholder; known_input_count],
                vec![Placeholder; known_output_count],
            )?
            .into_simplified()?;
        let residual_program = evaluation.program;

        Ok(PartitionedProgram { known_program, residual_program, known_input_indices, residual_inputs, outputs })
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::contexts::{Context, StagingContext};
    use crate::operations::arithmetic::{AddOperation, MulOperation, NegOperation};
    use crate::operations::constants::ConstantOperation;
    use crate::operations::debugging::PrintOperation;
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::SinOperation;
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
    use crate::scalars::{Scalar, ScalarTracingContext};
    use crate::types::DataType;

    use super::*;

    #[test]
    fn test_partial_evaluation() {
        // Build a residual program `g(x, r) = x * r + 3`, with `x` standing for the original program's surviving
        // unknown input and `r` for a known residual feeder carrying the folded value `2`, and pair it with an
        // original output report whose first output folded to `5`.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let x = builder.add_input(DataType::F64);
        let r = builder.add_input(DataType::F64);
        let c = builder.add_constant(Scalar::from(3.0));
        let product = builder.add_instruction(MulOperation, vec![x, r]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![product, c]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = PartialEvaluation::<EagerContext<Scalar, ScalarOperation<Scalar>>> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Scalar::from(2.0))],
            outputs: vec![PartialEvaluationOutput::Known(Scalar::from(5.0)), PartialEvaluationOutput::Unknown(0)],
        };

        // Interpretation takes exactly one value per `Unknown` feeder, feeds `Known` feeders from their carried values,
        // returns folded outputs directly, and reads the rest from the replayed residual program:
        // `(5, 4 * 2 + 3) = (5, 11)`.
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Scalar::from(4.0)]),
            Ok(vec![Scalar::from(5.0), Scalar::from(11.0)]),
        );
        assert!(matches!(
            evaluation.interpret(&context, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            evaluation.interpret(&context, &[Scalar::from(4.0), Scalar::from(5.0)]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        ));

        // An output that references a residual output the residual program does not produce is reported as a
        // malformed program.
        let evaluation = PartialEvaluation::<EagerContext<Scalar, ScalarOperation<Scalar>>> {
            program: program.clone(),
            inputs: evaluation.inputs,
            outputs: vec![PartialEvaluationOutput::Unknown(1)],
        };
        assert!(matches!(
            evaluation.interpret(&context, &[Scalar::from(4.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message == "partial evaluation output references residual output 1 but the residual program \
                    produced 1 output(s)",
        ));

        // Under a staging known-side context, the same replay stages the residual program into the outer trace
        // instead of executing it. Its constant is lifted as a staged constant, its instructions are staged as outer
        // instructions, folded outputs return their tracers directly, and residual outputs are tracers naming the
        // staged atoms.
        let outer = ScalarTracingContext::new();
        let folded = outer.input(DataType::F64);
        let unknown = outer.input(DataType::F64);
        let feeder = outer.constant(Scalar::from(2.0));
        let evaluation = PartialEvaluation::<ScalarTracingContext> {
            program,
            inputs: vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(feeder)],
            outputs: vec![PartialEvaluationOutput::Known(folded.clone()), PartialEvaluationOutput::Unknown(0)],
        };
        let outputs = evaluation.interpret(&outer, &[unknown]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].atom_id(), folded.atom_id());
        let staged = outputs[1].atom_id().unwrap();
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![staged], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = const
                    %3:f64 = const
                    %4:f64 = mul %1 %2
                    %5:f64 = add %4 %3
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluator() {
        let mut evaluator = PartialEvaluator::new(EagerContext::<Scalar, ScalarOperation<Scalar>>::new());
        assert_eq!(
            evaluator.context().bind(AddOperation, &[Scalar::from(1.0), Scalar::from(2.0)]),
            Ok(vec![Scalar::from(3.0)]),
        );

        // `fold_or_residualize` folds an all-known operation through the known-side context, and so its outputs are
        // known values with no residual materialization decision yet.
        let inputs =
            [PartialEvaluationValue::known(Scalar::from(2.0)), PartialEvaluationValue::known(Scalar::from(3.0))];
        let folded = evaluator.fold_or_residualize(MulOperation, &inputs).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_known());
        assert!(!folded[0].is_unknown());
        assert_eq!(folded[0].as_known(), Some(&Scalar::from(6.0)));
        assert_eq!(folded[0].materialization(), PartialValueMaterialization::Undecided);
        assert_eq!(folded[0].r#type().into_owned(), DataType::F64);
        assert!(matches!(folded[0].value(), PartialValue::Known(value) if *value == Scalar::from(6.0)));

        // `residualize` emits the operation into the residual program, materializing each known input as a fresh
        // residual input (i.e., atoms 0 and 1) and returning the instruction output as a residual variable
        // (i.e., atom 2).
        let residual = evaluator.residualize(AddOperation, &inputs).unwrap();
        assert_eq!(residual.len(), 1);
        assert!(residual[0].is_unknown());
        assert_eq!(residual[0].as_known(), None);
        assert_eq!(residual[0].r#type().into_owned(), DataType::F64);
        assert_eq!(
            residual[0].materialization(),
            PartialValueMaterialization::Variable { residual_atom: AtomId::new(2) },
        );

        // `fold_or_residualize` residualizes as soon as any input is unknown. `neg` lands in the residual program
        // over the residual variable, producing the next residual atom.
        let mixed = evaluator.fold_or_residualize(NegOperation, &[residual[0].clone()]).unwrap();
        assert_eq!(mixed[0].materialization(), PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) });

        // Materializing a known value keyed by a source atom requires an active materialization scope,
        // which only `inline_program` pushes.
        assert!(matches!(
            evaluator.residualize(
                NegOperation,
                &[PartialEvaluationValue::known_input(Scalar::from(1.0), Some(AtomId::new(0)))],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "partial evaluation materialization has no active scope",
        ));

        // `inline_program` walks a program over seed values. All-known seeds fold every instruction, lifting the
        // program constant into the known-side context on first use, and so the walk returns folded values.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let c = builder.add_constant(Scalar::from(1.0));
        let product = builder.add_instruction(MulOperation, vec![a, x]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![product, c]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outputs = evaluator
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Scalar::from(2.0)),
                    PartialEvaluationValue::known(Scalar::from(3.0)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Scalar::from(7.0)));

        // Mixed seeds fold the known work and residualize the rest, and so the walk returns residual variables.
        let outputs = evaluator
            .inline_program(&program, vec![PartialEvaluationValue::known(Scalar::from(2.0)), residual[0].clone()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        // `inline_partitioned_program` inlines a `Program::partition` result as two boundary operations. The known
        // side folds through the known-side context, its trailing residual-edge output feeds the residual boundary
        // operation, and the original outputs are reassembled from the two operations' outputs. The partitioned sides
        // of `f(a, x) = sin(a) * x` are a single `sin` and a single `mul`, and so those operations themselves serve as
        // arity-matching boundary operations.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let sine = builder.add_instruction(SinOperation, vec![a]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![sine, x]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let outputs = evaluator
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Scalar::from(2.0)), residual[0].clone()],
                |_known_program| ScalarOperation::Sin(SinOperation),
                |_residual_program| ScalarOperation::Mul(MulOperation),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        // An all-known partitioned program folds entirely through the known-side boundary operation, and so the
        // reassembled outputs are known values even though the (empty) residual operation is still emitted.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let sine = builder.add_instruction(SinOperation, vec![a]).unwrap()[0];
        let program =
            builder.build::<Vec<Scalar>, Vec<Scalar>>(vec![sine], vec![Placeholder], vec![Placeholder]).unwrap();
        let partition = program.partition(&[true]).unwrap();
        let outputs = evaluator
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Scalar::from(2.0))],
                |_known_program| ScalarOperation::Sin(SinOperation),
                |_residual_program| ScalarOperation::Constant(ConstantOperation::new(Scalar::from(0.0))),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Scalar::from(2.0f64.sin())));

        // `known_constant` recovers a known value's staged-constant payload. An eager known value is always concrete,
        // while under a staging known-side context only literal-backed tracers concretize.
        assert_eq!(evaluator.known_constant(&Scalar::from(5.0)), Ok(Scalar::from(5.0)));
        let staging = ScalarTracingContext::new();
        let staging_evaluator = PartialEvaluator::new(staging.clone());
        let symbolic = staging.input(DataType::F64);
        let literal = staging.constant(Scalar::from(4.0));
        assert_eq!(staging_evaluator.known_constant(&literal), Ok(Scalar::from(4.0)));
        assert!(matches!(
            staging_evaluator.known_constant(&symbolic),
            Err(ProgramError::MalformedProgram(message))
                if message == "a known value crossing into a nested residual program is not concretizable in the \
                    active known-side context",
        ));

        // `all_knowns_are_concrete` checks every known feeder and folded output of a partial evaluation, which is only
        // non-trivial under a staging known-side context where knowns can be live tracers.
        let empty = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new()
            .build::<Vec<Scalar>, Vec<Scalar>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert!(evaluator.all_knowns_are_concrete(
            &PartialEvaluation::<EagerContext<Scalar, ScalarOperation<Scalar>>> {
                program: empty.clone(),
                inputs: vec![PartialEvaluationInput::Known(Scalar::from(1.0)), PartialEvaluationInput::Unknown(0)],
                outputs: vec![PartialEvaluationOutput::Known(Scalar::from(2.0))],
            }
        ));
        assert!(!staging_evaluator.all_knowns_are_concrete(&PartialEvaluation::<ScalarTracingContext> {
            program: empty.clone(),
            inputs: vec![PartialEvaluationInput::Known(symbolic.clone())],
            outputs: Vec::new(),
        }));
        assert!(staging_evaluator.all_knowns_are_concrete(&PartialEvaluation::<ScalarTracingContext> {
            program: empty,
            inputs: vec![PartialEvaluationInput::Known(literal.clone())],
            outputs: Vec::new(),
        }));

        // `any_known_is_symbolic` is the signal online boundary rules split on. Only a known value that does not
        // resolve to a concrete constant counts, and so eager knowns and unknowns never do.
        assert!(!evaluator.any_known_is_symbolic(&[PartialEvaluationValue::known(Scalar::from(1.0))]));
        assert!(!staging_evaluator.any_known_is_symbolic(&[PartialEvaluationValue::known(literal)]));
        assert!(staging_evaluator.any_known_is_symbolic(&[PartialEvaluationValue::known(symbolic)]));
        assert!(
            !staging_evaluator
                .any_known_is_symbolic(&[PartialEvaluationValue::variable(DataType::F64, AtomId::new(0))]),
        );
    }

    #[test]
    fn test_program_partially_evaluate() {
        // `f(a, x) = (a * a, a * a * x + 1, a * a + x)` with `a` known and `x` unknown: the `a * a` subcomputation
        // folds to a known output, its two residual consumers share one residual feeder, and the literal is rebuilt
        // inline as a residual constant instead of becoming a feeder.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let c = builder.add_constant(Scalar::from(1.0));
        let squared = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation, vec![squared, x]).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation, vec![scaled, c]).unwrap()[0];
        let offset = builder.add_instruction(AddOperation, vec![squared, x]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![squared, shifted, offset],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Scalar::from(3.0)), PartialValue::Unknown(DataType::F64)])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Scalar::from(9.0)),]
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Scalar::from(9.0)),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %1 %0
                    %3:f64 = const
                    %4:f64 = add %2 %3
                    %5:f64 = add %1 %0
                in (%4, %5)
            "}
            .trim_end(),
        );

        // Replaying the partial evaluation at a concrete unknown input matches interpreting the original program.
        assert_eq!(
            evaluation.interpret(&EagerContext::<Scalar, ScalarOperation<Scalar>>::new(), &[Scalar::from(4.0)]),
            Ok(vec![Scalar::from(9.0), Scalar::from(37.0), Scalar::from(13.0)]),
        );
        assert_eq!(
            program.interpret(vec![Scalar::from(3.0), Scalar::from(4.0)]),
            Ok(vec![Scalar::from(9.0), Scalar::from(37.0), Scalar::from(13.0)]),
        );

        // All-known inputs fold the whole program away: every output is known and the residual program is empty.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Scalar::from(3.0)), PartialValue::Known(Scalar::from(4.0))])
            .unwrap();
        assert_eq!(evaluation.inputs, Vec::new());
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Scalar::from(9.0)),
                PartialEvaluationOutput::Known(Scalar::from(37.0)),
                PartialEvaluationOutput::Known(Scalar::from(13.0)),
            ]
        );
        assert!(evaluation.program.instructions().is_empty());

        // All-unknown inputs residualize the whole program unchanged, with the literal rebuilt inline at its first
        // residual use rather than up front.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Unknown(DataType::F64), PartialValue::Unknown(DataType::F64)])
            .unwrap();
        assert_eq!(evaluation.inputs, vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Unknown(1)]);
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
                PartialEvaluationOutput::Unknown(2),
            ]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %0
                    %3:f64 = mul %2 %1
                    %4:f64 = const
                    %5:f64 = add %3 %4
                    %6:f64 = add %2 %1
                in (%2, %5, %6)
            "}
            .trim_end(),
        );

        // Effectful operations place by input known-ness. An all-known `print` folds (firing its effect at partial
        // evaluation time), while a mixed-input `print` residualizes and is kept in the residual program even when
        // no output consumes it.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let printed = builder.add_instruction(PrintOperation::new("known"), vec![a]).unwrap()[0];
        builder.add_instruction(PrintOperation::new("dead"), vec![x]).unwrap();
        let product = builder.add_instruction(MulOperation, vec![printed, x]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Scalar::from(2.0)), PartialValue::Unknown(DataType::F64)])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Scalar::from(2.0)),]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = print [label=dead] %0
                    %3:f64 = mul %1 %0
                in (%3)
            "}
            .trim_end(),
        );

        // The number of provided inputs must match the number of program inputs.
        assert!(matches!(
            program.partially_evaluate(&[PartialValue::Known(Scalar::from(1.0))]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        ));
    }

    #[test]
    fn test_program_partially_evaluate_in_context() {
        // `f(a, x) = (a * a) * x + 1` with `a` known as a live tracer of an enclosing trace and `x` unknown: the known
        // `a * a` folds by staging into the outer program, the residual program consumes its staged result through a
        // known feeder naming the outer atom, and the literal is rebuilt inline as a residual constant.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let c = builder.add_constant(Scalar::from(1.0));
        let squared = builder.add_instruction(MulOperation, vec![a, a]).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation, vec![squared, x]).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation, vec![scaled, c]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![shifted], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = ScalarTracingContext::new();
        let known = outer.input(DataType::F64);
        let evaluation = program
            .partially_evaluate_in_context(&outer, &[PartialValue::Known(known), PartialValue::Unknown(DataType::F64)])
            .unwrap();

        // The known feeder is a tracer naming the staged `a * a` atom of the outer program.
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(
            &evaluation.inputs[1],
            PartialEvaluationInput::Known(feeder) if feeder.atom_id() == Ok(AtomId::new(1)),
        ));
        assert_eq!(evaluation.outputs.len(), 1);
        assert!(matches!(&evaluation.outputs[0], PartialEvaluationOutput::Unknown(0)));
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %1 %0
                    %3:f64 = const
                    %4:f64 = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // The outer trace accumulated the folded known work plus the lifted literal, which stays dead in the outer
        // trace because the residual program rebuilds it inline.
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![AtomId::new(1)], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = const
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_partition() {
        // `f(a, x) = (a + a, sin(a) * x)` partitioned with `a` known and `x` unknown: `a + a` is a fully known output,
        // `sin(a)` is a residual edge trailing it among the known program's outputs, and the residual program computes
        // the mixed output over the surviving unknown input plus the edge.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let doubled = builder.add_instruction(AddOperation, vec![a, a]).unwrap()[0];
        let sine = builder.add_instruction(SinOperation, vec![a]).unwrap()[0];
        let product = builder.add_instruction(MulOperation, vec![sine, x]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled, product], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(partition.known_input_indices, vec![0]);
        assert_eq!(
            partition.residual_inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(0),]
        );
        assert_eq!(partition.outputs, vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            partition.known_program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = add %0 %0
                    %2:f64 = sin %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            partition.residual_program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );

        // The two sides recombine to the original program: interpret the known program at `a`, feed its trailing
        // residual-edge output to the residual program together with `x`, and interleave per the outputs report.
        let known_outputs = partition.known_program.interpret(vec![Scalar::from(2.0)]).unwrap();
        let residual_outputs = partition.residual_program.interpret(vec![Scalar::from(3.0), known_outputs[1]]).unwrap();
        assert_eq!(known_outputs[0], Scalar::from(4.0));
        assert_eq!(residual_outputs, vec![Scalar::from(3.0 * 2.0f64.sin())]);

        // All-unknown known-ness produces an empty known program and residualizes everything.
        let partition = program.partition(&[false, false]).unwrap();
        assert_eq!(partition.known_input_indices, Vec::<usize>::new());
        assert!(partition.known_program.instructions().is_empty());
        assert!(partition.known_program.output_ids().is_empty());
        assert_eq!(
            partition.residual_inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Unknown(1),]
        );
        assert_eq!(partition.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);

        // All-known known-ness folds everything into the known program and leaves an empty residual program.
        let partition = program.partition(&[true, true]).unwrap();
        assert_eq!(partition.known_input_indices, vec![0, 1]);
        assert_eq!(partition.residual_inputs, Vec::new());
        assert_eq!(partition.outputs, vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Known(1)]);
        assert!(partition.residual_program.instructions().is_empty());

        // The provided known-ness must cover every program input.
        assert!(matches!(program.partition(&[true]), Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 })));
    }
}
