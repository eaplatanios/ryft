//! Contains machinery for _partially evaluating_ [`Program`]s into known work and residual programs.
//!
//! Partial evaluation runs the portions of a program whose inputs are available now and stages the remaining work for
//! later. It is both a public program-partitioning transform and infrastructure for other transforms (most notably
//! differentiation, where primals are known, tangents are unknown, and the residual tangent program is a reusable
//! linearization).
//!
//! ```text
//!        ┌───────────────────────────────────┐
//!        │ Program + Known / Unknown Inputs  │
//!        └─────────────────┬─────────────────┘
//!                          │ bind each operation
//!                          ▼
//!           ┌─────────────────────────────┐
//!           │ Partial Evaluation Context  │
//!           └──────────────┬──────────────┘
//!           ┌──────────────┴──────────────┐
//!   all inputs known              any input unknown
//!           │ evaluate now                │ apply the operation's rule
//!           ▼                             ▼
//!    ┌─────────────┐      ┌───────────────────────────────┐
//!    │ Known Value │      │ Residual Instruction + Tracer │
//!    └──────┬──────┘      └───────────────┬───────────────┘
//!           └──────────────┬──────────────┘
//!                          ▼
//!     ┌─────────────────────────────────────────┐
//!     │ Known Outputs + Residual Program Wiring │
//!     └─────────────────────────────────────────┘
//! ```
//!
//! # Entry Points
//!
//! [`Program::partially_evaluate`] is the eager convenience entry point for a flat program. Supply one [`PartialValue`]
//! per input. Known values are evaluated immediately and unknown inputs are represented only by type. The result is a
//! [`PartialEvaluation`] containing the residual program, its runtime input wiring, and output descriptors.
//!
//! [`Program::partially_evaluate_in_context`] performs the same transform through an explicitly supplied known-side
//! [`Context`]. This matters when known work should itself be staged into an enclosing trace rather than executed
//! concretely. [`Program::partition`] is the index-oriented convenience that returns a [`PartitionedProgram`] with
//! separate known and residual programs plus the wiring between them.
//!
//! # Values and Materialization
//!
//! [`PartialValue`] carries only semantic known/unknown classification: `Known(V)` contains a value available
//! now, while `Unknown(Type)` carries abstract metadata for a future value. [`PartialEvaluationValue`] adds the
//! residual-boundary state needed while building the split. Its shared [`PartialValueMaterialization`] slot records
//! whether the logical value becomes a residual input, an embedded constant, or a residual variable and remembers the
//! assigned residual atom. Clones share this slot, so the first materialization establishes one residual identity and
//! later consumers reuse it. [`PartialEvaluationInput`] describes how each residual input is supplied (i.e., from an
//! original unknown input, a known value forwarded across the boundary, or other recorded boundary state).
//! [`PartialEvaluationOutput`] distinguishes outputs already known after the first stage from outputs produced
//! by the residual program.
//!
//! # Context and Tracer
//!
//! [`PartialEvaluationContext`] wraps the context in which known work runs and owns the residual [`ProgramBuilder`].
//! Its bind protocol applies a [`PartiallyEvaluatableOperation`] rule, which may fold an operation when all inputs are
//! known, residualize it when future data is required, or apply operation-specific splitting logic.
//!
//! [`PartialTracer`] is the flowing value exposed to interpreted or traced closures. It contains a
//! [`PartialEvaluationValue`] while live and propagates poisoning after an error. Known values can participate in
//! host control flow only when the parent context resolves them as concrete; symbolic or opaque knowns require a
//! conservative residual rewrite.
//!
//! # Effects and Nested Programs
//!
//! Effectful operations follow the same placement rule as pure operations: all-known operations run on the known side,
//! while mixed or unknown work is residualized. Probe-based rewrites of higher-order programs must not speculatively
//! execute effectful bodies, and residual projections keep explicitly requested effectful atoms alive.
//!
//! [`PartiallyEvaluatableProgramOperation`] is the recursive fixed point for operation families containing nested
//! flat programs. Higher-order operations may split their bodies more precisely than the default, but their replay
//! and residualization logic stays with the operation that owns those programs.
//!
//! # Extending Partial Evaluation
//!
//! Implement [`PartiallyEvaluatableOperation`] for primitive operation payloads. The usual rule is to fold when every
//! input is known and to residualize otherwise, but control flow, loops, scans, and other higher-order operations can
//! preserve more known work with a dedicated rule. Use the supplied context's materialization and residualization APIs
//! rather than constructing duplicate boundary atoms manually. Implement [`PartiallyEvaluatableProgramOperation`] on an
//! operation family that recursively evaluates nested programs. Rules inspecting known payloads must first establish a
//! concrete [`ValueResolution`] and fall back conservatively otherwise.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::operations::Operation;
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::tracing::TracingContext;
use crate::types::Typed;

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial context
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
/// [`PartialValueMaterialization`] records how that value is represented at the residual boundary. Each materialization
/// lives in a slot shared by every clone of one logical [`PartialEvaluationValue`], and so the residual
/// [`Atom`](crate::Atom) assigned when a known value is first materialized (as a residual input or an inline residual
/// constant) is visible to every later consumer of the same value, which reuses that atom instead of materializing the
/// value again. By contrast, [`Variable`](Self::Variable) values were *created* in the residual program and so always
/// carry their residual atom.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PartialValueMaterialization {
    /// Known value with no residual materialization decision yet. If residual work depends on it, the corresponding
    /// [`PartialEvaluationContext`] will materialize it as a fresh residual input.
    Undecided,

    /// Known value that should be materialized as a residual program input.
    Input {
        /// Residual input atom assigned when this value was first materialized, if it was, so that later consumers
        /// of the same value reuse it. When absent, the value has not been materialized yet, and the first residualized
        /// consumer creates a fresh residual input and records it here.
        residual_atom: Option<AtomId>,
    },

    /// Known value that should be materialized as an inline residual program constant.
    Constant {
        /// Residual constant atom assigned when this value was first materialized, if it was, so that later consumers
        /// of the same value reuse it. When absent, the value has not been materialized yet, and the first residualized
        /// consumer creates a fresh residual constant and records it here.
        residual_atom: Option<AtomId>,
    },

    /// Unknown value already represented as a residual program variable.
    Variable {
        /// Atom in the residual program that carries this value. Residual operations consume it directly, and so it
        /// is not optional.
        residual_atom: AtomId,
    },
}

/// Represents the [`Value`] type used by [`PartialEvaluationContext`]s while partially evaluating [`Program`]s.
#[derive(Clone)]
pub struct PartialEvaluationValue<V: Value> {
    /// Underlying [`PartialValue`] that represents the abstract known/unknown classification of the value.
    value: PartialValue<V>,

    /// [`PartialValueMaterialization`] that describes how the underlying value is represented at the residual program
    /// boundary. This is deliberately separate from the underlying [`PartialValue`] because it answers a different
    /// question. A [`Known`](PartialValue::Known) value can still be consumed by residual work, materializing as a
    /// residual input or an inline residual constant according to its [`PartialValueMaterialization`], while an
    /// [`Unknown`](PartialValue::Unknown) value is always represented by a residual program variable that already
    /// exists. The slot is shared via [`Rc`] across every clone of this value, so that the residual atom assigned by
    /// the first materialization is reused by every other residualized consumer, which is what deduplicates residual
    /// inputs and inline constants without keying on source-program atoms. Furthermore, the [`Cell`] supplies the
    /// interior mutability that this lazy assignment needs. The residual atom is recorded at *first residual use*,
    /// long after the value has been cloned and shared, and so the write must go through `&self`. Because
    /// [`PartialValueMaterialization`] is a small [`Copy`] value, [`Cell`] suffices without [`RefCell`]'s borrow
    /// tracking.
    materialization: Rc<Cell<PartialValueMaterialization>>,
}

impl<V: Value> PartialEvaluationValue<V> {
    /// Creates a known [`PartialEvaluationValue`] with [`PartialValueMaterialization::Undecided`].
    #[inline]
    pub fn known(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Undecided)),
        }
    }

    /// Creates a known [`PartialEvaluationValue`] with an unassigned [`PartialValueMaterialization::Input`].
    #[inline]
    pub fn known_input(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Input { residual_atom: None })),
        }
    }

    /// Creates a known [`PartialEvaluationValue`] with an unassigned [`PartialValueMaterialization::Constant`].
    #[inline]
    pub fn known_constant(value: V) -> Self {
        Self {
            value: PartialValue::Known(value),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Constant { residual_atom: None })),
        }
    }

    /// Creates an unknown [`PartialEvaluationValue`] with [`PartialValueMaterialization::Variable`].
    #[inline]
    pub fn variable(r#type: V::Type, residual_atom: AtomId) -> Self {
        Self {
            value: PartialValue::Unknown(r#type),
            materialization: Rc::new(Cell::new(PartialValueMaterialization::Variable { residual_atom })),
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
        self.materialization.get()
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

impl<V: Value> Debug for PartialEvaluationValue<V> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PartialEvaluationValue")
            .field("value", &self.value)
            .field("materialization", &self.materialization.get())
            .finish()
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
/// [`PartialEvaluationContext::inline_partitioned_program`] to inline it as part of an ongoing partial evaluation
/// transform.
pub struct PartitionedProgram<V: Value, O: Operation<V::Type>> {
    /// Refer to the documentation of [`known_program`](Self::known_program) for more information.
    known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`residual_program`](Self::residual_program) for more information.
    residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`known_input_indices`](Self::known_input_indices) for more information.
    known_input_indices: Vec<usize>,

    /// Refer to the documentation of [`residual_inputs`](Self::residual_inputs) for more information.
    residual_inputs: Vec<PartialEvaluationInput<usize>>,

    /// Refer to the documentation of [`outputs`](Self::outputs) for more information.
    outputs: Vec<PartialEvaluationOutput<usize>>,
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

    /// Consumes this [`PartitionedProgram`] and returns its [`known_program`](Self::known_program),
    /// [`residual_program`](Self::residual_program), [`known_input_indices`](Self::known_input_indices),
    /// [`residual_inputs`](Self::residual_inputs), and [`outputs`](Self::outputs), in that order.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        Program<V, O, Vec<V>, Vec<V>>,
        Program<V, O, Vec<V>, Vec<V>>,
        Vec<usize>,
        Vec<PartialEvaluationInput<usize>>,
        Vec<PartialEvaluationOutput<usize>>,
    ) {
        (self.known_program, self.residual_program, self.known_input_indices, self.residual_inputs, self.outputs)
    }
}

/// [`Operation`] that supports partial evaluation via [`Program::partially_evaluate`]. This trait lets an individual
/// operation decide how partial evaluation treats it. It can be implemented with an empty implementation block,
/// deferring to [`PartialEvaluationContext::fold_or_residualize`], which is what most operations do, or its behavior
/// can be customized by overriding the [`PartiallyEvaluatableOperation::partially_evaluate`] function.
///
/// # Type Parameters
///
///   - `C`: Known-side [`Context`] that partial evaluation folds known work through. Its
///     [`Operation`](crate::DispatchDomain::Operation) is the operation family of the residual [`Program`] and of any
///     inlined nested programs (e.g., the enum this operation may belong to). Its
///     [`Constant`](crate::DispatchDomain::Constant) is the staged constant space those programs store. Finally, its
///     [`Value`](crate::DispatchDomain::Value) is the space known values flow in (i.e., concrete values under eager
///     contexts and [`Tracer`](crate::Tracer)s into the outer program under [`StagingContext`]s).
///
/// # Deriving Partially Evaluatable Operation Enums
///
/// The `#[derive(Operation)]` macro generates a [`PartiallyEvaluatableOperation`] implementation for operation enums
/// that forwards every variant to its payload's own rule, and extra value capabilities that recursive payload rules
/// require can be supplied with `#[ryft(bounds(partial_evaluation(Bound1 + Bound2 + ...)))]`. Refer to the
/// documentation of [`Operation`] for information on how to use that macro and on the shape of the generated code.
pub trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
    /// Partially evaluates this [`PartiallyEvaluatableOperation`] for the provided [`PartialEvaluationValue`]s. Unless
    /// overridden, this function will default to calling [`PartialEvaluationContext::fold_or_residualize`] which uses
    /// the following semantics:
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
    /// [`Known`](PartialValue::Known) and concretizable may ask the context to inline the selected branch and return
    /// that branch's output trace values, so that the condition disappears from the residual program and only the taken
    /// branch's work survives. Rules that inspect known *payloads* must gate that inspection on a
    /// [`Concrete`](ValueResolution::Concrete) [`Context::resolve`] resolution because a known value under a staging
    /// known-side context is a [`Tracer`](crate::Tracer) into the outer program rather than a concrete value, and
    /// partial evaluation should fall back to a conservative rewrite otherwise.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`PartialEvaluationContext`] that owns residual emission, inlining, and materialization.
    ///   - `inputs`: [`PartialEvaluationValue`] for each of this operation's inputs, in input order.
    #[inline]
    fn partially_evaluate(
        &self,
        context: &PartialEvaluationContext<C>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        context.fold_or_residualize(self.clone(), inputs)
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

/// [`Context`] which is used for _partial evaluation_, serving both as the engine behind the program-replay entry
/// points (i.e., [`Program::partially_evaluate`] and [`Program::partially_evaluate_in_context`]) and as a [`Context`]
/// in its own right, so that closures and transform interpreters (e.g., forward-mode differentiation) can drive partial
/// evaluation directly by [`bind`](Context::bind)ing operations over [`PartialTracer`]s. A [`PartialEvaluationContext`]
/// carries out partial evaluation by folding known subcomputations through the known-side inner [`Context`] `C` (where
/// folding [`bind`](Context::bind)s the [`Operation`] in that context, which interprets it immediately under an eager
/// context and stages it into the outer program under a staging context) and accumulating the unknown subcomputation in
/// a residual [`ProgramBuilder`]. [`bind`](Context::bind) dispatches each operation's
/// [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, which defaults to
/// [`fold_or_residualize`](Self::fold_or_residualize). Like [`TracingContext`], the mutable state lives behind
/// per-field `Rc<RefCell<…>>` handles so that cloning the context keeps every clone accumulating into the same residual
/// program, and rules receive `&self` and can freely re-enter the context (e.g., a known-predicate `condition` rule
/// inlining its selected branch through [`inline_program`](Self::inline_program)).
pub struct PartialEvaluationContext<C: Context> {
    /// Known-side parent [`Context`] used to fold [`Instruction`](crate::Instruction)s whose inputs are all known.
    parent: C,

    /// [`ProgramBuilder`] accumulating the residual program's [`Atom`](crate::Atom)s and
    /// [`Instruction`](crate::Instruction)s. Shared across clones of this context (and held behind a [`RefCell`] for
    /// interior mutability) for the same reason that [`TracingContext`] shares its builder: values stamped with cloned
    /// contexts must keep accumulating into the *same* residual program.
    builder: Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>>,

    /// [`PartialEvaluationInput`]s for the residual program, in residual program input order.
    /// This is shared across clones like the [`builder`](Self::builder).
    inputs: Rc<RefCell<Vec<PartialEvaluationInput<C::Value>>>>,

    /// Map that is used for deduplication of residual *input* feeders by the known value's *staged* identity
    /// (i.e., the outer program [`Atom`](crate::Atom) a known value names when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in a staging known-side context), mapping the staged atom to the
    /// residual input already created for it. This complements the per-value shared [`PartialValueMaterialization`]
    /// slots along the axis that value-identity deduplication cannot reach: two *distinct* known values (with distinct
    /// slots) naming the same outer atom collapse to one residual input, even when rule-produced. It holds only under a
    /// *staging* known-side context because an eager context resolves knowns as [`Concrete`](ValueResolution::Concrete)
    /// rather than [`Staged`](ValueResolution::Staged), and so nothing is ever recorded in that case, and inline
    /// constants are excluded because they carry no staged identity.
    staged_feeders: Rc<RefCell<HashMap<AtomId, AtomId>>>,
}

impl<C: Context> PartialEvaluationContext<C> {
    /// Creates a fresh [`PartialEvaluationContext`] that folds known work through `context` and accumulates
    /// residual work in a new residual [`ProgramBuilder`].
    #[inline]
    pub fn new(parent: C) -> Self {
        Self {
            parent,
            builder: Rc::new(RefCell::new(ProgramBuilder::new())),
            inputs: Rc::new(RefCell::new(Vec::new())),
            staged_feeders: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    /// Returns the known-side parent [`Context`] of this [`PartialEvaluationContext`] which is used
    /// to fold known subcomputations.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Creates a fresh unknown value backed by a new residual-program input of the provided type, recorded as an
    /// [`Unknown`](PartialEvaluationInput::Unknown) feeder carrying `index`. This is how drivers seed the unknown
    /// inputs of an evaluation. The program-replay driver behind [`Program::partially_evaluate_in_context`] seeds one
    /// per unknown program input (with the original program input index as the ordinal), and closure drivers seed one
    /// per traced unknown (e.g., one tangent input per primal in linearization), in order.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`Type`](crate::Type) of the unknown value.
    ///   - `index`: Index recorded in the resulting [`Unknown`](PartialEvaluationInput::Unknown) feeder, which
    ///     [`PartialEvaluation::interpret`] uses to align runtime values with unknown feeders.
    #[inline]
    pub fn unknown_input(&self, r#type: C::Type, index: usize) -> PartialEvaluationValue<C::Value> {
        let atom = self.builder.borrow_mut().add_input(r#type.clone());
        self.inputs.borrow_mut().push(PartialEvaluationInput::Unknown(index));
        PartialEvaluationValue::variable(r#type, atom)
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
        &self,
        operation: P,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let operation = operation.into();
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
            Ok(self.parent.bind(operation, &known)?.into_iter().map(PartialEvaluationValue::known).collect())
        } else {
            self.residualize(operation, inputs)
        }
    }

    /// _Residualizes_ the provided [`Operation`] into the residual [`Program`], materializing each known input into
    /// a residual program [`Atom`](crate::Atom) according to its [`PartialValueMaterialization`], and returns the
    /// operation's outputs as [`PartialEvaluationValue`]s, in output order. Materializing a known value deduplicates
    /// it two ways so a value consumed by several residualized [`Instruction`](crate::Instruction)s yields one
    /// residual input (or inline constant): through the value's shared [`PartialValueMaterialization`] slot, which
    /// records the residual atom assigned on first materialization and is visible to every clone of the value, and,
    /// for inputs, by its *staged* identity across the whole evaluation when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in the known-side context. A
    /// [`Constant`](PartialValueMaterialization::Constant) materialization is only ever attached to values that
    /// originated as literals (i.e., replayed-program constants lifted into the known-side context, or rule-produced
    /// [`known_constant`](PartialEvaluationValue::known_constant) values), and so recovering its payload through
    /// [`Context::resolve`] is expected to succeed. This is what keeps the residual program in the staged-constant
    /// space, since under a staging known-side context a known value is a [`Tracer`](crate::Tracer) that can never
    /// itself be a residual-program constant.
    pub fn residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Materialize each known input into a residual-program atom. The deduplication fast-paths return early,
        // and a genuine error rides `?` out through the `collect` into `residualize`.
        let input_atoms = inputs
            .iter()
            .map(|input| -> Result<AtomId, ProgramError> {
                // A residual variable is already a residual atom, and an already-materialized known value's shared
                // slot carries the residual atom assigned on first materialization. Every other known value differs
                // only in whether it materializes as an inline constant.
                let constant = match input.materialization() {
                    PartialValueMaterialization::Undecided => false,
                    PartialValueMaterialization::Input { residual_atom: None } => false,
                    PartialValueMaterialization::Input { residual_atom: Some(atom) } => return Ok(atom),
                    PartialValueMaterialization::Constant { residual_atom: None } => true,
                    PartialValueMaterialization::Constant { residual_atom: Some(atom) } => return Ok(atom),
                    PartialValueMaterialization::Variable { residual_atom } => return Ok(residual_atom),
                };

                let known = input.as_known().ok_or_else(|| {
                    ProgramError::MalformedProgram(
                        "residual materialization marked an unknown value as a known residual".to_string(),
                    )
                })?;

                // Inputs (but not inline constants) additionally deduplicate across the whole evaluation by the value's
                // staged identity in the known-side context. Reuse the residual input already created for it, if any.
                let staged_atom = if constant {
                    None
                } else {
                    match self.parent.resolve(known) {
                        ValueResolution::Staged(atom) => Some(atom),
                        _ => None,
                    }
                };

                // Reuse the residual input already registered for this staged identity, or create a fresh residual
                // constant (recovering the literal payload) or residual input and register it under that identity.
                let existing = staged_atom.and_then(|staged| self.staged_feeders.borrow().get(&staged).copied());
                let atom = match existing {
                    Some(existing) => existing,
                    None => {
                        let atom = if constant {
                            let constant = self.parent.resolve(known).into_concrete().ok_or_else(|| {
                                ProgramError::MalformedProgram(
                                    "residual materialization required a constant payload for a known value that is \
                                     not concretizable in the active known-side context"
                                        .to_string(),
                                )
                            })?;
                            self.builder.borrow_mut().add_constant(constant)
                        } else {
                            let atom = self.builder.borrow_mut().add_input(known.r#type().into_owned());
                            self.inputs.borrow_mut().push(PartialEvaluationInput::Known(known.clone()));
                            atom
                        };
                        if let Some(staged_atom) = staged_atom {
                            self.staged_feeders.borrow_mut().insert(staged_atom, atom);
                        }
                        atom
                    }
                };

                // Record the assignment in the value's shared slot so that every clone of this value reuses the same
                // residual atom instead of materializing the value again.
                input.materialization.set(match constant {
                    true => PartialValueMaterialization::Constant { residual_atom: Some(atom) },
                    false => PartialValueMaterialization::Input { residual_atom: Some(atom) },
                });
                Ok(atom)
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;

        let output_atoms = self.builder.borrow_mut().add_instruction(operation, input_atoms)?.to_vec();
        let builder = self.builder.borrow();
        Ok(output_atoms
            .into_iter()
            .map(|atom| {
                let r#type = builder.atoms()[atom.index()].r#type().into_owned();
                PartialEvaluationValue::variable(r#type, atom)
            })
            .collect())
    }

    /// Replays the provided [`Program`] through this context using the provided `inputs` bound to its input
    /// [`Atom`](crate::Atom)s in input order, and returns the replay value of each program output, in output order.
    /// The replay is [`Program::interpret_with`] instantiated at this context's protocol (i.e., the same one
    /// [`bind`](Context::bind) applies operation by operation). Each live program constant is [`lift`](Context::lift)ed
    /// into the known-side context as an inline-constant known (rebuilt in the residual program through
    /// [`Context::resolve`] if residual work consumes it), and each [`Instruction`](crate::Instruction) dispatches to
    /// its [`PartiallyEvaluatableOperation::partially_evaluate`] implementation, folding all-known work through the
    /// known-side [`Context`] and emitting residual work into this context's residual [`ProgramBuilder`].
    /// [`Operation`]-specific rules can call this function to recursively replay nested programs over selected inputs,
    /// so that an operation can rewrite itself into transformed work. For example, a known-predicate `condition` can
    /// inline its selected branch.
    pub fn inline_program(
        &self,
        program: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation: PartiallyEvaluatableOperation<C>,
    {
        program.interpret_with(
            inputs,
            |_, constant| Ok(PartialEvaluationValue::known_constant(self.parent.lift(constant.clone())?)),
            |instruction, inputs| instruction.operation().partially_evaluate(self, inputs),
        )
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
        &self,
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
        self.parent.resolve(value).into_concrete().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "a known value crossing into a nested residual program is not concretizable in the active \
                 known-side context"
                    .to_string(),
            )
        })
    }

    /// Returns `true` if every [`Known`](PartialEvaluationInput::Known) residual input and
    /// [`Known`](PartialEvaluationOutput::Known) output of the provided [`PartialEvaluation`] resolves to a concrete
    /// constant in the known-side [`Context`] of this [`PartialEvaluationContext`] (i.e., if a nested program rebuild
    /// that embeds those knowns as inline program constants through [`Self::known_constant`] can succeed). Under a
    /// staging known-side context, a probe's folds can produce known values that are genuine tracers into the live
    /// trace (e.g., a constant-only chain staged by the fold). Rules that rebuild nested programs from a live context
    /// probe must check this and fall back to a conservative rewrite when it returns `false`.
    #[inline]
    pub fn all_knowns_are_concrete(&self, evaluation: &PartialEvaluation<C>) -> bool {
        evaluation.inputs.iter().all(|input| match input {
            PartialEvaluationInput::Known(value) => self.parent.resolve(value).is_concrete(),
            PartialEvaluationInput::Unknown(_) => true,
        }) && evaluation.outputs.iter().all(|output| match output {
            PartialEvaluationOutput::Known(value) => self.parent.resolve(value).is_concrete(),
            PartialEvaluationOutput::Unknown(_) => true,
        })
    }

    /// Returns `true` when any of the provided `inputs` is known but does not [`resolve`](Context::resolve)
    /// to a [`Concrete`](ValueResolution::Concrete) constant in the known-side [`Context`] of this
    /// [`PartialEvaluationContext`] (i.e., it is a genuine [`Tracer`](crate::Tracer) into a live outer trace). This
    /// is the signal online boundary rules split on: all-concrete knowledge keeps the default fold-or-residualize
    /// behavior.
    #[inline]
    pub fn any_known_is_symbolic(&self, inputs: &[PartialEvaluationValue<C::Value>]) -> bool {
        inputs.iter().any(|input| match input.value() {
            PartialValue::Known(value) => !self.parent.resolve(value).is_concrete(),
            PartialValue::Unknown(_) => false,
        })
    }

    /// Consumes this [`PartialEvaluationContext`] and finalizes it into a [`PartialEvaluation`] whose outputs are the
    /// provided evaluation values: known outputs fold to their carried values, unknown outputs become the residual
    /// program's outputs in order, and the accumulated residual program is built and simplified. This is the shared
    /// epilogue of every partial-evaluation driver (the program-replay entry points and the closure-driven trace).
    /// Finalization recovers sole ownership of the accumulated residual state, and so every clone of this context
    /// (e.g., contexts stamped on [`PartialTracer`]s) must have been dropped by the time this is called; otherwise
    /// this returns [`ProgramError::EscapedProgramBuilder`], mirroring [`TracingContext`]'s trace boundary.
    ///
    /// # Parameters
    ///
    ///   - `outputs`: [`PartialEvaluationValue`] of each original output, in original output order.
    pub fn into_evaluation(
        self,
        outputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        // Assemble outputs. Folded values return directly and residual values index the residual program's outputs.
        let mut evaluation_outputs = Vec::with_capacity(outputs.len());
        let mut residual_output_atoms: Vec<AtomId> = Vec::new();
        for output in outputs {
            let materialization = output.materialization();
            match output.value {
                PartialValue::Known(value) => evaluation_outputs.push(PartialEvaluationOutput::Known(value)),
                PartialValue::Unknown(_) => {
                    let PartialValueMaterialization::Variable { residual_atom } = materialization else {
                        return Err(ProgramError::MalformedProgram(
                            "partial evaluation produced an unknown output without a residual atom".to_string(),
                        ));
                    };
                    evaluation_outputs.push(PartialEvaluationOutput::Unknown(residual_output_atoms.len()));
                    residual_output_atoms.push(residual_atom);
                }
            }
        }

        let output_count = residual_output_atoms.len();
        let inputs = Rc::try_unwrap(self.inputs).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let builder = Rc::try_unwrap(self.builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder
            .build::<Vec<C::Constant>, Vec<C::Constant>>(
                residual_output_atoms,
                vec![Placeholder; inputs.len()],
                vec![Placeholder; output_count],
            )?
            .into_simplified()?;
        Ok(PartialEvaluation { program, inputs, outputs: evaluation_outputs })
    }
}

impl<C: Context> Clone for PartialEvaluationContext<C> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            builder: self.builder.clone(),
            inputs: self.inputs.clone(),
            staged_feeders: self.staged_feeders.clone(),
        }
    }
}

impl<C: Context> Domain for PartialEvaluationContext<C> {
    type Type = C::Type;
    type Value = PartialTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context<Operation: PartiallyEvaluatableOperation<C>>> Context for PartialEvaluationContext<C> {
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<PartialTracer<C>, ProgramError> {
        // Lifting a staged constant produces a known value carrying an inline-constant materialization, so that
        // residual work consuming it rebuilds it as a residual-program constant rather than a residual input.
        Ok(PartialTracer::new(self.clone(), PartialEvaluationValue::known_constant(self.parent.lift(constant)?)))
    }

    fn bind<O: Into<C::Operation>>(
        &self,
        operation: O,
        inputs: &[PartialTracer<C>],
    ) -> Result<Vec<PartialTracer<C>>, ProgramError> {
        // Unwrap the input tracers into context-free partial-evaluation values, dispatch the operation's partial
        // evaluation rule against those, and rewrap the produced values with this context, mirroring how
        // `DifferentiationContext::bind` unwraps to `DifferentiationDual`s and rewraps.
        let operation = operation.into();
        let input_values = inputs.iter().map(|input| input.value()).collect::<Result<Vec<_>, _>>();
        let error = match input_values {
            Ok(input_values) => {
                let input_values = input_values.into_iter().cloned().collect::<Vec<_>>();
                match operation.partially_evaluate(self, input_values.as_slice()) {
                    Ok(outputs) => {
                        return Ok(outputs.into_iter().map(|value| PartialTracer::new(self.clone(), value)).collect());
                    }
                    Err(error) => error,
                }
            }
            // A poisoned input means an earlier bind already failed and deferred. Propagate its error.
            Err(error) => error,
        };

        // Defer the error by poisoning the outputs (propagating it value to value until an evaluation boundary reports
        // it), so the infallible operator sugar driving closures over this context never observes a failed bind and
        // never panics. The poisoned outputs are typed by output type inference. When even that fails, the output arity
        // is unknowable and the error surfaces immediately instead.
        let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let Ok(output_types) = operation.infer_output_types(input_types.as_slice()) else {
            return Err(error);
        };

        Ok(output_types
            .into_iter()
            .map(|r#type| PartialTracer::poisoned(self.clone(), error.clone(), r#type))
            .collect())
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // A partial-evaluation context is eager exactly when its known-side inner context is: known values are then
        // concrete, so concretizing extractions (e.g., branching on a known predicate) succeed on the known side,
        // while unknown (residual) values never concretize regardless of the inner context.
        self.parent.is_eager()
    }

    #[inline]
    fn resolve(&self, value: &PartialTracer<C>) -> ValueResolution<C::Constant> {
        // A known value resolves exactly as the known-side inner context resolves its payload (concrete under an eager
        // inner context, staged for live tracers of an enclosing trace), while an unknown value is opaque: it names a
        // residual program variable whose value does not exist until the residual program runs.
        match &value.state {
            PartialTracerState::Live(value) => match value.value() {
                PartialValue::Known(known) => self.parent.resolve(known),
                PartialValue::Unknown(_) => ValueResolution::Opaque,
            },
            PartialTracerState::Poison { .. } => ValueResolution::Opaque,
        }
    }
}

/// State carried by a [`PartialTracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`PartialEvaluationValue`], or a *poison* recording an error that a failed [`bind`](Context::bind) deferred
/// mirroring [`Tracer`](crate::Tracer)'s poison state. Because the closures driven through a
/// [`PartialEvaluationContext`] use infallible operator sugar with no deferral point of their own,
/// [`bind`](Context::bind) turns its errors into poisoned outputs (and propagates poison from inputs to outputs), so
/// the original error surfaces as a plain `Err` at the evaluation boundary instead of panicking mid-closure. Unlike
/// [`Tracer`](crate::Tracer)'s poison, the deferred [`ProgramError`] itself is carried, so boundaries report the
/// original failure rather than a generic poison error.
#[derive(Clone)]
pub enum PartialTracerState<C: Context> {
    /// The corresponding [`PartialTracer`] is _live_ and has a corresponding [`PartialEvaluationValue`].
    Live(PartialEvaluationValue<C::Value>),

    /// The corresponding [`PartialTracer`] has been _poisoned_, meaning that it corresponds to an error and
    /// will propagate that error wherever it is used (i.e., it will _poison_ those corresponding downstream
    /// [`PartialTracer`]s too).
    Poison {
        /// [`ProgramError`] that the failed bind deferred.
        error: ProgramError,

        /// [`Type`](crate::Type) of the output the failed bind would have produced.
        r#type: C::Type,
    },
}

/// Value flowing through [`PartialEvaluationContext`]s. This is a [`PartialEvaluationValue`] stamped with the context
/// it flows through, so that closures and transform interpreters can drive partial evaluation directly (it is the
/// closure-facing counterpart of the program-replay driver behind [`Program::partially_evaluate_in_context`]). A known
/// [`PartialTracer`] carries a concrete known-side value (i.e., a concrete value under an eager known-side inner
/// context, and a [`Tracer`](crate::Tracer) into the enclosing trace under a staging one), so concretizing extractions
/// such as [`BooleanLike::boolean`](crate::BooleanLike::boolean) succeed on it exactly when they succeed on the carried
/// value. This is what lets host control flow branch on known values while partial evaluation is in progress. An
/// unknown [`PartialTracer`] names a residual program variable and carries only its type.
#[derive(Clone)]
pub struct PartialTracer<C: Context> {
    /// [`PartialEvaluationContext`] this value flows through, used to dispatch [`Operation`]s that involve it.
    context: PartialEvaluationContext<C>,

    /// [`PartialTracerState`] of this [`PartialTracer`].
    state: PartialTracerState<C>,
}

impl<C: Context> PartialTracer<C> {
    /// Creates a new live [`PartialTracer`] from a context-free [`PartialEvaluationValue`] and the
    /// [`PartialEvaluationContext`] it flows through.
    #[inline]
    pub fn new(context: PartialEvaluationContext<C>, value: PartialEvaluationValue<C::Value>) -> Self {
        Self { context, state: PartialTracerState::Live(value) }
    }

    /// Creates a poisoned [`PartialTracer`] deferring the provided error. Refer to the documentation of
    /// [`PartialTracerState`] for more information.
    #[inline]
    fn poisoned(context: PartialEvaluationContext<C>, error: ProgramError, r#type: C::Type) -> Self {
        Self { context, state: PartialTracerState::Poison { error, r#type } }
    }

    /// Returns the [`PartialEvaluationContext`] this value flows through.
    #[inline]
    pub fn context(&self) -> &PartialEvaluationContext<C> {
        &self.context
    }

    /// Returns the underlying context-free [`PartialEvaluationValue`], or the deferred error if this value
    /// is poisoned.
    #[inline]
    pub fn value(&self) -> Result<&PartialEvaluationValue<C::Value>, ProgramError> {
        match &self.state {
            PartialTracerState::Live(value) => Ok(value),
            PartialTracerState::Poison { error, .. } => Err(error.clone()),
        }
    }

    /// Consumes this value and returns the underlying context-free [`PartialEvaluationValue`], or the deferred error
    /// if this value is poisoned.
    #[inline]
    pub fn into_value(self) -> Result<PartialEvaluationValue<C::Value>, ProgramError> {
        match self.state {
            PartialTracerState::Live(value) => Ok(value),
            PartialTracerState::Poison { error, .. } => Err(error),
        }
    }
}

// `PartialTracer` equality is *value identity* and not payload equality. Two values are equal if and only if they are
// clones of one logical partial-evaluation value )witnessed by sharing one materialization slot). Two values that would
// evaluate to equal payloads but were produced separately are considered unequal, which is the conservative answer
// analyses such as the scan/while loop-invariance fixed points of partial evaluation need (they degrade to passthrough
// detection, mirroring `Tracer`'s staging-identity `PartialEq`).
impl<C: Context> PartialEq for PartialTracer<C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (&self.state, &other.state) {
            (PartialTracerState::Live(left), PartialTracerState::Live(right)) => {
                Rc::ptr_eq(&left.materialization, &right.materialization)
            }
            // Poisoned values never compare equal: equality answers identity questions for analyses such as the
            // loop-invariance probes, and a deferred error has no identity to assert.
            _ => false,
        }
    }
}

impl<C: Context> Debug for PartialTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            PartialTracerState::Live(value) => formatter.debug_struct("PartialTracer").field("value", value).finish(),
            PartialTracerState::Poison { error, r#type } => {
                formatter.debug_struct("PartialTracer").field("error", error).field("type", r#type).finish()
            }
        }
    }
}

impl<C: Context> Display for PartialTracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let value = match &self.state {
            PartialTracerState::Live(value) => value,
            PartialTracerState::Poison { r#type, .. } => return write!(formatter, "<poison:{}>", r#type),
        };
        match (value.value(), value.materialization()) {
            (PartialValue::Known(value), _) => write!(formatter, "{value}"),
            (PartialValue::Unknown(_), PartialValueMaterialization::Variable { residual_atom }) => {
                write!(formatter, "{residual_atom}")
            }
            (PartialValue::Unknown(r#type), _) => write!(formatter, "<unknown:{}>", r#type),
        }
    }
}

impl<C: Context> Typed for PartialTracer<C> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        match &self.state {
            PartialTracerState::Live(value) => value.r#type(),
            PartialTracerState::Poison { r#type, .. } => Cow::Borrowed(r#type),
        }
    }
}

impl<C: Context> Parameter for PartialTracer<C> {}

impl<C: Context> Value for PartialTracer<C> {
    type DispatchDomain = PartialEvaluationContext<C>;
    type ExecutionDomain = PartialEvaluationContext<C>;

    #[inline]
    fn dispatch_domain(&self) -> PartialEvaluationContext<C> {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> PartialEvaluationContext<C> {
        self.context().clone()
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
    /// For example, a `condition` with a concretizable known predicate calls
    /// [`PartialEvaluationContext::inline_program`] to inline its selected branch in place of the operation, so that
    /// the condition disappears from the residual program. Building the residual program with a [`ProgramBuilder`]
    /// (rather than projecting the original) is what lets these rules emit *transformed* work; flat instructions with
    /// no override are emitted unchanged. The walk is flat per program but can recurse through operation rules into
    /// inlined nested programs, such as a selected `condition` branch; an instruction carrying a nested program that
    /// is *not* inlined is folded only when all of its inputs are known and is otherwise emitted unchanged.
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
        let context = PartialEvaluationContext::new(context.clone());
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, knowledge) in inputs.iter().enumerate() {
            match knowledge {
                PartialValue::Known(value) => seed.push(PartialEvaluationValue::known_input(value.clone())),
                PartialValue::Unknown(r#type) => seed.push(context.unknown_input(r#type.clone(), index)),
            }
        }

        // Replay this program through the context and finalize the accumulated residual state.
        let outputs = context.inline_program(self, seed)?;
        context.into_evaluation(outputs)
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

    use crate::backends::scalars::{Scalar, ScalarOperation, ScalarTracingContext};
    use crate::contexts::{Context, StagingContext};
    use crate::operations::BooleanLike;
    use crate::operations::constants::{ConstantOperation, Zero};
    use crate::operations::debugging::PrintOperation;
    use crate::operations::math::{AddOperation, MulOperation, NegOperation, SinOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
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
    fn test_partial_evaluation_context() {
        let context = PartialEvaluationContext::new(EagerContext::<Scalar, ScalarOperation<Scalar>>::new());
        assert_eq!(
            context.parent().bind(AddOperation, &[Scalar::from(1.0), Scalar::from(2.0)]),
            Ok(vec![Scalar::from(3.0)]),
        );

        // `fold_or_residualize` folds an all-known operation through the known-side context, and so its outputs are
        // known values with no residual materialization decision yet.
        let inputs =
            [PartialEvaluationValue::known(Scalar::from(2.0)), PartialEvaluationValue::known(Scalar::from(3.0))];
        let folded = context.fold_or_residualize(MulOperation, &inputs).unwrap();
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
        let residual = context.residualize(AddOperation, &inputs).unwrap();
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
        let mixed = context.fold_or_residualize(NegOperation, &[residual[0].clone()]).unwrap();
        assert_eq!(mixed[0].materialization(), PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) });

        // Materializing the same known value twice reuses the residual atom assigned on first materialization through
        // the value's shared materialization slot, so a value consumed by several residualized instructions yields a
        // single residual input.
        let shared = PartialEvaluationValue::known_input(Scalar::from(4.0));
        let first = context.residualize(NegOperation, &[shared.clone()]).unwrap();
        let second = context.residualize(SinOperation, &[shared.clone()]).unwrap();
        assert_eq!(
            shared.materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(AtomId::new(4)) }
        );
        assert!(first[0].is_unknown() && second[0].is_unknown());
        assert_eq!(context.inputs.borrow().len(), 3);

        // `inline_program` replays a program over seed values. All-known seeds fold every instruction, lifting the
        // program constant into the known-side context, and so the replay returns folded values.
        let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
        let a = builder.add_input(DataType::F64);
        let x = builder.add_input(DataType::F64);
        let c = builder.add_constant(Scalar::from(1.0));
        let product = builder.add_instruction(MulOperation, vec![a, x]).unwrap()[0];
        let sum = builder.add_instruction(AddOperation, vec![product, c]).unwrap()[0];
        let program = builder
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![sum], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let outputs = context
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
        let outputs = context
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
        let outputs = context
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
        let outputs = context
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
        assert_eq!(context.known_constant(&Scalar::from(5.0)), Ok(Scalar::from(5.0)));
        let staging = ScalarTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(DataType::F64);
        let literal = staging.constant(Scalar::from(4.0));
        assert_eq!(staging_context.known_constant(&literal), Ok(Scalar::from(4.0)));
        assert!(matches!(
            staging_context.known_constant(&symbolic),
            Err(ProgramError::MalformedProgram(message))
                if message == "a known value crossing into a nested residual program is not concretizable in the \
                    active known-side context",
        ));

        // `all_knowns_are_concrete` checks every known feeder and folded output of a partial evaluation, which is only
        // non-trivial under a staging known-side context where knowns can be live tracers.
        let empty = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new()
            .build::<Vec<Scalar>, Vec<Scalar>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert!(context.all_knowns_are_concrete(&PartialEvaluation::<EagerContext<Scalar, ScalarOperation<Scalar>>> {
            program: empty.clone(),
            inputs: vec![PartialEvaluationInput::Known(Scalar::from(1.0)), PartialEvaluationInput::Unknown(0)],
            outputs: vec![PartialEvaluationOutput::Known(Scalar::from(2.0))],
        }));
        assert!(!staging_context.all_knowns_are_concrete(&PartialEvaluation::<ScalarTracingContext> {
            program: empty.clone(),
            inputs: vec![PartialEvaluationInput::Known(symbolic.clone())],
            outputs: Vec::new(),
        }));
        assert!(staging_context.all_knowns_are_concrete(&PartialEvaluation::<ScalarTracingContext> {
            program: empty,
            inputs: vec![PartialEvaluationInput::Known(literal.clone())],
            outputs: Vec::new(),
        }));

        // `any_known_is_symbolic` is the signal online boundary rules split on. Only a known value that does not
        // resolve to a concrete constant counts, and so eager knowns and unknowns never do.
        assert!(!context.any_known_is_symbolic(&[PartialEvaluationValue::known(Scalar::from(1.0))]));
        assert!(!staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(literal)]));
        assert!(staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(symbolic)]));
        assert!(
            !staging_context.any_known_is_symbolic(&[PartialEvaluationValue::variable(DataType::F64, AtomId::new(0))]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_as_context() {
        // Over an eager known-side inner context, the partial-evaluation context is itself eager: lifted constants
        // and all-known binds fold to concrete known values that resolve `Concrete` and support concretizing
        // extractions such as `boolean`, which is what lets host control flow branch on known values mid-evaluation.
        let context = PartialEvaluationContext::new(EagerContext::<Scalar, ScalarOperation<Scalar>>::new());
        assert!(context.is_eager());
        let lifted = context.lift(Scalar::from(2.0)).unwrap();
        assert!(matches!(
            lifted.value().unwrap().materialization(),
            PartialValueMaterialization::Constant { residual_atom: None },
        ));
        assert!(matches!(context.resolve(&lifted), ValueResolution::Concrete(value) if value == Scalar::from(2.0)));
        let folded = context.bind(AddOperation, &[lifted.clone(), lifted.clone()]).unwrap();
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].value().unwrap().as_known(), Some(&Scalar::from(4.0)));
        assert_eq!(folded[0].boolean(), Ok(true));
        assert_eq!(folded[0].r#type().into_owned(), DataType::F64);

        // The `Zero` capability binds a nullary `ZeroOperation`, which is vacuously all-known and folds through the
        // inner context to a known zero.
        let zero = context.zero(&DataType::F64).unwrap();
        assert_eq!(zero.value().unwrap().as_known(), Some(&Scalar::from(0.0)));
        assert_eq!(zero.boolean(), Ok(false));

        // A mixed bind residualizes: the unknown input names a residual program variable, the known input
        // materializes as a residual input, and the output is an unknown value that resolves `Opaque` and rejects
        // concretizing extractions.
        let unknown_atom = context.builder.borrow_mut().add_input(DataType::F64);
        context.inputs.borrow_mut().push(PartialEvaluationInput::Unknown(0));
        let unknown =
            PartialTracer::new(context.clone(), PartialEvaluationValue::variable(DataType::F64, unknown_atom));
        let mixed = context.bind(MulOperation, &[folded[0].clone(), unknown.clone()]).unwrap();
        assert!(mixed[0].value().unwrap().is_unknown());
        assert!(matches!(context.resolve(&mixed[0]), ValueResolution::Opaque));
        assert!(matches!(mixed[0].boolean(), Err(ProgramError::Concretization { .. })));

        // Finalizing the context (after dropping every stamped clone) produces the accumulated residual program:
        // `(ẏ) = folded * ẋ` over the unknown input plus the materialized known feeder.
        let output = mixed[0].value().unwrap().clone();
        drop((lifted, folded, zero, unknown, mixed));
        let evaluation = context.into_evaluation(vec![output]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Scalar::from(4.0))],
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluation_context_defers_bind_errors_by_poisoning() {
        // A failed bind (in this case, folding an operation whose known inputs belong to two different traces) does
        // not return an error; it poisons its outputs so the infallible operator sugar driving closures never panics.
        // The poison propagates through later binds, resolves `Opaque`, rejects concretizing extractions with the
        // deferred error, and surfaces that original error at the value boundary.
        use crate::operations::BooleanLike;

        let outer_a = ScalarTracingContext::new();
        let outer_b = ScalarTracingContext::new();
        let context = PartialEvaluationContext::new(outer_a.clone());
        let known_a = PartialTracer::new(context.clone(), PartialEvaluationValue::known(outer_a.input(DataType::F64)));
        let known_b = PartialTracer::new(context.clone(), PartialEvaluationValue::known(outer_b.input(DataType::F64)));
        let poisoned = context.bind(AddOperation, &[known_a.clone(), known_b]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert_eq!(format!("{}", poisoned[0]), "<poison:f64>");
        assert_eq!(poisoned[0].r#type().into_owned(), DataType::F64);
        assert!(matches!(context.resolve(&poisoned[0]), ValueResolution::Opaque));
        assert!(matches!(poisoned[0].boolean(), Err(ProgramError::MismatchedProgramBuilders)));

        // Poison propagates from inputs to outputs of later binds, and unwrapping at a boundary reports the original
        // deferred error rather than a generic poison error.
        let propagated = context.bind(MulOperation, &[known_a, poisoned[0].clone()]).unwrap();
        assert!(matches!(propagated[0].value(), Err(ProgramError::MismatchedProgramBuilders)));
        assert!(matches!(propagated[0].clone().into_value(), Err(ProgramError::MismatchedProgramBuilders)));
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

        // The known feeder is a tracer naming the staged `a * a` atom of the outer program (atom 2, since the replay
        // lifts the live program constant into the outer trace up front, before replaying any instruction).
        assert_eq!(evaluation.inputs.len(), 2);
        assert!(matches!(&evaluation.inputs[0], PartialEvaluationInput::Unknown(1)));
        assert!(matches!(
            &evaluation.inputs[1],
            PartialEvaluationInput::Known(feeder) if feeder.atom_id() == Ok(AtomId::new(2)),
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

        // The outer trace accumulated the lifted literal followed by the folded known work. The literal stays dead in
        // the outer trace because the residual program rebuilds it inline.
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(vec![AtomId::new(2)], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = mul %0 %0
                in (%2)
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
