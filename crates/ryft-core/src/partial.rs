//! Partially evaluates [`Program`]s into work available in a known-side [`Context`] and work deferred to a residual
//! program.
//!
//! Partial evaluation is a transform boundary. Each input is classified as a concrete or symbolic value available
//! to the parent context, or as an unknown value represented only by its [`Type`]. Operations whose results can be
//! established from known inputs bind through the parent context. Work that depends on an unknown value is recorded in
//! a residual [`ProgramBuilder`], together with the minimum boundary needed to run it later. Finalization returns the
//! residual program plus descriptors that reconnect its inputs and outputs to the original program. Refer to the
//! documentation of [`PartialEvaluationContext`] for a rendered diagram of this split and to the documentation of
//! [`PartitionedProgram`] for the corresponding two-program wiring.
//!
//! Partial evaluation is both a public specialization transform and infrastructure for other transforms. In
//! linearization, for example, primals are known, tangents are unknown, and the residual tangent program becomes the
//! reusable linear computation.
//!
//! # Choosing an Entry Point
//!
//!   - [`Program::partially_evaluate`] is the eager specialization entry point for a flat program. It executes known
//!     work immediately and returns a [`PartialEvaluation`] carrying concrete known values and a residual program.
//!   - [`Program::partially_evaluate_in_context`] performs the same split relative to an explicit known-side context.
//!     With a staging context, known work is appended to an enclosing program instead of executed. The matching
//!     [`RegionRef::partially_evaluate_in_context`] method applies the transform to a borrowed sealed region without
//!     first materializing it as a standalone program.
//!   - [`Program::partition`] and [`RegionRef::partition`] reify both sides of the split as a [`PartitionedProgram`]: a
//!     known program, a residual program, and positional wiring between them.
//!   - [`PartialEvaluation::interpret`] supplies the surviving unknown inputs, runs the residual program in the same
//!     context family, and reconstructs the original outputs in their original order.
//!
//! # Known Work and Residual Work
//!
//! _Known_ means available in the parent context; it does not necessarily mean host-concrete. An eager parent executes
//! an all-known operation immediately. A staging parent binds the same operation into its enclosing program, making
//! the resulting tracer known to this partial-evaluation level. Mixed or unknown operations are offered to their
//! [`PartiallyEvaluatableOperation`] rule and ordinarily emitted into the residual program.
//!
//! Operation-owned rules may make a more precise split. A condition with a concretizable known predicate can inline
//! only its selected branch, for example. If a known value cannot be resolved or concretized through the parent
//! context, the rule must preserve it conservatively rather than inspect unavailable runtime data.
//!
//! # Values and Residual Materialization
//!
//! [`PartialValue`] carries only semantic classification: [`Known`](PartialValue::Known) contains a parent-context
//! value available now, while [`Unknown`](PartialValue::Unknown) carries the type of a future value.
//! [`PartialEvaluationValue`] adds a shared [`PartialValueMaterialization`] slot describing how that logical value
//! crosses into residual work. A known value may become a residual input or an inline residual constant. An unknown
//! value is already a residual variable. The first residual consumer assigns an atom, and every clone reuses it.
//! Staged-identity deduplication additionally merges distinct known values that name the same outer-program atom.
//!
//! Literal constants remain constants in the residual program. Known variables needed by residual work become
//! [`Known`](PartialEvaluationInput::Known) feeders, while original unknown inputs become
//! [`Unknown`](PartialEvaluationInput::Unknown) feeders. This distinction keeps runtime values out of staged constant
//! payloads while avoiding duplicate boundary inputs.
//!
//! # Results and Wiring
//!
//! [`PartialEvaluation`] owns one residual program. Its [`PartialEvaluationInput`] sequence is ordered like that
//! program's inputs and carries either a known feeder value or an original unknown-input index. Its
//! [`PartialEvaluationOutput`] sequence is ordered like the original outputs and carries either a folded value or a
//! residual-output index. [`PartialEvaluation::interpret`] follows those two mappings to replay and reassemble.
//!
//! [`PartitionedProgram`] expresses the same split without retaining parent-context values. It replaces feeder and
//! output values with positions, yielding a known program whose trailing outputs are residual edges and a residual
//! program that consumes those edges together with the original unknown inputs.
//!
//! # Identity, Concretization, and Failure Propagation
//!
//! [`PartialTracer`] equality is logical transform identity—two live tracers compare equal only when they share one
//! materialization slot, not when their eventual payloads are equal. This conservative identity is used by fixed-point
//! and passthrough analyses. Host control flow can inspect a known tracer only when its parent context resolves it to
//! a constant supporting the requested concretization; unknown and opaque values remain residual.
//!
//! Binding failures are deferred through poisoned [`PartialTracer`]s so infallible operator syntax can continue to
//! construct the surrounding closure. Poison propagates through later binds, and the partial-evaluation boundary
//! reports the original [`ProgramError`]. Escaped context or value clones keep shared builders alive and are rejected
//! during finalization with [`ProgramError::EscapedProgramBuilder`].
//!
//! # Control Flow, Effects, and Recursion
//!
//! Higher-order rules receive a [`PartialEvaluationDriver`] for recursively transforming attached regions. A rule may
//! inline selected nested work; uninlined mixed work remains attached to a residual operation. Ordinary effectful
//! operations follow the same placement rule as pure ones: all-known effects execute under an eager parent or stage
//! under a staging parent, while unknown-dependent effects remain residual. Unresolved ordered state is excluded from
//! both placements and must be discharged before partial evaluation. Probe-based fixed points must not speculatively
//! execute effectful bodies, because every probe would otherwise repeat or duplicate the effect.
//!
//! # Extending Partial Evaluation
//!
//! Implement [`PartiallyEvaluatableOperation`] for operation payloads. Most operations use the default
//! [`PartialEvaluationContext::fold_or_residualize`] policy; control flow, loops, scans, and other higher-order
//! operations may override it to preserve more known work. Use the supplied context and driver to materialize values,
//! residualize operations, and recurse into regions rather than constructing boundary atoms independently. Rules that
//! inspect known payloads must first establish [`Constant`](ValueResolution::Constant) resolution and fall back
//! conservatively when it is unavailable.

use std::borrow::Cow;
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use crate::contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    AtomId, BindingRegionDriver, Effect, EmptyRegionDriver, FlatProgram, Operation, Program, ProgramBuilder,
    ProgramError, ProjectedValue, Provenance, ProvenanceScope, ProvenanceState, RegionDriver, RegionRef,
    RegionReplayMappings, ReplayRegionDriver, Type, TypeError, TypeIdentityPosition, Typed, Value, ValueProjection,
};
use crate::tracing::TracingContext;

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial context
/// interprets a [`Program`] over. Every [`Atom`](crate::Atom) and every intermediate result is either
/// [`Known`](Self::Known) (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its
/// [`Type`](crate::Type) is available until the residual program runs). For more information on partial evaluation,
/// refer to the documentation of [`Program::partially_evaluate`].
#[derive(Clone, Debug)]
pub enum PartialValue<V: Value> {
    /// [`Value`] that is fully known at partial-evaluation time and can be folded forward.
    Known(V),

    /// [`Value`] that is not known until the residual program runs and only its [`Type`](crate::Type) is known.
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

/// Descriptor for one original output after partial evaluation. Partial evaluation splits the original outputs
/// into those it could fold to a known value and those that remain computed by the residual [`Program`]. A
/// [`PartialEvaluation`] stores these descriptors in original output order so that it can reconstruct the full
/// result after interpreting the residual program.
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

impl<C: Context> PartialEvaluation<C> {
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

/// Result of partitioning a [`Program`] into a known-side program and a residual program based on which original inputs
/// are known. Unlike [`PartialEvaluation`], this representation carries only programs and positional wiring. It does
/// not retain values from a parent [`Context`]. It is returned by [`Program::partition`] and is typically passed to
/// [`PartialEvaluationContext::inline_partitioned_program`] when recursively transforming an attached region.
///
/// # Boundary Wiring
///
/// ```mermaid
/// flowchart LR
///   known_inputs["Selected Original Known Inputs"] --> known_program["Known Program"]
///   known_program --> known_outputs["Known Original Outputs"]
///   known_program --> residual_edges["Residual Edge Values"]
///   unknown_inputs["Original Unknown Inputs"] --> residual_program["Residual Program"]
///   residual_edges --> residual_program
///   residual_program --> residual_outputs["Residual Original Outputs"]
///   known_outputs --> descriptors["Output Descriptors"]
///   residual_outputs --> descriptors
///   descriptors --> outputs["Outputs in Original Order"]
/// ```
///
/// The known program receives only the original inputs selected by [`known_input_indices`](Self::known_input_indices).
/// Its outputs place fully known original outputs before residual edge values. The residual program consumes the
/// original unknown inputs together with those edges, while [`outputs`](Self::outputs) records which side supplies
/// each original output.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct PartitionedProgram<V: Value, O: Operation<Type = V::Type>> {
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

impl<V: Value, O: Operation<Type = V::Type>> PartitionedProgram<V, O> {
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

/// [`RegionDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to [`Region`](crate::Region)s
/// attached to a partially evaluated [`Operation`] application. A [`PartialEvaluationDriver`] borrows the current
/// instruction's regions and supports recursive partial evaluation. Operation rules receive it separately from their
/// durable [`PartialEvaluationContext`], so the borrowed region access cannot escape through a [`PartialTracer`].
/// [`RegionDriver`] provides structural region access, while this trait adds partial-evaluation-specific recursion.
pub trait PartialEvaluationDriver<C: Context>: RegionDriver<C::Constant, C::Operation> {
    /// Partially evaluates the [`Region`](crate::Region) at `index` over the provided partial-evaluation values
    /// by re-entering the active partial-evaluation transform.
    fn partially_evaluate_region(
        &self,
        context: &PartialEvaluationContext<C>,
        index: usize,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>;

    /// Partially evaluates `region` against the provided input knowledge through the active known-side context and
    /// returns the region's residual split.
    fn partially_evaluate_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>;

    /// Partitions `region` into known and residual programs through a fresh staging context whose inputs encode the
    /// provided known-ness mask.
    fn partition_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError>;
}

impl<C: Context> PartialEvaluationDriver<C> for EmptyRegionDriver {
    #[inline]
    fn partially_evaluate_region(
        &self,
        _context: &PartialEvaluationContext<C>,
        _index: usize,
        _inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partially evaluate a region".to_string()))
    }

    #[inline]
    fn partially_evaluate_program(
        &self,
        _context: &PartialEvaluationContext<C>,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partially evaluate a program".to_string()))
    }

    #[inline]
    fn partition_program(
        &self,
        _region: RegionRef<'_, C::Constant, C::Operation>,
        _input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot partition a program".to_string()))
    }
}

/// [`PartialEvaluationDriver`] scoped to one [`Operation`] application. It borrows the application's complete
/// [`RegionDriver`], preserving the operation-defined ordering of owned [`Region`](crate::Region)s, borrowed regions,
/// and shared callees without collecting [`Program`]s or region views. Recursive requests re-enter partial evaluation
/// for a selected region or partition it into known and residual programs.
struct RecursivePartialEvaluationDriver<'r, D> {
    /// Application-scoped [`RegionDriver`], in [`Operation`]-defined order.
    driver: &'r D,
}

impl<V: Value, O: Operation<Type = V::Type>, D: RegionDriver<V, O>> RegionDriver<V, O>
    for RecursivePartialEvaluationDriver<'_, D>
{
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.driver.regions()
    }
}

impl<C: Context, D: RegionDriver<C::Constant, C::Operation>> PartialEvaluationDriver<C>
    for RecursivePartialEvaluationDriver<'_, D>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    #[inline]
    fn partially_evaluate_region(
        &self,
        context: &PartialEvaluationContext<C>,
        index: usize,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        context.inline_region(self.region(index)?, inputs)
    }

    #[inline]
    fn partially_evaluate_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        region.partially_evaluate_in_context(context.parent(), knowledge)
    }

    #[inline]
    fn partition_program(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError> {
        region.partition(input_known)
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
///     [`Operation`](Domain::Operation) is the operation family of the residual [`Program`] and of any inlined nested
///     programs (e.g., the enum this operation may belong to). Its [`Constant`](Domain::Constant) is the staged
///     constant space those programs store. Finally, its [`Value`](Domain::Value) is the space known values flow in
///     (i.e., concrete values under eager contexts and [`Tracer`](crate::Tracer)s into the outer program under
///     [`StagingContext`]s).
///
/// # Deriving Partially Evaluatable Operation Enums
///
/// The `#[derive(Operation)]` macro generates a [`PartiallyEvaluatableOperation`] implementation for operation enums.
/// Native variants forward to their payload's own rule, and the generated per-payload predicates transport that rule's
/// value and context requirements to the enum's use site. Declared member variants instead use the enclosing enum's
/// canonical fold-or-residualize path because member-side partial values cannot represent values belonging to other
/// members of the composite universe. This preserves correct folding and residualization without a second projected
/// partial-value protocol. Refer to the documentation of [`Operation`] for the full derive contract. Partial evaluation
/// is always generated and does not require a `dispatch(...)` selection.
pub trait PartiallyEvaluatableOperation<C: Context>: Clone + Into<C::Operation> {
    /// Partially evaluates this [`PartiallyEvaluatableOperation`] for the provided [`PartialEvaluationValue`]s. Unless
    /// overridden, this function will default to calling [`PartialEvaluationContext::fold_or_residualize`] which uses
    /// the following semantics:
    ///
    ///   - An operation that declares [`Effect::OrderedState`], or whose complete attached region closure contains that
    ///     effect, is rejected before input knownness is inspected. Unresolved state may neither execute on the known
    ///     side nor survive in the residual program.
    ///   - When *all* of the operation's inputs are [`Known`](PartialValue::Known), it **folds** the operation by
    ///     [`bind`](Context::bind)ing it in the known-side context, interpreting it immediately under an eager context,
    ///     and staging it into the outer program under a [`StagingContext`], so that the operation's outputs become
    ///     known values and the operation contributes nothing to the residual [`Program`].
    ///   - Otherwise, it **residualizes** the operation unchanged, meaning that it emits the operation into the
    ///     residual program over its inputs' residual program [`Atom`](crate::Atom)s, materializing each known input as
    ///     a residual input for a known variable or as an inlined residual program constant for a literal, so that the
    ///     operation runs at residual program execution time.
    ///
    /// There are situations where overriding this function can result in improved performance and better partitioning
    /// of a computation into known and unknown parts. For example, a `condition` instruction whose predicate is
    /// [`Known`](PartialValue::Known) and Boolean-concretizable may ask the context to inline the selected branch and
    /// return that branch's output trace values, so that the condition disappears from the residual program and only
    /// the taken branch's work survives. Rules that inspect known *payloads* must gate that inspection on a
    /// [`Constant`](ValueResolution::Constant) [`Context::resolve`] resolution because a known value under a staging
    /// known-side context may be a [`Tracer`](crate::Tracer) into the outer program rather than a program constant,
    /// and partial evaluation should fall back to a conservative rewrite otherwise. Resolving to a constant alone
    /// does not guarantee that the payload is host-inspectable; rules that inspect it require the corresponding
    /// capability separately.
    ///
    /// # Parameters
    ///
    ///   - `context`: Durable [`PartialEvaluationContext`] that owns residual emission, inlining, and materialization.
    ///   - `driver`: [`PartialEvaluationDriver`] that provides [`Instruction`](crate::Instruction)-scoped access to the
    ///     application [`Region`](crate::Region)s.
    ///   - `inputs`: [`PartialEvaluationValue`] for each of this [`Operation`]'s inputs, in input order.
    #[inline]
    fn partially_evaluate<D: PartialEvaluationDriver<C>>(
        &self,
        context: &PartialEvaluationContext<C>,
        driver: &D,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        context.fold_or_residualize(self.clone(), driver.regions().map(|region| region.to_program()).collect(), inputs)
    }
}

/// Active [`Context`] that folds known work through a parent context `C` and records unknown-dependent work in a
/// residual [`ProgramBuilder`]. It drives [`Program::partially_evaluate`], [`Program::partially_evaluate_in_context`],
/// and transform interpreters that bind operations directly over [`PartialTracer`]s.
///
/// Each [`Context::bind`] dispatches the operation's [`PartiallyEvaluatableOperation::partially_evaluate`]
/// implementation. The default [`fold_or_residualize`](Self::fold_or_residualize) policy binds an all-known operation
/// through the parent context and emits a mixed or unknown operation into the residual builder. Specialized rules can
/// instead inline nested programs or preserve more known work.
///
/// # Evaluation Pipeline
///
/// ```mermaid
/// flowchart TD
///   inputs["Inputs Classified as Known Values or Unknown Types"] --> context["PartialEvaluationContext"]
///   context --> rules["Operation Partial-Evaluation Rules"]
///   rules -->|"all inputs known"| parent["Known-Side Parent Context"]
///   parent -->|"execute eagerly or append to an outer program"| known["Known Result Values"]
///   rules -->|"mixed or unknown"| residualize["Residualize Operation"]
///   unknown["Unknown Inputs and Residual Variables"] --> residualize
///   known -->|"needed by residual work"| materialize["Materialization Policy"]
///   materialize -->|"known variable"| residual_input["Residual Input Feeder"]
///   materialize -->|"literal or designated constant"| residual_constant["Inline Residual Constant"]
///   residual_input --> builder["Residual Program Builder"]
///   residual_constant --> builder
///   residualize --> builder
///   known --> finalize["Finalize Boundary Mappings"]
///   builder --> finalize
///   finalize --> result["Residual Program with Input and Output Wiring"]
/// ```
///
/// Mutable state is shared behind `Rc<RefCell<...>>` handles. Cloning this context therefore keeps every clone writing
/// to the same residual program, input descriptors, and staged-feeder table, and lets rules re-enter the context (e.g.,
/// to inline a selected condition branch through [`inline_program`](Self::inline_program)).
#[cfg_attr(doc, aquamarine::aquamarine)]
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
    /// *staging* known-side context because an eager context resolves knowns as [`Constant`](ValueResolution::Constant)
    /// rather than [`Staged`](ValueResolution::Staged), and so nothing is ever recorded in that case, and inline
    /// constants are excluded because they carry no staged identity.
    staged_feeders: Rc<RefCell<HashMap<AtomId, AtomId>>>,

    /// Active [`ProvenanceState`] that residual [`Instruction`](crate::Instruction)s snapshot.
    /// [`PartialEvaluationContext`]s are staging boundaries (i.e., they own the residual [`ProgramBuilder`] and emit
    /// into it directly), and so they own provenance state exactly like tracing contexts instead of delegating reads
    /// to their known-side parents, which are often terminal eager contexts that would erase source provenance from
    /// every residual program. The state is seeded from the parent context's current provenance at construction and
    /// shared across clones.
    provenance: Rc<ProvenanceState>,
}

impl<C: Context> PartialEvaluationContext<C> {
    /// Creates a fresh [`PartialEvaluationContext`] that folds known work through `context` and accumulates
    /// residual work in a new residual [`ProgramBuilder`].
    #[inline]
    pub fn new(parent: C) -> Self {
        let provenance = Rc::new(ProvenanceState::seeded(parent.provenance()));
        Self {
            parent,
            builder: Rc::new(RefCell::new(ProgramBuilder::new())),
            inputs: Rc::new(RefCell::new(Vec::new())),
            staged_feeders: Rc::new(RefCell::new(HashMap::new())),
            provenance,
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
    /// Operations whose [`effects`](Operation::effects) are not [`Effects::PURE`](crate::Effects::PURE) ordinarily
    /// follow the same known-ness placement: an all-known effectful operation folds into the known side, and a
    /// mixed-input one residualizes. [`Effect::OrderedState`] is the exception: unresolved state is rejected before the
    /// knownness branch, including state anywhere in the complete attached-region closure, because neither executing it
    /// during specialization nor preserving it past the transform boundary is valid. For the remaining effects, the
    /// split's execution contract requires all known work to run before residual work, so that an effect's side is
    /// determined by its input known-ness:
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
    /// then discards), each speculative fold has a real consequence: it executes the effect when in an eager context or
    /// stages it into the outer program when in a [`StagingContext`]. A rule that iterates would then fire or stage the
    /// effect once per round instead of exactly once, and so it must skip effectful programs and residualize them
    /// unchanged.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] to fold into the known-side context when all inputs are known, or to emit into
    ///     the residual [`Program`] otherwise.
    ///   - `regions`: Owned [`Program`]s whose entry [`Region`](crate::Region)s are attached to `operation`, in
    ///     the order defined by [`Operation::region_slots`]. Folding binds these regions with the operation, while
    ///     residualization imports them into the residual [`Program`].
    ///   - `inputs`: Partially evaluated inputs/operands supplied to `operation`, in [`Operation`]-defined order.
    ///     Their known-ness determines whether the operation is folded or residualized.
    #[inline]
    pub fn fold_or_residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let operation = operation.into();
        self.validate_no_unresolved_references_or_state(
            &operation,
            inputs,
            regions.iter().map(|region| region.entry_region_ref()),
        )?;
        if inputs.iter().all(PartialEvaluationValue::is_known) {
            let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
            Ok(self
                .parent
                .bind(operation, regions, &known)?
                .into_iter()
                .map(|value| {
                    // A folded value that owns a type identity must remain a producer when it crosses into residual
                    // work. Embedding its cheap constant payload does that structurally. Symbolic known values remain
                    // residual inputs because their parent-context producer stays live.
                    let defines_identity =
                        value.r#type().identities().any(|(position, _)| position == TypeIdentityPosition::Definition);
                    if defines_identity && self.parent.resolve(&value).is_constant() {
                        PartialEvaluationValue::known_constant(value)
                    } else {
                        PartialEvaluationValue::known(value)
                    }
                })
                .collect())
        } else {
            self.residualize(operation, regions, inputs)
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
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] to emit into the residual [`Program`].
    ///   - `regions`: Owned [`Program`]s whose entry [`Region`](crate::Region)s are attached to `operation`, in the
    ///     order defined by [`Operation::region_slots`]. These programs are imported into the residual [`Program`]
    ///     before the operation is emitted.
    ///   - `inputs`: Partially evaluated inputs/operands supplied to `operation`, in [`Operation`]-defined order. Known
    ///     inputs are materialized as residual inputs or constants, while unknown inputs reuse their existing residual
    ///     atoms.
    pub fn residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let operation = operation.into();
        self.validate_no_unresolved_references_or_state(
            &operation,
            inputs,
            regions.iter().map(Program::entry_region_ref),
        )?;

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
                            let constant = self.parent.resolve(known).into_constant().ok_or_else(|| {
                                ProgramError::MalformedProgram(
                                    "residual materialization required a constant payload for a known value that is \
                                     not resolvable to a constant in the active known-side context"
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

        // Residualized regions splice into the residual builder's arena directly (i.e., owned move), in region order.
        let region_ids = {
            let mut builder = self.builder.borrow_mut();
            regions.into_iter().map(|region| builder.import_program(region)).collect::<Vec<_>>()
        };
        let output_atoms = self
            .builder
            .borrow_mut()
            .add_instruction(operation, region_ids, input_atoms, Some(self.provenance.current()))?
            .to_vec();

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
    ///
    /// # Parameters
    ///
    ///   - `program`: [`Program`] whose entry [`Region`](crate::Region) is replayed through this
    ///     [`PartialEvaluationContext`].
    ///   - `inputs`: Partially evaluated values bound to `program`'s input [`Atom`](crate::Atom)s in input order.
    #[inline]
    pub fn inline_program(
        &self,
        program: &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        self.inline_region(program.entry_region_ref(), inputs)
    }

    /// Replays a borrowed [`Region`](crate::Region) through this context without materializing it as a standalone
    /// source [`Program`]. This is the borrowed-region counterpart of [`inline_program`](Self::inline_program) and
    /// otherwise uses the same partial-evaluation replay, input binding, and output ordering described there.
    ///
    /// # Parameters
    ///
    ///   - `region`: Borrowed [`Region`](crate::Region) to replay through this [`PartialEvaluationContext`].
    ///   - `inputs`: Partially evaluated values bound to `region`'s input [`Atom`](crate::Atom)s in input order.
    pub(crate) fn inline_region(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        // This replay loop and `Context::bind` are the only two production paths into partial evaluation rules, so
        // rejecting unresolved state and references once over the complete entering closure here (with
        // `fold_or_residualize` and `residualize` retaining their own gates as direct invocation defense) covers all
        // higher-order operations centrally, including future ones that would otherwise each need a per-rule preamble.
        if let Some(occurrence) = region.effect_occurrences_in_closure(Effect::OrderedState).next() {
            // TODO(eaplatanios): Aggregate all effect occurrences into one diagnostic once `ProgramError` supports
            //  multi-diagnostic reporting.
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` must be discharged before partial evaluation", occurrence.operation().name()),
            });
        }

        if inputs.iter().any(|input| input.r#type().is_reference())
            || region.contains_atom_type_in_closure(Type::is_reference)
        {
            return Err(ProgramError::UnsupportedOperation {
                message: "references must be discharged before partial evaluation".to_string(),
            });
        }

        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(PartialEvaluationValue::known_constant(self.parent.lift(constant.clone())?)),
            |instruction, inputs| {
                // Evaluate inside the source instruction's recorded origin so residualized instructions record
                // where they came from.
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let driver = RecursivePartialEvaluationDriver { driver: &regions };
                self.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    instruction.operation().partially_evaluate(self, &driver, inputs)
                })
            },
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
    /// single step keeps it whole until it is gone, so no partially moved partition state can ever be observed.
    ///
    /// # Parameters
    ///
    ///   - `partition`: [`PartitionedProgram`] to inline, produced by [`Program::partition`].
    ///   - `inputs`: Input [`PartialEvaluationValue`]s in the order of the original program's input, pre-partitioning.
    ///   - `build_known_operation`: Wraps the provided known [`Program`] in the known-side boundary [`Operation`]
    ///     together with the owned [`Region`](crate::Region) programs (in region order) that the emitted instruction
    ///     attaches. Operations that carry the program in their payload return no regions.
    ///   - `build_residual_operation`: Wraps the provided residual [`Program`] in the residual boundary [`Operation`],
    ///     with the same [`Region`](crate::Region) contract as `build_known_operation`.
    pub fn inline_partitioned_program<
        V: Value<Type = C::Type>,
        O: Operation<Type = C::Type>,
        P: Into<C::Operation>,
        BuildKnownProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> (P, Vec<FlatProgram<C>>),
        BuildResidualProgramOperation: FnOnce(Program<V, O, Vec<V>, Vec<V>>) -> (P, Vec<FlatProgram<C>>),
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
        let (known_program_operation, known_regions) = build_known_operation(program.known_program);
        let known_outputs =
            self.fold_or_residualize(known_program_operation, known_regions, known_inputs.as_slice())?;

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
        let (residual_program_operation, residual_regions) = build_residual_operation(program.residual_program);
        let residual_outputs =
            self.residualize(residual_program_operation, residual_regions, residual_inputs.as_slice())?;

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
    /// a [`ProgramError`] when the known-side [`Context`] cannot resolve the provided value to a program constant.
    /// Higher-order rules use this when they must embed a known value *inside* a nested residual program (e.g., a
    /// folded loop-invariant carry spliced into a rebuilt `scan` body), where only a program constant can represent
    /// it (nested programs cannot reference atoms of the enclosing residual program or of the outer known-side
    /// program). Under an eager known-side context this always succeeds. Under a [`StagingContext`] it succeeds only
    /// for literal-backed values, and the caller must treat the error as "this rewrite is not available" and fall back
    /// to a conservative alternative.
    #[inline]
    pub fn known_constant(&self, value: &C::Value) -> Result<C::Constant, ProgramError> {
        self.parent.resolve(value).into_constant().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "a known value crossing into a nested residual program does not resolve to a constant in the active \
                 known-side context"
                    .to_string(),
            )
        })
    }

    /// Returns `true` if every [`Known`](PartialEvaluationInput::Known) residual input and
    /// [`Known`](PartialEvaluationOutput::Known) output of the provided [`PartialEvaluation`] resolves to a program
    /// constant in the known-side [`Context`] of this [`PartialEvaluationContext`] (i.e., if a nested program rebuild
    /// that embeds those knowns as inline program constants through [`Self::known_constant`] can succeed). Under a
    /// staging known-side context, a probe's folds can produce known values that are genuine tracers into the live
    /// trace (e.g., a constant-only chain staged by the fold). Rules that rebuild nested programs from a live context
    /// probe must check this and fall back to a conservative rewrite when it returns `false`.
    #[inline]
    pub fn all_knowns_are_constants(&self, evaluation: &PartialEvaluation<C>) -> bool {
        evaluation.inputs.iter().all(|input| match input {
            PartialEvaluationInput::Known(value) => self.parent.resolve(value).is_constant(),
            PartialEvaluationInput::Unknown(_) => true,
        }) && evaluation.outputs.iter().all(|output| match output {
            PartialEvaluationOutput::Known(value) => self.parent.resolve(value).is_constant(),
            PartialEvaluationOutput::Unknown(_) => true,
        })
    }

    /// Returns `true` when any of the provided `inputs` is known but does not [`resolve`](Context::resolve)
    /// to a [`Constant`](ValueResolution::Constant) in the known-side [`Context`] of this
    /// [`PartialEvaluationContext`] (i.e., it is a genuine [`Tracer`](crate::Tracer) into a live outer trace). This
    /// is the signal online boundary rules split on: all-constant knowledge keeps the default fold-or-residualize
    /// behavior.
    #[inline]
    pub fn any_known_is_symbolic(&self, inputs: &[PartialEvaluationValue<C::Value>]) -> bool {
        inputs.iter().any(|input| match input.value() {
            PartialValue::Known(value) => !self.parent.resolve(value).is_constant(),
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
        let builder = self.builder.borrow();
        if outputs.iter().any(|output| output.r#type().is_reference())
            || builder.atoms().iter().any(|atom| atom.r#type().is_reference())
            || builder.regions.iter().any(|region| region.atoms().iter().any(|atom| atom.r#type().is_reference()))
        {
            return Err(ProgramError::UnsupportedOperation {
                message: "references must be discharged before partial evaluation".to_string(),
            });
        }
        drop(builder);

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

    /// Rejects an operation application whose inputs contain references or whose operation or complete attached region
    /// closure contains unresolved references or ordered state. The partial-evaluation bind, replay, fold, and residual
    /// emission boundaries call this before executing, partitioning, importing, or emitting the application.
    ///
    /// The operation-local diagnostic intentionally matches the conservative reference-operation rules. The attached
    /// region diagnostic identifies the higher-order carrier whose region closure is unsafe, including state hidden
    /// under dormant rule regions that ordinary executable region effect summaries exclude.
    fn validate_no_unresolved_references_or_state<
        'r,
        'i,
        V: Typed<Type = C::Type> + 'i,
        I: IntoIterator<Item = &'i V>,
        R: IntoIterator<Item = RegionRef<'r, C::Constant, C::Operation>>,
    >(
        &self,
        operation: &C::Operation,
        inputs: I,
        regions: R,
    ) -> Result<(), ProgramError>
    where
        C::Constant: 'r,
        C::Operation: 'r,
    {
        if operation.effects().contains(Effect::OrderedState) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` must be discharged before partial evaluation", operation.name()),
            });
        }

        if inputs.into_iter().any(|input| input.r#type().is_reference()) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` consumes unresolved references and must be discharged before partial evaluation",
                    operation.name(),
                ),
            });
        }

        let mut contains_reference = false;
        let mut contains_state = false;
        for region in regions {
            contains_reference |= region.contains_atom_type_in_closure(Type::is_reference);
            contains_state |= region.contains_effect_in_closure(Effect::OrderedState);
        }

        if contains_reference {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` carries unresolved references in an attached region and must be discharged before partial \
                    evaluation",
                    operation.name(),
                ),
            });
        }

        if contains_state {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "`{}` carries unresolved state in an attached region and must be discharged before partial \
                    evaluation",
                    operation.name(),
                ),
            });
        }

        Ok(())
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
            provenance: self.provenance.clone(),
        }
    }
}

impl<C: Context> Domain for PartialEvaluationContext<C> {
    type Type = C::Type;
    type Value = PartialTracer<C>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context> Context for PartialEvaluationContext<C>
where
    C::Operation:
        PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
{
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<PartialTracer<C>, ProgramError> {
        // Lifting a staged constant produces a known value carrying an inline-constant materialization, so that
        // residual work consuming it rebuilds it as a residual-program constant rather than a residual input.
        Ok(PartialTracer::new(self.clone(), PartialEvaluationValue::known_constant(self.parent.lift(constant)?)))
    }

    fn bind<O: Into<C::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[PartialTracer<C>],
    ) -> Result<Vec<PartialTracer<C>>, ProgramError> {
        // Unwrap the input tracers into context-free partial-evaluation values, dispatch the operation's partial
        // evaluation rule against those, and rewrap the produced values with this context, mirroring how
        // `DifferentiationContext::bind` unwraps to `DifferentiationDual`s and rewraps.
        let operation = operation.into();
        operation.validate_region_count(driver.region_count())?;
        // Unresolved state is a transform-boundary error, not a value-producing rule failure. Report it before the
        // generic poison propagation below, where a zero-result operation or dead result could otherwise erase it.
        self.validate_no_unresolved_references_or_state(&operation, inputs, driver.regions())?;
        let input_values = inputs.iter().map(|input| input.value()).collect::<Result<Vec<_>, _>>();
        let error = match input_values {
            Ok(input_values) => {
                let input_values = input_values.into_iter().cloned().collect::<Vec<_>>();
                let partial_evaluation_driver = RecursivePartialEvaluationDriver { driver: &driver };
                let outputs = operation.partially_evaluate(self, &partial_evaluation_driver, input_values.as_slice());
                match outputs {
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
        let region_interfaces = driver.regions().map(RegionRef::interface).collect::<Vec<_>>();
        let Ok(output_types) = operation.infer_output_types(input_types.as_slice(), region_interfaces.as_slice())
        else {
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
    fn provenance(&self) -> Provenance {
        // A `PartialEvaluationContext` is a staging boundary for the residual program. It owns provenance state seeded
        // from its parent instead of delegating reads to the (often terminal eager context) known side.
        self.provenance.current()
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.provenance.invoke_with_origin(origin, function)
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.provenance.invoke_with_scope(scope, function)
    }

    #[inline]
    fn resolve(&self, value: &PartialTracer<C>) -> ValueResolution<C::Constant> {
        // A known value resolves exactly as the known-side inner context resolves its payload (a constant under an
        // eager inner context, staged for live tracers of an enclosing trace), while an unknown value is opaque: it
        // names a residual program variable whose value does not exist until the residual program runs.
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
/// such as [`Concretizable::concretize`](crate::Concretizable::concretize) succeed on it exactly when they succeed on
/// the carried value. This is what lets host control flow branch on known values while partial evaluation is in
/// progress. An unknown [`PartialTracer`] names a residual program variable and carries only its type.
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
// clones of one logical partial-evaluation value (witnessed by sharing one materialization slot). Two values that would
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

impl<C: Context, T: Type> ValueProjection<T> for PartialTracer<C>
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

impl<V: Value, O: Operation<Type = V::Type>> RegionRef<'_, V, O> {
    /// Partially evaluates this borrowed [`Region`](crate::Region) through the provided known-side context without
    /// materializing it. Refer to the documentation of [`Program::partially_evaluate_in_context`] for more information.
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        if inputs.len() != self.input_ids().len() {
            return Err(ProgramError::InvalidInputCount { expected: self.input_ids().len(), actual: inputs.len() });
        }

        let context = PartialEvaluationContext::new(context.clone());
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, knowledge) in inputs.iter().enumerate() {
            match knowledge {
                PartialValue::Known(value) => seed.push(PartialEvaluationValue::known_input(value.clone())),
                PartialValue::Unknown(r#type) => seed.push(context.unknown_input(r#type.clone(), index)),
            }
        }

        let outputs = context.inline_region(self, seed)?;
        context.into_evaluation(outputs)
    }

    /// Partitions this borrowed [`Region`](crate::Region) based on per-input known-ness without first detaching its
    /// source computation. Refer to the documentation of [`Program::partition`] for more information.
    pub fn partition(self, input_known: &[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>
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

        Ok(PartitionedProgram {
            known_program,
            residual_program: evaluation.program,
            known_input_indices,
            residual_inputs,
            outputs,
        })
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work eagerly.
    /// This is the main partial evaluation entry point, instantiated at this program's own [`EagerContext`] so that
    /// known values are concrete values and folding interprets each all-known [`Instruction`](crate::Instruction)
    /// immediately. [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) is the [`Context`]-taking
    /// core it delegates to and must be used instead with a [`StagingContext`] to fold known work into an enclosing
    /// trace.
    ///
    /// Partial evaluation classifies each [`Atom`](crate::Atom) as *known* (i.e., computable _now_ from the provided
    /// values) or *unknown* (i.e., dependent on a runtime input), folds the known subcomputation away, and carves the
    /// remaining unknown subcomputation into a residual [`Program`] that consumes only the unknown inputs plus the
    /// known values it actually needs. During partial evaluation, each instruction is first offered to its own
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
        O: InterpretableOperation<EagerContext<V, O>>
            + PartiallyEvaluatableOperation<EagerContext<V, O>>
            + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.partially_evaluate_in_context(&EagerContext::new(), inputs)
    }

    /// Partially evaluates this [`Program`] against the provided [`PartialValue`] inputs, folding known work through
    /// the provided known-side [`Context`]. This is the context-taking core behind
    /// [`partially_evaluate`](Self::partially_evaluate).
    #[inline]
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        &self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.entry_region_ref().partially_evaluate_in_context(context, inputs)
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
    #[inline]
    pub fn partition(&self, input_known: &[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.entry_region_ref().partition(input_known)
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayTracingContext,
        ArrayType, DataType, ReferenceIndexOperation,
    };
    use crate::contexts::{Context, StagingContext};
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::{
        AddOperation, ConditionOperation, ConstantOperation, MulOperation, NegOperation, PrintOperation, ScanOperation,
        SinOperation, Zero,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        AtomId, Concretizable, Effects, FreezeReference, FreezeReferenceOperation, NewReference, NewReferenceOperation,
        ProgramBuilder, ProgramError, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceDischarge,
        ReferenceType, RegionInterface, RegionSlot,
    };

    use super::*;

    #[test]
    fn test_partial_evaluation() {
        // Build a residual program `g(x, r) = x * r + 3`, with `x` standing for the original program's surviving
        // unknown input and `r` for a known residual feeder carrying the folded value `2`, and pair it with an
        // original output report whose first output folded to `5`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let r = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(3.0));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, r], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let evaluation = PartialEvaluation::<EagerContext<Array, ArrayOperation<Array>>> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(2.0))],
            outputs: vec![PartialEvaluationOutput::Known(Array::scalar(5.0)), PartialEvaluationOutput::Unknown(0)],
        };

        // Interpretation takes exactly one value per `Unknown` feeder, feeds `Known` feeders from their carried
        // values, returns folded outputs directly, and reads the rest from the replayed residual program:
        // `(5, 4 * 2 + 3) = (5, 11)`.
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(4.0)]),
            Ok(vec![Array::scalar(5.0), Array::scalar(11.0)]),
        );
        assert!(matches!(
            evaluation.interpret(&context, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ));
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0), Array::scalar(5.0)]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        ));

        // An output that references a residual output the residual program does not produce is reported as a
        // malformed program.
        let evaluation = PartialEvaluation::<EagerContext<Array, ArrayOperation<Array>>> {
            program: program.clone(),
            inputs: evaluation.inputs,
            outputs: vec![PartialEvaluationOutput::Unknown(1)],
        };
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0)]),
            Err(ProgramError::MalformedProgram(message))
                if message == "partial evaluation output references residual output 1 but the residual program \
                    produced 1 output(s)",
        ));

        // Under a staging known-side context, the same replay stages the residual program into the outer trace
        // instead of executing it. Its constant is lifted as a staged constant, its instructions are staged as outer
        // instructions, folded outputs return their tracers directly, and residual outputs are tracers naming the
        // staged atoms.
        let outer = ArrayTracingContext::new();
        let folded = outer.input(ArrayType::scalar(DataType::F64));
        let unknown = outer.input(ArrayType::scalar(DataType::F64));
        let feeder = outer.constant(Array::scalar(2.0));
        let evaluation = PartialEvaluation::<ArrayTracingContext> {
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
            .build::<Vec<Array>, Vec<Array>>(vec![staged], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = const 2.0
                    %3:f64[] = const 3.0
                    %4:f64[] = mul %1 %2
                    %5:f64[] = add %4 %3
                in (%5)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partial_evaluation_context() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert_eq!(
            context.parent().bind(AddOperation::new(), Vec::new(), &[Array::scalar(1.0), Array::scalar(2.0)]),
            Ok(vec![Array::scalar(3.0)]),
        );

        // `fold_or_residualize` folds an all-known operation through the known-side context, and so its outputs
        // are known values with no residual materialization decision yet.
        let inputs =
            [PartialEvaluationValue::known(Array::scalar(2.0)), PartialEvaluationValue::known(Array::scalar(3.0))];
        let folded = context.fold_or_residualize(MulOperation::new(), Vec::new(), &inputs).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_known());
        assert!(!folded[0].is_unknown());
        assert_eq!(folded[0].as_known(), Some(&Array::scalar(6.0)));
        assert_eq!(folded[0].materialization(), PartialValueMaterialization::Undecided);
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(folded[0].value(), PartialValue::Known(value) if *value == Array::scalar(6.0)));

        // `residualize` emits the operation into the residual program, materializing each known input as a fresh
        // residual input (i.e., atoms 0 and 1) and returning the instruction output as a residual variable
        // (i.e., atom 2).
        let residual = context.residualize(AddOperation::new(), Vec::new(), &inputs).unwrap();
        assert_eq!(residual.len(), 1);
        assert!(residual[0].is_unknown());
        assert_eq!(residual[0].as_known(), None);
        assert_eq!(residual[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(
            residual[0].materialization(),
            PartialValueMaterialization::Variable { residual_atom: AtomId::new(2) },
        );

        // `fold_or_residualize` residualizes as soon as any input is unknown. `neg` lands in the residual program
        // over the residual variable, producing the next residual atom.
        let mixed = context.fold_or_residualize(NegOperation::new(), Vec::new(), &[residual[0].clone()]).unwrap();
        assert_eq!(mixed[0].materialization(), PartialValueMaterialization::Variable { residual_atom: AtomId::new(3) });

        // Materializing the same known value twice reuses the residual atom assigned on first materialization through
        // the value's shared materialization slot, so a value consumed by several residualized instructions yields a
        // single residual input.
        let shared = PartialEvaluationValue::known_input(Array::scalar(4.0));
        let first = context.residualize(NegOperation::new(), Vec::new(), &[shared.clone()]).unwrap();
        let second = context.residualize(SinOperation::new(), Vec::new(), &[shared.clone()]).unwrap();
        assert_eq!(
            shared.materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(AtomId::new(4)) }
        );
        assert!(first[0].is_unknown() && second[0].is_unknown());
        assert_eq!(context.inputs.borrow().len(), 3);

        // `inline_program` replays a program over seed values. All-known seeds fold every instruction, lifting the
        // program constant into the known-side context, and so the replay returns folded values.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0));
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, x], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let outputs = context
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Array::scalar(2.0)),
                    PartialEvaluationValue::known(Array::scalar(3.0)),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(7.0)));

        // Mixed seeds fold the known work and residualize the rest, and so the walk returns residual variables.
        let outputs = context
            .inline_program(&program, vec![PartialEvaluationValue::known(Array::scalar(2.0)), residual[0].clone()])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        // `inline_partitioned_program` inlines a `Program::partition` result as two boundary operations. The known
        // side folds through the known-side context, its trailing residual-edge output feeds the residual boundary
        // operation, and the original outputs are reassembled from the two operations' outputs. The partitioned sides
        // of `f(a, x) = sin(a) * x` are a single `sin` and a single `mul`, and so those operations themselves serve as
        // arity-matching boundary operations.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Array::scalar(2.0)), residual[0].clone()],
                |_| (ArrayOperation::Sin(SinOperation::new()), Vec::new()),
                |_| (ArrayOperation::Mul(MulOperation::new()), Vec::new()),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        // An all-known partitioned program folds entirely through the known-side boundary operation, and so the
        // reassembled outputs are known values even though the (empty) residual operation is still emitted.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sine], vec![Placeholder], vec![Placeholder]).unwrap();
        let partition = program.partition(&[true]).unwrap();
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Array::scalar(2.0))],
                |_| (ArrayOperation::Sin(SinOperation::new()), Vec::new()),
                |_| (ArrayOperation::Constant(ConstantOperation::new(Array::scalar(0.0))), Vec::new()),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(2.0f64.sin())));

        // `known_constant` recovers a known value's staged-constant payload. An eager known value always resolves to a
        // constant, while under a staging known-side context only literal-backed tracers do.
        assert_eq!(context.known_constant(&Array::scalar(5.0)), Ok(Array::scalar(5.0)));
        let staging = ArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0));
        assert_eq!(staging_context.known_constant(&literal), Ok(Array::scalar(4.0)));
        assert!(matches!(
            staging_context.known_constant(&symbolic),
            Err(ProgramError::MalformedProgram(message))
                if message == "a known value crossing into a nested residual program does not resolve to a constant \
                    in the active known-side context",
        ));

        // `all_knowns_are_constants` checks every known feeder and folded output of a partial evaluation, which is only
        // non-trivial under a staging known-side context where knowns can be live tracers.
        let empty = ProgramBuilder::<Array, ArrayOperation<Array>>::new()
            .build::<Vec<Array>, Vec<Array>>(Vec::new(), Vec::new(), Vec::new())
            .unwrap();
        assert!(context.all_knowns_are_constants(&PartialEvaluation::<EagerContext<Array, ArrayOperation<Array>>> {
            program: empty.clone(),
            inputs: vec![PartialEvaluationInput::Known(Array::scalar(1.0)), PartialEvaluationInput::Unknown(0)],
            outputs: vec![PartialEvaluationOutput::Known(Array::scalar(2.0))],
        }));
        assert!(!staging_context.all_knowns_are_constants(&PartialEvaluation::<ArrayTracingContext> {
            program: empty.clone(),
            inputs: vec![PartialEvaluationInput::Known(symbolic.clone())],
            outputs: Vec::new(),
        }));
        assert!(staging_context.all_knowns_are_constants(&PartialEvaluation::<ArrayTracingContext> {
            program: empty,
            inputs: vec![PartialEvaluationInput::Known(literal.clone())],
            outputs: Vec::new(),
        }));

        // `any_known_is_symbolic` is the signal online boundary rules split on. Only a known value that does not
        // resolve to a program constant counts, and so eager knowns and unknowns never do.
        assert!(!context.any_known_is_symbolic(&[PartialEvaluationValue::known(Array::scalar(1.0))]));
        assert!(!staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(literal)]));
        assert!(staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(symbolic)]));
        assert!(!staging_context.any_known_is_symbolic(&[PartialEvaluationValue::variable(
            ArrayType::scalar(DataType::F64),
            AtomId::new(0)
        )]),);
    }

    #[test]
    fn test_partial_evaluation_rejects_state_before_folding_or_residual_emission() {
        /// Test operation family whose higher-order carrier encloses state under a dormant rule region.
        #[derive(Clone, Debug)]
        enum HigherOrderStateOperation {
            /// Binary operation carrying one executable computation region.
            HigherOrder,

            /// Unary operation carrying one dormant transformation-rule region.
            RuleCarrier,

            /// Unary unresolved-state operation.
            State,
        }

        impl Operation for HigherOrderStateOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                match self {
                    Self::HigherOrder => "higher_order",
                    Self::RuleCarrier => "rule_carrier",
                    Self::State => "state",
                }
            }

            fn region_slots(&self) -> &'static [RegionSlot] {
                match self {
                    Self::HigherOrder => const { &[RegionSlot::computation("body")] },
                    Self::RuleCarrier => const { &[RegionSlot::rule("rule")] },
                    Self::State => &[],
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let expected = if matches!(self, Self::HigherOrder) { 2 } else { 1 };
                check_count!("input", input_types, expected, TypeError);
                Ok(vec![input_types[0].clone()])
            }

            fn effects(&self) -> Effects {
                if matches!(self, Self::State) { Effects::single(Effect::OrderedState) } else { Effects::PURE }
            }
        }

        impl InterpretableOperation<EagerContext<Array, Self>> for HigherOrderStateOperation {
            fn interpret<D: InterpretationDriver<EagerContext<Array, Self>>>(
                &self,
                _context: &EagerContext<Array, Self>,
                _driver: &D,
                _inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                panic!("the state-bearing higher-order operation must be rejected before interpretation")
            }
        }

        impl PartiallyEvaluatableOperation<EagerContext<Array, Self>> for HigherOrderStateOperation {}

        // The executable body is superficially pure: its only state lives in a dormant rule attached to its carrier.
        // The complete closure scan must still find that state rather than trusting ordinary executable effects.
        let scalar_type = ArrayType::scalar(DataType::F64);
        let mut state_builder = ProgramBuilder::<Array, HigherOrderStateOperation>::new();
        let state_input = state_builder.add_input(scalar_type.clone());
        let state_output = state_builder
            .add_instruction(HigherOrderStateOperation::State, Vec::new(), vec![state_input], None)
            .unwrap()[0];
        let state = state_builder
            .build::<Vec<Array>, Vec<Array>>(vec![state_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut body_builder = ProgramBuilder::<Array, HigherOrderStateOperation>::new();
        let state_region = body_builder.import_region(state.entry_region_ref());
        let body_input = body_builder.add_input(scalar_type.clone());
        let body_output = body_builder
            .add_instruction(HigherOrderStateOperation::RuleCarrier, vec![state_region], vec![body_input], None)
            .unwrap()[0];
        let body = body_builder
            .build::<Vec<Array>, Vec<Array>>(vec![body_output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(body.effects(), Effects::PURE);
        assert!(body.entry_region_ref().contains_effect_in_closure(Effect::OrderedState));

        let expected = ProgramError::UnsupportedOperation {
            message: "`higher_order` carries unresolved state in an attached region and must be discharged before \
                partial evaluation"
                .to_string(),
        };

        // With every operand known, the pre-branch gate prevents eager interpretation of the higher-order carrier.
        let context = PartialEvaluationContext::new(EagerContext::<Array, HigherOrderStateOperation>::new());
        let known = [
            PartialEvaluationValue::known(Array::scalar(1.0_f64)),
            PartialEvaluationValue::known(Array::scalar(2.0_f64)),
        ];
        assert_eq!(
            context
                .fold_or_residualize(HigherOrderStateOperation::HigherOrder, vec![body.clone()], &known)
                .map(|_| ()),
            Err(expected.clone()),
        );

        // With one unknown operand, the same gate runs before residual emission can import the state-bearing closure.
        let context = PartialEvaluationContext::new(EagerContext::<Array, HigherOrderStateOperation>::new());
        let mixed = [PartialEvaluationValue::known(Array::scalar(1.0_f64)), context.unknown_input(scalar_type, 0)];
        assert_eq!(
            context
                .fold_or_residualize(HigherOrderStateOperation::HigherOrder, vec![body.clone()], &mixed)
                .map(|_| ()),
            Err(expected.clone()),
        );

        // The public residual-emission sink independently enforces the same closure check so custom rules cannot
        // bypass the default fold-or-residualize policy and import unresolved state directly.
        let residual_state = {
            let builder = context.builder.borrow();
            (
                builder.atoms().len(),
                builder.input_ids().len(),
                builder.instructions().len(),
                builder.regions.len(),
                context.inputs.borrow().len(),
                context.staged_feeders.borrow().len(),
            )
        };
        assert_eq!(
            context.residualize(HigherOrderStateOperation::HigherOrder, vec![body], &mixed).map(|_| ()),
            Err(expected),
        );
        {
            let builder = context.builder.borrow();
            assert_eq!(
                (
                    builder.atoms().len(),
                    builder.input_ids().len(),
                    builder.instructions().len(),
                    builder.regions.len(),
                    context.inputs.borrow().len(),
                    context.staged_feeders.borrow().len(),
                ),
                residual_state,
            );
        }
        assert_eq!(
            context.residualize(HigherOrderStateOperation::State, Vec::new(), &mixed[..1]).map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "`state` must be discharged before partial evaluation".to_string(),
            }),
        );
        {
            let builder = context.builder.borrow();
            assert_eq!(
                (
                    builder.atoms().len(),
                    builder.input_ids().len(),
                    builder.instructions().len(),
                    builder.regions.len(),
                    context.inputs.borrow().len(),
                    context.staged_feeders.borrow().len(),
                ),
                residual_state,
            );
        }
    }

    #[test]
    fn test_partial_evaluation_rejects_reference_types_before_seeding_or_replay() {
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ArrayIrType::Reference(ReferenceType::new(array_type.clone()));
        let expected = ProgramError::UnsupportedOperation {
            message: "references must be discharged before partial evaluation".to_string(),
        };

        let mut passthrough_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = passthrough_builder.add_input(reference_type.clone());
        let passthrough = passthrough_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            passthrough.partially_evaluate(&[PartialValue::Unknown(reference_type.clone())]).map(|_| ()),
            Err(expected.clone()),
        );

        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let residual_state = {
            let builder = context.builder.borrow();
            (
                builder.atoms().len(),
                builder.input_ids().len(),
                builder.instructions().len(),
                builder.regions.len(),
                context.inputs.borrow().len(),
                context.staged_feeders.borrow().len(),
            )
        };
        assert_eq!(
            context
                .residualize(
                    ArrayIrOperation::Condition(ConditionOperation::new()),
                    vec![passthrough.clone(), passthrough.clone()],
                    &[],
                )
                .map(|_| ()),
            Err(ProgramError::UnsupportedOperation {
                message: "`condition` carries unresolved references in an attached region and must be discharged \
                    before partial evaluation"
                    .to_string(),
            }),
        );
        {
            let builder = context.builder.borrow();
            assert_eq!(
                (
                    builder.atoms().len(),
                    builder.input_ids().len(),
                    builder.instructions().len(),
                    builder.regions.len(),
                    context.inputs.borrow().len(),
                    context.staged_feeders.borrow().len(),
                ),
                residual_state,
            );
        }

        let mut unused_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        unused_builder.add_input(reference_type.clone());
        let array = unused_builder.add_input(array_type.clone().into());
        let unused = unused_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![array],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let outer = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        assert_eq!(
            unused
                .partially_evaluate_in_context(
                    &outer,
                    &[PartialValue::Unknown(reference_type.clone()), PartialValue::Unknown(array_type.clone().into()),],
                )
                .map(|_| ()),
            Err(expected.clone()),
        );
        let outer_builder = outer.builder().borrow();
        assert!(outer_builder.atoms().is_empty());
        assert!(outer_builder.input_ids().is_empty());
        assert!(outer_builder.instructions().is_empty());
        assert!(outer_builder.regions.is_empty());
        drop(outer_builder);

        let mut array_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = array_builder.add_input(array_type.clone().into());
        let array_passthrough = array_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let concrete_reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        assert_eq!(
            array_passthrough.partially_evaluate(&[PartialValue::Known(concrete_reference.clone())]).map(|_| ()),
            Err(expected.clone()),
        );
        assert_eq!(
            array_passthrough.partially_evaluate(&[PartialValue::Unknown(reference_type.clone())]).map(|_| ()),
            Err(expected.clone()),
        );

        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let unknown_reference = context.unknown_input(reference_type, 0);

        // A reference-view operation is pure, so it never reaches the ordered-state rejection above. The
        // reference-typed operand itself is what the operation-named diagnostic reports at the fold boundary.
        assert!(matches!(
            context.fold_or_residualize(
                ArrayIrOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0)),
                Vec::new(),
                std::slice::from_ref(&unknown_reference),
            ),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`reference_index` consumes unresolved references and must be discharged before \
                    partial evaluation",
        ));

        let observer = context.clone();
        assert_eq!(context.into_evaluation(Vec::new()).map(|_| ()), Err(expected.clone()));
        let builder = observer.builder.borrow();
        assert_eq!(builder.atoms().len(), 1);
        assert_eq!(builder.input_ids().len(), 1);
        assert!(builder.instructions().is_empty());
        assert!(builder.regions.is_empty());
        drop(builder);
        drop(unknown_reference);

        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let observer = context.clone();
        assert_eq!(
            context.into_evaluation(vec![PartialEvaluationValue::known(concrete_reference)]).map(|_| ()),
            Err(expected.clone()),
        );
        let builder = observer.builder.borrow();
        assert!(builder.atoms().is_empty());
        assert!(builder.input_ids().is_empty());
        assert!(builder.instructions().is_empty());
        assert!(builder.regions.is_empty());
        drop(builder);

        let context =
            PartialEvaluationContext::new(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        context.builder.borrow_mut().import_region(passthrough.entry_region_ref());
        let observer = context.clone();
        assert_eq!(context.into_evaluation(Vec::new()).map(|_| ()), Err(expected));
        let builder = observer.builder.borrow();
        assert!(builder.atoms().is_empty());
        assert!(builder.input_ids().is_empty());
        assert!(builder.instructions().is_empty());
        assert_eq!(builder.regions.len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_as_context() {
        // Over an eager known-side inner context, the partial-evaluation context is itself eager: lifted constants and
        // all-known binds fold to known values that resolve `Constant`. A known `bool[]` also supports boolean
        // concretization, which lets host control flow branch on known values during evaluation.
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        assert!(context.is_eager());
        let lifted = context.lift(Array::scalar(2.0)).unwrap();
        assert!(matches!(
            lifted.value().unwrap().materialization(),
            PartialValueMaterialization::Constant { residual_atom: None },
        ));
        assert!(matches!(context.resolve(&lifted), ValueResolution::Constant(value) if value == Array::scalar(2.0)));
        let truth = context.lift(Array::scalar(true)).unwrap();
        assert_eq!(truth.concretize(), Ok(true));
        let folded = context.bind(AddOperation::new(), Vec::new(), &[lifted.clone(), lifted.clone()]).unwrap();
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].value().unwrap().as_known(), Some(&Array::scalar(4.0)));
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));

        // The `Zero` capability binds a nullary `ZeroOperation`, which is vacuously all-known and folds through the
        // inner context to a known zero.
        let zero = context.zero(&ArrayType::scalar(DataType::Boolean)).unwrap();
        assert_eq!(zero.value().unwrap().as_known(), Some(&Array::scalar(false)));
        assert_eq!(zero.concretize(), Ok(false));

        // A mixed bind residualizes: the unknown input names a residual program variable, the known input
        // materializes as a residual input, and the output is an unknown value that resolves `Opaque` and rejects
        // concretizing extractions.
        let unknown_atom = context.builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        context.inputs.borrow_mut().push(PartialEvaluationInput::Unknown(0));
        let unknown = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::variable(ArrayType::scalar(DataType::F64), unknown_atom),
        );
        let mixed = context.bind(MulOperation::new(), Vec::new(), &[folded[0].clone(), unknown.clone()]).unwrap();
        assert!(mixed[0].value().unwrap().is_unknown());
        assert!(matches!(context.resolve(&mixed[0]), ValueResolution::Opaque));
        assert!(matches!(mixed[0].concretize(), Err(ProgramError::Concretization { .. })));

        // Finalizing the context (after dropping every stamped clone) produces the accumulated residual program:
        // `(ẏ) = folded * ẋ` over the unknown input plus the materialized known feeder.
        let output = mixed[0].value().unwrap().clone();
        drop((lifted, truth, folded, zero, unknown, mixed));
        let evaluation = context.into_evaluation(vec![output]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(4.0))],
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
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
        let outer_a = ArrayTracingContext::new();
        let outer_b = ArrayTracingContext::new();
        let context = PartialEvaluationContext::new(outer_a.clone());
        let known_a = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::known(outer_a.input(ArrayType::scalar(DataType::F64))),
        );
        let known_b = PartialTracer::new(
            context.clone(),
            PartialEvaluationValue::known(outer_b.input(ArrayType::scalar(DataType::F64))),
        );
        let poisoned = context.bind(AddOperation::new(), Vec::new(), &[known_a.clone(), known_b]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert_eq!(format!("{}", poisoned[0]), "<poison:f64[]>");
        assert_eq!(poisoned[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(context.resolve(&poisoned[0]), ValueResolution::Opaque));
        assert!(matches!(poisoned[0].concretize(), Err(ProgramError::MismatchedProgramBuilders)));

        // Poison propagates from inputs to outputs of later binds, and unwrapping at a boundary reports the original
        // deferred error rather than a generic poison error.
        let propagated = context.bind(MulOperation::new(), Vec::new(), &[known_a, poisoned[0].clone()]).unwrap();
        assert!(matches!(propagated[0].value(), Err(ProgramError::MismatchedProgramBuilders)));
        assert!(matches!(propagated[0].clone().into_value(), Err(ProgramError::MismatchedProgramBuilders)));
    }

    #[test]
    fn test_program_partially_evaluate() {
        // `f(a, x) = (a * a, a * a * x + 1, a * a + x)` with `a` known and `x` unknown: the `a * a` subcomputation
        // folds to a known output, its two residual consumers share one residual feeder, and the literal is rebuilt
        // inline as a residual constant instead of becoming a feeder.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0));
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let offset = builder.add_instruction(AddOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared, shifted, offset], vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0)),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(9.0)),]
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0)),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = const 1.0
                    %4:f64[] = add %2 %3
                    %5:f64[] = add %1 %0
                in (%4, %5)
            "}
            .trim_end(),
        );

        // Replaying the partial evaluation at a concrete unknown input matches interpreting the original program.
        assert_eq!(
            evaluation.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[Array::scalar(4.0)]),
            Ok(vec![Array::scalar(9.0), Array::scalar(37.0), Array::scalar(13.0)]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(3.0), Array::scalar(4.0)]),
            Ok(vec![Array::scalar(9.0), Array::scalar(37.0), Array::scalar(13.0)]),
        );

        // All-known inputs fold the whole program away: every output is known and the residual program is empty.
        let evaluation = program
            .partially_evaluate(&[PartialValue::Known(Array::scalar(3.0)), PartialValue::Known(Array::scalar(4.0))])
            .unwrap();
        assert_eq!(evaluation.inputs, Vec::new());
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0)),
                PartialEvaluationOutput::Known(Array::scalar(37.0)),
                PartialEvaluationOutput::Known(Array::scalar(13.0)),
            ]
        );
        assert!(evaluation.program.instructions().is_empty());

        // All-unknown inputs residualize the whole program unchanged, with the literal rebuilt inline at its first
        // residual use rather than up front.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
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
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %0
                    %3:f64[] = mul %2 %1
                    %4:f64[] = const 1.0
                    %5:f64[] = add %3 %4
                    %6:f64[] = add %2 %1
                in (%2, %5, %6)
            "}
            .trim_end(),
        );

        // Effectful operations place by input known-ness. An all-known `print` folds (firing its effect at partial
        // evaluation time), while a mixed-input `print` residualizes and is kept in the residual program even when
        // no output consumes it.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let printed = builder.add_instruction(PrintOperation::new("known"), Vec::new(), vec![a], None).unwrap()[0];
        builder.add_instruction(PrintOperation::new("dead"), Vec::new(), vec![x], None).unwrap();
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![printed, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(2.0)),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(2.0)),]
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = print [label=dead] %0
                    %3:f64[] = mul %1 %0
                in (%3)
            "}
            .trim_end(),
        );

        // The number of provided inputs must match the number of program inputs.
        assert!(matches!(
            program.partially_evaluate(&[PartialValue::Known(Array::scalar(1.0))]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        ));
    }

    #[test]
    fn test_program_partially_evaluate_preserves_residual_provenance() {
        // `f(a, x) = (a * a * x + 1, print(x))` with `a` known and `x` unknown. Every residual instruction is a
        // deferred rewrite of one source instruction, so it must carry that instruction's provenance. The folded
        // known-side `a * a` contributes no residual instruction and so its scope must not appear.
        let scoped = |name: &str| Provenance::scope(ProvenanceScope::new(name), Provenance::unknown());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let one = builder.add_constant(Array::scalar(1.0));
        let squared =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], Some(scoped("known"))).unwrap()[0];
        let scaled = builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], Some(scoped("scaled")))
            .unwrap()[0];
        let shifted = builder
            .add_instruction(AddOperation::new(), Vec::new(), vec![scaled, one], Some(scoped("shifted")))
            .unwrap()[0];
        let printed = builder
            .add_instruction(PrintOperation::new("x"), Vec::new(), vec![x], Some(scoped("printed")))
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![shifted, printed], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0)),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>(),
            // The effectful `print` residualizes ahead of the pure chain, which is a placement property of partial
            // evaluation; what matters here is that each residual instruction carries its own source provenance.
            vec![("print", scoped("printed")), ("mul", scoped("scaled")), ("add", scoped("shifted"))],
        );
    }

    #[test]
    fn test_program_partially_evaluate_rejects_unresolved_references() {
        let input_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let (_, source) = TracingContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |input| {
                let reference = input.new_reference()?;
                reference.add_update(&input)?;
                reference.freeze()
            },
            input_type,
        )
        .unwrap();
        let source = source.to_flat_program();
        assert!(matches!(
            source.partially_evaluate(&[PartialValue::Known(ArrayIrValue::Array(Array::scalar(3.0_f32)))]),
            Err(ProgramError::UnsupportedOperation { message })
                if message == "`new_reference` must be discharged before partial evaluation",
        ));
    }

    #[test]
    fn test_program_partially_evaluate_after_local_reference_discharge_across_scan() {
        let array_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(array_type.clone());
        let mut body_builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let update = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0_f32)));
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        let mut builder = ProgramBuilder::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new();
        let input = builder.add_input(array_type.clone().into());
        let reference =
            builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder
            .add_instruction(ScanOperation::<ArrayIrValue<Array>>::new(1, 3), vec![body], vec![reference], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let source = builder
            .build::<Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // Discharge turns the scan's mutated reference into an ordinary carry before partial evaluation runs, so an
        // unknown boundary residualizes the whole three-iteration loop as a pure reference-free program.
        let evaluation = source
            .discharge_local_references(0, "partial evaluation")
            .unwrap()
            .partially_evaluate(&[PartialValue::Unknown(array_type.into())])
            .unwrap();
        assert!(evaluation.program().effects().is_pure());
        assert!(!evaluation.program().entry_region_ref().contains_atom_type_in_closure(Type::is_reference));
        assert_eq!(
            evaluation.program().interpret(vec![ArrayIrValue::Array(Array::scalar(3.0_f32))]),
            Ok(vec![ArrayIrValue::Array(Array::scalar(6.0_f32))]),
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context() {
        // `f(a, x) = (a * a) * x + 1` with `a` known as a live tracer of an enclosing trace and `x` unknown: the known
        // `a * a` folds by staging into the outer program, the residual program consumes its staged result through a
        // known feeder naming the outer atom, and the literal is rebuilt inline as a residual constant.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0));
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![shifted], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = ArrayTracingContext::new();
        let known = outer.input(ArrayType::scalar(DataType::F64));
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(known), PartialValue::Unknown(ArrayType::scalar(DataType::F64))],
            )
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
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = const 1.0
                    %4:f64[] = add %2 %3
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
            .build::<Vec<Array>, Vec<Array>>(vec![AtomId::new(2)], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = const 1.0
                    %2:f64[] = mul %0 %0
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
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sine, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled, product], vec![Placeholder; 2], vec![Placeholder; 2])
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
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                    %2:f64[] = sin %0
                in (%1, %2)
            "}
            .trim_end(),
        );
        assert_eq!(
            partition.residual_program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                in (%2)
            "}
            .trim_end(),
        );

        // The two sides recombine to the original program: interpret the known program at `a`, feed its trailing
        // residual-edge output to the residual program together with `x`, and interleave per the outputs report.
        let known_outputs = partition.known_program.interpret(vec![Array::scalar(2.0)]).unwrap();
        let residual_outputs =
            partition.residual_program.interpret(vec![Array::scalar(3.0), known_outputs[1].clone()]).unwrap();
        assert_eq!(known_outputs[0], Array::scalar(4.0));
        assert_eq!(residual_outputs, vec![Array::scalar(3.0 * 2.0f64.sin())]);

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
