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
//! construct the surrounding closure. The context retains the first failure even when outputs are discarded or absent,
//! skips subsequent operations, and reports the original [`ProgramError`] at the partial-evaluation boundary. Escaped
//! context or value clones keep shared builders alive and are rejected during finalization with
//! [`ProgramError::EscapedProgramBuilder`].
//!
//! # Control Flow, Effects, and Recursion
//!
//! Higher-order rules receive a [`PartialEvaluationDriver`] for recursively transforming attached regions. A rule may
//! inline selected nested work. Uninlined mixed work remains attached to a residual operation. Effectful operations
//! fold when their inputs are known, subject to the ordering rules on [`PartialEvaluationContext`]: once an ordered
//! operation residualizes, every later ordered operation residualizes, including accesses of other references and
//! other effect classes. This preserves observable failures
//! and synchronization before later I/O and mutations. Splitting separately invoked programs requires source-level
//! validation of reference dependencies and allocation lifetimes. Live references cross the
//! boundary by identity as [`Known`](PartialEvaluationInput::Known) reference feeders (refer to
//! [`PartialEvaluation::known_reference_inputs`] for more information), reference-typed constants embed inline,
//! and a [`ReferencePlacement`] decides whether known reference operations execute or stage under an eager parent.
//! Speculative fixed-point probes never execute effectful bodies.
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
use std::collections::{HashMap, HashSet};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;

use crate::contexts::{Context, Domain, EagerContext, StagingContext, ValueResolution};
use crate::interpretation::InterpretableOperation;
use crate::macros::check_count;
use crate::parameters::{Parameter, Placeholder};
use crate::programs::{
    AtomId, BindingRegionDriver, EffectClass, EffectClasses, EffectsSummary, EmptyRegionDriver, FlatProgram,
    InstructionId, Operation, Program, ProgramBuilder, ProgramError, ProjectedValue, Provenance, ProvenanceScope,
    ProvenanceState, ReferenceAccessMode, ReferenceAnalysis, ReferenceIdentity, ReferenceRoot, RegionDriver, RegionRef,
    RegionReplayMappings, RegionRole, ReplayRegionDriver, Type, TypeError, TypeIdentityPosition, Typed, Value, ValueId,
    ValueProjection,
};
use crate::tracing::TracingContext;

/// State of a [`Value`] during partial evaluation. A [`PartialValue`] is the value domain the partial context
/// interprets a [`Program`] over. Every [`Atom`](crate::Atom) and every intermediate result is either
/// [`Known`](Self::Known) (i.e., a concrete value available now) or [`Unknown`](Self::Unknown) (i.e., only its
/// [`Type`] is available until the residual program runs). For more information on partial evaluation, refer to
/// the documentation of [`Program::partially_evaluate`].
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
    /// Residual input fed by a value that partial evaluation folded to a concrete known residual value. Note that a
    /// reference-typed known feeder is the live reference handle itself, threaded by identity and never as a snapshot
    /// of its contents, so that the residual program accesses the state as it is when the residual program runs. Refer
    /// to the documentation of [`PartialEvaluation::known_reference_inputs`] for more information.
    Known(V),

    /// Residual input fed by an unknown input of the original program, identified by that input's index in the
    /// original program's inputs.
    Unknown(usize),
}

impl<V> PartialEvaluationInput<V> {
    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Known`].
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationInput`] is [`Self::Unknown`].
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
    pub const fn is_known(&self) -> bool {
        matches!(self, Self::Known(_))
    }

    /// Returns `true` if this [`PartialEvaluationOutput`] is [`Self::Unknown`].
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

    /// Returns the positions of reference-typed [`Known`](PartialEvaluationInput::Known) values among
    /// [`inputs`](Self::inputs). These positions index the residual program's inputs and not the original program's
    /// inputs. Unknown reference inputs are excluded.
    ///
    /// A known reference input carries a live handle and not a snapshot of its mutable contents. Under an eager known
    /// side it is the handle itself. Under a staging known side it is the tracer naming the outer program's reference
    /// atom. The residual program therefore observes the state when it runs. Such an input is needed when an access
    /// must stage (e.g., because of an unknown operand, an earlier deferred ordered effect, or
    /// [`Stage`](ReferencePlacement::Stage) placement) or when the residual program forwards the handle.
    ///
    /// Reference-typed program constants are excluded too: replay lifts them inline, so the residual program reaches
    /// them without an input carrying a known value.
    #[inline]
    pub fn known_reference_inputs(&self) -> impl '_ + Iterator<Item = usize> {
        self.inputs.iter().enumerate().filter_map(|(index, input)| match input {
            PartialEvaluationInput::Known(value) if value.r#type().is_reference() => Some(index),
            _ => None,
        })
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
                PartialEvaluationInput::Known(value) => value.clone(),
                PartialEvaluationInput::Unknown(_) => {
                    // The `.unwrap()` here is safe because of the earlier check for `inputs.len()`.
                    remaining_inputs.next().cloned().unwrap()
                }
            })
            .collect::<Vec<_>>();
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

/// Reference identity shared across a [`PartitionedProgram`] boundary. Allocations belong to their emitted invocation
/// even when the two independently constructed programs reuse region and instruction identifiers.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum PartitionReferenceRoot {
    /// Original input shared by the partition's boundary wiring.
    Input(usize),

    /// Allocation created by the known invocation, including nested local allocations.
    KnownAllocation(ReferenceRoot),

    /// Allocation created afresh by a residual invocation, including nested local allocations.
    ResidualAllocation(ReferenceRoot),
}

/// Records which effects must keep their relative execution order when partial evaluation separates known work from
/// residual work. During source replay, this type accumulates the constraints of work already deferred to the residual
/// invocation: a later instruction that conflicts with those constraints cannot move ahead of that work into the known
/// invocation. A [`PartitionedProgram`] also records constraints for each half so callers can check whether separating
/// their execution would cross an effect dependency.
///
/// Ordinary partitioning uses global ordering for ordered effects. Repeated invocation partitioning can use ordering
/// per reference allocation when reference analysis and the caller's independence contract justify it. For example,
/// a deferred write to one allocation prevents a later read of that allocation from moving ahead of it, but need not
/// prevent work on an independent allocation from moving. These constraints address effects only; value dependencies
/// and other requirements for a valid partition are checked separately.
///
/// The key `K` identifies an allocation, using either [`ReferenceRoot`] during source replay or
/// [`PartitionReferenceRoot`] when comparing the two halves of a partition. Views of the same allocation share a key.
#[derive(Clone, Debug)]
enum EffectOrdering<K: Eq + Hash> {
    /// Requires work on any of these reference allocations to stay ordered relative to other work on the same
    /// allocation. For example, sets containing `{a, b}` and `{b, c}` conflict because both include `b`, while `{a}`
    /// and `{c}` do not. Accesses through different views of an allocation still conflict, even when the views select
    /// different elements. This is an allocation-level ordering requirement and not a read/write hazard analysis.
    /// An empty set imposes no effect-ordering constraint and does not conflict even with [`Global`](Self::Global).
    PerReference(HashSet<K>),

    /// Requires work to stay ordered relative to any other work with a nonempty ordering constraint, whether that
    /// constraint is global or per reference. Pure work does not become ordered merely because this variant is present.
    /// Ordinary partitioning uses this variant for every ordered effect. Repeated invocation partitioning also uses
    /// it when ordering cannot be limited to identified reference allocations, such as for ordered I/O or opaque state.
    /// For example, a deferred operation with global ordering prevents a later ordered reference write from moving
    /// ahead of it, even if that write touches an otherwise independent allocation.
    Global,
}

impl<K: Eq + Hash> EffectOrdering<K> {
    /// Returns whether this ordering imposes no constraints.
    fn is_empty(&self) -> bool {
        matches!(self, Self::PerReference(references) if references.is_empty())
    }

    /// Returns whether the work described by `self` and `other` must keep its relative ordering. Two per-reference
    /// constraints conflict when they share an allocation and a global constraint conflicts with every nonempty
    /// constraint. An empty constraint never conflicts.
    ///
    /// During replay, `self` can describe previously deferred work and `other` a later instruction. A conflict then
    /// prevents that instruction from moving ahead of the deferred work. A `false` result only establishes that these
    /// effect constraints do not prevent the move; it does not establish that the move satisfies value dependencies.
    fn conflicts(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::PerReference(left), Self::PerReference(right)) => !left.is_disjoint(right),
            _ => !self.is_empty() && !other.is_empty(),
        }
    }

    /// Updates `self` to describe the constraints of both pieces of work. For example, after deferring work on
    /// allocation `a` and then work on allocation `b`, the accumulated constraint contains both allocations, so later
    /// work on either one conflicts with it.
    ///
    /// If either constraint is global, the result is global and individual allocation keys are no longer retained.
    /// Adding an empty constraint leaves `self` unchanged. Replay uses this function to accumulate constraints as
    /// more work is deferred; the function itself does not place or reorder any instructions.
    fn extend(&mut self, other: &Self)
    where
        K: Clone,
    {
        match (self, other) {
            (Self::PerReference(left), Self::PerReference(right)) => left.extend(right.iter().cloned()),
            (ordering, Self::Global) => *ordering = Self::Global,
            (Self::Global, _) => {}
        }
    }
}

impl<K: Eq + Hash> Default for EffectOrdering<K> {
    #[inline]
    fn default() -> Self {
        Self::PerReference(HashSet::new())
    }
}

impl EffectsSummary {
    /// Derives a conservative partition [`EffectOrdering`] from the provided resolved reference allocations. Ordinary
    /// partitioning uses [`EffectOrdering::Global`] for every ordered effect instead. [`EffectOrdering::PerReference`]
    /// applies only to repeated invocation; opaque state, ordered non-reference effects, and unaccounted ordered work
    /// retain global ordering.
    fn effect_ordering<K: Eq + Hash, R: IntoIterator<Item = K>>(self, references: R) -> EffectOrdering<K> {
        if !self.classes().is_ordered() {
            return EffectOrdering::default();
        }
        if self.has_explicit_ordered_state()
            || self.classes().into_iter().any(|class| class.is_ordered() && class != EffectClass::OrderedState)
        {
            return EffectOrdering::Global;
        }
        let ordering = EffectOrdering::PerReference(references.into_iter().collect());
        if ordering.is_empty() { EffectOrdering::Global } else { ordering }
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

    /// Effect-ordering constraints for the known program and residual program, respectively. By default, all ordered
    /// effects must retain their relative execution order. When partitioning work into a known invocation followed by
    /// repeated residual invocations (for example, linearization followed by pushforward calls), reference analysis
    /// can establish separate ordering constraints for independent allocations. References in both programs are then
    /// identified relative to the original inputs so that accesses to the same allocation can be compared across the
    /// partition boundary; allocations created within either program have separate identities.
    effect_ordering: [EffectOrdering<PartitionReferenceRoot>; 2],
}

impl<V: Value, O: Operation<Type = V::Type>> PartitionedProgram<V, O> {
    /// Reassembles an internally constructed [`PartitionedProgram`] whose programs and boundary wiring were validated
    /// together. The caller preserves the known-output prefix and residual feeder ordering documented on
    /// [`PartitionedProgram`].
    pub(crate) fn from_parts(
        known_program: Program<V, O, Vec<V>, Vec<V>>,
        residual_program: Program<V, O, Vec<V>, Vec<V>>,
        known_input_indices: Vec<usize>,
        residual_inputs: Vec<PartialEvaluationInput<usize>>,
        outputs: Vec<PartialEvaluationOutput<usize>>,
    ) -> Self {
        let effect_ordering = [&known_program, &residual_program].map(|program| {
            if program.effects().classes().is_ordered() { EffectOrdering::Global } else { EffectOrdering::default() }
        });
        Self { known_program, residual_program, known_input_indices, residual_inputs, outputs, effect_ordering }
    }

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
    /// _value_ erased to a position/index: [`Unknown`](PartialEvaluationInput::Unknown) entries keep their original
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

    /// Returns the residual input positions that receive known reference values from the known program, in residual
    /// input order. Unknown reference inputs, known non-reference inputs, and inline constants are excluded. These
    /// positions describe boundary wiring, not whether the two programs access overlapping reference allocations.
    #[inline]
    pub fn known_reference_inputs(&self) -> impl '_ + Iterator<Item = usize> {
        self.residual_inputs
            .iter()
            .zip(self.residual_program.inputs())
            .enumerate()
            .filter_map(|(index, (input, atom))| (input.is_known() && atom.r#type().is_reference()).then_some(index))
    }

    /// Returns whether effects in the known program must stay ordered relative to effects in the residual program.
    /// This compares the ordering constraints recorded when the partition was built. Constraints on the same reference
    /// allocation conflict, and global ordering conflicts with any nonempty constraint in the other program.
    ///
    /// For example, splitting a loop into one loop that runs all known work followed by another that runs all residual
    /// work changes the order of work across iterations. If the known work reads a reference that the residual work
    /// writes, the split could make the next iteration's read run before the preceding iteration's write. This
    /// function returns `true` for that conflict, allowing the caller to keep the original loop intact.
    ///
    /// A `false` result means only that the recorded effect constraints do not prevent such a split. The caller must
    /// still check value dependencies, loop-carried values, shapes, and how intermediate values are stored. The query
    /// does not inspect runtime reference identities; it relies on the assumptions used to construct the partition.
    pub(crate) fn has_effect_ordering_conflicts(&self) -> bool {
        self.effect_ordering[0].conflicts(&self.effect_ordering[1])
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

    /// Builds separate known and residual programs for `region`, treating input `i` as known when `input_known[i]`
    /// is `true`. Both programs are constructed in a fresh staging context, so even known operations are recorded
    /// rather than executed through the active context.
    ///
    /// The new evaluation uses `context`'s execution configuration but tracks effect ordering and pending errors
    /// separately. For example, a loop rule can try a partition, discover that a loop-carried value must be unknown,
    /// and try again without executing reference writes or changing which later operations the active evaluation
    /// must defer. Effects already deferred by the active evaluation likewise do not constrain this fresh partition.
    fn partition_program(
        &self,
        context: &PartialEvaluationContext<C>,
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
        _context: &PartialEvaluationContext<C>,
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

    /// Specifies whether partitioning must account for a known computation running once and its results being reused
    /// across separate residual calls. The same requirement applies when recursively partitioning nested computations.
    ///
    /// For example, linearizing a fused custom Jacobian-Vector Product (JVP) function at a fixed primal input computes
    /// the primal result and reusable coefficients once, then produces a pushforward callable with different tangent
    /// inputs. With this flag set, an accumulator used only by the pushforward must be allocated afresh on each call,
    /// even if its initial value is a known zero. Nested partitions determine which allocations belong to each call
    /// and use reference analysis to preserve ordering between accesses to the same allocation.
    ///
    /// Without this flag, nested computations use the standard partial evaluation rules. For example, specializing a
    /// function with a fixed scalar argument can fold pure arithmetic involving that argument while leaving work that
    /// depends on other arguments in the residual program. Ordered effects retain their relative order, and reference
    /// placement follows the context's configuration rather than the repeated-call allocation analysis. The resulting
    /// specialized program can still be called more than once (this flag selects how partitioning assigns state and
    /// preserves effect ordering, and does not impose a limit on the number of residual calls).
    repeated_residual: bool,
}

impl<D> RecursivePartialEvaluationDriver<'_, D> {
    /// Evaluates one source instruction while preserving the ordering and allocation requirements of its replay.
    /// Without reference analysis, this dispatches through the active context unless the caller explicitly requires
    /// residual execution. With analysis, independent allocations may be evaluated separately, but an instruction
    /// cannot move ahead of earlier deferred work on the same allocation or work requiring global ordering.
    ///
    /// Nested partitioning replaces arguments with fresh symbolic inputs. If two source arguments name the same
    /// allocation, their alias relationship would be lost in that replacement. Such an ordered application is kept
    /// whole (i.e., it can fold with known inputs, or remain residual, but cannot be split recursively).
    ///
    /// # Parameters
    ///
    ///   - `context`: Active evaluation whose residual builder receives emitted work. Only replay with reference
    ///     analysis creates a clone with instruction-local ordering state; ordinary replay uses this context directly.
    ///   - `region`: Source region containing the instruction and its reference identities.
    ///   - `instruction_index`: Instruction index within the source region.
    ///   - `inputs`: Partially evaluated operands in source operand order.
    ///   - `deferred_instructions`: Source instructions explicitly required to remain residual, including allocations
    ///     that must be created afresh for each residual call.
    ///   - `deferred_effect_ordering`: Accumulated ordering constraints of earlier deferred source work. Updated only
    ///     when reference analysis is supplied; nested emission still enforces ordering within this instruction.
    ///   - `reference_analysis`: Canonical source-reference analysis for repeated residual calls. Omitting it retains
    ///     the active context's ordering rules without performing source-reference analysis.
    fn partially_evaluate_instruction<C: Context>(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        instruction_index: usize,
        inputs: &[PartialEvaluationValue<C::Value>],
        deferred_instructions: &HashSet<InstructionId>,
        deferred_effect_ordering: &RefCell<EffectOrdering<ReferenceRoot>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        D: RegionDriver<C::Constant, C::Operation>,
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let instruction = &region.instructions()[instruction_index];
        let instruction_id = InstructionId::new(region.id(), instruction_index);
        let mut shared_reference_boundary = false;
        let effect_ordering = if let Some(analysis) = reference_analysis {
            let effects = region.instruction_effects(instruction_index)?;
            if effects.classes().is_ordered() {
                // Resolve input roots once for both ordering and alias detection. Allocation identity includes views,
                // even when the views select different elements of the allocation.
                let mut roots = HashSet::new();
                for &atom in instruction.inputs() {
                    if let Some(root) = analysis.root_of(ValueId::new(region.id(), atom)) {
                        shared_reference_boundary |= !roots.insert(root);
                    }
                }
                roots.extend(
                    instruction.outputs().iter().filter_map(|&atom| analysis.root_of(ValueId::new(region.id(), atom))),
                );
                roots.extend(analysis.transitive_access(instruction_id).into_iter().flat_map(|access| access.roots()));

                // Captured handles cannot prove independence from symbolic incoming references. Empty reference
                // sets conservatively give ordered work global ordering, rather than inventing independent identities.
                if roots.iter().any(|root| matches!(root, ReferenceRoot::Constant { .. })) {
                    roots.clear();
                }

                Some(effects.effect_ordering(roots))
            } else {
                Some(EffectOrdering::default())
            }
        } else {
            None
        };

        let instruction_context = effect_ordering.as_ref().map(|ordering| {
            let mut instruction_context = context.clone();
            instruction_context.defer_ordered_effects =
                Rc::new(Cell::new(deferred_effect_ordering.borrow().conflicts(ordering)));
            instruction_context
        });

        let context = instruction_context.as_ref().unwrap_or(context);
        let must_defer = deferred_instructions.contains(&instruction_id);
        let outputs = if must_defer || shared_reference_boundary {
            let programs = self.regions().map(RegionRef::to_program).collect();
            if must_defer {
                context.residualize(instruction.operation().clone(), programs, inputs)
            } else {
                context.fold_or_residualize(instruction.operation().clone(), programs, inputs)
            }
        } else {
            instruction.operation().partially_evaluate(context, self, inputs)
        }?;

        if let Some(ordering) = effect_ordering
            && context.defer_ordered_effects.get()
        {
            deferred_effect_ordering.borrow_mut().extend(&ordering);
        }

        Ok(outputs)
    }
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
    fn partially_evaluate_region(
        &self,
        context: &PartialEvaluationContext<C>,
        index: usize,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        let region = self.region(index)?;

        // Inlining during ordinary specialization shares the caller's residual builder and accumulated effect
        // ordering. It needs no separate allocation discovery or per-reference ordering analysis.
        if !self.repeated_residual {
            return context.inline_region(region, inputs, &HashSet::new(), None, None);
        }

        // Analyze a staged copy using only input knownness to discover allocations that must be fresh on each
        // residual invocation. Discard the staged programs as the replay below must use the caller's actual values.
        let knowledge = inputs.iter().map(PartialEvaluationValue::is_known).collect::<Vec<_>>();
        let (_, deferred_instructions) = region.partition_with_configuration(&knowledge, true, true, None)?;

        // Replay into the active context, explicitly deferring the discovered allocations. Source reference roots
        // let replay distinguish independent accesses while keeping accesses to the same state in order.
        context.inline_region(
            region,
            inputs,
            &deferred_instructions,
            None,
            Some(region.reference_analysis_with_configuration(None, true, &[])?.as_ref()),
        )
    }

    fn partially_evaluate_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        knowledge: &[PartialValue<C::Value>],
    ) -> Result<PartialEvaluation<C>, ProgramError> {
        // Unlike inlining a region, constructing a separate residual program needs fresh builder and ordering
        // state. Ordinary specialization still inherits the caller's permission to fold effectful work.
        if !self.repeated_residual {
            return region.partially_evaluate_in_context(context.parent(), knowledge, context.allow_effect_folding);
        }

        // Give the nested evaluation its own residual program while folding through the same known-side parent.
        // Stage placement keeps live reference operations residual when that parent is eager.
        let nested =
            PartialEvaluationContext::new(context.parent().clone()).with_reference_placement(ReferencePlacement::Stage);

        // Repeated residual calls use their own allocation placement and reference-ordering analysis. Keep the
        // fresh context's effect folding enabled so that this analysis determines which effects must be deferred.
        let known = knowledge.iter().map(PartialValue::is_known).collect::<Vec<_>>();
        let (_, deferred_instructions) = region.partition_with_configuration(&known, true, true, None)?;

        // Retain actual known values and create residual inputs for unknowns. Their indices refer to the original
        // region inputs so the returned evaluation can reconstruct its residual arguments in the correct order.
        let inputs = knowledge
            .iter()
            .enumerate()
            .map(|(index, value)| match value {
                PartialValue::Known(value) => PartialEvaluationValue::known_input(value.clone()),
                PartialValue::Unknown(r#type) => nested.unknown_input(r#type.clone(), index),
            })
            .collect();

        // Apply the allocation decisions to replay of the original region, using its reference analysis to preserve
        // dependencies between accesses. The discovery pass already recorded which allocations need deferring, so
        // this replay does not need to collect another list of residual source instructions.
        let outputs = nested.inline_region(
            region,
            inputs,
            &deferred_instructions,
            None,
            Some(region.reference_analysis_with_configuration(None, true, &[])?.as_ref()),
        )?;

        // Finalize the residual program and report how its inputs and outputs relate to the original computation.
        nested.into_evaluation(outputs)
    }

    fn partition_program(
        &self,
        context: &PartialEvaluationContext<C>,
        region: RegionRef<'_, C::Constant, C::Operation>,
        input_known: &[bool],
    ) -> Result<PartitionedProgram<C::Constant, C::Operation>, ProgramError> {
        // Both paths stage fresh known and residual programs without executing effects or changing the caller's
        // accumulated ordering state. Only the programs are needed here; allocation IDs are for replaying source
        // regions, whereas these returned programs already incorporate the allocation decisions.
        if self.repeated_residual {
            // Let repeated-call allocation and reference-ordering analysis decide which effects can remain known,
            // rather than inheriting a restriction intended for the caller's current residual program.
            region.partition_with_configuration(input_known, true, true, None).map(|(partition, _)| partition)
        } else {
            // Ordinary specialization preserves the caller's effect-folding policy and uses a single partition pass.
            region
                .partition_with_configuration(input_known, context.allow_effect_folding, false, None)
                .map(|(partition, _)| partition)
        }
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
    ///   - When *all* of the operation's inputs are [`Known`](PartialValue::Known), it **folds** the operation by
    ///     [`bind`](Context::bind)ing it in the known-side context, interpreting it immediately under an eager context,
    ///     and staging it into the outer program under a [`StagingContext`], so that the operation's outputs become
    ///     known values and the operation contributes nothing to the residual [`Program`].
    ///   - Otherwise, it **residualizes** the operation unchanged, meaning that it emits the operation into the
    ///     residual program over its inputs' residual program [`Atom`](crate::Atom)s, materializing each known input as
    ///     a residual input for a known variable or as an inlined residual program constant for a literal, so that the
    ///     operation runs at residual program execution time.
    ///   - An operation with an ordered effect (i.e., [`EffectClasses::is_ordered`] over the operation and its
    ///     executable computation regions) also follows the ordering rules on [`PartialEvaluationContext`] where once
    ///     an ordered operation has been staged, every later ordered operation is staged too, even when all of its
    ///     inputs are known, and reference operations under an eager known side follow the context's
    ///     [`ReferencePlacement`].
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

/// Placement of _known_ reference operations when the known-side [`Context`] of a [`PartialEvaluationContext`] is
/// eager. Under a [`StagingContext`] known side the two placements coincide, because folding stages an operation into
/// the outer program instead of executing it, and both follow the ordering rules on
/// [`PartialEvaluationContext`]: once an ordered effect is deferred, later ordered effects remain residual too.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ReferencePlacement {
    /// Known reference operations fold like every other ordered effect. Under an eager known side they execute at
    /// partial evaluation time unless an earlier ordered effect has been deferred. This is the placement of
    /// linearization, whose known side _is_ the evaluation of the primal program at the linearization point, so the
    /// primal accesses must run (and mutate the primal references) exactly as forward mode would, while the tangent
    /// accesses, which consume unknown tangent references, stage into the linear program.
    Execute,

    /// Under an eager known side, every reference operation (i.e., an operation with a reference-typed input or output,
    /// or a region-carrying operation whose closure touches references) stages into the residual program, and the live
    /// handles it touches are passed as known reference inputs (i.e., [`PartialEvaluation::known_reference_inputs`]),
    /// so partial evaluation never reads, mutates, or allocates live state and every access runs once per residual
    /// program run, observing the state of that run. This is the placement of specialization (i.e.,
    /// [`Program::partially_evaluate`] and the other program-replay entry points): the residual program is a reusable
    /// program that runs at some later time, possibly repeatedly and after the state has changed. An eager known side
    /// also has no way to prove that an allocation's complete lifecycle stays internal to the folded prefix, which is
    /// why allocations stage too. Under a staging known side this placement is inert, exactly like
    /// [`Execute`](Self::Execute): a known reference operation folds into the outer program and runs once per execution
    /// of that program, ahead of the residual program, unless an earlier ordered operation has residualized. The
    /// per-run guarantee therefore holds for the residual program relative to the outer program that consumes it,
    /// and not for the folded known accesses on their own.
    Stage,
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
/// # Preserving Effect Order
///
/// The split executes all known work before residual work. Once any ordered operation residualizes, every later
/// ordered operation residualizes too, irrespective of its input knowledge, reference root, or effect class. This
/// preserves the order in which reference accesses can fail or synchronize relative to later mutations, assertions,
/// and I/O. Pure operations keep folding whenever their operands permit.
///
///   | Work                                    | Default Placement                                     |
///   | --------------------------------------- | ----------------------------------------------------- |
///   | Pure operation with known operands      | Parent context                                        |
///   | Operation with an unknown operand       | Residual program                                      |
///   | Ordered effect with none yet deferred   | Parent context when its operands are known            |
///   | Ordered effect after one is deferred    | Residual program, including effects on distinct roots |
///   | Any effect in a deferred sibling        | Residual program, even with known operands            |
///   | Eager reference work with `Stage`       | Residual program, including allocation and views      |
///
/// [`deferred_sibling`](Self::deferred_sibling) retains every effect in residual execution, including allocations with
/// known initializers. Pure known computations still fold through the parent. Separately invoked programs require an
/// explicit source-level partition that validates their dependencies and allocation lifetimes; changing a context does
/// not implicitly establish independence between reference roots.
///
/// Control flow rules preserve the same contract. A known `condition` inlines only its selected branch under the
/// enclosing context's ordering constraints, and an unknown ordered `condition` residualizes whole. Ordinary `scan`
/// partitioning keeps a body whole when ordered effects would run on both sides. One-sided effects retain their
/// iteration order. Ordered `while` operations stay whole. Dormant derivative regions contribute neither effects nor
/// reference accesses until a transform invokes them. Speculative probes never execute effects through the live parent
/// or change its recorded ordering constraints; structural discovery uses fresh staging builders.
///
/// Mutable state is shared behind `Rc<RefCell<...>>` handles. Cloning this context therefore keeps every clone writing
/// to the same residual program, input descriptors, staged-feeder table, and ordering state, and lets rules re-enter
/// the context (e.g., to inline a selected condition branch through [`inline_program`](Self::inline_program)).
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct PartialEvaluationContext<C: Context> {
    /// Parent context in which known operations are evaluated or staged. An `Rc` shares the exact parent object across
    /// clones and deferred siblings instead of cloning `C`, whose `Clone` implementation need not preserve object
    /// identity. Imports compare these shared allocations to verify that a value comes from the same parent, even
    /// when the parent cannot resolve the value's identity. Independently constructed contexts have distinct parent
    /// allocations, while deferred siblings share the parent but build separate residual programs.
    parent: Rc<C>,

    /// [`ProgramBuilder`] accumulating the residual program's [`Atom`](crate::Atom)s and
    /// [`Instruction`](crate::Instruction)s. Shared across clones of this context (and held behind a [`RefCell`] for
    /// interior mutability) for the same reason that [`TracingContext`] shares its builder (i.e., values stamped with
    /// cloned contexts must keep accumulating into the _same_ residual program).
    builder: Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>>,

    /// Describes where each residual program input comes from, in input order, and is shared alongside the builder.
    inputs: Rc<RefCell<Vec<PartialEvaluationInput<C::Value>>>>,

    /// Map that is used for deduplication of residual _input_ feeders by the known value's _staged_ identity
    /// (i.e., the outer program [`Atom`](crate::Atom) a known value names when it [`resolve`](Context::resolve)s as a
    /// [`Staged`](ValueResolution::Staged) instance in a staging known-side context), mapping the staged atom to the
    /// residual input already created for it. This complements the per-value shared [`PartialValueMaterialization`]
    /// slots along the axis that value-identity deduplication cannot reach: two _distinct_ known values (with distinct
    /// slots) naming the same outer atom collapse to one residual input, even when rule-produced. It holds only under a
    /// _staging_ known-side context because an eager context resolves knowns as [`Constant`](ValueResolution::Constant)
    /// rather than [`Staged`](ValueResolution::Staged), and so nothing is ever recorded in that case, and inline
    /// constants are excluded because they carry no staged identity.
    staged_feeders: Rc<RefCell<HashMap<AtomId, AtomId>>>,

    /// Cache containing known values imported from sibling contexts keyed by the address of their source
    /// materialization slot. Each import needs a slot in this context because source atom identifiers belong to a
    /// different residual builder; repeated imports reuse that slot. Retaining the source slot prevents its address
    /// from being reused for another value while the cache entry exists. Clones share the cache alongside the builder.
    imported_known_values:
        Rc<RefCell<HashMap<usize, (Rc<Cell<PartialValueMaterialization>>, PartialEvaluationValue<C::Value>)>>>,

    /// Maps materialized reference constants to their identities in the parent context. Residual views of those
    /// constants must retain the same allocation identity, but constants have no entry in the input descriptors
    /// from which to recover it. A stored `None` records that the parent identity is unresolved, preventing fallback
    /// to a misleading identity in the residual builder. Clones share this map alongside the builder.
    constant_reference_identities: Rc<RefCell<HashMap<AtomId, Option<ReferenceIdentity>>>>,

    /// Active [`ProvenanceState`] that residual [`Instruction`](crate::Instruction)s snapshot.
    /// [`PartialEvaluationContext`]s are staging boundaries (i.e., they own the residual [`ProgramBuilder`] and emit
    /// into it directly), and so they own provenance state exactly like tracing contexts instead of delegating reads
    /// to their known-side parents, which are often terminal eager contexts that would erase source provenance from
    /// every residual program. The state is seeded from the parent context's current provenance at construction and
    /// shared across clones.
    provenance: Rc<ProvenanceState>,

    /// Specifies whether known reference operations execute or remain residual when the parent is eager.
    /// Specialization retains these operations so they observe state when the specialized program is called.
    reference_placement: ReferencePlacement,

    /// Specifies whether effectful operations with known inputs may run in the parent context. Deferred sibling
    /// contexts disable this so that every residual call performs its own effects, including allocations with known
    /// initial values. Pure known work can still fold; reference placement and previously deferred ordered effects
    /// can further restrict folding.
    allow_effect_folding: bool,

    /// Specifies whether ordered effects must remain residual to preserve execution order. Clones share this flag so
    /// that nested operations cannot move effects ahead of already deferred work. During replay with per-reference
    /// ordering, each instruction receives a separate flag initialized from its conflicts with earlier deferred work;
    /// the reference sets themselves remain local to that replay.
    defer_ordered_effects: Rc<Cell<bool>>,

    /// First binding error retained for finalization, even if the failed operation has no outputs or its poisoned
    /// outputs are discarded. Once set, later binds propagate poison without executing operations. Clones share the
    /// error so a failure in nested work cannot be lost when returning to the enclosing computation.
    error: Rc<RefCell<Option<ProgramError>>>,
}

impl<C: Context> PartialEvaluationContext<C> {
    /// Creates a fresh [`PartialEvaluationContext`] that folds known work through `parent` and accumulates residual
    /// work in a new residual [`ProgramBuilder`], with the [`Execute`](ReferencePlacement::Execute) placement of known
    /// reference operations. Refer to [`Self::with_reference_placement`] for the specialization placement.
    #[inline]
    pub fn new(parent: C) -> Self {
        Self::from_shared_parent(Rc::new(parent))
    }

    /// Creates fresh residual state around a shared parent, retaining its identity for known-value imports.
    fn from_shared_parent(parent: Rc<C>) -> Self {
        let provenance = Rc::new(ProvenanceState::seeded(parent.provenance()));
        Self {
            parent,
            builder: Rc::new(RefCell::new(ProgramBuilder::new())),
            inputs: Rc::new(RefCell::new(Vec::new())),
            staged_feeders: Rc::new(RefCell::new(HashMap::new())),
            imported_known_values: Rc::new(RefCell::new(HashMap::new())),
            constant_reference_identities: Rc::new(RefCell::new(HashMap::new())),
            provenance,
            reference_placement: ReferencePlacement::Execute,
            allow_effect_folding: true,
            defer_ordered_effects: Rc::new(Cell::new(false)),
            error: Rc::new(RefCell::new(None)),
        }
    }

    /// Returns a copy of this context with the provided [`ReferencePlacement`] for known reference operations under an
    /// eager parent. For example, use [`Stage`](ReferencePlacement::Stage) for specialization so reference operations
    /// run when the residual program is called rather than while it is constructed. Under a staging parent, both
    /// placements record known work in the parent program, subject to effect-ordering constraints.
    ///
    /// Note that this function changes the configuration of the returned context without creating a new residual
    /// program or changing previously emitted work. Existing clones retain their own placement setting.
    #[inline]
    pub fn with_reference_placement(mut self, reference_placement: ReferencePlacement) -> Self {
        self.reference_placement = reference_placement;
        self
    }

    /// Returns a copy of this context with the provided permission to fold effectful operations with known inputs into
    /// the parent context. When `false`, effectful operations remain in the residual program while pure known work can
    /// still fold. When `true`, reference placement and effect-ordering constraints can still prevent folding.
    ///
    /// This function changes only the returned context's configuration, without creating a new residual program or
    /// changing previously emitted work. Existing clones retain their own effect-folding setting.
    #[inline]
    pub fn with_allow_effect_folding(mut self, allow_effect_folding: bool) -> Self {
        self.allow_effect_folding = allow_effect_folding;
        self
    }

    /// Creates a fresh sibling [`PartialEvaluationContext`] that folds pure known work into this context's parent
    /// and retains every effectful operation in its residual program, even when all operands are known. Each residual
    /// invocation therefore executes its own effects, including fresh reference allocations. The shared parent identity
    /// permits explicit known-value transfers between these contexts without relying on constant-value resolution.
    /// The sibling preserves this context's reference placement, including staging pure reference views under an eager
    /// parent.
    #[inline]
    pub fn deferred_sibling(&self) -> Self {
        Self::from_shared_parent(self.parent.clone())
            .with_reference_placement(self.reference_placement)
            .with_allow_effect_folding(false)
    }

    /// Returns the known-side parent [`Context`] of this [`PartialEvaluationContext`] which is used
    /// to fold known subcomputations.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ReferencePlacement`] of known reference operations under an eager known-side parent for this
    /// [`PartialEvaluationContext`]. That placement specifies whether known reference operations execute or remain
    /// residual when the parent is eager. Specialization retains these operations so they observe state when the
    /// specialized program is called.
    #[inline]
    pub fn reference_placement(&self) -> ReferencePlacement {
        self.reference_placement
    }

    /// Imports a known value from another [`PartialEvaluationContext`] sharing this context's parent. The first import
    /// gets a fresh residual materialization slot. Repeated imports reuse that slot, preserving input/constant intent
    /// without copying source atom identifiers. Values already belonging to this context are returned unchanged,
    /// including unknown values. Foreign unknown values and different parent identities are rejected. A deferred
    /// binding failure instead produces a poisoned value in this context and is retained until finalization, preserving
    /// infallible operator syntax. Importing a reference preserves its live handle; it does not snapshot the referenced
    /// contents.
    pub fn import_known(&self, value: &PartialTracer<C>) -> Result<PartialTracer<C>, ProgramError> {
        let error = self
            .error
            .borrow()
            .clone()
            .or_else(|| value.context.error.borrow().clone())
            .or_else(|| value.value().err());
        if let Some(error) = error {
            self.error.borrow_mut().get_or_insert_with(|| error.clone());
            return Ok(PartialTracer::poisoned(self.clone(), error, value.r#type().into_owned()));
        }
        let partial = value.value()?;
        if Rc::ptr_eq(&self.builder, &value.context.builder) {
            return Ok(value.clone());
        }
        if !Rc::ptr_eq(&self.parent, &value.context.parent) {
            return Err(ProgramError::MalformedProgram(
                "cannot import a value from a partial-evaluation context with a different parent context".to_string(),
            ));
        }
        let known = partial.as_known().ok_or_else(|| {
            ProgramError::MalformedProgram(
                "cannot import an unknown value from another partial-evaluation context".to_string(),
            )
        })?;
        let source_key = Rc::as_ptr(&partial.materialization) as usize;
        if let Some((_, imported)) = self.imported_known_values.borrow().get(&source_key) {
            return Ok(PartialTracer::new(self.clone(), imported.clone()));
        }
        let imported = match partial.materialization() {
            PartialValueMaterialization::Constant { .. } => PartialEvaluationValue::known_constant(known.clone()),
            PartialValueMaterialization::Input { .. } => PartialEvaluationValue::known_input(known.clone()),
            PartialValueMaterialization::Undecided => PartialEvaluationValue::known(known.clone()),
            PartialValueMaterialization::Variable { .. } => unreachable!("known values never name residual variables"),
        };
        self.imported_known_values
            .borrow_mut()
            .insert(source_key, (partial.materialization.clone(), imported.clone()));
        Ok(PartialTracer::new(self.clone(), imported))
    }

    /// Creates a fresh unknown value backed by a new residual-program input of the provided type, recorded as an
    /// [`Unknown`](PartialEvaluationInput::Unknown) feeder carrying `index`. This is how drivers seed the unknown
    /// inputs of an evaluation. The program-replay driver behind [`Program::partially_evaluate_in_context`] seeds one
    /// per unknown program input (with the original program input index as the ordinal), and closure drivers seed one
    /// per traced unknown (e.g., one tangent input per primal in linearization), in order.
    ///
    /// # Parameters
    ///
    ///   - `r#type`: [`Type`] of the unknown value.
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
    /// Known inputs permit folding only when doing so preserves effect order. Once an ordered operation is deferred,
    /// later ordered operations remain residual so they cannot run ahead of it, even when their inputs are known.
    ///
    /// An eager parent executes folded operations immediately. [`ReferencePlacement::Stage`] instead residualizes
    /// every operation touching references, including allocation and aliasing, so eager specialization cannot observe
    /// or mutate live reference state. [`ReferencePlacement::Execute`] permits known reference operations to execute.
    /// A staging parent appends folded operations to its outer program, which executes before the residual program;
    /// reference placement adds no restriction there. A deferred sibling still retains all effects regardless of
    /// whether its parent is eager or staged.
    ///
    /// Fixed-point probes must never fold effectful programs through the live parent context as each probe would
    /// execute or stage the effects again. Control flow rules instead retain such programs whole, without changing
    /// the active context's ordering state during speculative analysis.
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
    pub fn fold_or_residualize<P: Into<C::Operation>>(
        &self,
        operation: P,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        if let Some(error) = self.error.borrow().clone() {
            return Err(error);
        }

        let operation = operation.into();

        // Combine the operation's effect classes with those of its executable computation regions. Dormant
        // derivative rules and other non-computation regions do not contribute effects when this operation runs.
        let effects = regions
            .iter()
            .enumerate()
            .filter(|(index, _)| operation.region_role(*index) == Some(RegionRole::Computation))
            .fold(operation.effects().summary(), |effects, (_, region)| effects.union(region.effects()))
            .classes();

        // Check reference placement only if the cheaper conditions have not already required residual execution.
        // Under eager specialization, even pure reference views must remain residual so they do not observe live
        // state before the specialized program runs. Dormant rule regions do not execute with this operation.
        let must_defer_references = || {
            self.reference_placement == ReferencePlacement::Stage
                && self.parent.is_eager()
                && (inputs.iter().any(|input| input.r#type().is_reference())
                    || operation.effects().has_reference_declarations()
                    || regions.iter().enumerate().any(|(index, region)| {
                        operation.region_role(index) == Some(RegionRole::Computation)
                    // Checked reference-effect endpoints are reference-typed, so inspecting atoms covers
                    // allocations, accesses, and aliases without rescanning their declarations.
                    && region.entry_region_ref().computation_regions().any(|region| {
                    region.atoms().iter().any(|atom| atom.r#type().is_reference())
                })
                    }))
        };

        if !inputs.iter().all(PartialEvaluationValue::is_known)
            || (!self.allow_effect_folding && !effects.is_empty())
            || (effects.is_ordered() && self.defer_ordered_effects.get())
            || must_defer_references()
        {
            return self.residualize_with_effects(operation, regions, inputs, effects);
        }

        let known = inputs.iter().map(|value| value.as_known().cloned().unwrap()).collect::<Vec<_>>();
        let outputs = self
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
            .collect::<Vec<_>>();
        Ok(outputs)
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
    /// itself be a residual-program constant. A known reference-typed input value materializes as a residual input
    /// (i.e., a residual reference) holding the live handle itself, while a reference-typed program constant (e.g., a
    /// captured reference) follows its [`Constant`](PartialValueMaterialization::Constant) materialization and embeds
    /// inline like every other constant, so it is never reported by [`PartialEvaluation::known_reference_inputs`].
    ///
    /// Note that emitting an ordered operation records that later ordered effects must remain residual too. This also
    /// applies to rules that emit residual work directly (e.g., the residual half of a partitioned boundary), so those
    /// rules preserve the same execution order as the default partial evaluation rule.
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
        if let Some(error) = self.error.borrow().clone() {
            return Err(error);
        }

        let operation = operation.into();

        // Combine the operation's effect classes with those of its executable computation regions. Dormant
        // derivative rules and other non-computation regions do not contribute effects when this operation runs.
        let effects = regions
            .iter()
            .enumerate()
            .filter(|(index, _)| operation.region_role(*index) == Some(RegionRole::Computation))
            .fold(operation.effects().summary(), |effects, (_, region)| effects.union(region.effects()))
            .classes();

        self.residualize_with_effects(operation, regions, inputs, effects)
    }

    /// Residualizes `operation` using its already computed [`EffectClasses`]. This is the emission behind
    /// [`Self::residualize`] and the residual branch of [`Self::fold_or_residualize`].
    fn residualize_with_effects(
        &self,
        operation: C::Operation,
        regions: Vec<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>,
        inputs: &[PartialEvaluationValue<C::Value>],
        effects: EffectClasses,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError> {
        // Record the ordering restriction before emission, which may fail partway through. Even if binding retains that
        // failure as poisoned outputs, later ordered operations must not execute ahead of the failed work. Retaining
        // the restriction without an emitted instruction can only defer additional ordered operations.
        if effects.is_ordered() {
            self.defer_ordered_effects.set(true);
        }

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

                let atom = if constant {
                    let constant = self.parent.resolve(known).into_constant().ok_or_else(|| {
                        ProgramError::MalformedProgram(
                            "residual materialization required a constant payload for a known value that is \
                             not resolvable to a constant in the active known-side context"
                                .to_string(),
                        )
                    })?;
                    let reference_identity =
                        known.r#type().is_reference().then(|| self.parent.reference_identity(known)).transpose()?;
                    let atom = self.builder.borrow_mut().add_constant(constant);
                    if let Some(key) = reference_identity {
                        self.constant_reference_identities.borrow_mut().insert(atom, key);
                    }
                    atom
                } else {
                    // Only input feeders deduplicate across distinct values naming the same known-side staged atom.
                    let staged_atom = match self.parent.resolve(known) {
                        ValueResolution::Staged(atom) => Some(atom),
                        _ => None,
                    };
                    let existing = staged_atom.and_then(|staged| self.staged_feeders.borrow().get(&staged).copied());
                    if let Some(atom) = existing {
                        atom
                    } else {
                        let atom = self.builder.borrow_mut().add_input(known.r#type().into_owned());
                        self.inputs.borrow_mut().push(PartialEvaluationInput::Known(known.clone()));
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
        self.inline_region(program.entry_region_ref(), inputs, &HashSet::new(), None, None)
    }

    /// Replays a borrowed [`Region`](crate::Region) without materializing it as a standalone source [`Program`].
    /// Input binding and output ordering follow [`inline_program`](Self::inline_program). Each call tracks its own
    /// source instruction indices, including recursive calls into shared regions.
    ///
    /// # Parameters
    ///
    ///   - `region`: Borrowed region to replay through this context.
    ///   - `inputs`: Partially evaluated values bound to the region's input atoms in input order.
    ///   - `deferred_instructions`: Region-qualified instruction IDs that must emit residual work regardless of input
    ///     knownness. Replay checks IDs in the source region, before rebuilding instructions in the residual program.
    ///   - `residual_instructions`: Optional vector to which replay appends the IDs of source instructions that emit
    ///     residual work or produce unknown outputs, in source order. This includes instructions with no outputs.
    ///     Recording these observations does not affect placement.
    ///   - `reference_analysis`: Canonical source reference analysis for replay with repeated residual calls. When
    ///     absent, replay preserves ordering across all ordered effects rather than separating independent references.
    fn inline_region(
        &self,
        region: RegionRef<'_, C::Constant, C::Operation>,
        inputs: Vec<PartialEvaluationValue<C::Value>>,
        deferred_instructions: &HashSet<InstructionId>,
        residual_instructions: Option<&RefCell<Vec<InstructionId>>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> Result<Vec<PartialEvaluationValue<C::Value>>, ProgramError>
    where
        C::Operation:
            PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>,
    {
        let deferred_effect_ordering = RefCell::new(EffectOrdering::default());
        let instruction_index = Cell::new(0);
        let region_mappings = RegionReplayMappings::new();
        region.interpret_with(
            inputs,
            |_, constant| Ok(PartialEvaluationValue::known_constant(self.parent.lift(constant.clone())?)),
            |instruction, inputs| {
                // Evaluate inside the source instruction's recorded origin so that residualized instructions record
                // where they came from.
                let regions = ReplayRegionDriver::new(region, instruction.regions(), &region_mappings)?;
                let index = instruction_index.get();
                instruction_index.set(index + 1);
                let driver = RecursivePartialEvaluationDriver {
                    driver: &regions,
                    repeated_residual: reference_analysis.is_some(),
                };

                // Count emitted instructions only when recording observations. Output knownness alone cannot detect
                // residual effects with no outputs, or an operation that emits effects and returns known outputs.
                let residual_instruction_count =
                    residual_instructions.map(|_| self.builder.borrow().instructions().len());

                let outputs = self.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                    driver.partially_evaluate_instruction(
                        self,
                        region,
                        index,
                        inputs,
                        deferred_instructions,
                        &deferred_effect_ordering,
                        reference_analysis,
                    )
                })?;

                // Record the source instruction if it produces unknown outputs or emits residual work. Checking
                // builder growth also catches effects with no outputs and rules that emit effects but return known
                // values. The initial count exists whenever observations are requested, so the `unwrap` is safe.
                if let Some(residual_instructions) = residual_instructions
                    && (outputs.iter().any(PartialEvaluationValue::is_unknown)
                        || self.builder.borrow().instructions().len() > residual_instruction_count.unwrap())
                {
                    residual_instructions.borrow_mut().push(InstructionId::new(region.id(), index));
                }

                Ok(outputs)
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
    /// Any deferred binding failure is returned first, even if the failed operation's outputs were discarded.
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
        if let Some(error) = self.error.borrow().clone() {
            return Err(error);
        }

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
            imported_known_values: self.imported_known_values.clone(),
            constant_reference_identities: self.constant_reference_identities.clone(),
            provenance: self.provenance.clone(),
            reference_placement: self.reference_placement,
            allow_effect_folding: self.allow_effect_folding,
            defer_ordered_effects: self.defer_ordered_effects.clone(),
            error: self.error.clone(),
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
        let input_values = match self.error.borrow().clone() {
            Some(error) => Err(error),
            None => inputs
                .iter()
                .map(|input| {
                    if !Rc::ptr_eq(&self.builder, &input.context.builder) {
                        return Err(ProgramError::MalformedProgram(
                            "cannot bind a value belonging to another partial-evaluation context; \
                         import known values explicitly"
                                .to_string(),
                        ));
                    }
                    input.value()
                })
                .collect::<Result<Vec<_>, _>>(),
        };
        let error = match input_values {
            Ok(input_values) => {
                let input_values = input_values.into_iter().cloned().collect::<Vec<_>>();
                let driver = RecursivePartialEvaluationDriver { driver: &driver, repeated_residual: false };
                let outputs = operation.partially_evaluate(self, &driver, input_values.as_slice());
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

        // Retain the first error even when this operation has no outputs or its outputs are discarded. Poison lets
        // infallible operator syntax finish the closure without executing subsequent operations. If output inference
        // also fails, the output arity is unknowable and the error surfaces immediately instead.
        self.error.borrow_mut().get_or_insert_with(|| error.clone());
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

    fn reference_identity(&self, value: &PartialTracer<C>) -> Result<Option<ReferenceIdentity>, ProgramError> {
        // The default implementation of this function recovers only runtime identities from a value or its resolved
        // constant. Partial evaluation also needs the parent's symbolic identities for known references and the
        // residual builder's allocation identities for unknown references, which resolve as opaque values. This
        // override follows those identities through aliases and checks that the tracer belongs to the correct
        // underlying residual builder.
        if !Rc::ptr_eq(&self.builder, &value.context().builder) {
            return Ok(None);
        }

        // Resolve through the parent for known values and through the residual builder for unknown values. Known input
        // feeders and materialized constants retain their parent identities. Inherited capture constants without
        // recorded parent identities remain unresolved; querying must not lift constants or invent identities.
        let value = value.value()?;
        if !value.r#type().is_reference() {
            return Ok(None);
        }
        match value.value() {
            PartialValue::Known(known) => self.parent.reference_identity(known),
            PartialValue::Unknown(_) => {
                let PartialValueMaterialization::Variable { residual_atom } = value.materialization() else {
                    unreachable!("unknown values always carry a residual variable")
                };
                let builder = self.builder.borrow();
                let (root, constant) = builder.resolve_reference(residual_atom)?;
                if let Some(key) = self.constant_reference_identities.borrow().get(&root) {
                    return Ok(*key);
                }
                if let Some(constant) = constant {
                    return Ok(if constant.capture_index().is_some() {
                        None
                    } else {
                        constant.reference_id().map(ReferenceIdentity::Runtime)
                    });
                }
                if let Some(index) = builder.input_ids().iter().position(|input| *input == root)
                    && let PartialEvaluationInput::Known(known) = &self.inputs.borrow()[index]
                {
                    return self.parent.reference_identity(known);
                }
                builder.reference_identity(residual_atom, Rc::as_ptr(&self.builder) as usize)
            }
        }
    }

    #[inline]
    fn invoke_with_provenance_origin<R, F: FnOnce() -> R>(&self, origin: Provenance, function: F) -> R {
        self.provenance.invoke_with_origin(origin, function)
    }

    #[inline]
    fn invoke_with_provenance_scope<R, F: FnOnce() -> R>(&self, scope: ProvenanceScope, function: F) -> R {
        self.provenance.invoke_with_scope(scope, function)
    }
}

/// State carried by a [`PartialTracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`PartialEvaluationValue`], or a *poison* recording an error that a failed [`bind`](Context::bind) deferred
/// mirroring [`Tracer`](crate::Tracer)'s poison state. Because the closures driven through a
/// [`PartialEvaluationContext`] use infallible operator sugar with no deferral point of their own,
/// [`bind`](Context::bind) turns its errors into poisoned outputs (and propagates poison from inputs to outputs), so
/// the original error surfaces as a plain `Err` at the evaluation boundary instead of panicking mid-closure. The
/// context also retains the first error independently of these outputs and skips subsequent operations. Unlike
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

        /// [`Type`] of the output the failed bind would have produced.
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
    /// materializing it. Refer to [`Program::partially_evaluate_in_context`] for the input and output conventions.
    ///
    /// Reference operations use [`ReferencePlacement::Stage`]. With an eager context they remain in the residual
    /// program, so specialization does not read or mutate live reference state. With a staging context, folding records
    /// work in the parent program instead of executing it, so reference placement adds no restriction. Disabling effect
    /// folding retains all effectful operations in the residual program in either case; pure known work can still fold.
    ///
    /// # Parameters
    ///
    ///   - `context`: Parent context through which known work is evaluated or staged.
    ///   - `inputs`: Known values or unknown input types, in region input order.
    ///   - `allow_effect_folding`: Specifies whether effects may fold into the parent when input knownness, reference
    ///     placement, and execution order constraints permit it. When `false`, every effectful operation remains
    ///     residual.
    pub fn partially_evaluate_in_context<C: Context<Type = V::Type, Constant = V, Operation = O>>(
        self,
        context: &C,
        inputs: &[PartialValue<C::Value>],
        allow_effect_folding: bool,
    ) -> Result<PartialEvaluation<C>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<C> + PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        check_count!("input", inputs, self.input_ids().len(), ProgramError);
        let context = PartialEvaluationContext::new(context.clone())
            .with_reference_placement(ReferencePlacement::Stage)
            .with_allow_effect_folding(allow_effect_folding);
        let mut seed = Vec::with_capacity(inputs.len());
        for (index, knowledge) in inputs.iter().enumerate() {
            match knowledge {
                PartialValue::Known(value) => seed.push(PartialEvaluationValue::known_input(value.clone())),
                PartialValue::Unknown(r#type) => seed.push(context.unknown_input(r#type.clone(), index)),
            }
        }
        let outputs = context.inline_region(self, seed, &HashSet::new(), None, None)?;
        context.into_evaluation(outputs)
    }

    /// Partitions this borrowed [`Region`](crate::Region) based on per-input known-ness without first detaching its
    /// source computation. Refer to the documentation of [`Program::partition`] for more information.
    #[inline]
    pub fn partition(self, input_known: &[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.partition_with_configuration(input_known, true, false, None).map(|(partition, _)| partition)
    }

    /// Partitions this region through fresh staging contexts, returning the known and residual programs together
    /// with source instructions that recursive replay must explicitly defer. No reference effects execute during
    /// construction. Ordinary partitioning builds once and preserves ordering across all ordered effects.
    ///
    /// For repeated residual calls, allocation discovery can require additional passes. An allocation used only by
    /// residual work must be created afresh on each call, even when its initializer is known. The first pass fixes the
    /// required known outputs and their source dependencies; later passes move residual allocations and their uses
    /// without weakening those requirements. Iteration stops when no additional allocation must be deferred.
    ///
    /// Repeated call partitioning requires the caller to validate independence of entering reference roots. Symbolic
    /// input positions alone cannot establish runtime independence. Independent known effects may execute before an
    /// earlier source access fails in a later residual call; this contract does not preserve ordinary specialization's
    /// interleaved failure order. Reference feeders and captured references retain conservative ordering constraints.
    ///
    /// # Parameters
    ///
    ///   - `input_known`: Whether each source input is available to the known invocation, in input order.
    ///   - `allow_effect_folding`: Whether known effectful work may be recorded in the known program, subject to
    ///     ordering constraints. Pure known work can fold even when this is false.
    ///   - `repeated_residual`: Whether to analyze allocations for a known invocation followed by repeated residual
    ///     calls. When false, returns after one pass with no explicitly deferred instruction IDs.
    ///   - `required_known_outputs`: Output positions that repeated-call partitioning must keep known. When absent,
    ///     preserves the outputs known after the first pass. An empty slice imposes no output requirement. Ignored
    ///     when repeated-call partitioning is disabled.
    pub(crate) fn partition_with_configuration(
        self,
        input_known: &[bool],
        allow_effect_folding: bool,
        repeated_residual: bool,
        required_known_outputs: Option<&[usize]>,
    ) -> Result<(PartitionedProgram<V, O>, HashSet<InstructionId>), ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        let input_types = self.input_types();
        check_count!("input", input_known, input_types.len(), ProgramError);
        let reference_analysis =
            repeated_residual.then(|| self.reference_analysis_with_configuration(None, true, &[])).transpose()?;
        let residual_instructions = RefCell::new(Vec::new());
        let mut deferred_instructions = HashSet::new();
        let mut required_known_outputs = required_known_outputs.map(<[usize]>::to_vec);
        let mut required_roots = None;

        // Repeated residual calls need fresh local state. Each pass below can discover allocations that must move into
        // the residual program. Moving those allocations can make more work residual and reveal further allocations to
        // defer. Therefore, we rebuild until no new allocations are found. Each retry adds source instructions to a
        // finite set, and so this process is guaranteed to terminate. Ordinary partitioning needs no allocation
        // discovery and returns after the first pass.
        loop {
            // Rebuild both programs from the source after each new allocation-deferral decision. Observations and
            // builders belong to this pass; required outputs and source allocation identities survive retries.
            residual_instructions.borrow_mut().clear();
            let context = TracingContext::<V, O>::new();
            let evaluation_context = PartialEvaluationContext::new(context.clone())
                .with_reference_placement(ReferencePlacement::Stage)
                .with_allow_effect_folding(allow_effect_folding);
            let seed = input_types
                .iter()
                .zip(input_known)
                .enumerate()
                .map(|(index, (input_type, &known))| match known {
                    true => PartialEvaluationValue::known_input(context.input(input_type.clone())),
                    false => evaluation_context.unknown_input(input_type.clone(), index),
                })
                .collect();
            let outputs = evaluation_context.inline_region(
                self,
                seed,
                &deferred_instructions,
                repeated_residual.then_some(&residual_instructions),
                reference_analysis.as_deref(),
            )?;
            let evaluation = evaluation_context.into_evaluation(outputs)?;

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

            let mut partition = PartitionedProgram::from_parts(
                known_program,
                evaluation.program,
                known_input_indices,
                residual_inputs,
                outputs,
            );

            // Ordinary partitioning is complete after this first pass. Only repeated residual calls need reference
            // analysis to refine ordering and discover allocations that must be deferred before rebuilding.
            let Some(analysis) = reference_analysis.as_deref() else {
                return Ok((partition, deferred_instructions));
            };

            // Resolve per-reference ordering only when replay used the caller's reference-independence contract.
            // Replay already kept aliased applications whole before nested input bindings could be lost. References
            // passed from known work to residual work, and executable reference constants, still require the
            // conservative global constraints established by `from_parts`.
            let programs = [&partition.known_program, &partition.residual_program];
            if partition.known_reference_inputs().next().is_none()
                && !programs.into_iter().any(|program| {
                    program.entry_region_ref().computation_regions().any(|region| {
                        region.atoms().iter().any(|atom| atom.as_constant().is_some() && atom.r#type().is_reference())
                    })
                })
            {
                for (index, program) in programs.into_iter().enumerate() {
                    let analysis = program.entry_region_ref().reference_analysis_with_configuration(None, true, &[])?;
                    let roots = analysis.roots().filter_map(|root| match root {
                        ReferenceRoot::RegionInput { region, input_index } if region == analysis.region() => {
                            if index == 0 {
                                Some(PartitionReferenceRoot::Input(partition.known_input_indices[input_index]))
                            } else if let PartialEvaluationInput::Unknown(original) =
                                partition.residual_inputs[input_index]
                            {
                                Some(PartitionReferenceRoot::Input(original))
                            } else {
                                // Known reference feeders were excluded before entering this analysis.
                                None
                            }
                        }
                        ReferenceRoot::Allocation { .. } => Some(if index == 0 {
                            PartitionReferenceRoot::KnownAllocation(root)
                        } else {
                            PartitionReferenceRoot::ResidualAllocation(root)
                        }),
                        _ => {
                            // Nested input accesses are already included transitively in their caller's roots.
                            // Nested-local allocations stay visible above even though the transitive access summary
                            // omits them.
                            None
                        }
                    });
                    partition.effect_ordering[index] = program.effects().effect_ordering(roots);
                }
            }

            // Preserve the caller's required known outputs, or default to those known after the first pass. Keep this
            // set unchanged across retries so that deferring an allocation cannot silently weaken the requirement.
            let required_outputs = required_known_outputs.get_or_insert_with(|| {
                partition
                    .outputs()
                    .iter()
                    .enumerate()
                    .filter_map(|(index, output)| output.is_known().then_some(index))
                    .collect()
            });

            // On the first pass, validate the required outputs and trace their value dependencies backward through
            // the source instructions. Retain the reference roots on those dependencies across retries: residual
            // work must not mutate or consume local state that also contributes to the required known results.
            if required_roots.is_none() {
                let mut required_atoms = HashSet::new();
                for &index in required_outputs.iter() {
                    required_atoms.insert(*self.output_ids().get(index).ok_or_else(|| {
                        ProgramError::MalformedProgram(format!("required known output index {index} is out of bounds"))
                    })?);
                    if !partition.outputs()[index].is_known() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "required output {index} depends on deferred work in a repeated residual computation",
                        )));
                    }
                }
                for instruction in self.instructions().iter().rev() {
                    if instruction.outputs().iter().any(|output| required_atoms.contains(output)) {
                        required_atoms.extend(instruction.inputs().iter().copied());
                    }
                }
                let roots = required_atoms
                    .iter()
                    .filter_map(|&atom| analysis.root_of(ValueId::new(self.id(), atom)))
                    .collect::<HashSet<_>>();
                required_roots = Some(roots);
            }

            // Inspect reference accesses from source instructions that produced residual work. Defer each local
            // allocation they use unless it contributes to required known results, so that each residual call creates
            // its own state. For allocations needed by those results, permit reads but reject other access modes.
            let previous_count = deferred_instructions.len();
            for &instruction in residual_instructions.borrow().iter() {
                if let Some(access) = analysis.transitive_access(instruction) {
                    for root in access.roots() {
                        if let ReferenceRoot::Allocation { instruction, .. } = root
                            && instruction.region() == self.id()
                        {
                            if required_roots.as_ref().unwrap().contains(&root) {
                                if access.access_modes_for(root).any(|mode| mode != ReferenceAccessMode::Read) {
                                    return Err(ProgramError::MalformedProgram(
                                        "local reference allocation contributes to both \
                                         required known outputs and deferred state"
                                            .to_string(),
                                    ));
                                }
                            } else {
                                deferred_instructions.insert(instruction);
                            }
                        }
                    }
                }
            }

            // If no additional allocations need deferring, the partition is stable. Check that earlier deferrals
            // have not made a required output residual before returning. Otherwise, rebuild with the enlarged set
            // of deferred instructions so the allocations and dependent work move into the residual program.
            if deferred_instructions.len() == previous_count {
                for &index in required_outputs.iter() {
                    if !partition.outputs()[index].is_known() {
                        return Err(ProgramError::MalformedProgram(format!(
                            "required output {index} depends on deferred work in a repeated residual computation",
                        )));
                    }
                }
                return Ok((partition, deferred_instructions));
            }
        }
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
        self.entry_region_ref().partially_evaluate_in_context(context, inputs, true)
    }

    /// Partitions this [`Program`] based on the provided per-input known-ness into a known-side program and a
    /// residual program joined by residual edges, packaged as a [`PartitionedProgram`]. This function invokes
    /// [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) with a **fresh** [`TracingContext`]
    /// whose inputs stand in for the known program inputs and so, instead of folding the known work into a
    /// caller-supplied context, the fresh trace reifies it as the known-side program. The same per-[`Operation`]
    /// rules drive both entry points, and they differ only in what happens to the known side.
    ///
    /// The known program executes before the residual program under the global effect ordering contract of
    /// [`PartialEvaluationContext`]. Known operands do not permit moving later effects ahead of a deferred effect,
    /// even when their reference roots differ. This function constructs programs without executing effects. Therefore,
    /// callers must preserve the resulting invocation order and the lifetime of any reference-valued residuals.
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
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        ReferenceIndexOperation,
    };
    use crate::captures::CaptureReference;
    use crate::contexts::{Context, StagingContext};
    use crate::operations::{
        AddOperation, ConditionOperation, LinearCallOperation, MulOperation, NegOperation, PrintOperation,
        ReferenceAddUpdateOperation, ReferenceNewOperation, ReferenceReadOperation, ReferenceSwapOperation,
        ReferenceWriteOperation, SubOperation, Zero,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, Concretizable, EffectClasses, Effects, ProgramBuilder, ProgramError, ReferenceType};
    use crate::tests::{
        TestArrayContext, TestArrayIrContext, TestArrayIrOperation, TestArrayOperation, TestArrayTracingContext,
        TestOrderedStateOperation,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type TestCapture = CaptureReference<ArrayIrType>;

    /// Builds a nested residual read followed by a write to an independently supplied reference.
    fn reference_ordering_program() -> Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>> {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let source = branch.add_input(reference_type.clone());
        branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap();
        let branch = branch.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let destination = builder.add_input(reference_type.clone());
        let source = builder.add_input(reference_type);
        let predicate = builder.add_constant(TestValue::Array(Array::scalar(true).unwrap()));
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![predicate, source], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, zero], None)
            .unwrap();
        builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap()
    }

    /// Replays a region with optional instruction observation and returns both emitted programs for comparison.
    fn replay_reference_ordering_program(
        region: RegionRef<'_, TestValue, TestOperation>,
        observations: Option<&RefCell<Vec<InstructionId>>>,
        reference_analysis: Option<&ReferenceAnalysis>,
    ) -> (
        Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>,
        Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>,
    ) {
        let parent = TracingContext::<TestValue, TestOperation>::new();
        let context = PartialEvaluationContext::new(parent.clone()).with_reference_placement(ReferencePlacement::Stage);
        let inputs = vec![
            PartialEvaluationValue::known_input(parent.input(region.input_types()[0].clone())),
            context.unknown_input(region.input_types()[1].clone(), 1),
        ];
        let outputs = context.inline_region(region, inputs, &HashSet::new(), observations, reference_analysis).unwrap();
        assert!(outputs.is_empty());
        let residual = context.into_evaluation(outputs).unwrap().program;
        let known = parent
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        (known, residual)
    }

    #[test]
    fn test_partial_evaluation_known_reference_inputs() {
        // Forward a mixture of known arrays, known references, and unknown references through a valid residual
        // boundary. Only the known reference positions are reported.
        let reference = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32).unwrap()));
        let inputs = vec![
            PartialEvaluationInput::Unknown(0),
            PartialEvaluationInput::Known(ArrayIrValue::Array(Array::scalar(2.0_f32).unwrap())),
            PartialEvaluationInput::Known(reference.clone()),
            PartialEvaluationInput::Unknown(1),
            PartialEvaluationInput::Known(reference.clone()),
        ];
        let mut builder = ProgramBuilder::<TestValue, TestArrayIrOperation>::new();
        let outputs = inputs
            .iter()
            .map(|input| {
                builder.add_input(match input {
                    PartialEvaluationInput::Known(value) => value.r#type().into_owned(),
                    PartialEvaluationInput::Unknown(_) => reference.r#type().into_owned(),
                })
            })
            .collect::<Vec<_>>();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(outputs, vec![Placeholder; 5], vec![Placeholder; 5])
            .unwrap();
        let evaluation = PartialEvaluation::<TestArrayIrContext> {
            program,
            inputs,
            outputs: (0..5).map(PartialEvaluationOutput::Unknown).collect(),
        };
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2, 4]);
    }

    #[test]
    fn test_partial_evaluation_interpret() {
        // Build a residual program `g(x, r) = x * r + 3`, with `x` standing for the original program's surviving
        // unknown input and `r` for a known residual feeder carrying the folded value `2`, and pair it with an
        // original output report whose first output folded to `5`.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let r = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(3.0).unwrap());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, r], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let evaluation = PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: vec![
                PartialEvaluationInput::Unknown(1),
                PartialEvaluationInput::Known(Array::scalar(2.0).unwrap()),
            ],
            outputs: vec![
                PartialEvaluationOutput::Known(Array::scalar(5.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
            ],
        };

        // Interpretation takes exactly one value per `Unknown` feeder, feeds `Known` feeders from their carried
        // values, returns folded outputs directly, and reads the rest from the replayed residual program:
        // `(5, 4 * 2 + 3) = (5, 11)`.
        let context = TestArrayContext::new();
        assert_eq!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(5.0).unwrap(), Array::scalar(11.0).unwrap()]),
        );
        assert!(matches!(
            evaluation.interpret(&context, &[]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }),
        ),);
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap(), Array::scalar(5.0).unwrap()]),
            Err(ProgramError::InvalidInputCount { expected: 1, actual: 2 }),
        ),);

        // An output that references a residual output the residual program does not produce is reported as a
        // malformed program.
        let evaluation = PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: evaluation.inputs,
            outputs: vec![PartialEvaluationOutput::Unknown(1)],
        };
        assert!(matches!(
            evaluation.interpret(&context, &[Array::scalar(4.0).unwrap()]),
            Err(ProgramError::MalformedProgram(message))
                if message == "partial evaluation output references residual output 1 but the residual program \
                    produced 1 output(s)",
        ),);

        // Under a staging known-side context, the same replay stages the residual program into the outer trace
        // instead of executing it. Its constant is lifted as a staged constant, its instructions are staged as outer
        // instructions, folded outputs return their tracers directly, and residual outputs are tracers naming the
        // staged atoms.
        let outer = TestArrayTracingContext::new();
        let folded = outer.input(ArrayType::scalar(DataType::F64));
        let unknown = outer.input(ArrayType::scalar(DataType::F64));
        let feeder = outer.constant(Array::scalar(2.0).unwrap());
        let evaluation = PartialEvaluation::<TestArrayTracingContext> {
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
    fn test_effect_ordering_is_empty() {
        assert!(EffectOrdering::<usize>::default().is_empty());
        assert!(EffectOrdering::<usize>::PerReference(HashSet::new()).is_empty());
        assert!(!EffectOrdering::PerReference(HashSet::from([1])).is_empty());
        assert!(!EffectOrdering::<usize>::Global.is_empty());
    }

    #[test]
    fn test_effect_ordering_conflicts() {
        let empty = EffectOrdering::<usize>::default();
        assert!(!empty.conflicts(&EffectOrdering::Global));
        assert!(!EffectOrdering::Global.conflicts(&empty));
        assert!(EffectOrdering::<usize>::Global.conflicts(&EffectOrdering::Global));
        assert!(EffectOrdering::Global.conflicts(&EffectOrdering::PerReference(HashSet::from([1]))));
        assert!(EffectOrdering::PerReference(HashSet::from([1])).conflicts(&EffectOrdering::Global));
        let dependencies = EffectOrdering::PerReference(HashSet::from([1, 2]));
        let overlapping = EffectOrdering::PerReference(HashSet::from([2, 3]));
        let disjoint = EffectOrdering::PerReference(HashSet::from([3, 4]));
        assert!(dependencies.conflicts(&overlapping));
        assert!(overlapping.conflicts(&dependencies));
        assert!(!dependencies.conflicts(&disjoint));
        assert!(!disjoint.conflicts(&dependencies));
        assert!(!dependencies.conflicts(&EffectOrdering::default()));
        assert!(!EffectOrdering::default().conflicts(&dependencies));
    }

    #[test]
    fn test_effect_ordering_extend() {
        let mut dependencies = EffectOrdering::PerReference(HashSet::from([1, 2]));
        dependencies.extend(&EffectOrdering::PerReference(HashSet::from([2, 3])));
        dependencies.extend(&EffectOrdering::default());
        assert!(
            matches!(&dependencies, EffectOrdering::PerReference(references) if *references == HashSet::from([1, 2, 3])),
        );
        assert!(!matches!(dependencies, EffectOrdering::Global));
        dependencies.extend(&EffectOrdering::Global);
        dependencies.extend(&EffectOrdering::PerReference(HashSet::from([4])));
        assert!(matches!(dependencies, EffectOrdering::Global));
        assert!(dependencies.conflicts(&EffectOrdering::PerReference(HashSet::from([5]))));
    }

    #[test]
    fn test_effects_summary_effect_ordering() {
        assert!(EffectsSummary::PURE.effect_ordering([0]).is_empty());
        let read = ReferenceReadOperation::<ArrayType, ArrayIrType>::new().effects().summary();
        assert!(
            matches!(read.effect_ordering([0, 1]), EffectOrdering::PerReference(roots) if roots == HashSet::from([0, 1])),
        );
        // Unresolved reference roots and explicitly ordered state cannot establish per-reference independence.
        assert!(matches!(read.effect_ordering(Vec::<usize>::new()), EffectOrdering::Global));
        let explicit = Effects::explicit(EffectClasses::single(EffectClass::OrderedState)).summary();
        assert!(matches!(explicit.effect_ordering([0]), EffectOrdering::Global));
        let io = Effects::explicit(EffectClasses::single(EffectClass::OrderedIo)).summary();
        assert!(matches!(io.effect_ordering([0]), EffectOrdering::Global));
    }

    #[test]
    fn test_partitioned_program_known_reference_inputs() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut builder = ProgramBuilder::<TestValue, TestArrayIrOperation>::new();
        let value = builder.add_input(scalar_type.clone());
        let reference = builder.add_input(reference_type.clone());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();

        // The unknown value occupies residual input 0 and the known reference enters at residual input 1.
        assert_eq!(program.partition(&[false, true]).unwrap().known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(
            program.partition(&[false, false]).unwrap().known_reference_inputs().collect::<Vec<_>>(),
            Vec::<usize>::new(),
        );
        assert_eq!(
            program.partition(&[true, false]).unwrap().known_reference_inputs().collect::<Vec<_>>(),
            Vec::<usize>::new(),
        );

        // A known reference consumed entirely on the known side does not cross the partition. This does not prove
        // it is disjoint from the unknown reference supplied at invocation.
        let mut builder = ProgramBuilder::<TestValue, TestArrayIrOperation>::new();
        let source = builder.add_input(reference_type.clone());
        let destination = builder.add_input(reference_type.clone());
        let value = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.partition(&[true, false]).unwrap().known_reference_inputs().collect::<Vec<_>>(),
            Vec::<usize>::new(),
        );

        // Inline reference constants are not input feeders, on either side of the partition.
        let mut builder = ProgramBuilder::<TestCapture, ReferenceWriteOperation<ArrayType, ArrayIrType>>::new();
        let value = builder.add_input(scalar_type);
        let captured = builder.add_constant(CaptureReference::new(0, reference_type));
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![captured, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestCapture>, Vec<TestCapture>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        assert_eq!(
            program.partition(&[true]).unwrap().known_reference_inputs().collect::<Vec<_>>(),
            Vec::<usize>::new(),
        );
        assert_eq!(
            program.partition(&[false]).unwrap().known_reference_inputs().collect::<Vec<_>>(),
            Vec::<usize>::new(),
        );
    }

    #[test]
    fn test_partitioned_program_has_effect_ordering_conflicts() {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(reference_type.clone());
        let unknown = builder.add_input(reference_type);
        let known = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![known], None).unwrap()[0];
        let unknown =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![known, unknown], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        // Each emitted program numbers its single reference input as zero. Resolve them back to original boundary
        // inputs before comparing identities; ordinary specialization still preserves their global ordering.
        let ordinary = program.partition(&[true, false]).unwrap();
        assert!(ordinary.has_effect_ordering_conflicts());
        let (repeated, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, true, Some(&[0]))
            .unwrap();
        assert!(!repeated.has_effect_ordering_conflicts());
        assert!(!program.partition(&[false, false]).unwrap().has_effect_ordering_conflicts());

        // Reconstructing through the unqualified boundary constructor carries no repeated-invocation evidence.
        let (known, residual, input_indices, residual_inputs, outputs) = repeated.into_parts();
        let reconstructed = PartitionedProgram::from_parts(known, residual, input_indices, residual_inputs, outputs);
        assert!(reconstructed.has_effect_ordering_conflicts());
    }

    #[test]
    fn test_recursive_partial_evaluation_driver_partition_program() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let destination = builder.add_input(reference_type.clone());
        let source = builder.add_input(reference_type);
        let update = builder.add_input(scalar_type);
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![destination, update], None)
            .unwrap();
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let regions = vec![program];
        let driver = RecursivePartialEvaluationDriver { driver: &regions, repeated_residual: false };
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        context.defer_ordered_effects.set(true);

        // A fresh partition inherits configuration, but not the active context's recorded ordering constraints.
        let partition = driver.partition_program(&context, regions[0].entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert!(context.defer_ordered_effects.get());

        // The same driver uses the effect placement passed to each call.
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new()).deferred_sibling();
        let partition = driver.partition_program(&context, regions[0].entry_region_ref(), &[true, true, true]).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Unknown(0)]);
        assert!(!context.defer_ordered_effects.get());
    }

    #[test]
    fn test_partial_evaluation_context_new() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(context.reference_placement(), ReferencePlacement::Execute);
        assert!(context.allow_effect_folding);
        assert!(context.builder.borrow().instructions().is_empty());
        assert!(context.inputs.borrow().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_with_reference_placement() {
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let input = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let configured = context.clone().with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(context.reference_placement(), ReferencePlacement::Execute);
        assert_eq!(configured.reference_placement(), ReferencePlacement::Stage);
        let restored = configured.clone().with_reference_placement(ReferencePlacement::Execute);
        assert_eq!(restored.reference_placement(), ReferencePlacement::Execute);
        assert_eq!(configured.reference_placement(), ReferencePlacement::Stage);
        // Reconfiguration shares the accumulated residual state rather than starting a fresh program.
        let outputs = configured.residualize(ReferenceNewOperation::new(), Vec::new(), &[input]).unwrap();
        drop((context, configured));
        let evaluation = restored.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.inputs, vec![PartialEvaluationInput::Unknown(0)]);
        assert_eq!(evaluation.program.instructions().len(), 1);
        let outputs = evaluation
            .interpret(&EagerContext::new(), &[TestValue::Array(Array::scalar(3.0_f32).unwrap())])
            .unwrap();
        assert!(
            matches!(&outputs[0], TestValue::Reference(reference) if reference.read() == Ok(Array::scalar(3.0_f32).unwrap())),
        );
    }

    #[test]
    fn test_partial_evaluation_context_with_allow_effect_folding() {
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let input = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let configured = context.clone().with_allow_effect_folding(false);
        assert!(context.allow_effect_folding);
        assert!(!configured.allow_effect_folding);
        let restored = configured.clone().with_allow_effect_folding(true);
        assert!(restored.allow_effect_folding);
        assert!(!configured.allow_effect_folding);
        // Enabling folding again must restore behavior, while preserving the previously created residual input.
        let initial = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        let allocated =
            restored.fold_or_residualize(ReferenceNewOperation::new(), Vec::new(), &[initial.clone()]).unwrap();
        assert!(
            matches!(allocated[0].as_known(), Some(TestValue::Reference(reference)) if reference.read() == Ok(Array::scalar(2.0_f32).unwrap())),
        );
        let deferred = configured.fold_or_residualize(ReferenceNewOperation::new(), Vec::new(), &[initial]).unwrap();
        assert!(deferred[0].is_unknown());
        drop((configured, restored));
        let evaluation = context.into_evaluation(vec![input, deferred[0].clone()]).unwrap();
        assert_eq!(evaluation.inputs.len(), 2);
        let outputs = evaluation
            .interpret(&EagerContext::new(), &[TestValue::Array(Array::scalar(5.0_f32).unwrap())])
            .unwrap();
        assert_eq!(outputs[0], TestValue::Array(Array::scalar(5.0_f32).unwrap()));
        assert!(
            matches!(&outputs[1], TestValue::Reference(reference) if reference.read() == Ok(Array::scalar(2.0_f32).unwrap())),
        );
    }

    #[test]
    fn test_partial_evaluation_context_deferred_sibling() {
        let parent = TracingContext::<TestValue, TestOperation>::new();
        let source = PartialEvaluationContext::new(parent.clone());
        let context = source.deferred_sibling();
        assert!(Rc::ptr_eq(&source.parent, &context.parent));
        assert!(Rc::ptr_eq(&context.parent, &context.clone().parent));
        assert!(!Rc::ptr_eq(&source.builder, &context.builder));
        assert!(!Rc::ptr_eq(&source.parent, &PartialEvaluationContext::new(parent.clone()).parent));
        let zero = context.lift(TestValue::Array(Array::scalar(0.0_f32).unwrap())).unwrap();
        let allocated = context.bind(ReferenceNewOperation::new(), Vec::new(), &[zero.clone()]).unwrap();
        assert!(allocated[0].value().unwrap().is_unknown());
        assert!(parent.builder().borrow().instructions().is_empty());

        // Pure known work can still execute in the parent after a residual effect.
        let folded = context.clone().bind(AddOperation::new(), Vec::new(), &[zero.clone(), zero]).unwrap();
        assert!(folded[0].value().unwrap().is_known());
        assert_eq!(parent.builder().borrow().instructions().len(), 1);
        assert_eq!(context.builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_deferred_sibling_preserves_reference_placement() {
        let source = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        let target = source.deferred_sibling();
        assert_eq!(target.reference_placement(), ReferencePlacement::Stage);
        let reference = ArrayReference::new(Array::vector(vec![1.0_f32, 2.0]).unwrap());
        let source_reference = source.lift(TestValue::Reference(reference)).unwrap();
        let target_reference = target.import_known(&source_reference).unwrap();
        let viewed = target.bind(ReferenceIndexOperation::new(0, 0), Vec::new(), &[target_reference]).unwrap();
        assert!(viewed[0].value().unwrap().is_unknown());
        assert_eq!(target.builder.borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_parent() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(
            context.parent().bind(
                AddOperation::new(),
                Vec::new(),
                &[Array::scalar(1.0).unwrap(), Array::scalar(2.0).unwrap()]
            ),
            Ok(vec![Array::scalar(3.0).unwrap()]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_import_known() {
        let parent = TracingContext::<TestValue, TestArrayIrOperation>::new();
        let source = PartialEvaluationContext::new(parent.clone());
        let target = source.deferred_sibling();
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let known =
            PartialTracer::new(source.clone(), PartialEvaluationValue::known(parent.input(scalar_type.clone())));
        source
            .residualize(
                TestArrayOperation::Add(AddOperation::new()),
                Vec::new(),
                &[known.value().unwrap().clone(), known.value().unwrap().clone()],
            )
            .unwrap();
        assert!(matches!(
            known.value().unwrap().materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(_) },
        ),);
        let imported = target.import_known(&known).unwrap();
        assert_eq!(
            imported.value().unwrap().materialization(),
            PartialValueMaterialization::Input { residual_atom: None },
        );
        assert_eq!(target.import_known(&imported).unwrap(), imported);

        let reference = parent.input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let known_reference = PartialTracer::new(source.clone(), PartialEvaluationValue::known(reference));
        let imported_reference = target.import_known(&known_reference).unwrap();
        assert_eq!(target.reference_identity(&imported_reference), source.reference_identity(&known_reference));

        let unknown = PartialTracer::new(source.clone(), source.unknown_input(scalar_type.clone(), 0));
        assert!(matches!(target.import_known(&unknown), Err(ProgramError::MalformedProgram(message))
            if message == "cannot import an unknown value from another partial-evaluation context",),);
        let other_parent = TracingContext::<TestValue, TestArrayIrOperation>::new();
        let foreign = PartialTracer::new(
            PartialEvaluationContext::new(other_parent.clone()),
            PartialEvaluationValue::known(other_parent.input(scalar_type)),
        );
        assert!(matches!(target.import_known(&foreign), Err(ProgramError::MalformedProgram(message))
            if message == "cannot import a value from a partial-evaluation context with a different parent context",),);

        // Binding a foreign value cannot reinterpret its atom identifier in this builder, including in release builds.
        let poisoned = target
            .bind(TestArrayOperation::Add(AddOperation::new()), Vec::new(), &[known.clone(), known])
            .unwrap();
        assert!(matches!(poisoned[0].value(), Err(ProgramError::MalformedProgram(message))
            if message == "cannot bind a value belonging to another partial-evaluation context; import known values explicitly",),);
        assert!(target.builder.borrow().instructions().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_import_known_deduplicates_eager_inputs() {
        let source = PartialEvaluationContext::new(TestArrayContext::new());
        let target = source.deferred_sibling();
        let input =
            PartialTracer::new(source.clone(), PartialEvaluationValue::known_input(Array::scalar(3.0).unwrap()));
        let first = target.import_known(&input).unwrap();
        let second = target.import_known(&input).unwrap();
        assert_eq!(first, second);
        let outputs = target
            .residualize(
                AddOperation::new(),
                Vec::new(),
                &[first.value().unwrap().clone(), second.value().unwrap().clone()],
            )
            .unwrap();
        drop(first);
        drop(second);
        let evaluation = target.into_evaluation(outputs).unwrap();
        assert_eq!(evaluation.inputs(), &[PartialEvaluationInput::Known(Array::scalar(3.0).unwrap())]);
        assert_eq!(evaluation.program().inputs().count(), 1);
        assert_eq!(evaluation.interpret(source.parent(), &[]), Ok(vec![Array::scalar(6.0).unwrap()]));
    }

    #[test]
    fn test_partial_evaluation_context_import_known_retains_discarded_errors() {
        let source = PartialEvaluationContext::new(TestArrayContext::new());
        let target = source.deferred_sibling();
        let known = source.lift(Array::scalar(3.0).unwrap()).unwrap();
        assert_eq!(
            source.bind(AddOperation::new(), Vec::new(), &[known.clone()]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        );
        let imported = target.import_known(&known).unwrap();
        assert!(matches!(imported.value(), Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 })));
        drop(imported);
        assert!(matches!(
            target.into_evaluation(Vec::new()),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        ),);
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        // `fold_or_residualize` folds an all-known operation through the known-side context, and so its outputs
        // are known values with no residual materialization decision yet.
        let inputs = [
            PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
            PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
        ];
        let folded = context.fold_or_residualize(MulOperation::new(), Vec::new(), &inputs).unwrap();
        assert_eq!(folded.len(), 1);
        assert!(folded[0].is_known());
        assert!(!folded[0].is_unknown());
        assert_eq!(folded[0].as_known(), Some(&Array::scalar(6.0).unwrap()));
        assert_eq!(folded[0].materialization(), PartialValueMaterialization::Undecided);
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(folded[0].value(), PartialValue::Known(value) if *value == Array::scalar(6.0).unwrap()));

        let unknown = context.unknown_input(ArrayType::scalar(DataType::F64), 0);
        let mixed = context.fold_or_residualize(NegOperation::new(), Vec::new(), &[unknown]).unwrap();
        let evaluation = context.into_evaluation(mixed).unwrap();
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(7.0).unwrap()]),
            Ok(vec![Array::scalar(-7.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_order_for_unrooted_state() {
        let scalar_type = ArrayType::scalar(DataType::F64);
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestOrderedStateOperation>::new());
        let known = [PartialEvaluationValue::known(Array::scalar(1.0).unwrap())];
        let unknown = [context.unknown_input(scalar_type, 0)];

        // Ordered state folds like any other all-known operation before any ordered effect has been deferred.
        let folded = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert_eq!(folded[0].as_known(), Some(&Array::scalar(1.0).unwrap()));

        // A state access with an unknown operand stages. All later ordered operations must then stage too, even
        // with known operands and no shared reference allocation. Pure work keeps folding.
        let staged = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &unknown).unwrap();
        assert!(staged[0].is_unknown());
        let later = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert!(later[0].is_unknown());
        let pure = context.fold_or_residualize(TestOrderedStateOperation::Pure, Vec::new(), &known).unwrap();
        assert_eq!(pure[0].as_known(), Some(&Array::scalar(1.0).unwrap()));
        let evaluation = context.into_evaluation(vec![later[0].clone()]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(1.0).unwrap())],
        );
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = state %0
                    %3:f64[] = state %1
                in (%3)
            "}
            .trim_end(),
        );

        // Direct residual emission also records that ordered work was deferred, so later ordered operations
        // remain residual just as they do after the default rule defers an operation.
        let context = PartialEvaluationContext::new(EagerContext::<Array, TestOrderedStateOperation>::new());
        let known = [PartialEvaluationValue::known(Array::scalar(1.0).unwrap())];
        context.residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        let later = context.fold_or_residualize(TestOrderedStateOperation::State(0), Vec::new(), &known).unwrap();
        assert!(later[0].is_unknown());
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_stages_reference_operations_under_stage_placement() {
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        assert_eq!(context.reference_placement(), ReferencePlacement::Stage);
        assert_eq!(
            PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new()).reference_placement(),
            ReferencePlacement::Execute,
        );

        // Under the specialization placement every reference operation stages, all-known operands notwithstanding:
        // the update leaves the live state untouched, the allocation is deferred to the residual program, and only
        // reference-free work (pure or ordered) folds.
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        let initial = PartialEvaluationValue::known(TestValue::Array(Array::scalar(3.0_f32).unwrap()));
        let add_update = TestOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new());
        assert!(
            context
                .fold_or_residualize(add_update, Vec::new(), &[reference.clone(), update])
                .unwrap()
                .is_empty(),
        );
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));
        let allocation = TestOperation::ReferenceNew(ReferenceNewOperation::new());
        let allocated = context.fold_or_residualize(allocation, Vec::new(), &[initial.clone()]).unwrap();
        assert!(allocated[0].is_unknown());
        let product = TestOperation::from(ArrayOperation::from(MulOperation::new()));
        let folded = context.fold_or_residualize(product, Vec::new(), &[initial.clone(), initial.clone()]).unwrap();
        assert_eq!(folded[0].as_known(), Some(&TestValue::Array(Array::scalar(9.0_f32).unwrap())));

        // The live handle crosses into the residual program by identity as a residual reference, so replaying the
        // residual program performs the deferred accesses against the same allocation.
        let evaluation = context.into_evaluation(allocated).unwrap();
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[], %2:f32[] .
                let reference_add_update %0 %1
                    %3:ref<f32[]> = reference_new %2
                in (%3)
            "}
            .trim_end(),
        );
        let outputs = evaluation.interpret(&EagerContext::<TestValue, TestOperation>::new(), &[]).unwrap();
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        let TestValue::Reference(allocated) = &outputs[0] else {
            panic!("residual replay returned {:?} instead of the deferred allocation", outputs[0]);
        };
        assert_eq!(allocated.read(), Ok(Array::scalar(3.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_reference_failure_order() {
        use crate::programs::ReferenceError;

        // An unused read can fail, so later I/O must not execute during specialization ahead of that failure.
        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        frozen.freeze().unwrap();
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new())
            .with_reference_placement(ReferencePlacement::Stage);
        let reference = PartialEvaluationValue::known(TestValue::Reference(frozen));
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let output = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        assert!(output[0].is_unknown());
        let value = PartialEvaluationValue::known(TestValue::Array(Array::scalar(42.0_f32).unwrap()));
        let print = TestOperation::from(ArrayOperation::from(PrintOperation::new("after_read")));
        let printed = context.fold_or_residualize(print, Vec::new(), &[value]).unwrap();
        assert!(printed[0].is_unknown());
        let evaluation = context.into_evaluation(printed).unwrap();
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_read", "print"],
        );
        let error = evaluation.interpret(&EagerContext::<TestValue, TestOperation>::new(), &[]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_preserves_order_across_reference_roots() {
        use crate::programs::ReferenceError;

        // The default execution policy must also preserve order when the earlier reference is an unknown input.
        // Its failure prevents a later write to another root, including when that write has entirely known operands.
        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        frozen.freeze().unwrap();
        let other = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(TestArrayIrContext::new());
        let reference = context.unknown_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into(), 0);
        let read = TestArrayIrOperation::ReferenceRead(ReferenceReadOperation::new());
        let output = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        let target = PartialEvaluationValue::known(TestValue::Reference(other.clone()));
        let value = PartialEvaluationValue::known(TestValue::Array(Array::scalar(7.0_f32).unwrap()));
        let write = TestArrayIrOperation::ReferenceWrite(ReferenceWriteOperation::new());
        assert!(context.fold_or_residualize(write, Vec::new(), &[target, value]).unwrap().is_empty());
        assert_eq!(other.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let evaluation = context.into_evaluation(output).unwrap();
        let error = evaluation.interpret(&TestArrayIrContext::new(), &[TestValue::Reference(frozen)]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(other.read(), Ok(Array::scalar(2.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_excludes_dormant_effects() {
        use crate::programs::RegionSlot;
        use crate::tests::TestRegionOperation;

        let mut builder = ProgramBuilder::<Array, TestRegionOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let output = builder
            .add_instruction(TestRegionOperation::Effectful(EffectClass::OrderedState), Vec::new(), vec![input], None)
            .unwrap()[0];
        let rule = builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();
        let outer = TracingContext::<Array, TestRegionOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone());
        let known = PartialEvaluationValue::known(outer.input(ArrayType::scalar(DataType::F32)));
        let unknown = context.unknown_input(ArrayType::scalar(DataType::F32), 0);
        let operation = TestRegionOperation::WithRegions(const { &[RegionSlot::rule("derivative")] });
        context.fold_or_residualize(operation, vec![rule], &[unknown]).unwrap();
        let retained = context
            .fold_or_residualize(TestRegionOperation::Effectful(EffectClass::OrderedIo), Vec::new(), &[known])
            .unwrap();
        assert!(retained[0].is_known());
        assert_eq!(outer.builder().borrow().instructions().len(), 1);
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_executes_reference_operations_under_execute_placement() {
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let other = ArrayReference::new(Array::scalar(5.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let other_reference = PartialEvaluationValue::known(TestValue::Reference(other.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let u = context.unknown_input(ArrayType::scalar(DataType::F32).into(), 0);
        let add_update = TestOperation::ReferenceAddUpdate(ReferenceAddUpdateOperation::new());
        let write = TestOperation::ReferenceWrite(ReferenceWriteOperation::new());
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());

        // Under the execution placement an all-known reference operation executes at partial-evaluation time like any
        // other known ordered effect. This is the known side of an eager linearization (i.e., the forward pass).
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        assert!(
            context
                .fold_or_residualize(add_update, Vec::new(), &[reference.clone(), update])
                .unwrap()
                .is_empty(),
        );
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        assert!(context.builder.borrow().instructions().is_empty());

        // A write of the unknown value stages and prevents later ordered effects from folding: later known reads stage
        // even on a different allocation, preserving failure and synchronization order.
        assert!(context.fold_or_residualize(write, Vec::new(), &[reference.clone(), u]).unwrap().is_empty());
        assert_eq!(live.read(), Ok(Array::scalar(3.0_f32).unwrap()));
        let staged = context.fold_or_residualize(read.clone(), Vec::new(), &[reference]).unwrap();
        assert!(staged[0].is_unknown());
        let other_read = context.fold_or_residualize(read, Vec::new(), &[other_reference]).unwrap();
        assert!(other_read[0].is_unknown());
    }

    #[test]
    fn test_partial_evaluation_context_fold_or_residualize_retains_ordering_after_type_error() {
        // A write of an unknown `f64` into an `f32` reference fails at residual emission (type inference rejects the
        // mismatch) after recording that later ordered effects must remain residual, so a later all-known read of that
        // root stages instead of executing against state the failed write attempted to mutate.
        let live = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        let reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let mismatched = context.unknown_input(ArrayType::scalar(DataType::F64).into(), 0);
        let write = TestOperation::ReferenceWrite(ReferenceWriteOperation::new());
        assert!(matches!(
            context.fold_or_residualize(write, Vec::new(), &[reference.clone(), mismatched]),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`reference_write` replacement type `f64[]` must exactly \
                               match reference referent type `f32[]`",
        ),);
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let staged = context.fold_or_residualize(read, Vec::new(), &[reference]).unwrap();
        assert!(staged[0].is_unknown());
        assert_eq!(live.read(), Ok(Array::scalar(1.0_f32).unwrap()));
    }

    #[test]
    fn test_partial_evaluation_context_residualize() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let inputs = [
            PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
            PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
        ];
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
        let shared = PartialEvaluationValue::known_input(Array::scalar(4.0).unwrap());
        let first = context.residualize(NegOperation::new(), Vec::new(), &[shared.clone()]).unwrap();
        let second = context.residualize(AddOperation::new(), Vec::new(), &[shared.clone(), shared.clone()]).unwrap();
        assert_eq!(
            shared.materialization(),
            PartialValueMaterialization::Input { residual_atom: Some(AtomId::new(4)) },
        );
        assert!(first[0].is_unknown() && second[0].is_unknown());
        assert_eq!(context.inputs.borrow().len(), 3);
    }

    #[test]
    fn test_partial_evaluation_context_inline_program() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        // `inline_program` replays a program over seed values. All-known seeds fold every instruction, lifting the
        // program constant into the known-side context, and so the replay returns folded values.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, x], None).unwrap()[0];
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![product, c], None).unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![sum], vec![Placeholder; 2], vec![Placeholder]).unwrap();
        let outputs = context
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    PartialEvaluationValue::known(Array::scalar(3.0).unwrap()),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(7.0).unwrap()));

        // Mixed seeds fold the known work and residualize the rest, and so the walk returns residual variables.
        let outputs = context
            .inline_program(
                &program,
                vec![
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    context.unknown_input(ArrayType::scalar(DataType::F64), 0),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::scalar(11.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let observations = RefCell::new(Vec::new());
        let (known, residual) = replay_reference_ordering_program(region, None, None);
        let (observed_known, observed_residual) = replay_reference_ordering_program(region, Some(&observations), None);

        // Recording source instructions must not change ordering or either emitted program. The nested read has no
        // public outputs, so it must be recorded based on the residual instructions it emits.
        assert_eq!(known.to_string(), observed_known.to_string());
        assert_eq!(residual.to_string(), observed_residual.to_string());
        assert_eq!(known.instructions().len(), 0);
        assert_eq!(
            *observations.borrow(),
            vec![InstructionId::new(region.id(), 0), InstructionId::new(region.id(), 1)],
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation_with_reference_analysis() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_configuration(None, true, &[]).unwrap();
        let observations = RefCell::new(Vec::new());
        let (known, residual) = replay_reference_ordering_program(region, None, Some(analysis.as_ref()));
        let (observed_known, observed_residual) =
            replay_reference_ordering_program(region, Some(&observations), Some(analysis.as_ref()));

        // Recording source instructions must not change ordering or either emitted program. The nested read has no
        // public outputs, so it must be recorded based on the residual instructions it emits.
        assert_eq!(known.to_string(), observed_known.to_string());
        assert_eq!(residual.to_string(), observed_residual.to_string());
        assert_eq!(known.instructions().len(), 1);
        assert_eq!(*observations.borrow(), vec![InstructionId::new(region.id(), 0)]);
    }

    #[test]
    fn test_partial_evaluation_context_inline_region_observation_preserves_validation() {
        let program = reference_ordering_program();
        let region = program.entry_region_ref();
        let analysis = region.reference_analysis_with_configuration(None, true, &[]).unwrap();
        let observations = RefCell::new(Vec::new());
        let context = PartialEvaluationContext::new(TracingContext::<TestValue, TestOperation>::new());

        // Invalid input arity is rejected before replay, regardless of observation or reference analysis.
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), None, None),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), Some(&observations), None),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), None, Some(analysis.as_ref())),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(matches!(
            context.inline_region(region, Vec::new(), &HashSet::new(), Some(&observations), Some(analysis.as_ref())),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 0 }),
        ),);
        assert!(observations.borrow().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_inline_partitioned_program() {
        // Partition `x - -a` with `a` known. Its single-operation halves can serve as boundary operations;
        // subtraction makes swapping the known coefficient and unknown input observable in the result.
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let difference = builder.add_instruction(SubOperation::new(), Vec::new(), vec![x, negated], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![difference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[
                    PartialEvaluationValue::known(Array::scalar(2.0).unwrap()),
                    context.unknown_input(ArrayType::scalar(DataType::F64), 0),
                ],
                |_| (ArrayOperation::Neg(NegOperation::new()), Vec::new()),
                |_| (ArrayOperation::Sub(SubOperation::new()), Vec::new()),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(outputs[0].is_unknown());
        assert!(matches!(outputs[0].materialization(), PartialValueMaterialization::Variable { .. }));

        let evaluation = context.into_evaluation(outputs).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(-2.0_f64).unwrap())],
        );
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(evaluation.program.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(1)]);
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(5.0).unwrap()]),
            Ok(vec![Array::scalar(5.0 - (-2.0_f64)).unwrap()]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_inline_partitioned_program_all_known() {
        let context = PartialEvaluationContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![negated], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let partition = program.partition(&[true]).unwrap();

        // Preserve the empty residual program's boundary with a region wrapper. It emits no outputs and is removed
        // by finalization because it has no observable effects.
        let outputs = context
            .inline_partitioned_program(
                partition,
                &[PartialEvaluationValue::known(Array::scalar(2.0).unwrap())],
                |_| (ArrayOperation::Neg(NegOperation::new()), Vec::new()),
                |program| (ArrayOperation::LinearCall(LinearCallOperation::new(0)), vec![program.clone(), program]),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].as_known(), Some(&Array::scalar(-2.0_f64).unwrap()));
        let evaluation = context.into_evaluation(outputs).unwrap();
        assert!(evaluation.inputs.is_empty());
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Known(Array::scalar(-2.0_f64).unwrap())]);
        assert!(evaluation.program.instructions().is_empty());
        assert!(evaluation.program.input_ids().is_empty());
        assert!(evaluation.program.output_ids().is_empty());
        assert_eq!(evaluation.interpret(&EagerContext::new(), &[]), Ok(vec![Array::scalar(-2.0_f64).unwrap()]));
    }

    #[test]
    fn test_partial_evaluation_context_known_constant() {
        // `known_constant` recovers a known value's staged-constant payload. An eager known value always resolves to a
        // constant, while under a staging known-side context only literal-backed tracers do.
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert_eq!(context.known_constant(&Array::scalar(5.0).unwrap()), Ok(Array::scalar(5.0).unwrap()));
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());
        assert_eq!(staging_context.known_constant(&literal), Ok(Array::scalar(4.0).unwrap()));
        assert!(matches!(
            staging_context.known_constant(&symbolic),
            Err(ProgramError::MalformedProgram(message))
                if message == "a known value crossing into a nested residual program does not resolve to a constant \
                    in the active known-side context",
        ),);
    }

    #[test]
    fn test_partial_evaluation_context_all_knowns_are_constants() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());

        // A known feeder still occupies an input of the residual program.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();

        assert!(context.all_knowns_are_constants(&PartialEvaluation::<TestArrayContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(Array::scalar(1.0).unwrap())],
            outputs: vec![
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Known(Array::scalar(2.0).unwrap())
            ],
        }),);

        assert!(!staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(symbolic.clone())],
            outputs: vec![PartialEvaluationOutput::Unknown(0)],
        }),);

        assert!(staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program: program.clone(),
            inputs: vec![PartialEvaluationInput::Known(literal.clone())],
            outputs: vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Known(literal)],
        }),);

        // Symbolic outputs must also reject constant-only reconstruction even when every feeder is constant.
        assert!(!staging_context.all_knowns_are_constants(&PartialEvaluation::<TestArrayTracingContext> {
            program,
            inputs: vec![PartialEvaluationInput::Unknown(0)],
            outputs: vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Known(symbolic)],
        }),);
    }

    #[test]
    fn test_partial_evaluation_context_any_known_is_symbolic() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let staging = TestArrayTracingContext::new();
        let staging_context = PartialEvaluationContext::new(staging.clone());
        let symbolic = staging.input(ArrayType::scalar(DataType::F64));
        let literal = staging.constant(Array::scalar(4.0).unwrap());

        // `any_known_is_symbolic` is the signal online boundary rules split on. Only a known value that does not
        // resolve to a program constant counts, and so eager knowns and unknowns never do.
        assert!(!context.any_known_is_symbolic(&[PartialEvaluationValue::known(Array::scalar(1.0).unwrap())]));
        assert!(!staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(literal)]));
        assert!(staging_context.any_known_is_symbolic(&[PartialEvaluationValue::known(symbolic)]));
        assert!(
            !staging_context
                .any_known_is_symbolic(&[staging_context.unknown_input(ArrayType::scalar(DataType::F64), 0)]),
        );
    }

    #[test]
    fn test_partial_evaluation_context_lift() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        assert_eq!(
            lifted.value().unwrap().materialization(),
            PartialValueMaterialization::Constant { residual_atom: None },
        );
        assert_eq!(lifted.value().unwrap().as_known(), Some(&Array::scalar(2.0).unwrap()));
        let truth = context.lift(Array::scalar(true).unwrap()).unwrap();
        assert_eq!(truth.concretize(), Ok(true));
    }

    #[test]
    fn test_partial_evaluation_context_bind() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        let folded = context.bind(AddOperation::new(), Vec::new(), &[lifted.clone(), lifted.clone()]).unwrap();
        assert_eq!(folded.len(), 1);
        assert_eq!(folded[0].value().unwrap().as_known(), Some(&Array::scalar(4.0).unwrap()));
        assert_eq!(folded[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        // A mixed bind retains the known operand as a feeder and stages the multiplication.
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F64), 0));
        let mixed = context.bind(MulOperation::new(), Vec::new(), &[folded[0].clone(), unknown.clone()]).unwrap();
        assert!(mixed[0].value().unwrap().is_unknown());
        let output = mixed[0].value().unwrap().clone();
        drop((lifted, folded, unknown, mixed));
        let evaluation = context.into_evaluation(vec![output]).unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(Array::scalar(4.0).unwrap())],
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
        assert_eq!(
            evaluation.interpret(&EagerContext::new(), &[Array::scalar(3.0).unwrap()]),
            Ok(vec![Array::scalar(12.0).unwrap()])
        );
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_materialization_error() {
        // A staged reference incorrectly marked as a literal has no constant payload to materialize. Binding a
        // swap retains that error, so later reads with valid operands also fail without emitting parent work.
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let reference = outer.input(reference_type);
        let context = PartialEvaluationContext::new(outer.clone());
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(scalar_type, 0));
        let malformed = PartialTracer::new(context.clone(), PartialEvaluationValue::known_constant(reference.clone()));
        let swap = TestOperation::ReferenceSwap(ReferenceSwapOperation::new());
        let poisoned = context.bind(swap, Vec::new(), &[malformed, unknown]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert!(matches!(poisoned[0].value(), Err(ProgramError::MalformedProgram(message))
                if message == "residual materialization required a constant payload for a known value \
                               that is not resolvable to a constant in the active known-side context"),);
        let read = TestOperation::ReferenceRead(ReferenceReadOperation::new());
        let known = PartialTracer::new(context.clone(), PartialEvaluationValue::known(reference));
        let later = context.bind(read, Vec::new(), &[known]).unwrap();
        assert!(matches!(later[0].value(), Err(ProgramError::MalformedProgram(message))
                if message == "residual materialization required a constant payload for a known value \
                               that is not resolvable to a constant in the active known-side context"),);
        assert!(outer.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_mismatched_builder_error() {
        // A failed bind (in this case, folding an operation whose known inputs belong to two different traces) does
        // not return an error; it poisons its outputs so the infallible operator sugar driving closures never panics.
        // The poison propagates through later binds, resolves `Opaque`, rejects concretizing extractions with the
        // deferred error, and surfaces that original error at the value boundary.
        let outer_a = TestArrayTracingContext::new();
        let outer_b = TestArrayTracingContext::new();
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
    fn test_partial_evaluation_context_bind_retains_zero_output_failure() {
        use crate::programs::ReferenceError;

        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let failed_reference = context.lift(TestValue::Reference(frozen)).unwrap();
        let later_reference = context.lift(TestValue::Reference(live.clone())).unwrap();
        let update = context.lift(TestValue::Array(Array::scalar(3.0_f32).unwrap())).unwrap();
        let failed = context.bind(ReferenceWriteOperation::new(), Vec::new(), &[failed_reference, update.clone()]);
        assert!(failed.unwrap().is_empty());

        // Clones retain the same failure and prevent unrelated later mutations, even without a poisoned operand.
        let later = context.clone().bind(ReferenceWriteOperation::new(), Vec::new(), &[later_reference, update]);
        assert!(later.unwrap().is_empty());
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let error = context.into_evaluation(Vec::new()).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_bind_retains_discarded_output_failure() {
        use crate::programs::ReferenceError;

        let frozen = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(frozen.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let context = PartialEvaluationContext::new(EagerContext::<TestValue, TestOperation>::new());
        let failed_reference = context.lift(TestValue::Reference(frozen)).unwrap();
        let poisoned = context.bind(ReferenceReadOperation::new(), Vec::new(), &[failed_reference]).unwrap();
        assert_eq!(poisoned.len(), 1);
        assert_eq!(poisoned[0].value().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen),);
        drop(poisoned);

        let later_reference = PartialEvaluationValue::known(TestValue::Reference(live.clone()));
        let update = PartialEvaluationValue::known(TestValue::Array(Array::scalar(3.0_f32).unwrap()));
        let error = context
            .fold_or_residualize(ReferenceWriteOperation::new(), Vec::new(), &[later_reference, update])
            .unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        let output = PartialEvaluationValue::known(TestValue::Array(Array::scalar(4.0_f32).unwrap()));
        let error = context.into_evaluation(vec![output]).unwrap_err();
        assert_eq!(error.downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_partial_evaluation_context_is_eager() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        assert!(context.is_eager());
        assert!(!PartialEvaluationContext::new(TestArrayTracingContext::new()).is_eager());
    }

    #[test]
    fn test_partial_evaluation_context_provenance() {
        let parent = TestArrayTracingContext::new();
        let (context, initial) = parent.invoke_with_provenance_scope(ProvenanceScope::new("parent"), || {
            (PartialEvaluationContext::new(parent.clone()), parent.provenance())
        });
        let cloned = context.clone();
        assert_eq!(context.provenance(), initial);
        assert_eq!(cloned.provenance(), initial);
        assert_eq!(parent.provenance(), Provenance::unknown());

        // The residual context snapshots its parent's provenance at construction, then shares its own active state
        // across clones. A local scope changes neither the parent nor the state restored after the scope exits.
        context.invoke_with_provenance_scope(ProvenanceScope::new("residual"), || {
            assert_eq!(context.provenance(), Provenance::scope(ProvenanceScope::new("residual"), initial.clone()));
            assert_eq!(cloned.provenance(), context.provenance());
            assert_eq!(parent.provenance(), Provenance::unknown());
        });
        assert_eq!(context.provenance(), initial);
        assert_eq!(cloned.provenance(), initial);
    }

    #[test]
    fn test_partial_evaluation_context_resolve() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let lifted = context.lift(Array::scalar(2.0).unwrap()).unwrap();
        assert!(
            matches!(context.resolve(&lifted), ValueResolution::Constant(value) if value == Array::scalar(2.0).unwrap())
        );
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F64), 0));
        assert!(matches!(context.resolve(&unknown), ValueResolution::Opaque));
        assert!(matches!(unknown.concretize(), Err(ProgramError::Concretization { message })
                if message == "cannot extract a concrete boolean from an unknown partial-evaluation value"),);
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity() {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let parent_reference = outer.input(reference_type.clone());
        let expected = outer.reference_identity(&parent_reference).unwrap();
        let context = PartialEvaluationContext::new(outer);
        let known = PartialEvaluationValue::known(parent_reference);
        let original = PartialTracer::new(context.clone(), known.clone());
        assert_eq!(context.reference_identity(&original), Ok(expected));

        // A residual view of a known feeder must retain its parent's root identity rather than acquire the residual
        // builder's namespace. An unknown reference input does belong to that separate namespace.
        let operation = TestOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0));
        let viewed = context.residualize(operation, Vec::new(), &[known]).unwrap();
        let viewed = PartialTracer::new(context.clone(), viewed[0].clone());
        assert_eq!(context.reference_identity(&viewed), Ok(expected));
        let unknown = PartialTracer::new(context.clone(), context.unknown_input(reference_type, 0));
        assert!(matches!(context.reference_identity(&unknown), Ok(Some(ReferenceIdentity::Staged { .. }))));
        assert_ne!(context.reference_identity(&unknown).unwrap(), expected);

        let value =
            PartialTracer::new(context.clone(), context.unknown_input(ArrayType::scalar(DataType::F32).into(), 1));
        assert_eq!(context.reference_identity(&value), Ok(None));
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity_for_captured_views() {
        let outer = TracingContext::<TestCapture, ReferenceIndexOperation>::new();
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let capture = outer.constant(CaptureReference::new(0, reference_type.clone()));
        let expected = outer.reference_identity(&capture).unwrap();
        let context = PartialEvaluationContext::new(outer.clone());
        let captured = PartialEvaluationValue::known_constant(capture);
        let view = context
            .residualize(ReferenceIndexOperation::new(0, 0), Vec::new(), &[captured.clone()])
            .unwrap()
            .remove(0);
        let original = PartialTracer::new(context.clone(), captured);
        let view = PartialTracer::new(context.clone(), view);
        let parent_atom_count = outer.builder().borrow().atoms().len();
        assert_eq!(context.reference_identity(&original), Ok(expected));
        assert_eq!(context.reference_identity(&view), Ok(expected));
        assert_eq!(context.clone().reference_identity(&view), Ok(expected));
        assert_eq!(outer.builder().borrow().atoms().len(), parent_atom_count);
        let other = context.lift(CaptureReference::new(1, reference_type)).unwrap();
        assert_ne!(context.reference_identity(&other).unwrap(), expected);
        assert_eq!(
            context.reference_identity(&other),
            outer.reference_identity(other.value().unwrap().as_known().unwrap()),
        );
    }

    #[test]
    fn test_partial_evaluation_context_reference_identity_for_unresolved_inherited_capture() {
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let outer = TracingContext::<TestCapture, ReferenceIndexOperation>::new();
        let context = PartialEvaluationContext::new(outer.clone());

        // Inherited constants enter the residual builder without being materialized through the parent context.
        // Seed that state directly: no parent identity has been recorded for this symbolic capture.
        let captured = context.builder.borrow_mut().add_constant(CaptureReference::new(0, reference_type.clone()));
        let captured = PartialTracer::new(context.clone(), PartialEvaluationValue::variable(reference_type, captured));
        assert_eq!(context.reference_identity(&captured), Ok(None));
        assert!(outer.builder().borrow().atoms().is_empty());
    }

    #[test]
    fn test_partial_evaluation_context_zero() {
        let context = PartialEvaluationContext::new(TestArrayContext::new());
        let zero = context.zero(&ArrayType::scalar(DataType::Boolean)).unwrap();
        assert_eq!(zero.value().unwrap().as_known(), Some(&Array::scalar(false).unwrap()));
        assert_eq!(zero.concretize(), Ok(false));
    }

    #[test]
    fn test_region_partition_with_configuration() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let primal = builder.add_input(scalar_type.clone());
        let tangent = builder.add_input(scalar_type);
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, tangent], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![primal, output], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (partition, deferred_instructions) = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, true, Some(&[0]))
            .unwrap();

        assert_eq!(deferred_instructions, HashSet::from([InstructionId::new(program.entry_region_ref().id(), 0)]));
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(partition.known_reference_inputs().collect::<Vec<_>>(), Vec::<usize>::new());
        assert_eq!(
            partition.known_program().interpret(vec![TestValue::Array(Array::scalar(3.0_f32).unwrap())]),
            Ok(vec![TestValue::Array(Array::scalar(3.0_f32).unwrap())]),
        );

        // Each invocation starts from zero, including when an earlier tangent value is used again.
        assert_eq!(
            partition.residual_program().interpret(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
            Ok(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
        );
        assert_eq!(
            partition.residual_program().interpret(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );
        assert_eq!(
            partition.residual_program().interpret(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
            Ok(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
        );
        assert!(
            matches!(program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[1])),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 1 depends on deferred work in a repeated residual computation",),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_preserves_failure_order() {
        let program = reference_ordering_program();
        let partition = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, false, None)
            .unwrap()
            .0;
        let destination = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let source = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(source.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let known = partition.known_program().interpret(vec![TestValue::Reference(destination.clone())]).unwrap();
        let residual_inputs = partition
            .residual_inputs()
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => {
                    assert_eq!(*index, 1);
                    TestValue::Reference(source.clone())
                }
                PartialEvaluationInput::Known(index) => known[*index].clone(),
            })
            .collect();
        let error = partition.residual_program().interpret(residual_inputs).unwrap_err();
        assert_eq!(
            error.downcast_custom::<crate::programs::ReferenceError>(),
            Some(&crate::programs::ReferenceError::Frozen),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(2.0_f32).unwrap()));
    }

    #[test]
    fn test_region_partition_with_configuration_allows_independent_effects_before_residual_failure() {
        let program = reference_ordering_program();
        let partition =
            program.entry_region_ref().partition_with_configuration(&[true, false], true, true, None).unwrap().0;
        let destination = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let source = ArrayReference::new(Array::scalar(1.0_f32).unwrap());
        assert_eq!(source.freeze(), Ok(Array::scalar(1.0_f32).unwrap()));
        let known = partition.known_program().interpret(vec![TestValue::Reference(destination.clone())]).unwrap();
        let residual_inputs = partition
            .residual_inputs()
            .iter()
            .map(|input| match input {
                PartialEvaluationInput::Unknown(index) => {
                    assert_eq!(*index, 1);
                    TestValue::Reference(source.clone())
                }
                PartialEvaluationInput::Known(index) => known[*index].clone(),
            })
            .collect();
        let error = partition.residual_program().interpret(residual_inputs).unwrap_err();
        assert_eq!(
            error.downcast_custom::<crate::programs::ReferenceError>(),
            Some(&crate::programs::ReferenceError::Frozen),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(0.0_f32).unwrap()));
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_ignores_unused_unknown_inputs() {
        let scalar_type: ArrayIrType = ArrayType::scalar(DataType::F32).into();
        let mut branch = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        branch.add_input(scalar_type.clone());
        let one = branch.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        branch
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        let output =
            branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let branch = branch
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let unused = builder.add_input(scalar_type);
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let predicate = builder.add_constant(TestValue::Array(Array::scalar(true).unwrap()));
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let true_branch = builder.import_program(branch.clone());
        let false_branch = builder.import_program(branch);
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, unused],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let (partition, _) =
            program.entry_region_ref().partition_with_configuration(&[false], true, true, Some(&[0])).unwrap();
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0)]);
        assert!(partition.residual_program().instructions().is_empty());
        assert_eq!(
            partition.known_program().interpret(Vec::new()),
            Ok(vec![TestValue::Array(Array::scalar(1.0_f32).unwrap())])
        );
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_tracks_empty_output_effects() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let print = TestOperation::from(ArrayOperation::from(PrintOperation::new("deferred")));
        builder.add_instruction(print, Vec::new(), vec![tangent], None).unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, zero], None)
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();

        // The write has known operands and no outputs. Its deferred execution must still pull the allocation into
        // each residual call, rather than retain a reference created once by the known program.
        let (partition, _) =
            program.entry_region_ref().partition_with_configuration(&[false], true, true, Some(&[])).unwrap();
        assert!(partition.known_program().instructions().is_empty());
        assert!(partition.known_reference_inputs().next().is_none());
        assert_eq!(
            partition.residual_program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = print [label=deferred] %0
                    %2:f32[] = const 0.0
                    %3:ref<f32[]> = reference_new %2
                    reference_write %3 %2
                in ()
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_preserves_cross_class_order() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let source = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let print = TestOperation::from(ArrayOperation::from(PrintOperation::new("before_read")));
        builder.add_instruction(print, Vec::new(), vec![tangent], None).unwrap();
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert!(
            matches!(program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[0])),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 0 depends on deferred work in a repeated residual computation",),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_preserves_reference_constant_order() {
        let mut builder = ProgramBuilder::<TestCapture, ReferenceReadOperation<ArrayType, ArrayIrType>>::new();
        let source = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let captured =
            builder.add_constant(CaptureReference::new(0, ReferenceType::new(ArrayType::scalar(DataType::F32)).into()));
        builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestCapture>, Vec<TestCapture>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        // A captured reference and a symbolic input have different analysis roots, but that alone cannot establish
        // their runtime independence. Keep the captured access behind the earlier deferred read.
        assert!(matches!(
            program.entry_region_ref().partition_with_configuration(&[false], true, true, Some(&[0])),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 0 depends on deferred work in a repeated residual computation",
        ),);
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_rejects_shared_local_state() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let tangent = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let initial =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, tangent], None)
            .unwrap();
        let final_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![initial, final_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        // Recursive discovery preserves the initially known output. Until its ownership is explicit, keeping the
        // allocation outside repeated execution would silently retain updates from earlier calls.
        assert!(matches!(program.entry_region_ref().partition_with_configuration(&[false], true, true, None),
            Err(ProgramError::MalformedProgram(message))
                if message == "local reference allocation contributes to both required known outputs and deferred state",),);
    }

    #[test]
    fn test_region_partition_with_configuration_rejects_invalid_required_output() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32));
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap();
        assert!(matches!(
            program.entry_region_ref().partition_with_configuration(&[true], true, true, Some(&[1])),
            Err(ProgramError::MalformedProgram(message))
                if message == "required known output index 1 is out of bounds",
        ),);
    }

    #[test]
    fn test_region_partition_with_configuration_discovers_allocations_across_retries() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let first = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let second = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        builder
            .add_instruction(ArrayOperation::Print(PrintOperation::new("read")), Vec::new(), vec![read], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![first, zero], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![second, update], None)
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let region = program.entry_region_ref();

        // Initially only the final write is residual, so it defers the second allocation. Replaying then defers its
        // read and print; that print keeps the first allocation's write residual, requiring another discovery pass.
        let (partition, deferred_instructions) =
            region.partition_with_configuration(&[false], true, true, Some(&[])).unwrap();
        assert_eq!(
            deferred_instructions,
            HashSet::from([InstructionId::new(region.id(), 0), InstructionId::new(region.id(), 1)]),
        );
        assert!(partition.known_program().instructions().is_empty());
        assert_eq!(
            partition
                .residual_program()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_new", "reference_read", "print", "reference_new", "reference_write", "reference_write"],
        );
        assert_eq!(
            partition.residual_program().interpret(vec![TestValue::Array(Array::scalar(3.0_f32).unwrap())]),
            Ok(Vec::new()),
        );
        assert_eq!(
            partition.residual_program().interpret(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
            Ok(Vec::new()),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_rejects_required_output_deferred_by_retry() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let source = builder.add_input(ReferenceType::new(ArrayType::scalar(DataType::F32)).into());
        let update = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let zero = builder.add_constant(TestValue::Array(Array::scalar(0.0_f32).unwrap()));
        let local = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        let initial = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![local], None).unwrap()[0];
        builder
            .add_instruction(ArrayOperation::Print(PrintOperation::new("initial")), Vec::new(), vec![initial], None)
            .unwrap();
        let output = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![source], None).unwrap()[0];
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![local, update], None)
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // The required read folds on the first pass. Deferring the local allocation also defers the preceding print,
        // so preserving ordered effects makes the required read residual on the next pass.
        assert!(matches!(
            program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[0])),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 0 depends on deferred work in a repeated residual computation",
        ),);
    }

    #[test]
    fn test_region_partition_with_configuration_allows_shared_read_only_local_state() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let initial = builder.add_constant(TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        let local = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let known = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![local], None).unwrap()[0];
        builder
            .add_instruction(ArrayOperation::Print(PrintOperation::new("input")), Vec::new(), vec![input], None)
            .unwrap();
        let residual =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![local], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![known, residual], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        // The local allocation contributes to a required known output, but residual invocations only read it.
        // It can remain on the known side and cross the boundary without accumulating state between calls.
        let (partition, deferred_instructions) =
            program.entry_region_ref().partition_with_configuration(&[false], true, true, Some(&[0])).unwrap();
        assert!(deferred_instructions.is_empty());
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(partition.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        let known = partition.known_program().interpret(Vec::new()).unwrap();
        assert_eq!(known[0], TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(
            partition.residual_inputs(),
            &[PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Known(0)],
        );
        assert_eq!(
            partition
                .residual_program()
                .interpret(vec![TestValue::Array(Array::scalar(3.0_f32).unwrap()), known[1].clone()]),
            Ok(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
        );
        assert_eq!(
            partition
                .residual_program()
                .interpret(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap()), known[1].clone()]),
            Ok(vec![TestValue::Array(Array::scalar(2.0_f32).unwrap())]),
        );
    }

    #[test]
    fn test_program_partially_evaluate() {
        // `f(a, x) = (a * a, a * a * x + 1, a * a + x)` with `a` known and `x` unknown: the `a * a` subcomputation
        // folds to a known output, its two residual consumers share one residual feeder, and the literal is rebuilt
        // inline as a residual constant instead of becoming a feeder.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let offset = builder.add_instruction(AddOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![squared, shifted, offset], vec![Placeholder; 2], vec![Placeholder; 3])
            .unwrap();
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(9.0).unwrap()),],
        );
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0).unwrap()),
                PartialEvaluationOutput::Unknown(0),
                PartialEvaluationOutput::Unknown(1),
            ],
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
            evaluation.interpret(&EagerContext::<Array, ArrayOperation<Array>>::new(), &[Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(9.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(13.0).unwrap()]),
        );
        assert_eq!(
            program.interpret(vec![Array::scalar(3.0).unwrap(), Array::scalar(4.0).unwrap()]),
            Ok(vec![Array::scalar(9.0).unwrap(), Array::scalar(37.0).unwrap(), Array::scalar(13.0).unwrap()]),
        );

        // All-known inputs fold the whole program away: every output is known and the residual program is empty.
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(Array::scalar(3.0).unwrap()),
                PartialValue::Known(Array::scalar(4.0).unwrap()),
            ])
            .unwrap();
        assert_eq!(evaluation.inputs, Vec::new());
        assert_eq!(
            evaluation.outputs,
            vec![
                PartialEvaluationOutput::Known(Array::scalar(9.0).unwrap()),
                PartialEvaluationOutput::Known(Array::scalar(37.0).unwrap()),
                PartialEvaluationOutput::Known(Array::scalar(13.0).unwrap()),
            ],
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
            ],
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
                PartialValue::Known(Array::scalar(2.0).unwrap()),
                PartialValue::Unknown(ArrayType::scalar(DataType::F64)),
            ])
            .unwrap();
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(Array::scalar(2.0).unwrap()),],
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
            program.partially_evaluate(&[PartialValue::Known(Array::scalar(1.0).unwrap())]),
            Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 }),
        ),);
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
        let one = builder.add_constant(Array::scalar(1.0).unwrap());
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
                PartialValue::Known(Array::scalar(3.0).unwrap()),
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
    fn test_program_partially_evaluate_with_residual_references() {
        // `f(r, x) = { add_update(r, 1); write(r, x); read(r) }` over a live reference `r` and an unknown `x`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        let one = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Specialization never touches live state: every reference operation residualizes even though the reference is
        // known, and the live handle becomes a residual reference threaded by identity.
        let live = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let evaluation = program
            .partially_evaluate(&[
                PartialValue::Known(TestValue::Reference(live.clone())),
                PartialValue::Unknown(scalar_type.clone().into()),
            ])
            .unwrap();
        assert_eq!(live.read(), Ok(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(
            evaluation.inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(TestValue::Reference(live.clone()))],
        );
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]> .
                let %2:f32[] = const 1.0
                    reference_add_update %1 %2
                    reference_write %1 %0
                    %3:f32[] = reference_read %1
                in (%3)
            "}
            .trim_end(),
        );

        // Replaying the residual program binds the live reference by identity: the deferred accesses run against the
        // same allocation, so the read observes the write and the handle reflects it afterwards.
        assert_eq!(
            evaluation.interpret(
                &EagerContext::<TestValue, TestOperation>::new(),
                &[TestValue::Array(Array::scalar(5.0_f32).unwrap())],
            ),
            Ok(vec![TestValue::Array(Array::scalar(5.0_f32).unwrap())]),
        );
        assert_eq!(live.read(), Ok(Array::scalar(5.0_f32).unwrap()));
    }

    #[test]
    fn test_program_partially_evaluate_in_context() {
        // `f(a, x) = (a * a) * x + 1` with `a` known as a live tracer of an enclosing trace and `x` unknown: the known
        // `a * a` folds by staging into the outer program, the residual program consumes its staged result through a
        // known feeder naming the outer atom, and the literal is rebuilt inline as a residual constant.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_constant(Array::scalar(1.0).unwrap());
        let squared = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let scaled = builder.add_instruction(MulOperation::new(), Vec::new(), vec![squared, x], None).unwrap()[0];
        let shifted = builder.add_instruction(AddOperation::new(), Vec::new(), vec![scaled, c], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![shifted], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let outer = TestArrayTracingContext::new();
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
        ),);
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

        // The outer trace accumulated the lifted literal followed by the folded known work. The literal stays dead
        // in the outer trace because the residual program rebuilds it inline.
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
    fn test_program_partially_evaluate_in_context_with_residual_references() {
        // `f(r, x) = { add_update(r, 1); write(r, x); read(r) }` over a live reference `r` and an unknown `x`.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        let one = builder.add_constant(TestValue::Array(Array::scalar(1.0_f32).unwrap()));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, one], None)
            .unwrap();
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, x], None)
            .unwrap();
        let read =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Under a staging known side the known update folds into the outer program, the write of the unknown value
        // stages and prevents later ordered effects from folding, and the read stages over the root as a residual
        // reference. Replaying the residual program in the outer trace threads the outer reference atom by identity,
        // never a snapshot.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let reference = outer.input(reference_type.into());
        let x = outer.input(scalar_type.clone().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[PartialValue::Known(reference), PartialValue::Unknown(scalar_type.into())],
            )
            .unwrap();
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        let outputs = evaluation.interpret(&outer, &[x]).unwrap();
        let outer_program = outer
            .builder()
            .borrow()
            .clone()
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![outputs[0].atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        assert_eq!(
            outer_program.to_string(),
            indoc! {"
                lambda %0:ref<f32[]>, %1:f32[] .
                let %2:f32[] = const 1.0
                    reference_add_update %0 %2
                    reference_write %0 %1
                    %3:f32[] = reference_read %0
                in (%3)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_partially_evaluate_in_context_preserves_order_in_selected_condition_branch() {
        // `f(p, a, b, x) = { if p { write(a, x) } else { write(b, x) }; (read(a), read(b)) }` over two reference roots.
        let scalar_type = ArrayType::scalar(DataType::F32);
        let reference_type = ReferenceType::new(scalar_type.clone());
        let branch = |written: usize| {
            let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
            let references =
                [builder.add_input(reference_type.clone().into()), builder.add_input(reference_type.clone().into())];
            let x = builder.add_input(scalar_type.clone().into());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![references[written], x], None)
                .unwrap();
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let true_branch = builder.import_program(branch(0));
        let false_branch = builder.import_program(branch(1));
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let a = builder.add_input(reference_type.clone().into());
        let b = builder.add_input(reference_type.clone().into());
        let x = builder.add_input(scalar_type.clone().into());
        builder
            .add_instruction(ConditionOperation::new(), vec![true_branch, false_branch], vec![predicate, a, b, x], None)
            .unwrap();
        let read_a = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let read_b = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![b], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read_a, read_b], vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();

        // A known predicate inlines only the selected branch. Its residual write prevents later ordered effects from
        // folding, so both reads stage after it; the unselected branch contributes no write.
        let outer = TracingContext::<TestValue, TestOperation>::new();
        let a = outer.input(reference_type.clone().into());
        let b = outer.input(reference_type.clone().into());
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.constant(TestValue::Array(Array::scalar(true).unwrap()))),
                    PartialValue::Known(a.clone()),
                    PartialValue::Known(b.clone()),
                    PartialValue::Unknown(scalar_type.clone().into()),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![1, 2]);
        assert_eq!(
            evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                let reference_write %1 %0
                    %3:f32[] = reference_read %1
                    %4:f32[] = reference_read %2
                in (%3, %4)
            "}
            .trim_end(),
        );
        let first = ArrayReference::new(Array::scalar(2.0_f32).unwrap());
        let second = ArrayReference::new(Array::scalar(3.0_f32).unwrap());
        assert_eq!(
            evaluation.program.interpret(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Reference(first.clone()),
                TestValue::Reference(second.clone()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(3.0_f32).unwrap())
            ]),
        );
        assert_eq!(first.read(), Ok(Array::scalar(5.0_f32).unwrap()));
        assert_eq!(second.read(), Ok(Array::scalar(3.0_f32).unwrap()));

        // Specializing a false predicate selects the other root before the residual program is built.
        let false_evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Known(outer.constant(TestValue::Array(Array::scalar(false).unwrap()))),
                    PartialValue::Known(a.clone()),
                    PartialValue::Known(b.clone()),
                    PartialValue::Unknown(scalar_type.clone().into()),
                ],
            )
            .unwrap();
        assert_eq!(
            false_evaluation.program.to_string(),
            indoc! {"
                lambda %0:f32[], %1:ref<f32[]>, %2:ref<f32[]> .
                let reference_write %1 %0
                    %3:f32[] = reference_read %2
                    %4:f32[] = reference_read %1
                in (%3, %4)
            "}
            .trim_end(),
        );

        assert!(matches!(
            &false_evaluation.inputs[1],
            PartialEvaluationInput::Known(reference) if reference.atom_id() == b.atom_id(),
        ),);
        assert!(matches!(
            &false_evaluation.inputs[2],
            PartialEvaluationInput::Known(reference) if reference.atom_id() == a.atom_id(),
        ),);

        // An unknown predicate residualizes the whole conditional. Its ordered effects require both later reads
        // to remain residual, regardless of which branch eventually runs.
        let evaluation = program
            .partially_evaluate_in_context(
                &outer,
                &[
                    PartialValue::Unknown(ArrayType::scalar(DataType::Boolean).into()),
                    PartialValue::Known(a),
                    PartialValue::Known(b),
                    PartialValue::Unknown(scalar_type.into()),
                ],
            )
            .unwrap();
        assert_eq!(evaluation.outputs, vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);
        assert_eq!(evaluation.known_reference_inputs().collect::<Vec<_>>(), vec![2, 3]);
        assert_eq!(
            evaluation
                .program
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["condition", "reference_read", "reference_read"],
        );
        // Taking the other branch at runtime must update the second root and preserve the first root.
        assert_eq!(
            evaluation.program.interpret(vec![
                TestValue::Array(Array::scalar(false).unwrap()),
                TestValue::Array(Array::scalar(7.0_f32).unwrap()),
                TestValue::Reference(first.clone()),
                TestValue::Reference(second.clone()),
            ]),
            Ok(vec![
                TestValue::Array(Array::scalar(5.0_f32).unwrap()),
                TestValue::Array(Array::scalar(7.0_f32).unwrap())
            ]),
        );
        assert_eq!(first.read(), Ok(Array::scalar(5.0_f32).unwrap()));
        assert_eq!(second.read(), Ok(Array::scalar(7.0_f32).unwrap()));
    }

    #[test]
    fn test_program_partition() {
        // `f(a, x) = (a + a, -a * x)` partitioned with `a` known and `x` unknown: `a + a` is a fully known output,
        // `-a` is a residual edge trailing it among the known program's outputs, and the residual program computes
        // the mixed output over the surviving unknown input plus the edge.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![negated, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![doubled, product], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let partition = program.partition(&[true, false]).unwrap();
        assert_eq!(partition.known_input_indices, vec![0]);
        assert_eq!(
            partition.residual_inputs,
            vec![PartialEvaluationInput::Unknown(1), PartialEvaluationInput::Known(0),],
        );
        assert_eq!(partition.outputs, vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(
            partition.known_program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                    %2:f64[] = neg %0
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
        let known_outputs = partition.known_program.interpret(vec![Array::scalar(2.0).unwrap()]).unwrap();
        let residual_outputs = partition
            .residual_program
            .interpret(vec![Array::scalar(3.0).unwrap(), known_outputs[1].clone()])
            .unwrap();
        assert_eq!(known_outputs[0], Array::scalar(4.0).unwrap());
        assert_eq!(residual_outputs, vec![Array::scalar(3.0 * (-2.0_f64)).unwrap()]);

        // All-unknown known-ness produces an empty known program and residualizes everything.
        let partition = program.partition(&[false, false]).unwrap();
        assert_eq!(partition.known_input_indices, Vec::<usize>::new());
        assert!(partition.known_program.instructions().is_empty());
        assert!(partition.known_program.output_ids().is_empty());
        assert_eq!(
            partition.residual_inputs,
            vec![PartialEvaluationInput::Unknown(0), PartialEvaluationInput::Unknown(1),],
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
