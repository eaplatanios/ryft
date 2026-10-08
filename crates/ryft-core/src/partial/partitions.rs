use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::rc::Rc;

use crate::contexts::StagingContext;
use crate::macros::check_count;
use crate::parameters::Placeholder;
use crate::partial::contexts::{PartialEvaluationContext, ReferencePlacement};
use crate::partial::operations::PartiallyEvaluatableOperation;
use crate::partial::residuals::{ResidualPlacement, ResidualPolicyReference};
use crate::partial::values::{
    PartialEvaluationInput, PartialEvaluationOutput, PartialEvaluationValue, ResidualInputSource,
};
use crate::programs::{
    EffectClass, EffectsSummary, InstructionId, Operation, OperationFormatter, OperationPayloadProjection, Program,
    ProgramError, ProgramRenderingMode, ReferenceAccessMode, ReferenceRoot, RegionRef, Type, Typed, Value, ValueId,
};
use crate::tracing::TracingContext;

#[cfg(doc)]
use crate::partial::evaluations::PartialEvaluation;

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
#[derive(Clone, Debug, PartialEq)]
pub(super) enum EffectOrdering<K: Eq + Hash> {
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
    pub(super) fn conflicts(&self, other: &Self) -> bool {
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
    pub(super) fn extend(&mut self, other: &Self)
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
    pub(super) fn effect_ordering<K: Eq + Hash, R: IntoIterator<Item = K>>(self, references: R) -> EffectOrdering<K> {
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

/// Boundary wiring and effect-ordering constraints of a [`PartitionedProgram`], retained separately from its two
/// [`Program`]s so that region transform caches can store and reassemble the complete partition.
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionMetadata {
    /// Refer to the documentation of [`PartitionedProgram::original_input_count`] for more information.
    original_input_count: usize,

    /// Refer to the documentation of [`PartitionedProgram::known_input_indices`] for more information.
    known_input_indices: Vec<usize>,

    /// Refer to the documentation of [`PartitionedProgram::residual_inputs`] for more information.
    residual_inputs: Vec<ResidualInputSource>,

    /// Refer to the documentation of [`PartitionedProgram::outputs`] for more information.
    outputs: Vec<PartialEvaluationOutput<usize>>,

    /// Effect-ordering constraints for the known program and residual program, respectively. By default, all ordered
    /// effects must retain their relative execution order. When partitioning work into a known invocation followed by
    /// repeated residual invocations (for example, linearization followed by pushforward calls), reference analysis
    /// can establish separate ordering constraints for independent allocations. References in both programs are then
    /// identified relative to the original inputs so that accesses to the same allocation can be compared across the
    /// partition boundary; allocations created within either program have separate identities.
    effect_ordering: [EffectOrdering<PartitionReferenceRoot>; 2],
}

impl PartitionMetadata {
    /// Returns this [`PartitionMetadata`] with its residual inputs replaced by `residual_inputs`,
    /// keeping its original input count, known inputs, outputs, and effect-ordering constraints.
    /// [`PartitionedProgram::with_residual_policy`] uses this after it rewrites which residual
    /// edges the residual program consumes.
    pub(super) fn with_residual_inputs(mut self, residual_inputs: Vec<ResidualInputSource>) -> Self {
        self.residual_inputs = residual_inputs;
        self
    }
}

/// Result of partitioning a [`Program`] into a known-side program and a residual program based on which original
/// inputs are known. Unlike [`PartialEvaluation`], this representation carries only programs and [`PartitionMetadata`]
/// describing their positional wiring and effect-ordering constraints. It does not retain values from a parent
/// [`Context`]. It is returned by [`Program::partition`] and is typically passed to
/// [`PartialEvaluationContext::inline_partitioned_program`] when recursively
/// transforming an attached region.
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
///   known_inputs -. "forwarded" .-> residual_program
///   known_outputs -. "forwarded" .-> residual_program
///   residual_program --> residual_outputs["Residual Original Outputs"]
///   known_outputs --> descriptors["Output Descriptors"]
///   residual_outputs --> descriptors
///   descriptors --> outputs["Outputs in Original Order"]
/// ```
///
/// The known program receives only the original inputs selected by [`known_input_indices`](Self::known_input_indices).
/// Its outputs place fully known original outputs before residual edge values. Each residual program input names its
/// [`ResidualInputSource`] in [`residual_inputs`](Self::residual_inputs): an original unknown input or a residual edge
/// and, after [`forward_residuals`](Self::forward_residuals), possibly also an original known input or a fully known
/// output that the residual program reads directly (the dotted arrows above). [`outputs`](Self::outputs) records which
/// side supplies each original output.
///
/// # Rendering
///
/// The [`Display`] implementation renders the boundary wiring of a partition like the metadata of an operation,
/// followed by its two programs, each indented beneath its label like the attached regions of an instruction. For
/// example, partitioning `f(x, t) = (sin(dot(x, x)), cos(dot(x, x)) * t)` with `x` known renders as follows:
///
/// ```text
/// partition [
///     known_inputs=[0],
///     residual_inputs=[UnknownInput(1), ResidualEdge(0)],
///     outputs=[Known(0), Unknown(0)],
/// ]
/// known={
///     lambda %0:f64[3] .
///     let %1:f64[] = dot [...] %0 %0
///         %2:f64[] = sin %1
///         %3:f64[] = cos %1
///     in (%2, %3)
/// }
/// residual={
///     lambda %0:f64[], %1:f64[] .
///     let %2:f64[] = mul %1 %0
///     in (%2)
/// }
/// ```
///
/// Here, `residual_inputs` lists the original input `1` followed by the edge `0` (i.e., known output `%3`), and
/// `outputs` takes the first original output from the known program and the second one from the residual program.
/// The wiring renders on one line when it is short enough. The effect-ordering constraints are not rendered, because
/// reference analysis derives them from the two programs, and neither is the
/// [`original_input_count`](Self::original_input_count).
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct PartitionedProgram<V: Value, O: Operation<Type = V::Type>> {
    /// Refer to the documentation of [`known_program`](Self::known_program) for more information.
    known_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Refer to the documentation of [`residual_program`](Self::residual_program) for more information.
    residual_program: Program<V, O, Vec<V>, Vec<V>>,

    /// Boundary wiring and effect-ordering constraints shared by the two programs.
    metadata: PartitionMetadata,
}

impl<V: Value, O: Operation<Type = V::Type>> PartitionedProgram<V, O> {
    /// Assembles a [`PartitionedProgram`] from its two programs and their boundary wiring, which is the inverse of
    /// [`into_parts`](Self::into_parts) apart from the [`original_input_count`](Self::original_input_count) that
    /// [`into_parts`](Self::into_parts) omits. The wiring must follow the layout documented on [`PartitionedProgram`],
    /// and this function validates that it addresses the two programs and the original boundary consistently, so that
    /// every [`PartitionedProgram`] upholds that layout. Types are not checked here, because binding or interpreting
    /// either program validates them. Effect-ordering constraints are derived conservatively: each program with ordered
    /// effects must keep them ordered relative to every ordered effect of the other program.
    ///
    /// # Parameters
    ///
    ///   - `known_program`: Known-side [`Program`], whose inputs are the original inputs listed in
    ///     `known_input_indices` and whose outputs are the fully known original outputs followed by the residual edges.
    ///   - `residual_program`: Residual-side [`Program`], whose inputs are described by `residual_inputs` and whose
    ///     outputs are the original outputs that `outputs` assigns to the residual side.
    ///   - `original_input_count`: Number of inputs of the original (i.e., pre-partitioning) boundary, which callers
    ///     must supply in full and which the input indices in `known_input_indices` and `residual_inputs` address.
    ///   - `known_input_indices`: Index of the original input feeding each input of `known_program`, in order.
    ///   - `residual_inputs`: [`ResidualInputSource`] feeding each input of `residual_program`, in order.
    ///   - `outputs`: Source of each original output, in original output order, where
    ///     [`Known`](PartialEvaluationOutput::Known) indexes the fully known outputs of `known_program` and
    ///     [`Unknown`](PartialEvaluationOutput::Unknown) indexes the outputs of `residual_program`.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] describing the first inconsistency in the wiring (e.g., a known
    /// input index or residual input source that is out of bounds, or a source count that differs from the number of
    /// inputs of `residual_program`).
    pub fn from_parts(
        known_program: Program<V, O, Vec<V>, Vec<V>>,
        residual_program: Program<V, O, Vec<V>, Vec<V>>,
        original_input_count: usize,
        known_input_indices: Vec<usize>,
        residual_inputs: Vec<ResidualInputSource>,
        outputs: Vec<PartialEvaluationOutput<usize>>,
    ) -> Result<Self, ProgramError> {
        let effect_ordering = [&known_program, &residual_program].map(|program| {
            if program.effects().classes().is_ordered() { EffectOrdering::Global } else { EffectOrdering::default() }
        });
        let partition = Self {
            known_program,
            residual_program,
            metadata: PartitionMetadata {
                original_input_count,
                known_input_indices,
                residual_inputs,
                outputs,
                effect_ordering,
            },
        };
        partition.validate_wiring()?;
        Ok(partition)
    }

    /// Reassembles a [`PartitionedProgram`] from the programs and metadata that
    /// [`into_programs_and_metadata`](Self::into_programs_and_metadata) returned, preserving its effect-ordering
    /// constraints (unlike [`from_parts`](Self::from_parts), which recomputes conservative ones). This is how region
    /// transform caches retain partitions.
    pub(crate) fn from_programs_and_metadata(
        known_program: Program<V, O, Vec<V>, Vec<V>>,
        residual_program: Program<V, O, Vec<V>, Vec<V>>,
        metadata: PartitionMetadata,
    ) -> Self {
        // Transformations that rebuild a partition (e.g., residual placement, rounding, forwarding, and cache
        // reassembly) preserve its wiring by construction, so only debug builds repeat the check of `from_parts`.
        let partition = Self { known_program, residual_program, metadata };
        debug_assert_eq!(partition.validate_wiring(), Ok(()));
        partition
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

    /// Returns the number of inputs of the original (i.e., pre-partitioning) boundary, which callers must supply in
    /// full when invoking this [`PartitionedProgram`]. The [`known_input_indices`](Self::known_input_indices) and the
    /// input sources among the [`residual_inputs`](Self::residual_inputs) address this boundary, but they need not
    /// mention every position. For example, [`forward_residuals`](Self::forward_residuals) stops passing a known input
    /// to the known program when only the residual program reads it, or when no program reads it at all.
    #[inline]
    pub fn original_input_count(&self) -> usize {
        self.metadata.original_input_count
    }

    /// Returns the indices of the original program inputs feeding the known-side [`Program`]
    /// (i.e., [`known_program`](Self::known_program)), in order.
    #[inline]
    pub fn known_input_indices(&self) -> &[usize] {
        &self.metadata.known_input_indices
    }

    /// Returns the [`ResidualInputSource`] feeding each residual [`Program`] (i.e.,
    /// [`residual_program`](Self::residual_program)) input, in residual program input order. This is the callee's
    /// [`PartialEvaluation::inputs`] with each feeder _value_ erased to a position/index: an unknown feeder becomes an
    /// [`UnknownInput`](ResidualInputSource::UnknownInput) naming its original boundary input, and each known feeder
    /// becomes a [`ResidualEdge`](ResidualInputSource::ResidualEdge) naming its residual edge ordinal which is also,
    /// offset by the fully known output count, the position of the edge among the known-side operation's outputs.
    /// [`forward_residuals`](Self::forward_residuals) can replace edges with
    /// [`KnownInput`](ResidualInputSource::KnownInput) and [`KnownOutput`](ResidualInputSource::KnownOutput) sources.
    #[inline]
    pub fn residual_inputs(&self) -> &[ResidualInputSource] {
        &self.metadata.residual_inputs
    }

    /// Returns the source of each original (i.e., pre-partitioning) [`Program`] output, in original output order.
    /// This is the callee's [`PartialEvaluation::outputs`] with each folded *value* erased to a position/index:
    /// [`Known`](PartialEvaluationOutput::Known) entries carry the output's position among the known-side operation's
    /// outputs, and [`Unknown`](PartialEvaluationOutput::Unknown) entries keep their ordinal among the residual
    /// program's outputs.
    #[inline]
    pub fn outputs(&self) -> &[PartialEvaluationOutput<usize>] {
        &self.metadata.outputs
    }

    /// Returns the residual input positions that receive known reference values, whether through residual edges of
    /// the known program or forwarded from original known inputs or fully known outputs, in residual input order.
    /// Unknown reference inputs, known non-reference inputs, and inline constants are excluded. These positions
    /// describe boundary wiring, not whether the two programs access overlapping reference allocations.
    #[inline]
    pub fn known_reference_inputs(&self) -> impl '_ + Iterator<Item = usize> {
        self.metadata.residual_inputs.iter().zip(self.residual_program.inputs()).enumerate().filter_map(
            |(index, (source, atom))| {
                (!matches!(source, ResidualInputSource::UnknownInput(_)) && atom.r#type().is_reference())
                    .then_some(index)
            },
        )
    }

    /// Returns whether effects in the known program must stay ordered relative to effects in the residual program.
    /// This compares the ordering constraints recorded when the partition was built. Constraints on the same reference
    /// allocation conflict, and global ordering conflicts with any nonempty constraint in the other program.
    ///
    /// For example, splitting a loop into one loop that runs all known work followed by another that runs all residual
    /// work changes the order of work across iterations. If the known work reads a reference that the residual work
    /// writes, the split could make the next iteration's read run before the preceding iteration's write. This
    /// function returns `true` for that conflict, allowing the caller to keep the original loop intact. Split rules of
    /// custom region-carrying operations that run the two halves of a partition as separate invocations (e.g., a custom
    /// loop operation, like the split rule of `scan`) must perform the same check before splitting.
    ///
    /// The precision of the result depends on how the partition was constructed:
    ///
    ///   - [`Program::partition`], [`Program::partition_with_residual_policy`], and [`from_parts`](Self::from_parts)
    ///     record global ordering for every ordered effect, so the result is `true` whenever both programs have
    ///     ordered effects.
    ///   - The partitions that drivers return for repeated residual invocations (i.e.,
    ///     [`PartialEvaluationDriver::partition_program`](crate::PartialEvaluationDriver::partition_program) and
    ///     [`DifferentiationDriver::partition_jvp_program`](crate::DifferentiationDriver::partition_jvp_program)) can
    ///     record ordering per reference allocation, established by reference analysis, so the result can be `false`
    ///     when the two programs only access independent allocations.
    ///   - When [`forward_residuals`](Self::forward_residuals) rebuilds the known program, the ordering of the known
    ///     program is reset to global ordering, so the result can only become more conservative.
    ///
    /// A `false` result means only that the recorded effect constraints do not prevent such a split. The caller must
    /// still check value dependencies, loop-carried values, shapes, and how intermediate values are stored. The query
    /// does not inspect runtime reference identities; it relies on the assumptions used to construct the partition.
    #[inline]
    pub fn has_effect_ordering_conflicts(&self) -> bool {
        self.metadata.effect_ordering[0].conflicts(&self.metadata.effect_ordering[1])
    }

    /// Returns whether any [`residual_inputs`](Self::residual_inputs) entry is a
    /// [`KnownInput`](ResidualInputSource::KnownInput) or a [`KnownOutput`](ResidualInputSource::KnownOutput) source
    /// (i.e., whether [`forward_residuals`](Self::forward_residuals) forwarded any residual edge). Consumers whose
    /// algorithms require every known feeder to be a residual edge (e.g., residual placement and the split rules of
    /// `condition` and `scan`) use this to reject forwarded wiring with their own diagnostics.
    #[inline]
    pub fn has_forwarded_residual_inputs(&self) -> bool {
        self.metadata
            .residual_inputs
            .iter()
            .any(|source| matches!(source, ResidualInputSource::KnownInput(_) | ResidualInputSource::KnownOutput(_)))
    }

    // TODO(eaplatanios): Review this.
    /// Consumes this [`PartitionedProgram`] and returns an equivalent partition in which the residual program reads
    /// the values that its residual edges merely repeat directly from where the caller of the partition already has
    /// them: the original known inputs and the fully known outputs. The residual program itself is unchanged; only its
    /// [`residual_inputs`](Self::residual_inputs) and the known program change.
    ///
    /// # Notation
    ///
    /// Write the known program as `K` and the residual program as `R`. In the outputs of `K`, `‖` separates the fully
    /// known original outputs `y` from the residual edges `e`, and `p ← s` states that input `p` of `R` reads its value
    /// from the [`ResidualInputSource`] `s`:
    ///
    /// ```text
    ///     K(k₀, …, kₘ)  = (y₀, …, yₙ ‖ e₀, …, eₗ)
    ///     R(r₀, …, rₜ),   rⱼ ← UnknownInput(i) | KnownInput(i) | KnownOutput(i) | ResidualEdge(i)
    /// ```
    ///
    /// Here, `UnknownInput(i)` and `KnownInput(i)` name original input `i`, `KnownOutput(i)` names `yᵢ`, and
    /// `ResidualEdge(i)` names `eᵢ` (i.e., output `n + 1 + i` of `K`). [`Program::partition`] produces only
    /// `UnknownInput` and `ResidualEdge` sources, with one edge per known value that `R` reads.
    ///
    /// # Forwarding
    ///
    /// Each `ResidualEdge` source is classified by the atom that `K` returns for it:
    ///
    ///   - an edge that returns an input of `K` becomes a [`KnownInput`](ResidualInputSource::KnownInput) naming the
    ///     corresponding original input (the analogue of the input forwarding of JAX's
    ///     `trace_to_subjaxpr_nounits_fwd2`),
    ///   - an edge that returns the same atom as a fully known output becomes a
    ///     [`KnownOutput`](ResidualInputSource::KnownOutput) naming the first such output (the analogue of its output
    ///     forwarding), and
    ///   - every other edge stays a [`ResidualEdge`](ResidualInputSource::ResidualEdge), renumbered in order of first
    ///     use, with edges that return the same atom sharing one edge.
    ///
    /// An edge that is both an input and a fully known output of `K` is fed by the input, which does not depend on `K`
    /// at all. `K` then keeps every fully known output slot (including repeated ones), stops returning the edges that
    /// are no longer read, and stops receiving the inputs that it no longer uses. Its
    /// [`known_input_indices`](Self::known_input_indices) can therefore shrink, while
    /// [`original_input_count`](Self::original_input_count) is unchanged, so that callers still supply the complete
    /// original boundary. Existing `UnknownInput`, `KnownInput`, and `KnownOutput` sources are kept unchanged, which
    /// makes forwarding idempotent.
    ///
    /// # Example
    ///
    /// Partitioning `f(a, b, c, x) = (a + a, (a + a) · x, b · x, -c · x)` with `x` unknown produces one fully known
    /// output, `a + a`, and three residual edges. The first edge repeats that output, the second repeats the known
    /// input `b`, and only the third, `-c`, is computed for the residual program alone:
    ///
    /// ```text
    ///     R(x, p, q, r) = (p · x, q · x, r · x)
    ///
    ///     before:  K(a, b, c) = (a + a ‖ a + a, b, -c)
    ///              x ← UnknownInput(3)   p ← ResidualEdge(0)   q ← ResidualEdge(1)   r ← ResidualEdge(2)
    ///
    ///     after:   K(a, c)    = (a + a ‖ -c)
    ///              x ← UnknownInput(3)   p ← KnownOutput(0)   q ← KnownInput(1)   r ← ResidualEdge(0)
    /// ```
    ///
    /// After forwarding, `K` returns `a + a` once and no longer receives `b`, whose value the caller passes to `R`
    /// directly. When no edge is forwarded or deduplicated, the partition is returned unchanged. Otherwise, the
    /// effect-ordering constraints of the rebuilt `K` are conservatively reset to global ordering of its ordered
    /// effects, because rebuilding renumbers the instructions that identify its reference allocations. The
    /// constraints of `R` stay valid, because `R` is unchanged.
    ///
    /// # Usage
    ///
    /// Forwarding is a separate step, rather than part of [`Program::partition`], because an edge and the value it
    /// repeats are interchangeable only at the boundary of a single invocation. Residual placement (refer to
    /// [`with_residual_policy`](Self::with_residual_policy)), precision rounding, and the split rules of `condition`
    /// and `scan` rely on the edges forming a contiguous suffix of the outputs of `K`, and they reject forwarded
    /// partitions. In a `scan` body, for example, an edge that repeats a body input carries the value of one
    /// iteration rather than the input of the enclosing `scan`. Boundary operations whose two halves become separate
    /// operations (e.g., through [`PartialEvaluationContext::inline_partitioned_program`]) and rematerialized calls
    /// forward residuals after placing and rounding them, so that their known half does not return the same value
    /// twice and does not route its own inputs back out.
    ///
    /// # Errors
    ///
    /// Returns a [`ProgramError`] when the known program cannot be restricted to the remaining outputs.
    pub fn forward_residuals(self) -> Result<Self, ProgramError>
    where
        O: Clone,
    {
        let Self { known_program, residual_program, metadata } = self;
        let known_output_count = metadata.outputs.iter().filter(|output| output.is_known()).count();
        let known_input_ids = known_program.input_ids();
        let known_output_ids = known_program.output_ids();

        // Atom IDs index the known arena, so one source table handles forwarding and edge deduplication without
        // repeatedly scanning the boundary. Original inputs take precedence, then the first fully known output.
        let mut sources = vec![None; known_program.atoms().len()];
        for (&atom, &index) in known_input_ids.iter().zip(&metadata.known_input_indices) {
            sources[atom.index()] = Some(ResidualInputSource::KnownInput(index));
        }
        for (index, atom) in known_output_ids[..known_output_count].iter().enumerate() {
            sources[atom.index()].get_or_insert(ResidualInputSource::KnownOutput(index));
        }

        let mut kept_output_ids = known_output_ids[..known_output_count].to_vec();
        let residual_inputs = metadata
            .residual_inputs
            .iter()
            .map(|&source| match source {
                ResidualInputSource::ResidualEdge(edge) => {
                    let atom = known_output_ids[known_output_count + edge];
                    *sources[atom.index()].get_or_insert_with(|| {
                        let edge = kept_output_ids.len() - known_output_count;
                        kept_output_ids.push(atom);
                        ResidualInputSource::ResidualEdge(edge)
                    })
                }
                source => source,
            })
            .collect::<Vec<_>>();

        if kept_output_ids == known_output_ids {
            let metadata = metadata.with_residual_inputs(residual_inputs);
            return Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata));
        }

        let (known_program, live_input_positions) = known_program.filtered(known_input_ids, &kept_output_ids, &[])?;
        let known_input_indices =
            live_input_positions.into_iter().map(|position| metadata.known_input_indices[position]).collect();

        // Without resolved reference roots, the ordering derivation falls back to global ordering for ordered effects.
        // The residual program is unchanged, so its constraints remain valid.
        let [_, residual_effect_ordering] = metadata.effect_ordering;
        let effect_ordering = [known_program.effects().effect_ordering(std::iter::empty()), residual_effect_ordering];
        let metadata = PartitionMetadata {
            original_input_count: metadata.original_input_count,
            known_input_indices,
            residual_inputs,
            outputs: metadata.outputs,
            effect_ordering,
        };
        Ok(Self::from_programs_and_metadata(known_program, residual_program, metadata))
    }

    /// Consumes this [`PartitionedProgram`] and returns its [`known_program`](Self::known_program),
    /// [`residual_program`](Self::residual_program), [`known_input_indices`](Self::known_input_indices),
    /// [`residual_inputs`](Self::residual_inputs), and [`outputs`](Self::outputs), in that order. The result omits the
    /// [`original_input_count`](Self::original_input_count) and the effect-ordering constraints, so it describes how
    /// to invoke the two programs but cannot be reassembled into an equivalent [`PartitionedProgram`].
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn into_parts(
        self,
    ) -> (
        Program<V, O, Vec<V>, Vec<V>>,
        Program<V, O, Vec<V>, Vec<V>>,
        Vec<usize>,
        Vec<ResidualInputSource>,
        Vec<PartialEvaluationOutput<usize>>,
    ) {
        (
            self.known_program,
            self.residual_program,
            self.metadata.known_input_indices,
            self.metadata.residual_inputs,
            self.metadata.outputs,
        )
    }

    /// Consumes this [`PartitionedProgram`] and returns its [`known_program`](Self::known_program),
    /// its [`residual_program`](Self::residual_program), and its [`PartitionMetadata`], in that order.
    pub(crate) fn into_programs_and_metadata(
        self,
    ) -> (Program<V, O, Vec<V>, Vec<V>>, Program<V, O, Vec<V>, Vec<V>>, PartitionMetadata) {
        (self.known_program, self.residual_program, self.metadata)
    }

    /// Validates that the boundary wiring of this [`PartitionedProgram`] addresses its two programs and its original
    /// boundary consistently: the known input indices match the known program inputs, each residual program input has
    /// exactly one source, input sources address the [`original_input_count`](Self::original_input_count) inputs of
    /// the original boundary, [`KnownOutput`](ResidualInputSource::KnownOutput) sources and known output descriptors
    /// address the fully known outputs of the known program, [`ResidualEdge`](ResidualInputSource::ResidualEdge)
    /// sources address its residual edges, and unknown output descriptors address the residual program outputs. Types
    /// are not checked, because binding or interpreting either program validates them. [`from_parts`](Self::from_parts)
    /// calls this, so that every [`PartitionedProgram`] upholds these invariants.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] describing the first inconsistency that it finds.
    fn validate_wiring(&self) -> Result<(), ProgramError> {
        let original_input_count = self.metadata.original_input_count;
        let known_input_count = self.known_program.input_ids().len();
        if self.metadata.known_input_indices.len() != known_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "partition lists {} known input indices but its known program has {} inputs",
                self.metadata.known_input_indices.len(),
                known_input_count,
            )));
        }

        if let Some(index) = self.metadata.known_input_indices.iter().find(|&&index| index >= original_input_count) {
            return Err(ProgramError::MalformedProgram(format!(
                "partition known input index {index} is out of bounds for {original_input_count} original inputs",
            )));
        }

        let residual_input_count = self.residual_program.input_ids().len();
        if self.metadata.residual_inputs.len() != residual_input_count {
            return Err(ProgramError::MalformedProgram(format!(
                "partition lists {} residual input sources but its residual program has {} inputs",
                self.metadata.residual_inputs.len(),
                residual_input_count,
            )));
        }

        let known_output_count = self.metadata.outputs.iter().filter(|output| output.is_known()).count();
        let edge_count = self.known_program.output_ids().len().checked_sub(known_output_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "partition declares {} known outputs but its known program has {} outputs",
                known_output_count,
                self.known_program.output_ids().len(),
            ))
        })?;

        for (position, source) in self.metadata.residual_inputs.iter().enumerate() {
            let (index, count) = match *source {
                ResidualInputSource::UnknownInput(index) | ResidualInputSource::KnownInput(index) => {
                    (index, original_input_count)
                }
                ResidualInputSource::KnownOutput(index) => (index, known_output_count),
                ResidualInputSource::ResidualEdge(index) => (index, edge_count),
            };
            if index >= count {
                return Err(ProgramError::MalformedProgram(format!(
                    "partition residual input {position} names `{source:?}` outside of its index space of size {count}",
                )));
            }
        }

        let residual_output_count = self.residual_program.output_ids().len();
        for (position, output) in self.metadata.outputs.iter().enumerate() {
            let (index, count) = match *output {
                PartialEvaluationOutput::Known(index) => (index, known_output_count),
                PartialEvaluationOutput::Unknown(index) => (index, residual_output_count),
            };
            if index >= count {
                return Err(ProgramError::MalformedProgram(format!(
                    "partition output {position} names `{output:?}` outside of its index space of size {count}",
                )));
            }
        }

        Ok(())
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Display for PartitionedProgram<V, O> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        OperationFormatter::new(formatter, 0, "partition")?.bracketed(|partition| {
            partition.list("known_inputs", self.known_input_indices())?;
            partition.list("residual_inputs", self.residual_inputs().iter().map(|input| format!("{input:?}")))?;
            partition.list("outputs", self.outputs().iter().map(|output| format!("{output:?}")))
        })?;

        // The programs render at the top level rather than as program-valued fields, which would nest them deeper.
        write!(formatter, "\nknown={{\n")?;
        self.known_program.render(formatter, 4, ProgramRenderingMode::Semantic)?;
        write!(formatter, "\n}}\nresidual={{\n")?;
        self.residual_program.render(formatter, 4, ProgramRenderingMode::Semantic)?;
        write!(formatter, "\n}}")
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionRef<'_, V, O> {
    /// Partitions this borrowed [`Region`](crate::Region) based on per-input known-ness without first detaching its
    /// source computation. Refer to the documentation of [`Program::partition`] for more information.
    #[inline]
    pub fn partition(self, input_known: &[bool]) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        O: PartiallyEvaluatableOperation<TracingContext<V, O>>,
    {
        self.partition_with_configuration(input_known, true, false, None, None)
            .map(|(partition, _)| partition)
    }

    /// Partitions this borrowed [`Region`](crate::Region) based on per-input known-ness while placing its residuals
    /// according to `policy`, without first detaching its source computation. Refer to the documentation of
    /// [`Program::partition_with_residual_policy`] for more information.
    #[inline]
    pub fn partition_with_residual_policy(
        self,
        input_known: &[bool],
        policy: &ResidualPolicyReference<V::Type>,
    ) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        V::Type: 'static,
        O: PartiallyEvaluatableOperation<TracingContext<V, O>> + OperationPayloadProjection,
    {
        let residual_placement: Rc<dyn ResidualPlacement<V, O>> = Rc::new(policy.clone());
        self.partition_with_configuration(input_known, true, false, None, Some(residual_placement))
            .map(|(partition, _)| partition)
    }

    /// Partitions this region through fresh staging contexts, returning the known and residual programs together
    /// with source instructions that recursive replay must explicitly defer. No reference effects execute during
    /// construction. Ordinary partitioning builds once and preserves ordering across all ordered effects. When a
    /// residual placement is provided, it places the residuals of the resulting partition, and the partial evaluation
    /// contexts that construct the partition carry it too, so that the split rules of region-carrying operations in
    /// this region place the residuals of the partitions of their bodies with it as well.
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
    ///   - `residual_placement`: Placement of the residuals of the partition and of the partitions nested within it,
    ///     if any. When absent, the partition keeps the residuals that partitioning chose.
    pub(crate) fn partition_with_configuration(
        self,
        input_known: &[bool],
        allow_effect_folding: bool,
        repeated_residual: bool,
        required_known_outputs: Option<&[usize]>,
        residual_placement: Option<Rc<dyn ResidualPlacement<V, O>>>,
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

        let place_residuals = |partition: PartitionedProgram<V, O>| match &residual_placement {
            Some(residual_placement) => residual_placement.place_residuals(partition),
            None => Ok(partition),
        };

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
                .with_allow_effect_folding(allow_effect_folding)
                .with_residual_placement(residual_placement.clone());
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
                .scan(0, |edge_count, input| {
                    Some(match input {
                        PartialEvaluationInput::Unknown(index) => ResidualInputSource::UnknownInput(*index),
                        PartialEvaluationInput::Known(_) => {
                            let edge = *edge_count;
                            *edge_count += 1;
                            ResidualInputSource::ResidualEdge(edge)
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
                input_known.len(),
                known_input_indices,
                residual_inputs,
                outputs,
            )?;

            // Ordinary partitioning is complete after this first pass. Only repeated residual calls need reference
            // analysis to refine ordering and discover allocations that must be deferred before rebuilding.
            let Some(analysis) = reference_analysis.as_deref() else {
                return Ok((place_residuals(partition)?, deferred_instructions));
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
                                Some(PartitionReferenceRoot::Input(partition.metadata.known_input_indices[input_index]))
                            } else if let ResidualInputSource::UnknownInput(original) =
                                partition.metadata.residual_inputs[input_index]
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
                    partition.metadata.effect_ordering[index] = program.effects().effect_ordering(roots);
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
                return Ok((place_residuals(partition)?, deferred_instructions));
            }
        }
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Program<V, O, Vec<V>, Vec<V>> {
    /// Partitions this [`Program`] based on the provided per-input known-ness into a known-side program and a
    /// residual program joined by residual edges, packaged as a [`PartitionedProgram`]. This function invokes
    /// [`partially_evaluate_in_context`](Self::partially_evaluate_in_context) with a **fresh** [`TracingContext`]
    /// whose inputs stand in for the known program inputs and so, instead of folding the known work into a
    /// caller-supplied context, the fresh trace reifies it as the known-side program. The same per-[`Operation`]
    /// rules drive both entry points, and they differ only in what happens to the known side.
    ///
    /// The known program executes before the residual program under the global effect ordering contract of
    /// [`PartialEvaluationContext`]. Known instruction inputs do not permit moving later effects ahead of a deferred
    /// effect, even when their reference roots differ. This function constructs programs without executing effects.
    /// Therefore, callers must preserve the resulting invocation order and the lifetime of any reference-valued
    /// residuals.
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

    /// Partitions this [`Program`] like [`partition`](Self::partition) while placing the known values that its residual
    /// program consumes according to `policy` (refer to [`PartitionedProgram::with_residual_policy`] for how a policy
    /// places them). Unlike placing the residuals of the result of [`partition`](Self::partition), the policy also
    /// places the residuals of the partitions that the split rules of region-carrying operations construct for their
    /// bodies (refer to [`PartialEvaluationContext::with_residual_policy`]), so that, for example, a policy that saves
    /// only the outputs of dot products saves the per-iteration dot products of a `scan` body and recomputes the rest
    /// of the body in the residual `scan`.
    ///
    /// # Parameters
    ///
    ///   - `input_known`: Known-ness of each program input, in input order. The length of this slice must match
    ///     the number of inputs of this [`Program`].
    ///   - `policy`: Policy that places the residuals of the partition and of the partitions nested within it.
    #[inline]
    pub fn partition_with_residual_policy(
        &self,
        input_known: &[bool],
        policy: &ResidualPolicyReference<V::Type>,
    ) -> Result<PartitionedProgram<V, O>, ProgramError>
    where
        V::Type: 'static,
        O: PartiallyEvaluatableOperation<TracingContext<V, O>> + OperationPayloadProjection,
    {
        self.entry_region_ref().partition_with_residual_policy(input_known, policy)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrType, ArrayOperation, ArrayReference, ArrayReferenceTransform, ArrayType, DataType,
    };
    use crate::captures::CaptureReference;
    use crate::differentiation::NothingSavable;
    use crate::operations::{
        AddOperation, ConditionOperation, CosOperation, DotDimensionNumbers, DotOperation, MulOperation, NegOperation,
        PrintOperation, ReferenceAddUpdateOperation, ReferenceNewOperation, ReferenceReadOperation,
        ReferenceWriteOperation, ScanOperation, SinOperation, SubOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::residuals::{
        NoStorage, ResidualCandidate, ResidualDecision, ResidualPolicy, ResidualRejection,
    };
    use crate::partial::tests::{TestCapture, TestOperation, TestValue, reference_ordering_program};
    use crate::partial::values::PartialEvaluationOutput;
    use crate::programs::{
        EffectClass, EffectClasses, Effects, EffectsSummary, InstructionId, Operation, ProgramBuilder, ProgramError,
        ReferenceType,
    };
    use crate::tests::{TestArrayIrOperation, TestArrayOperation};

    use super::*;

    /// Returns a residual policy over [`ArrayType`] that saves the residuals produced by dot products and recomputes
    /// every other residual.
    fn save_dots() -> ResidualPolicyReference<ArrayType> {
        struct SaveDots;

        impl ResidualPolicy<ArrayType> for SaveDots {
            type Storage = NoStorage;

            fn name(&self) -> &str {
                "save_dots"
            }

            fn classify(
                &self,
                candidate: &ResidualCandidate<'_, ArrayType>,
            ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
                Ok(match candidate.producers().iter().any(|producer| producer.payload::<DotOperation>().is_some()) {
                    true => ResidualDecision::Save,
                    false => ResidualDecision::Recompute,
                })
            }
        }

        ResidualPolicyReference::new(SaveDots)
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
        let read = ReferenceReadOperation::<ArrayType, ArrayIrType, ArrayReferenceTransform>::new().effects().summary();
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
    fn test_partitioned_program_from_parts() {
        // The known program returns `-a` as a fully known output followed by `-a` as a residual edge, and the residual
        // program computes `x - e` from the unknown input `x` and the edge `e`.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let known = builder
            .build::<Vec<Array>, Vec<Array>>(vec![negated, negated], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let edge = builder.add_input(ArrayType::scalar(DataType::F64));
        let difference = builder.add_instruction(SubOperation::new(), Vec::new(), vec![x, edge], None).unwrap()[0];
        let residual = builder
            .build::<Vec<Array>, Vec<Array>>(vec![difference], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let from_parts = |known_input_indices: Vec<usize>,
                          residual_inputs: Vec<ResidualInputSource>,
                          outputs: Vec<PartialEvaluationOutput<usize>>| {
            PartitionedProgram::from_parts(
                known.clone(),
                residual.clone(),
                2,
                known_input_indices,
                residual_inputs,
                outputs,
            )
            .map(|_| ())
        };
        let error = |message: &str| Err(ProgramError::MalformedProgram(message.to_string()));
        let outputs = || vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)];
        let sources = |source| vec![ResidualInputSource::UnknownInput(1), source];

        assert_eq!(from_parts(vec![0], sources(ResidualInputSource::ResidualEdge(0)), outputs()), Ok(()));
        assert_eq!(from_parts(vec![0], sources(ResidualInputSource::KnownOutput(0)), outputs()), Ok(()));
        assert_eq!(from_parts(vec![0], sources(ResidualInputSource::KnownInput(0)), outputs()), Ok(()));
        assert_eq!(
            from_parts(vec![0, 1], sources(ResidualInputSource::ResidualEdge(0)), outputs()),
            error("partition lists 2 known input indices but its known program has 1 inputs"),
        );
        assert_eq!(
            from_parts(vec![2], sources(ResidualInputSource::ResidualEdge(0)), outputs()),
            error("partition known input index 2 is out of bounds for 2 original inputs"),
        );
        assert_eq!(
            from_parts(vec![0], vec![ResidualInputSource::UnknownInput(1)], outputs()),
            error("partition lists 1 residual input sources but its residual program has 2 inputs"),
        );
        assert_eq!(
            from_parts(
                vec![0],
                sources(ResidualInputSource::ResidualEdge(0)),
                vec![
                    PartialEvaluationOutput::Known(0),
                    PartialEvaluationOutput::Known(1),
                    PartialEvaluationOutput::Known(2),
                ],
            ),
            error("partition declares 3 known outputs but its known program has 2 outputs"),
        );
        assert_eq!(
            from_parts(vec![0], sources(ResidualInputSource::UnknownInput(2)), outputs()),
            error("partition residual input 1 names `UnknownInput(2)` outside of its index space of size 2"),
        );
        assert_eq!(
            from_parts(vec![0], sources(ResidualInputSource::KnownOutput(1)), outputs()),
            error("partition residual input 1 names `KnownOutput(1)` outside of its index space of size 1"),
        );
        assert_eq!(
            from_parts(vec![0], sources(ResidualInputSource::ResidualEdge(1)), outputs()),
            error("partition residual input 1 names `ResidualEdge(1)` outside of its index space of size 1"),
        );
        assert_eq!(
            from_parts(
                vec![0],
                sources(ResidualInputSource::ResidualEdge(0)),
                vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(1)],
            ),
            error("partition output 1 names `Unknown(1)` outside of its index space of size 1"),
        );
    }

    #[test]
    fn test_partitioned_program_original_input_count() {
        // `f(a, x, b, u) = (-a * x, b * x)` partitioned with `x` unknown: forwarding feeds `b` to the residual program
        // directly and the known program uses neither `b` nor `u`, yet the original boundary still has four inputs.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let b = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let negated_product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![negated, x], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![b, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![negated_product, product], vec![Placeholder; 4], vec![Placeholder; 2])
            .unwrap();
        let partition = program.partition(&[true, false, true, true]).unwrap();
        assert_eq!(partition.original_input_count(), 4);
        assert_eq!(partition.known_input_indices(), &[0, 2, 3]);
        let forwarded = partition.forward_residuals().unwrap();
        assert_eq!(forwarded.original_input_count(), 4);
        assert_eq!(forwarded.known_input_indices(), &[0]);
        assert_eq!(
            forwarded.residual_inputs(),
            &[
                ResidualInputSource::UnknownInput(1),
                ResidualInputSource::ResidualEdge(0),
                ResidualInputSource::KnownInput(2)
            ],
        );

        // Cache reassembly preserves the original input count.
        let (known, residual, metadata) = forwarded.into_programs_and_metadata();
        assert_eq!(PartitionedProgram::from_programs_and_metadata(known, residual, metadata).original_input_count(), 4);
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

        // The unknown value occupies residual input 0 and the known reference enters at residual input 1, either
        // through a residual edge or, once forwarded, directly as the original known input.
        assert_eq!(program.partition(&[false, true]).unwrap().known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        let forwarded = program.partition(&[false, true]).unwrap().forward_residuals().unwrap();
        assert_eq!(
            forwarded.residual_inputs(),
            &[ResidualInputSource::UnknownInput(0), ResidualInputSource::KnownInput(1)],
        );
        assert_eq!(forwarded.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
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
        let mut builder = ProgramBuilder::<
            TestCapture,
            ReferenceWriteOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
        >::new();
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
            .partition_with_configuration(&[true, false], true, true, Some(&[0]), None)
            .unwrap();
        assert!(!repeated.has_effect_ordering_conflicts());
        assert!(!program.partition(&[false, false]).unwrap().has_effect_ordering_conflicts());

        // Keeping the metadata preserves the evidence that the two programs access independent references.
        let (known, residual, metadata) = repeated.into_programs_and_metadata();
        let repeated = PartitionedProgram::from_programs_and_metadata(known, residual, metadata);
        assert!(!repeated.has_effect_ordering_conflicts());
        assert_eq!(repeated.known_input_indices(), &[0]);
        assert_eq!(repeated.residual_inputs(), &[ResidualInputSource::UnknownInput(1)]);
        assert_eq!(repeated.outputs(), &[PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);

        // Reconstructing through the unqualified boundary constructor carries no repeated-invocation evidence.
        let (known, residual, input_indices, residual_inputs, outputs) = repeated.into_parts();
        let reconstructed =
            PartitionedProgram::from_parts(known, residual, 2, input_indices, residual_inputs, outputs).unwrap();
        assert!(reconstructed.has_effect_ordering_conflicts());
    }

    #[test]
    fn test_partitioned_program_has_forwarded_residual_inputs() {
        // `f(a, x) = (-a, -a * x, a * x)` partitioned with `x` unknown has residual edges that repeat a known output
        // and a known input until they are forwarded.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let negated_product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![negated, x], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![a, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![negated, negated_product, product],
                vec![Placeholder; 2],
                vec![Placeholder; 3],
            )
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        assert!(!partition.has_forwarded_residual_inputs());
        let forwarded = partition.forward_residuals().unwrap();
        assert_eq!(
            forwarded.residual_inputs(),
            &[
                ResidualInputSource::UnknownInput(1),
                ResidualInputSource::KnownOutput(0),
                ResidualInputSource::KnownInput(0),
            ],
        );
        assert!(forwarded.has_forwarded_residual_inputs());
        assert!(
            !program
                .partition(&[false, false])
                .unwrap()
                .forward_residuals()
                .unwrap()
                .has_forwarded_residual_inputs()
        );
    }

    #[test]
    fn test_partitioned_program_forward_residuals() {
        // `f(a, b, c, x) = (a + a, (a + a) * x, b * x, -c * x)` partitioned with `x` unknown: the residual edges are
        // the known output `a + a`, the known input `b`, and `-c`. Forwarding feeds the first two from that output and
        // that input, so the known program returns each value once and no longer receives `b`.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let b = builder.add_input(ArrayType::scalar(DataType::F64));
        let c = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let doubled_product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![doubled, x], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![b, x], None).unwrap()[0];
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![c], None).unwrap()[0];
        let negated_product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![negated, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![doubled, doubled_product, product, negated_product],
                vec![Placeholder; 4],
                vec![Placeholder; 4],
            )
            .unwrap();
        let partition = program.partition(&[true, true, true, false]).unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1, 2],
                    residual_inputs=[UnknownInput(3), ResidualEdge(0), ResidualEdge(1), ResidualEdge(2)],
                    outputs=[Known(0), Unknown(0), Unknown(1), Unknown(2)],
                ]
                known={
                    lambda %0:f64[], %1:f64[], %2:f64[] .
                    let %3:f64[] = add %0 %0
                        %4:f64[] = neg %2
                    in (%3, %3, %1, %4)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = mul %2 %0
                        %6:f64[] = mul %3 %0
                    in (%4, %5, %6)
                }"},
        );

        // The residual program is unchanged; only its sources and the known program change.
        assert_eq!(
            partition.forward_residuals().unwrap().to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 2],
                    residual_inputs=[UnknownInput(3), KnownOutput(0), KnownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0), Unknown(1), Unknown(2)],
                ]
                known={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = add %0 %0
                        %3:f64[] = neg %1
                    in (%2, %3)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
                    let %4:f64[] = mul %1 %0
                        %5:f64[] = mul %2 %0
                        %6:f64[] = mul %3 %0
                    in (%4, %5, %6)
                }"},
        );

        // Edges that carry the same known program atom share one edge.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![a], None).unwrap()[0];
        let known_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![negated, negated], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![sum, x], None).unwrap()[0];
        let residual_program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![product], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let partition = PartitionedProgram::from_parts(
            known_program,
            residual_program,
            2,
            vec![0],
            vec![
                ResidualInputSource::UnknownInput(1),
                ResidualInputSource::ResidualEdge(0),
                ResidualInputSource::ResidualEdge(1),
            ],
            vec![PartialEvaluationOutput::Unknown(0)],
        )
        .unwrap();
        let forwarded = partition.forward_residuals().unwrap();
        assert_eq!(forwarded.known_input_indices(), &[0]);
        assert_eq!(
            forwarded.residual_inputs(),
            &[
                ResidualInputSource::UnknownInput(1),
                ResidualInputSource::ResidualEdge(0),
                ResidualInputSource::ResidualEdge(0),
            ],
        );
        assert_eq!(
            forwarded.known_program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partitioned_program_forward_residuals_preserves_repeated_known_outputs() {
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let negated = builder.add_instruction(NegOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let known = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![input, negated, negated, input, negated],
                vec![Placeholder],
                vec![Placeholder; 5],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
        let residual = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let partition = PartitionedProgram::from_parts(
            known,
            residual,
            1,
            vec![0],
            vec![ResidualInputSource::ResidualEdge(0), ResidualInputSource::ResidualEdge(1)],
            vec![
                PartialEvaluationOutput::Known(0),
                PartialEvaluationOutput::Known(1),
                PartialEvaluationOutput::Known(2),
                PartialEvaluationOutput::Unknown(0),
            ],
        )
        .unwrap();
        let forwarded = partition.forward_residuals().unwrap();
        assert_eq!(
            forwarded.residual_inputs(),
            &[ResidualInputSource::KnownInput(0), ResidualInputSource::KnownOutput(1)]
        );
        assert_eq!(
            forwarded.known_program().to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%0, %1, %1)
            "}
            .trim_end(),
        );
        assert_eq!(
            forwarded.residual_program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_partitioned_program_forward_residuals_large_boundary() {
        let count = 4096;
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let unknown = builder.add_input(ArrayType::scalar(DataType::F64));
        let mut value = known;
        let mut outputs = Vec::new();
        for _ in 0..count {
            value = builder.add_instruction(NegOperation::new(), Vec::new(), vec![value], None).unwrap()[0];
            outputs
                .push(builder.add_instruction(MulOperation::new(), Vec::new(), vec![value, unknown], None).unwrap()[0]);
        }
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2], vec![Placeholder; count])
            .unwrap();
        let partition = program.partition(&[true, false]).unwrap();
        let known_rendering = partition.known_program().to_string();
        let residual_rendering = partition.residual_program().to_string();
        let known = partition.known_program().clone();
        let input_indices = partition.known_input_indices().to_vec();
        let forwarded = partition.forward_residuals().unwrap();
        assert_eq!(forwarded.known_program().to_string(), known_rendering);
        assert_eq!(forwarded.residual_program().to_string(), residual_rendering);
        assert_eq!(
            forwarded.residual_inputs(),
            std::iter::once(ResidualInputSource::UnknownInput(1))
                .chain((0..count).map(ResidualInputSource::ResidualEdge))
                .collect::<Vec<_>>(),
        );

        // Repeated feeders share edges in first-use order, even when that order reverses the known outputs.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let unknown = builder.add_input(ArrayType::scalar(DataType::F64));
        let mut sources = vec![ResidualInputSource::UnknownInput(1)];
        let mut outputs = Vec::new();
        for edge in (0..count).rev() {
            let left = builder.add_input(ArrayType::scalar(DataType::F64));
            let right = builder.add_input(ArrayType::scalar(DataType::F64));
            let sum = builder.add_instruction(AddOperation::new(), Vec::new(), vec![left, right], None).unwrap()[0];
            outputs
                .push(builder.add_instruction(MulOperation::new(), Vec::new(), vec![sum, unknown], None).unwrap()[0]);
            sources.extend([ResidualInputSource::ResidualEdge(edge); 2]);
        }
        let residual = builder
            .build::<Vec<Array>, Vec<Array>>(outputs, vec![Placeholder; 2 * count + 1], vec![Placeholder; count])
            .unwrap();
        let residual_rendering = residual.to_string();
        let expected_outputs = known.output_ids().iter().rev().copied().collect::<Vec<_>>();
        let expected_known = known.filtered(known.input_ids(), &expected_outputs, &[]).unwrap().0.to_string();
        let forwarded = PartitionedProgram::from_parts(
            known,
            residual,
            2,
            input_indices,
            sources,
            (0..count).map(PartialEvaluationOutput::Unknown).collect(),
        )
        .unwrap()
        .forward_residuals()
        .unwrap();
        assert_eq!(forwarded.known_program().to_string(), expected_known);
        assert_eq!(forwarded.residual_program().to_string(), residual_rendering);
        assert_eq!(
            forwarded.residual_inputs(),
            std::iter::once(ResidualInputSource::UnknownInput(1))
                .chain((0..count).flat_map(|edge| [ResidualInputSource::ResidualEdge(edge); 2]))
                .collect::<Vec<_>>(),
        );
        let values = forwarded.known_program().interpret(vec![Array::scalar(2f64).unwrap()]).unwrap();
        let inputs = std::iter::once(Array::scalar(3f64).unwrap())
            .chain(values.iter().flat_map(|value| [value.clone(), value.clone()]))
            .collect::<Vec<_>>();
        let expected = (0..count)
            .map(|index| Array::scalar(if index % 2 == 0 { 12f64 } else { -12f64 }).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(forwarded.residual_program().interpret(inputs), Ok(expected));
    }

    #[test]
    fn test_partitioned_program_forward_residuals_is_idempotent() {
        // `f(a, b, x) = (a + a, (a + a) * x, b * x)` partitioned with `x` unknown forwards a known output and a known
        // input that the known program then stops receiving. Forwarding again keeps both sources and both programs.
        let mut builder = ProgramBuilder::<Array, TestArrayOperation>::new();
        let a = builder.add_input(ArrayType::scalar(DataType::F64));
        let b = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![a, a], None).unwrap()[0];
        let doubled_product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![doubled, x], None).unwrap()[0];
        let product = builder.add_instruction(MulOperation::new(), Vec::new(), vec![b, x], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![doubled, doubled_product, product],
                vec![Placeholder; 3],
                vec![Placeholder; 3],
            )
            .unwrap();
        let forwarded = program.partition(&[true, true, false]).unwrap().forward_residuals().unwrap();
        let rendering = indoc! {"
            partition [
                known_inputs=[0],
                residual_inputs=[UnknownInput(2), KnownOutput(0), KnownInput(1)],
                outputs=[Known(0), Unknown(0), Unknown(1)],
            ]
            known={
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                in (%1)
            }
            residual={
                lambda %0:f64[], %1:f64[], %2:f64[] .
                let %3:f64[] = mul %1 %0
                    %4:f64[] = mul %2 %0
                in (%3, %4)
            }"};
        assert_eq!(forwarded.to_string(), rendering);
        let forwarded_twice = forwarded.forward_residuals().unwrap();
        assert_eq!(forwarded_twice.to_string(), rendering);
        assert_eq!(forwarded_twice.original_input_count(), 3);
    }

    #[test]
    fn test_partitioned_program_forward_residuals_effect_ordering() {
        // Forwarding that leaves the known program unchanged preserves the evidence that the two programs access
        // independent references.
        let reference_type: ArrayIrType = ReferenceType::new(ArrayType::scalar(DataType::F32)).into();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(reference_type.clone());
        let unknown = builder.add_input(reference_type.clone());
        let known = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![known], None).unwrap()[0];
        let unknown =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![unknown], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![known, unknown], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let (repeated, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, true, Some(&[0]), None)
            .unwrap();
        assert!(!repeated.forward_residuals().unwrap().has_effect_ordering_conflicts());

        // Rebuilding the known program renumbers the instructions that identify its reference allocations, so its
        // constraints conservatively become global. Here, forwarding the scale `s` removes it from the known outputs.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let known = builder.add_input(reference_type.clone());
        let unknown = builder.add_input(reference_type);
        let scale = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let known = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![known], None).unwrap()[0];
        let unknown =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![unknown], None).unwrap()[0];
        let product = builder
            .add_instruction(ArrayOperation::<Array>::from(MulOperation::new()), Vec::new(), vec![unknown, scale], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![known, product], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let (repeated, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false, true], true, true, Some(&[0]), None)
            .unwrap();
        assert!(!repeated.has_effect_ordering_conflicts());
        let forwarded = repeated.forward_residuals().unwrap();
        assert_eq!(
            forwarded.residual_inputs(),
            &[ResidualInputSource::UnknownInput(1), ResidualInputSource::KnownInput(2)],
        );
        assert_eq!(forwarded.known_input_indices(), &[0]);
        assert!(forwarded.has_effect_ordering_conflicts());
    }

    #[test]
    fn test_region_partition_with_residual_policy() {
        // `f(x, t) = (sin(dot(x, x)), cos(dot(x, x)) * t)` partitioned with `x` known: the policy places the residuals
        // of the top-level partition exactly as placing them in the ordinary partition would, saving the dot product
        // and recomputing its cosine.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let t = builder.add_input(ArrayType::scalar(DataType::F64));
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]));
        let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
        let sine = builder.add_instruction(SinOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let tangent = builder.add_instruction(MulOperation::new(), Vec::new(), vec![cosine, t], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![sine, tangent], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let policy = save_dots();
        let partition = program.entry_region_ref().partition_with_residual_policy(&[true, false], &policy).unwrap();
        assert_eq!(
            partition.to_string(),
            program.partition(&[true, false]).unwrap().with_residual_policy(&policy).unwrap().to_string()
        );
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0],
                    residual_inputs=[UnknownInput(1), ResidualEdge(0)],
                    outputs=[Known(0), Unknown(0)],
                ]
                known={
                    lambda %0:f64[3] .
                    let %1:f64[] = dot [
                        dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                    ] %0 %0
                        %2:f64[] = sin %1
                    in (%2, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = cos %1
                        %3:f64[] = mul %2 %0
                    in (%3)
                }"},
        );
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
            .partition_with_configuration(&[true, false], true, true, Some(&[0]), None)
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
            matches!(program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[1]), None),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 1 depends on deferred work in a repeated residual computation",),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_preserves_failure_order() {
        let program = reference_ordering_program();
        let partition = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, false, None, None)
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
                ResidualInputSource::UnknownInput(index) => {
                    assert_eq!(*index, 1);
                    TestValue::Reference(source.clone())
                }
                ResidualInputSource::ResidualEdge(index) => known[*index].clone(),
                source => panic!("ordinary partitions have no `{source:?}` sources"),
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
        let partition = program
            .entry_region_ref()
            .partition_with_configuration(&[true, false], true, true, None, None)
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
                ResidualInputSource::UnknownInput(index) => {
                    assert_eq!(*index, 1);
                    TestValue::Reference(source.clone())
                }
                ResidualInputSource::ResidualEdge(index) => known[*index].clone(),
                source => panic!("ordinary partitions have no `{source:?}` sources"),
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
        let (partition, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[false], true, true, Some(&[0]), None)
            .unwrap();
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

        // The write has known inputs and no outputs. Its deferred execution must still pull the allocation into
        // each residual call, rather than retain a reference created once by the known program.
        let (partition, _) = program
            .entry_region_ref()
            .partition_with_configuration(&[false], true, true, Some(&[]), None)
            .unwrap();
        assert!(partition.known_program().instructions().is_empty());
        assert!(partition.known_reference_inputs().next().is_none());
        assert_eq!(
            partition.residual_program().to_string(),
            indoc! {"
                lambda %0:f32[] .
                let %1:f32[] = print [label=deferred] %0
                    %2:f32[] = const 0.0
                    %3:ref<f32[]> = reference_new %2
                    () = reference_write %3 %2
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
            matches!(program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[0]), None),
            Err(ProgramError::MalformedProgram(message))
                if message == "required output 0 depends on deferred work in a repeated residual computation",),
        );
    }

    #[test]
    fn test_region_partition_with_configuration_repeated_residual_preserves_reference_constant_order() {
        let mut builder = ProgramBuilder::<
            TestCapture,
            ReferenceReadOperation<ArrayType, ArrayIrType, ArrayReferenceTransform>,
        >::new();
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
            program.entry_region_ref().partition_with_configuration(&[false], true, true, Some(&[0]), None),
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
        assert!(matches!(program.entry_region_ref().partition_with_configuration(&[false], true, true, None, None),
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
            program.entry_region_ref().partition_with_configuration(&[true], true, true, Some(&[1]), None),
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
            region.partition_with_configuration(&[false], true, true, Some(&[]), None).unwrap();
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
            program.entry_region_ref().partition_with_configuration(&[true, false], true, true, Some(&[0]), None),
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
        let (partition, deferred_instructions) = program
            .entry_region_ref()
            .partition_with_configuration(&[false], true, true, Some(&[0]), None)
            .unwrap();
        assert!(deferred_instructions.is_empty());
        assert_eq!(partition.outputs(), &[PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
        assert_eq!(partition.known_reference_inputs().collect::<Vec<_>>(), vec![1]);
        let known = partition.known_program().interpret(Vec::new()).unwrap();
        assert_eq!(known[0], TestValue::Array(Array::scalar(2.0_f32).unwrap()));
        assert_eq!(
            partition.residual_inputs(),
            &[ResidualInputSource::UnknownInput(0), ResidualInputSource::ResidualEdge(0)]
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
        assert_eq!(partition.known_input_indices(), &[0]);
        assert_eq!(
            partition.residual_inputs(),
            vec![ResidualInputSource::UnknownInput(1), ResidualInputSource::ResidualEdge(0),],
        );
        assert_eq!(partition.outputs(), vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Unknown(0)]);
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
        assert_eq!(partition.known_input_indices(), Vec::<usize>::new());
        assert!(partition.known_program.instructions().is_empty());
        assert!(partition.known_program.output_ids().is_empty());
        assert_eq!(
            partition.residual_inputs(),
            vec![ResidualInputSource::UnknownInput(0), ResidualInputSource::UnknownInput(1),],
        );
        assert_eq!(partition.outputs(), vec![PartialEvaluationOutput::Unknown(0), PartialEvaluationOutput::Unknown(1)]);

        // All-known known-ness folds everything into the known program and leaves an empty residual program.
        let partition = program.partition(&[true, true]).unwrap();
        assert_eq!(partition.known_input_indices(), &[0, 1]);
        assert_eq!(partition.residual_inputs(), &[]);
        assert_eq!(partition.outputs(), vec![PartialEvaluationOutput::Known(0), PartialEvaluationOutput::Known(1)]);
        assert!(partition.residual_program.instructions().is_empty());

        // The provided known-ness must cover every program input.
        assert!(matches!(program.partition(&[true]), Err(ProgramError::InvalidInputCount { expected: 2, actual: 1 })));
    }

    #[test]
    fn test_program_partition_with_residual_policy() {
        // `f(c, xs)` scans `c * cos(dot(x, x))` over the two rows `x` of `xs`, with the accumulator `c` unknown and
        // `xs` known. The split rule of the scan partitions its body: without a policy, the known scan stacks the
        // per-iteration cosines for the residual scan, while a policy that saves only dot products makes the known scan
        // stack the per-iteration dot products and the residual scan recompute their cosines.
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let _index = builder.add_input(ArrayType::scalar(DataType::I64));
            let c = builder.add_input(ArrayType::scalar(DataType::F64));
            let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
            let dot = DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]));
            let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
            let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
            let next = builder.add_instruction(MulOperation::new(), Vec::new(), vec![c, cosine], None).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder; 3], vec![Placeholder])
                .unwrap()
        };
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_region(body.entry_region_ref());
        let c = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]));
        let scan = ArrayOperation::Scan(ScanOperation::new(1, 2));
        let output = builder.add_instruction(scan, vec![body], vec![c, xs], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let partition = program.partition(&[false, true]).unwrap();
        assert_eq!(
            partition.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[2, 3] .
                    let %1:f64[2] = scan [carry_count=0, length=2, reverse=false] %0 [
                        body={
                            lambda %0:i64[], %1:f64[3] .
                            let %2:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %1 %1
                                %3:f64[] = cos %2
                            in (%3)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[2] .
                    let %2:f64[] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[] .
                            let %3:f64[] = mul %1 %2
                            in (%3)
                        },
                    ]
                    in (%2)
                }"},
        );

        let planned = program.partition_with_residual_policy(&[false, true], &save_dots()).unwrap();
        assert_eq!(
            planned.to_string(),
            indoc! {"
                partition [
                    known_inputs=[1],
                    residual_inputs=[UnknownInput(0), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[2, 3] .
                    let %1:f64[2] = scan [carry_count=0, length=2, reverse=false] %0 [
                        body={
                            lambda %0:i64[], %1:f64[3] .
                            let %2:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %1 %1
                            in (%2)
                        },
                    ]
                    in (%1)
                }
                residual={
                    lambda %0:f64[], %1:f64[2] .
                    let %2:f64[] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[] .
                            let %3:f64[] = cos %2
                                %4:f64[] = mul %1 %3
                            in (%4)
                        },
                    ]
                    in (%2)
                }"},
        );

        // Running the known program and then the residual program reproduces the original output.
        let xs = Array::new(
            ArrayType::new_static(DataType::F64, [2, 3]),
            [1.0f64, 2.0, 3.0, 0.5, 0.25, 0.125].into_iter().flat_map(f64::to_ne_bytes).collect(),
        )
        .unwrap();
        let c = Array::scalar(2.0f64).unwrap();
        let known_outputs = planned.known_program().interpret(vec![xs.clone()]).unwrap();
        let residual_outputs = planned.residual_program().interpret(vec![c.clone(), known_outputs[0].clone()]).unwrap();
        assert_eq!(residual_outputs, program.interpret(vec![c, xs]).unwrap());
    }

    #[test]
    fn test_program_partition_with_residual_policy_recomputes_known_region_operations() {
        // `f(c, xs, y)` scans `c * cos(dot(x, x))` over the two rows `x` of `xs` and multiplies the result by `y`, with
        // `c` and `xs` known and `y` unknown. No split rule runs for the scan, because all of its inputs are known, so
        // the known program computes it whole. Saving nothing recomputes it whole in the residual program, while saving
        // dot products saves its output, because recomputing the scan whole would also recompute its dot products.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let _index = builder.add_input(ArrayType::scalar(DataType::I64));
        let c = builder.add_input(ArrayType::scalar(DataType::F64));
        let x = builder.add_input(ArrayType::new_static(DataType::F64, [3]));
        let dot = DotOperation::new(DotDimensionNumbers::new(vec![0], vec![0], vec![], vec![]));
        let product = builder.add_instruction(dot, Vec::new(), vec![x, x], None).unwrap()[0];
        let cosine = builder.add_instruction(CosOperation::new(), Vec::new(), vec![product], None).unwrap()[0];
        let next = builder.add_instruction(MulOperation::new(), Vec::new(), vec![c, cosine], None).unwrap()[0];
        let body = builder
            .build::<Vec<Array>, Vec<Array>>(vec![next], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let body = builder.import_region(body.entry_region_ref());
        let c = builder.add_input(ArrayType::scalar(DataType::F64));
        let xs = builder.add_input(ArrayType::new_static(DataType::F64, [2, 3]));
        let y = builder.add_input(ArrayType::scalar(DataType::F64));
        let scan = ArrayOperation::Scan(ScanOperation::new(1, 2));
        let scanned = builder.add_instruction(scan, vec![body], vec![c, xs], None).unwrap()[0];
        let output = builder.add_instruction(MulOperation::new(), Vec::new(), vec![scanned, y], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
            .unwrap();

        let save_nothing = ResidualPolicyReference::<ArrayType>::new(NothingSavable);
        let planned = program.partition_with_residual_policy(&[true, true, false], &save_nothing).unwrap();
        assert_eq!(
            planned.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0), ResidualEdge(1)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[], %1:f64[2, 3] .
                    in (%0, %1)
                }
                residual={
                    lambda %0:f64[], %1:f64[], %2:f64[2, 3] .
                    let %3:f64[] = scan [carry_count=1, length=2, reverse=false] %1 %2 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[3] .
                            let %3:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %2
                                %4:f64[] = cos %3
                                %5:f64[] = mul %1 %4
                            in (%5)
                        },
                    ]
                        %4:f64[] = mul %3 %0
                    in (%4)
                }"},
        );
        let c = Array::scalar(2.0f64).unwrap();
        let xs = Array::new(
            ArrayType::new_static(DataType::F64, [2, 3]),
            [1.0f64, 2.0, 3.0, 0.5, 0.25, 0.125].into_iter().flat_map(f64::to_ne_bytes).collect(),
        )
        .unwrap();
        let y = Array::scalar(3.0f64).unwrap();
        let known_outputs = planned.known_program().interpret(vec![c.clone(), xs.clone()]).unwrap();
        let mut residual_inputs = vec![y.clone()];
        residual_inputs.extend(known_outputs);
        assert_eq!(
            planned.residual_program().interpret(residual_inputs),
            program.interpret(vec![c.clone(), xs.clone(), y.clone()]),
        );

        let planned = program.partition_with_residual_policy(&[true, true, false], &save_dots()).unwrap();
        assert_eq!(
            planned.to_string(),
            indoc! {"
                partition [
                    known_inputs=[0, 1],
                    residual_inputs=[UnknownInput(2), ResidualEdge(0)],
                    outputs=[Unknown(0)],
                ]
                known={
                    lambda %0:f64[], %1:f64[2, 3] .
                    let %2:f64[] = scan [carry_count=1, length=2, reverse=false] %0 %1 [
                        body={
                            lambda %0:i64[], %1:f64[], %2:f64[3] .
                            let %3:f64[] = dot [
                                dimensions=(lhs_contracting=[0], rhs_contracting=[0], lhs_batching=[], rhs_batching=[]),
                            ] %2 %2
                                %4:f64[] = cos %3
                                %5:f64[] = mul %1 %4
                            in (%5)
                        },
                    ]
                    in (%2)
                }
                residual={
                    lambda %0:f64[], %1:f64[] .
                    let %2:f64[] = mul %1 %0
                    in (%2)
                }"},
        );
    }
}
