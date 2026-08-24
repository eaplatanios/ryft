//! Traces Rust function execution into immutable, typed [`Program`]s.
//!
//! Tracing is a staging boundary. It replaces runtime inputs with symbolic [`Tracer`] values and invokes the supplied
//! Rust closure once. Operations applied to those values bind through an active [`StagingContext`], which records typed
//! instructions in a shared [`ProgramBuilder`]. Finalization validates the resulting graph, preserves the structured
//! input and output boundaries, and returns the program together with its inferred output types. Refer to the
//! [`TracingContext`] documentation for a rendered diagram of this pipeline and the relationship between root and
//! nested tracing.
//!
//! # What Gets Traced
//!
//! Tracing records operations bound through [`Context::bind`]; it does not record arbitrary Rust execution. Ordinary
//! host computation runs immediately while the closure is being traced. Consequently, a Rust `if`, loop, or container
//! traversal may depend only on information concretely available at trace time. Runtime-dependent behavior must be
//! represented explicitly by staged control-flow or data operations.
//!
//! The closure is part of specialization semantics: changing static host values can change which operations are
//! recorded. Dynamic runtime data should remain in tracer inputs rather than being inspected by host code.
//!
//! # Choosing an Entry Point
//!
//!   - [`Trace::trace`] accepts abstract input types and traces in the implementing [`Domain`]'s type, constant, and
//!     operation universe. [`Trace::infer_output_type`] performs the same trace but returns only the output types.
//!   - [`trace`] and [`infer_output_type`] accept example values. The values contribute only their abstract types; they
//!     are neither executed nor captured, and their statically known execution domain selects the tracing universe.
//!   - [`TracingContext::trace`] names the value, operation, and capture representations directly.
//!     [`TracingContext::trace_with_named_axes`] additionally seeds named-axis bindings for operations such as
//!     collectives.
//!   - [`NestedTracingContext::trace`] records a flat inner program in an enclosing [`Context`]'s universe. It is used
//!     for loop bodies, branches, derivative rules, and other regions that need a fresh builder while retaining parent
//!     capabilities. [`NestedTracingContext::trace_with_named_axes`] lets local bindings shadow parent axes.
//!
//! [`DomainTracingContext`] and [`DomainTracer`] are the usual aliases for root domain tracing;
//! [`NestedTracer`] is the corresponding nested-trace value alias.
//!
//! # Tracer Identity and Failure Propagation
//!
//! A [`Tracer`] carries its context, abstract type, and [`TracerState`]. A live tracer names one [`AtomId`] in exactly
//! one builder. Equality is staging identity (i.e., same builder and same atom) and not equality of eventual runtime
//! values. Trace boundaries reject foreign output tracers because reusing an atom index from another builder would
//! silently alias an unrelated value.
//!
//! Convenience operators defer binding errors by returning [`TracerState::Poison`]. Poison then propagates through
//! downstream binds, and finalization reports the builder's original [`ProgramError`]. Explicitly fallible staging APIs
//! may instead return errors immediately. A tracer that escapes the closure keeps the shared builder alive, so
//! finalization rejects it with [`ProgramError::EscapedProgramBuilder`].
//!
//! # Context State, Captures, and Nesting
//!
//! [`TracingContext`] shares one builder, capture table, and set of named-axis bindings across all of its clones.
//! The staged constant type may differ from the captured runtime value type: compilation can therefore place stable
//! [`CaptureReference`](crate::CaptureReference)s in source IR while retaining concrete runtime values separately.
//! The result is later packaged as a [`ClosedProgram`](crate::ClosedProgram) without embedding runtime buffers in
//! the program.
//!
//! [`NestedTracingContext`] owns a fresh builder and local named-axis bindings but delegates capture registration and
//! other parent capabilities through the enclosing context. Transform and nested contexts can therefore compose by
//! following the same context stack for both operation binding and capture handling.
//!
//! # Extending Tracing
//!
//! Most operations require no tracing-specific rule: implement the operation and its type inference, add it to the
//! domain's operation family, and bind it through [`Context::bind`]. Tracer capability implementations should pass the
//! concrete payload into the context and let the operation family's [`From`] conversion select its wrapper variant.
//!
//! A new staging context should implement [`Context`] with `Value = Tracer<Self>` and then [`StagingContext`]. Keep
//! mutable builder state behind shared handles, use checked builder APIs, resolve foreign or poisoned tracers
//! conservatively, and finalize only after all context and tracer clones have been dropped.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::axes::NamedAxis;
use crate::contexts::{Context, Domain, StagingContext, ValueResolution};
use crate::macros::{check_builders, check_count};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::{
    AtomId, BindingRegionDriver, Operation, Program, ProgramBuilder, ProgramError, ProjectedValue, Provenance,
    ProvenanceScope, ProvenanceState, Type, TypeError, Typed, Value, ValueProjection,
};

/// State carried by a [`Tracer`] that indicates whether this tracer is _live_ and has a corresponding
/// [`Atom`](crate::Atom) or _poisoned_, meaning that it corresponds to an error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TracerState {
    /// The corresponding [`Tracer`] is _live_ and has a corresponding [`Atom`](crate::Atom).
    Live(AtomId),

    /// The corresponding [`Tracer`] has been _poisoned_, meaning that it corresponds to an error and will propagate
    /// that error wherever it is used (i.e., it will _poison_ those corresponding downstream [`Tracer`]s too).
    Poison,
}

/// Value used while tracing [`Program`]s through an active [`Context`], substituting actual runtime values, and
/// recording the executed [`Operation`]s in that [`Context`]. When tracing fails, later operations return _poisoned_
/// tracers which are represented using [`TracerState::Poison`].
#[derive(Clone, Parameter)]
pub struct Tracer<C: Context> {
    /// [`Context`] associated with this [`Tracer`].
    context: C,

    /// [`TracerState`] of this [`Tracer`].
    state: TracerState,

    /// [`Type`] of the value that this [`Tracer`] represents.
    r#type: C::Type,
}

impl<C: Context> Tracer<C> {
    /// Creates a new [`Tracer`].
    #[inline]
    pub fn new(context: C, state: TracerState, r#type: C::Type) -> Self {
        Self { context, state, r#type }
    }

    /// Returns the [`TracerState`] of this [`Tracer`].
    #[inline]
    pub fn state(&self) -> &TracerState {
        &self.state
    }

    /// Returns the [`Context`] associated with this [`Tracer`].
    #[inline]
    pub fn context(&self) -> &C {
        &self.context
    }

    /// Returns the staged [`AtomId`] for this [`Tracer`] if it is _live_,
    /// and [`ProgramError::PoisonedValue`] otherwise.
    #[inline]
    pub fn atom_id(&self) -> Result<AtomId, ProgramError> {
        match &self.state {
            TracerState::Live(atom) => Ok(*atom),
            TracerState::Poison => Err(ProgramError::PoisonedValue),
        }
    }
}

// `Tracer` equality is *staging identity*, not value equality. Two tracers are equal if and only if they correspond to
// the same staged `Atom` of the same `ProgramBuilder` (or are both poisoned in the same builder). Two tracers that
// would evaluate to equal runtime values but were staged as distinct atoms are considered unequal, which is the
// conservative answer trace-time analyses need. For example, the loop invariance fixed points of the `scan` and `while`
// partial evaluation rules degrade to syntactic passthrough detection under a staging known-side context precisely
// because of these semantics.
impl<C: StagingContext> PartialEq for Tracer<C> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(self.context.builder(), other.context.builder()) && self.state == other.state
    }
}

impl<C: StagingContext> Tracer<C> {
    /// Returns the [`ProgramBuilder`] associated with this [`Tracer`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>> {
        self.context.builder()
    }

    /// Applies the provided _unary_ [`Operation`] to this [`Tracer`] returning the resulting [`Tracer`]. _Unary_
    /// operations are operations that have a single input and a single output. If the provided operation is not a
    /// unary operation, then the resulting [`Tracer`] will contain a [`TracerState::Poison`].
    pub fn unary<P: Into<C::Operation>>(&self, operation: P) -> Self {
        let operation = operation.into();
        match self.context.stage_operation(operation, [], &[self]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() });
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }

    /// Applies the provided _binary_ [`Operation`] to this [`Tracer`] and the provided [`Tracer`] returning the
    /// resulting [`Tracer`]. _Binary_ operations are operations that have two inputs and a single output. If the
    /// provided operation is not a binary operation, then the resulting [`Tracer`] will contain a
    /// [`TracerState::Poison`].
    pub fn binary<P: Into<C::Operation>>(&self, rhs: &Self, operation: P) -> Self {
        let operation = operation.into();
        match self.context.stage_operation(operation, Vec::new(), &[self, rhs]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() });
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
            Err(error) => {
                self.context.error(error);
                Self { state: TracerState::Poison, r#type: self.r#type.clone(), context: self.context.clone() }
            }
        }
    }
}

impl<C: Context> Debug for Tracer<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tracer")
            .field("state", &self.state)
            .field("type", &self.r#type)
            .finish_non_exhaustive()
    }
}

impl<C: Context> Display for Tracer<C> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom_id) => write!(formatter, "{atom_id}"),
            TracerState::Poison => write!(formatter, "<poison:{}>", self.r#type),
        }
    }
}

impl<C: Context> Typed for Tracer<C> {
    type Type = C::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<C: StagingContext> Value for Tracer<C> {
    type DispatchDomain = C;
    type ExecutionDomain = C;

    #[inline]
    fn dispatch_domain(&self) -> C {
        self.context().clone()
    }

    #[inline]
    fn execution_domain(&self) -> C {
        self.context().clone()
    }
}

impl<C: StagingContext, T: Type> ValueProjection<T> for Tracer<C>
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
        Ok(ProjectedValue::new(self, <&T>::try_from(&self.r#type)?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(&self.r#type)?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

/// Ordinary active tracing [`Context`] over a [`Type`]/[`Value`]/[`Operation`] universe. [`TracingContext`] pairs
/// the staged-constant and operation representations `(V, O)` of a program with the [`ProgramBuilder`] used for one
/// tracing invocation. It presents itself as a [`Domain`] whose [`Value`] is [`Tracer<Self>`](Tracer) and whose
/// [`Domain::Constant`] is `V`. Its default [`StagingContext::stage_operation`] behavior records each primitive bind
/// as a program instruction. Transform contexts wrap or replace this context when they need different binding behavior,
/// but they still share the same [`Context`] protocol used by [`Tracer`] values.
///
/// The optional capture parameter `C` names the concrete runtime value type stored in the capture table while tracing
/// a captured [`Program`]. It is deliberately distinct from the staged-constant type `V`.
/// [`capture`](crate::CapturingContext::capture) takes a runtime value of type `C` and returns a symbolic constant of
/// type `V` (i.e., a [`CaptureReference`](crate::CaptureReference)). In a capturing context the two genuinely differ.
/// For example, we might use a runtime device buffer for `C` versus a [`CaptureReference`](crate::CaptureReference)
/// for `V`. `C` defaults to `V` for the common non-capturing case, where no capture table exists and the distinction
/// is moot. Refer to [`CapturingContext`](crate::CapturingContext) and [`CaptureReference`](crate::CaptureReference)
/// for more information on what captures are and how they are used in practice.
///
/// # Tracing Pipeline
///
/// ```mermaid
/// %%{init: {"themeCSS": ".edgeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   values["Example Input Values"]
///   root["TracingContext"]
///   values -->|"&lt;code&gt;trace&lt;/code&gt; or &lt;code&gt;infer_output_type&lt;/code&gt;"| root
///   types["Domain + Input Types"] -->|"&lt;code&gt;Trace&lt;/code&gt; Trait"| root
///   request["Nested Trace Request"] --> nested["NestedTracingContext"]
///   parent["Enclosing Context"] -->|"universe and capabilities"| nested
///   root --> root_state["Shared Builder, Capture Table, and Named Axes"]
///   nested --> nested_state["Fresh Builder and Local Named Axes"]
///   nested -.->|"delegates capture registration"| parent
///   root_state --> tracers["Symbolic Input Tracers"]
///   nested_state --> tracers
///   tracers -->|"invoke once"| closure["Rust Closure"]
///   closure -->|"bind typed operations"| builder["Active Program Builder"]
///   builder -->|"validate and build"| result["Program + Structured Boundary Metadata"]
/// ```
///
/// Both paths use the same staging protocol after creating their context. The root path owns capture storage. The
/// nested path expresses its program in the parent's type, constant, and operation universe and delegates capture
/// registration to that parent.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub struct TracingContext<V: Value, O: Operation<Type = V::Type>, C = V> {
    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced. The builder is held behind
    /// an [`Rc`] rather than being outright owned because a single trace shares one builder across many contexts.
    /// A [`Tracer`] holds its [`Context`] by value, and tracing freely clones tracers (and hence their contexts) as
    /// values flow through the traced function, and so a cloned [`TracingContext`] must keep pointing at the *same*
    /// accumulating builder. Cloning the [`Rc`] does exactly that. An owned [`ProgramBuilder`] would instead be
    /// *forked* on every context clone, and so [`Tracer`]s created at different points in the trace would accumulate
    /// into divergent programs rather than the one program the trace is building. Furthermore, the nested [`RefCell`]
    /// supplies the interior mutability that staging needs.
    builder: Rc<RefCell<ProgramBuilder<V, O>>>,

    /// Capture table of closed-over runtime values, referenced symbolically from the staged [`Program`] via
    /// [`CaptureReference`](crate::CaptureReference)s. It stays empty for ordinary (i.e., non-capturing) tracing and
    /// is filled only when tracing a captured [`Program`] (e.g., when just-in-time-compiling a function that closes
    /// over device buffers), in which case those values are passed to the compiled program as runtime arguments rather
    /// than being baked into it. Capturing is gated at the type level: [`capture`](crate::CapturingContext::capture)
    /// is implemented only when the staged constant type can embed a [`CaptureReference`](crate::CaptureReference),
    /// and so an ordinary trace over plain runtime constants can never push into this table. Capture-owning traces
    /// construct their context directly and read this table back out through [`captures`](Self::captures). The
    /// [`trace`](Self::trace) entry points instead discard it and therefore reject traces that registered captures.
    /// Refer to the documentation of [`CaptureReference`](crate::CaptureReference) for more information.
    ///
    /// Like the [`builder`](Self::builder), the table is held behind an [`Rc`] and a [`RefCell`] for the same reason:
    /// one capturing trace shares a single table across its many cloned contexts, so the [`Rc`] keeps every clone
    /// pushing into the *same* accumulating table (which is what keeps [`CaptureReference`](crate::CaptureReference)
    /// indices consistent) instead of forking it, and the [`RefCell`] supplies the interior mutability
    /// [`capture`](crate::CapturingContext::capture) needs to push through a shared `&self`.
    captures: Rc<RefCell<Vec<C>>>,

    /// Named axes this [`TracingContext`] was seeded with, resolved by its [`NamedAxes`] implementation. An ordinary
    /// trace binds no named axes and this stays empty. Traces that run inside a manual-parallelism region (e.g., a
    /// `shard_map` body) are seeded with that region's device mesh axes so that named-axis readers (e.g., collectives)
    /// can validate and resolve them. The list is immutable for the [`TracingContext`]'s lifetime and shared across
    /// cloned contexts.
    named_axes: Rc<Vec<(String, NamedAxis)>>,

    /// Active [`ProvenanceState`] recording the non-semantic provenance scopes and origins that staged
    /// [`Instruction`](crate::Instruction)s snapshot. It is shared behind an [`Rc`] (separately from [`Self::builder`]
    /// so that [`ProvenanceScope`] transitions never hold a builder borrow) so that every cloned context observes the
    /// same active scopes, while independent traces own independent states.
    provenance: Rc<ProvenanceState>,
}

impl<V: Value, O: Operation<Type = V::Type>, C> TracingContext<V, O, C> {
    /// Creates a new [`TracingContext`] over the `(V, O)` type universe with a fresh, empty [`ProgramBuilder`] and a
    /// fresh, empty capture table. Use [`builder`](Self::builder) afterward to read or finalize the staged program, and
    /// [`captures`](Self::captures) to read values registered through [`capture`](crate::CapturingContext::capture). To
    /// instead compose further staging onto a trace that already owns prior instructions, do not create a context at
    /// all: an input [`Tracer`]'s [`context`](Tracer::context) shares that trace's [`ProgramBuilder`], and so staging
    /// on it (e.g., via [`stage_operation`](StagingContext::stage_operation)) appends to the same program.
    #[inline]
    pub fn new() -> Self {
        Self {
            builder: Rc::new(RefCell::new(ProgramBuilder::<V, O>::new())),
            captures: Rc::new(RefCell::new(Vec::new())),
            named_axes: Rc::new(Vec::new()),
            provenance: Rc::new(ProvenanceState::new()),
        }
    }

    /// Returns the [`ProgramBuilder`] that this [`TracingContext`] stages into.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<V, O>>> {
        &self.builder
    }

    /// Returns the shared capture table that [`capture`](crate::CapturingContext::capture) fills while tracing. That
    /// table stays empty for ordinary traces, since [`capture`](crate::CapturingContext::capture) is only implemented
    /// when the staged constant type is [`CaptureReference`](crate::CaptureReference).
    #[inline]
    pub fn captures(&self) -> &Rc<RefCell<Vec<C>>> {
        &self.captures
    }

    /// Returns the named axes this trace was seeded with, resolved by its [`NamedAxes`](crate::NamedAxes)
    /// implementation. Ordinary traces are seeded with no axes.
    #[inline]
    pub fn named_axes(&self) -> &[(String, NamedAxis)] {
        self.named_axes.as_slice()
    }

    /// Traces `function` into a [`Program`] for the provided input types. This is the symbolic ordinary-tracing entry
    /// point. It creates a fresh [`TracingContext`] over the `(V, O)` type universe, executes `function` once on
    /// [`Tracer`] inputs standing in for `input_type`, and returns the output types plus the finalized program.
    /// Operation binds are handled by the context's [`StagingContext::stage_operation`] implementation. The type
    /// universe only supplies the staged constant and operation types used by that program. The capture parameter `C`
    /// is preserved on the staged [`Tracer`] leaves so that callers tracing in a context with a non-default capture
    /// type (such as a backend whose runtime [`Domain::Value`] differs from its staged [`Domain::Constant`]) observe
    /// that same context type. The trace must not register captures. Refer to the documentation of
    /// [`trace_with_named_axes`](Self::trace_with_named_axes) for the rationale and the rejection semantics.
    #[inline]
    pub fn trace<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
    ) -> Result<(Output::To<V::Type>, Program<V, O, Input::To<V>, Output::To<V>>), ProgramError> {
        Self::trace_with_named_axes(function, input_type, Vec::new())
    }

    /// Traces `function` against `input_type` like [`trace`](Self::trace), but seeds the trace's context with the
    /// provided named axes. Named-axis readers staged by `function` (e.g., collectives) resolve against these bindings.
    /// This entry point returns only the traced [`Program`] and therefore has no way to hand a capture table to its
    /// caller. A capture registered through [`capture`](crate::CapturingContext::capture) during the trace would leave
    /// capture-referencing constants in the returned program while the values they name are silently discarded, and a
    /// later use of that program inside a capture-owning scope (e.g., as an attached region of a compiled function)
    /// would resolve those references against that unrelated scope's capture table, silently aliasing whichever value
    /// occupies the referenced slot. Such traces are therefore rejected with [`ProgramError::DiscardedCaptures`].
    /// Traces that need to capture must construct their [`TracingContext`] directly and pair the traced program with
    /// the context's [`captures`](Self::captures) table (e.g., through a [`ClosedProgram`](crate::ClosedProgram)).
    pub fn trace_with_named_axes<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
        named_axes: Vec<(String, NamedAxis)>,
    ) -> Result<(Output::To<V::Type>, Program<V, O, Input::To<V>, Output::To<V>>), ProgramError> {
        let (output_types, program, capture_count) =
            Self::trace_with_named_axes_counting_captures(function, input_type, named_axes)?;
        if capture_count > 0 {
            return Err(ProgramError::DiscardedCaptures { count: capture_count });
        }
        Ok((output_types, program))
    }

    /// Traces `function` against `input_type` and returns the output type, without retaining the traced [`Program`].
    /// Use this when callers only need the output types of an ordinary symbolic trace. Unlike [`trace`](Self::trace),
    /// captures registered during the trace are tolerated: the traced program is discarded together with them, so no
    /// dangling capture reference can escape (e.g., output-type inference over a function that stages calls to
    /// captured compiled functions remains valid).
    #[inline]
    pub fn infer_output_type<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
    ) -> Result<Output::To<V::Type>, ProgramError> {
        Ok(Self::trace_with_named_axes_counting_captures(function, input_type, Vec::new())?.0)
    }

    /// Traces like [`trace_with_named_axes`](Self::trace_with_named_axes) but additionally reports the number of
    /// captures the trace registered into its local (and discarded) capture table, leaving the decision of whether
    /// discarding them is acceptable to the caller. [`trace_with_named_axes`](Self::trace_with_named_axes) rejects
    /// such traces because it retains the program, while [`infer_output_type`](Self::infer_output_type) tolerates
    /// them because it discards the program along with the captures.
    fn trace_with_named_axes_counting_captures<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
        named_axes: Vec<(String, NamedAxis)>,
    ) -> Result<(Output::To<V::Type>, Program<V, O, Input::To<V>, Output::To<V>>, usize), ProgramError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let captures = Rc::new(RefCell::new(Vec::new()));
        let input_structure = input_type.parameter_structure();

        let (output_types, outputs, output_structure) = {
            let context = Self {
                builder: builder.clone(),
                captures: captures.clone(),
                named_axes: Rc::new(named_axes),
                provenance: Rc::new(ProvenanceState::new()),
            };
            let input = input_type.map_parameters(|t| context.input(t)).map_err(ProgramError::from)?;
            let output = function(input).map_err(|e| builder.borrow_mut().error.take().unwrap_or(e))?;

            // The outputs must belong to this tracing context. A foreign tracer's atom ID would silently alias
            // whichever atom shares its index in this builder, and so we check for this here.
            check_builders!(&builder, [output.parameters().map(|output| output.builder())])?;

            builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            let outputs = output.parameters().map(|o| o.atom_id()).collect::<Result<Vec<_>, _>>()?;
            let output_types = output.map_parameters(|o| o.r#type().into_owned()).map_err(ProgramError::from)?;
            (output_types, outputs, output_structure)
        };

        let capture_count = captures.borrow().len();
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(outputs, input_structure, output_structure)?;

        Ok((output_types, program, capture_count))
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> Clone for TracingContext<V, O, C> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            builder: self.builder.clone(),
            captures: self.captures.clone(),
            named_axes: self.named_axes.clone(),
            provenance: self.provenance.clone(),
        }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> Debug for TracingContext<V, O, C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> Default for TracingContext<V, O, C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> Domain for TracingContext<V, O, C> {
    type Type = V::Type;
    type Value = Tracer<Self>;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<Type = V::Type>, C> Context for TracingContext<V, O, C> {
    #[inline]
    fn lift(&self, constant: V) -> Result<Tracer<Self>, ProgramError> {
        // A value family that forbids constant storage (most notably a mutable reference holder) must be rejected at
        // the lift that attempts it because region sealing re-checks every stored constant, but only at build time,
        // when the tracing call that stored the constant is long gone.
        if let Err(error) = constant.validate_as_constant() {
            return Err(self.error(error.into()));
        }
        Ok(self.constant(constant))
    }

    #[inline]
    fn bind<P: Into<O>, D: BindingRegionDriver<V, O>>(
        &self,
        operation: P,
        driver: D,
        inputs: &[Tracer<Self>],
    ) -> Result<Vec<Tracer<Self>>, ProgramError> {
        self.stage_operation(operation, driver, inputs)
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // `TracingContext`s stage values as `Tracer`s rather than computing them and so they are never eager.
        false
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // A `TracingContext` is a staging boundary. It owns the builder that instructions are emitted into, and so it
        // also owns the shared provenance state those instructions snapshot.
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
    fn resolve(&self, value: &Tracer<Self>) -> ValueResolution<V> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Constant(constant.clone()),
            None => ValueResolution::Staged(atom_id),
        }
    }
}

impl<V: Value, O: Operation<Type = V::Type>, C> StagingContext for TracingContext<V, O, C> {
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Constant, Self::Operation>>> {
        &self.builder
    }
}

/// Represents a nested [`TracingContext`] that is used to trace a closure into a [`Program`] expressed in an
/// *enclosing* [`Context`]'s universe rather than in a raw `(V, O)` type universe of its own. Where [`TracingContext`]
/// is keyed by the `(V, O)` types it stages and owns its own capture table, [`NestedTracingContext`] is keyed by
/// the enclosing [`Context`] `C`. It derives its `Type`, `Constant`, and `Operation` types from `C`, owns a fresh
/// [`ProgramBuilder`] for the nested [`Program`] it stages, and holds a clone of `C`. Runtime capture registration
/// is *not* owned by this context but is rather delegated to the enclosing context through
/// [`CapturingContext`](crate::CapturingContext), and so values captured while tracing the nested program flow
/// into `C`'s table along the same nesting path as ordinary operation staging. As with [`TracingContext`], the
/// [`ProgramBuilder`] is shared behind an [`Rc`] so cloned contexts keep appending to the *same* nested program.
pub struct NestedTracingContext<C: Context> {
    /// [`Context`] that this [`NestedTracingContext`] is nested into.
    parent: C,

    /// [`ProgramBuilder`] that this [`NestedTracingContext`] stages the nested [`Program`] into.
    builder: Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>>,

    /// Named axes this [`NestedTracingContext`] was seeded with, resolved by its [`NamedAxes`] implementation ahead
    /// of the parent context's bindings. An ordinary nested trace binds no named axes of its own and this stays empty,
    /// in which case every lookup delegates to the parent. The list is immutable for the [`NestedTracingContext`]'s
    /// lifetime and shared across cloned contexts.
    named_axes: Rc<Vec<(String, NamedAxis)>>,

    /// Active [`ProvenanceState`] for the nested program, shared across cloned contexts and independent of the
    /// parent's state. It is seeded with the parent context's current provenance at depth zero, so that instructions
    /// staged in the nested program record where the nested program itself came from, while later replaying them
    /// under the same ambient scopes preserves each instruction's provenance exactly.
    provenance: Rc<ProvenanceState>,
}

impl<C: Context> NestedTracingContext<C> {
    /// Creates a new [`NestedTracingContext`] that owns a fresh [`ProgramBuilder`] and traces on behalf of `parent`.
    pub fn new(parent: C) -> Self {
        let provenance = Rc::new(ProvenanceState::seeded(parent.provenance()));
        Self {
            parent,
            builder: Rc::new(RefCell::new(ProgramBuilder::new())),
            named_axes: Rc::new(Vec::new()),
            provenance,
        }
    }

    /// Returns the [`Context`] that this [`NestedTracingContext`] is nested into.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ProgramBuilder`] that this [`NestedTracingContext`] stages the nested [`Program`] into.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Constant, C::Operation>>> {
        &self.builder
    }

    /// Returns the named axes this nested trace was seeded with, resolved by its [`NamedAxes`](crate::NamedAxes)
    /// implementation ahead of the parent context's bindings. Ordinary nested traces are seeded with no axes.
    #[inline]
    pub fn named_axes(&self) -> &[(String, NamedAxis)] {
        self.named_axes.as_slice()
    }

    /// Traces `function` into a flat [`Program`] expressed in the enclosing context `parent`'s universe. This is the
    /// nested-tracing counterpart of [`TracingContext::trace`], following the same tracing protocol, but with two
    /// nested-specific differences: runtime captures registered while tracing delegate to `parent` through
    /// [`CapturingContext`](crate::CapturingContext), so nested traces compose with enclosing capturing traces, and the
    /// traced program is flat (i.e., [`Vec`]-parameterized) on both boundaries (the canonical shape for nested programs
    /// that are replayed positionally) with the closure's output [`Parameter`] structure returned alongside it so that
    /// callers can reassemble structured outputs from the program's flat outputs.
    #[inline]
    pub fn trace<F, Output>(
        parent: C,
        function: F,
        input_types: Vec<C::Type>,
    ) -> Result<
        (Output::ParameterStructure, Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>),
        ProgramError,
    >
    where
        F: FnOnce(Vec<Tracer<Self>>) -> Result<Output, ProgramError>,
        Output: Parameterized<Tracer<Self>>,
    {
        Self::trace_with_named_axes(parent, function, input_types, Vec::new())
    }

    /// Traces `function` against `input_types` like [`trace`](Self::trace), but seeds the nested trace's context with
    /// the provided named axes, which shadow same-named bindings of `parent`. This is the nested-tracing counterpart
    /// of [`TracingContext::trace_with_named_axes`].
    pub fn trace_with_named_axes<F, Output>(
        parent: C,
        function: F,
        input_types: Vec<C::Type>,
        named_axes: Vec<(String, NamedAxis)>,
    ) -> Result<
        (Output::ParameterStructure, Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>),
        ProgramError,
    >
    where
        F: FnOnce(Vec<Tracer<Self>>) -> Result<Output, ProgramError>,
        Output: Parameterized<Tracer<Self>>,
    {
        let input_count = input_types.len();
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let named_axes = Rc::new(named_axes);
        let provenance = Rc::new(ProvenanceState::seeded(parent.provenance()));
        let context = Self { parent, builder, named_axes, provenance };

        let (output_structure, output_atoms) = {
            let input_tracers = input_types.into_iter().map(|r#type| context.input(r#type)).collect::<Vec<_>>();
            let output = function(input_tracers)
                .map_err(|error| context.builder().borrow_mut().error.take().unwrap_or(error))?;
            context.builder().borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();

            // The outputs must belong to this trace. A foreign tracer's atom ID would silently alias whichever atom
            // shares its index in this builder, so the boundary rejects it with a builder-identity check.
            check_builders!(context.builder(), [output.parameters().map(|output| output.builder())])?;

            let output_atoms = output.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
            (output_structure, output_atoms)
        };

        // Clone out the builder handle and drop the context so that the clone is the sole owner, letting
        // `Rc::try_unwrap` recover the builder below unless a `Tracer` escaped the trace and still holds a reference.
        let builder = context.builder().clone();
        drop(context);
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();

        let output_count = output_atoms.len();
        let program = builder.build::<Vec<C::Constant>, Vec<C::Constant>>(
            output_atoms,
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
        )?;

        Ok((output_structure, program))
    }
}

impl<C: Context> Clone for NestedTracingContext<C> {
    #[inline]
    fn clone(&self) -> Self {
        Self {
            parent: self.parent.clone(),
            builder: self.builder.clone(),
            named_axes: self.named_axes.clone(),
            provenance: self.provenance.clone(),
        }
    }
}

impl<C: Context> Debug for NestedTracingContext<C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("NestedTracingContext").finish_non_exhaustive()
    }
}

impl<C: Context> Domain for NestedTracingContext<C> {
    type Type = C::Type;
    type Value = Tracer<Self>;
    type Constant = C::Constant;
    type Operation = C::Operation;
}

impl<C: Context> Context for NestedTracingContext<C> {
    #[inline]
    fn lift(&self, constant: C::Constant) -> Result<Tracer<Self>, ProgramError> {
        // A value family that forbids constant storage (most notably a mutable reference holder) must be rejected at
        // the lift that attempts it because region sealing re-checks every stored constant, but only at build time,
        // when the tracing call that stored the constant is long gone.
        if let Err(error) = constant.validate_as_constant() {
            return Err(self.error(error.into()));
        }
        Ok(self.constant(constant))
    }

    #[inline]
    fn bind<P: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: P,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation(operation, driver, inputs)
    }

    #[inline]
    fn is_eager(&self) -> bool {
        // `NestedTracingContext`s stage values as `Tracer`s rather than computing them and so they are never eager.
        false
    }

    #[inline]
    fn provenance(&self) -> Provenance {
        // A `NestedTracingContext` is a staging boundary for the nested program. It owns an independent provenance
        // state seeded from its parent and not a delegation to the parent's state.
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
    fn resolve(&self, value: &Tracer<Self>) -> ValueResolution<C::Constant> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Constant(constant.clone()),
            None => ValueResolution::Staged(atom_id),
        }
    }
}

impl<C: Context> StagingContext for NestedTracingContext<C> {
    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Constant, Self::Operation>>> {
        &self.builder
    }
}

/// [`TracingContext`] named by an enclosing [`Domain`] `D`'s associated types. This is `TracingContext` over `D`'s
/// type, staged constant, and [`Operation`] representations, and so it is the active tracing context that stages a
/// [`Program`] expressed in `D`'s universe. The optional capture parameter `C` defaults to `D`'s staged constant
/// representation, matching the default capture type of [`TracingContext::new`]. Closed program traces over a backend
/// whose runtime `Value` type differs from its staged `Constant` type pin `C` to that runtime value type explicitly.
/// Use this alias at call sites that already hold a [`Domain`] and want to name the matching tracing context. Use
/// [`TracingContext`] directly at sites that already work in terms of a `(V, O)` universe.
pub type DomainTracingContext<D, C = <D as Domain>::Constant> =
    TracingContext<<D as Domain>::Constant, <D as Domain>::Operation, C>;

/// [`Tracer`] flowing through a [`DomainTracingContext`] for a backend [`Domain`] `D`. This is the value that stands in
/// for a `D`-typed runtime value while a function is being traced into a [`Program`]. Each [`Operation`] bound on these
/// tracers records a program instruction and yields further [`DomainTracer`]s, and so ordinary backend traces flow
/// entirely in them. The [`Domain`] is a pure type witness, and so the tracer borrows nothing from it. The backend-less
/// specialization used during symbolic program tracing and transposition is a [`Tracer`] over a plain
/// [`TracingContext<V, O>`](TracingContext).
pub type DomainTracer<D> = Tracer<DomainTracingContext<D>>;

/// [`Tracer`] flowing through a [`NestedTracingContext`] over an enclosing context `C`. This is the value used while
/// tracing a nested closure into a [`Program`] expressed in the enclosing context's universe. The closure receives
/// these tracers in place of `C`-typed runtime values, each [`Operation`] bound on them records an instruction in the
/// nested [`Program`], and the staged program is then interpreted, differentiated, transposed, etc. back in `C`. Use
/// this alias at call sites that trace a closure into a nested program over an enclosing context `C`.
pub type NestedTracer<C> = Tracer<NestedTracingContext<C>>;

/// Extension trait that exposes ordinary symbolic tracing on any [`Domain`], with the tracing universe named explicitly
/// through the implementing type and the signature supplied as abstract input types. Refer to the documentation of the
/// [`trace`] and [`infer_output_type`] functions for information on the trace and its arguments.
pub trait Trace: Domain {
    /// Traces `function` into a [`Program`] for the provided input types, in this [`Domain`]'s type, staged-constant,
    /// and operation universe. This is the explicitly named counterpart of the [`trace`] function, which recovers the
    /// tracing universe and signature from example values instead. Refer to its documentation for more information.
    #[inline]
    fn trace<
        F: FnOnce(Input::To<DomainTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Type,
                Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<DomainTracer<Self>>,
            >,
        Output: Parameterized<
                DomainTracer<Self>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        function: F,
        input_type: Input,
    ) -> Result<
        (
            Output::To<Self::Type>,
            Program<Self::Constant, Self::Operation, Input::To<Self::Constant>, Output::To<Self::Constant>>,
        ),
        ProgramError,
    > {
        DomainTracingContext::<Self>::trace(function, input_type)
    }

    /// Traces `function` for the provided input types in this [`Domain`]'s universe and returns only the inferred
    /// output types, without retaining the traced [`Program`]. This is the explicitly named counterpart of the
    /// [`infer_output_type`] function. Refer to its documentation for more information.
    #[inline]
    fn infer_output_type<
        F: FnOnce(Input::To<DomainTracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Type,
                Family: ParameterizedFamily<Self::Constant> + ParameterizedFamily<DomainTracer<Self>>,
            >,
        Output: Parameterized<
                DomainTracer<Self>,
                Family: ParameterizedFamily<Self::Type> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        function: F,
        input_type: Input,
    ) -> Result<Output::To<Self::Type>, ProgramError> {
        DomainTracingContext::<Self>::infer_output_type(function, input_type)
    }
}

impl<D: Domain> Trace for D {}

/// Traces `function` into a [`Program`] at the abstract signature of the provided `input` values (i.e., the analogue
/// of [JAX's `make_jaxpr`](https://docs.jax.dev/en/latest/_autosummary/jax.make_jaxpr.html)). The provided values
/// contribute only their abstract [`Type`]s; no runtime computation is performed on them, and they are not
/// captured by the resulting program. The trace runs in the input value type's statically known
/// [`ExecutionDomain`](Value::ExecutionDomain), and so, unlike [`batch`](crate::batch) and the differentiation entry
/// points, no context instance needs to be recovered from the input leaves and inputs with no leaf values are still
/// traceable. The trace invokes `function` once over [`DomainTracer`] inputs standing in for the input types through
/// a fresh [`DomainTracingContext`], and returns the inferred output types together with the finalized program.
/// [`Trace::trace`] exposes the same trace with the tracing universe named explicitly and the abstract input types
/// supplied directly.
#[inline]
pub fn trace<
    V: Value,
    F: FnOnce(InputType::To<DomainTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V::Type> = InputType, Family: ParameterizedFamily<V::Type>>,
    InputType: Parameterized<
            V::Type,
            Family: ParameterizedFamily<<V::ExecutionDomain as Domain>::Constant>
                        + ParameterizedFamily<DomainTracer<V::ExecutionDomain>>,
        >,
    Output: Parameterized<
            DomainTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V::Type> + ParameterizedFamily<<V::ExecutionDomain as Domain>::Constant>,
        >,
>(
    function: F,
    input: Input,
) -> Result<
    (
        Output::To<V::Type>,
        Program<
            <V::ExecutionDomain as Domain>::Constant,
            <V::ExecutionDomain as Domain>::Operation,
            InputType::To<<V::ExecutionDomain as Domain>::Constant>,
            Output::To<<V::ExecutionDomain as Domain>::Constant>,
        >,
    ),
    ProgramError,
> {
    V::ExecutionDomain::trace(function, input.map_parameters(|value| value.r#type().into_owned())?)
}

/// Traces `function` at the abstract signature of the provided `input` values and returns only the inferred
/// output types, without retaining the traced [`Program`] (i.e., the analogue of [JAX's `eval_shape`](
/// https://docs.jax.dev/en/latest/_autosummary/jax.eval_shape.html)). This is the output-type-only counterpart
/// of [`trace`], and [`Trace::infer_output_type`] exposes the same trace with the tracing universe named explicitly
/// and the abstract input types supplied directly.
#[inline]
pub fn infer_output_type<
    V: Value,
    F: FnOnce(InputType::To<DomainTracer<V::ExecutionDomain>>) -> Result<Output, ProgramError>,
    Input: Parameterized<V, To<V::Type> = InputType, Family: ParameterizedFamily<V::Type>>,
    InputType: Parameterized<
            V::Type,
            Family: ParameterizedFamily<<V::ExecutionDomain as Domain>::Constant>
                        + ParameterizedFamily<DomainTracer<V::ExecutionDomain>>,
        >,
    Output: Parameterized<
            DomainTracer<V::ExecutionDomain>,
            Family: ParameterizedFamily<V::Type> + ParameterizedFamily<<V::ExecutionDomain as Domain>::Constant>,
        >,
>(
    function: F,
    input: Input,
) -> Result<Output::To<V::Type>, ProgramError> {
    V::ExecutionDomain::infer_output_type(function, input.map_parameters(|value| value.r#type().into_owned())?)
}

impl<
    T: Type,
    V: Value<Type = T>,
    O: Clone + Operation<Type = T>,
    Input: Parameterized<V, Family: ParameterizedFamily<Tracer<TracingContext<V, O>>>, ParameterStructure: Debug + PartialEq>,
    Output: Parameterized<V, Family: ParameterizedFamily<Tracer<TracingContext<V, O>>>>,
> Program<V, O, Input, Output>
{
    /// Specializes this [`Program`] to the provided refined input types by replaying every
    /// [`Instruction`](crate::Instruction) into a fresh trace, so that type inference propagates the refinements
    /// through the entire body and the rebuilt program's boundary and instruction output types are exactly as if it had
    /// been traced at the refined types. The program is returned unchanged when the provided types equal its declared
    /// input types, and each provided type must refine the corresponding declared type (refer to the documentation of
    /// [`Type::is_refined_by`] for more information on refinement requirements). Anything else is rejected before the
    /// replay begins.
    ///
    /// Replay refines the boundary and every inferred instruction output type, but never rewrites stored operation
    /// payloads. An operation whose payload itself stores a type referencing a refined identity re-runs its own
    /// inference against the refined operands and surfaces its own diagnostic, while payloads that carry geometry
    /// as explicit operands specialize cleanly.
    pub fn specialize(self, input_types: &[T]) -> Result<Self, ProgramError> {
        check_count!("input", input_types, self.input_count(), ProgramError);
        let declared_input_types = self.input_types();
        if declared_input_types == input_types {
            return Ok(self);
        }
        for (declared, actual) in declared_input_types.iter().zip(input_types) {
            if !declared.is_refined_by(actual) {
                return Err(TypeError::invalid(format!(
                    "specialized input type {actual} does not refine declared input type {declared}",
                ))
                .into());
            }
        }
        let (builder, output_ids) = {
            let context = TracingContext::<V, O>::new();
            let builder = context.builder().clone();
            let inputs = Input::To::<Tracer<TracingContext<V, O>>>::from_parameters(
                self.input_structure().clone(),
                input_types.iter().cloned().map(|r#type| context.input(r#type)),
            )
            .map_err(ProgramError::from)?;
            let outputs = self.interpret_in_context(&context, inputs)?;
            let output_ids = outputs.parameters().map(Tracer::atom_id).collect::<Result<Vec<_>, _>>()?;
            (builder, output_ids)
        };
        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        builder.build(output_ids, self.input_structure.clone(), self.output_structure.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };
    use crate::axes::NamedAxes;
    use crate::captures::{CaptureReference, CapturingContext};
    use crate::contexts::EagerContext;
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::operations::{AddOperation, NegOperation, OneLike, OneOperation, Sin, ZeroLike, ZeroOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, Operation, ProgramError, RegionInterface, TypeError, Typed};

    use super::*;

    #[test]
    fn test_trace() {
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| Ok(x.clone() * x),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret(Array::scalar(3.0)), Ok(Array::scalar(9.0)));

        // The free function traces at the abstract signature of example values, which contribute only their types
        // (i.e., the resulting program neither captures nor depends on the example values themselves).
        let (output_type, program) = trace(|x| Ok(x.clone() * x), Array::scalar(2.0)).unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret(Array::scalar(3.0)), Ok(Array::scalar(9.0)));

        // Structured inputs trace at the structured signature of the example values.
        let (output_type, program) = trace(|(x, y)| Ok(x * y), (Array::scalar(2.0), Array::scalar(3.0))).unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret((Array::scalar(4.0), Array::scalar(5.0))), Ok(Array::scalar(20.0)));
    }

    #[test]
    fn test_context_trace_rejects_captures_registered_into_its_discarded_capture_table() {
        /// Capturing trace universe whose staged constants are capture references into a runtime `Array` table.
        type CapturingTrace =
            TracingContext<CaptureReference<ArrayType>, ArrayOperation<CaptureReference<ArrayType>>, Array>;

        /// Stages `x + capture#0` with the capture registered through the trace's own context.
        fn capturing_body(x: Tracer<CapturingTrace>) -> Result<Vec<Tracer<CapturingTrace>>, ProgramError> {
            let context = x.context().clone();
            let reference = context.capture(Array::scalar(3.0))?;
            let captured = StagingContext::constant(&context, reference);
            context.bind(AddOperation::new(), Vec::new(), &[x, captured])
        }

        // `trace` retains the traced program but discards the trace's local capture table, so the registered
        // capture would leave a dangling `capture#0` reference behind (silently aliasing whatever capture table
        // later surrounds the program) and the trace is rejected instead.
        let result = CapturingTrace::trace(capturing_body, ArrayType::scalar(DataType::F64));
        assert!(matches!(result, Err(ProgramError::DiscardedCaptures { count: 1 })));

        // `infer_output_type` discards the traced program together with the captures, and so the same body still
        // infers output types successfully (e.g., shape inference over functions that call captured compiled
        // functions remains valid).
        let output_types = CapturingTrace::infer_output_type(capturing_body, ArrayType::scalar(DataType::F64)).unwrap();
        assert_eq!(output_types, vec![ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_context_interpret_and_trace() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (output, program) =
            domain.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()?), Array::scalar(2.0)).unwrap();
        assert_eq!(output, Array::scalar(2.0 * 2.0 + 2.0f64.sin()));
        assert_eq!(program.interpret(Array::scalar(3.0)), Ok(Array::scalar(3.0 * 3.0 + 3.0f64.sin())));
    }

    #[test]
    fn test_context_infer_output_type() {
        let output_type = EagerContext::<Array, ArrayOperation<Array>>::infer_output_type(
            |x| Ok(x.sin()?),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));

        // The free function infers output types at the abstract signature of example values.
        let output_type = infer_output_type(|x| Ok(x.sin()?), Array::scalar(1.5)).unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
    }

    #[test]
    fn test_tracer_state_clone_debug_and_equality() {
        let live = TracerState::Live(AtomId::new(3));
        assert_eq!(live.clone(), TracerState::Live(AtomId::new(3)));
        assert_eq!(TracerState::Poison.clone(), TracerState::Poison);
        assert_ne!(live, TracerState::Poison);
        assert_eq!(format!("{live:?}"), "Live(AtomId { index: 3 })");
        assert_eq!(format!("{:?}", TracerState::Poison), "Poison");
    }

    #[test]
    fn test_tracer() {
        // Test handles, atom lookup, cloning, typing, and rendering.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let tracer = tracing_context.tracer(atom, None);
        let poisoned: Tracer<_> =
            Tracer::new(tracing_context.clone(), TracerState::Poison, ArrayType::scalar(DataType::F64));
        let cloned_tracer = tracer.clone();
        assert!(Rc::ptr_eq(tracer.builder(), &builder));
        assert_eq!(tracer.atom_id(), Ok(atom));
        assert_eq!(poisoned.atom_id(), Err(ProgramError::PoisonedValue));
        assert_eq!(cloned_tracer.state(), tracer.state());
        assert_eq!(cloned_tracer.r#type(), tracer.r#type());
        assert!(Rc::ptr_eq(cloned_tracer.builder(), &builder));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(DataType::F64)));
        assert_eq!(tracer.to_string(), "%0");
        assert_eq!(
            format!("{tracer:?}"),
            "Tracer { state: Live(AtomId { index: 0 }), type: ArrayType { data_type: F64, shape: Shape { dimensions: \
             [] }, layout: None, sharding: None, memory: Device }, .. }",
        );
        assert_eq!(poisoned.to_string(), "<poison:f64[]>");
        assert_eq!(
            format!("{poisoned:?}"),
            "Tracer { state: Poison, type: ArrayType { data_type: F64, shape: Shape { dimensions: [] }, layout: None, \
             sharding: None, memory: Device }, .. }",
        );

        // Test staging value-level identity helpers through the tracer convenience API.
        let zero = tracer.zero_like();
        let one = tracer.one_like();
        assert_eq!(zero.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(one.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let zero_atom = zero.atom_id().expect("zero_like output should remain live");
        let one_atom = one.atom_id().expect("one_like output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<Array, Vec<Array>>(vec![zero_atom, one_atom], Placeholder, vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0)), Ok(vec![Array::scalar(0.0), Array::scalar(1.0)]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero_like %0
                    %2:f64[] = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );

        // Test staging a unary operation through the tracer convenience API.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let tracer = tracing_context.tracer(atom, None);
        let output = tracer.unary(NegOperation::new());
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let output_atom = output.atom_id().expect("unary output should remain live");
        let program =
            builder.borrow().clone().build::<Array, Array>(vec![output_atom], Placeholder, Placeholder).unwrap();
        assert_eq!(program.interpret(Array::scalar(2.0)), Ok(Array::scalar(-2.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test staging a binary operation through the tracer convenience API.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let rhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let output = lhs.binary(&rhs, AddOperation::new());
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let output_atom = output.atom_id().expect("binary output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Array::scalar(2.0), Array::scalar(3.0))), Ok(Array::scalar(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test that binary operations poison the result when inputs belong to different builders.
        let context_a = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let context_b = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder_a = context_a.builder().clone();
        let atom_a = builder_a.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let atom_b = context_b.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let tracer_a = context_a.tracer(atom_a, None);
        let tracer_b = context_b.tracer(atom_b, None);
        let output = tracer_a.binary(&tracer_b, AddOperation::new());
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_tracer_unary_records_invalid_output_count_and_returns_poisoned_tracer() {
        #[derive(Copy, Clone, Debug)]
        struct NoOutputOperation;

        impl Operation for NoOutputOperation {
            type Type = ArrayType;

            #[inline]
            fn name(&self) -> &'static str {
                "no_output"
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                Ok(Vec::new())
            }
        }

        impl<C: Domain<Type = ArrayType, Value = Array>> InterpretableOperation<C> for NoOutputOperation {
            #[inline]
            fn interpret<D: InterpretationDriver<C>>(
                &self,
                _context: &C,
                _driver: &D,
                _inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                Ok(Vec::new())
            }
        }

        let context = TracingContext::<Array, NoOutputOperation>::new();
        let builder = context.builder().clone();
        let input_type = ArrayType::scalar(DataType::F64);
        let tracer = context.input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(
            builder.borrow().error().cloned(),
            Some(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_tracing_context() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();

        // Test construction, cloning, and debug formatting.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let cloned_context = tracing_context.clone();
        assert!(Rc::ptr_eq(tracing_context.builder(), &builder));
        assert!(Rc::ptr_eq(cloned_context.builder(), &builder));
        assert_eq!(format!("{tracing_context:?}"), "TracingContext { .. }");

        // Test creating a program constant in the staged program.
        let constant = tracing_context.constant(Array::scalar(2.5));
        assert_eq!(constant.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let constant_atom = constant.atom_id().expect("constant tracer should remain live");
        assert_eq!(constant_atom.index(), 0);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Array>, Array>(vec![constant_atom], Vec::<Placeholder>::new(), Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(Array::scalar(2.5)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = const 2.5
                in (%0)
            "}
            .trim_end(),
        );

        // Test constructing tracers from builder-owned and explicitly cached types.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let atom = tracing_context.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let builder_typed = tracing_context.tracer(atom, None);
        let cached_typed = tracing_context.tracer(atom, Some(ArrayType::scalar(DataType::F64)));
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(DataType::F64)));

        // Test that only the first recorded builder error is retained.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(tracing_context.error(first_error.clone()), first_error);
        assert_eq!(tracing_context.error(second_error), ProgramError::InvalidOutputCount { expected: 1, actual: 0 });
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        // Test staging a valid operation through the context.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let rhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let outputs = tracing_context.stage_operation(AddOperation::new(), Vec::new(), &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].state(), &TracerState::Live(AtomId::new(2)));
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Array::scalar(2.0), Array::scalar(3.0))), Ok(Array::scalar(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test rejecting inputs that belong to a different program builder.
        let context_a = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let context_b = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder_a = context_a.builder().clone();
        let atom_a = builder_a.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let atom_b = context_b.builder().borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let tracer_a = context_a.tracer(atom_a, None);
        let tracer_b = context_b.tracer(atom_b, None);
        assert!(matches!(
            context_a.stage_operation(AddOperation::new(), Vec::new(), &[&tracer_a, &tracer_b]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));

        // Test tracing after a builder failure by returning poisoned tracers when output types can still be inferred.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F64));
        let builder_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        builder.borrow_mut().error = Some(builder_error.clone());
        let tracer = tracing_context.tracer(atom, None);
        let outputs = tracing_context.stage_operation(NegOperation::new(), Vec::new(), &[&tracer]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), &TracerState::Poison));
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error.clone()));
        assert!(matches!(
            tracing_context.stage_operation(AddOperation::new(), Vec::new(), &[&tracer]),
            Err(ProgramError::Type(TypeError::Invalid { message })) if message == "expected 2 inputs but got 1",
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error));

        // Test propagating abstract-evaluation errors and recording them on the builder.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F8E3M4));
        let rhs_atom = builder.borrow_mut().add_input(ArrayType::scalar(DataType::F32));
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let result = tracing_context.stage_operation(AddOperation::new(), Vec::new(), &[&lhs, &rhs]);
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));
        assert!(matches!(
            builder.borrow().error().cloned(),
            Some(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));

        // Test staging program constants through the context without requiring the context itself to be a domain.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = tracing_context.builder().clone();
        let zero = tracing_context.constant(
            domain
                .bind(ZeroOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), &[])
                .unwrap()
                .into_iter()
                .next()
                .unwrap(),
        );
        let one = tracing_context.constant(
            domain
                .bind(OneOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), &[])
                .unwrap()
                .into_iter()
                .next()
                .unwrap(),
        );
        assert_eq!(zero.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(one.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let zero_atom = zero.atom_id().expect("zero tracer should remain live");
        let one_atom = one.atom_id().expect("one tracer should remain live");
        assert_eq!(zero_atom.index(), 0);
        assert_eq!(one_atom.index(), 1);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![zero_atom, one_atom],
                Vec::<Placeholder>::new(),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(vec![Array::scalar(0.0), Array::scalar(1.0)]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = const 0.0
                    %1:f64[] = const 1.0
                in (%0, %1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracing_context_trace() {
        let (output_type, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| Ok(x.clone() * x.clone() + x.one_like()),
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();
        assert_eq!(output_type, ArrayType::scalar(DataType::F64));
        assert_eq!(program.interpret(Array::scalar(3.0)), Ok(Array::scalar(10.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = one_like %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using an escaped `ProgramBuilder`.
        let escaped_builder = Rc::new(RefCell::new(None));
        assert!(matches!(
            EagerContext::<Array, ArrayOperation<Array>>::trace(
                |x| {
                    *escaped_builder.borrow_mut() = Some(x.builder().clone());
                    Ok(x)
                },
                ArrayType::scalar(DataType::F64),
            ),
            Err(ProgramError::EscapedProgramBuilder),
        ));

        // Test that `TypeError`s are returned in certain cases.
        assert!(matches!(
            EagerContext::<Array, ArrayOperation<Array>>::trace(
                |inputs| Ok(inputs.0 + inputs.1),
                (ArrayType::scalar(DataType::F8E3M4), ArrayType::scalar(DataType::F32)),
            ),
            Err(ProgramError::Type(TypeError::Invalid { message }))
                if message == "`add` input types are not broadcast-compatible",
        ));
    }

    #[test]
    fn test_tracing_context_interpret_and_trace() {
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (output, program) =
            domain.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()?), Array::scalar(2.0)).unwrap();
        assert_eq!(output, Array::scalar(2.0f64 * 2.0f64 + 2.0f64.sin()));
        assert_eq!(program.interpret(Array::scalar(0.5)), Ok(Array::scalar(0.5f64 * 0.5f64 + 0.5f64.sin())));
        assert_eq!(program.input_ids().len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                    %3:f64[] = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using a function with a tuple argument.
        let (_, compiled) = domain
            .interpret_and_trace(|(x, y)| Ok(x.clone() * y + x.sin()?), (Array::scalar(2.0), Array::scalar(3.0)))
            .unwrap();
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %0 %1
                    %3:f64[] = sin %0
                    %4:f64[] = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Test using a function that contains unused code.
        let (output, program) = domain
            .interpret_and_trace(
                |x| {
                    let _ = x.sin()?;
                    Ok(x.clone() * x)
                },
                Array::scalar(2.0),
            )
            .unwrap();
        assert_eq!(output, Array::scalar(4.0));
        assert_eq!(program.interpret(Array::scalar(0.5)), Ok(Array::scalar(0.25)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test tracing value-level identity helpers as ordinary operations.
        let (output, program) =
            domain.interpret_and_trace(|x| Ok((x.zero_like(), x.one_like())), Array::scalar(2.0)).unwrap();
        assert_eq!(output, (Array::scalar(0.0), Array::scalar(1.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero_like %0
                    %2:f64[] = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_nested_tracing_context() {
        // A nested trace over an eager `EagerContext<Array, ArrayOperation<Array>>` parent stages its own independent
        // primal program and, like the root `TracingContext`, shares that program's builder across cloned contexts.
        let nested = NestedTracingContext::new(EagerContext::<Array, ArrayOperation<Array>>::new());
        let builder = nested.builder().clone();
        let cloned_context = nested.clone();
        assert!(Rc::ptr_eq(nested.builder(), &builder));
        assert!(Rc::ptr_eq(cloned_context.builder(), &builder));
        assert_eq!(format!("{nested:?}"), "NestedTracingContext { .. }");

        // Staging an operation appends to the nested program, which interprets and renders exactly
        // as a root trace would.
        let lhs = nested.input(ArrayType::scalar(DataType::F64));
        let rhs = nested.input(ArrayType::scalar(DataType::F64));
        let outputs = nested.stage_operation(AddOperation::new(), Vec::new(), &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), ArrayType::scalar(DataType::F64));
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Array::scalar(2.0), Array::scalar(3.0))), Ok(Array::scalar(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Runtime captures are not owned by the nested context. They delegate to the enclosing capturing context,
        // so a value captured through the nested context lands in the parent's shared capture table.
        let capturing_parent = TracingContext::<CaptureReference<ArrayType>, NegOperation<ArrayType>, Array>::new();
        let nested = NestedTracingContext::new(capturing_parent.clone());
        let reference = nested.capture(Array::scalar(7.0)).expect("capture should delegate to the enclosing context");
        assert_eq!(reference.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(capturing_parent.captures().borrow().as_slice(), &[Array::scalar(7.0)]);
    }

    #[test]
    fn test_nested_tracing_context_trace() {
        // `trace` runs the closure once on tracer inputs standing in for the provided input types and finalizes
        // the staged flat program together with the closure's output structure.
        let (output_structure, program) = NestedTracingContext::trace(
            EagerContext::<Array, ArrayOperation<Array>>::new(),
            |inputs: Vec<Tracer<_>>| Ok(vec![inputs[0].clone() * inputs[0].clone(), inputs[0].sin()?]),
            vec![ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        assert_eq!(output_structure, vec![Placeholder, Placeholder]);
        assert_eq!(
            program.interpret(vec![Array::scalar(2.0)]),
            Ok(vec![Array::scalar(4.0), Array::scalar(2.0f64.sin())])
        );
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = mul %0 %0
                    %2:f64[] = sin %0
                in (%1, %2)
            "}
            .trim_end(),
        );

        // A tracer escaping the closure keeps the shared builder alive and is reported at the trace boundary.
        let escaped_tracer = Rc::new(RefCell::new(None));
        assert!(matches!(
            NestedTracingContext::trace(
                EagerContext::<Array, ArrayOperation<Array>>::new(),
                |inputs: Vec<Tracer<_>>| {
                    *escaped_tracer.borrow_mut() = Some(inputs[0].clone());
                    Ok(inputs)
                },
                vec![ArrayType::scalar(DataType::F64)],
            ),
            Err(ProgramError::EscapedProgramBuilder),
        ));

        // `trace_with_named_axes` seeds the nested context with axis bindings that named-axis readers resolve inside
        // the closure, while unseeded names keep delegating to the parent (an eager parent binds none).
        let (_, program) = NestedTracingContext::trace_with_named_axes(
            EagerContext::<Array, ArrayOperation<Array>>::new(),
            |inputs: Vec<Tracer<_>>| {
                assert_eq!(inputs[0].context().named_axis("model"), Some(NamedAxis::Batched { size: Some(4) }));
                assert_eq!(inputs[0].context().named_axis("unbound"), None);
                Ok(inputs)
            },
            vec![ArrayType::scalar(DataType::F64)],
            vec![("model".to_string(), NamedAxis::Batched { size: Some(4) })],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![Array::scalar(3.0)]), Ok(vec![Array::scalar(3.0)]));
    }

    #[test]
    fn test_program_specialize() {
        let variable = DimensionVariable::new("n", DimensionBounds::new(1, Some(5)).unwrap());
        let dynamic_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Dynamic(variable)]));
        let (_, program) =
            EagerContext::<Array, ArrayOperation<Array>>::trace(|x| Ok(x.clone() * x), dynamic_type.clone()).unwrap();

        // Equal input types take the fast path and return the program unchanged.
        let unchanged = program.clone().specialize(std::slice::from_ref(&dynamic_type)).unwrap();
        assert_eq!(unchanged.to_string(), program.to_string());

        // Refined input types replay the program so inference propagates the refinement through the entire body.
        let static_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(3)]));
        let specialized = program.clone().specialize(std::slice::from_ref(&static_type)).unwrap();
        assert_eq!(specialized.input_types(), vec![static_type.clone()]);
        assert_eq!(specialized.output_types(), vec![static_type.clone()]);
        assert_eq!(specialized.interpret(Array::vector(vec![2.0, 3.0, 4.0])), Ok(Array::vector(vec![4.0, 9.0, 16.0])));

        // Non-refining input types are rejected before the replay begins.
        let unrelated_type = ArrayType::new(DataType::F64, Shape::new(vec![Dimension::Static(7)]));
        let error = program.specialize(std::slice::from_ref(&unrelated_type)).unwrap_err();
        assert_eq!(
            error.to_string(),
            format!("specialized input type {unrelated_type} does not refine declared input type {dynamic_type}"),
        );
    }

    #[test]
    fn test_tracing_provenance_snapshot_and_clone_sharing() {
        let scope = ProvenanceScope::new("labeled");
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |x| {
                let context = x.context().clone();
                let scoped = context.invoke_with_provenance_scope(scope.clone(), || {
                    // A clone of the context observes the same active scope state.
                    assert_eq!(context.clone().provenance(), Provenance::scope(scope.clone(), Provenance::unknown()),);
                    x.clone() * x.clone()
                });
                Ok(scoped + x)
            },
            ArrayType::scalar(DataType::F64),
        )
        .unwrap();

        // Each staged instruction snapshots the provenance that was active when it was staged: the multiplication
        // carries the scope while the addition staged outside it stays unknown.
        let provenances = program
            .instructions()
            .iter()
            .map(|instruction| instruction.provenance().clone())
            .collect::<Vec<_>>();
        assert_eq!(provenances, vec![Provenance::scope(scope, Provenance::unknown()), Provenance::unknown()]);
    }

    #[test]
    fn test_tracing_provenance_independent_traces() {
        let scope = ProvenanceScope::new("scope");
        let first = TracingContext::<Array, ArrayOperation<Array>>::new();
        let second = TracingContext::<Array, ArrayOperation<Array>>::new();
        first.invoke_with_provenance_scope(scope.clone(), || {
            assert_eq!(first.provenance(), Provenance::scope(scope.clone(), Provenance::unknown()));
            // Independent traces own independent states, so scopes never leak across traces.
            assert!(second.provenance().is_unknown());
        });
        assert!(first.provenance().is_unknown());
    }

    #[test]
    fn test_eager_provenance_is_a_no_op() {
        // Terminal eager contexts run scope closures directly, record no provenance, and execute unaffected.
        let eager = EagerContext::<Array, ArrayOperation<Array>>::new();
        let sum = eager
            .invoke_with_provenance_scope(ProvenanceScope::new("scope"), || {
                assert!(eager.provenance().is_unknown());
                eager.bind(AddOperation::new(), Vec::new(), &[Array::scalar(1.0), Array::scalar(2.0)])
            })
            .unwrap();
        assert_eq!(sum, vec![Array::scalar(3.0)]);
    }

    #[test]
    fn test_nested_tracing_provenance_seeding() {
        let outer_scope = ProvenanceScope::new("outer");
        let nested_scope = ProvenanceScope::new("nested");
        let parent = TracingContext::<Array, ArrayOperation<Array>>::new();
        parent.invoke_with_provenance_scope(outer_scope.clone(), || {
            let seed = parent.provenance();

            // A nested tracing context seeds its origin from the parent's current provenance at depth zero of its
            // own scope stack, and owns independent scope state for the nested program.
            let nested = NestedTracingContext::new(parent.clone());
            assert_eq!(nested.provenance(), seed);
            nested.invoke_with_provenance_scope(nested_scope.clone(), || {
                assert_eq!(nested.provenance(), Provenance::scope(nested_scope.clone(), seed.clone()));
                assert_eq!(parent.provenance(), seed);
            });

            // Instructions staged in a nested trace record the seeded origin.
            let (_, nested_program) = NestedTracingContext::trace(
                parent.clone(),
                |inputs| Ok(vec![inputs[0].clone() + inputs[1].clone()]),
                vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
            )
            .unwrap();
            assert_eq!(nested_program.instructions()[0].provenance(), &seed);
        });
    }
}
