//! Contains machinery for _tracing_ Rust execution into typed [`Program`]s.
//!
//! Tracing substitutes symbolic [`Tracer`] values for runtime inputs and executes a Rust closure once. Operations
//! on those tracers bind through an active staging context, which appends typed instructions to a shared
//! [`ProgramBuilder`], and finalization validates the resulting atom graph and records the closure's structured
//! outputs as an immutable program.
//!
//! Tracing records the operations bound by the host execution path. It does not record arbitrary Rust. Dynamic behavior
//! must be represented by staged operations, and host branches execute during tracing only when their conditions are
//! concretely available.
//!
//! ```text
//!               ┌─────────────┐
//!               │ Input Types │
//!               └──────┬──────┘
//!                      │ create input tracers
//!                      ▼
//!     ┌─────────────────────────────────┐
//!     │ Tracing Context + Input Tracers │
//!     └────────────────┬────────────────┘
//!                      │ execute the Rust closure once, binding operations as instructions
//!                      ▼
//! ┌─────────────────────────────────────────┐
//! │ Output Tracers + Recorded Instructions  │
//! └────────────────────┬────────────────────┘
//!                      │ build
//!                      ▼
//!                 ┌─────────┐
//!                 │ Program │
//!                 └─────────┘
//! ```
//!
//! # Entry Points
//!
//! [`Trace::trace`] is the usual domain-oriented entry point. It creates a fresh [`DomainTracingContext`], converts
//! the abstract input structure into tracers, invokes the closure, and returns both output types and the finalized
//! program, while [`Trace::infer_output_type`] runs the same trace but retains only abstract outputs.
//!
//! Use [`TracingContext::trace`] directly when the value, operation, and capture types are named independently rather
//! than through a domain. [`TracingContext::trace_with_named_axes`] additionally installs named-axis bindings for the
//! trace, and [`NestedTracingContext::trace`] creates a fresh inner builder while retaining a parent context for
//! lifting, captures, and composition inside higher-order transforms.
//!
//! The free [`trace`] and [`infer_output_type`] functions expose the same entry points for callers holding example
//! values rather than abstract types: the values contribute only their types, and the trace runs in their statically
//! known execution domain.
//!
//! # Tracers and Errors
//!
//! A [`Tracer`] stores its context, abstract type, and [`TracerState`]. A live tracer names one [`AtomId`] in its
//! context's builder, and tracer equality is staging identity (i.e., same builder and same atom) and not runtime value
//! equality.
//!
//! Operator conveniences use poisoned tracers to defer a staging failure until the trace boundary. Once a bind fails,
//! downstream operations propagate [`TracerState::Poison`], and finalization returns the builder's original
//! [`ProgramError`]. Explicitly fallible staging paths may return errors immediately. Code must never use an atom ID
//! from one builder in another builder.
//!
//! # Tracing Contexts
//!
//! [`TracingContext`] is the ordinary root staging context. It owns shared handles to a program builder, a capture
//! table, and named-axis bindings. Its constant type may differ from the concrete captured value type, which lets
//! compiled domains store lifetime-free capture references in IR while retaining concrete runtime arrays separately.
//!
//! [`DomainTracingContext`] selects those generic parameters from a [`Domain`], and [`DomainTracer`] is its common
//! tracer alias. [`NestedTracingContext`] wraps a parent context but allocates a new builder, making it suitable for
//! tracing loop bodies, branches, derivative rules, and other nested programs without losing access to parent
//! capabilities. [`NestedTracer`] is the corresponding alias.
//!
//! # Captures and Context Composition
//!
//! A tracing context implementing [`CapturingContext`](crate::CapturingContext) registers a concrete runtime value in
//! its capture table and returns the staged constant payload referring to it. Transform and nested contexts delegate
//! capture registration to their parent, so a captured value follows the same context stack as ordinary operations.
//! Compilation uses this to build [`ClosedProgram`](crate::ClosedProgram)s without embedding runtime data in source IR.
//!
//! # Extending Tracing
//!
//! Most new operations need no tracing-specific implementation: implement the operation and its type inference, add it
//! to the domain's operation family, and stage it through [`Context::bind`]. Tracer capability implementations should
//! pass the concrete operation payload into the context and let the operation family's [`From`] conversion select the
//! wrapper variant.
//!
//! A new staging context should implement [`Context`] with `Value = Tracer<Self>` and then [`StagingContext`]. Keep
//! mutable builder state in shared handles, use checked builder APIs, make resolution conservative for foreign or
//! poisoned tracers, and finalize only after all context and tracer clones have been dropped.

use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::axes::NamedAxis;
use crate::contexts::{Context, Domain, StagingContext, ValueResolution};
use crate::macros::check_builders;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::programs::ProgramError;
use crate::programs::atoms::AtomId;
use crate::programs::builders::ProgramBuilder;
use crate::programs::operations::Operation;
use crate::programs::programs::Program;
use crate::programs::regions::BindingRegionDriver;
use crate::programs::types::Typed;
use crate::programs::values::Value;

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

/// Ordinary active tracing [`Context`] over a [`Type`](crate::Type)/[`Value`]/[`Operation`] universe.
/// [`TracingContext`] pairs the staged-constant and operation representations `(V, O)` of a program with the
/// [`ProgramBuilder`] used for one tracing invocation. It presents itself as a [`Domain`] whose [`Value`] is
/// [`Tracer<Self>`](Tracer) and whose [`Domain::Constant`] is `V`. Its default [`StagingContext::stage_operation`]
/// behavior records each primitive bind as a program instruction. Transform contexts wrap or replace this context when
/// they need different binding behavior, but they still share the same [`Context`] protocol used by [`Tracer`] values.
///
/// The optional capture parameter `C` names the concrete runtime value type stored in the capture table while tracing
/// a captured [`Program`]. It is deliberately distinct from the staged-constant type `V`.
/// [`capture`](crate::CapturingContext::capture) takes a runtime value of type `C` and returns a symbolic constant of
/// type `V` (i.e., a [`CaptureReference`](crate::CaptureReference)). In a capturing context the two genuinely differ.
/// For example, we might use a runtime device buffer for `C` versus a [`CaptureReference`](crate::CaptureReference)
/// for `V`. `C` defaults to `V` for the common non-capturing case, where no capture table exists and the distinction
/// is moot. Refer to [`CapturingContext`](crate::CapturingContext) and [`CaptureReference`](crate::CaptureReference)
/// for more information on what captures are and how they are used in practice.
pub struct TracingContext<V: Value, O: Operation<V::Type>, C = V> {
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
    /// is implemented only when the staged constant type is [`CaptureReference`](crate::CaptureReference), and so an
    /// ordinary trace can never push into this table. Refer to the documentation of
    /// [`CaptureReference`](crate::CaptureReference) for more information.
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
}

impl<V: Value, O: Operation<V::Type>, C> TracingContext<V, O, C> {
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
    /// that same context type.
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
    pub fn trace_with_named_axes<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
        named_axes: Vec<(String, NamedAxis)>,
    ) -> Result<(Output::To<V::Type>, Program<V, O, Input::To<V>, Output::To<V>>), ProgramError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_type.parameter_structure();

        let (output_types, outputs, output_structure) = {
            let context = Self {
                builder: builder.clone(),
                captures: Rc::new(RefCell::new(Vec::new())),
                named_axes: Rc::new(named_axes),
            };
            let input = input_type.map_parameters(|t| context.input(t)).map_err(ProgramError::from)?;
            let output = function(input).map_err(|e| builder.borrow_mut().error.take().unwrap_or_else(|| e))?;

            // The outputs must belong to this tracing context. A foreign tracer's atom ID would silently alias
            // whichever atom shares its index in this builder, and so we check for this here.
            check_builders!(&builder, [output.parameters().map(|output| output.builder())])?;

            builder.borrow_mut().error.take().map_or(Ok(()), Err)?;
            let output_structure = output.parameter_structure();
            let outputs = output.parameters().map(|o| o.atom_id()).collect::<Result<Vec<_>, _>>()?;
            let output_types = output.map_parameters(|o| o.r#type().into_owned()).map_err(ProgramError::from)?;
            (output_types, outputs, output_structure)
        };

        let builder = Rc::try_unwrap(builder).map_err(|_| ProgramError::EscapedProgramBuilder)?.into_inner();
        let program = builder.build(outputs, input_structure, output_structure)?;

        Ok((output_types, program))
    }

    /// Traces `function` against `input_type` and returns the output type, without retaining the traced [`Program`].
    /// Use this when callers only need the output types of an ordinary symbolic trace.
    #[inline]
    pub fn infer_output_type<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<V::Type, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<V::Type> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
    ) -> Result<Output::To<V::Type>, ProgramError> {
        Ok(Self::trace(function, input_type)?.0)
    }
}

impl<V: Value, O: Operation<V::Type>, C> Clone for TracingContext<V, O, C> {
    #[inline]
    fn clone(&self) -> Self {
        Self { builder: self.builder.clone(), captures: self.captures.clone(), named_axes: self.named_axes.clone() }
    }
}

impl<V: Value, O: Operation<V::Type>, C> Debug for TracingContext<V, O, C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<V: Value, O: Operation<V::Type>, C> Default for TracingContext<V, O, C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Value, O: Operation<V::Type>, C> Domain for TracingContext<V, O, C> {
    type Type = V::Type;
    type Value = Tracer<Self>;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<V::Type>, C> Context for TracingContext<V, O, C> {
    #[inline]
    fn lift(&self, constant: V) -> Result<Tracer<Self>, ProgramError> {
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
    fn resolve(&self, value: &Tracer<Self>) -> ValueResolution<V> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Concrete(constant.clone()),
            None => ValueResolution::Staged(atom_id),
        }
    }
}

impl<V: Value, O: Operation<V::Type>, C> StagingContext for TracingContext<V, O, C> {
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
}

impl<C: Context> NestedTracingContext<C> {
    /// Creates a new [`NestedTracingContext`] that owns a fresh [`ProgramBuilder`] and traces on behalf of `parent`.
    pub fn new(parent: C) -> Self {
        Self { parent, builder: Rc::new(RefCell::new(ProgramBuilder::new())), named_axes: Rc::new(Vec::new()) }
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
        let context = Self { parent, builder, named_axes };

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
        Self { parent: self.parent.clone(), builder: self.builder.clone(), named_axes: self.named_axes.clone() }
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
    fn resolve(&self, value: &Tracer<Self>) -> ValueResolution<C::Constant> {
        if !Rc::ptr_eq(self.builder(), value.context().builder()) {
            return ValueResolution::Opaque;
        }
        let Ok(atom_id) = value.atom_id() else {
            return ValueResolution::Opaque;
        };
        match self.builder().borrow().atoms().get(atom_id.index()).and_then(|atom| atom.as_constant()) {
            Some(constant) => ValueResolution::Concrete(constant.clone()),
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
/// contribute only their abstract [`Type`](crate::Type)s; no runtime computation is performed on them, and they are
/// not captured by the resulting program. The trace runs in the input value type's statically known
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
