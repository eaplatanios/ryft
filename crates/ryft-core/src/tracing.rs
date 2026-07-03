use std::borrow::Cow;
use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::rc::Rc;

use ryft_macros::Parameter;

use crate::compilation::captures::CaptureReference;
use crate::compilation::context::CapturingContext;
use crate::contexts::{Context, StagingContext, ValueResolution};
use crate::domains::Domain;
use crate::operations::Operation;
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::types::{Type, Typed};

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
///
/// The `Meta` type parameter carries transform-specific metadata alongside each staged value. Ordinary tracing leaves
/// it as the default `()`, while transform contexts can specialize it (e.g., to a partial-evaluation known/unknown
/// classification or a batch axis).
#[derive(Clone, Parameter)]
pub struct Tracer<C: Context, Meta = ()> {
    /// [`Context`] associated with this [`Tracer`].
    context: C,

    /// [`TracerState`] of this [`Tracer`].
    state: TracerState,

    /// [`Type`] of the value that this [`Tracer`] represents.
    r#type: C::Type,

    /// Transform-specific metadata carried alongside this [`Tracer`].
    meta: Meta,
}

impl<C: Context, Meta> Tracer<C, Meta> {
    /// Creates a new [`Tracer`] with a default `Meta`.
    #[inline]
    pub fn new(context: C, state: TracerState, r#type: C::Type) -> Self
    where
        Meta: Default,
    {
        Self { context, state, r#type, meta: Meta::default() }
    }

    /// Creates a new [`Tracer`] carrying the provided `meta`.
    #[inline]
    pub fn new_with_meta(context: C, state: TracerState, r#type: C::Type, meta: Meta) -> Self {
        Self { context, state, r#type, meta }
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

    /// Returns the transform-specific metadata carried by this [`Tracer`].
    #[inline]
    pub fn meta(&self) -> &Meta {
        &self.meta
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

// `Tracer` equality is *staging identity*, not value equality. Two tracers are equal if and only if they carry equal
// metadata and correspond to the same staged `Atom` of the same `ProgramBuilder` (or are both poisoned in the same
// builder). Two tracers that would evaluate to equal runtime values but were staged as distinct atoms are considered
// unequal, which is the conservative answer trace-time analyses need. For example, the loop invariance fixed points of
// the `scan` and `while` partial evaluation rules degrade to syntactic passthrough detection under a staging known-side
// context precisely because of these semantics.
impl<C: StagingContext, Meta: PartialEq> PartialEq for Tracer<C, Meta> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(self.context.builder(), other.context.builder())
            && self.state == other.state
            && self.meta == other.meta
    }
}

impl<C: StagingContext> Tracer<C, C::Meta> {
    /// Returns the [`ProgramBuilder`] associated with this [`Tracer`].
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>> {
        self.context.builder()
    }

    /// Applies the provided _unary_ [`Operation`] to this [`Tracer`] returning the resulting [`Tracer`]. _Unary_
    /// operations are operations that have a single input and a single output. If the provided operation is not a
    /// unary operation, then the resulting [`Tracer`] will contain a [`TracerState::Poison`].
    pub fn unary<P: Into<C::Operation>>(&self, operation: P) -> Self {
        let operation = operation.into();
        match self.context.stage_operation(operation, &[self]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
                Self {
                    state: TracerState::Poison,
                    r#type: self.r#type.clone(),
                    context: self.context.clone(),
                    meta: self.meta.clone(),
                }
            }
            Err(error) => {
                self.context.error(error);
                Self {
                    state: TracerState::Poison,
                    r#type: self.r#type.clone(),
                    context: self.context.clone(),
                    meta: self.meta.clone(),
                }
            }
        }
    }

    /// Applies the provided _binary_ [`Operation`] to this [`Tracer`] and the provided [`Tracer`] returning the
    /// resulting [`Tracer`]. _Binary_ operations are operations that have two inputs and a single output. If the
    /// provided operation is not a binary operation, then the resulting [`Tracer`] will contain a
    /// [`TracerState::Poison`].
    pub fn binary<P: Into<C::Operation>>(&self, rhs: &Self, operation: P) -> Self {
        let operation = operation.into();
        match self.context.stage_operation(operation, &[self, rhs]) {
            Ok(mut outputs) if outputs.len() == 1 => outputs.remove(0),
            Ok(outputs) => {
                self.context.error(ProgramError::InvalidOutputCount { expected: 1, actual: outputs.len() }.into());
                Self {
                    state: TracerState::Poison,
                    r#type: self.r#type.clone(),
                    context: self.context.clone(),
                    meta: self.meta.clone(),
                }
            }
            Err(error) => {
                self.context.error(error);
                Self {
                    state: TracerState::Poison,
                    r#type: self.r#type.clone(),
                    context: self.context.clone(),
                    meta: self.meta.clone(),
                }
            }
        }
    }
}

impl<C: Context, Meta: Debug> Debug for Tracer<C, Meta> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Tracer")
            .field("state", &self.state)
            .field("type", &self.r#type)
            .field("meta", &self.meta)
            .finish_non_exhaustive()
    }
}

impl<C: Context, Meta> Display for Tracer<C, Meta> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.state {
            TracerState::Live(atom_id) => write!(formatter, "{atom_id}"),
            TracerState::Poison => write!(formatter, "<poison:{}>", self.r#type),
        }
    }
}

impl<C: Context, Meta> Typed<C::Type> for Tracer<C, Meta> {
    #[inline]
    fn r#type(&self) -> Cow<'_, C::Type> {
        Cow::Borrowed(&self.r#type)
    }
}

impl<C: Context, Meta: Clone + Debug> Value<C::Type> for Tracer<C, Meta> {}

/// Ordinary active tracing [`Context`] over a [`Type`]/[`Value`]/[`Operation`] universe. [`TracingContext`] pairs the
/// type, staged-constant, and operation representations `(T, V, O)` of a program with the [`ProgramBuilder`] used for
/// one tracing invocation. It presents itself as a [`Domain`] whose [`Value`] is [`Tracer<Self>`](Tracer) and whose
/// [`Constant`](Domain::Constant) is `V`. Its default [`StagingContext::stage_operation`] behavior records each
/// primitive bind as a program instruction. Transform contexts wrap or replace this context when they need different
/// binding behavior, but they still share the same [`Context`] protocol used by [`Tracer`] values.
///
/// The optional capture parameter `C` names the concrete runtime value type stored in the capture table while tracing
/// a captured [`Program`]. It is deliberately distinct from the staged-constant type `V`.
/// [`capture`](CapturingContext::capture) takes a runtime value of type `C` and returns a symbolic constant of type
/// `V` (i.e., a [`CaptureReference`] reference). In a capturing context the two genuinely differ. For example, we might
/// use a runtime device buffer for `C` versus a [`CaptureReference`] reference for `V`. `C` defaults to `V` for the
/// common non-capturing case, where no capture table exists and the distinction is moot. Refer to [`CapturingContext`]
/// and [`CaptureReference`] for more information on what captures are and how they are used in practice.
pub struct TracingContext<T: Type, V: Value<T>, O: Operation<T>, C = V> {
    /// [`ProgramBuilder`] that owns the staged [`Program`] that is currently being traced. The builder is held behind
    /// an [`Rc`] rather than being outright owned because a single trace shares one builder across many contexts.
    /// A [`Tracer`] holds its [`Context`] by value, and tracing freely clones tracers (and hence their contexts) as
    /// values flow through the traced function, and so a cloned [`TracingContext`] must keep pointing at the *same*
    /// accumulating builder. Cloning the [`Rc`] does exactly that. An owned [`ProgramBuilder`] would instead be
    /// *forked* on every context clone, and so [`Tracer`]s created at different points in the trace would accumulate
    /// into divergent programs rather than the one program the trace is building. Furthermore, the nested [`RefCell`]
    /// supplies the interior mutability that staging needs.
    builder: Rc<RefCell<ProgramBuilder<T, V, O>>>,

    /// Capture table of closed-over runtime values, referenced symbolically from the staged [`Program`] via
    /// [`CaptureReference`]s. It stays empty for ordinary (i.e., non-capturing) tracing and is filled only when tracing
    /// a captured [`Program`] (e.g., when just-in-time-compiling a function that closes over device buffers), in which
    /// case those values are passed to the compiled program as runtime arguments rather than being baked into it.
    /// Capturing is gated at the type level: [`capture`](CapturingContext::capture) is implemented only when the staged
    /// constant type is [`CaptureReference`], and so an ordinary trace can never push into this table. Refer to the
    /// documentation of [`CaptureReference`] for more information.
    ///
    /// Like the [`builder`](Self::builder), the table is held behind an [`Rc`] and a [`RefCell`] for the same reason:
    /// one capturing trace shares a single table across its many cloned contexts, so the [`Rc`] keeps every clone
    /// pushing into the *same* accumulating table (which is what keeps [`CaptureReference`] indices consistent) instead
    /// of forking it, and the [`RefCell`] supplies the interior mutability [`capture`](CapturingContext::capture) needs
    /// to push through a shared `&self`.
    captures: Rc<RefCell<Vec<C>>>,
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> TracingContext<T, V, O, C> {
    /// Creates a new [`TracingContext`] over the `(T, V, O)` type universe with a fresh, empty [`ProgramBuilder`] and a
    /// fresh, empty capture table. Use [`builder`](Self::builder) afterward to read or finalize the staged program, and
    /// [`captures`](Self::captures) to read any values registered through [`capture`](CapturingContext::capture). To
    /// instead compose further staging onto a trace that already owns prior instructions, do not create a context at
    /// all: an input [`Tracer`]'s [`context`](Tracer::context) shares that trace's [`ProgramBuilder`], and so staging
    /// on it (e.g., via [`stage_operation`](StagingContext::stage_operation)) appends to the same program.
    #[inline]
    pub fn new() -> Self {
        Self {
            builder: Rc::new(RefCell::new(ProgramBuilder::<T, V, O>::new())),
            captures: Rc::new(RefCell::new(Vec::new())),
        }
    }

    /// Returns the [`ProgramBuilder`] that this [`TracingContext`] stages into.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<T, V, O>>> {
        &self.builder
    }

    /// Returns the shared capture table that [`capture`](CapturingContext::capture) fills while tracing. That table
    /// stays empty for ordinary traces, since [`capture`](CapturingContext::capture) is only implemented when the
    /// staged constant type is [`CaptureReference`].
    #[inline]
    pub fn captures(&self) -> &Rc<RefCell<Vec<C>>> {
        &self.captures
    }

    /// Traces `function` into a [`Program`] for the provided input types. This is the symbolic ordinary-tracing entry
    /// point. It creates a fresh [`TracingContext`] over the `(T, V, O)` type universe, executes `function` once on
    /// [`Tracer`] inputs standing in for `input_type`, and returns the output types plus the finalized program.
    /// Operation binds are handled by the context's [`StagingContext::stage_operation`] implementation. The type
    /// universe only supplies the staged constant and operation types used by that program. The capture parameter `C`
    /// is preserved on the staged [`Tracer`] leaves so that callers tracing in a context with a non-default capture
    /// type (such as a backend whose runtime [`Value`](Domain::Value) differs from its staged
    /// [`Constant`](Domain::Constant)) observe that same context type.
    pub fn trace<
        F: FnOnce(Input::To<Tracer<Self>>) -> Result<Output, ProgramError>,
        Input: Parameterized<T, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<T> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
    ) -> Result<(Output::To<T>, Program<T, V, O, Input::To<V>, Output::To<V>>), ProgramError> {
        let builder = Rc::new(RefCell::new(ProgramBuilder::new()));
        let input_structure = input_type.parameter_structure();
        let (output_types, outputs, output_structure) = {
            let context = Self { builder: builder.clone(), captures: Rc::new(RefCell::new(Vec::new())) };
            let input = input_type.map_parameters(|t| context.input(t)).map_err(ProgramError::from)?;
            let output = function(input).map_err(|e| builder.borrow_mut().error.take().unwrap_or_else(|| e))?;
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
        Input: Parameterized<T, Family: ParameterizedFamily<V> + ParameterizedFamily<Tracer<Self>>>,
        Output: Parameterized<Tracer<Self>, Family: ParameterizedFamily<T> + ParameterizedFamily<V>>,
    >(
        function: F,
        input_type: Input,
    ) -> Result<Output::To<T>, ProgramError> {
        Ok(Self::trace(function, input_type)?.0)
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> Clone for TracingContext<T, V, O, C> {
    fn clone(&self) -> Self {
        Self { builder: self.builder.clone(), captures: self.captures.clone() }
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> Debug for TracingContext<T, V, O, C> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("TracingContext").finish_non_exhaustive()
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> Default for TracingContext<T, V, O, C> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> Domain for TracingContext<T, V, O, C> {
    type Type = T;
    type Value = Tracer<Self>;
    type Constant = V;
    type Operation = O;
}

impl<T: Type, V: Value<T>, O: Operation<T>, C> Context for TracingContext<T, V, O, C> {
    #[inline]
    fn lift(&self, constant: V) -> Result<Tracer<Self>, ProgramError> {
        Ok(self.constant(constant))
    }

    #[inline]
    fn bind<P: Into<O>>(&self, operation: P, inputs: &[Tracer<Self>]) -> Result<Vec<Tracer<Self>>, ProgramError> {
        self.stage_operation(operation.into(), inputs)
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

impl<T: Type, V: Value<T>, O: Operation<T>, C> StagingContext for TracingContext<T, V, O, C> {
    type Meta = ();

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.builder
    }
}

impl<T: Type, O: Operation<T>, C: Value<T>> CapturingContext<C> for TracingContext<T, CaptureReference<T>, O, C> {
    #[inline]
    fn capture(&self, value: C) -> Result<Self::Constant, ProgramError> {
        let mut captures = self.captures.borrow_mut();
        let constant = CaptureReference::new(captures.len(), value.r#type().into_owned());
        captures.push(value);
        Ok(constant)
    }
}

/// Represents a nested [`TracingContext`] that is used to trace a closure into a [`Program`] expressed
/// in an *enclosing* [`Context`]'s universe rather than in a raw `(T, V, O)` type universe of its own.
/// Where [`TracingContext`] is keyed by the `(T, V, O)` types it stages and owns its own capture table,
/// [`NestedTracingContext`] is keyed by the enclosing [`Context`] `C`. It derives its [`Type`](Domain::Type),
/// [`Constant`](Domain::Constant), and [`Operation`](Domain::Operation) from `C`, owns a fresh [`ProgramBuilder`] for
/// the nested [`Program`] it stages, and holds a clone of `C`. Runtime capture registration is *not* owned by this
/// context but is rather delegated to the enclosing context through [`CapturingContext`], and so values captured while
/// tracing the nested program flow into `C`'s table along the same nesting path as ordinary operation staging. As with
/// [`TracingContext`], the [`ProgramBuilder`] is shared behind an [`Rc`] so cloned contexts keep appending to the
/// *same* nested program.
pub struct NestedTracingContext<C: Context> {
    /// [`Context`] that this [`NestedTracingContext`] is nested into.
    parent: C,

    /// [`ProgramBuilder`] that this [`NestedTracingContext`] stages the nested [`Program`] into.
    builder: Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>>,
}

impl<C: Context> NestedTracingContext<C> {
    /// Creates a new [`NestedTracingContext`] that owns a fresh [`ProgramBuilder`] and traces on behalf of `parent`.
    pub fn new(parent: C) -> Self {
        Self { parent, builder: Rc::new(RefCell::new(ProgramBuilder::new())) }
    }

    /// Returns the [`Context`] that this [`NestedTracingContext`] is nested into.
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }

    /// Returns the [`ProgramBuilder`] that this [`NestedTracingContext`] stages the nested [`Program`] into.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<C::Type, C::Constant, C::Operation>>> {
        &self.builder
    }
}

impl<C: Context> Clone for NestedTracingContext<C> {
    fn clone(&self) -> Self {
        Self { parent: self.parent.clone(), builder: self.builder.clone() }
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
    fn bind<P: Into<Self::Operation>>(
        &self,
        operation: P,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation(operation.into(), inputs)
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
    type Meta = ();

    #[inline]
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Type, Self::Constant, Self::Operation>>> {
        &self.builder
    }
}

/// [`TracingContext`] named by an enclosing [`Domain`] `D`'s associated types. This is `TracingContext` over `D`'s
/// type, staged constant, and [`Operation`] representations, and so it is the active tracing context that stages a
/// [`Program`] expressed in `D`'s universe. The optional capture parameter `C` defaults to `D`'s staged constant
/// representation, matching the default capture type of [`TracingContext::new`]. Closed program traces over a backend
/// whose runtime [`Value`](Domain::Value) differs from its staged [`Constant`](Domain::Constant) pin `C` to that
/// runtime value type explicitly. Use this alias at call sites that already hold a [`Domain`] and want to name the
/// matching tracing context. Use [`TracingContext`] directly at sites that already work in terms of a `(T, V, O)`
/// universe.
pub type DomainTracingContext<D, C = <D as Domain>::Constant> =
    TracingContext<<D as Domain>::Type, <D as Domain>::Constant, <D as Domain>::Operation, C>;

/// [`Tracer`] flowing through a [`DomainTracingContext`] for a backend [`Domain`] `D`. This is the value that stands in
/// for a `D`-typed runtime value while a function is being traced into a [`Program`]. Each [`Operation`] bound on these
/// tracers records a program instruction and yields further [`DomainTracer`]s, and so ordinary backend traces flow
/// entirely in them. The [`Domain`] is a pure type witness, and so the tracer borrows nothing from it. The backend-less
/// specialization used during symbolic program tracing and transposition is a [`Tracer`] over a plain
/// [`TracingContext<T, V, O>`](TracingContext).
pub type DomainTracer<D> = Tracer<DomainTracingContext<D>>;

/// [`Tracer`] flowing through a [`NestedTracingContext`] over an enclosing context `C`. This is the value used while
/// tracing a nested closure into a [`Program`] expressed in the enclosing context's universe. The closure receives
/// these tracers in place of `C`-typed runtime values, each [`Operation`] bound on them records an instruction in the
/// nested [`Program`], and the staged program is then interpreted, differentiated, transposed, etc. back in `C`. Use
/// this alias at call sites that trace a closure into a nested program over an enclosing context `C`.
pub type NestedTracer<C> = Tracer<NestedTracingContext<C>>;

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::rc::Rc;

    use indoc::indoc;
    use pretty_assertions::assert_eq;

    use crate::interpretation::InterpretableOperation;
    use crate::operations::Operation;
    use crate::operations::arithmetic::{AddOperation, NegOperation};
    use crate::operations::constants::{OneLike, OneOperation, ZeroLike, ZeroOperation};
    use crate::operations::scalars::ScalarOperation;
    use crate::operations::trigonometric::Sin;
    use crate::parameters::Placeholder;
    use crate::programs::{AtomId, ProgramBuilder, ProgramError};
    use crate::scalars::{Scalar, ScalarDomain};
    use crate::types::{DataType, TypeError, Typed};

    use super::*;

    #[test]
    fn test_trace() {
        let (output_type, program) = ScalarDomain::trace(|x| Ok(x.clone() * x), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(Scalar::from(3.0)), Ok(Scalar::from(9.0)));
    }

    #[test]
    fn test_interpret_and_trace() {
        let domain = ScalarDomain::new();
        let (output, program) =
            domain.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()?), Scalar::from(2.0)).unwrap();
        assert_eq!(output, 2.0 * 2.0 + 2.0f64.sin());
        assert_eq!(program.interpret(Scalar::from(3.0)), Ok(Scalar::from(3.0 * 3.0 + 3.0f64.sin())));
    }

    #[test]
    fn test_infer_output_type() {
        let output_type = ScalarDomain::infer_output_type(|x| Ok(x.sin()?), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
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
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracer = tracing_context.tracer(atom, None);
        let poisoned: Tracer<_> = Tracer::new(tracing_context.clone(), TracerState::Poison, DataType::F64);
        let cloned_tracer = tracer.clone();
        assert!(Rc::ptr_eq(tracer.builder(), &builder));
        assert_eq!(tracer.atom_id(), Ok(atom));
        assert_eq!(poisoned.atom_id(), Err(ProgramError::PoisonedValue));
        assert_eq!(cloned_tracer.state(), tracer.state());
        assert_eq!(cloned_tracer.r#type(), tracer.r#type());
        assert!(Rc::ptr_eq(cloned_tracer.builder(), &builder));
        assert!(matches!(tracer.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert_eq!(tracer.to_string(), "%0");
        assert_eq!(format!("{tracer:?}"), "Tracer { state: Live(AtomId { index: 0 }), type: F64, meta: (), .. }");
        assert_eq!(poisoned.to_string(), "<poison:f64>");
        assert_eq!(format!("{poisoned:?}"), "Tracer { state: Poison, type: F64, meta: (), .. }");

        // Test staging value-level identity helpers through the tracer convenience API.
        let zero = tracer.zero_like();
        let one = tracer.one_like();
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero_like output should remain live");
        let one_atom = one.atom_id().expect("one_like output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<Scalar, Vec<Scalar>>(vec![zero_atom, one_atom], Placeholder, vec![Placeholder, Placeholder])
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(2.0)), Ok(vec![Scalar::from(0.0), Scalar::from(1.0)]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );

        // Test staging a unary operation through the tracer convenience API.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let tracer = tracing_context.tracer(atom, None);
        let output = tracer.unary(NegOperation);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("unary output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<Scalar, Scalar>(vec![output_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(2.0)), Ok(Scalar::from(-2.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = neg %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test staging a binary operation through the tracer convenience API.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let output = lhs.binary(&rhs, AddOperation);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        let output_atom = output.atom_id().expect("binary output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Scalar, Scalar), Scalar>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.0))), Ok(Scalar::from(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test that binary operations poison the result when inputs belong to different builders.
        let context_a = DomainTracingContext::<ScalarDomain>::new();
        let context_b = DomainTracingContext::<ScalarDomain>::new();
        let builder_a = context_a.builder().clone();
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = context_b.builder().borrow_mut().add_input(DataType::F64);
        let tracer_a = context_a.tracer(atom_a, None);
        let tracer_b = context_b.tracer(atom_b, None);
        let output = tracer_a.binary(&tracer_b, AddOperation);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_tracer_unary_records_invalid_output_count_and_returns_poisoned_tracer() {
        #[derive(Copy, Clone, Debug)]
        struct NoOutputOperation;

        impl Operation<DataType> for NoOutputOperation {
            #[inline]
            fn name(&self) -> &'static str {
                "no_output"
            }

            fn infer_output_types(&self, _input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
                Ok(Vec::new())
            }
        }

        impl<C> InterpretableOperation<DataType, Scalar, C> for NoOutputOperation {
            #[inline]
            fn interpret(&self, _context: &C, _inputs: &[Scalar]) -> Result<Vec<Scalar>, ProgramError> {
                Ok(Vec::new())
            }
        }

        let context = TracingContext::<DataType, Scalar, NoOutputOperation>::new();
        let builder = context.builder().clone();
        let input_type = DataType::F64;
        let tracer = context.input(input_type);
        let output = tracer.unary(NoOutputOperation);
        assert!(matches!(output.state(), TracerState::Poison));
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(
            builder.borrow().error().cloned(),
            Some(ProgramError::InvalidOutputCount { expected: 1, actual: 0 }),
        );
    }

    #[test]
    fn test_tracing_context() {
        let domain = ScalarDomain::new();

        // Test construction, cloning, and debug formatting.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let cloned_context = tracing_context.clone();
        assert!(Rc::ptr_eq(tracing_context.builder(), &builder));
        assert!(Rc::ptr_eq(cloned_context.builder(), &builder));
        assert_eq!(format!("{tracing_context:?}"), "TracingContext { .. }");

        // Test creating a concrete constant in the staged program.
        let constant = tracing_context.constant(Scalar::from(2.5));
        assert_eq!(constant.r#type().into_owned(), DataType::F64);
        let constant_atom = constant.atom_id().expect("constant tracer should remain live");
        assert_eq!(constant_atom.index(), 0);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Scalar>(vec![constant_atom], Vec::<Placeholder>::new(), Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(Scalar::from(2.5)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                in (%0)
            "}
            .trim_end(),
        );

        // Test constructing tracers from builder-owned and explicitly cached types.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let atom = tracing_context.builder().borrow_mut().add_input(DataType::F64);
        let builder_typed = tracing_context.tracer(atom, None);
        let cached_typed = tracing_context.tracer(atom, Some(DataType::F64));
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));

        // Test that only the first recorded builder error is retained.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(tracing_context.error(first_error.clone()), first_error);
        assert_eq!(tracing_context.error(second_error), ProgramError::InvalidOutputCount { expected: 1, actual: 0 });
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        // Test staging a valid operation through the context.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F64);
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let outputs = tracing_context.stage_operation(AddOperation, &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].state(), &TracerState::Live(AtomId::new(2)));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Scalar, Scalar), Scalar>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.0))), Ok(Scalar::from(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Test rejecting inputs that belong to a different program builder.
        let context_a = DomainTracingContext::<ScalarDomain>::new();
        let context_b = DomainTracingContext::<ScalarDomain>::new();
        let builder_a = context_a.builder().clone();
        let atom_a = builder_a.borrow_mut().add_input(DataType::F64);
        let atom_b = context_b.builder().borrow_mut().add_input(DataType::F64);
        let tracer_a = context_a.tracer(atom_a, None);
        let tracer_b = context_b.tracer(atom_b, None);
        assert!(matches!(
            context_a.stage_operation(AddOperation, &[&tracer_a, &tracer_b]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder_a.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));

        // Test tracing after a builder failure by returning poisoned tracers when output types can still be inferred.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let atom = builder.borrow_mut().add_input(DataType::F64);
        let builder_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        builder.borrow_mut().error = Some(builder_error.clone());
        let tracer = tracing_context.tracer(atom, None);
        let outputs = tracing_context.stage_operation(NegOperation, &[&tracer]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(matches!(outputs[0].state(), &TracerState::Poison));
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error.clone()));
        assert!(matches!(
            tracing_context.stage_operation(AddOperation, &[&tracer]),
            Err(ProgramError::Type(TypeError { message })) if message == "expected 2 inputs but got 1",
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(builder_error));

        // Test propagating abstract-evaluation errors and recording them on the builder.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let lhs_atom = builder.borrow_mut().add_input(DataType::F8E3M4);
        let rhs_atom = builder.borrow_mut().add_input(DataType::F32);
        let lhs = tracing_context.tracer(lhs_atom, None);
        let rhs = tracing_context.tracer(rhs_atom, None);
        let result = tracing_context.stage_operation(AddOperation, &[&lhs, &rhs]);
        assert!(matches!(
            result,
            Err(ProgramError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));
        assert!(matches!(
            builder.borrow().error().cloned(),
            Some(ProgramError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible"
        ));

        // Test staging concrete constants through the context without requiring the context itself to be a domain.
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let zero = tracing_context
            .constant(domain.bind(ZeroOperation::new(DataType::F64), &[]).unwrap().into_iter().next().unwrap());
        let one = tracing_context
            .constant(domain.bind(OneOperation::new(DataType::F64), &[]).unwrap().into_iter().next().unwrap());
        assert_eq!(zero.r#type().into_owned(), DataType::F64);
        assert_eq!(one.r#type().into_owned(), DataType::F64);
        let zero_atom = zero.atom_id().expect("zero tracer should remain live");
        let one_atom = one.atom_id().expect("one tracer should remain live");
        assert_eq!(zero_atom.index(), 0);
        assert_eq!(one_atom.index(), 1);
        let program = builder
            .borrow()
            .clone()
            .build::<Vec<Scalar>, Vec<Scalar>>(
                vec![zero_atom, one_atom],
                Vec::<Placeholder>::new(),
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        assert_eq!(program.interpret(Vec::new()), Ok(vec![Scalar::from(0.0), Scalar::from(1.0)]));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda  .
                let %0:f64 = const
                    %1:f64 = const
                in (%0, %1)
            "}
            .trim_end(),
        );

        // Test staging an existing program through the context, including lifting embedded constants.
        let mut builder = ProgramBuilder::<DataType, Scalar, ScalarOperation<Scalar>>::new();
        let input = builder.add_input(DataType::F64);
        let constant = builder.add_constant(Scalar::from(4.0));
        let output = builder.add_instruction(AddOperation, vec![input, constant]).unwrap()[0];
        let program = builder.build::<Scalar, Scalar>(vec![output], Placeholder, Placeholder).unwrap();
        let tracing_context = DomainTracingContext::<ScalarDomain>::new();
        let builder = tracing_context.builder().clone();
        let input = tracing_context.input(DataType::F64);
        let outputs = tracing_context.stage_program(&program, vec![input]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("staged program output should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<Scalar, Scalar>(vec![output_atom], Placeholder, Placeholder)
            .unwrap();
        assert_eq!(program.interpret(Scalar::from(3.0)), Ok(Scalar::from(7.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = const
                    %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_tracing_context_trace() {
        let (output_type, program) =
            ScalarDomain::trace(|x| Ok(x.clone() * x.clone() + x.one_like()), DataType::F64).unwrap();
        assert_eq!(output_type, DataType::F64);
        assert_eq!(program.interpret(Scalar::from(3.0)), Ok(Scalar::from(10.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = one_like %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using an escaped [`ProgramBuilder`].
        let escaped_builder = Rc::new(RefCell::new(None));
        assert!(matches!(
            ScalarDomain::trace(
                |x| {
                    *escaped_builder.borrow_mut() = Some(x.builder().clone());
                    Ok(x)
                },
                DataType::F64,
            ),
            Err(ProgramError::EscapedProgramBuilder),
        ));

        // Test that [`TypeError`]s are returned in certain cases.
        assert!(matches!(
            ScalarDomain::trace(|inputs| Ok(inputs.0 + inputs.1), (DataType::F8E3M4, DataType::F32)),
            Err(ProgramError::Type(TypeError { message }))
                if message == "add input types are not broadcast-compatible",
        ));
    }

    #[test]
    fn test_tracing_context_interpret_and_trace() {
        let domain = ScalarDomain::new();
        let (output, program) =
            domain.interpret_and_trace(|x| Ok(x.clone() * x.clone() + x.sin()?), Scalar::from(2.0)).unwrap();
        assert_eq!(output, 2.0f64 * 2.0f64 + 2.0f64.sin());
        assert_eq!(program.interpret(Scalar::from(0.5)), Ok(Scalar::from(0.5f64 * 0.5f64 + 0.5f64.sin())));
        assert_eq!(program.input_ids().len(), 1);
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                    %2:f64 = sin %0
                    %3:f64 = add %1 %2
                in (%3)
            "}
            .trim_end(),
        );

        // Test using a function with a tuple argument.
        let (_, compiled) = domain
            .interpret_and_trace(|(x, y)| Ok(x.clone() * y + x.sin()?), (Scalar::from(2.0), Scalar::from(3.0)))
            .unwrap();
        assert_eq!(
            compiled.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = mul %0 %1
                    %3:f64 = sin %0
                    %4:f64 = add %2 %3
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
                Scalar::from(2.0),
            )
            .unwrap();
        assert_eq!(output, 4.0);
        assert_eq!(program.interpret(Scalar::from(0.5)), Ok(Scalar::from(0.25)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = mul %0 %0
                in (%1)
            "}
            .trim_end(),
        );

        // Test tracing value-level identity helpers as ordinary operations.
        let (output, program) =
            domain.interpret_and_trace(|x| Ok((x.zero_like(), x.one_like())), Scalar::from(2.0)).unwrap();
        assert_eq!(output, (Scalar::from(0.0), Scalar::from(1.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64 .
                let %1:f64 = zero_like %0
                    %2:f64 = one_like %0
                in (%1, %2)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_nested_tracing_context() {
        // A nested trace over an eager `ScalarDomain` parent stages its own independent primal program and,
        // like the root `TracingContext`, shares that program's builder across cloned contexts.
        let nested = NestedTracingContext::new(ScalarDomain::new());
        let builder = nested.builder().clone();
        let cloned_context = nested.clone();
        assert!(Rc::ptr_eq(nested.builder(), &builder));
        assert!(Rc::ptr_eq(cloned_context.builder(), &builder));
        assert_eq!(format!("{nested:?}"), "NestedTracingContext { .. }");

        // Staging an operation appends to the nested program, which interprets and renders exactly
        // as a root trace would.
        let lhs = nested.input(DataType::F64);
        let rhs = nested.input(DataType::F64);
        let outputs = nested.stage_operation(AddOperation, &[&lhs, &rhs]).unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].r#type().into_owned(), DataType::F64);
        let output_atom = outputs[0].atom_id().expect("output tracer should remain live");
        let program = builder
            .borrow()
            .clone()
            .build::<(Scalar, Scalar), Scalar>(vec![output_atom], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.0))), Ok(Scalar::from(5.0)));
        assert_eq!(
            program.to_string(),
            indoc! {"
                lambda %0:f64, %1:f64 .
                let %2:f64 = add %0 %1
                in (%2)
            "}
            .trim_end(),
        );

        // Runtime captures are not owned by the nested context. They delegate to the enclosing capturing context,
        // so a value captured through the nested context lands in the parent's shared capture table.
        let capturing_parent = TracingContext::<DataType, CaptureReference<DataType>, NegOperation, Scalar>::new();
        let nested = NestedTracingContext::new(capturing_parent.clone());
        let reference = nested.capture(Scalar::from(7.0)).expect("capture should delegate to the enclosing context");
        assert_eq!(reference.r#type().into_owned(), DataType::F64);
        assert_eq!(capturing_parent.captures().borrow().as_slice(), &[Scalar::from(7.0)]);
    }
}
