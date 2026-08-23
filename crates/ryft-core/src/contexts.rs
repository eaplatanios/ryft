//! Defines the passive [`Domain`] vocabulary and active [`Context`] protocol used by interpretation, tracing, and
//! program transforms.
//!
//! A domain selects one coherent family of [`Type`]s, flowing [`Value`]s, stored constants, and [`Operation`]s. It says
//! what a program may contain, but not what applying an operation does. A context adds that behavior: it lifts stored
//! constants into flowing values, binds operations, resolves values conservatively, and reports whether its innermost
//! semantics execute eagerly. Keeping those roles separate lets one program universe participate in eager execution,
//! staging, batching, differentiation, partial evaluation, and nested combinations of those transforms. Refer to the
//! documentation of [`Context`] for a rendered diagram of this relationship.
//!
//! # Where Contexts Participate
//!
//!   - Ordinary [`trace`](crate::trace) creates a [`TracingContext`] whose values are symbolic [`Tracer`]s and whose
//!     binds append instructions to a new [`ProgramBuilder`]. [`Trace::trace`] selects the same path through a named
//!     domain, while [`infer_output_type`](crate::tracing::infer_output_type) uses it only for abstract outputs.
//!   - [`Program::interpret`] constructs an [`EagerContext`] and replays a program immediately.
//!     [`Program::interpret_in_context`] accepts an explicit context, so the same replay can instead stage into an
//!     enclosing program or pass through a transform.
//!   - Value-level transform entry points construct their own contexts and feed transform-specific values through the
//!     same binding protocol. Nested transforms compose by wrapping one context around another.
//!
//! # Domains and Contexts
//!
//! [`Domain`] is descriptive. Its associated types form the closed vocabulary shared by a program and every compatible
//! context. It does not own an active trace, execute operations, or carry mutable builder state.
//!
//! [`Context`] is behavioral. It is a cloneable handle to one active binding semantics. Stateful contexts share their
//! mutable builders, error slots, or transform state behind internal handles so that values may retain cloned contexts
//! without forking the active computation.
//!
//! # Binding Protocol
//!
//! [`Context::lift`] translates a stored [`Domain::Constant`] into the context's flowing [`Domain::Value`].
//! [`Context::bind`] applies one operation to flowing inputs and receives an application-scoped [`BindingRegionDriver`]
//! containing the operation's ordered attached computations. Eager contexts expose those regions to interpretation
//! rules, staging contexts import or intern them into the destination program, and transform contexts recursively
//! rewrite them according to the operation's rule.
//!
//! [`Context::resolve`] is a conservative identity query, not a general concretization API. It may prove that a value
//! is a stored constant or a live staged atom, but callers must preserve an [`Opaque`](ValueResolution::Opaque) value
//! when neither fact is safe to establish. [`Context::is_eager`] describes the innermost semantics that ultimately
//! handles binding, even when several transform contexts wrap it.
//!
//! # Context Families and Composition
//!
//! [`EagerContext`] binds immediately through [`InterpretableOperation`] and ordinarily lifts constants by identity.
//! [`StagingContext`] refines [`Context`] for symbolic [`Tracer`] values, exposes the active [`ProgramBuilder`], and
//! records each bind as an instruction. Transform contexts intercept binds, propagate transform-specific values, and
//! delegate rewritten primitive work to a parent context.
//!
//! Delegation makes contexts a semantic stack. A differentiation context may delegate residual tangent work through a
//! partial-evaluation context, which may in turn delegate primitive operations to a tracing context. The outer layers
//! determine the rewrite, while the innermost eager or staging context determines whether the final operation executes
//! now or becomes part of a program.
//!
//! # Composite Domains and Projection
//!
//! [`ProjectedContext`] presents one member kind of a composite domain as an ordinary context while forwarding every
//! lift, bind, resolution, and eagerness query to the composite parent. It preserves symbolic identity and creates no
//! second interpreter or program. Projection is intentionally limited to region-free member operations because an
//! attached region may mix several member kinds and therefore belongs to the composite operation contract.
//!
//! # Value Resolution
//!
//! [`ValueResolution`] reports exactly what the resolving context can prove:
//!
//!   - [`ValueResolution::Constant`] carries a payload that may be embedded as a program constant
//!     but is not necessarily host-inspectable.
//!   - [`ValueResolution::Staged`] carries the value's [`AtomId`] in the active builder.
//!   - [`ValueResolution::Opaque`] proves neither fact and requires conservative handling.
//!
//! A rule must never infer concreteness, builder ownership, or staged identity from an opaque result.
//!
//! # Extending Contexts
//!
//! Define a new backend or program universe by implementing [`Domain`] and one closed operation family. Implement
//! [`Context`] only for a concrete binding semantics. A staging implementation uses `Value = Tracer<Self>`, implements
//! [`StagingContext`], and stores builder and error state in shared handles. A transform context wraps the narrowest
//! parent context it needs, defines its flowing value, and applies operation-owned rules in [`Context::bind`]. Capture
//! registration and compatible capabilities delegate to the parent.

use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::interpretation::{EagerInterpretationDriver, InterpretableOperation};
use crate::macros::check_builders;
use crate::operations::ConstantOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{
    AtomId, BindingRegionDriver, EagerInterpretationValidation, Operation, OperationProjection, Program,
    ProgramBuilder, ProgramError, Type, Typed, Value, ValueProjection,
};
use crate::tracing::{Trace, Tracer, TracerState, TracingContext};

/// Passive type/value universe shared by program interpretation, tracing, and transformation contexts. A [`Domain`]
/// selects the [`Type`], flowing [`Value`], stored constant, and [`Operation`] families that may participate in one
/// program universe. It carries no active builder or binding behavior. Applying operations and lifting stored constants
/// into flowing values are responsibilities of [`Context`] implementations over that universe.
pub trait Domain: Sized {
    /// [`Type`]s that this [`Domain`] uses to represent the abstract metadata associated with its [`Value`]s.
    /// A commonly used [`Type`] is [`ArrayType`](crate::ArrayType), though scalar-only domains can use
    /// [`DataType`](crate::DataType) and richer backends may use richer type representations.
    type Type: Type;

    /// [`Value`] types supported by this [`Domain`]. Instances of this type are what [`Program`] interpretation and
    /// eager transforms operate on. [`Domain::Type`] represents abstract staging metadata, while [`Domain::Value`]
    /// represents the runtime values that inhabit traced programs during execution.
    type Value: Value<Type = Self::Type>;

    /// Constant payload type stored in traced [`Program`]s for this [`Domain`]. For eager domains this is usually the
    /// same type as [`Domain::Value`]. Compiled backends may use a lifetime-free abstract representation here while
    /// reserving [`Domain::Value`] for concrete runtime values.
    type Constant: Value<Type = Self::Type>;

    /// [`Operation`] representation supported by this [`Domain`] for ordinary traced [`Program`]s.
    type Operation: Operation<Type = Self::Type>;
}

/// Active binding semantics layered on a passive [`Domain`]. A [`Context`] decides how stored constants become flowing
/// values, what applying an operation means, what identity information can be recovered from a value, and whether the
/// innermost binding semantics execute eagerly. Eager contexts interpret operations immediately, staging contexts
/// record instructions, and transform contexts apply operation-owned rules before delegating rewritten work to a
/// parent.
///
/// # Context Model
///
/// ```mermaid
/// %%{init: {"themeCSS": ".nodeLabel code { white-space: nowrap !important; }"}}%%
/// flowchart TD
///   domain["Domain"] --> families["Type, Value, Constant, and Operation Families"]
///   domain --> context["Context"]
///   context --> lift["&lt;code&gt;lift&lt;/code&gt;: Stored Constant to Flowing Value"]
///   context --> bind["&lt;code&gt;bind&lt;/code&gt;: Apply Operation with Attached Regions"]
///   context --> resolve["&lt;code&gt;resolve&lt;/code&gt;: Constant, Staged, or Opaque"]
///   context --> eager_query["&lt;code&gt;is_eager&lt;/code&gt;: Innermost Execution Semantics"]
///   bind --> eager["Eager Binding"]
///   bind --> staging["Staging Binding"]
///   bind --> transform["Transform Binding"]
///   eager --> interpretation["InterpretableOperation Rule"]
///   staging --> builder["Append Instruction to Program Builder"]
///   transform --> rule["Apply Operation-Owned Transform Rule"]
///   rule --> parent["Delegate Rewritten Primitive Work to Parent Context"]
///   parent --> terminal["Eventually Execute Eagerly or Stage an Instruction"]
/// ```
///
/// All contexts are cloneable handles. Stateful implementations must ensure that clones share one active builder or
/// transform state rather than fork it.
#[cfg_attr(doc, aquamarine::aquamarine)]
pub trait Context: Domain + Clone {
    /// Lifts a staged [`Program`] constant into this [`Context`]'s runtime value representation. Most eager contexts
    /// use the same representation for [`Domain::Constant`] and [`Domain::Value`], and so this is just an identity
    /// function. Backends that use abstract, lifetime-free constants for compiled programs can either materialize
    /// a runtime value here when that is semantically valid, or return an error when an abstract constant cannot
    /// be interpreted as a concrete runtime value. [`StagingContext`]s lift constants by recording them as constant
    /// [`Tracer`]s.
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError>;

    /// Applies an [`Operation`] in this [`Context`] and returns its output values. The context determines what applying
    /// the operation means: [`EagerContext`]s interpret it over concrete values, [`StagingContext`]s record an
    /// [`Instruction`](crate::Instruction) in their [`ProgramBuilder`], and transform contexts apply the corresponding
    /// transformation rule before forwarding rewritten operations to a parent context.
    ///
    /// Nested computations belong to an operation application rather than to the operation payload itself. They are
    /// therefore supplied alongside the operation and its inputs through `driver`, which may provide owned regions,
    /// borrowed regions from a replayed program, or shared callee programs through one ordered
    /// [`Region`](crate::Region) namespace. Eager and transform contexts make those regions available to the
    /// operation's interpretation or transform driver. Staging contexts import or intern them into the destination
    /// program and attach their [`RegionId`](crate::RegionId)s to the recorded instruction. The resulting sequence
    /// must match the number and order returned by [`Operation::region_slots`].
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] being applied.
    ///   - `driver`: Application-scoped [`BindingRegionDriver`] providing the complete ordered
    ///     [`Region`](crate::Region)s supplied by this [`Operation`] application. Binding consumes the driver because
    ///     staging may import or intern its roots into the destination [`Program`].
    ///   - `inputs`: Input values supplied as the operation's operands, in operation-defined order.
    fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError>;

    /// Returns `true` if this [`Context`] is *eager* meaning that its [`bind`](Self::bind) function computes
    /// concrete values immediately so that concretizing extractions such as
    /// [`Concretizable::concretize`](crate::Concretizable::concretize) on values it produces can succeed and, for
    /// example, the trip count of a data-dependent loop is decidable while differentiating. Transform contexts that
    /// wrap other contexts (e.g., batching and forward-mode differentiation) delegate to the wrapped context, so that
    /// the answer reflects the innermost context that actually executes the bound operations.
    fn is_eager(&self) -> bool;

    /// Resolves the provided value in this [`Context`]. Refer to [`ValueResolution`] for the possible
    /// [`ValueResolution`]s and their semantics.
    #[inline]
    fn resolve(&self, value: &Self::Value) -> ValueResolution<Self::Constant> {
        let _ = value;
        ValueResolution::Opaque
    }

    /// Traces `function` into a [`Program`] and interprets that program on the provided `input`. This creates an
    /// ordinary symbolic trace over this context's `(Self::Type, Self::Constant, Self::Operation)` universe through a
    /// fresh [`TracingContext`], simplifies the resulting flat program, and interprets it with the provided concrete
    /// input values. Use this when a caller needs both the staged program and the corresponding concrete output for
    /// the same input. The runtime values flow through `self`, which supplies the concrete value type and the
    /// constant-lifting behavior.
    fn interpret_and_trace<
        F: FnOnce(Input::To<Tracer<TracingContext<Self::Constant, Self::Operation>>>) -> Result<Output, ProgramError>,
        Input: Parameterized<
                Self::Value,
                Family: ParameterizedFamily<Self::Constant>
                            + ParameterizedFamily<Tracer<TracingContext<Self::Constant, Self::Operation>>>,
            >,
        Output: Parameterized<
                Tracer<TracingContext<Self::Constant, Self::Operation>>,
                Family: ParameterizedFamily<Self::Value> + ParameterizedFamily<Self::Constant>,
            >,
    >(
        &self,
        function: F,
        input: Input,
    ) -> Result<
        (
            Output::To<Self::Value>,
            Program<Self::Constant, Self::Operation, Input::To<Self::Constant>, Output::To<Self::Constant>>,
        ),
        ProgramError,
    > {
        let input_structure = input.parameter_structure();
        let input_values = input.into_parameters().collect::<Vec<_>>();
        let input_types = input_values.iter().map(|value| value.r#type().into_owned()).collect::<Vec<_>>();
        let mut output_structure = None;
        let (_, flat_program) = Self::trace(
            |flat_input| {
                let input = <Input::To<Tracer<TracingContext<Self::Constant, Self::Operation>>>>::from_parameters(
                    input_structure.clone(),
                    flat_input,
                )?;
                let output = function(input)?;
                output_structure = Some(output.parameter_structure());
                Ok(output.into_parameters().collect::<Vec<_>>())
            },
            input_types,
        )?;
        let output_structure = output_structure.unwrap();
        let flat_program = flat_program.into_simplified()?;
        let output_values = flat_program.interpret_in_context(self, input_values)?;
        let output = Output::To::<Self::Value>::from_parameters(output_structure.clone(), output_values)?;
        let program = flat_program.restructured(input_structure, output_structure)?;
        Ok((output, program))
    }
}

/// Zero-sized [`Context`] used for a concrete `(type, value, operation)` universe whose operations can be interpreted
/// directly without backend-owned context state. Values may themselves carry explicit resources, such as reference
/// holders, but all state needed to interpret an operation arrives through those values and the attached regions.
/// [`EagerContext`] makes this direct interpretation mode explicit in generic code that otherwise has no backend-owned
/// eager context value to pass around.
///
/// The default operation family is [`ConstantOperation`], which is the minimal operation family needed by ordinary
/// eager value contexts that only materialize constants and expose context capabilities such as zero, one, fill, etc.
/// Code that binds or batches a richer operation family should still specify `O` explicitly, such as
/// `EagerContext<V, ArrayOperation<V>>`.
pub struct EagerContext<V: Value, O: Operation<Type = V::Type> = ConstantOperation<V>> {
    /// [`PhantomData`] marker tying this zero-sized context to its associated types.
    marker: PhantomData<fn() -> (V, O)>,
}

impl<V: Value, O: Operation<Type = V::Type>> EagerContext<V, O> {
    /// Creates a new [`EagerContext`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Copy for EagerContext<V, O> {}

impl<V: Value, O: Operation<Type = V::Type>> Clone for EagerContext<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Value, O: Operation<Type = V::Type>> std::fmt::Debug for EagerContext<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("EagerContext")
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Default for EagerContext<V, O> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Domain for EagerContext<V, O> {
    type Type = V::Type;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<Type = V::Type> + InterpretableOperation<Self>> Context for EagerContext<V, O> {
    #[inline]
    fn lift(&self, constant: V) -> Result<V, ProgramError> {
        Ok(constant)
    }

    #[inline]
    fn bind<P: Into<O>, D: BindingRegionDriver<V, O>>(
        &self,
        operation: P,
        driver: D,
        inputs: &[V],
    ) -> Result<Vec<V>, ProgramError> {
        let operation = operation.into();
        operation.validate_region_count(driver.region_count())?;
        // A resource-bearing value family requires the complete attached region closure to be validated before its
        // eager rule runs. A driver carrying evidence was already covered by its root's boundary validation. Otherwise,
        // this bind is itself a validation boundary. Either way, the eager rule receives fresh evidence so that nested
        // replay does not revalidate the regions it selects.
        let validation = if V::VALIDATES_EAGER_INTERPRETATION {
            if driver.eager_interpretation_validation().is_none() {
                for region in driver.regions() {
                    V::validate_eager_interpretation(region)?;
                }
            }
            Some(EagerInterpretationValidation::new())
        } else {
            None
        };
        operation.interpret(self, &EagerInterpretationDriver::new(&driver, validation), inputs)
    }

    #[inline]
    fn is_eager(&self) -> bool {
        true
    }

    #[inline]
    fn resolve(&self, value: &V) -> ValueResolution<V> {
        ValueResolution::Constant(value.clone())
    }
}

/// [`Context`] adapter that presents one member kind of a composite parent context as an ordinary context of that
/// member kind. [`Program`]s that mix several value kinds store one composite [`Type`] / [`Value`] / [`Operation`]
/// universe, but most operations, transform rules, and capability implementations are written against exactly one
/// member kind (e.g., array-only rules bind through a context whose values are `Value<Type = ArrayType>`). Without an
/// adapter, every such rule would need a second, composite-typed implementation. `ProjectedContext<C, T>` is that
/// adapter. It *is* a context of the `T`-typed member (its [`Domain`] associated types are the member value, constant,
/// and operation families selected by [`ValueProjection<T>`] and [`OperationProjection<T>`]), while every actual
/// effect happens in the composite parent `C`.
///
/// [`bind`](Context::bind) works by round-tripping through the parent. It lifts each member operand into the composite
/// value family (i.e., using [`ValueProjection::from_projected`]), lifts the member operation into the composite
/// operation family (i.e., using the [`From`] super-trait of [`OperationProjection`]), binds *once* through the parent
/// context, and projects the results back to the member kind (i.e., using [`ValueProjection::into_projected`]). Staged
/// member values are identity-preserving views of composite tracers and so the [`Instruction`](crate::Instruction)
/// recorded in the parent consumes the original Single Static Assignment (SSA) [`Atom`](crate::Atom)s (i.e., projection
/// adds no copies, no new instructions, and no indirection in the staged program).
///
/// The adapter is deliberately zero-state. It stores its parent and nothing else. It never inspects a staged program,
/// never reconstructs dependencies, and never carries dimensions, source arrays, identity mappings, or replay
/// substitutions. Every dependency of a bound operation must therefore arrive as an explicit operand, which keeps
/// the program graph the sole source of data dependencies.
///
/// Note that projected binding is intentionally limited to [`Region`](crate::Region)-free member operations. A region
/// can carry values of several member kinds at once, so higher-order operations own composite region contracts instead
/// of projecting them; a bound operation that declares or receives regions is rejected.
pub struct ProjectedContext<C: Domain, T: Type> {
    /// Composite parent [`Domain`] or [`Context`].
    parent: C,

    /// [`PhantomData`] marker tying this context to its associated type.
    marker: PhantomData<fn() -> T>,
}

impl<C: Domain, T: Type> ProjectedContext<C, T> {
    /// Creates a [`ProjectedContext`] view of `parent`.
    #[inline]
    pub const fn new(parent: C) -> Self {
        Self { parent, marker: PhantomData }
    }

    /// Returns the parent [`Domain`] or [`Context`].
    #[inline]
    pub fn parent(&self) -> &C {
        &self.parent
    }
}

impl<C: Domain + Clone, T: Type> Clone for ProjectedContext<C, T> {
    #[inline]
    fn clone(&self) -> Self {
        Self::new(self.parent.clone())
    }
}

impl<C: Domain, T: Type> Domain for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
{
    type Type = T;
    type Value = <C::Value as ValueProjection<T>>::Projected;
    type Constant = <C::Constant as ValueProjection<T>>::Projected;
    type Operation = <C::Operation as OperationProjection<T>>::Projected;
}

impl<C: Context, T: Type> Context for ProjectedContext<C, T>
where
    C::Value: ValueProjection<T, Projected: Value<Type = T>>,
    C::Constant: ValueProjection<T, Projected: Value<Type = T>>,
    C::Operation: OperationProjection<T>,
{
    #[inline]
    fn lift(&self, constant: Self::Constant) -> Result<Self::Value, ProgramError> {
        self.parent
            .lift(<C::Constant as ValueProjection<T>>::from_projected(constant))?
            .into_projected()
            .map_err(Into::into)
    }

    fn bind<O: Into<Self::Operation>, D: BindingRegionDriver<Self::Constant, Self::Operation>>(
        &self,
        operation: O,
        driver: D,
        inputs: &[Self::Value],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();
        if !operation.region_slots().is_empty() || driver.region_count() != 0 {
            return Err(ProgramError::MalformedProgram(format!(
                "projected operation `{}` cannot carry regions",
                operation.name(),
            )));
        }

        let operation: C::Operation = operation.into();
        let regions = Vec::<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>::new();

        // `Context::bind` borrows its inputs while embedding a projected member consumes it, so clone each member
        // representation to construct temporary composite values for the parent. Keep the common nullary, unary, and
        // binary cases on the stack; only uncommon wider homogeneous operations allocate an input vector. Symbolic
        // projected values retain their parent value and therefore preserve SSA identity through this clone. Concrete
        // eager values ordinarily dispatch through their native member contexts rather than through this adapter.
        let outputs = match inputs {
            [] => self.parent.bind(operation, regions, &[]),
            [input] => {
                let inputs = [<C::Value as ValueProjection<T>>::from_projected(input.clone())];
                self.parent.bind(operation, regions, &inputs)
            }
            [left, right] => {
                let inputs = [
                    <C::Value as ValueProjection<T>>::from_projected(left.clone()),
                    <C::Value as ValueProjection<T>>::from_projected(right.clone()),
                ];
                self.parent.bind(operation, regions, &inputs)
            }
            inputs => {
                let inputs =
                    inputs.iter().cloned().map(<C::Value as ValueProjection<T>>::from_projected).collect::<Vec<_>>();
                self.parent.bind(operation, regions, inputs.as_slice())
            }
        }?;

        outputs
            .into_iter()
            .map(<C::Value as ValueProjection<T>>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .map_err(Into::into)
    }

    #[inline]
    fn is_eager(&self) -> bool {
        self.parent.is_eager()
    }

    #[inline]
    fn resolve(&self, value: &Self::Value) -> ValueResolution<Self::Constant> {
        // Resolution must ask the parent about the original composite value. For the symbolic values that use this
        // context through `ProjectedValue`, cloning preserves the same underlying tracer or transform state.
        let value = <C::Value as ValueProjection<T>>::from_projected(value.clone());
        match self.parent.resolve(&value) {
            ValueResolution::Constant(constant) => match constant.into_projected() {
                Ok(constant) => ValueResolution::Constant(constant),
                Err(_) => ValueResolution::Opaque,
            },
            ValueResolution::Staged(atom) => ValueResolution::Staged(atom),
            ValueResolution::Opaque => ValueResolution::Opaque,
        }
    }
}

/// Staging [`Context`] whose flowing [`Domain::Value`] is a [`Tracer`] into an active [`ProgramBuilder`].
/// Binding records [`Operation`] invocations as [`Program`] [`Instruction`](crate::Instruction)s rather than
/// interpreting them, and this trait owns the builder-dependent staging API: [`constant`](StagingContext::constant),
/// [`input`](StagingContext::input), [`tracer`](StagingContext::tracer), [`error`](StagingContext::error),
/// [`stage_nullary_operation`](StagingContext::stage_nullary_operation), and
/// [`stage_operation`](StagingContext::stage_operation). Ordinary and nested tracing implement it through
/// [`TracingContext`] and [`NestedTracingContext`](crate::NestedTracingContext), respectively. Transform contexts have
/// their own flowing value types and delegate rewritten operations to a parent staging context instead of implementing
/// this trait themselves.
///
/// The flowing value is pinned to [`Tracer<Self>`](Tracer). Every staging context records operation invocations
/// as [`Program`] instructions and hands back [`Tracer`]s standing in for their results.
pub trait StagingContext: Context<Value = Tracer<Self>> {
    /// Returns the shared [`ProgramBuilder`] owned by this [`StagingContext`].
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Constant, Self::Operation>>>;

    /// Creates a constant [`Tracer`] in this context with the provided constant payload. Note that this is the
    /// raw staging primitive and deliberately performs no constant-storage validation. It is meant to be used
    /// by transform infrastructure that stages values it constructed itself. [`Context::lift`] validates
    /// [`Value::validate_as_constant`] at the trace boundary instead, and [`Region`](crate::Region) sealing
    /// re-checks every stored constant at [`build`](ProgramBuilder::build) time as the backstop.
    #[inline]
    fn constant(&self, value: Self::Constant) -> Self::Value {
        let r#type = value.r#type().into_owned();
        let atom = self.builder().borrow_mut().add_constant(value);
        self.tracer(atom, Some(r#type))
    }

    /// Creates an input [`Tracer`] in this context with the provided type.
    #[inline]
    fn input(&self, r#type: Self::Type) -> Self::Value {
        let atom = self.builder().borrow_mut().add_input(r#type.clone());
        self.tracer(atom, Some(r#type))
    }

    /// Constructs a _live_ [`Tracer`] in this context referring to the provided [`AtomId`]. If the provided
    /// `r#type` is [`None`], the staged [`Atom`](crate::programs::Atom)'s type is read from the owned
    /// [`ProgramBuilder`].
    #[inline]
    fn tracer(&self, atom: AtomId, r#type: Option<Self::Type>) -> Self::Value {
        let r#type = r#type.unwrap_or_else(|| self.builder().borrow().atoms()[atom.index()].r#type().into_owned());
        Tracer::new(self.clone(), TracerState::Live(atom), r#type)
    }

    /// Records the provided [`ProgramError`] in the underlying [`ProgramBuilder`] and returns it. If the underlying
    /// [`ProgramBuilder`] already has an error recorded, then it is left unchanged and this function acts simply as
    /// an identity function.
    #[inline]
    fn error(&self, error: ProgramError) -> ProgramError {
        let mut builder = self.builder().borrow_mut();
        if builder.error.is_none() {
            builder.error = Some(error.clone());
        }
        error
    }

    /// Stages an application of the provided **nullary** [`Operation`] (i.e., an operation with no inputs) in this
    /// [`StagingContext`] and returns [`Tracer`]s for its outputs.
    #[inline]
    fn stage_nullary_operation<O: Into<Self::Operation>>(
        &self,
        operation: O,
    ) -> Result<Vec<Self::Value>, ProgramError> {
        self.stage_operation::<O, _, Self::Value>(operation, [], &[])
    }

    /// Stages an application of the provided [`Operation`] in this [`StagingContext`] and returns [`Tracer`]s for
    /// its outputs. The application-scoped [`BindingRegionDriver`] provides the operation's complete ordered region
    /// sequence. Staging consumes that driver through [`BindingRegionDriver::import_into`] to obtain destination
    /// [`RegionId`](crate::RegionId)s (owned region programs are imported, replayed regions preserve source-arena
    /// sharing, and shared callees are interned by [`Rc`] identity). The resulting identifiers are recorded on the new
    /// [`Instruction`](crate::Instruction) in the same order.
    ///
    /// # Parameters
    ///
    ///   - `operation`: [`Operation`] being staged.
    ///   - `driver`: [`BindingRegionDriver`] scoped to this [`Operation`] application.
    ///   - `inputs`: Input [`Tracer`]s, in [`Operation`]-defined order.
    fn stage_operation<
        O: Into<Self::Operation>,
        D: BindingRegionDriver<Self::Constant, Self::Operation>,
        I: std::borrow::Borrow<Self::Value>,
    >(
        &self,
        operation: O,
        driver: D,
        inputs: &[I],
    ) -> Result<Vec<Self::Value>, ProgramError> {
        let operation = operation.into();

        // Every live input atom is local to exactly one builder. Validate that ownership before inspecting regions or
        // mutating this trace so a foreign tracer cannot alias an unrelated atom with the same numeric identifier.
        check_builders!(self.builder(), [inputs.iter().map(|input| input.borrow().context().builder())])
            .map_err(|error| self.error(error))?;

        // Region input identities are instantiated from the caller operand types before the regions enter this builder.
        // First validate the operation's complete attachment declaration so later zips cannot silently omit either a
        // region or its instantiation request. Region-free operations avoid collecting input types here as the checked
        // builder path will infer them directly from its atoms.
        let declared_region_count = operation.region_slots().len();
        let (input_types, region_input_types) = if declared_region_count == 0 {
            operation.validate_region_count(driver.region_count()).map_err(|error| self.error(error))?;
            (None, Vec::new())
        } else {
            let region_interfaces = driver.regions().map(|region| region.interface()).collect::<Vec<_>>();
            operation.validate_region_count(region_interfaces.len()).map_err(|error| self.error(error))?;
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let region_input_types = operation
                .infer_region_input_types(input_types.as_slice(), region_interfaces.as_slice())
                .map_err(|error| self.error(error.into()))?;
            if region_input_types.len() != declared_region_count {
                return Err(self.error(ProgramError::MalformedProgram(format!(
                    "operation `{}` returned {} region instantiation entries for {} attached regions",
                    operation.name(),
                    region_input_types.len(),
                    declared_region_count,
                ))));
            }
            (Some(input_types), region_input_types)
        };

        if self.builder().borrow().error.is_some() {
            // A previously failed trace cannot append atoms, regions, or instructions, but callers still need correctly
            // typed poison tracers so tracing can continue to the boundary that reports the original error. Reconstruct
            // each hypothetical instantiated interface without importing it, then run ordinary output inference.
            let input_types = input_types
                .unwrap_or_else(|| inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>());
            let region_interfaces = driver
                .regions()
                .zip(&region_input_types)
                .map(|(region, input_types)| match input_types {
                    Some(input_types) => {
                        Ok(region.to_program().with_instantiated_type_identities(input_types)?.interface())
                    }
                    None => Ok(region.interface()),
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let output_types = operation.infer_output_types(input_types.as_slice(), region_interfaces.as_slice())?;
            Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(self.clone(), TracerState::Poison, r#type))
                .collect())
        } else {
            // Only a healthy trace consumes the driver and mutates the builder. Importing with the operation-derived
            // input types makes each attached region boundary agree with this invocation before `add_instruction`
            // derives its output types and records the final instruction.
            let inputs = match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<Vec<_>, _>>() {
                Ok(input_atom_ids) => input_atom_ids,
                Err(error) => return Err(self.error(error)),
            };
            let region_ids =
                driver.import_into(self.builder(), &region_input_types).map_err(|error| self.error(error))?;
            let outputs = {
                let mut builder = self.builder().borrow_mut();
                match builder.add_instruction(operation, region_ids, inputs) {
                    Ok(outputs) => outputs.to_vec(),
                    Err(error) => {
                        if builder.error.is_none() {
                            builder.error = Some(error.clone());
                        }
                        return Err(error);
                    }
                }
            };
            Ok(outputs.into_iter().map(|atom| self.tracer(atom, None)).collect::<Vec<_>>())
        }
    }
}

/// Resolution of a flowing [`Domain::Value`] against a [`Context`], as returned by [`Context::resolve`].
/// It represents what, if anything, the context can prove that the value denotes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ValueResolution<V> {
    /// The value denotes this program-constant payload. Eager [`Context`]s whose flowing values and constants use the
    /// same representation always resolve to [`ValueResolution::Constant`]. [`StagingContext`]s resolve here only for
    /// *literal-backed* [`Tracer`]s whose staged [`Atom`](crate::Atom) is a constant atom in the context's
    /// [`ProgramBuilder`]. This resolution guarantees that the payload can be embedded as a program constant.
    Constant(V),

    /// The value is a live, staged, value identified by this [`AtomId`] in the resolving context's [`Program`], with no
    /// concrete payload until the traced program runs. The carried [`AtomId`] is a stable identity for the value within
    /// that program.
    Staged(AtomId),

    /// The resolving context can prove nothing about the value. This is the conservative default, and the answer that
    /// [`StagingContext`]s provide for poisoned [`Tracer`]s and for values belonging to different [`ProgramBuilder`]s.
    Opaque,
}

impl<V> ValueResolution<V> {
    /// Returns `true` if this [`ValueResolution`] is [`Constant`](Self::Constant).
    #[inline]
    pub fn is_constant(&self) -> bool {
        matches!(self, Self::Constant(_))
    }

    /// Returns the program-constant payload of this [`ValueResolution`] if it is a [`Constant`](Self::Constant)
    /// resolution, and [`None`] otherwise.
    #[inline]
    pub fn into_constant(self) -> Option<V> {
        match self {
            Self::Constant(constant) => Some(constant),
            _ => None,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::borrow::Cow;
    use std::fmt::Display;
    use std::sync::Arc;

    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayOperation, ArrayType, DataType, Dimension, DimensionBounds, DimensionVariable, Shape,
    };
    use crate::differentiation::{
        DifferentiableOperation, DifferentiableType, DifferentiationDriver, DifferentiationDual, DifferentiationError,
        DifferentiationTracer, TransposableOperation, TranspositionDriver,
    };
    use crate::interpretation::{InterpretableOperation, InterpretationDriver};
    use crate::macros::check_count;
    use crate::operations::{
        AddOperation, CompareOperation, ComparisonDirection, NegOperation, OneOperation, WhileOperation, ZeroOperation,
    };
    use crate::parameters::{Parameter, Placeholder};
    use crate::partial::{PartialTracer, PartialValue};
    use crate::programs::{
        Atom, AtomId, CalleeRegionDriver, MaybeZero, NoIdentity, OperationProjection, ProgramBuilder, ProgramError,
        ProjectedValue, RegionInterface, Type, TypeError, Typed, ValueProjection,
    };
    use crate::tracing::{DomainTracingContext, Tracer, TracerState, TracingContext};

    use super::*;

    /// Test-only homogeneous member type used by the generic projected-context fixtures.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct ProjectedMemberType<const MEMBER: u8>;

    impl<const MEMBER: u8> Display for ProjectedMemberType<MEMBER> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "member_{MEMBER}")
        }
    }

    impl<const MEMBER: u8> Parameter for ProjectedMemberType<MEMBER> {}

    impl<const MEMBER: u8> Type for ProjectedMemberType<MEMBER> {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl<const MEMBER: u8> DifferentiableType for ProjectedMemberType<MEMBER> {
        fn is_zero_space(&self) -> bool {
            false
        }

        fn tangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }

        fn cotangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }
    }

    /// Test-only concrete value for one homogeneous projected member.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct ProjectedMemberValue<const MEMBER: u8>(pub(crate) usize);

    impl<const MEMBER: u8> Display for ProjectedMemberValue<MEMBER> {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "{}", self.0)
        }
    }

    impl<const MEMBER: u8> Parameter for ProjectedMemberValue<MEMBER> {}

    impl<const MEMBER: u8> Typed for ProjectedMemberValue<MEMBER> {
        type Type = ProjectedMemberType<MEMBER>;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Owned(ProjectedMemberType)
        }
    }

    impl<const MEMBER: u8> Value for ProjectedMemberValue<MEMBER> {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    /// Test-only composite storage type with three distinct member kinds.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramType {
        /// First member kind, used by the ordinary projection tests.
        First(ProjectedMemberType<0>),

        /// Second member kind, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberType<1>),

        /// Third member kind, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberType<2>),
    }

    impl Display for ProjectedProgramType {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::First(r#type) => Display::fmt(r#type, formatter),
                Self::Second(r#type) => Display::fmt(r#type, formatter),
                Self::Third(r#type) => Display::fmt(r#type, formatter),
            }
        }
    }

    impl Parameter for ProjectedProgramType {}

    impl Type for ProjectedProgramType {
        type Identity = NoIdentity;
        type Refinements = ();

        fn is_compatible_with(&self, other: &Self) -> bool {
            self == other
        }

        fn is_refined_by(&self, other: &Self) -> bool {
            self == other
        }

        fn is_scalar(&self) -> bool {
            true
        }

        fn is_complex(&self) -> bool {
            false
        }
    }

    impl DifferentiableType for ProjectedProgramType {
        fn is_zero_space(&self) -> bool {
            false
        }

        fn tangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }

        fn cotangent(&self) -> Result<Self, DifferentiationError> {
            Ok(self.clone())
        }
    }

    /// Test-only composite storage value mirroring [`ProjectedProgramType`].
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramValue {
        /// First member value, used by the ordinary projection tests.
        First(ProjectedMemberValue<0>),

        /// Second member value, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberValue<1>),

        /// Third member value, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberValue<2>),
    }

    impl Display for ProjectedProgramValue {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::First(value) => Display::fmt(value, formatter),
                Self::Second(value) => Display::fmt(value, formatter),
                Self::Third(value) => Display::fmt(value, formatter),
            }
        }
    }

    impl Parameter for ProjectedProgramValue {}

    impl Typed for ProjectedProgramValue {
        type Type = ProjectedProgramType;

        fn r#type(&self) -> Cow<'_, Self::Type> {
            Cow::Owned(match self {
                Self::First(_) => ProjectedProgramType::First(ProjectedMemberType),
                Self::Second(_) => ProjectedProgramType::Second(ProjectedMemberType),
                Self::Third(_) => ProjectedProgramType::Third(ProjectedMemberType),
            })
        }
    }

    impl Value for ProjectedProgramValue {
        type DispatchDomain = EagerContext<Self>;
        type ExecutionDomain = EagerContext<Self>;

        fn dispatch_domain(&self) -> Self::DispatchDomain {
            EagerContext::new()
        }

        fn execution_domain(&self) -> Self::ExecutionDomain {
            EagerContext::new()
        }
    }

    /// Implements type/value conversion and projection for one named member of the test composite family.
    macro_rules! impl_projected_test_member {
        ($member:literal, $variant:ident) => {
            impl From<ProjectedMemberType<$member>> for ProjectedProgramType {
                fn from(r#type: ProjectedMemberType<$member>) -> Self {
                    Self::$variant(r#type)
                }
            }

            impl<'t> TryFrom<&'t ProjectedProgramType> for &'t ProjectedMemberType<$member> {
                type Error = TypeError;

                fn try_from(r#type: &'t ProjectedProgramType) -> Result<Self, Self::Error> {
                    match r#type {
                        ProjectedProgramType::$variant(r#type) => Ok(r#type),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, r#type))),
                    }
                }
            }

            impl ValueProjection<ProjectedMemberType<$member>> for ProjectedProgramValue {
                type Projected = ProjectedMemberValue<$member>;
                type ProjectedRef<'v>
                    = &'v ProjectedMemberValue<$member>
                where
                    Self: 'v;

                fn from_projected(value: Self::Projected) -> Self {
                    Self::$variant(value)
                }

                fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
                where
                    ProjectedMemberType<$member>: 'v,
                {
                    match self {
                        Self::$variant(value) => Ok(value),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, self.r#type()))),
                    }
                }

                fn into_projected(self) -> Result<Self::Projected, TypeError> {
                    match self {
                        Self::$variant(value) => Ok(value),
                        _ => Err(TypeError::invalid(format!("expected member {} but got {}", $member, self.r#type()))),
                    }
                }
            }

            impl From<ProjectedMemberOperation<$member>> for ProjectedProgramOperation {
                fn from(operation: ProjectedMemberOperation<$member>) -> Self {
                    Self::$variant(operation)
                }
            }

            impl OperationProjection<ProjectedMemberType<$member>> for ProjectedProgramOperation {
                type Projected = ProjectedMemberOperation<$member>;
            }
        };
    }

    impl_projected_test_member!(0, First);
    impl_projected_test_member!(1, Second);
    impl_projected_test_member!(2, Third);

    /// Test-only homogeneous identity operation for one projected member kind.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) struct ProjectedMemberOperation<const MEMBER: u8>;

    impl<const MEMBER: u8> Operation for ProjectedMemberOperation<MEMBER> {
        type Type = ProjectedMemberType<MEMBER>;

        fn name(&self) -> &'static str {
            "projected_member"
        }

        fn infer_output_types(
            &self,
            input_types: &[ProjectedMemberType<MEMBER>],
            _region_interfaces: &[RegionInterface<ProjectedMemberType<MEMBER>>],
        ) -> Result<Vec<ProjectedMemberType<MEMBER>>, TypeError> {
            check_count!("input", input_types, 1, TypeError);
            Ok(input_types.to_vec())
        }
    }

    impl<const MEMBER: u8, C: Context<Type = ProjectedMemberType<MEMBER>, Operation: From<Self>>>
        DifferentiableOperation<C> for ProjectedMemberOperation<MEMBER>
    {
        fn jvp<D: DifferentiationDriver<C>>(
            &self,
            context: &C,
            _driver: &D,
            inputs: &[DifferentiationDual<C::Value>],
        ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
            check_count!("input", inputs, 1, ProgramError);
            let primal = context.bind(self.clone(), Vec::new(), std::slice::from_ref(inputs[0].primal()))?.remove(0);
            Ok(vec![DifferentiationDual::new(primal, inputs[0].tangent().clone())?])
        }
    }

    impl<
        const MEMBER: u8,
        V: Value<Type = ProjectedMemberType<MEMBER>>,
        O: Operation<Type = ProjectedMemberType<MEMBER>>,
    > TransposableOperation<V, O> for ProjectedMemberOperation<MEMBER>
    {
        fn transpose<D: TranspositionDriver<V, O>>(
            &self,
            _context: &mut TracingContext<V, O>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
            check_count!("input", inputs, 1, ProgramError);
            check_count!("output", outputs, 1, ProgramError);
            Ok(vec![match &inputs[0] {
                PartialValue::Unknown(_) => outputs[0].clone(),
                PartialValue::Known(_) => MaybeZero::Zero(inputs[0].r#type().cotangent()?),
            }])
        }
    }

    /// Test-only composite operation family embedding all three member families.
    #[derive(Clone, Debug, PartialEq, Eq)]
    pub(crate) enum ProjectedProgramOperation {
        /// First member operation, used by the ordinary projection tests.
        First(ProjectedMemberOperation<0>),

        /// Second member operation, which exercises the additional-member extensibility gate.
        Second(ProjectedMemberOperation<1>),

        /// Third member operation, used by transform tests to prove that generic machinery is member-kind-agnostic.
        Third(ProjectedMemberOperation<2>),
    }

    impl ProjectedProgramOperation {
        /// Delegates composite inference to one member operation. Fixture operations declare no region slots, so
        /// staging rejects attached regions before inference and the member sees none.
        fn infer_member<const MEMBER: u8>(
            operation: &ProjectedMemberOperation<MEMBER>,
            input_types: &[ProjectedProgramType],
        ) -> Result<Vec<ProjectedProgramType>, TypeError>
        where
            for<'t> &'t ProjectedMemberType<MEMBER>: TryFrom<&'t ProjectedProgramType, Error = TypeError>,
            ProjectedProgramType: From<ProjectedMemberType<MEMBER>>,
        {
            let input_types = input_types
                .iter()
                .map(|r#type| <&ProjectedMemberType<MEMBER>>::try_from(r#type).cloned())
                .collect::<Result<Vec<_>, _>>()?;
            operation
                .infer_output_types(input_types.as_slice(), &[])
                .map(|types| types.into_iter().map(Into::into).collect())
        }
    }

    impl Operation for ProjectedProgramOperation {
        type Type = ProjectedProgramType;

        fn name(&self) -> &'static str {
            match self {
                Self::First(operation) => operation.name(),
                Self::Second(operation) => operation.name(),
                Self::Third(operation) => operation.name(),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ProjectedProgramType],
            _region_interfaces: &[RegionInterface<ProjectedProgramType>],
        ) -> Result<Vec<ProjectedProgramType>, TypeError> {
            match self {
                Self::First(operation) => Self::infer_member(operation, input_types),
                Self::Second(operation) => Self::infer_member(operation, input_types),
                Self::Third(operation) => Self::infer_member(operation, input_types),
            }
        }
    }

    impl InterpretableOperation<EagerContext<ProjectedProgramValue, Self>> for ProjectedProgramOperation {
        fn interpret<D: InterpretationDriver<EagerContext<ProjectedProgramValue, Self>>>(
            &self,
            _context: &EagerContext<ProjectedProgramValue, Self>,
            _driver: &D,
            inputs: &[ProjectedProgramValue],
        ) -> Result<Vec<ProjectedProgramValue>, ProgramError> {
            let input_types = inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
            self.infer_output_types(input_types.as_slice(), &[])?;
            Ok(inputs.to_vec())
        }
    }

    #[test]
    fn test_domain() {
        // `EagerContext<Array, ArrayOperation<Array>>` is an eager context over self-describing arrays, so binding a
        // nullary zero or one operation interprets it directly as a rank-zero array of the requested `ArrayType`.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert_eq!(
            domain.bind(ZeroOperation::new(ArrayType::scalar(DataType::BF16)), Vec::new(), &[]),
            Ok(vec![Array::scalar(bf16::ZERO)])
        );
        assert_eq!(
            domain.bind(OneOperation::new(ArrayType::scalar(DataType::BF16)), Vec::new(), &[]),
            Ok(vec![Array::scalar(bf16::ONE)])
        );
        assert_eq!(
            domain.bind(ZeroOperation::new(ArrayType::scalar(DataType::F16)), Vec::new(), &[]),
            Ok(vec![Array::scalar(f16::ZERO)])
        );
        assert_eq!(
            domain.bind(OneOperation::new(ArrayType::scalar(DataType::F16)), Vec::new(), &[]),
            Ok(vec![Array::scalar(f16::ONE)])
        );
        assert_eq!(
            domain.bind(ZeroOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), &[]),
            Ok(vec![Array::scalar(0.0f32)])
        );
        assert_eq!(
            domain.bind(OneOperation::new(ArrayType::scalar(DataType::F32)), Vec::new(), &[]),
            Ok(vec![Array::scalar(1.0f32)])
        );
        assert_eq!(
            domain.bind(ZeroOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), &[]),
            Ok(vec![Array::scalar(0.0)])
        );
        assert_eq!(
            domain.bind(OneOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), &[]),
            Ok(vec![Array::scalar(1.0)])
        );
    }

    #[test]
    fn test_eager_context_binds_and_lifts_values() {
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let default_context = EagerContext::<Array, ArrayOperation<Array>>::default();
        let copied_context = context;
        let cloned_context = copied_context.clone();
        assert_eq!(format!("{context:?}"), "EagerContext");
        assert_eq!(format!("{default_context:?}"), "EagerContext");
        assert_eq!(format!("{cloned_context:?}"), "EagerContext");
        assert_eq!(context.lift(Array::scalar(2.5)), Ok(Array::scalar(2.5)));
        assert_eq!(
            context.bind(ZeroOperation::new(ArrayType::scalar(DataType::F64)), [], &[]),
            Ok(vec![Array::scalar(0.0)])
        );
        assert_eq!(
            context.bind(OneOperation::new(ArrayType::scalar(DataType::F64)), Vec::new(), &[]),
            Ok(vec![Array::scalar(1.0)])
        );
        assert_eq!(
            context.bind(AddOperation::new(), Vec::new(), &[Array::scalar(2.0), Array::scalar(3.5)]),
            Ok(vec![Array::scalar(5.5)]),
        );
    }

    #[test]
    fn test_eager_context_bind_validates_and_interprets_shared_callee_programs() {
        // Shared callee programs bind through the same eager interpretation driver as owned regions. Eager binding
        // validates the complete attachment count before handing that driver to the operation's interpretation rule.
        let condition = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let carry = builder.add_input(ArrayType::scalar(DataType::F64));
            let eight = builder.add_constant(Array::scalar(8.0));
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), Vec::new(), vec![carry, eight])
                .unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let carry = builder.add_input(ArrayType::scalar(DataType::F64));
            let doubled = builder.add_instruction(AddOperation::new(), Vec::new(), vec![carry, carry]).unwrap()[0];
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![doubled], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let condition = Arc::new(condition);
        let body = Arc::new(body);
        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            context.bind(
                AddOperation::new(),
                CalleeRegionDriver::new(std::slice::from_ref(&condition)),
                &[Array::scalar(1.0), Array::scalar(2.0)],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `add` declares no region slots but 1 regions were attached",
        ));
        assert!(matches!(
            context.bind(
                ArrayOperation::While(WhileOperation::new()),
                CalleeRegionDriver::new(&[]),
                &[Array::scalar(1.0)],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `while` declares 2 region slots but 0 regions were attached",
        ));
        assert!(matches!(
            context.bind(
                ArrayOperation::While(WhileOperation::new()),
                CalleeRegionDriver::new(&[condition.clone(), body.clone(), body.clone()]),
                &[Array::scalar(1.0)],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `while` declares 2 region slots but 3 regions were attached",
        ));
        assert_eq!(
            context.bind(
                ArrayOperation::While(WhileOperation::new()),
                CalleeRegionDriver::new(&[condition, body]),
                &[Array::scalar(1.0)],
            ),
            Ok(vec![Array::scalar(8.0)]),
        );
    }

    #[test]
    fn test_projected_context_binds_and_lifts_values() {
        type TestEagerContext = EagerContext<ProjectedProgramValue, ProjectedProgramOperation>;

        // The adapter is zero-state. It stores its parent and nothing else.
        let context = ProjectedContext::<_, ProjectedMemberType<0>>::new(TestEagerContext::new());
        assert_eq!(
            size_of::<ProjectedContext<TestEagerContext, ProjectedMemberType<0>>>(),
            size_of::<TestEagerContext>()
        );
        assert!(context.is_eager());
        assert_eq!(
            context.bind(ProjectedMemberOperation, Vec::new(), &[ProjectedMemberValue::<0>(7)]),
            Ok(vec![ProjectedMemberValue::<0>(7)]),
        );
        assert_eq!(context.lift(ProjectedMemberValue::<0>(11)), Ok(ProjectedMemberValue::<0>(11)));
        assert_eq!(
            context.resolve(&ProjectedMemberValue::<0>(13)),
            ValueResolution::Constant(ProjectedMemberValue::<0>(13)),
        );

        // Homogeneous projected operations cannot smuggle composite values through attached regions.
        let mut builder = ProgramBuilder::<ProjectedMemberValue<0>, ProjectedMemberOperation<0>>::new();
        let input = builder.add_input(ProjectedMemberType);
        let region = builder
            .build::<Vec<ProjectedMemberValue<0>>, Vec<ProjectedMemberValue<0>>>(
                vec![input],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            context.bind(
                ProjectedMemberOperation,
                vec![region],
                &[ProjectedMemberValue::<0>(17)],
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "projected operation `projected_member` cannot carry regions",
        ));
    }

    #[test]
    fn test_projected_context_stages_without_implicit_dependencies() {
        type TestTracingContext = TracingContext<ProjectedProgramValue, ProjectedProgramOperation>;

        let parent = TestTracingContext::new();
        let input = parent.input(ProjectedProgramType::First(ProjectedMemberType));
        let input_atom = input.atom_id().unwrap();
        let input =
            <Tracer<TestTracingContext> as ValueProjection<ProjectedMemberType<0>>>::into_projected(input).unwrap();
        let context = input.dispatch_domain();
        let output = context.bind(ProjectedMemberOperation, Vec::new(), &[input]).unwrap().remove(0);

        assert_eq!(output.value().atom_id(), Ok(AtomId::new(1)));
        assert_eq!(context.resolve(&output), ValueResolution::Staged(AtomId::new(1)));
        let builder = parent.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one projected instruction");
        };
        assert_eq!(instruction.inputs(), &[input_atom]);
        assert_eq!(instruction.outputs(), &[AtomId::new(1)]);
        assert!(matches!(instruction.operation(), ProjectedProgramOperation::First(ProjectedMemberOperation)));
        assert!(instruction.regions().is_empty());
    }

    #[test]
    fn test_projected_context_supports_additional_members() {
        type TestEagerContext = EagerContext<ProjectedProgramValue, ProjectedProgramOperation>;
        type TestTracingContext = TracingContext<ProjectedProgramValue, ProjectedProgramOperation>;

        // This is the additional-member extensibility gate. Adding another member kind to a composite family requires
        // only that member's composite type/value conversions and operation-family projection. The generic projected
        // values for ordinary tracing and both transform carriers remain usable without any changes.
        fn assert_value<V: Value>() {}
        assert_value::<ProjectedValue<ProjectedMemberType<1>, Tracer<TestTracingContext>>>();
        assert_value::<ProjectedValue<ProjectedMemberType<1>, PartialTracer<TestEagerContext>>>();
        assert_value::<ProjectedValue<ProjectedMemberType<1>, DifferentiationTracer<TestEagerContext>>>();

        let parent = TestTracingContext::new();
        let input = parent.input(ProjectedProgramType::Second(ProjectedMemberType));
        let input =
            <Tracer<TestTracingContext> as ValueProjection<ProjectedMemberType<1>>>::into_projected(input).unwrap();
        let output = input.dispatch_domain().bind(ProjectedMemberOperation, Vec::new(), &[input]).unwrap().remove(0);
        assert_eq!(output.value().atom_id(), Ok(AtomId::new(1)));
        assert!(matches!(
            parent.builder().borrow().instructions()[0].operation(),
            ProjectedProgramOperation::Second(ProjectedMemberOperation),
        ));
    }

    #[test]
    fn test_staging_context_creates_inputs_constants_and_tracers() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();

        let input = context.input(ArrayType::scalar(DataType::F64));
        let constant = context.constant(Array::scalar(2.5));
        let builder_typed = context.tracer(AtomId::new(0), None);
        let cached_typed = context.tracer(AtomId::new(0), Some(ArrayType::scalar(DataType::F64)));

        assert_eq!(input.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(constant.atom_id(), Ok(AtomId::new(1)));
        assert_eq!(input.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(constant.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == ArrayType::scalar(DataType::F64)));

        let builder = builder.borrow();
        assert_eq!(builder.input_ids(), &[AtomId::new(0)]);
        assert!(builder.instructions().is_empty());
        assert!(matches!(&builder.atoms()[0], Atom::Variable(r#type) if *r#type == ArrayType::scalar(DataType::F64)));
        assert!(matches!(&builder.atoms()[1], Atom::Constant(value) if *value == Array::scalar(2.5)));
    }

    #[test]
    fn test_staging_context_stages_nullary_and_regular_operations() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();

        let mut nullary_outputs =
            context.stage_nullary_operation(ZeroOperation::new(ArrayType::scalar(DataType::F64))).unwrap();
        assert_eq!(nullary_outputs.len(), 1);
        let zero = nullary_outputs.remove(0);
        assert_eq!(zero.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(zero.r#type().into_owned(), ArrayType::scalar(DataType::F64));

        let lhs = context.input(ArrayType::scalar(DataType::F64));
        let rhs = context.input(ArrayType::scalar(DataType::F64));
        let mut add_outputs = context.stage_operation(AddOperation::new(), [], &[&lhs, &rhs]).unwrap();
        assert_eq!(add_outputs.len(), 1);
        let sum = add_outputs.remove(0);
        assert_eq!(sum.atom_id(), Ok(AtomId::new(3)));
        assert_eq!(sum.r#type().into_owned(), ArrayType::scalar(DataType::F64));

        {
            let builder = builder.borrow();
            assert_eq!(builder.instructions().len(), 2);
            assert_eq!(builder.instructions()[0].inputs(), &[]);
            assert_eq!(builder.instructions()[0].outputs(), &[AtomId::new(0)]);
            assert!(builder.instructions()[0].operation().is_zero(0));
            assert_eq!(builder.instructions()[1].inputs(), &[AtomId::new(1), AtomId::new(2)]);
            assert_eq!(builder.instructions()[1].outputs(), &[AtomId::new(3)]);
        }

        let program = builder
            .borrow()
            .clone()
            .build::<(Array, Array), Array>(vec![sum.atom_id().unwrap()], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Array::scalar(2.0), Array::scalar(3.5))), Ok(Array::scalar(5.5)));
    }

    #[test]
    fn test_staging_context_records_errors_and_returns_poisoned_outputs_after_failure() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();
        let input = context.input(ArrayType::scalar(DataType::F64));

        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(context.error(first_error.clone()), first_error);
        assert_eq!(context.error(second_error.clone()), second_error);
        assert_eq!(builder.borrow().error().cloned(), Some(first_error.clone()));

        let mut outputs = context.stage_operation(NegOperation::new(), Vec::new(), &[&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output = outputs.remove(0);
        assert_eq!(output.state(), &TracerState::Poison);
        assert_eq!(output.r#type().into_owned(), ArrayType::scalar(DataType::F64));
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let builder = context.builder().clone();
        let input = context.input(ArrayType::scalar(DataType::F64));
        let foreign_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let foreign_input = foreign_context.input(ArrayType::scalar(DataType::F64));

        assert!(matches!(
            context.stage_operation(AddOperation::new(), Vec::new(), &[&input, &foreign_input]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_staging_context_validates_and_instantiates_region_attachments() {
        #[derive(Clone)]
        struct StagingRegionOperation {
            region_input_type_count: usize,
            fail_region_input_inference: bool,
        }

        impl Operation for StagingRegionOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "staging_region"
            }

            fn region_slots(&self) -> &'static [crate::RegionSlot] {
                const { &[crate::RegionSlot::computation("body")] }
            }

            fn infer_region_input_types(
                &self,
                _input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<Option<Vec<ArrayType>>>, TypeError> {
                if self.fail_region_input_inference {
                    return Err(TypeError::invalid("failed to infer staging region input types"));
                }
                Ok(vec![None; self.region_input_type_count])
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let [region_interface] = region_interfaces else {
                    return Err(TypeError::invalid(format!(
                        "staging region expects 1 attached region but got {}",
                        region_interfaces.len(),
                    )));
                };
                if region_interface.input_types() != input_types {
                    return Err(TypeError::invalid("staging region input types do not match its operand types"));
                }
                Ok(region_interface.output_types().to_vec())
            }
        }

        #[derive(Clone)]
        struct IdentityInstantiatingRegionOperation;

        impl Operation for IdentityInstantiatingRegionOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                "identity_instantiating_region"
            }

            fn region_slots(&self) -> &'static [crate::RegionSlot] {
                const { &[crate::RegionSlot::computation("body")] }
            }

            fn infer_region_input_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<Option<Vec<ArrayType>>>, TypeError> {
                Ok(vec![Some(input_types.to_vec())])
            }

            fn infer_output_types(
                &self,
                _input_types: &[ArrayType],
                region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                let [region_interface] = region_interfaces else {
                    return Err(TypeError::invalid(format!(
                        "identity-instantiating region expects 1 attached region but got {}",
                        region_interfaces.len(),
                    )));
                };
                Ok(region_interface.output_types().to_vec())
            }
        }

        let region = {
            let mut builder = ProgramBuilder::<Array, StagingRegionOperation>::new();
            let input = builder.add_input(ArrayType::scalar(DataType::F64));
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };

        // Attachment-count mismatches are rejected before inference or import and become the trace's recorded error.
        let context = TracingContext::<Array, StagingRegionOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            context.stage_operation(
                StagingRegionOperation { region_input_type_count: 1, fail_region_input_inference: false },
                Vec::new(),
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::MalformedProgram(message))
                if message == "operation `staging_region` declares 1 region slots but 0 regions were attached",
        ));
        assert!(matches!(
            context.builder().borrow().error(),
            Some(ProgramError::MalformedProgram(message))
                if message == "operation `staging_region` declares 1 region slots but 0 regions were attached",
        ));

        // The operation must return exactly one instantiation request per declared region.
        let context = TracingContext::<Array, StagingRegionOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        assert!(matches!(
            context.stage_operation(
                StagingRegionOperation { region_input_type_count: 0, fail_region_input_inference: false },
                vec![region.clone()],
                std::slice::from_ref(&input),
            ),
            Err(ProgramError::MalformedProgram(actual))
            if actual == "operation `staging_region` returned 0 region instantiation entries for 1 attached regions",
        ));
        assert!(matches!(
            context.builder().borrow().error(),
            Some(ProgramError::MalformedProgram(actual))
            if actual == "operation `staging_region` returned 0 region instantiation entries for 1 attached regions",
        ));

        // Region-input inference failures are recorded on the builder just like output-inference and import failures.
        let context = TracingContext::<Array, StagingRegionOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        let error = ProgramError::Type(TypeError::invalid("failed to infer staging region input types"));
        assert!(matches!(
            context.stage_operation(
                StagingRegionOperation { region_input_type_count: 1, fail_region_input_inference: true },
                vec![region.clone()],
                std::slice::from_ref(&input),
            ),
            Err(actual) if actual == error,
        ));
        assert_eq!(context.builder().borrow().error(), Some(&error));

        // A poisoned trace infers the outputs against the hypothetical instantiated interface but imports and records
        // nothing, preserving the original builder error.
        let context = TracingContext::<Array, StagingRegionOperation>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        let original_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        context.error(original_error.clone());
        let outputs = context
            .stage_operation(
                StagingRegionOperation { region_input_type_count: 1, fail_region_input_inference: false },
                vec![region],
                std::slice::from_ref(&input),
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].state(), &TracerState::Poison);
        assert_eq!(outputs[0].r#type().as_ref(), &ArrayType::scalar(DataType::F64));
        let builder = context.builder().borrow();
        assert_eq!(builder.error(), Some(&original_error));
        assert!(builder.instructions().is_empty());
        assert!(builder.regions.is_empty());

        // A live trace imports the region only after instantiating its formal input identity from the caller type.
        let bounds = DimensionBounds::non_negative(Some(8)).unwrap();
        let formal_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("batch", bounds))]),
        );
        let caller_type = ArrayType::new(
            DataType::F64,
            Shape::new(vec![Dimension::Dynamic(DimensionVariable::new("caller_batch", bounds))]),
        );
        let region = {
            let mut builder = ProgramBuilder::<Array, IdentityInstantiatingRegionOperation>::new();
            let input = builder.add_input(formal_type);
            builder.build::<Vec<Array>, Vec<Array>>(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
        };
        let context = TracingContext::<Array, IdentityInstantiatingRegionOperation>::new();
        let input = context.input(caller_type.clone());
        let outputs = context
            .stage_operation(IdentityInstantiatingRegionOperation, vec![region], std::slice::from_ref(&input))
            .unwrap();
        assert_eq!(outputs[0].r#type().as_ref(), &caller_type);
        let builder = context.builder().borrow();
        let [instruction] = builder.instructions() else {
            panic!("expected one staged instruction");
        };
        let [region_id] = instruction.regions() else {
            panic!("expected one attached region");
        };
        let interface = builder.region_ref(*region_id).unwrap().interface();
        assert_eq!(interface.input_types(), std::slice::from_ref(&caller_type));
        assert_eq!(interface.output_types(), std::slice::from_ref(&caller_type));
    }

    #[test]
    fn test_staging_context_resolve() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let input = context.input(ArrayType::scalar(DataType::F64));
        let constant = context.constant(Array::scalar(2.5));
        let mut add_outputs = context.stage_operation(AddOperation::new(), Vec::new(), &[&input, &constant]).unwrap();
        let sum = add_outputs.remove(0);

        // Literal-backed tracers resolve to their program-constant payload, while inputs and operation outputs
        // resolve to their staged atoms.
        assert_eq!(context.resolve(&input), ValueResolution::Staged(AtomId::new(0)));
        assert_eq!(context.resolve(&constant), ValueResolution::Constant(Array::scalar(2.5)));
        assert_eq!(context.resolve(&sum), ValueResolution::Staged(AtomId::new(2)));

        // Tracers belonging to a different builder are opaque, in both directions.
        let foreign_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let foreign_input = foreign_context.input(ArrayType::scalar(DataType::F64));
        assert_eq!(context.resolve(&foreign_input), ValueResolution::Opaque);
        assert_eq!(foreign_context.resolve(&input), ValueResolution::Opaque);

        // Poisoned tracers are opaque even in their own context.
        let poisoning_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        assert_eq!(foreign_context.error(poisoning_error.clone()), poisoning_error);
        let mut poisoned_outputs =
            foreign_context.stage_operation(NegOperation::new(), Vec::new(), &[&foreign_input]).unwrap();
        let poisoned = poisoned_outputs.remove(0);
        assert_eq!(poisoned.state(), &TracerState::Poison);
        assert_eq!(foreign_context.resolve(&poisoned), ValueResolution::Opaque);
    }
}
