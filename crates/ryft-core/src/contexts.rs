//! Contains types for representing _domains_ and operation-binding _contexts_.
//!
//! A [`Domain`] names a type/value universe. A [`Context`] gives operations meaning within that universe. Keeping these
//! roles separate lets the same domain participate in eager execution, ordinary tracing, batching, differentiation,
//! partial evaluation, and nested transforms without changing its type, value, constant, or operation families.
//!
//! ```text
//! ┌─────────┐
//! │ Domain  │
//! └────┬────┘
//!      ├── Type ──────── abstract type metadata
//!      ├── Value ─────── values flowing through the domain
//!      ├── Constant ──── payloads stored in staged programs
//!      └── Operation ─── the closed operation family
//!           │
//!           │ adds binding semantics
//!           ▼
//!      ┌─────────┐
//!      │ Context │
//!      └────┬────┘
//!           ├── Eager Context ────── interpret each operation now
//!           ├── Staging Context ──── append each operation as an instruction
//!           └── Transform Context ── apply an operation rule, then delegate to the parent
//! ```
//!
//! # Entry Points
//!
//! Use the free [`trace`](crate::trace) function for ordinary symbolic tracing (or [`Trace::trace`] to name a domain
//! explicitly) and [`infer_output_type`](crate::tracing::infer_output_type) when only abstract outputs are needed.
//! Use [`Program::interpret`] for eager replay or [`Program::interpret_in_context`] to replay through a chosen context.
//! Most transform modules also expose a free value-level entry point and a context capability built on this same bind
//! protocol.
//!
//! # Domains versus Contexts
//!
//! [`Domain`] is _descriptive_. It associates one [`Type`], runtime [`Value`], staged constant type, and [`Operation`]
//! family and provides ordinary tracing conveniences. It does not represent an active trace and does not decide how an
//! operation is evaluated.
//!
//! [`Context`] is _behavioral_. Its [`Context::bind`] method accepts an operation and flowing values and returns
//! result values. [`Context::lift`] materializes a staged constant into the flowing value semantics, while
//! [`Context::resolve`] reports whether a flowing value is concrete, staged, or opaque. Contexts are cloneable
//! handles. Mutable tracing state is shared behind their internal handles rather than passed as `&mut self`.
//!
//! # Eager, Staging, and Transform Contexts
//!
//! [`EagerContext`] is the no-state eager implementation. Binding immediately calls the operation's
//! [`InterpretableOperation`] implementation, and lifting is the identity when constants and values coincide.
//! It is the context used by [`Program::interpret`] and by transforms over concrete values.
//!
//! [`StagingContext`] refines [`Context`] for contexts whose flowing value is [`Tracer`]. It exposes a
//! [`ProgramBuilder`] and records bound operations as instructions.
//! Ordinary [`TracingContext`] and nested tracing contexts implement this trait.
//!
//! Transform contexts wrap another context and change bind semantics locally. Batching maps an operation over logical
//! batch axes, differentiation propagates primal/tangent duals, and partial evaluation folds known work or emits
//! residual work. Each delegates primitive work to its parent, so contexts compose into a stack. For example, this is
//! what a staged forward-mode differentiation context stack might look like:
//!
//! ```text
//!       ┌─────────────────┐
//!       │ User Operation  │
//!       └────────┬────────┘
//!                │ bind
//!                ▼
//!   ┌─────────────────────────┐
//!   │ Differentiation Context │
//!   └────────────┬────────────┘
//!                │ delegate primitive work
//!                ▼
//! ┌─────────────────────────────┐
//! │ Partial Evaluation Context  │
//! └──────────────┬──────────────┘
//!                │ delegate primitive work
//!                ▼
//!       ┌─────────────────┐
//!       │ Tracing Context │
//!       └────────┬────────┘
//!                │ record instructions
//!                ▼
//!     ┌─────────────────────┐
//!     │ Transformed Program │
//!     └─────────────────────┘
//! ```
//!
//! # Value Resolution
//!
//! [`ValueResolution`] is the single conservative query used when a rule needs to inspect a flowing value:
//!
//! - [`ValueResolution::Concrete`] carries a materialized constant and permits host-side decisions.
//! - [`ValueResolution::Staged`] carries the corresponding [`AtomId`] in the active builder.
//! - [`ValueResolution::Opaque`] means the context cannot safely prove either fact.
//!
//! Rules must treat `Opaque` conservatively. In particular, they must not assume that an arbitrary tracer
//! is concrete or belongs to the current builder.
//!
//! # Extending Contexts
//!
//! To define a new backend or value universe, implement [`Domain`] first. Define one closed operation family rather
//! than coupling operation payload modules to a concrete wrapper enum, and implement [`Context`] only when that
//! universe has a specific binding semantics.
//!
//! To add a staging context, implement [`Context`] with `Value = Tracer<Self>` and then [`StagingContext`], keeping the
//! builder and error state in shared handles. To add a transform, wrap the narrowest parent context needed, define the
//! transform's flowing value or tracer, apply operation-owned rules in `bind`, and delegate capture registration and
//! context capabilities to the parent where appropriate.

use std::cell::RefCell;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::rc::Rc;

use crate::interpretation::{EagerInterpretationDriver, InterpretableOperation};
use crate::macros::check_builders;
use crate::operations::Operation;
use crate::operations::constants::ConstantOperation;
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::{AtomId, Program, ProgramBuilder, ProgramError, Value};
use crate::regions::BindingRegionDriver;
use crate::tracing::{Trace, Tracer, TracerState, TracingContext};
use crate::types::{Type, Typed};

/// Type/value universe at the core of Ryft that is used by program interpretation, tracing, and transformations like
/// batching and automatic differentiation. A [`Domain`] is purely the type, value, constant, and operation universe
/// that a backend or value model understands. It carries no behavior. It does not describe an active tracing run, and
/// it does not decide what happens when a primitive operation is bound. Active bind handling, and lifting of staged
/// constants into runtime values, live in [`Context`] implementations. This separation allows the same [`Domain`] to
/// be reused by ordinary tracing, batching, linearization, and other transformation contexts. A [`Domain`] that can
/// additionally *apply* operations to values is a [`Context`]. Eager backends do so by implementing [`Context`]
/// directly (i.e., binding interprets operations over concrete values), while staging contexts bind through
/// [`StagingContext`]s.
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
    type Operation: Operation<Self::Type>;
}

/// Active context that can *apply* an [`Operation`] to values, layered on top of the passive [`Domain`] substrate.
/// Where a [`Domain`] only describes the type, value, constant, and operation universe, a [`Context`] additionally
/// decides what *binding* a primitive means in this context and how to [`lift`](Context::lift) a staged constant into
/// a runtime value. There are two flavors:
///
/// - *Eager* contexts, whose flowing [`Domain::Value`] is a concrete value (equal to [`Domain::Constant`]).
///   [`Context::bind`] interprets the operation immediately and there is no [`ProgramBuilder`] involved anywhere.
///   Eager backends implement [`Context`] directly. An eager context is also where interpreters and program transforms
///   synthesize a type's additive or multiplicative identity from metadata alone, via `bind(ZeroOperation, &[])` or
///   `bind(OneOperation, &[])`).
/// - [**Staging**](StagingContext) contexts, whose flowing [`Domain::Value`] is a [`Tracer`] into an active
///   [`ProgramBuilder`]. [`Context::bind`] records the operation as a program instruction. Ordinary tracing appends
///   the operation to a program. Transform contexts such as batching or linearization intercept the same bind, update
///   transform-local state, and usually stage rewritten operations into a parent context.
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
    /// program and attach their [`RegionId`](crate::RegionId)s to the recorded instruction. The resulting sequence must
    /// match the number and order returned by [`Operation::region_names`].
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

    /// Returns `true` if this [`Context`] is *eager* meaning that its [`bind`](Self::bind) computes concrete values
    /// immediately so that concretizing extractions such as [`BooleanLike::boolean`](crate::BooleanLike::boolean) on
    /// values it produces can succeed and, for example, the trip count of a data-dependent loop is decidable while
    /// differentiating. Transform contexts that wrap other contexts (e.g., batching and forward-mode differentiation)
    /// delegate to the wrapped context, so that the answer reflects the innermost context that actually executes the
    /// bound operations.
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
        let program = Program {
            input_structure,
            output_structure,
            regions: flat_program.regions,
            entry: flat_program.entry,
            marker: PhantomData,
        };
        Ok((output, program))
    }
}

/// [`Context`] used for a concrete `(type, value, operation)` universe that carries no runtime state and for which,
/// binding an operation to some input values corresponds to directly interpreting/evaluating/executing that operation
/// for those input values using the value type's default interpretation context. [`EagerContext`] exists to make direct
/// interpretation contexts explicit in generic code that otherwise has no backend-owned eager context value to pass
/// around.
///
/// The default operation family is [`ConstantOperation`], which is the minimal operation family needed by ordinary
/// eager value contexts that only materialize constants and expose context capabilities such as zero, one, fill, etc.
/// Code that binds or batches a richer operation family should still specify `O` explicitly, such as
/// `EagerContext<V, ArrayOperation<V>>`.
pub struct EagerContext<V: Value, O: Operation<V::Type> = ConstantOperation<V>> {
    /// [`PhantomData`] marker tying this zero-sized context to its associated types.
    marker: PhantomData<fn() -> (V, O)>,
}

impl<V: Value, O: Operation<V::Type>> EagerContext<V, O> {
    /// Creates a new [`EagerContext`].
    #[inline]
    pub const fn new() -> Self {
        Self { marker: PhantomData }
    }
}

impl<V: Value, O: Operation<V::Type>> Copy for EagerContext<V, O> {}

impl<V: Value, O: Operation<V::Type>> Clone for EagerContext<V, O> {
    #[inline]
    fn clone(&self) -> Self {
        *self
    }
}

impl<V: Value, O: Operation<V::Type>> std::fmt::Debug for EagerContext<V, O> {
    #[inline]
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("EagerContext")
    }
}

impl<V: Value, O: Operation<V::Type>> Default for EagerContext<V, O> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<V: Value, O: Operation<V::Type>> Domain for EagerContext<V, O> {
    type Type = V::Type;
    type Value = V;
    type Constant = V;
    type Operation = O;
}

impl<V: Value, O: Operation<V::Type> + InterpretableOperation<Self>> Context for EagerContext<V, O> {
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
        operation.into().interpret(self, &EagerInterpretationDriver::new(&driver), inputs)
    }

    #[inline]
    fn is_eager(&self) -> bool {
        true
    }

    #[inline]
    fn resolve(&self, value: &V) -> ValueResolution<V> {
        ValueResolution::Concrete(value.clone())
    }
}

/// Staging [`Context`] whose flowing [`Domain::Value`] is a [`Tracer`] into an active [`ProgramBuilder`].
/// Binding records [`Operation`] invocations as [`Program`] [`Instruction`](crate::Instruction)s rather than
/// interpreting them, and this trait owns the builder-dependent staging API: [`constant`](StagingContext::constant),
/// [`input`](StagingContext::input), [`tracer`](StagingContext::tracer), [`error`](StagingContext::error),
/// and [`stage_operation`](StagingContext::stage_operation). Ordinary and nested tracing implement it through
/// [`TracingContext`] and [`NestedTracingContext`](crate::NestedTracingContext), respectively. Transform contexts have
/// their own flowing value types and delegate rewritten operations to a parent staging context instead of implementing
/// this trait themselves.
///
/// The flowing value is pinned to [`Tracer<Self>`](Tracer). Every staging context records operation invocations
/// as [`Program`] instructions and hands back [`Tracer`]s standing in for their results.
pub trait StagingContext: Context<Value = Tracer<Self>> {
    /// Returns the shared [`ProgramBuilder`] owned by this [`StagingContext`].
    fn builder(&self) -> &Rc<RefCell<ProgramBuilder<Self::Constant, Self::Operation>>>;

    /// Creates a constant [`Tracer`] in this context with the provided constant payload.
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
        check_builders!(self.builder(), [inputs.iter().map(|input| input.borrow().context().builder())])
            .map_err(|error| self.error(error))?;
        if self.builder().borrow().error.is_some() {
            let input_types = inputs.iter().map(|input| input.borrow().r#type().into_owned()).collect::<Vec<_>>();
            let region_interfaces = driver.regions().map(|region| region.interface()).collect::<Vec<_>>();
            let output_types = operation.infer_output_types(input_types.as_slice(), region_interfaces.as_slice())?;
            Ok(output_types
                .into_iter()
                .map(|r#type| Tracer::new(self.clone(), TracerState::Poison, r#type))
                .collect())
        } else {
            let inputs = match inputs.iter().map(|input| input.borrow().atom_id()).collect::<Result<Vec<_>, _>>() {
                Ok(input_atom_ids) => input_atom_ids,
                Err(error) => return Err(self.error(error)),
            };
            let region_ids = driver.import_into(self.builder()).map_err(|error| self.error(error))?;
            let outputs = {
                let mut builder = self.builder().borrow_mut();
                match builder.add_instruction(operation, inputs, region_ids) {
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
    /// The value denotes this concrete constant payload. This is the strongest outcome. Eager [`Context`]s, whose
    /// flowing values *are* constants, always resolve to [`ValueResolution::Concrete`]. [`StagingContext`]s resolve
    /// here only for *literal-backed* [`Tracer`]s, whose staged [`Atom`](crate::Atom) is a constant atom in the
    /// context's [`ProgramBuilder`].
    Concrete(V),

    /// The value is a live, staged, value identified by this [`AtomId`] in the resolving context's [`Program`], with no
    /// concrete payload until the traced program runs. The carried [`AtomId`] is a stable identity for the value within
    /// that program.
    Staged(AtomId),

    /// The resolving context can prove nothing about the value. This is the conservative default, and the answer that
    /// [`StagingContext`]s provide for poisoned [`Tracer`]s and for values belonging to different [`ProgramBuilder`]s.
    Opaque,
}

impl<V> ValueResolution<V> {
    /// Returns `true` if this [`ValueResolution`] is [`Concrete`](Self::Concrete).
    #[inline]
    pub fn is_concrete(&self) -> bool {
        matches!(self, Self::Concrete(_))
    }

    /// Returns the concrete constant payload of this [`ValueResolution`] if it is a [`Concrete`](Self::Concrete)
    /// resolution, and [`None`] otherwise.
    #[inline]
    pub fn into_concrete(self) -> Option<V> {
        match self {
            Self::Concrete(constant) => Some(constant),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use half::{bf16, f16};
    use pretty_assertions::assert_eq;

    use crate::backends::scalars::{Scalar, ScalarOperation};
    use crate::operations::compare::{CompareOperation, ComparisonDirection};
    use crate::operations::constants::{MaybeZeroOperation, OneOperation, ZeroOperation};
    use crate::operations::control_flow::WhileOperation;
    use crate::operations::math::{AddOperation, NegOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{Atom, AtomId, ProgramBuilder, ProgramError};
    use crate::regions::CalleeRegionDriver;
    use crate::tracing::{DomainTracingContext, TracerState};
    use crate::types::{DataType, Typed};

    use super::*;

    #[test]
    fn test_domain() {
        // `EagerContext<Scalar, ScalarOperation<Scalar>>` is an eager `Context` over the self-describing `Scalar` value
        // type, so binding a nullary zero/one `Operation` interprets it directly to the `Scalar` variant matching the
        // requested `DataType`.
        let domain = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(
            domain.bind(ZeroOperation::new(DataType::BF16), Vec::new(), &[]),
            Ok(vec![Scalar::BF16(bf16::ZERO)])
        );
        assert_eq!(domain.bind(OneOperation::new(DataType::BF16), Vec::new(), &[]), Ok(vec![Scalar::BF16(bf16::ONE)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F16), Vec::new(), &[]), Ok(vec![Scalar::F16(f16::ZERO)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F16), Vec::new(), &[]), Ok(vec![Scalar::F16(f16::ONE)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F32), Vec::new(), &[]), Ok(vec![Scalar::F32(0.0)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F32), Vec::new(), &[]), Ok(vec![Scalar::F32(1.0)]));
        assert_eq!(domain.bind(ZeroOperation::new(DataType::F64), Vec::new(), &[]), Ok(vec![Scalar::F64(0.0)]));
        assert_eq!(domain.bind(OneOperation::new(DataType::F64), Vec::new(), &[]), Ok(vec![Scalar::F64(1.0)]));
    }

    #[test]
    fn test_eager_context_binds_and_lifts_values() {
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        let default_context = EagerContext::<Scalar, ScalarOperation<Scalar>>::default();
        let copied_context = context;
        let cloned_context = copied_context.clone();
        assert_eq!(format!("{context:?}"), "EagerContext");
        assert_eq!(format!("{default_context:?}"), "EagerContext");
        assert_eq!(format!("{cloned_context:?}"), "EagerContext");
        assert_eq!(context.lift(Scalar::from(2.5)), Ok(Scalar::from(2.5)));
        assert_eq!(context.bind(ZeroOperation::new(DataType::F64), [], &[]), Ok(vec![Scalar::from(0.0)]));
        assert_eq!(context.bind(OneOperation::new(DataType::F64), Vec::new(), &[]), Ok(vec![Scalar::from(1.0)]));
        assert_eq!(
            context.bind(AddOperation, Vec::new(), &[Scalar::from(2.0), Scalar::from(3.5)]),
            Ok(vec![Scalar::from(5.5)]),
        );
    }

    #[test]
    fn test_staging_context_creates_inputs_constants_and_tracers() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();

        let input = context.input(DataType::F64);
        let constant = context.constant(Scalar::from(2.5));
        let builder_typed = context.tracer(AtomId::new(0), None);
        let cached_typed = context.tracer(AtomId::new(0), Some(DataType::F64));

        assert_eq!(input.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(constant.atom_id(), Ok(AtomId::new(1)));
        assert_eq!(input.r#type().into_owned(), DataType::F64);
        assert_eq!(constant.r#type().into_owned(), DataType::F64);
        assert!(matches!(builder_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));
        assert!(matches!(cached_typed.r#type(), Cow::Borrowed(r#type) if *r#type == DataType::F64));

        let builder = builder.borrow();
        assert_eq!(builder.input_ids(), &[AtomId::new(0)]);
        assert!(builder.instructions().is_empty());
        assert!(matches!(&builder.atoms()[0], Atom::Variable(r#type) if *r#type == DataType::F64));
        assert!(matches!(&builder.atoms()[1], Atom::Constant(value) if *value == 2.5));
    }

    #[test]
    fn test_staging_context_stages_nullary_and_regular_operations() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();

        let mut nullary_outputs = context.stage_nullary_operation(ZeroOperation::new(DataType::F64)).unwrap();
        assert_eq!(nullary_outputs.len(), 1);
        let zero = nullary_outputs.remove(0);
        assert_eq!(zero.atom_id(), Ok(AtomId::new(0)));
        assert_eq!(zero.r#type().into_owned(), DataType::F64);

        let lhs = context.input(DataType::F64);
        let rhs = context.input(DataType::F64);
        let mut add_outputs = context.stage_operation(AddOperation, [], &[&lhs, &rhs]).unwrap();
        assert_eq!(add_outputs.len(), 1);
        let sum = add_outputs.remove(0);
        assert_eq!(sum.atom_id(), Ok(AtomId::new(3)));
        assert_eq!(sum.r#type().into_owned(), DataType::F64);

        {
            let builder = builder.borrow();
            assert_eq!(builder.instructions().len(), 2);
            assert_eq!(builder.instructions()[0].inputs(), &[]);
            assert_eq!(builder.instructions()[0].outputs(), &[AtomId::new(0)]);
            assert!(builder.instructions()[0].operation().is_zero_operation());
            assert_eq!(builder.instructions()[1].inputs(), &[AtomId::new(1), AtomId::new(2)]);
            assert_eq!(builder.instructions()[1].outputs(), &[AtomId::new(3)]);
        }

        let program = builder
            .borrow()
            .clone()
            .build::<(Scalar, Scalar), Scalar>(vec![sum.atom_id().unwrap()], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        assert_eq!(program.interpret((Scalar::from(2.0), Scalar::from(3.5))), Ok(Scalar::from(5.5)));
    }

    #[test]
    fn test_staging_context_records_errors_and_returns_poisoned_outputs_after_failure() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);

        let first_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        let second_error = ProgramError::InvalidOutputCount { expected: 1, actual: 0 };
        assert_eq!(context.error(first_error.clone()), first_error);
        assert_eq!(context.error(second_error.clone()), second_error);
        assert_eq!(builder.borrow().error().cloned(), Some(first_error.clone()));

        let mut outputs = context.stage_operation(NegOperation, Vec::new(), &[&input]).unwrap();
        assert_eq!(outputs.len(), 1);
        let output = outputs.remove(0);
        assert_eq!(output.state(), &TracerState::Poison);
        assert_eq!(output.r#type().into_owned(), DataType::F64);
        assert_eq!(builder.borrow().error().cloned(), Some(first_error));

        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let builder = context.builder().clone();
        let input = context.input(DataType::F64);
        let foreign_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let foreign_input = foreign_context.input(DataType::F64);

        assert!(matches!(
            context.stage_operation(AddOperation, Vec::new(), &[&input, &foreign_input]),
            Err(ProgramError::MismatchedProgramBuilders),
        ));
        assert_eq!(builder.borrow().error().cloned(), Some(ProgramError::MismatchedProgramBuilders));
    }

    #[test]
    fn test_staging_context_resolve() {
        let context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let input = context.input(DataType::F64);
        let constant = context.constant(Scalar::from(2.5));
        let mut add_outputs = context.stage_operation(AddOperation, Vec::new(), &[&input, &constant]).unwrap();
        let sum = add_outputs.remove(0);

        // Literal-backed tracers resolve to their concrete constant payload, while inputs and operation outputs
        // resolve to their staged atoms.
        assert_eq!(context.resolve(&input), ValueResolution::Staged(AtomId::new(0)));
        assert_eq!(context.resolve(&constant), ValueResolution::Concrete(Scalar::from(2.5)));
        assert_eq!(context.resolve(&sum), ValueResolution::Staged(AtomId::new(2)));

        // Tracers belonging to a different builder are opaque, in both directions.
        let foreign_context = DomainTracingContext::<EagerContext<Scalar, ScalarOperation<Scalar>>>::new();
        let foreign_input = foreign_context.input(DataType::F64);
        assert_eq!(context.resolve(&foreign_input), ValueResolution::Opaque);
        assert_eq!(foreign_context.resolve(&input), ValueResolution::Opaque);

        // Poisoned tracers are opaque even in their own context.
        let poisoning_error = ProgramError::InvalidInputCount { expected: 1, actual: 0 };
        assert_eq!(foreign_context.error(poisoning_error.clone()), poisoning_error);
        let mut poisoned_outputs =
            foreign_context.stage_operation(NegOperation, Vec::new(), &[&foreign_input]).unwrap();
        let poisoned = poisoned_outputs.remove(0);
        assert_eq!(poisoned.state(), &TracerState::Poison);
        assert_eq!(foreign_context.resolve(&poisoned), ValueResolution::Opaque);
    }

    #[test]
    fn test_eager_context_bind_interprets_shared_callee_programs() {
        // Shared callee programs bind through the same eager interpretation driver as owned regions. The while loop
        // below receives its condition and body as callee attachments and runs to completion.
        let condition = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let eight = builder.add_constant(Scalar::from(8.0));
            let predicate = builder
                .add_instruction(CompareOperation::new(ComparisonDirection::LessThan), vec![carry, eight], Vec::new())
                .unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![predicate], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let body = {
            let mut builder = ProgramBuilder::<Scalar, ScalarOperation<Scalar>>::new();
            let carry = builder.add_input(DataType::F64);
            let doubled = builder.add_instruction(AddOperation, vec![carry, carry], Vec::new()).unwrap()[0];
            builder
                .build::<Vec<Scalar>, Vec<Scalar>>(vec![doubled], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let context = EagerContext::<Scalar, ScalarOperation<Scalar>>::new();
        assert_eq!(
            context.bind(
                ScalarOperation::While(WhileOperation::new()),
                CalleeRegionDriver::new(&[Rc::new(condition), Rc::new(body)]),
                &[Scalar::from(1.0)],
            ),
            Ok(vec![Scalar::from(8.0)]),
        );
    }
}
