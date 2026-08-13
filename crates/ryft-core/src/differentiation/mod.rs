//! Contains machinery for forward and reverse mode automatic _differentiation_.
//!
//! Differentiation is expressed as [`Context`](crate::Context) composition and [`Program`](crate::Program)
//! transformation rather than as a backend-specific facility. Values flowing through a [`DifferentiationContext`]
//! carry a primal and a tangent, operation-owned Jacobian-Vector Product (JVP) rules propagate those duals, partial
//! evaluation splits primal work from reusable linear work, and transposition turns the resulting linear program into
//! a pullback for reverse mode differentiation.
//!
//! # Entry Points
//!
//! [`differentiate_at`] takes the active primals and returns a typed [`DifferentiationBuilder`] for composing
//! value-level transforms with orthogonal options. Each terminal method receives the transformed closure as its last
//! argument, so the builder state fixes the closure's parameter types before they are deduced and call sites rarely
//! need to annotate them. The builder recovers its execution context from the active primals by default;
//! [`DifferentiationBuilder::in_context`] binds one explicitly, and the blanket [`Differentiate`] trait provides the
//! equivalent `context.differentiate_at(primals)` syntax. Its terminal methods provide the complete value-level API:
//!
//!   - [`DifferentiationBuilder::jvp`] computes a Jacobian-Vector Product (JVP) for one set of primals and tangents.
//!     It is the direct forward-mode transform and is usually best when there are few input directions or when both
//!     primal and tangent outputs are needed immediately.
//!   - [`DifferentiationBuilder::linearize`] runs the primal computation once and returns its value together with a
//!     reusable [`Pushforward`] that can be applied to many tangent inputs without repeating nonlinear primal work.
//!   - [`DifferentiationBuilder::vjp`] runs the primal computation and returns a reusable [`Pullback`] mapping output
//!     cotangents to input cotangents. Conceptually, `vjp = linearize + transpose` and corresponds to what is known as
//!     the Vector-Jacobian Product (VJP).
//!   - [`DifferentiationBuilder::value_and_gradient`] and [`DifferentiationBuilder::gradient`] are scalar-output
//!     conveniences that seed the pullback with one.
//!   - [`DifferentiationBuilder::jacobian_forward`] and [`DifferentiationBuilder::jacobian_reverse`] materialize
//!     complete [`Jacobian`]s by batching coordinate directions through JVPs or pullbacks. Their relative cost depends
//!     on the input and output coordinate counts.
//!   - [`DifferentiationBuilder::hessian`] differentiates a scalar-output gradient and returns the resulting structured
//!     [`Hessian`] block matrix.
//!
//! [`DifferentiationBuilder::with_aux`] preserves non-differentiated auxiliary output, while
//! [`DifferentiationBuilder::holomorphic`] explicitly opts into the complex holomorphic contract.
//! Furthermore, [`DifferentiationBuilder::with_captures`] accepts dynamic runtime values that affect the primal
//! computation while only the first closure argument is differentiated. Captures are useful for model training,
//! for example, where the model being trained is the active argument (or the primal), and a potentially large data
//! batch is a captured input whose derivatives we do not want to compute. In this case, [`Jacobian`] and [`Hessian`]
//! materialization will only enumerate basis directions for model parameters and not for the data batch.
//!
//! Active inputs, differentiated outputs, captures, and auxiliary outputs are independently polymorphic
//! [`Parameterized`] trees. Each may be a leaf, tuple, array, [`Vec`], map, or user-defined struct or enum deriving
//! `Parameterized`. Derived types may also contain optional parameter fields. The trees need neither share a container
//! type nor have the same number of leaves. A transform replaces each dynamic leaf with the tracer type required by
//! that transform, invokes the closure with the reparameterized trees, and reconstructs the same public tree shapes
//! with runtime values. Non-parameter metadata in a derived type is preserved unchanged.
//! [`DifferentiationBuilder::with_aux`] applies this treatment to the second member of a closure result
//! `(differentiated_output, auxiliary_output)`, returning the entire auxiliary tree without seeding or transposing it.
//!
//! The trees are structurally polymorphic but homogeneous in their dynamic leaf family meaning that one builder
//! invocation uses a single runtime value type `V` for all active, output, capture, and auxiliary leaves. Individual
//! leaves may still carry different element types, shapes, shardings, and placements when `V` represents those
//! properties dynamically, as [`Array`](crate::Array) does. Unrelated host configuration belongs in non-parameter
//! fields of a derived tree or in the closure itself.
//!
//! Differentiation captures are not static arguments and do not embed host values into a trace. They retain their
//! runtime types, shapes, shardings, and placement, and changing capture values alone does not specialize the
//! computation. They also differ from [`stop_gradient`](crate::stop_gradient), which blocks derivative flow but leaves
//! the stopped value inside the active parameter structure and its derivative bookkeeping. Finally, they differ from
//! [`ClosedProgram`](crate::ClosedProgram) and Just-In-Time (JIT) compilation captures as those hide lifted runtime
//! values from a compiled function's public signature, whereas differentiation captures are explicit inputs to the
//! binary closure. Auxiliary outputs are similarly runtime outputs rather than differentiation variables. They remain
//! part of the closure result but are excluded from derivative seeding and pullback construction.
//!
//! Context-generic code can use [`ForwardModeDifferentiate`] and [`ReverseModeDifferentiate`] directly, and already
//! traced programs expose the corresponding program-level JVP, linearization, and transposition methods documented on
//! [`Program`](crate::programs::Program).
//!
//! # Differentiation Pipeline
//!
//! Residuals are runtime values from the chosen linearization point. The reusable programs describe derivative
//! structure, while each [`Pushforward`] or [`Pullback`] binds that structure to the residual values captured for one
//! invocation. Refer to [`Linearization`] for a rendered diagram of the complete forward/reverse pipeline.
//!
//! # Forward Mode Differentiation
//!
//! [`DifferentiationDual`] pairs each primal with a [`MaybeZero`](crate::programs::MaybeZero) tangent, and a
//! [`DifferentiationTracer`] carries that dual through a [`DifferentiationContext`]. When an operation is bound, its
//! [`DifferentiableOperation`] rule receives primal and tangent inputs, stages or evaluates the primal operation, and
//! produces tangent outputs. Symbolic zero tangents avoid materializing unnecessary zero arrays.
//!
//! [`DifferentiationBuilder::linearize`] composes differentiation with [`PartialEvaluationContext`]. Primals are known
//! and tangents are unknown. Nonlinear primal work is evaluated once, values needed by the tangent computation become
//! residuals, and the residual program becomes a reusable [`Pushforward`]. Host control flow may branch on concrete
//! primals during this process, so only the taken path is linearized.
//!
//! # Reverse Mode Differentiation
//!
//! Reverse mode differentiation reuses forward linearization instead of maintaining an independent nonlinear trace.
//! The linearized tangent program is transposed by applying [`TransposableOperation`] rules in reverse dataflow order,
//! and the result is a [`Pullback`] that accepts output cotangents, consumes saved residuals, and accumulates input
//! cotangents. This architecture keeps primal execution, residualization, and linear algebra as separate, composable
//! concerns.
//!
//! # Zero Differential Spaces
//!
//! [`DifferentiableType`] may describe a tangent or cotangent space containing only zero. Such leaves remain present
//! in public primal structures but need no Single Static Assignment (SSA) slots in linear tangent or pullback programs.
//! Derivative entry points omit those internal slots, preserve their boundary positions as metadata, and reconstruct
//! typed zeros when rebuilding public results. [`MaybeZero`](crate::MaybeZero) likewise lets operation rules propagate
//! symbolic zeros without materializing arrays.
//!
//! # Structural Transform Reuse
//!
//! Higher-order rules frequently differentiate the same sealed [`Region`](crate::Region) from several outer programs.
//! Region-level JVP, linearization, and transposition paths retain their context-free derived programs against the
//! region's complete reachable contents and the transform's structural arguments. Faithful copies reuse those
//! artifacts, while content-changing rewrites and changed descendant closures invalidate them. These built-ins use
//! the same [`Transform`](crate::Transform) extension mechanism available to external structural program transforms;
//! differentiation-specific artifact layouts remain private adapters around that general API.
//!
//! Runtime callables are deliberately different. A [`Pushforward`] or [`Pullback`] owns concrete residual values from
//! one linearization point and is never cached as a type-only structural artifact. Immediate value-level transforms
//! likewise retain their existing semantics, including concrete host control flow chosen from the current primals.
//!
//! # Extending Differentiation
//!
//! Implement [`DifferentiableType`] for types that possess tangent and cotangent spaces. Implement
//! [`DifferentiableOperation`] for primitive JVP rules. Rules should express tangent behavior through the provided
//! context and preserve symbolic zeros where possible. Implement [`MemberDifferentiableOperation`] when a homogeneous
//! member operation needs values from its projection's parent context while constructing its derivative. Ordinary
//! homogeneous projected rules should use [`jvp_projected_operation`]. Implement [`TransposableOperation`] for linear
//! primitives that may occur in a homogeneous pushforward. A member payload whose parent instruction has a mixed
//! signature needs no separate rule, because [`transpose_mixed_operation`] delegates that instruction's member-typed
//! data operands, wherever they sit in its operand list, to that same homogeneous rule. Higher-order
//! [`Operation`](crate::Operation) logic belongs with the operation whose instruction attaches to the nested
//! [`Region`](crate::Region). Wrapper operation enums should provide family dispatch and forward to those
//! payload rules.

use std::fmt::{Debug, Display, Formatter};

use thiserror::Error;

use crate::contexts::{Context, Domain};
use crate::differentiation::hessian::hessian_in_context;
use crate::differentiation::jacobian::{jacobian_forward_in_context, jacobian_reverse_in_context};
use crate::differentiation::reverse::{value_and_gradient_auxiliary_in_context, value_and_gradient_in_context};
use crate::errors::MaybeFallible;
use crate::operations::{AddOperation, OneOperation, Zero, ZeroLikeOperation};
use crate::parameters::{ParameterError, Parameterized, ParameterizedFamily};
use crate::partial::{PartialEvaluationContext, PartiallyEvaluatableOperation};
use crate::programs::{ProgramError, TypeError, Value};
use crate::tracing::TracingContext;

pub mod batching;
pub mod elementwise;
pub mod forward;
pub mod hessian;
pub mod jacobian;
pub mod linear;
pub mod reverse;
pub mod types;

pub use batching::CotangentBatchingPolicy;
pub use elementwise::{
    BinaryElementwiseJvpOperands, BroadcastDerivativeAlignment, ElementwiseDerivativeAlignment,
    UnaryElementwiseJvpOperands, binary_elementwise_jvp, unary_elementwise_jvp,
};
pub use forward::{
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationTracer,
    ForwardModeDifferentiate, Linearization, LinearizationTracer, MemberDifferentiableOperation, Pushforward,
    jvp_projected_operation,
};
pub use hessian::{Hessian, HessianBlock};
pub use jacobian::{Jacobian, JacobianBlock};
pub use linear::{LinearCallOperation, ResidualZeroProvider};
pub use reverse::{
    Pullback, ReverseModeDifferentiate, TransposableOperation, TranspositionDriver, transpose_mixed_operation,
    transpose_projected_operation,
};
pub use types::{DenseDifferentiableType, DifferentiableType};

/// Represents differentiation-related errors.
///
/// [`DifferentiationError`] and [`ProgramError`] form the same normalized conversion cycle as
/// [`BatchingError`](crate::BatchingError) and [`ProgramError`] do. Every differentiation surface including
/// the value-level entry points (i.e., `jvp`, `linearize`, `vjp`, etc.), the program-level transforms (i.e.,
/// `Program::jvp`, `Program::linearize`, `Program::transpose`, etc.), and the per-operation rule traits (i.e.,
/// [`DifferentiableOperation`] and [`TransposableOperation`]), returns [`DifferentiationError`], while the errors those
/// rules produce *through* the kernel (i.e., binding and staging operations) are [`ProgramError`]s. The paired [`From`]
/// implementations keep this cycle normalized instead of letting the two types nest: converting to [`ProgramError`]
/// unwraps a [`DifferentiationError::Program`] back into the program error that it carries and wraps every other
/// variant in [`ProgramError::Custom`], while converting to [`DifferentiationError`] unwraps a [`ProgramError::Custom`]
/// payload holding a [`DifferentiationError`] and wraps every other program error in [`DifferentiationError::Program`].
/// Roundtrips therefore never nest one error type inside the other, and `?` re-types errors correctly at both
/// boundaries. Outside of these conversions, a [`DifferentiationError`] carried by a [`ProgramError`] can be recovered
/// using [`ProgramError::downcast_custom`].
#[derive(Clone, Debug, Error, PartialEq, Eq, Hash)]
pub enum DifferentiationError {
    /// Error returned when a differentiation entry point is invoked on an active input with no leaf values/parameters.
    /// A function without differentiation variables has no direction to perturb, so its Jacobian has no columns.
    /// Captures are deliberately not used to recover a context because they are not differentiation variables.
    #[error("differentiation requires an input with at least one leaf value")]
    EmptyInput,

    /// Error returned when reverse mode differentiation is requested for a function whose output is not a single
    /// scalar. Reverse mode differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar
    /// value of `1`) and pulls it back to the inputs, which yields a gradient only when the output is a rank-0 scalar.
    /// A non-scalar output describes a vector-valued function whose full derivative is a Jacobian. Seeding that output
    /// with an all-ones cotangent would instead compute the gradient of the sum of its elements, which is a different
    /// operation and not a canonical gradient of the vector-valued function. So, the gradient entry points reject it
    /// up front using this error variant. Use a Jacobian transform for non-scalar outputs.
    #[error("gradient output must be a rank-0 scalar but got {output_type}")]
    NonScalarGradientOutput { output_type: String },

    /// Error returned when reverse mode differentiation is requested for a function whose output type carries no
    /// cotangent space (i.e., a non-differentiable type such as a Boolean or an integer scalar). Reverse mode
    /// differentiation seeds the output cotangent with the multiplicative identity (i.e., a scalar value of `1`), but
    /// a non-differentiable output has no such value to seed. So, in this case, the gradient is degenerate and the
    /// gradient entry points reject it up front rather than fabricating a seed.
    #[error("gradient output type {output_type} is non-differentiable and carries no cotangent space")]
    NonDifferentiableGradientOutput { output_type: String },

    /// Error returned when reverse mode differentiation is requested through a plain (i.e., non-holomorphic) gradient
    /// entry point for a function whose scalar output is complex. A single reverse mode seed recovers the derivative
    /// of a complex-output function only when the function is holomorphic (i.e., complex-differentiable), a promise
    /// the plain entry points do not ask for, so they reject complex outputs up front instead of silently computing a
    /// value that is not a derivative. Use [`DifferentiationBuilder::holomorphic`] when the function is holomorphic,
    /// or split the function into its real and imaginary parts and differentiate those otherwise.
    #[error(
        "gradient output type {output_type} is complex; use a holomorphic gradient entry point \
        if the function is holomorphic"
    )]
    ComplexGradientOutput { output_type: String },

    /// Error returned when a Jacobian or Hessian transform encounters a structured input or output parameter
    /// with no tangent or cotangent space.
    #[error("{transform} {role} parameter at {path} has non-differentiable type {type}")]
    NonDifferentiableParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a non-holomorphic Jacobian or Hessian transform receives a complex active input or
    /// differentiated output parameter whose derivative requires an explicit holomorphy promise or a real-coordinate
    /// representation that Ryft does not yet expose. Captures are exempt because they are not part of the
    /// differentiated map.
    #[error("{transform} {role} parameter at {path} has complex type {type}; use holomorphic {transform} instead")]
    ComplexParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a holomorphic Jacobian or Hessian transform receives a non-complex active input or
    /// differentiated output parameter. Captures are exempt because they are not part of the complex-linear map.
    #[error("holomorphic {transform} {role} parameter at {path} must be complex but has type {type}")]
    NonComplexParameter {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when a Jacobian or Hessian transform cannot enumerate a finite coordinate space for one of its
    /// structured input or output parameters.
    #[error("{transform} {role} parameter at {path} does not have a finite static coordinate space: {type}")]
    NonFiniteCoordinateSpace {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    /// Error returned when the finite coordinate count of a Jacobian or Hessian parameter structure exceeds `usize`.
    #[error("{transform} {role} coordinate count overflows usize at parameter {path}: {type}")]
    CoordinateCountOverflow {
        transform: DerivativeTransform,
        role: DifferentiationParameterRole,
        path: String,
        r#type: String,
    },

    #[error(transparent)]
    Program(ProgramError),
}

impl From<TypeError> for DifferentiationError {
    #[inline]
    fn from(error: TypeError) -> Self {
        DifferentiationError::Program(error.into())
    }
}

impl From<ParameterError> for DifferentiationError {
    #[inline]
    fn from(error: ParameterError) -> Self {
        DifferentiationError::Program(error.into())
    }
}

impl From<ProgramError> for DifferentiationError {
    #[inline]
    fn from(error: ProgramError) -> Self {
        if let Some(differentiation) = error.downcast_custom::<DifferentiationError>() {
            differentiation.clone()
        } else {
            DifferentiationError::Program(error)
        }
    }
}

impl From<DifferentiationError> for ProgramError {
    #[inline]
    fn from(error: DifferentiationError) -> Self {
        match error {
            DifferentiationError::Program(error) => error,
            error => ProgramError::custom(error),
        }
    }
}

/// Derivative transform family that was active when an error occurred. Holomorphic entry points use the same forward-
/// or reverse-Jacobian family as their non-holomorphic counterparts. Holomorphy changes the admissible element types,
/// not the mathematical derivative object being materialized.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeTransform {
    /// Forward-mode Jacobian materialization, including its holomorphic entry point.
    JacobianForward,

    /// Reverse-mode Jacobian materialization, including its holomorphic entry point.
    JacobianReverse,

    /// Hessian materialization.
    Hessian,
}

impl Display for DerivativeTransform {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::JacobianForward => write!(formatter, "forward Jacobian"),
            Self::JacobianReverse => write!(formatter, "reverse Jacobian"),
            Self::Hessian => write!(formatter, "hessian"),
        }
    }
}

/// Role of a structured parameter in a derivative transform.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DifferentiationParameterRole {
    /// Parameter belongs to the differentiated input structure.
    Input,

    /// Parameter belongs to the differentiated output structure.
    Output,

    /// Parameter describes a materialized derivative block.
    Derivative,
}

impl Display for DifferentiationParameterRole {
    #[inline]
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input => write!(formatter, "input"),
            Self::Output => write!(formatter, "output"),
            Self::Derivative => write!(formatter, "derivative"),
        }
    }
}

mod private {
    /// Seals implementation-only differentiation builder type-state traits.
    pub trait Sealed {}
}

/// Sealed behavior shared by the ordinary and holomorphic builder type states.
pub trait DifferentiationBuilderLinearityMode: private::Sealed {
    /// Specified whether the differentiation transform uses the holomorphic complex-linearity contract or not.
    const HOLOMORPHIC: bool;
}

/// Resolves the [`Context`] used by a [`DifferentiationBuilder`] terminal function.
pub trait DifferentiationBuilderContext<V: Value, Input: Parameterized<V>>: private::Sealed {
    /// [`Context`] selected for the transform.
    type Context: Context<Type = V::Type, Value = V>;

    /// Returns the [`Context`] in which the differentiation transform must execute.
    fn resolve(self, primal: &Input) -> Result<Self::Context, DifferentiationError>;
}

/// Type state indicating that a [`DifferentiationBuilder`] has no runtime captures.
#[derive(Copy, Clone, Debug, Default)]
pub struct WithoutCapture;

/// Type state containing the dynamic non-differentiated runtime values supplied to a [`DifferentiationBuilder`].
#[derive(Clone, Debug)]
pub struct WithCaptures<Capture>(Capture);

/// Type state indicating that a [`DifferentiationBuilder`] differentiates only its primary output.
#[derive(Copy, Clone, Debug, Default)]
pub struct WithoutAuxiliary;

/// Type state indicating that a [`DifferentiationBuilder`] returns a non-differentiated auxiliary output.
#[derive(Copy, Clone, Debug, Default)]
pub struct WithAuxiliary;

/// Type state selecting ordinary real-valued differentiation semantics.
#[derive(Copy, Clone, Debug, Default)]
pub struct RealLinearity;

impl private::Sealed for RealLinearity {}

impl DifferentiationBuilderLinearityMode for RealLinearity {
    const HOLOMORPHIC: bool = false;
}

/// Type state selecting complex-linear differentiation under a holomorphy promise.
#[derive(Copy, Clone, Debug, Default)]
pub struct HolomorphicLinearity;

impl private::Sealed for HolomorphicLinearity {}

impl DifferentiationBuilderLinearityMode for HolomorphicLinearity {
    const HOLOMORPHIC: bool = true;
}

/// Type state indicating that the execution context is recovered from the active primals.
#[derive(Copy, Clone, Debug, Default)]
pub struct WithoutContext;

impl private::Sealed for WithoutContext {}

impl<V: Value<ExecutionDomain: Context<Type = V::Type, Value = V>>, Input: Parameterized<V>>
    DifferentiationBuilderContext<V, Input> for WithoutContext
{
    type Context = V::ExecutionDomain;

    #[inline]
    fn resolve(self, primal: &Input) -> Result<V::ExecutionDomain, DifferentiationError> {
        primal.parameters().next().map(Value::execution_domain).ok_or(DifferentiationError::EmptyInput)
    }
}

/// Type state containing the execution context explicitly selected for a [`DifferentiationBuilder`].
#[derive(Clone, Debug)]
pub struct WithContext<C>(C);

impl<C> private::Sealed for WithContext<C> {}

impl<C: Context<Type = V::Type, Value = V>, V: Value, Input: Parameterized<V>> DifferentiationBuilderContext<V, Input>
    for WithContext<C>
{
    type Context = C;

    #[inline]
    fn resolve(self, _primal: &Input) -> Result<C, DifferentiationError> {
        Ok(self.0)
    }
}

/// Execution [`Context`] selected by a [`DifferentiationBuilder`] type state.
pub type DifferentiationBuilderExecutionContext<ContextMode, V, Input> =
    <ContextMode as DifferentiationBuilderContext<V, Input>>::Context;

// TODO(eaplatanios): Review from here onwards.

/// Configures a value-level automatic differentiation transform.
///
/// Use [`differentiate_at`] or [`Differentiate::differentiate_at`] to construct this builder from the active primals.
/// Each terminal method takes the differentiated closure as its last argument, so the builder state determines the
/// closure's parameter types and they rarely need to be annotated at the call site.
/// [`in_context`](Self::in_context) selects a specific execution context; otherwise, terminal methods recover one from
/// the first active primal leaf.
/// [`with_captures`](Self::with_captures) supplies dynamic runtime values that participate in the primal computation
/// but are excluded from derivative bookkeeping. The active input and captures can have unrelated [`Parameterized`]
/// structures: for example, a derived model struct can be active while a tuple containing batches, targets, and
/// optional runtime state is captured. The builder reparameterizes both structures from runtime leaves `V` to the
/// terminal's tracer leaves before invoking the closure. Unlike [`stop_gradient()`](crate::stop_gradient()), captures
/// do not allocate Jacobian or Hessian basis families at all. They are also distinct from compilation captures stored
/// by [`ClosedProgram`](crate::ClosedProgram): differentiation captures remain explicit runtime inputs to this
/// closure, and changing their values does not make them static specialization arguments.
/// Captures follow the same operational validity rules as ordinary values. If a capture is incompatible with the
/// selected context, the operation or execution boundary that uses it returns its normal context- or backend-specific
/// error; an unused capture is harmless and does not fail preflight validation.
///
/// [`with_aux`](Self::with_aux) declares that the closure returns `(output, auxiliary)`. `auxiliary` may be any third,
/// independently shaped [`Parameterized`] tree and is reconstructed with runtime leaves without being differentiated.
/// [`holomorphic`](Self::holomorphic) opts the active inputs and differentiated outputs into complex-linear validation.
/// These modifiers are orthogonal type-state transitions, avoiding combined function-name variants.
///
/// All parameter leaves in one invocation use the same runtime value family `V`. The structures are otherwise fully
/// polymorphic: tuples, arrays, [`Vec`]s, maps, and types deriving `Parameterized` may be nested freely. Derived types
/// additionally support optional parameter fields and retain non-parameter metadata. A dynamic value family such as
/// [`Array`](crate::Array) may represent different element types, shardings, and placements at individual
/// leaves.
#[derive(Clone, Debug)]
pub struct DifferentiationBuilder<
    Input,
    Capture = WithoutCapture,
    Auxiliary = WithoutAuxiliary,
    Linearity = RealLinearity,
    ContextMode = WithoutContext,
> {
    /// Active primal inputs at which the transform is evaluated.
    primals: Input,

    /// Dynamic nondifferentiated runtime captures, or [`WithoutCapture`].
    captures: Capture,

    /// Auxiliary-output type state.
    auxiliary: Auxiliary,

    /// Linearity type state.
    linearity: Linearity,

    /// Execution-context selection state.
    context: ContextMode,
}

impl<Input, Capture, Auxiliary, Linearity>
    DifferentiationBuilder<Input, Capture, Auxiliary, Linearity, WithoutContext>
{
    /// Selects the context in which the differentiation transform executes.
    ///
    /// Without this modifier, terminal methods recover the context from the first active primal leaf.
    #[inline]
    pub fn in_context<C: Context>(
        self,
        context: &C,
    ) -> DifferentiationBuilder<Input, Capture, Auxiliary, Linearity, WithContext<C>> {
        DifferentiationBuilder {
            primals: self.primals,
            captures: self.captures,
            auxiliary: self.auxiliary,
            linearity: self.linearity,
            context: WithContext(context.clone()),
        }
    }
}

impl<Input, Auxiliary, Linearity, ContextMode>
    DifferentiationBuilder<Input, WithoutCapture, Auxiliary, Linearity, ContextMode>
{
    /// Supplies dynamic runtime values that affect primal evaluation without becoming differentiation variables.
    ///
    /// `captures` may be any [`Parameterized`] tree whose leaves use the same runtime value family as the active input.
    /// The terminal method passes the closure a reparameterized copy whose leaves are its tracer type. Container shape
    /// and non-parameter metadata are preserved. Captures remain explicit runtime inputs but contribute no tangent,
    /// cotangent, Jacobian, or Hessian coordinates.
    #[inline]
    pub fn with_captures<Capture>(
        self,
        captures: Capture,
    ) -> DifferentiationBuilder<Input, WithCaptures<Capture>, Auxiliary, Linearity, ContextMode> {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(captures),
            auxiliary: self.auxiliary,
            linearity: self.linearity,
            context: self.context,
        }
    }
}

impl<Input, Capture, Linearity, ContextMode>
    DifferentiationBuilder<Input, Capture, WithoutAuxiliary, Linearity, ContextMode>
{
    /// Declares that the transformed closure returns `(differentiated_output, auxiliary_output)`.
    ///
    /// The auxiliary output may be any [`Parameterized`] tree. Its leaves are traced during primal evaluation and
    /// reconstructed as runtime values in the terminal result, but they are excluded from derivative seeding and
    /// transposition. Its structure is independent of both the active input and any captures.
    #[inline]
    pub fn with_aux(self) -> DifferentiationBuilder<Input, Capture, WithAuxiliary, Linearity, ContextMode> {
        DifferentiationBuilder {
            primals: self.primals,
            captures: self.captures,
            auxiliary: WithAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
    }
}

impl<Input, Capture, Auxiliary, ContextMode>
    DifferentiationBuilder<Input, Capture, Auxiliary, RealLinearity, ContextMode>
{
    /// Treats the differentiated computation as complex linear under a holomorphy promise.
    ///
    /// The promise applies only to active inputs and differentiated outputs. Runtime captures are exempt because they
    /// are fixed coefficients rather than coordinates of the differentiated map. Holomorphic mode is available for
    /// Jacobian, Hessian, and scalar-gradient terminals; JVP, linearization, and VJP do not define a separate
    /// holomorphic validation contract.
    #[inline]
    pub fn holomorphic(self) -> DifferentiationBuilder<Input, Capture, Auxiliary, HolomorphicLinearity, ContextMode> {
        DifferentiationBuilder {
            primals: self.primals,
            captures: self.captures,
            auxiliary: self.auxiliary,
            linearity: HolomorphicLinearity,
            context: self.context,
        }
    }
}

impl<Input, ContextMode> DifferentiationBuilder<Input, WithoutCapture, WithoutAuxiliary, RealLinearity, ContextMode> {
    /// Evaluates `function` at the builder's active primals and propagates `tangents` through its Jacobian.
    ///
    /// For a function `y = f(x)`, this returns the dual `(y, ẏ) = (f(x), J_f(x) · ẋ)`, where `ẋ` is the supplied
    /// tangent and `J_f(x) = ∂f/∂x`. This is direct forward-mode differentiation, analogous to
    /// [JAX's `jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html): the closure runs once on
    /// [`DifferentiationTracer`] duals, and every operation propagates its primal and tangent together through its
    /// [`DifferentiableOperation`] rule.
    ///
    /// The selected context determines execution. An eager context computes both dual halves immediately, so host
    /// control flow may inspect concrete primals. A staging context records primal and tangent operations in one fused
    /// trace, and data-dependent Rust control flow on a traced primal is unavailable. Structural-zero tangents remain
    /// symbolic between operations and are materialized at the output boundary through [`ResidualZeroProvider`],
    /// preserving runtime geometry for dynamically shaped values. Nested transforms differentiate through these duals
    /// and therefore compose forward, reverse-over-forward, and higher-order differentiation.
    ///
    /// The closure executes exactly as written: the transform does not trim dead code, and observable effects fire as
    /// the closure runs.
    ///
    /// The active primal tree must contain at least one leaf; otherwise this returns
    /// [`DifferentiationError::EmptyInput`].
    ///
    /// # Parameters
    ///
    ///   - `tangents`: Tangent tree matching the structure of the builder's active primals.
    ///   - `function`: Function whose primal output and directional derivative are evaluated.
    pub fn jvp<V, Output, F>(
        self,
        tangents: Input::To<V>,
        function: F,
    ) -> Result<(Output::To<V>, Output::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: DifferentiableOperation<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>
                               + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                Family: ParameterizedFamily<
                    DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        Output: Parameterized<
                DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .jvp(tangents, |input, ()| function(input))
    }

    /// Linearizes `function` at the builder's active primals, returning its value and a reusable [`Pushforward`].
    ///
    /// For `y = f(x)`, this returns `y` together with the linear map `ẋ ↦ ẏ = J_f(x) · ẋ` at the fixed
    /// linearization point `x`, analogous to
    /// [JAX's `linearize`](https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html).
    /// Unlike [`jvp`](Self::jvp), which evaluates one `(primal, tangent)` pair, linearization performs the nonlinear
    /// primal work once. The returned [`Pushforward`] can then apply the same Jacobian to any number of tangent trees
    /// without retracing or redifferentiating `function`.
    ///
    /// Internally, the closure runs on [`LinearizationTracer`] duals over a [`PartialEvaluationContext`]. Primal halves
    /// are known and execute in the selected context, while tangent halves are unknown and residualize into a linear
    /// program `(ẋ, r) ↦ ẏ`. The residuals `r` contain the values from the linearization point needed by that
    /// program, and the returned pushforward closes over them. Under an eager context, concrete primal values may drive
    /// host control flow and only the taken path is linearized; under a staging context, primal work composes into the
    /// enclosing trace.
    ///
    /// The active primal tree must contain at least one leaf; otherwise this returns
    /// [`DifferentiationError::EmptyInput`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Function to evaluate and linearize at the builder's active primals.
    pub fn linearize<V, Output, F>(
        self,
        function: F,
    ) -> Result<
        (
            Output::To<V>,
            Pushforward<DifferentiationBuilderExecutionContext<ContextMode, V, Input>, Input, Output::To<V>>,
        ),
        DifferentiationError,
    >
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .linearize(|input, ()| function(input))
    }

    /// Reverse-mode-differentiates `function`, returning its value and a reusable [`Pullback`].
    ///
    /// For `y = f(x)`, this returns `y` together with the transposed linear map `ȳ ↦ x̄ = J_f(x)ᵀ · ȳ`,
    /// analogous to [JAX's `vjp`](https://docs.jax.dev/en/latest/_autosummary/jax.vjp.html). Applying the pullback maps
    /// an output cotangent tree to the corresponding input cotangent tree at the builder's fixed primal point.
    ///
    /// Reverse mode first performs the partial-evaluation-backed linearization described by
    /// [`linearize`](Self::linearize), then transposes its linear program by applying [`TransposableOperation`] rules
    /// in reverse dataflow order. The returned pullback closes that transposed program over the saved linearization
    /// residuals, so [`Pullback::apply`] handles residual arguments and reconstructs the structured input cotangents;
    /// callers only provide output cotangents.
    ///
    /// The active primal tree must contain at least one leaf; otherwise this returns
    /// [`DifferentiationError::EmptyInput`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Function to evaluate and reverse-mode-differentiate at the builder's active primals.
    pub fn vjp<V, Output, F>(
        self,
        function: F,
    ) -> Result<
        (Output::To<V>, Pullback<DifferentiationBuilderExecutionContext<ContextMode, V, Input>, Input, Output::To<V>>),
        DifferentiationError,
    >
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .vjp(|input, ()| function(input))
    }
}

impl<Input, Linearity: DifferentiationBuilderLinearityMode, ContextMode>
    DifferentiationBuilder<Input, WithoutCapture, WithoutAuxiliary, Linearity, ContextMode>
{
    /// Computes both the scalar value of `function` and its reverse-mode gradient.
    ///
    /// For a real scalar `y = f(x)`, this returns `(f(x), ∇f(x))`, where `∇f(x) = J_f(x)ᵀ · 1` is the pullback of the
    /// multiplicative-identity cotangent seed. In [`HolomorphicLinearity`] mode, a holomorphic complex scalar function
    /// `y = f(z)` instead returns `(f(z), ∂f/∂z)`: the complex derivative itself, rather than a conjugate
    /// steepest-ascent direction. Calling [`holomorphic`](Self::holomorphic) promises that the function is holomorphic;
    /// the transform validates complex input and output types but cannot prove the Cauchy-Riemann equations.
    ///
    /// `function` may return its traced scalar directly or in a [`Result`] whose error converts into
    /// [`DifferentiationError`], as specified by [`MaybeFallible`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Scalar-valued function to evaluate and differentiate at the builder's active primals.
    pub fn value_and_gradient<V, Output, F>(self, function: F) -> Result<(V, Input::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                DifferentiationError,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .value_and_gradient(|input, ()| function(input))
    }

    /// Computes the reverse-mode gradient of a scalar-valued `function`.
    ///
    /// For a real scalar `y = f(x)`, this returns `∇f(x) = J_f(x)ᵀ · 1`, analogous to
    /// [JAX's `grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html). In [`HolomorphicLinearity`] mode, a
    /// holomorphic complex scalar function returns `∂f/∂z`. This is the gradient-only counterpart of
    /// [`value_and_gradient`](Self::value_and_gradient), discarding the primal scalar output while retaining the same
    /// output validation and [`MaybeFallible`] closure contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Scalar-valued function to differentiate at the builder's active primals.
    pub fn gradient<V, Output, F>(self, function: F) -> Result<Input::To<V>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                DifferentiationError,
            >,
    {
        self.value_and_gradient(function).map(|(_, gradient)| gradient)
    }

    /// Materializes the complete [`Jacobian`] of `function` using forward-mode differentiation.
    ///
    /// For `y = f(x)`, the Jacobian is the linear map `J_f(x) = ∂f/∂x` satisfying `ẏ = J_f(x) · ẋ`. This
    /// method linearizes `function` once, applies the resulting [`Pushforward`] to a packed basis of the finite input
    /// coordinate space, and assembles the resulting columns into a Jacobian. Each block corresponds to one output
    /// leaf and one input leaf in deterministic output-major/input-minor order; for arrays, a block places output axes
    /// before input axes.
    ///
    /// Let `n` and `m` be the total input and output coordinate counts, `T_linearize` the one-time cost of constructing
    /// the pushforward, and `T_pushforward` the cost of one tangent direction. Forward materialization evaluates an
    /// `n`-way packed pushforward, so its derivative work is approximately
    /// `O(T_linearize + n · T_pushforward)`. If `R_forward` is the shared residual storage and `M_pushforward` the
    /// additional peak memory for one direction, its working memory excluding the result is approximately
    /// `O(R_forward + n · M_pushforward)`. The materialized Jacobian itself necessarily occupies `Θ(mn)` storage.
    /// [`jacobian_reverse`](Self::jacobian_reverse) produces the same representation and is generally preferable when
    /// `m < n`; use forward mode when `n <= m`. Packing may execute directions in parallel, but does not change these
    /// total-work or storage scalings.
    ///
    /// Ordinary mode requires real active input leaves but permits complex differentiated outputs.
    /// [`HolomorphicLinearity`] mode treats the derivative as complex linear and requires every active input and
    /// differentiated output leaf to be complex; selecting it is a promise of holomorphy that type validation cannot
    /// prove.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose complete Jacobian is materialized at the builder's active primals.
    pub fn jacobian_forward<V, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .jacobian_forward(|input, ()| function(input))
    }

    /// Materializes the complete [`Jacobian`] of `function` using reverse-mode differentiation.
    ///
    /// For `y = f(x)`, the pullback maps `ȳ` to `x̄ = J_f(x)ᵀ · ȳ`, where `J_f(x) = ∂f/∂x`. This method
    /// constructs one [`Pullback`], applies it to a packed basis of the finite output coordinate space, and reorients
    /// the resulting rows into the same output-major/input-minor [`Jacobian`] representation returned by
    /// [`jacobian_forward`](Self::jacobian_forward). Array blocks place output axes before input axes.
    ///
    /// Let `n` and `m` be the total input and output coordinate counts, `T_vjp` the one-time cost of constructing the
    /// pullback, and `T_pullback` the cost of one cotangent direction. Reverse materialization evaluates an `m`-way
    /// packed pullback, so its derivative work is approximately `O(T_vjp + m · T_pullback)`. If `R_reverse` is the
    /// shared residual storage and `M_pullback` the additional peak memory for one direction, its working memory
    /// excluding the result is approximately `O(R_reverse + m · M_pullback)`. The materialized Jacobian itself
    /// necessarily occupies `Θ(mn)` storage. Reverse mode is generally preferable when `m < n`; use
    /// [`jacobian_forward`](Self::jacobian_forward) when `n <= m`. Packing may execute directions in parallel, but does
    /// not change these total-work or storage scalings.
    ///
    /// Ordinary mode requires real differentiated output leaves but permits complex active inputs.
    /// [`HolomorphicLinearity`] mode treats the derivative as complex linear and requires every active input and
    /// differentiated output leaf to be complex; selecting it is a promise of holomorphy that type validation cannot
    /// prove.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose complete Jacobian is materialized at the builder's active primals.
    pub fn jacobian_reverse<V, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + TransposableOperation<
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                > + ResidualZeroProvider<V::Type>
                               + From<AddOperation<V::Type>>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .jacobian_reverse(|input, ()| function(input))
    }

    /// Materializes the complete [`Hessian`] of `function` using forward-over-reverse differentiation.
    ///
    /// For `y = f(x)`, each entry is `H_f(x)[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])`. The inner reverse transform
    /// materializes `J_f(x)` by applying a [`Pullback`] to packed output-coordinate basis cotangents; the outer forward
    /// transform differentiates those Jacobian entries by applying a [`Pushforward`] to packed input-coordinate basis
    /// tangents. The result stores blocks in output-major/first-input-major/second-input-minor order. For arrays, a
    /// block places output axes first, followed by first-input axes and then second-input axes.
    ///
    /// Let `n` and `m` be the total input and output coordinate counts. The transform performs an `m`-way packed inner
    /// pullback and an `n`-way packed outer pushforward. If `T_inner_vjp` and `T_outer_linearize` are the one-time
    /// costs of those transforms and `T_inner_pullback` and `T_outer_pushforward` are their per-direction costs, the
    /// derivative work is approximately
    /// `O(T_inner_vjp + m · T_inner_pullback + T_outer_linearize + n · T_outer_pushforward)`. If `R_hessian` is the
    /// nested residual storage and `M_inner_pullback` and `M_outer_pushforward` are the additional peak memories for
    /// one direction, working memory excluding the result is approximately
    /// `O(R_hessian + m · M_inner_pullback + n · M_outer_pushforward)`. The materialized Hessian necessarily occupies
    /// `Θ(mn²)` storage. This decomposition is particularly well suited to scalar-output functions, where `m = 1`:
    /// reverse mode constructs the gradient with one cotangent direction and forward mode differentiates that
    /// input-to-gradient map. Packing may execute directions in parallel, but does not change these scalings.
    ///
    /// Complete materialization requires finite, statically enumerable coordinate spaces. Ordinary mode requires real
    /// active input and differentiated output leaves. [`HolomorphicLinearity`] mode requires them all to be complex and
    /// treats both nested transforms as complex linear; selecting it promises holomorphy but cannot prove it.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function whose complete Hessian is materialized at the builder's active primals.
    pub fn hessian<C, V, Output, F>(
        self,
        function: F,
    ) -> Result<Hessian<C::Type, V, Input::To<C::Type>, Output::To<C::Type>>, DifferentiationError>
    where
        C: Context<
                Type: DenseDifferentiableType<C>
                          + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                Value = V,
                Operation: PartiallyEvaluatableOperation<C>
                               + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
                               + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<PartialEvaluationContext<C>>
                               + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<
                    PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>,
                > + TransposableOperation<C::Constant, C::Operation>
                               + ResidualZeroProvider<C::Type>
                               + From<AddOperation<C::Type>>,
            >,
        V: Value<Type = C::Type>,
        ContextMode: DifferentiationBuilderContext<V, Input, Context = C>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                To<C::Type>: Clone,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Input::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Input::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                    To<C::Type> = Input::To<C::Type>,
                >,
                Family: ParameterizedFamily<C::Type>
                            + ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<C::Type>: Clone,
                Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithoutAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .hessian(|input, ()| function(input))
    }
}

impl<Input, Linearity: DifferentiationBuilderLinearityMode, ContextMode>
    DifferentiationBuilder<Input, WithoutCapture, WithAuxiliary, Linearity, ContextMode>
{
    /// Computes a scalar value, its reverse-mode gradient, and non-differentiated auxiliary output.
    ///
    /// For `(y, a) = f(x)` with a real scalar `y`, this returns `((y, a), J_y(x)ᵀ · 1)`. Only `y` receives the
    /// multiplicative-identity cotangent seed; every auxiliary leaf receives a zero seed and is reconstructed as a
    /// primal runtime value. In [`HolomorphicLinearity`] mode, complex `y` instead yields the complex derivative
    /// `∂y/∂z`, under the caller's holomorphy promise. The auxiliary tree may have a structure unrelated to the active
    /// input.
    ///
    /// The closure may return `(y, a)` directly or in a [`Result`] through the [`MaybeFallible`] contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function returning the differentiated scalar and auxiliary output.
    pub fn value_and_gradient<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                (
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                ),
                DifferentiationError,
            >,
        Aux: Parameterized<
                V,
                To<V> = Aux,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    To = Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                >,
            >,
        (
            LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
            Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ): Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                To<V> = (V, Aux),
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .value_and_gradient(|input, ()| function(input))
    }

    /// Computes a scalar reverse-mode gradient and returns non-differentiated auxiliary output.
    ///
    /// For `(y, a) = f(x)` with real scalar `y`, this returns `(J_y(x)ᵀ · 1, a)`. In [`HolomorphicLinearity`] mode it
    /// returns `(∂y/∂z, a)` under the caller's holomorphy promise. This is the gradient-only counterpart of
    /// [`value_and_gradient`](Self::value_and_gradient): it discards `y`, while auxiliary leaves remain primal values
    /// with zero cotangent seeds. The closure supports the same [`MaybeFallible`] result forms.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function returning the differentiated scalar and auxiliary output.
    pub fn gradient<V, Output, Aux, F>(self, function: F) -> Result<(Input::To<V>, Aux), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                (
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                ),
                DifferentiationError,
            >,
        Aux: Parameterized<
                V,
                To<V> = Aux,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    To = Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                >,
            >,
        (
            LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
            Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ): Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                To<V> = (V, Aux),
                Family: ParameterizedFamily<V>,
            >,
    {
        self.value_and_gradient(function).map(|((_, auxiliary), gradient)| (gradient, auxiliary))
    }

    /// Materializes a complete forward-mode [`Jacobian`] and returns non-differentiated auxiliary output.
    ///
    /// For `(y, a) = f(x)`, only `J_y(x) = ∂y/∂x` is materialized: the method linearizes once, applies its
    /// [`Pushforward`] to a packed input-coordinate basis, and assembles output-major/input-minor blocks. Auxiliary
    /// leaves are recovered from the primal trace and returned unchanged; they add no Jacobian coordinates.
    /// Forward mode is generally preferable when the input coordinate count does not exceed the differentiated output
    /// coordinate count. Ordinary and [`HolomorphicLinearity`] element-type rules are the same as for the non-auxiliary
    /// forward Jacobian.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function returning the differentiated output and auxiliary output.
    pub fn jacobian_forward<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, Aux::To<V>), DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .jacobian_forward(|input, ()| function(input))
    }

    /// Materializes a complete reverse-mode [`Jacobian`] and returns non-differentiated auxiliary output.
    ///
    /// For `(y, a) = f(x)`, only `J_y(x) = ∂y/∂x` is materialized: the method constructs one [`Pullback`], applies it
    /// to a packed differentiated-output basis, and assembles the same output-major/input-minor blocks as forward mode.
    /// Auxiliary leaves are recovered from the primal trace and returned unchanged; they receive zero cotangent seeds
    /// and add no Jacobian coordinates. Reverse mode is generally preferable when the differentiated output coordinate
    /// count is smaller than the input coordinate count. Ordinary and [`HolomorphicLinearity`] element-type rules are
    /// the same as for the non-auxiliary reverse Jacobian.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function returning the differentiated output and auxiliary output.
    pub fn jacobian_reverse<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, Aux::To<V>), DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + TransposableOperation<
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                > + ResidualZeroProvider<V::Type>
                               + From<AddOperation<V::Type>>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .jacobian_reverse(|input, ()| function(input))
    }

    /// Materializes a complete forward-over-reverse [`Hessian`] and returns non-differentiated auxiliary output.
    ///
    /// For `(y, a) = f(x)`, this returns the blocks `H_y(x)[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])` in
    /// output-major/first-input-major/second-input-minor order. Only `y` participates in the inner reverse Jacobian and
    /// outer forward differentiation; auxiliary leaves are recovered as primal runtime values and introduce no
    /// derivative coordinates. Complete materialization requires finite coordinate spaces and occupies `Θ(mn²)` for `n`
    /// active-input and `m` differentiated-output coordinates. Ordinary and [`HolomorphicLinearity`] element-type rules
    /// are the same as for the non-auxiliary Hessian.
    ///
    /// # Parameters
    ///
    ///   - `function`: Function returning the differentiated output and auxiliary output.
    pub fn hessian<C, V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Hessian<C::Type, V, Input::To<C::Type>, Output::To<C::Type>>, Aux::To<V>), DifferentiationError>
    where
        C: Context<
                Type: DenseDifferentiableType<C>
                          + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                Value = V,
                Operation: PartiallyEvaluatableOperation<C>
                               + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
                               + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<PartialEvaluationContext<C>>
                               + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<
                    PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>,
                > + TransposableOperation<C::Constant, C::Operation>
                               + ResidualZeroProvider<C::Type>
                               + From<AddOperation<C::Type>>,
            >,
        V: Value<Type = C::Type>,
        ContextMode: DifferentiationBuilderContext<V, Input, Context = C>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                To<C::Type>: Clone,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Input::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Input::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                    To<C::Type> = Input::To<C::Type>,
                >,
                Family: ParameterizedFamily<C::Type>
                            + ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<C::Type>: Clone,
                Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<LinearizationTracer<C>>: Parameterized<LinearizationTracer<C>, To<V> = Aux::To<V>>,
                Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<V>,
            >,
    {
        DifferentiationBuilder {
            primals: self.primals,
            captures: WithCaptures(()),
            auxiliary: WithAuxiliary,
            linearity: self.linearity,
            context: self.context,
        }
        .hessian(|input, ()| function(input))
    }
}

impl<Input, Capture, ContextMode>
    DifferentiationBuilder<Input, WithCaptures<Capture>, WithoutAuxiliary, RealLinearity, ContextMode>
{
    /// Evaluates `function` and its Jacobian-vector product while holding runtime captures fixed.
    ///
    /// For `y = f(x; c)`, this returns `(y, ẏ) = (f(x; c), (∂f/∂x)(x; c) · ẋ)`. Only the active primals `x`
    /// pair with the supplied tangents; captured values `c` affect the primal and tangent computations as runtime
    /// coefficients but receive no tangents
    /// and contribute no derivative coordinates. The closure otherwise follows the direct forward-mode execution,
    /// structural-zero, nesting, and eager-versus-staged semantics of the capture-free [`jvp`](Self::jvp).
    ///
    /// The active primal tree, not the capture tree, must contain at least one leaf.
    ///
    /// # Parameters
    ///
    ///   - `tangents`: Tangent tree matching the structure of the builder's active primals.
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn jvp<V, Output, F>(
        self,
        tangents: Input::To<V>,
        function: F,
    ) -> Result<(Output::To<V>, Output::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: DifferentiableOperation<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>
                               + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                Family: ParameterizedFamily<
                    DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
                ParameterStructure: Debug + PartialEq,
            >,
        Capture: Parameterized<
                V,
                Family: ParameterizedFamily<
                    DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                DifferentiationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        context.jvp(function, self.primals, tangents, self.captures.0)
    }

    /// Linearizes `function` with respect to the active primals while holding runtime captures fixed.
    ///
    /// For `y = f(x; c)`, this returns `f(x; c)` and the reusable map
    /// `ẋ ↦ ẏ = (∂f/∂x)(x; c) · ẋ`. Partial evaluation executes known primal and capture work in the
    /// selected context and residualizes the unknown tangent computation. The returned [`Pushforward`] closes over
    /// both ordinary linearization residuals and any runtime capture values it needs, so repeated applications require
    /// only active tangent trees and never tangent values for captures.
    ///
    /// Host-control-flow and eager-versus-staged behavior are the same as for the capture-free
    /// [`linearize`](Self::linearize). The active primal tree, not the capture tree, must contain at least one leaf.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn linearize<V, Output, F>(
        self,
        function: F,
    ) -> Result<
        (
            Output::To<V>,
            Pushforward<DifferentiationBuilderExecutionContext<ContextMode, V, Input>, Input, Output::To<V>>,
        ),
        DifferentiationError,
    >
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        context.linearize(function, self.primals, self.captures.0)
    }

    /// Reverse-mode-differentiates `function` with respect to the active primals while holding captures fixed.
    ///
    /// For `y = f(x; c)`, this returns `f(x; c)` and the reusable map
    /// `ȳ ↦ x̄ = (∂f/∂x)(x; c)ᵀ · ȳ`. Reverse mode linearizes the function and transposes only the active
    /// tangent program. Captures may remain among the residual values closed over by the [`Pullback`], but callers
    /// provide only output cotangents and receive cotangents only for `x`; no capture cotangent tree is constructed.
    ///
    /// The active primal tree, not the capture tree, must contain at least one leaf.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn vjp<V, Output, F>(
        self,
        function: F,
    ) -> Result<
        (Output::To<V>, Pullback<DifferentiationBuilderExecutionContext<ContextMode, V, Input>, Input, Output::To<V>>),
        DifferentiationError,
    >
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        context.vjp(function, self.primals, self.captures.0)
    }
}

impl<Input, Capture, Linearity: DifferentiationBuilderLinearityMode, ContextMode>
    DifferentiationBuilder<Input, WithCaptures<Capture>, WithoutAuxiliary, Linearity, ContextMode>
{
    /// Materializes the complete forward-mode [`Jacobian`] with respect to the active primals only.
    ///
    /// For `y = f(x; c)`, this materializes `J_x f(x; c) = ∂f/∂x` by applying one [`Pushforward`] to a packed
    /// basis of active-input coordinates. Captures affect every evaluated column as fixed runtime coefficients but
    /// contribute no basis directions or Jacobian blocks. Blocks use deterministic
    /// output-major/active-input-minor order, with output axes before input axes for arrays.
    ///
    /// Forward materialization is generally preferable when the number of active-input coordinates does not exceed the
    /// number of differentiated-output coordinates. Ordinary mode requires real active inputs but permits complex
    /// outputs; [`HolomorphicLinearity`] mode requires complex active inputs and differentiated outputs and treats the
    /// map as complex linear under the caller's holomorphy promise.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn jacobian_forward<V, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        let (jacobian, ()) = jacobian_forward_in_context(
            &context,
            |input, captures| Ok((function(input, captures)?, ())),
            self.primals,
            self.captures.0,
            Linearity::HOLOMORPHIC,
        )?;
        Ok(jacobian)
    }

    /// Materializes the complete reverse-mode [`Jacobian`] with respect to the active primals only.
    ///
    /// For `y = f(x; c)`, this constructs a [`Pullback`] for `J_x f(x; c)ᵀ`, applies it to a packed basis of
    /// differentiated-output coordinates, and assembles the same output-major/active-input-minor blocks as forward
    /// mode. Captures affect the pullback as fixed runtime coefficients but receive no cotangents and contribute no
    /// Jacobian blocks.
    ///
    /// Reverse materialization is generally preferable when the differentiated-output coordinate count is smaller than
    /// the active-input coordinate count. Ordinary mode requires real differentiated outputs but permits complex active
    /// inputs; [`HolomorphicLinearity`] mode requires complex active inputs and outputs and treats the map as complex
    /// linear under the caller's holomorphy promise.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn jacobian_reverse<V, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + TransposableOperation<
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                > + ResidualZeroProvider<V::Type>
                               + From<AddOperation<V::Type>>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        let (jacobian, ()) = jacobian_reverse_in_context(
            &context,
            |input, captures| Ok((function(input, captures)?, ())),
            self.primals,
            self.captures.0,
            Linearity::HOLOMORPHIC,
        )?;
        Ok(jacobian)
    }

    /// Materializes the complete forward-over-reverse [`Hessian`] with respect to the active primals only.
    ///
    /// For `y = f(x; c)`, this materializes
    /// `H_x f(x; c)[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])`. Captures remain fixed through both the inner reverse
    /// Jacobian and the outer forward transform: they may affect every derivative value, but introduce neither first-
    /// nor second-order coordinates. Blocks use output-major/first-active-input-major/second-active-input-minor order.
    ///
    /// With `n` active-input and `m` differentiated-output coordinates, the result occupies `Θ(mn²)` storage. Complete
    /// materialization requires finite coordinate spaces. Ordinary mode requires real active inputs and outputs;
    /// [`HolomorphicLinearity`] mode requires complex active inputs and outputs and treats both nested transforms as
    /// complex linear under the caller's holomorphy promise.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function receiving reparameterized active primals and captures.
    pub fn hessian<C, V, Output, F>(
        self,
        function: F,
    ) -> Result<Hessian<C::Type, V, Input::To<C::Type>, Output::To<C::Type>>, DifferentiationError>
    where
        C: Context<
                Type: DenseDifferentiableType<C>
                          + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                Value = V,
                Operation: PartiallyEvaluatableOperation<C>
                               + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
                               + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<PartialEvaluationContext<C>>
                               + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<
                    PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>,
                > + TransposableOperation<C::Constant, C::Operation>
                               + ResidualZeroProvider<C::Type>
                               + From<AddOperation<C::Type>>,
            >,
        V: Value<Type = C::Type>,
        ContextMode: DifferentiationBuilderContext<V, Input, Context = C>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
            Capture::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                To<C::Type>: Clone,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Input::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Input::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                    To<C::Type> = Input::To<C::Type>,
                >,
                Family: ParameterizedFamily<C::Type>
                            + ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Capture::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Capture::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                >,
                Family: ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<C::Type>: Clone,
                Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        let (hessian, ()) = hessian_in_context(
            &context,
            |input, captures| Ok((function(input, captures)?, ())),
            self.primals,
            self.captures.0,
            Linearity::HOLOMORPHIC,
        )?;
        Ok(hessian)
    }

    /// Computes a scalar value and its reverse-mode gradient with respect to the active primals only.
    ///
    /// For `y = f(x; c)` with real scalar `y`, this returns `(f(x; c), (∂f/∂x)(x; c)ᵀ · 1)`. Captures are fixed
    /// runtime coefficients: they may change the value and gradient but receive no gradient result. In
    /// [`HolomorphicLinearity`] mode, complex `y` yields `(f(x; c), ∂f/∂x)` under the caller's holomorphy promise. The
    /// closure supports the [`MaybeFallible`] result contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary scalar-valued function receiving reparameterized active primals and captures.
    pub fn value_and_gradient<V, Output, F>(self, function: F) -> Result<(V, Input::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                DifferentiationError,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        value_and_gradient_in_context(&context, function, self.primals, self.captures.0, Linearity::HOLOMORPHIC)
    }

    /// Computes a scalar reverse-mode gradient with respect to the active primals only.
    ///
    /// For real scalar `y = f(x; c)`, this returns `(∂f/∂x)(x; c)ᵀ · 1`; in [`HolomorphicLinearity`] mode it returns
    /// the complex derivative with respect to `x`. This discards the primal scalar from
    /// [`value_and_gradient`](Self::value_and_gradient). Captures remain fixed runtime coefficients and receive no
    /// gradient result. The closure supports the same [`MaybeFallible`] result contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary scalar-valued function receiving reparameterized active primals and captures.
    #[inline]
    pub fn gradient<V, Output, F>(self, function: F) -> Result<Input::To<V>, DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                DifferentiationError,
            >,
    {
        self.value_and_gradient(function).map(|(_, gradient)| gradient)
    }
}

impl<Input, Capture, Linearity: DifferentiationBuilderLinearityMode, ContextMode>
    DifferentiationBuilder<Input, WithCaptures<Capture>, WithAuxiliary, Linearity, ContextMode>
{
    /// Materializes a forward-mode [`Jacobian`] over active primals and returns auxiliary output with captures fixed.
    ///
    /// For `(y, a) = f(x; c)`, this materializes `∂y/∂x` from a packed active-input basis. Captures `c` remain runtime
    /// coefficients, and auxiliary leaves `a` are reconstructed as primal values; neither introduces Jacobian
    /// coordinates. Blocks use output-major/active-input-minor order. Forward mode is generally preferable when the
    /// active-input coordinate count does not exceed the differentiated-output coordinate count.
    /// [`HolomorphicLinearity`] mode applies the same complex-linearity promise and validation as the non-auxiliary
    /// transform.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function returning the differentiated output and auxiliary output.
    pub fn jacobian_forward<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, Aux::To<V>), DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + ResidualZeroProvider<V::Type>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        jacobian_forward_in_context(&context, function, self.primals, self.captures.0, Linearity::HOLOMORPHIC)
    }

    /// Materializes a reverse-mode [`Jacobian`] over active primals and returns auxiliary output with captures fixed.
    ///
    /// For `(y, a) = f(x; c)`, this materializes `∂y/∂x` from a packed differentiated-output cotangent basis. Captures
    /// `c` remain runtime coefficients, while auxiliary leaves `a` receive zero cotangent seeds and are reconstructed
    /// as primal values; neither contributes derivative coordinates. Reverse mode is generally preferable when the
    /// differentiated-output coordinate count is smaller than the active-input coordinate count.
    /// [`HolomorphicLinearity`] mode applies the same complex-linearity promise and validation as the non-auxiliary
    /// transform.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function returning the differentiated output and auxiliary output.
    pub fn jacobian_reverse<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Jacobian<V::Type, V, Input::To<V::Type>, Output::To<V::Type>>, Aux::To<V>), DifferentiationError>
    where
        V: Value<Type: DenseDifferentiableType<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: Context<
                Type = V::Type,
                Value = V,
                Operation: PartiallyEvaluatableOperation<
                    DifferentiationBuilderExecutionContext<ContextMode, V, Input>,
                > + PartiallyEvaluatableOperation<
                    TracingContext<
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                    >,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + TransposableOperation<
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Constant,
                    <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation,
                > + ResidualZeroProvider<V::Type>
                               + From<AddOperation<V::Type>>,
            >,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                > + ParameterizedFamily<V::Type>,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V> + ParameterizedFamily<V::Type>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        jacobian_reverse_in_context(&context, function, self.primals, self.captures.0, Linearity::HOLOMORPHIC)
    }

    /// Materializes a forward-over-reverse [`Hessian`] over active primals and returns auxiliary output.
    ///
    /// For `(y, a) = f(x; c)`, this materializes `H_x y[k, i, j] = ∂²y[k]/(∂x[i] ∂x[j])`. Captures remain fixed through
    /// both derivative levels, while auxiliary leaves are reconstructed from primal values; neither contributes first-
    /// or second-order coordinates. Blocks use output-major/first-active-input-major/second-active-input-minor order,
    /// and the result occupies `Θ(mn²)` for `n` active-input and `m` differentiated-output coordinates. Complete
    /// materialization requires finite coordinate spaces. [`HolomorphicLinearity`] mode treats both nested transforms
    /// as complex linear under the caller's promise.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function returning the differentiated output and auxiliary output.
    pub fn hessian<C, V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<(Hessian<C::Type, V, Input::To<C::Type>, Output::To<C::Type>>, Aux::To<V>), DifferentiationError>
    where
        C: Context<
                Type: DenseDifferentiableType<C>
                          + DenseDifferentiableType<DifferentiationContext<PartialEvaluationContext<C>>>,
                Value = V,
                Operation: PartiallyEvaluatableOperation<C>
                               + PartiallyEvaluatableOperation<DifferentiationContext<PartialEvaluationContext<C>>>
                               + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<PartialEvaluationContext<C>>
                               + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
                               + DifferentiableOperation<
                    PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>,
                > + DifferentiableOperation<
                    PartialEvaluationContext<DifferentiationContext<PartialEvaluationContext<C>>>,
                > + TransposableOperation<C::Constant, C::Operation>
                               + ResidualZeroProvider<C::Type>
                               + From<AddOperation<C::Type>>,
            >,
        V: Value<Type = C::Type>,
        ContextMode: DifferentiationBuilderContext<V, Input, Context = C>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
            Capture::To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>>,
        ) -> Result<(Output, Aux), ProgramError>,
        Input: Parameterized<
                V,
                To<V> = Input,
                To<C::Type>: Clone,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Input::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Input::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                    To<C::Type> = Input::To<C::Type>,
                >,
                Family: ParameterizedFamily<C::Type>
                            + ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                To<LinearizationTracer<C>>: Parameterized<
                    LinearizationTracer<C>,
                    To<LinearizationTracer<C>> = Capture::To<LinearizationTracer<C>>,
                    To<LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>> = Capture::To<
                        LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                    >,
                >,
                Family: ParameterizedFamily<LinearizationTracer<C>>
                            + ParameterizedFamily<
                    LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                >,
            >,
        Output: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<C::Type>: Clone,
                Family: ParameterizedFamily<C::Type> + ParameterizedFamily<LinearizationTracer<C>>,
            >,
        Aux: Parameterized<
                LinearizationTracer<DifferentiationContext<PartialEvaluationContext<C>>>,
                To<LinearizationTracer<C>>: Parameterized<LinearizationTracer<C>, To<V> = Aux::To<V>>,
                Family: ParameterizedFamily<LinearizationTracer<C>> + ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        hessian_in_context(&context, function, self.primals, self.captures.0, Linearity::HOLOMORPHIC)
    }

    /// Computes a scalar value, its active-primal gradient, and auxiliary output while holding captures fixed.
    ///
    /// For `(y, a) = f(x; c)` with real scalar `y`, this returns `((y, a), (∂y/∂x)(x; c)ᵀ · 1)`. Captures affect
    /// the computation as fixed runtime coefficients, auxiliary leaves receive zero cotangent seeds, and neither
    /// receives a gradient result. In [`HolomorphicLinearity`] mode, complex `y` yields `∂y/∂x` under the caller's
    /// holomorphy promise. The closure supports the [`MaybeFallible`] result contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function returning the differentiated scalar and auxiliary output.
    pub fn value_and_gradient<V, Output, Aux, F>(
        self,
        function: F,
    ) -> Result<((V, Aux), Input::To<V>), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                (
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                ),
                DifferentiationError,
            >,
        Aux: Parameterized<
                V,
                To<V> = Aux,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    To = Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                >,
            >,
        (
            LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
            Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ): Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                To<V> = (V, Aux),
                Family: ParameterizedFamily<V>,
            >,
    {
        let context = self.context.resolve(&self.primals)?;
        value_and_gradient_auxiliary_in_context(
            &context,
            function,
            self.primals,
            self.captures.0,
            Linearity::HOLOMORPHIC,
        )
    }

    /// Computes an active-primal scalar gradient and auxiliary output while holding captures fixed.
    ///
    /// For `(y, a) = f(x; c)` with real scalar `y`, this returns `((∂y/∂x)(x; c)ᵀ · 1, a)`. In [`HolomorphicLinearity`]
    /// mode, complex `y` yields the complex derivative with respect to `x`. This discards the primal scalar from
    /// [`value_and_gradient`](Self::value_and_gradient); captures remain fixed and auxiliary leaves remain primal
    /// values, so neither receives a gradient result. The closure supports the [`MaybeFallible`] result contract.
    ///
    /// # Parameters
    ///
    ///   - `function`: Binary function returning the differentiated scalar and auxiliary output.
    #[inline]
    pub fn gradient<V, Output, Aux, F>(self, function: F) -> Result<(Input::To<V>, Aux), DifferentiationError>
    where
        V: Value<Type: DifferentiableType>,
        ContextMode: DifferentiationBuilderContext<V, Input>,
        DifferentiationBuilderExecutionContext<ContextMode, V, Input>: ReverseModeDifferentiate + Zero<V>,
        <DifferentiationBuilderExecutionContext<ContextMode, V, Input> as Domain>::Operation:
            From<OneOperation<V::Type>> + From<ZeroLikeOperation<V::Type>>,
        F: FnOnce(
            Input::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
            Capture::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ) -> Output,
        Input: Parameterized<
                V,
                To<V> = Input,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Capture: Parameterized<
                V,
                To<V> = Capture,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                >,
            >,
        Output: MaybeFallible<
                (
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                ),
                DifferentiationError,
            >,
        Aux: Parameterized<
                V,
                To<V> = Aux,
                Family: ParameterizedFamily<
                    LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                    To = Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
                >,
            >,
        (
            LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
            Aux::To<LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>>,
        ): Parameterized<
                LinearizationTracer<DifferentiationBuilderExecutionContext<ContextMode, V, Input>>,
                To<V> = (V, Aux),
                Family: ParameterizedFamily<V>,
            >,
    {
        self.value_and_gradient(function).map(|((_, auxiliary), gradient)| (gradient, auxiliary))
    }
}

/// Starts configuring a differentiation transform that executes in this context.
///
/// This trait is blanket-implemented for every differentiable [`Context`]. Calling `context.differentiate_at(primals)`
/// is equivalent to `differentiate_at(primals).in_context(&context)`.
///
/// ```
/// use std::ops::Mul;
///
/// use ryft_core::{Array, ArrayOperation, Differentiate, EagerContext};
///
/// fn square<A: Clone + Mul<Output = A>>(input: A) -> A {
///     input.clone() * input
/// }
///
/// let context = EagerContext::<Array, ArrayOperation<Array>>::new();
/// let (_, gradient) = context.differentiate_at(Array::scalar(2.0)).value_and_gradient(square)?;
/// assert_eq!(gradient.to_f64s(), vec![4.0]);
/// # Ok::<(), ryft_core::DifferentiationError>(())
/// ```
pub trait Differentiate: Context<Type: DifferentiableType> {
    /// Starts configuring a transform at `primals` with this context selected explicitly.
    #[inline]
    fn differentiate_at<Input>(
        &self,
        primals: Input,
    ) -> DifferentiationBuilder<Input, WithoutCapture, WithoutAuxiliary, RealLinearity, WithContext<Self>> {
        differentiate_at(primals).in_context(self)
    }
}

impl<C: Context<Type: DifferentiableType>> Differentiate for C {}

/// Starts configuring a value-level automatic differentiation transform at `primals`.
///
/// `primals` is the active input tree that the transform differentiates with respect to. Each terminal method takes
/// the transformed closure as its last argument. The closure takes one active argument by default; calling
/// [`DifferentiationBuilder::with_captures`] changes its contract to a binary closure whose second argument is the
/// traced capture structure. Because the active input, captures, and context are already fixed when the terminal
/// receives the closure, the closure's parameter and result types are deduced from the builder state and usually need
/// no annotations.
///
/// ```
/// use std::ops::Mul;
///
/// use ryft_core::{Array, differentiate_at};
///
/// fn scale<A: Mul<Output = A>>(input: A, scale: A) -> A {
///     input * scale
/// }
///
/// let (_, gradient) = differentiate_at(Array::scalar(2.0))
///     .with_captures(Array::scalar(3.0))
///     .value_and_gradient(scale)?;
/// assert_eq!(gradient.to_f64s(), vec![3.0]);
/// # Ok::<(), ryft_core::DifferentiationError>(())
/// ```
///
/// Capture and auxiliary structures are independent trees:
///
/// ```
/// use std::ops::{Add, Mul};
///
/// use ryft_core::{Array, Parameter, differentiate_at};
/// use ryft_macros::Parameterized;
///
/// #[derive(Clone, Parameterized)]
/// #[ryft(crate = "ryft_core")]
/// struct Auxiliary<P: Parameter> {
///     prediction: P,
///     diagnostic: Option<P>,
/// }
///
/// fn evaluate<A: Clone + Parameter + Add<Output = A> + Mul<Output = A>>(
///     input: A,
///     (scale, offsets): (A, Vec<A>),
/// ) -> (A, Auxiliary<A>) {
///     let output = input * scale + offsets[0].clone();
///     (output.clone(), Auxiliary { prediction: output, diagnostic: offsets.into_iter().next() })
/// }
///
/// let ((value, auxiliary), gradient): ((Array, Auxiliary<Array>), Array) = differentiate_at(Array::scalar(2.0))
///     .with_captures((Array::scalar(3.0), vec![Array::scalar(4.0)]))
///     .with_aux()
///     .value_and_gradient(evaluate)?;
/// assert_eq!(value.to_f64s(), vec![10.0]);
/// assert_eq!(auxiliary.prediction.to_f64s(), vec![10.0]);
/// assert_eq!(auxiliary.diagnostic.unwrap().to_f64s(), vec![4.0]);
/// assert_eq!(gradient.to_f64s(), vec![3.0]);
/// # Ok::<(), ryft_core::DifferentiationError>(())
/// ```
///
/// Closure arity follows the builder state and is checked where the terminal receives the closure. A capture-free
/// builder requires a unary closure, so a binary closure literal fails to match the expected signature:
///
/// ```compile_fail
/// use ryft_core::{Array, differentiate_at};
///
/// let _ = differentiate_at(Array::scalar(2.0)).gradient(|input, capture| input * capture);
/// ```
///
/// A builder with captures requires a binary closure, so a unary closure literal fails the same way:
///
/// ```compile_fail
/// use ryft_core::{Array, differentiate_at};
///
/// let _ = differentiate_at(Array::scalar(2.0)).with_captures(Array::scalar(3.0)).gradient(|input| input);
/// ```
///
/// An explicitly selected context cannot be silently replaced, because [`DifferentiationBuilder::in_context`] only
/// exists in the unconfigured [`WithoutContext`] state:
///
/// ```compile_fail
/// use ryft_core::{Array, ArrayOperation, EagerContext, differentiate_at};
///
/// let first = EagerContext::<Array, ArrayOperation<Array>>::new();
/// let second = EagerContext::<Array, ArrayOperation<Array>>::new();
/// let _ = differentiate_at(Array::scalar(2.0)).in_context(&first).in_context(&second);
/// ```
///
/// Holomorphic validation is intentionally unavailable for JVP, linearization, and VJP terminals, so those terminals
/// do not exist in the [`HolomorphicLinearity`] state:
///
/// ```compile_fail
/// use std::ops::Mul;
///
/// use ryft_core::{Array, ProgramError, differentiate_at};
///
/// fn square<A: Clone + Mul<Output = A>>(input: A) -> Result<A, ProgramError> {
///     Ok(input.clone() * input)
/// }
///
/// let _ = differentiate_at(Array::scalar(2.0)).holomorphic().linearize(square);
/// ```
///
/// # Parameters
///
///   - `primals`: Active input tree the transform differentiates with respect to. It may be a leaf, tuple, array,
///     [`Vec`], map, or any type deriving `Parameterized`.
#[inline]
pub fn differentiate_at<Input>(
    primals: Input,
) -> DifferentiationBuilder<Input, WithoutCapture, WithoutAuxiliary, RealLinearity, WithoutContext> {
    DifferentiationBuilder {
        primals,
        captures: WithoutCapture,
        auxiliary: WithoutAuxiliary,
        linearity: RealLinearity,
        context: WithoutContext,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::ops::{Add, Mul};

    use indoc::indoc;
    use num_complex::Complex as ComplexNumber;
    use pretty_assertions::assert_eq;
    use ryft_macros::Parameterized;

    use crate::arrays::{Array, ArrayOperation, ArrayType, DataType};
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::{Context, EagerContext, StagingContext, ValueResolution};
    use crate::operations::complex::{Complex, Real};
    use crate::operations::{One, Reduce, ReductionKind};
    use crate::parameters::Parameter;
    use crate::programs::BindingRegionDriver;
    use crate::tracing::DomainTracingContext;

    use super::*;

    fn square<A: Clone + Mul<Output = A>>(input: A) -> A {
        input.clone() * input
    }

    fn fallible_square<A: Clone + Mul<Output = A>>(input: A) -> Result<A, ProgramError> {
        Ok(square(input))
    }

    #[derive(Clone, Parameterized)]
    struct StructuredCaptures<P: Parameter> {
        scale: P,
        offsets: Vec<P>,
        extra: Option<P>,
    }

    #[derive(Clone, Parameterized)]
    struct StructuredAuxiliary<P: Parameter> {
        prediction: P,
        intermediates: Vec<P>,
        diagnostic: Option<P>,
        training: bool,
    }

    fn structured_capture_function<A: Clone + Parameter + Add<Output = A> + Mul<Output = A>>(
        input: A,
        (captures, tail): (StructuredCaptures<A>, A),
    ) -> (A, StructuredAuxiliary<A>) {
        let scaled = input * captures.scale;
        let offset = captures.offsets[0].clone();
        let output = scaled.clone() + offset.clone() + captures.extra.unwrap() + tail;
        (
            output.clone(),
            StructuredAuxiliary {
                prediction: output,
                intermediates: vec![scaled],
                diagnostic: Some(offset),
                training: true,
            },
        )
    }

    fn scaled_square<A: Clone + Mul<Output = A>>(input: A, scale: A) -> Result<A, ProgramError> {
        Ok(square(input) * scale)
    }

    fn multiply_with_captured_auxiliary<A: Clone + Mul<Output = A>>(
        input: A,
        scale: A,
    ) -> Result<(A, A), ProgramError> {
        let output = input * scale;
        Ok((output.clone(), output))
    }

    fn scaled_square_with_auxiliary<A: Clone + Mul<Output = A>>(input: A, scale: A) -> Result<(A, A), ProgramError> {
        let output = square(input) * scale;
        Ok((output.clone(), output))
    }

    fn capture_only<A>(_: Vec<A>, capture: A) -> A {
        capture
    }

    fn first_scaled<A: Clone + Mul<Output = A>>(inputs: Vec<A>, scale: A) -> Result<A, ProgramError> {
        Ok(inputs[0].clone() * scale)
    }

    #[test]
    fn test_differentiation_error_display() {
        let cases = [
            (DifferentiationError::EmptyInput, "differentiation requires an input with at least one leaf value"),
            (
                DifferentiationError::NonScalarGradientOutput { output_type: "f32[2]".to_string() },
                "gradient output must be a rank-0 scalar but got f32[2]",
            ),
            (
                DifferentiationError::NonDifferentiableGradientOutput { output_type: "i32[]".to_string() },
                "gradient output type i32[] is non-differentiable and carries no cotangent space",
            ),
            (
                DifferentiationError::ComplexGradientOutput { output_type: "c64[]".to_string() },
                "gradient output type c64[] is complex; use a holomorphic gradient entry point if the function is \
                 holomorphic",
            ),
            (
                DifferentiationError::NonDifferentiableParameter {
                    transform: DerivativeTransform::JacobianForward,
                    role: DifferentiationParameterRole::Input,
                    path: "$.value".to_string(),
                    r#type: "i32[]".to_string(),
                },
                "forward Jacobian input parameter at $.value has non-differentiable type i32[]",
            ),
            (
                DifferentiationError::ComplexParameter {
                    transform: DerivativeTransform::JacobianForward,
                    role: DifferentiationParameterRole::Input,
                    path: "$.value".to_string(),
                    r#type: "c64[]".to_string(),
                },
                "forward Jacobian input parameter at $.value has complex type c64[]; use holomorphic forward \
                 Jacobian instead",
            ),
            (
                DifferentiationError::NonComplexParameter {
                    transform: DerivativeTransform::JacobianReverse,
                    role: DifferentiationParameterRole::Output,
                    path: "$".to_string(),
                    r#type: "f64[]".to_string(),
                },
                "holomorphic reverse Jacobian output parameter at $ must be complex but has type f64[]",
            ),
            (
                DifferentiationError::NonFiniteCoordinateSpace {
                    transform: DerivativeTransform::Hessian,
                    role: DifferentiationParameterRole::Derivative,
                    path: "$.0".to_string(),
                    r#type: "f64[dynamic]".to_string(),
                },
                "hessian derivative parameter at $.0 does not have a finite static coordinate space: f64[dynamic]",
            ),
            (
                DifferentiationError::CoordinateCountOverflow {
                    transform: DerivativeTransform::JacobianReverse,
                    role: DifferentiationParameterRole::Input,
                    path: "$.1".to_string(),
                    r#type: "f64[]".to_string(),
                },
                "reverse Jacobian input coordinate count overflows usize at parameter $.1: f64[]",
            ),
            (
                DifferentiationError::Program(ProgramError::MalformedProgram("invalid program".to_string())),
                "encountered malformed program: invalid program",
            ),
        ];
        for (error, expected) in cases {
            assert_eq!(error.to_string(), expected);
        }
    }

    #[test]
    fn test_differentiation_error_conversions() {
        let program_error = ProgramError::MalformedProgram("invalid program".to_string());
        assert_eq!(ProgramError::from(DifferentiationError::from(program_error.clone())), program_error);

        let differentiation_error = DifferentiationError::EmptyInput;
        assert_eq!(
            DifferentiationError::from(ProgramError::from(differentiation_error.clone())),
            differentiation_error
        );
    }

    #[test]
    fn test_derivative_transform() {
        assert_eq!(DerivativeTransform::JacobianForward.to_string(), "forward Jacobian");
        assert_eq!(DerivativeTransform::JacobianReverse.to_string(), "reverse Jacobian");
        assert_eq!(DerivativeTransform::Hessian.to_string(), "hessian");
    }

    #[test]
    fn test_differentiation_parameter_role() {
        assert_eq!(DifferentiationParameterRole::Input.to_string(), "input");
        assert_eq!(DifferentiationParameterRole::Output.to_string(), "output");
        assert_eq!(DifferentiationParameterRole::Derivative.to_string(), "derivative");
    }

    #[test]
    fn test_differentiation_builder_forward_and_reverse_transforms_with_captures() {
        let primal = Array::scalar(2.0);
        let capture = Array::scalar(3.0);

        // JVP differentiates only the active primal while treating the capture as a fixed runtime coefficient.
        let (value, tangent) = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .jvp(Array::scalar(1.0), |input, scale| Ok(input * scale))
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(tangent.to_f64s(), vec![3.0]);

        // Linearization retains that runtime coefficient in the reusable pushforward.
        let (value, pushforward) = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .linearize(|input, scale| Ok(input * scale))
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(pushforward.apply(Array::scalar(2.0)).unwrap().to_f64s(), vec![6.0]);

        // Reverse mode likewise excludes captures from the pullback result.
        let (value, pullback) = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .vjp(|input, scale| Ok(input * scale))
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(pullback.apply(Array::scalar(2.0)).unwrap().to_f64s(), vec![6.0]);

        // Scalar-gradient projection returns no gradient leaf for the capture.
        let (value, gradient) = differentiate_at(primal)
            .with_captures(capture)
            .value_and_gradient(|input, scale| input * scale)
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![3.0]);

        // Tangents must remain structurally isomorphic to the active primals.
        let error = differentiate_at(vec![Array::scalar(2.0)])
            .with_captures(Array::scalar(3.0))
            .jvp(Vec::<Array>::new(), first_scaled)
            .unwrap_err();
        assert_eq!(
            error,
            DifferentiationError::Program(ProgramError::Parameter(ParameterError::MismatchedParameterStructures {
                left_structure: "[<Parameter>]".to_string(),
                right_structure: "[]".to_string(),
            })),
        );
    }

    #[test]
    fn test_differentiation_builder_jacobians_and_hessian_with_captures() {
        let primal = Array::scalar(2.0);
        let capture = Array::scalar(3.0);

        // Both Jacobian directions materialize derivatives only with respect to the active primal.
        let forward = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .jacobian_forward(|input, scale| Ok(input * scale))
            .unwrap();
        let reverse = differentiate_at(primal.clone())
            .with_captures(capture.clone())
            .jacobian_reverse(|input, scale| Ok(input * scale))
            .unwrap();
        assert_eq!(forward.values()[0].to_f64s(), vec![3.0]);
        assert_eq!(reverse.values()[0].to_f64s(), vec![3.0]);

        // Hessian materialization holds the same capture fixed through both derivative levels.
        let hessian = differentiate_at(primal).with_captures(capture).hessian(scaled_square).unwrap();
        assert_eq!(hessian.values()[0].to_f64s(), vec![6.0]);

        // Auxiliary output is reconstructed from primal values and excluded from both Jacobian directions.
        let (jacobian_with_auxiliary, auxiliary) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .with_aux()
            .jacobian_reverse(multiply_with_captured_auxiliary)
            .unwrap();
        assert_eq!(jacobian_with_auxiliary.values()[0].to_f64s(), vec![3.0]);
        assert_eq!(auxiliary.to_f64s(), vec![6.0]);

        let (jacobian_with_auxiliary, auxiliary) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .with_aux()
            .jacobian_forward(multiply_with_captured_auxiliary)
            .unwrap();
        assert_eq!(jacobian_with_auxiliary.values()[0].to_f64s(), vec![3.0]);
        assert_eq!(auxiliary.to_f64s(), vec![6.0]);

        // Hessian auxiliary output follows the same nondifferentiated contract.
        let (hessian_with_auxiliary, auxiliary) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .with_aux()
            .hessian(scaled_square_with_auxiliary)
            .unwrap();
        assert_eq!(hessian_with_auxiliary.values()[0].to_f64s(), vec![6.0]);
        assert_eq!(auxiliary.to_f64s(), vec![12.0]);
    }

    #[test]
    fn test_differentiation_builder_capture_free_terminals() {
        let primal = Array::scalar(2.0);

        // Capture-free scalar gradients retain the ergonomic unary closure shape.
        let (value, gradient) = differentiate_at(primal.clone()).value_and_gradient(square).unwrap();
        assert_eq!(value.to_f64s(), vec![4.0]);
        assert_eq!(gradient.to_f64s(), vec![4.0]);

        // Fallible and infallible unary VJP closures produce equivalent pullbacks.
        let (expected_value, expected_pullback) =
            differentiate_at(primal.clone()).vjp(|input| Ok(input.clone() * input)).unwrap();
        let (actual_value, actual_pullback) = differentiate_at(primal).vjp(fallible_square).unwrap();
        assert_eq!(actual_value, expected_value);
        assert_eq!(
            actual_pullback.apply(Array::scalar(1.0)).unwrap(),
            expected_pullback.apply(Array::scalar(1.0)).unwrap(),
        );

        // Capture-free linearization stages the expected unary pushforward program.
        let (_, actual_pushforward) = differentiate_at(Array::scalar(2.0)).linearize(fallible_square).unwrap();
        assert_eq!(
            actual_pushforward.program().to_string(),
            indoc! {"
                lambda %0:f64[], %1:f64[] .
                let %2:f64[] = mul %1 %0
                    %3:f64[] = mul %1 %0
                    %4:f64[] = add %2 %3
                in (%4)
            "}
            .trim_end(),
        );

        // Higher-order terminals retain the same unary closure shape.
        let hessian = differentiate_at(Array::scalar(2.0)).hessian(fallible_square).unwrap();
        assert_eq!(hessian.values()[0].to_f64s(), vec![2.0]);
    }

    #[test]
    fn test_differentiation_builder_capture_values_do_not_specialize_linearization() {
        let (_, pushforward_at_three) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .linearize(|input, scale| Ok(input * scale))
            .unwrap();
        let (_, pushforward_at_five) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(5.0))
            .linearize(|input, scale| Ok(input * scale))
            .unwrap();

        assert_eq!(pushforward_at_three.program().to_string(), pushforward_at_five.program().to_string());
        assert_eq!(pushforward_at_three.apply(Array::scalar(1.0)).unwrap().to_f64s(), vec![3.0]);
        assert_eq!(pushforward_at_five.apply(Array::scalar(1.0)).unwrap().to_f64s(), vec![5.0]);
    }

    #[test]
    fn test_differentiation_builder_captures_compose_with_batching() {
        let capture_tangent_was_zero = Cell::new(false);
        let per_example = batch(
            |(input, scale)| {
                differentiate_at(input)
                    .with_captures(scale)
                    .value_and_gradient(|input, scale| {
                        capture_tangent_was_zero.set(scale.tangent().is_zero());
                        input * scale
                    })
                    .map_err(ProgramError::from)
            },
            (Array::vector(vec![2.0, 3.0]), Array::vector(vec![4.0, 5.0])),
            (BatchAxis::new(0), BatchAxis::new(0)),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        )
        .unwrap();
        assert_eq!(per_example.0.to_f64s(), vec![8.0, 15.0]);
        assert_eq!(per_example.1.to_f64s(), vec![4.0, 5.0]);
        assert!(capture_tangent_was_zero.get());

        let (value, gradient) = differentiate_at(Array::vector(vec![2.0, 3.0]))
            .with_captures(Array::vector(vec![4.0, 5.0]))
            .value_and_gradient(|input, scale| {
                let mapped = batch(
                    |(item, factor)| Ok(item * factor),
                    (input, scale),
                    (BatchAxis::new(0), BatchAxis::new(0)),
                    BatchAxis::new(0),
                    None,
                )?;
                Ok::<_, ProgramError>(mapped.reduce(&[0], ReductionKind::Sum))
            })
            .unwrap();
        assert_eq!(value.to_f64s(), vec![23.0]);
        assert_eq!(gradient.to_f64s(), vec![4.0, 5.0]);
    }

    #[test]
    fn test_differentiation_builder_explicit_context() {
        #[derive(Clone)]
        struct ExplicitContext(EagerContext<Array, ArrayOperation<Array>>);

        impl Domain for ExplicitContext {
            type Type = ArrayType;
            type Value = Array;
            type Constant = Array;
            type Operation = ArrayOperation<Array>;
        }

        impl Context for ExplicitContext {
            fn lift(&self, constant: Array) -> Result<Array, ProgramError> {
                self.0.lift(constant)
            }

            fn bind<Operation: Into<Self::Operation>, Driver: BindingRegionDriver<Array, Self::Operation>>(
                &self,
                operation: Operation,
                driver: Driver,
                inputs: &[Array],
            ) -> Result<Vec<Array>, ProgramError> {
                self.0.bind(operation, driver, inputs)
            }

            fn is_eager(&self) -> bool {
                self.0.is_eager()
            }

            fn resolve(&self, value: &Array) -> ValueResolution<Array> {
                self.0.resolve(value)
            }
        }

        impl Zero<Array> for ExplicitContext {
            fn zero(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
                self.0.zero(r#type)
            }
        }

        impl One<Array> for ExplicitContext {
            fn one(&self, r#type: &ArrayType) -> Result<Array, ProgramError> {
                self.0.one(r#type)
            }
        }

        // The selected context may differ from `Array::ExecutionDomain`; its type drives all tracer and transform
        // machinery when the builder is context-bound.
        let context = ExplicitContext(EagerContext::new());
        let (_, gradient) = context.differentiate_at(Array::scalar(2.0)).value_and_gradient(square).unwrap();
        assert_eq!(gradient.to_f64s(), vec![4.0]);

        let context = EagerContext::<Array, ArrayOperation<Array>>::new();
        let (_, gradient) = context
            .differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .value_and_gradient(|input, scale| input * scale)
            .unwrap();
        assert_eq!(gradient.to_f64s(), vec![3.0]);

        // Binding a different live tracing context must fail. Deriving the context from the primal would incorrectly
        // make both calls succeed, so these assertions pin the explicit-context semantics of both entry points.
        let primal_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let explicit_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = primal_context.input(ArrayType::scalar(DataType::F64));
        let result = explicit_context.differentiate_at(primal).value_and_gradient(square);
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));

        let primal_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let explicit_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = primal_context.input(ArrayType::scalar(DataType::F64));
        let result = differentiate_at(primal).in_context(&explicit_context).value_and_gradient(square);
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }

    #[test]
    fn test_differentiation_builder_holomorphic_terminals_with_captures() {
        let input = Array::scalar(ComplexNumber::new(2.0f32, 1.0));
        let capture = Array::scalar(ComplexNumber::new(3.0f32, -1.0));
        let (_, gradient) = differentiate_at(input)
            .with_captures(capture.clone())
            .holomorphic()
            .value_and_gradient(|input, capture| input * capture)
            .unwrap();
        assert_eq!(gradient.elements::<ComplexNumber<f32>>(), Ok(vec![ComplexNumber::new(3.0, -1.0)]));

        let jacobian = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 1.0)))
            .with_captures(capture.clone())
            .holomorphic()
            .jacobian_reverse(|input, capture| Ok(input * capture))
            .unwrap();
        assert_eq!(jacobian.values()[0].elements::<ComplexNumber<f32>>(), Ok(vec![ComplexNumber::new(3.0, -1.0)]));

        let forward_jacobian = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 1.0)))
            .with_captures(capture.clone())
            .holomorphic()
            .jacobian_forward(|input, capture| Ok(input * capture))
            .unwrap();
        assert_eq!(
            forward_jacobian.values()[0].elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(3.0, -1.0)]),
        );

        let hessian = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 1.0)))
            .with_captures(capture.clone())
            .holomorphic()
            .hessian(scaled_square)
            .unwrap();
        assert_eq!(hessian.values()[0].elements::<ComplexNumber<f32>>(), Ok(vec![ComplexNumber::new(6.0, -2.0)]));

        // Linearity-mode validation constrains only active inputs and differentiated outputs, so captures are exempt
        // in both directions. A real capture under `holomorphic()` must not raise `NonComplexParameter`, even though
        // the same type in the active input position would.
        let real_capture = Array::scalar(3.0f32);
        let (value, gradient) = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 1.0)))
            .with_captures(real_capture.clone())
            .holomorphic()
            .value_and_gradient(|input, capture| Ok::<_, ProgramError>(input * capture.complex(&capture)?))
            .unwrap();
        assert_eq!(value.elements::<ComplexNumber<f32>>(), Ok(vec![ComplexNumber::new(3.0, 9.0)]));
        assert_eq!(gradient.elements::<ComplexNumber<f32>>(), Ok(vec![ComplexNumber::new(3.0, 3.0)]));

        let real_capture_jacobian = differentiate_at(Array::scalar(ComplexNumber::new(2.0f32, 1.0)))
            .with_captures(real_capture)
            .holomorphic()
            .jacobian_forward(|input, capture| Ok(input * capture.complex(&capture)?))
            .unwrap();
        assert_eq!(
            real_capture_jacobian.values()[0].elements::<ComplexNumber<f32>>(),
            Ok(vec![ComplexNumber::new(3.0, 3.0)]),
        );

        // Symmetrically, a complex capture in ordinary (i.e., real-valued) mode must not raise `ComplexParameter`.
        let (value, gradient) = differentiate_at(Array::scalar(2.0f32))
            .with_captures(capture.clone())
            .value_and_gradient(|input, capture| Ok::<_, ProgramError>(input * capture.real()?))
            .unwrap();
        assert_eq!(value.to_f64s(), vec![6.0]);
        assert_eq!(gradient.to_f64s(), vec![3.0]);

        let complex_capture_jacobian = differentiate_at(Array::scalar(2.0f32))
            .with_captures(capture)
            .jacobian_reverse(|input, capture| Ok(input * capture.real()?))
            .unwrap();
        assert_eq!(complex_capture_jacobian.values()[0].to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_differentiation_builder_empty_active_input_ignores_captures() {
        let error = differentiate_at(Vec::<Array>::new())
            .with_captures(Array::scalar(3.0))
            .value_and_gradient(capture_only)
            .unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_differentiation_builder_validates_captures_when_used() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let foreign_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F64));
        let foreign_capture = foreign_context.input(ArrayType::scalar(DataType::F64));
        let invoked = Cell::new(false);

        // An unused capture never enters the staged program, so its foreign context is irrelevant.
        let result = differentiate_at(primal.clone()).with_captures(foreign_capture.clone()).value_and_gradient(
            |input, _capture| {
                invoked.set(true);
                input
            },
        );
        let (value, gradient) = result.unwrap();
        assert!(invoked.get());
        assert_eq!(value, primal);
        assert!(matches!(context.resolve(&gradient), ValueResolution::Staged(_)));

        // Using that capture binds it into the active program and therefore validates its context identity.
        let result = differentiate_at(primal)
            .with_captures(foreign_capture)
            .value_and_gradient(|input, capture| input * capture);
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));
    }

    #[test]
    fn test_differentiation_builder_hessian_composes_over_staging_context() {
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F64));
        let capture = context.input(ArrayType::scalar(DataType::F64));
        let hessian = differentiate_at(primal).with_captures(capture).hessian(scaled_square).unwrap();

        assert_eq!(hessian.values().len(), 1);
        assert!(matches!(context.resolve(&hessian.values()[0]), ValueResolution::Staged(_)));
    }

    #[test]
    fn test_differentiation_builder_structured_captures_and_auxiliary_output() {
        let captures = StructuredCaptures {
            scale: Array::scalar(3.0),
            offsets: vec![Array::scalar(4.0)],
            extra: Some(Array::scalar(5.0)),
        };
        let ((value, auxiliary), gradient): ((Array, StructuredAuxiliary<Array>), Array) =
            differentiate_at(Array::scalar(2.0))
                .with_captures((captures, Array::scalar(1.0)))
                .with_aux()
                .value_and_gradient(structured_capture_function)
                .unwrap();
        assert_eq!(value.to_f64s(), vec![16.0]);
        assert_eq!(auxiliary.prediction.to_f64s(), vec![16.0]);
        assert_eq!(auxiliary.intermediates[0].to_f64s(), vec![6.0]);
        assert_eq!(auxiliary.diagnostic.unwrap().to_f64s(), vec![4.0]);
        assert!(auxiliary.training);
        assert_eq!(gradient.to_f64s(), vec![3.0]);
    }

    #[test]
    fn test_differentiation_builder_modifier_order_is_orthogonal() {
        let first: (Array, Array) = differentiate_at(Array::scalar(2.0))
            .with_captures(Array::scalar(3.0))
            .with_aux()
            .gradient(multiply_with_captured_auxiliary)
            .unwrap();
        let second: (Array, Array) = differentiate_at(Array::scalar(2.0))
            .with_aux()
            .with_captures(Array::scalar(3.0))
            .gradient(multiply_with_captured_auxiliary)
            .unwrap();

        assert_eq!(first.0.to_f64s(), vec![3.0]);
        assert_eq!(first.1.to_f64s(), vec![6.0]);
        assert_eq!(second.0.to_f64s(), vec![3.0]);
        assert_eq!(second.1.to_f64s(), vec![6.0]);
    }
}
