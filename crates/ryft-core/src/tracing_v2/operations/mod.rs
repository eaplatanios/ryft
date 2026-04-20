//! Concrete staged operations and core operation traits for [`crate::tracing_v2`].
//!
//! This module owns the operation universe used by `tracing_v2`. It bundles three layers:
//!
//! - **Core traits** ([`Operation`], [`InterpretableOperation`], [`LinearOperation`], [`DifferentiableOperation`],
//!   [`VectorizableOperation`]) Ã¢â‚¬â€ the operation-neutral dispatch interfaces every staged primitive must
//!   satisfy in order to participate in program construction, replay, and the various transforms.
//! - **Per-primitive submodules** ([`add`], [`mul`], [`neg`], Ã¢â‚¬Â¦) Ã¢â‚¬â€ the concrete semantic op types
//!   ([`AddOperation`], [`MulOperation`], Ã¢â‚¬Â¦) and their associated hidden staging traits
//!   ([`AddTracingOperation`](add::AddTracingOperation), [`MulTracingOperation`](mul::MulTracingOperation),
//!   etc.) used to construct closed staged op carriers.
//! - **Closed default carriers** ([`primitive`], [`custom`]) Ã¢â‚¬â€ [`PrimitiveOperation`] / [`LinearPrimitiveOperation`]
//!   and the rule-based [`CustomPrimitive`] / [`LinearCustomPrimitive`] escape hatch.
//!
//! # Trait hierarchy
//!
//! ```text
//! Operation<T: Type>                      - shape-level, generic over type descriptor T
//! InterpretableOperation<T, V>           - concrete execution on values of type V
//! LinearOperation<T, V>           - semantic reverse-mode transpose rule
//! DifferentiableOperation<T, V, Tangent> - forward-mode JVP rule, generic over tangent type
//! VectorizableOperation<T, V>            - batching rule for vmap
//! ```
//!
//! [`Operation`] is generic over the type descriptor `T` so that the same trait can describe abstract
//! evaluation for different type metadata systems. The default `T = ArrayType` means that existing
//! code which writes `Operation` without a type parameter continues to work unchanged. Sub-traits like
//! [`InterpretableOperation`] are also generic over the type descriptor `T`, so the type descriptor always
//! precedes the value type in all generic parameter lists.
//!
//! # Operation selection through `Engine`
//!
//! The public tracing surface ([`jvp`](crate::tracing_v2::jvp), [`vjp`](crate::tracing_v2::vjp),
//! [`interpret_and_trace`](crate::tracing_v2::interpret_and_trace),
//! [`trace`](crate::tracing_v2::trace), and friends) is
//! parameterized by an [`Engine`], and the staged op carriers used inside those
//! transforms are picked by that engine via [`Engine::TracingOperation`] and
//! [`Engine::LinearOperation`]. This is what keeps the op universe open: a backend contributes
//! its own closed carrier (for example, `XlaPrimitiveOperation`) by implementing [`Engine`] with those
//! associated types pointing at its backend-specific enum, without editing any central dispatch
//! layer in `tracing_v2`.
//!
//! Do **not** reintroduce a `Supports*` umbrella trait that bundles "all capabilities a transform
//! might need" onto a single bound. Per-op staging is expressed through the small hidden
//! capability traits that live next to each operation (for example, `add::AddTracingOperation`
//! and `mul::MulTracingOperation`), and transform code should bound itself on the concrete
//! engine-selected carrier or on the specific per-op capability traits it actually exercises Ã¢â‚¬â€
//! never on a catch-all faÃƒÂ§ade. The [`TracingOperation`] and [`LinearProgramOperation`] bundles defined
//! in this module are additive aliases used only to name the bundle locally; they are not an
//! extension point and should not grow new "is-supported" requirements.

use std::{
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::{
    parameters::Parameterized,
    tracing_v2::{
        AtomId, Traceable, TracingError, batch::Batch, engine::Engine, forward::JvpTracer, jit::Tracer,
        linear::LinearTerm,
    },
    types::{ArrayType, Type, Typed},
};

/// Elementwise addition.
pub mod add;

/// Elementwise cosine.
pub mod cos;

/// Custom-primitive escape hatch.
pub mod custom;

/// Linear left matrix multiplication.
pub mod left_matmul;

/// Matrix capability layer shared by matrix staged operations.
pub mod matrix;

/// Matrix multiplication.
pub mod matmul;

/// Matrix transposition.
pub mod matrix_transpose;

/// Elementwise multiplication.
pub mod mul;

/// Elementwise negation.
pub mod neg;

/// Closed default carriers for the built-in operation set.
pub mod primitive;

/// Traced rematerialization boundary.
pub mod rematerialize;

/// Reshaping primitive.
pub mod reshape;

/// Linear right matrix multiplication.
pub mod right_matmul;

/// Scalar and tensor scaling.
pub mod scale;

/// Elementwise sine.
pub mod sin;

/// Traced `vmap` operations.
pub mod vmap;

pub use add::{AddOperation, AddTracingOperation, LinearAddOperation};
pub use cos::{Cos, CosOperation, CosTracingOperation};
pub use custom::{
    CustomPrimitive, CustomPrimitiveExtensions, CustomTracingOperation, LinearCustomOperation, LinearCustomPrimitive,
};
pub use left_matmul::{LeftMatMulOperation, LeftMatMulTracingOperation, LinearLeftMatMulOperation};
pub use matmul::{MatMulOperation, MatMulTracingOperation};
pub use matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeOperation, MatrixTransposeTracingOperation};
pub use mul::{MulOperation, MulTracingOperation};
pub use neg::{LinearNegOperation, NegOperation, NegTracingOperation};
pub use primitive::{LinearPrimitiveOperation, PrimitiveOperation};
pub use rematerialize::{
    FlatTracedRematerialize, LinearRematerializeCarrierOperation, LinearRematerializeOperation, RematerializeOperation,
    RematerializeTracingOperation,
};
pub use reshape::{LinearReshapeOperation, ReshapeOperation, ReshapeTracingOperation};
pub use right_matmul::{LinearRightMatMulOperation, RightMatMulOperation, RightMatMulTracingOperation};
pub use scale::{LinearScaleOperation, ScaleOperation, ScaleTracingOperation};
pub use sin::{Sin, SinOperation, SinTracingOperation};
pub use vmap::{FlatTracedVMap, LinearVMapCarrierOperation, LinearVMapOperation, VMapOperation, VMapTracingOperation};

/// Returns an input-count error when one staged op receives the wrong arity.
pub fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TracingError> {
    if inputs == expected { Ok(()) } else { Err(TracingError::InvalidInputCount { expected, got: inputs }) }
}

/// Returns a batch-size error when two batched inputs disagree on their lane count.
pub fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TracingError> {
    if left.len() == right.len() { Ok(()) } else { Err(TracingError::MismatchedBatchSize) }
}

/// Lifts one concrete value into the staged program owned by a JIT tracer.
pub fn lift_jit_constant<
    'engine,
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType>,
    L,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
>(
    constant: &V,
    exemplar: &Tracer<'engine, E>,
) -> Tracer<'engine, E> {
    let builder = exemplar.builder_handle();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    Tracer::from_engine(atom, builder, exemplar.engine())
}

/// Propagates one unary input type through a shape-preserving staged op.
pub fn unary_abstract(inputs: &[ArrayType]) -> Result<ArrayType, TracingError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

/// Shape-level operation interface for staged programs.
///
/// This trait covers the metadata surface needed for program construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`InterpretableOperation`] trait. Staged-program differentiation rules
/// are split between [`LinearOperation`] (transpose/replay) and [`DifferentiableOperation`] (forward-mode JVP).
///
/// The type parameter `T` determines which abstract type descriptor is used for shape-level reasoning. The default
/// is [`ArrayType`], which covers the entire core tracing infrastructure. Future instantiations with different type
/// descriptors can reuse the same trait without modifying existing implementations.
pub trait Operation<T: Type = ArrayType>: Debug + Display {
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract output types from abstract input types without executing the operation.
    fn abstract_eval(&self, inputs: &[T]) -> Result<Vec<T>, TracingError>;

    /// Returns simplified output atoms if this operation is a trivial algebraic identity.
    ///
    /// Called during program construction to eliminate no-op operations like `x + 0`, `x * 1`,
    /// or `scale(x, 1)`. The callbacks check whether an input atom is a constant zero or one.
    /// Returns `None` if no simplification applies.
    fn try_simplify(
        &self,
        _inputs: &[AtomId],
        _is_zero_constant: &dyn Fn(AtomId) -> bool,
        _is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        None
    }
}

/// Concrete execution capability for staged operations.
///
/// Separated from [`Operation`] so that program construction, display, and simplification can work without value-type bounds.
/// Only code paths that actually execute operations (program replay, JIT example propagation) require this trait.
pub trait InterpretableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError>;
}

/// Semantic contract for staged operations that can live in linear programs.
///
/// A [`LinearOperation`] is not a separate IR container by itself. Instead, it is the capability
/// an operation type must provide in order to participate in tangent and cotangent programs after
/// one primal program has been linearized. In practice, this trait is implemented both by
/// primitive semantic op types like [`AddOperation`] and by closed carrier enums such as
/// [`LinearPrimitiveOperation`], which delegate the rule to the wrapped semantic primitive.
///
/// For one linear operation `y = L(x)`, the transpose rule builds the reverse linear map `L^T`
/// that pulls cotangents on `y` back to cotangents on `x`. The rule does not receive concrete
/// primal witnesses because those are not part of the transpose trace. Instead, it operates
/// directly on staged output cotangents and emits staged cotangent contributions for the op
/// inputs.
///
/// A few concrete examples:
///
/// - For [`ScaleOperation`], `y = a * x`, the transpose stages one new [`LinearTerm`] representing
///   `a * c`, where `c` is the output cotangent:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ryft_core::tracing_v2::{LinearOperation, LinearPrimitiveOperation, LinearTerm, ProgramBuilder, ScaleOperation};
///
///   let builder =
///       Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<ArrayType, f64>>::new()));
///   let cotangent_atom = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = ScaleOperation::new(3.0f64).transpose(&[cotangent]).unwrap();
///   let dx = contributions[0].clone().expect("scale contributes one cotangent");
///   // `dx` is a staged `LinearTerm` representing `3.0 * cotangent`.
///   ```
/// - For [`AddOperation`], `y = x0 + x1`, the transpose duplicates the same staged cotangent for both
///   inputs:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ryft_core::tracing_v2::{AddOperation, LinearOperation, LinearPrimitiveOperation, LinearTerm, ProgramBuilder};
///
///   let builder =
///       Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<ArrayType, f64>>::new()));
///   let cotangent_atom = builder.borrow_mut().add_input(1.0f64.r#type().into_owned());
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = AddOperation.transpose(&[cotangent]).unwrap();
///   let dx0 = contributions[0].clone().expect("add contributes to lhs");
///   let dx1 = contributions[1].clone().expect("add contributes to rhs");
///   // `dx0` and `dx1` are staged `LinearTerm`s representing the same cotangent.
///   ```
/// - For [`MatrixTransposeOperation`], `Y = X^T`, the transpose stages another transpose on the output
///   cotangent:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ndarray::arr2;
///   use ryft_core::tracing_v2::{
///       LinearOperation, LinearPrimitiveOperation, LinearTerm, MatrixTransposeOperation, ProgramBuilder,
///   };
///
///   let builder = Rc::new(RefCell::new(
///       ProgramBuilder::<ArrayType, ndarray::Array2<f64>, LinearPrimitiveOperation<ArrayType, ndarray::Array2<f64>>>::new(),
///   ));
///   let cotangent_atom = builder.borrow_mut().add_input(arr2(&[[1.0, 2.0], [3.0, 4.0]]).r#type().into_owned());
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = MatrixTransposeOperation.transpose(&[cotangent]).unwrap();
///   let dx = contributions[0].clone().expect("transpose contributes one cotangent");
///   // `dx` is a staged `LinearTerm` representing `cotangent.transpose()`.
///   ```
/// - For [`ReshapeOperation`], the transpose reshapes the output cotangent back to the input shape because
///   reshape only changes layout metadata.
///
/// Structural validation happens when the forward linear program is built and when any staged ops
/// emitted by the rule are added to the transpose program.
pub trait LinearOperation<
    T: Type + Display,
    V: Traceable<T>,
    LinearCarrier: Clone = primitive::LinearPrimitiveOperation<T, V>,
>: Operation<T>
{
    /// Applies the transpose rule for reverse-mode differentiation.
    ///
    /// `output_cotangents` is aligned with the op outputs in forward order. The returned vector
    /// must be aligned with the op inputs in forward order.
    ///
    /// Returning `Some(term)` means that input receives the staged cotangent contribution `term`.
    /// Returning `None` means the contribution is structurally zero and the transpose pass does not
    /// need to materialize an explicit zero term for that input.
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<T, V, LinearCarrier>],
    ) -> Result<Vec<Option<LinearTerm<T, V, LinearCarrier>>>, TracingError>
    where
        LinearCarrier: Operation<T>;
}

/// Forward-mode differentiation rule, generic over the tangent type `T` and staged carrier types.
///
/// Each operation implements this trait with the exact bounds on `T` that its JVP rule requires.
/// For example, [`AddOperation`] only needs `T: TangentSpace<V>`, while [`MatMulOperation`] needs
/// `T: TangentSpace<V> + MatrixTangentSpace<V>`.
///
/// [`TangentSpace`]: crate::tracing_v2::forward::TangentSpace
/// [`MatrixTangentSpace`]: crate::tracing_v2::MatrixTangentSpace
pub trait DifferentiableOperation<T: Type + Display, V: Traceable<T>, Tangent, O: Clone, L: Clone>:
    Operation<T>
{
    /// Applies the forward-mode JVP rule.
    ///
    /// The `engine` argument carries the context needed to synthesize zero values for higher-order
    /// ops that replay staged sub-programs (such as [`RematerializeOperation`] and [`VMapOperation`]). Pure
    /// arithmetic ops ignore it.
    fn jvp(
        &self,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, Tangent>],
    ) -> Result<Vec<JvpTracer<V, Tangent>>, TracingError>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub trait VectorizableOperation<T: Type, V: Typed<T>>: Operation<T> {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError>;
}

/// Capability bundle for the ordinary staged operation type stored in traced programs.
///
/// A [`TracingOperation`] is the operation flavor carried by the ordinary staged program produced by
/// transforms like [`interpret_and_trace`](crate::tracing_v2::interpret_and_trace) and
/// [`trace`](crate::tracing_v2::trace). In practice this is
/// usually one backend-owned closed
/// enum such as [`PrimitiveOperation`] or `XlaPrimitiveOperation`, but the trait is written as an additive bundle
/// so any type that provides the same capabilities can serve as the carrier.
///
/// The required capabilities are exactly what replaying and transforming an ordinary staged
/// program need:
///
/// - [`Operation`] for abstract evaluation and shape-level reasoning,
/// - [`InterpretableOperation`] for concrete replay on example values, and
/// - [`DifferentiableOperation`] for linearization/JVP construction.
///
/// Any op type that already implements those supertraits automatically implements
/// [`TracingOperation`] via the blanket impl below. The trait exists so that downstream code can
/// talk about "the ordinary staged operation type" in one place instead of repeating the full
/// bundle at every boundary.
///
/// [`VectorizableOperation`] is intentionally **not** part of the bundle: `batch()` is only invoked on
/// concrete ops while `vmap` traces through a Rust closure, never on ops stored in an ordinary
/// program, so pinning it here would unnecessarily restrict which op types can satisfy the bundle.
pub trait TracingOperation<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone + Operation<T>>:
    Operation<T> + InterpretableOperation<T, V> + DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L>
{
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone,
    L: Clone + Operation<T>,
    Carrier: Operation<T> + InterpretableOperation<T, V> + DifferentiableOperation<T, V, LinearTerm<T, V, L>, O, L>,
> TracingOperation<T, V, O, L> for Carrier
{
}

/// Capability bundle for operations that can appear in a staged linear program.
///
/// Like [`TracingOperation`], this is additive Ã¢â‚¬â€ any op that already satisfies the three supertraits
/// automatically satisfies [`LinearProgramOperation`]. The bundle lists what a linear program needs from
/// each stored op: shape metadata ([`Operation`]), concrete interpretation for replay
/// ([`InterpretableOperation`]), and the reverse-mode transpose rule ([`LinearOperation`]).
pub trait LinearProgramOperation<T: Type + Display, V: Traceable<T>>:
    Clone + Operation<T> + InterpretableOperation<T, V> + LinearOperation<T, V, Self>
{
}

impl<
    T: Type + Display,
    V: Traceable<T>,
    O: Clone + Operation<T> + InterpretableOperation<T, V> + LinearOperation<T, V, O>,
> LinearProgramOperation<T, V> for O
{
}

/// Default linear-op carrier capability: eager replay on concrete values.
pub(crate) trait CoreLinearReplayOperation<V: Traceable<ArrayType>>:
    Operation<ArrayType> + InterpretableOperation<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
}

impl<V: Traceable<ArrayType>, O> CoreLinearReplayOperation<V> for O
where
    O: Operation<ArrayType> + InterpretableOperation<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
}

/// Default linear-op carrier capability: eager replay plus transpose support.
pub(crate) trait CoreLinearProgramOperation<V: Traceable<ArrayType>>:
    Clone + CoreLinearReplayOperation<V> + LinearOperation<ArrayType, V, Self>
{
}

impl<V: Traceable<ArrayType>, O: Clone + CoreLinearReplayOperation<V> + LinearOperation<ArrayType, V, O>>
    CoreLinearProgramOperation<V> for O
{
}

/// Capability bundle gathering the linear staging traits needed to drive `Tracer` replay.
///
/// This bundle is `'static` because it must satisfy the `'static` requirements imposed by the JIT
/// tracer's storage of staged instructions and is bounded over the [`Tracer`] flavor that backs
/// linearized JIT replay rules. Any inner linear operation type that implements
/// [`LinearAddOperation`](add::LinearAddOperation),
/// [`LinearNegOperation`](neg::LinearNegOperation), and
/// [`LinearScaleOperation`](scale::LinearScaleOperation) for the appropriate Tracer leaf
/// automatically satisfies it.
#[doc(hidden)]
pub trait TracerLinearOperation<
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType> + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation> + ?Sized
        + 'static = dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = O,
            LinearOperation = OuterLinearOperation,
        >,
>: Clone + Operation<ArrayType> + 'static
where
    for<'engine> Self: add::LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + neg::LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + scale::LinearScaleOperation<ArrayType, Tracer<'engine, E>>
{
}

impl<
    V: Traceable<ArrayType>,
    O: Clone + Operation<ArrayType> + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
    InnerLinearOperation: Clone + Operation<ArrayType> + 'static,
> TracerLinearOperation<V, O, OuterLinearOperation, E> for InnerLinearOperation
where
    for<'engine> InnerLinearOperation: add::LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + neg::LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + scale::LinearScaleOperation<ArrayType, Tracer<'engine, E>>,
{
}

// ---------------------------------------------------------------------------
// Arc forwarding impls
// ---------------------------------------------------------------------------

impl<O: Operation<T> + ?Sized, T: Type> Operation<T> for Arc<O> {
    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[T]) -> Result<Vec<T>, TracingError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[AtomId],
        is_zero_constant: &dyn Fn(AtomId) -> bool,
        is_one_constant: &dyn Fn(AtomId) -> bool,
    ) -> Option<Vec<AtomId>> {
        (**self).try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<O: InterpretableOperation<T, V> + ?Sized, T: Type, V: Traceable<T>> InterpretableOperation<T, V> for Arc<O> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        (**self).interpret(inputs)
    }
}

impl<O: LinearOperation<T, V, LinearCarrier> + ?Sized, T: Type + Display, V: Traceable<T>, LinearCarrier: Clone>
    LinearOperation<T, V, LinearCarrier> for Arc<O>
{
    #[inline]
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<T, V, LinearCarrier>],
    ) -> Result<Vec<Option<LinearTerm<T, V, LinearCarrier>>>, TracingError>
    where
        LinearCarrier: Operation<T>,
    {
        (**self).transpose(output_cotangents)
    }
}

impl<
    InnerOperation: DifferentiableOperation<T, V, Tangent, O, L> + ?Sized,
    T: Type + Display,
    V: Traceable<T>,
    Tangent,
    O: Clone,
    L: Clone,
> DifferentiableOperation<T, V, Tangent, O, L> for Arc<InnerOperation>
{
    #[inline]
    fn jvp(
        &self,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, Tangent>],
    ) -> Result<Vec<JvpTracer<V, Tangent>>, TracingError> {
        (**self).jvp(engine, inputs)
    }
}

impl<O: VectorizableOperation<T, V> + ?Sized, T: Type, V: Traceable<T>> VectorizableOperation<T, V> for Arc<O> {
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TracingError> {
        (**self).batch(inputs)
    }
}
