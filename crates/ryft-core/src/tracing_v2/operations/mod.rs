//! Concrete staged operations and core operation traits for [`crate::tracing_v2`].
//!
//! This module owns the operation universe used by `tracing_v2`. It bundles three layers:
//!
//! - **Core traits** ([`Op`], [`InterpretableOp`], [`LinearOperation`], [`DifferentiableOp`],
//!   [`VectorizableOp`]) Ã¢â‚¬â€ the operation-neutral dispatch interfaces every staged primitive must
//!   satisfy in order to participate in program construction, replay, and the various transforms.
//! - **Per-primitive submodules** ([`add`], [`mul`], [`neg`], Ã¢â‚¬Â¦) Ã¢â‚¬â€ the concrete semantic op types
//!   ([`AddOp`], [`MulOp`], Ã¢â‚¬Â¦) and their associated hidden staging traits
//!   ([`AddTracingOperation`](add::AddTracingOperation), [`MulTracingOperation`](mul::MulTracingOperation),
//!   etc.) used to construct closed staged op carriers.
//! - **Closed default carriers** ([`primitive`], [`custom`]) Ã¢â‚¬â€ [`PrimitiveOp`] / [`LinearPrimitiveOp`]
//!   and the rule-based [`CustomPrimitive`] / [`LinearCustomPrimitive`] escape hatch.
//!
//! # Trait hierarchy
//!
//! ```text
//! Op<T: Type>                      - shape-level, generic over type descriptor T
//! InterpretableOp<T, V>           - concrete execution on values of type V
//! LinearOperation<T, V>           - semantic reverse-mode transpose rule
//! DifferentiableOp<T, V, Tangent> - forward-mode JVP rule, generic over tangent type
//! VectorizableOp<T, V>            - batching rule for vmap
//! ```
//!
//! [`Op`] is generic over the type descriptor `T` so that the same trait can describe abstract
//! evaluation for different type metadata systems. The default `T = ArrayType` means that existing
//! code which writes `Op` without a type parameter continues to work unchanged. Sub-traits like
//! [`InterpretableOp`] are also generic over the type descriptor `T`, so the type descriptor always
//! precedes the value type in all generic parameter lists.
//!
//! # Op selection through `Engine`
//!
//! The public tracing surface ([`jvp`](crate::tracing_v2::jvp), [`vjp`](crate::tracing_v2::vjp),
//! [`interpret_and_trace`](crate::tracing_v2::interpret_and_trace),
//! [`trace`](crate::tracing_v2::trace), and friends) is
//! parameterized by an [`Engine`], and the staged op carriers used inside those
//! transforms are picked by that engine via [`Engine::TracingOperation`] and
//! [`Engine::LinearOperation`]. This is what keeps the op universe open: a backend contributes
//! its own closed carrier (for example, `XlaPrimitiveOp`) by implementing [`Engine`] with those
//! associated types pointing at its backend-specific enum, without editing any central dispatch
//! layer in `tracing_v2`.
//!
//! Do **not** reintroduce a `Supports*` umbrella trait that bundles "all capabilities a transform
//! might need" onto a single bound. Per-op staging is expressed through the small hidden
//! capability traits that live next to each operation (for example, `add::AddTracingOperation`
//! and `mul::MulTracingOperation`), and transform code should bound itself on the concrete
//! engine-selected carrier or on the specific per-op capability traits it actually exercises Ã¢â‚¬â€
//! never on a catch-all faÃƒÂ§ade. The [`TracingOperation`] and [`LinearProgramOp`] bundles defined
//! in this module are additive aliases used only to name the bundle locally; they are not an
//! extension point and should not grow new "is-supported" requirements.

use std::{
    collections::BTreeSet,
    fmt::{Debug, Display},
    sync::Arc,
};

use crate::{
    parameters::{Parameter, Parameterized},
    sharding::Sharding,
    tracing_v2::{
        TraceError, Traceable, batch::Batch, engine::Engine, forward::JvpTracer, jit::Tracer, linear::LinearTerm,
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

pub use add::{AddOp, AddTracingOperation, LinearAddOperation};
pub use cos::{Cos, CosOp, CosTracingOperation};
pub use custom::{
    CustomPrimitive, CustomPrimitiveExtensions, CustomTracingOperation, LinearCustomOperation, LinearCustomPrimitive,
};
pub use left_matmul::{LeftMatMulOp, LeftMatMulTracingOperation, LinearLeftMatMulOperation};
pub use matmul::{MatMulOp, MatMulTracingOperation};
pub use matrix_transpose::{LinearMatrixTransposeOperation, MatrixTransposeOp, MatrixTransposeTracingOperation};
pub use mul::{MulOp, MulTracingOperation};
pub use neg::{LinearNegOperation, NegOp, NegTracingOperation};
pub use primitive::{LinearPrimitiveOp, PrimitiveOp};
pub use rematerialize::{
    FlatTracedRematerialize, LinearRematerializeOp, LinearRematerializeOperation, RematerializeOp,
    RematerializeTracingOperation,
};
pub use reshape::{LinearReshapeOperation, ReshapeOp, ReshapeTracingOperation};
pub use right_matmul::{LinearRightMatMulOperation, RightMatMulOp, RightMatMulTracingOperation};
pub use scale::{LinearScaleOperation, ScaleOp, ScaleTracingOperation};
pub use sin::{Sin, SinOp, SinTracingOperation};
pub use vmap::{FlatTracedVMap, LinearVMapOp, LinearVMapOperation, VMapOp, VMapTracingOperation};

fn is_replicated_sharding(sharding: &Sharding) -> bool {
    sharding
        .dimensions
        .iter()
        .all(|dimension| matches!(dimension, crate::sharding::ShardingDimension::Replicated))
}

fn merge_unique_axes(left: &BTreeSet<String>, right: &BTreeSet<String>) -> BTreeSet<String> {
    left.union(right).cloned().collect()
}

fn merge_sharding_state(base: &Sharding, other: &Sharding) -> Sharding {
    let mut sharding = base.clone();
    sharding.unreduced_axes = merge_unique_axes(&base.unreduced_axes, &other.unreduced_axes);
    sharding.reduced_manual_axes = merge_unique_axes(&base.reduced_manual_axes, &other.reduced_manual_axes);
    sharding.varying_manual_axes = merge_unique_axes(&base.varying_manual_axes, &other.varying_manual_axes);
    sharding
}

fn binary_output_sharding(inputs: &[ArrayType]) -> Option<Sharding> {
    match (&inputs[0].sharding, &inputs[1].sharding) {
        (Some(left), Some(right))
            if left.mesh == right.mesh
                && left.dimensions == right.dimensions
                && left.unreduced_axes == right.unreduced_axes
                && left.reduced_manual_axes == right.reduced_manual_axes =>
        {
            Some(merge_sharding_state(left, right))
        }
        (Some(left), Some(right)) if is_replicated_sharding(left) => Some(merge_sharding_state(right, left)),
        (Some(left), Some(right)) if is_replicated_sharding(right) => Some(merge_sharding_state(left, right)),
        (Some(left), None) => Some(left.clone()),
        (None, Some(right)) => Some(right.clone()),
        _ => None,
    }
}

/// Returns an input-count error when one staged op receives the wrong arity.
pub fn expect_input_count(inputs: usize, expected: usize) -> Result<(), TraceError> {
    if inputs == expected { Ok(()) } else { Err(TraceError::InvalidInputCount { expected, got: inputs }) }
}

/// Returns a batch-size error when two batched inputs disagree on their lane count.
pub fn expect_batch_sizes_match<V>(left: &Batch<V>, right: &Batch<V>) -> Result<(), TraceError> {
    if left.len() == right.len() { Ok(()) } else { Err(TraceError::MismatchedBatchSize) }
}

/// Lifts one concrete value into the staged program owned by a JIT tracer.
pub fn lift_jit_constant<V: Traceable<ArrayType>, O: Clone + 'static, L: Clone + 'static, E>(
    constant: &V,
    exemplar: &Tracer<E>,
) -> Tracer<E>
where
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L> + ?Sized,
{
    let builder = exemplar.builder_handle();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    Tracer::from_staged_parts(atom, builder, exemplar.staging_error_handle(), exemplar.engine())
}

/// Propagates one unary input type through a shape-preserving staged op.
pub fn unary_abstract(inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 1)?;
    Ok(inputs[0].clone())
}

/// Propagates one binary input type through a shape-preserving staged op.
pub fn binary_same_abstract(op: &'static str, inputs: &[ArrayType]) -> Result<ArrayType, TraceError> {
    expect_input_count(inputs.len(), 2)?;
    if inputs[0].data_type != inputs[1].data_type || inputs[0].shape != inputs[1].shape {
        Err(TraceError::IncompatibleAbstractValues { op })
    } else {
        let sharding = binary_output_sharding(inputs);
        ArrayType::new(
            inputs[0].data_type,
            inputs[0].shape.clone(),
            if inputs[0].layout == inputs[1].layout { inputs[0].layout.clone() } else { None },
            sharding,
        )
        .map_err(|_| TraceError::InternalInvariantViolation("binary output sharding should match operand rank"))
    }
}

/// Shape-level operation interface for staged programs.
///
/// This trait covers the metadata surface needed for program construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`InterpretableOp`] trait. Staged-program differentiation rules
/// are split between [`LinearOperation`] (transpose/replay) and [`DifferentiableOp`] (forward-mode JVP).
///
/// The type parameter `T` determines which abstract type descriptor is used for shape-level reasoning. The default
/// is [`ArrayType`], which covers the entire core tracing infrastructure. Future instantiations with different type
/// descriptors can reuse the same trait without modifying existing implementations.
pub trait Op<T: Type = ArrayType>: Debug + Display {
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract output types from abstract input types without executing the operation.
    fn abstract_eval(&self, inputs: &[T]) -> Result<Vec<T>, TraceError>;

    /// Returns simplified output atoms if this operation is a trivial algebraic identity.
    ///
    /// Called during program construction to eliminate no-op operations like `x + 0`, `x * 1`,
    /// or `scale(x, 1)`. The callbacks check whether an input atom is a constant zero or one.
    /// Returns `None` if no simplification applies.
    fn try_simplify(
        &self,
        _inputs: &[usize],
        _is_zero_constant: &dyn Fn(usize) -> bool,
        _is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        None
    }
}

/// Concrete execution capability for staged operations.
///
/// Separated from [`Op`] so that program construction, display, and simplification can work without value-type bounds.
/// Only code paths that actually execute operations (program replay, JIT example propagation) require this trait.
pub trait InterpretableOp<T: Type, V: Typed<T>>: Op<T> {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Semantic contract for staged operations that can live in linear programs.
///
/// A [`LinearOperation`] is not a separate IR container by itself. Instead, it is the capability
/// an operation type must provide in order to participate in tangent and cotangent programs after
/// one primal program has been linearized. In practice, this trait is implemented both by
/// primitive semantic op types like [`AddOp`] and by closed carrier enums such as
/// [`LinearPrimitiveOp`], which delegate the rule to the wrapped semantic primitive.
///
/// For one linear operation `y = L(x)`, the transpose rule builds the reverse linear map `L^T`
/// that pulls cotangents on `y` back to cotangents on `x`. The rule does not receive concrete
/// primal witnesses because those are not part of the transpose trace. Instead, it operates
/// directly on staged output cotangents and emits staged cotangent contributions for the op
/// inputs.
///
/// A few concrete examples:
///
/// - For [`ScaleOp`], `y = a * x`, the transpose stages one new [`LinearTerm`] representing
///   `a * c`, where `c` is the output cotangent:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ryft_core::tracing_v2::{LinearOperation, LinearProgramBuilder, LinearTerm, ScaleOp};
///
///   let builder = Rc::new(RefCell::new(LinearProgramBuilder::<f64>::new()));
///   let cotangent_atom = builder.borrow_mut().add_input(&1.0f64);
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = ScaleOp::new(3.0f64).transpose(&[cotangent]).unwrap();
///   let dx = contributions[0].clone().expect("scale contributes one cotangent");
///   // `dx` is a staged `LinearTerm` representing `3.0 * cotangent`.
///   ```
/// - For [`AddOp`], `y = x0 + x1`, the transpose duplicates the same staged cotangent for both
///   inputs:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ryft_core::tracing_v2::{AddOp, LinearOperation, LinearProgramBuilder, LinearTerm};
///
///   let builder = Rc::new(RefCell::new(LinearProgramBuilder::<f64>::new()));
///   let cotangent_atom = builder.borrow_mut().add_input(&1.0f64);
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = AddOp.transpose(&[cotangent]).unwrap();
///   let dx0 = contributions[0].clone().expect("add contributes to lhs");
///   let dx1 = contributions[1].clone().expect("add contributes to rhs");
///   // `dx0` and `dx1` are staged `LinearTerm`s representing the same cotangent.
///   ```
/// - For [`MatrixTransposeOp`], `Y = X^T`, the transpose stages another transpose on the output
///   cotangent:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ndarray::arr2;
///   use ryft_core::tracing_v2::{LinearOperation, LinearProgramBuilder, LinearTerm, MatrixTransposeOp};
///
///   let builder = Rc::new(RefCell::new(LinearProgramBuilder::<ndarray::Array2<f64>>::new()));
///   let cotangent_atom = builder.borrow_mut().add_input(&arr2(&[[1.0, 2.0], [3.0, 4.0]]));
///   let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder.clone());
///
///   let contributions = MatrixTransposeOp.transpose(&[cotangent]).unwrap();
///   let dx = contributions[0].clone().expect("transpose contributes one cotangent");
///   // `dx` is a staged `LinearTerm` representing `cotangent.transpose()`.
///   ```
/// - For [`ReshapeOp`], the transpose reshapes the output cotangent back to the input shape because
///   reshape only changes layout metadata.
///
/// Structural validation happens when the forward linear program is built and when any staged ops
/// emitted by the rule are added to the transpose program.
pub trait LinearOperation<
    T: Type + Display,
    V: Traceable<T> + Parameter,
    LinearCarrier: Clone = primitive::LinearPrimitiveOp<T, V>,
>: Op<T>
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
    ) -> Result<Vec<Option<LinearTerm<T, V, LinearCarrier>>>, TraceError>;
}

/// Forward-mode differentiation rule, generic over the tangent type `T` and staged carrier types.
///
/// Each operation implements this trait with the exact bounds on `T` that its JVP rule requires.
/// For example, [`AddOp`] only needs `T: TangentSpace<V>`, while [`MatMulOp`] needs
/// `T: TangentSpace<V> + MatrixTangentSpace<V>`.
///
/// [`TangentSpace`]: crate::tracing_v2::forward::TangentSpace
/// [`MatrixTangentSpace`]: crate::tracing_v2::MatrixTangentSpace
pub trait DifferentiableOp<T: Type + Display, V: Traceable<T>, Tangent, O: Clone, L: Clone>: Op<T> {
    /// Applies the forward-mode JVP rule.
    ///
    /// The `engine` argument carries the context needed to synthesize zero values for higher-order
    /// ops that replay staged sub-programs (such as [`RematerializeOp`] and [`VMapOp`]). Pure
    /// arithmetic ops ignore it.
    fn jvp(
        &self,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, Tangent>],
    ) -> Result<Vec<JvpTracer<V, Tangent>>, TraceError>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub trait VectorizableOp<T: Type, V: Typed<T>>: Op<T> {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

/// Capability bundle for the ordinary staged operation type stored in traced programs.
///
/// A [`TracingOperation`] is the operation flavor carried by the ordinary staged program produced by
/// transforms like [`interpret_and_trace`](crate::tracing_v2::interpret_and_trace) and
/// [`trace`](crate::tracing_v2::trace). In practice this is
/// usually one backend-owned closed
/// enum such as [`PrimitiveOp`] or `XlaPrimitiveOp`, but the trait is written as an additive bundle
/// so any type that provides the same capabilities can serve as the carrier.
///
/// The required capabilities are exactly what replaying and transforming an ordinary staged
/// program need:
///
/// - [`Op`] for abstract evaluation and shape-level reasoning,
/// - [`InterpretableOp`] for concrete replay on example values, and
/// - [`DifferentiableOp`] for linearization/JVP construction.
///
/// Any op type that already implements those supertraits automatically implements
/// [`TracingOperation`] via the blanket impl below. The trait exists so that downstream code can
/// talk about "the ordinary staged operation type" in one place instead of repeating the full
/// bundle at every boundary.
///
/// [`VectorizableOp`] is intentionally **not** part of the bundle: `batch()` is only invoked on
/// concrete ops while `vmap` traces through a Rust closure, never on ops stored in an ordinary
/// program, so pinning it here would unnecessarily restrict which op types can satisfy the bundle.
pub trait TracingOperation<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone>:
    Op<T> + InterpretableOp<T, V> + DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L>
{
}

impl<T: Type + Display, V: Traceable<T>, O: Clone, L: Clone, Operation> TracingOperation<T, V, O, L> for Operation where
    Operation: Op<T> + InterpretableOp<T, V> + DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L>
{
}

/// Capability bundle for operations that can appear in a staged linear program.
///
/// Like [`TracingOperation`], this is additive Ã¢â‚¬â€ any op that already satisfies the three supertraits
/// automatically satisfies [`LinearProgramOp`]. The bundle lists what a linear program needs from
/// each stored op: shape metadata ([`Op`]), concrete interpretation for replay
/// ([`InterpretableOp`]), and the reverse-mode transpose rule ([`LinearOperation`]).
pub trait LinearProgramOp<T: Type + Display, V: Traceable<T>>:
    Clone + Op<T> + InterpretableOp<T, V> + LinearOperation<T, V, Self>
{
}

impl<T: Type + Display, V: Traceable<T>, O: Clone> LinearProgramOp<T, V> for O where
    O: Op<T> + InterpretableOp<T, V> + LinearOperation<T, V, O>
{
}

/// Default linear-op carrier capability: eager replay on concrete values.
pub(crate) trait CoreLinearReplayOp<V: Traceable<ArrayType>>:
    Op<ArrayType> + InterpretableOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
}

impl<V: Traceable<ArrayType>, O> CoreLinearReplayOp<V> for O
where
    O: Op<ArrayType> + InterpretableOp<ArrayType, V>,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
}

/// Default linear-op carrier capability: eager replay plus transpose support.
pub(crate) trait CoreLinearProgramOp<V: Traceable<ArrayType>>:
    Clone + CoreLinearReplayOp<V> + LinearOperation<ArrayType, V, Self>
{
}

impl<V: Traceable<ArrayType>, O: Clone> CoreLinearProgramOp<V> for O where
    O: CoreLinearReplayOp<V> + LinearOperation<ArrayType, V, O>
{
}

/// Capability bundle gathering the linear staging traits needed to drive `Tracer` replay.
///
/// This bundle is `'static` because it must satisfy the `'static` requirements imposed by the JIT
/// tracer's storage of staged equations and is bounded over the [`Tracer`] flavor that backs
/// linearized JIT replay rules. Any inner linear operation type that implements
/// [`LinearAddOperation`](add::LinearAddOperation),
/// [`LinearNegOperation`](neg::LinearNegOperation), and
/// [`LinearScaleOperation`](scale::LinearScaleOperation) for the appropriate Tracer leaf
/// automatically satisfies it.
#[doc(hidden)]
pub trait TracerLinearOperation<
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation> + ?Sized
        + 'static = dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = O,
            LinearOperation = OuterLinearOperation,
        >,
>: Clone
    + 'static
    + add::LinearAddOperation<ArrayType, Tracer<E>>
    + neg::LinearNegOperation<ArrayType, Tracer<E>>
    + scale::LinearScaleOperation<ArrayType, Tracer<E>>
{
}

impl<
    V: Traceable<ArrayType>,
    O: Clone + 'static,
    OuterLinearOperation: Clone + 'static,
    E: Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = OuterLinearOperation>
        + ?Sized
        + 'static,
    InnerLinearOperation,
> TracerLinearOperation<V, O, OuterLinearOperation, E> for InnerLinearOperation
where
    InnerLinearOperation: Clone
        + 'static
        + add::LinearAddOperation<ArrayType, Tracer<E>>
        + neg::LinearNegOperation<ArrayType, Tracer<E>>
        + scale::LinearScaleOperation<ArrayType, Tracer<E>>,
{
}

// ---------------------------------------------------------------------------
// Arc forwarding impls
// ---------------------------------------------------------------------------

impl<O: Op<T> + ?Sized, T: Type> Op<T> for Arc<O> {
    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[T]) -> Result<Vec<T>, TraceError> {
        (**self).abstract_eval(inputs)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        (**self).try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<O: InterpretableOp<T, V> + ?Sized, T: Type, V: Traceable<T>> InterpretableOp<T, V> for Arc<O> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
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
    ) -> Result<Vec<Option<LinearTerm<T, V, LinearCarrier>>>, TraceError> {
        (**self).transpose(output_cotangents)
    }
}

impl<InnerOperation, T: Type + Display, V: Traceable<T>, Tangent, O: Clone, L: Clone>
    DifferentiableOp<T, V, Tangent, O, L> for Arc<InnerOperation>
where
    InnerOperation: DifferentiableOp<T, V, Tangent, O, L> + ?Sized,
{
    #[inline]
    fn jvp(
        &self,
        engine: &dyn Engine<Type = T, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, Tangent>],
    ) -> Result<Vec<JvpTracer<V, Tangent>>, TraceError> {
        (**self).jvp(engine, inputs)
    }
}

impl<O: VectorizableOp<T, V> + ?Sized, T: Type, V: Traceable<T>> VectorizableOp<T, V> for Arc<O> {
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        (**self).batch(inputs)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use super::*;
    use crate::sharding::{LogicalMesh, MeshAxis, MeshAxisType, Sharding, ShardingDimension};
    use crate::types::{DataType, Shape, Size};

    fn test_mesh() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    #[test]
    fn test_binary_same_abstract_unions_varying_axes() {
        let mesh = test_mesh();
        let left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["x"],
                )
                .unwrap(),
            ),
        )
        .unwrap();
        let right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh.clone(),
                    vec![ShardingDimension::sharded(["x"])],
                    Vec::<&str>::new(),
                    Vec::<&str>::new(),
                    ["y"],
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            binary_same_abstract("add", &[left, right]).map(|output| output.sharding.unwrap().varying_manual_axes),
            Ok(BTreeSet::from(["x".to_string(), "y".to_string()]))
        );
    }

    #[test]
    fn test_binary_same_abstract_preserves_reduced_axes_from_replicated_input() {
        let mesh = test_mesh();
        let left = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap()),
        )
        .unwrap();
        let right = ArrayType::new(
            DataType::F32,
            Shape::new(vec![Size::Static(8)]),
            None,
            Some(
                Sharding::with_manual_axes(
                    mesh,
                    vec![ShardingDimension::replicated()],
                    Vec::<&str>::new(),
                    ["y"],
                    Vec::<&str>::new(),
                )
                .unwrap(),
            ),
        )
        .unwrap();

        assert_eq!(
            binary_same_abstract("add", &[left, right]).map(|output| output.sharding.unwrap().reduced_manual_axes),
            Ok(BTreeSet::from(["y".to_string()]))
        );
    }
}
