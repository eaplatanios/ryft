//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.
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
//! [`Op`] is generic over the type descriptor `T` so that the same trait can describe abstract evaluation for
//! different type metadata systems. The default `T = ArrayType` means that existing code which writes `Op` without
//! a type parameter continues to work unchanged. Sub-traits like [`InterpretableOp`] are also generic over the
//! type descriptor `T`, so the type descriptor always precedes the value type in all generic parameter lists.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use crate::{
    parameters::{Parameter, Parameterized},
    tracing_v2::{
        Cos, MatrixOps, OneLike, Sin, TraceError, Traceable, ZeroLike,
        batch::Batch,
        engine::Engine,
        forward::JvpTracer,
        jit::JitTracer,
        linear::{LinearTerm, Linearized},
        program::LinearProgramOpRef,
    },
    types::{ArrayType, Type, Typed},
};

/// Shape-level operation interface for staged graphs.
///
/// This trait covers the metadata surface needed for graph construction, display, simplification, and MLIR lowering.
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
    /// Called during graph construction to eliminate no-op operations like `x + 0`, `x * 1`,
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
/// Separated from [`Op`] so that graph construction, display, and simplification can work without value-type bounds.
/// Only code paths that actually execute operations (graph replay, JIT example propagation) require this trait.
pub trait InterpretableOp<T: Type, V: Typed<T>>: Op<T> {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Semantic contract for staged operations that can live in linear programs.
///
/// A [`LinearOperation`] is not a separate IR container by itself. Instead, it is the capability
/// an operation type must provide in order to participate in tangent and cotangent programs after
/// one primal program has been linearized. In practice, this trait is implemented both by
/// primitive semantic op types like [`AddOp`](crate::tracing_v2::operations::AddOp) and by closed
/// carrier enums such as [`LinearPrimitiveOp`], which delegate the rule to the wrapped semantic
/// primitive.
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
    LinearCarrier: Clone = LinearPrimitiveOp<T, V>,
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
/// For example, [`AddOp`](crate::tracing_v2::operations::AddOp) only needs `T: TangentSpace<V>`,
/// while [`MatMulOp`](crate::tracing_v2::operations::MatMulOp) needs
/// `T: TangentSpace<V> + MatrixTangentSpace<V>`.
///
/// [`TangentSpace`]: crate::tracing_v2::forward::TangentSpace
/// [`MatrixTangentSpace`]: crate::tracing_v2::MatrixTangentSpace
pub trait DifferentiableOp<T: Type + Display, V: Traceable<T>, Tangent, O: Clone, L: Clone>: Op<T> {
    /// Applies the forward-mode JVP rule.
    ///
    /// The `engine` argument carries the context needed to synthesize zero values for higher-order
    /// ops that replay staged sub-programs (such as [`RematerializeOp`](crate::tracing_v2::operations::RematerializeOp)
    /// and [`VMapOp`](crate::tracing_v2::operations::VMapOp)). Pure arithmetic ops ignore it.
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
/// A [`TracingOperation`] is the operation flavor carried by the ordinary staged graph produced by
/// transforms like [`try_trace_program`](crate::tracing_v2::try_trace_program) and
/// [`try_jit`](crate::tracing_v2::try_jit). In practice this is usually one backend-owned closed
/// enum such as [`PrimitiveOp`] or `XlaPrimitiveOp`, but the trait is written as an additive
/// bundle so any type that provides the same capabilities can serve as the carrier.
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
/// graph, so pinning it here would unnecessarily restrict which op types can satisfy the bundle.
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
/// Like [`TracingOperation`], this is additive — any op that already satisfies the three supertraits
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

#[doc(hidden)]
pub trait AddTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn add_op() -> Self;
}

#[doc(hidden)]
pub trait MulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn mul_op() -> Self;
}

#[doc(hidden)]
pub trait NegTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn neg_op() -> Self;
}

#[doc(hidden)]
pub trait SinTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn sin_op() -> Self;
}

#[doc(hidden)]
pub trait CosTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn cos_op() -> Self;
}

#[doc(hidden)]
pub trait MatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn matmul_op() -> Self;
}

#[doc(hidden)]
pub trait MatrixTransposeTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn matrix_transpose_op() -> Self;
}

#[doc(hidden)]
pub trait ScaleTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn scale_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait LeftMatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn left_matmul_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait RightMatMulTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn right_matmul_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait ReshapeTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn reshape_op(input_type: T, output_type: T) -> Self;
}

#[doc(hidden)]
pub trait VMapTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone {
    fn vmap_op(op: crate::tracing_v2::operations::VMapOp<T, V, Self, L>) -> Self;
}

#[doc(hidden)]
pub trait RematerializeTracingOperation<T: Type + Display, V: Traceable<T>, L: Clone>: Clone {
    fn rematerialize_op(op: crate::tracing_v2::operations::RematerializeOp<T, V, Self, L>) -> Self;
}

#[doc(hidden)]
pub trait CustomTracingOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self;
}

#[doc(hidden)]
pub trait LinearAddOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_add_op() -> Self;
}

#[doc(hidden)]
pub trait LinearNegOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_neg_op() -> Self;
}

#[doc(hidden)]
pub trait LinearMatrixTransposeOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_matrix_transpose_op() -> Self;
}

#[doc(hidden)]
pub trait LinearScaleOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_scale_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait JitTracerLinearOperation<V: Traceable<ArrayType>, O: Clone + 'static, OuterLinearOperation: Clone + 'static>:
    Clone
    + 'static
    + LinearAddOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>
    + LinearNegOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>
    + LinearScaleOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>
{
}

impl<V: Traceable<ArrayType>, O: Clone + 'static, OuterLinearOperation: Clone + 'static, InnerLinearOperation>
    JitTracerLinearOperation<V, O, OuterLinearOperation> for InnerLinearOperation
where
    InnerLinearOperation: Clone
        + 'static
        + LinearAddOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>
        + LinearNegOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>
        + LinearScaleOperation<ArrayType, JitTracer<ArrayType, V, O, OuterLinearOperation>>,
{
}

#[doc(hidden)]
pub trait LinearLeftMatMulOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_left_matmul_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait LinearRightMatMulOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_right_matmul_op(factor: V) -> Self;
}

#[doc(hidden)]
pub trait LinearReshapeOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_reshape_op(input_type: T, output_type: T) -> Self;
}

#[doc(hidden)]
pub trait LinearVMapOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_vmap_op(op: crate::tracing_v2::operations::LinearVMapOp<T, V, Self>) -> Self;
}

#[doc(hidden)]
pub trait LinearRematerializeOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_rematerialize_op(op: crate::tracing_v2::operations::LinearRematerializeOp<T, V, Self>) -> Self;
}

#[doc(hidden)]
pub trait LinearCustomOperation<T: Type + Display, V: Traceable<T>>: Clone {
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self, TraceError>;

    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TraceError>;
}

pub trait SupportsAdd<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn add_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsAdd<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: AddTracingOperation<T, V>,
{
    fn add_op() -> Self::TracingOperation {
        <E::TracingOperation as AddTracingOperation<T, V>>::add_op()
    }
}

pub trait SupportsMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn mul_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: MulTracingOperation<T, V>,
{
    fn mul_op() -> Self::TracingOperation {
        <E::TracingOperation as MulTracingOperation<T, V>>::mul_op()
    }
}

pub trait SupportsNeg<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn neg_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsNeg<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: NegTracingOperation<T, V>,
{
    fn neg_op() -> Self::TracingOperation {
        <E::TracingOperation as NegTracingOperation<T, V>>::neg_op()
    }
}

pub trait SupportsSin<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn sin_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsSin<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: SinTracingOperation<T, V>,
{
    fn sin_op() -> Self::TracingOperation {
        <E::TracingOperation as SinTracingOperation<T, V>>::sin_op()
    }
}

pub trait SupportsCos<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn cos_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsCos<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: CosTracingOperation<T, V>,
{
    fn cos_op() -> Self::TracingOperation {
        <E::TracingOperation as CosTracingOperation<T, V>>::cos_op()
    }
}

pub trait SupportsMatMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn matmul_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsMatMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: MatMulTracingOperation<T, V>,
{
    fn matmul_op() -> Self::TracingOperation {
        <E::TracingOperation as MatMulTracingOperation<T, V>>::matmul_op()
    }
}

pub trait SupportsMatrixTranspose<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn matrix_transpose_op() -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsMatrixTranspose<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: MatrixTransposeTracingOperation<T, V>,
{
    fn matrix_transpose_op() -> Self::TracingOperation {
        <E::TracingOperation as MatrixTransposeTracingOperation<T, V>>::matrix_transpose_op()
    }
}

pub trait SupportsScale<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn scale_op(factor: V) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsScale<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: ScaleTracingOperation<T, V>,
{
    fn scale_op(factor: V) -> Self::TracingOperation {
        <E::TracingOperation as ScaleTracingOperation<T, V>>::scale_op(factor)
    }
}

pub trait SupportsLeftMatMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn left_matmul_op(factor: V) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLeftMatMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: LeftMatMulTracingOperation<T, V>,
{
    fn left_matmul_op(factor: V) -> Self::TracingOperation {
        <E::TracingOperation as LeftMatMulTracingOperation<T, V>>::left_matmul_op(factor)
    }
}

pub trait SupportsRightMatMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn right_matmul_op(factor: V) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsRightMatMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: RightMatMulTracingOperation<T, V>,
{
    fn right_matmul_op(factor: V) -> Self::TracingOperation {
        <E::TracingOperation as RightMatMulTracingOperation<T, V>>::right_matmul_op(factor)
    }
}

pub trait SupportsReshape<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn reshape_op(input_type: T, output_type: T) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsReshape<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: ReshapeTracingOperation<T, V>,
{
    fn reshape_op(input_type: T, output_type: T) -> Self::TracingOperation {
        <E::TracingOperation as ReshapeTracingOperation<T, V>>::reshape_op(input_type, output_type)
    }
}

pub trait SupportsCoreSyntax<T: Type + Display, V: Traceable<T>>:
    SupportsAdd<T, V>
    + SupportsMul<T, V>
    + SupportsNeg<T, V>
    + SupportsSin<T, V>
    + SupportsCos<T, V>
    + SupportsMatMul<T, V>
    + SupportsMatrixTranspose<T, V>
    + SupportsReshape<T, V>
{
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsCoreSyntax<T, V> for E where
    E: SupportsAdd<T, V>
        + SupportsMul<T, V>
        + SupportsNeg<T, V>
        + SupportsSin<T, V>
        + SupportsCos<T, V>
        + SupportsMatMul<T, V>
        + SupportsMatrixTranspose<T, V>
        + SupportsReshape<T, V>
{
}

pub trait SupportsVMap<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn vmap_op(
        op: crate::tracing_v2::operations::VMapOp<T, V, Self::TracingOperation, Self::LinearOperation>,
    ) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsVMap<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: VMapTracingOperation<T, V, E::LinearOperation>,
{
    fn vmap_op(
        op: crate::tracing_v2::operations::VMapOp<T, V, Self::TracingOperation, Self::LinearOperation>,
    ) -> Self::TracingOperation {
        <E::TracingOperation as VMapTracingOperation<T, V, E::LinearOperation>>::vmap_op(op)
    }
}

pub trait SupportsRematerialize<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn rematerialize_op(
        op: crate::tracing_v2::operations::RematerializeOp<T, V, Self::TracingOperation, Self::LinearOperation>,
    ) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsRematerialize<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: RematerializeTracingOperation<T, V, E::LinearOperation>,
{
    fn rematerialize_op(
        op: crate::tracing_v2::operations::RematerializeOp<T, V, Self::TracingOperation, Self::LinearOperation>,
    ) -> Self::TracingOperation {
        <E::TracingOperation as RematerializeTracingOperation<T, V, E::LinearOperation>>::rematerialize_op(op)
    }
}

pub trait SupportsCustom<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self::TracingOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsCustom<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::TracingOperation: CustomTracingOperation<T, V>,
{
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self::TracingOperation {
        <E::TracingOperation as CustomTracingOperation<T, V>>::custom_op(primitive)
    }
}

pub trait SupportsLinearAdd<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_add_op() -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearAdd<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearAddOperation<T, V>,
{
    fn linear_add_op() -> Self::LinearOperation {
        <E::LinearOperation as LinearAddOperation<T, V>>::linear_add_op()
    }
}

pub trait SupportsLinearNeg<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_neg_op() -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearNeg<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearNegOperation<T, V>,
{
    fn linear_neg_op() -> Self::LinearOperation {
        <E::LinearOperation as LinearNegOperation<T, V>>::linear_neg_op()
    }
}

pub trait SupportsLinearMatrixTranspose<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_matrix_transpose_op() -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearMatrixTranspose<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearMatrixTransposeOperation<T, V>,
{
    fn linear_matrix_transpose_op() -> Self::LinearOperation {
        <E::LinearOperation as LinearMatrixTransposeOperation<T, V>>::linear_matrix_transpose_op()
    }
}

pub trait SupportsLinearScale<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_scale_op(factor: V) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearScale<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearScaleOperation<T, V>,
{
    fn linear_scale_op(factor: V) -> Self::LinearOperation {
        <E::LinearOperation as LinearScaleOperation<T, V>>::linear_scale_op(factor)
    }
}

pub trait SupportsLinearLeftMatMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_left_matmul_op(factor: V) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearLeftMatMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearLeftMatMulOperation<T, V>,
{
    fn linear_left_matmul_op(factor: V) -> Self::LinearOperation {
        <E::LinearOperation as LinearLeftMatMulOperation<T, V>>::linear_left_matmul_op(factor)
    }
}

pub trait SupportsLinearRightMatMul<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_right_matmul_op(factor: V) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearRightMatMul<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearRightMatMulOperation<T, V>,
{
    fn linear_right_matmul_op(factor: V) -> Self::LinearOperation {
        <E::LinearOperation as LinearRightMatMulOperation<T, V>>::linear_right_matmul_op(factor)
    }
}

pub trait SupportsLinearReshape<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_reshape_op(input_type: T, output_type: T) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearReshape<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearReshapeOperation<T, V>,
{
    fn linear_reshape_op(input_type: T, output_type: T) -> Self::LinearOperation {
        <E::LinearOperation as LinearReshapeOperation<T, V>>::linear_reshape_op(input_type, output_type)
    }
}

pub trait SupportsLinearVMap<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_vmap_op(
        op: crate::tracing_v2::operations::LinearVMapOp<T, V, Self::LinearOperation>,
    ) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearVMap<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearVMapOperation<T, V>,
{
    fn linear_vmap_op(
        op: crate::tracing_v2::operations::LinearVMapOp<T, V, Self::LinearOperation>,
    ) -> Self::LinearOperation {
        <E::LinearOperation as LinearVMapOperation<T, V>>::linear_vmap_op(op)
    }
}

pub trait SupportsLinearRematerialize<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_rematerialize_op(
        op: crate::tracing_v2::operations::LinearRematerializeOp<T, V, Self::LinearOperation>,
    ) -> Self::LinearOperation;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearRematerialize<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearRematerializeOperation<T, V>,
{
    fn linear_rematerialize_op(
        op: crate::tracing_v2::operations::LinearRematerializeOp<T, V, Self::LinearOperation>,
    ) -> Self::LinearOperation {
        <E::LinearOperation as LinearRematerializeOperation<T, V>>::linear_rematerialize_op(op)
    }
}

pub trait SupportsLinearCustom<T: Type + Display, V: Traceable<T>>: Engine<Type = T, Value = V> {
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self::LinearOperation, TraceError>;

    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self::LinearOperation, TraceError>;
}

impl<T: Type + Display, V: Traceable<T>, E> SupportsLinearCustom<T, V> for E
where
    E: Engine<Type = T, Value = V>,
    E::LinearOperation: LinearCustomOperation<T, V>,
{
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self::LinearOperation, TraceError> {
        <E::LinearOperation as LinearCustomOperation<T, V>>::linear_custom_op(primitive)
    }

    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self::LinearOperation, TraceError> {
        <E::LinearOperation as LinearCustomOperation<T, V>>::linear_custom_arc_op(primitive)
    }
}

/// Typed extension registry carried by one [`CustomPrimitive`].
#[derive(Clone, Default)]
pub struct CustomPrimitiveExtensions<T: Type, V: Typed<T>> {
    entries: HashMap<TypeId, Arc<dyn Any>>,
    _marker: std::marker::PhantomData<(T, V)>,
}

impl<T: Type, V: Traceable<T>> Debug for CustomPrimitiveExtensions<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CustomPrimitiveExtensions").field("count", &self.entries.len()).finish()
    }
}

impl<T: Type, V: Traceable<T>> CustomPrimitiveExtensions<T, V> {
    /// Inserts one typed extension into the registry, replacing any previous extension of the same type.
    pub fn insert<E: 'static>(&mut self, extension: E) {
        self.entries.insert(TypeId::of::<E>(), Arc::new(extension));
    }

    /// Returns the registered extension of type `E`, if present.
    pub fn get<E: 'static>(&self) -> Option<&E> {
        self.entries.get(&TypeId::of::<E>()).and_then(|extension| extension.as_ref().downcast_ref::<E>())
    }
}

/// Type-erased wrapper for a linearized-JIT replay rule stored inside [`CustomPrimitiveExtensions`].
///
/// This wrapper is `'static` so it can live inside the extension registry. The `V: Traceable<ArrayType>`
/// bound is required at construction time but does not appear on the outer [`CustomPrimitive`] struct.
struct LinearizedJitRule<
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + 'static,
    OuterLinearOperation: Clone + 'static,
    InnerLinearOperation: JitTracerLinearOperation<V, O, OuterLinearOperation>,
>(
    Arc<
        dyn InterpretableOp<
                ArrayType,
                Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>,
            >,
    >,
);

impl<
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + 'static,
    OuterLinearOperation: Clone + 'static,
    InnerLinearOperation: JitTracerLinearOperation<V, O, OuterLinearOperation>,
> LinearizedJitRule<V, O, OuterLinearOperation, InnerLinearOperation>
{
    fn interpret(
        &self,
        inputs: &[Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>],
    ) -> Result<Vec<Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>>, TraceError>
    {
        self.0.interpret(inputs)
    }
}

/// Type-erased wrapper for a staged-carrier-specific JVP rule stored inside [`CustomPrimitiveExtensions`].
struct JvpRule<T: Type + Display, V: Traceable<T> + Parameter, O: Clone, L: Clone>(
    Arc<dyn DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L>>,
);

impl<T: Type + Display, V: Traceable<T> + Parameter, O: Clone, L: Clone> JvpRule<T, V, O, L> {
    fn rule(&self) -> &dyn DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L> {
        self.0.as_ref()
    }
}

trait CustomBaseOp<T: Type, V: Typed<T>>: Op<T> + InterpretableOp<T, V> {}

impl<Ty: Type, V: Traceable<Ty>, O: Op<Ty> + InterpretableOp<Ty, V>> CustomBaseOp<Ty, V> for O {}

/// Rule-based registration object used by [`PrimitiveOp::Custom`].
///
/// The base op always supplies shape metadata and eager interpretation. Optional transform rules are
/// registered using the existing tracing traits directly:
///
/// - [`LinearOperation<ArrayType, V>`] for reverse-mode transpose,
/// - [`DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>>`] for forward-mode JVP,
/// - [`VectorizableOp<ArrayType, V>`] for `vmap`, and
/// - [`InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>>`] for fully general linearized-JIT replay.
#[derive(Clone)]
pub struct CustomPrimitive<T: Type + Display, V: Traceable<T> + Parameter> {
    base: Arc<dyn CustomBaseOp<T, V>>,
    transpose_rule: Option<Arc<dyn LinearOperation<T, V>>>,
    vectorization_rule: Option<Arc<dyn VectorizableOp<T, V>>>,
    extensions: CustomPrimitiveExtensions<T, V>,
}

impl<T: Type + Display + 'static, V: Traceable<T> + Parameter + 'static> CustomPrimitive<T, V> {
    /// Creates one custom primitive from its required base operation.
    pub fn new<Base>(base: Base) -> Self
    where
        Base: Op<T> + InterpretableOp<T, V> + 'static,
    {
        Self {
            base: Arc::new(base),
            transpose_rule: None,
            vectorization_rule: None,
            extensions: CustomPrimitiveExtensions { entries: HashMap::new(), _marker: std::marker::PhantomData },
        }
    }

    /// Registers one transpose rule for reverse-mode differentiation.
    pub fn with_transpose_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: LinearOperation<T, V> + 'static,
    {
        self.transpose_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one staged-carrier-specific forward-mode JVP rule.
    pub fn with_jvp_rule_for<O, L, Rule>(mut self, rule: Rule) -> Self
    where
        O: Clone + 'static,
        L: Clone + 'static,
        Rule: DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L> + 'static,
    {
        self.extensions.insert(JvpRule::<T, V, O, L>(Arc::new(rule)));
        self
    }

    /// Registers one batching rule.
    pub fn with_vectorization_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: VectorizableOp<T, V> + 'static,
    {
        self.vectorization_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one staged-carrier-specific linearized-JIT replay rule for nested custom primitives.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule_for<O, OuterLinearOperation, InnerLinearOperation, Rule>(
        mut self,
        rule: Rule,
    ) -> Self
    where
        O: Clone + 'static,
        OuterLinearOperation: Clone + 'static,
        InnerLinearOperation: JitTracerLinearOperation<V, O, OuterLinearOperation>,
        Rule: InterpretableOp<
                ArrayType,
                Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>,
            > + 'static,
        Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>: Traceable<ArrayType>,
        V: Traceable<ArrayType> + ZeroLike,
    {
        self.extensions
            .insert(LinearizedJitRule::<V, O, OuterLinearOperation, InnerLinearOperation>(Arc::new(rule)));
        self
    }

    /// Registers one typed extension.
    pub fn with_extension<E: 'static>(mut self, extension: E) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Returns the typed extension registry carried by this primitive.
    #[inline]
    pub fn extensions(&self) -> &CustomPrimitiveExtensions<T, V> {
        &self.extensions
    }

    /// Returns one linear-only wrapper for this primitive after verifying that it provides a transpose rule.
    pub fn into_linear(self) -> Result<LinearCustomPrimitive<T, V>, TraceError> {
        LinearCustomPrimitive::from_custom_primitive(Arc::new(self))
    }

    /// Clones this primitive into one linear-only wrapper after verifying that it provides a transpose rule.
    pub fn to_linear(&self) -> Result<LinearCustomPrimitive<T, V>, TraceError> {
        self.clone().into_linear()
    }

    fn missing_rule(&self, transform: &'static str) -> TraceError {
        TraceError::MissingCustomRule { op: self.base.name(), transform }
    }

    fn jvp_rule<O: Clone + 'static, L: Clone + 'static>(
        &self,
    ) -> Result<&dyn DifferentiableOp<T, V, LinearTerm<T, V, L>, O, L>, TraceError> {
        self.extensions
            .get::<JvpRule<T, V, O, L>>()
            .map(JvpRule::rule)
            .ok_or_else(|| self.missing_rule("jvp"))
    }
}

impl<V: Traceable<ArrayType> + Parameter + ZeroLike + 'static> CustomPrimitive<ArrayType, V> {
    /// Registers one forward-mode JVP rule for the canonical core staged carriers.
    pub fn with_jvp_rule<Rule>(self, rule: Rule) -> Self
    where
        Rule: DifferentiableOp<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            > + 'static,
    {
        self.with_jvp_rule_for::<PrimitiveOp<ArrayType, V>, LinearPrimitiveOp<ArrayType, V>, _>(rule)
    }

    /// Registers one linearized-JIT replay rule for nested custom primitives using the canonical
    /// core staged carriers.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule<Rule>(self, rule: Rule) -> Self
    where
        Rule: InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>, LinearProgramOpRef<JitTracer<ArrayType, V>>>>
            + 'static,
        Linearized<JitTracer<ArrayType, V>, LinearProgramOpRef<JitTracer<ArrayType, V>>>: Traceable<ArrayType>,
    {
        self.with_linearized_jit_rule_for::<
            PrimitiveOp<ArrayType, V>,
            LinearPrimitiveOp<ArrayType, V>,
            LinearProgramOpRef<JitTracer<ArrayType, V>>,
            _,
        >(rule)
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<T: Type + Display, V: Traceable<T>> Display for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.base.as_ref(), formatter)
    }
}

impl<V: Traceable<ArrayType>> Op for CustomPrimitive<ArrayType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.base.name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        self.base.abstract_eval(inputs)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        self.base.try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<V: Traceable<ArrayType>> InterpretableOp<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        self.base.interpret(inputs)
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        self.transpose_rule
            .as_deref()
            .ok_or_else(|| self.missing_rule("transpose"))?
            .transpose(output_cotangents)
    }
}

impl<V: Traceable<ArrayType>> VectorizableOp<ArrayType, V> for CustomPrimitive<ArrayType, V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        self.vectorization_rule.as_deref().ok_or_else(|| self.missing_rule("vectorize"))?.batch(inputs)
    }
}

impl<V: Traceable<ArrayType>, O: Clone + 'static, L: Clone + 'static>
    DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V, L>, O, L> for CustomPrimitive<ArrayType, V>
{
    fn jvp(
        &self,
        engine: &dyn Engine<Type = ArrayType, Value = V, TracingOperation = O, LinearOperation = L>,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V, L>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V, L>>>, TraceError> {
        self.jvp_rule::<O, L>()?.jvp(engine, inputs)
    }
}

impl<
    V: Traceable<ArrayType> + ZeroLike,
    O: Clone + 'static,
    OuterLinearOperation: Clone + 'static,
    InnerLinearOperation: JitTracerLinearOperation<V, O, OuterLinearOperation>,
> InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>>
    for CustomPrimitive<ArrayType, V>
where
    Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>: Traceable<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>],
    ) -> Result<Vec<Linearized<JitTracer<ArrayType, V, O, OuterLinearOperation>, InnerLinearOperation>>, TraceError>
    {
        self.extensions
            .get::<LinearizedJitRule<V, O, OuterLinearOperation, InnerLinearOperation>>()
            .ok_or_else(|| self.missing_rule("linearized JIT replay"))?
            .interpret(inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
#[derive(Clone)]
pub struct LinearCustomPrimitive<T: Type + Display, V: Traceable<T> + Parameter> {
    primitive: Arc<CustomPrimitive<T, V>>,
}

impl<T: Type + Display + 'static, V: Traceable<T>> LinearCustomPrimitive<T, V> {
    /// Creates one linear-only wrapper from a custom primitive that already provides a transpose rule.
    pub fn from_custom_primitive(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TraceError> {
        primitive.transpose_rule.as_ref().ok_or_else(|| primitive.missing_rule("transpose"))?;
        Ok(Self { primitive })
    }

    /// Returns the wrapped custom primitive.
    #[inline]
    pub fn primitive(&self) -> &Arc<CustomPrimitive<T, V>> {
        &self.primitive
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: Type + Display, V: Traceable<T>> Display for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<V: Traceable<ArrayType>> Op for LinearCustomPrimitive<ArrayType, V> {
    #[inline]
    fn name(&self) -> &'static str {
        self.primitive.name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        self.primitive.abstract_eval(inputs)
    }

    #[inline]
    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        self.primitive.try_simplify(inputs, is_zero_constant, is_one_constant)
    }
}

impl<V: Traceable<ArrayType>> InterpretableOp<ArrayType, V> for LinearCustomPrimitive<ArrayType, V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        self.primitive.interpret(inputs)
    }
}

impl<V: Traceable<ArrayType>> LinearOperation<ArrayType, V> for LinearCustomPrimitive<ArrayType, V> {
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        self.primitive
            .transpose_rule
            .as_deref()
            .expect("linear custom primitives must carry a transpose rule")
            .transpose(output_cotangents)
    }
}

/// Closed set of built-in staged operations.
///
/// Every known primitive is a zero-cost enum variant. Operations originating outside
/// `ryft-core` (e.g., shard-map ops in `ryft-xla`) go through the [`Custom`](PrimitiveOp::Custom) escape
/// hatch, which still uses dynamic dispatch.
pub enum PrimitiveOp<T: Type + Display, V: Traceable<T> + Parameter> {
    /// Elementwise addition.
    Add,

    /// Elementwise multiplication.
    Mul,

    /// Elementwise negation.
    Neg,

    /// Elementwise sine.
    Sin,

    /// Elementwise cosine.
    Cos,

    /// Matrix multiplication.
    MatMul,

    /// Matrix transposition.
    MatrixTranspose,

    /// Linear matrix transposition used in cotangent programs.
    LinearMatrixTranspose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape between two statically known shapes.
    Reshape { input_type: T, output_type: T },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<T, V, PrimitiveOp<T, V>, LinearPrimitiveOp<T, V>>>),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(
        Box<crate::tracing_v2::operations::RematerializeOp<T, V, PrimitiveOp<T, V>, LinearPrimitiveOp<T, V>>>,
    ),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<T, V>>),
}

impl<T: Type + Display, V: Traceable<T>> Clone for PrimitiveOp<T, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Add => Self::Add,
            Self::Mul => Self::Mul,
            Self::Neg => Self::Neg,
            Self::Sin => Self::Sin,
            Self::Cos => Self::Cos,
            Self::MatMul => Self::MatMul,
            Self::MatrixTranspose => Self::MatrixTranspose,
            Self::LinearMatrixTranspose => Self::LinearMatrixTranspose,
            Self::Scale { factor } => Self::Scale { factor: factor.clone() },
            Self::LeftMatMul { factor } => Self::LeftMatMul { factor: factor.clone() },
            Self::RightMatMul { factor } => Self::RightMatMul { factor: factor.clone() },
            Self::Reshape { input_type, output_type } => {
                Self::Reshape { input_type: input_type.clone(), output_type: output_type.clone() }
            }
            Self::VMap(vmap) => Self::VMap(vmap.clone()),
            Self::Rematerialize(remat) => Self::Rematerialize(remat.clone()),
            Self::Custom(op) => Self::Custom(op.clone()),
        }
    }
}

/// Canonical operation type used by the staged program IR.
pub type PrimitiveOpRef<T, V> = PrimitiveOp<T, V>;

/// Closed set of operations that may appear in staged linear programs.
pub enum LinearPrimitiveOp<T: Type + Display, V: Traceable<T> + Parameter> {
    /// Elementwise addition.
    Add,

    /// Elementwise negation.
    Neg,

    /// Matrix transposition.
    MatrixTranspose,

    /// Linear matrix transposition used in cotangent programs.
    LinearMatrixTranspose,

    /// Scalar or tensor scaling by a captured factor.
    Scale { factor: V },

    /// Left matrix multiplication by a captured factor: `factor @ input`.
    LeftMatMul { factor: V },

    /// Right matrix multiplication by a captured factor: `input @ factor`.
    RightMatMul { factor: V },

    /// Reshape between two statically known shapes.
    Reshape { input_type: T, output_type: T },

    /// Higher-order `vmap` restricted to linear bodies and linear transpose bodies.
    VMap(Box<crate::tracing_v2::operations::LinearVMapOp<T, V, LinearPrimitiveOp<T, V>>>),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(Box<crate::tracing_v2::operations::LinearRematerializeOp<T, V, LinearPrimitiveOp<T, V>>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<T: Type + Display, V: Traceable<T>> Clone for LinearPrimitiveOp<T, V> {
    fn clone(&self) -> Self {
        match self {
            Self::Add => Self::Add,
            Self::Neg => Self::Neg,
            Self::MatrixTranspose => Self::MatrixTranspose,
            Self::LinearMatrixTranspose => Self::LinearMatrixTranspose,
            Self::Scale { factor } => Self::Scale { factor: factor.clone() },
            Self::LeftMatMul { factor } => Self::LeftMatMul { factor: factor.clone() },
            Self::RightMatMul { factor } => Self::RightMatMul { factor: factor.clone() },
            Self::Reshape { input_type, output_type } => {
                Self::Reshape { input_type: input_type.clone(), output_type: output_type.clone() }
            }
            Self::VMap(vmap) => Self::VMap(vmap.clone()),
            Self::Rematerialize(remat) => Self::Rematerialize(remat.clone()),
            Self::Custom(op) => Self::Custom(op.clone()),
        }
    }
}

impl<V: Traceable<ArrayType>> LinearPrimitiveOp<ArrayType, V> {
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<ArrayType, V>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<ArrayType, V>>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

impl<T: Type + Display, V: Traceable<T>> AddTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn add_op() -> Self {
        PrimitiveOp::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> MulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn mul_op() -> Self {
        PrimitiveOp::Mul
    }
}

impl<T: Type + Display, V: Traceable<T>> NegTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn neg_op() -> Self {
        PrimitiveOp::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> SinTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn sin_op() -> Self {
        PrimitiveOp::Sin
    }
}

impl<T: Type + Display, V: Traceable<T>> CosTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn cos_op() -> Self {
        PrimitiveOp::Cos
    }
}

impl<T: Type + Display, V: Traceable<T>> MatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn matmul_op() -> Self {
        PrimitiveOp::MatMul
    }
}

impl<T: Type + Display, V: Traceable<T>> MatrixTransposeTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn matrix_transpose_op() -> Self {
        PrimitiveOp::MatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> ScaleTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn scale_op(factor: V) -> Self {
        PrimitiveOp::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LeftMatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn left_matmul_op(factor: V) -> Self {
        PrimitiveOp::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> RightMatMulTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn right_matmul_op(factor: V) -> Self {
        PrimitiveOp::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> ReshapeTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn reshape_op(input_type: T, output_type: T) -> Self {
        PrimitiveOp::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> VMapTracingOperation<T, V, LinearPrimitiveOp<T, V>> for PrimitiveOp<T, V> {
    #[inline]
    fn vmap_op(op: crate::tracing_v2::operations::VMapOp<T, V, Self, LinearPrimitiveOp<T, V>>) -> Self {
        PrimitiveOp::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> RematerializeTracingOperation<T, V, LinearPrimitiveOp<T, V>>
    for PrimitiveOp<T, V>
{
    #[inline]
    fn rematerialize_op(
        op: crate::tracing_v2::operations::RematerializeOp<T, V, Self, LinearPrimitiveOp<T, V>>,
    ) -> Self {
        PrimitiveOp::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> CustomTracingOperation<T, V> for PrimitiveOp<T, V> {
    #[inline]
    fn custom_op(primitive: Arc<CustomPrimitive<T, V>>) -> Self {
        PrimitiveOp::Custom(primitive)
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearAddOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_add_op() -> Self {
        LinearPrimitiveOp::Add
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearNegOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_neg_op() -> Self {
        LinearPrimitiveOp::Neg
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearMatrixTransposeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_matrix_transpose_op() -> Self {
        LinearPrimitiveOp::LinearMatrixTranspose
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearScaleOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_scale_op(factor: V) -> Self {
        LinearPrimitiveOp::Scale { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearLeftMatMulOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_left_matmul_op(factor: V) -> Self {
        LinearPrimitiveOp::LeftMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRightMatMulOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_right_matmul_op(factor: V) -> Self {
        LinearPrimitiveOp::RightMatMul { factor }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearReshapeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_reshape_op(input_type: T, output_type: T) -> Self {
        LinearPrimitiveOp::Reshape { input_type, output_type }
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearVMapOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_vmap_op(op: crate::tracing_v2::operations::LinearVMapOp<T, V, Self>) -> Self {
        LinearPrimitiveOp::VMap(Box::new(op))
    }
}

impl<T: Type + Display, V: Traceable<T>> LinearRematerializeOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_rematerialize_op(op: crate::tracing_v2::operations::LinearRematerializeOp<T, V, Self>) -> Self {
        LinearPrimitiveOp::Rematerialize(Box::new(op))
    }
}

impl<T: Type + Display + 'static, V: Traceable<T>> LinearCustomOperation<T, V> for LinearPrimitiveOp<T, V> {
    #[inline]
    fn linear_custom_op(primitive: CustomPrimitive<T, V>) -> Result<Self, TraceError> {
        Ok(LinearPrimitiveOp::Custom(Arc::new(primitive.into_linear()?)))
    }

    #[inline]
    fn linear_custom_arc_op(primitive: Arc<CustomPrimitive<T, V>>) -> Result<Self, TraceError> {
        Ok(LinearPrimitiveOp::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
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

// ---------------------------------------------------------------------------
// PrimitiveOp — Debug, Display, Op, InterpretableOp, LinearOperation, DifferentiableOp, VectorizableOp
// ---------------------------------------------------------------------------

use crate::tracing_v2::operations::{
    AddOp, CosOp, LeftMatMulOp, LinearMatrixTransposeOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, ReshapeOp,
    RightMatMulOp, ScaleOp, SinOp, left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
};

impl<T: Type + Display, V: Traceable<T>> Debug for PrimitiveOp<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Mul => write!(formatter, "Mul"),
            Self::Neg => write!(formatter, "Neg"),
            Self::Sin => write!(formatter, "Sin"),
            Self::Cos => write!(formatter, "Cos"),
            Self::MatMul => write!(formatter, "MatMul"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
            Self::LinearMatrixTranspose => write!(formatter, "LinearMatrixTranspose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_type, output_type } => {
                write!(formatter, "Reshape({input_type} -> {output_type})")
            }
            Self::VMap(vmap) => Debug::fmt(vmap, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: Traceable<ArrayType>> Display for PrimitiveOp<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<T: Type + Display, V: Traceable<T>> Debug for LinearPrimitiveOp<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Add => write!(formatter, "Add"),
            Self::Neg => write!(formatter, "Neg"),
            Self::MatrixTranspose => write!(formatter, "MatrixTranspose"),
            Self::LinearMatrixTranspose => write!(formatter, "LinearMatrixTranspose"),
            Self::Scale { .. } => write!(formatter, "Scale"),
            Self::LeftMatMul { .. } => write!(formatter, "LeftMatMul"),
            Self::RightMatMul { .. } => write!(formatter, "RightMatMul"),
            Self::Reshape { input_type, output_type } => {
                write!(formatter, "Reshape({input_type} -> {output_type})")
            }
            Self::VMap(vmap) => Debug::fmt(vmap, formatter),
            Self::Rematerialize(remat) => Debug::fmt(remat, formatter),
            Self::Custom(op) => Debug::fmt(op.as_ref(), formatter),
        }
    }
}

impl<V: Traceable<ArrayType>> Display for LinearPrimitiveOp<ArrayType, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// [`Op`] for [`PrimitiveOp`] requires NO value-type bounds — shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Op for PrimitiveOp<ArrayType, V> {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Mul => "mul",
            Self::Neg => "neg",
            Self::Sin => "sin",
            Self::Cos => "cos",
            Self::MatMul => "matmul",
            Self::MatrixTranspose => "matrix_transpose",
            Self::LinearMatrixTranspose => "linear_matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::VMap(vmap) => vmap.name(),
            Self::Rematerialize(remat) => remat.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        match self {
            Self::Add => AddOp.abstract_eval(inputs),
            Self::Mul => MulOp.abstract_eval(inputs),
            Self::Neg => NegOp.abstract_eval(inputs),
            Self::Sin => SinOp.abstract_eval(inputs),
            Self::Cos => CosOp.abstract_eval(inputs),
            Self::MatMul => MatMulOp.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.abstract_eval(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOp::<ArrayType, V>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::Reshape { input_type, output_type } => {
                <ReshapeOp as Op>::abstract_eval(&ReshapeOp::new(input_type.clone(), output_type.clone()), inputs)
            }
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
            Self::Custom(op) => op.abstract_eval(inputs),
        }
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        match self {
            Self::Add => AddOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Mul => MulOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Neg => NegOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => {
                ScaleOp::<ArrayType, V>::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
            }
            Self::LeftMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::Custom(op) => op.try_simplify(inputs, is_zero_constant, is_one_constant),
            _ => None,
        }
    }
}

/// [`Op`] for [`LinearPrimitiveOp`] requires NO value-type bounds — shape validation works for any `V: Traceable<ArrayType>`.
impl<V: Traceable<ArrayType>> Op for LinearPrimitiveOp<ArrayType, V> {
    fn name(&self) -> &'static str {
        match self {
            Self::Add => "add",
            Self::Neg => "neg",
            Self::MatrixTranspose => "matrix_transpose",
            Self::LinearMatrixTranspose => "linear_matrix_transpose",
            Self::Scale { .. } => "scale",
            Self::LeftMatMul { .. } => "left_matmul",
            Self::RightMatMul { .. } => "right_matmul",
            Self::Reshape { .. } => "reshape",
            Self::VMap(vmap) => vmap.name(),
            Self::Rematerialize(remat) => remat.name(),
            Self::Custom(op) => op.name(),
        }
    }

    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
        match self {
            Self::Add => AddOp.abstract_eval(inputs),
            Self::Neg => NegOp.abstract_eval(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.abstract_eval(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.abstract_eval(inputs),
            Self::Scale { .. } => ScaleOp::<ArrayType, V>::abstract_eval_static(inputs),
            Self::LeftMatMul { factor } => left_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::RightMatMul { factor } => right_matmul_abstract_eval(&Typed::tpe(factor), inputs),
            Self::Reshape { input_type, output_type } => {
                <ReshapeOp as Op>::abstract_eval(&ReshapeOp::new(input_type.clone(), output_type.clone()), inputs)
            }
            Self::VMap(vmap) => vmap.abstract_eval(inputs),
            Self::Rematerialize(remat) => remat.abstract_eval(inputs),
            Self::Custom(op) => op.abstract_eval(inputs),
        }
    }

    fn try_simplify(
        &self,
        inputs: &[usize],
        is_zero_constant: &dyn Fn(usize) -> bool,
        is_one_constant: &dyn Fn(usize) -> bool,
    ) -> Option<Vec<usize>> {
        match self {
            Self::Add => AddOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Neg => NegOp.try_simplify(inputs, is_zero_constant, is_one_constant),
            Self::Scale { factor } => {
                ScaleOp::<ArrayType, V>::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
            }
            Self::LeftMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::RightMatMul { factor } => {
                if crate::tracing_v2::graph::is_identity_one(factor) {
                    Some(inputs.to_vec())
                } else {
                    None
                }
            }
            Self::Custom(op) => op.try_simplify(inputs, is_zero_constant, is_one_constant),
            _ => None,
        }
    }
}

/// [`InterpretableOp`] for [`PrimitiveOp`] requires the full union of value capabilities used by
/// the closed default ordinary-op carrier.
///
/// That broad union is local to [`PrimitiveOp`] itself. The higher-level tracing APIs avoid
/// exposing it as one public value-bundle trait and instead express their requirements through the
/// specific staged op carrier bounds they actually exercise.
impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, V> for PrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, V> for LinearPrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Neg<Output = V>
        + Mul<Output = V>
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> LinearOperation<ArrayType, V> for LinearPrimitiveOp<ArrayType, V>
where
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn transpose(
        &self,
        output_cotangents: &[LinearTerm<ArrayType, V>],
    ) -> Result<Vec<Option<LinearTerm<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => AddOp.transpose(output_cotangents),
            Self::Neg => NegOp.transpose(output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOp.transpose(output_cotangents),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.transpose(output_cotangents),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).transpose(output_cotangents),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).transpose(output_cotangents),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).transpose(output_cotangents),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).transpose(output_cotangents)
            }
            Self::VMap(vmap) => vmap.transpose(output_cotangents),
            Self::Rematerialize(remat) => remat.transpose(output_cotangents),
            Self::Custom(op) => op.transpose(output_cotangents),
        }
    }
}

/// Linearized JIT replay: evaluates staged operations on [`Linearized<JitTracer<V>>`] values.
///
/// For pure (non-capturing) ops, this is covered by their generic [`InterpretableOp<V>`] implementations
/// because [`JvpTracer`] already implements all necessary arithmetic, matrix, and reshape traits.
/// Capturing ops ([`ScaleOp`], [`LeftMatMulOp`], [`RightMatMulOp`]) and higher-order ops
/// ([`VMapOp`](crate::tracing_v2::operations::VMapOp),
/// [`RematerializeOp`](crate::tracing_v2::operations::RematerializeOp)) provide dedicated
/// [`InterpretableOp`] implementations that lift captured constants into the JIT trace.
///
/// [`Linearized<JitTracer<V>>`]: crate::tracing_v2::linear::Linearized
/// [`ScaleOp`]: crate::tracing_v2::operations::ScaleOp
/// [`LeftMatMulOp`]: crate::tracing_v2::operations::LeftMatMulOp
/// [`RightMatMulOp`]: crate::tracing_v2::operations::RightMatMulOp
impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V>>>
    for PrimitiveOp<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => AddOp.interpret(inputs),
            Self::Mul => MulOp.interpret(inputs),
            Self::Neg => NegOp.interpret(inputs),
            Self::Sin => SinOp.interpret(inputs),
            Self::Cos => CosOp.interpret(inputs),
            Self::MatMul => MatMulOp.interpret(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.interpret(inputs),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.interpret(inputs),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).interpret(inputs),
            Self::LeftMatMul { factor } => LeftMatMulOp::new(factor.clone()).interpret(inputs),
            Self::RightMatMul { factor } => RightMatMulOp::new(factor.clone()).interpret(inputs),
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).interpret(inputs)
            }
            Self::VMap(vmap) => vmap.interpret(inputs),
            Self::Rematerialize(remat) => remat.interpret(inputs),
            Self::Custom(op) => op.interpret(inputs),
        }
    }
}

impl<
    V: Traceable<ArrayType>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + Sin
        + Cos
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>, PrimitiveOp<ArrayType, V>, LinearPrimitiveOp<ArrayType, V>>
    for PrimitiveOp<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn jvp(
        &self,
        engine: &dyn Engine<
            Type = ArrayType,
            Value = V,
            TracingOperation = PrimitiveOp<ArrayType, V>,
            LinearOperation = LinearPrimitiveOp<ArrayType, V>,
        >,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&AddOp, engine, inputs),
            Self::Mul => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MulOp, engine, inputs),
            Self::Neg => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&NegOp, engine, inputs),
            Self::Sin => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&SinOp, engine, inputs),
            Self::Cos => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&CosOp, engine, inputs),
            Self::Scale { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&ScaleOp::new(factor.clone()), engine, inputs),
            Self::MatMul => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MatMulOp, engine, inputs),
            Self::MatrixTranspose => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&MatrixTransposeOp, engine, inputs),
            Self::LinearMatrixTranspose => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&LinearMatrixTransposeOp, engine, inputs),
            Self::LeftMatMul { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&LeftMatMulOp::new(factor.clone()), engine, inputs),
            Self::RightMatMul { factor } => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(&RightMatMulOp::new(factor.clone()), engine, inputs),
            Self::Reshape { input_type, output_type } => {
                DifferentiableOp::<
                    ArrayType,
                    V,
                    LinearTerm<ArrayType, V>,
                    PrimitiveOp<ArrayType, V>,
                    LinearPrimitiveOp<ArrayType, V>,
                >::jvp(&ReshapeOp::new(input_type.clone(), output_type.clone()), engine, inputs)
            }
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => DifferentiableOp::<
                ArrayType,
                V,
                LinearTerm<ArrayType, V>,
                PrimitiveOp<ArrayType, V>,
                LinearPrimitiveOp<ArrayType, V>,
            >::jvp(remat.as_ref(), engine, inputs),
            Self::Custom(op) => op.jvp(engine, inputs),
        }
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + Sin + Cos + MatrixOps>
    VectorizableOp<ArrayType, V> for PrimitiveOp<ArrayType, V>
{
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        match self {
            Self::Add => AddOp.batch(inputs),
            Self::Mul => MulOp.batch(inputs),
            Self::Neg => NegOp.batch(inputs),
            Self::Sin => SinOp.batch(inputs),
            Self::Cos => CosOp.batch(inputs),
            Self::MatMul => MatMulOp.batch(inputs),
            Self::MatrixTranspose => MatrixTransposeOp.batch(inputs),
            Self::Custom(op) => op.batch(inputs),
            _ => Err(TraceError::HigherOrderOpFailure {
                op: "vectorize",
                message: format!("vectorization rule for staged op '{}' is not implemented", self.name()),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{cell::RefCell, rc::Rc, sync::Arc};

    use pretty_assertions::assert_eq;

    use crate::tracing_v2::{
        Batch, CompiledFunction, LinearProgramBuilder, ProgramOpRef, TraceError, engine::ArrayScalarEngine, grad, jvp,
        try_jit, vmap,
    };
    use crate::types::{ArrayType, DataType, Shape};

    use super::*;

    /// Simple unary custom op used to exercise the rule-based custom primitive API.
    #[derive(Clone, Debug)]
    struct ShiftOp {
        amount: f64,
    }

    impl ShiftOp {
        /// Creates one shift op with the provided additive amount.
        fn new(amount: f64) -> Self {
            Self { amount }
        }
    }

    impl Display for ShiftOp {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(formatter, "test_shift")
        }
    }

    impl Op for ShiftOp {
        fn name(&self) -> &'static str {
            "test_shift"
        }

        fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0].clone()])
        }
    }

    impl InterpretableOp<ArrayType, f64> for ShiftOp {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0] + self.amount])
        }
    }

    impl LinearOperation<ArrayType, f64> for ShiftOp {
        fn transpose(
            &self,
            output_cotangents: &[LinearTerm<ArrayType, f64>],
        ) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TraceError> {
            if output_cotangents.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl DifferentiableOp<ArrayType, f64, LinearTerm<ArrayType, f64>, ProgramOpRef<f64>, LinearProgramOpRef<f64>>
        for ShiftOp
    {
        fn jvp(
            &self,
            _engine: &dyn Engine<
                Type = ArrayType,
                Value = f64,
                TracingOperation = ProgramOpRef<f64>,
                LinearOperation = LinearProgramOpRef<f64>,
            >,
            inputs: &[JvpTracer<f64, LinearTerm<ArrayType, f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<ArrayType, f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![JvpTracer { primal: inputs[0].primal + self.amount, tangent: inputs[0].tangent.clone() }])
        }
    }

    impl VectorizableOp<ArrayType, f64> for ShiftOp {
        fn batch(&self, inputs: &[Batch<f64>]) -> Result<Vec<Batch<f64>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![Batch::new(inputs[0].lanes().iter().map(|lane| lane + self.amount).collect::<Vec<_>>())])
        }
    }

    impl InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, f64>>> for ShiftOp {
        fn interpret(
            &self,
            inputs: &[Linearized<JitTracer<ArrayType, f64>>],
        ) -> Result<Vec<Linearized<JitTracer<ArrayType, f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            let primal = apply_custom_traced_unary(
                inputs[0].primal.clone(),
                CustomPrimitive::<ArrayType, f64>::new(self.clone()),
            )?;
            Ok(vec![Linearized { primal, tangent: inputs[0].tangent.clone() }])
        }
    }

    /// Applies one unary custom primitive to one traced scalar.
    fn apply_custom_traced_unary(
        input: JitTracer<ArrayType, f64>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Result<JitTracer<ArrayType, f64>, TraceError> {
        let output_values = primitive.interpret(std::slice::from_ref(&input.value))?;
        Ok(JitTracer::apply_staged_op(
            std::slice::from_ref(&input),
            PrimitiveOp::Custom(Arc::new(primitive)),
            output_values,
        )?
        .into_iter()
        .next()
        .expect("unary custom primitive should produce one output"))
    }

    /// Applies one unary custom primitive to one traced scalar and expects staging to succeed.
    fn stage_custom_traced_unary(
        input: JitTracer<ArrayType, f64>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> JitTracer<ArrayType, f64> {
        apply_custom_traced_unary(input, primitive).expect("custom primitive staging should succeed")
    }

    /// Applies one unary custom primitive to one batched scalar.
    fn apply_custom_batched_unary(
        input: Batch<f64>,
        primitive: CustomPrimitive<ArrayType, f64>,
    ) -> Result<Batch<f64>, TraceError> {
        Ok(VectorizableOp::batch(&PrimitiveOp::Custom(Arc::new(primitive)), &[input])?
            .into_iter()
            .next()
            .expect("unary custom primitive should produce one batched output"))
    }

    /// Returns one scalar array type used by these custom-primitive tests.
    fn scalar_type() -> ArrayType {
        ArrayType::new(DataType::F64, Shape::scalar(), None, None).expect("scalar array types should be valid")
    }

    #[test]
    fn test_linear_custom_primitive_requires_transpose_rule_up_front() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert!(matches!(
            primitive.into_linear(),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_base_execution_replays_without_optional_rules() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = try_jit(
            &engine,
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| Ok(stage_custom_traced_unary(x, primitive.clone()))
            },
            3.0f64,
        )
        .unwrap();

        assert_eq!(output, 5.0);
        assert_eq!(compiled.call(4.0f64), Ok(6.0));
    }

    #[test]
    fn test_custom_primitive_missing_transpose_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let builder = Rc::new(RefCell::new(LinearProgramBuilder::<f64>::new()));
        let cotangent_atom = builder.borrow_mut().add_input(&0.0);
        let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder);

        assert!(matches!(
            primitive.transpose(&[cotangent]),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_missing_jvp_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let result: Result<(f64, f64), TraceError> = jvp(
            &engine,
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| stage_custom_traced_unary(x, primitive.clone())
            },
            3.0f64,
            1.0f64,
        );

        assert_eq!(result, Err(TraceError::MissingCustomRule { op: "test_shift", transform: "jvp" }),);
    }

    #[test]
    fn test_custom_primitive_missing_linearized_jit_rule_reports_targeted_error() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_jvp_rule(ShiftOp::new(2.0));
        let result: Result<(f64, CompiledFunction<ArrayType, f64, f64, f64>), TraceError> = try_jit(
            &engine,
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| {
                    let (primal, tangent) = jvp(
                        &engine,
                        {
                            let primitive = primitive.clone();
                            move |inner: JitTracer<ArrayType, f64>| stage_custom_traced_unary(inner, primitive.clone())
                        },
                        x.clone(),
                        x.one_like(),
                    )?;
                    Ok(primal + tangent)
                }
            },
            3.0f64,
        );

        assert!(matches!(
            result,
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "linearized JIT replay" })
        ));
    }

    #[test]
    fn test_custom_primitive_jvp_rule_participates_in_grad_and_linearized_jit_replay() {
        let engine = ArrayScalarEngine::<f64>::new();
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0))
            .with_jvp_rule(ShiftOp::new(2.0))
            .with_linearized_jit_rule(ShiftOp::new(2.0));

        assert_eq!(
            grad(
                &engine,
                {
                    let primitive = primitive.clone();
                    move |x: JitTracer<ArrayType, f64>| stage_custom_traced_unary(x, primitive.clone())
                },
                3.0f64,
            ),
            Ok(1.0f64),
        );

        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = try_jit(
            &engine,
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| {
                    let (primal, tangent) = jvp(
                        &engine,
                        {
                            let primitive = primitive.clone();
                            move |inner: JitTracer<ArrayType, f64>| stage_custom_traced_unary(inner, primitive.clone())
                        },
                        x.clone(),
                        x.one_like(),
                    )?;
                    Ok(primal + tangent)
                }
            },
            3.0f64,
        )
        .unwrap();

        assert_eq!(output, 6.0);
        assert_eq!(compiled.call(4.0f64), Ok(7.0));
    }

    #[test]
    fn test_custom_primitive_batch_rule_reports_targeted_error_when_missing() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert_eq!(
            apply_custom_batched_unary(Batch::new(vec![1.0f64, 2.0]), primitive),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "vectorize" }),
        );
    }

    #[test]
    fn test_custom_primitive_batch_rule_participates_in_vmap() {
        let primitive =
            CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_vectorization_rule(ShiftOp::new(2.0));

        assert_eq!(
            vmap(
                {
                    let primitive = primitive.clone();
                    move |batch: Batch<f64>| apply_custom_batched_unary(batch, primitive.clone()).unwrap()
                },
                vec![1.0f64, 2.0, 3.0],
            ),
            Ok(vec![3.0f64, 4.0, 5.0]),
        );
    }

    #[test]
    fn test_custom_primitive_abstract_eval_uses_the_registered_base_op() {
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));

        assert_eq!(primitive.abstract_eval(&[scalar_type()]), Ok(vec![scalar_type()]));
    }
}
