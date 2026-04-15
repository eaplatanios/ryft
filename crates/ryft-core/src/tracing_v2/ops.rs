//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.
//!
//! # Trait hierarchy
//!
//! ```text
//! Op (shape-level, NOT generic over V)
//! ├─ InterpretableOp<V>        — concrete execution on values of type V
//! ├─ LinearOp<V>               — semantic reverse-mode transpose rule
//! ├─ DifferentiableOp<V, T>    — forward-mode JVP rule, generic over tangent type T
//! └─ VectorizableOp<V>         — batching rule for vmap
//! ```
//!
//! [`Op`] is intentionally non-generic: `abstract_eval` and `name` operate on [`ArrayType`] metadata only.
//! This means `Graph<O: Op, V, ...>` can be constructed, displayed, and simplified for *any* value type
//! without requiring operation-specific value bounds.

use std::{
    any::{Any, TypeId},
    collections::HashMap,
    fmt::{Debug, Display},
    ops::{Add, Mul, Neg},
    sync::Arc,
};

use crate::tracing_v2::{
    FloatExt, MatrixOps, OneLike, TraceError, TraceValue, ZeroLike,
    batch::Batch,
    forward::JvpTracer,
    jit::JitTracer,
    linear::{LinearTerm, Linearized},
};
use crate::types::{ArrayType, Typed};

/// Shape-level operation interface for staged graphs.
///
/// This trait covers the metadata surface needed for graph construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`InterpretableOp`] trait. Staged-program differentiation rules are split
/// between [`LinearOp`] (transpose/replay) and [`DifferentiableOp`] (forward-mode JVP).
pub trait Op: Debug + Display {
    /// Returns the stable primitive name used in diagnostics and pretty-printing.
    fn name(&self) -> &'static str;

    /// Computes abstract output types from abstract input types without executing the operation.
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError>;

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
pub trait InterpretableOp<V: TraceValue>: Op {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Operations that can appear in tangent/cotangent programs and support reverse-mode transposition.
///
/// The `inputs` and `outputs` are the representative concrete values recorded while staging the
/// forward program. The `output_cotangents` are staged cotangents in the transpose program and can
/// be transformed with existing [`LinearTerm`] helpers such as [`LinearTerm::apply_staged_op`].
pub trait LinearOp<V: TraceValue>: Op {
    /// Applies the transpose rule for reverse-mode differentiation.
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError>;
}

/// Forward-mode differentiation rule, generic over the tangent type `T`.
///
/// Each operation implements this trait with the exact bounds on `T` that its JVP rule requires.
/// For example, [`AddOp`](crate::tracing_v2::operations::AddOp) only needs `T: TangentSpace<V>`,
/// while [`MatMulOp`](crate::tracing_v2::operations::MatMulOp) needs
/// `T: TangentSpace<V> + MatrixTangentSpace<V>`.
///
/// [`TangentSpace`]: crate::tracing_v2::forward::TangentSpace
/// [`MatrixTangentSpace`]: crate::tracing_v2::MatrixTangentSpace
pub trait DifferentiableOp<V: TraceValue, T>: Op {
    /// Applies the forward-mode JVP rule.
    fn jvp(&self, inputs: &[JvpTracer<V, T>]) -> Result<Vec<JvpTracer<V, T>>, TraceError>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub trait VectorizableOp<V: TraceValue>: Op {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

/// Typed extension registry carried by one [`CustomPrimitive`].
#[derive(Clone, Default)]
pub struct CustomPrimitiveExtensions {
    entries: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
}

impl Debug for CustomPrimitiveExtensions {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CustomPrimitiveExtensions").field("count", &self.entries.len()).finish()
    }
}

impl CustomPrimitiveExtensions {
    /// Inserts one typed extension into the registry, replacing any previous extension of the same type.
    pub fn insert<T: Send + Sync + 'static>(&mut self, extension: T) {
        self.entries.insert(TypeId::of::<T>(), Arc::new(extension));
    }

    /// Returns the registered extension of type `T`, if present.
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.entries.get(&TypeId::of::<T>()).and_then(|extension| extension.as_ref().downcast_ref::<T>())
    }
}

trait CustomBaseOp<V: TraceValue>: Op + InterpretableOp<V> + Send + Sync {}

impl<V: TraceValue, T: Op + InterpretableOp<V> + Send + Sync> CustomBaseOp<V> for T {}

/// Rule-based registration object used by [`PrimitiveOp::Custom`].
///
/// The base op always supplies shape metadata and eager interpretation. Optional transform rules are
/// registered using the existing tracing traits directly:
///
/// - [`LinearOp<V>`] for reverse-mode transpose,
/// - [`DifferentiableOp<V, LinearTerm<V>>`] for forward-mode JVP,
/// - [`VectorizableOp<V>`] for `vmap`, and
/// - [`InterpretableOp<Linearized<JitTracer<V>>>`] for fully general linearized-JIT replay.
#[derive(Clone)]
pub struct CustomPrimitive<V: TraceValue> {
    base: Arc<dyn CustomBaseOp<V>>,
    transpose_rule: Option<Arc<dyn LinearOp<V> + Send + Sync>>,
    jvp_rule: Option<Arc<dyn DifferentiableOp<V, LinearTerm<V>> + Send + Sync>>,
    vectorization_rule: Option<Arc<dyn VectorizableOp<V> + Send + Sync>>,
    linearized_jit_rule: Option<Arc<dyn InterpretableOp<Linearized<JitTracer<V>>> + Send + Sync>>,
    extensions: CustomPrimitiveExtensions,
}

impl<V: TraceValue> CustomPrimitive<V> {
    /// Creates one custom primitive from its required base operation.
    pub fn new<Base>(base: Base) -> Self
    where
        Base: Op + InterpretableOp<V> + Send + Sync + 'static,
    {
        Self {
            base: Arc::new(base),
            transpose_rule: None,
            jvp_rule: None,
            vectorization_rule: None,
            linearized_jit_rule: None,
            extensions: CustomPrimitiveExtensions::default(),
        }
    }

    /// Registers one transpose rule for reverse-mode differentiation.
    pub fn with_transpose_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: LinearOp<V> + Send + Sync + 'static,
    {
        self.transpose_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one forward-mode JVP rule.
    pub fn with_jvp_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: DifferentiableOp<V, LinearTerm<V>> + Send + Sync + 'static,
    {
        self.jvp_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one batching rule.
    pub fn with_vectorization_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: VectorizableOp<V> + Send + Sync + 'static,
    {
        self.vectorization_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one linearized-JIT replay rule for nested custom primitives.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: InterpretableOp<Linearized<JitTracer<V>>> + Send + Sync + 'static,
        Linearized<JitTracer<V>>: TraceValue,
    {
        self.linearized_jit_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one typed extension.
    pub fn with_extension<T: Send + Sync + 'static>(mut self, extension: T) -> Self {
        self.extensions.insert(extension);
        self
    }

    /// Returns the typed extension registry carried by this primitive.
    #[inline]
    pub fn extensions(&self) -> &CustomPrimitiveExtensions {
        &self.extensions
    }

    /// Returns one linear-only wrapper for this primitive after verifying that it provides a transpose rule.
    pub fn into_linear(self) -> Result<LinearCustomPrimitive<V>, TraceError> {
        LinearCustomPrimitive::from_custom_primitive(Arc::new(self))
    }

    /// Clones this primitive into one linear-only wrapper after verifying that it provides a transpose rule.
    pub fn to_linear(&self) -> Result<LinearCustomPrimitive<V>, TraceError> {
        self.clone().into_linear()
    }

    fn missing_rule(&self, transform: &'static str) -> TraceError {
        TraceError::MissingCustomRule { op: self.name(), transform }
    }

    fn jvp_rule(&self) -> Result<&(dyn DifferentiableOp<V, LinearTerm<V>> + Send + Sync), TraceError> {
        self.jvp_rule.as_deref().ok_or_else(|| self.missing_rule("jvp"))
    }
}

impl<V: TraceValue> Debug for CustomPrimitive<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<V: TraceValue> Display for CustomPrimitive<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.base.as_ref(), formatter)
    }
}

impl<V: TraceValue> Op for CustomPrimitive<V> {
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

impl<V: TraceValue> InterpretableOp<V> for CustomPrimitive<V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        self.base.interpret(inputs)
    }
}

impl<V: TraceValue> LinearOp<V> for CustomPrimitive<V> {
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        self.transpose_rule.as_deref().ok_or_else(|| self.missing_rule("transpose"))?.transpose(
            inputs,
            outputs,
            output_cotangents,
        )
    }
}

impl<V: TraceValue> VectorizableOp<V> for CustomPrimitive<V> {
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        self.vectorization_rule.as_deref().ok_or_else(|| self.missing_rule("vectorize"))?.batch(inputs)
    }
}

impl<V: TraceValue> DifferentiableOp<V, LinearTerm<V>> for CustomPrimitive<V> {
    fn jvp(&self, inputs: &[JvpTracer<V, LinearTerm<V>>]) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        self.jvp_rule()?.jvp(inputs)
    }
}

impl<V: TraceValue> InterpretableOp<Linearized<JitTracer<V>>> for CustomPrimitive<V>
where
    Linearized<JitTracer<V>>: TraceValue,
{
    fn interpret(&self, inputs: &[Linearized<JitTracer<V>>]) -> Result<Vec<Linearized<JitTracer<V>>>, TraceError> {
        self.linearized_jit_rule
            .as_deref()
            .ok_or_else(|| self.missing_rule("linearized JIT replay"))?
            .interpret(inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
#[derive(Clone)]
pub struct LinearCustomPrimitive<V: TraceValue> {
    primitive: Arc<CustomPrimitive<V>>,
}

impl<V: TraceValue> LinearCustomPrimitive<V> {
    /// Creates one linear-only wrapper from a custom primitive that already provides a transpose rule.
    pub fn from_custom_primitive(primitive: Arc<CustomPrimitive<V>>) -> Result<Self, TraceError> {
        primitive.transpose_rule.as_ref().ok_or_else(|| primitive.missing_rule("transpose"))?;
        Ok(Self { primitive })
    }

    /// Returns the wrapped custom primitive.
    #[inline]
    pub fn primitive(&self) -> &Arc<CustomPrimitive<V>> {
        &self.primitive
    }
}

impl<V: TraceValue> Debug for LinearCustomPrimitive<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<V: TraceValue> Display for LinearCustomPrimitive<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<V: TraceValue> Op for LinearCustomPrimitive<V> {
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

impl<V: TraceValue> InterpretableOp<V> for LinearCustomPrimitive<V> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        self.primitive.interpret(inputs)
    }
}

impl<V: TraceValue> LinearOp<V> for LinearCustomPrimitive<V> {
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        self.primitive
            .transpose_rule
            .as_deref()
            .expect("linear custom primitives must carry a transpose rule")
            .transpose(inputs, outputs, output_cotangents)
    }
}

/// Closed set of built-in staged operations.
///
/// Every known primitive is a zero-cost enum variant. Operations originating outside
/// `ryft-core` (e.g., shard-map ops in `ryft-xla`) go through the [`Custom`](PrimitiveOp::Custom) escape
/// hatch, which still uses dynamic dispatch.
#[derive(Clone)]
pub enum PrimitiveOp<V: TraceValue> {
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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order `vmap` carrying a compiled per-lane body and optional transpose body.
    VMap(Box<crate::tracing_v2::operations::VMapOp<V>>),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(Box<crate::tracing_v2::operations::RematerializeOp<V>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<V>>),
}

/// Canonical operation type used by the staged program IR.
pub type PrimitiveOpRef<V> = PrimitiveOp<V>;

/// Closed set of operations that may appear in staged linear programs.
#[derive(Clone)]
pub enum LinearPrimitiveOp<V: TraceValue> {
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
    Reshape { input_type: ArrayType, output_type: ArrayType },

    /// Higher-order `vmap` restricted to linear bodies and linear transpose bodies.
    VMap(Box<crate::tracing_v2::operations::LinearVMapOp<V>>),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(Box<crate::tracing_v2::operations::LinearRematerializeOp<V>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<V>>),
}

impl<V: TraceValue> LinearPrimitiveOp<V> {
    /// Wraps one custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom(primitive: CustomPrimitive<V>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(primitive.into_linear()?)))
    }

    /// Wraps one shared custom primitive in the linear-only operation universe after verifying transpose support.
    pub fn custom_arc(primitive: Arc<CustomPrimitive<V>>) -> Result<Self, TraceError> {
        Ok(Self::Custom(Arc::new(LinearCustomPrimitive::from_custom_primitive(primitive)?)))
    }
}

// ---------------------------------------------------------------------------
// Arc forwarding impls
// ---------------------------------------------------------------------------

impl<T: Op + ?Sized> Op for Arc<T> {
    #[inline]
    fn name(&self) -> &'static str {
        (**self).name()
    }

    #[inline]
    fn abstract_eval(&self, inputs: &[ArrayType]) -> Result<Vec<ArrayType>, TraceError> {
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

impl<T: InterpretableOp<V> + ?Sized, V: TraceValue> InterpretableOp<V> for Arc<T> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).interpret(inputs)
    }
}

impl<T: LinearOp<V> + ?Sized, V: TraceValue> LinearOp<V> for Arc<T> {
    #[inline]
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        (**self).transpose(inputs, outputs, output_cotangents)
    }
}

impl<T: DifferentiableOp<V, U> + ?Sized, V: TraceValue, U> DifferentiableOp<V, U> for Arc<T> {
    #[inline]
    fn jvp(&self, inputs: &[JvpTracer<V, U>]) -> Result<Vec<JvpTracer<V, U>>, TraceError> {
        (**self).jvp(inputs)
    }
}

impl<T: VectorizableOp<V> + ?Sized, V: TraceValue> VectorizableOp<V> for Arc<T> {
    #[inline]
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError> {
        (**self).batch(inputs)
    }
}

// ---------------------------------------------------------------------------
// PrimitiveOp — Debug, Display, Op, InterpretableOp, LinearOp, DifferentiableOp, VectorizableOp
// ---------------------------------------------------------------------------

use crate::tracing_v2::operations::{
    AddOp, CosOp, LeftMatMulOp, LinearMatrixTransposeOp, MatMulOp, MatrixTransposeOp, MulOp, NegOp, ReshapeOp,
    RightMatMulOp, ScaleOp, SinOp, left_matmul::left_matmul_abstract_eval, right_matmul::right_matmul_abstract_eval,
};

impl<V: TraceValue> Debug for PrimitiveOp<V> {
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

impl<V: TraceValue> Display for PrimitiveOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

impl<V: TraceValue> Debug for LinearPrimitiveOp<V> {
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

impl<V: TraceValue> Display for LinearPrimitiveOp<V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reshape { output_type, .. } => write!(formatter, "reshape{}", output_type.shape),
            _ => write!(formatter, "{}", self.name()),
        }
    }
}

/// [`Op`] for [`PrimitiveOp`] requires NO value-type bounds — shape validation works for any `V: TraceValue`.
impl<V: TraceValue> Op for PrimitiveOp<V> {
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
            Self::Scale { .. } => ScaleOp::<V>::abstract_eval_static(inputs),
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
                ScaleOp::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
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

/// [`Op`] for [`LinearPrimitiveOp`] requires NO value-type bounds — shape validation works for any `V: TraceValue`.
impl<V: TraceValue> Op for LinearPrimitiveOp<V> {
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
            Self::Scale { .. } => ScaleOp::<V>::abstract_eval_static(inputs),
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
                ScaleOp::new(factor.clone()).try_simplify(inputs, is_zero_constant, is_one_constant)
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

/// [`InterpretableOp`] for [`PrimitiveOp`] requires the full value capability set.
impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    InterpretableOp<V> for PrimitiveOp<V>
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

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    InterpretableOp<V> for LinearPrimitiveOp<V>
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

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    LinearOp<V> for LinearPrimitiveOp<V>
{
    fn transpose(
        &self,
        inputs: &[V],
        outputs: &[V],
        output_cotangents: &[LinearTerm<V>],
    ) -> Result<Vec<Option<LinearTerm<V>>>, TraceError> {
        match self {
            Self::Add => AddOp.transpose(inputs, outputs, output_cotangents),
            Self::Neg => NegOp.transpose(inputs, outputs, output_cotangents),
            Self::MatrixTranspose => MatrixTransposeOp.transpose(inputs, outputs, output_cotangents),
            Self::LinearMatrixTranspose => LinearMatrixTransposeOp.transpose(inputs, outputs, output_cotangents),
            Self::Scale { factor } => ScaleOp::new(factor.clone()).transpose(inputs, outputs, output_cotangents),
            Self::LeftMatMul { factor } => {
                LeftMatMulOp::new(factor.clone()).transpose(inputs, outputs, output_cotangents)
            }
            Self::RightMatMul { factor } => {
                RightMatMulOp::new(factor.clone()).transpose(inputs, outputs, output_cotangents)
            }
            Self::Reshape { input_type, output_type } => {
                ReshapeOp::new(input_type.clone(), output_type.clone()).transpose(inputs, outputs, output_cotangents)
            }
            Self::VMap(vmap) => vmap.transpose(inputs, outputs, output_cotangents),
            Self::Rematerialize(remat) => remat.transpose(inputs, outputs, output_cotangents),
            Self::Custom(op) => op.transpose(inputs, outputs, output_cotangents),
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
    V: TraceValue
        + FloatExt
        + ZeroLike
        + OneLike
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<crate::tracing_v2::linear::Linearized<JitTracer<V>>> for PrimitiveOp<V>
{
    fn interpret(
        &self,
        inputs: &[crate::tracing_v2::linear::Linearized<JitTracer<V>>],
    ) -> Result<Vec<crate::tracing_v2::linear::Linearized<JitTracer<V>>>, TraceError> {
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

impl<V: TraceValue + FloatExt + ZeroLike + OneLike + MatrixOps + crate::tracing_v2::operations::reshape::ReshapeOps>
    DifferentiableOp<V, LinearTerm<V>> for PrimitiveOp<V>
{
    fn jvp(&self, inputs: &[JvpTracer<V, LinearTerm<V>>]) -> Result<Vec<JvpTracer<V, LinearTerm<V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<V, LinearTerm<V>>::jvp(&AddOp, inputs),
            Self::Mul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MulOp, inputs),
            Self::Neg => DifferentiableOp::<V, LinearTerm<V>>::jvp(&NegOp, inputs),
            Self::Sin => DifferentiableOp::<V, LinearTerm<V>>::jvp(&SinOp, inputs),
            Self::Cos => DifferentiableOp::<V, LinearTerm<V>>::jvp(&CosOp, inputs),
            Self::Scale { factor } => DifferentiableOp::<V, LinearTerm<V>>::jvp(&ScaleOp::new(factor.clone()), inputs),
            Self::MatMul => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatMulOp, inputs),
            Self::MatrixTranspose => DifferentiableOp::<V, LinearTerm<V>>::jvp(&MatrixTransposeOp, inputs),
            Self::LinearMatrixTranspose => DifferentiableOp::<V, LinearTerm<V>>::jvp(&LinearMatrixTransposeOp, inputs),
            Self::LeftMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&LeftMatMulOp::new(factor.clone()), inputs)
            }
            Self::RightMatMul { factor } => {
                DifferentiableOp::<V, LinearTerm<V>>::jvp(&RightMatMulOp::new(factor.clone()), inputs)
            }
            Self::Reshape { input_type, output_type } => DifferentiableOp::<V, LinearTerm<V>>::jvp(
                &ReshapeOp::new(input_type.clone(), output_type.clone()),
                inputs,
            ),
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => DifferentiableOp::<V, LinearTerm<V>>::jvp(remat.as_ref(), inputs),
            Self::Custom(op) => op.jvp(inputs),
        }
    }
}

impl<V: TraceValue + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + FloatExt + MatrixOps> VectorizableOp<V>
    for PrimitiveOp<V>
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

    use crate::tracing_v2::{Batch, CompiledFunction, LinearProgramBuilder, TraceError, grad, jvp, try_jit, vmap};
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

    impl InterpretableOp<f64> for ShiftOp {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0] + self.amount])
        }
    }

    impl LinearOp<f64> for ShiftOp {
        fn transpose(
            &self,
            inputs: &[f64],
            outputs: &[f64],
            output_cotangents: &[LinearTerm<f64>],
        ) -> Result<Vec<Option<LinearTerm<f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            if outputs.len() != 1 {
                return Err(TraceError::InvalidOutputCount { expected: 1, got: outputs.len() });
            }
            if output_cotangents.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl DifferentiableOp<f64, LinearTerm<f64>> for ShiftOp {
        fn jvp(
            &self,
            inputs: &[JvpTracer<f64, LinearTerm<f64>>],
        ) -> Result<Vec<JvpTracer<f64, LinearTerm<f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![JvpTracer { primal: inputs[0].primal + self.amount, tangent: inputs[0].tangent.clone() }])
        }
    }

    impl VectorizableOp<f64> for ShiftOp {
        fn batch(&self, inputs: &[Batch<f64>]) -> Result<Vec<Batch<f64>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![Batch::new(inputs[0].lanes().iter().map(|lane| lane + self.amount).collect::<Vec<_>>())])
        }
    }

    impl InterpretableOp<Linearized<JitTracer<f64>>> for ShiftOp {
        fn interpret(
            &self,
            inputs: &[Linearized<JitTracer<f64>>],
        ) -> Result<Vec<Linearized<JitTracer<f64>>>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            let primal =
                apply_custom_traced_unary(inputs[0].primal.clone(), CustomPrimitive::<f64>::new(self.clone()))?;
            Ok(vec![Linearized { primal, tangent: inputs[0].tangent.clone() }])
        }
    }

    /// Applies one unary custom primitive to one traced scalar.
    fn apply_custom_traced_unary(
        input: JitTracer<f64>,
        primitive: CustomPrimitive<f64>,
    ) -> Result<JitTracer<f64>, TraceError> {
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
    fn stage_custom_traced_unary(input: JitTracer<f64>, primitive: CustomPrimitive<f64>) -> JitTracer<f64> {
        apply_custom_traced_unary(input, primitive).expect("custom primitive staging should succeed")
    }

    /// Applies one unary custom primitive to one batched scalar.
    fn apply_custom_batched_unary(
        input: Batch<f64>,
        primitive: CustomPrimitive<f64>,
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
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));

        assert!(matches!(
            primitive.into_linear(),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_base_execution_replays_without_optional_rules() {
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));
        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) = try_jit(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<f64>| Ok(stage_custom_traced_unary(x, primitive.clone()))
            },
            3.0f64,
        )
        .unwrap();

        assert_eq!(output, 5.0);
        assert_eq!(compiled.call(4.0f64), Ok(6.0));
    }

    #[test]
    fn test_custom_primitive_missing_transpose_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));
        let builder = Rc::new(RefCell::new(LinearProgramBuilder::<f64>::new()));
        let cotangent_atom = builder.borrow_mut().add_input(&0.0);
        let cotangent = LinearTerm::from_staged_parts(cotangent_atom, builder);

        assert!(matches!(
            primitive.transpose(&[3.0f64], &[5.0f64], &[cotangent]),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "transpose" })
        ));
    }

    #[test]
    fn test_custom_primitive_missing_jvp_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));
        let result: Result<(f64, f64), TraceError> = jvp(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<f64>| stage_custom_traced_unary(x, primitive.clone())
            },
            3.0f64,
            1.0f64,
        );

        assert_eq!(result, Err(TraceError::MissingCustomRule { op: "test_shift", transform: "jvp" }),);
    }

    #[test]
    fn test_custom_primitive_missing_linearized_jit_rule_reports_targeted_error() {
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0)).with_jvp_rule(ShiftOp::new(2.0));
        let result: Result<(f64, CompiledFunction<f64, f64, f64>), TraceError> = try_jit(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<f64>| {
                    let (primal, tangent) = jvp(
                        {
                            let primitive = primitive.clone();
                            move |inner: JitTracer<f64>| stage_custom_traced_unary(inner, primitive.clone())
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
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0))
            .with_jvp_rule(ShiftOp::new(2.0))
            .with_linearized_jit_rule(ShiftOp::new(2.0));

        assert_eq!(
            grad(
                {
                    let primitive = primitive.clone();
                    move |x: JitTracer<f64>| stage_custom_traced_unary(x, primitive.clone())
                },
                3.0f64,
            ),
            Ok(1.0f64),
        );

        let (output, compiled): (f64, CompiledFunction<f64, f64, f64>) = try_jit(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<f64>| {
                    let (primal, tangent) = jvp(
                        {
                            let primitive = primitive.clone();
                            move |inner: JitTracer<f64>| stage_custom_traced_unary(inner, primitive.clone())
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
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));

        assert_eq!(
            apply_custom_batched_unary(Batch::new(vec![1.0f64, 2.0]), primitive),
            Err(TraceError::MissingCustomRule { op: "test_shift", transform: "vectorize" }),
        );
    }

    #[test]
    fn test_custom_primitive_batch_rule_participates_in_vmap() {
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0)).with_vectorization_rule(ShiftOp::new(2.0));

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
        let primitive = CustomPrimitive::<f64>::new(ShiftOp::new(2.0));

        assert_eq!(primitive.abstract_eval(&[scalar_type()]), Ok(vec![scalar_type()]));
    }
}
