//! Primitive operation traits for `tracing_v2`.
//!
//! The staged op set is intentionally open: each primitive is represented by its own concrete type implementing one
//! or more transform-specific traits. This module keeps only the operation-neutral dispatch interfaces.
//!
//! # Trait hierarchy
//!
//! ```text
//! Op<T: Type>                       — shape-level, generic over type descriptor T
//! ├─ InterpretableOp<T, V>          — concrete execution on values of type V
//! ├─ LinearOp<T, V>                 — semantic reverse-mode transpose rule
//! ├─ DifferentiableOp<T, V, Tangent>— forward-mode JVP rule, generic over tangent type
//! └─ VectorizableOp<T, V>           — batching rule for vmap
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
        FloatExt, MatrixOps, OneLike, TraceError, Traceable, ZeroLike,
        batch::Batch,
        forward::JvpTracer,
        jit::JitTracer,
        linear::{LinearTerm, Linearized},
    },
    types::{ArrayType, Type, Typed},
};

/// Shape-level operation interface for staged graphs.
///
/// This trait covers the metadata surface needed for graph construction, display, simplification, and MLIR lowering.
/// Concrete execution is provided by the separate [`InterpretableOp`] trait. Staged-program differentiation rules
/// are split between [`LinearOp`] (transpose/replay) and [`DifferentiableOp`] (forward-mode JVP).
///
/// The type parameter `T` determines which abstract type descriptor is used for shape-level reasoning. The default
/// is [`ArrayType`], which covers the entire core tracing infrastructure. Future instantiations with different type
/// descriptors can reuse the same trait without modifying existing implementations.
pub trait Op<T: Type + Clone = ArrayType>: Debug + Display {
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
pub trait InterpretableOp<T: Type + Clone, V: Typed<T>>: Op<T> {
    /// Executes the operation on concrete values.
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError>;
}

/// Represents [`Op`]s that can appear in tangent/cotangent programs and support reverse-mode transposition.
///
/// For one linear operation `y = L(x)`, the transpose rule builds the reverse linear map `L^T`
/// that pulls cotangents on `y` back to cotangents on `x`.
///
/// In other words, this trait models the adjoint used by reverse-mode differentiation once one
/// primal program has already been linearized. The rule does not receive concrete primal witnesses
/// because those are not part of the transpose trace. Instead, it operates directly on staged
/// output cotangents and emits staged cotangent contributions for the op inputs.
///
/// A few concrete examples:
///
/// - For [`ScaleOp`], `y = a * x`, the transpose stages one new [`LinearTerm`] representing
///   `a * c`, where `c` is the output cotangent:
///   ```rust,ignore
///   use std::{cell::RefCell, rc::Rc};
///
///   use ryft_core::tracing_v2::{LinearOp, LinearProgramBuilder, LinearTerm, ScaleOp};
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
///   use ryft_core::tracing_v2::{AddOp, LinearOp, LinearProgramBuilder, LinearTerm};
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
///   use ryft_core::tracing_v2::{LinearOp, LinearProgramBuilder, LinearTerm, MatrixTransposeOp};
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
pub trait LinearOp<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter>: Op<T> {
    /// Applies the transpose rule for reverse-mode differentiation.
    ///
    /// `output_cotangents` is aligned with the op outputs in forward order. The returned vector
    /// must be aligned with the op inputs in forward order.
    ///
    /// Returning `Some(term)` means that input receives the staged cotangent contribution `term`.
    /// Returning `None` means the contribution is structurally zero and the transpose pass does not
    /// need to materialize an explicit zero term for that input.
    fn transpose(&self, output_cotangents: &[LinearTerm<T, V>]) -> Result<Vec<Option<LinearTerm<T, V>>>, TraceError>;
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
pub trait DifferentiableOp<T: Type + Clone, V: Typed<T>, Tangent>: Op<T> {
    /// Applies the forward-mode JVP rule.
    fn jvp(&self, inputs: &[JvpTracer<V, Tangent>]) -> Result<Vec<JvpTracer<V, Tangent>>, TraceError>;
}

/// Primitive operation with a batching rule used by `vmap`.
pub trait VectorizableOp<T: Type + Clone, V: Typed<T>>: Op<T> {
    /// Applies the primitive's batching rule to batched inputs.
    fn batch(&self, inputs: &[Batch<V>]) -> Result<Vec<Batch<V>>, TraceError>;
}

/// Typed extension registry carried by one [`CustomPrimitive`].
#[derive(Clone, Default)]
pub struct CustomPrimitiveExtensions<T: Type + Clone, V: Typed<T>> {
    entries: HashMap<TypeId, Arc<dyn Any + Send + Sync>>,
    _marker: std::marker::PhantomData<(T, V)>,
}

impl<T: Type + Clone, V: Traceable<T>> Debug for CustomPrimitiveExtensions<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.debug_struct("CustomPrimitiveExtensions").field("count", &self.entries.len()).finish()
    }
}

impl<T: Type + Clone, V: Traceable<T>> CustomPrimitiveExtensions<T, V> {
    /// Inserts one typed extension into the registry, replacing any previous extension of the same type.
    pub fn insert<E: Send + Sync + 'static>(&mut self, extension: E) {
        self.entries.insert(TypeId::of::<E>(), Arc::new(extension));
    }

    /// Returns the registered extension of type `E`, if present.
    pub fn get<E: Send + Sync + 'static>(&self) -> Option<&E> {
        self.entries.get(&TypeId::of::<E>()).and_then(|extension| extension.as_ref().downcast_ref::<E>())
    }
}

/// Type-erased wrapper for a linearized-JIT replay rule stored inside [`CustomPrimitiveExtensions`].
///
/// This wrapper is `Send + Sync + 'static` so it can live inside the extension registry. The `V: Traceable<ArrayType>`
/// bound is required at construction time but does not appear on the outer [`CustomPrimitive`] struct.
struct LinearizedJitRule<V: Traceable<ArrayType> + ZeroLike>(
    Arc<dyn InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>> + Send + Sync>,
);

impl<V: Traceable<ArrayType> + ZeroLike> LinearizedJitRule<V> {
    fn interpret(
        &self,
        inputs: &[Linearized<JitTracer<ArrayType, V>>],
    ) -> Result<Vec<Linearized<JitTracer<ArrayType, V>>>, TraceError> {
        self.0.interpret(inputs)
    }
}

trait CustomBaseOp<T: Type + Clone, V: Typed<T>>: Op<T> + InterpretableOp<T, V> + Send + Sync {}

impl<Ty: Type + Clone, V: Traceable<Ty>, O: Op<Ty> + InterpretableOp<Ty, V> + Send + Sync> CustomBaseOp<Ty, V> for O {}


/// Rule-based registration object used by [`PrimitiveOp::Custom`].
///
/// The base op always supplies shape metadata and eager interpretation. Optional transform rules are
/// registered using the existing tracing traits directly:
///
/// - [`LinearOp<ArrayType, V>`] for reverse-mode transpose,
/// - [`DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>>`] for forward-mode JVP,
/// - [`VectorizableOp<ArrayType, V>`] for `vmap`, and
/// - [`InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>>`] for fully general linearized-JIT replay.
#[derive(Clone)]
pub struct CustomPrimitive<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter> {
    base: Arc<dyn CustomBaseOp<T, V>>,
    transpose_rule: Option<Arc<dyn LinearOp<T, V> + Send + Sync>>,
    jvp_rule: Option<Arc<dyn DifferentiableOp<T, V, LinearTerm<T, V>> + Send + Sync>>,
    vectorization_rule: Option<Arc<dyn VectorizableOp<T, V> + Send + Sync>>,
    extensions: CustomPrimitiveExtensions<T, V>,
}

impl<T: Type + Clone + Display, V: Traceable<T>> CustomPrimitive<T, V> {
    /// Creates one custom primitive from its required base operation.
    pub fn new<Base>(base: Base) -> Self
    where
        Base: Op<T> + InterpretableOp<T, V> + Send + Sync + 'static,
    {
        Self {
            base: Arc::new(base),
            transpose_rule: None,
            jvp_rule: None,
            vectorization_rule: None,
            extensions: CustomPrimitiveExtensions { entries: HashMap::new(), _marker: std::marker::PhantomData },
        }
    }

    /// Registers one transpose rule for reverse-mode differentiation.
    pub fn with_transpose_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: LinearOp<T, V> + Send + Sync + 'static,
    {
        self.transpose_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one forward-mode JVP rule.
    pub fn with_jvp_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: DifferentiableOp<T, V, LinearTerm<T, V>> + Send + Sync + 'static,
    {
        self.jvp_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one batching rule.
    pub fn with_vectorization_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: VectorizableOp<T, V> + Send + Sync + 'static,
    {
        self.vectorization_rule = Some(Arc::new(rule));
        self
    }

    /// Registers one linearized-JIT replay rule for nested custom primitives.
    ///
    /// The rule is stored inside the extensions registry so that `CustomPrimitive<T, V>` does not
    /// require `V: Traceable<ArrayType>` on its struct definition. The concrete rule is recovered
    /// via the extensions in the `InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>> for CustomPrimitive<ArrayType, V>` impl.
    #[doc(hidden)]
    pub fn with_linearized_jit_rule<Rule>(mut self, rule: Rule) -> Self
    where
        Rule: InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>> + Send + Sync + 'static,
        Linearized<JitTracer<ArrayType, V>>: Traceable<ArrayType>,
        V: Traceable<ArrayType> + ZeroLike,
    {
        self.extensions.insert(LinearizedJitRule::<V>(Arc::new(rule)));
        self
    }

    /// Registers one typed extension.
    pub fn with_extension<E: Send + Sync + 'static>(mut self, extension: E) -> Self {
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

    fn jvp_rule(&self) -> Result<&(dyn DifferentiableOp<T, V, LinearTerm<T, V>> + Send + Sync), TraceError> {
        self.jvp_rule.as_deref().ok_or_else(|| self.missing_rule("jvp"))
    }
}

impl<T: Type + Clone + Display, V: Traceable<T>> Debug for CustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.base.as_ref(), formatter)
    }
}

impl<T: Type + Clone + Display, V: Traceable<T>> Display for CustomPrimitive<T, V> {
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

impl<V: Traceable<ArrayType>> LinearOp<ArrayType, V> for CustomPrimitive<ArrayType, V> {
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

impl<V: Traceable<ArrayType>> DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>> for CustomPrimitive<ArrayType, V> {
    fn jvp(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        self.jvp_rule()?.jvp(inputs)
    }
}

impl<V: Traceable<ArrayType> + ZeroLike>
    InterpretableOp<ArrayType, Linearized<JitTracer<ArrayType, V>>> for CustomPrimitive<ArrayType, V>
where
    Linearized<JitTracer<ArrayType, V>>: Traceable<ArrayType>,
{
    fn interpret(
        &self,
        inputs: &[Linearized<JitTracer<ArrayType, V>>],
    ) -> Result<Vec<Linearized<JitTracer<ArrayType, V>>>, TraceError> {
        self.extensions
            .get::<LinearizedJitRule<V>>()
            .ok_or_else(|| self.missing_rule("linearized JIT replay"))?
            .interpret(inputs)
    }
}

/// Linear-only wrapper around one [`CustomPrimitive`] that guarantees a transpose rule is present.
#[derive(Clone)]
pub struct LinearCustomPrimitive<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter> {
    primitive: Arc<CustomPrimitive<T, V>>,
}

impl<T: Type + Clone + Display, V: Traceable<T>> LinearCustomPrimitive<T, V> {
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

impl<T: Type + Clone + Display, V: Traceable<T>> Debug for LinearCustomPrimitive<T, V> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Debug::fmt(self.primitive.as_ref(), formatter)
    }
}

impl<T: Type + Clone + Display, V: Traceable<T>> Display for LinearCustomPrimitive<T, V> {
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

impl<V: Traceable<ArrayType>> LinearOp<ArrayType, V> for LinearCustomPrimitive<ArrayType, V> {
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
pub enum PrimitiveOp<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter> {
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
    VMap(Box<crate::tracing_v2::operations::VMapOp<T, V>>),

    /// Higher-order rematerialization boundary carrying a compiled body and optional transpose body.
    Rematerialize(Box<crate::tracing_v2::operations::RematerializeOp<T, V>>),

    /// Escape hatch for user- or crate-defined operations outside `ryft-core`.
    Custom(Arc<CustomPrimitive<T, V>>),
}

impl<T: Type + Clone + Display, V: Traceable<T>> Clone for PrimitiveOp<T, V> {
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
pub enum LinearPrimitiveOp<T: Type + Clone + Display, V: Typed<T> + Clone + Parameter> {
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
    VMap(Box<crate::tracing_v2::operations::LinearVMapOp<T, V>>),

    /// Higher-order rematerialization boundary restricted to linear bodies and transpose bodies.
    Rematerialize(Box<crate::tracing_v2::operations::LinearRematerializeOp<T, V>>),

    /// Escape hatch for user- or crate-defined linear custom operations.
    Custom(Arc<LinearCustomPrimitive<T, V>>),
}

impl<T: Type + Clone + Display, V: Traceable<T>> Clone for LinearPrimitiveOp<T, V> {
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

// ---------------------------------------------------------------------------
// Arc forwarding impls
// ---------------------------------------------------------------------------

impl<O: Op<T> + ?Sized, T: Type + Clone> Op<T> for Arc<O> {
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

impl<O: InterpretableOp<T, V> + ?Sized, T: Type + Clone, V: Traceable<T>> InterpretableOp<T, V> for Arc<O> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TraceError> {
        (**self).interpret(inputs)
    }
}

impl<O: LinearOp<T, V> + ?Sized, T: Type + Clone + Display, V: Traceable<T>> LinearOp<T, V> for Arc<O> {
    #[inline]
    fn transpose(&self, output_cotangents: &[LinearTerm<T, V>]) -> Result<Vec<Option<LinearTerm<T, V>>>, TraceError> {
        (**self).transpose(output_cotangents)
    }
}

impl<O: DifferentiableOp<T, V, Tangent> + ?Sized, T: Type + Clone, V: Traceable<T>, Tangent>
    DifferentiableOp<T, V, Tangent> for Arc<O>
{
    #[inline]
    fn jvp(&self, inputs: &[JvpTracer<V, Tangent>]) -> Result<Vec<JvpTracer<V, Tangent>>, TraceError> {
        (**self).jvp(inputs)
    }
}

impl<O: VectorizableOp<T, V> + ?Sized, T: Type + Clone, V: Traceable<T>> VectorizableOp<T, V> for Arc<O> {
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

impl<T: Type + Clone + Display, V: Traceable<T>> Debug for PrimitiveOp<T, V> {
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

impl<T: Type + Clone + Display, V: Traceable<T>> Debug for LinearPrimitiveOp<T, V> {
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

/// [`InterpretableOp`] for [`PrimitiveOp`] requires the full value capability set.
impl<
    V: Traceable<ArrayType>
        + FloatExt
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
        + FloatExt
        + ZeroLike
        + OneLike
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
        + FloatExt
        + ZeroLike
        + OneLike
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> LinearOp<ArrayType, V> for LinearPrimitiveOp<ArrayType, V>
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
        + FloatExt
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + Add<Output = V>
        + Mul<Output = V>
        + Neg<Output = V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> InterpretableOp<ArrayType, crate::tracing_v2::linear::Linearized<JitTracer<ArrayType, V>>> for PrimitiveOp<ArrayType, V>
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
        + FloatExt
        + ZeroLike
        + OneLike
        + Parameterized<V>
        + MatrixOps
        + crate::tracing_v2::operations::reshape::ReshapeOps,
> DifferentiableOp<ArrayType, V, LinearTerm<ArrayType, V>> for PrimitiveOp<ArrayType, V>
where
    V::ParameterStructure: Clone + PartialEq,
    Vec<V>: Parameterized<V, ParameterStructure: Clone + PartialEq>,
{
    fn jvp(
        &self,
        inputs: &[JvpTracer<V, LinearTerm<ArrayType, V>>],
    ) -> Result<Vec<JvpTracer<V, LinearTerm<ArrayType, V>>>, TraceError> {
        match self {
            Self::Add => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&AddOp, inputs),
            Self::Mul => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&MulOp, inputs),
            Self::Neg => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&NegOp, inputs),
            Self::Sin => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&SinOp, inputs),
            Self::Cos => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&CosOp, inputs),
            Self::Scale { factor } => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&ScaleOp::new(factor.clone()), inputs)
            }
            Self::MatMul => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&MatMulOp, inputs),
            Self::MatrixTranspose => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&MatrixTransposeOp, inputs)
            }
            Self::LinearMatrixTranspose => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(&LinearMatrixTransposeOp, inputs)
            }
            Self::LeftMatMul { factor } => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(
                    &LeftMatMulOp::new(factor.clone()),
                    inputs,
                )
            }
            Self::RightMatMul { factor } => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(
                    &RightMatMulOp::new(factor.clone()),
                    inputs,
                )
            }
            Self::Reshape { input_type, output_type } => DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(
                &ReshapeOp::new(input_type.clone(), output_type.clone()),
                inputs,
            ),
            Self::VMap(vmap) => Err(TraceError::HigherOrderOpFailure {
                op: "linearize_program",
                message: format!("JVP rule for staged op '{}' is not implemented", vmap.name()),
            }),
            Self::Rematerialize(remat) => {
                DifferentiableOp::<ArrayType, V, LinearTerm<ArrayType, V>>::jvp(remat.as_ref(), inputs)
            }
            Self::Custom(op) => op.jvp(inputs),
        }
    }
}

impl<V: Traceable<ArrayType> + Add<Output = V> + Mul<Output = V> + Neg<Output = V> + FloatExt + MatrixOps>
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

    impl InterpretableOp<ArrayType, f64> for ShiftOp {
        fn interpret(&self, inputs: &[f64]) -> Result<Vec<f64>, TraceError> {
            if inputs.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: inputs.len() });
            }
            Ok(vec![inputs[0] + self.amount])
        }
    }

    impl LinearOp<ArrayType, f64> for ShiftOp {
        fn transpose(&self, output_cotangents: &[LinearTerm<ArrayType, f64>]) -> Result<Vec<Option<LinearTerm<ArrayType, f64>>>, TraceError> {
            if output_cotangents.len() != 1 {
                return Err(TraceError::InvalidInputCount { expected: 1, got: output_cotangents.len() });
            }
            Ok(vec![Some(output_cotangents[0].clone())])
        }
    }

    impl DifferentiableOp<ArrayType, f64, LinearTerm<ArrayType, f64>> for ShiftOp {
        fn jvp(
            &self,
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
            let primal =
                apply_custom_traced_unary(inputs[0].primal.clone(), CustomPrimitive::<ArrayType, f64>::new(self.clone()))?;
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
    fn stage_custom_traced_unary(input: JitTracer<ArrayType, f64>, primitive: CustomPrimitive<ArrayType, f64>) -> JitTracer<ArrayType, f64> {
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
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = try_jit(
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
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0));
        let result: Result<(f64, f64), TraceError> = jvp(
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
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_jvp_rule(ShiftOp::new(2.0));
        let result: Result<(f64, CompiledFunction<ArrayType, f64, f64, f64>), TraceError> = try_jit(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| {
                    let (primal, tangent) = jvp(
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
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0))
            .with_jvp_rule(ShiftOp::new(2.0))
            .with_linearized_jit_rule(ShiftOp::new(2.0));

        assert_eq!(
            grad(
                {
                    let primitive = primitive.clone();
                    move |x: JitTracer<ArrayType, f64>| stage_custom_traced_unary(x, primitive.clone())
                },
                3.0f64,
            ),
            Ok(1.0f64),
        );

        let (output, compiled): (f64, CompiledFunction<ArrayType, f64, f64, f64>) = try_jit(
            {
                let primitive = primitive.clone();
                move |x: JitTracer<ArrayType, f64>| {
                    let (primal, tangent) = jvp(
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
        let primitive = CustomPrimitive::<ArrayType, f64>::new(ShiftOp::new(2.0)).with_vectorization_rule(ShiftOp::new(2.0));

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
