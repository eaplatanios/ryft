use std::sync::Arc;

use crate::{
    parameters::Parameterized,
    tracing::{InterpretableOperation, Operation, Traceable, TracingError},
    tracing_v2::{
        engines::{DifferentiableEngine, Engine},
        forward::{Differentiable, EngineTangent, JvpTracer},
        jit::Tracer,
        linear::LinearTerm,
    },
    types::{ArrayType, Type, TypeError},
};

/// Elementwise addition.
pub mod add;

/// Elementwise cosine.
pub mod cos;

/// Value-level identity helpers and built-in scalar constant traits.
pub mod constants;

/// Higher-order condition and while-loop operations.
pub mod control_flow;

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

pub use add::{AddOperation, AddTracingOperation, LinearAddOperation};
pub use control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram, LinearConditionOperation,
    WhileOperation, flat_program_input_types, flat_program_output_types,
};
pub use cos::{Cos, CosOperation, CosTracingOperation};
pub use custom::{
    CustomOperationError, CustomPrimitive, CustomPrimitiveExtensions, CustomTracingOperation, LinearCustomOperation,
    LinearCustomPrimitive,
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

/// Lifts one concrete value into the staged program owned by a JIT tracer.
pub fn lift_jit_constant<'engine, V: Traceable<ArrayType>, E: Engine<Type = ArrayType, Value = V> + ?Sized>(
    constant: &V,
    exemplar: &Tracer<'engine, E>,
) -> Tracer<'engine, E> {
    let builder = exemplar.builder.clone();
    let r#type = constant.r#type().into_owned();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    Tracer::from_staged_parts(atom, r#type, builder, exemplar.engine)
}

/// Propagates one unary input type through a shape-preserving staged op.
pub fn unary_abstract(inputs: &[ArrayType]) -> Result<ArrayType, TypeError> {
    if inputs.len() != 1 {
        return Err(TypeError { message: format!("expected 1 input type but got {}", inputs.len()) });
    }
    Ok(inputs[0].clone())
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
///       Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new(Vec::new())));
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
///       Rc::new(RefCell::new(ProgramBuilder::<ArrayType, f64, LinearPrimitiveOperation<f64>>::new(Vec::new())));
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
///       ProgramBuilder::<ArrayType, ndarray::Array2<f64>, LinearPrimitiveOperation<ndarray::Array2<f64>>>::new(
///           Vec::new(),
///       ),
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
pub trait LinearOperation<T: Type, V: Traceable<T>, LinearCarrier: Clone = primitive::LinearPrimitiveOperation<V>>:
    Operation<T>
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

/// Forward-mode differentiation rule keyed only by the engine that owns the staged carriers.
///
/// Primitive JVP rules recover their tangent representation from [`Differentiable`] on
/// `E::Value` at `E::Type`, so the operation trait no longer needs separate `T`, `V`, `O`, `L`, or
/// tangent parameters. Per-op capability requirements still live in the individual impl blocks
/// through bounds on `E::Value` and [`EngineTangent<E>`].
pub trait DifferentiableOperation<E: DifferentiableEngine + ?Sized>: Operation<E::Type>
where
    E::Value: Differentiable<E::Type>,
{
    /// Applies the forward-mode JVP rule.
    ///
    /// The `engine` argument carries the context needed to synthesize zero values for higher-order
    /// ops that replay staged sub-programs such as [`RematerializeOperation`]. Pure arithmetic ops
    /// ignore it.
    fn jvp(
        &self,
        engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError>;
}

/// Default linear-op carrier capability: eager replay on concrete values.
///
/// This captures the minimum replay surface a stored linear carrier needs: shape metadata through
/// [`Operation`] and concrete evaluation through [`InterpretableOperation`].
///
/// It is intentionally narrower than the ordinary staged carrier used during tracing: linear
/// programs only need metadata reasoning plus concrete replay, not forward-mode differentiation.
///
/// The linear-program surface consists of shape metadata through [`Operation`], concrete replay
/// through [`InterpretableOperation`], and a separate transpose rule through [`LinearOperation`].
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
pub trait TracerLinearOperation<V: Traceable<ArrayType>, E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static>:
    Clone + Operation<ArrayType> + 'static
where
    for<'engine> Self: add::LinearAddOperation<ArrayType, Tracer<'engine, E>>
        + neg::LinearNegOperation<ArrayType, Tracer<'engine, E>>
        + scale::LinearScaleOperation<ArrayType, Tracer<'engine, E>>,
{
}

impl<
    V: Traceable<ArrayType>,
    E: Engine<Type = ArrayType, Value = V> + ?Sized + 'static,
    InnerLinearOperation: Clone + Operation<ArrayType> + 'static,
> TracerLinearOperation<V, E> for InnerLinearOperation
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
    fn infer_output_types(&self, input_types: &[T]) -> Result<Vec<T>, TypeError> {
        (**self).infer_output_types(input_types)
    }

    #[inline]
    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        (**self).render(formatter, indentation)
    }
}

impl<O: InterpretableOperation<T, V> + ?Sized, T: Type, V: Traceable<T>> InterpretableOperation<T, V> for Arc<O> {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, TracingError> {
        (**self).interpret(inputs)
    }
}

impl<O: LinearOperation<T, V, LinearCarrier> + ?Sized, T: Type, V: Traceable<T>, LinearCarrier: Clone>
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

impl<InnerOperation: DifferentiableOperation<E> + ?Sized, E: DifferentiableEngine + ?Sized> DifferentiableOperation<E>
    for Arc<InnerOperation>
where
    E::Value: Differentiable<E::Type>,
{
    #[inline]
    fn jvp(
        &self,
        engine: &E,
        inputs: &[JvpTracer<E::Value, EngineTangent<E>>],
    ) -> Result<Vec<JvpTracer<E::Value, EngineTangent<E>>>, TracingError> {
        (**self).jvp(engine, inputs)
    }
}
