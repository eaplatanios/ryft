use std::cell::RefCell;
use std::rc::Rc;

use crate::tracing::{AtomId, Instruction, InterpretableOperation, Operation, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::DifferentiableEngine;
use crate::tracing_v2::engines::StagingEngine;
use crate::tracing_v2::forward::{JvpContext, JvpTracer};
use crate::tracing_v2::jit::Tracer;
use crate::types::{ArrayType, Type, TypeError, Typed};

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

pub use add::{AddOperation, SupportsAdd};
pub use constants::{SupportsZero, ZeroOperation};
pub use control_flow::{
    ConditionOperation, ConditionPredicate, ControlFlowError, ControlFlowValue, FlatProgram, WhileOperation,
    flat_program_input_types, flat_program_output_types,
};
pub use cos::{Cos, CosOperation, SupportsCos};
pub use custom::{
    CustomOperationError, CustomPrimitive, CustomPrimitiveExtensions, LinearCustomPrimitive, SupportsCustom,
    SupportsLinearCustom,
};
pub use left_matmul::{LeftMatMulOperation, SupportsLeftMatMul};
pub use matmul::{MatMulOperation, SupportsMatMul};
pub use matrix_transpose::{MatrixTransposeOperation, SupportsMatrixTranspose};
pub use mul::{MulOperation, SupportsMul};
pub use neg::{NegOperation, SupportsNeg};
pub use primitive::{LinearPrimitiveOperation, PrimitiveOperation};
pub use rematerialize::{
    FlatTracedRematerialize, LinearRematerializeOperation, RematerializeOperation, SupportsLinearRematerialize,
    SupportsRematerialize,
};
pub use reshape::{ReshapeOperation, SupportsReshape};
pub use right_matmul::{RightMatMulOperation, SupportsRightMatMul};
pub use scale::{ScaleOperation, SupportsScale};
pub use sin::{Sin, SinOperation, SupportsSin};

/// Carrier capability required by [`TracingEngine`](crate::tracing_v2::TracingEngine).
///
/// Traced linearization runs ordinary JVP rules with [`Tracer`] primals. Those rules may stage the
/// primal side of built-in arithmetic through the outer operation carrier, so the carrier must know
/// how to represent the primitive operations that can appear while evaluating those traced primals.
/// Keeping that requirement as one semantic bound prevents every traced-linearization impl from
/// repeating the individual primitive support traits.
#[doc(hidden)]
pub trait TracedLinearizationCarrier<T: Type, V: Traceable<T>>:
    Clone
    + Operation<T>
    + SupportsAdd<T, V>
    + SupportsMul<T, V>
    + SupportsNeg<T, V>
    + SupportsScale<T, V>
    + SupportsMatMul<T, V>
    + SupportsMatrixTranspose<T, V>
    + SupportsReshape<T, V>
    + 'static
{
}

impl<T, V, O> TracedLinearizationCarrier<T, V> for O
where
    T: Type,
    V: Traceable<T>,
    O: Clone
        + Operation<T>
        + SupportsAdd<T, V>
        + SupportsMul<T, V>
        + SupportsNeg<T, V>
        + SupportsScale<T, V>
        + SupportsMatMul<T, V>
        + SupportsMatrixTranspose<T, V>
        + SupportsReshape<T, V>
        + 'static,
{
}

/// Lifts one concrete value into the staged program owned by a JIT tracer.
pub fn lift_jit_constant<'engine, V: Traceable<ArrayType>, E: StagingEngine<Type = ArrayType, Value = V> + ?Sized>(
    constant: &V,
    exemplar: &Tracer<'engine, E>,
) -> Tracer<'engine, E> {
    let builder = exemplar.builder().clone();
    let r#type = constant.r#type().into_owned();
    let atom = builder.borrow_mut().add_constant(constant.clone());
    exemplar.engine.tracer_from_staged_parts(atom, r#type)
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
/// inputs. The transpose context is available for higher-order rules that need to recursively
/// transpose nested programs and synthesize zeros for disconnected leaves.
///
/// A few concrete examples:
///
/// - For [`ScaleOperation`], `y = a * x`, the transpose stages one new scale instruction that
///   computes `a * c`, where `c` is the output cotangent atom.
/// - For [`AddOperation`], `y = x0 + x1`, the transpose returns the same cotangent atom for both
///   inputs.
/// - For [`MatrixTransposeOperation`], `Y = X^T`, the transpose stages another transpose on the
///   output cotangent atom.
/// - For [`ReshapeOperation`], the transpose reshapes the output cotangent back to the input shape because
///   reshape only changes layout metadata.
///
/// Structural validation happens when the forward linear program is built and when any staged ops
/// emitted by the rule are added to the transpose program.
///
/// Concrete state threaded through linear transposition rules.
///
/// [`TranspositionContext`] owns the currently active transpose-program builder. Primitive rules
/// stage new linear instructions through it, and higher-order rules use it to recursively transpose
/// nested linear programs into the same builder. The pass propagates structural zeros via
/// `Option<AtomId>`, so no cotangent-synthesis policy lives on the context anymore.
#[doc(hidden)]
pub struct TranspositionContext<'a, T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> {
    /// Builder for the currently active transpose program.
    builder: Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>>,

    /// Phantom marker reserving a context lifetime for future per-pass borrows without forcing an
    /// API change when one is added.
    marker: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: Type, V: Traceable<T>, LinearCarrier: Clone + Operation<T>> TranspositionContext<'a, T, V, LinearCarrier> {
    /// Creates a transposition context that stages into `builder`.
    #[doc(hidden)]
    pub fn new(builder: Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>>) -> Self {
        Self { builder, marker: std::marker::PhantomData }
    }

    /// Returns the builder for the currently active transpose program.
    #[inline]
    pub fn builder(&self) -> &Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>> {
        &self.builder
    }

    /// Stages one operation in the currently active transpose program.
    ///
    /// `inputs` are atom ids that already live in the transpose builder. Output types are inferred
    /// via [`Operation::infer_output_types`] and the resulting variable atoms are returned in
    /// forward order.
    pub fn apply_operation(
        &self,
        inputs: &[AtomId],
        operation: LinearCarrier,
        output_count: usize,
    ) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        if output_types.len() != output_count {
            return Err(TracingError::InvalidOutputCount { expected: output_count, got: output_types.len() });
        }
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }

    /// Replaces the active transpose-program builder and returns the previous one.
    #[inline]
    pub(crate) fn replace_builder(
        &mut self,
        builder: Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>>,
    ) -> Rc<RefCell<ProgramBuilder<T, V, LinearCarrier>>> {
        std::mem::replace(&mut self.builder, builder)
    }

    /// Takes ownership of the active builder, leaving an empty builder behind.
    pub(crate) fn take_builder(&mut self) -> Result<ProgramBuilder<T, V, LinearCarrier>, TracingError> {
        let builder = self.replace_builder(Rc::new(RefCell::new(ProgramBuilder::new())));
        match Rc::try_unwrap(builder) {
            Ok(builder) => Ok(builder.into_inner()),
            Err(_) => Err(TracingError::EscapedProgramBuilder),
        }
    }
}

pub trait LinearOperation<T: Type, V: Traceable<T>, LinearCarrier: Clone = primitive::LinearPrimitiveOperation<V>>:
    Operation<T>
{
    /// Applies the transpose rule for reverse-mode differentiation.
    ///
    /// `output_cotangents` is aligned with the op outputs in forward order. Each entry is
    /// `Some(atom)` when the corresponding output has an accumulated symbolic cotangent atom in
    /// the active transpose builder and `None` when its cotangent is structurally zero. The
    /// returned vector must be aligned with the op inputs in forward order.
    ///
    /// Returning `Some(atom)` means that input receives the staged cotangent contribution `atom`.
    /// Returning `None` means the contribution is structurally zero and the transpose pass does
    /// not need to materialize an explicit zero atom for that input. Rules stage new linear ops in
    /// the transpose builder via [`TranspositionContext::apply_operation`].
    fn transpose(
        &self,
        context: &mut TranspositionContext<'_, T, V, LinearCarrier>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError>
    where
        LinearCarrier: Operation<T>;
}

/// Forward-mode differentiation rule keyed only by the engine that owns the staged carriers.
///
/// Primitive JVP rules consume `JvpTracer<E::Value, AtomId>` inputs — primal value plus tangent
/// atom id in the active linear-program builder — and stage tangent ops via
/// [`JvpContext::apply_operation`]. Higher-order rules (e.g., the rematerialization and
/// control-flow ops) use `engine` to recurse into nested sub-programs.
pub trait DifferentiableOperation<E: DifferentiableEngine + ?Sized>: Operation<E::Type> {
    /// Applies the forward-mode JVP rule.
    fn jvp(
        &self,
        engine: &E,
        context: &mut JvpContext<'_, E::Value, E::LinearOperation, E::Type>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError>;
}
