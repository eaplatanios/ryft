use std::cell::RefCell;
use std::rc::Rc;

use crate::tracing::{AtomId, Instruction, InterpretableOperation, Operation, ProgramBuilder, Traceable, TracingError};
use crate::tracing_v2::DifferentiableEngine;
use crate::tracing_v2::forward::{JvpContext, JvpTracer};
use crate::types::{Type, Typed};

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
pub use constants::{
    OneLikeOperation, OneOperation, SupportsOne, SupportsOneLike, SupportsZero, SupportsZeroLike, ZeroLikeOperation,
    ZeroOperation,
};
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
pub use primitive::{ArrayOperation, LinearArrayOperation, LinearScalarOperation, ScalarOperation};
pub use rematerialize::{
    FlatTracedRematerialize, LinearRematerializeOperation, RematerializeOperation, SupportsLinearRematerialize,
    SupportsRematerialize,
};
pub use reshape::{ReshapeOperation, SupportsReshape};
pub use right_matmul::{RightMatMulOperation, SupportsRightMatMul};
pub use scale::{ScaleOperation, SupportsScale};
pub use sin::{Sin, SinOperation, SupportsSin};

/// State threaded through [`LinearOperation::transpose`] while building a reverse linear program.
///
/// [`TranspositionContext`] owns the active [`ProgramBuilder`] for the transposed program. Rules
/// use [`stage`](Self::stage) to append operations whose inputs are [`AtomId`]s already present in
/// that builder, and higher-order rules may temporarily replace [`builder`](Self::builder) while
/// transposing nested linear programs.
///
/// The context deliberately carries only builder state. Transpose rules operate on already-linear
/// operations and abstract atom metadata, and structural zeros are represented by `Option<AtomId>`
/// in [`LinearOperation::transpose`].
pub struct TranspositionContext<T: Type, V: Traceable<T>, O: Clone + Operation<T>> {
    /// [`ProgramBuilder`] that owns the reverse linear [`Program`](crate::tracing::Program) currently being staged.
    pub builder: Rc<RefCell<ProgramBuilder<T, V, O>>>,
}

impl<T: Type, V: Traceable<T>, O: Clone + Operation<T>> TranspositionContext<T, V, O> {
    /// Creates a new [`TranspositionContext`] that stages into the provided [`ProgramBuilder`].
    ///
    /// # Parameters
    ///
    ///   - `builder`: Shared builder that will own the staged reverse linear program.
    pub fn new(builder: Rc<RefCell<ProgramBuilder<T, V, O>>>) -> Self {
        Self { builder }
    }

    /// Stages `operation` in the active transpose builder and returns its output atoms.
    ///
    /// Output types are inferred with [`Operation::infer_output_types`] from the current types of
    /// `inputs`. New variable atoms are allocated before the instruction is recorded, and the
    /// returned atom ids are ordered like the operation outputs.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Operation to append to the active transpose builder.
    ///   - `inputs`: Atom ids in the active transpose builder that feed `operation`.
    pub fn stage(&self, operation: O, inputs: &[AtomId]) -> Result<Vec<AtomId>, TracingError> {
        let mut builder_borrow = self.builder.borrow_mut();
        let input_types =
            inputs.iter().map(|atom| builder_borrow.atoms[atom.index].r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(&input_types)?;
        let outputs = output_types.into_iter().map(|r#type| builder_borrow.add_variable(r#type)).collect::<Vec<_>>();
        builder_borrow
            .instructions
            .push(Instruction { operation, inputs: inputs.to_vec(), outputs: outputs.clone() });
        Ok(outputs)
    }
}

/// Operation-level contract for staged linear maps that can be transposed.
///
/// A [`LinearOperation`] is the capability an operation carrier provides after a primal program has
/// been linearized. Implementors describe how one staged linear instruction contributes to the
/// reverse linear program used by VJP and reverse-mode gradient transforms. The trait is
/// implemented by primitive operation types, such as [`AddOperation`], and by carrier enums, such
/// as [`LinearArrayOperation`], that delegate to primitive rules.
///
/// For a linear instruction `y = L(x)`, [`transpose`](Self::transpose) receives symbolic cotangent
/// atoms for `y` and returns symbolic cotangent contributions for `x`. Rules may reuse existing
/// cotangent atoms, return `None` for structural zeros, or stage additional linear operations in
/// the active [`TranspositionContext`]. The rule does not receive concrete primal values; any
/// required metadata must be encoded in the operation itself or in staged atom types.
///
/// Structural validation happens when the linear program is built and when transpose rules stage
/// additional operations in the transpose builder.
pub trait LinearOperation<T: Type, V: Traceable<T>, O: Clone + Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to symbolic output cotangents.
    ///
    /// The returned vector must contain one entry per operation input. Each `Some(atom)` is a
    /// staged cotangent contribution in the active transpose builder, and each `None` means the
    /// corresponding input receives a structural zero from this operation.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active transpose context used to stage any new linear operations required by
    ///     the rule.
    ///   - `output_cotangents`: Cotangent atoms aligned with this operation's outputs. `None`
    ///     entries represent structural zeros.
    fn transpose(
        &self,
        context: &mut TranspositionContext<T, V, O>,
        output_cotangents: &[Option<AtomId>],
    ) -> Result<Vec<Option<AtomId>>, TracingError>;
}

/// Operation-level contract for forward-mode Jacobian-Vector Product (JVP) staging.
///
/// A [`DifferentiableOperation`] is keyed by the [`DifferentiableEngine`] that supplies the value,
/// type, and linear-operation families used while differentiating. Implementors consume
/// [`JvpTracer`] inputs, each carrying a primal value and a tangent atom in the active linear
/// builder, and return traced primal/tangent outputs.
///
/// Primitive rules usually stage tangent operations through [`JvpContext::apply_operation`].
/// Higher-order rules use [`JvpContext::engine`] to recurse into nested programs with the same
/// engine.
pub trait DifferentiableOperation<E: DifferentiableEngine + ?Sized>: Operation<E::Type> {
    /// Applies this operation's forward-mode Jacobian-Vector Product (JVP) rule.
    ///
    /// The returned vector must be aligned with this operation's outputs and must carry both the
    /// primal output values and the staged tangent atoms for those outputs.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active JVP context used to stage tangent operations and access the
    ///     differentiable engine.
    ///   - `inputs`: Traced inputs aligned with this operation's inputs.
    fn jvp(
        &self,
        context: &mut JvpContext<'_, E>,
        inputs: &[JvpTracer<E::Value, AtomId>],
    ) -> Result<Vec<JvpTracer<E::Value, AtomId>>, TracingError>;
}
