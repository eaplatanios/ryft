use std::cell::RefCell;
use std::rc::Rc;

use crate::operations::Operation;
use crate::tracing::{AtomId, Instruction, ProgramBuilder, Traceable, TracingError};
use crate::types::{Type, Typed};

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
/// implemented by primitive operation types, such as
/// [`AddOperation`](crate::tracing_v2::operations::AddOperation), and by carrier enums, such as
/// [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation), that delegate to primitive
/// rules.
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
