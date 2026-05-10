use crate::operations::Operation;
use crate::parameters::Parameter;
use crate::tracing::domains::{ProgramTracer, ProgramTracingContext};
use crate::tracing::{Traceable, TracingError};
use crate::types::Type;

/// Operation-level contract for staged linear maps that can be transposed.
///
/// A [`LinearOperation`] is the capability an operation carrier provides after a primal program has
/// been linearized. Implementors describe how one staged linear instruction contributes to the
/// reverse linear program used by VJP and reverse-mode gradient transforms. The trait is
/// implemented by primitive operation types, such as [`AddOperation`](crate::AddOperation), and by carrier enums,
/// such as [`LinearArrayOperation`](crate::tracing_v2::LinearArrayOperation), that delegate to primitive rules.
///
/// For a linear instruction `y = L(x)`, [`transpose`](Self::transpose) receives symbolic cotangent
/// [`Tracer`]s for `y` and returns symbolic cotangent contributions for `x`. Rules may reuse
/// existing cotangents, return `None` for structural zeros, or stage additional linear operations
/// in the active [`ProgramTracingContext`]. The rule does not receive concrete primal values; any
/// required metadata must be encoded in the operation itself or in staged atom types.
///
/// Structural validation happens when the linear program is built and when transpose rules stage
/// additional operations in the transpose builder.
pub trait LinearOperation<T: Type + Parameter, V: Traceable<T>, O: Operation<T>>: Operation<T> {
    /// Applies this operation's transpose rule to symbolic output cotangents.
    ///
    /// The returned vector must contain one entry per operation input. Each `Some(cotangent)` is a
    /// staged cotangent contribution in the active transpose builder, and each `None` means the
    /// corresponding input receives a structural zero from this operation.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active transpose context used to stage any new linear operations required by
    ///     the rule.
    ///   - `output_cotangents`: Cotangent tracers aligned with this operation's outputs. `None`
    ///     entries represent structural zeros.
    fn transpose<'transpose>(
        &self,
        context: &mut ProgramTracingContext<'transpose, T, V, O>,
        output_cotangents: &[Option<ProgramTracer<'transpose, T, V, O>>],
    ) -> Result<Vec<Option<ProgramTracer<'transpose, T, V, O>>>, TracingError>;
}
