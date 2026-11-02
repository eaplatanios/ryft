use crate::programs::{Operation, TypeError, Value};

/// Operation-family capability for representing a call to a staged program. The call operation carries only metadata.
/// Its callee is a flat program, whose captures have been lifted into leading inputs, composed into the region driver
/// passed to [`Context::bind`](crate::Context::bind), which interns it as a shared callee root region by
/// [`Arc`](std::sync::Arc) identity. The concrete operation family decides how that boundary lowers and how batching,
/// differentiation, partial evaluation, and other transforms rewrite it. This keeps higher-order call semantics with
/// the operation that owns them while the lifecycle and capture plumbing remain backend-neutral.
pub trait StagedCallOperation<Constant: Value>: Operation<Type = Constant::Type> + Sized {
    /// Constructs a call operation. The accompanying [`Context::bind`](crate::Context::bind) supplies its callee
    /// through a shared-callee region driver, while `capture_count` records the exact leading lifted-capture prefix
    /// of that callee boundary.
    fn staged_call(capture_count: usize) -> Self;

    /// Constructs one call with its exact callee signature. Families that export callee-local nominal identities
    /// override this hook to allocate instruction-owned result identities while forwarding input identities.
    /// Allocation happens when the call is built, and output inference must remain deterministic. The default returns
    /// [`Self::staged_call`], leaving result inference to the family. A family whose inference reports the callee's
    /// result types then makes each call instruction define the callee-local identities itself. Program construction
    /// rejects a second call to the same callee in one region, because it would define those identities again although
    /// sharing a callee root does not make the runtime results of separate calls the same quantity. Override this
    /// function to instantiate fresh result identities for every call instead.
    #[inline]
    fn staged_call_with_types(
        capture_count: usize,
        input_types: &[Constant::Type],
        output_types: &[Constant::Type],
    ) -> Result<Self, TypeError> {
        let _ = input_types;
        let _ = output_types;
        Ok(Self::staged_call(capture_count))
    }
}
