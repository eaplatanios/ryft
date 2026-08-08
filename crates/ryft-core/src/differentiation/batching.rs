use crate::axes::Axis;
use crate::batching::{BatchingError, BatchingPolicy};
use crate::contexts::Context;
use crate::differentiation::types::DifferentiableType;
use crate::tracing::{Tracer, TracingContext};

/// [`BatchingPolicy`] that collapses a mapped cotangent, which is the one capability that batching an operation with
/// an attached *backward* [`Region`](crate::Region) needs beyond ordinary batching. Batching such an operation may
/// replicate one of its linear inputs across the mapped axis. If the batched backward program subsequently produces one
/// cotangent `ūᵢ` for each batch item, the transpose of that replication is summation, so the single cotangent for the
/// original replicated input is:
///
/// ```text
/// ū = Σᵢ ūᵢ.
/// ```
///
/// The capability is shared rather than being operation-specific. [`LinearCallOperation`](crate::LinearCallOperation),
/// [`CustomVjpOperation`](crate::CustomVjpOperation), and [`RematerializeOperation`](crate::RematerializeOperation)
/// each pass [`Self::sum_mapped_cotangents`] to [`BatchingPolicy::adapt_batched_program`] while adapting their batched
/// backward region back to its plain boundary. Operations whose attached regions are all forward-shaped (e.g.,
/// [`CustomJvpOperation`](crate::CustomJvpOperation)) require plain [`BatchingPolicy`] instead.
///
/// Those batching rules own every universe-independent part of the transformation, structurally batching the attached
/// regions, aligning their boundaries, threading policy-owned bookkeeping values, and rebuilding the call. The
/// representation of `ūᵢ` is the one step they cannot determine generically. An ordinary array policy reduces the
/// cotangent directly along its mapped axis, while a composite policy may first need to project the cotangent to its
/// differentiable member, perform that member's reduction, and lift the result back. This capability supplies exactly
/// that representation-dependent step and lets each generic [`BatchableOperation`](crate::BatchableOperation)
/// implementation retain its complete algorithm.
///
/// Implement this trait for a [`BatchingPolicy`] only when its program universe supports batching operations with
/// backward regions. An implementation must return a value owned by `context`, of the same program type as `cotangent`,
/// with `axis` removed and all batch-item cotangents combined by addition. Policies that do not support them should
/// omit the implementation. This is deliberately an opt-in capability rather than a method on [`BatchingPolicy`],
/// because ordinary batching policies need not provide differentiation semantics; other operation families must not
/// acquire parallel policy traits unless they expose an independently irreducible universe-specific step of their own.
///
/// # Parameters
///
///   - `context`: [`TracingContext`] that owns the structurally batched backward program being adapted.
///   - `cotangent`: Mapped cotangent produced by that program.
///   - `axis`: Physical axis containing the packed family of per-item cotangents.
pub trait CotangentBatchingPolicy<C: Context<Type: DifferentiableType>>: BatchingPolicy<C> {
    /// Sums the per-item cotangents packed along `axis`.
    fn sum_mapped_cotangents(
        context: &TracingContext<C::Constant, C::Operation>,
        cotangent: Tracer<TracingContext<C::Constant, C::Operation>>,
        axis: Axis,
    ) -> Result<Tracer<TracingContext<C::Constant, C::Operation>>, BatchingError>;
}
