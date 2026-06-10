//! Gradient rematerialization / rematerialization — the analogue of JAX's
//! [`jax.checkpoint` / `jax.remat`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html).
//!
//! [`rematerialize`] wraps a function so that reverse-mode differentiation through it trades memory for compute:
//! instead of storing every linearization residual produced inside the wrapped region, only the region's inputs
//! (plus any values selected by a [`RematerializationPolicy`]) are saved, and everything else is recomputed from them in
//! the backward pass.
//!
//! Rematerializeing is not a primitive operation in `ryft`. Each [`Rematerialize::call`] expands into a
//! [`CustomVjpOperation`] — the same reduction JAX documents for `jax.checkpoint` — by deriving the forward and
//! backward programs symbolically: the forward program computes the region outputs plus the saved values, and the
//! backward program recomputes the remaining linearization residuals from the saved values before replaying the
//! transposed tangent map. All downstream behavior (interpretation, batching, lowering, the reverse-mode rule)
//! therefore reuses the custom-derivative machinery unchanged.
//!
//! This module also owns the [`rematerialization_name`](RematerializationName) value-tagging primitive
//! ([`RematerializationNameOperation`]) consumed by the name-based [`RematerializationPolicy`] members.

use std::cell::RefCell;
use std::fmt::{Debug, Display};
use std::marker::PhantomData;
use std::rc::Rc;

use half::{bf16, f16};

use crate::contexts::StagingContext;
use crate::differentiation::SupportsTransposition;
use crate::domains::Domain;
use crate::macros::check_count;
use crate::operations::constants::{SupportsOne, SupportsZero};
use crate::operations::{ElementwiseOperation, InterpretableOperation, Operation};
use crate::parameters::{Parameterized, ParameterizedFamily};
use crate::programs::ProgramError;
use crate::tracing::{DomainTracer, Tracer, TracingContext};
use crate::tracing_v2::differentiation::{
    DifferentiableOperation, DifferentiationContext, DirectLinearOperationOf, JvpTracer, LinearOperationOf,
    ResidualizedOperation, TangentContext,
};
use crate::tracing_v2::operations::custom_derivatives::{CustomVjpOperation, SupportsCustomVjp};
use crate::tracing_v2::operations::dot::{DotDimensionNumbers, MaybeDot};
use crate::types::{ArrayType, DataType, Type, TypeError, Typed};

/// Canonical operation name for [`RematerializationNameOperation`].
pub const REMATERIALIZATION_NAME_OPERATION_NAME: &'static str = "rematerialization_name";

/// [`Operation`] that returns its input unchanged while tagging it with a name visible to rematerialization policies — the
/// direct analogue of JAX's
/// [`jax.ad_checkpoint.checkpoint_name`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-offloadable-and-saveable-names).
///
/// Interpretation, batching, and backend lowering all treat this operation as the identity, and differentiation
/// passes the tangent through unchanged while re-tagging the primal — so the tag is visible on the instructions that
/// define linearization residuals, which is exactly what the name-based members of
/// [`RematerializationPolicy`] classify.
#[derive(Clone, Debug)]
pub struct RematerializationNameOperation {
    /// Name tagging the operation's output value.
    name: String,
}

impl RematerializationNameOperation {
    /// Creates a new [`RematerializationNameOperation`] with the provided tag name.
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }

    /// Returns the tag name carried by this operation.
    #[inline]
    pub fn tag(&self) -> &str {
        self.name.as_str()
    }
}

impl Display for RematerializationNameOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{REMATERIALIZATION_NAME_OPERATION_NAME}[{}]", self.name)
    }
}

impl Operation<DataType> for RematerializationNameOperation {
    #[inline]
    fn name(&self) -> &'static str {
        REMATERIALIZATION_NAME_OPERATION_NAME
    }

    #[inline]
    fn infer_output_types(&self, input_types: &[DataType]) -> Result<Vec<DataType>, TypeError> {
        check_count!("input", input_types, 1, TypeError);
        Ok(vec![input_types[0].clone()])
    }
}

impl ElementwiseOperation for RematerializationNameOperation {
    #[inline]
    fn name(&self) -> &'static str {
        REMATERIALIZATION_NAME_OPERATION_NAME
    }

    #[inline]
    fn input_count(&self) -> usize {
        1
    }
}

impl<V: Clone + Typed<DataType>> InterpretableOperation<DataType, V> for RematerializationNameOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

impl<V: Clone + Typed<ArrayType>> InterpretableOperation<ArrayType, V> for RematerializationNameOperation {
    #[inline]
    fn interpret(&self, inputs: &[V]) -> Result<Vec<V>, ProgramError> {
        check_count!("input", inputs, 1, ProgramError);
        Ok(vec![inputs[0].clone()])
    }
}

/// Trait that represents [`Operation`] types that support/include [`RematerializationNameOperation`]. Backend-owned closed
/// [`Operation`] types implement this trait so that generic transform code can stage [`RematerializationNameOperation`]s
/// without knowing which operation type is in use.
pub trait SupportsRematerializationName<T: Type> {
    /// Constructs an instance of [`RematerializationNameOperation`] for this [`Operation`] type.
    fn rematerialization_name_operation(name: String) -> Self;
}

/// Query trait classifying operations as rematerialization-name tags. Backend-owned closed operation enums implement this
/// trait so that the name-based members of [`RematerializationPolicy`] can
/// classify staged instructions without knowing the concrete operation enum.
pub trait MaybeRematerializationName {
    /// Returns the tag name when this operation is a [`RematerializationNameOperation`], and [`None`] otherwise.
    fn rematerialization_name(&self) -> Option<&str>;
}

/// Value-level rematerialization-name tagging. [`RematerializationName`] is the identity on concrete values, while on traced
/// values it stages a [`RematerializationNameOperation`] so the tag is visible to rematerialization policies.
pub trait RematerializationName: Sized {
    /// Returns this value unchanged while tagging it with `name` for rematerialization policies.
    fn rematerialization_name(self, name: &str) -> Self;
}

macro_rules! impl_rematerialization_name_identity {
    ($($ty:ty),* $(,)?) => {
        $(
            impl RematerializationName for $ty {
                #[inline]
                fn rematerialization_name(self, _name: &str) -> Self {
                    self
                }
            }
        )*
    };
}

impl_rematerialization_name_identity!(bf16, f16, f32, f64);

impl<C: StagingContext<Operation: SupportsRematerializationName<C::Type>>> RematerializationName for Tracer<C> {
    #[inline]
    fn rematerialization_name(self, name: &str) -> Self {
        self.unary(C::Operation::rematerialization_name_operation(name.to_string()))
    }
}

/// JVP rule for [`RematerializationNameOperation`]: the tangent passes through unchanged and the primal is re-tagged via
/// [`RematerializationName`], so the tag stays visible on the instructions that define linearization residuals (which is
/// what the name-based rematerialization policies classify). The rule stages no linear operation, so `rematerialization_name`
/// never appears in a pushforward program and needs no transpose rule.
impl<D: DifferentiationContext> DifferentiableOperation<D> for RematerializationNameOperation
where
    RematerializationNameOperation: Operation<D::Type>,
    D::Value: RematerializationName,
{
    #[inline]
    fn jvp<'jvp>(
        &self,
        _context: &mut TangentContext<'jvp, D>,
        inputs: &[JvpTracer<'jvp, D>],
    ) -> Result<Vec<JvpTracer<'jvp, D>>, ProgramError>
    where
        D: 'jvp,
    {
        check_count!("input", inputs, 1, ProgramError);
        let primal = inputs[0].primal().clone().rematerialization_name(self.tag());
        Ok(vec![JvpTracer::new(primal, inputs[0].tangent().clone())])
    }
}

/// Classification facts about one linearization residual, exposed to [`RematerializationPolicy::Custom`] policies.
///
/// A candidate describes the instruction that produced the residual — its operation name, dot classification, and
/// [`rematerialization_name`](RematerializationName) tag — together with the residual's staged type, mirroring the
/// information JAX exposes to custom `jax.checkpoint` policies (the primitive and its abstract values).
#[derive(Clone, Debug)]
pub struct RematerializationCandidate<'a, T: Type> {
    /// Name of the operation that produced the residual.
    operation_name: &'a str,

    /// Dimension numbers of the producing operation when it is a dot-like contraction.
    dot_dimensions: Option<&'a DotDimensionNumbers>,

    /// [`rematerialization_name`](RematerializationName) tag on the producing operation, if any.
    rematerialization_name: Option<&'a str>,

    /// Staged type of the residual value.
    residual_type: T,
}

impl<'a, T: Type> RematerializationCandidate<'a, T> {
    /// Returns the name of the operation that produced the residual.
    #[inline]
    pub fn operation_name(&self) -> &'a str {
        self.operation_name
    }

    /// Returns whether the producing operation is a dot-like contraction.
    #[inline]
    pub fn is_dot(&self) -> bool {
        self.dot_dimensions.is_some()
    }

    /// Returns the dimension numbers of the producing operation when it is a dot-like contraction.
    #[inline]
    pub fn dot_dimensions(&self) -> Option<&'a DotDimensionNumbers> {
        self.dot_dimensions
    }

    /// Returns the [`rematerialization_name`](RematerializationName) tag on the producing operation, if any.
    #[inline]
    pub fn rematerialization_name(&self) -> Option<&'a str> {
        self.rematerialization_name
    }

    /// Returns the staged type of the residual value.
    #[inline]
    pub fn residual_type(&self) -> &T {
        &self.residual_type
    }
}

/// Policy selecting which linearization residuals a [`Rematerialize`] saves instead of recomputing — the analogue of
/// the named members of JAX's
/// [`jax.checkpoint_policies`](https://docs.jax.dev/en/latest/gradient-checkpointing.html#custom-policies-for-what-s-saveable).
///
/// A residual is a value captured during linearization as a coefficient of the staged linear (tangent) map — for
/// example, `cos(x)` for `sin`, or the operand values for `mul`. Saved residuals are emitted as extra outputs of the
/// rematerialization's forward program and consumed directly by its backward program; unsaved residuals are recomputed in
/// the backward program from the saved values. Residuals that are region inputs or constants are never stored: the
/// backward program always receives the region inputs, and constants are re-created in place.
///
/// The name-based members classify residuals by [`rematerialization_name`](RematerializationName)
/// tags applied inside the body, [`Custom`](Self::Custom) policies classify them through a
/// [`RematerializationCandidate`], and [`SaveFromBothPolicies`](Self::SaveFromBothPolicies) combines two policies.
/// Offloading policies (JAX's `save_and_offload_only_these_names` / `offloadable`) require a memory-space model
/// (host/device placement for residuals) that `ryft` does not have yet, and remain future work.
#[derive(Clone, Default)]
pub enum RematerializationPolicy<T: Type = ArrayType> {
    /// Save nothing beyond the region inputs; recompute every residual in the backward pass. This is the default,
    /// matching JAX's `nothing_saveable` (the default policy of `jax.checkpoint`).
    #[default]
    NothingSaveable,

    /// Save every instruction-produced residual, making the rematerialization inert: the backward pass recomputes nothing.
    /// Matches JAX's `everything_saveable`.
    EverythingSaveable,

    /// Save residuals produced by dot-like contractions (classified via [`MaybeDot`]) and recompute the rest.
    /// Matches JAX's `dots_saveable`.
    DotsSaveable,

    /// Save residuals produced by dot-like contractions whose [`DotDimensionNumbers`] have no batch dimensions and
    /// recompute the rest. Batched contractions behave more like cheap elementwise work per batch element, so
    /// saving only the unbatched ones targets the genuinely expensive matrix products. Matches JAX's
    /// `dots_with_no_batch_dims_saveable`.
    DotsWithNoBatchDimsSaveable,

    /// Save only residuals tagged with one of the provided
    /// [`rematerialization_name`](RematerializationName) names and recompute everything else.
    /// Matches JAX's `save_only_these_names`.
    SaveOnlyTheseNames(Vec<String>),

    /// Save every *named* residual except those tagged with one of the provided names; unnamed residuals are
    /// recomputed. Matches JAX's `save_any_names_but_these`.
    SaveAnyNamesButThese(Vec<String>),

    /// Save every instruction-produced residual except those tagged with one of the provided names. Matches JAX's
    /// `save_anything_except_these_names`.
    SaveAnythingExceptTheseNames(Vec<String>),

    /// Save every residual that either of the two combined policies saves. Matches JAX's
    /// `save_from_both_policies`.
    SaveFromBothPolicies(Box<RematerializationPolicy<T>>, Box<RematerializationPolicy<T>>),

    /// Save exactly the residuals for which the provided callable returns `true`, classifying each through a
    /// [`RematerializationCandidate`]. This is the analogue of passing an arbitrary callable as a JAX
    /// `jax.checkpoint` policy. The callable is shared by reference counting, so cloned policies (and the wrappers
    /// holding them) observe the same classification function; policies never travel inside staged programs.
    Custom(Rc<dyn Fn(&RematerializationCandidate<'_, T>) -> bool>),
}

impl<T: Type> Debug for RematerializationPolicy<T> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NothingSaveable => formatter.write_str("NothingSaveable"),
            Self::EverythingSaveable => formatter.write_str("EverythingSaveable"),
            Self::DotsSaveable => formatter.write_str("DotsSaveable"),
            Self::DotsWithNoBatchDimsSaveable => formatter.write_str("DotsWithNoBatchDimsSaveable"),
            Self::SaveOnlyTheseNames(names) => formatter.debug_tuple("SaveOnlyTheseNames").field(names).finish(),
            Self::SaveAnyNamesButThese(names) => formatter.debug_tuple("SaveAnyNamesButThese").field(names).finish(),
            Self::SaveAnythingExceptTheseNames(names) => {
                formatter.debug_tuple("SaveAnythingExceptTheseNames").field(names).finish()
            }
            Self::SaveFromBothPolicies(first, second) => {
                formatter.debug_tuple("SaveFromBothPolicies").field(first).field(second).finish()
            }
            Self::Custom(_) => formatter.write_str("Custom(..)"),
        }
    }
}

/// [`Custom`](RematerializationPolicy::Custom) policies compare by callable identity ([`Rc::ptr_eq`]): two custom
/// policies are equal exactly when they share the same classification function. Every other variant compares
/// structurally.
impl<T: Type> PartialEq for RematerializationPolicy<T> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::NothingSaveable, Self::NothingSaveable)
            | (Self::EverythingSaveable, Self::EverythingSaveable)
            | (Self::DotsSaveable, Self::DotsSaveable)
            | (Self::DotsWithNoBatchDimsSaveable, Self::DotsWithNoBatchDimsSaveable) => true,
            (Self::SaveOnlyTheseNames(left), Self::SaveOnlyTheseNames(right))
            | (Self::SaveAnyNamesButThese(left), Self::SaveAnyNamesButThese(right))
            | (Self::SaveAnythingExceptTheseNames(left), Self::SaveAnythingExceptTheseNames(right)) => left == right,
            (
                Self::SaveFromBothPolicies(left_first, left_second),
                Self::SaveFromBothPolicies(right_first, right_second),
            ) => left_first == right_first && left_second == right_second,
            (Self::Custom(left), Self::Custom(right)) => Rc::ptr_eq(left, right),
            _ => false,
        }
    }
}

impl<T: Type> Eq for RematerializationPolicy<T> {}

impl<T: Type> RematerializationPolicy<T> {
    /// Returns whether the residual `value` should be saved by the rematerialization's forward program. Residuals that are
    /// not produced by an instruction (region inputs and constants) are never saved; see the type-level
    /// documentation.
    fn saves_residual<'d, D>(&self, residual: &DomainTracer<'d, D>) -> Result<bool, ProgramError>
    where
        D: Domain<Type = T, Operation: MaybeDot + MaybeRematerializationName> + 'd,
    {
        if matches!(self, Self::NothingSaveable) {
            return Ok(false);
        }
        let atom_id = residual.atom_id()?;
        let context = residual.context();
        let builder = context.builder().borrow();
        let Some(instruction) =
            builder.instructions.iter().rev().find(|instruction| instruction.outputs().contains(&atom_id))
        else {
            return Ok(false);
        };
        let operation = instruction.operation();
        Ok(match self {
            Self::NothingSaveable => false,
            Self::EverythingSaveable => true,
            Self::DotsSaveable => operation.is_dot(),
            Self::DotsWithNoBatchDimsSaveable => operation.dot_dimensions().is_some_and(|dimensions| {
                dimensions.lhs_batching_dimensions().is_empty() && dimensions.rhs_batching_dimensions().is_empty()
            }),
            Self::SaveOnlyTheseNames(names) => {
                operation.rematerialization_name().is_some_and(|name| names.iter().any(|n| n == name))
            }
            Self::SaveAnyNamesButThese(names) => {
                operation.rematerialization_name().is_some_and(|name| !names.iter().any(|n| n == name))
            }
            Self::SaveAnythingExceptTheseNames(names) => {
                !operation.rematerialization_name().is_some_and(|name| names.iter().any(|n| n == name))
            }
            Self::SaveFromBothPolicies(first, second) => {
                first.saves_residual(residual)? || second.saves_residual(residual)?
            }
            Self::Custom(policy) => policy(&RematerializationCandidate {
                operation_name: operation.name(),
                dot_dimensions: operation.dot_dimensions(),
                rematerialization_name: operation.rematerialization_name(),
                residual_type: residual.r#type().into_owned(),
            }),
        })
    }
}

/// Function whose reverse-mode differentiation rematerializes interior values instead of storing them — the
/// ergonomic analogue of JAX's [`jax.checkpoint`](https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html),
/// built by [`rematerialize`].
///
/// The wrapped body is stored as a plain closure over [`DomainTracer`]s and nothing is derived at construction
/// time: each [`call`](Self::call) reads the input types off its tracer arguments, traces the body, derives the
/// forward and backward programs symbolically (specialized to those types and to the configured
/// [`RematerializationPolicy`]), and stages one [`CustomVjpOperation`]. The derived forward program returns the body
/// outputs followed by the region inputs and the policy-saved residual values; the derived backward program
/// recomputes the remaining residuals from those saved values and replays the transposed tangent map. Reverse-mode
/// differentiation through the staged call therefore stores exactly the saved values — nothing interior — and both
/// derived programs are pruned of unreachable instructions, so saved residuals are genuinely not recomputed.
///
/// Unlike user-authored custom VJPs, the expansion also carries a derived *tangent program*, so forward-mode
/// differentiation works through rematerialized calls — matching `jax.checkpoint`, which supports `jvp`.
/// Un-differentiated calls replay the lean primal program and pay for neither residual computation nor saving.
///
/// Each [`call`](Self::call) caches its derivation inside the wrapper keyed by the flat input types — the analogue
/// of JAX caching traced rules on `(function, avals)` — so repeated calls with equal input types stage the
/// previously derived operation without re-tracing anything. Nothing invalidates the cache: the wrapper owns both
/// the body closure and the policy, and both are immutable after construction ([`with_policy`](Self::with_policy)
/// and [`with_prevent_cse`](Self::with_prevent_cse) consume `self`).
pub struct Rematerialize<'d, D: Domain, B, IT, OT>
where
    D::Type: PartialEq,
    OT: Parameterized<DomainTracer<'d, D>>,
{
    /// Domain whose constant and operation types the derived programs are traced over.
    domain: &'d D,

    /// Closure computing the region output tree from the region input tree.
    body: B,

    /// Policy selecting which linearization residuals are saved instead of recomputed.
    policy: RematerializationPolicy<D::Type>,

    /// Whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier;
    /// see [`Self::with_prevent_cse`].
    prevent_cse: bool,

    /// Derivations already produced by [`call`](Self::call), keyed by the flat input types they were specialized
    /// to. Entries hold the staged operation together with the body's output-tree structure. The handful of
    /// distinct input signatures a wrapper sees makes a linear scan cheaper and simpler than hashing types.
    cache: RefCell<Vec<CachedDerivation<'d, D, OT>>>,

    /// Phantom marker pinning the input and output tracer-tree types named by the body's signature.
    marker: PhantomData<fn() -> (IT, OT)>,
}

/// One [`Rematerialize`] cache entry: the flat input types a derivation was specialized to, the derived
/// custom-VJP operation, and the structure of the body's output tree.
type CachedDerivation<'d, D, OT> = (
    Vec<<D as Domain>::Type>,
    CustomVjpOperation<<D as Domain>::Constant, <D as Domain>::Operation, <D as Domain>::Type>,
    <OT as Parameterized<DomainTracer<'d, D>>>::ParameterStructure,
);

/// Creates a [`Rematerialize`] function from a body closure over `domain`'s tracers, with the default
/// [`RematerializationPolicy::NothingSaveable`] policy. Use [`Rematerialize::with_policy`] to select a different policy.
/// Refer to the documentation of [`Rematerialize`] for the derivation and rematerialization semantics.
pub fn rematerialize<'d, D, B, IT, OT>(domain: &'d D, body: B) -> Rematerialize<'d, D, B, IT, OT>
where
    D: Domain<Type: PartialEq>,
    B: Fn(IT) -> Result<OT, ProgramError>,
    OT: Parameterized<DomainTracer<'d, D>>,
{
    Rematerialize {
        domain,
        body,
        policy: RematerializationPolicy::NothingSaveable,
        prevent_cse: true,
        cache: RefCell::new(Vec::new()),
        marker: PhantomData,
    }
}

impl<'d, D, B, IT, OT> Rematerialize<'d, D, B, IT, OT>
where
    D: Domain<Type: PartialEq>,
    OT: Parameterized<DomainTracer<'d, D>>,
{
    /// Replaces this rematerialization's [`RematerializationPolicy`].
    #[inline]
    pub fn with_policy(mut self, policy: RematerializationPolicy<D::Type>) -> Self {
        self.policy = policy;
        self
    }

    /// Sets whether backends should wrap the lowered backward/tangent program outputs in an optimization barrier
    /// (e.g., StableHLO's `optimization_barrier`), preventing the compiler from common-subexpression-eliminating
    /// the recomputed values against the forward pass — which would silently restore the memory cost the
    /// rematerialization was meant to avoid. Enabled by default, mirroring `jax.checkpoint`'s `prevent_cse=True`;
    /// disable it when the rematerialized region is staged somewhere CSE cannot reach (for example, under
    /// `jax.checkpoint`'s documented `scan` carve-out) and the barrier would inhibit useful optimizations.
    #[inline]
    pub fn with_prevent_cse(mut self, prevent_cse: bool) -> Self {
        self.prevent_cse = prevent_cse;
        self
    }
}

impl<'d, D, B, IT, OT> Rematerialize<'d, D, B, IT, OT>
where
    D: DifferentiationContext<Type: PartialEq> + 'd,
    B: Fn(IT) -> Result<OT, ProgramError>,
    IT: Parameterized<
            DomainTracer<'d, D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
        >,
    OT: Parameterized<
            DomainTracer<'d, D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
        >,
    IT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = IT::Family,
            To<DomainTracer<'d, D>> = IT,
            To<<D as Domain>::Constant> = IT::To<<D as Domain>::Constant>,
        >,
    OT::To<D::Type>: Clone
        + Parameterized<
            D::Type,
            Family = OT::Family,
            To<DomainTracer<'d, D>> = OT,
            To<<D as Domain>::Constant> = OT::To<<D as Domain>::Constant>,
        >,
    <D as Domain>::Operation: Clone
        + MaybeDot
        + MaybeRematerializationName
        + SupportsCustomVjp<D::Type, <D as Domain>::Constant>
        + SupportsZero<D::Type>
        + SupportsOne<D::Type>
        + DifferentiableOperation<TracingContext<'d, D>>,
    LinearOperationOf<TracingContext<'d, D>>: ResidualizedOperation<TracingContext<'d, D>>,
    DirectLinearOperationOf<TracingContext<'d, D>>: SupportsTransposition<D::Type, DomainTracer<'d, D>>
        + crate::operations::InterpretableOperation<D::Type, DomainTracer<'d, D>>,
    Vec<D::Type>: Parameterized<
            D::Type,
            Family: ParameterizedFamily<<D as Domain>::Constant> + ParameterizedFamily<DomainTracer<'d, D>>,
            To<DomainTracer<'d, D>> = Vec<DomainTracer<'d, D>>,
            To<<D as Domain>::Constant> = Vec<<D as Domain>::Constant>,
        >,
    Vec<DomainTracer<'d, D>>: Parameterized<
            DomainTracer<'d, D>,
            Family: ParameterizedFamily<D::Type> + ParameterizedFamily<<D as Domain>::Constant>,
            To<D::Type> = Vec<D::Type>,
            To<<D as Domain>::Constant> = Vec<<D as Domain>::Constant>,
            ParameterStructure: Debug + PartialEq,
        >,
    Vec<<D as Domain>::Constant>: Parameterized<
            <D as Domain>::Constant,
            Family: ParameterizedFamily<DomainTracer<'d, D>>,
            To<DomainTracer<'d, D>> = Vec<DomainTracer<'d, D>>,
            ParameterStructure: Debug + PartialEq,
        >,
{
    /// Stages this rematerialized function on the provided tracer inputs and returns its outputs, deriving the
    /// forward/backward programs specialized to the inputs' types. Reverse-mode differentiation of the staged call
    /// stores only the region inputs plus the policy-saved residuals and recomputes everything else.
    pub fn call<C, ICT>(
        &self,
        input: ICT,
    ) -> Result<<OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>, ProgramError>
    where
        C: StagingContext<Type = D::Type, Constant = <D as Domain>::Constant, Operation = <D as Domain>::Operation>,
        IT::Family: ParameterizedFamily<Tracer<C>>,
        OT::Family: ParameterizedFamily<Tracer<C>>,
        ICT: Parameterized<Tracer<C>, Family = IT::Family, To<D::Type> = IT::To<D::Type>>,
        <OT::To<D::Type> as Parameterized<D::Type>>::To<Tracer<C>>: Parameterized<
                Tracer<C>,
                Family = OT::Family,
                ParameterStructure = <OT::To<D::Type> as Parameterized<D::Type>>::ParameterStructure,
            >,
    {
        let mut input_tracers = Vec::new();
        let structured_input_types = input
            .map_parameters(|tracer| {
                let r#type = tracer.r#type().into_owned();
                input_tracers.push(tracer);
                r#type
            })
            .map_err(ProgramError::from)?;
        let Some(first) = input_tracers.first() else {
            return Err(TypeError { message: "rematerialization requires at least one input".to_string() }.into());
        };
        let input_types = structured_input_types.parameters().cloned().collect::<Vec<_>>();

        // Stage a previously cached derivation when one exists for these input types, without re-tracing anything.
        let cached = self
            .cache
            .borrow()
            .iter()
            .find(|(cached_input_types, ..)| *cached_input_types == input_types)
            .map(|(_, operation, output_structure)| (operation.clone(), output_structure.clone()));
        if let Some((operation, output_structure)) = cached {
            let operation = <D as Domain>::Operation::custom_vjp_operation(operation);
            let outputs = first.context().stage_operation(operation, &input_tracers)?;
            return Ok(Parameterized::from_parameters(output_structure, outputs)?);
        }

        let (structured_output_types, primal) =
            TracingContext::trace(self.domain, |xs| (self.body)(xs), structured_input_types.clone())?;
        let primal = primal.to_flat_program();
        let output_types = structured_output_types.parameters().cloned().collect::<Vec<_>>();

        // Pass 1: derive the forward program — the body outputs followed by the region inputs and the policy-saved
        // residual values — and record which residual indices were saved. Linearizing on tracers gives full
        // provenance: each residual is a tracer whose defining instruction identifies the operation that produced
        // it, which is exactly what the policy classifies.
        let mut saved_indices = Vec::new();
        let mut residual_count = 0;
        let (forward_output_types, forward) = TracingContext::trace(
            self.domain,
            |xs: Vec<DomainTracer<'d, D>>| {
                let context = xs.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let (mut outputs, pushforward) = context.linearize_program(&primal, xs.clone())?;
                outputs.extend(xs.iter().cloned());
                residual_count = pushforward.residuals().len();
                for (index, residual) in pushforward.residuals().iter().enumerate() {
                    if self.policy.saves_residual(residual)? {
                        saved_indices.push(index);
                        outputs.push(residual.clone());
                    }
                }
                Ok(outputs)
            },
            input_types.clone(),
        )?;
        let forward = forward.into_simplified()?;

        // Pass 2: derive the backward program over `(inputs..., saved..., cotangents...)`. Re-linearizing the body
        // inside this trace stages the recomputation graph; substituting the saved residuals with the backward
        // program's own input tracers short-circuits exactly the policy-saved values (residuals that are region
        // inputs already instantiate to the backward inputs, with no storage). Residual enumeration is
        // deterministic, so the indices recorded in pass 1 align with this pass's residual table.
        let input_count = input_types.len();
        let saved_count = saved_indices.len();
        let saved_types = forward_output_types[output_types.len() + input_count..].to_vec();
        let backward_input_types =
            input_types.iter().chain(saved_types.iter()).chain(output_types.iter()).cloned().collect::<Vec<_>>();
        let (_, backward) = TracingContext::trace(
            self.domain,
            |flat: Vec<DomainTracer<'d, D>>| {
                let context = flat.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let primal_tracers = flat[..input_count].to_vec();
                let saved_tracers = &flat[input_count..input_count + saved_count];
                let cotangent_tracers = flat[input_count + saved_count..].to_vec();
                let (_, pushforward) = context.linearize_program(&primal, primal_tracers)?;
                let mut residuals = pushforward.residuals().to_vec();
                if residuals.len() != residual_count {
                    return Err(ProgramError::MalformedProgram(format!(
                        "rematerialization backward derivation produced {} residual(s) but the forward derivation \
                         produced {residual_count}",
                        residuals.len(),
                    )));
                }
                for (slot, index) in saved_indices.iter().enumerate() {
                    residuals[*index] = saved_tracers[slot].clone();
                }
                let instantiated = pushforward
                    .program()
                    .map_operations(|operation| operation.instantiate_residuals(residuals.as_slice()))?;
                let pullback = context.transpose_linear_program(&instantiated)?;
                pullback.interpret(cotangent_tracers)
            },
            backward_input_types,
        )?;
        let backward = backward.into_simplified()?;

        // Pass 3: derive the tangent program over `(inputs..., saved..., input_tangents...)` so that forward-mode
        // differentiation works through the rematerialized call (JAX's `jax.checkpoint` also supports `jvp`). The
        // derivation mirrors the backward pass without the transposition: re-linearize, substitute the saved
        // residuals, and apply the pushforward to the tangent tracers.
        let tangent_input_types =
            input_types.iter().chain(saved_types.iter()).chain(input_types.iter()).cloned().collect::<Vec<_>>();
        let (_, tangent) = TracingContext::trace(
            self.domain,
            |flat: Vec<DomainTracer<'d, D>>| {
                let context = flat.first().ok_or(ProgramError::InvalidInputCount { expected: 1, actual: 0 })?.context();
                let context = context.clone();
                let primal_tracers = flat[..input_count].to_vec();
                let saved_tracers = &flat[input_count..input_count + saved_count];
                let tangent_tracers = flat[input_count + saved_count..].to_vec();
                let (_, pushforward) = context.linearize_program(&primal, primal_tracers)?;
                let mut residuals = pushforward.residuals().to_vec();
                if residuals.len() != residual_count {
                    return Err(ProgramError::MalformedProgram(format!(
                        "rematerialization tangent derivation produced {} residual(s) but the forward derivation \
                         produced {residual_count}",
                        residuals.len(),
                    )));
                }
                for (slot, index) in saved_indices.iter().enumerate() {
                    residuals[*index] = saved_tracers[slot].clone();
                }
                let instantiated = pushforward
                    .program()
                    .map_operations(|operation| operation.instantiate_residuals(residuals.as_slice()))?;
                instantiated.interpret(tangent_tracers)
            },
            tangent_input_types,
        )?;
        let tangent = tangent.into_simplified()?;

        let operation = CustomVjpOperation::new(primal, forward, backward)?
            .with_prevent_cse(self.prevent_cse)
            .with_tangent_program(tangent)?;
        let output_structure = structured_output_types.parameter_structure();
        self.cache.borrow_mut().push((input_types, operation.clone(), output_structure.clone()));
        let outputs = first
            .context()
            .stage_operation(<D as Domain>::Operation::custom_vjp_operation(operation), &input_tracers)?;
        Ok(Parameterized::from_parameters(output_structure, outputs)?)
    }
}

#[cfg(test)]
mod tests {
    use crate::operations::trigonometric::Sin;
    use crate::scalars::ScalarDomain;
    use crate::tracing_v2::operations::control_flow::flat_program_output_types;
    use crate::tracing_v2::operations::dot::{Dot, DotDimensionNumbers};
    use crate::tracing_v2::test_util::{TestArray, TestArrayDomain, assert_close};
    use crate::tracing_v2::{ArrayOperation, value_and_grad};
    use crate::types::{ArrayType, DataType, Shape, Size};

    use super::*;

    /// Computes `f(x) = u * sin(u)` with `u = x · x`, whose linearization residuals span all three policy classes:
    /// `u` is produced by a dot, `sin(u)` by a sine, and the sine rule's `cos(u)` factor by a cosine.
    fn dot_sine<V>(x: V) -> V
    where
        V: Clone + Sin + Dot + std::ops::Mul<Output = V>,
    {
        let u = x.clone().dot(x, &DotDimensionNumbers::inner_product());
        u.clone() * u.sin()
    }

    /// [`dot_sine`] in the closure shape consumed by [`rematerialization`].
    fn dot_sine_body<'d>(
        input: DomainTracer<'d, TestArrayDomain>,
    ) -> Result<DomainTracer<'d, TestArrayDomain>, ProgramError> {
        Ok(dot_sine(input))
    }

    /// Reference gradient of [`dot_sine_body`]: `∇f(x) = (sin(u) + u * cos(u)) * 2x` with `u = x · x`.
    fn dot_sine_gradient(x: &[f64]) -> Vec<f64> {
        let u: f64 = x.iter().map(|value| value * value).sum();
        x.iter().map(|value| (u.sin() + u * u.cos()) * 2.0 * value).collect()
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(size)]))
    }

    #[test]
    fn test_rematerialization_matches_the_unrematerialized_gradient_under_every_policy() {
        let domain = TestArrayDomain;
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        let (direct_value, direct_gradient) = value_and_grad(&domain, |x| dot_sine(x), input.clone()).unwrap();
        for policy in [
            RematerializationPolicy::NothingSaveable,
            RematerializationPolicy::EverythingSaveable,
            RematerializationPolicy::DotsSaveable,
        ] {
            let function = rematerialize(&domain, dot_sine_body).with_policy(policy);
            let (value, gradient) = value_and_grad(&domain, |x| function.call(x).unwrap(), input.clone()).unwrap();
            assert_close(value.values[0], direct_value.values[0]);
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_close(gradient.values[index], *expected);
                assert_close(direct_gradient.values[index], *expected);
            }
        }
    }

    #[test]
    fn test_rematerialization_policies_control_the_saved_residuals() {
        // `dot_sine_body` has one output and one input, and three instruction-produced residuals: the dot output
        // `u`, the sine output `sin(u)`, and the sine rule's `cos(u)` factor. The forward program therefore outputs
        // 2 values under `NothingSaveable` (output + input), 3 under `DotsSaveable` (+`u`), and 5 under
        // `EverythingSaveable`; and the backward program shrinks as more residuals are saved instead of recomputed.
        let domain = TestArrayDomain;
        let mut forward_output_counts = Vec::new();
        let mut backward_instruction_counts = Vec::new();
        for policy in [
            RematerializationPolicy::NothingSaveable,
            RematerializationPolicy::DotsSaveable,
            RematerializationPolicy::EverythingSaveable,
        ] {
            let function = rematerialize(&domain, dot_sine_body).with_policy(policy);
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
            assert_eq!(program.instructions().len(), 1);
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            forward_output_counts.push(flat_program_output_types(operation.forward()).len());
            backward_instruction_counts.push(operation.backward().instructions().len());
        }
        assert_eq!(forward_output_counts, vec![2, 3, 5]);
        // Saving everything prunes the whole recomputation graph from the backward program. Saving only the dot
        // output does not shrink it here because the unsaved `sin(u)` and `cos(u)` residuals still recompute from
        // `u`, keeping the dot instruction reachable; the saved value only short-circuits the factor use itself.
        assert!(
            backward_instruction_counts[0] >= backward_instruction_counts[1]
                && backward_instruction_counts[1] > backward_instruction_counts[2],
            "saving more residuals should never grow the backward program and saving everything should shrink it, \
             but instruction counts were {backward_instruction_counts:?}",
        );
    }

    #[test]
    fn test_rematerialization_name_is_transparent_to_differentiation() {
        let domain = ScalarDomain::<f64>::new();
        let (primal, tangent) =
            domain.jvp(|x| (x.clone() * x).rematerialization_name("square"), 2.0f64, 1.0f64).unwrap();
        assert_eq!(primal, 4.0);
        assert_eq!(tangent, 4.0);
        let (value, gradient) =
            value_and_grad(&domain, |x| (x.clone() * x).rematerialization_name("square"), 3.0f64).unwrap();
        assert_eq!(value, 9.0);
        assert_eq!(gradient, 6.0);
    }

    #[test]
    fn test_name_based_rematerialization_policies_classify_tagged_residuals() {
        // `f(x) = u * sin(u)` with `u = rematerialization_name(x · x, "u")`: the tagged dot output is one of the three
        // instruction-produced residuals (`u`, `sin(u)`, and the sine rule's `cos(u)` factor), so name-based
        // policies can select it (or its complement) by tag.
        fn body<'d>(x: DomainTracer<'d, TestArrayDomain>) -> Result<DomainTracer<'d, TestArrayDomain>, ProgramError> {
            let u = x.clone().dot(x, &DotDimensionNumbers::inner_product()).rematerialization_name("u");
            Ok(u.clone() * u.sin())
        }
        let domain = TestArrayDomain;
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_gradient = dot_sine_gradient(&[0.5, 1.5]);
        // Forward output counts: 2 base outputs (output + input), plus the residuals each policy saves.
        let cases = [
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["u".to_string()]), 3),
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["other".to_string()]), 2),
            (RematerializationPolicy::SaveAnyNamesButThese(vec!["u".to_string()]), 2),
            (RematerializationPolicy::SaveAnyNamesButThese(vec!["other".to_string()]), 3),
            (RematerializationPolicy::SaveAnythingExceptTheseNames(vec!["u".to_string()]), 4),
            (RematerializationPolicy::SaveAnythingExceptTheseNames(vec!["other".to_string()]), 5),
        ];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize(&domain, body).with_policy(policy.clone());
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            assert_eq!(
                flat_program_output_types(operation.forward()).len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Every policy preserves the gradient; only the save/recompute split changes.
            let (_, gradient) = value_and_grad(&domain, |x| function.call(x).unwrap(), input.clone()).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_close(gradient.values[index], *expected);
            }
        }
    }

    #[test]
    fn test_scalar_rematerialization_matches_the_unrematerialized_gradient() {
        let domain = ScalarDomain::<f64>::new();
        for policy in [RematerializationPolicy::NothingSaveable, RematerializationPolicy::EverythingSaveable] {
            let function = rematerialize(&domain, |x: DomainTracer<'_, ScalarDomain<f64>>| Ok((x.clone() * x).sin()))
                .with_policy(policy);
            let (value, gradient) = value_and_grad(&domain, |x| function.call(x).unwrap(), 2.0).unwrap();
            assert_close(value, 4.0f64.sin());
            assert_close(gradient, 4.0f64.cos() * 4.0);
        }
    }

    #[test]
    fn test_jvp_through_rematerialization_uses_the_derived_tangent_program() {
        // Unlike user-authored custom VJPs (which reject forward mode, matching JAX), rematerialized calls carry a
        // derived tangent program, so `jvp` works through them — matching `jax.checkpoint`.
        let domain = TestArrayDomain;
        let function = rematerialize(&domain, dot_sine_body);
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| function.call(x).unwrap(),
                TestArray::new(vector_type(2), vec![0.5, 1.5]),
                TestArray::new(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        // f(x) = u * sin(u) with u = x · x; the tangent against seed e_0 is the first gradient component.
        let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
        assert_close(primal.values[0], u * u.sin());
        assert_close(tangent.values[0], dot_sine_gradient(&[0.5, 1.5])[0]);
    }

    #[test]
    fn test_prevent_cse_is_carried_on_the_staged_custom_vjp_operation() {
        // `prevent_cse` defaults to enabled (JAX parity) and is carried on the staged operation as a backend
        // lowering hint; user-authored custom VJPs (constructed directly) leave it disabled.
        let domain = TestArrayDomain;
        for (function, expected) in [
            (rematerialize(&domain, dot_sine_body), true),
            (rematerialize(&domain, dot_sine_body).with_prevent_cse(false), false),
        ] {
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            assert_eq!(operation.prevent_cse(), expected);
        }
    }

    #[test]
    fn test_nested_rematerialization_matches_the_unrematerialized_gradient() {
        // The analogue of JAX's sqrt-schedule idiom: rematerialized regions nest, with each level storing only its
        // own region inputs. Differentiating the outer call replays the inner call's backward program inside the
        // outer backward derivation, which interprets the inner (transposed) custom_vjp call over tracers.
        let domain = TestArrayDomain;
        let inner = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let outer = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| {
            let y = inner.call(x.clone())?;
            Ok(y.dot(x, &DotDimensionNumbers::inner_product()))
        });
        // f(x) = Σᵢ sin(xᵢ²) xᵢ, so ∂f/∂xⱼ = sin(xⱼ²) + 2 xⱼ² cos(xⱼ²).
        let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_gradient = [0.5f64, 1.5].map(|x| (x * x).sin() + 2.0 * x * x * (x * x).cos());
        let (value, gradient) = value_and_grad(&domain, |x| outer.call(x).unwrap(), input).unwrap();
        assert_close(value.values[0], expected_value);
        for (index, expected) in expected_gradient.iter().enumerate() {
            assert_close(gradient.values[index], *expected);
        }
    }

    #[test]
    fn test_nested_rematerialization_preserves_the_nested_call_structure_and_residual_accounting() {
        let domain = TestArrayDomain;
        let inner = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let outer = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| {
            let y = inner.call(x.clone())?;
            Ok(y.dot(x, &DotDimensionNumbers::inner_product()))
        });
        let (_, program) = TracingContext::trace(&domain, |x| outer.call(x), vector_type(2)).unwrap();
        assert_eq!(program.instructions().len(), 1);
        let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
            panic!("nested rematerialization should stage a custom_vjp call");
        };
        // The outer primal program preserves the inner rematerialized call instead of inlining its body.
        assert!(
            operation
                .primal()
                .instructions()
                .iter()
                .any(|instruction| matches!(instruction.operation(), ArrayOperation::CustomVjp(_))),
            "the outer primal program should contain the inner rematerialized call",
        );
        // `NothingSaveable` everywhere: the outer forward program outputs only the body output plus the region
        // input, storing no interior residuals — in particular nothing produced inside the inner region.
        assert_eq!(flat_program_output_types(operation.forward()).len(), 2);
    }

    #[test]
    fn test_jvp_through_nested_rematerialization_uses_the_derived_tangent_programs() {
        // Forward mode through nested rematerialized calls exercises the un-transposed custom_vjp call replay over
        // tracers: deriving the outer tangent program interprets the inner call's tangent program inside the outer
        // tangent trace.
        let domain = TestArrayDomain;
        let inner = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let outer = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| {
            let y = inner.call(x.clone())?;
            Ok(y.dot(x, &DotDimensionNumbers::inner_product()))
        });
        let (primal, tangent) = TestArrayDomain
            .jvp(
                |x| outer.call(x).unwrap(),
                TestArray::new(vector_type(2), vec![0.5, 1.5]),
                TestArray::new(vector_type(2), vec![1.0, 0.0]),
            )
            .unwrap();
        let expected_value: f64 = [0.5f64, 1.5].iter().map(|x| (x * x).sin() * x).sum();
        let expected_tangent = {
            let x = 0.5f64;
            (x * x).sin() + 2.0 * x * x * (x * x).cos()
        };
        assert_close(primal.values[0], expected_value);
        assert_close(tangent.values[0], expected_tangent);
    }

    #[test]
    fn test_nested_scalar_rematerialization_matches_the_unrematerialized_gradient() {
        let domain = ScalarDomain::<f64>::new();
        let inner = rematerialize(&domain, |x: DomainTracer<'_, ScalarDomain<f64>>| Ok((x.clone() * x).sin()));
        let outer = rematerialize(&domain, |x: DomainTracer<'_, ScalarDomain<f64>>| {
            let y = inner.call(x.clone())?;
            Ok(y * x)
        });
        // f(x) = sin(x²) x, so f'(x) = sin(x²) + 2 x² cos(x²).
        let (value, gradient) = value_and_grad(&domain, |x| outer.call(x).unwrap(), 0.7f64).unwrap();
        assert_close(value, 0.49f64.sin() * 0.7);
        assert_close(gradient, 0.49f64.sin() + 2.0 * 0.49 * 0.49f64.cos());
    }

    #[test]
    fn test_rematerialization_survives_batching_with_preserved_residual_structure() {
        use crate::tracing_v2::batching::BatchContext;

        // Batching a rematerialized call re-wraps it around batched programs instead of inlining the primal, so
        // the memory-saving structure survives `vmap`: the staged program holds exactly one custom_vjp call whose
        // batched forward program still stores only the body output and the region input plus the policy-saved
        // residuals — each now carrying the lane axis.
        let domain = TestArrayDomain;
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        for (policy, expected_forward_outputs) in [
            (RematerializationPolicy::NothingSaveable, 2),
            (RematerializationPolicy::DotsSaveable, 3),
            (RematerializationPolicy::EverythingSaveable, 5),
        ] {
            let function = rematerialize(&domain, dot_sine_body).with_policy(policy.clone());
            let (_, program) = TracingContext::trace(
                &domain,
                |x| {
                    let context = x.context().clone();
                    BatchContext::batch(&context, |lane| function.call(lane), x, Some(0), Some(0), None)
                        .map_err(ProgramError::from)
                },
                matrix_type.clone(),
            )
            .unwrap();
            assert_eq!(program.instructions().len(), 1, "unexpected batched program shape for policy {policy:?}");
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("batching a rematerialized call should re-wrap the staged custom_vjp call");
            };
            let forward_output_types = flat_program_output_types(operation.forward());
            assert_eq!(
                forward_output_types.len(),
                expected_forward_outputs,
                "unexpected batched forward output count for policy {policy:?}",
            );
            // Every batched forward output carries the lane axis at position 0.
            for output_type in &forward_output_types {
                assert_eq!(
                    output_type.shape().dimensions().first().copied(),
                    Some(Size::Static(2)),
                    "batched forward outputs should carry the lane axis for policy {policy:?}",
                );
            }
        }
    }

    #[test]
    fn test_rematerialized_gradients_are_correct_through_batching() {
        use crate::tracing_v2::LinearizationTracer;
        use crate::tracing_v2::batching::BatchContext;
        use crate::tracing_v2::operations::reduce::{Reduce, ReductionKind};

        // `grad(vmap(rematerialize(f)))`: the gradient flows through the re-wrapped batched call's derived
        // backward program and matches the analytic per-lane gradients.
        let domain = TestArrayDomain;
        let function = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let (value, gradient) = value_and_grad(
            &domain,
            |x| {
                let context = x.context().clone();
                let mapped: LinearizationTracer<'_, TestArrayDomain> =
                    BatchContext::batch(&context, |lane| function.call(lane), x, Some(0), Some(0), None).unwrap();
                mapped.reduce(&[0], ReductionKind::Sum)
            },
            TestArray::new(vector_type(2), vec![0.5, 1.0]),
        )
        .unwrap();
        // f(x) = Σᵢ sin(xᵢ²), so ∂f/∂xⱼ = 2 xⱼ cos(xⱼ²).
        assert_close(value.values[0], 0.25f64.sin() + 1.0f64.sin());
        assert_close(gradient.values[0], 2.0 * 0.5 * 0.25f64.cos());
        assert_close(gradient.values[1], 2.0 * 1.0 * 1.0f64.cos());
    }

    #[test]
    fn test_call_caches_derivations_per_input_types() {
        use std::cell::Cell;

        // The body closure runs only while deriving (the primal trace; the remaining passes replay the traced
        // program), so the closure invocation count equals the number of derivations.
        let domain = TestArrayDomain;
        let trace_count = Rc::new(Cell::new(0));
        let counter = trace_count.clone();
        let function = rematerialize(&domain, move |x: DomainTracer<'_, TestArrayDomain>| {
            counter.set(counter.get() + 1);
            Ok((x.clone() * x).sin())
        });

        // Two calls with equal input types derive once, both within one trace and across separate traces.
        let (_, program) = TracingContext::trace(
            &domain,
            |x: DomainTracer<'_, TestArrayDomain>| {
                let first = function.call(x.clone())?;
                let second = function.call(x)?;
                Ok(first + second)
            },
            vector_type(2),
        )
        .unwrap();
        assert_eq!(trace_count.get(), 1);
        let custom_vjp_count = program
            .instructions()
            .iter()
            .filter(|instruction| matches!(instruction.operation(), ArrayOperation::CustomVjp(_)))
            .count();
        assert_eq!(custom_vjp_count, 2, "both calls should stage their own custom_vjp instruction");
        TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
        assert_eq!(trace_count.get(), 1);

        // A different input type re-derives.
        TracingContext::trace(&domain, |x| function.call(x), vector_type(3)).unwrap();
        assert_eq!(trace_count.get(), 2);

        // Cache hits still differentiate correctly: the second gradient call reuses the derivation staged by the
        // first one.
        let (_, first_gradient) =
            value_and_grad(&domain, |x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        let derivations_after_first_gradient = trace_count.get();
        let (_, second_gradient) =
            value_and_grad(&domain, |x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        assert_eq!(trace_count.get(), derivations_after_first_gradient);
        assert_close(first_gradient.values[0], 2.0 * 0.7 * 0.49f64.cos());
        assert_close(second_gradient.values[0], first_gradient.values[0]);
    }

    #[test]
    fn test_dots_with_no_batch_dims_saveable_skips_batched_contractions() {
        // The body stages two dots: a batched per-row inner product `u = dot(x, x; batch=[0])` and an unbatched
        // inner product `v = u · u`. `DotsSaveable` saves both dot residuals while
        // `DotsWithNoBatchDimsSaveable` saves only the unbatched one, so the forward output counts differ by one
        // (2 base outputs = body output + region input).
        fn body<'d>(x: DomainTracer<'d, TestArrayDomain>) -> Result<DomainTracer<'d, TestArrayDomain>, ProgramError> {
            let batched = DotDimensionNumbers::new(vec![1], vec![1], vec![0], vec![0]);
            let u = x.clone().dot(x, &batched);
            let v = u.clone().dot(u, &DotDimensionNumbers::inner_product());
            Ok(v.clone() * v.sin())
        }
        let domain = TestArrayDomain;
        let matrix_type = ArrayType::new(DataType::F64, Shape::new(vec![Size::Static(2), Size::Static(3)]));
        let cases =
            [(RematerializationPolicy::DotsSaveable, 4), (RematerializationPolicy::DotsWithNoBatchDimsSaveable, 3)];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize(&domain, body).with_policy(policy.clone());
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), matrix_type.clone()).unwrap();
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            assert_eq!(
                flat_program_output_types(operation.forward()).len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
    }

    #[test]
    fn test_save_from_both_policies_saves_the_union_of_both_policies() {
        // The body produces one dot residual `u`, one named residual `s`, and one unnamed `cos` residual; the
        // combinator saves the union of what its two members save.
        fn body<'d>(x: DomainTracer<'d, TestArrayDomain>) -> Result<DomainTracer<'d, TestArrayDomain>, ProgramError> {
            let u = x.clone().dot(x, &DotDimensionNumbers::inner_product());
            let s = u.clone().sin().rematerialization_name("s");
            Ok(u * s)
        }
        let domain = TestArrayDomain;
        let cases = [
            (RematerializationPolicy::DotsSaveable, 3),
            (RematerializationPolicy::SaveOnlyTheseNames(vec!["s".to_string()]), 3),
            (
                RematerializationPolicy::SaveFromBothPolicies(
                    Box::new(RematerializationPolicy::DotsSaveable),
                    Box::new(RematerializationPolicy::SaveOnlyTheseNames(vec!["s".to_string()])),
                ),
                4,
            ),
        ];
        for (policy, expected_forward_outputs) in cases {
            let function = rematerialize(&domain, body).with_policy(policy.clone());
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            assert_eq!(
                flat_program_output_types(operation.forward()).len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
        }
    }

    #[test]
    fn test_custom_policies_classify_residuals_through_candidates() {
        // Custom policies see each residual's classification facts. The first policy reproduces the
        // `SaveFromBothPolicies` union from the test above through candidate queries; the second selects by
        // operation name; and both observe the residuals' staged types.
        fn body<'d>(x: DomainTracer<'d, TestArrayDomain>) -> Result<DomainTracer<'d, TestArrayDomain>, ProgramError> {
            let u = x.clone().dot(x, &DotDimensionNumbers::inner_product());
            let s = u.clone().sin().rematerialization_name("s");
            Ok(u * s)
        }
        let domain = TestArrayDomain;
        let dot_or_named =
            RematerializationPolicy::Custom(Rc::new(|candidate: &RematerializationCandidate<'_, ArrayType>| {
                assert!(candidate.residual_type().shape().dimensions().is_empty());
                candidate.is_dot() || candidate.rematerialization_name() == Some("s")
            }));
        let cosines_only =
            RematerializationPolicy::Custom(Rc::new(|candidate: &RematerializationCandidate<'_, ArrayType>| {
                candidate.operation_name() == "cos"
            }));
        let expected_gradient = {
            // f(x) = u sin(u) with u = x · x, so ∇f(x) = (sin(u) + u cos(u)) · 2x.
            let u: f64 = 0.5 * 0.5 + 1.5 * 1.5;
            [0.5f64, 1.5].map(|value| (u.sin() + u * u.cos()) * 2.0 * value)
        };
        for (policy, expected_forward_outputs) in [(dot_or_named, 4), (cosines_only, 3)] {
            let function = rematerialize(&domain, body).with_policy(policy.clone());
            let (_, program) = TracingContext::trace(&domain, |x| function.call(x), vector_type(2)).unwrap();
            let ArrayOperation::CustomVjp(operation) = program.instructions()[0].operation() else {
                panic!("rematerialization should stage a custom_vjp call");
            };
            assert_eq!(
                flat_program_output_types(operation.forward()).len(),
                expected_forward_outputs,
                "unexpected forward output count for policy {policy:?}",
            );
            // Custom policies only change the save/recompute split, never the gradient.
            let input = TestArray::new(vector_type(2), vec![0.5, 1.5]);
            let (_, gradient) = value_and_grad(&domain, |x| function.call(x).unwrap(), input).unwrap();
            for (index, expected) in expected_gradient.iter().enumerate() {
                assert_close(gradient.values[index], *expected);
            }
        }
    }

    #[test]
    fn test_second_order_reverse_through_rematerialization_matches_the_analytic_second_derivative() {
        use crate::tracing_v2::DifferentiableDomainExtension;

        // Second-order differentiation through a rematerialized call: the inner reverse pass replays the derived
        // backward program over tracers (inlining it into the gradient program), and the outer pass differentiates
        // the result. f(x) = sin(x²), so f''(x) = 2 cos(x²) - 4x² sin(x²).
        let domain = TestArrayDomain;
        let function = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let hessian = domain.hessian(|x| function.call(x).unwrap(), TestArray::scalar(0.7)).unwrap();
        let (_, _, block) = hessian.iter_blocks().next().unwrap();
        let x: f64 = 0.7;
        assert_close(block.values()[0], 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin());
    }

    #[test]
    fn test_scalar_second_order_through_rematerialization_matches_the_analytic_second_derivative() {
        use crate::tracing_v2::{DifferentiationContext, DifferentiationError};

        // The scalar counterpart of the test above, staged manually (the dense `hessian` API is array-only):
        // trace the rematerialized gradient, then push a unit tangent through its linearization at the same point.
        let domain = ScalarDomain::<f64>::new();
        let function = rematerialize(&domain, |x: DomainTracer<'_, ScalarDomain<f64>>| Ok((x.clone() * x).sin()));
        let (_, gradient_program) = TracingContext::interpret_and_trace(
            &domain,
            |x: DomainTracer<'_, ScalarDomain<f64>>| {
                let context = x.context().clone();
                DifferentiationContext::value_and_gradient(&context, |y| function.call(y).unwrap(), x).map_err(
                    |error| match error {
                        DifferentiationError::Program(error) => error,
                        error => ProgramError::MalformedProgram(error.to_string()),
                    },
                )
            },
            0.7f64,
        )
        .unwrap();
        let (gradient, pushforward) = domain.linearize_program(&gradient_program, vec![0.7]).unwrap();
        let second_derivative = pushforward.apply(1.0).unwrap();
        let x: f64 = 0.7;
        assert_close(gradient, 2.0 * x * (x * x).cos());
        assert_close(second_derivative, 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin());
    }

    #[test]
    fn test_linear_transpose_of_the_rematerialized_pullback_recovers_the_tangent_map() {
        // The pullback of a rematerialized call carries a derived tangent program, so transposing it again — the
        // analogue of JAX's `jax.linear_transpose` of a vjp function — recovers the tangent map instead of
        // erroring. f(x) = sin(x²), so the recovered map sends dx to f'(0.7) · dx.
        let domain = TestArrayDomain;
        let function = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let (_, pullback) = domain.vjp(|x| function.call(x), TestArray::scalar(0.7)).unwrap();
        let tangent_map = domain.transpose_linear_program(&pullback).unwrap();
        let output = tangent_map.interpret(TestArray::scalar(1.0)).unwrap();
        let x: f64 = 0.7;
        assert_close(output.values[0], 2.0 * x * (x * x).cos());
    }

    #[test]
    fn test_jacrev_through_rematerialization_uses_the_rematerializing_backward_program() {
        use crate::tracing_v2::jacrev;

        // The Jacobian of elementwise `sin(x * x)` is the diagonal matrix `diag(cos(x²) * 2x)`; `jacrev` exercises
        // the batched replay of the derived backward program.
        let domain = TestArrayDomain;
        let function = rematerialize(&domain, |x: DomainTracer<'_, TestArrayDomain>| Ok((x.clone() * x).sin()));
        let jacobian = jacrev(&domain, |x| function.call(x), TestArray::new(vector_type(2), vec![0.5, 1.0])).unwrap();
        let (_, _, block) = jacobian.iter_blocks().next().unwrap();
        assert_close(block.values()[0], 0.25f64.cos());
        assert_close(block.values()[1], 0.0);
        assert_close(block.values()[2], 0.0);
        assert_close(block.values()[3], 1.0f64.cos() * 2.0);
    }
}
