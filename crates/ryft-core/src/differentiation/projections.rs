//! Differentiation across [`ProjectedContext`]s, in both directions of a context projection.
//!
//! # Member Rules in Projected Contexts
//!
//! An operation family that includes the operations of a member [`Type`] (e.g., the array operations of the array IR
//! family) differentiates and transposes those member operations through their own rules. Family dispatchers (e.g.,
//! the ones that `#[derive(Operation)]` generates) project the inputs of such an operation into the member value
//! family, apply its [`DifferentiableOperation`] or [`TransposableOperation`] rule in a projected view of the
//! enclosing [`DifferentiationContext`] or [`TranspositionContext`], and lift its outputs back. Refer to
//! [`DifferentiationContext::jvp_projected_operation`] and [`TranspositionContext::transpose_projected_operation`] for
//! more information on that last part. [`ProjectedDifferentiationPolicy`] keeps the projected view's primal and tangent
//! separation identical to the enclosing one. Member operations whose rules need capabilities of the parent family
//! implement [`MemberDifferentiableOperation`] and [`MemberTransposableOperation`] instead.
//!
//! # Projected Values in Parent Contexts
//!
//! [`differentiate_in_parent_at`] is the differentiation counterpart of [`batch_in_parent`](crate::batch_in_parent),
//! and goes the other way than the member rules above (i.e., from a projected context to its parent). Its active
//! leaves are projections of values of a richer parent context (i.e., leaves of a [`ProjectedContext`], such as the
//! [`ArrayType`](crate::ArrayType) views of array IR values), and its terminal functions differentiate in that parent
//! context rather than in the projected one. The differentiated closures still receive projected values, but every
//! operation that they bind is differentiated by the rules of the parent's universe, which can contain operations
//! that the projected one lacks (e.g., calls of compiled functions). Results are projected back at their boundary:
//! derivative values directly, and the reusable [`ParentPushforward`] and [`ParentPullback`] functions at every
//! application, so that they take and return projected values as well.
//!
//! The terminal functions flatten the active leaves, lift them into the parent context, run the ordinary
//! [`DifferentiationBuilder`](crate::DifferentiationBuilder) of that context, and rebuild the caller's structures from
//! their [`Parameterized::ParameterStructure`]s, so they accept the same (arbitrarily nested) structures as that
//! builder.

use std::marker::PhantomData;

use crate::captures::ImportingContext;
use crate::contexts::{Context, ContextProjection, Domain, ProjectedContext, StagingContext};
use crate::differentiation::forward::{
    DifferentiableOperation, DifferentiationContext, DifferentiationDriver, DifferentiationDual, DifferentiationPolicy,
    DifferentiationTracer, LinearizationContext, LinearizationTracer, Pushforward,
};
use crate::differentiation::hessian::Hessian;
use crate::differentiation::jacobian::Jacobian;
use crate::differentiation::reverse::{
    CotangentAccumulator, CotangentDestination, CotangentSeed, Pullback, PullbackOperation, ReverseModeDifferentiate,
    TransposableOperation, TranspositionContext, TranspositionDriver,
};
use crate::differentiation::types::{DenseDifferentiableType, DifferentiableType};
use crate::differentiation::zeros::ResidualZeroProvider;
use crate::differentiation::{DifferentiationError, DifferentiationRule, differentiate_at};
use crate::macros::{check_builders, check_count};
use crate::operations::{
    AddOperation, OneOperation, ReferenceFreezeOperation, ReferenceNewOperation, Zero, ZeroOperation,
};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::{
    EmptyRegionDriver, MaybeZero, Operation, OperationProjection, OperationProvider, ProgramError, ProjectedValue,
    ReferenceMemberType, Type, TypeError, TypeIdentityPosition, Typed, Value, ValueProjection,
};
use crate::tracing::{Tracer, TracingContext};

#[cfg(doc)]
use crate::differentiation::forward::{FusedDifferentiationPolicy, PartitionedDifferentiationPolicy};
#[cfg(doc)]
use crate::programs::Region;

/// Adapts an existing [`DifferentiationPolicy`] `P` to a [`ProjectedContext`] while preserving how
/// it separates primal and tangent work. This wraps a policy such as [`FusedDifferentiationPolicy`]
/// or [`PartitionedDifferentiationPolicy`], rather than another choice of how to execute differentiation.
///
/// [`DifferentiationContext::project`] preserves existing contexts. When used with [`DifferentiationContext::new`],
/// the wrapper asks `P` to create a tangent context for the enclosing context, then projects that context into the
/// member family. If `P` shares the primal context, the projected rule shares its primal context too. To transfer a
/// value, the wrapper lifts it into the enclosing value family, applies `P`'s transfer, then projects the result back.
/// For example, an array-only rule inside a heterogeneous context retains partitioned known value imports even though
/// it sees only array values. All of these calls use static dispatch through `P`.
#[derive(Copy, Clone, Debug)]
pub struct ProjectedDifferentiationPolicy<P>(PhantomData<P>);

impl<T: Type, C: ContextProjection<T>, P: DifferentiationPolicy<C>> DifferentiationPolicy<ProjectedContext<C, T>>
    for ProjectedDifferentiationPolicy<P>
{
    const PRIMAL_COMPLETES_BEFORE_TANGENT: bool = P::PRIMAL_COMPLETES_BEFORE_TANGENT;

    #[inline]
    fn tangent_context(primal: &ProjectedContext<C, T>) -> Option<ProjectedContext<C, T>> {
        P::tangent_context(primal.parent()).map(ProjectedContext::new)
    }

    #[inline]
    fn primal_to_tangent(
        tangent: &ProjectedContext<C, T>,
        value: <C::Value as ValueProjection<T>>::Projected,
    ) -> Result<<C::Value as ValueProjection<T>>::Projected, DifferentiationError> {
        let value = <C::Value as ValueProjection<T>>::from_projected(value);
        P::primal_to_tangent(tangent.parent(), value)?.into_projected().map_err(Into::into)
    }

    #[inline]
    fn tangent_to_primal(
        primal: &ProjectedContext<C, T>,
        value: <C::Value as ValueProjection<T>>::Projected,
    ) -> Result<<C::Value as ValueProjection<T>>::Projected, DifferentiationError> {
        let value = <C::Value as ValueProjection<T>>::from_projected(value);
        P::tangent_to_primal(primal.parent(), value)?.into_projected().map_err(Into::into)
    }
}

/// Derivative rules for a homogeneous member [`Operation`] whose [`jvp`](DifferentiableOperation::jvp) or
/// [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule must execute in the parent [`Context`]
/// enclosing its projection. If this operation has member type `T` and the enclosing context has type `U`,
/// [`jvp_projected_operation`](DifferentiationContext::jvp_projected_operation) converts every input from
/// the enclosing value type to the member value type, applies the ordinary member rule in
/// [`ProjectedContext<C, T>`](ProjectedContext), and converts its outputs back. A rule executed that way can work only
/// with values of member type `T`. This trait instead gives the rule the original values and the projected context's
/// parent `C` itself, allowing the rule to use values from other members of `U` when constructing the derivative.
///
/// This distinction matters whenever the primal operation belongs to one projected member but its linearization must
/// retain values belonging to another member of the enclosing type universe. Implementing [`DifferentiableOperation`]
/// directly cannot express that relationship because its [`jvp`](DifferentiableOperation::jvp) function deliberately
/// requires `Self::Type = C::Type`. This trait preserves that same-universe invariant while making member
/// differentiation in the parent universe explicit.
///
/// Implementations bound the parent context by the projection vocabulary they
/// actually use (typically [`ValueProjection<T>`](ValueProjection) for values and constants,
/// [`OperationProjection<T>`](OperationProjection) for the member operation family, and the member and mixed
/// operations they stage) rather than this trait imposing one fixed vocabulary on every implementation.
/// Operation-family dispatchers should use this trait only for projected members whose derivative requires
/// parent-universe values. Members whose inputs, outputs, and derivative all remain within `T` should use
/// [`jvp_projected_operation`](DifferentiationContext::jvp_projected_operation) and
/// [`jvp_for_transpose_projected_operation`](DifferentiationContext::jvp_for_transpose_projected_operation)
/// in their respective named hooks. A parent dispatcher must forward both hooks to preserve member overrides.
pub trait MemberDifferentiableOperation<C: Context>: Operation<Type: DifferentiableType> {
    /// Applies this projected member's Jacobian-Vector Product (JVP) rule (i.e., its [`DifferentiableOperation::jvp`])
    /// in the parent context enclosing the member's projection, using that parent's own values.
    ///
    /// # Parameters
    ///
    ///   - `context`: Parent [`Context`] through which the rule stages member and mixed operations.
    ///   - `driver`: Instruction-scoped [`DifferentiationDriver`] that exposes any attached [`Region`]s.
    ///   - `inputs`: Parent-universe primal/tangent pairs aligned with this operation's inputs.
    fn jvp_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>;

    /// Applies this projected member's derivative rule for subsequent transposition in the parent context enclosing
    /// its projection, using that parent's own values. The default delegates to [`Self::jvp_in_parent`]. Overrides
    /// follow the primal, residual, and transposition contract of [`DifferentiableOperation::jvp_for_transpose`].
    #[inline]
    fn jvp_for_transpose_in_parent<D: DifferentiationDriver<C>, P: DifferentiationPolicy<C>>(
        &self,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        self.jvp_in_parent(context, driver, inputs)
    }
}

/// Transpose rule for a member operation that runs in its enclosing operation family's [`TranspositionContext`].
/// Unlike [`TransposableOperation`], this trait permits the operation's [`Type`] to differ from the context's type.
/// For example, an array slice operates on arrays, but its parent context may also represent references. Its member
/// rule can then add a cotangent into a reference slice without trying to project that reference into the array family.
///
/// This is the reverse mode counterpart of [`MemberDifferentiableOperation`].
/// Generated operation family dispatchers call it for computational projected members. A member family delegates to
/// its individual operations when they need parent family capabilities, and can use
/// [`transpose_projected_operation`](TranspositionContext::transpose_projected_operation) for region-free rules that
/// remain within the member family. Implementations specify the projection and operation capabilities they need; this
/// trait does not require every enclosing family to support references.
///
/// Accumulators remain owned by the enclosing context. The contribution, ownership, and deterministic staging
/// requirements of [`TransposableOperation::transpose`] apply unchanged, including when a rule delegates to projection.
pub trait MemberTransposableOperation<V: Value, O: Operation<Type = V::Type>>:
    Operation<Type: DifferentiableType>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Applies this member's transpose rule using the enclosing family's values and cotangent accumulators.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`TranspositionContext`] for the enclosing operation family, which dereferences to the
    ///     [`TracingContext`] in which rules stage additional linear operations. Reference-state rules access the
    ///     instruction's reference inputs through this context; their state updates do not use the `accumulators`
    ///     handles.
    ///   - `driver`: Call-scoped nested-region [`TranspositionDriver`] for the current instruction, exposing attached
    ///     regions in the enclosing operation family.
    ///   - `inputs`: Per-input [`PartialValue`] knowledge, in operation input order, expressed using the enclosing
    ///     family's types and tracers. As in [`TransposableOperation::transpose`], a [`PartialValue::Unknown`] entry
    ///     marks a linear input that receives a cotangent contribution, while a [`Known`](PartialValue::Known) entry
    ///     supplies a runtime value in the pullback and receives no contribution. Each input's [`Type`] is available
    ///     either way through the [`Typed`] trait.
    ///   - `outputs`: Symbolic cotangents for the instruction's outputs, in operation output order, expressed using
    ///     the enclosing family's tracers. A reference-typed output never carries a live cotangent (its state cotangent
    ///     lives in the accumulator of the root it forwards) and always arrives as a structural zero.
    ///   - `accumulators`: [`CotangentAccumulator`]s, one per input in operation input order, owned by `context`.
    ///     Known, ignored, and reference-state inputs have unneeded handles. Contributions must have the input's
    ///     cotangent type and belong to this context's builder, including when the handle is unneeded.
    fn transpose_in_parent<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>;
}

impl<C: Context, T: Type, P: DifferentiationPolicy<C>> ValueProjection<T> for DifferentiationTracer<C, P>
where
    for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
{
    type Projected = ProjectedValue<T, Self>;
    type ProjectedRef<'v>
        = ProjectedValue<T, &'v Self>
    where
        Self: 'v,
        T: 'v;

    #[inline]
    fn from_projected(value: Self::Projected) -> Self {
        value.into_value()
    }

    #[inline]
    fn projected<'v>(&'v self) -> Result<Self::ProjectedRef<'v>, TypeError>
    where
        T: 'v,
    {
        Ok(ProjectedValue::new(self, <&T>::try_from(self.r#type().as_ref())?.clone()))
    }

    #[inline]
    fn into_projected(self) -> Result<Self::Projected, TypeError> {
        let r#type = <&T>::try_from(self.r#type().as_ref())?.clone();
        Ok(ProjectedValue::new(self, r#type))
    }
}

impl<C: Context, P: DifferentiationPolicy<C>> DifferentiationContext<C, P> {
    /// Returns a projected view of this [`DifferentiationContext`] for a member type, preserving its primal/tangent
    /// separation and value transfer policy. The projected contexts share the existing contexts' state (i.e.,
    /// projection does not create a new tangent computation or discard values already transferred to it).
    ///
    /// When using the [`FusedDifferentiationPolicy`], [`primal`](Self::primal) and [`tangent`](Self::tangent) still
    /// return the same projected context. When using [`PartitionedDifferentiationPolicy`], rules append to the original
    /// primal and tangent programs through their projected contexts.
    pub fn project<T: Type>(&self) -> DifferentiationContext<ProjectedContext<C, T>, ProjectedDifferentiationPolicy<P>>
    where
        C: ContextProjection<T>,
    {
        // Reuse the existing contexts instead of invoking the policy's tangent context constructor again.
        DifferentiationContext::from_parts(
            ProjectedContext::new(self.primal().clone()),
            self.separate_tangent().map(|tangent| ProjectedContext::new(tangent.clone())),
            self.rule(),
        )
    }
}

// The following functions are the forward mode rules that operation family dispatchers (e.g., the ones generated by
// `#[derive(Operation)]`) apply to member operations whose rules run through a projected or structural view.
impl<C: Context, P: DifferentiationPolicy<C>> DifferentiationContext<C, P> {
    /// Applies a member operation's Jacobian-Vector Product (JVP) rule through a projected view of this heterogeneous
    /// [`DifferentiationContext`]. Use this function from a family-level operation dispatcher when the operation is
    /// [`Region`]-free and every input and output belongs to the same projectable member type `T`. It projects primal
    /// values and live tangent values into the member value family, carries structural-zero tangents as types without
    /// materializing values, runs the member's existing [`DifferentiableOperation`] rule, and lifts the resulting duals
    /// back into the heterogeneous value family. This named adapter always selects [`DifferentiableOperation::jvp`],
    /// including when the context is constructing a reverse derivative. Use
    /// [`jvp_for_transpose_projected_operation`](Self::jvp_for_transpose_projected_operation) to select the reverse
    /// rule explicitly.
    ///
    /// Operations whose derivative crosses member types or whose rule needs attached regions require an explicit
    /// family-level Jacobian-Vector Product (JVP) rule instead. A member operation that declares
    /// [`RegionSlot`](crate::RegionSlot)s is rejected with an exact diagnostic naming it, because projection reaches
    /// the member rule with no region access: the attached regions are programs in the heterogeneous universe, and no
    /// projected driver can present them in the member universe.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Region-free operation expressed in the projected member operation family.
    ///   - `inputs`: Heterogeneous [`DifferentiationDual`]s corresponding to the operation's inputs.
    #[inline]
    pub fn jvp_projected_operation<
        T: DifferentiableType,
        O: Operation<Type = T> + DifferentiableOperation<ProjectedContext<C, T>>,
    >(
        &self,
        operation: &O,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType>,
    {
        DifferentiationRule::Jvp.apply_projected(self, operation, inputs)
    }

    /// Applies a member operation's [`jvp_for_transpose`](DifferentiableOperation::jvp_for_transpose) rule through a
    /// projected view of this heterogeneous [`DifferentiationContext`]. It has the same projection requirements and
    /// diagnostics as [`jvp_projected_operation`](Self::jvp_projected_operation), but the resulting tangent computation
    /// need only support transposition. The named adapter selects the reverse rule even when
    /// [`DifferentiationContext::rule`] is `Jvp`. It does not change the context's selection
    /// for recursive differentiation.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation carries regions, an input cannot be projected into the member family, or the
    /// selected member rule fails. Callers must retain the resulting tangent computation for transposition rather than
    /// execute it as a pushforward.
    #[inline]
    pub fn jvp_for_transpose_projected_operation<
        T: DifferentiableType,
        O: Operation<Type = T> + DifferentiableOperation<ProjectedContext<C, T>>,
    >(
        &self,
        operation: &O,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType>,
    {
        DifferentiationRule::JvpForTranspose.apply_projected(self, operation, inputs)
    }

    /// Applies the forward mode rule of a structural operation, which stages the operation on the primal inputs and
    /// pairs every output with a structural zero tangent. Structural operations are bookkeeping (e.g., the first-class
    /// dimension operations of a heterogeneous operation family), and so they have no derivative to compute.
    /// `#[derive(Operation)]` uses this function for the `#[ryft(projected(U, structural))]` variants of the
    /// operation families that it derives.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Structural operation in the operation family of this context.
    ///   - `inputs`: [`DifferentiationDual`]s that correspond to the inputs of `operation`.
    ///
    /// # Errors
    ///
    /// Returns an error if `operation` declares regions, because it is staged again without them,
    /// or if staging it fails.
    pub fn jvp_structural_operation(
        &self,
        operation: C::Operation,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C::Type: DifferentiableType,
    {
        // This path stages the operation without regions. Reject them first so that downstream inference or region
        // count errors do not obscure the unsupported operation.
        if !operation.region_slots().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("structural operation `{}` must not declare regions", operation.name()),
            }
            .into());
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        self.primal()
            .bind(operation, Vec::new(), primal_inputs.as_slice())?
            .into_iter()
            .map(DifferentiationDual::new_with_zero_tangent)
            .collect()
    }

    /// Applies the forward mode rule of a structural operation with a mixed boundary,
    /// which stages the operation on the primal inputs and pairs every output with a zero tangent. Unlike
    /// [`jvp_structural_operation`](Self::jvp_structural_operation), this function stages an explicit zero tangent for
    /// every output that belongs to a computational member universe, over the same primal inputs, so that the runtime
    /// extents that those inputs carry stay available to the tangent computation (e.g., for a zero-filled array whose
    /// shape is given by first-class dimension inputs). Outputs that belong to no such universe (e.g., first-class
    /// dimensions) have a zero differential space and receive structural zero tangents instead. `#[derive(Operation)]`
    /// uses this function for the `#[ryft(mixed(U, structural))]` variants of the operation families that it derives.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Structural operation in the operation family of this context.
    ///   - `inputs`: [`DifferentiationDual`]s that correspond to the inputs of `operation`.
    ///   - `member_zero_tangent`: Function that receives this context, the primal inputs, and the type of one output,
    ///     and that returns the zero tangent which it staged for that output, or [`None`] if the output belongs to no
    ///     computational member universe. It is typically implemented by trying
    ///     [`stage_member_zero_tangent`](Self::stage_member_zero_tangent) for every computational member universe of
    ///     the operation family.
    ///
    /// # Errors
    ///
    /// Returns an error if `operation` declares regions, because it is staged again without them, if its output types
    /// cannot be inferred, or if staging any primal or tangent operation fails.
    pub fn jvp_structural_mixed_operation<
        F: Fn(&Self, &[C::Value], &C::Type) -> Result<Option<C::Value>, DifferentiationError>,
    >(
        &self,
        operation: C::Operation,
        inputs: &[DifferentiationDual<C::Value>],
        member_zero_tangent: F,
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError>
    where
        C::Type: DifferentiableType,
    {
        // This path stages the operation without regions. Reject them first so that downstream inference or region
        // count errors do not obscure the unsupported operation.
        if !operation.region_slots().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!("structural operation `{}` must not declare regions", operation.name()),
            }
            .into());
        }

        let primal_inputs = inputs.iter().map(|input| input.primal().clone()).collect::<Vec<_>>();
        let input_types = primal_inputs.iter().map(|input| input.r#type().into_owned()).collect::<Vec<_>>();
        let output_types = operation.infer_output_types(input_types.as_slice(), &[])?;
        let primals = self.primal().bind(operation, Vec::new(), primal_inputs.as_slice())?;
        check_count!("output", primals, output_types.len(), ProgramError);
        primals
            .into_iter()
            .zip(output_types)
            .map(|(primal, output_type)| match member_zero_tangent(self, primal_inputs.as_slice(), &output_type)? {
                Some(tangent) => DifferentiationDual::new(primal, MaybeZero::Value(tangent)),
                None => DifferentiationDual::new_with_zero_tangent(primal),
            })
            .collect()
    }

    /// Stages a zero tangent for an output of the provided type if that type belongs to member universe `U`, and
    /// returns [`None`] otherwise. The zero is staged in the tangent context of this context over the tangent views of
    /// the provided primal inputs, so that it can read the runtime extents that they carry. This is meant to be used as
    /// part of the `member_zero_tangent` function of
    /// [`jvp_structural_mixed_operation`](Self::jvp_structural_mixed_operation).
    ///
    /// # Parameters
    ///
    ///   - `primal_inputs`: Primal inputs of the structural operation whose output tangent is being staged.
    ///   - `output_type`: Type of the output whose tangent is being staged.
    ///
    /// # Errors
    ///
    /// Returns an error if the output type has no tangent representation, if the primal inputs cannot be converted to
    /// tangent values, or if staging the zero fails.
    pub fn stage_member_zero_tangent<U: DifferentiableType>(
        &self,
        primal_inputs: &[C::Value],
        output_type: &C::Type,
    ) -> Result<Option<C::Value>, DifferentiationError>
    where
        C::Type: DifferentiableType,
        C::Operation: From<ZeroOperation<U>>,
        for<'t> &'t U: TryFrom<&'t C::Type, Error = TypeError>,
    {
        let Ok(output_type) = <&U>::try_from(output_type) else {
            return Ok(None);
        };
        let tangent_type = output_type.tangent()?;
        let tangent_inputs = primal_inputs
            .iter()
            .map(|value| self.primal_to_tangent(value.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let mut tangents =
            self.tangent().bind(ZeroOperation::new(tangent_type), Vec::new(), tangent_inputs.as_slice())?;
        check_count!("output", tangents, 1, ProgramError);
        Ok(Some(tangents.remove(0)))
    }
}

impl DifferentiationRule {
    /// Dispatches the selected parent-universe member rule without changing the context or recursive driver.
    #[inline]
    pub(crate) fn apply_member<
        O: MemberDifferentiableOperation<C>,
        C: Context,
        D: DifferentiationDriver<C>,
        P: DifferentiationPolicy<C>,
    >(
        self,
        operation: &O,
        context: &DifferentiationContext<C, P>,
        driver: &D,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        match self {
            Self::Jvp => operation.jvp_in_parent(context, driver, inputs),
            Self::JvpForTranspose => operation.jvp_for_transpose_in_parent(context, driver, inputs),
        }
    }

    /// Projects inputs, applies the explicitly selected member rule, and lifts its outputs into the parent universe.
    pub(crate) fn apply_projected<
        T: DifferentiableType,
        O: Operation<Type = T> + DifferentiableOperation<ProjectedContext<C, T>>,
        C: ContextProjection<T, Type: DifferentiableType>,
        P: DifferentiationPolicy<C>,
    >(
        self,
        context: &DifferentiationContext<C, P>,
        operation: &O,
        inputs: &[DifferentiationDual<C::Value>],
    ) -> Result<Vec<DifferentiationDual<C::Value>>, DifferentiationError> {
        if !operation.region_slots().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "projected operation `{}` carries regions and cannot be differentiated through its member family; \
                     differentiate it through a family-level carrier for that operation instead",
                    operation.name(),
                ),
            }
            .into());
        }
        let projected_inputs = inputs
            .iter()
            .map(|input| {
                let primal = <C::Value as ValueProjection<T>>::into_projected(input.primal().clone())?;
                match input.tangent() {
                    MaybeZero::Zero(_) => DifferentiationDual::new_with_zero_tangent(primal),
                    MaybeZero::Value(value) => {
                        let tangent = <C::Value as ValueProjection<T>>::into_projected(value.clone())?;
                        DifferentiationDual::new(primal, tangent)
                    }
                }
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let projected_context = context.project::<T>();
        let outputs = self.apply(operation, &projected_context, &EmptyRegionDriver, &projected_inputs)?;
        outputs
            .into_iter()
            .map(|output| {
                let (primal, tangent) = output.into_parts();
                let primal = <C::Value as ValueProjection<T>>::from_projected(primal);
                let tangent = match tangent {
                    MaybeZero::Zero(r#type) => MaybeZero::Zero(r#type.into()),
                    MaybeZero::Value(value) => {
                        MaybeZero::Value(<C::Value as ValueProjection<T>>::from_projected(value))
                    }
                };
                DifferentiationDual::new(primal, tangent)
            })
            .collect::<Result<Vec<_>, _>>()
    }
}

// The following functions are the transposition rules that operation family dispatchers (e.g., the ones generated by
// `#[derive(Operation)]`) apply to member operations whose rules run through a projected view.
impl<V: Value, O: Operation<Type = V::Type>> TranspositionContext<V, O>
where
    TracingContext<V, O>: StagingContext<Type = V::Type, Constant = V, Operation = O>,
{
    /// Applies a member operation's transpose rule through a projected view of this heterogeneous
    /// [`TranspositionContext`]. Use this function from a family-level operation dispatcher when the linear operation
    /// is [`Region`]-free and every input and output belongs to the same projectable member type `T`. Because
    /// [`TransposableOperation`] rules stage through a member-typed [`TracingContext`], this function records the rule
    /// in a short-lived member program, converts that program to the heterogeneous type, and splices it into the active
    /// trace. Known primal inputs and live output cotangents become splice inputs in encounter order; structural zeros
    /// remain types and do not materialize values. The member rule receives the enclosing handles' cotangent mask but
    /// accumulates values locally, because its type family may not represent reference buffers. Each resulting
    /// contribution is lifted and submitted to the corresponding enclosing handle. Family-level rules that can update
    /// reference views directly should bypass this dense fallback.
    ///
    /// Operations whose transpose crosses member types or whose rule needs attached regions require an explicit
    /// family-level transpose rule instead. A member operation that declares [`RegionSlot`](crate::RegionSlot)s is
    /// rejected with an exact diagnostic naming it, because projection reaches the member rule with no region access:
    /// the attached regions are programs in the _heterogeneous_ universe, and no projected driver can present them in
    /// the member universe.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Region-free linear operation expressed in the projected member operation family.
    ///   - `inputs`: Per-input primal knowledge, preserving whether each primal is known or is a linear unknown.
    ///   - `outputs`: Heterogeneous output cotangents, represented as live traced values or structural zeros.
    ///   - `accumulators`: Enclosing value cotangent handles, aligned with `inputs` and owned by this context.
    pub fn transpose_projected_operation<T: DifferentiableType, P: Operation<Type = T>>(
        &mut self,
        operation: &P,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>
    where
        V: ValueProjection<T, Projected: Value<Type = T>>,
        V::Type: DifferentiableType + From<T>,
        P: TransposableOperation<<V as ValueProjection<T>>::Projected, P>,
        O: OperationProjection<T, Projected = P> + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
        TracingContext<<V as ValueProjection<T>>::Projected, P>:
            StagingContext<Type = T, Constant = <V as ValueProjection<T>>::Projected, Operation = P>,
        for<'t> &'t T: TryFrom<&'t V::Type, Error = TypeError>,
    {
        if !operation.region_slots().is_empty() {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "projected operation `{}` carries regions and cannot be transposed through its member family; \
                     transpose it through a family-level carrier for that operation instead",
                    operation.name(),
                ),
            }
            .into());
        }

        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        accumulators.iter().try_for_each(|accumulator| self.accumulated_value(accumulator).map(|_| ()))?;

        // Stage the native member rule in an isolated member-typed trace. The completed rule program is converted back
        // to the heterogeneous type before it is attached to this context, so primitive transpose rules never need
        // heterogeneous types. The trace is detached from any region, so the member rule cannot reach reference
        // accumulators, which is what this helper's region-free, reference-free contract requires.
        let mut rule_context =
            TranspositionContext::<<V as ValueProjection<T>>::Projected, P>::new(TracingContext::new());

        // Build the member rule's boundary and the matching source atoms together. Unknown primals contribute only
        // their projected types. Known primals become leading member-program inputs and their heterogeneous atoms
        // become the leading splice inputs, preserving the transposition rule's input order. Validate builder ownership
        // before taking atom IDs: the same index in another trace names an unrelated value.
        let mut splice_inputs = Vec::new();
        let rule_inputs = inputs
            .iter()
            .map(|input| -> Result<_, DifferentiationError> {
                match input {
                    PartialValue::Unknown(r#type) => Ok(PartialValue::Unknown(<&T>::try_from(r#type)?.clone())),
                    PartialValue::Known(value) => {
                        check_builders!(self.builder(), value.builder())?;
                        splice_inputs.push(value.atom_id()?);
                        Ok(PartialValue::Known(rule_context.input(<&T>::try_from(value.r#type().as_ref())?.clone())))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Live output cotangents follow known primals in both boundaries. Structural zeros remain type-only and
        // therefore consume neither a member-program input nor a heterogeneous splice input.
        let rule_outputs = outputs
            .iter()
            .map(|output| -> Result<_, DifferentiationError> {
                match output {
                    MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(<&T>::try_from(r#type)?.clone())),
                    MaybeZero::Value(value) => {
                        check_builders!(self.builder(), value.builder())?;
                        splice_inputs.push(value.atom_id()?);
                        Ok(MaybeZero::Value(rule_context.input(<&T>::try_from(value.r#type().as_ref())?.clone())))
                    }
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Apply the native transpose rule to the member boundary. Any operations synthesized by the rule are recorded
        // in `rule_context`. `EmptyRegionDriver` enforces this helper's region-free contract.
        let cotangent_mask = accumulators.iter().map(CotangentAccumulator::is_needed).collect::<Vec<_>>();
        let rule_accumulators = rule_context.cotangent_accumulators(&rule_inputs, &cotangent_mask)?;
        (*rule_context).clone().invoke_with_provenance_origin(self.provenance(), || {
            operation.transpose(&mut rule_context, &EmptyRegionDriver, &rule_inputs, &rule_outputs, &rule_accumulators)
        })?;

        // Export each accumulated member cotangent. Its additions are part of the member program, while reference
        // updates remain the responsibility of the enclosing self.
        let (contribution_inputs, output_ids): (Vec<_>, Vec<_>) = rule_accumulators
            .iter()
            .enumerate()
            .filter_map(|(index, accumulator)| match rule_context.accumulated_value(accumulator) {
                Ok(Some((value, provenance))) => Some(Ok((index, value, provenance))),
                Ok(None) => None,
                Err(error) => Some(Err(error)),
            })
            .map(|accumulated| {
                let (index, value, provenance) = accumulated?;
                Ok(((index, provenance.clone()), value.atom_id()?))
            })
            .collect::<Result<_, DifferentiationError>>()?;

        // Convert the complete member program into the heterogeneous universe, then splice it with the source atoms
        // collected in precisely the same order as the temporary member-program inputs.
        let rule_program = rule_context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<<V as ValueProjection<T>>::Projected>, Vec<<V as ValueProjection<T>>::Projected>>(
                output_ids,
                vec![Placeholder; splice_inputs.len()],
                vec![Placeholder; contribution_inputs.len()],
            )?
            .into_unprojected::<V, O>()?;
        let splice_outputs = self.builder().borrow_mut().splice_program(&rule_program, splice_inputs.as_slice())?;
        check_count!("output", splice_outputs, contribution_inputs.len(), ProgramError);
        contribution_inputs.into_iter().zip(splice_outputs).try_for_each(|((input, provenance), output)| {
            // Preserve scopes entered inside the member rule when outer storage later combines its contributions.
            let contribution = MaybeZero::Value(self.tracer(output, None));
            (**self)
                .clone()
                .invoke_with_provenance_origin(provenance, || accumulators[input].accumulate(self, contribution))
        })
    }

    /// Applies a member operation's transpose rule to an instruction whose parent boundary is _mixed_, meaning that the
    /// instruction consumes its `T`-typed member inputs together with inputs belonging to other members of the parent
    /// type universe (e.g., the first-class dimensions that supply a dynamic result shape), in any arrangement. Use
    /// this function from a family-level operation dispatcher for a [`Region`]-free payload that keeps its native
    /// member operation type while its instruction crosses member kinds.
    ///
    /// Each instruction input is classified individually rather than by position. An input whose type projects into `T`
    /// is a _data_ input and every other input is a parent-universe shape input. The data inputs, in input order, are
    /// delegated to the payload's homogeneous [`TransposableOperation`] rule through
    /// [`transpose_projected_operation`](Self::transpose_projected_operation), together with their corresponding
    /// accumulator handles. Shape inputs only select the result shape and carry no differential contribution, so they
    /// receive no contribution. This classification makes the helper independent of how the two input kinds are
    /// arranged, so it handles interleaved signatures exactly like the "data-inputs-first" arrangement every current
    /// payload uses. A payload with no data inputs at all (i.e., a dynamic constructor whose inputs are all extents) is
    /// a constant linear map, so no member rule runs.
    ///
    /// Delegating reconstructs the member instruction from type metadata alone, so a mixed instruction whose inputs
    /// carry runtime (i.e., [`Reference`](TypeIdentityPosition::Reference)-position) identities is rejected. Recovering
    /// that runtime-dependent type metadata requires linearization, which retains the relevant primal information as
    /// explicit residuals.
    ///
    /// # Parameters
    ///
    ///   - `operation`: Region-free linear member operation whose instruction has a mixed parent boundary. Its
    ///     homogeneous rule sees exactly the data inputs, in the order they appear in the mixed instruction.
    ///   - `inputs`: Per-input primal knowledge in the mixed instruction's input order.
    ///   - `outputs`: Heterogeneous output cotangents, represented as live traced values or structural zeros.
    ///   - `accumulators`: Enclosing value cotangent handles, aligned with `inputs` and owned by this context.
    pub fn transpose_mixed_operation<T: DifferentiableType, P: Operation<Type = T>>(
        &mut self,
        operation: &P,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>
    where
        V: ValueProjection<T, Projected: Value<Type = T>>,
        V::Type: DifferentiableType + From<T>,
        O: OperationProjection<T> + OperationProvider<V::Type, AddOperation<V::Type>, Operation = O>,
        <O as OperationProjection<T>>::Projected: From<P>
            + TransposableOperation<<V as ValueProjection<T>>::Projected, <O as OperationProjection<T>>::Projected>,
        TracingContext<<V as ValueProjection<T>>::Projected, <O as OperationProjection<T>>::Projected>: StagingContext<
                Type = T,
                Constant = <V as ValueProjection<T>>::Projected,
                Operation = <O as OperationProjection<T>>::Projected,
            >,
        for<'t> &'t T: TryFrom<&'t V::Type, Error = TypeError>,
    {
        check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
        accumulators.iter().try_for_each(|accumulator| self.accumulated_value(accumulator).map(|_| ()))?;

        // Classify each input by whether its type projects into the member universe. Data inputs keep their input order
        // so the delegated member rule sees the same boundary it would see in a homogeneous instruction. Filter handles
        // alongside inputs to preserve the association between each input and its enclosing storage.
        let (data_inputs, data_accumulators): (Vec<_>, Vec<_>) = inputs
            .iter()
            .zip(accumulators)
            .filter(|(input, _)| <&T>::try_from(input.r#type().as_ref()).is_ok())
            .map(|(input, accumulator)| (input.clone(), accumulator.clone()))
            .unzip();

        // A mixed instruction with no member-typed inputs stages a value that does not depend on any of them, so every
        // input receives a structural zero and the member rule is never consulted.
        if data_inputs.is_empty() {
            return Ok(());
        }

        // The delegated member rule sees only the member-typed inputs, so it must be able to derive every cotangent
        // shape from those inputs alone. A runtime identity anywhere in the mixed signature means the runtime shape
        // information lives in the parent-universe inputs instead, and only linearization can retain it.
        if inputs
            .iter()
            .any(|input| input.r#type().identities().any(|(position, _)| position == TypeIdentityPosition::Reference))
        {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "direct `{}` transposition with runtime-dependent type metadata requires linearization so that the \
                     relevant primal information can be retained as residuals",
                    operation.name(),
                ),
            }
            .into());
        }

        let member_operation: <O as OperationProjection<T>>::Projected = operation.clone().into();
        self.transpose_projected_operation(&member_operation, &data_inputs, outputs, &data_accumulators)
    }
}

/// Returns a [`ParentDifferentiationBuilder`] for differentiating functions at `primal`, whose leaves are projections
/// of values of a parent context, in that parent context. Refer to the [module documentation](self) for more
/// information, and to [`differentiate_at`] for the semantics of the terminal functions.
#[inline]
pub fn differentiate_in_parent_at<Input>(primal: Input) -> ParentDifferentiationBuilder<Input> {
    ParentDifferentiationBuilder { primal }
}

/// Builder returned by [`differentiate_in_parent_at`]. Each terminal function mirrors the function of the same name of
/// [`DifferentiationBuilder`](crate::DifferentiationBuilder), recovers the parent context from the first active leaf,
/// and returns [`DifferentiationError`]s for structures with no active leaves.
#[derive(Clone, Debug)]
pub struct ParentDifferentiationBuilder<Input> {
    /// Active primal inputs, whose leaves are projections of values of the parent context.
    primal: Input,
}

// TODO(eaplatanios): Review this module.

/// Returns the parent context of the leaves of `primal`, its parameter structure, and its leaves lifted into that
/// parent context.
fn lift_primal<C, T, Input>(
    primal: Input,
) -> Result<(C, Input::ParameterStructure, Vec<C::Value>), DifferentiationError>
where
    C: ContextProjection<T>,
    T: Type,
    <C::Value as ValueProjection<T>>::Projected: Value<Domain = ProjectedContext<C, T>>,
    Input: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
{
    let context = primal.parameters().next().map(|leaf| leaf.domain().parent().clone()).ok_or_else(|| {
        ProgramError::InvalidArgument {
            message: "differentiation in a parent context requires at least one active input leaf".to_string(),
        }
    })?;
    let structure = primal.parameter_structure();
    let leaves = primal.into_parameters().map(<C::Value as ValueProjection<T>>::from_projected).collect();
    Ok((context, structure, leaves))
}

/// Flattens `tree` and lifts its leaves into their parent values `P`, returning the parameter structure of `tree` as
/// well.
fn flatten_and_lift<T: Type, P: ValueProjection<T>, Tree: Parameterized<P::Projected>>(
    tree: Tree,
) -> (Tree::ParameterStructure, Vec<P>)
where
    P::Projected: Parameter,
{
    let structure = tree.parameter_structure();
    (structure, tree.into_parameters().map(P::from_projected).collect())
}

/// Projects the parent values `leaves` and rebuilds the tree with parameter structure `structure` from them.
fn project_and_rebuild<T: Type, P: ValueProjection<T>, Tree: Parameterized<P::Projected>>(
    structure: Tree::ParameterStructure,
    leaves: Vec<P>,
) -> Result<Tree, ProgramError>
where
    P::Projected: Parameter,
{
    let leaves = leaves.into_iter().map(P::into_projected).collect::<Result<Vec<_>, _>>()?;
    Ok(Tree::from_parameters(structure, leaves)?)
}

/// Returns the projected [`Type`] of a parent type.
fn project_type<S, T: Type>(r#type: &S) -> Result<T, TypeError>
where
    for<'t> &'t T: TryFrom<&'t S, Error = TypeError>,
{
    Ok(<&T>::try_from(r#type)?.clone())
}

/// Projects the value of a possibly structural zero parent value, keeping structural zeros symbolic.
fn project_maybe_zero<T: Type, V: Value + ValueProjection<T, Projected: Typed<Type = T>>>(
    value: MaybeZero<V>,
) -> Result<MaybeZero<V::Projected>, TypeError>
where
    for<'t> &'t T: TryFrom<&'t V::Type, Error = TypeError>,
{
    match value {
        MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(project_type(&r#type)?)),
        MaybeZero::Value(value) => Ok(MaybeZero::Value(value.into_projected()?)),
    }
}

/// Lifts the value of a projected [`CotangentSeed`] into its parent value `P`.
fn lift_seed<T: Type, P: ValueProjection<T>>(seed: CotangentSeed<P::Projected>) -> CotangentSeed<P> {
    match seed {
        CotangentSeed::Value(value) => CotangentSeed::Value(P::from_projected(value)),
        CotangentSeed::Zero => CotangentSeed::Zero,
        CotangentSeed::NoCotangent => CotangentSeed::NoCotangent,
    }
}

/// Lifts the reference of a projected [`CotangentDestination`] into its parent value `P`.
fn lift_destination<T: Type, P: ValueProjection<T>>(
    destination: CotangentDestination<P::Projected>,
) -> CotangentDestination<P> {
    match destination {
        CotangentDestination::Return => CotangentDestination::Return,
        CotangentDestination::Reference(reference) => CotangentDestination::Reference(P::from_projected(reference)),
        CotangentDestination::Ignore => CotangentDestination::Ignore,
    }
}

impl<Input> ParentDifferentiationBuilder<Input> {
    /// Projected counterpart of [`DifferentiationBuilder::jvp`](crate::DifferentiationBuilder::jvp), whose `tangent`
    /// has the structure and leaf type of the active primal.
    pub fn jvp<C, T, L, Output, F>(
        self,
        tangent: Input,
        function: F,
    ) -> Result<(Output::To<L>, Output::To<L>), DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType>,
        C::Operation: DifferentiableOperation<C> + ResidualZeroProvider<C::Type, Operation = C::Operation>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<L, Family: ParameterizedFamily<ProjectedValue<T, DifferentiationTracer<C>>>>,
        Output: Parameterized<ProjectedValue<T, DifferentiationTracer<C>>, Family: ParameterizedFamily<L>>,
        F: FnOnce(Input::To<ProjectedValue<T, DifferentiationTracer<C>>>) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let tangent = tangent.into_parameters().map(<C::Value as ValueProjection<T>>::from_projected).collect();
        let mut output_structure = None;
        let (output, output_tangent) =
            differentiate_at(primal).in_context(&context).jvp(tangent, |input: Vec<DifferentiationTracer<C>>| {
                let (structure, output) =
                    flatten_and_lift::<T, DifferentiationTracer<C>, _>(function(project_and_rebuild::<
                        T,
                        _,
                        Input::To<_>,
                    >(
                        structure, input
                    )?)?);
                output_structure = Some(structure);
                Ok(output)
            })?;
        let output_structure = output_structure.expect("the differentiated function ran");
        Ok((
            project_and_rebuild::<T, C::Value, Output::To<_>>(output_structure.clone(), output)?,
            project_and_rebuild::<T, C::Value, Output::To<_>>(output_structure, output_tangent)?,
        ))
    }

    /// Projected counterpart of [`DifferentiationBuilder::linearize`](crate::DifferentiationBuilder::linearize), whose
    /// [`ParentPushforward`] takes and returns projected values.
    #[allow(clippy::type_complexity)]
    pub fn linearize<C, T, L, Output, F>(
        self,
        function: F,
    ) -> Result<(Output::To<L>, ParentPushforward<C, T, Input, Output::To<L>>), DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType>,
        C::Operation: PartiallyEvaluatableOperation<C>
            + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
            + ResidualZeroProvider<C::Type, Operation = C::Operation>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<L, Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>>>,
        Output: Parameterized<ProjectedValue<T, LinearizationTracer<C>>, Family: ParameterizedFamily<L>>,
        F: FnOnce(Input::To<ProjectedValue<T, LinearizationTracer<C>>>) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let mut output_structure = None;
        let (output, pushforward) =
            differentiate_at(primal).in_context(&context).linearize(|input: Vec<LinearizationTracer<C>>| {
                let input = project_and_rebuild::<T, _, Input::To<_>>(structure, input)?;
                let (structure, output) = flatten_and_lift::<T, LinearizationTracer<C>, _>(function(input)?);
                output_structure = Some(structure);
                Ok(output)
            })?;
        let output_structure = output_structure.expect("the differentiated function ran");
        Ok((
            project_and_rebuild::<T, C::Value, Output::To<_>>(output_structure.clone(), output)?,
            ParentPushforward { inner: pushforward, output_structure, marker: PhantomData },
        ))
    }

    /// Projected counterpart of [`DifferentiationBuilder::vjp`](crate::DifferentiationBuilder::vjp), whose
    /// [`ParentPullback`] takes and returns projected values.
    #[allow(clippy::type_complexity)]
    pub fn vjp<C, T, L, Output, F>(
        self,
        function: F,
    ) -> Result<(Output::To<L>, ParentPullback<C, T, Input, Output::To<L>>), DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType + ReferenceMemberType> + ReverseModeDifferentiate,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<L, Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>>>,
        Output: Parameterized<ProjectedValue<T, LinearizationTracer<C>>, Family: ParameterizedFamily<L>>,
        F: FnOnce(Input::To<ProjectedValue<T, LinearizationTracer<C>>>) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        <PartialEvaluationContext<C> as Domain>::Operation: DifferentiableOperation<PartialEvaluationContext<C>>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let input_structure = structure.clone();
        let mut output_structure = None;
        let (output, pullback) =
            differentiate_at(primal).in_context(&context).vjp(|input: Vec<LinearizationTracer<C>>| {
                let input = project_and_rebuild::<T, _, Input::To<_>>(structure, input)?;
                let (structure, output) = flatten_and_lift::<T, LinearizationTracer<C>, _>(function(input)?);
                output_structure = Some(structure);
                Ok(output)
            })?;
        let output_structure = output_structure.expect("the differentiated function ran");
        Ok((
            project_and_rebuild::<T, C::Value, Output::To<_>>(output_structure, output)?,
            ParentPullback { inner: pullback, input_structure, marker: PhantomData },
        ))
    }

    /// Projected counterpart of
    /// [`DifferentiationBuilder::value_and_gradient`](crate::DifferentiationBuilder::value_and_gradient).
    pub fn value_and_gradient<C, T, L, F>(self, function: F) -> Result<(L, Input::To<L>), DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType + ReferenceMemberType>
            + ReverseModeDifferentiate<
                Operation: OperationProvider<
                    C::Type,
                    ReferenceNewOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                    Operation = C::Operation,
                > + OperationProvider<
                    C::Type,
                    ReferenceFreezeOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                    Operation = C::Operation,
                > + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
            > + Zero<C::Value>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<L, Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>>>,
        F: FnOnce(
            Input::To<ProjectedValue<T, LinearizationTracer<C>>>,
        ) -> Result<ProjectedValue<T, LinearizationTracer<C>>, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        <PartialEvaluationContext<C> as Domain>::Operation: DifferentiableOperation<PartialEvaluationContext<C>>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let gradient_structure = structure.clone();
        let (value, gradient) = differentiate_at(primal).in_context(&context).value_and_gradient(
            |input: Vec<LinearizationTracer<C>>| {
                let input = project_and_rebuild::<T, _, Input::To<ProjectedValue<T, LinearizationTracer<C>>>>(
                    structure, input,
                )?;
                Ok::<_, ProgramError>(<LinearizationTracer<C> as ValueProjection<T>>::from_projected(function(input)?))
            },
        )?;
        let gradient = project_and_rebuild::<T, C::Value, Input::To<L>>(gradient_structure, gradient)?;
        Ok((value.into_projected()?, gradient))
    }

    /// Projected counterpart of [`DifferentiationBuilder::gradient`](crate::DifferentiationBuilder::gradient).
    pub fn gradient<C, T, L, F>(self, function: F) -> Result<Input::To<L>, DifferentiationError>
    where
        C: ContextProjection<T, Type: DifferentiableType + ReferenceMemberType>
            + ReverseModeDifferentiate<
                Operation: OperationProvider<
                    C::Type,
                    ReferenceNewOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                    Operation = C::Operation,
                > + OperationProvider<
                    C::Type,
                    ReferenceFreezeOperation<<C::Type as ReferenceMemberType>::Referent, C::Type>,
                    Operation = C::Operation,
                > + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
            > + Zero<C::Value>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<L, Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>>>,
        F: FnOnce(
            Input::To<ProjectedValue<T, LinearizationTracer<C>>>,
        ) -> Result<ProjectedValue<T, LinearizationTracer<C>>, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        <PartialEvaluationContext<C> as Domain>::Operation: DifferentiableOperation<PartialEvaluationContext<C>>,
    {
        self.value_and_gradient::<C, T, L, F>(function).map(|(_, gradient)| gradient)
    }

    /// Projected counterpart of
    /// [`DifferentiationBuilder::jacobian_forward`](crate::DifferentiationBuilder::jacobian_forward), whose blocks are
    /// projected values.
    #[allow(clippy::type_complexity)]
    pub fn jacobian_forward<C, T, L, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<T, L, Input::To<T>, Output::To<T>>, DifferentiationError>
    where
        C: ContextProjection<T, Type: DenseDifferentiableType<C>>,
        C::Operation: PartiallyEvaluatableOperation<C>
            + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
            + ResidualZeroProvider<C::Type, Operation = C::Operation>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<
                L,
                Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>> + ParameterizedFamily<T>,
            >,
        Output: Parameterized<ProjectedValue<T, LinearizationTracer<C>>, Family: ParameterizedFamily<T>>,
        F: FnOnce(Input::To<ProjectedValue<T, LinearizationTracer<C>>>) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let input_types =
            primal.iter().map(|leaf| project_type(leaf.r#type().as_ref())).collect::<Result<Vec<_>, _>>()?;
        let input_types =
            Input::To::<T>::from_parameters(structure.clone(), input_types).map_err(ProgramError::from)?;
        let mut output_structure = None;
        let jacobian =
            differentiate_at(primal)
                .in_context(&context)
                .jacobian_forward(|input: Vec<LinearizationTracer<C>>| {
                    let input = project_and_rebuild::<T, _, Input::To<_>>(structure, input)?;
                    let (structure, output) = flatten_and_lift::<T, LinearizationTracer<C>, _>(function(input)?);
                    output_structure = Some(structure);
                    Ok(output)
                })?;
        let output_structure = output_structure.expect("the differentiated function ran");
        let output_types = jacobian.output_type().iter().map(project_type).collect::<Result<Vec<_>, _>>()?;
        let output_types =
            Output::To::<T>::from_parameters(output_structure, output_types).map_err(ProgramError::from)?;
        let values = jacobian
            .into_values()
            .into_iter()
            .map(ValueProjection::<T>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Jacobian::new(input_types, output_types, values)?)
    }

    /// Projected counterpart of
    /// [`DifferentiationBuilder::jacobian_reverse`](crate::DifferentiationBuilder::jacobian_reverse), whose blocks are
    /// projected values.
    #[allow(clippy::type_complexity)]
    pub fn jacobian_reverse<C, T, L, Output, F>(
        self,
        function: F,
    ) -> Result<Jacobian<T, L, Input::To<T>, Output::To<T>>, DifferentiationError>
    where
        C: ContextProjection<T, Type: DenseDifferentiableType<C> + ReferenceMemberType> + ReverseModeDifferentiate,
        C::Operation: DifferentiableOperation<PartialEvaluationContext<C>>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<
                L,
                Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<C>>> + ParameterizedFamily<T>,
            >,
        Output: Parameterized<ProjectedValue<T, LinearizationTracer<C>>, Family: ParameterizedFamily<T>>,
        F: FnOnce(Input::To<ProjectedValue<T, LinearizationTracer<C>>>) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let input_types =
            primal.iter().map(|leaf| project_type(leaf.r#type().as_ref())).collect::<Result<Vec<_>, _>>()?;
        let input_types =
            Input::To::<T>::from_parameters(structure.clone(), input_types).map_err(ProgramError::from)?;
        let mut output_structure = None;
        let jacobian =
            differentiate_at(primal)
                .in_context(&context)
                .jacobian_reverse(|input: Vec<LinearizationTracer<C>>| {
                    let input = project_and_rebuild::<T, _, Input::To<_>>(structure, input)?;
                    let (structure, output) = flatten_and_lift::<T, LinearizationTracer<C>, _>(function(input)?);
                    output_structure = Some(structure);
                    Ok(output)
                })?;
        let output_structure = output_structure.expect("the differentiated function ran");
        let output_types = jacobian.output_type().iter().map(project_type).collect::<Result<Vec<_>, _>>()?;
        let output_types =
            Output::To::<T>::from_parameters(output_structure, output_types).map_err(ProgramError::from)?;
        let values = jacobian
            .into_values()
            .into_iter()
            .map(ValueProjection::<T>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Jacobian::new(input_types, output_types, values)?)
    }

    /// Projected counterpart of [`DifferentiationBuilder::hessian`](crate::DifferentiationBuilder::hessian), whose
    /// blocks are projected values.
    #[allow(clippy::type_complexity)]
    pub fn hessian<C, T, L, Output, F>(
        self,
        function: F,
    ) -> Result<Hessian<T, L, Input::To<T>, Output::To<T>>, DifferentiationError>
    where
        C: ContextProjection<
                T,
                Type: DenseDifferentiableType<C>
                          + DenseDifferentiableType<LinearizationContext<C>>
                          + ReferenceMemberType,
            >,
        C::Operation: PartiallyEvaluatableOperation<C>
            + PartiallyEvaluatableOperation<LinearizationContext<C>>
            + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<C>>
            + DifferentiableOperation<TracingContext<C::Constant, C::Operation>>
            + DifferentiableOperation<PartialEvaluationContext<TracingContext<C::Constant, C::Operation>>>
            + DifferentiableOperation<PartialEvaluationContext<LinearizationContext<C>>>
            + PullbackOperation<C::Type, C::Constant>,
        T: Type,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        L: Value<Type = T, Domain = ProjectedContext<C, T>>,
        C::Value: ValueProjection<T, Projected = L>,
        Input: Parameterized<
                L,
                Family: ParameterizedFamily<ProjectedValue<T, LinearizationTracer<LinearizationContext<C>>>>
                            + ParameterizedFamily<T>,
            >,
        Output: Parameterized<
                ProjectedValue<T, LinearizationTracer<LinearizationContext<C>>>,
                Family: ParameterizedFamily<T>,
            >,
        F: FnOnce(
            Input::To<ProjectedValue<T, LinearizationTracer<LinearizationContext<C>>>>,
        ) -> Result<Output, ProgramError>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        PartialEvaluationContext<C>: Context<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let (context, structure, primal) = lift_primal::<C, T, Input>(self.primal)?;
        let input_types =
            primal.iter().map(|leaf| project_type(leaf.r#type().as_ref())).collect::<Result<Vec<_>, _>>()?;
        let input_types =
            Input::To::<T>::from_parameters(structure.clone(), input_types).map_err(ProgramError::from)?;
        let mut output_structure = None;
        let hessian = differentiate_at(primal).in_context(&context).hessian(
            |input: Vec<LinearizationTracer<LinearizationContext<C>>>| {
                let input = project_and_rebuild::<T, _, Input::To<_>>(structure, input)?;
                let (structure, output) =
                    flatten_and_lift::<T, LinearizationTracer<LinearizationContext<C>>, _>(function(input)?);
                output_structure = Some(structure);
                Ok(output)
            },
        )?;
        let output_structure = output_structure.expect("the differentiated function ran");
        let output_types = hessian.output_types().iter().map(project_type).collect::<Result<Vec<_>, _>>()?;
        let output_types =
            Output::To::<T>::from_parameters(output_structure, output_types).map_err(ProgramError::from)?;
        let values = hessian
            .into_values()
            .into_iter()
            .map(ValueProjection::<T>::into_projected)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Hessian::new(input_types, output_types, values)?)
    }
}

/// [`Pushforward`] returned by [`ParentDifferentiationBuilder::linearize`], whose program and residuals live in the
/// parent context `C` and whose applications take and return projected values with the structures `Input` and `Output`
/// of the linearized function.
pub struct ParentPushforward<
    C: ContextProjection<T>,
    T: Type,
    Input: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
    Output: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
> {
    /// Pushforward in the parent context, over the flattened leaves of the linearized function.
    inner: Pushforward<C, Vec<C::Value>, Vec<C::Value>>,

    /// Parameter structure of the tangents that this pushforward returns.
    output_structure: Output::ParameterStructure,

    /// [`PhantomData`] marker tying this pushforward to its projected [`Type`] and tangent structures.
    marker: PhantomData<fn(Input) -> (T, Output)>,
}

impl<
    C: ContextProjection<T>,
    T: Type,
    Input: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
    Output: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
> ParentPushforward<C, T, Input, Output>
where
    C::Type: DifferentiableType,
{
    /// Returns the [`Pushforward`] in the parent context, over the flattened leaves of the linearized function.
    #[inline]
    pub fn inner(&self) -> &Pushforward<C, Vec<C::Value>, Vec<C::Value>> {
        &self.inner
    }

    /// Projected counterpart of [`Pushforward::apply`].
    pub fn apply(&self, tangents: Input) -> Result<Output, ProgramError>
    where
        C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let tangents = tangents.into_parameters().map(<C::Value as ValueProjection<T>>::from_projected).collect();
        project_and_rebuild::<T, C::Value, Output>(self.output_structure.clone(), self.inner.apply(tangents)?)
    }

    /// Projected counterpart of [`Pushforward::apply_in_context`], whose tangents are the projected leaves of the
    /// caller context.
    pub fn apply_in_context<CallerContext, Path>(
        &self,
        context: &CallerContext,
        tangents: Input::To<<CallerContext::Value as ValueProjection<T>>::Projected>,
    ) -> Result<Output::To<<CallerContext::Value as ValueProjection<T>>::Projected>, ProgramError>
    where
        CallerContext: ContextProjection<T, Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        Input::Family: ParameterizedFamily<<CallerContext::Value as ValueProjection<T>>::Projected>,
        Output::Family: ParameterizedFamily<<CallerContext::Value as ValueProjection<T>>::Projected>,
        C::Operation: ResidualZeroProvider<C::Type, Operation = C::Operation>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let tangents = tangents
            .into_parameters()
            .map(<CallerContext::Value as ValueProjection<T>>::from_projected)
            .collect();
        project_and_rebuild::<T, CallerContext::Value, Output::To<_>>(
            self.output_structure.clone(),
            self.inner.apply_in_context(context, tangents)?,
        )
    }
}

/// [`Pullback`] returned by [`ParentDifferentiationBuilder::vjp`], whose program and residuals live in the parent
/// context `C` and whose applications take and return projected values with the structures `Input` and `Output` of the
/// differentiated function.
pub struct ParentPullback<
    C: ContextProjection<T>,
    T: Type,
    Input: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
    Output: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
> {
    /// Pullback in the parent context, over the flattened leaves of the differentiated function.
    inner: Pullback<C, Vec<C::Value>, Vec<C::Value>>,

    /// Parameter structure of the cotangents that this pullback returns.
    input_structure: Input::ParameterStructure,

    /// [`PhantomData`] marker tying this pullback to its projected [`Type`] and cotangent structures.
    marker: PhantomData<fn(Output) -> (T, Input)>,
}

impl<
    C: ContextProjection<T>,
    T: Type,
    Input: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
    Output: Parameterized<<C::Value as ValueProjection<T>>::Projected>,
> ParentPullback<C, T, Input, Output>
where
    C::Type: DifferentiableType + ReferenceMemberType,
{
    /// Returns the [`Pullback`] in the parent context, over the flattened leaves of the differentiated function.
    #[inline]
    pub fn inner(&self) -> &Pullback<C, Vec<C::Value>, Vec<C::Value>> {
        &self.inner
    }

    /// Projected counterpart of [`Pullback::apply`], whose structural zeros carry projected types.
    pub fn apply(
        &self,
        cotangents: Output,
    ) -> Result<Input::To<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>, ProgramError>
    where
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        Input::Family: ParameterizedFamily<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let cotangents = cotangents.into_parameters().map(<C::Value as ValueProjection<T>>::from_projected).collect();
        let cotangents = self
            .inner
            .apply(cotangents)?
            .into_iter()
            .map(project_maybe_zero::<T, C::Value>)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Input::To::<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>::from_parameters(
            self.input_structure.clone(),
            cotangents,
        )?)
    }

    /// Projected counterpart of [`Pullback::apply_in_context`], whose cotangents are the projected leaves of the
    /// caller context.
    pub fn apply_in_context<CallerContext, Path>(
        &self,
        context: &CallerContext,
        cotangents: Output::To<<CallerContext::Value as ValueProjection<T>>::Projected>,
    ) -> Result<Input::To<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>, ProgramError>
    where
        CallerContext: ContextProjection<T, Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
        Input::Family: ParameterizedFamily<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>,
        Output::Family: ParameterizedFamily<<CallerContext::Value as ValueProjection<T>>::Projected>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let cotangents = cotangents
            .into_parameters()
            .map(<CallerContext::Value as ValueProjection<T>>::from_projected)
            .collect();
        let cotangents = self
            .inner
            .apply_in_context(context, cotangents)?
            .into_iter()
            .map(project_maybe_zero::<T, CallerContext::Value>)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Input::To::<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>::from_parameters(
            self.input_structure.clone(),
            cotangents,
        )?)
    }

    /// Projected counterpart of [`Pullback::apply_materialized`].
    pub fn apply_materialized(&self, cotangents: Output) -> Result<Input, ProgramError>
    where
        C: Zero<C::Value>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let cotangents = cotangents.into_parameters().map(<C::Value as ValueProjection<T>>::from_projected).collect();
        project_and_rebuild::<T, C::Value, Input>(
            self.input_structure.clone(),
            self.inner.apply_materialized(cotangents)?,
        )
    }

    /// Projected counterpart of [`Pullback::apply_materialized_in_context`], whose cotangents are the projected leaves
    /// of the caller context.
    pub fn apply_materialized_in_context<CallerContext, Path>(
        &self,
        context: &CallerContext,
        cotangents: Output::To<<CallerContext::Value as ValueProjection<T>>::Projected>,
    ) -> Result<Input::To<<CallerContext::Value as ValueProjection<T>>::Projected>, ProgramError>
    where
        CallerContext: ContextProjection<T, Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>
            + Zero<CallerContext::Value>,
        Input::Family: ParameterizedFamily<<CallerContext::Value as ValueProjection<T>>::Projected>,
        Output::Family: ParameterizedFamily<<CallerContext::Value as ValueProjection<T>>::Projected>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
    {
        let cotangents = cotangents
            .into_parameters()
            .map(<CallerContext::Value as ValueProjection<T>>::from_projected)
            .collect();
        project_and_rebuild::<T, CallerContext::Value, Input::To<_>>(
            self.input_structure.clone(),
            self.inner.apply_materialized_in_context(context, cotangents)?,
        )
    }

    /// Projected counterpart of [`Pullback::apply_with_destinations`], whose seeds, destinations, and returned
    /// cotangents hold projected values.
    #[allow(clippy::type_complexity)]
    pub fn apply_with_destinations(
        &self,
        seeds: Output::To<CotangentSeed<<C::Value as ValueProjection<T>>::Projected>>,
        destinations: Input::To<CotangentDestination<<C::Value as ValueProjection<T>>::Projected>>,
    ) -> Result<Input::To<Option<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>>, ProgramError>
    where
        Input::Family: ParameterizedFamily<CotangentDestination<<C::Value as ValueProjection<T>>::Projected>>
            + ParameterizedFamily<Option<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>>,
        Output::Family: ParameterizedFamily<CotangentSeed<<C::Value as ValueProjection<T>>::Projected>>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
    {
        let seeds = seeds.into_parameters().map(lift_seed::<T, C::Value>).collect();
        let destinations = destinations.into_parameters().map(lift_destination::<T, C::Value>).collect();
        let cotangents = self
            .inner
            .apply_with_destinations(seeds, destinations)?
            .into_iter()
            .map(|cotangent| cotangent.map(project_maybe_zero::<T, C::Value>).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Input::To::<Option<MaybeZero<<C::Value as ValueProjection<T>>::Projected>>>::from_parameters(
            self.input_structure.clone(),
            cotangents,
        )?)
    }

    /// Projected counterpart of [`Pullback::apply_with_destinations_in_context`], whose seeds and destinations are the
    /// projected leaves of the caller context.
    #[allow(clippy::type_complexity)]
    pub fn apply_with_destinations_in_context<CallerContext, Path>(
        &self,
        context: &CallerContext,
        seeds: Output::To<CotangentSeed<<CallerContext::Value as ValueProjection<T>>::Projected>>,
        destinations: Input::To<CotangentDestination<<CallerContext::Value as ValueProjection<T>>::Projected>>,
    ) -> Result<Input::To<Option<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>>, ProgramError>
    where
        CallerContext: ContextProjection<T, Type = C::Type, Constant = C::Constant, Operation = C::Operation>
            + ImportingContext<C, Path>,
        Input::Family: ParameterizedFamily<CotangentDestination<<CallerContext::Value as ValueProjection<T>>::Projected>>
            + ParameterizedFamily<Option<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>>,
        Output::Family: ParameterizedFamily<CotangentSeed<<CallerContext::Value as ValueProjection<T>>::Projected>>,
        C::Operation: PullbackOperation<C::Type, C::Constant>,
        TracingContext<C::Constant, C::Operation>:
            StagingContext<Type = C::Type, Constant = C::Constant, Operation = C::Operation>,
        for<'t> &'t T: TryFrom<&'t C::Type, Error = TypeError>,
    {
        let seeds = seeds.into_parameters().map(lift_seed::<T, CallerContext::Value>).collect();
        let destinations = destinations.into_parameters().map(lift_destination::<T, CallerContext::Value>).collect();
        let cotangents = self
            .inner
            .apply_with_destinations_in_context(context, seeds, destinations)?
            .into_iter()
            .map(|cotangent| cotangent.map(project_maybe_zero::<T, CallerContext::Value>).transpose())
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Input::To::<Option<MaybeZero<<CallerContext::Value as ValueProjection<T>>::Projected>>>::from_parameters(
            self.input_structure.clone(),
            cotangents,
        )?)
    }
}

#[cfg(test)]
mod tests {
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayType, DataType, DimensionOperation,
        DimensionValue,
    };
    use crate::batching::{BatchAxis, BatchAxisSpecification, batch_in_parent};
    use crate::contexts::EagerContext;
    use crate::differentiation::forward::{FusedDifferentiationPolicy, PartitionedDifferentiationPolicy};
    use crate::differentiation::tests::RuleProbeOperation;
    use crate::operations::{
        ConditionOperation, ConvertElementType, DimensionAddOperation, DimensionSize, DimensionToScalar, NegOperation,
        Reduce, ReductionKind,
    };
    use crate::partial::PartialTracer;
    use crate::programs::{Program, Provenance, ProvenanceScope};
    use crate::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedMemberValue, ProjectedProgramOperation,
        ProjectedProgramType, ProjectedProgramValue,
    };

    use super::*;

    /// Concrete values of the heterogeneous parent universe.
    type TestValue = ArrayIrValue<Array>;

    /// Operations of the heterogeneous parent universe.
    type TestOperation = ArrayIrOperation<Array>;

    /// Eager parent used to test policy and member views.
    type ArrayIrContext = EagerContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Parent trace in which the projected callable adapters run.
    type ArrayIrTracingContext = TracingContext<ArrayIrValue<Array>, ArrayIrOperation<Array>>;

    /// Array view of a value of the parent trace.
    type ArrayTracer = ProjectedValue<ArrayType, Tracer<ArrayIrTracingContext>>;

    /// Program whose array IR boundary is flattened for interpretation assertions.
    type ArrayIrProgram =
        Program<ArrayIrValue<Array>, ArrayIrOperation<Array>, Vec<ArrayIrValue<Array>>, Vec<ArrayIrValue<Array>>>;

    /// Traces projected-array adapters in a full array IR context, retaining their array IR program for execution.
    fn trace_arrays<F>(input_types: Vec<ArrayType>, function: F) -> ArrayIrProgram
    where
        F: FnOnce(Vec<ArrayTracer>) -> Result<Vec<ArrayTracer>, ProgramError>,
    {
        let (_, program) = ArrayIrTracingContext::trace(
            |inputs: Vec<Tracer<ArrayIrTracingContext>>| {
                let inputs = inputs
                    .into_iter()
                    .map(ValueProjection::<ArrayType>::into_projected)
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(function(inputs)?.into_iter().map(ProjectedValue::into_value).collect::<Vec<_>>())
            },
            input_types.into_iter().map(ArrayIrType::Array).collect::<Vec<_>>(),
        )
        .unwrap();
        program
    }

    /// Executes an array IR program and requires every returned member to remain an array, preserving its full type.
    fn interpret_arrays(program: &ArrayIrProgram, inputs: Vec<Array>) -> Vec<Array> {
        program
            .interpret(inputs.into_iter().map(ArrayIrValue::Array).collect::<Vec<_>>())
            .unwrap()
            .into_iter()
            .map(ValueProjection::<ArrayType>::into_projected)
            .collect::<Result<Vec<_>, _>>()
            .unwrap()
    }

    #[test]
    fn test_projected_differentiation_policy_tangent_context() {
        let primal = ProjectedContext::<_, ArrayType>::new(ArrayIrContext::new());
        assert!(
            <ProjectedDifferentiationPolicy<FusedDifferentiationPolicy> as DifferentiationPolicy<_>>::tangent_context(
                &primal,
            )
            .is_none()
        );
        let primal = ProjectedContext::<_, ArrayType>::new(PartialEvaluationContext::new(ArrayIrContext::new()));
        let tangent =
            <ProjectedDifferentiationPolicy<PartitionedDifferentiationPolicy> as DifferentiationPolicy<_>>::tangent_context(
                &primal,
            )
            .unwrap();
        let unknown = PartialTracer::new(
            tangent.parent().clone(),
            tangent.parent().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        assert_eq!(tangent.parent().import_known(&unknown), Ok(unknown.clone()));
        assert!(matches!(
            primal.parent().import_known(&unknown),
            Err(ProgramError::MalformedProgram(message))
                if message == "cannot import an unknown value from another partial-evaluation context",
        ));
    }

    #[test]
    fn test_projected_differentiation_policy_primal_to_tangent() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(ArrayIrContext::new()));
        let primal = context.primal().lift(ArrayIrValue::Array(Array::scalar(3f32).unwrap())).unwrap();
        let projected = ValueProjection::<ArrayType>::into_projected(primal.clone()).unwrap();
        let tangent = context.project::<ArrayType>().primal_to_tangent(projected).unwrap().into_value();
        assert_eq!(context.tangent().import_known(&primal), Ok(tangent.clone()));
        assert_eq!(tangent.value().unwrap().as_known(), Some(&ArrayIrValue::Array(Array::scalar(3f32).unwrap())));

        let unknown = PartialTracer::new(
            context.primal().clone(),
            context.primal().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        assert!(matches!(
            context.project::<ArrayType>().primal_to_tangent(ValueProjection::<ArrayType>::into_projected(unknown).unwrap()),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "cannot import an unknown value from another partial-evaluation context",
        ));
    }

    #[test]
    fn test_projected_differentiation_policy_tangent_to_primal() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(ArrayIrContext::new()));
        let tangent = context.tangent().lift(ArrayIrValue::Array(Array::scalar(5f32).unwrap())).unwrap();
        let projected = ValueProjection::<ArrayType>::into_projected(tangent.clone()).unwrap();
        let primal = context.project::<ArrayType>().tangent_to_primal(projected).unwrap().into_value();
        assert_eq!(context.primal().import_known(&tangent), Ok(primal.clone()));
        assert_eq!(primal.value().unwrap().as_known(), Some(&ArrayIrValue::Array(Array::scalar(5f32).unwrap())));

        let unknown = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        assert!(matches!(
            context.project::<ArrayType>().tangent_to_primal(ValueProjection::<ArrayType>::into_projected(unknown).unwrap()),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "cannot import an unknown value from another partial-evaluation context",
        ));
    }

    #[test]
    fn test_differentiation_tracer_value_projection() {
        let context = DifferentiationContext::fused(ArrayIrContext::new());
        let tracer = DifferentiationTracer::new(
            DifferentiationDual::new(
                ArrayIrValue::Array(Array::scalar(2f32).unwrap()),
                ArrayIrValue::Array(Array::scalar(3f32).unwrap()),
            )
            .unwrap(),
            context.clone(),
        );
        let borrowed = ValueProjection::<ArrayType>::projected(&tracer).unwrap();
        assert!(std::ptr::eq(borrowed.into_value(), &tracer));
        let projected = ValueProjection::<ArrayType>::into_projected(tracer.clone()).unwrap();
        assert_eq!(projected.r#type().as_ref(), &ArrayType::scalar(DataType::F32));
        assert_eq!(
            <DifferentiationTracer<ArrayIrContext> as ValueProjection<ArrayType>>::from_projected(projected),
            tracer,
        );

        let dimension = DifferentiationTracer::new(
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()))
                .unwrap(),
            context,
        );
        assert_eq!(
            ValueProjection::<ArrayType>::projected(&dimension),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
        assert_eq!(
            ValueProjection::<ArrayType>::into_projected(dimension),
            Err(TypeError::invalid("expected array type but got dimension type")),
        );
    }

    #[test]
    fn test_differentiation_context_project() {
        let fused = DifferentiationContext::fused(EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::new());
        let projected = fused.project::<ArrayType>();
        assert!(std::ptr::eq(projected.primal(), projected.tangent()));
        assert_eq!(projected.primal_to_tangent(Array::scalar(3.0f32).unwrap()), Ok(Array::scalar(3.0f32).unwrap()));

        // Projection must reuse the existing tangent context, including its unknown inputs and earlier transfers.
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            ArrayIrValue<Array>,
            ArrayIrOperation<Array>,
        >::new()))
        .with_rule(DifferentiationRule::JvpForTranspose);
        let primal = context.primal().lift(ArrayIrValue::Array(Array::scalar(3.0f32).unwrap())).unwrap();
        let transferred = context.primal_to_tangent(primal.clone()).unwrap();
        let unknown = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F32).into(), 0),
        );
        let projected = context.project::<ArrayType>();
        assert_eq!(projected.rule(), DifferentiationRule::JvpForTranspose);
        assert_eq!(projected.clone().rule(), DifferentiationRule::JvpForTranspose);
        assert!(!std::ptr::eq(projected.primal(), projected.tangent()));
        assert_eq!(projected.tangent().parent().import_known(&unknown), Ok(unknown));
        assert_eq!(
            projected.primal_to_tangent(ValueProjection::<ArrayType>::into_projected(primal).unwrap()),
            Ok(ValueProjection::<ArrayType>::into_projected(transferred).unwrap()),
        );
    }

    #[test]
    fn test_differentiation_context_jvp_projected_operation() {
        // The third fixture member is intentionally unrelated to arrays. Its identity JVP proves that the adapter
        // projects both halves of a live dual and lifts the resulting member values back into the heterogeneous family.
        let context =
            DifferentiationContext::fused(EagerContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)),
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3)),
        )
        .unwrap();
        let output = context
            .jvp_projected_operation(&ProjectedMemberOperation::<2>::Identity, &[input])
            .unwrap()
            .remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(7)));
        assert!(matches!(tangent, MaybeZero::Value(ProjectedProgramValue::Third(ProjectedMemberValue::<2>(3))),));

        // Structural zeros cross the same adapter as types and therefore do not stage or materialize member values.
        let input = DifferentiationDual::new(
            ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)),
            MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),
        )
        .unwrap();
        let output = context
            .jvp_projected_operation(&ProjectedMemberOperation::<2>::Identity, &[input])
            .unwrap()
            .remove(0);
        let (primal, tangent) = output.into_parts();
        assert_eq!(primal, ProjectedProgramValue::Third(ProjectedMemberValue::<2>(11)));
        assert!(matches!(tangent, MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>)),));
    }

    #[test]
    fn test_differentiation_context_jvp_projected_operation_explicit_rule() {
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new())
            .with_rule(DifferentiationRule::JvpForTranspose);
        assert!(matches!(
            context.jvp_projected_operation(&RuleProbeOperation, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "forward with rule `JvpForTranspose`",
        ));
    }

    #[test]
    fn test_differentiation_context_jvp_projected_operation_partitioned() {
        let context = DifferentiationContext::partitioned(PartialEvaluationContext::new(EagerContext::<
            TestValue,
            TestOperation,
        >::new()));
        let primal = context.primal().lift(Array::scalar(0.0f64).unwrap().into()).unwrap();
        let tangent = PartialTracer::new(
            context.tangent().clone(),
            context.tangent().unknown_input(ArrayType::scalar(DataType::F64).into(), 0),
        );
        let input = DifferentiationDual::new(primal, tangent.clone()).unwrap();
        let outputs = context
            .jvp_projected_operation(&ArrayOperation::<Array>::Neg(NegOperation::new()), &[input])
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].primal().value().unwrap().as_known(), Some(&Array::scalar(-0.0f64).unwrap().into()));
        let tangent_output = outputs[0].tangent().as_value().unwrap();
        assert_eq!(tangent_output.context().import_known(&tangent), Ok(tangent.clone()));
        assert!(matches!(outputs[0].primal().context().import_known(&tangent),
            Err(ProgramError::MalformedProgram(message)) if message == "cannot import an unknown value from another \
                partial-evaluation context",
        ));
    }

    #[test]
    fn test_differentiation_context_jvp_for_transpose_projected_operation() {
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        assert!(matches!(
            context.jvp_for_transpose_projected_operation(&RuleProbeOperation, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "reverse with rule `Jvp`",
        ));
    }

    #[test]
    fn test_differentiation_context_jvp_structural_operation() {
        // Structural operations stage their primals and carry structural zero tangents.
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let three = DimensionValue::constant(3).unwrap();
        let four = DimensionValue::constant(4).unwrap();
        let operation = TestOperation::Dimension(DimensionOperation::Add(
            DimensionAddOperation::new(three.r#type().as_ref(), four.r#type().as_ref()).unwrap(),
        ));
        let inputs = [
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Dimension(three)).unwrap(),
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Dimension(four)).unwrap(),
        ];
        let mut outputs = context.jvp_structural_operation(operation, &inputs).unwrap();
        assert_eq!(outputs.len(), 1);
        let (primal, tangent) = outputs.remove(0).into_parts();
        assert!(matches!(primal, ArrayIrValue::Dimension(value) if value.extent() == 7));
        assert!(matches!(tangent, MaybeZero::Zero(_)));

        // Structural operations are staged again without regions, and so operations that declare regions are rejected.
        assert!(matches!(
            context.jvp_structural_operation(TestOperation::Condition(ConditionOperation::new()), &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "structural operation `condition` must not declare regions",
        ));
    }

    #[test]
    fn test_differentiation_context_jvp_structural_mixed_operation() {
        // Every output receives the zero tangent that the provided function stages for it, or a structural zero tangent
        // when the function stages none.
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let r#type = ArrayType::scalar(DataType::F64);
        let operation = TestOperation::Zero(ZeroOperation::new(r#type.clone()));
        let mut outputs = context
            .jvp_structural_mixed_operation(operation.clone(), &[], |context, inputs, r#type| {
                context.stage_member_zero_tangent::<ArrayType>(inputs, r#type)
            })
            .unwrap();
        assert_eq!(outputs.len(), 1);
        let (primal, tangent) = outputs.remove(0).into_parts();
        assert_eq!(primal, ArrayIrValue::Array(Array::scalar(0.0f64).unwrap()));
        assert!(
            matches!(tangent, MaybeZero::Value(ArrayIrValue::Array(value)) if value == Array::scalar(0.0f64).unwrap())
        );
        let mut outputs = context.jvp_structural_mixed_operation(operation, &[], |_, _, _| Ok(None)).unwrap();
        assert!(matches!(outputs.remove(0).into_parts().1, MaybeZero::Zero(ArrayIrType::Array(tangent_type))
            if tangent_type == r#type));

        // Errors of the provided function and of the operation itself are propagated.
        let operation = TestOperation::Zero(ZeroOperation::new(r#type));
        assert!(matches!(
            context.jvp_structural_mixed_operation(operation, &[], |_, _, _| {
                Err(ProgramError::UnsupportedOperation { message: "zero tangent".to_string() }.into())
            }),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "zero tangent",
        ));
        assert!(matches!(
            context.jvp_structural_mixed_operation(
                TestOperation::Condition(ConditionOperation::new()),
                &[],
                |_, _, _| Ok(None),
            ),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "structural operation `condition` must not declare regions",
        ));
    }

    #[test]
    fn test_differentiation_context_stage_member_zero_tangent() {
        // Outputs of the requested member universe receive staged zero tangents, and all other outputs receive none.
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        let array_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        assert_eq!(
            context.stage_member_zero_tangent::<ArrayType>(&[], &array_type),
            Ok(Some(ArrayIrValue::Array(Array::scalar(0.0f32).unwrap()))),
        );
        let dimension_type = TestValue::Dimension(DimensionValue::constant(3).unwrap()).r#type().into_owned();
        assert_eq!(context.stage_member_zero_tangent::<ArrayType>(&[], &dimension_type), Ok(None));
    }

    #[test]
    fn test_differentiation_rule_apply_member() {
        let context = DifferentiationContext::fused(EagerContext::<TestValue, TestOperation>::new());
        assert!(matches!(
            DifferentiationRule::JvpForTranspose.apply_member(&RuleProbeOperation, &context, &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "parent reverse with rule `Jvp`",
        ));
        let context = context.with_rule(DifferentiationRule::JvpForTranspose);
        assert!(matches!(
            DifferentiationRule::Jvp.apply_member(&RuleProbeOperation, &context, &EmptyRegionDriver, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "parent forward with rule `JvpForTranspose`",
        ));
    }

    #[test]
    fn test_differentiation_rule_apply_projected() {
        let context = DifferentiationContext::fused(ArrayIrContext::new());
        let operation = ArrayOperation::<Array>::Condition(ConditionOperation::new());
        assert_eq!(
            DifferentiationRule::Jvp.apply_projected(&context, &operation, &[]).unwrap_err(),
            DifferentiationError::Program(ProgramError::UnsupportedOperation {
                message: "projected operation `condition` carries regions and cannot be differentiated through its \
                    member family; differentiate it through a family-level carrier for that operation instead"
                    .to_string(),
            }),
        );
        let dimension =
            DifferentiationDual::new_with_zero_tangent(ArrayIrValue::Dimension(DimensionValue::constant(3).unwrap()))
                .unwrap();
        assert_eq!(
            DifferentiationRule::Jvp
                .apply_projected(&context, &ArrayOperation::<Array>::Neg(NegOperation::new()), &[dimension])
                .unwrap_err(),
            DifferentiationError::Program(ProgramError::Type(TypeError::invalid(
                "expected array type but got dimension type",
            ))),
        );
    }

    #[test]
    fn test_transposition_context_transpose_projected_operation() {
        // The third fixture member is intentionally unrelated to arrays. Its identity transpose proves that the
        // adapter records the member rule, converts its program, and splices the live cotangent into the heterogeneous
        // trace without changing its SSA identity or type.
        let mut context =
            TranspositionContext::new(TracingContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let member_type = ProjectedProgramType::Third(ProjectedMemberType::<2>);
        let output_cotangent = context.input(member_type.clone());
        let inputs = &[PartialValue::Unknown(member_type.clone())];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_projected_operation(
                &ProjectedMemberOperation::<2>::Identity,
                inputs,
                &[MaybeZero::Value(output_cotangent.clone())],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert_eq!(cotangents.len(), 1);
        let MaybeZero::Value(input_cotangent) = &cotangents[0] else {
            panic!("identity transpose must preserve its live output cotangent");
        };
        assert_eq!(input_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(input_cotangent.r#type(), output_cotangent.r#type());

        // Known primal inputs are replay values rather than linear inputs, so the member rule returns a structural
        // zero for them. Structural-zero output cotangents also cross the adapter without becoming replay values.
        let known_input = context.input(member_type.clone());
        let inputs = &[PartialValue::Known(known_input)];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_projected_operation(
                &ProjectedMemberOperation::<2>::Identity,
                inputs,
                &[MaybeZero::Value(output_cotangent)],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(
            cotangents.as_slice(),
            [MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>))],
        ));
        let inputs = &[PartialValue::Unknown(member_type.clone())];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_projected_operation(
                &ProjectedMemberOperation::<2>::Identity,
                inputs,
                &[MaybeZero::Zero(member_type)],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(
            cotangents.as_slice(),
            [MaybeZero::Zero(ProjectedProgramType::Third(ProjectedMemberType::<2>))],
        ));
    }

    #[test]
    fn test_transposition_context_transpose_projected_operation_preserves_contribution_provenance() {
        // The member rule submits two identity contributions under different local scopes.
        let mut context =
            TranspositionContext::new(TracingContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let r#type = ProjectedProgramType::Third(ProjectedMemberType::<2>);
        let seed = context.input(r#type.clone());
        let inputs = [PartialValue::Unknown(r#type)];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        let origin = Provenance::scope(ProvenanceScope::new("source"), Provenance::unknown());
        (*context)
            .clone()
            .invoke_with_provenance_origin(origin.clone(), || {
                context.transpose_projected_operation(
                    &ProjectedMemberOperation::<2>::Scoped,
                    &inputs,
                    &[MaybeZero::Value(seed.clone())],
                    &accumulators,
                )
            })
            .unwrap();

        // The member rule adds its contributions before projection. Splicing preserves the fused member-local
        // scopes and source origin, and extracting the accumulated value does not stage another addition.
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        let addition = &builder.instructions()[0];
        assert_eq!(addition.operation().name(), "add");
        assert_eq!(addition.inputs(), &[seed.atom_id().unwrap(), seed.atom_id().unwrap()]);
        assert_eq!(cotangents[0].as_value().unwrap().atom_id().unwrap(), addition.outputs()[0]);
        assert_eq!(
            addition.provenance(),
            &Provenance::fused([
                Provenance::scope(ProvenanceScope::new("first"), origin.clone()),
                Provenance::scope(ProvenanceScope::new("second"), origin),
            ]),
        );
    }

    #[test]
    fn test_transposition_context_transpose_projected_operation_rejects_foreign_builders() {
        let mut context =
            TranspositionContext::new(TracingContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let foreign_context = TracingContext::<ProjectedProgramValue, ProjectedProgramOperation>::new();
        let member_type = ProjectedProgramType::Third(ProjectedMemberType::<2>);
        let local = context.input(member_type.clone());
        let foreign = foreign_context.input(member_type.clone());

        // Both traces assign their first input the same atom index. Checking only that index would silently replace
        // the foreign value with the local input when the member program is spliced.
        assert_eq!(local.atom_id(), foreign.atom_id());
        let accumulators = context.cotangent_accumulators(&[PartialValue::Unknown(member_type.clone())], &[]).unwrap();
        assert!(matches!(
            context.transpose_projected_operation(
                &ProjectedMemberOperation::<2>::Identity,
                &[PartialValue::Known(foreign.clone())],
                &[MaybeZero::Value(local)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
        assert!(matches!(
            context.transpose_projected_operation(
                &ProjectedMemberOperation::<2>::Identity,
                &[PartialValue::Unknown(member_type)],
                &[MaybeZero::Value(foreign)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
    }

    #[test]
    fn test_transposition_context_transpose_mixed_operation() {
        // The member addition is linear in its two member-typed data inputs. No production mixed payload interleaves
        // its data and shape inputs, so this test is what pins that mixed transposition classifies inputs one by one
        // instead of splitting the input list at its first shape input.
        type Context = TracingContext<ProjectedProgramValue, ProjectedProgramOperation>;

        // Data inputs at positions 0 and 2 are delegated to the member rule in that order, and the shape inputs
        // between and after them receive structural zeros in their own member universe.
        let data_type = ProjectedProgramType::Third(ProjectedMemberType::<2>);
        let shape_type = ProjectedProgramType::First(ProjectedMemberType::<0>);
        let mut context = TranspositionContext::new(Context::new());
        let output_cotangent = context.input(data_type.clone());
        let inputs = &[
            PartialValue::Unknown(data_type.clone()),
            PartialValue::Unknown(shape_type.clone()),
            PartialValue::Unknown(data_type.clone()),
            PartialValue::Unknown(shape_type.clone()),
        ];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_mixed_operation(
                &ProjectedMemberOperation::<2>::Add,
                inputs,
                &[MaybeZero::Value(output_cotangent.clone())],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let [
            MaybeZero::Value(first_cotangent),
            MaybeZero::Zero(second_cotangent_type),
            MaybeZero::Value(third_cotangent),
            MaybeZero::Zero(fourth_cotangent_type),
        ] = cotangents.as_slice()
        else {
            panic!("mixed transposition must classify each input individually: {cotangents:?}");
        };
        assert_eq!(first_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(third_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(second_cotangent_type, &shape_type);
        assert_eq!(fourth_cotangent_type, &shape_type);

        // Interleaved known data inputs stay known inputs of the member rule, so their structural zeros are the member
        // rule's own and remain at their input positions.
        let known_data = context.input(data_type.clone());
        let inputs = &[
            PartialValue::Known(known_data),
            PartialValue::Unknown(shape_type.clone()),
            PartialValue::Unknown(data_type.clone()),
        ];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_mixed_operation(
                &ProjectedMemberOperation::<2>::Add,
                inputs,
                &[MaybeZero::Value(output_cotangent.clone())],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        assert!(matches!(
            cotangents.as_slice(),
            [
                MaybeZero::Zero(ProjectedProgramType::Third(_)),
                MaybeZero::Zero(ProjectedProgramType::First(_)),
                MaybeZero::Value(_),
            ],
        ));

        // The data-inputs-first arrangement every current payload uses degenerates to the same result, so the
        // classification is a strict generalization of splitting the input list at its first shape input.
        let inputs = &[
            PartialValue::Unknown(data_type.clone()),
            PartialValue::Unknown(data_type),
            PartialValue::Unknown(shape_type.clone()),
        ];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        context
            .transpose_mixed_operation(
                &ProjectedMemberOperation::<2>::Add,
                inputs,
                &[MaybeZero::Value(output_cotangent.clone())],
                &accumulators,
            )
            .unwrap();
        let cotangents = context.take_cotangents(&accumulators).unwrap();
        let [MaybeZero::Value(first_cotangent), MaybeZero::Value(second_cotangent), MaybeZero::Zero(shape)] =
            cotangents.as_slice()
        else {
            panic!("prefix-arranged mixed transposition must keep its previous result: {cotangents:?}");
        };
        assert_eq!(first_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(second_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(shape, &shape_type);
    }

    #[test]
    fn test_differentiate_in_parent_at() {
        let mut invoked = false;
        let result = differentiate_in_parent_at(Vec::<ArrayTracer>::new()).jvp(Vec::new(), |inputs| {
            invoked = true;
            Ok(inputs)
        });
        assert_eq!(
            result,
            Err(DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "differentiation in a parent context requires at least one active input leaf".to_string(),
            })),
        );
        assert!(!invoked);
    }

    #[test]
    fn test_parent_differentiation_builder_jvp() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar, vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let primal = (x, vec![y]);
            let ((value, values), (tangent, tangents)) = differentiate_in_parent_at(primal.clone())
                .jvp(primal, |(x, mut ys)| {
                    let y = ys.remove(0);
                    // A first-class extent travels through the parent while arrays stay projected.
                    let count =
                        y.clone().into_value().dimension_size(0)?.to_scalar()?.convert_element_type(DataType::F32)?;
                    let count = ValueProjection::<ArrayType>::into_projected(count)?;
                    Ok((x.clone() * x, vec![y * count]))
                })
                .unwrap();
            assert_eq!((values.len(), tangents.len()), (1, 1));
            Ok(vec![value, values[0].clone(), tangent, tangents[0].clone()])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(2f32).unwrap(), Array::from_elements(vector.clone(), &[3f32, 4.0]).unwrap()]
            ),
            vec![
                Array::scalar(4f32).unwrap(),
                Array::from_elements(vector.clone(), &[6f32, 8.0]).unwrap(),
                Array::scalar(8f32).unwrap(),
                Array::from_elements(vector, &[6f32, 8.0]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_linearize() {
        let scalar = ArrayType::scalar(DataType::F32);
        let program = trace_arrays(vec![scalar.clone(), scalar], |mut inputs| {
            let seed = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (value, pushforward) =
                differentiate_in_parent_at(x.clone()).linearize(|x| Ok(x.clone() * x.clone() * x)).unwrap();
            Ok(vec![value, pushforward.apply(seed)?, pushforward.apply(x)?])
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::scalar(3f32).unwrap(), Array::scalar(2f32).unwrap()]),
            vec![Array::scalar(27f32).unwrap(), Array::scalar(54f32).unwrap(), Array::scalar(81f32).unwrap()],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_vjp() {
        let scalar = ArrayType::scalar(DataType::F32);
        let program = trace_arrays(vec![scalar.clone(), scalar], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (values, pullback) = differentiate_in_parent_at((x.clone(), vec![y]))
                .vjp(|(x, mut ys)| Ok(vec![x * ys.remove(0)]))
                .unwrap();
            assert_eq!(values.len(), 1);
            let (dx, dys) = pullback.apply_materialized(vec![x])?;
            assert_eq!(dys.len(), 1);
            Ok(vec![values[0].clone(), dx, dys[0].clone()])
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::scalar(2f32).unwrap(), Array::scalar(3f32).unwrap()]),
            vec![Array::scalar(6f32).unwrap(), Array::scalar(6f32).unwrap(), Array::scalar(4f32).unwrap()],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_value_and_gradient() {
        let scalar = ArrayType::scalar(DataType::F32);
        let program = trace_arrays(vec![scalar.clone(), scalar], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (value, (dx, dys)) = differentiate_in_parent_at((x, vec![y]))
                .value_and_gradient(|(x, mut ys)| Ok(x * ys.remove(0)))
                .unwrap();
            assert_eq!(dys.len(), 1);
            Ok(vec![value, dx, dys[0].clone()])
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::scalar(2f32).unwrap(), Array::scalar(3f32).unwrap()]),
            vec![Array::scalar(6f32).unwrap(), Array::scalar(3f32).unwrap(), Array::scalar(2f32).unwrap()],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_gradient() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar, vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (dx, dy) = differentiate_in_parent_at((x, y)).gradient(|(x, _)| Ok(x.clone() * x)).unwrap();
            Ok(vec![dx, dy])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(2f32).unwrap(), Array::from_elements(vector.clone(), &[3f32, 4.0]).unwrap()]
            ),
            vec![Array::scalar(4f32).unwrap(), Array::from_elements(vector, &[0f32, 0.0]).unwrap()],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_jacobian_forward() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar.clone(), vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let jacobian = differentiate_in_parent_at((x, vec![y]))
                .jacobian_forward(|(x, mut ys)| Ok(x * ys.remove(0)))
                .unwrap();
            assert_eq!(jacobian.input_type(), &(scalar.clone(), vec![vector.clone()]));
            assert_eq!(jacobian.output_type(), &vector);
            let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].input_path().to_string(), "$.0");
            assert_eq!(blocks[1].input_path().to_string(), "$.1[0]");
            Ok(jacobian.into_values())
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(2f32).unwrap(), Array::from_elements(vector.clone(), &[3f32, 4.0]).unwrap()]
            ),
            vec![
                Array::from_elements(vector, &[3f32, 4.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[2f32, 0.0, 0.0, 2.0]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_jacobian_reverse() {
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![vector.clone()], |mut inputs| {
            let jacobian = differentiate_in_parent_at(inputs.remove(0))
                .jacobian_reverse(|x| Ok((x.reduce(&[0], ReductionKind::Sum)?, vec![x.clone() * x])))
                .unwrap();
            assert_eq!(jacobian.input_type(), &vector);
            assert_eq!(jacobian.output_type(), &(ArrayType::scalar(DataType::F32), vec![vector.clone()]));
            let blocks = jacobian.iter_blocks().collect::<Vec<_>>();
            assert_eq!(blocks.len(), 2);
            assert_eq!(blocks[0].output_path().to_string(), "$.0");
            assert_eq!(blocks[1].output_path().to_string(), "$.1[0]");
            Ok(jacobian.into_values())
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::from_elements(vector.clone(), &[2f32, 3.0]).unwrap()]),
            vec![
                Array::from_elements(vector, &[1f32, 1.0]).unwrap(),
                Array::from_elements(ArrayType::new_static(DataType::F32, [2, 2]), &[4f32, 0.0, 0.0, 6.0]).unwrap(),
            ],
        );
    }

    #[test]
    fn test_parent_differentiation_builder_hessian() {
        let scalar = ArrayType::scalar(DataType::F32);
        let program = trace_arrays(vec![scalar.clone(), scalar.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let hessian = differentiate_in_parent_at((x, y)).hessian(|(x, y)| Ok(x.clone() * x * y)).unwrap();
            assert_eq!(hessian.input_types(), &(scalar.clone(), scalar.clone()));
            assert_eq!(hessian.output_types(), &scalar);
            let blocks = hessian.iter_blocks().collect::<Vec<_>>();
            assert_eq!(blocks.len(), 4);
            assert_eq!(blocks[1].first_input_path().to_string(), "$.0");
            assert_eq!(blocks[1].second_input_path().to_string(), "$.1");
            Ok(hessian.into_values())
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::scalar(2f32).unwrap(), Array::scalar(3f32).unwrap()]),
            vec![
                Array::scalar(6f32).unwrap(),
                Array::scalar(4f32).unwrap(),
                Array::scalar(4f32).unwrap(),
                Array::scalar(0f32).unwrap()
            ],
        );
    }

    #[test]
    fn test_parent_pushforward_inner() {
        let program = trace_arrays(vec![ArrayType::scalar(DataType::F32)], |mut inputs| {
            let x = inputs.remove(0);
            let (_, pushforward) = differentiate_in_parent_at(x.clone()).linearize(|x| Ok(x.clone() * x)).unwrap();
            pushforward
                .inner()
                .apply(vec![x.into_value()])?
                .into_iter()
                .map(ValueProjection::<ArrayType>::into_projected)
                .collect::<Result<Vec<_>, _>>()
                .map_err(Into::into)
        });
        assert_eq!(interpret_arrays(&program, vec![Array::scalar(3f32).unwrap()]), vec![Array::scalar(18f32).unwrap()]);
    }

    #[test]
    fn test_parent_pushforward_apply() {
        let program = trace_arrays(vec![ArrayType::scalar(DataType::F32)], |mut inputs| {
            let x = inputs.remove(0);
            let (_, pushforward) = differentiate_in_parent_at(vec![x.clone()])
                .linearize(|mut xs| {
                    let x = xs.remove(0);
                    Ok((x.clone(), vec![x.clone() * x]))
                })
                .unwrap();
            assert_eq!(pushforward.apply(Vec::new()), Err(ProgramError::InvalidInputCount { expected: 1, actual: 0 }));
            let (linear, nonlinear) = pushforward.apply(vec![x])?;
            assert_eq!(nonlinear.len(), 1);
            Ok(vec![linear, nonlinear[0].clone()])
        });
        assert_eq!(
            interpret_arrays(&program, vec![Array::scalar(3f32).unwrap()]),
            vec![Array::scalar(3f32).unwrap(), Array::scalar(18f32).unwrap()]
        );
    }

    #[test]
    fn test_parent_pushforward_apply_in_context() {
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let matrix = ArrayType::new_static(DataType::F32, [2, 2]);
        let program = trace_arrays(vec![vector.clone(), matrix.clone()], |mut inputs| {
            let seeds = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pushforward) = differentiate_in_parent_at(x).linearize(|x| Ok(x.clone() * x)).unwrap();
            let tangents = batch_in_parent(
                |seed| pushforward.apply_in_context(&seed.domain().parent().clone(), seed),
                seeds,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
            .unwrap();
            Ok(vec![tangents])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![
                    Array::from_elements(vector, &[2f32, 3.0]).unwrap(),
                    Array::from_elements(matrix.clone(), &[1f32, 2.0, 3.0, 4.0]).unwrap()
                ]
            ),
            vec![Array::from_elements(matrix, &[4f32, 12.0, 12.0, 24.0]).unwrap()]
        );
    }

    #[test]
    fn test_parent_pullback_inner() {
        let program = trace_arrays(vec![ArrayType::scalar(DataType::F32)], |mut inputs| {
            let x = inputs.remove(0);
            let (_, pullback) = differentiate_in_parent_at(x.clone()).vjp(|x| Ok(x.clone() * x)).unwrap();
            let mut cotangents = pullback.inner().apply(vec![x.into_value()])?;
            let MaybeZero::Value(cotangent) = cotangents.remove(0) else {
                panic!("a connected input must have a live cotangent");
            };
            assert!(cotangents.is_empty());
            Ok(vec![ValueProjection::<ArrayType>::into_projected(cotangent)?])
        });
        assert_eq!(interpret_arrays(&program, vec![Array::scalar(3f32).unwrap()]), vec![Array::scalar(18f32).unwrap()]);
    }

    #[test]
    fn test_parent_pullback_apply() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar, vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) =
                differentiate_in_parent_at((x.clone(), vec![y])).vjp(|(x, _)| Ok(x.clone() * x)).unwrap();
            let (dx, dys) = pullback.apply(x)?;
            assert_eq!(dys, vec![MaybeZero::Zero(vector.clone())]);
            let MaybeZero::Value(dx) = dx else {
                panic!("a connected input must have a live cotangent");
            };
            Ok(vec![dx])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(3f32).unwrap(), Array::from_elements(vector, &[4f32, 5.0]).unwrap()]
            ),
            vec![Array::scalar(18f32).unwrap()]
        );
    }

    #[test]
    fn test_parent_pullback_apply_in_context() {
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let matrix = ArrayType::new_static(DataType::F32, [2, 2]);
        let program = trace_arrays(vec![vector.clone(), matrix.clone()], |mut inputs| {
            let seeds = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) = differentiate_in_parent_at(x).vjp(|x| Ok(x.clone() * x)).unwrap();
            let cotangents = batch_in_parent(
                |seed| {
                    let MaybeZero::Value(cotangent) =
                        pullback.apply_in_context(&seed.domain().parent().clone(), seed)?
                    else {
                        panic!("a connected input must have a live cotangent");
                    };
                    Ok(cotangent)
                },
                seeds,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
            .unwrap();
            Ok(vec![cotangents])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![
                    Array::from_elements(vector, &[2f32, 3.0]).unwrap(),
                    Array::from_elements(matrix.clone(), &[1f32, 2.0, 3.0, 4.0]).unwrap()
                ]
            ),
            vec![Array::from_elements(matrix, &[4f32, 12.0, 12.0, 24.0]).unwrap()]
        );
    }

    #[test]
    fn test_parent_pullback_apply_materialized() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar, vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) =
                differentiate_in_parent_at((x.clone(), vec![y])).vjp(|(x, _)| Ok(x.clone() * x)).unwrap();
            let (dx, mut dys) = pullback.apply_materialized(x)?;
            assert_eq!(dys.len(), 1);
            Ok(vec![dx, dys.remove(0)])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(3f32).unwrap(), Array::from_elements(vector.clone(), &[4f32, 5.0]).unwrap()]
            ),
            vec![Array::scalar(18f32).unwrap(), Array::from_elements(vector, &[0f32, 0.0]).unwrap()]
        );
    }

    #[test]
    fn test_parent_pullback_apply_materialized_in_context() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar.clone(), scalar, vector.clone()], |mut inputs| {
            let seeds = inputs.pop().unwrap();
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) = differentiate_in_parent_at((x, y)).vjp(|(x, _)| Ok(x.clone() * x)).unwrap();
            let (dx, dy) = batch_in_parent(
                |seed| pullback.apply_materialized_in_context(&seed.domain().parent().clone(), seed),
                seeds,
                BatchAxis::new(0),
                (BatchAxis::new(0), BatchAxis::new(0)),
                BatchAxisSpecification::default(),
            )
            .unwrap();
            Ok(vec![dx, dy])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![
                    Array::scalar(3f32).unwrap(),
                    Array::scalar(4f32).unwrap(),
                    Array::from_elements(vector.clone(), &[1f32, 2.0]).unwrap()
                ]
            ),
            vec![
                Array::from_elements(vector.clone(), &[6f32, 12.0]).unwrap(),
                Array::from_elements(vector, &[0f32, 0.0]).unwrap()
            ]
        );
    }

    #[test]
    fn test_parent_pullback_apply_with_destinations() {
        let scalar = ArrayType::scalar(DataType::F32);
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let program = trace_arrays(vec![scalar, vector.clone()], |mut inputs| {
            let y = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) =
                differentiate_in_parent_at((x.clone(), vec![y])).vjp(|(x, _)| Ok(x.clone() * x)).unwrap();

            // A returned cotangent keeps its projected type, while an ignored one has no result.
            let (dx, dys) = pullback.apply_with_destinations(
                CotangentSeed::Value(x),
                (CotangentDestination::Return, vec![CotangentDestination::Ignore]),
            )?;
            assert_eq!(dys, vec![None]);
            let Some(MaybeZero::Value(dx)) = dx else {
                panic!("a connected returned input must have a live cotangent");
            };

            // A structural zero seed returns a structural zero cotangent of the projected type.
            let (zero, _) = pullback.apply_with_destinations(
                CotangentSeed::Zero,
                (CotangentDestination::Return, vec![CotangentDestination::Ignore]),
            )?;
            assert_eq!(zero, Some(MaybeZero::Zero(ArrayType::scalar(DataType::F32))));
            Ok(vec![dx])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![Array::scalar(3f32).unwrap(), Array::from_elements(vector, &[4f32, 5.0]).unwrap()]
            ),
            vec![Array::scalar(18f32).unwrap()]
        );
    }

    #[test]
    fn test_parent_pullback_apply_with_destinations_in_context() {
        let vector = ArrayType::new_static(DataType::F32, [2]);
        let matrix = ArrayType::new_static(DataType::F32, [2, 2]);
        let program = trace_arrays(vec![vector.clone(), matrix.clone()], |mut inputs| {
            let seeds = inputs.pop().unwrap();
            let x = inputs.pop().unwrap();
            let (_, pullback) = differentiate_in_parent_at(x).vjp(|x| Ok(x.clone() * x)).unwrap();
            let cotangents = batch_in_parent(
                |seed| {
                    let Some(MaybeZero::Value(cotangent)) = pullback.apply_with_destinations_in_context(
                        &seed.domain().parent().clone(),
                        CotangentSeed::Value(seed),
                        CotangentDestination::Return,
                    )?
                    else {
                        panic!("a connected returned input must have a live cotangent");
                    };
                    Ok(cotangent)
                },
                seeds,
                BatchAxis::new(0),
                BatchAxis::new(0),
                BatchAxisSpecification::default(),
            )
            .unwrap();
            Ok(vec![cotangents])
        });
        assert_eq!(
            interpret_arrays(
                &program,
                vec![
                    Array::from_elements(vector, &[2f32, 3.0]).unwrap(),
                    Array::from_elements(matrix.clone(), &[1f32, 2.0, 3.0, 4.0]).unwrap()
                ]
            ),
            vec![Array::from_elements(matrix, &[4f32, 12.0, 12.0, 24.0]).unwrap()]
        );
    }
}
