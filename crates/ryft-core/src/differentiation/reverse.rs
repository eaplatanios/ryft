use std::borrow::Cow;
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};
use std::rc::Rc;
use std::sync::Arc;

use ryft_macros::Parameter;

use crate::contexts::{Context, StagingContext};
use crate::differentiation::forward::{
    DifferentiableOperation, ForwardModeDifferentiate, LinearizationTracer, Pushforward,
};
use crate::differentiation::types::DifferentiableType;
use crate::differentiation::zeros::{ResidualZeroProvider, ZeroSpaceBoundaryReconstruction, ZeroSpaceBoundaryRole};
use crate::differentiation::{DifferentiationBoundaryPosition, DifferentiationError};
use crate::errors::MaybeFallible;
use crate::macros::{check_builders, check_count};
use crate::operations::{
    AddOperation, OneOperation, ReferenceAddUpdateOperation, ReferenceFreezeOperation, ReferenceNewOperation, Zero,
};
use crate::parameters::{Parameter, Parameterized, ParameterizedFamily, Placeholder};
use crate::partial::{PartialEvaluationContext, PartialValue, PartiallyEvaluatableOperation};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::{
    Atom, AtomId, BindingRegionDriver, EffectClass, EmptyRegionDriver, Instruction, InstructionId, MaybeZero,
    Operation, OperationProjection, OperationProvider, Program, ProgramBuilder, ProgramError, Provenance,
    ReferenceAccessMode, ReferenceAliasKind, ReferenceAnalysis, ReferenceBoundary, ReferenceRoot,
    ReferenceViewOperation, Region, RegionDriver, RegionRef, RegionReplayMappings, ReplayRegionDriver, Type, TypeError,
    TypeIdentityPosition, Typed, Value, ValueId, ValueProjection, ViewSymbolBinding,
};
use crate::tracing::{Tracer, TracingContext};

/// Cotangent seed for one flattened primal output leaf of a [`Pullback`] application, aligned leaf-for-leaf with
/// the differentiated closure's output structure. Non-reference output leaves are seeded with a [`Value`](Self::Value)
/// of their cotangent type, while reference-typed output leaves carry [`NoCotangent`](Self::NoCotangent) since a
/// reference output forwards an input root, and the cotangent of that root's state lives in the input's
/// [`CotangentDestination`] rather than in a seed. [`Pullback::apply_with_destinations`] rejects every other pairing.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum CotangentSeed<V> {
    /// Cotangent value of a non-reference primal output leaf, typed with that leaf's cotangent type.
    Value(V),

    /// No cotangent, for a reference-typed primal output leaf whose state cotangent is owned by an input destination.
    NoCotangent,
}

/// Cotangent _destination_ for one flattened primal input leaf of a [`Pullback`] application, aligned leaf-for-leaf
/// with the differentiated closure's input structure. A non-reference input leaf either [`returns`](Self::Return) its
/// cotangent, accumulates it into a caller-owned [`Reference`](Self::Reference), or [`ignores`](Self::Ignore) it. A
/// reference-typed input leaf uses a caller-owned cotangent reference or ignores its state cotangent.
/// [`Pullback::apply_with_destinations`] rejects every other pairing. The [`CotangentDestinationKind`]s form the
/// structural mask that selects the retained transposition. Refer to the documentation of [`CotangentDestinationKind`]
/// for more information.
#[derive(Clone, Debug, PartialEq, Eq, Parameter)]
pub enum CotangentDestination<V> {
    /// Returns the input cotangent as a non-reference value in the pullback's output structure.
    Return,

    /// The input cotangent is stored in a caller-owned reference. For a non-reference input of type `T`, it has type
    /// `ref<cotangent(T)>` and the computed cotangent is added to its existing contents. For a primal input of type
    /// `ref<T>`, it has the same referent cotangent type and carries the state adjoint through the reverse computation.
    /// It holds the cotangent of the reference's post-execution state when the pullback starts. The transposes of the
    /// `reference_write` and `reference_swap` operations may replace it. It holds the cotangent of the pre-execution
    /// state when the pullback returns. It must not alias any reference bound at the primal boundary of the
    /// differentiated closure or any other destination of the same application.
    Reference(V),

    /// The input cotangent is discarded (i.e., ignored). For a reference-typed input this means that the caller does
    /// not want the pre-execution state cotangent and not that state adjoints are unnecessary. The pullback still
    /// accumulates through an internal cotangent reference so that values stored into the reference receive their
    /// cotangents.
    Ignore,
}

impl<V> CotangentDestination<V> {
    /// Returns the [`CotangentDestinationKind`] of this [`CotangentDestination`].
    #[inline]
    pub fn kind(&self) -> CotangentDestinationKind {
        match self {
            Self::Return => CotangentDestinationKind::Return,
            Self::Reference(_) => CotangentDestinationKind::Reference,
            Self::Ignore => CotangentDestinationKind::Ignore,
        }
    }
}

/// Structural kind of a [`CotangentDestination`], without the runtime reference value. For non-reference inputs
/// this selects whether the pullback returns a cotangent value, adds contributions directly into a caller-provided
/// reference, or omits the cotangent. Reference-typed primal inputs have separate state-adjoint semantics: a
/// caller-provided reference carries the state cotangent backward through reads and writes, while an ignored state
/// cotangent uses internal storage whenever intermediate state still contributes to another input's gradient.
///
/// These kinds participate in the retained transposition's cache key. Applications with the same kinds reuse a
/// program even when they supply different buffers. Changing a kind selects a different program boundary.
/// Refer to [`Program::transpose_with_respect_to`] for information on the input and output ordering.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CotangentDestinationKind {
    /// Returns a non-reference input's cotangent value. This is the default for non-reference inputs and is invalid
    /// for reference-typed primal inputs, whose state cotangents require reference storage.
    Return,

    /// Accepts a caller-owned cotangent reference as a pullback input. Value contributions are added into it and
    /// produce no cotangent output. For a reference-typed primal input, the buffer carries its state adjoint and is
    /// also returned by identity. This is the default for reference-typed primal inputs.
    Reference,

    /// Omits the input cotangent from the boundary. Non-reference inputs need no accumulation. Reference-typed primal
    /// inputs may still need an internal state accumulator to compute other input cotangents that is discarded
    /// after use.
    Ignore,
}

/// Information about the cotangent destinations of an [`Instruction`] whose [`Region`] inputs mirror its operands.
/// Non-reference operands follow their accumulation handles: requested values use [`CotangentDestinationKind::Return`],
/// supplied buffers use [`CotangentDestinationKind::Reference`], and unrequested gradients use
/// [`CotangentDestinationKind::Ignore`]. Known operands remain as [`CotangentDestinationKind::Return`]
/// placeholders and are not transposed.
///
/// Reference-state operands have a different contract. A live state cotangent passes its reference into the nested
/// region and returns that reference by identity. An unused state cotangent uses [`CotangentDestinationKind::Ignore`],
/// allowing the nested region to allocate temporary state only if needed. An gradient buffer for a non-reference input
/// is updated without producing an output. [`returns_cotangent`](Self::returns_cotangent) records this distinction for
/// reconstruction of the nested results.
///
/// [`kinds`](Self::kinds) and [`references`](Self::references) both follow operand order. For example, for the
/// `condition` operation, the known predicate is omitted when selecting the branch boundary and the remaining operands
/// keep that order. Reference-state liveness can require a nested transpose even when every non-reference output
/// cotangent is zero.
#[derive(Clone, Debug)]
pub struct CotangentDestinations<V> {
    /// [`CotangentDestinationKind`] of every operand, in operand order.
    kinds: Vec<CotangentDestinationKind>,

    /// Cotangent references of the [`Reference`](CotangentDestinationKind::Reference)-kind operands, in operand order.
    references: Vec<V>,

    /// Contains a boolean value for each operand specifying whether it carries reference state,
    /// rather than a non-reference gradient that may use a buffer.
    reference_inputs: Vec<bool>,
}

impl<V> CotangentDestinations<V> {
    /// Creates a new [`CotangentDestinations`] instance from the provided operand-ordered components.
    /// Callers supply one `reference_inputs` entry per [`CotangentDestinationKind`] and one reference per
    /// [`Reference`](CotangentDestinationKind::Reference)-kind operand, preserving operand order.
    ///
    /// # Parameters
    ///
    ///   - `kinds`: Destination kind of every operand.
    ///   - `references`: Cotangent references for the `Reference`-kind operands.
    ///   - `reference_inputs`: Whether each operand carries reference state rather than a non-reference value.
    fn new(kinds: Vec<CotangentDestinationKind>, references: Vec<V>, reference_inputs: Vec<bool>) -> Self {
        Self { kinds, references, reference_inputs }
    }

    /// Creates a [`CotangentDestinations`] instance for a reference-free boundary, using
    /// [`CotangentDestinationKind::Return`] for requested gradients for non-reference inputs and
    /// [`CotangentDestinationKind::Ignore`] for the others. The mask iterator follows operand order. An all-`true`
    /// iterator requests the conservative value-returning boundary used for `scan` operation carries and scanned
    /// inputs.
    ///
    /// # Parameters
    ///
    ///   - `cotangent_mask`: One boolean per operand, in operand order, specifying whether its gradient is requested.
    ///     `true` selects [`CotangentDestinationKind::Return`], so the nested transpose returns that operand's
    ///     cotangent. `false` selects [`CotangentDestinationKind::Ignore`], so it returns no cotangent for that
    ///     operand. For example, `[true, false, true]` requests gradients for the first and third operands. This
    ///     describes which gradients are needed, and not whether their values are nonzero; a requested gradient
    ///     may still be zero.
    pub fn without_references<R: IntoIterator<Item = bool>>(cotangent_mask: R) -> Self {
        let kinds = cotangent_mask
            .into_iter()
            .map(|needed| if needed { CotangentDestinationKind::Return } else { CotangentDestinationKind::Ignore })
            .collect::<Vec<_>>();
        let reference_inputs = vec![false; kinds.len()];
        Self::new(kinds, Vec::new(), reference_inputs)
    }

    /// Returns the [`CotangentDestinationKind`] of the operand at `index`.
    #[inline]
    pub fn kind(&self, index: usize) -> CotangentDestinationKind {
        self.kinds[index]
    }

    /// Returns the [`CotangentDestinationKind`] of every operand, in operand order.
    #[inline]
    pub fn kinds(&self) -> &[CotangentDestinationKind] {
        self.kinds.as_slice()
    }

    /// Returns the cotangent references of the [`Reference`](CotangentDestinationKind::Reference)-kind operands,
    /// in operand order.
    #[inline]
    pub fn references(&self) -> &[V] {
        self.references.as_slice()
    }

    /// Returns whether the operand at `index` carries reference state whose cotangent is handled by the transpose.
    /// Known reference operands and non-reference values whose gradients use reference buffers return `false`.
    #[inline]
    pub fn is_reference_input(&self, index: usize) -> bool {
        self.reference_inputs[index]
    }

    /// Returns whether the operand at `index` has a corresponding output in the nested transposed program. A
    /// [`Return`](CotangentDestinationKind::Return) destination produces the gradient value. A reference-state input
    /// with a [`Reference`](CotangentDestinationKind::Reference) destination returns the same accumulator reference it
    /// received, so the enclosing operation can pass that state onward. A non-reference value with a `Reference`
    /// destination instead adds its gradient into the supplied buffer without returning an output. An
    /// [`Ignore`](CotangentDestinationKind::Ignore) destination produces no output.
    ///
    /// Callers use this when matching the nested program's outputs back to the operands being transposed. Known
    /// operands are excluded separately as their `Return` entries are placeholders and not requests for gradients.
    #[inline]
    pub fn returns_cotangent(&self, index: usize) -> bool {
        self.kind(index) == CotangentDestinationKind::Return
            || (self.is_reference_input(index) && self.kind(index) == CotangentDestinationKind::Reference)
    }

    /// Returns whether any reference-state input has a [`Reference`](CotangentDestinationKind::Reference) destination.
    /// Such a destination passes a state cotangent through the nested transposed program, so that program may still
    /// need to run even when every non-reference output cotangent is zero. This checks the destination kind, not
    /// whether the buffer's contents are nonzero. Gradient buffers for non-reference inputs and reference-state inputs
    /// with an [`Ignore`](CotangentDestinationKind::Ignore) destination do not satisfy this check.
    #[inline]
    pub fn has_reference_state_destinations(&self) -> bool {
        self.reference_inputs
            .iter()
            .zip(&self.kinds)
            .any(|(reference, kind)| *reference && *kind == CotangentDestinationKind::Reference)
    }
}

/// Cotangent accumulator of a reference root during transposition. The accumulator of a root is the cotangent
/// reference `ref<cotangent(T)>` into which the reverse sweep accumulates the cotangents of the values read
/// from the root's state (through the `reference_read` and `reference_freeze` operations) and out of which the
/// transposes of the stores (e.g., the `reference_write`, `reference_swap`, and `reference_add_update` operations)
/// take the cotangents of the stored values. Every linear reference input of a transposed region has an accumulator,
/// either [`Allocated`](Self::Allocated) from a caller-supplied cotangent reference (i.e., a
/// [`CotangentDestinationKind::Reference`] input) or [`Unallocated`](Self::Unallocated) until its first use (i.e., a
/// [`CotangentDestinationKind::Ignore`] input, whose final contents are discarded), and so does every linear allocation
/// (i.e., through the `reference_new` operation), whose accumulator is [`Unallocated`](Self::Unallocated) until first
/// use and is frozen into the cotangent of the initial value when the sweep reaches the allocation.
#[derive(Clone, Debug)]
pub enum CotangentReferenceAccumulator<V: Typed> {
    /// No cotangent has reached the root yet. The accumulator is allocated lazily on first use with a zero initial
    /// value. A root whose accumulator is still unallocated when the sweep finishes with it contributes a symbolic
    /// zero and emits nothing.
    Unallocated {
        /// Type `ref<cotangent(T)>` of the cotangent reference that would be allocated.
        cotangent_type: V::Type,
    },

    /// The root's cotangent reference, either supplied by the caller or allocated on first use.
    Allocated {
        /// Cotangent reference of type `ref<cotangent(T)>`.
        reference: V,
    },
}

impl<V: Typed> CotangentReferenceAccumulator<V> {
    /// Returns whether this [`CotangentReferenceAccumulator`] holds an already allocated cotangent reference.
    #[inline]
    pub fn is_allocated(&self) -> bool {
        matches!(self, Self::Allocated { .. })
    }

    /// Returns the cotangent reference of this [`CotangentReferenceAccumulator`],
    /// or [`None`] if it has not been allocated yet.
    #[inline]
    pub fn reference(&self) -> Option<&V> {
        match self {
            Self::Unallocated { .. } => None,
            Self::Allocated { reference } => Some(reference),
        }
    }
}

impl<
    V: Value<Type: DifferentiableType>,
    O: Operation<Type = V::Type>
        + ResidualZeroProvider<V::Type>
        + OperationProvider<V::Type, ReferenceNewOperation<V::Type, V::Type>, Operation = O>,
> CotangentReferenceAccumulator<Tracer<TracingContext<V, O>>>
{
    /// Returns this [`CotangentReferenceAccumulator`]'s cotangent reference, allocating it with a zero initial value
    /// first if it has not been allocated yet. The zero referent is materialized through the operation family's
    /// [`ResidualZeroProvider`] implementation and then allocated through its [`OperationProvider`] implementation,
    /// both in `context`.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`TracingContext`] in which to stage the zero initial value and allocate the cotangent reference.
    ///     If the accumulator is already allocated, its existing reference is returned without staging any operations.
    ///   - `sources`: Values available in `context` from which to obtain the runtime quantities needed to construct the
    ///     zero initial value, such as dynamic dimensions. They are searched in order for each residual declared by the
    ///     referent's [`ResidualZeroProvider`]. They are unused if no residuals are required or the accumulator is
    ///     already allocated.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::UnsupportedOperation`] when the accumulator's type does not project onto a referent
    /// through [`Type::referent`], and propagates zero materialization and allocation errors otherwise.
    fn allocate_in(
        &mut self,
        context: &TracingContext<V, O>,
        sources: &[Tracer<TracingContext<V, O>>],
    ) -> Result<&Tracer<TracingContext<V, O>>, DifferentiationError> {
        match self {
            Self::Unallocated { cotangent_type } => {
                let referent = cotangent_type.referent().ok_or_else(|| ProgramError::UnsupportedOperation {
                    message: format!(
                        "cannot allocate a cotangent reference of type {cotangent_type} because its universe does not \
                         project the type onto a referent"
                    ),
                })?;
                let zero =
                    O::materialize_zero_from_residual_sources(context, MaybeZero::Zero(referent), sources.iter())?;
                let mut references = context.bind(
                    O::provide(ReferenceNewOperation::new(), &[zero.r#type().as_ref()])?,
                    Vec::new(),
                    &[zero],
                )?;
                check_count!("output", references, 1, ProgramError);
                *self = Self::Allocated { reference: references.remove(0) };
                Ok(self.reference().unwrap())
            }
            Self::Allocated { reference } => Ok(reference),
        }
    }
}

impl<V: Typed> Typed for CotangentReferenceAccumulator<V> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        match self {
            Self::Unallocated { cotangent_type } => Cow::Borrowed(cotangent_type),
            Self::Allocated { reference } => reference.r#type(),
        }
    }
}

/// Handle to one non-reference input's cotangent storage in a [`TranspositionContext`]. Cloning a handle keeps the same
/// underlying storage, so repeated operands can each contribute to one gradient value. Handles are valid only in their
/// [`TranspositionContext`]s that created them; reference-state adjoints use the context's separate reference
/// operations instead.
#[derive(Clone, Debug)]
pub struct CotangentAccumulator {
    /// Identity token shared with the [`TranspositionContext`] that owns this handle's storage slot. The context
    /// checks pointer equality with this token before accessing the slot, rejecting handles from other contexts.
    /// Holding only the token keeps the context available for mutable access without retaining its trace and program
    /// or borrowing the context for the handle's lifetime. The token outlives a dropped context while handles remain,
    /// preventing an old handle from accidentally identifying a new context whose allocation reuses the same address.
    context_identity: Rc<()>,

    /// Index into the owning context's value cotangent storage.
    storage_index: usize,

    /// Whether a value cotangent is requested for this input.
    needed: bool,
}

impl CotangentAccumulator {
    /// Returns whether this [`CotangentAccumulator`] needs a value cotangent. Rules can check this before constructing
    /// an expensive contribution. Known operands and reference-state operands do not request non-reference cotangents.
    #[inline]
    pub fn is_needed(&self) -> bool {
        self.needed
    }

    /// Returns the underlying caller-provided gradient buffer for a non-reference input, if this accumulator has one.
    /// Value accumulators never allocate a reference merely because this function is called. Using another context's
    /// handle returns an error. A rule using this buffer must preserve the existing cotangent and add a contribution
    /// independent of the buffer's contents. It must not consume the reference or discard earlier contributions. A
    /// read-modify-write sequence is valid when it implements that addition without intervening mutations. Views can
    /// restrict the update to the entries affected by the rule.
    #[inline]
    pub fn reference<V: Value, O: Operation<Type = V::Type>>(
        &self,
        context: &TranspositionContext<V, O>,
    ) -> Result<Option<Tracer<TracingContext<V, O>>>, DifferentiationError> {
        Ok(match context.cotangent_storage(self)? {
            CotangentStorage::Value { .. } => None,
            CotangentStorage::Buffer { reference, .. } => Some(reference.clone()),
        })
    }

    /// Adds a contribution to this input's cotangent. Structural zeros emit no work, ignored contributions are
    /// discarded, and reference destinations are updated immediately. Value contributions are summed as they arrive
    /// and extracted by [`TranspositionContext::take_cotangents`]. Every contribution is validated before any storage
    /// is changed, including contributions to ignored inputs. Live reference-state cotangents cannot use this additive
    /// interface. The first value is reused directly; subsequent additions preserve submission order and combine the
    /// provenance of their contributions.
    pub fn accumulate<V: Value, O: Operation<Type = V::Type> + From<AddOperation<V::Type>>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        contribution: MaybeZero<Tracer<TracingContext<V, O>>>,
    ) -> Result<(), DifferentiationError> {
        // Validate the handle's context identity and the contribution's type before changing any storage,
        // even when this input does not need a cotangent.
        let storage = context.cotangent_storage(self)?;
        if contribution.r#type() != storage.r#type() {
            return Err(TypeError::invalid(format!(
                "cotangent contribution has type {} but its accumulator expects {}",
                contribution.r#type(),
                storage.r#type(),
            ))
            .into());
        }

        // A correctly typed symbolic zero needs no operation. Concrete contributions must belong to this trace;
        // reference-state updates use the context's separate reference functions rather than value addition.
        let MaybeZero::Value(contribution) = contribution else { return Ok(()) };
        check_builders!(context.builder(), contribution.builder())?;
        if storage.r#type().is_reference() {
            return Err(ProgramError::InvalidArgument {
                message: "reference-state cotangents cannot be contributed to a value cotangent accumulator".into(),
            }
            .into());
        }

        // Discard unrequested contributions only after validation so ignored inputs cannot hide invalid rule output.
        if !self.needed {
            return Ok(());
        }

        // Caller-provided buffers receive the contribution immediately through their selected update operation.
        // Otherwise, update the running value immediately so earlier contributions can be released during execution.
        match storage {
            CotangentStorage::Value { cotangent_type, value } => {
                let cotangent_type = cotangent_type.clone();
                let mut provenance = context.provenance();

                // Reuse the first value without an addition; later values extend the sum in submission order.
                let value = if let Some((existing, existing_provenance)) = value {
                    provenance = Provenance::fused([existing_provenance.clone(), provenance]);
                    let output = {
                        let mut builder = context.builder().borrow_mut();
                        let outputs = builder.add_instruction(
                            AddOperation::<V::Type>::new(),
                            Vec::new(),
                            vec![existing.atom_id()?, contribution.atom_id()?],
                            Some(provenance.clone()),
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        outputs[0]
                    };
                    context.tracer(output, None)
                } else {
                    contribution
                };

                // Keep the previous sum intact if staging the addition fails.
                context.cotangent_storage[self.storage_index] =
                    CotangentStorage::Value { cotangent_type, value: Some((value, provenance)) };
            }
            CotangentStorage::Buffer { reference, operation, .. } => {
                context.bind(operation.clone(), Vec::new(), &[reference.clone(), contribution])?;
            }
        }

        Ok(())
    }
}

/// Cotangent storage for non-reference inputs, holding either a running value sum or a caller-supplied buffer. Its
/// [`Typed`] implementation describes the expected contribution type, including when no contributions have arrived; it
/// does not describe the reference used to store those contributions. Both variants accept only additive updates,
/// unlike reference-state accumulators, whose contents may also be taken or replaced by reference transpose rules.
enum CotangentStorage<V: Value, O: Operation<Type = V::Type>> {
    /// Contributions are summed as they arrive, allowing earlier values to be released during execution.
    Value {
        /// Expected cotangent type, also retained for disconnected and ignored inputs.
        cotangent_type: V::Type,

        /// Running sum and combined provenance, or [`None`] before a contribution or after extraction.
        value: Option<(Tracer<TracingContext<V, O>>, Provenance)>,
    },

    /// Contributions are added directly into a caller-supplied reference, preserving its existing contents.
    Buffer {
        /// Expected contribution type, rather than the type of the reference itself.
        cotangent_type: V::Type,

        /// Reference receiving the accumulated contributions.
        reference: Tracer<TracingContext<V, O>>,

        /// Selected [`ReferenceAddUpdateOperation`] instance. Retaining it preserves operation-family dispatch without
        /// adding reference-operation provider bounds to every transpose rule for non-reference values.
        operation: O,
    },
}

impl<V: Value, O: Operation<Type = V::Type>> Typed for CotangentStorage<V, O> {
    type Type = V::Type;

    #[inline]
    fn r#type(&self) -> Cow<'_, Self::Type> {
        match self {
            Self::Value { cotangent_type, .. } | Self::Buffer { cotangent_type, .. } => Cow::Borrowed(cotangent_type),
        }
    }
}

/// Context passed to [`TransposableOperation::transpose`] rules. It wraps the parent [`TracingContext`] into which
/// the transposed [`Program`] is staged and owns the cotangent storage used by those rules. It dereferences to the
/// parent so rules can stage operations directly. Each rule receives [`CotangentAccumulator`] handles aligned with
/// its operands and accesses reference-state cotangents through this context's reference functions.
///
/// [`Self::new`] supports direct rule invocations for non-reference values, including projected member rules, without
/// reference analysis. During program transposition, the engine also installs the source region's [`ReferenceAnalysis`]
/// and reference-state accumulators. The [`TranspositionDriver`] supplies the source region and current instruction and
/// this context retains the analysis and generated values needed across rule invocations, without borrowing the source
/// region or storing a current instruction index.
///
/// # Cotangent Storage
///
/// Value cotangents and reference-state cotangents use separate storage. A value cotangent accumulator maintains a
/// running value sum, adds contributions directly into a caller-supplied gradient buffer, or discards contributions
/// when no cotangent is requested. Its updates are always additive. Handles identify the context that owns their
/// storage: two contexts may share a parent trace while owning unrelated slots, so sharing a parent does not make their
/// handles interchangeable.
///
/// Reference-state cotangents use [`CotangentReferenceAccumulator`]s keyed by the canonical [`ReferenceRoot`] assigned
/// by reference analysis, so aliases share the same state. Reference rules can also take or replace that state, as
/// required when transposing writes. They use the reference functions instead of submitting reference values through
/// value cotangent accumulator handles; such contributions are rejected.
///
/// Reference-state buffers are allocated lazily when needed. [`Self::dimension_sources`] retains values from which
/// their runtime dimensions can be obtained, including known primal values and previously encountered cotangents.
/// These sources remain available after value cotangents are extracted, so a later rule can still construct a
/// correctly shaped zero even when its type alone does not supply the required dimensions.
///
/// # Reference Views
///
/// A rule accessing a primal reference view receives the corresponding view of its root's cotangent buffer. For
/// example, a primal slice `r[i..i + k]` needs a slice of the cotangent buffer using the same coordinates. The context
/// obtains the view path from the region's [`ReferenceViewAnalysis`](crate::ReferenceViewAnalysis) and reapplies its
/// steps through [`ReferenceViewOperation::reapply_view`]. The root buffer is allocated if needed, and reconstruction
/// validates the resulting view's type. An unavailable view path is rejected rather than treated as the whole root.
///
/// The reverse sweep materializes known primal coordinates into the transposed program before a rule needs them.
/// The context retains these coordinates separately from the generated cotangent views: two slices may share the
/// same dynamic index while requiring distinct views. Generated views are cached by the primal view's [`ValueId`]
/// so repeated accesses reuse staged operations; consuming a root accumulator removes its cached views.
///
/// Generic view reconstruction requires known coordinate values. It rejects a view bound to a linear coordinate or a
/// nested region's iteration counter. An enclosing operation's transpose rule must handle iteration-bound views when
/// supporting that case; the context cannot reconstruct them from the region's non-reference operands alone.
///
/// # Nested Regions
///
/// A nested [`Region`] is transposed in its own context. Inputs with [`CotangentDestinationKind::Reference`] receive
/// buffers from the enclosing rule. Reference state inputs return the same references so control flow rules can thread
/// state through nested regions; gradient buffers for non-reference inputs receive additive updates and produce no
/// output. A root whose state cotangent does not need to cross the region boundary can use
/// [`CotangentDestinationKind::Ignore`], allowing the nested region to allocate and discard its own accumulator
/// if needed. State cotangents needed by the enclosing computation must instead use its accumulators.
pub struct TranspositionContext<V: Value, O: Operation<Type = V::Type>> {
    /// [`TracingContext`] that the transposed [`Program`] is staged into.
    parent: TracingContext<V, O>,

    /// Identity token for this context's cotangent storage, shared with its accumulator handles. Two transposition
    /// contexts may stage into the same parent trace but own different slots: slot zero in one context is unrelated
    /// to slot zero in the other. This token therefore identifies the storage owner, not the parent or its builder.
    /// Handles retain only the token so they neither keep the trace alive nor borrow the context. Pointer equality
    /// rejects foreign handles, and retaining the token prevents stale handles from matching a later allocation.
    context_identity: Rc<()>,

    /// Storage accessed by [`Self::cotangent_storage`]. Refer to that function for more information.
    cotangent_storage: Vec<CotangentStorage<V, O>>,

    /// Reference analysis returned by [`Self::reference_analysis`]. Refer to that function for more information.
    reference_analysis: Option<Arc<ReferenceAnalysis>>,

    /// Root accumulators accessed by [`Self::cotangent_accumulator_reference`]. Refer to that function for more information.
    reference_accumulators: BTreeMap<ReferenceRoot, CotangentReferenceAccumulator<Tracer<TracingContext<V, O>>>>,

    /// Generated cotangent reference views, keyed by the primal view's [`ValueId`]. Repeated accesses to one primal
    /// view reuse the same staged cotangent view. For example, a primal slice `r[i..i + k]` maps to the corresponding
    /// slice of `r`'s cotangent buffer. [`Self::cotangent_view_coordinates`] supplies the preserved primal coordinates
    /// needed to construct that slice; this map retains the resulting reference, not those coordinates. Entries are
    /// removed when the root accumulator is consumed. Refer to [`Self::cotangent_reference_view`] for information
    /// on view reconstruction.
    cotangent_views: BTreeMap<ValueId, Tracer<TracingContext<V, O>>>,

    /// Known primal coordinate values materialized in the transposed program, keyed by each coordinate's primal
    /// [`ValueId`]. These are the original indices used to reconstruct cotangent views, not derivatives of the indices.
    /// For example, two primal slices that use the same dynamic index `i` share its entry here, while each slice has
    /// its own entry in [`Self::cotangent_views`]. A single view can also require several coordinates. The maps thus
    /// have different keys and are not paired entry by entry. This map supplies coordinates during reconstruction,
    /// and [`Self::cotangent_views`] caches the resulting references.
    cotangent_view_coordinates: BTreeMap<ValueId, Tracer<TracingContext<V, O>>>,

    /// Values supplying runtime dimensions, returned by [`Self::dimension_sources`].
    /// Refer to that function for more information.
    dimension_sources: Vec<Tracer<TracingContext<V, O>>>,
}

impl<V: Value, O: Operation<Type = V::Type>> TranspositionContext<V, O> {
    /// Creates a new [`TranspositionContext`] over `parent` with empty cotangent storage and no reference analysis.
    /// Rules staged through it can use every [`TracingContext`] capability and cotangent accumulators for non-reference
    /// inputs. Reference-state queries require the analysis installed by the program transposition engine.
    #[inline]
    pub fn new(parent: TracingContext<V, O>) -> Self {
        Self {
            parent,
            context_identity: Rc::new(()),
            cotangent_storage: Vec::new(),
            reference_analysis: None,
            reference_accumulators: BTreeMap::new(),
            cotangent_views: BTreeMap::new(),
            cotangent_view_coordinates: BTreeMap::new(),
            dimension_sources: Vec::new(),
        }
    }

    /// Creates a [`TranspositionContext`] over `parent`, retaining reference analysis for `region` when its closure
    /// contains references. `consumable_input_indices` lists the positions of linear inputs whose tangent references
    /// belong to this invocation. Other external references remain borrowed during analysis.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::MalformedProgram`] when the region closure violates the reference model.
    #[inline]
    fn for_region(
        parent: TracingContext<V, O>,
        region: RegionRef<'_, V, O>,
        consumable_input_indices: Vec<usize>,
    ) -> Result<Self, ProgramError> {
        let analysis = if region.contains_references_in_closure() {
            Some(region.reference_analysis_with_consumable_inputs(0, consumable_input_indices)?)
        } else {
            None
        };
        Ok(Self { reference_analysis: analysis, ..Self::new(parent) })
    }

    /// Creates empty cotangent storage and returns its [`CotangentAccumulator`] handle. Each call creates an
    /// independent storage, even if the provided type is identical. Clone the returned handle when several
    /// contributions should share the same running sum. Creating the handle stages no operations; contributions are
    /// submitted through [`CotangentAccumulator::accumulate`] and values are extracted through
    /// [`Self::take_cotangents`].
    ///
    /// This function does not derive a cotangent type from a primal type or allocate a reference-state buffer.
    /// Use [`Self::cotangent_accumulators`] to derive types and eligibility from rule operands, and the context's
    /// reference functions to work with reference-state cotangents.
    ///
    /// # Parameters
    ///
    ///   - `cotangent_type`: Expected type of each contribution, already derived from the primal type. It is retained
    ///     even before any contributions arrive, so extracting an empty accumulator returns a zero of this type.
    ///     Concrete reference-typed contributions are rejected by the value accumulation API.
    ///   - `needed`: Whether this accumulator should retain contributions. With `false`, contributions are discarded
    ///     after validation and extraction returns a structural zero. With `true`, contributions are summed as they
    ///     arrive; this requests a cotangent but does not imply that its value is nonzero.
    pub fn cotangent_accumulator(&mut self, cotangent_type: V::Type, needed: bool) -> CotangentAccumulator {
        let storage_index = self.cotangent_storage.len();
        self.cotangent_storage.push(CotangentStorage::Value { cotangent_type, value: None });
        CotangentAccumulator { context_identity: self.context_identity.clone(), storage_index, needed }
    }

    /// Creates new cotangent storage and returns [`CotangentAccumulator`] handles aligned with `inputs` for a direct
    /// transposition rule invocation. Each call creates independent storage. Clone a returned handle when two operand
    /// positions must share their contributions.
    ///
    /// # Parameters
    ///
    ///   - `inputs`: The rule's operands in operand order, each carrying either a known primal value or the type of
    ///     an unknown value. One handle is returned per operand, including operands whose cotangent is not requested.
    ///     Known values must belong to this context's parent trace.
    ///   - `cotangent_mask`: Whether a value cotangent is needed for each input, in the same order as `inputs`.
    ///     An empty slice requests cotangents for all eligible inputs. Otherwise, it must have exactly one flag per
    ///     input. For example, `[true, false, true]` requests cotangents for the first and third inputs if they are
    ///     unknown non-reference values. Known inputs and reference-state inputs never receive value contributions,
    ///     even with a `true` flag. Their handles report [`CotangentAccumulator::is_needed`] as `false`, as do handles
    ///     selected by a `false` flag, and discard contributions after validating them. These flags describe which
    ///     cotangents are needed, not whether they are nonzero: a requested cotangent can still be zero.
    pub fn cotangent_accumulators(
        &mut self,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        cotangent_mask: &[bool],
    ) -> Result<Vec<CotangentAccumulator>, DifferentiationError>
    where
        V::Type: DifferentiableType,
    {
        if !cotangent_mask.is_empty() {
            check_count!("input", cotangent_mask, inputs.len(), ProgramError);
        }

        // Validate the complete boundary before adding slots, so that malformed inputs cannot partly install a scope.
        let types = inputs
            .iter()
            .map(|input| {
                if let PartialValue::Known(value) = input {
                    check_builders!(self.builder(), value.builder())?;
                }
                input.r#type().cotangent()
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(inputs
            .iter()
            .zip(types)
            .enumerate()
            .map(|(index, (input, cotangent_type))| {
                let needed = matches!(input, PartialValue::Unknown(_))
                    && !cotangent_type.is_reference()
                    && cotangent_mask.get(index).copied().unwrap_or(true);
                self.cotangent_accumulator(cotangent_type, needed)
            })
            .collect())
    }

    /// Returns the retained [`ReferenceAnalysis`] of the source region's reference roots, views, and accesses, or
    /// [`None`] when the context is detached or the region closure contains no references. The analysis accounts for
    /// which reference inputs may be consumed by this invocation. Retaining it keeps operand resolution and access
    /// checks consistent throughout the reverse mode differentiation sweep.
    #[inline]
    pub fn reference_analysis(&self) -> Option<&Arc<ReferenceAnalysis>> {
        self.reference_analysis.as_ref()
    }

    /// Returns retained values from which the runtime dimensions needed to construct reference state zeros can be
    /// obtained. A source can be an array whose dimensions can be queried or an explicit dimension value; these are
    /// candidate sources and not a list of already extracted dimensions. They include known primal values and
    /// cotangents encountered during transposition, and remain available after value cotangent storage is drained.
    ///
    /// For example, constructing a zero buffer of type `f32[n]` requires the runtime value of `n`, which the type alone
    /// does not supply. An available array of type `f64[n]` can supply that dimension even though its element type is
    /// different. A slice's length does not necessarily supply the length of its root buffer, so view coordinates
    /// alone are insufficient. Zero materialization searches these values for the dimension identities it needs.
    ///
    /// A transpose rule can chain its known operands after these sources and pass the resulting iterator to
    /// [`ResidualZeroProvider::materialize_zero_from_residual_sources`] when it needs a concrete zero, for example
    /// to clear the state cotangent when transposing a reference write or swap operation.
    #[inline]
    pub fn dimension_sources(&self) -> impl Iterator<Item = &Tracer<TracingContext<V, O>>> {
        self.dimension_sources.iter()
    }

    /// Returns the cotangent reference of the reference-typed operand at `input_index` of the current instruction,
    /// allocating the root's accumulator with a zero initial value first when it is still unallocated, and viewing it
    /// as the operand views its root. This is the accumulator lookup for rules that must produce a cotangent reference
    /// regardless of prior use: the transposes of the `reference_read` and `reference_freeze` operations accumulate a
    /// live output cotangent into it, `reference_swap` swaps a live output cotangent into it, and structured operations
    /// thread it into their nested regions.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`TranspositionDriver`] for the source instruction whose cotangents are being processed.
    ///     Reference operands require a source region consistent with this context's reference analysis.
    ///   - `input_index`: Position of the reference input/operand among the current instruction's inputs/operands.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand is not a linear reference operand of the transposed region, when its view
    /// path cannot be reapplied, or when the accumulator cannot be allocated.
    pub fn cotangent_reference<D: TranspositionDriver<V, O>>(
        &mut self,
        driver: &D,
        input_index: usize,
    ) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: ReferenceViewOperation
            + ResidualZeroProvider<V::Type>
            + OperationProvider<V::Type, ReferenceNewOperation<V::Type, V::Type>, Operation = O>,
    {
        let (_, value, root) = self.reference_input(driver, input_index)?;
        let root_reference = {
            // `reference_input` established that the root has an accumulator.
            let accumulator = self.reference_accumulators.get_mut(&root).unwrap();
            accumulator.allocate_in(&self.parent, &self.dimension_sources)?.clone()
        };
        self.cotangent_reference_view(driver, value, root_reference)
    }

    /// Returns the existing cotangent reference for the reference operand at `input_index`, applying the same view
    /// as the primal operand. Returns [`None`] if the root's cotangent buffer has not been allocated; in that case,
    /// its state cotangent is represented by a symbolic zero, and this function leaves it that way.
    ///
    /// For example, when transposing `reference_write(r, x)`, the current cotangent of `r` determines the contribution
    /// to `x`'s cotangent. If `r` has no allocated cotangent buffer, that contribution is zero. The rule can return
    /// without creating a buffer or staging operations to read and clear it. The `reference_add_update` operation uses
    /// the same check. The `reference_swap` operation does so when its output cotangent is also zero; a nonzero output
    /// cotangent requires a buffer and uses [`Self::cotangent_reference`] instead.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`TranspositionDriver`] for the source instruction whose cotangents are being processed.
    ///     Reference operands require a source region consistent with this context's reference analysis.
    ///   - `input_index`: Position of the reference input/operand among the current instruction's inputs/operands.
    ///
    /// # Errors
    ///
    /// Returns an error when the operand is not a linear reference operand of the transposed region or when its view
    /// path cannot be reapplied.
    pub fn cotangent_reference_if_allocated<D: TranspositionDriver<V, O>>(
        &mut self,
        driver: &D,
        input_index: usize,
    ) -> Result<Option<Tracer<TracingContext<V, O>>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: ReferenceViewOperation,
    {
        let (_, value, root) = self.reference_input(driver, input_index)?;
        match self.cotangent_accumulator_reference(root).cloned() {
            Some(root_reference) => self.cotangent_reference_view(driver, value, root_reference).map(Some),
            None => Ok(None),
        }
    }

    /// Resolves the reference-typed input/operand at `input_index` of the current instruction to the [`InstructionId`]
    /// of that instruction, the operand's [`ValueId`], and its canonical linear [`ReferenceRoot`], rejecting operands
    /// that are not linear reference operands of the region.
    fn reference_input<D: TranspositionDriver<V, O>>(
        &self,
        driver: &D,
        input_index: usize,
    ) -> Result<(InstructionId, ValueId, ReferenceRoot), DifferentiationError> {
        let (Some((region, id, instruction)), Some(analysis)) = (driver.scope()?, &self.reference_analysis) else {
            return Err(ProgramError::MalformedProgram(format!(
                "input {input_index} has no reference root in a transposition context that is not scoped to a \
                 reference-carrying instruction"
            ))
            .into());
        };
        let atom = instruction.inputs().get(input_index).copied().ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "input {} is out of range for `{}` with {} operands",
                input_index,
                instruction.operation().name(),
                instruction.inputs().len(),
            ))
        })?;
        let value = ValueId::new(region.id(), atom);
        let root = analysis.root_of(value).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "input {} of `{}` is not a reference-typed value of the transposed region",
                input_index,
                instruction.operation().name(),
            ))
        })?;
        if !self.reference_accumulators.contains_key(&root) {
            return Err(ProgramError::UnsupportedOperation {
                message: format!(
                    "operand {} of `{}` denotes a known reference; transposing a linear access to a known \
                     reference is not supported",
                    input_index,
                    instruction.operation().name(),
                ),
            }
            .into());
        }
        Ok((id, value, root))
    }

    /// Returns the cotangent reference of `root`'s [`CotangentReferenceAccumulator`], or [`None`] when it is
    /// unallocated or absent. These accumulators track the region's linear reference roots, so aliases share state
    /// instead of accumulating separate values. An unallocated entry represents a symbolic-zero state cotangent;
    /// this lookup never allocates it.
    fn cotangent_accumulator_reference(&self, root: ReferenceRoot) -> Option<&Tracer<TracingContext<V, O>>> {
        self.reference_accumulators.get(&root).and_then(CotangentReferenceAccumulator::reference)
    }

    /// Returns the cotangent reference viewed exactly as the primal `value` views its root, deriving it from the root's
    /// cotangent reference `root_reference` through the region's retained view overlay on first use, and validating
    /// that the result has the operand's cotangent type in either case. Derived views are cached by primal value so
    /// repeated accesses reuse the same staged slice or index operations. The cache holds emitted cotangent values;
    /// the reference analysis holds descriptions of the original primal views.
    fn cotangent_reference_view<D: TranspositionDriver<V, O>>(
        &mut self,
        driver: &D,
        value: ValueId,
        root_reference: Tracer<TracingContext<V, O>>,
    ) -> Result<Tracer<TracingContext<V, O>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: ReferenceViewOperation,
    {
        if let Some(viewed) = self.cotangent_views.get(&value) {
            return Ok(viewed.clone());
        }

        // `reference_input` established that the driver supplies a source instruction and the context has reference
        // analysis. The driver's source scope stays fixed throughout this invocation. A root skips the view overlay.
        let (region, _, _) = driver.scope()?.unwrap();
        let is_view = self.reference_analysis.as_ref().unwrap().is_view(value);
        let expected = region.atoms()[value.atom().index()].r#type().cotangent()?;
        let mut view = root_reference;
        if is_view {
            // Each reapplied step stages the view operation over the current transformed source, whose type inference
            // validates the description against that source.
            let overlay = region.reference_view_analysis(0).map_err(ProgramError::from)?;
            let path = overlay.path(value).ok_or_else(|| {
                ProgramError::MalformedProgram(format!(
                    "the view overlay of the transposed region has no path for reference operand {value:?}"
                ))
            })?;

            view = path.steps().iter().try_fold(view, |view, step| {
                // The overlay bound each symbolic coordinate to a known primal operand of the view-creating
                // instruction, whose transposed program value the reverse sweep materialized before this rule
                // ran. An iteration-bound boundary view has no such value: the transpose rule of the attaching
                // operation re-creates it.
                let symbols = step
                    .bindings()
                    .iter()
                    .map(|binding| match binding {
                        ViewSymbolBinding::Value(id) => self.cotangent_view_coordinates.get(id).cloned().ok_or_else(
                            || ProgramError::UnsupportedOperation {
                                message: format!(
                                    "the view coordinate {id:?} of reference operand {value:?} is a linear value, so \
                                     the view cannot be reapplied to the operand's cotangent reference"
                                ),
                            },
                        ),
                        ViewSymbolBinding::Iteration(region) => Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "reference operand {value:?} views its root through the iteration counter of region \
                                 {region:?}; boundary views are re-created by the transpose rule of the attaching \
                                 operation and cannot be reapplied to a cotangent reference"
                            ),
                        }),
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                O::reapply_view(&self.parent, step.view(), view, symbols.as_slice())
            })?;
        }

        // The final type is checked against the operand's cotangent type for roots and views alike, so that a view
        // chain that composes but lands elsewhere, or a root accumulator whose type disagrees with the operand it is
        // handed out for, is rejected rather than accumulated into.
        if view.r#type().as_ref() != &expected {
            return Err(ProgramError::MalformedProgram(format!(
                "the cotangent reference of reference operand {:?} has type {} but the operand's cotangent type is {}",
                value,
                view.r#type().as_ref(),
                expected,
            ))
            .into());
        }

        if is_view {
            self.cotangent_views.insert(value, view.clone());
        }

        Ok(view)
    }

    /// Resolves the [`CotangentDestinations`] of the instruction whose operands are described by `inputs`. A linear
    /// reference operand receives [`CotangentDestinationKind::Reference`] when its root already has a cotangent buffer,
    /// or when the instruction reads, swaps, or consumes that root, directly or in a nested [`Region`]. Its buffer is
    /// obtained through [`Self::cotangent_reference`], allocating it if needed. Other linear reference operands receive
    /// [`CotangentDestinationKind::Ignore`], avoiding an allocation when the instruction only writes into a root whose
    /// state cotangent is zero. Structured operations such as `scan` and `condition` use these destinations to pass
    /// buffers and value cotangents into their transposed regions in operand order.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`TranspositionDriver`] for the source instruction whose cotangents are being processed.
    ///     Reference operands require a source region consistent with this context's reference analysis.
    ///   - `inputs`: The [`Instruction`]'s inputs/operands in order. Each entry contains either a known primal value
    ///     or the type of an unknown value whose cotangent is being propagated.
    ///   - `accumulators`: [`CotangentAccumulator`]s in input/operand order. Empty requests returned non-reference
    ///     cotangents (i.e., cotangents of non-reference values), used when a nested recurrence must compute carry
    ///     gradients independently of the outer requested outputs.
    pub fn cotangent_destinations<D: TranspositionDriver<V, O>>(
        &mut self,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<CotangentDestinations<Tracer<TracingContext<V, O>>>, DifferentiationError>
    where
        V::Type: DifferentiableType,
        O: ReferenceViewOperation
            + ResidualZeroProvider<V::Type>
            + OperationProvider<V::Type, ReferenceNewOperation<V::Type, V::Type>, Operation = O>,
    {
        if !accumulators.is_empty() {
            check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
            // Validate every handle, including ignored and known operands, before resolving any reference state.
            accumulators.iter().try_for_each(|accumulator| self.cotangent_storage(accumulator).map(|_| ()))?;
        }

        let mut references = Vec::new();
        let kinds = inputs
            .iter()
            .enumerate()
            .map(|(operand_index, input)| {
                Ok(match input {
                    PartialValue::Unknown(r#type) if r#type.is_reference() => {
                        let (id, _, root) = self.reference_input(driver, operand_index)?;
                        // An existing buffer carries state cotangents from later instructions; even a write operation
                        // must receive it. Otherwise, reads, swaps, or consumes in this instruction or its nested
                        // regions can introduce a state cotangent, so they also need a shared buffer. Pure writes into
                        // a root with zero state cotangent need no buffer passed from the enclosing region. Note that
                        // `reference_input` above already checked that reference analysis is available, so unwrapping
                        // it is safe.
                        let needs_reference = self.cotangent_accumulator_reference(root).is_some()
                            || self.reference_analysis.as_ref().unwrap().transitive_access(id).is_some_and(|access| {
                                access.access_modes_for(root).any(|mode| {
                                    matches!(
                                        mode,
                                        ReferenceAccessMode::Read
                                            | ReferenceAccessMode::ReadWrite
                                            | ReferenceAccessMode::Consume
                                    )
                                })
                            });
                        if needs_reference {
                            references.push(self.cotangent_reference(driver, operand_index)?);
                            CotangentDestinationKind::Reference
                        } else {
                            CotangentDestinationKind::Ignore
                        }
                    }
                    PartialValue::Unknown(_) if !accumulators.is_empty() => {
                        let accumulator = &accumulators[operand_index];
                        if !accumulator.is_needed() {
                            CotangentDestinationKind::Ignore
                        } else if let Some(reference) = accumulator.reference(self)? {
                            references.push(reference);
                            CotangentDestinationKind::Reference
                        } else {
                            CotangentDestinationKind::Return
                        }
                    }
                    PartialValue::Unknown(_) | PartialValue::Known(_) => CotangentDestinationKind::Return,
                })
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;

        let reference_inputs = inputs.iter().map(|input| input.is_unknown() && input.r#type().is_reference()).collect();
        Ok(CotangentDestinations::new(kinds, references, reference_inputs))
    }

    /// Returns the accumulated cotangent value for each handle in `accumulators`, in the same order, and empties the
    /// value storage from which it was taken. The handles remain valid. Subsequent contributions start a new sum, and
    /// another call without intervening contributions returns structural zeros (i.e., [`MaybeZero::Zero`]). A handle
    /// with no retained contributions, including one whose cotangent was not requested, also returns a structural zero
    /// of its expected contribution type. This function stages no operations as contributions were summed as they
    /// arrived.
    ///
    /// Repeated handles return the same value at every occurrence and empty their shared storage only once. For
    /// example, if a handle has accumulated `a + b`, passing that handle twice returns `[a + b, a + b]`, rather than
    /// returning the sum followed by zero.
    ///
    /// When a handle accumulates directly into a caller-supplied gradient buffer, its result here is a structural
    /// zero as the contributions are already accounted for in that buffer. This function does not read, clear, or
    /// remove that buffer. It also leaves reference state accumulators untouched. Those track the contents of mutable
    /// references and are separate from the value sums extracted here; use [`Self::take_reference_cotangent`] to remove
    /// the cotangent buffer for an allocating reference output.
    ///
    /// # Parameters
    ///
    ///   - `accumulators`: [`CotangentAccumulator`]s created by this context, in the desired result order.
    ///     Handles may be repeated, including through clones. An empty slice returns an empty vector.
    ///
    /// # Errors
    ///
    /// Returns an error if any handle belongs to another transposition context. All handles are checked before any
    /// values are taken, so an invalid handle leaves every accumulator unchanged.
    pub fn take_cotangents(
        &mut self,
        accumulators: &[CotangentAccumulator],
    ) -> Result<Vec<MaybeZero<Tracer<TracingContext<V, O>>>>, DifferentiationError> {
        accumulators.iter().try_for_each(|accumulator| self.cotangent_storage(accumulator).map(|_| ()))?;
        let mut extracted = BTreeMap::new();
        Ok(accumulators
            .iter()
            .map(|accumulator| {
                // Extract a shared slot once; repeated handles reuse the value already taken from it.
                extracted
                    .entry(accumulator.storage_index)
                    .or_insert_with(|| match &mut self.cotangent_storage[accumulator.storage_index] {
                        CotangentStorage::Value { cotangent_type, value } => value.take().map_or_else(
                            || MaybeZero::Zero(cotangent_type.clone()),
                            |(value, _)| MaybeZero::Value(value),
                        ),
                        CotangentStorage::Buffer { cotangent_type, .. } => MaybeZero::Zero(cotangent_type.clone()),
                    })
                    .clone()
            })
            .collect())
    }

    /// Removes the [`CotangentReferenceAccumulator`] for the reference allocated by the `output_index`-th output of the
    /// current instruction and returns its buffer, if allocated. Returns [`None`] if the accumulator was unallocated or
    /// has already been removed. Cached cotangent views of that reference are removed as well. This function stages no
    /// operations and does not freeze or read the returned buffer.
    ///
    /// This extracts reference state storage, independently of the value sums extracted by [`Self::take_cotangents`].
    /// For example, when transposing `r = reference_new(x)`, the rule takes `r`'s cotangent buffer here, freezes it
    /// into a value, and contributes that value to `x`'s value cotangent accumulator. That contribution can then be
    /// extracted through [`Self::take_cotangents`]. If no buffer was allocated, the contribution is zero.
    ///
    /// Taking the buffer marks the end of the reverse sweep for this allocation: its allocation precedes every use
    /// in forward order, so all of those uses have already been transposed. The context no longer retains the root's
    /// accumulator or derived views after this function returns.
    ///
    /// # Parameters
    ///
    ///   - `driver`: [`TranspositionDriver`] identifying the current source instruction. It must identify the
    ///     instruction that allocated the reference and not a later instruction that reads, updates, or views it.
    ///   - `output_index`: Position of the allocating output among that instruction's outputs. This identifies a
    ///     reference allocation, not a value cotangent handle or an operand index.
    ///
    /// # Errors
    ///
    /// Propagates errors from the driver. Returns [`ProgramError::MalformedProgram`] if the driver supplies no
    /// current instruction or the selected output is not declared as a reference allocation.
    pub fn take_reference_cotangent<D: TranspositionDriver<V, O>>(
        &mut self,
        driver: &D,
        output_index: usize,
    ) -> Result<Option<Tracer<TracingContext<V, O>>>, DifferentiationError> {
        let Some((_, id, instruction)) = driver.scope()? else {
            return Err(ProgramError::MalformedProgram(format!(
                "output {output_index} has no reference allocation in a transposition context that is not scoped to \
                 a reference-carrying instruction"
            ))
            .into());
        };
        if !instruction.operation().effects().allocation_output_indices().any(|i| i == output_index) {
            return Err(ProgramError::MalformedProgram(format!(
                "output {} of `{}` is not a reference allocation",
                output_index,
                instruction.operation().name(),
            ))
            .into());
        }
        let root = ReferenceRoot::Allocation { instruction: id, output_index };
        if let Some(analysis) = &self.reference_analysis {
            self.cotangent_views.retain(|value, _| analysis.root_of(*value) != Some(root));
        }
        Ok(match self.reference_accumulators.remove(&root) {
            Some(CotangentReferenceAccumulator::Allocated { reference }) => Some(reference),
            Some(CotangentReferenceAccumulator::Unallocated { .. }) | None => None,
        })
    }

    /// Returns the [`CotangentStorage`] for `accumulator` after checking its context identity. This storage receives
    /// additive contributions and is separate from reference state accumulators, whose transpose rules can also take or
    /// replace the state cotangent. Retained identity tokens prevent an old context's storage index from accidentally
    /// becoming valid when a new context is allocated at the same address.
    fn cotangent_storage(
        &self,
        accumulator: &CotangentAccumulator,
    ) -> Result<&CotangentStorage<V, O>, DifferentiationError> {
        if !Rc::ptr_eq(&self.context_identity, &accumulator.context_identity) {
            return Err(ProgramError::InvalidArgument {
                message: "cotangent accumulator belongs to another transposition context".into(),
            }
            .into());
        }

        // Handles have private fields and slots are never removed, so ownership guarantees that the index exists.
        Ok(&self.cotangent_storage[accumulator.storage_index])
    }
}

impl<V: Value, O: Operation<Type = V::Type>> Deref for TranspositionContext<V, O> {
    type Target = TracingContext<V, O>;

    #[inline]
    fn deref(&self) -> &TracingContext<V, O> {
        &self.parent
    }
}

impl<V: Value, O: Operation<Type = V::Type>> DerefMut for TranspositionContext<V, O> {
    #[inline]
    fn deref_mut(&mut self) -> &mut TracingContext<V, O> {
        &mut self.parent
    }
}

/// Pullback of a function `f` at a linearization point `x` (i.e., the transposed linear map `ȳ ↦ x̄ = (∂f/∂x)(x)ᵀ · ȳ`),
/// packaged as a reusable callable. This is the reverse-mode dual of [`Pushforward`], whose callable applies the
/// un-transposed map `ẋ ↦ (∂f/∂x)(x) · ẋ` instead. It retains the linear pushforward [`Program`] `(ẋ, r) ↦ ẏ` together
/// with the saved linearization values `r`, called residuals. With `r` fixed, that program is linear in `ẋ`.
///
/// [`Self::apply`] returns input cotangent values for reference-free boundaries. [`Self::apply_with_destinations`]
/// chooses how each input's cotangent is delivered through [`CotangentDestinationKind`]: return a value, accumulate
/// into a caller-supplied reference buffer, or omit the cotangent. Reference inputs use the reference-state contract;
/// they cannot request a returned cotangent value through these application functions.
///
/// # Why the Pushforward Program Is Retained
///
/// The requested destinations determine the backward program's inputs, outputs, and accumulation operations. They are
/// supplied at application time, so the pullback retains the pushforward and transposes it for that destination pattern
/// when needed. One pullback can therefore serve different patterns without retracing or differentiating the original
/// function again. A fixed backward program that returned values could be followed by additions into buffers for
/// non-reference inputs, but it would miss opportunities to accumulate directly. For example, a slice transpose can
/// update a slice of the caller's buffer instead of constructing a full-sized intermediate gradient.
///
/// Derived backward programs are retained in the [`Region`] transform cache, with the destination kinds included
/// in the cache key. The first application of a pattern derives its program unless it is already cached; subsequent
/// applications reuse it, including applications with different destination buffer instances. Each application still
/// validates its arguments and looks up the cached program. Using several patterns can retain several backward
/// programs, and transposition errors surface when a program is first requested rather than during pullback
/// construction. This design supports destinations chosen independently on each application; it does not promise
/// lower overhead than eager transposition for callers that only ever request returned values.
///
/// # Application and Structured Values
///
/// Application appends destination references, when needed, and the saved residuals to the flattened output cotangents
/// `ȳ`, interprets the selected backward program, and rebuilds the requested input cotangent structure. The same
/// pullback can process many cotangents (e.g., successive coordinate basis cotangents to build a Jacobian row by row)
/// reusing the work done during linearization and cached transposition.
///
/// The stored programs are _compact_ meaning that leaves whose differential types contain only zero are omitted from
/// their inputs and outputs. Application removes those output cotangent leaves before replay and restores typed zeros
/// for omitted input cotangent leaves when returned values are requested. This omission is determined by types, not by
/// whether a particular cotangent value happens to be zero.
///
/// The context `C` supplies the value semantics and operation family. `Input` is the closure's structured input type;
/// its [`ParameterStructure`](Parameterized::ParameterStructure) is retained so flat input cotangents can be reshaped
/// into `Input::To<C::Value>`. `Output` is the structured output type, carried as a type parameter so [`Self::apply`]
/// infers the cotangent family from the pullback itself without requiring explicit generic arguments.
pub struct Pullback<C: Context, Input: Parameterized<C::Value>, Output> {
    /// [`Context`] that the pullback was built in. [`apply`](Self::apply) replays the transposed program in it,
    /// mirroring how [`Pushforward`] replays its pushforward program.
    context: C,

    /// Linear pushforward [`Program`] `(live(ẋ), r) ↦ live(ẏ)` over the primal operation family in the context's
    /// staged [`Constant`](Context::Constant) space, whose trailing inputs consume the residuals. It is transposed on
    /// application and its literal constants are lifted through the context's [`lift`](Context::lift) when the
    /// transposed program is replayed.
    linear_program: Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>,

    /// Linearization-point residuals consumed by [`linear_program`](Self::linear_program), appended after the output
    /// cotangents and destination references when interpreting its transpose.
    residuals: Vec<C::Value>,

    /// Reconstruction plan used to restore zero space input cotangents omitted from the transposed program.
    cotangent_reconstruction: ZeroSpaceBoundaryReconstruction<C::Value>,

    /// Complete public primal input boundary, including leaves whose tangent spaces contain only zero and which are
    /// consequently absent from [`linear_program`](Self::linear_program)'s inputs.
    primal_input_types: Vec<C::Type>,

    /// Complete public primal output boundary, including leaves whose tangent spaces contain only zero and which are
    /// consequently absent from [`linear_program`](Self::linear_program)'s outputs.
    primal_output_types: Vec<C::Type>,

    /// Runtime identities of every reference bound at the primal boundary (inputs and captures alike), retained so that
    /// a cotangent destination aliasing one of them is rejected on application.
    primal_references: ReferenceBoundary<DifferentiationBoundaryPosition>,

    /// Parameter structure of the closure's input, used to reshape the flat input cotangents.
    input_structure: Input::ParameterStructure,

    /// Encodes the closure's output family `Output` so that [`apply`](Self::apply) can flatten the cotangents without
    /// a turbofish. No `Output::ParameterStructure` is stored alongside it because [`apply`](Self::apply) only
    /// _flattens_ its structured cotangent argument, which needs no stored structure, and rebuilds structure only on
    /// the input-cotangent side through `input_structure`. [`Pushforward`] mirrors this with a stored output structure
    /// and a phantom `Input`.
    marker: PhantomData<fn() -> Output>,
}

impl<
    C: Context<Type: DifferentiableType>,
    Input: Parameterized<C::Value, Family: ParameterizedFamily<C::Value>>,
    Output: Parameterized<C::Value>,
> Pullback<C, Input, Output>
{
    /// Converts a validated [`Pushforward`] into a [`Pullback`], retaining its linear program, residuals, and primal
    /// boundary without repeating their validation. Only the reconstruction of zero space input cotangents is new.
    /// The pushforward's output tangent reconstruction is discarded.
    fn from_pushforward(
        pushforward: Pushforward<C, Input, Output>,
        primal_inputs: &[C::Value],
        input_structure: Input::ParameterStructure,
    ) -> Result<Self, ProgramError>
    where
        C::Operation: ResidualZeroProvider<C::Type>,
    {
        let (context, linear_program, residuals, primal_input_types, primal_output_types, primal_references) =
            pushforward.into_parts();
        let cotangent_reconstruction = ZeroSpaceBoundaryReconstruction::capture(
            &context,
            primal_inputs,
            &primal_input_types,
            ZeroSpaceBoundaryRole::InputCotangent,
        )?;
        Ok(Self {
            context,
            linear_program,
            residuals,
            cotangent_reconstruction,
            primal_input_types,
            primal_output_types,
            primal_references,
            input_structure,
            marker: PhantomData,
        })
    }

    /// Returns the compact linear _pushforward_ [`Program`] `(live(ẋ), r) ↦ live(ẏ)` retained by this pullback.
    /// For `y = f(x)` at the fixed linearization point `x`, it computes the tangent map `ẏ = (∂f/∂x)(x) · ẋ`. The
    /// residuals `r`, returned by [`Self::residuals`], contain saved values from linearization that this computation
    /// needs. With those residuals fixed, the map is linear in the input tangents `ẋ`; it need not be linear in `r`.
    /// Here, `live(...)` omits leaves whose differential types contain only zero. The program receives the remaining
    /// input tangents followed by `r` and produces the remaining output tangents. This omission depends on the types,
    /// not on whether a particular tangent value happens to be zero. Note that this is the pushforward program and not
    /// the program computing the pullback `x̄ = (∂f/∂x)(x)ᵀ · ȳ`. Application transposes it for the requested cotangent
    /// destinations and reuses the resulting program through the transform cache. Retaining the pushforward allows one
    /// pullback to return cotangent values, update caller-supplied buffers, or omit unrequested cotangents on different
    /// applications.
    #[inline]
    pub fn linear_program(&self) -> &Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>> {
        &self.linear_program
    }

    /// Returns the transposition of [`linear_program`](Self::linear_program) under the destination kinds of its tangent
    /// inputs, served from the linear program's retained transform cache. An empty `destination_kinds` selects the
    /// default kinds (i.e., [`Return`](CotangentDestinationKind::Return) for non-reference inputs and
    /// [`Reference`](CotangentDestinationKind::Reference) for reference inputs).
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn transposed_program(
        &self,
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, DifferentiationError>
    where
        C::Operation: TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
    {
        self.linear_program
            .transpose_with_trailing_residuals_shared(self.residuals.len(), destination_kinds)
    }

    /// Returns the linearization-point residuals `r` that this callable closes over, aligned with the trailing inputs
    /// of [`linear_program`](Self::linear_program).
    #[inline]
    pub fn residuals(&self) -> &[C::Value] {
        &self.residuals
    }

    /// Consumes this [`Pullback`] and returns its open parts (i.e., the compact linear pushforward program `(live(ẋ),
    /// r) ↦ live(ẏ)` and the linearization-point residuals `r` its trailing inputs consume, in that order). Unlike the
    /// application functions, the returned program does not insert typed zero values for public leaves omitted from its
    /// Single Static Assignment (SSA) boundary because their differential spaces contain only zero, and it is the
    /// _linear_ program. [`into_transposed_parts`](Self::into_transposed_parts) returns its transpose.
    #[inline]
    pub fn into_linear_parts(
        self,
    ) -> (Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>, Vec<C::Value>) {
        (self.linear_program, self.residuals)
    }

    /// Consumes this [`Pullback`] and returns the compact transposed program `(live(ȳ), r) ↦ live(x̄)` together
    /// with the linearization-point residuals `r` its trailing inputs consume, in that order. The program is the
    /// all-[`Return`](CotangentDestinationKind::Return) transposition served from the linear program's retained
    /// transform cache, so it is valid only for a reference-free boundary and, like
    /// [`into_linear_parts`](Self::into_linear_parts), it omits zero-space leaves from its Single Static Assignment
    /// (SSA) boundary.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] when a public input or output leaf is reference-typed, and propagates
    /// transposition errors otherwise.
    #[allow(clippy::type_complexity)]
    #[inline]
    pub fn into_transposed_parts(
        self,
    ) -> Result<
        (Arc<Program<C::Constant, C::Operation, Vec<C::Constant>, Vec<C::Constant>>>, Vec<C::Value>),
        DifferentiationError,
    >
    where
        C::Operation: TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
    {
        self.validate_reference_boundary()?;
        let program = self.transposed_program(&[])?;
        Ok((program, self.residuals))
    }

    /// Pulls the provided structured output cotangents `cotangents` back to the closure's input cotangents.
    /// This is the all-[`Value`](CotangentSeed::Value), all-[`Return`](CotangentDestination::Return) application of
    /// [`apply_with_destinations`](Self::apply_with_destinations) where the cotangents are flattened, the
    /// linearization-point residuals are appended, the transposed program is interpreted at that vector in the context
    /// that this pullback was built in (i.e., the single replay path for both context flavors; an eager context
    /// interprets the pullback immediately, while a staging context stages it into the enclosing trace and returns
    /// tracers), and the flat input cotangents are reshaped against the closure's input structure. It is valid only
    /// when no public input or output leaf is reference-typed (references used _inside_ the differentiated closure are
    /// fine), and rejects a reference boundary with an [`InvalidArgument`](ProgramError::InvalidArgument) error
    /// directing the caller to [`apply_with_destinations`](Self::apply_with_destinations). Because transposition is
    /// late-bound, the first application also reports transposition errors.
    pub fn apply(&self, cotangents: Output::To<C::Value>) -> Result<Input::To<C::Value>, ProgramError>
    where
        C::Operation: TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
    {
        self.validate_reference_boundary()?;
        let seeds = cotangents.into_parameters().map(CotangentSeed::Value).collect::<Vec<_>>();
        let destinations = (0..self.primal_input_types.len()).map(|_| CotangentDestination::Return).collect::<Vec<_>>();
        let cotangents = self
            .apply_impl(seeds, &destinations)?
            .into_iter()
            .map(|cotangent| {
                cotangent.ok_or_else(|| {
                    ProgramError::MalformedProgram("pullback omitted a returned input cotangent".to_string())
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Restore the closure's original structured input shape after rebuilding every flattened cotangent leaf.
        Ok(Input::To::<C::Value>::from_parameters(self.input_structure.clone(), cotangents)?)
    }

    /// Pulls the provided structured output cotangent `seeds` back to the closure's inputs under the provided
    /// structured cotangent `destinations`, returning [`Some`] exactly at the [`Return`](CotangentDestination::Return)
    /// positions. Seeds and destinations are aligned leaf for leaf with the closure's output and input structures and
    /// are validated against this matrix:
    ///
    /// | Primal Boundary Leaf | Legal Seed or Destination                                                    |
    /// |----------------------|------------------------------------------------------------------------------|
    /// | Non-Reference Output | [`CotangentSeed::Value`]                                                     |
    /// | Reference Output     | [`CotangentSeed::NoCotangent`]                                               |
    /// | Non-Reference Input  | `Return`, `Ignore`, or `Reference` storing the input's cotangent type        |
    /// | Reference Input      | [`CotangentDestination::Reference`] with `r: ref<cotangent(T)>`, or `Ignore` |
    ///
    /// Every destination reference must resolve to an allocation in the originating context, must not alias a reference
    /// bound at the primal boundary of the differentiated closure (an input or a capture), and must not alias another
    /// destination of the same application; aliasing is compared by allocation identity, never by state generation.
    /// Destination kinds form the structural mask under which the linear program is transposed (refer to
    /// [`CotangentDestinationKind`]). Reference destinations for non-reference inputs receive contributions
    /// during replay and ignored cotangents need not be computed. The first application with a given mask pays for
    /// transposition, and later applications with the same mask hit the region transform cache. The transposed program
    /// is interpreted at `[live output cotangents, destination references, residuals]` in the context that this
    /// pullback was built in. For a non-reference input, a [`Reference`](CotangentDestination::Reference) destination
    /// adds the computed value cotangent to the destination's existing contents. For a reference input, the destination
    /// holds the cotangent of the reference's post-execution state on entry and the cotangent of its pre-execution
    /// state on return. An [`Ignore`](CotangentDestination::Ignore) reference destination accumulates through an
    /// internal cotangent reference whose final contents are discarded. A reference-primal destination is threaded
    /// through the raw transposed program by identity. Reference destinations for non-reference inputs and all ignored
    /// inputs have no program output. The returned structure carries [`None`] at every non-`Return` position.
    ///
    /// # Parameters
    ///
    ///   - `seeds`: Output cotangent seeds, aligned with the closure's output structure.
    ///   - `destinations`: Input [`CotangentDestination`]s, aligned with the closure's input structure.
    ///
    /// # Errors
    ///
    /// Returns [`ProgramError::InvalidArgument`] for a seed or destination that violates the matrix above or a
    /// destination that aliases a primal reference or another destination, and propagates transposition and replay
    /// errors otherwise.
    pub fn apply_with_destinations(
        &self,
        seeds: Output::To<CotangentSeed<C::Value>>,
        destinations: Input::To<CotangentDestination<C::Value>>,
    ) -> Result<Input::To<Option<C::Value>>, ProgramError>
    where
        C::Operation: TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
        Input::Family: ParameterizedFamily<CotangentDestination<C::Value>> + ParameterizedFamily<Option<C::Value>>,
        Output::Family: ParameterizedFamily<CotangentSeed<C::Value>>,
    {
        let seeds = seeds.into_parameters().collect::<Vec<_>>();
        let destinations = destinations.into_parameters().collect::<Vec<_>>();
        let cotangents = self.apply_impl(seeds, &destinations)?;
        Ok(Input::To::<Option<C::Value>>::from_parameters(self.input_structure.clone(), cotangents)?)
    }

    /// Replays a scalar gradient pullback and returns non-reference values at every input position. Reference
    /// destinations were initialized to zero before primal execution; freezing them after replay extracts the
    /// derivative with respect to the initial state and prevents the internal accumulators from escaping through
    /// the result.
    fn apply_gradient(
        &self,
        seeds: Vec<CotangentSeed<C::Value>>,
        destinations: Vec<CotangentDestination<C::Value>>,
    ) -> Result<Input::To<C::Value>, ProgramError>
    where
        C::Operation: OperationProvider<C::Type, ReferenceFreezeOperation<C::Type, C::Type>, Operation = C::Operation>
            + TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
    {
        let cotangents = self.apply_impl(seeds, &destinations)?;
        let gradients = destinations
            .into_iter()
            .zip(cotangents)
            .map(|(destination, cotangent)| match destination {
                CotangentDestination::Reference(reference) => {
                    let mut outputs = self.context.bind(
                        C::Operation::provide(ReferenceFreezeOperation::new(), &[reference.r#type().as_ref()])?,
                        Vec::new(),
                        &[reference],
                    )?;
                    check_count!("output", outputs, 1, ProgramError);
                    Ok(outputs.remove(0))
                }
                CotangentDestination::Return => {
                    // `apply_impl` guarantees a returned cotangent for every non-reference `Return` input,
                    // including disconnected and zero-space inputs.
                    Ok(cotangent.unwrap())
                }
                CotangentDestination::Ignore => {
                    // Both callers of this private function obtain destinations from `gradient_destinations`, which
                    // creates only `Return` for non-reference inputs and `Reference` for reference inputs. Neither
                    // caller accepts user-supplied destinations, so `Ignore` cannot reach this branch.
                    unreachable!()
                }
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        Ok(Input::To::<C::Value>::from_parameters(self.input_structure.clone(), gradients)?)
    }

    /// Validates and replays the provided flattened seeds and [`CotangentDestination`].
    /// Only non-reference [`Return`](CotangentDestination::Return) inputs produce values here;
    /// [`Reference`](CotangentDestination::Reference) destinations are updated during replay,
    /// and [`Ignore`](CotangentDestination::Ignore) inputs expose no result.
    fn apply_impl(
        &self,
        seeds: Vec<CotangentSeed<C::Value>>,
        destinations: &[CotangentDestination<C::Value>],
    ) -> Result<Vec<Option<C::Value>>, ProgramError>
    where
        C::Operation: TransposableOperation<C::Constant, C::Operation>
            + ResidualZeroProvider<C::Type>
            + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
            + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
            + From<AddOperation<C::Type>>,
    {
        // Validate the seeds against the complete primal output boundary, including the leaves whose differential
        // spaces contain only zero. Only information-carrying values become program inputs because the compact program
        // has no Single Static Assignment (SSA) input for a zero differential space, and none for a reference output,
        // whose state cotangent is owned by the destination of the input root it forwards.
        if seeds.len() != self.primal_output_types.len() {
            return Err(ProgramError::InvalidInputCount {
                expected: self.primal_output_types.len(),
                actual: seeds.len(),
            });
        }
        let mut program_inputs = Vec::new();
        for (index, (seed, primal_type)) in seeds.into_iter().zip(&self.primal_output_types).enumerate() {
            let cotangent_type = primal_type.cotangent()?;
            match seed {
                CotangentSeed::Value(value) => {
                    if primal_type.is_reference() {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "pullback cotangent seed {index} supplies a value for a primal output of reference \
                                 type {primal_type}; use `CotangentSeed::NoCotangent` for reference outputs"
                            ),
                        });
                    }
                    if value.r#type().as_ref() != &cotangent_type {
                        return Err(ProgramError::MalformedProgram(format!(
                            "pullback cotangent {} has type {} but its primal boundary requires cotangent type {}",
                            index,
                            value.r#type().as_ref(),
                            cotangent_type,
                        )));
                    }
                    if !cotangent_type.is_zero_space() {
                        program_inputs.push(value);
                    }
                }
                CotangentSeed::NoCotangent => {
                    if !primal_type.is_reference() {
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "pullback cotangent seed {index} omits the cotangent of a primal output of type \
                                 {primal_type}; use `CotangentSeed::Value` for non-reference outputs"
                            ),
                        });
                    }
                }
            }
        }

        // Validate the destinations against the complete primal input boundary and derive the structural destination
        // mask. Each destination kind selects its own retained program; runtime buffer identities are never keys.
        if destinations.len() != self.primal_input_types.len() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "pullback received {} cotangent destinations for a primal input boundary with {} leaves",
                    destinations.len(),
                    self.primal_input_types.len(),
                ),
            });
        }
        let mut destination_references = Vec::new();
        for (index, (destination, primal_type)) in destinations.iter().zip(&self.primal_input_types).enumerate() {
            let is_reference = primal_type.is_reference();
            match destination {
                CotangentDestination::Return if is_reference => {
                    return Err(ProgramError::InvalidArgument {
                        message: format!(
                            "pullback destination {index} returns the cotangent of a primal input of reference type \
                             {primal_type} as a value; use `CotangentDestination::Reference` or \
                             `CotangentDestination::Ignore` for reference inputs"
                        ),
                    });
                }
                CotangentDestination::Return | CotangentDestination::Ignore => {}
                CotangentDestination::Reference(reference) => {
                    let cotangent_type = primal_type.cotangent()?;
                    let valid_type = if is_reference {
                        reference.r#type().as_ref() == &cotangent_type
                    } else {
                        reference.r#type().referent().as_ref() == Some(&cotangent_type)
                    };
                    if !valid_type {
                        let requirement = if is_reference {
                            format!("the cotangent reference type {cotangent_type}")
                        } else {
                            format!("a reference storing the cotangent type {cotangent_type}")
                        };
                        return Err(ProgramError::InvalidArgument {
                            message: format!(
                                "pullback destination {} has type {} but its primal input of type {} requires {}",
                                index,
                                reference.r#type().as_ref(),
                                primal_type,
                                requirement,
                            ),
                        });
                    }
                    destination_references.push((index, reference));
                }
            }
        }

        // Every destination reference must denote an allocation distinct from every primal reference and from every
        // other destination. Identity, not generation, is compared: a primal reference that advanced generations after
        // the differentiated closure ran is still the same allocation and still rejected.
        self.primal_references.validate_differentiation_arguments(
            &self.context,
            destination_references
                .iter()
                .map(|&(index, reference)| (DifferentiationBoundaryPosition::Cotangent(index), reference)),
        )?;

        // Transpose under the derived mask (served from the retained transform cache after the first application),
        // close the compact cotangent boundary over the destination references and the primal residuals, and replay it
        // in the originating context. The mask handed to the transposition covers the linear program's tangent inputs
        // only, so the leaves whose tangent spaces contain only zero (and which the compact program omits) drop out.
        let mut transposition_kinds = Vec::with_capacity(destinations.len());
        for (destination, primal_type) in destinations.iter().zip(&self.primal_input_types) {
            if !primal_type.cotangent()?.is_zero_space() {
                transposition_kinds.push(destination.kind());
            }
        }
        let program = self.transposed_program(&transposition_kinds)?;
        for (index, reference) in destination_references {
            if !self.primal_input_types[index].cotangent()?.is_zero_space() {
                program_inputs.push(reference.clone());
            }
        }
        program_inputs.extend(self.residuals.iter().cloned());
        let mut program_input_cotangents = program.interpret_in_context(&self.context, program_inputs)?.into_iter();

        // Reconstruct the public input boundary. Non-reference `Return` leaves consume one result or materialize their
        // typed zero. Non-reference `Reference`/`Ignore` leaves consume nothing. Reference-primal `Reference` leaves
        // consumetheir identity output but expose `None` to the caller. Reference-primal `Ignored` leaves consume
        // nothing.
        let materialize_zeros = destinations
            .iter()
            .map(|destination| matches!(destination, CotangentDestination::Return))
            .collect::<Vec<_>>();
        let input_cotangents =
            self.cotangent_reconstruction.rebuild_with(&self.context, &materialize_zeros, |index, zero| {
                let destination = &destinations[index];
                let primal_type = &self.primal_input_types[index];
                if primal_type.is_reference() {
                    if matches!(destination, CotangentDestination::Reference(_)) {
                        program_input_cotangents.next().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "pullback program omitted the cotangent reference output of input {index}",
                            ))
                        })?;
                    }
                    return Ok(None);
                }
                if !matches!(destination, CotangentDestination::Return) {
                    return Ok(None);
                }
                let cotangent = match zero {
                    Some(zero) => zero,
                    None => program_input_cotangents.next().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "pullback program omitted the cotangent of input {index}",
                        ))
                    })?,
                };
                Ok(Some(cotangent))
            })?;

        if program_input_cotangents.next().is_some() {
            return Err(ProgramError::MalformedProgram("pullback program produced too many cotangents".to_string()));
        }

        Ok(input_cotangents)
    }

    /// Validates the reference-free public boundary required when returning cotangent values without explicit
    /// destinations. Reference inputs or outputs require [`apply_with_destinations`](Self::apply_with_destinations)
    /// instead; references used only inside the differentiated computation are allowed.
    fn validate_reference_boundary(&self) -> Result<(), ProgramError> {
        let leaf = self
            .primal_input_types
            .iter()
            .map(|r#type| ("input", r#type))
            .chain(self.primal_output_types.iter().map(|r#type| ("output", r#type)))
            .find(|(_, r#type)| r#type.is_reference());
        match leaf {
            Some((role, r#type)) => Err(ProgramError::InvalidArgument {
                message: format!(
                    "returning pullback cotangent values requires a reference-free boundary but a primal {role} leaf \
                     has reference type {type}; use `Pullback::apply_with_destinations` to supply cotangent seeds and \
                     destinations",
                ),
            }),
            None => Ok(()),
        }
    }
}

/// [`RegionDriver`] that provides call-scoped access to the [`Region`]s attached to the [`Instruction`] being
/// transposed and to nested transposition work over those regions. The transposition engine constructs a
/// [`TranspositionDriver`] for every instruction. [`RegionDriver`] provides access to its attached child regions
/// and this trait adds access to the current source instruction and transposition recursion.
pub trait TranspositionDriver<V: Value, O: Operation<Type = V::Type>>: RegionDriver<V, O> {
    /// Transposes `region` with respect to `input_indices`, re-entering the active transposition machinery, and returns
    /// a shared handle to the transposed standalone [`Program`]. The result is shared rather than owned because rules
    /// commonly re-attach the transposed program as a callee. An [`Arc`] lets a caching driver serve one artifact for a
    /// callee that several programs share instead of re-transposing it per program, and it lets repeated attachments of
    /// one artifact intern by [`Arc`] identity. The built-in recursive driver therefore serves the region's retained
    /// transposition through the cached counterpart of [`RegionRef::transpose`], while a custom driver that retains
    /// nothing simply transposes the region uncached and wraps the result in [`Arc::new`].
    ///
    /// # Parameters
    ///
    ///   - `region`: Reference to the [`Region`] to transpose.
    ///   - `input_indices`: Unique indices of the region inputs to transpose with respect to, in the desired cotangent
    ///     order. Each index must be in range. Unselected inputs are known parameters of the linear map. An empty slice
    ///     selects no inputs. This is the same selection convention as [`RegionRef::transpose`].
    ///   - `destination_kinds`: One [`CotangentDestinationKind`] per selected input, in `input_indices` order, or empty
    ///     to select the default kinds (i.e., [`Return`](CotangentDestinationKind::Return) for non-reference inputs and
    ///     [`Reference`](CotangentDestinationKind::Reference) for reference inputs). For example, indices `[2, 0]` with
    ///     kinds `[Return, Ignore]` return input 2's cotangent and omit input 0's cotangent. Refer to
    ///     [`RegionRef::transpose`] for the complete boundary contract.
    fn transpose_program(
        &self,
        region: RegionRef<'_, V, O>,
        input_indices: &[usize],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError>;

    /// Returns the source scope for this rule invocation that consists of its containing [`Region`], [`InstructionId`],
    /// and [`Instruction`]. Reference rules use this scope to resolve operand positions to primal values and reference
    /// views. The source instruction is distinct from its attached child regions exposed by [`RegionDriver::regions`].
    ///
    /// Drivers for detached rules may return [`None`], which is the default. A driver that provides a source scope
    /// must return the same instruction throughout the rule invocation, with its identity identifying that instruction
    /// within the returned region. Reference-state queries require this scope.
    fn scope(&self) -> Result<Option<(RegionRef<'_, V, O>, InstructionId, &Instruction<O>)>, ProgramError> {
        Ok(None)
    }
}

impl<V: Value, O: Operation<Type = V::Type>> TranspositionDriver<V, O> for EmptyRegionDriver {
    #[inline]
    fn transpose_program(
        &self,
        _region: RegionRef<'_, V, O>,
        _input_indices: &[usize],
        _destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        Err(ProgramError::MalformedProgram("empty region driver cannot transpose a program".to_string()).into())
    }
}

/// [`TranspositionDriver`] scoped to one replayed linear [`Instruction`], retaining its source region and borrowing
/// the child [`Region`]s attached to that instruction.
struct RecursiveTranspositionDriver<'r, V: Value, O: Operation<Type = V::Type>> {
    /// Borrowed source [`Region`] containing the [`Instruction`] being transposed.
    source_region: RegionRef<'r, V, O>,

    /// Validated index of the source [`Instruction`] within `source_region`.
    source_instruction_index: usize,

    /// Borrowed attached [`Region`]s, in region order.
    attached_regions: Vec<RegionRef<'r, V, O>>,
}

impl<'r, V: Value, O: Operation<Type = V::Type>> RecursiveTranspositionDriver<'r, V, O> {
    /// Creates a new [`RecursiveTranspositionDriver`] for the [`Instruction`] at `index` in `region`, resolving its
    /// attached child [`Region`]s.
    fn new(region: RegionRef<'r, V, O>, index: usize) -> Result<Self, ProgramError> {
        let instruction = region.instructions().get(index).ok_or_else(|| {
            ProgramError::MalformedProgram(format!("transposition driver refers to missing instruction {index}"))
        })?;
        let attached_regions =
            instruction.regions().iter().map(|id| region.with_id(*id)).collect::<Result<Vec<_>, _>>()?;
        Ok(Self { source_region: region, source_instruction_index: index, attached_regions })
    }
}

impl<V: Value, O: Operation<Type = V::Type>> RegionDriver<V, O> for RecursiveTranspositionDriver<'_, V, O> {
    #[inline]
    fn regions<'r>(&'r self) -> impl Iterator<Item = RegionRef<'r, V, O>>
    where
        V: 'r,
        O: 'r,
    {
        self.attached_regions.iter().copied()
    }
}

impl<
    V: Value<Type: DifferentiableType>,
    O: TransposableOperation<V, O>
        + ResidualZeroProvider<V::Type>
        + OperationProvider<V::Type, ReferenceNewOperation<V::Type, V::Type>, Operation = O>
        + OperationProvider<V::Type, ReferenceAddUpdateOperation<V::Type, V::Type>, Operation = O>
        + From<AddOperation<V::Type>>,
> TranspositionDriver<V, O> for RecursiveTranspositionDriver<'_, V, O>
{
    #[inline]
    fn transpose_program(
        &self,
        region: RegionRef<'_, V, O>,
        input_indices: &[usize],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        region.transpose_shared(input_indices, &[], destination_kinds)
    }

    #[inline]
    fn scope(&self) -> Result<Option<(RegionRef<'_, V, O>, InstructionId, &Instruction<O>)>, ProgramError> {
        Ok(Some((
            self.source_region,
            InstructionId::new(self.source_region.id(), self.source_instruction_index),
            &self.source_region.instructions()[self.source_instruction_index],
        )))
    }
}

/// Represents [`Operation`]s that provide a transposition rule for linear [`Program`]s. Reading a linear
/// [`Instruction`] as a linear map `y = L(x)` (in differentiation, `L` is a piece of the tangent map `(∂f/∂x)(x)`
/// produced by linearization), the [`transpose`](Self::transpose) function computes the action of the _transposed_
/// (i.e., adjoint) map: given a cotangent `ȳ` for the output, it accumulates the cotangent contribution `x̄ = Lᵀ(ȳ)`
/// for each input, where `Lᵀ` is the unique linear map satisfying `⟨ȳ, L(x)⟩ = ⟨Lᵀ(ȳ), x⟩`. Applied instruction by
/// instruction in reverse program order, these rules compute the Vector-Jacobian Product (VJP) `x̄ = (∂f/∂x)(x)ᵀ · ȳ`
/// that reverse mode differentiation is built on. Cotangents flow symbolically as [`MaybeZero`]s: rules may reuse
/// existing cotangents, omit structural-zero contributions, or stage additional linear operations in the active
/// [`TracingContext`]. The rule does not receive concrete primal values. Instead, it receives each input's/operand's
/// [`PartialValue`] knowledge (i.e., its [`Type`] when the operand is linear, or the staged [`Tracer`] carrying its
/// runtime value when the operand is a known factor) and any further metadata must be encoded in the operation itself.
///
/// Refer to the documentation of [`Program::transpose`] for more information on what _transposition_ means here and
/// how it relates to the algebraic notion of transposition.
///
/// # Design
///
/// A transpose rule can express its mathematics by returning one cotangent contribution per input and letting the
/// transposition engine sum those contributions. The accumulator interface preserves that value-based algorithm while
/// also letting a rule use existing gradient storage and avoid computing unrequested gradients. Each input receives a
/// [`CotangentAccumulator`] handle owned by the active [`TranspositionContext`]. The handle describes whether a
/// contribution is needed and provides a common way to submit it, whether the caller wants a returned value, supplies
/// a gradient reference, or ignores that gradient.
///
/// Accumulating through a handle does not inherently stage a reference mutation. For a returned gradient, the context
/// adds each value contribution to a running sum as it arrives. The generated computation can therefore remain
/// functional, and value-only operation families do not need reference capabilities merely to submit contributions.
/// For a caller-provided gradient reference, accumulation instead stages an additive update that preserves the buffer's
/// existing contents. The handle itself exists during program construction; it is not a new runtime value type or a
/// requirement to allocate gradient buffers.
///
/// Exposing an available buffer matters especially for sparse contributions. For `y = x[i]`, the input cotangent is
/// zero everywhere except at `i`. A value-based rule can construct that full-sized contribution and submit it. A rule
/// with access to an existing gradient reference can instead emit `gradient[i] += ȳ`, touching only the affected entry.
/// A compiler may recover this optimization from value operations, but doing so requires recognizing the pattern and
/// proving that storage can be reused. Buffer access lets a rule express the update directly, including across custom
/// operation boundaries. Dense rules can keep their value formulas; direct updates are an opportunity rather than a
/// guarantee of better performance for every operation.
///
/// Passing raw gradient references to every rule would also expose storage, but would require references even when the
/// caller only wants a value. Handles keep that choice in the context and expose reference access only when a buffer is
/// available in the rule's value family. They do not make an enclosing composite family's reference representable in a
/// homogeneous array family: projected rules use value contributions when they cannot represent the buffer, and direct
/// indexed updates require a reference-capable implementation. Additive gradient storage for non-reference inputs also
/// remains distinct from reference-state adjoints, whose transpose rules may need to read, replace, or clear state
/// through the context's separate reference operations.
///
/// # Deriving Transposable Operation Enums
///
/// `#[derive(Operation)]` generates a [`TransposableOperation`] dispatcher when the enum specifies
/// `#[ryft(dispatch(transposition))]`. Selecting it enables *reverse-mode* differentiation for programs staged in the
/// operation family. The independent `differentiation` selection adds forward-mode (JVP) support, and the transposition
/// dispatcher is what [`Program::transpose`] and the reverse-mode entry points build on. The generated implementation
/// is intentionally only a dispatcher: it matches on the enum variant and forwards [`transpose`](Self::transpose) to
/// the wrapped payload. Operation-specific transpose semantics still live on the concrete payload types. The derived
/// implementation follows the same enum-shape rules as `#[derive(Operation)]`:
///
///   - The derivation macro input must be an enum.
///   - Every variant must be a tuple variant with exactly one payload field.
///   - A payload may be stored directly as `Payload` or indirectly as `Box<Payload>`.
///   - The enum derives [`Operation`] and selects `transposition`.
///
/// The generated implementation is:
///
///   - `impl TransposableOperation<V, Enum> for Enum`, where the transposition value type `V` carries the primary
///     type `T` selected using the same rules that are used for inferring `T` in the `#[derive(Operation)]` macro.
///   - For enums with one `Value<Type = T>` parameter, that parameter is treated as the operation family's stored
///     constant type and the generated implementation is generic over a separate transposition value type `V`. For
///     enums with two or more `Value<Type = T>` parameters, the first value parameter is treated as the
///     tangent/cotangent value type and later value parameters are constants or captured factors.
///   - Concrete payload variants forward directly to their payload implementations and receive a generated
///     `Payload: TransposableOperation<V, Enum>` `where` predicate. Payload-specific capability requirements should
///     live on the payload's own [`TransposableOperation`] implementation; the enum derivation carries them through
///     this generated payload bound.
///   - Bare generic payload variants such as `Extension(Extension)` receive the same generated
///     `Extension: TransposableOperation<V, Enum>` bound, because the macro cannot know which concrete extension
///     type will be substituted by the caller.
///
/// Higher-order payloads that need to transpose nested linear programs request that work through their
/// [`TranspositionDriver`] (whose built-in implementation calls [`RegionRef::transpose_shared`]) instead of carrying
/// a direct `Enum: TransposableOperation<V, Enum>` bound, which keeps the enum's bound graph finite. When payload rules
/// need value capabilities, those requirements belong on the payload implementations themselves, and the generated
/// dispatcher inherits them through its per-payload trait bounds.
///
/// ## Example
///
/// ```rust
/// # use ryft_core as ryft;
/// # use ryft_core::arrays::ArrayType;
/// # use ryft_core::{ConstantOperation, Value, ZeroOperation};
/// # use ryft_macros::Operation;
///
/// #[derive(Clone, Debug, Operation)]
/// #[ryft(dispatch(transposition))]
/// enum LinearOperation<V: Value<Type = ArrayType>> {
///     Zero(ZeroOperation<ArrayType>),
///     Constant(ConstantOperation<V>),
/// }
/// ```
pub trait TransposableOperation<V: Value, O: Operation<Type = V::Type>>: Operation<Type = V::Type> {
    /// Applies this operation's transpose rule to symbolic output cotangents, accumulating `x̄ = Lᵀ(ȳ)` for the linear
    /// map `y = L(x)`. Each operand has an opaque [`CotangentAccumulator`], and the implementor must check
    /// [`is_needed`](CotangentAccumulator::is_needed) before computing an expensive contribution, and then submit it
    /// with [`accumulate`](CotangentAccumulator::accumulate). A rule may submit multiple contributions or none.
    /// Repeated operands can share storage, so each operand's mathematical contribution must be submitted
    /// independently.
    ///
    /// A rule capable of updating a gradient buffer directly can query [`reference`](CotangentAccumulator::reference).
    /// For example, a slice rule can add its output cotangent into a view of that buffer instead of constructing a
    /// full-sized padded gradient. When no buffer is available, it uses its value formula. Handles expose whether a
    /// cotangent is needed and its optional storage without changing which primal operands are linear or known.
    ///
    /// Rules must be deterministic structural functions of their inputs (i.e., of this operation, the input knowledge,
    /// the output cotangents, and the attached regions reachable through `driver`), because the programs derived from
    /// them may be retained and replayed by the per-region transform cache behind
    /// [`transpose_program`](TranspositionDriver::transpose_program). When the `debug_assertions` feature is enabled,
    /// every cache hit checks this with a rendering-based diagnostic whose fidelity is bounded by [`Operation::render`]
    /// on operation metadata and by [`Display`](std::fmt::Display) on constants.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`TranspositionContext`], which dereferences to the [`TracingContext`] in which rules
    ///     stage additional linear operations. Reference-state rules access the instruction's reference operands
    ///     through this context; their state updates do not use the `accumulators` handles.
    ///   - `driver`: Call-scoped nested-region [`TranspositionDriver`] for the current instruction.
    ///   - `inputs`: Per-input [`PartialValue`] knowledge, in operation input order. A [`PartialValue::Unknown`]
    ///     entry marks an input that is linear in the transposed program and therefore receives a cotangent
    ///     contribution of its cotangent type. The type also recovers cotangent shapes not derivable from the operation
    ///     payload alone (e.g., a broadcast operation's pre-broadcast shape). A [`Known`](PartialValue::Known) entry
    ///     marks an input whose runtime value rides in the pullback function as a tracer: bilinear rules such as the
    ///     one for `Mul` read it directly to scale the output cotangent into the linear input's contribution, and the
    ///     rule contributes nothing for that input. Each input's [`Type`] is available either way through the [`Typed`]
    ///     trait. Rules that transpose fully linear operations need only read the types, since every input is then
    ///     [`PartialValue::Unknown`].
    ///   - `outputs`: Symbolic cotangents for the instruction's outputs, in operation output order. A reference-typed
    ///     output never carries a live cotangent (its state cotangent lives in the accumulator of the root it forwards)
    ///     and always arrives as a structural zero.
    ///   - `accumulators`: [`CotangentAccumulator`]s, one per input in operand order, owned by `context`. Known,
    ///     ignored, and reference-state operands have unneeded handles. Contributions must have the operand's cotangent
    ///     type and belong to this context's builder, including when the handle is unneeded.
    fn transpose<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>;
}

/// Transpose rule for a member operation that runs in its enclosing operation family's [`TranspositionContext`].
/// Unlike [`TransposableOperation`], this trait permits the operation's [`Type`] to differ from the context's type.
/// For example, an array slice operates on arrays, but its parent context may also represent references. Its member
/// rule can then add a cotangent into a reference slice without trying to project that reference into the array family.
///
/// This is the reverse mode counterpart of [`MemberDifferentiableOperation`](crate::MemberDifferentiableOperation).
/// Generated operation family dispatchers call it for computational projected members. A member family delegates to
/// its individual operations when they need parent family capabilities, and can use [`transpose_projected_operation`]
/// for region-free rules that remain within the member family. Implementations specify the projection and operation
/// capabilities they need; this trait does not require every enclosing family to support references.
///
/// Accumulators remain owned by the enclosing context. The contribution, ownership, and deterministic staging
/// requirements of [`TransposableOperation::transpose`] apply unchanged, including when a rule delegates to projection.
pub trait MemberTransposableOperation<V: Value, O: Operation<Type = V::Type>>:
    Operation<Type: DifferentiableType>
{
    /// Applies this member's transpose rule using the enclosing family's values and cotangent accumulators.
    ///
    /// # Parameters
    ///
    ///   - `context`: Active [`TranspositionContext`] for the enclosing operation family, which dereferences to the
    ///     [`TracingContext`] in which rules stage additional linear operations. Reference-state rules access the
    ///     instruction's reference operands through this context; their state updates do not use the `accumulators`
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
    ///   - `accumulators`: [`CotangentAccumulator`]s, one per input in operand order, owned by `context`. Known,
    ///     ignored, and reference-state operands have unneeded handles. Contributions must have the operand's cotangent
    ///     type and belong to this context's builder, including when the handle is unneeded.
    fn transpose_in_parent<D: TranspositionDriver<V, O>>(
        &self,
        context: &mut TranspositionContext<V, O>,
        driver: &D,
        inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
        outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
        accumulators: &[CotangentAccumulator],
    ) -> Result<(), DifferentiationError>;
}

impl<
    T: DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O>
        + ResidualZeroProvider<T>
        + OperationProvider<T, ReferenceNewOperation<T, T>, Operation = O>
        + OperationProvider<T, ReferenceAddUpdateOperation<T, T>, Operation = O>
        + From<AddOperation<T>>,
> RegionRef<'_, V, O>
{
    /// Transposes this borrowed linear _pushforward_ [`Region`] into its reverse-mode _pullback_ with respect to
    /// `input_indices`, using `destination_kinds` to select how their cotangents cross the pullback boundary. Refer
    /// to [`Program::transpose_with_respect_to`] for the resulting program's boundary contract.
    ///
    /// This function constructs an owned program without using the region's transform cache. Use
    /// [`Self::transpose_shared`] to reuse a retained program across calls or attach it to another program.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Selected linear input indices, in requested cotangent order. Each index must be in range
    ///     and appear at most once. An empty slice selects no inputs.
    ///   - `zero_residual_input_indices`: Per-selected-input residual indices supplying runtime dimensions for
    ///     disconnected cotangent zeros, or empty when no mappings are needed. For example, a dynamically shaped
    ///     input can need a residual dimension to construct its zero gradient. Linearization appends these dimension
    ///     residuals to its tangent program. A nonempty slice must have one entry per selected input.
    ///   - `destination_kinds`: One kind per selected input, or empty to use `Return` for non-reference inputs and
    ///     `Reference` for reference inputs. Non-reference inputs also accept `Reference` and `Ignore`; reference
    ///     inputs accept `Reference` and `Ignore` but cannot return their state cotangent as a value.
    #[inline]
    pub fn transpose(
        &self,
        input_indices: &[usize],
        zero_residual_input_indices: &[Vec<usize>],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.transpose_impl(&TranspositionTransformArguments::new(
            *self,
            input_indices,
            zero_residual_input_indices,
            destination_kinds,
        )?)
    }

    /// Transposes this borrowed linear _pushforward_ [`Region`] through its retained transform cache, returning a
    /// shared handle to the resulting _pullback_ [`Program`]. The arguments and boundary contract are the same as
    /// [`Self::transpose`].
    ///
    /// Content-preserving copies of one sealed region share one artifact per argument list. This avoids repeating
    /// transposition for a shared callee and lets consumers intern repeated attachments by [`Arc`] identity.
    /// Destination kinds are resolved to their defaults before cache lookup, so an explicit selection equal to the
    /// defaults shares the default artifact. Input order and zero residual mappings remain part of the cache key.
    ///
    /// Recursive transposition of the region and argument list currently in flight on this thread is served without
    /// the cache, using the same implementation as [`Self::transpose`].
    pub fn transpose_shared(
        &self,
        input_indices: &[usize],
        zero_residual_input_indices: &[Vec<usize>],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        let arguments =
            TranspositionTransformArguments::new(*self, input_indices, zero_residual_input_indices, destination_kinds)?;
        let artifact =
            (*self).transform::<TranspositionTransform, _, DifferentiationError>(arguments, |region, arguments| {
                let program = region.transpose_impl(arguments)?;
                Ok(TransformArtifact::new(vec![Arc::new(program)], ()))
            })?;
        let (programs, ()) = artifact.into_parts();
        let mut programs = programs.into_iter();
        let program = programs.next().unwrap();
        assert!(programs.next().is_none(), "transposition transform retained more than one program");
        Ok(program)
    }

    /// Constructs the pullback for validated input indices and resolved destination kinds. Both public entry points
    /// use this implementation, so cache misses and debug cache checks do not repeat argument normalization.
    fn transpose_impl(
        &self,
        arguments: &TranspositionTransformArguments,
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        let input_indices = arguments.input_indices.as_slice();
        let zero_residual_input_indices = arguments.zero_residual_input_indices.as_slice();
        let destination_kinds = arguments.destination_kinds.as_slice();

        // Index the validated selection and its destination kinds by program input for linearity propagation and
        // pullback boundary construction below.
        let input_count = self.input_ids().len();
        let mut kind_by_input = vec![None; input_count];
        input_indices.iter().zip(destination_kinds).for_each(|(&index, &kind)| {
            kind_by_input[index] = Some(kind);
        });

        /// Helper internal enum for the [`materialize_known`] implementation.
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum MaterializationState {
            Unseen,
            Visiting,
            Complete,
        }

        /// Helper internal enum for the [`materialize_known`] implementation.
        #[derive(Copy, Clone, PartialEq, Eq)]
        enum MaterializationStep {
            Visit(AtomId),
            Replay(usize),
        }

        /// Replays the pure producer subgraph of one known atom into the pullback builder using an iterative postorder
        /// traversal. Program inputs are seeded in `known_map` by the caller, constants are copied directly, and all
        /// outputs of a replayed instruction are memoized together so shared producers and sibling results are emitted
        /// only once. `materialization_state` distinguishes scheduled producers from completed ones, both detecting a
        /// malformed cycle and keeping the traversal independent of the native call stack.
        ///
        /// # Parameters
        ///
        ///   - `program`: Source program containing the known atom and its producer subgraph.
        ///   - `instruction_by_output`: Source-instruction index for each produced atom, or `None` for atoms
        ///     without an instruction producer.
        ///   - `linear`: Per-source-atom mask indicating whether the atom depends on a selected linear input.
        ///   - `region_mappings`: Replay-scoped region remapping shared by every known producer
        ///     materialized from the source arena.
        ///   - `builder`: Destination pullback builder into which demanded pure producers are replayed.
        ///   - `known_map`: Per-source-atom mapping to an already materialized pullback atom.
        ///   - `materialization_state`: Per-source-instruction traversal state used for memoization
        ///      and cycle detection.
        ///   - `atom`: ID of the known source atom to materialize in the pullback builder.
        fn materialize_known<V: Value, O: Operation<Type = V::Type>>(
            program: RegionRef<'_, V, O>,
            instruction_by_output: &[Option<usize>],
            linear: &[bool],
            region_mappings: &RegionReplayMappings<V, O>,
            builder: &Rc<RefCell<ProgramBuilder<V, O>>>,
            known_map: &mut [Option<AtomId>],
            materialization_state: &mut [MaterializationState],
            atom: AtomId,
        ) -> Result<AtomId, ProgramError> {
            let mut steps = vec![MaterializationStep::Visit(atom)];
            while let Some(step) = steps.pop() {
                match step {
                    MaterializationStep::Visit(current) => {
                        if known_map.get(current.index()).copied().flatten().is_some() {
                            continue;
                        }
                        if *linear.get(current.index()).ok_or(ProgramError::UnboundAtomId { id: current })? {
                            return Err(ProgramError::MalformedProgram(
                                "a linear atom was requested as a known transpose operand".to_string(),
                            ));
                        }
                        let source =
                            program.atoms().get(current.index()).ok_or(ProgramError::UnboundAtomId { id: current })?;
                        if let Atom::Constant(value) = source {
                            let mapped = builder.borrow_mut().add_constant(value.clone());
                            known_map[current.index()] = Some(mapped);
                            continue;
                        }
                        let instruction_index =
                            instruction_by_output.get(current.index()).copied().flatten().ok_or_else(|| {
                                ProgramError::MalformedProgram("known variable atom has no owning instruction".into())
                            })?;
                        let instruction = program
                            .instructions()
                            .get(instruction_index)
                            .ok_or_else(|| ProgramError::MalformedProgram("known atom producer is missing".into()))?;

                        // Replaying an ordered assertion is safe. It validates the same saved primal value and cannot
                        // introduce a failure that successful primal execution did not already admit. Other effects
                        // could be duplicated or reordered and therefore require prior residualization.
                        let effects = program.instruction_effects(instruction_index)?.classes();
                        if effects.into_iter().any(|effect| effect != EffectClass::OrderedAssertion) {
                            return Err(ProgramError::UnsupportedOperation {
                                message: format!(
                                    "partition-aware transpose cannot replay effectful known intermediate producer \
                                     `{}`; partial-evaluate it into a residual input first",
                                    instruction.operation().name(),
                                ),
                            });
                        }

                        match materialization_state.get_mut(instruction_index).ok_or_else(|| {
                            ProgramError::MalformedProgram("known atom producer state is missing".into())
                        })? {
                            state @ MaterializationState::Unseen => *state = MaterializationState::Visiting,
                            MaterializationState::Visiting => {
                                return Err(ProgramError::MalformedProgram(
                                    "known intermediate producer graph contains a cycle".into(),
                                ));
                            }
                            MaterializationState::Complete => {
                                return Err(ProgramError::MalformedProgram(
                                    "materialized known producer output was not remapped".into(),
                                ));
                            }
                        }
                        steps.push(MaterializationStep::Replay(instruction_index));
                        steps.extend(instruction.inputs().iter().rev().copied().map(MaterializationStep::Visit));
                    }
                    MaterializationStep::Replay(instruction_index) => {
                        let instruction = program
                            .instructions()
                            .get(instruction_index)
                            .ok_or_else(|| ProgramError::MalformedProgram("known atom producer is missing".into()))?;
                        let inputs = instruction
                            .inputs()
                            .iter()
                            .map(|input| {
                                known_map.get(input.index()).copied().flatten().ok_or_else(|| {
                                    ProgramError::MalformedProgram(
                                        "known producer input was not remapped before replay".into(),
                                    )
                                })
                            })
                            .collect::<Result<Vec<_>, _>>()?;
                        let driver = ReplayRegionDriver::new(program, instruction.regions(), region_mappings)?;
                        let region_input_types = vec![None; instruction.regions().len()];
                        let regions = driver.import_into(builder, &region_input_types)?;
                        // Replaying an unchanged known producer into the pullback is a structural copy,
                        // so that its provenance is preserved verbatim.
                        let outputs = builder
                            .borrow_mut()
                            .add_instruction(
                                instruction.operation().clone(),
                                regions,
                                inputs,
                                Some(instruction.provenance().clone()),
                            )?
                            .to_vec();
                        check_count!("output", outputs, instruction.outputs().len(), ProgramError);
                        for (source, mapped) in instruction.outputs().iter().copied().zip(outputs) {
                            known_map[source.index()] = Some(mapped);
                        }
                        *materialization_state.get_mut(instruction_index).ok_or_else(|| {
                            ProgramError::MalformedProgram("known atom producer state is missing".into())
                        })? = MaterializationState::Complete;
                    }
                }
            }
            known_map
                .get(atom.index())
                .copied()
                .flatten()
                .ok_or_else(|| ProgramError::MalformedProgram("known producer output was not remapped".into()))
        }

        // Propagate operand linearity forward over the primal atoms. A program-input atom is linear when it has a
        // selected destination kind, a constant atom is always known (non-linear), and an instruction result is linear
        // when any of its operands is linear. Because instructions are stored in evaluation order, a single forward
        // pass suffices: every operand atom of an instruction is defined before that instruction. Known-only producers
        // remain primal computations whose outputs may supply coefficients to transpose rules. A reference allocated
        // inside the linear program is linear regardless of its initial value: it is mutable state of the linear map
        // whose contents are linear values (a tangent reference allocated from a materialized zero tangent is the
        // common case), so its root receives a cotangent accumulator and the stores into it transpose into on-reference
        // cotangents. Hand-built linear programs that allocate a reference holding known (non-linear) contents are
        // therefore not supported by the direct entry points (such an allocation is treated as linear state as well).
        let mut linear = vec![false; self.atoms().len()];
        self.input_ids().iter().zip(&kind_by_input).try_for_each(|(&input, kind)| {
            *linear.get_mut(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? = kind.is_some();
            Ok::<_, ProgramError>(())
        })?;
        for instruction in self.instructions().iter() {
            let mut output_is_linear = false;
            for input in instruction.inputs().iter().copied() {
                if *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                    output_is_linear = true;
                    break;
                }
            }
            let effects = instruction.operation().effects();
            for (output_index, output) in instruction.outputs().iter().copied().enumerate() {
                let allocates = effects.allocation_output_indices().any(|index| index == output_index);
                *linear.get_mut(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })? =
                    output_is_linear || allocates;
            }
        }

        // Stage the pullback into a fresh tracing context's builder, and reserve the main structural vectors up
        // front. These are conservative lower bounds that cover cotangent inputs, one instruction per reversed primal
        // instruction, and possible zero outputs for disconnected primal inputs. The context is scoped to this region
        // so that reference rules can reach the accumulators of their operands. A linear input represents tangent state
        // owned by this invocation and may be consumed by its source program. Known residual references remain
        // borrowed. Transposition replaces consumption with accumulation into the input's cotangent reference,
        // so the caller's cotangent destination remains live after replay.
        let consumable_inputs = self
            .input_ids()
            .iter()
            .zip(&kind_by_input)
            .enumerate()
            .filter_map(|(index, (input, kind))| {
                (kind.is_some() && self.atoms()[input.index()].r#type().is_reference()).then_some(index)
            })
            .collect();
        let mut context = TranspositionContext::for_region(TracingContext::<V, O>::new(), *self, consumable_inputs)?;
        let builder = context.builder().clone();
        {
            let mut builder_borrow = builder.borrow_mut();
            builder_borrow
                .atoms
                .reserve(self.output_ids().len() + self.instructions().len() + self.input_ids().len());
            builder_borrow.input_ids.reserve(self.output_ids().len());
            builder_borrow.instructions.reserve(self.instructions().len() + self.input_ids().len());
        }

        // A reference-typed output either forwards an input root, whose state cotangent lives in that input's
        // accumulator, or escapes a local allocation, which cannot be pulled back because its later uses are unknown
        // to the program. A derived view at the boundary is rejected as well, because its cotangent would have to be
        // a view of a root cotangent that the boundary does not expose. The analysis handle is shared with the reverse
        // walk below, which consults it for state liveness.
        let analysis = context.reference_analysis().cloned();
        if let Some(analysis) = &analysis {
            for (output_index, (output, root)) in self.output_ids().iter().zip(analysis.output_roots()).enumerate() {
                match root {
                    Some(ReferenceRoot::Allocation { .. }) => {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "output {output_index} is a reference allocated inside the transposed program and \
                                 cannot be pulled back because its later uses are unknown to the program",
                            ),
                        }
                        .into());
                    }
                    Some(ReferenceRoot::RegionInput { .. }) if analysis.is_view(ValueId::new(self.id(), *output)) => {
                        return Err(ProgramError::UnsupportedOperation {
                            message: format!(
                                "output {output_index} is a derived view of a reference and cannot be transposed; \
                                 return the viewed reference and apply the view outside the transposed program",
                            ),
                        }
                        .into());
                    }
                    _ => {}
                }
            }
        }

        // Seed the reverse pass with one cotangent input for each non-reference primal output, typed with that output's
        // cotangent type. A differentiable output carries its cotangent dual (e.g., swapping unreduced and reduced
        // sharding axes for arrays). A non-differentiable output, such as a Boolean, integer, or token, uses the
        // first-class zero-space type, which keeps the numbering of the boundary stable. A reference-typed output gets
        // no slot: it forwards an input root whose state cotangent is the accumulator of that input. The adjoint table
        // is indexed by source atoms. Value cotangent slots maintain a running sum in submission order until their
        // producer is visited; supplied gradient buffers receive updates directly. Propagate the cotangent mask forward
        // from requested inputs independently of primal linearity. Until nested regions use specialized masks, keep
        // state and region-bearing paths conservative so backward effects remain live.
        let conservative_cotangent_mask =
            analysis.is_some() || self.instructions().iter().any(|instruction| !instruction.regions().is_empty());
        let mut cotangent_mask =
            if conservative_cotangent_mask { linear.clone() } else { vec![false; self.atoms().len()] };
        input_indices.iter().zip(destination_kinds).for_each(|(&index, &kind)| {
            let input = self.input_ids()[index];
            cotangent_mask[input.index()] = kind != CotangentDestinationKind::Ignore;
        });
        if !conservative_cotangent_mask {
            for instruction in self.instructions() {
                let needed = instruction.inputs().iter().any(|input| cotangent_mask[input.index()]);
                instruction.outputs().iter().for_each(|output| cotangent_mask[output.index()] = needed);
            }
        }
        let atom_accumulators = self
            .atoms()
            .iter()
            .enumerate()
            .map(|(index, atom)| {
                let cotangent_type = atom.r#type().cotangent()?;
                let needed = linear[index] && cotangent_mask[index] && !cotangent_type.is_reference();
                Ok(context.cotangent_accumulator(cotangent_type, needed))
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;
        let mut cotangent_inputs = Vec::new();
        for output in self.output_ids().iter().copied() {
            let output_atom = self.atoms().get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
            let output_type = output_atom.r#type();
            if output_type.is_reference() {
                continue;
            }
            let cotangent_type = output_type.cotangent()?;
            let has_cotangent = !cotangent_type.is_zero_space();
            let cotangent_input = builder.borrow_mut().add_input(cotangent_type);
            if has_cotangent && *linear.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })? {
                cotangent_inputs.push((output, cotangent_input));
            }
        }

        // Install one cotangent accumulator per linear reference input, in program-input order. A `Reference` kind
        // exposes the caller-owned cotangent reference as a pullback input following the output cotangents, while an
        // `Ignore` kind starts unallocated and is allocated lazily by the first rule that needs it.
        self.input_ids()
            .iter()
            .copied()
            .zip(&kind_by_input)
            .enumerate()
            .filter(|(_, (_, kind))| kind.is_some())
            .try_for_each(|(index, (input, _))| -> Result<(), DifferentiationError> {
                let input_type =
                    self.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?.r#type();
                if !input_type.is_reference() {
                    if kind_by_input[index] == Some(CotangentDestinationKind::Reference) {
                        let cotangent_type = input_type.cotangent()?;
                        // Ask the operation family for its reference type without allocating a runtime reference.
                        let allocation = O::provide(ReferenceNewOperation::<T, T>::new(), &[&cotangent_type])?;
                        let reference_types =
                            allocation.infer_output_types(std::slice::from_ref(&cotangent_type), &[])?;
                        check_count!("output", reference_types, 1, ProgramError);
                        let reference_type = reference_types.into_iter().next().unwrap();
                        let update = O::provide(
                            ReferenceAddUpdateOperation::<T, T>::new(),
                            &[&reference_type, &cotangent_type],
                        )?;
                        let reference = context.input(reference_type);
                        let handle = &atom_accumulators[input.index()];
                        let slot = &mut context.cotangent_storage[handle.storage_index];
                        *slot = CotangentStorage::Buffer { cotangent_type, reference, operation: update };
                    }
                    return Ok(());
                }

                let cotangent_type = input_type.cotangent()?;
                let root = ReferenceRoot::RegionInput { region: self.id(), input_index: index };
                let accumulator = match kind_by_input[index] {
                    Some(CotangentDestinationKind::Reference) => {
                        let destination = builder.borrow_mut().add_input(cotangent_type.clone());
                        CotangentReferenceAccumulator::Allocated {
                            reference: context.tracer(destination, Some(cotangent_type)),
                        }
                    }
                    Some(CotangentDestinationKind::Ignore) => {
                        CotangentReferenceAccumulator::Unallocated { cotangent_type }
                    }
                    Some(CotangentDestinationKind::Return) | None => {
                        return Err(ProgramError::MalformedProgram(format!(
                            "linear reference input {index} has no resolved cotangent destination kind"
                        ))
                        .into());
                    }
                };

                context.reference_accumulators.insert(root, accumulator);
                Ok(())
            })?;

        // Submit seeds only after caller-provided destinations are installed. An identity output can refer directly
        // to an input, and repeated outputs must update its buffer without first constructing a separate value sum.
        cotangent_inputs.into_iter().try_for_each(|(output, cotangent_input)| {
            let contribution = MaybeZero::Value(context.tracer(cotangent_input, None));
            atom_accumulators[output.index()].accumulate(&mut context, contribution)
        })?;

        // Add a pullback input carrying the runtime value of each unselected program input, after the output cotangents
        // and supplied destination buffers. These known values retain their source types and program-input order.
        // Record them in `known_map` by source atom so transpose rules can read their pullback values.
        let mut known_map = vec![None; self.atoms().len()];
        self.input_ids().iter().zip(&kind_by_input).filter(|(_, kind)| kind.is_none()).try_for_each(
            |(&input, _)| {
                let input_atom = self.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                let known_input = builder.borrow_mut().add_input(input_atom.r#type().into_owned());
                known_map[input.index()] = Some(known_input);
                Ok::<_, ProgramError>(())
            },
        )?;

        // Retain known primal values that can supply runtime dimensions for reference-state zeros. Static types
        // need no dimension source, and reference values cannot supply dimensions through this protocol.
        if context.reference_analysis.is_some() {
            for atom in known_map.iter().flatten() {
                let source = context.tracer(*atom, None);
                if !source.r#type().is_reference() && source.r#type().identities().next().is_some() {
                    context.dimension_sources.push(source);
                }
            }
        }

        // Every linear reference allocation of this region gets an accumulator that starts unallocated: the first rule
        // that accumulates into the root allocates it, and the allocation's own transpose freezes it into the cotangent
        // of the initial value (or yields a symbolic zero when nothing ever accumulated).
        for (instruction_index, instruction) in self.instructions().iter().enumerate() {
            for output_index in instruction.operation().effects().allocation_output_indices() {
                let output = *instruction.outputs().get(output_index).ok_or_else(|| {
                    ProgramError::MalformedProgram(format!(
                        "`{}` declares a reference allocation at output {} but has {} outputs",
                        instruction.operation().name(),
                        output_index,
                        instruction.outputs().len(),
                    ))
                })?;
                let cotangent_type = self.atoms()[output.index()].r#type().cotangent()?;
                context.reference_accumulators.insert(
                    ReferenceRoot::Allocation {
                        instruction: InstructionId::new(self.id(), instruction_index),
                        output_index,
                    },
                    CotangentReferenceAccumulator::Unallocated { cotangent_type },
                );
            }
        }

        // Constants and pure known intermediates are materialized lazily below, only when a live transpose rule
        // needs them. This avoids copying dead constants and replaying dead known-side work into the pullback.
        let instruction_by_output = self.region().instruction_by_output();
        let region_mappings = RegionReplayMappings::new();
        let mut materialization_state = vec![MaterializationState::Unseen; self.instructions().len()];

        // Walk the primal program backward. Value output cotangents and live reference-state accumulators
        // determine which rules must run; known factors are materialized only when a rule needs them.
        for (instruction_index, instruction) in self.instructions().iter().enumerate().rev() {
            // Validate effects before omitting dead reverse work. Instructions with linear outputs, and sinks that
            // access reference state, may carry only state effects or primal assertions. Reference rules transpose
            // state through its accumulators; assertions remain preconditions executed by the forward program.
            // Other sinks have no cotangent result and are omitted. Known-only producers are checked separately
            // when their values are materialized for a transpose rule.
            let has_linear_output = instruction.outputs().iter().copied().try_fold(false, |any, output| {
                let is_linear = *linear.get(output.index()).ok_or(ProgramError::UnboundAtomId { id: output })?;
                Ok::<_, ProgramError>(any || is_linear)
            })?;
            let has_linear_input = instruction.inputs().iter().copied().try_fold(false, |any, input| {
                let is_linear = *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
                Ok::<_, ProgramError>(any || is_linear)
            })?;
            if has_linear_output || has_linear_input {
                // Ordered assertions are primal preconditions. The primal/forward program executes them before its
                // pullback is applied, so transposition must not replay them in reverse order. Other observable effects
                // cannot be omitted from or replayed by the transposed linear program.
                let effects = self.instruction_effects(instruction_index)?.classes();
                let takes_reference_path = has_linear_output || effects.contains(EffectClass::OrderedState);
                if takes_reference_path
                    && effects
                        .into_iter()
                        .any(|effect| !matches!(effect, EffectClass::OrderedAssertion | EffectClass::OrderedState))
                {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!(
                            "partition-aware transpose cannot transpose effectful linear instruction `{}`; \
                             transposition cannot replay observable effects",
                            instruction.operation().name(),
                        ),
                    }
                    .into());
                }
            }

            // Skip dead reverse edges early. If none of an instruction's outputs carries an adjoint and none of the
            // reference roots it mutates (directly or inside its regions) or allocates has an allocated accumulator,
            // the instruction cannot contribute to any input cotangent. Reads and consumes are driven by their output
            // adjoints alone, while stores, swaps, accumulations, and allocations are driven by the state of their
            // root's accumulator, because their transposes produce value cotangents from the accumulator even though
            // they have no Single Static Assignment (SSA) outputs. This is the only operand-side guard. A live
            // transpose rule may read non-linear operands; pure known producer subgraphs are materialized lazily below,
            // while effectful known producers are rejected rather than duplicated or reordered in the pullback.
            let mut has_output_adjoint = false;
            for output in instruction.outputs().iter().copied() {
                if matches!(
                    context.cotangent_storage(&atom_accumulators[output.index()])?,
                    CotangentStorage::Value { value: Some(_), .. },
                ) {
                    has_output_adjoint = true;
                    break;
                }
            }
            let has_live_state = match &analysis {
                None => {
                    // A region without references has no state to drive the walk.
                    false
                }
                Some(analysis) => {
                    let id = InstructionId::new(self.id(), instruction_index);

                    // Stores, swaps, and accumulations run when the accumulator of a root they mutate is allocated.
                    let mutates_live_root = analysis.transitive_access(id).is_some_and(|access| {
                        access.roots().any(|root| {
                            access.is_mutated(root) && context.cotangent_accumulator_reference(root).is_some()
                        })
                    });

                    // Allocations run when the accumulator of the allocated root became allocated.
                    let allocates_live_root =
                        instruction.operation().effects().allocation_output_indices().any(|output_index| {
                            context
                                .cotangent_accumulator_reference(ReferenceRoot::Allocation {
                                    instruction: id,
                                    output_index,
                                })
                                .is_some()
                        });
                    mutates_live_root || allocates_live_root
                }
            };

            if !has_output_adjoint && !has_live_state {
                // Backward rule effects are absent from execution effect summaries. Keeping the rule live here
                // lets it preserve those effects without treating a dormant region as an executed forward region.
                let has_rule_effects = instruction
                    .regions()
                    .iter()
                    .any(|id| self.with_id(*id).unwrap().has_observable_effects_in_closure());
                if !has_rule_effects {
                    continue;
                }
            }

            // Materialize the instruction's output cotangents in operation-result order. Missing adjoint slots become
            // structural zeros so transpose rules can distinguish unused outputs without staging zero operations.
            // Structural zeros carry the output's cotangent type: a differentiable output's cotangent dual or the
            // first-class zero-space type for a non-differentiable output. Accumulated adjoints are always live: rules
            // communicate zero-ness symbolically through `MaybeZero` (opaque program splices such as the custom-VJP
            // backward replay recover it at their own boundary), and so no staged canonical zero ever needs to be
            // recognized here.
            let output_accumulators = instruction
                .outputs()
                .iter()
                .map(|output| atom_accumulators[output.index()].clone())
                .collect::<Vec<_>>();
            let instruction_output_cotangents = context.take_cotangents(&output_accumulators)?;

            // Prepare the primitive rule's primal knowledge separately from its cotangent mask. Each input/operand
            // becomes a self-describing `PartialValue`: a linear operand is `Unknown` of its type (the rule produces
            // a cotangent of that type), and a known operand is `Known` of the tracer reading its pullback value atom
            // from `known_map`. Known inputs are seeded above, constants are copied lazily, and pure known
            // intermediates iteratively replay their producer subgraphs exactly once before the rule runs.
            let inputs = instruction
                .inputs()
                .iter()
                .copied()
                .map(|input| {
                    let r#type = self
                        .atoms()
                        .get(input.index())
                        .ok_or(ProgramError::UnboundAtomId { id: input })?
                        .r#type()
                        .into_owned();
                    if *linear.get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })? {
                        Ok(PartialValue::Unknown(r#type))
                    } else {
                        let atom = materialize_known(
                            *self,
                            instruction_by_output.as_slice(),
                            linear.as_slice(),
                            &region_mappings,
                            &builder,
                            known_map.as_mut_slice(),
                            materialization_state.as_mut_slice(),
                            input,
                        )?;
                        Ok(PartialValue::Known(context.tracer(atom, Some(r#type))))
                    }
                })
                .collect::<Result<Vec<_>, ProgramError>>()?;
            let transposition_driver = RecursiveTranspositionDriver::new(*self, instruction_index)?;

            // Retain concrete cotangents with runtime dimensions before the rule consumes them. Later reference rules
            // may need these dimensions to construct zeros after value cotangent storage has been drained.
            if context.reference_analysis.is_some() {
                context.dimension_sources.extend(
                    instruction_output_cotangents
                        .iter()
                        .filter_map(MaybeZero::as_value)
                        .filter(|source| {
                            !source.r#type().is_reference() && source.r#type().identities().next().is_some()
                        })
                        .cloned(),
                );
            }

            // Walk each operand's alias chain to find the coordinates needed to reconstruct its cotangent view.
            // The view descriptions bind coordinates to non-reference inputs of the view-creating instructions.
            if let Some(analysis) = &analysis {
                for operand in instruction.inputs() {
                    let mut value = ValueId::new(self.id(), *operand);
                    while let Some(edge) = analysis.alias(value) {
                        // Views created outside this region have no local coordinate operands. The enclosing
                        // operation's transpose rule must handle those boundary views, including iteration-bound ones.
                        if edge.kind() == ReferenceAliasKind::View && edge.instruction().region() == self.id() {
                            for coordinate in self.instructions()[edge.instruction().index()].inputs() {
                                let id = ValueId::new(self.id(), *coordinate);
                                let r#type = self.atoms()[coordinate.index()].r#type();

                                // Several views can share a coordinate. Reuse its staged value, and exclude the
                                // source reference itself from the candidate coordinates.
                                if r#type.is_reference() || context.cotangent_view_coordinates.contains_key(&id) {
                                    continue;
                                }

                                // Only known primal values can supply coordinates. Skip linear values here;
                                // view reconstruction rejects them if a view actually requires them.
                                if *linear
                                    .get(coordinate.index())
                                    .ok_or(ProgramError::UnboundAtomId { id: *coordinate })?
                                {
                                    continue;
                                }

                                let atom = materialize_known(
                                    *self,
                                    instruction_by_output.as_slice(),
                                    linear.as_slice(),
                                    &region_mappings,
                                    &builder,
                                    known_map.as_mut_slice(),
                                    materialization_state.as_mut_slice(),
                                    *coordinate,
                                )?;

                                context
                                    .cotangent_view_coordinates
                                    .insert(id, context.tracer(atom, Some(r#type.into_owned())));
                            }
                        }
                        value = edge.source();
                    }
                }
            }

            let operand_accumulators = instruction
                .inputs()
                .iter()
                .map(|input| atom_accumulators[input.index()].clone())
                .collect::<Vec<_>>();

            // Run the rule under the source instruction's recorded origin. Enter through a cloned tracing handle
            // because the rule borrows the context mutably; both handles share the provenance state.
            (*context).clone().invoke_with_provenance_origin(instruction.provenance().clone(), || {
                instruction.operation().transpose(
                    &mut context,
                    &transposition_driver,
                    inputs.as_slice(),
                    instruction_output_cotangents.as_slice(),
                    &operand_accumulators,
                )
            })?;
        }

        // The pullback outputs are the accumulated cotangents of the selected inputs, emitted directly in
        // `input_indices` order. Known inputs receive no cotangent output. A reference input transposed with the
        // `Reference` kind returns its caller-owned cotangent reference by identity (so that structured rules can
        // thread it positionally as a carry), one transposed with the `Ignore` kind returns nothing, and a disconnected
        // non-reference selected input is materialized through the operation family's canonical zero representation,
        // using any mapped residual inputs needed to provide runtime dimensions that its cotangent type does not
        // contain.
        let mut outputs = Vec::with_capacity(input_indices.len());
        for (output_index, &index) in input_indices.iter().enumerate() {
            let input = self.input_ids()[index];
            let input_atom = self.atoms().get(input.index()).ok_or(ProgramError::UnboundAtomId { id: input })?;
            let input_type = input_atom.r#type();
            if input_type.is_reference() {
                match destination_kinds[output_index] {
                    CotangentDestinationKind::Reference => {
                        let root = ReferenceRoot::RegionInput { region: self.id(), input_index: index };
                        let reference = context.cotangent_accumulator_reference(root).ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "linear reference input {index} lost its cotangent destination accumulator",
                            ))
                        })?;
                        outputs.push(reference.atom_id()?);
                    }
                    CotangentDestinationKind::Ignore | CotangentDestinationKind::Return => {}
                }
                continue;
            }

            if destination_kinds[output_index] != CotangentDestinationKind::Return {
                continue;
            }

            match context.take_cotangents(std::slice::from_ref(&atom_accumulators[input.index()]))?.remove(0) {
                MaybeZero::Value(adjoint) => outputs.push(adjoint.atom_id()?),
                MaybeZero::Zero(_) => {
                    let cotangent_type = input_type.cotangent()?;
                    let mut builder_borrow = builder.borrow_mut();

                    // These indices name source-program inputs, but zero construction occurs in the pullback
                    // builder. Resolve them through `known_map`: dimension residuals are known primal quantities,
                    // never linear inputs whose cotangents the pullback should compute.
                    let residuals = zero_residual_input_indices
                        .get(output_index)
                        .into_iter()
                        .flatten()
                        .map(|&residual_index| {
                            let residual =
                                *self.input_ids().get(residual_index).ok_or_else(|| ProgramError::InvalidArgument {
                                    message: format!(
                                        "transposition zero-residual input index {residual_index} is out of range \
                                         for a program with {input_count} input(s)",
                                    ),
                                })?;
                            known_map.get(residual.index()).copied().flatten().ok_or_else(|| {
                                ProgramError::MalformedProgram(
                                    "transposition zero residual is not a known pullback input".to_string(),
                                )
                            })
                        })
                        .collect::<Result<Vec<_>, ProgramError>>()?;
                    let (operation, operands) = O::zero_operation_with_residuals(cotangent_type, residuals.as_slice())?;
                    outputs.push(builder_borrow.add_instruction(operation, Vec::new(), operands, None)?[0]);
                }
            }
        }

        // Release the context and the staged tracers it retains so the cloned `builder` handle can be unwrapped.
        // Accumulator handles retain only the separate context identity token, so they do not keep the builder alive.
        drop(context);

        // Build a flat pullback boundary: output cotangents, supplied destination buffers, then known input values.
        // Results follow the selected input order and destination kinds. Fully linear callers recover their structured
        // boundary by reattaching the source program's input and output structures.
        let pullback_input_count = builder.borrow().input_ids().len();
        let pullback_output_count = outputs.len();
        let builder = match Rc::try_unwrap(builder) {
            Ok(builder) => builder.into_inner(),
            Err(_) => return Err(ProgramError::EscapedProgramBuilder.into()),
        };
        builder
            .build(outputs, vec![Placeholder; pullback_input_count], vec![Placeholder; pullback_output_count])
            .map_err(DifferentiationError::from)
    }
}

impl<T, V, O, Input, Output> Program<V, O, Input, Output>
where
    T: DifferentiableType,
    V: Value<Type = T>,
    O: TransposableOperation<V, O>
        + ResidualZeroProvider<T>
        + OperationProvider<T, ReferenceNewOperation<T, T>, Operation = O>
        + OperationProvider<T, ReferenceAddUpdateOperation<T, T>, Operation = O>
        + From<AddOperation<T>>,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Transposes this linear [`Program`] with respect to every input while preserving its structured boundary.
    /// Refer to [`Program::transpose_with_respect_to`] for more information.
    #[inline]
    pub fn transpose(&self) -> Result<Program<V, O, Output, Input>, DifferentiationError> {
        Ok(self
            .entry_region_ref()
            .transpose(&(0..self.input_ids().len()).collect::<Vec<_>>(), &[], &[])?
            .restructured(self.output_structure().clone(), self.input_structure().clone())?)
    }

    /// Transposes this linear _pushforward_ [`Program`] into its reverse-mode _pullback_ with respect to selected
    /// inputs. In the algebraic sense, _transposing_ a linear map `L: X -> Y` gives a map on _dual_ spaces
    /// `L^T: Y* -> X*`. In finite dimensions this is the same operation represented by a matrix transpose. Here the
    /// linear map is not stored as a matrix. It is a staged [`Program`] that maps input tangents to output tangents,
    /// and transposition builds the dual program that maps output cotangents back to input cotangents. Operationally,
    /// transposition creates cotangent inputs for this program's outputs, walks the instructions in reverse order,
    /// and applies each primitive operation's [`TransposableOperation::transpose`] rule to accumulate cotangent
    /// contributions for the original inputs. This is the same decomposition of reverse-mode automatic differentiation
    /// as in [this paper](https://arxiv.org/abs/2204.10923).
    ///
    /// Over complex types, transposition is defined with respect to the **bilinear** (i.e., conjugation-free) pairing
    /// `⟨a, b⟩ = Real(a · b)`: the transpose of multiplying by a known complex factor multiplies by that same factor
    /// (never its conjugate), which keeps transposition an involution and keeps every bilinear transpose rule identical
    /// across real and complex types. Conjugation enters only through the transpose rules of the
    /// ℝ-linear-but-not-ℂ-linear primitives (i.e., `conjugate`, `real`, `imaginary`, and `complex`), whose adjoints
    /// under this pairing carry the conjugations and negations explicitly. The user-facing consequence is documented on
    /// the gradient entry points: the holomorphic ones return the complex derivative `∂f/∂z`, and the plain ones return
    /// `2 · ∂f/∂z̄` for ℂ → ℝ functions.
    ///
    /// Disconnected primal inputs are emitted as [`ZeroOperation`](crate::ZeroOperation)s, which the value type's
    /// [`Zero`] implementation evaluates at interpretation time. This applies uniformly to linear programs whose values
    /// are [`Tracer`]s from an outer trace. Interpreting such a pullback [`ZeroOperation`](crate::ZeroOperation) over
    /// outer-trace [`Tracer`]s stages a typed zero into the surrounding tracing context, so that backends whose traced
    /// constants are abstract metadata do not need to materialize a runtime value just to transpose an enclosing traced
    /// program.
    ///
    /// `input_indices` selects the inputs to transpose with respect to, while the remaining inputs are held as constant
    /// parameters of the linear map. The program must be linear in the selected inputs, but it can depend arbitrarily
    /// on the known ones. This is the partial entry point behind the fully linear [`Program::transpose`].
    ///
    /// Linearity is propagated forward from the program inputs: a program-input [`Atom`] is linear exactly when its
    /// index appears in `input_indices`, constant atoms are always known, and an operation result is linear when any
    /// of its operands is linear. Local reference allocations are always treated as linear state, including allocations
    /// initialized from known zeros; their stores and reads participate in the reverse sweep through
    /// [`CotangentAccumulator`]s. Each operation's [`transpose`](TransposableOperation::transpose) rule
    /// receives the per-operand linearity knowledge derived from this propagation.
    ///
    /// For programs with no reference inputs or outputs, the pullback's inputs are the cotangents of this program's
    /// outputs followed by the runtime values of the known inputs (in program-input order). Its outputs are the
    /// accumulated cotangents of the selected inputs, **in `input_indices` order**. Known inputs receive no cotangent
    /// output. Because this layout depends on `input_indices`, the pullback's inputs and outputs are returned as flat
    /// [`Vec`]s rather than reusing this program's structured input and output types. The fully linear
    /// [`Program::transpose`] recovers the structured form. Reference inputs and outputs follow the
    /// cotangent destination rules below.
    ///
    /// # Known Intermediates and Rematerialization
    ///
    /// The normal differentiation path linearizes and partially evaluates before transposition. Values computed only
    /// from primals then cross the linear boundary as residual inputs, and so transposing such a normalized pushforward
    /// does **not** rebuild their producer instructions in the pullback. This function nevertheless accepts a
    /// hand-built or otherwise unnormalized linear program whose live transpose rules read internal known values. For
    /// such a value, transposition lazily copies the demanded, pure known-producer ancestor subgraph into the generated
    /// pullback. This is _rematerialization_: the copied instructions execute every time the pullback is interpreted,
    /// trading saved residuals for recomputation. Only ancestors of a known value actually demanded by a live transpose
    /// rule are copied; dead known instructions and dead constants remain absent. Each source producer is copied at
    /// most once, all of its output atoms are memoized together, and every later consumer reuses those mapped outputs.
    /// The producer walk is iterative, so its call-stack usage does not grow with producer-chain depth. This behavior
    /// is a correctness fallback, and not an implicit recommendation to rematerialize hot or expensive primal work.
    /// Callers that want predictable pullback cost should partially evaluate and carry such values as residual inputs.
    /// Known producers with observable effects are rejected with [`ProgramError::UnsupportedOperation`] (callers must
    /// partially evaluate their values into residual inputs to avoid duplicating or reordering primal effects). Ordered
    /// assertions are the exception because replay validates the same saved primal values. On the linear side,
    /// reference-state effects transpose through cotangent accumulators, and ordered assertions remain primal
    /// preconditions rather than running in reverse order. Other effects on instructions with linear outputs are
    /// rejected before dead edge elimination. Known program inputs are always exposed as pullback inputs, while
    /// literal constants are copied lazily under the same demand-driven policy.
    ///
    /// The pullback is staged into a fresh internal [`TracingContext`]: transposition records one cotangent input per
    /// non-reference program output, walks this program in reverse instruction order applying each [`Operation`]'s
    /// [`transpose`](TransposableOperation::transpose) rule, and accumulates the per-input cotangent contributions
    /// (summing repeated contributions with staged adds). Nested subprograms are transposed directly through their
    /// borrowed [`RegionRef`]s in their own fresh contexts.
    ///
    /// # Cotangent Destinations
    ///
    /// `destination_kinds` controls how the cotangents of the selected inputs cross the pullback boundary. An empty
    /// slice selects each input's default kind. Otherwise, provide one kind per input in `input_indices` order.
    ///
    ///   - A non-reference selected input defaults to [`Return`](CotangentDestinationKind::Return), exposing its
    ///     cotangent as a pullback output. [`Reference`](CotangentDestinationKind::Reference) instead accepts a
    ///     caller-owned `ref<cotangent(T)>` buffer and adds contributions into its existing contents during replay,
    ///     with no output for that input. [`Ignore`](CotangentDestinationKind::Ignore) omits its cotangent entirely.
    ///   - A reference-typed selected input of type `ref<T>` with kind
    ///     [`Reference`](CotangentDestinationKind::Reference) (i.e., its default) exposes one caller-owned cotangent
    ///     reference input of type `ref<cotangent(T)>` in the pullback and returns that same reference, by identity,
    ///     as the input's cotangent output. On entry the reference holds the cotangent of the reference's
    ///     post-execution state, and on return it holds the cotangent of its pre-execution state, because the reference
    ///     rules accumulate the cotangents of values read from the state and take the cotangents of stored values out
    ///     of it.
    ///   - A reference-typed selected input with kind [`Ignore`](CotangentDestinationKind::Ignore) accumulates through
    ///     an internal cotangent reference that is allocated lazily on first use and discarded, exposing neither an
    ///     input nor an output for it.
    ///
    /// The pullback's inputs are the cotangents of non-reference outputs, followed by the destinations for selected
    /// [`Reference`](CotangentDestinationKind::Reference) inputs in program-input order, followed by the known inputs.
    /// Its outputs follow `input_indices` order, including non-reference [`Return`](CotangentDestinationKind::Return)
    /// values and reference-primal [`Reference`](CotangentDestinationKind::Reference) identities, and omitting
    /// non-reference [`Reference`](CotangentDestinationKind::Reference) inputs and every
    /// [`Ignore`](CotangentDestinationKind::Ignore) input. A reference-typed output that forwards an input root shares
    /// that input's accumulator and has no cotangent input, while a reference-typed output that escapes a local
    /// allocation is rejected because its later uses are unknown to the program. Reference operations take an explicit
    /// reference path through transposition. Other effects follow the restrictions described above. Nested regions are
    /// transposed with [`Reference`](CotangentDestinationKind::Reference) kinds and their enclosing rules thread the
    /// accumulators positionally, so a staged unbounded `while` operation remains rejected while the `scan`, bounded
    /// `while`, and `condition` operation transpose through their reference carries.
    ///
    /// The source may consume a selected reference input directly in its entry region. Transposing that consumption
    /// transfers the returned value's cotangent into the input accumulator and leaves the cotangent destination live.
    /// Known references remain borrowed, and attached regions cannot consume references owned by their callers.
    ///
    /// This raw program transformation sees types rather than runtime reference identities. When executing its result,
    /// callers must supply mutually independent cotangent references that do not alias primal references or captured
    /// references. Distinct reference inputs of the source program must also denote independent roots. Aliases created
    /// inside the program are tracked by reference analysis. [`Pullback::apply_with_destinations`] validates these
    /// runtime boundary requirements for callable pullbacks.
    ///
    /// # Parameters
    ///
    ///   - `input_indices`: Indices of the program inputs the program is transposed with respect to. Each index
    ///     must be in range and appear at most once, otherwise this returns [`ProgramError::InvalidArgument`].
    ///     The order of the indices defines the order of the pullback's cotangent outputs. Only non-reference
    ///     [`Return`](CotangentDestinationKind::Return) inputs and reference-primal
    ///     [`Reference`](CotangentDestinationKind::Reference) inputs produce outputs.
    ///   - `destination_kinds`: One kind per selected input, aligned with `input_indices`, or empty to select
    ///     every input's default kind ([`Return`](CotangentDestinationKind::Return) for non-reference inputs and
    ///     [`Reference`](CotangentDestinationKind::Reference) for reference inputs). A nonempty slice of the wrong
    ///     length or a kind that is inadmissible for its input's type returns [`ProgramError::InvalidArgument`].
    #[inline]
    pub fn transpose_with_respect_to(
        &self,
        input_indices: &[usize],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Program<V, O, Vec<V>, Vec<V>>, DifferentiationError> {
        self.entry_region_ref().transpose(input_indices, &[], destination_kinds)
    }

    /// Transposes this compact linearization [`Program`] using the entry region's retained transform cache. Its
    /// trailing `residual_count` inputs are known primal residuals; its leading inputs are tangents whose cotangent
    /// destinations are selected by `destination_kinds` as in [`Self::transpose_with_respect_to`]. Repeated calls
    /// with the same arguments share the resulting program.
    ///
    /// Linearization reserves the final residual positions for dimensions needed to construct disconnected cotangent
    /// zeros, in tangent-input order. This function recovers those mappings before transposition as selecting only the
    /// leading tangent inputs would not supply enough information to construct dynamically shaped zeros.
    pub(crate) fn transpose_with_trailing_residuals_shared(
        &self,
        residual_count: usize,
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Arc<Program<V, O, Vec<V>, Vec<V>>>, DifferentiationError> {
        let tangent_input_count = self.input_ids().len().checked_sub(residual_count).ok_or_else(|| {
            ProgramError::MalformedProgram(format!(
                "linearization program consumes {} inputs which is fewer than its {} residuals",
                self.input_ids().len(),
                residual_count,
            ))
        })?;

        // Linearization appends the dimensions needed for disconnected cotangent zeros after its other residuals.
        // Recover each tangent input's share of that suffix from liveness and the zero provider's declared residuals,
        // preserving their input order without storing parallel metadata.
        let live_sets = self.live_sets();
        let zero_residual_counts = self
            .input_ids()
            .iter()
            .copied()
            .take(tangent_input_count)
            .map(|input| {
                if live_sets.atoms()[input.index()] {
                    0
                } else {
                    O::zero_residual_types(self.atoms()[input.index()].r#type().as_ref()).len()
                }
            })
            .collect::<Vec<_>>();
        let zero_residual_count = zero_residual_counts.iter().sum::<usize>();
        if zero_residual_count > residual_count {
            return Err(ProgramError::MalformedProgram(format!(
                "linearization program declares {residual_count} residuals but disconnected cotangent zeros require \
                 {zero_residual_count}",
            ))
            .into());
        }

        // Partial-evaluation residuals precede the dimensions reserved for zeros. Partition that final suffix
        // into contiguous per-input ranges in the same order used by linearization.
        let mut next_zero_residual = self.input_ids().len() - zero_residual_count;
        let zero_residual_input_indices = zero_residual_counts
            .into_iter()
            .map(|count| {
                let indices = (next_zero_residual..next_zero_residual + count).collect::<Vec<_>>();
                next_zero_residual += count;
                indices
            })
            .collect::<Vec<_>>();
        self.entry_region_ref().transpose_shared(
            &(0..tangent_input_count).collect::<Vec<_>>(),
            &zero_residual_input_indices,
            destination_kinds,
        )
    }
}

/// Extension trait carrying the primitive value-level reverse-mode transform on every [`Context`]. Reverse mode
/// is implemented as forward linearization followed by transposition. This trait is blanket-implemented for every
/// [`ForwardModeDifferentiate`] context whose operation family supports the required partial-evaluation and
/// transposition machinery. Value cotangents flow through the same context as the primals while reference-state
/// cotangents are carried by cotangent references.
///
/// User-facing scalar gradients, auxiliary outputs, and holomorphic validation are composed through
/// [`DifferentiationBuilder`](crate::DifferentiationBuilder). Keeping those orthogonal choices in builder type state
/// leaves this trait with the reusable reverse-mode engine [`vjp`](Self::vjp) and the low-level scalar cotangent-seed
/// primitive [`gradient_seed`](Self::gradient_seed), instead of a method for every option combination.
pub trait ReverseModeDifferentiate:
    ForwardModeDifferentiate
    + Context<
        Operation: PartiallyEvaluatableOperation<Self>
                       + PartiallyEvaluatableOperation<TracingContext<Self::Constant, Self::Operation>>
                       + DifferentiableOperation<PartialEvaluationContext<Self>>
                       + TransposableOperation<Self::Constant, Self::Operation>
                       + ResidualZeroProvider<Self::Type>
                       + OperationProvider<
            Self::Type,
            ReferenceNewOperation<Self::Type, Self::Type>,
            Operation = Self::Operation,
        > + OperationProvider<
            Self::Type,
            ReferenceAddUpdateOperation<Self::Type, Self::Type>,
            Operation = Self::Operation,
        > + From<AddOperation<Self::Type>>,
    >
{
    /// Reverse-mode-differentiates `function` at `primals`, returning the primal output and a reusable [`Pullback`],
    /// with this [`Context`] executing (or staging) the primal-side operations. Refer to the documentation of
    /// [`DifferentiationBuilder::vjp`](crate::DifferentiationBuilder::vjp) for the reverse-mode transform.
    /// The returned [`Pullback`] retains the linear program and transposes it on its first application under the
    /// [`CotangentDestination`]s chosen then (refer to the documentation of [`Pullback`]), so transposition errors
    /// surface on that first application rather than here.
    #[allow(clippy::type_complexity)]
    fn vjp<
        F: FnOnce(
            Input::To<LinearizationTracer<Self>>,
            Capture::To<LinearizationTracer<Self>>,
        ) -> Result<Output, ProgramError>,
        Input: Parameterized<Self::Value, To<Self::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Capture: Parameterized<Self::Value, To<Self::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<Self>>>,
        Output: Parameterized<LinearizationTracer<Self>, Family: ParameterizedFamily<Self::Value>>,
    >(
        &self,
        function: F,
        primal: Input,
        capture: Capture,
    ) -> Result<(Output::To<Self::Value>, Pullback<Self, Input, Output::To<Self::Value>>), DifferentiationError> {
        let input_structure = primal.parameter_structure();
        let primal_input_values = primal.parameters().cloned().collect::<Vec<_>>();
        let (output, pushforward) = self.linearize(function, primal, capture)?;
        let pullback = Pullback::from_pushforward(pushforward, &primal_input_values, input_structure)?;
        Ok((output, pullback))
    }

    /// Validates the scalar `output` of a gradient entry point and constructs its cotangent seed. The output must be a
    /// single rank-0 scalar with a cotangent space, and complex outputs additionally require `holomorphic`. A single
    /// reverse-mode seed recovers the derivative of a complex-output function only when the function is holomorphic,
    /// so without that promise a complex output is rejected with an error instead of silently computing a value that
    /// is not a derivative (i.e., `holomorphic` changes nothing for real outputs). The seed is the multiplicative
    /// identity typed with the output's cotangent type (e.g., swapping unreduced and reduced sharding axes for arrays)
    /// and bound through this [`Context`], so an eager context constructs a concrete value while a staging context
    /// stages into its enclosing trace.
    ///
    /// Gradient entry points share this seeding step. Custom gradient entry points built on top of
    /// [`vjp`](crate::DifferentiationBuilder::vjp) can reuse the same validation and seeding contract.
    ///
    /// # Parameters
    ///
    ///   - `output`: Scalar primal output whose cotangent space determines the seed type.
    ///   - `holomorphic`: Whether a complex output is accepted under the caller's holomorphy promise.
    fn gradient_seed(&self, output: &Self::Value, holomorphic: bool) -> Result<Self::Value, DifferentiationError>
    where
        Self::Operation: OperationProvider<Self::Type, OneOperation<Self::Type>, Operation = Self::Operation>,
    {
        // Reverse mode only defines a gradient for scalar-output functions.
        let output_type = output.r#type();
        if !output_type.is_scalar() {
            return Err(DifferentiationError::NonScalarGradientOutput { output_type: output_type.to_string() });
        }

        if !holomorphic && output_type.is_complex() {
            return Err(DifferentiationError::ComplexGradientOutput { output_type: output_type.to_string() });
        }

        // A non-differentiable scalar output carries no cotangent space and thus no "one" to seed, so reverse mode
        // is degenerate and is rejected up front.
        let output_cotangent_type = output_type.cotangent()?;
        if output_cotangent_type.is_zero_space() {
            return Err(DifferentiationError::NonDifferentiableGradientOutput { output_type: output_type.to_string() });
        }

        let one_operation = Self::Operation::provide(OneOperation::new(output_cotangent_type), &[])?;
        let mut seeds = self.bind(one_operation, Vec::new(), &[])?;
        check_count!("output", seeds, 1, ProgramError);
        Ok(seeds.pop().unwrap())
    }
}

impl<C: ForwardModeDifferentiate + Context> ReverseModeDifferentiate for C where
    C::Operation: PartiallyEvaluatableOperation<C>
        + PartiallyEvaluatableOperation<TracingContext<C::Constant, C::Operation>>
        + DifferentiableOperation<PartialEvaluationContext<C>>
        + TransposableOperation<C::Constant, C::Operation>
        + ResidualZeroProvider<C::Type>
        + OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
        + OperationProvider<C::Type, ReferenceAddUpdateOperation<C::Type, C::Type>, Operation = C::Operation>
        + From<AddOperation<C::Type>>
{
}

/// Computes a value and its gradient in `context` capturing `capture` as a non-differentiated `function` input
/// and selecting real or holomorphic complex output validation.
pub(crate) fn value_and_gradient_in_context<
    C: ReverseModeDifferentiate + Zero<C::Value>,
    F: FnOnce(Input::To<LinearizationTracer<C>>, Capture::To<LinearizationTracer<C>>) -> Output,
    Input: Parameterized<C::Value, To<C::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    Capture: Parameterized<C::Value, To<C::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    Output: MaybeFallible<LinearizationTracer<C>, ProgramError>,
>(
    context: &C,
    function: F,
    primals: Input,
    capture: Capture,
    holomorphic: bool,
) -> Result<(C::Value, Input::To<C::Value>), DifferentiationError>
where
    C::Operation: OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
        + OperationProvider<C::Type, ReferenceFreezeOperation<C::Type, C::Type>, Operation = C::Operation>
        + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
{
    let destinations = gradient_destinations(context, &primals)?;
    let (output, pullback) = context.vjp(|input, capture| function(input, capture).into_result(), primals, capture)?;
    let seed = context.gradient_seed(&output, holomorphic)?;
    let gradient = pullback.apply_gradient(vec![CotangentSeed::Value(seed)], destinations)?;
    Ok((output, gradient))
}

/// Computes a value, auxiliary outputs, and a gradient in `context` capturing `capture` as a non-differentiated
/// `function` input and selecting real or holomorphic complex output validation. Only the scalar value is
/// differentiated. Auxiliary leaves receive zero cotangent seeds.
pub(crate) fn value_and_gradient_auxiliary_in_context<
    C: ReverseModeDifferentiate + Zero<C::Value>,
    F: FnOnce(Input::To<LinearizationTracer<C>>, Capture::To<LinearizationTracer<C>>) -> Output,
    Input: Parameterized<C::Value, To<C::Value> = Input, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    Capture: Parameterized<C::Value, To<C::Value> = Capture, Family: ParameterizedFamily<LinearizationTracer<C>>>,
    Output: MaybeFallible<(LinearizationTracer<C>, AuxiliaryOutput::To<LinearizationTracer<C>>), ProgramError>,
    AuxiliaryOutput: Parameterized<
            C::Value,
            To<C::Value> = AuxiliaryOutput,
            Family: ParameterizedFamily<LinearizationTracer<C>, To = AuxiliaryOutput::To<LinearizationTracer<C>>>,
        >,
>(
    context: &C,
    function: F,
    primals: Input,
    capture: Capture,
    holomorphic: bool,
) -> Result<((C::Value, AuxiliaryOutput), Input::To<C::Value>), DifferentiationError>
where
    C::Operation: OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
        + OperationProvider<C::Type, ReferenceFreezeOperation<C::Type, C::Type>, Operation = C::Operation>
        + OperationProvider<C::Type, OneOperation<C::Type>, Operation = C::Operation>,
    (LinearizationTracer<C>, AuxiliaryOutput::To<LinearizationTracer<C>>): Parameterized<
            LinearizationTracer<C>,
            To<C::Value> = (C::Value, AuxiliaryOutput),
            Family: ParameterizedFamily<C::Value>,
        >,
{
    let destinations = gradient_destinations(context, &primals)?;
    let ((output, auxiliary), pullback): ((C::Value, AuxiliaryOutput), _) =
        context.vjp(|input, capture| function(input, capture).into_result(), primals, capture)?;

    // Each non-reference auxiliary leaf supplies the runtime shape of its zero cotangent. Cotangent types preserve
    // shapes even when their element representation differs, so the leaf supplies every required dynamic extent.
    // Reference auxiliary outputs remain outside the scalar gradient convenience contract.
    let auxiliary_cotangents = auxiliary
        .parameters()
        .map(|value| {
            if value.r#type().is_reference() {
                return Err(ProgramError::InvalidArgument {
                    message: "gradient auxiliary outputs must be non-reference values; \
                              use `Pullback::apply_with_destinations` for reference outputs"
                        .to_string(),
                });
            }
            C::Operation::materialize_zero_from_residual_sources(
                context,
                MaybeZero::Zero(value.r#type().cotangent()?),
                std::iter::once(value),
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let seeds = std::iter::once(context.gradient_seed(&output, holomorphic)?)
        .chain(auxiliary_cotangents)
        .map(CotangentSeed::Value)
        .collect();
    let gradient = pullback.apply_gradient(seeds, destinations)?;
    Ok(((output, auxiliary), gradient))
}

/// Applies a member operation's transpose rule through a projected view of a composite [`TracingContext`]. Use this
/// function from a composite operation dispatcher when the linear operation is [`Region`]-free and every operand and
/// result belongs to the same projectable member type `T`. Because [`TransposableOperation`] rules stage through a
/// member-typed [`TracingContext`], this function records the rule in a short-lived member program, converts that
/// program to the composite type, and splices it into the active trace. Known primal inputs and live output cotangents
/// become splice inputs in encounter order; structural zeros remain types and do not materialize values. The member
/// rule receives the enclosing handles' cotangent mask but accumulates values locally, because its type family may not
/// represent reference buffers. Each resulting contribution is lifted and submitted to the corresponding enclosing
/// handle. Composite rules that can update reference views directly should bypass this dense fallback.
///
/// Operations whose transpose crosses member types or whose rule needs attached regions require an explicit composite
/// transpose rule instead. A member operation that declares [`RegionSlot`](crate::RegionSlot)s is rejected with an
/// exact diagnostic naming it, because projection reaches the member rule with no region access: the attached regions
/// are programs in the _composite_ universe, and no projected driver can present them in the member universe.
///
/// # Parameters
///
///   - `context`: Active composite [`TranspositionContext`] into which the projected transpose program is spliced.
///   - `operation`: Region-free linear operation expressed in the projected member operation family.
///   - `inputs`: Per-operand primal knowledge, preserving whether each primal is known or is a linear unknown.
///   - `outputs`: Composite output cotangents, represented as live traced values or structural zeros.
///   - `accumulators`: Enclosing value cotangent handles, aligned with `inputs` and owned by `context`.
pub fn transpose_projected_operation<
    T: DifferentiableType,
    P: Operation<Type = T> + TransposableOperation<<V as ValueProjection<T>>::Projected, P>,
    V: Value<Type: DifferentiableType + From<T>> + ValueProjection<T, Projected: Value<Type = T>>,
    O: Operation<Type = V::Type> + OperationProjection<T, Projected = P> + From<AddOperation<V::Type>>,
>(
    context: &mut TranspositionContext<V, O>,
    operation: &P,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    accumulators: &[CotangentAccumulator],
) -> Result<(), DifferentiationError>
where
    for<'t> &'t T: TryFrom<&'t V::Type, Error = TypeError>,
{
    if !operation.region_slots().is_empty() {
        return Err(ProgramError::UnsupportedOperation {
            message: format!(
                "projected operation `{}` carries regions and cannot be transposed through its member family; \
                 transpose it through a composite carrier for that operation instead",
                operation.name(),
            ),
        }
        .into());
    }

    check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
    accumulators.iter().try_for_each(|accumulator| context.cotangent_storage(accumulator).map(|_| ()))?;

    // Stage the native member rule in an isolated member-typed trace. The completed rule program is converted back to
    // the composite type before it is attached to `context`, so primitive transpose rules never need composite types.
    // The trace is detached from any region, so the member rule cannot reach reference accumulators, which is what this
    // helper's region-free, reference-free contract requires.
    let mut rule_context = TranspositionContext::new(TracingContext::<<V as ValueProjection<T>>::Projected, P>::new());

    // Build the member rule's boundary and the matching source atoms together. Unknown primals contribute only their
    // projected types. Known primals become leading member-program inputs and their composite atoms become the leading
    // splice inputs, preserving the transposition rule's operand order. Validate builder ownership before taking atom
    // IDs: the same index in another trace names an unrelated value.
    let mut splice_inputs = Vec::new();
    let rule_inputs = inputs
        .iter()
        .map(|input| -> Result<_, DifferentiationError> {
            match input {
                PartialValue::Unknown(r#type) => Ok(PartialValue::Unknown(<&T>::try_from(r#type)?.clone())),
                PartialValue::Known(value) => {
                    check_builders!(context.builder(), value.builder())?;
                    splice_inputs.push(value.atom_id()?);
                    Ok(PartialValue::Known(rule_context.input(<&T>::try_from(value.r#type().as_ref())?.clone())))
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Live output cotangents follow known primals in both boundaries. Structural zeros remain type-only and therefore
    // consume neither a member-program input nor a composite splice input.
    let rule_outputs = outputs
        .iter()
        .map(|output| -> Result<_, DifferentiationError> {
            match output {
                MaybeZero::Zero(r#type) => Ok(MaybeZero::Zero(<&T>::try_from(r#type)?.clone())),
                MaybeZero::Value(value) => {
                    check_builders!(context.builder(), value.builder())?;
                    splice_inputs.push(value.atom_id()?);
                    Ok(MaybeZero::Value(rule_context.input(<&T>::try_from(value.r#type().as_ref())?.clone())))
                }
            }
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Apply the native transpose rule to the member boundary. Any operations synthesized by the rule are recorded in
    // `rule_context`. `EmptyRegionDriver` enforces this helper's region-free contract.
    let cotangent_mask = accumulators.iter().map(CotangentAccumulator::is_needed).collect::<Vec<_>>();
    let rule_accumulators = rule_context.cotangent_accumulators(&rule_inputs, &cotangent_mask)?;
    (*rule_context).clone().invoke_with_provenance_origin(context.provenance(), || {
        operation.transpose(&mut rule_context, &EmptyRegionDriver, &rule_inputs, &rule_outputs, &rule_accumulators)
    })?;

    // Export each accumulated member cotangent. Its additions are part of the member program, while reference
    // updates remain the responsibility of the enclosing context.
    let (contribution_inputs, output_ids): (Vec<_>, Vec<_>) = rule_accumulators
        .iter()
        .enumerate()
        .filter_map(|(index, accumulator)| match &rule_context.cotangent_storage[accumulator.storage_index] {
            CotangentStorage::Value { value: Some((value, provenance)), .. } => Some((index, value, provenance)),
            _ => None,
        })
        .map(|(index, value, provenance)| Ok(((index, provenance.clone()), value.atom_id()?)))
        .collect::<Result<_, DifferentiationError>>()?;

    // Convert the complete member program into the composite universe, then splice it with the source atoms collected
    // in precisely the same order as the temporary member-program inputs.
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
    let splice_outputs = context.builder().borrow_mut().splice_program(&rule_program, splice_inputs.as_slice())?;
    check_count!("output", splice_outputs, contribution_inputs.len(), ProgramError);
    contribution_inputs.into_iter().zip(splice_outputs).try_for_each(|((input, provenance), output)| {
        // Preserve scopes entered inside the member rule when outer storage later combines its contributions.
        let contribution = MaybeZero::Value(context.tracer(output, None));
        (**context)
            .clone()
            .invoke_with_provenance_origin(provenance, || accumulators[input].accumulate(context, contribution))
    })
}

/// Applies a member operation's transpose rule to an instruction whose parent boundary is _mixed_, meaning that the
/// instruction consumes its `T`-typed member operands together with operands belonging to other members of the parent
/// type universe (e.g., the first-class dimensions that supply a dynamic result shape), in any arrangement. Use this
/// function from a composite operation dispatcher for a [`Region`]-free payload that keeps its native member operation
/// type while its instruction crosses member kinds.
///
/// Each operand is classified individually rather than by position. An operand whose type projects into `T` is a _data_
/// operand and every other operand is a parent-universe shape operand. The data operands, in operand order, are
/// delegated to the payload's homogeneous [`TransposableOperation`] rule through [`transpose_projected_operation`],
/// together with their corresponding accumulator handles. Shape operands only select the result shape and carry no
/// differential contribution, so they receive no contribution. This classification makes the helper independent of how
/// the two operand kinds are arranged, so it handles interleaved signatures exactly like the "data-operands-first"
/// arrangement every current payload uses. A payload with no data operands at all (i.e., a dynamic constructor whose
/// operands are all extents) is a constant linear map, so no member rule runs.
///
/// Delegating reconstructs the member instruction from type metadata alone, so a mixed instruction whose operands carry
/// runtime (i.e., [`Reference`](TypeIdentityPosition::Reference)-position) identities is rejected. Recovering that
/// runtime-dependent type metadata requires linearization, which retains the relevant primal information as explicit
/// residuals.
///
/// # Parameters
///
///   - `context`: Active composite [`TranspositionContext`] the delegated member rule is spliced into.
///   - `operation`: Region-free linear member operation whose instruction has a mixed parent boundary. Its homogeneous
///     rule sees exactly the data operands, in the order they appear in the mixed instruction.
///   - `inputs`: Per-operand primal knowledge in the mixed instruction's operand order.
///   - `outputs`: Composite output cotangents, represented as live traced values or structural zeros.
///   - `accumulators`: Enclosing value cotangent handles, aligned with `inputs` and owned by `context`.
pub fn transpose_mixed_operation<
    T: DifferentiableType,
    P: Operation<Type = T>,
    V: Value<Type: DifferentiableType + From<T>> + ValueProjection<T, Projected: Value<Type = T>>,
    O: Operation<Type = V::Type> + OperationProjection<T> + From<AddOperation<V::Type>>,
>(
    context: &mut TranspositionContext<V, O>,
    operation: &P,
    inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
    outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
    accumulators: &[CotangentAccumulator],
) -> Result<(), DifferentiationError>
where
    <O as OperationProjection<T>>::Projected:
        From<P> + TransposableOperation<<V as ValueProjection<T>>::Projected, <O as OperationProjection<T>>::Projected>,
    for<'t> &'t T: TryFrom<&'t V::Type, Error = TypeError>,
{
    check_count!("accumulator", accumulators, inputs.len(), DifferentiationError);
    accumulators.iter().try_for_each(|accumulator| context.cotangent_storage(accumulator).map(|_| ()))?;

    // Classify each operand by whether its type projects into the member universe. Data operands keep their operand
    // order so the delegated member rule sees the same boundary it would see in a homogeneous instruction. Filter
    // handles alongside inputs to preserve the association between each operand and its enclosing storage.
    let (data_inputs, data_accumulators): (Vec<_>, Vec<_>) = inputs
        .iter()
        .zip(accumulators)
        .filter(|(input, _)| <&T>::try_from(input.r#type().as_ref()).is_ok())
        .map(|(input, accumulator)| (input.clone(), accumulator.clone()))
        .unzip();

    // A mixed instruction with no member-typed operands stages a value that does not depend on any of them, so every
    // operand receives a structural zero and the member rule is never consulted.
    if data_inputs.is_empty() {
        return Ok(());
    }

    // The delegated member rule sees only the member-typed operands, so it must be able to derive every cotangent shape
    // from those operands alone. A runtime identity anywhere in the mixed signature means the runtime shape information
    // lives in the parent-universe operands instead, and only linearization can retain it.
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

    let member_operation = <O as OperationProjection<T>>::Projected::from(operation.clone());
    transpose_projected_operation(context, &member_operation, &data_inputs, outputs, &data_accumulators)
}

/// [`Region`] [`Transform`] marker for retained transposed [`Program`]s.
struct TranspositionTransform;

impl<V: Value, O: Operation<Type = V::Type>> Transform<Region<V, O>> for TranspositionTransform {
    type Arguments = TranspositionTransformArguments;
    type Artifact = TransformArtifact<V, O, ()>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

/// Argument key for one retained [`TranspositionTransform`].
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TranspositionTransformArguments {
    /// Selected linear input indices, in requested output order.
    input_indices: Vec<usize>,

    /// Per-selected-input residual indices supplying runtime dimensions for disconnected cotangent zeros.
    zero_residual_input_indices: Vec<Vec<usize>>,

    /// Resolved cotangent [`CotangentDestinationKind`] of every selected input, aligned with `input_indices`.
    destination_kinds: Vec<CotangentDestinationKind>,
}

impl TranspositionTransformArguments {
    /// Creates a new [`TranspositionTransformArguments`] instance by validating the selected inputs and mapping counts
    /// and resolving default [`CotangentDestinationKind`]s before cache lookup. Input order and zero residual mappings
    /// are preserved because they determine the pullback boundary and the runtime dimensions used to construct
    /// disconnected cotangent zeros.
    fn new<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
        input_indices: &[usize],
        zero_residual_input_indices: &[Vec<usize>],
        destination_kinds: &[CotangentDestinationKind],
    ) -> Result<Self, DifferentiationError> {
        // Each supplied mapping identifies residual inputs needed to construct one selected input's zero cotangent.
        // Validate the outer count here; an empty slice means no mappings were supplied.
        if !zero_residual_input_indices.is_empty() && zero_residual_input_indices.len() != input_indices.len() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "transposition received {} zero-residual mappings for {} selected inputs",
                    zero_residual_input_indices.len(),
                    input_indices.len(),
                ),
            }
            .into());
        }

        // Explicit destination kinds follow the selected inputs, rather than every input in the region.
        // An empty slice requests the type-dependent defaults resolved below.
        if !destination_kinds.is_empty() && destination_kinds.len() != input_indices.len() {
            return Err(ProgramError::InvalidArgument {
                message: format!(
                    "transposition received {} destination kinds for {} selected inputs",
                    destination_kinds.len(),
                    input_indices.len(),
                ),
            }
            .into());
        }

        // Track membership without sorting: duplicate selections are invalid, but the requested input order
        // determines the order of cotangents at the pullback boundary and must be preserved.
        let input_count = region.input_ids().len();
        let mut selected_inputs = vec![false; input_count];
        let destination_kinds = input_indices
            .iter()
            .enumerate()
            .map(|(position, &index)| {
                // Validate the input position before indexing the membership mask. The selection position is
                // separate from this input index (e.g., for `[2, 0]` destination kind 0 describes input 2).
                let input = *region.input_ids().get(index).ok_or_else(|| ProgramError::InvalidArgument {
                    message: format!(
                        "transposition input index {index} is out of range for a program with {input_count} input(s)",
                    ),
                })?;
                // Reject a second occurrence instead of silently merging potentially different destinations.
                if selected_inputs[index] {
                    return Err(ProgramError::InvalidArgument {
                        message: format!("transposition input index {index} appears more than once"),
                    }
                    .into());
                }
                selected_inputs[index] = true;

                // Resolve defaults to explicit kinds so omitted defaults and their explicit spelling share a cache
                // entry. Reference inputs use state accumulators; other inputs return cotangent values by default.
                let input_type = region.atoms()[input.index()].r#type();
                let is_reference = input_type.is_reference();
                let default =
                    if is_reference { CotangentDestinationKind::Reference } else { CotangentDestinationKind::Return };
                let kind = destination_kinds.get(position).copied().unwrap_or(default);

                // Reference-state cotangents must use reference storage or be ignored; they cannot be returned
                // as value cotangents. Non-reference inputs support all destination kinds.
                match kind {
                    CotangentDestinationKind::Return if is_reference => Err(ProgramError::InvalidArgument {
                        message: format!(
                            "linear reference input {index} of type {input_type} cannot return its cotangent as a \
                             value; transpose it with a `Reference` or `Ignore` cotangent destination",
                        ),
                    }
                    .into()),
                    kind => Ok(kind),
                }
            })
            .collect::<Result<Vec<_>, DifferentiationError>>()?;

        // Retain the selection order and residual mappings alongside the normalized kinds. Together they identify
        // the requested pullback in the transform cache, including how disconnected zeros obtain runtime dimensions.
        Ok(Self {
            input_indices: input_indices.to_vec(),
            zero_residual_input_indices: zero_residual_input_indices.to_vec(),
            destination_kinds,
        })
    }
}

/// Prepares the [`CotangentDestination`]s for a scalar gradient before the primal function runs. Non-reference inputs
/// return their cotangents. Reference inputs receive fresh zero state cotangents, so only the scalar result contributes
/// to the derivative. Preparing the zeros now captures runtime dimensions while every primal reference is still live,
/// even when the function later consumes it. The pullback validates the resulting destinations against its primal
/// boundary.
fn gradient_destinations<C: Context + Zero<C::Value>, Input: Parameterized<C::Value>>(
    context: &C,
    primals: &Input,
) -> Result<Vec<CotangentDestination<C::Value>>, ProgramError>
where
    C::Type: DifferentiableType,
    C::Operation: OperationProvider<C::Type, ReferenceNewOperation<C::Type, C::Type>, Operation = C::Operation>
        + ResidualZeroProvider<C::Type>,
{
    primals
        .parameters()
        .map(|primal| {
            if !primal.r#type().is_reference() {
                return Ok(CotangentDestination::Return);
            }
            let cotangent_type = primal.r#type().cotangent()?;
            let referent = cotangent_type.referent().ok_or_else(|| ProgramError::UnsupportedOperation {
                message: format!(
                    "gradient cotangent type `{cotangent_type}` cannot represent its referent in this value family",
                ),
            })?;
            let zero = C::Operation::materialize_zero_from_residual_sources(
                context,
                MaybeZero::Zero(referent),
                std::iter::once(primal),
            )?;
            let mut outputs = context.bind(
                C::Operation::provide(ReferenceNewOperation::new(), &[zero.r#type().as_ref()])?,
                Vec::new(),
                &[zero],
            )?;
            check_count!("output", outputs, 1, ProgramError);
            Ok(CotangentDestination::Reference(outputs.remove(0)))
        })
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use approx::assert_abs_diff_eq;
    use indoc::indoc;
    use num_complex::Complex;
    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayOperation, ArrayReference, ArrayType, DataType,
        Dimension, DimensionBounds, DimensionVariable, ReferenceIndexOperation, Shape,
    };
    use crate::batching::{BatchAxis, batch};
    use crate::contexts::tests::{
        ProjectedMemberOperation, ProjectedMemberType, ProjectedProgramOperation, ProjectedProgramType,
        ProjectedProgramValue,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::differentiation::forward::LinearizationTracer;
    use crate::differentiation::{Differentiate, differentiate_at};
    use crate::macros::{check_count, check_gradient, check_types};
    use crate::operations::{
        AddOperation, ConditionOperation, Constant, CumulativeSum, Dot, DotDimensionNumbers, MulOperation, Reduce,
        ReduceOperation, ReductionKind, ReferenceAddUpdate, ReferenceAddUpdateOperation, ReferenceFreeze,
        ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead, ReferenceReadOperation,
        ReferenceSwap, ReferenceSwapOperation, ReferenceWrite, ReferenceWriteOperation, ScanOperation, Sin,
        ZeroOperation,
    };
    use crate::parameters::Placeholder;
    use crate::partial::PartialValue;
    use crate::programs::{
        AtomId, Concretizable, EffectClass, EffectClasses, Effects, MaybeZero, Operation, Program, ProgramBuilder,
        ProgramError, ProvenanceScope, ReferenceError, ReferenceType, RegionInterface, RegionSlot, TypeError, Typed,
        Value,
    };
    use crate::specialization::SpecializationCacheStatistics;
    use crate::tracing::{DomainTracer, DomainTracingContext, Trace, Tracer, TracingContext};

    use super::*;

    /// Returns retained transposition statistics for control-flow tests without exposing the transform's private
    /// cache marker or argument representation.
    pub(crate) fn transposition_statistics<V: Value, O: Operation<Type = V::Type>>(
        region: RegionRef<'_, V, O>,
    ) -> Option<SpecializationCacheStatistics> {
        region.transform_statistics::<TranspositionTransform>()
    }

    type TestTracingValue = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;

    /// Test-only linear operation type used to exercise transposition validation paths. Most variants model tiny
    /// rank-zero array primitives so the generated programs stay readable. The sentinel variants intentionally violate
    /// transpose rule contracts or builder ownership rules. Built-in array operations cannot represent those failures
    /// because their transpose implementations are valid by construction.
    #[derive(Clone, Debug)]
    enum TestLinearOperation {
        /// Single-input passthrough used when a test needs a live instruction whose transpose forwards its cotangent.
        Identity,

        /// Effectful passthrough used to verify that known-side producer replay never duplicates observable effects.
        EffectfulIdentity,

        /// Effectful single-input sink without outputs, used to verify that a non-state effect over a linear operand
        /// contributes no cotangent and is dropped from the pullback rather than rejected.
        EffectfulSink,

        /// Passthrough with one attached region used to verify that known-side producer replay preserves nested region
        /// closures and observes their recursively derived effects.
        RegionIdentity,

        /// Two-input addition used to verify cotangent accumulation through repeated primal inputs.
        Add,

        /// Named linear rule with a runtime coefficient followed by its linear operand. Its transpose is retained
        /// as executable Rust behavior rather than an attached backward program or captured closure.
        ScaleLinear,

        /// Single-input, two-output operation used to verify that unused operation results are passed to transpose
        /// rules as structural zero cotangents.
        TwoOutputs,

        /// Single-input operation whose transpose stages a [`ZeroOperation`] as the input cotangent contribution. The
        /// staged zero remains an input-free [`ZeroOperation`] instruction in the pullback and is materialized at
        /// interpretation time. Built-in array operations do not stage that exact structural-zero contribution, so
        /// this sentinel keeps that path directly covered.
        StagedZeroContribution,

        /// Single-input operation whose transpose submits no contributions, leaving a structural zero cotangent.
        NoContribution,

        /// Single-input operation whose transpose deliberately returns a cotangent staged in another builder. This
        /// verifies that the transposition pass rejects contributions from foreign builders before their atom IDs can
        /// alias unrelated atoms in the destination pullback.
        ForeignContribution,

        /// Real zero operation wrapper used by the `From<ZeroOperation>`/`TryFrom<ZeroOperation>` conversions for this
        /// test operation enum.
        Zero(ZeroOperation<ArrayType>),
    }

    impl Operation for TestLinearOperation {
        type Type = ArrayType;

        #[inline]
        fn name(&self) -> &'static str {
            match self {
                Self::Identity => "identity",
                Self::EffectfulIdentity => "effectful_identity",
                Self::EffectfulSink => "effectful_sink",
                Self::RegionIdentity => "region_identity",
                Self::Add => "add",
                Self::ScaleLinear => "scale_linear",
                Self::TwoOutputs => "two_outputs",
                Self::StagedZeroContribution => "staged_zero_contribution",
                Self::NoContribution => "no_contribution",
                Self::ForeignContribution => "foreign_contribution",
                Self::Zero(_) => "zero",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::RegionIdentity => const { &[RegionSlot::computation("body")] },
                _ => &[],
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayType],
            region_interfaces: &[RegionInterface<ArrayType>],
        ) -> Result<Vec<ArrayType>, TypeError> {
            match self {
                Self::Identity
                | Self::EffectfulIdentity
                | Self::StagedZeroContribution
                | Self::NoContribution
                | Self::ForeignContribution => {
                    check_count!("region", region_interfaces, 0, TypeError);
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::EffectfulSink => {
                    check_count!("region", region_interfaces, 0, TypeError);
                    check_count!("input", input_types, 1, TypeError);
                    Ok(Vec::new())
                }
                Self::RegionIdentity => {
                    check_count!("region", region_interfaces, 1, TypeError);
                    check_count!("input", input_types, 1, TypeError);
                    check_types!(@same, "region identity input", [input_types, region_interfaces[0].input_types()]);
                    check_types!(@same, "region identity output", [input_types, region_interfaces[0].output_types()]);
                    Ok(vec![input_types[0].clone()])
                }
                Self::Add | Self::ScaleLinear => {
                    check_count!("region", region_interfaces, 0, TypeError);
                    check_count!("input", input_types, 2, TypeError);
                    Ok(vec![input_types[0].clone()])
                }
                Self::TwoOutputs => {
                    check_count!("region", region_interfaces, 0, TypeError);
                    check_count!("input", input_types, 1, TypeError);
                    Ok(vec![input_types[0].clone(), input_types[0].clone()])
                }
                Self::Zero(zero) => zero.infer_output_types(input_types, region_interfaces),
            }
        }

        fn effects(&self) -> Cow<'_, Effects> {
            Cow::Owned(Effects::explicit(match self {
                Self::EffectfulIdentity | Self::EffectfulSink => EffectClasses::single(EffectClass::OrderedIo),
                _ => EffectClasses::NONE,
            }))
        }

        fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
            match self {
                Self::Zero(zero) => zero.render(formatter, indentation),
                _ => formatter.write_str(self.name()),
            }
        }
    }

    impl From<AddOperation<ArrayType>> for TestLinearOperation {
        #[inline]
        fn from(_operation: AddOperation<ArrayType>) -> Self {
            Self::Add
        }
    }

    impl From<ZeroOperation<ArrayType>> for TestLinearOperation {
        #[inline]
        fn from(operation: ZeroOperation<ArrayType>) -> Self {
            Self::Zero(operation)
        }
    }

    // This mirrors the borrowed payload projection that `#[derive(Operation)]` generates,
    // including its canonical wrong-payload diagnostic.
    impl<'o> TryFrom<&'o TestLinearOperation> for &'o ZeroOperation<ArrayType> {
        type Error = TypeError;

        #[inline]
        fn try_from(value: &'o TestLinearOperation) -> Result<Self, TypeError> {
            match value {
                TestLinearOperation::Zero(zero) => Ok(zero),
                _ => Err(TypeError::invalid(format!(
                    "cannot project operation `{}` into a `ZeroOperation<ArrayType>` payload",
                    value.name(),
                ))),
            }
        }
    }

    impl<V: Value<Type = ArrayType>> TransposableOperation<V, TestLinearOperation> for TestLinearOperation {
        fn transpose<D: TranspositionDriver<V, TestLinearOperation>>(
            &self,
            context: &mut TranspositionContext<V, TestLinearOperation>,
            _driver: &D,
            inputs: &[PartialValue<Tracer<TracingContext<V, TestLinearOperation>>>],
            outputs: &[MaybeZero<Tracer<TracingContext<V, TestLinearOperation>>>],
            accumulators: &[CotangentAccumulator],
        ) -> Result<(), DifferentiationError> {
            match self {
                Self::Identity | Self::EffectfulIdentity | Self::RegionIdentity => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    accumulators[0].accumulate(context, outputs[0].clone())
                }
                Self::EffectfulSink => {
                    check_count!("input", inputs, 1, ProgramError);
                    check_count!("output", outputs, 0, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    Ok(())
                }
                Self::Add => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 2, DifferentiationError);
                    accumulators[0].accumulate(context, outputs[0].clone())?;
                    accumulators[1].accumulate(context, outputs[0].clone())
                }
                Self::ScaleLinear => {
                    check_count!("input", inputs, 2, ProgramError);
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 2, DifferentiationError);
                    if accumulators[1].is_needed()
                        && let MaybeZero::Value(seed) = &outputs[0]
                    {
                        let coefficient = inputs[0].as_known().unwrap().clone();
                        let contribution = context
                            .stage_operation(Self::ScaleLinear, Vec::new(), &[coefficient, seed.clone()])?
                            .remove(0);
                        accumulators[1].accumulate(context, MaybeZero::Value(contribution))?;
                    }
                    Ok(())
                }
                Self::TwoOutputs => {
                    check_count!("output", outputs, 2, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    assert!(outputs[1].is_zero());
                    accumulators[0].accumulate(context, outputs[0].clone())
                }
                Self::StagedZeroContribution => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    let zero = {
                        let mut builder = context.builder().borrow_mut();
                        let outputs = builder.add_instruction(
                            Self::Zero(ZeroOperation::new(ArrayType::scalar(DataType::F64))),
                            Vec::new(),
                            Vec::new(),
                            None,
                        )?;
                        check_count!("output", outputs, 1, ProgramError);
                        outputs[0]
                    };
                    let contribution = MaybeZero::Value(context.tracer(zero, None));
                    accumulators[0].accumulate(context, contribution)
                }
                Self::NoContribution => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    Ok(())
                }
                Self::ForeignContribution => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 1, DifferentiationError);
                    let foreign_context = TracingContext::<V, TestLinearOperation>::new();
                    accumulators[0]
                        .accumulate(context, MaybeZero::Value(foreign_context.input(ArrayType::scalar(DataType::F64))))
                }
                Self::Zero(_) => {
                    check_count!("output", outputs, 1, ProgramError);
                    check_count!("accumulator", accumulators, 0, DifferentiationError);
                    Ok(())
                }
            }
        }
    }

    type ReferenceTestValue = ArrayIrValue<Array>;
    type ReferenceTestOperation = ArrayIrOperation<Array>;
    type ReferenceTestContext = EagerContext<ReferenceTestValue, ReferenceTestOperation>;
    type ReferenceTestTracer = LinearizationTracer<ReferenceTestContext>;
    type ReferenceTestProgram =
        Program<ReferenceTestValue, ReferenceTestOperation, Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>;

    /// Wraps a scalar `f32` array into the composite value used by the reference transposition tests.
    fn reference_test_scalar(value: f32) -> ReferenceTestValue {
        ArrayIrValue::Array(Array::scalar(value))
    }

    /// Squares a value in the composite reference test family, preserving the active transform context.
    fn reference_test_square<V: Value<Type = ArrayIrType>>(value: V) -> Result<V, ProgramError>
    where
        V::DispatchDomain: Context<Value = V, Operation: From<ArrayOperation<Array>>>,
    {
        Ok(value
            .dispatch_domain()
            .bind(ArrayOperation::from(MulOperation::<ArrayType>::new()), Vec::new(), &[value.clone(), value.clone()])?
            .remove(0))
    }

    /// Runs a transposed program whose boundary is `[cotangents..., cotangent references..., known...]` by allocating
    /// each cotangent reference locally from `destinations`, splicing the transposed program behind those allocations
    /// so the interpreted program owns the state it mutates, and freezing the allocations afterwards. Returns the
    /// transposed program's non-reference outputs followed by the final contents of every destination, in order.
    fn run_transposed_with_destinations(
        transposed: &ReferenceTestProgram,
        cotangents: Vec<Array>,
        destinations: Vec<Array>,
        known: Vec<Array>,
    ) -> Vec<Array> {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let mut inputs = Vec::new();
        let mut values = Vec::new();
        for cotangent in cotangents {
            inputs.push(builder.add_input(cotangent.r#type().into_owned().into()));
            values.push(ArrayIrValue::Array(cotangent));
        }
        let mut references = Vec::new();
        for destination in destinations {
            let state = builder.add_input(destination.r#type().into_owned().into());
            values.push(ArrayIrValue::Array(destination));
            let reference =
                builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![state], None).unwrap()[0];
            inputs.push(reference);
            references.push(reference);
        }
        for value in known {
            inputs.push(builder.add_input(value.r#type().into_owned().into()));
            values.push(ArrayIrValue::Array(value));
        }
        let mut outputs = builder
            .splice_program(transposed, inputs.as_slice())
            .unwrap()
            .into_iter()
            .zip(transposed.output_types())
            .filter_map(|(output, r#type)| (!r#type.is_reference()).then_some(output))
            .collect::<Vec<_>>();
        for reference in references {
            outputs.push(
                builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0],
            );
        }
        let input_count = values.len();
        let output_count = outputs.len();
        let runnable = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                outputs,
                vec![Placeholder; input_count],
                vec![Placeholder; output_count],
            )
            .unwrap();
        runnable
            .interpret(values)
            .unwrap()
            .into_iter()
            .map(|value| match value {
                ArrayIrValue::Array(array) => array,
                other => panic!("expected an array output but got {other:?}"),
            })
            .collect()
    }

    #[test]
    fn test_cotangent_destination_kind() {
        let reference = reference_test_scalar(1.0);
        assert_eq!(CotangentDestination::<ReferenceTestValue>::Return.kind(), CotangentDestinationKind::Return);
        assert_eq!(CotangentDestination::Reference(reference).kind(), CotangentDestinationKind::Reference);
        assert_eq!(CotangentDestination::<ReferenceTestValue>::Ignore.kind(), CotangentDestinationKind::Ignore);
    }

    #[test]
    fn test_cotangent_destinations_without_references() {
        // A universe without references resolves every operand to the `Return` kind.
        let cotangents = CotangentDestinations::<ReferenceTestTracer>::without_references([true; 3]);
        assert_eq!(cotangents.kinds(), &[CotangentDestinationKind::Return; 3]);
        assert!(cotangents.references().is_empty());
        assert!(!cotangents.has_reference_state_destinations());
        assert!(!cotangents.is_reference_input(1));

        let cotangents = CotangentDestinations::<ReferenceTestTracer>::without_references([true, false]);
        assert_eq!(cotangents.kinds(), &[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore]);
        assert!(!cotangents.is_reference_input(1));
        assert!(!cotangents.returns_cotangent(1));
    }

    #[test]
    fn test_cotangent_destinations_kind() {
        let destinations = CotangentDestinations::<Array>::without_references([true, false]);
        assert_eq!(destinations.kind(0), CotangentDestinationKind::Return);
        assert_eq!(destinations.kind(1), CotangentDestinationKind::Ignore);
    }

    #[test]
    fn test_cotangent_destinations_kinds() {
        let destinations = CotangentDestinations::<Array>::without_references([false, true, false]);
        assert_eq!(
            destinations.kinds(),
            &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return, CotangentDestinationKind::Ignore,]
        );
    }

    #[test]
    fn test_cotangent_destinations_references() {
        let first = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32)));
        let second = ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0_f32)));
        let destinations = CotangentDestinations::new(
            vec![
                CotangentDestinationKind::Reference,
                CotangentDestinationKind::Return,
                CotangentDestinationKind::Reference,
            ],
            vec![first.clone(), second.clone()],
            vec![false, false, true],
        );
        assert_eq!(destinations.references(), &[first, second]);
    }

    #[test]
    fn test_cotangent_destinations_is_reference_input() {
        // A supplied gradient buffer does not turn its non-reference primal input into reference state.
        let destinations = CotangentDestinations::new(
            vec![CotangentDestinationKind::Reference; 2],
            vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0_f32))); 2],
            vec![false, true],
        );
        assert!(!destinations.is_reference_input(0));
        assert!(destinations.is_reference_input(1));
    }

    #[test]
    fn test_cotangent_destinations_returns_cotangent() {
        let destinations = CotangentDestinations::new(
            vec![
                CotangentDestinationKind::Return,
                CotangentDestinationKind::Reference,
                CotangentDestinationKind::Reference,
                CotangentDestinationKind::Ignore,
            ],
            vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0_f32))); 2],
            vec![false, false, true, true],
        );
        assert!(destinations.returns_cotangent(0));
        assert!(!destinations.returns_cotangent(1));
        assert!(destinations.returns_cotangent(2));
        assert!(!destinations.returns_cotangent(3));
    }

    #[test]
    fn test_cotangent_destinations_has_reference_state_destinations() {
        let references = vec![ArrayIrValue::Reference(ArrayReference::new(Array::scalar(0.0_f32)))];
        let buffer =
            CotangentDestinations::new(vec![CotangentDestinationKind::Reference], references.clone(), vec![false]);
        assert!(!buffer.has_reference_state_destinations());
        let state = CotangentDestinations::new(vec![CotangentDestinationKind::Reference], references, vec![true]);
        assert!(state.has_reference_state_destinations());
        let ignored = CotangentDestinations::<ArrayIrValue<Array>>::new(
            vec![CotangentDestinationKind::Ignore],
            Vec::new(),
            vec![true],
        );
        assert!(!ignored.has_reference_state_destinations());
    }

    #[test]
    fn test_cotangent_reference_accumulator_allocate_in() {
        let context = TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new();
        let unallocated = CotangentReferenceAccumulator::<Tracer<TracingContext<ReferenceTestValue, _>>>::Unallocated {
            cotangent_type: ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
        };
        assert!(!unallocated.is_allocated());
        assert!(unallocated.reference().is_none());
        assert_eq!(
            unallocated.r#type().as_ref(),
            &ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
        );

        // Allocation stages exactly one zero and one `reference_new`, and the allocated accumulator reports them.
        let mut accumulator = unallocated;
        let reference = accumulator.allocate_in(&context, &[]).unwrap().clone();
        assert!(accumulator.is_allocated());
        assert_eq!(accumulator.reference().map(Tracer::atom_id), Some(reference.atom_id()));
        assert_eq!(
            accumulator.r#type().as_ref(),
            &ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
        );
        let names = context
            .builder()
            .borrow()
            .instructions()
            .iter()
            .map(|instruction| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["zero", "reference_new"]);

        // Allocating again is the identity on an allocated accumulator.
        let again = accumulator.allocate_in(&context, &[]).unwrap().clone();
        assert_eq!(again.atom_id(), reference.atom_id());
        assert_eq!(context.builder().borrow().instructions().len(), 2);

        // A type that does not project onto a referent cannot be allocated.
        let mut accumulator =
            CotangentReferenceAccumulator::<Tracer<TracingContext<ReferenceTestValue, _>>>::Unallocated {
                cotangent_type: ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
            };
        assert!(matches!(
            accumulator.allocate_in(&context, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot allocate a cotangent reference of type f32[] because its universe does not \
                    project the type onto a referent",
        ));
    }

    #[test]
    fn test_cotangent_reference_accumulator_allocate_in_requires_dynamic_dimensions() {
        let context = TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new();
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let mut accumulator = CotangentReferenceAccumulator::Unallocated {
            cotangent_type: ReferenceType::new(ArrayType::new(
                DataType::F32,
                Shape::new(vec![Dimension::Dynamic(extent)]),
            ))
            .into(),
        };
        // A dynamic root cannot be allocated from its symbolic type alone. Missing evidence fails during staging,
        // before any allocation or update is appended to the destination program.
        assert!(matches!(
            accumulator.allocate_in(&context, &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "cannot materialize a zero of type f32[extent] because no value in scope supplies the \
                    runtime geometry dimension<extent ∈ [2, 8)> that its residual 0 names",
        ));
        assert!(!accumulator.is_allocated());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_cotangent_accumulator_is_needed() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let known = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let inputs =
            [PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32))), PartialValue::Known(known)];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert!(accumulators[0].is_needed());
        assert!(!accumulators[1].is_needed());
        let ignored = context.cotangent_accumulators(&inputs, &[false, false]).unwrap();
        assert!(!ignored[0].is_needed());
    }

    #[test]
    fn test_cotangent_accumulator_reference() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let accumulator = context
            .cotangent_accumulators(&[PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))], &[])
            .unwrap()
            .remove(0);
        assert!(accumulator.reference(&context).unwrap().is_none());
        let other = TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        assert!(matches!(accumulator.reference(&other),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "cotangent accumulator belongs to another transposition context"));
    }

    #[test]
    fn test_cotangent_accumulator_accumulate() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let accumulator = context
            .cotangent_accumulators(&[PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))], &[])
            .unwrap()
            .remove(0);
        let first = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let second = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let third = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        accumulator.accumulate(&mut context, MaybeZero::Value(first.clone())).unwrap();
        assert!(context.builder().borrow().instructions().is_empty());
        accumulator.clone().accumulate(&mut context, MaybeZero::Value(second.clone())).unwrap();
        assert_eq!(context.builder().borrow().instructions().len(), 1);
        accumulator.accumulate(&mut context, MaybeZero::Value(third.clone())).unwrap();
        // Additions are staged as contributions arrive, in submission order, rather than at extraction.
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 2);
        assert_eq!(builder.instructions()[0].operation().name(), "add");
        assert_eq!(builder.instructions()[1].operation().name(), "add");
        assert_eq!(builder.instructions()[0].inputs(), &[first.atom_id().unwrap(), second.atom_id().unwrap()]);
        assert_eq!(
            builder.instructions()[1].inputs(),
            &[builder.instructions()[0].outputs()[0], third.atom_id().unwrap()],
        );
        let sum = builder.instructions()[1].outputs()[0];
        drop(builder);
        // Aliased handles share the sum, rather than consuming a contribution once per operand occurrence.
        let values = context.take_cotangents(&[accumulator.clone(), accumulator.clone()]).unwrap();
        assert_eq!(values[0].as_value().unwrap().atom_id().unwrap(), sum);
        assert_eq!(values[1].as_value().unwrap().atom_id().unwrap(), sum);
        assert_eq!(context.builder().borrow().instructions().len(), 2);
        assert!(context.take_cotangents(&[accumulator]).unwrap()[0].is_zero());
    }

    #[test]
    fn test_cotangent_accumulator_accumulate_validates_ignored_contributions() {
        let mut context = TranspositionContext::new(TracingContext::<Array, ArrayOperation<Array>>::new());
        let ignored = context.cotangent_accumulator(ArrayType::scalar(DataType::F32), false);
        // Ignoring a gradient does not make malformed rule output acceptable, including a mistyped symbolic zero.
        assert_eq!(
            ignored.accumulate(&mut context, MaybeZero::Zero(ArrayType::scalar(DataType::F64))),
            Err(TypeError::invalid("cotangent contribution has type f64[] but its accumulator expects f32[]").into()),
        );
        let foreign = TracingContext::<Array, ArrayOperation<Array>>::new();
        assert!(matches!(
            ignored.accumulate(&mut context, MaybeZero::Value(foreign.input(ArrayType::scalar(DataType::F32)))),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
        assert_eq!(ignored.accumulate(&mut context, MaybeZero::Zero(ArrayType::scalar(DataType::F32))), Ok(()));
        assert!(context.builder().borrow().instructions().is_empty());
        assert!(context.take_cotangents(&[ignored]).unwrap()[0].is_zero());
    }

    #[test]
    fn test_cotangent_accumulator_accumulate_reference() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let reference = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let contribution = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let accumulator = context.cotangent_accumulator(ArrayIrType::Array(ArrayType::scalar(DataType::F32)), true);
        context.cotangent_storage[accumulator.storage_index] = CotangentStorage::Buffer {
            cotangent_type: ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
            reference: reference.clone(),
            operation: ReferenceAddUpdateOperation::new().into(),
        };
        assert_eq!(accumulator.reference(&context).unwrap().unwrap().atom_id(), reference.atom_id());
        accumulator.accumulate(&mut context, MaybeZero::Value(contribution)).unwrap();
        assert_eq!(context.builder().borrow().instructions()[0].operation().name(), "reference_add_update");
        assert!(context.take_cotangents(&[accumulator]).unwrap()[0].is_zero());
    }

    #[test]
    fn test_cotangent_accumulator_accumulate_rejects_reference_state() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let reference = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let accumulator = context
            .cotangent_accumulators(
                &[PartialValue::Unknown(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))))],
                &[],
            )
            .unwrap()
            .remove(0);
        assert!(matches!(accumulator.accumulate(&mut context, MaybeZero::Value(reference)),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "reference-state cotangents cannot be contributed to a value cotangent accumulator"));
    }

    #[test]
    fn test_transposition_context_new() {
        // A driver without a source instruction cannot identify reference operands or allocating outputs.
        let driver = EmptyRegionDriver;
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        assert!(matches!(
            context.cotangent_reference(&driver, 0),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 0 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));
        assert!(matches!(
            context.cotangent_reference_if_allocated(&driver, 1),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "input 1 has no reference root in a transposition context that is not scoped to a \
                    reference-carrying instruction",
        ));
        assert!(matches!(
            context.take_reference_cotangent(&driver, 0),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "output 0 has no reference allocation in a transposition context that is not scoped to \
                    a reference-carrying instruction",
        ));

        // The context dereferences to its tracing context, so non-reference staging goes through unchanged and the staged
        // value belongs to the wrapped context's builder.
        let input = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        assert_eq!(input.atom_id(), Ok(AtomId::new(0)));
        assert!(Rc::ptr_eq(input.builder(), (*context).builder()));
    }

    #[test]
    fn test_transposition_context_cotangent_accumulator() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let first = context.cotangent_accumulator(scalar_type.clone(), true);
        let second = context.cotangent_accumulator(scalar_type.clone(), true);
        let ignored = context.cotangent_accumulator(scalar_type.clone(), false);
        assert!(first.is_needed());
        assert!(!ignored.is_needed());
        assert!(context.builder().borrow().instructions().is_empty());

        // Matching types do not share storage. A contribution to the first handle leaves the second empty,
        // while an ignored handle accepts the same valid contribution without retaining it.
        let contribution = context.input(scalar_type.clone());
        first.accumulate(&mut context, MaybeZero::Value(contribution.clone())).unwrap();
        ignored.accumulate(&mut context, MaybeZero::Value(contribution.clone())).unwrap();
        let values = context.take_cotangents(&[first, second, ignored]).unwrap();
        assert_eq!(values[0].as_value().unwrap().atom_id(), contribution.atom_id());
        for value in &values[1..] {
            assert!(value.is_zero());
            assert_eq!(value.r#type().as_ref(), &scalar_type);
        }
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_transposition_context_cotangent_accumulators() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let known = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let inputs = [
            PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32))),
            PartialValue::Known(known),
            PartialValue::Unknown(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))),
        ];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        assert_eq!(
            accumulators.iter().map(CotangentAccumulator::is_needed).collect::<Vec<_>>(),
            vec![true, false, false]
        );
        let ignored = context.cotangent_accumulators(&inputs, &[false; 3]).unwrap();
        assert_eq!(ignored.iter().map(CotangentAccumulator::is_needed).collect::<Vec<_>>(), vec![false; 3]);
        assert_ne!(accumulators[0].storage_index, ignored[0].storage_index);

        // A valid first input must not install a slot before a later foreign input fails validation.
        let foreign = TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new();
        let inputs = [
            PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32))),
            PartialValue::Known(foreign.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))),
        ];
        let slot_count = context.cotangent_storage.len();
        assert!(matches!(
            context.cotangent_accumulators(&inputs, &[]),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
        assert_eq!(context.cotangent_storage.len(), slot_count);
    }

    #[test]
    fn test_transposition_context_cotangent_reference_source_instructions() {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let first = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let second = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let first_output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first], None).unwrap()[0];
        let second_output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![first_output, second_output],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let mut context = TranspositionContext::for_region(
            TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new(),
            region,
            Vec::new(),
        )
        .unwrap();
        let first_destination = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let second_destination = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        context.reference_accumulators.insert(
            ReferenceRoot::RegionInput { region: region.id(), input_index: 0 },
            CotangentReferenceAccumulator::Allocated { reference: first_destination.clone() },
        );
        context.reference_accumulators.insert(
            ReferenceRoot::RegionInput { region: region.id(), input_index: 1 },
            CotangentReferenceAccumulator::Allocated { reference: second_destination.clone() },
        );
        let first_driver = RecursiveTranspositionDriver::new(region, 0).unwrap();
        let second_driver = RecursiveTranspositionDriver::new(region, 1).unwrap();

        // Operand zero names different roots in these instructions. Each driver preserves its source scope even
        // when the same context serves interleaved lookups for both instructions.
        assert_eq!(context.cotangent_reference(&first_driver, 0).unwrap().atom_id(), first_destination.atom_id());
        assert_eq!(context.cotangent_reference(&second_driver, 0).unwrap().atom_id(), second_destination.atom_id());
        assert_eq!(context.cotangent_reference(&first_driver, 0).unwrap().atom_id(), first_destination.atom_id());
        assert!(context.builder().borrow().instructions().is_empty());
    }

    #[test]
    fn test_transposition_context_cotangent_destinations() {
        // `condition(p, r, x)` whose branches only store into `r` (through `write` and `add_update`) and return `x`.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let storing_branch = |accumulate: bool| {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            if accumulate {
                builder
                    .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                    .unwrap();
            } else {
                builder
                    .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                    .unwrap();
            }
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                    vec![value],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let true_branch = builder.import_region(storing_branch(false).entry_region_ref());
        let false_branch = builder.import_region(storing_branch(true).entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let root = ReferenceRoot::RegionInput { region: region.id(), input_index: 1 };
        let mut context = TranspositionContext::for_region(
            TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new(),
            region,
            Vec::new(),
        )
        .unwrap();
        let driver = RecursiveTranspositionDriver::new(region, 0).unwrap();
        let inputs = [
            PartialValue::Known(context.input(predicate_type.clone())),
            PartialValue::Unknown(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))),
            PartialValue::Unknown(scalar_type.clone()),
        ];

        // With an unallocated accumulator and no read anywhere in the branches, the reference operand's state cotangent
        // is provably zero: it resolves to the `Ignore` kind and nothing is allocated.
        context.reference_accumulators.insert(
            root,
            CotangentReferenceAccumulator::Unallocated {
                cotangent_type: ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            },
        );
        let cotangents = context.cotangent_destinations(&driver, &inputs, &[]).unwrap();
        assert_eq!(
            cotangents.kinds(),
            &[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore, CotangentDestinationKind::Return],
        );
        assert_eq!(cotangents.kind(1), CotangentDestinationKind::Ignore);
        assert!(cotangents.is_reference_input(1));
        assert!(!cotangents.is_reference_input(2));
        assert!(cotangents.references().is_empty());
        assert!(!cotangents.has_reference_state_destinations());
        assert!(!context.cotangent_accumulator_reference(root).is_some());
        assert!(context.builder().borrow().instructions().is_empty());

        // An accumulator that a later instruction already allocated is live and is handed out as the destination.
        let destination = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        context
            .reference_accumulators
            .insert(root, CotangentReferenceAccumulator::Allocated { reference: destination.clone() });
        let cotangents = context.cotangent_destinations(&driver, &inputs, &[]).unwrap();
        assert_eq!(
            cotangents.kinds(),
            &[CotangentDestinationKind::Return, CotangentDestinationKind::Reference, CotangentDestinationKind::Return],
        );
        assert_eq!(cotangents.references().len(), 1);
        assert_eq!(cotangents.references()[0].atom_id(), destination.atom_id());
        assert!(cotangents.has_reference_state_destinations());

        // A branch that reads the reference makes its state cotangent live even when nothing accumulated into it yet,
        // so the accumulator is allocated on the spot.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        builder.add_input(scalar_type.clone());
        let current =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let reading_branch = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![current],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let true_branch = builder.import_region(reading_branch.entry_region_ref());
        let false_branch = builder.import_region(storing_branch(true).entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let root = ReferenceRoot::RegionInput { region: region.id(), input_index: 1 };
        let mut context = TranspositionContext::for_region(
            TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new(),
            region,
            Vec::new(),
        )
        .unwrap();
        let driver = RecursiveTranspositionDriver::new(region, 0).unwrap();
        context.reference_accumulators.insert(
            root,
            CotangentReferenceAccumulator::Unallocated {
                cotangent_type: ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            },
        );
        let inputs = [
            PartialValue::Known(context.input(predicate_type)),
            PartialValue::Unknown(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))),
            PartialValue::Unknown(scalar_type),
        ];
        let cotangents = context.cotangent_destinations(&driver, &inputs, &[]).unwrap();
        assert_eq!(cotangents.kind(1), CotangentDestinationKind::Reference);
        assert!(cotangents.has_reference_state_destinations());
        assert!(context.cotangent_accumulator_reference(root).is_some());
        assert_eq!(
            context
                .builder()
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero", "reference_new"],
        );
    }

    #[test]
    fn test_transposition_context_cotangent_destinations_non_reference_destinations() {
        let mut context =
            TranspositionContext::new(TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new());
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let destination = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let accumulators = [
            context.cotangent_accumulator(scalar_type.clone(), true),
            context.cotangent_accumulator(scalar_type.clone(), true),
            context.cotangent_accumulator(scalar_type.clone(), false),
        ];
        context.cotangent_storage[accumulators[1].storage_index] = CotangentStorage::Buffer {
            cotangent_type: scalar_type.clone(),
            reference: destination.clone(),
            operation: ReferenceAddUpdateOperation::new().into(),
        };
        let inputs = [
            PartialValue::Unknown(scalar_type.clone()),
            PartialValue::Unknown(scalar_type.clone()),
            PartialValue::Unknown(scalar_type),
        ];
        let cotangents = context.cotangent_destinations(&EmptyRegionDriver, &inputs, &accumulators).unwrap();
        assert_eq!(
            cotangents.kinds(),
            &[CotangentDestinationKind::Return, CotangentDestinationKind::Reference, CotangentDestinationKind::Ignore]
        );
        assert_eq!(cotangents.references().len(), 1);
        assert_eq!(cotangents.references()[0].atom_id(), destination.atom_id());
        assert!(cotangents.returns_cotangent(0));
        assert!(!cotangents.returns_cotangent(1));
        assert!(!cotangents.returns_cotangent(2));
        // A non-reference gradient buffer does not carry reference state or keep a zero-seeded transpose live.
        assert!(!cotangents.is_reference_input(1));
        assert!(!cotangents.has_reference_state_destinations());
    }

    #[test]
    fn test_transposition_context_take_cotangents() {
        let tracing_context = TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new();
        let mut context = TranspositionContext::new(tracing_context.clone());
        let inputs = [PartialValue::Unknown(ArrayIrType::Array(ArrayType::scalar(DataType::F32)))];
        let accumulator = context.cotangent_accumulators(&inputs, &[]).unwrap().remove(0);
        let first = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let second = context.input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        accumulator.accumulate(&mut context, MaybeZero::Value(first.clone())).unwrap();
        accumulator.accumulate(&mut context, MaybeZero::Value(second.clone())).unwrap();

        // Contexts sharing one builder still own separate cotangent slots. Validate every handle before draining
        // even the first valid slot.
        let mut foreign = TranspositionContext::new(tracing_context);
        let foreign_accumulator = foreign.cotangent_accumulators(&inputs, &[]).unwrap().remove(0);
        assert!(matches!(
            context.take_cotangents(&[accumulator.clone(), foreign_accumulator]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "cotangent accumulator belongs to another transposition context",
        ));
        assert!(matches!(
            context.cotangent_storage[accumulator.storage_index],
            CotangentStorage::Value { value: Some(_), .. },
        ));
        assert_eq!(context.builder().borrow().instructions().len(), 1);

        // Repeated handles share one sum and draining removes the submitted contributions only once.
        let values = context.take_cotangents(&[accumulator.clone(), accumulator.clone()]).unwrap();
        assert_eq!(values[0].as_value().unwrap().atom_id(), values[1].as_value().unwrap().atom_id());
        let builder = context.builder().borrow();
        assert_eq!(builder.instructions().len(), 1);
        assert_eq!(builder.instructions()[0].operation().name(), "add");
        assert_eq!(builder.instructions()[0].inputs(), &[first.atom_id().unwrap(), second.atom_id().unwrap()]);
        drop(builder);
        assert!(context.take_cotangents(&[accumulator.clone()]).unwrap()[0].is_zero());
        accumulator.accumulate(&mut context, MaybeZero::Value(first.clone())).unwrap();
        let values = context.take_cotangents(&[accumulator]).unwrap();
        assert_eq!(values[0].as_value().unwrap().atom_id(), first.atom_id());
    }

    #[test]
    fn test_transposition_context_take_reference_cotangent() {
        // `r = reference_new(x); add_update(r, y)`: the allocation's accumulator is consumed by the lookup that its
        // transpose performs, so the table entry is gone afterwards whether or not anything accumulated into it.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let update = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, update], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        let region = program.entry_region_ref();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(region.id(), 0), output_index: 0 };
        let mut context = TranspositionContext::for_region(
            TracingContext::<ReferenceTestValue, ReferenceTestOperation>::new(),
            region,
            Vec::new(),
        )
        .unwrap();

        // Only the allocating output of the allocating instruction can be looked up.
        let driver = RecursiveTranspositionDriver::new(region, 1).unwrap();
        assert!(matches!(
            context.take_reference_cotangent(&driver, 0),
            Err(DifferentiationError::Program(ProgramError::MalformedProgram(message)))
                if message == "output 0 of `reference_add_update` is not a reference allocation",
        ));

        // An accumulator that nothing reached yields no cotangent and is removed.
        let driver = RecursiveTranspositionDriver::new(region, 0).unwrap();
        context.reference_accumulators.insert(
            root,
            CotangentReferenceAccumulator::Unallocated {
                cotangent_type: ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            },
        );
        assert!(context.take_reference_cotangent(&driver, 0).unwrap().is_none());
        assert!(context.cotangent_accumulator_reference(root).is_none());
        assert!(context.builder().borrow().instructions().is_empty());

        // An allocated accumulator hands out its cotangent reference exactly once.
        let destination = context.input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        context
            .reference_accumulators
            .insert(root, CotangentReferenceAccumulator::Allocated { reference: destination.clone() });
        assert!(context.cotangent_accumulator_reference(root).is_some());
        let cotangent = context.take_reference_cotangent(&driver, 0).unwrap().unwrap();
        assert_eq!(cotangent.atom_id(), destination.atom_id());
        assert!(!context.cotangent_accumulator_reference(root).is_some());
        assert!(context.take_reference_cotangent(&driver, 0).unwrap().is_none());
    }

    #[test]
    fn test_pullback_linear_program() {
        let (_, pullback) = differentiate_at(Array::scalar(1.0)).vjp(|x| Ok(x.clone() * x)).unwrap();
        assert_eq!(pullback.linear_program().input_types().len(), 1 + pullback.residuals().len());
        assert_eq!(pullback.linear_program().output_types(), vec![ArrayType::scalar(DataType::F64)]);
    }

    #[test]
    fn test_pullback_transposed_program() {
        let (_, pullback) = differentiate_at(Array::scalar(2.0)).vjp(|x| Ok(x.clone() * x)).unwrap();
        let transposed = pullback.transposed_program(&[]).unwrap();
        let mut inputs = vec![Array::scalar(3.0)];
        inputs.extend_from_slice(pullback.residuals());
        assert_eq!(transposed.interpret(inputs), Ok(vec![Array::scalar(12.0)]));
        assert!(Arc::ptr_eq(&transposed, &pullback.transposed_program(&[]).unwrap()));
    }

    #[test]
    fn test_pullback_transposed_program_omits_unrequested_matrix_gradient() {
        let left = Array::matrix(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let right = Array::matrix(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let (_, pullback) = differentiate_at((left, right))
            .vjp(|(left, right)| Ok(left.dot(&right, &DotDimensionNumbers::matmul())))
            .unwrap();
        let both = pullback.transposed_program(&[CotangentDestinationKind::Return; 2]).unwrap();
        let selected = pullback
            .transposed_program(&[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore])
            .unwrap();
        // The caller's request reaches the production linearization and reverse sweep. The ignored right gradient
        // removes one whole dot product, rather than merely discarding its result after replay.
        assert_eq!(both.instructions().iter().filter(|instruction| instruction.operation().name() == "dot").count(), 2);
        assert_eq!(
            selected.instructions().iter().filter(|instruction| instruction.operation().name() == "dot").count(),
            1
        );
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(Array::matrix(2, 2, vec![1.0, -2.0, 0.5, 3.0])),
                (CotangentDestination::Return, CotangentDestination::Ignore),
            ),
            Ok((Some(Array::matrix(2, 3, vec![-9.0, -11.0, -13.0, 27.5, 34.5, 41.5])), None))
        );
    }

    #[test]
    fn test_pullback_residuals() {
        // Identity needs no saved primal values to evaluate its pullback.
        let (_, pullback) = differentiate_at(Array::scalar(2.0)).vjp(Ok).unwrap();
        assert_eq!(pullback.residuals(), &[]);
    }

    #[test]
    fn test_pullback_into_linear_parts() {
        // At x = 2, the retained pushforward maps a tangent to four times its value.
        let (_, pullback) = differentiate_at(Array::scalar(2.0)).vjp(|x| Ok(x.clone() * x)).unwrap();
        let (linear, residuals) = pullback.into_linear_parts();
        let mut inputs = vec![Array::scalar(3.0)];
        inputs.extend(residuals);
        assert_eq!(linear.interpret(inputs), Ok(vec![Array::scalar(12.0)]));
    }

    #[test]
    fn test_pullback_into_transposed_parts() {
        let (_, pullback) = differentiate_at(Array::scalar(2.0)).vjp(|x| Ok(x.clone() * x)).unwrap();
        let (transposed, residuals) = pullback.into_transposed_parts().unwrap();
        assert_eq!(transposed.input_types().len(), 1 + residuals.len());
        let mut inputs = vec![Array::scalar(3.0)];
        inputs.extend(residuals);
        assert_eq!(transposed.interpret(inputs), Ok(vec![Array::scalar(12.0)]));
    }

    #[test]
    fn test_pullback_apply() {
        let function = |(left, right): (
            LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>,
            LinearizationTracer<EagerContext<Array, ArrayOperation<Array>>>,
        )| Ok(left * right);
        let (_, pullback) = differentiate_at((Array::scalar(3.0), Array::scalar(2.0))).vjp(function).unwrap();
        assert_eq!(pullback.apply(Array::scalar(4.0)), Ok((Array::scalar(8.0), Array::scalar(12.0))),);
    }

    #[test]
    fn test_pullback_apply_with_destinations() {
        // `f(r, x) = { add_update(r, x); read(r) }` over a live reference. Under an `Ignore` destination the pullback
        // accumulates through an internal cotangent reference and returns `x̄ = ȳ` at the `Return` leaf only.
        fn function<V: ReferenceAddUpdate + ReferenceRead>((reference, x): (V, V)) -> Result<V, ProgramError> {
            reference.add_update(&x)?;
            reference.read()
        }
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let (value, pullback) =
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Array(Array::scalar(3.0_f32))))
                .vjp(function)
                .unwrap();
        assert_eq!(value, reference_test_scalar(4.0));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Ok((None, Some(reference_test_scalar(2.0)))),
        );
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Ignore, CotangentDestination::Ignore),
            ),
            Ok((None, None)),
        );

        // `apply` requires a reference-free boundary.
        assert!(matches!(
            pullback.apply(reference_test_scalar(2.0)),
            Err(ProgramError::InvalidArgument { message })
                if message == "returning pullback cotangent values requires a reference-free boundary but a primal input leaf has \
                    reference type ref<f32[]>; use `Pullback::apply_with_destinations` to supply cotangent seeds \
                    and destinations",
        ));

        // The seed and destination matrix is validated leaf for leaf.
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::NoCotangent,
                (CotangentDestination::Ignore, CotangentDestination::Return),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "pullback cotangent seed 0 omits the cotangent of a primal output of type f32[]; use \
                    `CotangentSeed::Value` for non-reference outputs",
        ));
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Return, CotangentDestination::Return),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "pullback destination 0 returns the cotangent of a primal input of reference type \
                    ref<f32[]> as a value; use `CotangentDestination::Reference` or `CotangentDestination::Ignore` for \
                    reference inputs",
        ));
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Ignore, CotangentDestination::Reference(reference_test_scalar(0.0))),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "pullback destination 1 has type f32[] but its primal input of type f32[] \
                    requires a reference storing the cotangent type f32[]",
        ));
        let mismatched = ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![0.0_f32, 0.0])));
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Reference(mismatched), CotangentDestination::Return),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "pullback destination 0 has type ref<f32[2]> but its primal input of type ref<f32[]> \
                    requires the cotangent reference type ref<f32[]>",
        ));

        // A destination must not alias the primal reference, even after the primal advanced generations, because
        // allocation identity rather than generation is compared.
        reference.write(Array::scalar(9.0_f32)).unwrap();
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(reference.clone())),
                    CotangentDestination::Return,
                ),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "cotangent 0 aliases a reference bound at the primal boundary of the \
                    differentiated function",
        ));

        // A destination must not alias a captured reference either, and two destinations must not alias each other.
        let captured = ArrayReference::new(Array::scalar(1.0_f32));
        let (_, pullback) = differentiate_at((
            ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32))),
            ArrayIrValue::Array(Array::scalar(3.0_f32)),
        ))
        .with_captures(ArrayIrValue::Reference(captured.clone()))
        .vjp(|input, _| function(input))
        .unwrap();
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (CotangentDestination::Reference(ArrayIrValue::Reference(captured)), CotangentDestination::Return),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "cotangent 0 aliases a reference bound at the primal boundary of the \
                    differentiated function",
        ));
        let shared = ArrayReference::new(Array::scalar(0.0_f32));
        let (_, pullback) = differentiate_at((
            ArrayIrValue::Reference(ArrayReference::new(Array::scalar(1.0_f32))),
            ArrayIrValue::Reference(ArrayReference::new(Array::scalar(2.0_f32))),
        ))
        .vjp(|(first, second): (ReferenceTestTracer, ReferenceTestTracer)| {
            second.add_update(&first.read()?)?;
            second.read()
        })
        .unwrap();
        assert!(matches!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(shared.clone())),
                    CotangentDestination::Reference(ArrayIrValue::Reference(shared)),
                ),
            ),
            Err(ProgramError::InvalidArgument { message })
                if message == "cotangent 1 and cotangent 0 bind the same reference allocation",
        ));
    }

    #[test]
    fn test_pullback_apply_with_destinations_reference_free_family() {
        let (_, pullback) = differentiate_at(Array::scalar(3.0_f32)).vjp(|value| Ok(value.clone() * value)).unwrap();
        assert_eq!(
            pullback
                .apply_with_destinations(CotangentSeed::Value(Array::scalar(2.0_f32)), CotangentDestination::Return,),
            Ok(Some(Array::scalar(12.0_f32)))
        );
        assert_eq!(
            pullback
                .apply_with_destinations(CotangentSeed::Value(Array::scalar(2.0_f32)), CotangentDestination::Ignore,),
            Ok(None)
        );
    }

    #[test]
    fn test_pullback_apply_with_destinations_staged_reference_free_family() {
        // Returned and ignored cotangents never request reference accumulation, including while staging.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (_, pullback) = differentiate_at(inputs[0].clone()).vjp(|value| Ok(value.clone() * value))?;
                let returned = pullback
                    .apply_with_destinations(CotangentSeed::Value(inputs[1].clone()), CotangentDestination::Return)?;
                let ignored = pullback
                    .apply_with_destinations(CotangentSeed::Value(inputs[1].clone()), CotangentDestination::Ignore)?;
                assert!(ignored.is_none());
                Ok(vec![returned.unwrap()])
            },
            vec![ArrayType::scalar(DataType::F32), ArrayType::scalar(DataType::F32)],
        )
        .unwrap();
        assert_eq!(
            program.interpret(vec![Array::scalar(3.0_f32), Array::scalar(2.0_f32)]),
            Ok(vec![Array::scalar(12.0_f32)]),
        );
    }

    #[test]
    fn test_pullback_apply_with_destinations_non_reference_input_reference() {
        let (_, pullback) = differentiate_at(reference_test_scalar(3.0))
            .vjp(|value| {
                Ok(value
                    .context()
                    .bind(
                        ArrayOperation::from(MulOperation::<ArrayType>::new()),
                        Vec::new(),
                        &[value.clone(), value.clone()],
                    )?
                    .remove(0))
            })
            .unwrap();
        let destination = ArrayReference::new(Array::scalar(5.0_f32));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                CotangentDestination::Reference(ArrayIrValue::Reference(destination.clone())),
            ),
            Ok(None)
        );
        assert_eq!(destination.read(), Ok(Array::scalar(17.0_f32)));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(1.0)),
                CotangentDestination::Reference(ArrayIrValue::Reference(destination.clone())),
            ),
            Ok(None)
        );
        assert_eq!(destination.read(), Ok(Array::scalar(23.0_f32)));
    }

    #[test]
    fn test_pullback_apply_with_destinations_non_reference_input_reference_batching() {
        // Each batch member adds its own 2x contribution to pre-populated storage without resetting another member.
        let destination = ArrayReference::new(Array::vector(vec![5.0_f32, 7.0]));
        let result = batch(
            |(value, destination)| {
                let seed = value.context().lift(reference_test_scalar(1.0))?;
                let (_, pullback) = differentiate_at(value).vjp(reference_test_square)?;
                pullback.apply_with_destinations(
                    CotangentSeed::Value(seed),
                    CotangentDestination::Reference(destination.clone()),
                )?;
                destination.read()
            },
            (ArrayIrValue::Array(Array::vector(vec![3.0_f32, 5.0])), ArrayIrValue::Reference(destination.clone())),
            (BatchAxis::new(0), BatchAxis::new(0)),
            BatchAxis::new(0),
            None,
        );
        assert_eq!(result, Ok(ArrayIrValue::Array(Array::vector(vec![11.0_f32, 17.0]))));
        assert_eq!(destination.read(), Ok(Array::vector(vec![11.0_f32, 17.0])));
    }

    #[test]
    fn test_pullback_apply_with_destinations_non_reference_input_reference_higher_order() {
        // The inner pullback adds 2x to an independently allocated buffer initialized to 5. Differentiating its
        // contents checks that accumulation preserves the derivative while the nonzero initial contents stay constant.
        let result = differentiate_at(reference_test_scalar(3.0)).value_and_gradient(|value: ReferenceTestTracer| {
            let initial = value.context().constant(reference_test_scalar(5.0))?;
            let destination = initial.reference_new()?;
            let seed = value.context().constant(reference_test_scalar(1.0))?;
            let (_, pullback) = differentiate_at(value).vjp(reference_test_square)?;
            pullback.apply_with_destinations(
                CotangentSeed::Value(seed),
                CotangentDestination::Reference(destination.clone()),
            )?;
            destination.read()
        });
        assert_eq!(result, Ok((reference_test_scalar(11.0), reference_test_scalar(2.0))));
    }

    #[test]
    fn test_pullback_apply_with_destinations_staged_references() {
        let reference_type = ArrayIrType::Reference(ReferenceType::new(ArrayType::scalar(DataType::F32)));
        let (_, program) = EagerContext::<ArrayIrValue<Array>, ArrayIrOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (primal, pullback) =
                    differentiate_at((inputs[0].clone(), inputs[2].clone())).vjp(|(reference, value)| {
                        let stored = reference.read()?;
                        let operation = ArrayOperation::from(AddOperation::<ArrayType>::new());
                        Ok(value.context().bind(operation, Vec::new(), &[stored, value.clone()])?.remove(0))
                    })?;
                assert!(matches!(pullback.apply_with_destinations(
                    CotangentSeed::Value(inputs[2].clone()),
                    (CotangentDestination::Reference(inputs[0].clone()), CotangentDestination::Return),
                ), Err(ProgramError::InvalidArgument { message })
                    if message == "cotangent 0 aliases a reference bound at the primal boundary of the \
                        differentiated function"));
                let (_, gradient) = pullback.apply_with_destinations(
                    CotangentSeed::Value(inputs[2].clone()),
                    (CotangentDestination::Reference(inputs[1].clone()), CotangentDestination::Return),
                )?;
                Ok(vec![primal, gradient.unwrap()])
            },
            vec![reference_type.clone(), reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32))],
        )
        .unwrap();
        let destination = ArrayReference::new(Array::scalar(5.0_f32));
        assert_eq!(
            program.interpret(vec![
                ArrayIrValue::Reference(ArrayReference::new(Array::scalar(3.0_f32))),
                ArrayIrValue::Reference(destination.clone()),
                reference_test_scalar(2.0),
            ]),
            Ok(vec![reference_test_scalar(5.0), reference_test_scalar(2.0)])
        );
        assert_eq!(destination.read(), Ok(Array::scalar(7.0_f32)));
    }

    #[test]
    fn test_pullback_apply_with_destinations_reference_destination() {
        // `f(r, x) = { write(r, x); read(r) }` under a `Reference` destination whose initial contents are a nonzero
        // post-state cotangent: the read accumulates `ȳ` into the destination, and the write hands the accumulated
        // `5 + ȳ` to `x̄` and leaves the destination holding the zero pre-state cotangent.
        fn write_then_read<V: ReferenceWrite + ReferenceRead>((reference, x): (V, V)) -> Result<V, ProgramError> {
            reference.write(&x)?;
            reference.read()
        }
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let (value, pullback) =
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Array(Array::scalar(3.0_f32))))
                .vjp(write_then_read)
                .unwrap();
        assert_eq!(value, reference_test_scalar(3.0));
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
        let destination = ArrayReference::new(Array::scalar(5.0_f32));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(destination.clone())),
                    CotangentDestination::Return,
                ),
            ),
            Ok((None, Some(reference_test_scalar(7.0)))),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(0.0_f32)));

        // `y = swap(r, x)` swaps `ȳ` into the destination and hands the previous post-state cotangent to `x̄`.
        fn swap<V: ReferenceSwap<V, V>>((reference, x): (V, V)) -> Result<V, ProgramError> {
            reference.swap(&x)
        }
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let (value, pullback) =
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Array(Array::scalar(3.0_f32))))
                .vjp(swap)
                .unwrap();
        assert_eq!(value, reference_test_scalar(1.0));
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
        let destination = ArrayReference::new(Array::scalar(5.0_f32));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(2.0)),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(destination.clone())),
                    CotangentDestination::Return,
                ),
            ),
            Ok((None, Some(reference_test_scalar(5.0)))),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(2.0_f32)));

        // Applications under the same structural mask share one retained transposition even when they supply different
        // destination references, while the `Ignore` variant at the reference leaf is a distinct retained program.
        let other = ArrayReference::new(Array::scalar(1.0_f32));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(4.0)),
                (CotangentDestination::Reference(ArrayIrValue::Reference(other.clone())), CotangentDestination::Return),
            ),
            Ok((None, Some(reference_test_scalar(1.0)))),
        );
        assert_eq!(other.read(), Ok(Array::scalar(4.0_f32)));
        let referenced = pullback
            .transposed_program(&[CotangentDestinationKind::Reference, CotangentDestinationKind::Return])
            .unwrap();
        assert!(Arc::ptr_eq(
            &pullback
                .transposed_program(&[CotangentDestinationKind::Reference, CotangentDestinationKind::Return])
                .unwrap(),
            &referenced,
        ));
        let ignored = pullback
            .transposed_program(&[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert!(!Arc::ptr_eq(&referenced, &ignored));
        assert_eq!(referenced.input_types().len(), ignored.input_types().len() + 1);
    }

    #[test]
    fn test_pullback_apply_with_destinations_consumed_reference_inputs() {
        // Consumption ends the primal lifetime, but its initial contents still contribute to the scalar result.
        // Replay writes the initial-state cotangent into a separate destination that remains live for extraction.
        let reference = reference_test_scalar(3.0).reference_new().unwrap();
        let (value, pullback) = differentiate_at(reference)
            .vjp(|reference: ReferenceTestTracer| {
                let value = reference.freeze()?;
                Ok::<_, ProgramError>(
                    value
                        .context()
                        .bind(
                            ArrayOperation::from(MulOperation::<ArrayType>::new()),
                            Vec::new(),
                            &[value.clone(), value.clone()],
                        )?
                        .remove(0),
                )
            })
            .unwrap();
        let destination = reference_test_scalar(0.0).reference_new().unwrap();
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(reference_test_scalar(1.0)),
                CotangentDestination::Reference(destination.clone()),
            ),
            Ok(None),
        );
        assert_eq!(value, reference_test_scalar(9.0));
        assert_eq!(destination.freeze(), Ok(reference_test_scalar(6.0)));
    }

    #[test]
    fn test_pullback_apply_with_destinations_scan_reference_carry() {
        // `f(r, xs) = scan { add_update(r, x_i); y_i = read(r) }` accumulates the scanned elements into the carried
        // reference and reports the running state: `y_i = r + Σ_{j ≤ i} x_j`. Its pure equivalent is the prefix sum of
        // `xs` shifted by the initial state, whose gradient under unit output cotangents is `x̄_j = length - j` and
        // `r̄ = length`.
        let body = {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let carry = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let element = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![carry, element], None)
                .unwrap();
            let current =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                    vec![carry, current],
                    vec![Placeholder; 2],
                    vec![Placeholder; 2],
                )
                .unwrap()
        };
        let function = move |(reference, elements): (ReferenceTestTracer, ReferenceTestTracer)| {
            let context = reference.context().clone();
            let mut outputs = context.bind(
                ScanOperation::<ReferenceTestValue>::new(1, 3),
                vec![body.clone()],
                &[reference, elements],
            )?;
            Ok(outputs.remove(1))
        };
        let reference = ArrayReference::new(Array::scalar(1.0_f32));
        let (value, pullback) = differentiate_at((
            ArrayIrValue::Reference(reference.clone()),
            ArrayIrValue::Array(Array::vector(vec![1.0_f32, 2.0, 3.0])),
        ))
        .vjp(function)
        .unwrap();
        assert_eq!(value, ArrayIrValue::Array(Array::vector(vec![2.0_f32, 4.0, 7.0])));
        assert_eq!(reference.read(), Ok(Array::scalar(7.0_f32)));

        // The destination starts at the zero post-state cotangent and ends holding the cotangent of the initial state.
        let destination = ArrayReference::new(Array::scalar(0.0_f32));
        assert_eq!(
            pullback.apply_with_destinations(
                CotangentSeed::Value(ArrayIrValue::Array(Array::vector(vec![1.0_f32, 1.0, 1.0]))),
                (
                    CotangentDestination::Reference(ArrayIrValue::Reference(destination.clone())),
                    CotangentDestination::Return,
                ),
            ),
            Ok((None, Some(ArrayIrValue::Array(Array::vector(vec![3.0_f32, 2.0, 1.0]))))),
        );
        assert_eq!(destination.read(), Ok(Array::scalar(3.0_f32)));

        // The pure equivalent (the discharged program's dataflow, summed to a scalar) agrees with the reference
        // pullback, and finite differences confirm its gradient.
        assert_eq!(
            differentiate_at(Array::vector(vec![1.0, 2.0, 3.0]))
                .gradient(|xs| Ok(xs.cumulative_sum(0)?.reduce(&[0], ReductionKind::Sum)))
                .unwrap(),
            Array::vector(vec![3.0, 2.0, 1.0]),
        );
        check_gradient!(
            |xs| Ok(xs.cumulative_sum(0)?.reduce(&[0], ReductionKind::Sum)),
            at = Array::vector(vec![1.0, 2.0, 3.0]),
            step = 1e-3,
            tolerance = 1e-6,
        );
    }

    #[test]
    fn test_recursive_transposition_driver_transpose_program() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let first = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_input(ArrayType::scalar(DataType::F64));
        let last = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![first], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output, last], vec![Placeholder; 3], vec![Placeholder; 2])
            .unwrap();
        let region = program.entry_region_ref();
        let driver = RecursiveTranspositionDriver::new(region, 0).unwrap();

        // Sparse selections preserve the requested order and share the region's cached specialization.
        let transposed = driver.transpose_program(region, &[2, 0], &[]).unwrap();
        assert_eq!(transposed.output_ids(), &[transposed.input_ids()[1], transposed.input_ids()[0]]);
        assert!(Arc::ptr_eq(&transposed, &region.transpose_shared(&[2, 0], &[], &[]).unwrap()));

        // Destination kinds follow the selection order, rather than the original input positions.
        let ignored = driver
            .transpose_program(region, &[2, 0], &[CotangentDestinationKind::Return, CotangentDestinationKind::Ignore])
            .unwrap();
        assert_eq!(ignored.output_ids(), &[ignored.input_ids()[1]]);
        assert!(driver.transpose_program(region, &[], &[]).unwrap().output_ids().is_empty());
        assert!(matches!(
            driver.transpose_program(region, &[3], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 3 is out of range for a program with 3 input(s)",
        ));
        assert!(matches!(
            driver.transpose_program(region, &[0, 0], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 0 appears more than once",
        ));
        assert!(matches!(
            driver.transpose_program(region, &[2, 0], &[CotangentDestinationKind::Return]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition received 1 destination kinds for 2 selected inputs",
        ));
    }

    #[test]
    fn test_region_transpose() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let region = program.entry_region_ref();

        let transposed = region.transpose(&[0], &[], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![ArrayType::scalar(DataType::F64)]);
        assert_eq!(transposed.output_ids(), transposed.input_ids());
        assert!(transposed.instructions().is_empty());

        // An empty selection requests no cotangents, rather than selecting every input.
        let transposed = region.transpose(&[], &[], &[]).unwrap();
        assert!(transposed.output_ids().is_empty());
        assert!(transposed.instructions().is_empty());
    }

    #[test]
    fn test_region_transpose_rejects_invalid_arguments() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let region = program.entry_region_ref();

        assert!(matches!(
            region.transpose(&[1], &[], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 1 is out of range for a program with 1 input(s)",
        ));
        assert!(matches!(
            region.transpose(&[0, 0], &[], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 0 appears more than once",
        ));
        assert!(matches!(
            region.transpose(&[0], &[Vec::new(), Vec::new()], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition received 2 zero-residual mappings for 1 selected inputs",
        ));
    }

    #[test]
    fn test_region_transpose_retained_named_rule() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let coefficient = builder.add_input(ArrayType::scalar(DataType::F64));
        let tangent = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestLinearOperation::ScaleLinear, Vec::new(), vec![coefficient, tangent], None)
            .unwrap()[0];
        let program = builder.build::<Vec<Array>, Array>(vec![output], vec![Placeholder; 2], Placeholder).unwrap();
        let cloned = program.clone();
        drop(program);
        let mut imported_builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let imported = imported_builder.import_program(cloned);
        let region = imported_builder.region_ref(imported).unwrap();

        // The originating builder and program are gone. The imported payload still dispatches its named rule with
        // the coefficient as a known runtime input; no callback lifetime or stored backward region is involved.
        let returned = region.transpose(&[1], &[], &[CotangentDestinationKind::Return]).unwrap();
        assert_eq!(returned.input_types(), vec![ArrayType::scalar(DataType::F64); 2]);
        assert_eq!(returned.instructions().len(), 1);
        assert!(matches!(returned.instructions()[0].operation(), TestLinearOperation::ScaleLinear));
        assert_eq!(returned.instructions()[0].inputs(), &[AtomId::new(1), AtomId::new(0)]);
        assert_eq!(returned.output_ids(), &[AtomId::new(2)]);

        // Demand is chosen only after importing the same source rule. Ignoring its linear operand emits no rule work
        // and does not manufacture a scratch reference. This fixture does not model nonlinear VJP registration.
        let ignored = region.transpose(&[1], &[], &[CotangentDestinationKind::Ignore]).unwrap();
        assert!(ignored.instructions().is_empty());
        assert!(ignored.output_ids().is_empty());
    }

    #[test]
    fn test_region_transpose_shared() {
        // A shared linear callee is transposed once per argument list and reused by every copy of its sealed region,
        // which is what removes the repeated re-transposition that outer programs interning one callee would pay.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Add, Vec::new(), vec![left, right], None).unwrap()[0];
        let callee = Arc::new(
            builder
                .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
                .unwrap(),
        );

        // Every argument of the shared entry point takes part in the key, so a different selection of linear inputs
        // produces its own transposed program rather than reusing the first one.
        let both = callee.entry_region_ref().transpose_shared(&[0, 1], &[], &[]).unwrap();
        let reversed = callee.entry_region_ref().transpose_shared(&[1, 0], &[], &[]).unwrap();
        let leading = callee.entry_region_ref().transpose_shared(&[0], &[], &[]).unwrap();
        assert!(!Arc::ptr_eq(&both, &reversed));
        assert!(!Arc::ptr_eq(&both, &leading));
        assert_eq!(both.to_string(), callee.transpose_with_respect_to(&[0, 1], &[]).unwrap().to_string());
        assert_eq!(leading.to_string(), callee.transpose_with_respect_to(&[0], &[]).unwrap().to_string());

        // The same argument list reuses one artifact, including through an independently built program that interned
        // the callee, because importing a region carries its retained transforms along with its contents.
        assert!(Arc::ptr_eq(&callee.entry_region_ref().transpose_shared(&[0, 1], &[], &[]).unwrap(), &both));
        let mut outer_builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let interned = outer_builder.intern_callee(&callee, None).unwrap();
        let interned = RegionRef::new(&outer_builder.regions, interned).unwrap();
        assert!(Arc::ptr_eq(&interned.transpose_shared(&[0, 1], &[], &[]).unwrap(), &both));

        // The zero-residual mappings change the transposed program too, so they separate entries as well.
        let residualized = interned.transpose_shared(&[0, 1], &[Vec::new(), Vec::new()], &[]).unwrap();
        assert!(!Arc::ptr_eq(&residualized, &both));
        assert_eq!(residualized.to_string(), both.to_string());
    }

    #[test]
    fn test_region_transpose_shared_rejects_invalid_arguments() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let region = program.entry_region_ref();

        assert!(matches!(
            region.transpose_shared(&[1], &[], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 1 is out of range for a program with 1 input(s)",
        ));
        assert!(matches!(
            region.transpose_shared(&[0, 0], &[], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 0 appears more than once",
        ));
        assert!(matches!(
            region.transpose_shared(&[0], &[Vec::new(), Vec::new()], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition received 2 zero-residual mappings for 1 selected inputs",
        ));
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_region_transpose_shared_debug_recheck_detects_corrupted_cached_program() {
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Add, Vec::new(), vec![left, right], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // Publish an artifact that disagrees with what transposing this region produces, which is exactly the state a
        // nondeterministic `transpose` rule would leave behind: a retained pullback of a linear map the region does
        // not stage.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let unrelated = Arc::new(
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap(),
        );
        program.entry_region_ref().insert_transform_artifact_for_testing::<TranspositionTransform, _>(
            TranspositionTransformArguments {
                input_indices: vec![0, 1],
                zero_residual_input_indices: Vec::new(),
                destination_kinds: vec![CotangentDestinationKind::Return; 2],
            },
            TransformArtifact::new(vec![unrelated], ()),
        );

        // The recheck runs on the hit and reports the contract violation rather than serving the wrong pullback.
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program.entry_region_ref().transpose_shared(&[0, 1], &[], &[])
        }))
        .unwrap_err();
        let message = panicked.downcast_ref::<String>().unwrap();
        assert!(message.starts_with("nondeterministic transform rule detected for `"), "{message}",);
        assert!(message.contains("TranspositionTransform"), "{message}");
    }

    #[test]
    fn test_region_transpose_shared_specializes_by_destination_kinds() {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let region = program.entry_region_ref();

        // The default mask and its explicit spelling share one retained artifact, while the `Ignore` variant at the
        // reference leaf is a distinct artifact with a different boundary.
        let default = region.transpose_shared(&[0, 1], &[], &[]).unwrap();
        let explicit = region
            .transpose_shared(&[0, 1], &[], &[CotangentDestinationKind::Reference, CotangentDestinationKind::Return])
            .unwrap();
        let ignored = region
            .transpose_shared(&[0, 1], &[], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert!(Arc::ptr_eq(&default, &explicit));
        assert!(!Arc::ptr_eq(&default, &ignored));
        assert_eq!(default.input_types().len(), 2);
        assert_eq!(ignored.input_types().len(), 1);
        assert!(Arc::ptr_eq(
            &region
                .transpose_shared(&[0, 1], &[], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return],)
                .unwrap(),
            &ignored,
        ));
    }

    #[test]
    fn test_program_transpose() {
        // Test that transposing an identity instruction forwards the output cotangent straight to the input.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_cotangent_representation() {
        // A primal representation may use a different cotangent representation. E8M0 cannot represent zero or
        // negative values, so both pullback boundaries use F32 while the source program remains E8M0-typed.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F8E8M0FNU));
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_types(), vec![ArrayType::scalar(DataType::F32)]);
        assert_eq!(transposed.output_types(), vec![ArrayType::scalar(DataType::F32)]);
        assert!(transposed.instructions().is_empty());
    }

    #[test]
    fn test_program_transpose_zero_space() {
        // Zero-space cotangent boundary leaves preserve the primal leaf structure, but can never carry live adjoints.
        // Transposing identity programs over non-differentiable types therefore returns the zero-space value.
        for data_type in
            [ArrayType::scalar(DataType::Token), ArrayType::scalar(DataType::Boolean), ArrayType::scalar(DataType::I32)]
        {
            let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
            let input = builder.add_input(data_type);
            let program = builder.build::<Array, Array>(vec![input], Placeholder, Placeholder).unwrap();
            let transposed = program.transpose().unwrap();
            assert_eq!(transposed.input_types(), vec![ArrayType::scalar(DataType::Zero)]);
            assert_eq!(transposed.output_types(), vec![ArrayType::scalar(DataType::Zero)]);
            let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
            assert_eq!(transposed.interpret(zero.clone()), Ok(zero));
        }
    }

    #[test]
    fn test_program_transpose_repeated_operands() {
        // Test that repeated uses of one input accumulate their cotangent contributions through a staged `add`.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output =
            builder.add_instruction(TestLinearOperation::Add, Vec::new(), vec![input, input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(matches!(transposed.instructions()[0].operation(), TestLinearOperation::Add));
        assert_eq!(transposed.instructions()[0].inputs(), &[AtomId::new(0), AtomId::new(0)]);
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = add %0 %0
                in (%1)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_unused_outputs() {
        // Test that unused instruction outputs are passed to transpose rules as structural zero cotangents (the
        // `TwoOutputs` rule asserts that its second output cotangent is a structural zero).
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let outputs = builder
            .add_instruction(TestLinearOperation::TwoOutputs, Vec::new(), vec![input], None)
            .unwrap()
            .to_vec();
        let program = builder.build::<Array, Array>(vec![outputs[0]], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(outputs, &[AtomId::new(1), AtomId::new(2)]);
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert!(transposed.instructions().is_empty());
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_disconnected_input() {
        // Test that a disconnected primal input's cotangent is emitted as an input-free `ZeroOperation` instruction,
        // which is materialized at interpretation time rather than at transpose time.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder.build::<Array, ()>(Vec::new(), Placeholder, ()).unwrap();
        let transposed = program.transpose().unwrap();
        assert!(transposed.input_ids().is_empty());
        assert_eq!(transposed.output_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(0)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &ArrayType::scalar(DataType::F64),
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda  .
                let %0:f64[] = zero [type=f64[]]
                in (%0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_skips_dead_rules() {
        // Test that instructions whose outputs carry no adjoint are skipped in the reverse walk, with the dead input
        // still receiving a zero cotangent output.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let dead_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let live_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let dead_output = builder
            .add_instruction(TestLinearOperation::ForeignContribution, Vec::new(), vec![dead_input], None)
            .unwrap()[0];
        let output =
            builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![live_input], None).unwrap()[0];
        let program = builder
            .build::<(Array, Array), Array>(vec![output], (Placeholder, Placeholder), Placeholder)
            .unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(dead_output, AtomId::new(2));
        assert_eq!(transposed.input_ids(), &[AtomId::new(0)]);
        assert_eq!(transposed.output_ids(), &[AtomId::new(1), AtomId::new(0)]);
        assert_eq!(transposed.instructions().len(), 1);
        assert!(transposed.instructions()[0].inputs().is_empty());
        assert_eq!(transposed.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &ArrayType::scalar(DataType::F64),
        ));
        assert_eq!(
            transposed.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero [type=f64[]]
                in (%1, %0)
            "}
            .trim_end(),
        );
    }

    #[test]
    fn test_program_transpose_preserves_outer_trace() {
        // Test that transposing a program whose values are tracers of an outer trace stays self-contained: the
        // disconnected input's zero is emitted as an instruction in the pullback and nothing is staged into the
        // outer tracing context.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let outer_builder = tracing_context.builder().clone();
        let mut builder = ProgramBuilder::<TestTracingValue, TestLinearOperation>::new();
        let connected_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let disconnected_input = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder
            .build::<Vec<TestTracingValue>, TestTracingValue>(
                vec![connected_input],
                vec![Placeholder, Placeholder],
                Placeholder,
            )
            .unwrap();
        let pullback = program.transpose().unwrap();
        assert_eq!(disconnected_input, AtomId::new(1));
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(0), AtomId::new(1)]);
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &ArrayType::scalar(DataType::F64),
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero [type=f64[]]
                in (%0, %1)
            "}
            .trim_end(),
        );
        assert!(outer_builder.borrow().atoms().is_empty());
        assert!(outer_builder.borrow().instructions().is_empty());
    }

    #[test]
    fn test_program_transpose_staged_zero_contribution() {
        // Test that a transpose-rule-staged structural zero contribution stays an input-free `ZeroOperation`
        // instruction in the pullback, again leaving the outer tracing context untouched.
        let tracing_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let outer_builder = tracing_context.builder().clone();
        let mut builder = ProgramBuilder::<TestTracingValue, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestLinearOperation::StagedZeroContribution, Vec::new(), vec![input], None)
            .unwrap()[0];
        let program =
            builder.build::<TestTracingValue, TestTracingValue>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose().unwrap();
        assert_eq!(pullback.input_ids(), &[AtomId::new(0)]);
        assert_eq!(pullback.output_ids(), &[AtomId::new(1)]);
        assert_eq!(pullback.instructions().len(), 1);
        assert!(pullback.instructions()[0].inputs().is_empty());
        assert_eq!(pullback.instructions()[0].outputs(), &[AtomId::new(1)]);
        assert!(matches!(
            pullback.instructions()[0].operation(),
            TestLinearOperation::Zero(zero) if zero.r#type() == &ArrayType::scalar(DataType::F64),
        ));
        assert_eq!(
            pullback.to_string(),
            indoc! {"
                lambda %0:f64[] .
                let %1:f64[] = zero [type=f64[]]
                in (%1)
            "}
            .trim_end(),
        );
        assert!(outer_builder.borrow().atoms().is_empty());
        assert!(outer_builder.borrow().instructions().is_empty());
    }

    #[test]
    fn test_program_transpose_omitted_contribution() {
        // A rule may omit a zero contribution entirely; the boundary still materializes the requested zero output.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output =
            builder.add_instruction(TestLinearOperation::NoContribution, Vec::new(), vec![input], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let transposed = program.transpose().unwrap();
        assert_eq!(transposed.instructions().len(), 1);
        assert!(matches!(transposed.instructions()[0].operation(), TestLinearOperation::Zero(_)));
    }

    #[test]
    fn test_program_transpose_rejects_foreign_contribution() {
        // Test that a cotangent contribution staged in a foreign builder is rejected before its atom ID can alias an
        // unrelated atom in the destination pullback.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F64));
        let output = builder
            .add_instruction(TestLinearOperation::ForeignContribution, Vec::new(), vec![input], None)
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.transpose(),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
    }

    #[test]
    fn test_program_transpose_preserves_and_fuses_source_provenance() {
        // `f(x) = x * 2 + x * 3` is linear in `x` and uses it twice, so transposing it stages one pullback `mul` per
        // source `mul` plus one accumulating `add`. Each staged `mul` records the source instruction it transposes,
        // while the accumulation intentionally merges contributions from two *different* source instructions and
        // therefore records the fused provenance of both.
        let first = Provenance::scope(ProvenanceScope::new("a"), Provenance::unknown());
        let second = Provenance::scope(ProvenanceScope::new("b"), Provenance::unknown());
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let x = builder.add_input(ArrayType::scalar(DataType::F64));
        let two = builder.add_constant(Array::scalar(2.0));
        let three = builder.add_constant(Array::scalar(3.0));
        let doubled =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![x, two], Some(first.clone())).unwrap()[0];
        let tripled = builder
            .add_instruction(MulOperation::new(), Vec::new(), vec![x, three], Some(second.clone()))
            .unwrap()[0];
        let output = builder
            .add_instruction(
                AddOperation::new(),
                Vec::new(),
                vec![doubled, tripled],
                Some(Provenance::scope(ProvenanceScope::new("c"), Provenance::unknown())),
            )
            .unwrap()[0];
        let program =
            builder.build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder], vec![Placeholder]).unwrap();

        // The reverse walk visits the source instructions in reverse program order, so the second `mul`'s contribution
        // is accumulated first and leads the fused origin list. The `add` transposes into pure cotangent routing and
        // therefore contributes no instruction of its own.
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            pullback
                .instructions()
                .iter()
                .map(|instruction| (instruction.operation().name(), instruction.provenance().clone()))
                .collect::<Vec<_>>(),
            vec![("mul", second.clone()), ("mul", first.clone()), ("add", Provenance::fused([second, first])),],
        );
        assert_eq!(pullback.interpret(vec![Array::scalar(1.0)]), Ok(vec![Array::scalar(5.0)]));
    }

    #[test]
    fn test_program_transpose_deep_known_chain_iteratively() {
        // This chain is intentionally much deeper than realistic scalar code. Materializing its tail exercises the
        // explicit postorder work stack and would make a recursive implementation consume one native stack frame per
        // producer. Keep the assertion structural so the test characterizes transformation behavior independently of
        // any interpretation backend.
        const CHAIN_LENGTH: usize = 10_000;

        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let mut known_intermediate = known;
        for _ in 0..CHAIN_LENGTH {
            known_intermediate = builder
                .add_instruction(TestLinearOperation::Identity, Vec::new(), vec![known_intermediate], None)
                .unwrap()[0];
        }
        let output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![known_intermediate, linear], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1], &[]).unwrap();
        assert_eq!(
            pullback
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), TestLinearOperation::Identity))
                .count(),
            CHAIN_LENGTH,
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to() {
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let left = builder.add_input(ArrayType::scalar(DataType::F64));
        let right = builder.add_input(ArrayType::scalar(DataType::F64));
        let program = builder
            .build::<(Array, Array), (Array, Array)>(
                vec![left, right],
                (Placeholder, Placeholder),
                (Placeholder, Placeholder),
            )
            .unwrap();

        // Distinct output seeds make selection order observable in both the program interface and its values.
        let forward = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        let reversed = program.transpose_with_respect_to(&[1, 0], &[]).unwrap();
        assert_eq!(forward.output_ids().len(), 2);
        assert_eq!(reversed.output_ids().len(), 2);
        assert_eq!(
            reversed.output_ids(),
            &[forward.output_ids()[1], forward.output_ids()[0]],
            "requested index order must permute the pullback outputs",
        );

        assert_eq!(
            reversed.interpret(vec![Array::scalar(3.0), Array::scalar(7.0)]),
            Ok(vec![Array::scalar(7.0), Array::scalar(3.0)]),
        );

        // Out-of-range and duplicate input indices are rejected.
        assert!(matches!(
            program.transpose_with_respect_to(&[2], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 2 is out of range for a program with 2 input(s)",
        ));
        assert!(matches!(
            program.transpose_with_respect_to(&[1, 1], &[]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition input index 1 appears more than once",
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to_replays_known_intermediate() {
        // A live transpose rule may need a pure value produced entirely from known inputs. For `f(a, x) = (a², a²x)`,
        // transposing only with respect to `x` must replay `a²` in the pullback, ignore the cotangent supplied for the
        // non-linear `a²` output, and produce `d_x = d_product · a²`.
        let mut builder = ProgramBuilder::<Array, ArrayOperation<Array>>::new();
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let known_square =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![known, known], None).unwrap()[0];
        let product =
            builder.add_instruction(MulOperation::new(), Vec::new(), vec![known_square, linear], None).unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![known_square, product],
                vec![Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1], &[]).unwrap();
        let outputs = pullback.interpret(vec![Array::scalar(100.0), Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        assert_eq!(outputs, vec![Array::scalar(18.0)]);
    }

    #[test]
    fn test_program_transpose_with_respect_to_shares_known_producer() {
        // Two live transpose rules that demand the same pure known intermediate must share one rematerialized producer.
        // The pullback contains one `identity`, not one copy per `add` consumer, and both linear inputs still receive
        // their corresponding output cotangents.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let first_linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let second_linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let known_intermediate =
            builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![known], None).unwrap()[0];
        let first_output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![known_intermediate, first_linear], None)
            .unwrap()[0];
        let second_output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![known_intermediate, second_linear], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_output, second_output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            pullback
                .instructions()
                .iter()
                .filter(|instruction| matches!(instruction.operation(), TestLinearOperation::Identity))
                .count(),
            1,
            "a shared pure known producer must be replayed exactly once",
        );
        assert_eq!(
            pullback.output_ids(),
            &pullback.input_ids()[..2],
            "each linear input must receive its corresponding output cotangent",
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_shares_attached_regions() {
        // Region-bearing known producers retain their attached closure when replayed, and two producers that attach
        // the same source region reuse one imported destination region rather than cloning equivalent closures.
        let mut region_builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region = region_builder.build::<Array, Array>(vec![region_input], Placeholder, Placeholder).unwrap();
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let shared_region = builder.import_region(region.entry_region_ref());
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let first_linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let second_linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let first_known = builder
            .add_instruction(TestLinearOperation::RegionIdentity, vec![shared_region], vec![known], None)
            .unwrap()[0];
        let second_known = builder
            .add_instruction(TestLinearOperation::RegionIdentity, vec![shared_region], vec![known], None)
            .unwrap()[0];
        let first_output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![first_known, first_linear], None)
            .unwrap()[0];
        let second_output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![second_known, second_linear], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Vec<Array>>(
                vec![first_output, second_output],
                vec![Placeholder, Placeholder, Placeholder],
                vec![Placeholder, Placeholder],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        let replayed_region_ids = pullback
            .instructions()
            .iter()
            .filter_map(|instruction| {
                matches!(instruction.operation(), TestLinearOperation::RegionIdentity).then(|| instruction.regions()[0])
            })
            .collect::<Vec<_>>();
        assert_eq!(replayed_region_ids.len(), 2);
        assert_eq!(replayed_region_ids[0], replayed_region_ids[1]);
        assert_eq!(pullback.regions().len(), 2, "the shared nested region must be imported only once");
    }

    #[test]
    fn test_program_transpose_with_respect_to_rejects_effectful_known_producer() {
        // Replaying a known producer with observable effects in the pullback could duplicate or reorder that effect,
        // so the partition-aware transpose must require partial evaluation to residualize the value instead.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let known_intermediate = builder
            .add_instruction(TestLinearOperation::EffectfulIdentity, Vec::new(), vec![known], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![known_intermediate, linear], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "partition-aware transpose cannot replay effectful known intermediate producer \
                    `effectful_identity`; partial-evaluate it into a residual input first",
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to_rejects_effectful_linear_instruction() {
        // An effectful instruction whose *output is linear* is rejected outright, even when its adjoint is dead:
        // skipping it would silently drop the effect from the pullback, and transposing it would replay the effect
        // in the reversed program.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let effectful = builder
            .add_instruction(TestLinearOperation::EffectfulIdentity, Vec::new(), vec![linear], None)
            .unwrap()[0];
        let program = builder.build::<Array, Array>(vec![effectful], Placeholder, Placeholder).unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message
                    == "partition-aware transpose cannot transpose effectful linear instruction \
                        `effectful_identity`; transposition cannot replay observable effects",
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to_drops_effectful_sink() {
        // An effectful *sink* over a linear operand (e.g., an I/O sink over a tangent) has no linear output and touches
        // no reference state, so it contributes no cotangent: the dead-edge skip drops it from the pullback instead of
        // the effect gate rejecting it, and the surrounding linear program stays transposable.
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        builder.add_instruction(TestLinearOperation::EffectfulSink, Vec::new(), vec![linear], None).unwrap();
        let output = builder.add_instruction(TestLinearOperation::Identity, Vec::new(), vec![linear], None).unwrap()[0];
        let program = builder.build::<Array, Array>(vec![output], Placeholder, Placeholder).unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert!(pullback.instructions().is_empty());
        assert_eq!(pullback.output_ids(), pullback.input_ids());
    }

    #[test]
    fn test_program_transpose_with_respect_to_rejects_nested_effectful_known_producer() {
        // EffectClasses nested inside a known producer's attached region are equally observable. The outer operation is
        // intrinsically pure, so this specifically verifies recursive effect accounting through the region closure.
        let mut region_builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let region_input = region_builder.add_input(ArrayType::scalar(DataType::F64));
        let region_output = region_builder
            .add_instruction(TestLinearOperation::EffectfulIdentity, Vec::new(), vec![region_input], None)
            .unwrap()[0];
        let region = region_builder.build::<Array, Array>(vec![region_output], Placeholder, Placeholder).unwrap();
        let mut builder = ProgramBuilder::<Array, TestLinearOperation>::new();
        let region = builder.import_region(region.entry_region_ref());
        let known = builder.add_input(ArrayType::scalar(DataType::F64));
        let linear = builder.add_input(ArrayType::scalar(DataType::F64));
        let known_intermediate = builder
            .add_instruction(TestLinearOperation::RegionIdentity, vec![region], vec![known], None)
            .unwrap()[0];
        let output = builder
            .add_instruction(TestLinearOperation::Add, Vec::new(), vec![known_intermediate, linear], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<Array>, Array>(vec![output], vec![Placeholder, Placeholder], Placeholder)
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[1], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "partition-aware transpose cannot replay effectful known intermediate producer \
                    `region_identity`; partial-evaluate it into a residual input first",
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to_non_reference_destinations() {
        // A repeated operand submits both contributions to the same buffer. Destination arguments retain primal
        // input order even when the selection is reversed; returned cotangents follow the selection order instead.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let first = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let second = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let doubled = builder
            .add_instruction(
                ArrayOperation::from(AddOperation::<ArrayType>::new()),
                Vec::new(),
                vec![first, first],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![doubled, second],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let accumulated =
            program.transpose_with_respect_to(&[1, 0], &[CotangentDestinationKind::Reference; 2]).unwrap();
        assert!(accumulated.output_ids().is_empty());
        assert_eq!(
            accumulated.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
            ]
        );
        let first_buffer = ArrayReference::new(Array::scalar(10.0_f32));
        let second_buffer = ArrayReference::new(Array::scalar(20.0_f32));
        assert_eq!(
            accumulated.interpret(vec![
                reference_test_scalar(3.0),
                reference_test_scalar(5.0),
                ArrayIrValue::Reference(first_buffer.clone()),
                ArrayIrValue::Reference(second_buffer.clone()),
            ]),
            Ok(vec![])
        );
        assert_eq!(first_buffer.read(), Ok(Array::scalar(16.0_f32)));
        assert_eq!(second_buffer.read(), Ok(Array::scalar(25.0_f32)));

        let returned = program.transpose_with_respect_to(&[1, 0], &[]).unwrap();
        assert_eq!(
            returned.interpret(vec![reference_test_scalar(3.0), reference_test_scalar(5.0)]),
            Ok(vec![reference_test_scalar(5.0), reference_test_scalar(6.0)])
        );

        // Ignoring every non-reference input removes the arithmetic without changing the seed boundary.
        let ignored = program.transpose_with_respect_to(&[1, 0], &[CotangentDestinationKind::Ignore; 2]).unwrap();
        assert!(ignored.instructions().is_empty());
        assert!(ignored.output_ids().is_empty());
        assert_eq!(ignored.interpret(vec![reference_test_scalar(3.0), reference_test_scalar(5.0)]), Ok(vec![]));
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_destination_identity() {
        // A program can return an input more than once without running a rule. Its seeds must still accumulate into
        // the supplied buffer instead of being lost when the engine installs the buffer-backed storage.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let input = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![input, input],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();
        let pullback = program.transpose_with_respect_to(&[0], &[CotangentDestinationKind::Reference]).unwrap();
        assert_eq!(
            pullback.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["reference_add_update", "reference_add_update"]
        );
        let buffer = ArrayReference::new(Array::scalar(10.0_f32));
        assert_eq!(
            pullback.interpret(vec![
                reference_test_scalar(2.0),
                reference_test_scalar(3.0),
                ArrayIrValue::Reference(buffer.clone())
            ]),
            Ok(vec![])
        );
        assert_eq!(buffer.read(), Ok(Array::scalar(15.0_f32)));
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_read() {
        // `y = read(r)` is the identity from the state to `y`, so its transpose accumulates `ȳ` into the destination.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
            ]
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))]
        );
        assert_eq!(transposed.output_ids(), &[AtomId::new(1)]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_add_update"],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(2.0_f32)],
                vec![Array::scalar(0.5_f32)],
                vec![]
            ),
            vec![Array::scalar(2.5_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_write() {
        // `write(r, x)` has no outputs, so the non-reference adjoint walk would never reach it; the state-driven walk runs
        // it because the destination accumulator is allocated, swaps a zero into the destination (the pre-execution
        // state no longer flows anywhere), and hands the previous contents to `x̄`.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))]
        );
        assert_eq!(
            transposed.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32))
            ]
        );
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero", "reference_swap"],
        );
        assert_eq!(
            run_transposed_with_destinations(&transposed, vec![], vec![Array::scalar(5.0_f32)], vec![]),
            vec![Array::scalar(5.0_f32), Array::scalar(0.0_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_swap() {
        // `y = swap(r, x)` maps `(state, x) ↦ (x, state)`: `ȳ` is swapped into the destination and the previous
        // contents (the post-state cotangent) become `x̄`.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let output = builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
            ]
        );
        assert_eq!(
            transposed.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32))
            ]
        );
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_swap"],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(2.0_f32)],
                vec![Array::scalar(5.0_f32)],
                vec![]
            ),
            vec![Array::scalar(5.0_f32), Array::scalar(2.0_f32)],
        );

        // A dead swap result with an unallocated accumulator stages nothing: the stored value's cotangent is zero.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert!(transposed.input_types().is_empty());
        assert_eq!(transposed.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero"],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_add_update() {
        // `add_update(r, x)` maps `(state, x) ↦ state + x`, so `x̄` reads the destination and the destination is left
        // unchanged for the earlier accesses.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))]
        );
        assert_eq!(
            transposed.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32))
            ]
        );
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_read"],
        );
        assert_eq!(
            run_transposed_with_destinations(&transposed, vec![], vec![Array::scalar(5.0_f32)], vec![]),
            vec![Array::scalar(5.0_f32), Array::scalar(5.0_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_freeze() {
        // `r = new(v); y = freeze(r)` reads the final state and consumes the allocation (only a local allocation may be
        // consumed), so the freeze's transpose accumulates `ȳ` exactly like a read and the allocation's transpose then
        // freezes the accumulator into `v̄ = ȳ`.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(transposed.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero", "reference_new", "reference_add_update", "reference_freeze"],
        );
        assert_eq!(transposed.interpret(vec![reference_test_scalar(2.0)]), Ok(vec![reference_test_scalar(2.0)]));
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_new() {
        // `r = new(v); y = read(r)` allocates the accumulator lazily when the read accumulates `ȳ` and freezes it into
        // `v̄` when the sweep reaches the allocation, so the transposed program allocates exactly once.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(transposed.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero", "reference_new", "reference_add_update", "reference_freeze"],
        );
        assert_eq!(transposed.interpret(vec![reference_test_scalar(3.0)]), Ok(vec![reference_test_scalar(3.0)]),);

        // An allocation that nothing accumulates into yields a symbolic zero for its initial value and stages neither
        // `reference_new` nor `reference_freeze`.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![value],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["zero"],
        );
        assert_eq!(
            transposed.interpret(vec![reference_test_scalar(4.0)]),
            Ok(vec![reference_test_scalar(0.0), reference_test_scalar(4.0)]),
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_dynamic_reference_allocation() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let input = builder.add_input(array_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // A live output cotangent supplies the root extent for the lazy accumulator's zero.
        let transpose = program.transpose_with_respect_to(&[0], &[]).unwrap();
        let cotangent = ReferenceTestValue::Array(Array::vector(vec![2.0_f32, 3.0, 5.0]));
        assert_eq!(transpose.interpret(vec![cotangent.clone()]), Ok(vec![cotangent]));

        // A non-differentiated initializer still allocates a tangent reference with the primal's runtime extent.
        let jvp = program.entry_region_ref().jvp(&[]).unwrap();
        let primal = ReferenceTestValue::Array(Array::vector(vec![7.0_f32, 11.0, 13.0]));
        assert_eq!(
            jvp.interpret(vec![primal.clone()]),
            Ok(vec![primal, ReferenceTestValue::Array(Array::vector(vec![0.0_f32, 0.0, 0.0]))]),
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_dynamic_reference_write() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ReferenceType::new(array_type.clone()).into());
        let replacement = builder.add_input(array_type.into());
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, replacement], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![reference],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        // A state-only output has no array seed from which to recover runtime dimensions. Clearing the caller's destination
        // obtains its extent from that same live reference and returns the old state cotangent for the replacement.
        let transpose = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        let destination = ArrayReference::new(Array::vector(vec![3.0_f32, 5.0, 7.0]));
        assert_eq!(
            transpose.interpret(vec![ReferenceTestValue::Reference(destination.clone())]),
            Ok(vec![
                ReferenceTestValue::Reference(destination.clone()),
                ReferenceTestValue::Array(Array::vector(vec![3.0_f32, 5.0, 7.0])),
            ]),
        );
        assert_eq!(destination.read(), Ok(Array::vector(vec![0.0_f32, 0.0, 0.0])));
    }

    #[test]
    fn test_program_transpose_with_respect_to_write_then_read() {
        // `write(r, x); y = read(r)` maps `x ↦ y` through the state. Under an `Ignore` destination the read allocates
        // the accumulator lazily, the write's transpose reads it back out as `x̄ = ȳ`, and the accumulator is allocated
        // exactly once even though both accesses touch it.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let ignored = program
            .transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(ignored.input_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(ignored.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(
            ignored.instructions().iter().map(|instruction| instruction.operation().name()).collect::<Vec<_>>(),
            vec!["zero", "reference_new", "reference_add_update", "zero", "reference_swap"],
        );
        assert_eq!(ignored.interpret(vec![reference_test_scalar(2.0)]), Ok(vec![reference_test_scalar(2.0)]));

        // Under a `Reference` destination whose initial contents are a nonzero post-state cotangent, that cotangent
        // joins `ȳ` in `x̄` and the destination is left holding the zero pre-state cotangent.
        let referenced = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            referenced.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::scalar(DataType::F32)),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
            ]
        );
        assert_eq!(
            referenced.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::F32))
            ]
        );
        assert!(
            !referenced
                .instructions()
                .iter()
                .any(|instruction| instruction.operation().name() == "reference_new")
        );
        assert_eq!(
            run_transposed_with_destinations(
                &referenced,
                vec![Array::scalar(2.0_f32)],
                vec![Array::scalar(5.0_f32)],
                vec![]
            ),
            vec![Array::scalar(7.0_f32), Array::scalar(0.0_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_unused_ignored_reference_allocates_nothing() {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![value],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(transposed.input_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert_eq!(transposed.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::F32))]);
        assert!(transposed.instructions().is_empty());
    }

    #[test]
    fn test_program_transpose_with_respect_to_rejects_inadmissible_kinds_and_escaping_allocations() {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Return; 2]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "linear reference input 0 of type ref<f32[]> cannot return its cotangent as a value; \
                    transpose it with a `Reference` or `Ignore` cotangent destination",
        ));
        let accumulated =
            program.transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Reference; 2]).unwrap();
        assert_eq!(
            accumulated.input_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))); 2]
        );
        assert_eq!(
            accumulated.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))]
        );
        assert!(matches!(
            program.transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Reference]),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument { message }))
                if message == "transposition received 1 destination kinds for 2 selected inputs",
        ));

        // An escaping local allocation cannot be pulled back because its later uses are unknown to the program.
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let initial = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![initial], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![reference],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        assert!(matches!(
            program.transpose_with_respect_to(&[0], &[]),
            Err(DifferentiationError::Program(ProgramError::UnsupportedOperation { message }))
                if message == "output 0 is a reference allocated inside the transposed program and cannot be pulled \
                    back because its later uses are unknown to the program",
        ));
    }

    #[test]
    fn test_program_transpose_with_respect_to_reference_view() {
        // A view operand accumulates through the same view of its root's cotangent reference: `add_update(r[1], x)`
        // transposes into `x̄ = read(r̄[1])`, so only the selected element of the destination flows into `x̄`.
        let vector_reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let reference = builder.add_input(vector_reference_type.clone());
        let value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let view = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![view, value], None)
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 2],
                Vec::<Placeholder>::new(),
            )
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![vector_reference_type.clone()]);
        assert_eq!(
            transposed.output_types(),
            vec![vector_reference_type, ArrayIrType::Array(ArrayType::scalar(DataType::F32))]
        );
        assert_eq!(
            transposed
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().name())
                .collect::<Vec<_>>(),
            vec!["reference_index", "reference_read"],
        );
        assert_eq!(
            run_transposed_with_destinations(&transposed, vec![], vec![Array::vector(vec![10.0_f32, 20.0])], vec![]),
            vec![Array::scalar(20.0_f32), Array::vector(vec![10.0_f32, 20.0])],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_scan_reference_carry() {
        // `scan { add_update(r, x_i); y_i = read(r) }` over a reference carry: `y_i = r + Σ_{j ≤ i} x_j`, so with unit
        // output cotangents `x̄_j = length - j` and the destination ends holding `Σ ȳ_i = length`. The reference carry
        // is threaded through the reversed scan positionally and the body's view of it accumulates in place.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut body_builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let carry = body_builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let element = body_builder.add_input(scalar_type.clone());
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![carry, element], None)
            .unwrap();
        let current =
            body_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![carry], None).unwrap()[0];
        let body = body_builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![carry, current],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let elements = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ReferenceTestValue>::new(1, 3),
                vec![body],
                vec![reference, elements],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        // The forwarded reference output shares the input's accumulator and has no cotangent slot, so the transposed
        // program consumes `[ȳs, r̄]` and produces `[r̄, x̄s]`.
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]))
            ],
        );
        let scan = transposed
            .instructions()
            .iter()
            .find(|instruction| instruction.operation().name() == "scan")
            .unwrap();
        assert!(matches!(
            scan.operation(),
            ReferenceTestOperation::Scan(operation) if operation.carry_count() == 1 && operation.reverse(),
        ));
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::vector(vec![1.0_f32, 1.0, 1.0])],
                vec![Array::scalar(0.0_f32)],
                vec![],
            ),
            vec![Array::vector(vec![3.0_f32, 2.0, 1.0]), Array::scalar(3.0_f32)],
        );

        // The same program with a view of the carry inside the body accumulates into the root's accumulator through
        // the view: `add_update(r[1], x_i)` over `r: ref<f32[2]>` gives `x̄_i = r̄[1]`.
        let vector_reference_type: ArrayIrType = ReferenceType::new(ArrayType::new_static(DataType::F32, [2])).into();
        let mut body_builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let carry = body_builder.add_input(vector_reference_type.clone());
        let element = body_builder.add_input(scalar_type.clone());
        let view = body_builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![carry], None)
            .unwrap()[0];
        body_builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![view, element], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![carry],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(vector_reference_type.clone());
        let elements = builder.add_input(ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3])));
        let outputs = builder
            .add_instruction(
                ScanOperation::<ReferenceTestValue>::new(1, 3),
                vec![body],
                vec![reference, elements],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(outputs, vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(transposed.input_types(), vec![vector_reference_type]);
        assert_eq!(
            run_transposed_with_destinations(&transposed, vec![], vec![Array::vector(vec![10.0_f32, 20.0])], vec![]),
            vec![Array::vector(vec![20.0_f32, 20.0, 20.0]), Array::vector(vec![10.0_f32, 20.0])],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_scan_write_only_reference_carry() {
        // `scan { write(r, x_i); y_i = x_i }` only stores into its reference carry. Under an `Ignore` destination for
        // the reference no later instruction accumulated into its root and the body never reads it, so its state
        // cotangent is provably zero: the body is transposed with an `Ignore` destination as well, the dead carry is
        // dropped from the reversed scan, and the pullback stages no cotangent reference at all instead of allocating,
        // zeroing, and freezing a dead accumulator around the reversed scan.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let stack_type = ArrayIrType::Array(ArrayType::new_static(DataType::F32, [3]));
        let mut body_builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let carry = body_builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let element = body_builder.add_input(scalar_type.clone());
        body_builder
            .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![carry, element], None)
            .unwrap();
        let body = body_builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![carry, element],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let elements = builder.add_input(stack_type.clone());
        let outputs = builder
            .add_instruction(
                ScanOperation::<ReferenceTestValue>::new(1, 3),
                vec![body],
                vec![reference, elements],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                outputs,
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[0, 1], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(transposed.input_types(), vec![stack_type.clone()]);
        assert_eq!(transposed.output_types(), vec![stack_type.clone()]);
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["scan"]);
        assert!(matches!(
            transposed.instructions()[0].operation(),
            ReferenceTestOperation::Scan(operation) if operation.carry_count() == 0 && operation.reverse(),
        ));
        assert_eq!(
            run_transposed_with_destinations(&transposed, vec![Array::vector(vec![1.0_f32, 2.0, 3.0])], vec![], vec![]),
            vec![Array::vector(vec![1.0_f32, 2.0, 3.0])],
        );

        // Under a `Reference` destination the carry's state cotangent is live, so the cotangent reference is threaded
        // through the reversed scan as a carry and the body's store transposes against it.
        let transposed = program.transpose_with_respect_to(&[0, 1], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![stack_type.clone(), ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32)))]
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), stack_type]
        );
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert!(names.contains(&"reference_swap"), "{names:?}");
        assert!(!names.contains(&"reference_new"), "{names:?}");
        assert!(matches!(
            transposed.instructions()[0].operation(),
            ReferenceTestOperation::Scan(operation) if operation.carry_count() == 1 && operation.reverse(),
        ));
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::vector(vec![1.0_f32, 2.0, 3.0])],
                vec![Array::scalar(5.0_f32)],
                vec![],
            ),
            vec![Array::vector(vec![1.0_f32, 2.0, 8.0]), Array::scalar(0.0_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_condition_reference_operand() {
        // Both branches receive the cotangent reference of the reference operand: the taken branch's transpose acts on
        // it in place (`add_update` reads the destination into `x̄`, `write` swaps a zero into it).
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let true_branch = {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                Vec::new(),
                vec![Placeholder; 3],
                Vec::<Placeholder>::new(),
            )
            .unwrap();

        // The predicate is a known parameter of the linear map, so the transposed program consumes `[r̄, p]`.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                ArrayIrType::Array(ArrayType::scalar(DataType::Boolean))
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type]
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0_f32)],
                vec![Array::scalar(true)]
            ),
            vec![Array::scalar(5.0_f32), Array::scalar(5.0_f32)],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![],
                vec![Array::scalar(5.0_f32)],
                vec![Array::scalar(false)]
            ),
            vec![Array::scalar(5.0_f32), Array::scalar(0.0_f32)],
        );
    }

    #[test]
    fn test_program_transpose_with_respect_to_condition_write_only_reference_operand() {
        // Both branches only store into the reference operand (`write` when taken, `add_update` otherwise) and forward
        // `x` as the live output. Under an `Ignore` destination for the reference no later instruction accumulated
        // into its root and neither branch reads it, so its state cotangent is provably zero: the branches are
        // transposed with an `Ignore` destination as well and the pullback stages no cotangent reference at all
        // instead of allocating, zeroing, and freezing a dead accumulator around the transposed condition.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let predicate_type = ArrayIrType::Array(ArrayType::scalar(DataType::Boolean));
        let true_branch = {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                    vec![value],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let false_branch = {
            let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
            let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
            let value = builder.add_input(scalar_type.clone());
            builder
                .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, value], None)
                .unwrap();
            builder
                .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                    vec![value],
                    vec![Placeholder; 2],
                    vec![Placeholder],
                )
                .unwrap()
        };
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let true_branch = builder.import_region(true_branch.entry_region_ref());
        let false_branch = builder.import_region(false_branch.entry_region_ref());
        let predicate = builder.add_input(predicate_type.clone());
        let reference = builder.add_input(ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))));
        let value = builder.add_input(scalar_type.clone());
        let output = builder
            .add_instruction(
                ConditionOperation::new(),
                vec![true_branch, false_branch],
                vec![predicate, reference, value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder; 3],
                vec![Placeholder],
            )
            .unwrap();
        let transposed = program
            .transpose_with_respect_to(&[1, 2], &[CotangentDestinationKind::Ignore, CotangentDestinationKind::Return])
            .unwrap();
        assert_eq!(transposed.input_types(), vec![scalar_type.clone(), predicate_type.clone()]);
        assert_eq!(transposed.output_types(), vec![scalar_type.clone()]);
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["condition"]);
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0_f32)],
                vec![],
                vec![Array::scalar(true)],
            ),
            vec![Array::scalar(3.0_f32)],
        );

        // Under a `Reference` destination the operand's state cotangent is live, so both branches receive the
        // cotangent reference and their stores transpose against it.
        let transposed = program.transpose_with_respect_to(&[1, 2], &[]).unwrap();
        assert_eq!(
            transposed.input_types(),
            vec![
                scalar_type.clone(),
                ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))),
                predicate_type
            ],
        );
        assert_eq!(
            transposed.output_types(),
            vec![ArrayIrType::from(ReferenceType::new(ArrayType::scalar(DataType::F32))), scalar_type]
        );
        let names = transposed
            .entry_region_ref()
            .instructions_in_closure()
            .map(|(_, instruction)| instruction.operation().name())
            .collect::<Vec<_>>();
        assert!(names.contains(&"reference_swap"), "{names:?}");
        assert!(names.contains(&"reference_read"), "{names:?}");
        assert!(!names.contains(&"reference_new"), "{names:?}");
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0_f32)],
                vec![Array::scalar(5.0_f32)],
                vec![Array::scalar(true)],
            ),
            vec![Array::scalar(8.0_f32), Array::scalar(0.0_f32)],
        );
        assert_eq!(
            run_transposed_with_destinations(
                &transposed,
                vec![Array::scalar(3.0_f32)],
                vec![Array::scalar(5.0_f32)],
                vec![Array::scalar(false)],
            ),
            vec![Array::scalar(8.0_f32), Array::scalar(5.0_f32)],
        );
    }

    #[test]
    fn test_program_linearize_pullback_with_local_tangent_accumulator() {
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let input = builder.add_input(ArrayType::scalar(DataType::F32).into());
        let zero = builder.add_constant(reference_test_scalar(0.0));
        let reference = builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![zero], None).unwrap()[0];
        builder
            .add_instruction(ReferenceAddUpdateOperation::new(), Vec::new(), vec![reference, input], None)
            .unwrap();
        let output =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();
        let linearization = program.linearize().unwrap();
        let primal = linearization.primal().interpret(vec![reference_test_scalar(3.0)]).unwrap();
        let pullback = linearization.pullback().unwrap();
        let discharged = program
            .discharge_references(0)
            .unwrap()
            .into_program_without_external_references()
            .unwrap()
            .linearize()
            .unwrap()
            .pullback()
            .unwrap();
        for seed in [2.0, 5.0, 2.0] {
            let mut inputs = vec![reference_test_scalar(seed)];
            inputs.extend_from_slice(&primal[1..]);
            assert_eq!(pullback.interpret(inputs), Ok(vec![reference_test_scalar(seed)]));
            assert_eq!(discharged.interpret(vec![reference_test_scalar(seed)]), Ok(vec![reference_test_scalar(seed)]));
        }
    }

    #[test]
    fn test_program_linearize_pullback_dynamic_reference_reduction() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let mut builder = ProgramBuilder::<ReferenceTestValue, ReferenceTestOperation>::new();
        let input = builder.add_input(array_type.into());
        let reference =
            builder.add_instruction(ReferenceNewOperation::new(), Vec::new(), vec![input], None).unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let output = builder
            .add_instruction(
                ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                Vec::new(),
                vec![value],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<ReferenceTestValue>, Vec<ReferenceTestValue>>(
                vec![output],
                vec![Placeholder],
                vec![Placeholder],
            )
            .unwrap();

        // The scalar seed has lost the root's dynamic extent. Linearization saves that extent so the transpose can
        // broadcast the seed and allocate the full cotangent root.
        let linearization = program.linearize().unwrap();
        let primal_outputs = linearization
            .primal()
            .interpret(vec![ReferenceTestValue::Array(Array::vector(vec![7.0_f32, 11.0, 13.0]))])
            .unwrap();
        assert_eq!(primal_outputs[0], ReferenceTestValue::Array(Array::scalar(31.0_f32)));
        let mut pullback_inputs = vec![ReferenceTestValue::Array(Array::scalar(5.0_f32))];
        pullback_inputs.extend_from_slice(&primal_outputs[1..]);
        assert_eq!(
            linearization.pullback().unwrap().interpret(pullback_inputs),
            Ok(vec![ReferenceTestValue::Array(Array::vector(vec![5.0_f32, 5.0, 5.0]))]),
        );
    }

    #[test]
    fn test_vjp() {
        // `ReverseModeDifferentiate::vjp` on an explicit context linearizes and transposes: for `f(x) = sin(x)` at
        // `x = 2` the primal output is `sin(2)`, and the returned pullback maps any number of output cotangents back
        // through the transposed Jacobian without re-tracing or re-differentiating.
        let (value, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(|x, ()| x.sin(), Array::scalar(2.0), ())
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pullback.apply(Array::scalar(1.0)).unwrap().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            pullback.apply(Array::scalar(3.0)).unwrap().to_f64s()[0],
            3.0 * 2.0f64.cos(),
            epsilon = 1e-9
        );

        // The builder's `vjp` terminal serves top-level concrete values through their `Value::ExecutionDomain`
        // declarations: a rank-zero `Array` input recovers the eager array domain.
        let (value, pullback) = differentiate_at(Array::scalar(2.0)).vjp(|x| x.sin()).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(pullback.apply(Array::scalar(1.0)).unwrap().to_f64s()[0], 2.0f64.cos(), epsilon = 1e-9);

        let token = Array::from_logical_bytes(ArrayType::scalar(DataType::Token), &[]).unwrap();
        let zero = Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap();
        let (value, pullback) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .vjp(|token, ()| Ok(token), token.clone(), ())
            .unwrap();
        assert_eq!(value, token.clone());
        assert_eq!(pullback.apply(zero.clone()), Ok(zero.clone()));
        assert!(matches!(
            pullback.apply(token.clone()),
            Err(ProgramError::MalformedProgram(message))
                if message
                    == "pullback cotangent 0 has type token[] but its primal boundary requires cotangent type zero[]",
        ));

        // Under an active trace, the builder's `vjp` terminal recovers the staging context from its tracer input,
        // so the primal work and the pullback replay both stage into the enclosing trace.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, pullback) = differentiate_at(inputs[0].clone()).vjp(|x| x.sin())?;
                let cotangent = pullback.apply(inputs[1].clone())?;
                Ok(vec![value, cotangent])
            },
            vec![ArrayType::scalar(DataType::F64), ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let outputs = program.interpret(vec![Array::scalar(2.0), Array::scalar(3.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1].to_f64s()[0], 3.0 * 2.0f64.cos(), epsilon = 1e-9);

        // Replaying a pullback inside an enclosing trace likewise carries absent token cotangents through zero-space
        // leaves and never constructs a token-valued zero operation.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, pullback) = differentiate_at(inputs[0].clone()).vjp(|token| Ok(token))?;
                let cotangent = pullback.apply(inputs[1].clone())?;
                Ok(vec![value, cotangent])
            },
            vec![ArrayType::scalar(DataType::Token), ArrayType::scalar(DataType::Zero)],
        )
        .unwrap();
        assert_eq!(program.interpret(vec![token.clone(), zero.clone()]), Ok(vec![token, zero]));

        // With no leaf value to recover a context from, the builder's `vjp` terminal reports that differentiation
        // requires at least one input leaf.
        let error = differentiate_at(Vec::<Array>::new()).vjp(|x| Ok(x)).map(|(outputs, _)| outputs).unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_vjp_stages_non_copy_array_pullbacks_into_an_enclosing_trace() {
        let vector_type = Array::vector(vec![0.0; 3]).r#type().into_owned();
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(vector_type.clone());
        let cotangent = context.input(vector_type);
        let (_, pullback) = context
            .vjp(|inputs: Vec<_>, ()| Ok(vec![inputs[0].clone() * inputs[0].sin()?]), vec![primal], ())
            .unwrap();
        let input_cotangents = pullback.apply(vec![cotangent]).unwrap();
        let program = context
            .builder()
            .borrow()
            .clone()
            .build::<Vec<Array>, Vec<Array>>(
                vec![input_cotangents[0].atom_id().unwrap()],
                vec![Placeholder; 2],
                vec![Placeholder],
            )
            .unwrap();

        let input = [0.7f64, -1.2, 2.0];
        let output_cotangent = [2.5f64, 1.0, -0.5];
        let outputs = program
            .interpret(vec![Array::vector(input.to_vec()), Array::vector(output_cotangent.to_vec())])
            .unwrap();
        let expected = input
            .into_iter()
            .zip(output_cotangent)
            .map(|(input, cotangent)| (input.sin() + input * input.cos()) * cotangent)
            .collect::<Vec<_>>();
        assert_eq!(outputs.len(), 1);
        for (actual, expected) in outputs[0].to_f64s().iter().zip(expected) {
            assert_abs_diff_eq!(actual, &expected, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_value_and_gradient() {
        // An explicitly selected context computes both the primal and pullback replay. For `f(x, y) = x * y + x`,
        // the value is `8` and the gradient is `(y + 1, x) = (4, 2)` at `(2, 3)`.
        let (value, gradient): (Array, (Array, Array)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
            .value_and_gradient(|(x, y)| x.clone() * y + x)
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 8.0, epsilon = 1e-9);
        assert_eq!(gradient, (Array::scalar(4.0), Array::scalar(2.0)));

        // The builder's `value_and_gradient` terminal recovers the eager domain from the concrete primals.
        let (value, gradient) =
            differentiate_at(Array::scalar(0.7)).value_and_gradient(|x| x.clone() * x.sin().unwrap()).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 0.7 * 0.7f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);

        // Under an active trace, the builder's `value_and_gradient` terminal recovers the staging context from its
        // tracer input instead, so the primal work and the pullback replay both stage into the enclosing trace.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, gradient) =
                    differentiate_at(inputs[0].clone()).value_and_gradient(|x| x.sin().unwrap()).unwrap();
                Ok(vec![value, gradient])
            },
            vec![ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let outputs = program.interpret(vec![Array::scalar(2.0)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], 2.0f64.sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1].to_f64s()[0], 2.0f64.cos(), epsilon = 1e-9);

        // JAX-parity marquee behavior: the closure can branch on a Boolean *primal* with host control flow, because the
        // duals' primal halves carry concrete known values under an eager context (exactly like branching on concrete
        // primals under JAX's `grad`). For a true predicate and `x = 3`, `f(x) = x * x` with gradient `2x = 6`, and
        // the untaken `sin(x)` branch is never traced at all. The Boolean's cotangent remains in the zero space.
        let (value, (predicate_gradient, gradient)) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(true), Array::scalar(3.0)))
            .value_and_gradient(
                |(predicate, x)| {
                    if predicate.concretize().unwrap() { x.clone() * x } else { x.sin().unwrap() }
                },
            )
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 9.0, epsilon = 1e-9);
        assert_eq!(predicate_gradient, Array::from_logical_bytes(ArrayType::scalar(DataType::Zero), &[]).unwrap(),);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 6.0, epsilon = 1e-9);

        // The closure is invoked exactly once: a single linearizing replay produces both the value and the gradient.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F64));
        let calls = Cell::new(0);
        let (_, gradient): (
            DomainTracer<EagerContext<Array, ArrayOperation<Array>>>,
            Vec<DomainTracer<EagerContext<Array, ArrayOperation<Array>>>>,
        ) = context
            .differentiate_at(vec![primal])
            .value_and_gradient(|inputs| {
                calls.set(calls.get() + 1);
                inputs[0].clone() * inputs[0].clone()
            })
            .unwrap();
        assert_eq!(calls.get(), 1);
        assert_eq!(gradient.len(), 1);

        // Mixing tracers of two different traces is rejected with `MismatchedProgramBuilders`. The closure runs on
        // differentiation duals whose operator sugar has no deferral point of its own, so the partial-evaluation
        // context defers the failed bind by poisoning its outputs, and the original error surfaces as a plain `Err`
        // at the evaluation boundary.
        let foreign_context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::F64));
        let foreign_primal = foreign_context.input(ArrayType::scalar(DataType::F64));
        let result = context
            .differentiate_at(vec![primal, foreign_primal])
            .value_and_gradient(|inputs| inputs[0].clone() + inputs[1].clone());
        assert!(matches!(result, Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders))));

        // A complex scalar output is rejected toward the holomorphic entry points, and inputs with no leaf values
        // report an invalid input count.
        let z = Complex::new(0.7f64, -0.3f64);
        let error = differentiate_at(Array::scalar(z)).value_and_gradient(|x| x.clone() * x).unwrap_err();
        assert!(matches!(error, DifferentiationError::ComplexGradientOutput { .. }));
        let error = differentiate_at(Vec::<Array>::new())
            .value_and_gradient(|x| x.into_iter().next().unwrap())
            .unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_value_and_gradient_handles_deep_elementwise_chains_on_the_default_stack() {
        // Linearization and transposition rebuild programs by walking use-def chains whose depth grows with the
        // primal chain length. That walk used to recurse once per producer instruction, so differentiating a chain
        // of a few hundred elementwise operations overflowed the default `libtest` thread stack in debug builds. This
        // chain is an order of magnitude past that threshold and must differentiate on a default-size test thread.
        const CHAIN_LENGTH: usize = 2000;
        let (expected_value, expected_gradient) =
            (0..CHAIN_LENGTH).fold((0.5f64, 1.0f64), |(value, gradient), _| (value.sin(), gradient * value.cos()));
        let (value, gradient) = differentiate_at(Array::scalar(0.5f64))
            .value_and_gradient(|mut value| {
                for _ in 0..CHAIN_LENGTH {
                    value = value.sin().unwrap();
                }
                value
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], expected_value, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], expected_gradient, epsilon = 1e-9);

        // The staged counterpart drives the same rebuilds over a traced program: the primal chain, its linearization,
        // and its transposition all stage into one deep program that is then interpreted (and dropped) on the same
        // default-size thread.
        let (_, program) = EagerContext::<Array, ArrayOperation<Array>>::trace(
            |inputs: Vec<_>| {
                let (value, gradient) = differentiate_at(inputs[0].clone())
                    .value_and_gradient(|mut value| {
                        for _ in 0..CHAIN_LENGTH {
                            value = value.sin().unwrap();
                        }
                        value
                    })
                    .unwrap();
                Ok(vec![value, gradient])
            },
            vec![ArrayType::scalar(DataType::F64)],
        )
        .unwrap();
        let outputs = program.interpret(vec![Array::scalar(0.5f64)]).unwrap();
        assert_eq!(outputs.len(), 2);
        assert_abs_diff_eq!(outputs[0].to_f64s()[0], expected_value, epsilon = 1e-9);
        assert_abs_diff_eq!(outputs[1].to_f64s()[0], expected_gradient, epsilon = 1e-9);
    }

    #[test]
    fn test_value_and_gradient_reference_inputs() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference.clone()))
                .value_and_gradient(|reference: ReferenceTestTracer| reference_test_square(reference.read()?)),
            Ok((reference_test_scalar(9.0), reference_test_scalar(6.0))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));

        // Each invocation differentiates the current input state with a fresh zero final-state cotangent.
        let function = |reference: ReferenceTestTracer| {
            let value = reference.read()?;
            reference.add_update(&value)?;
            reference_test_square(reference.read()?)
        };
        assert_eq!(
            ReferenceTestContext::new()
                .differentiate_at(ArrayIrValue::Reference(reference.clone()))
                .value_and_gradient(function),
            Ok((reference_test_scalar(36.0), reference_test_scalar(24.0))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(6.0_f32)));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference.clone())).value_and_gradient(function),
            Ok((reference_test_scalar(144.0), reference_test_scalar(48.0))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(12.0_f32)));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_mutations() {
        // Overwriting the state removes dependence on its initial contents and sends the adjoint to the value input.
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at((ArrayIrValue::Reference(reference.clone()), reference_test_scalar(5.0)))
                .value_and_gradient(|(reference, value): (ReferenceTestTracer, ReferenceTestTracer)| {
                    reference.write(&value)?;
                    reference_test_square(reference.read()?)
                }),
            Ok((reference_test_scalar(25.0), (reference_test_scalar(0.0), reference_test_scalar(10.0)))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(5.0_f32)));

        // Swapping exposes the old state as a value; the replacement contributes only through final-state seeds,
        // which the gradient convenience API sets to zero.
        assert_eq!(
            differentiate_at((ArrayIrValue::Reference(reference.clone()), reference_test_scalar(7.0)))
                .value_and_gradient(|(reference, value): (ReferenceTestTracer, ReferenceTestTracer)| {
                    reference_test_square(reference.swap(&value)?)
                }),
            Ok((reference_test_scalar(25.0), (reference_test_scalar(10.0), reference_test_scalar(0.0)))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(7.0_f32)));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_consumed() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference.clone()))
                .value_and_gradient(|reference: ReferenceTestTracer| reference_test_square(reference.freeze()?)),
            Ok((reference_test_scalar(9.0), reference_test_scalar(6.0))),
        );
        assert_eq!(reference.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_unused() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at((ArrayIrValue::Reference(reference.clone()), reference_test_scalar(5.0)))
                .value_and_gradient(|(_, value): (ReferenceTestTracer, ReferenceTestTracer)| reference_test_square(
                    value
                )),
            Ok((reference_test_scalar(25.0), (reference_test_scalar(0.0), reference_test_scalar(10.0)))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));

        // A zero-space referent still needs a reference identity while replaying, then becomes a non-reference zero.
        let reference = ArrayReference::new(Array::scalar(3_i32));
        let (_, (gradient, _)) = differentiate_at((ArrayIrValue::Reference(reference), reference_test_scalar(5.0)))
            .value_and_gradient(|(_, value): (ReferenceTestTracer, ReferenceTestTracer)| Ok::<_, ProgramError>(value))
            .unwrap();
        assert_eq!(gradient, ArrayIrValue::Array(Array::new(ArrayType::scalar(DataType::Zero), Vec::new()).unwrap()));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_dynamic_consumption() {
        let extent = DimensionVariable::new("extent", DimensionBounds::new(2, Some(8)).unwrap());
        let array_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Dynamic(extent)]));
        let (_, program) = ReferenceTestContext::trace(
            |inputs: Vec<_>| {
                let (_, gradient) = differentiate_at(inputs[0].clone()).value_and_gradient(|reference| {
                    let value = reference.freeze()?;
                    Ok::<_, ProgramError>(
                        value
                            .context()
                            .bind(
                                ArrayOperation::from(ReduceOperation::new(vec![0], ReductionKind::Sum)),
                                Vec::new(),
                                &[value.clone()],
                            )?
                            .remove(0),
                    )
                })?;
                Ok(vec![gradient])
            },
            vec![ArrayIrType::Reference(ReferenceType::new(array_type))],
        )
        .unwrap();

        // The zero's extent must be captured before the primal freeze. Only non-reference gradients leave the trace,
        // and its one internal accumulator is allocated afresh by each interpretation.
        assert!(program.output_types().iter().all(|r#type| !r#type.is_reference()));
        assert_eq!(
            program
                .entry_region()
                .instructions()
                .iter()
                .filter(|instruction| { instruction.operation().name() == "reference_new" })
                .count(),
            1
        );
        let reference = ArrayReference::new(Array::vector(vec![3.0_f32, 5.0, 7.0]));
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(reference.clone())]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32; 3]))])
        );
        assert_eq!(reference.read().unwrap_err().downcast_custom::<ReferenceError>(), Some(&ReferenceError::Frozen));
        assert_eq!(
            program.interpret(vec![ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![9.0_f32; 4])))]),
            Ok(vec![ArrayIrValue::Array(Array::vector(vec![1.0_f32; 4]))])
        );
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_batching() {
        let reference = ArrayReference::new(Array::vector(vec![3.0_f32, 5.0]));
        let result = batch(
            |reference| {
                differentiate_at(reference)
                    .value_and_gradient(|reference| reference_test_square(reference.read()?))
                    .map_err(ProgramError::from)
            },
            ArrayIrValue::Reference(reference.clone()),
            BatchAxis::new(0),
            (BatchAxis::new(0), BatchAxis::new(0)),
            None,
        );
        assert_eq!(
            result,
            Ok((
                ArrayIrValue::Array(Array::vector(vec![9.0_f32, 25.0])),
                ArrayIrValue::Array(Array::vector(vec![6.0_f32, 10.0]))
            ))
        );
        assert_eq!(reference.read(), Ok(Array::vector(vec![3.0_f32, 5.0])));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_rejects_aliases() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at((ArrayIrValue::Reference(reference.clone()), ArrayIrValue::Reference(reference.clone())))
                .value_and_gradient(|(first, _): (ReferenceTestTracer, ReferenceTestTracer)| first.read()),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument {
                message: "input 1 and input 0 bind the same reference allocation".to_string(),
            })),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_value_and_gradient_reference_inputs_higher_order() {
        // The inner gradient is 2x; differentiating the returned non-reference value agrees with d²(x²)/dx² = 2.
        assert_eq!(
            differentiate_at(reference_test_scalar(3.0)).value_and_gradient(|value: ReferenceTestTracer| {
                differentiate_at(value.reference_new()?)
                    .gradient(|reference| reference_test_square(reference.read()?))
                    .map_err(ProgramError::from)
            }),
            Ok((reference_test_scalar(6.0), reference_test_scalar(2.0))),
        );
    }

    #[test]
    fn test_gradient() {
        // The builder's `gradient` terminal is the gradient-only counterpart of `value_and_gradient`.
        let method_gradient: (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
            .gradient(|(x, y)| x.clone() * y + x)
            .unwrap();
        assert_eq!(method_gradient, (Array::scalar(4.0), Array::scalar(2.0)));

        // The builder's `gradient` terminal recovers the eager domain from the concrete primal and agrees with the
        // value-carrying form.
        let free_gradient = differentiate_at(Array::scalar(0.7)).gradient(|x| x.clone() * x.sin().unwrap()).unwrap();
        assert_abs_diff_eq!(free_gradient.to_f64s()[0], 0.7f64.sin() + 0.7 * 0.7f64.cos(), epsilon = 1e-9);

        // With no leaf value to recover a context from, the builder's `gradient` terminal reports that differentiation
        // requires at least one input leaf.
        let error = differentiate_at(Vec::<Array>::new()).gradient(|x| x.into_iter().next().unwrap()).unwrap_err();
        assert_eq!(error, DifferentiationError::EmptyInput);
    }

    #[test]
    fn test_builder_value_and_gradient_in_holomorphic_mode() {
        // An explicitly selected builder context recovers `∂z²/∂z = 2z` under the holomorphy promise.
        let z = Complex::new(0.7f64, -0.3f64);
        let (value, gradient) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(z))
            .holomorphic()
            .value_and_gradient(|x| x.clone() * x)
            .unwrap();
        assert_eq!(value, Array::scalar(z * z));
        assert_eq!(gradient, Array::scalar(z + z));

        // The builder recovers the eager domain from the concrete primal, and for real outputs the holomorphy
        // promise changes nothing.
        let (value, gradient) =
            differentiate_at(Array::scalar(2.0)).holomorphic().value_and_gradient(|x| x.clone() * x).unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 4.0, epsilon = 1e-9);
        assert_abs_diff_eq!(gradient.to_f64s()[0], 4.0, epsilon = 1e-9);

        // Under an active trace the guards run at the type level (the identity closure performs no complex arithmetic).
        // The plain entry point rejects a complex output toward the holomorphic one, which accepts it and seeds `one`
        // at the complex cotangent type, while a real output flows through the holomorphic entry point exactly like
        // the plain one.
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::C64));
        let result = context.differentiate_at(vec![primal]).value_and_gradient(|inputs: Vec<_>| inputs[0].clone());
        assert!(matches!(
            result,
            Err(DifferentiationError::ComplexGradientOutput { output_type }) if output_type == "c64[]",
        ));
        let primal = context.input(ArrayType::scalar(DataType::C64));
        let (value, gradient) = context
            .differentiate_at(vec![primal])
            .holomorphic()
            .value_and_gradient(|inputs: Vec<_>| inputs[0].clone())
            .unwrap();
        assert_eq!(*value.r#type(), ArrayType::scalar(DataType::C64));
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), ArrayType::scalar(DataType::C64));
        let primal = context.input(ArrayType::scalar(DataType::F64));
        let (value, gradient) = context
            .differentiate_at(vec![primal])
            .holomorphic()
            .value_and_gradient(|inputs: Vec<_>| inputs[0].clone())
            .unwrap();
        assert_eq!(*value.r#type(), ArrayType::scalar(DataType::F64));
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), ArrayType::scalar(DataType::F64));
    }

    #[test]
    fn test_builder_value_and_gradient_with_auxiliary_output() {
        // An explicitly selected builder context returns auxiliary outputs as non-reference primal values seeded with zero
        // cotangents, so they do not contribute to the gradient.
        let ((value, aux), gradient): ((Array, Array), (Array, Array)) =
            EagerContext::<Array, ArrayOperation<Array>>::new()
                .differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
                .with_auxiliary_output()
                .value_and_gradient(|(x, y)| (x.clone() * y.clone(), x + y))
                .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(aux.to_f64s()[0], 5.0, epsilon = 1e-9);
        assert_eq!(gradient, (Array::scalar(3.0), Array::scalar(2.0)));

        // The builder recovers the eager domain from the concrete primals, and the auxiliary structure can carry
        // multiple leaves (each rides along as a primal value with a zero cotangent seed, so none contributes to the
        // gradient of `x * y`).
        let ((value, aux), gradient): ((Array, (Array, Array)), (Array, Array)) =
            differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
                .with_auxiliary_output()
                .value_and_gradient(|(x, y)| {
                    let value = x.clone() * y.clone();
                    let aux = (x.clone() + y, x.clone() * x);
                    (value, aux)
                })
                .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 6.0, epsilon = 1e-9);
        assert_eq!(aux, (Array::scalar(5.0), Array::scalar(4.0)));
        assert_eq!(gradient, (Array::scalar(3.0), Array::scalar(2.0)));

        // Auxiliary cotangent seeds use each auxiliary leaf's cotangent type. This matters both for non-differentiable
        // leaves, whose cotangent type is the first-class zero space, and for differentiable storage representations
        // such as E8M0, whose cotangent type is widened to F32 and can represent zero.
        let ((value, aux), gradient): ((Array, (Array, Array)), Array) = differentiate_at(Array::scalar(2.0))
            .with_auxiliary_output()
            .value_and_gradient(|x| {
                let integer = x.context().constant(Array::scalar(7i32))?;
                let e8m0 = x
                    .context()
                    .constant(Array::from_logical_bytes(ArrayType::scalar(DataType::F8E8M0FNU), &[0x7f]).unwrap())?;
                Ok((x.clone() * x, (integer, e8m0)))
            })
            .unwrap();
        assert_eq!(value, Array::scalar(4.0));
        assert_eq!(
            aux,
            (Array::scalar(7i32), Array::from_logical_bytes(ArrayType::scalar(DataType::F8E8M0FNU), &[0x7f]).unwrap(),),
        );
        assert_eq!(gradient, Array::scalar(4.0));
    }

    #[test]
    fn test_builder_value_and_gradient_with_auxiliary_output_reference_inputs() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        let capture = ArrayReference::new(Array::scalar(2.0_f32));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference.clone()))
                .with_captures(ArrayIrValue::Reference(capture.clone()))
                .with_auxiliary_output()
                .value_and_gradient(|reference: ReferenceTestTracer, capture: ReferenceTestTracer| {
                    reference.add_update(&capture.read()?)?;
                    Ok::<_, ProgramError>((reference_test_square(reference.read()?)?, reference.read()?))
                }),
            Ok(((reference_test_scalar(25.0), reference_test_scalar(5.0)), reference_test_scalar(10.0))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(5.0_f32)));
        assert_eq!(capture.read(), Ok(Array::scalar(2.0_f32)));
    }

    #[test]
    fn test_builder_value_and_gradient_with_auxiliary_output_rejects_reference_outputs() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference))
                .with_auxiliary_output()
                .value_and_gradient::<_, _, _, ReferenceTestValue, _>(|reference: ReferenceTestTracer| Ok::<
                    _,
                    ProgramError,
                >(
                    (
                    reference.read()?,
                    reference
                )
                )),
            Err(DifferentiationError::Program(ProgramError::InvalidArgument {
                message:
                    "gradient auxiliary outputs must be non-reference values; use `Pullback::apply_with_destinations` \
                        for reference outputs"
                        .to_string(),
            })),
        );
    }

    #[test]
    fn test_builder_value_and_gradient_with_auxiliary_output_in_holomorphic_mode() {
        // Builder type state composes the holomorphy promise with auxiliary output: the gradient is `∂z²/∂z = 2z`
        // while the auxiliary value rides along with a zero cotangent seed.
        let z = Complex::new(0.7f64, -0.3f64);
        let ((value, aux), gradient): ((Array, Array), Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(z))
            .with_auxiliary_output()
            .holomorphic()
            .value_and_gradient(|x| (x.clone() * x.clone(), x))
            .unwrap();
        assert_eq!(value, Array::scalar(z * z));
        assert_eq!(aux, Array::scalar(z));
        assert_eq!(gradient, Array::scalar(z + z));

        // The builder recovers the eager domain from the concrete primal and agrees.
        let ((value, aux), gradient): ((Array, Array), Array) = differentiate_at(Array::scalar(z))
            .with_auxiliary_output()
            .holomorphic()
            .value_and_gradient(|x| (x.clone() * x.clone(), x))
            .unwrap();
        assert_eq!(value, Array::scalar(z * z));
        assert_eq!(aux, Array::scalar(z));
        assert_eq!(gradient, Array::scalar(z + z));

        // The holomorphic entry point uses the same zero-space cotangent rule for non-differentiable auxiliary leaves.
        let ((value, aux), gradient): ((Array, Array), Array) = differentiate_at(Array::scalar(z))
            .with_auxiliary_output()
            .holomorphic()
            .value_and_gradient(|x| {
                let aux = x.context().constant(Array::scalar(7i32))?;
                Ok((x.clone() * x, aux))
            })
            .unwrap();
        assert_eq!(value, Array::scalar(z * z));
        assert_eq!(aux, Array::scalar(7i32));
        assert_eq!(gradient, Array::scalar(z + z));

        // The holomorphy gate also runs at the type level under an active trace. A complex output with
        // an auxiliary output is accepted end to end and seeds `one` at the complex cotangent type.
        type TestTracer = DomainTracer<EagerContext<Array, ArrayOperation<Array>>>;
        let context = DomainTracingContext::<EagerContext<Array, ArrayOperation<Array>>>::new();
        let primal = context.input(ArrayType::scalar(DataType::C64));
        let ((value, aux), gradient): ((TestTracer, TestTracer), Vec<TestTracer>) = context
            .differentiate_at(vec![primal])
            .with_auxiliary_output()
            .holomorphic()
            .value_and_gradient(|inputs: Vec<_>| (inputs[0].clone(), inputs[0].clone()))
            .unwrap();
        assert_eq!(*value.r#type(), ArrayType::scalar(DataType::C64));
        assert_eq!(*aux.r#type(), ArrayType::scalar(DataType::C64));
        assert_eq!(gradient.len(), 1);
        assert_eq!(*gradient[0].r#type(), ArrayType::scalar(DataType::C64));
    }

    #[test]
    fn test_builder_gradient_in_holomorphic_mode() {
        // The holomorphic builder gradient computes `∂sin(z)/∂z = cos(z)` at a genuinely complex point.
        let z = Complex::new(0.7f64, -0.3f64);
        let method_gradient = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(z))
            .holomorphic()
            .gradient(|x| x.sin().unwrap())
            .unwrap();
        assert_eq!(method_gradient, Array::scalar(z.cos()));

        // The builder recovers the eager domain from the concrete primal and agrees.
        let free_gradient = differentiate_at(Array::scalar(z)).holomorphic().gradient(|x| x.sin().unwrap()).unwrap();
        assert_eq!(free_gradient, Array::scalar(z.cos()));
    }

    #[test]
    fn test_builder_gradient_with_auxiliary_output() {
        // The auxiliary builder's `gradient` terminal returns `(gradient, auxiliary)`.
        let (method_gradient, aux): ((Array, Array), Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
            .with_auxiliary_output()
            .gradient(|(x, y)| (x.clone() * y.clone(), x + y))
            .unwrap();
        assert_eq!(method_gradient, (Array::scalar(3.0), Array::scalar(2.0)));
        assert_abs_diff_eq!(aux.to_f64s()[0], 5.0, epsilon = 1e-9);

        // The builder recovers the eager domain from the concrete primals and agrees.
        let (free_gradient, aux): ((Array, Array), Array) = differentiate_at((Array::scalar(2.0), Array::scalar(3.0)))
            .with_auxiliary_output()
            .gradient(|(x, y)| (x.clone() * y.clone(), x + y))
            .unwrap();
        assert_eq!(free_gradient, (Array::scalar(3.0), Array::scalar(2.0)));
        assert_abs_diff_eq!(aux.to_f64s()[0], 5.0, epsilon = 1e-9);
    }

    #[test]
    fn test_builder_gradient_with_auxiliary_output_reference_inputs() {
        let reference = ArrayReference::new(Array::scalar(3.0_f32));
        assert_eq!(
            differentiate_at(ArrayIrValue::Reference(reference.clone())).with_auxiliary_output().gradient(
                |reference: ReferenceTestTracer| {
                    Ok::<_, ProgramError>((reference_test_square(reference.read()?)?, reference.read()?))
                }
            ),
            Ok((reference_test_scalar(6.0), reference_test_scalar(3.0))),
        );
        assert_eq!(reference.read(), Ok(Array::scalar(3.0_f32)));
    }

    #[test]
    fn test_builder_gradient_with_auxiliary_output_in_holomorphic_mode() {
        // The holomorphic auxiliary builder's `gradient` terminal returns `(gradient, auxiliary)`.
        let z = Complex::new(0.7f64, -0.3f64);
        let (method_gradient, aux): (Array, Array) = EagerContext::<Array, ArrayOperation<Array>>::new()
            .differentiate_at(Array::scalar(z))
            .with_auxiliary_output()
            .holomorphic()
            .gradient(|x| (x.clone() * x.clone(), x))
            .unwrap();
        assert_eq!(method_gradient, Array::scalar(z + z));
        assert_eq!(aux, Array::scalar(z));

        // The builder recovers the eager domain from the concrete primal and agrees.
        let (free_gradient, aux): (Array, Array) = differentiate_at(Array::scalar(z))
            .with_auxiliary_output()
            .holomorphic()
            .gradient(|x| (x.clone() * x.clone(), x))
            .unwrap();
        assert_eq!(free_gradient, Array::scalar(z + z));
        assert_eq!(aux, Array::scalar(z));
    }

    #[test]
    fn test_nested_differentiation() {
        // Every nesting shape differentiates `f(x) = sin(x²)` at `x = 0.7` through closure-level nesting. Inner
        // transforms run on the nested tracing context their tracers flow in, recovered either implicitly by the
        // free differentiation functions from their tracer inputs or explicitly through `x.context()`. Every closure
        // is fallible, propagating staging failures outward through `?` (or by adapting the inner transform's
        // `Result` with `.map_err(Into::into)`) instead of unwrapping.
        let domain = EagerContext::<Array, ArrayOperation<Array>>::new();
        let x: f64 = 0.7;

        // Reverse-over-reverse through builder terminals: the outer value is `f'(x) = 2x cos(x²)` and the
        // outer gradient is the analytic second derivative `f''(x) = 2 cos(x²) - 4x² sin(x²)`.
        let (value, second_derivative) = differentiate_at(Array::scalar(x))
            .value_and_gradient(|x| differentiate_at(x).gradient(|y| (y.clone() * y).sin()).map_err(Into::into))
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            second_derivative.to_f64s()[0],
            2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(),
            epsilon = 1e-9
        );

        // Three levels of nesting exercise the recursive `NestedTracingContext<NestedTracingContext<...>>` types
        // through the trait solver, with every inner transform run through an explicitly recovered context receiver.
        // The outer gradient is the analytic third derivative `f'''(x) = -12x sin(x²) - 8x³ cos(x²)`.
        let (value, third_derivative) = domain
            .differentiate_at(Array::scalar(x))
            .value_and_gradient(|x| {
                x.clone()
                    .context()
                    .differentiate_at(x)
                    .gradient(|y| {
                        y.clone().context().differentiate_at(y).gradient(|z| (z.clone() * z).sin()).map_err(Into::into)
                    })
                    .map_err(Into::into)
            })
            .unwrap();
        assert_abs_diff_eq!(value.to_f64s()[0], 2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            third_derivative.to_f64s()[0],
            -12.0 * x * (x * x).sin() - 8.0 * x * x * x * (x * x).cos(),
            epsilon = 1e-9,
        );

        // Forward-over-reverse through builder terminals: pushing the tangent `v = 2` through the gradient computes the
        // Hessian-vector product `f''(x) · v` without materializing a dense Hessian, because the `jvp` duals' stamped
        // `DifferentiationContext` is itself a `ReverseModeDifferentiate` the inner transform nests on.
        let (primal, tangent) = differentiate_at(Array::scalar(x))
            .jvp(Array::scalar(2.0), |x| Ok(differentiate_at(x).gradient(|y| (y.clone() * y).sin())?))
            .unwrap();
        assert_abs_diff_eq!(primal.to_f64s()[0], 2.0 * x * (x * x).cos(), epsilon = 1e-9);
        assert_abs_diff_eq!(
            tangent.to_f64s()[0],
            2.0 * (2.0 * (x * x).cos() - 4.0 * x * x * (x * x).sin()),
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_transpose_projected_operation() {
        // The third fixture member is intentionally unrelated to arrays. Its identity transpose proves that the
        // adapter records the member rule, converts its program, and splices the live cotangent into the composite
        // trace without changing its SSA identity or type.
        let mut context =
            TranspositionContext::new(TracingContext::<ProjectedProgramValue, ProjectedProgramOperation>::new());
        let member_type = ProjectedProgramType::Third(ProjectedMemberType::<2>);
        let output_cotangent = context.input(member_type.clone());
        let inputs = &[PartialValue::Unknown(member_type.clone())];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        transpose_projected_operation(
            &mut context,
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

        // Known primal inputs are replay operands rather than linear inputs, so the member rule returns a structural
        // zero for them. Structural-zero output cotangents also cross the adapter without becoming replay values.
        let known_input = context.input(member_type.clone());
        let inputs = &[PartialValue::Known(known_input)];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        transpose_projected_operation(
            &mut context,
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
        transpose_projected_operation(
            &mut context,
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
    fn test_transpose_projected_operation_preserves_contribution_provenance() {
        /// Member rule that submits two identity contributions under different local scopes.
        #[derive(Clone, Debug)]
        enum ScopedMemberOperation {
            Scoped,
            Add(AddOperation<ArrayType>),
        }

        impl Operation for ScopedMemberOperation {
            type Type = ArrayType;

            fn name(&self) -> &'static str {
                match self {
                    Self::Scoped => "scoped_member",
                    Self::Add(operation) => operation.name(),
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayType],
                _region_interfaces: &[RegionInterface<ArrayType>],
            ) -> Result<Vec<ArrayType>, TypeError> {
                if let Self::Add(operation) = self {
                    return operation.infer_output_types(input_types, _region_interfaces);
                }
                check_count!("input", input_types, 1, TypeError);
                Ok(input_types.to_vec())
            }
        }

        impl<V: Value<Type = ArrayType>, O: Operation<Type = ArrayType> + From<AddOperation<ArrayType>>>
            TransposableOperation<V, O> for ScopedMemberOperation
        {
            fn transpose<D: TranspositionDriver<V, O>>(
                &self,
                context: &mut TranspositionContext<V, O>,
                _driver: &D,
                inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
                outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
                accumulators: &[CotangentAccumulator],
            ) -> Result<(), DifferentiationError> {
                let input_count = if matches!(self, Self::Add(_)) { 2 } else { 1 };
                check_count!("input", inputs, input_count, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, input_count, DifferentiationError);
                if matches!(self, Self::Add(_)) {
                    for (input, accumulator) in inputs.iter().zip(accumulators) {
                        if matches!(input, PartialValue::Unknown(_)) {
                            accumulator.accumulate(context, outputs[0].clone())?;
                        }
                    }
                    return Ok(());
                }
                (*context).clone().invoke_with_provenance_scope(ProvenanceScope::new("first"), || {
                    accumulators[0].accumulate(context, outputs[0].clone())
                })?;
                (*context).clone().invoke_with_provenance_scope(ProvenanceScope::new("second"), || {
                    accumulators[0].accumulate(context, outputs[0].clone())
                })
            }
        }

        impl From<AddOperation<ArrayType>> for ScopedMemberOperation {
            fn from(operation: AddOperation<ArrayType>) -> Self {
                Self::Add(operation)
            }
        }

        /// Composite carrier that preserves additions staged inside the member rule.
        #[derive(Clone, Debug)]
        struct ScopedProgramOperation(ScopedMemberOperation);

        impl Operation for ScopedProgramOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                self.0.name()
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                let input_types = input_types
                    .iter()
                    .map(|value| <&ArrayType>::try_from(value).cloned())
                    .collect::<Result<Vec<_>, _>>()?;
                check_count!("region", region_interfaces, 0, TypeError);
                Ok(self.0.infer_output_types(&input_types, &[])?.into_iter().map(ArrayIrType::from).collect())
            }
        }

        impl From<ScopedMemberOperation> for ScopedProgramOperation {
            fn from(operation: ScopedMemberOperation) -> Self {
                Self(operation)
            }
        }

        impl From<AddOperation<ArrayIrType>> for ScopedProgramOperation {
            fn from(_operation: AddOperation<ArrayIrType>) -> Self {
                Self(ScopedMemberOperation::Add(AddOperation::new()))
            }
        }

        impl OperationProjection<ArrayType> for ScopedProgramOperation {
            type Projected = ScopedMemberOperation;
        }

        let mut context =
            TranspositionContext::new(TracingContext::<ArrayIrValue<Array>, ScopedProgramOperation>::new());
        let r#type = ArrayIrType::from(ArrayType::scalar(DataType::F64));
        let seed = context.input(r#type.clone());
        let inputs = [PartialValue::Unknown(r#type)];
        let accumulators = context.cotangent_accumulators(&inputs, &[]).unwrap();
        let origin = Provenance::scope(ProvenanceScope::new("source"), Provenance::unknown());
        (*context)
            .clone()
            .invoke_with_provenance_origin(origin.clone(), || {
                transpose_projected_operation(
                    &mut context,
                    &ScopedMemberOperation::Scoped,
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
    fn test_transpose_projected_operation_rejects_foreign_builders() {
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
            transpose_projected_operation(
                &mut context,
                &ProjectedMemberOperation::<2>::Identity,
                &[PartialValue::Known(foreign.clone())],
                &[MaybeZero::Value(local)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
        assert!(matches!(
            transpose_projected_operation(
                &mut context,
                &ProjectedMemberOperation::<2>::Identity,
                &[PartialValue::Unknown(member_type)],
                &[MaybeZero::Value(foreign)],
                &accumulators,
            ),
            Err(DifferentiationError::Program(ProgramError::MismatchedProgramBuilders)),
        ));
    }

    #[test]
    fn test_transpose_mixed_operation() {
        /// Test-only linear member operation consuming two member-typed data operands. No production mixed payload
        /// interleaves its data and shape operands, so this fixture is what pins that mixed transposition classifies
        /// operands one by one instead of splitting the operand list at its first shape operand.
        #[derive(Clone, Debug)]
        enum InterleavedMemberOperation {
            Interleaved,
            Add,
        }

        impl Operation for InterleavedMemberOperation {
            type Type = ProjectedMemberType<2>;

            fn name(&self) -> &'static str {
                match self {
                    Self::Interleaved => "interleaved_member",
                    Self::Add => "add",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ProjectedMemberType<2>],
                region_interfaces: &[RegionInterface<ProjectedMemberType<2>>],
            ) -> Result<Vec<ProjectedMemberType<2>>, TypeError> {
                check_count!("input", input_types, 2, TypeError);
                check_count!("region", region_interfaces, 0, TypeError);
                Ok(vec![ProjectedMemberType])
            }
        }

        impl<
            V: Value<Type = ProjectedMemberType<2>>,
            O: Operation<Type = ProjectedMemberType<2>> + From<AddOperation<ProjectedMemberType<2>>>,
        > TransposableOperation<V, O> for InterleavedMemberOperation
        {
            fn transpose<D: TranspositionDriver<V, O>>(
                &self,
                context: &mut TranspositionContext<V, O>,
                _driver: &D,
                inputs: &[PartialValue<Tracer<TracingContext<V, O>>>],
                outputs: &[MaybeZero<Tracer<TracingContext<V, O>>>],
                accumulators: &[CotangentAccumulator],
            ) -> Result<(), DifferentiationError> {
                check_count!("input", inputs, 2, ProgramError);
                check_count!("output", outputs, 1, ProgramError);
                check_count!("accumulator", accumulators, 2, DifferentiationError);
                for (input, accumulator) in inputs.iter().zip(accumulators) {
                    if matches!(input, PartialValue::Unknown(_)) {
                        accumulator.accumulate(context, outputs[0].clone())?;
                    }
                }
                Ok(())
            }
        }

        impl From<AddOperation<ProjectedMemberType<2>>> for InterleavedMemberOperation {
            fn from(_operation: AddOperation<ProjectedMemberType<2>>) -> Self {
                Self::Add
            }
        }

        /// Composite carrier whose canonical third-member projection is [`InterleavedMemberOperation`].
        #[derive(Clone, Debug)]
        struct InterleavedProgramOperation(InterleavedMemberOperation);

        impl Operation for InterleavedProgramOperation {
            type Type = ProjectedProgramType;

            fn name(&self) -> &'static str {
                self.0.name()
            }

            fn infer_output_types(
                &self,
                input_types: &[ProjectedProgramType],
                region_interfaces: &[RegionInterface<ProjectedProgramType>],
            ) -> Result<Vec<ProjectedProgramType>, TypeError> {
                check_count!("region", region_interfaces, 0, TypeError);
                if matches!(self.0, InterleavedMemberOperation::Add) {
                    check_count!("input", input_types, 2, TypeError);
                    check_types!(@same, "addition input", [&input_types[..1], &input_types[1..]]);
                    return Ok(vec![input_types[0].clone()]);
                }
                Ok(vec![ProjectedProgramType::Third(ProjectedMemberType)])
            }
        }

        impl From<InterleavedMemberOperation> for InterleavedProgramOperation {
            fn from(operation: InterleavedMemberOperation) -> Self {
                Self(operation)
            }
        }

        impl From<AddOperation<ProjectedProgramType>> for InterleavedProgramOperation {
            fn from(_operation: AddOperation<ProjectedProgramType>) -> Self {
                Self(InterleavedMemberOperation::Add)
            }
        }

        impl OperationProjection<ProjectedMemberType<2>> for InterleavedProgramOperation {
            type Projected = InterleavedMemberOperation;
        }

        type Context = TracingContext<ProjectedProgramValue, InterleavedProgramOperation>;

        // Data operands at positions 0 and 2 are delegated to the member rule in that order, and the shape operands
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
        transpose_mixed_operation(
            &mut context,
            &InterleavedMemberOperation::Interleaved,
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
            panic!("mixed transposition must classify each operand individually: {cotangents:?}");
        };
        assert_eq!(first_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(third_cotangent.atom_id(), output_cotangent.atom_id());
        assert_eq!(second_cotangent_type, &shape_type);
        assert_eq!(fourth_cotangent_type, &shape_type);

        // Interleaved known data operands stay known operands of the member rule, so their structural zeros are the
        // member rule's own and remain at their operand positions.
        let known_data = context.input(data_type.clone());
        let inputs = &[
            PartialValue::Known(known_data),
            PartialValue::Unknown(shape_type.clone()),
            PartialValue::Unknown(data_type.clone()),
        ];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        transpose_mixed_operation(
            &mut context,
            &InterleavedMemberOperation::Interleaved,
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

        // The data-operands-first arrangement every current payload uses degenerates to the same result, so the
        // classification is a strict generalization of splitting the operand list at its first shape operand.
        let inputs = &[
            PartialValue::Unknown(data_type.clone()),
            PartialValue::Unknown(data_type),
            PartialValue::Unknown(shape_type.clone()),
        ];
        let accumulators = context.cotangent_accumulators(inputs, &[]).unwrap();
        transpose_mixed_operation(
            &mut context,
            &InterleavedMemberOperation::Interleaved,
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
}
