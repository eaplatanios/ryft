//! Contains machinery for representing and working with _reference views_ and paths layered on the structural
//! [`ReferenceAnalysis`]. The generic analysis records _that_ a reference-typed value is a narrowing view of its
//! [`ReferenceRoot`], through [`ReferenceAliasEdge`](crate::ReferenceAliasEdge)s of kind [`ReferenceAliasKind::View`],
//! but leaves the view itself to the value family (e.g., an array view is an index or slice, while a downstream family
//! may split a register into halves). [`ReferenceViewOperation`] defines these views for an operation family, and
//! [`ReferenceViewAnalysis`] composes the per-edge views into a [`ReferenceViewPath`] for every reference-typed value.
//! Transforms that rebuild references (e.g., for tangent, cotangent, and residual reconstruction) consult the view
//! analysis and reapply views through the same contract, so no transform ever matches view operations by name, and
//! downstream operations such as the array family's `reference_index` and `reference_slice` operations are not
//! special-cased anywhere.
//!
//! This module uses static dispatch on the operation family `O`. Views are owned data; validation and reapplication are
//! associated functions of the family, so the contract composes with the closed operation enums that backends own.
//!
//! # Symbols And Bindings
//!
//! A view may depend on values supplied to its [`Instruction`](crate::Instruction), such as a scalar array index. Its
//! [`ReferenceView::symbols`] function lists the positions of those inputs. Analysis binds each position to the
//! corresponding [`ValueId`]. The view and these bindings form a [`ReferenceViewStep`]. Static views have no symbols
//! and carry empty bindings. Index values created by an enclosing loop enter as ordinary region inputs, so analysis
//! does not need loop-specific symbols or names.
//!
//! The binding type is a parameter of the path because the same path shape serves consumers that close symbols
//! differently: the view analysis binds program identities, an eager handle carries only static steps and uses
//! the uninhabited [`NoReferenceViewBinding`], and a discharge policy may close symbols over destination values.
//!
//! # Validating Views Against Transformed References
//!
//! A view is validated against the _current_ source reference type before it is reapplied. A tangent or cotangent root
//! may have a different referent type from the primal root (e.g., a widened floating-point tangent type), so a view
//! that was valid on the primal root is re-checked against the transformed root rather than assumed to transfer.
//!
//! # Batching Moves The Axis Through The Mapping
//!
//! [`ReferenceViewOperation::reapply_reference_view`] rebuilds a view over a root with the same dimensions as the one
//! it was derived on, which is what tangent, cotangent, and residual reconstruction need. Batching is different in that
//! it inserts an axis into the packed root, and a primal view reapplied unchanged to a batched root would index or
//! slice the wrong axis. The contract therefore splits the two concerns. [`BatchableReferenceView::batch`] is pure axis
//! arithmetic on the view (i.e., given the packed source type and the source's batch axis, it returns the view that
//! selects the same part of each item of the packed source together with the batch axis of the derived reference).
//! The shared rule [`ReferenceViewOperation::batch`] then binds that batched view through
//! [`reapply_reference_view`](ReferenceViewOperation::reapply_reference_view) on the parent context, so every view
//! operation of every family batches through one rule and no operation carries the axis arithmetic itself.
//!
//! # Overlap Queries
//!
//! Two paths of one root may select the same part, provably disjoint parts, or parts whose overlap is not decidable
//! statically. [`ReferenceView::overlap`] answers that question for two paths starting from the same complete reference
//! as a [`ReferenceViewOverlap`]; [`ReferenceViewPath::overlap`] and [`ReferenceViewAnalysis::overlap`] expose it on
//! paths and on analyzed values. Equal symbol bindings identify the same value, but the views must also agree for the
//! selected parts to be identical. For example, array indexing can clamp the same index differently after two different
//! slices. The value family accounts for these semantics; generic consumers must treat
//! [`MayOverlap`](ReferenceViewOverlap::MayOverlap) conservatively. Proving that two views select disjoint parts does
//! not establish independent reference lifetimes or permit a transform to split one allocation into independently
//! updated states.
//!
//! # Region Boundaries
//!
//! A reference passed into an attached [`Region`] must be a complete root handle. The region receives any indices as
//! ordinary inputs and constructs views with instructions in that same region. A shared region therefore has one view
//! path expressed in its own input identities, independent of which instruction calls it.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use thiserror::Error;

use ryft_macros::Parameter;

use crate::batching::{BatchAxis, BatchedOutputs, BatchingContext, BatchingError, BatchingPolicy};
use crate::contexts::Context;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::effects::ReferenceAliasKind;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::references::analysis::{
    ReferenceAnalysis, ReferenceAnalysisError, ReferenceAnalysisTransformArguments, ReferenceRoot,
};
use crate::programs::regions::{Region, RegionRef};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId};

/// Error produced by [`ReferenceViewOperation::validate_reference_view`] when a view does not compose onto its source
/// reference type or does not derive the declared output reference type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceViewValidationError {
    /// The view composes onto the source but derives a referent type that differs from the declared one.
    #[error("view declares referent type `{actual}` but derives referent type `{expected}` from its source")]
    TypeMismatch {
        /// Referent type derived by the view from the source.
        expected: String,

        /// Referent type declared by the view output.
        actual: String,
    },

    /// The view cannot be applied to the source reference type at all.
    #[error("invalid view composition: {message}")]
    InvalidComposition {
        /// Description of why the view is invalid for the source.
        message: String,
    },
}

/// Error produced by [`ReferenceViewAnalysis`] when the generic reference analysis fails or when a derived view path
/// cannot be reconciled with the program's declared reference types. Conversion to [`ProgramError`] preserves an
/// underlying [`ReferenceAnalysisError`] through its typed conversion. View-specific failures are preserved through
/// [`ReferenceError::ViewAnalysis`](crate::ReferenceError::ViewAnalysis). Invalid views retain their
/// [`ReferenceViewValidationError`] as an error source; positions and symbols identify the exact
/// declaration that failed without duplicating the validation error's variants.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceViewAnalysisError {
    /// The structural reference analysis rejected the region and its attached computation regions.
    #[error(transparent)]
    Analysis(#[from] ReferenceAnalysisError),

    /// An operation declares a view for an output but supplies no view for it.
    #[error(
        "operation `{operation}` at {instruction} declares a reference view at output {output_index} \
         but describes no view"
    )]
    MissingView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output whose view is missing.
        output_index: usize,
    },

    /// A view cannot be applied to its source or derives a type different from the declared type.
    /// The underlying validation error retains the type mismatch or invalid-composition diagnostic.
    #[error("operation `{operation}` at {instruction} has an invalid view at output {output_index}: {source}")]
    InvalidView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output described by the invalid view.
        output_index: usize,

        /// Failure reported when validating the view against its source and declared output types.
        #[source]
        source: ReferenceViewValidationError,
    },

    /// A symbolic input position is out of range or names a reference rather than an index value.
    #[error(
        "operation `{operation}` at {instruction} describes a view at output {output_index} \
         through input {symbol}, but {message}"
    )]
    InvalidViewSymbol {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output whose view uses the invalid symbol.
        output_index: usize,

        /// Symbol that cannot be bound at this position.
        symbol: usize,

        /// Explanation of why the symbol cannot be bound here.
        message: String,
    },
}

impl From<ReferenceViewAnalysisError> for ProgramError {
    #[inline]
    fn from(error: ReferenceViewAnalysisError) -> Self {
        match error {
            ReferenceViewAnalysisError::Analysis(error) => error.into(),
            error => ProgramError::Reference(error.into()),
        }
    }
}

/// Uninhabited binding of [`ReferenceViewPath`]s that only ever carry static [`ReferenceViewStep`]s, such as the path
/// of an eager array reference handle. Every step of such paths has empty bindings; consumers must reject views that
/// require symbols because no binding value can be supplied for them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum NoReferenceViewBinding {}

/// Represents whether two views of the same reference allocation select separate parts, exactly the same part, or
/// potentially overlapping parts, as determined by [`ReferenceView::overlap`]. Both paths apply views starting from
/// the complete allocation. For example, `root[0]` and `root[1]` select different elements and are disjoint, while
/// `root[i]` and `root[j]` may overlap when the values of `i` and `j` are unknown. Each symbolic index in a path has a
/// binding identifying the program value that supplies it; that binding does not imply that the index's runtime value
/// is known.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceViewOverlap {
    /// The two paths _provably_ select disjoint parts of the root.
    Disjoint,

    /// The two paths _provably_ select exactly the same part of the root.
    Same,

    /// The two paths may overlap: they select intersecting parts, or a view depends on a symbol whose binding cannot
    /// prove the paths identical or disjoint.
    MayOverlap,
}

/// Metadata describing a view from a source reference to a derived reference in the same allocation. Views contain
/// metadata, such as an array axis and a static index or an input position naming a dynamic index. They do not contain
/// the reference allocation or the dynamic index value itself; [`ReferenceViewStep`] pairs a view with those index
/// values or their program identities.
///
/// The `'static`, [`Send`], and [`Sync`] bounds allow [`ReferenceViewAnalysis`] to be retained as type-erased metadata
/// in the region's transform cache (refer to [`RegionRef::transform`] for more information on that). In particular,
/// `'static` prevents views from borrowing temporary data; it does not require their instances to live forever.
/// Owned view metadata satisfies this bound. Equality supports revalidation against a fresh analysis, and hashing
/// lets paths serve as part of eager reference handles' identities.
pub trait ReferenceView: 'static + Clone + Debug + PartialEq + Eq + Hash + Send + Sync {
    /// Reference type family this [`ReferenceView`] addresses.
    type Type: Type;

    /// Returns the symbols that this [`ReferenceView`] depends on, in the order that their bindings
    /// and values are supplied to every consumer. Each symbol is a non-reference input's position in the
    /// [`Instruction`](crate::Instruction) that produces the view. For example, a dynamic indexing instruction with
    /// inputs `[reference, index]` uses symbol `1` to name its index input and returns `vec![1]`. That symbol is the
    /// input position, not the index's runtime value. A static index stored directly in the view needs no input
    /// binding, so a view containing only static views returns an empty vector.
    fn symbols(&self) -> Vec<usize>;

    /// Returns whether `lhs` and `rhs` select separate parts, exactly the same part, or potentially overlapping parts
    /// of the same reference allocation, without executing the program. Both paths are assumed to start from the
    /// complete allocation. For example, paths selecting `reference[0]` and `reference[1]` are disjoint, while paths
    /// selecting `reference[i]` and `reference[j]` may overlap when their indices are unknown. Equal symbol bindings
    /// identify the same program value. Different bindings do not prove that the runtime values differ. Comparing
    /// symbolic views must also account for the views themselves, including any clamping.
    ///
    /// Note that an empty path selects the complete allocation, two empty paths are considered
    /// [`Same`](ReferenceViewOverlap::Same), and an empty path may overlap with a path selecting only part of the
    /// allocation. Paths are validated when they are derived, and implementations may conservatively return
    /// [`MayOverlap`](ReferenceViewOverlap::MayOverlap) for a malformed path instead of failing.
    ///
    /// # Parameters
    ///
    ///   - `type`: [`Type`] of the complete reference from which both paths start.
    ///   - `lhs`: [`ReferenceViewStep`]s selecting the first part to compare, starting from the complete reference.
    ///   - `rhs`: [`ReferenceViewStep`]s selecting the second part to compare, starting from the complete reference.
    fn overlap(
        r#type: &Self::Type,
        lhs: &[ReferenceViewStep<Self>],
        rhs: &[ReferenceViewStep<Self>],
    ) -> ReferenceViewOverlap;
}

/// Optional batching capability for a [`ReferenceView`]. Implementations adjust a view when its source reference
/// gains a batch axis. Reference families that support analysis and discharge without batching need only implement
/// [`ReferenceView`] but batching rules additionally require this trait.
pub trait BatchableReferenceView: ReferenceView {
    /// Moves the batch axis of a source reference through this [`ReferenceView`] mapping. The batch axis of a reference
    /// is an axis of its packed referent that the per-item view never sees, so the batched view must select the same
    /// part of each item of the packed source that this view selects of the unbatched one, and the derived reference
    /// has its own batch axis. This is pure axis arithmetic: the symbols of the view are untouched, and a replicated
    /// `batch_axis` returns the view unchanged and replicated.
    ///
    /// # Parameters
    ///
    ///   - `source`: Packed reference [`Type`] of the batched source (i.e., the type with the batch axis inserted).
    ///   - `batch_axis`: Batch axis of the source, positioned in the packed referent of `source`.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] when this family cannot carry `batch_axis` through the view (e.g., a family without
    /// axes rejects every mapped axis, and a static array slice cannot span a dynamically sized batch axis).
    fn batch(&self, r#type: &Self::Type, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError>;
}

/// View contract of an operation family that specifies a view for every view alias the family can derive, its
/// type-level validation, and its reapplication to another reference with compatible dimensions. An operation family
/// implements this trait once, and every transform that rebuilds references (e.g., tangent, cotangent, and residual
/// reconstruction) then reaches the family's views through it: [`ReferenceViewAnalysis`] composes the views into
/// per-value [`ReferenceViewPath`]s, and reconstruction reapplies them step by step to the transformed root. A tangent
/// or cotangent root may have a different referent type from the primal root, so each view is validated against the
/// current transformed source type before it is reapplied. Batching does not reapply a primal view unchanged, because
/// a batched root has an extra axis; it first moves the batch axis through the view with
/// [`BatchableReferenceView::batch`] and then reapplies the batched view, which is what
/// [`ReferenceViewOperation::batch`] does for every view operation.
pub trait ReferenceViewOperation: Operation {
    /// View type of this family, addressing this family's reference types.
    type View: ReferenceView<Type = Self::Type>;

    /// Returns the view this operation derives at output `output_index`, or [`None`] when that output
    /// is not a reference view. Exactly the outputs whose [`effects`](Operation::effects) declare a
    /// [`ReferenceAlias`](crate::ReferenceAlias) of kind [`ReferenceAliasKind::View`] return [`Some`].
    /// An operation that declares such an alias but returns [`None`] is rejected by the view analysis with
    /// [`ReferenceViewAnalysisError::MissingView`]. Symbol positions refer to the operation's non-reference inputs.
    fn reference_view(&self, output_index: usize) -> Option<Self::View>;

    /// Validates that applying `view` to a reference of type `source` produces a reference of type `target`. Both
    /// arguments are reference types, so implementations inspect their referent types when checking view and type
    /// compatibility. This function does not validate symbol bindings or an entire [`ReferenceViewPath`]. A path is
    /// checked by validating each view against the preceding view's resulting reference type.
    ///
    /// This function accepts one view, but its representation is defined by the operation family. For example, an array
    /// view represents one indexing or slicing operation. Another family may represent several operations as one view.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceViewValidationError::InvalidComposition`] when `view` cannot be applied to `source` at all,
    /// and [`ReferenceViewValidationError::TypeMismatch`] when it derives a referent type other than the one `target`
    /// declares.
    fn validate_reference_view(
        view: &Self::View,
        source: &Self::Type,
        target: &Self::Type,
    ) -> Result<(), ReferenceViewValidationError>;

    /// Applies `view` to the reference `source` through `context` and returns the resulting reference view. The result
    /// selects part of the same allocation as `source`. It does not copy the referenced data into a new allocation. The
    /// context determines whether the operation executes eagerly or is staged into a program.
    ///
    /// This function lets transforms recreate a view on another reference, such as a tangent or cotangent buffer. For
    /// example, to recreate `reference[index]` on a cotangent buffer, pass that buffer as `source` and the value of
    /// `index` in the reconstruction context as the only entry of `symbols`. A symbolic input position of `1` in the
    /// view still takes its value from `symbols[0]`. This slice contains only the symbol values, not all instruction
    /// inputs or their program identities. A static view needs an empty slice.
    ///
    /// This function reapplies one view, not an entire [`ReferenceViewPath`]. To reconstruct a path, apply each view to
    /// the preceding result. When a caller requires a particular target type, it must first check compatibility using
    /// [`validate_reference_view`](Self::validate_reference_view); this function receives no expected target type.
    ///
    /// # Parameters
    ///
    ///   - `context`: [`Context`] through which the reference-view operation is bound.
    ///   - `view`: [`ReferenceView`] to apply to `source`.
    ///   - `source`: Reference to select from, represented as a value in `context`.
    ///   - `symbols`: Values in `context` supplying the view's symbolic inputs, one per entry returned by
    ///     [`ReferenceView::symbols`] and in that order. Their types must satisfy the view operation's requirements;
    ///     for example, a dynamic array index requires a scalar integer value.
    ///
    /// # Errors
    ///
    /// Returns an error when the number of symbol values is incorrect or the view cannot be applied to the supplied
    /// reference and symbol values. Propagates errors from binding the operation through `context`.
    fn reapply_reference_view<C: Context<Type = Self::Type, Operation = Self>>(
        context: &C,
        view: &Self::View,
        source: C::Value,
        symbols: &[C::Value],
    ) -> Result<C::Value, ProgramError>;

    /// Batches this operation by reconstructing its reference views in the parent of `context`. This shared rule
    /// supports operations whose outputs are all views of one input reference, with no observable effects or attached
    /// regions. Every other input must be replicated across the batch. For example, indexing a batch of references at
    /// one shared index is supported; using a different index for each item is not.
    ///
    /// The rule adjusts each view through [`BatchableReferenceView::batch`], reapplies it through
    /// [`reapply_reference_view`](Self::reapply_reference_view), and returns the results in output order with their
    /// batch axes. Operations with multiple view outputs are reconstructed once per output. Member operation rules
    /// first convert their operation into the enclosing operation family and invoke this function on that value.
    /// Reference families that do not support batching need not implement [`BatchableReferenceView`].
    ///
    /// # Parameters
    ///
    ///   - `context`: [`BatchingContext`] whose parent is used to reconstruct the views.
    ///   - `inputs`: Operation inputs in input order, each carrying its value and batch axis according to `P`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchingError::UnsupportedOperation`] when the operation has no views, uses multiple source
    /// references, has other outputs, observable effects, or attached regions, or receives a mapped non-source
    /// input. Malformed input positions or missing views produce a [`ProgramError::MalformedProgram`] wrapped in
    /// [`BatchingError`]. Propagates errors from type inference, view batching, reapplication, and batch construction.
    fn batch<C: Context<Type = Self::Type, Operation = Self>, P: BatchingPolicy<C>>(
        &self,
        context: &BatchingContext<C, P>,
        inputs: &[P::Batch],
    ) -> Result<BatchedOutputs<C, P>, BatchingError>
    where
        Self::View: BatchableReferenceView,
    {
        let name = self.name();
        let effects = self.effects();
        let not_a_view = |output_index: usize| BatchingError::UnsupportedOperation {
            message: format!(
                "`{name}` has reference output {output_index} that is not a view, so it cannot batch as a \
                 view operation",
            ),
        };

        // A new allocation cannot be reconstructed by reapplying views of an existing input.
        if let Some(output_index) = effects.allocation_output_indices().next() {
            return Err(not_a_view(output_index));
        }

        // Collect the view outputs while checking that every alias is a view of the same input. Invalid aliases must
        // produce an error rather than being filtered out, since dropping them would change the operation's outputs.
        let (source_index, mut output_indices) = effects.reference_aliases().iter().try_fold(
            (None, Vec::with_capacity(effects.reference_aliases().len())),
            |(source_index, mut output_indices), alias| {
                let (output_index, input_index) = (alias.output_index(), alias.input_index());
                if alias.kind() != ReferenceAliasKind::View {
                    return Err(not_a_view(output_index));
                }
                if let Some(source_index) = source_index
                    && source_index != input_index
                {
                    return Err(BatchingError::UnsupportedOperation {
                        message: format!(
                            "`{name}` views inputs {source_index} and {input_index}, but a view operation \
                             views one source",
                        ),
                    });
                }
                output_indices.push(output_index);
                Ok((Some(input_index), output_indices))
            },
        )?;

        let Some(source_index) = source_index else {
            return Err(BatchingError::UnsupportedOperation { message: format!("`{name}` derives no reference view") });
        };

        // Replaying views preserves only the views themselves. Reject effects and attached computations whose
        // execution would otherwise be silently dropped by this rule.
        if effects.summary().has_observable_effects_when_unused() || !self.region_slots().is_empty() {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{name}` has effects or attached regions that cannot be preserved by batching only its \
                     reference views",
                ),
            });
        }

        // Return values must retain the operation's output order. Gaps identify outputs this rule cannot reconstruct.
        output_indices.sort_unstable();
        if output_indices.iter().enumerate().any(|(position, output_index)| position != *output_index) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{name}` has outputs other than its reference views"),
            });
        }

        // Resolve the shared source, then require all other inputs to have one value shared across the batch.
        // In particular, independently varying indices would need a different rule for constructing reference views.
        let Some(source) = inputs.get(source_index) else {
            return Err(ProgramError::MalformedProgram(format!(
                "`{name}` views input {source_index} but was applied to {} inputs",
                inputs.len(),
            ))
            .into());
        };

        if let Some((input_index, _)) = inputs
            .iter()
            .enumerate()
            .find(|(index, input)| *index != source_index && !P::batch_axis(input).is_replicated())
        {
            return Err(BatchingError::UnsupportedOperation {
                message: format!(
                    "`{name}` requires input {input_index} to be replicated; batching a reference view \
                     through a mapped index input is not supported",
                ),
            });
        }

        // Alias positions alone do not reveal a trailing non-reference output. Check the complete per-item signature
        // before emitting any views so the reconstructed outputs preserve the operation's full boundary.
        let input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
        let output_types = self.infer_output_types(&input_types, &[])?;
        if output_types.len() != output_indices.len() || output_types.iter().any(|r#type| !r#type.is_reference()) {
            return Err(BatchingError::UnsupportedOperation {
                message: format!("`{name}` has outputs other than its reference views"),
            });
        }

        // Each view is adjusted for the packed source type, then reapplied in the parent context. The returned batch
        // axis describes where the new reference carries its batch dimension after indexing or slicing.
        let source_value = P::value(source);
        let source_axis = P::batch_axis(source);
        let source_type = source_value.r#type();
        let outputs = output_indices
            .into_iter()
            .map(|output_index| {
                let Some(view) = self.reference_view(output_index) else {
                    return Err(ProgramError::MalformedProgram(format!(
                        "operation `{name}` derives a reference view at output {output_index} but exposes no \
                         view transform",
                    ))
                    .into());
                };

                // Symbol positions name instruction inputs; reapplication receives only their values, in symbol order.
                let symbols = view
                    .symbols()
                    .into_iter()
                    .map(|input_index| {
                        inputs.get(input_index).map(|input| P::value(input).clone()).ok_or_else(|| {
                            BatchingError::from(ProgramError::MalformedProgram(format!(
                                "`{}` describes output {} through input {} but was applied to {} inputs",
                                name,
                                output_index,
                                input_index,
                                inputs.len(),
                            )))
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let (view, output_axis) = view.batch(source_type.as_ref(), source_axis)?;
                let value =
                    Self::reapply_reference_view(context.parent(), &view, source_value.clone(), symbols.as_slice())?;
                P::batch(value, output_axis)
            })
            .collect::<Result<Vec<_>, BatchingError>>()?;

        Ok(outputs.into())
    }
}

/// A [`ReferenceView`] view together with bindings for the values it depends on. [`ReferenceViewPath`]s compose these
/// steps from a root reference to a derived reference.
///
/// `View` is the view type and must implement [`ReferenceView`]. For example,
/// [`ArrayReferenceView`](crate::ArrayReferenceView) describes indexing one axis or slicing an array. It stores
/// metadata and not the referenced array. A dynamic index is represented in the view by the position of the instruction
/// input supplying that index.
///
/// `Binding` is the type used to represent each such input. It is set to [`ValueId`] during program analysis, or a
/// context value during reference discharge. The `bindings` vector contains one entry per symbol reported by
/// [`ReferenceView::symbols`], in that order. Static views have no symbols and carry an empty vector.
///
/// For example, an instruction selecting `root[index]` has inputs `[root, index]`. Its array view is `Index { axis: 0,
/// index: Symbolic(1) }`: `1` names the instruction's second input, not the array element to select. During analysis,
/// the step binds that symbol to the [`ValueId`] of `index`. During discharge, it instead binds the symbol to the
/// context value representing `index`, which can be passed directly to a dynamic slice operation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceViewStep<View: ReferenceView, Binding = ValueId> {
    /// [`ReferenceView`] of this [`ReferenceViewStep`].
    view: View,

    /// Binding of each symbol of the [`ReferenceView`], in the order returned by [`ReferenceView::symbols`].
    bindings: Vec<Binding>,
}

impl<View: ReferenceView, Binding> ReferenceViewStep<View, Binding> {
    /// Returns the [`ReferenceView`] of this [`ReferenceViewStep`].
    #[inline]
    pub fn view(&self) -> &View {
        &self.view
    }

    /// Returns the binding of each symbol of the [`ReferenceView`], in the order returned
    /// by [`ReferenceView::symbols`].
    #[inline]
    pub fn bindings(&self) -> &[Binding] {
        self.bindings.as_slice()
    }
}

/// Sequence of [`ReferenceViewStep`] from a reference root to one derived reference, in the order they are applied.
/// `View` is the type of each view, such as [`ArrayReferenceView`](crate::ArrayReferenceView), and `Binding` represents
/// the inputs needed by a symbolic view. Each [`ReferenceViewStep`] pairs a `View` with a vector of `Binding`s. The
/// path stores these steps, but neither the root allocation nor its identity; [`ReferenceAnalysis`] identifies the root
/// when analyzing a [`Program`](crate::Program).
///
/// For example, selecting `root[row][column]` produces two steps: the first selects a row from the root, and the second
/// selects an element from that row. With `View = ArrayReferenceView` and `Binding = ValueId`, the steps describe the
/// two indexing operations and store the program identities of `row` and `column`. During discharge, `Binding =
/// C::Value` stores their values in the reconstruction context instead, so the same path traversal can reapply the
/// views without looking up source program identities.
///
/// Eager reference handles resolve indices immediately into static views and use [`NoReferenceViewBinding`]. Their
/// steps have empty binding vectors. See [`ReferenceViewStep`] for how an instruction input position in a view
/// corresponds to a binding.
///
/// The empty path denotes the complete root. Complete root handles, capture constants, and forwarded complete
/// references carry it. Equality and hashing compare step views and bindings, not the identities of the reference
/// handles or the array elements selected by different step sequences.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceViewPath<View: ReferenceView, Binding = ValueId> {
    /// [`ReferenceViewStep`]s in this [`ReferenceViewPath`] in the order they are applied,
    /// starting from the complete reference.
    steps: Vec<ReferenceViewStep<View, Binding>>,
}

impl<View: ReferenceView, Binding> ReferenceViewPath<View, Binding> {
    /// Returns the empty [`ReferenceViewPath`] denoting the complete root.
    pub const fn root() -> Self {
        Self { steps: Vec::new() }
    }

    /// Returns whether this [`ReferenceViewPath`] denotes the complete root (i.e., whether it is empty).
    #[inline]
    pub fn is_root(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns the [`ReferenceViewStep`]s in this [`ReferenceViewPath`] in the order they are applied,
    /// starting from the complete reference.
    #[inline]
    pub fn steps(&self) -> &[ReferenceViewStep<View, Binding>] {
        self.steps.as_slice()
    }

    /// Returns the ordered [`ReferenceView`]s applied from the root outward, without their bindings.
    #[inline]
    pub fn views(&self) -> impl ExactSizeIterator<Item = &View> + DoubleEndedIterator {
        self.steps.iter().map(ReferenceViewStep::view)
    }

    /// Returns a copy of this [`ReferenceViewPath`] extended by one more [`ReferenceViewStep`] applied to its current
    /// end, binding the symbols of `view` to `bindings`. The caller must supply one binding per symbol in the order
    /// returned by [`ReferenceView::symbols`]. This generic container does not validate the view or its bindings.
    pub fn with_step(&self, view: View, bindings: Vec<Binding>) -> Self
    where
        Binding: Clone,
    {
        let mut steps = Vec::with_capacity(self.steps.len() + 1);
        steps.extend(self.steps.iter().cloned());
        steps.push(ReferenceViewStep { view, bindings });
        Self { steps }
    }

    /// Returns a copy of this [`ReferenceViewPath`] extended by one more static [`ReferenceView`] applied to its
    /// current end. This is the shorthand of [`with_step`](Self::with_step) with no bindings. The caller must ensure
    /// that `view` requires no symbols.
    #[inline]
    pub fn with_view(&self, view: View) -> Self
    where
        Binding: Clone,
    {
        self.with_step(view, Vec::new())
    }
}

impl<View: ReferenceView> ReferenceViewPath<View, ValueId> {
    /// Returns the [`ReferenceViewOverlap`] between the parts this [`ReferenceViewPath`] and `other` select of one
    /// root of type `root`, through [`ReferenceView::overlap`]. Both paths must start from the same complete reference.
    /// Callers that compare analyzed values of one region use [`ReferenceViewAnalysis::overlap`], which checks the
    /// roots first, while this function serves callers that resolve roots across namespaces themselves.
    #[inline]
    pub fn overlap(&self, other: &Self, root: &View::Type) -> ReferenceViewOverlap {
        View::overlap(root, self.steps(), other.steps())
    }
}

impl<View: ReferenceView, Binding> Default for ReferenceViewPath<View, Binding> {
    #[inline]
    fn default() -> Self {
        Self::root()
    }
}

/// Structural [`ReferenceAnalysis`] of a [`Region`] closure together with the [`ReferenceViewPath`] of every
/// reference-typed value in that closure, derived through the [`ReferenceViewOperation`] contract of the closure's
/// operation family. Every reference-typed value has exactly one path. A root handle (i.e., a region input, an
/// allocation, a capture constant, or a forwarded region output) has the empty [`ReferenceViewPath`], an identity alias
/// copies the path of its source, and a view alias copies the path of its source and appends the view its producing
/// operation reports for that edge's output, after that view was validated against the source and output reference
/// types. Nested region inputs are separate roots of the structural analysis and carry empty paths. Instructions in
/// each region create views of those roots, with any symbolic input positions bound to values in that region.
///
/// The view analysis is retained in the region's transform cache under exactly the cache identity of the structural
/// analysis (refer to the documentation of [`RegionRef::reference_view_analysis`] for more information) and shares
/// that analysis through an [`Arc`] rather than re-deriving it. Transforms and other consumers invoke this validation
/// explicitly on their [`Program`](crate::Program)s.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceViewAnalysis<View: ReferenceView> {
    /// Structural [`ReferenceAnalysis`] of the [`Region`] closure.
    analysis: Arc<ReferenceAnalysis>,

    /// [`ReferenceViewPath`] of every reference-typed value of the [`Region`] closure, in canonical [`ValueId`] order.
    paths: BTreeMap<ValueId, ReferenceViewPath<View>>,
}

impl<View: ReferenceView> ReferenceViewAnalysis<View> {
    /// Analyzes the complete [`Region`] closure of `region` and derives the [`ReferenceViewPath`] of every
    /// reference-typed value in it, returning the resulting [`ReferenceViewAnalysis`]. The structural analysis is
    /// obtained through [`RegionRef::reference_analysis`], and so it is shared with every other consumer of the same
    /// closure. This function itself represents the uncached derivation of the view analysis, and
    /// [`RegionRef::reference_view_analysis`] is its retained counterpart. Refer to the documentation
    /// of [`RegionRef::reference_analysis`] for the meaning of `capture_count`.
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceAnalysisError`] of the structural analysis when the closure violates the reference model,
    /// and otherwise the first path derivation failure in canonical value order: an operation declaring a view alias
    /// without describing it, a view that is invalid for its source, or a declared output referent that differs from
    /// the derived one.
    #[inline]
    pub fn new<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        capture_count: usize,
    ) -> Result<Self, ReferenceViewAnalysisError> {
        Self::new_with_arguments(
            region,
            &ReferenceAnalysisTransformArguments::new(region, Vec::new(), Some(capture_count), false),
        )
    }

    /// Derives a [`ReferenceViewAnalysis`] exactly like [`new`](Self::new), obtaining the structural analysis under
    /// the already-derived cache key `arguments` so that the [`Region`] closure is walked only once per derivation.
    fn new_with_arguments<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        arguments: &ReferenceAnalysisTransformArguments,
    ) -> Result<Self, ReferenceViewAnalysisError> {
        // Reuse structural analysis to establish roots, alias edges, and valid reference lifetimes before building
        // the view paths. It also guarantees that following alias sources cannot form a cycle.
        let analysis = region.reference_analysis_impl(arguments)?;
        let mut paths: BTreeMap<ValueId, ReferenceViewPath<View>> = BTreeMap::new();
        for value in analysis.values() {
            // Atom identifiers need not follow dependency order. Follow unresolved alias sources until reaching a root
            // or an existing path, then process the collected values in reverse. This avoids recursion for long alias
            // chains and lets each value reuse the path already recorded for its source.
            let mut pending = Vec::new();
            let mut source = value;
            while !paths.contains_key(&source) {
                pending.push(source);
                let Some(edge) = analysis.alias(source) else {
                    break;
                };
                source = edge.source();
            }

            // Sources are installed first, so each alias can copy its source path without another traversal.
            for value in pending.into_iter().rev() {
                let path = match analysis.alias(value) {
                    None => {
                        // A value without an alias edge denotes the complete reference.
                        ReferenceViewPath::root()
                    }
                    Some(edge) => {
                        // Identity aliases retain the source path while view aliases append their own view.
                        let mut path = paths[&edge.source()].clone();
                        if edge.kind() == ReferenceAliasKind::View {
                            let id = edge.instruction();
                            let output_index = edge.output_index();
                            let source = edge.source();

                            // Structural analysis resolved both values and the instruction before recording this edge.
                            let current = region.with_id(id.region()).unwrap();
                            let instruction = &current.instructions()[id.index()];
                            let operation = instruction.operation();
                            let name = operation.name();

                            // Every declared view alias must supply a view before its types
                            // or symbolic inputs can be checked.
                            let view = operation.reference_view(output_index).ok_or(
                                ReferenceViewAnalysisError::MissingView {
                                    operation: name,
                                    instruction: id,
                                    output_index,
                                },
                            )?;

                            // Check that applying the view to its source produces the reference type
                            // declared by this output.
                            let atoms = current.atoms();
                            O::validate_reference_view(
                                &view,
                                current.atoms()[source.atom().index()].r#type().as_ref(),
                                region.with_id(value.region()).unwrap().atoms()[value.atom().index()].r#type().as_ref(),
                            )
                            .map_err(|source| ReferenceViewAnalysisError::InvalidView {
                                operation: name,
                                instruction: id,
                                output_index,
                                source,
                            })?;

                            // Resolve each symbolic input position to the ordinary program value supplied to this
                            // instruction. A reference cannot supply a scalar index; the operation family validates
                            // the remaining index type rules.
                            let inputs = instruction.inputs();
                            let bindings = view
                                .symbols()
                                .into_iter()
                                .map(|input_index| {
                                    let Some(atom) = inputs.get(input_index) else {
                                        return Err(ReferenceViewAnalysisError::InvalidViewSymbol {
                                            operation: name,
                                            instruction: id,
                                            output_index,
                                            symbol: input_index,
                                            message: format!("the instruction has only {} inputs", inputs.len()),
                                        });
                                    };
                                    if atoms[atom.index()].r#type().is_reference() {
                                        return Err(ReferenceViewAnalysisError::InvalidViewSymbol {
                                            operation: name,
                                            instruction: id,
                                            output_index,
                                            symbol: input_index,
                                            message: "that input is a reference rather than an index value".to_string(),
                                        });
                                    }
                                    Ok(ValueId::new(id.region(), *atom))
                                })
                                .collect::<Result<Vec<_>, ReferenceViewAnalysisError>>()?;

                            // Record the validated view and its bindings after the source's steps.
                            path.steps.push(ReferenceViewStep { view, bindings });
                        }
                        path
                    }
                };

                paths.insert(value, path);
            }
        }

        Ok(Self { analysis, paths })
    }

    /// Returns the structural [`ReferenceAnalysis`] of the [`Region`] closure.
    #[inline]
    pub fn analysis(&self) -> &ReferenceAnalysis {
        &self.analysis
    }

    /// Returns the [`ReferenceViewPath`] of every reference-typed value of the [`Region`] closure,
    /// in canonical [`ValueId`] order.
    #[inline]
    pub fn paths(&self) -> impl '_ + Iterator<Item = (ValueId, &ReferenceViewPath<View>)> {
        self.paths.iter().map(|(value, path)| (*value, path))
    }

    /// Returns the [`ReferenceViewPath`] from the root of the reference-typed `value` to the part it selects, or
    /// [`None`] when `value` is not a reference-typed value of the [`Region`] closure. Root handles carry the empty
    /// path, and so does a nested region input forwarded as a complete handle.
    #[inline]
    pub fn path(&self, value: ValueId) -> Option<&ReferenceViewPath<View>> {
        self.paths.get(&value)
    }

    /// Returns the [`ReferenceViewOverlap`] between the views represented by `lhs` and `rhs`, or [`None`] when
    /// either is not a reference-typed value of the closure or the values belong to different [`Region`]s. Roots are
    /// region-relative (a nested region input is a root of its own namespace even when it carries a caller root), so
    /// only values of one region have comparable roots: values of different roots are
    /// [`Disjoint`](ReferenceViewOverlap::Disjoint), and values of one root delegate to [`ReferenceView::overlap`] with
    /// the type of the root's defining atom, read from `region`, the closure this view analysis was derived for. The
    /// view analysis retains no types itself, because the region's transform cache holds it behind a `Send + Sync`
    /// erasure that the value family's type is not required to satisfy. Callers that compare paths across namespaces
    /// resolve the roots themselves and use [`ReferenceViewPath::overlap`].
    pub fn overlap<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        &self,
        region: RegionRef<'_, V, O>,
        lhs: ValueId,
        rhs: ValueId,
    ) -> Option<ReferenceViewOverlap>
    where
        View: ReferenceView<Type = V::Type>,
    {
        if lhs.region() != rhs.region() {
            return None;
        }

        let root = self.analysis.root_of(lhs)?;
        if root != self.analysis.root_of(rhs)? {
            return Some(ReferenceViewOverlap::Disjoint);
        }

        // Every root is defined exactly once, by the input atom or the allocating instruction output that names it,
        // and every reference-typed value has `lhs` path, so once both roots resolved the remaining lookups cannot fail
        // for the region this view analysis was derived for.
        let current = region.with_id(root.region()).ok()?;
        let atom = match root {
            ReferenceRoot::RegionInput { input_index, .. } => *current.input_ids().get(input_index)?,
            ReferenceRoot::Constant { value } => value.atom(),
            ReferenceRoot::Allocation { instruction, output_index } => {
                *current.instructions().get(instruction.index())?.outputs().get(output_index)?
            }
        };
        let root_type = current.atoms().get(atom.index())?.r#type();
        Some(self.paths[&lhs].overlap(&self.paths[&rhs], root_type.as_ref()))
    }
}

impl<'r, V: Value, O: ReferenceViewOperation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Returns the [`ReferenceViewAnalysis`] of this [`Region`]'s closure, retained in the region's transform cache
    /// under exactly the cache identity of [`reference_analysis`](Self::reference_analysis): the same `capture_count`
    /// and closure region identifiers key both, so a topology-preserving import that renumbers regions derives its own
    /// view analysis instead of being served paths keyed by another arena's identifiers, and repeated analysis of an
    /// unmoved region hits. The view analysis shares the retained structural analysis rather than deriving a second
    /// one. Refer to the documentation of [`ReferenceViewAnalysis::new`] for the view analysis itself; that function
    /// remains the uncached path.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading inputs of this region that originate in a lifted capture table.
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceViewAnalysisError`] naming the first violated rule. A failed analysis is not retained.
    pub fn reference_view_analysis(
        self,
        capture_count: usize,
    ) -> Result<Arc<ReferenceViewAnalysis<O::View>>, ReferenceViewAnalysisError> {
        let arguments = ReferenceAnalysisTransformArguments::new(self, Vec::new(), Some(capture_count), false);
        let artifact = self.transform::<ReferenceViewAnalysisTransform, _, ReferenceViewAnalysisError>(
            arguments,
            |region, arguments| {
                Ok(TransformArtifact::new(
                    Vec::new(),
                    Arc::new(ReferenceViewAnalysis::new_with_arguments(region, arguments)?),
                ))
            },
        )?;
        let (programs, analysis) = artifact.into_parts();
        assert!(programs.is_empty(), "reference view analysis transform retained a program");
        Ok(analysis)
    }
}

/// [`Region`] [`Transform`] marker for retained [`ReferenceViewAnalysis`] artifacts.
struct ReferenceViewAnalysisTransform;

impl<V: Value, O: ReferenceViewOperation<Type = V::Type>> Transform<Region<V, O>> for ReferenceViewAnalysisTransform {
    type Arguments = ReferenceAnalysisTransformArguments;
    type Artifact = TransformArtifact<V, O, Arc<ReferenceViewAnalysis<O::View>>>;

    const DEFAULT_CACHE_CAPACITY: usize = 8;
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::collections::HashMap;
    use std::error::Error as _;

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue, ArrayReferenceView,
        ArrayReferenceViewIndex, ArraySliceAxis, ArrayType, DataType, DimensionBounds, DimensionType, DimensionValue,
        DimensionVariable, REFERENCE_INDEX_OPERATION_NAME, ReferenceDynamicIndexOperation, ReferenceIndexOperation,
        ReferenceSliceOperation, reapply_array_reference_view,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{
        ConditionOperation, ReferenceFreezeOperation, ReferenceNew, ReferenceNewOperation, ReferenceRead,
        ReferenceReadOperation, ReferenceWriteOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceAlias, ReferenceEffect};
    use crate::programs::instructions::Instruction;
    use crate::programs::programs::Program;
    use crate::programs::references::analysis::ReferenceAliasEdge;
    use crate::programs::references::discharge::ReferenceSource;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{RegionId, RegionInterface};
    use crate::programs::types::TypeError;
    use crate::tracing::TracingContext;

    use super::*;

    /// Concrete array and reference values used by the test programs.
    type TestValue = ArrayIrValue<Array>;

    /// Existing array-IR operations used by the test programs.
    type TestOperation = ArrayIrOperation<Array>;

    /// Builder for test programs using existing array-IR operations.
    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    /// Test program with flat input and output parameters.
    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    /// Array view path whose symbols are bound to program values.
    type TestPath = ReferenceViewPath<ArrayReferenceView>;

    /// Returns a reference type over a statically shaped `f32` array.
    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    /// Returns a static index view for the given axis and index.
    fn index(axis: usize, index: usize) -> ArrayReferenceView {
        ArrayReferenceView::Index { axis, index: ArrayReferenceViewIndex::Static(index) }
    }

    /// Builds `f(matrix: ref<f32[2, 3]>) = read(matrix[0:1, 0:3][0])`, a two-step view chain over one root.
    fn chain_program() -> TestProgram {
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(reference_type([2, 3]));
        let axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let slice =
            builder.add_instruction(ReferenceSliceOperation::new(axes), Vec::new(), vec![matrix], None).unwrap()[0];
        let row =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![slice], None).unwrap()[0];
        let snapshot = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![row], None).unwrap()[0];
        builder.build(vec![snapshot], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Array-IR operations with explicit view contracts for testing analysis and shared batching validation.
    #[derive(Clone, Debug)]
    enum TestViewOperation {
        /// An existing array-IR operation used to construct or read a view.
        Native(TestOperation),

        /// A view whose symbol names an instruction input, including invalid positions for validation tests.
        Symbolic(usize),

        /// A declared view alias without the corresponding view.
        MissingView,

        /// Two index views, with the second view optionally referring to a different input.
        Pair { second_input: usize },

        /// A view accompanied by either a source read or an extra non-reference output.
        AdditionalBehavior { reads: bool },
    }

    impl TestViewOperation {
        /// Returns the view used by the symbolic test operation.
        fn view(symbol: usize) -> ArrayReferenceView {
            ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(symbol) }
        }
    }

    impl Operation for TestViewOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Native(operation) => operation.name(),
                Self::Symbolic(_) => "symbolic_view",
                Self::MissingView => "missing_view",
                Self::Pair { .. } => "view_pair",
                Self::AdditionalBehavior { .. } => "additional_behavior",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            match self {
                Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::MissingView => Ok(vec![input_types[0].clone()]),
                Self::Pair { second_input } => [0, *second_input]
                    .into_iter()
                    .enumerate()
                    .map(|(index, input)| {
                        ReferenceIndexOperation::new(0, index)
                            .infer_output_types(&input_types[input..=input], &[])
                            .map(|outputs| outputs.into_iter().next().unwrap())
                    })
                    .collect(),
                Self::AdditionalBehavior { reads } => {
                    let mut outputs = ReferenceIndexOperation::new(0, 0).infer_output_types(input_types, &[])?;
                    if !reads {
                        outputs.push(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
                    }
                    Ok(outputs)
                }
                Self::Symbolic(symbol) => {
                    let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
                    Ok(vec![ReferenceType::new(Self::view(*symbol).output_type(reference.referent())?).into()])
                }
            }
        }

        fn effects(&self) -> Cow<'_, Effects> {
            match self {
                Self::Native(operation) => operation.effects(),
                Self::AdditionalBehavior { reads } => Cow::Owned(
                    Effects::new(
                        EffectClasses::NONE,
                        if *reads {
                            vec![ReferenceEffect::Access { input_index: 0, mode: ReferenceAccessMode::Read }]
                        } else {
                            Vec::new()
                        },
                        vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)],
                    )
                    .unwrap(),
                ),
                Self::Pair { second_input } => Cow::Owned(
                    Effects::new(
                        EffectClasses::NONE,
                        Vec::new(),
                        vec![
                            ReferenceAlias::new(0, 0, ReferenceAliasKind::View),
                            ReferenceAlias::new(1, *second_input, ReferenceAliasKind::View),
                        ],
                    )
                    .unwrap(),
                ),
                Self::Symbolic(_) | Self::MissingView => Cow::Owned(
                    Effects::new(
                        EffectClasses::NONE,
                        Vec::new(),
                        vec![ReferenceAlias::new(0, 0, ReferenceAliasKind::View)],
                    )
                    .unwrap(),
                ),
            }
        }
    }

    impl From<ReferenceIndexOperation> for TestViewOperation {
        fn from(operation: ReferenceIndexOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl From<ReferenceDynamicIndexOperation> for TestViewOperation {
        fn from(operation: ReferenceDynamicIndexOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl From<ReferenceSliceOperation> for TestViewOperation {
        fn from(operation: ReferenceSliceOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl ReferenceViewOperation for TestViewOperation {
        type View = ArrayReferenceView;

        fn reference_view(&self, output_index: usize) -> Option<ArrayReferenceView> {
            match self {
                Self::Native(operation) => operation.reference_view(output_index),
                Self::Pair { .. } if output_index < 2 => Some(index(0, output_index)),
                Self::Symbolic(symbol) if output_index == 0 => Some(Self::view(*symbol)),
                Self::AdditionalBehavior { .. } if output_index == 0 => Some(index(0, 0)),
                Self::Symbolic(_) | Self::MissingView | Self::Pair { .. } | Self::AdditionalBehavior { .. } => None,
            }
        }

        fn validate_reference_view(
            view: &ArrayReferenceView,
            source: &ArrayIrType,
            target: &ArrayIrType,
        ) -> Result<(), ReferenceViewValidationError> {
            TestOperation::validate_reference_view(view, source, target)
        }

        fn reapply_reference_view<C: Context<Type = ArrayIrType, Operation = Self>>(
            context: &C,
            view: &ArrayReferenceView,
            source: C::Value,
            symbols: &[C::Value],
        ) -> Result<C::Value, ProgramError> {
            reapply_array_reference_view(context, view, source, symbols)
        }
    }

    /// Builds `f(vector: ref<f32[2]>, index: i64) = read(symbolic_view(vector, index))`, whose view describes its index
    /// through `symbol`.
    fn symbolic_view_program(symbol: usize) -> Program<TestValue, TestViewOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, TestViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let index_value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let view = builder
            .add_instruction(TestViewOperation::Symbolic(symbol), Vec::new(), vec![vector, index_value], None)
            .unwrap()[0];
        let snapshot = builder
            .add_instruction(
                TestViewOperation::Native(ReferenceReadOperation::new().into()),
                Vec::new(),
                vec![view],
                None,
            )
            .unwrap()[0];
        builder.build(vec![snapshot], vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    #[test]
    fn test_reference_view_validation_error() {
        assert_eq!(
            ReferenceViewValidationError::TypeMismatch { expected: "f32[3]".to_string(), actual: "f32[2]".to_string() }
                .to_string(),
            "view declares referent type `f32[2]` but derives referent type `f32[3]` from its source",
        );
        assert_eq!(
            ReferenceViewValidationError::InvalidComposition { message: "axis 2 is out of bounds".to_string() }
                .to_string(),
            "invalid view composition: axis 2 is out of bounds",
        );
    }

    #[test]
    fn test_reference_view_analysis_error() {
        let analysis =
            ReferenceAnalysisError::InvalidReferenceConstant { region: RegionId::new(0), atom: AtomId::new(1) };
        assert_eq!(
            ReferenceViewAnalysisError::from(analysis.clone()),
            ReferenceViewAnalysisError::Analysis(analysis.clone()),
        );
        assert_eq!(
            ProgramError::from(ReferenceViewAnalysisError::Analysis(analysis.clone())),
            ProgramError::from(analysis.clone()),
        );
        assert_eq!(
            ReferenceViewAnalysisError::Analysis(analysis).to_string(),
            "region ^0 stores reference-typed constant %1 that names no capture; references enter a program only \
             through inputs and captures",
        );
        assert_eq!(
            ReferenceViewAnalysisError::MissingView {
                operation: "view",
                instruction: InstructionId::new(RegionId::new(0), 2),
                output_index: 0
            }
            .to_string(),
            "operation `view` at ^0[2] declares a reference view at output 0 but describes no view",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidView {
                operation: "reference_index",
                instruction: InstructionId::new(RegionId::new(0), 2),
                output_index: 0,
                source: ReferenceViewValidationError::TypeMismatch {
                    expected: "f32[3]".to_string(),
                    actual: "f32[2]".to_string(),
                },
            }
            .to_string(),
            "operation `reference_index` at ^0[2] has an invalid view at output 0: view declares referent type `f32[2]` \
             but derives referent type `f32[3]` from its source",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidView {
                operation: "reference_index",
                instruction: InstructionId::new(RegionId::new(0), 2),
                output_index: 0,
                source: ReferenceViewValidationError::InvalidComposition {
                    message: "reference index axis 2 is out of bounds for rank 2".to_string(),
                },
            }
            .to_string(),
            "operation `reference_index` at ^0[2] has an invalid view at output 0: invalid view composition: reference \
             index axis 2 is out of bounds for rank 2",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidViewSymbol {
                operation: "symbolic_view",
                instruction: InstructionId::new(RegionId::new(0), 2),
                output_index: 0,
                symbol: 3,
                message: "the instruction has only 2 inputs".to_string(),
            }
            .to_string(),
            "operation `symbolic_view` at ^0[2] describes a view at output 0 through input 3, but the instruction has \
             only 2 inputs",
        );
        assert_eq!(
            ProgramError::from(ReferenceViewAnalysisError::MissingView {
                operation: "view",
                instruction: InstructionId::new(RegionId::new(0), 2),
                output_index: 0,
            }),
            ProgramError::Reference(crate::programs::references::ReferenceError::ViewAnalysis(Box::new(
                ReferenceViewAnalysisError::MissingView {
                    operation: "view",
                    instruction: InstructionId::new(RegionId::new(0), 2),
                    output_index: 0
                },
            ),)),
        );
    }

    #[test]
    fn test_reference_view_analysis_error_source() {
        let validation =
            ReferenceViewValidationError::TypeMismatch { expected: "f32[3]".to_string(), actual: "f32[2]".to_string() };
        let error = ReferenceViewAnalysisError::InvalidView {
            operation: "view",
            instruction: InstructionId::new(RegionId::new(0), 0),
            output_index: 2,
            source: validation.clone(),
        };
        assert_eq!(error.source().unwrap().downcast_ref::<ReferenceViewValidationError>(), Some(&validation));

        // The umbrella conversion keeps the contextual analysis failure, including the typed validation cause.
        assert_eq!(
            ProgramError::from(error.clone()),
            ProgramError::Reference(crate::programs::references::ReferenceError::ViewAnalysis(Box::new(error))),
        );
    }

    #[test]
    fn test_reference_view_operation_batch() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context =
            BatchingContext::<_, ArrayIrBatchingPolicy>::new(EagerContext::<TestValue, TestOperation>::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(Array::from_f64s(packed_type, (0..6).map(f64::from).collect()).unwrap())
            .reference_new()
            .unwrap();

        // A mapped source moves its batch axis through the view and binds the batched view on the parent: the
        // leading batch axis shifts the indexed per-item axis to packed axis 1 and the output keeps batch axis 0.
        let batch = ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap();
        let outputs = ReferenceViewOperation::batch(
            &TestOperation::from(ReferenceIndexOperation::new(0, 2)),
            &context,
            &[batch.clone()],
        )
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().read(), Ok(TestValue::Array(Array::vector(vec![2.0f32, 5.0]).unwrap())));

        // A dynamic index shared by every batch item follows the same axis adjustment as a static index.
        let outputs = ReferenceViewOperation::batch(
            &TestOperation::from(ReferenceDynamicIndexOperation::new(0)),
            &context,
            &[batch.clone(), ArrayIrBatch::replicated(TestValue::Array(Array::scalar(1i64).unwrap()))],
        )
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().read(), Ok(TestValue::Array(Array::vector(vec![1.0f32, 4.0]).unwrap())));

        // A replicated source is viewed unchanged and stays replicated.
        let replicated = ArrayIrBatch::replicated(reference);
        let outputs = ReferenceViewOperation::batch(
            &TestOperation::from(ReferenceIndexOperation::new(0, 1)),
            &context,
            &[replicated],
        )
        .unwrap()
        .into_parts()
        .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().read(), Ok(TestValue::Array(Array::vector(vec![3.0f32, 4.0, 5.0]).unwrap())));

        // Only view operations batch through the rule: an operation without a view alias and one with an allocation
        // output are rejected by name, and so is a mapped input other than the viewed source.
        assert_eq!(
            ReferenceViewOperation::batch(
                &TestOperation::from(ReferenceReadOperation::<ArrayType, ArrayIrType>::new()),
                &context,
                &[batch.clone()],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`reference_read` derives no reference view".to_string(),
            }),
        );
        assert_eq!(
            ReferenceViewOperation::batch(
                &TestOperation::from(ReferenceNewOperation::<ArrayType, ArrayIrType>::new()),
                &context,
                &[ArrayIrBatch::replicated(TestValue::Array(Array::scalar(0.0f32).unwrap()))],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`reference_new` has reference output 0 that is not a view, so it cannot batch as a view \
                          operation"
                    .to_string(),
            }),
        );
        assert_eq!(
            ReferenceViewOperation::batch(
                &TestOperation::from(ReferenceDynamicIndexOperation::new(0)),
                &context,
                &[
                    batch,
                    ArrayIrBatch::new(TestValue::Array(Array::vector(vec![0i64, 1]).unwrap()), BatchAxis::new(0))
                        .unwrap()
                ],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message:
                    "`reference_dynamic_index` requires input 1 to be replicated; batching a reference view through a \
                          mapped index input is not supported"
                        .to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_view_operation_batch_multiple_outputs() {
        let parent = TracingContext::<TestValue, TestViewOperation>::new();
        let extent = parent
            .lift(TestValue::Dimension(
                DimensionValue::new(
                    DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())),
                    2,
                )
                .unwrap(),
            ))
            .unwrap();
        let input = ArrayIrBatch::new(parent.input(reference_type([2, 3])), BatchAxis::new(0)).unwrap();
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);

        // Each output is reconstructed independently, in output order, with the batch axis preserved.
        let outputs =
            ReferenceViewOperation::batch(&TestViewOperation::Pair { second_input: 0 }, &context, &[input.clone()])
                .unwrap()
                .into_parts()
                .0;
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[1].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().r#type().as_ref(), &reference_type([2]));
        assert_eq!(outputs[1].value().r#type().as_ref(), &reference_type([2]));
        assert_eq!(
            parent
                .builder()
                .borrow()
                .instructions()
                .iter()
                .map(|instruction| instruction.operation().reference_view(0))
                .collect::<Vec<_>>(),
            vec![Some(index(1, 0)), Some(index(1, 1))],
        );

        // The returned handles must follow that same order, even though both outputs have the same type and axis.
        assert_eq!(outputs[0].value().atom_id(), Ok(parent.builder().borrow().instructions()[0].outputs()[0]));
        assert_eq!(outputs[1].value().atom_id(), Ok(parent.builder().borrow().instructions()[1].outputs()[0]));

        // The shared rule accepts several views of one source, but cannot reconstruct views of separate sources.
        let second = ArrayIrBatch::new(parent.input(reference_type([2, 3])), BatchAxis::new(0)).unwrap();
        assert_eq!(
            ReferenceViewOperation::batch(&TestViewOperation::Pair { second_input: 1 }, &context, &[input, second],)
                .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`view_pair` views inputs 0 and 1, but a view operation views one source".to_string(),
            }),
        );
        assert_eq!(parent.builder().borrow().instructions().len(), 2);
    }

    #[test]
    fn test_reference_view_operation_batch_rejects_missing_views() {
        let parent = TracingContext::<TestValue, TestViewOperation>::new();
        let extent = parent
            .lift(TestValue::Dimension(
                DimensionValue::new(
                    DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())),
                    2,
                )
                .unwrap(),
            ))
            .unwrap();
        let input = ArrayIrBatch::replicated(parent.input(reference_type([2])));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);

        // Declaring a view alias is not enough: reconstruction needs the view for that output.
        assert_eq!(
            ReferenceViewOperation::batch(&TestViewOperation::MissingView, &context, &[input]).err(),
            Some(BatchingError::from(ProgramError::MalformedProgram(
                "operation `missing_view` derives a reference view at output 0 but exposes no view transform"
                    .to_string(),
            ))),
        );
        assert_eq!(parent.builder().borrow().instructions().len(), 0);
    }

    #[test]
    fn test_reference_view_operation_batch_rejects_additional_behavior() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let parent = TracingContext::<TestValue, TestViewOperation>::new();
        let extent = parent.lift(extent).unwrap();
        let input = ArrayIrBatch::replicated(parent.input(reference_type([2])));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);

        // Replaying only the view would discard either the extra value result or the source read effect.
        assert_eq!(
            ReferenceViewOperation::batch(
                &TestViewOperation::AdditionalBehavior { reads: false },
                &context,
                &[input.clone()],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`additional_behavior` has outputs other than its reference views".to_string(),
            }),
        );
        assert_eq!(
            ReferenceViewOperation::batch(&TestViewOperation::AdditionalBehavior { reads: true }, &context, &[input]).err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`additional_behavior` has effects or attached regions that cannot be preserved by batching only its reference views".to_string(),
            }),
        );
        assert_eq!(parent.builder().borrow().instructions().len(), 0);
    }

    #[test]
    fn test_reference_view_step_view() {
        let view = index(0, 1);
        let path = TestPath::root().with_view(view.clone());
        assert_eq!(path.steps()[0].view(), &view);
    }

    #[test]
    fn test_reference_view_step_bindings() {
        let static_path = TestPath::root().with_view(index(0, 1));
        assert_eq!(static_path.steps()[0].bindings(), &[]);

        // The binding identifies the value supplying the dynamic index; it is not the index's runtime value.
        let path = TestPath::root()
            .with_step(TestViewOperation::view(1), vec![ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(path.steps()[0].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(3))]);
    }

    #[test]
    fn test_reference_view_path_root() {
        let root = TestPath::root();
        assert!(root.is_root());
        assert_eq!(root.steps(), &[]);
        assert_eq!(root.views().count(), 0);
        assert_eq!(root, TestPath::default());

        assert_eq!(format!("{root:?}"), "ReferenceViewPath { steps: [] }");
    }

    #[test]
    fn test_reference_view_path_is_root() {
        assert!(TestPath::root().is_root());
        assert!(!TestPath::root().with_view(index(0, 1)).is_root());
    }

    #[test]
    fn test_reference_view_path_steps() {
        let path = TestPath::root().with_view(index(0, 1)).with_view(index(0, 2));
        assert_eq!(
            path.steps(),
            &[
                ReferenceViewStep { view: index(0, 1), bindings: Vec::new() },
                ReferenceViewStep { view: index(0, 2), bindings: Vec::new() },
            ],
        );
    }

    #[test]
    fn test_reference_view_path_views() {
        let path = TestPath::root().with_view(index(0, 1)).with_view(index(0, 2));
        assert_eq!(path.views().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(path.views().rev().collect::<Vec<_>>(), vec![&index(0, 2), &index(0, 1)]);
    }

    #[test]
    fn test_reference_view_path_with_step() {
        let row = TestPath::root().with_view(index(0, 1));
        let symbolic = ArrayReferenceView::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        let bound = row.with_step(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(bound.views().collect::<Vec<_>>(), vec![&index(0, 1), &symbolic]);
        assert_eq!(bound.steps()[1].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(3))]);
        assert_eq!(row.views().collect::<Vec<_>>(), vec![&index(0, 1)]);

        // Equal views can select different indices when their source bindings differ.
        assert_eq!(bound, row.with_step(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(3))]));
        assert_ne!(bound, row.with_step(symbolic.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(4))]));
        assert_ne!(bound, row.with_step(symbolic, vec![ValueId::new(RegionId::new(1), AtomId::new(0))]));

        // Paths used as map keys distinguish bindings as well as views.
        let paths = HashMap::from([(bound.clone(), "bound")]);
        assert_eq!(paths.get(&bound), Some(&"bound"));
        assert_eq!(paths.get(&row), None);
    }

    #[test]
    fn test_reference_view_path_with_view() {
        let root = TestPath::root();
        let row = root.with_view(index(0, 1));
        let element = row.with_view(index(0, 2));

        // Appending preserves root-to-value order without modifying either source path.
        assert!(root.is_root());
        assert_eq!(row.views().collect::<Vec<_>>(), vec![&index(0, 1)]);
        assert_eq!(element.views().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(row, TestPath::root().with_view(index(0, 1)));
        assert_ne!(row, element);
        assert_ne!(row, TestPath::root().with_view(index(1, 1)));
        assert_eq!(
            format!("{row:?}"),
            "ReferenceViewPath { steps: [ReferenceViewStep { view: Index { axis: 0, index: Static(1) }, bindings: [] \
             }] }",
        );
    }

    #[test]
    fn test_reference_view_path_overlap() {
        // The path query delegates to the family's rule against the caller-supplied root type: two different rows are
        // disjoint, a row is the same as itself, and the complete root may overlap with any row but is the same as
        // itself.
        let root = reference_type([2, 3]);
        let row_0 = TestPath::root().with_view(index(0, 0));
        let row_1 = TestPath::root().with_view(index(0, 1));
        assert_eq!(row_0.overlap(&row_1, &root), ReferenceViewOverlap::Disjoint);
        assert_eq!(row_0.overlap(&TestPath::root().with_view(index(0, 0)), &root), ReferenceViewOverlap::Same);
        assert_eq!(TestPath::root().overlap(&row_0, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(TestPath::root().overlap(&TestPath::root(), &root), ReferenceViewOverlap::Same);

        // Symbolic views agree when their input bindings agree. Different bindings may still select the same row.
        let symbolic = TestPath::root()
            .with_step(TestViewOperation::view(1), vec![ValueId::new(RegionId::new(0), AtomId::new(0))]);
        let other = TestPath::root()
            .with_step(TestViewOperation::view(1), vec![ValueId::new(RegionId::new(1), AtomId::new(0))]);
        assert_eq!(symbolic.overlap(&symbolic, &root), ReferenceViewOverlap::Same);
        assert_eq!(symbolic.overlap(&TestPath::root(), &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(symbolic.overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(symbolic.overlap(&other, &root), ReferenceViewOverlap::MayOverlap);
    }

    #[test]
    fn test_reference_view_analysis_new() {
        // Each view appends a step to its source path: first the slice, then its row index. The complete reference
        // has an empty path, and the array returned by the read has no path.
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let slice =
            ArrayReferenceView::Slice { axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)] };
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(0))), Some(&TestPath::root()));
        assert_eq!(
            analysis.path(ValueId::new(RegionId::new(0), AtomId::new(1))),
            Some(&TestPath::root().with_view(slice.clone())),
        );
        assert_eq!(
            analysis.path(ValueId::new(RegionId::new(0), AtomId::new(2))),
            Some(&TestPath::root().with_view(slice.clone()).with_view(index(0, 0))),
        );
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(3))), None);
    }

    #[test]
    fn test_reference_view_analysis_new_handles_reverse_numbered_alias_chains() {
        let mut condition = TestBuilder::new();
        condition.add_input(reference_type([2]));
        let predicate = condition.add_constant(TestValue::Array(Array::scalar(false).unwrap()));
        let condition: TestProgram = condition.build(vec![predicate], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut body = TestBuilder::new();
        let reference = body.add_input(reference_type([2]));
        let body: TestProgram = body.build(vec![reference], vec![Placeholder], vec![Placeholder]).unwrap();
        let mut builder = TestBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let aliases = (0..4096).map(|_| builder.add_variable(reference_type([2]))).collect::<Vec<_>>();
        let mut reference = builder.add_input(reference_type([2]));

        // Instruction order is valid, but the first value in atom order depends on the complete alias chain.
        // Every loop forwards the reference unchanged, keeping path storage constant while testing deep derivation.
        for alias in aliases.iter().rev() {
            builder.add_instruction_unchecked(Instruction::new(
                WhileOperation::<ArrayIrType>::new().into(),
                vec![reference],
                vec![*alias],
                vec![condition, body],
            ));
            reference = *alias;
        }
        let program: TestProgram = builder.build(vec![reference], vec![Placeholder], vec![Placeholder]).unwrap();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(analysis.path(ValueId::new(region.id(), aliases[0])), Some(&TestPath::root()));
        assert_eq!(analysis.path(ValueId::new(region.id(), aliases[4095])), Some(&TestPath::root()));
    }

    #[test]
    fn test_reference_view_analysis_new_copies_identity_aliases_through_while_carries() {
        // The carried reference keeps the empty path across the loop through the identity edge of the loop's second
        // output, while the body derives its own element path from the carried root.
        let scalar_type = ArrayIrType::Array(ArrayType::scalar(DataType::F32));
        let mut condition = TestBuilder::new();
        condition.add_input(scalar_type.clone());
        condition.add_input(reference_type([2]));
        let predicate = condition.add_constant(TestValue::Array(Array::scalar(false).unwrap()));
        let condition = condition
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let mut body = TestBuilder::new();
        let counter = body.add_input(scalar_type.clone());
        let reference = body.add_input(reference_type([2]));
        let element =
            body.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None).unwrap()[0];
        body.add_instruction(ReferenceWriteOperation::new(), Vec::new(), vec![element, counter], None)
            .unwrap();
        let body = body
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![counter, reference],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let mut builder = TestBuilder::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let counter = builder.add_input(scalar_type);
        let reference = builder.add_input(reference_type([2]));
        let outputs = builder
            .add_instruction(
                WhileOperation::<ArrayIrType>::new(),
                vec![condition, body],
                vec![counter, reference],
                None,
            )
            .unwrap()
            .to_vec();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![outputs[0]], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(
            analysis.analysis().alias(ValueId::new(RegionId::new(2), AtomId::new(3))),
            Some(ReferenceAliasEdge::new(
                InstructionId::new(RegionId::new(2), 0),
                1,
                ValueId::new(RegionId::new(2), AtomId::new(1)),
                ReferenceAliasKind::Identity,
                false,
            )),
        );
        assert_eq!(
            analysis.paths().collect::<Vec<_>>(),
            vec![
                (ValueId::new(RegionId::new(0), AtomId::new(1)), &TestPath::root()),
                (ValueId::new(RegionId::new(1), AtomId::new(1)), &TestPath::root()),
                (ValueId::new(RegionId::new(1), AtomId::new(2)), &TestPath::root().with_view(index(0, 1))),
                (ValueId::new(RegionId::new(2), AtomId::new(1)), &TestPath::root()),
                (ValueId::new(RegionId::new(2), AtomId::new(3)), &TestPath::root()),
            ],
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_view_type_mismatches() {
        // The unchecked instruction declares the row view as `f32[2]` although indexing axis 0 of `f32[2, 3]` derives
        // `f32[3]`.
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(reference_type([2, 3]));
        let row = builder.add_variable(reference_type([2]));
        builder.add_instruction_unchecked(Instruction::new(
            ArrayIrOperation::ReferenceIndex(ReferenceIndexOperation::new(0, 0)),
            vec![matrix],
            vec![row],
            Vec::new(),
        ));
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidView {
                operation: REFERENCE_INDEX_OPERATION_NAME,
                instruction: InstructionId::new(RegionId::new(0), 0),
                output_index: 0,
                source: ReferenceViewValidationError::TypeMismatch {
                    expected: "f32[3]".to_string(),
                    actual: "f32[2]".to_string(),
                },
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_invalid_view_compositions() {
        // Bypass instruction inference so the view analysis must report the invalid array transform itself. Axis 2
        // does not exist on this rank-2 referent, regardless of the declared output type.
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(reference_type([2, 3]));
        let view = builder.add_variable(reference_type([2, 3]));
        builder.add_instruction_unchecked(Instruction::new(
            ArrayIrOperation::ReferenceIndex(ReferenceIndexOperation::new(2, 0)),
            vec![matrix],
            vec![view],
            Vec::new(),
        ));
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidView {
                operation: REFERENCE_INDEX_OPERATION_NAME,
                instruction: InstructionId::new(RegionId::new(0), 0),
                output_index: 0,
                source: ReferenceViewValidationError::InvalidComposition {
                    message: "reference index axis 2 is out of bounds for rank 2".to_string(),
                },
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_propagates_analysis_errors() {
        // Consuming an entry reference input fails structural analysis before any view is derived. The view analysis
        // preserves the typed error and its input ownership diagnostic.
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([2]));
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::Analysis(ReferenceAnalysisError::ExternalReferenceConsumption {
                operation: "reference_freeze",
                instruction: InstructionId::new(RegionId::new(0), 0),
                root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
                external_source: ReferenceSource::Input { index: 0 },
            })),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_missing_views() {
        let mut builder = ProgramBuilder::<TestValue, TestViewOperation>::new();
        let reference = builder.add_input(reference_type([2]));
        let view =
            builder.add_instruction(TestViewOperation::MissingView, Vec::new(), vec![reference], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                TestViewOperation::Native(ReferenceReadOperation::new().into()),
                Vec::new(),
                vec![view],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::MissingView {
                operation: "missing_view",
                instruction: InstructionId::new(RegionId::new(0), 0),
                output_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_binds_input_symbols() {
        // The symbol binds to the index input of the instruction that creates the view. The read returns an array,
        // so its output has no reference view path.
        let program = symbolic_view_program(1);
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let view = TestViewOperation::view(1);
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(0))), Some(&TestPath::root()));
        assert_eq!(
            analysis.path(ValueId::new(RegionId::new(0), AtomId::new(2))),
            Some(&TestPath::root().with_step(view.clone(), vec![ValueId::new(RegionId::new(0), AtomId::new(1))])),
        );
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(3))), None);
        assert_eq!(
            analysis
                .path(ValueId::new(RegionId::new(0), AtomId::new(2)))
                .map(|path| path.views().collect::<Vec<_>>()),
            Some(vec![&view])
        );
    }

    #[test]
    fn test_reference_view_analysis_new_binds_shared_region_inputs() {
        // The shared body receives a whole root and two ordinary indices. Its two view steps bind those inputs,
        // independently of the caller's values or which condition branch enters the region.
        let mut body = TestBuilder::new();
        let root = body.add_input(reference_type([2, 3]));
        let row_index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let column_index = body.add_input(ArrayType::scalar(DataType::I64).into());
        let row = body
            .add_instruction(ReferenceDynamicIndexOperation::new(0), Vec::new(), vec![root, row_index], None)
            .unwrap()[0];
        let element = body
            .add_instruction(ReferenceDynamicIndexOperation::new(0), Vec::new(), vec![row, column_index], None)
            .unwrap()[0];
        let result = body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        let body: TestProgram = body.build(vec![result], vec![Placeholder; 3], vec![Placeholder]).unwrap();
        let mut builder = TestBuilder::new();
        let body = builder.import_program(body);
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let root = builder.add_input(reference_type([2, 3]));
        let row_index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let column_index = builder.add_input(ArrayType::scalar(DataType::I64).into());
        let outputs = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![body, body],
                vec![predicate, root, row_index, column_index],
                None,
            )
            .unwrap()
            .to_vec();
        let program: TestProgram = builder.build(outputs, vec![Placeholder; 4], vec![Placeholder]).unwrap();
        let analysis = program.entry_region_ref().reference_view_analysis(0).unwrap();
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(0))), Some(&TestPath::root()));
        let path = analysis.path(ValueId::new(RegionId::new(0), AtomId::new(4))).unwrap();
        assert_eq!(path.steps().len(), 2);
        assert_eq!(path.steps()[0].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(1))]);
        assert_eq!(path.steps()[1].bindings(), &[ValueId::new(RegionId::new(0), AtomId::new(2))]);
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_invalid_symbols() {
        // An input symbol must name an input of the describing instruction.
        let program = symbolic_view_program(2);
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidViewSymbol {
                operation: "symbolic_view",
                instruction: InstructionId::new(RegionId::new(0), 0),
                output_index: 0,
                symbol: 2,
                message: "the instruction has only 2 inputs".to_string(),
            }),
        );

        // The named input must be a non-reference value, not the viewed reference itself.
        let program = symbolic_view_program(0);
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidViewSymbol {
                operation: "symbolic_view",
                instruction: InstructionId::new(RegionId::new(0), 0),
                output_index: 0,
                symbol: 0,
                message: "that input is a reference rather than an index value".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_analysis() {
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(analysis.analysis().region(), RegionId::new(0));
        assert!(analysis.analysis().is_view(ValueId::new(RegionId::new(0), AtomId::new(2))));

        // The structural analysis is the retained one, not a second derivation.
        let retained = program.entry_region_ref().reference_analysis(0).unwrap();
        assert!(std::ptr::eq(analysis.analysis(), &*retained));
    }

    #[test]
    fn test_reference_view_analysis_paths() {
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(
            analysis.paths().map(|(value, path)| (value, path.views().len())).collect::<Vec<_>>(),
            vec![
                (ValueId::new(RegionId::new(0), AtomId::new(0)), 0),
                (ValueId::new(RegionId::new(0), AtomId::new(1)), 1),
                (ValueId::new(RegionId::new(0), AtomId::new(2)), 2)
            ],
        );
    }

    #[test]
    fn test_reference_view_analysis_path() {
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(0))), Some(&TestPath::root()));
        assert_eq!(
            analysis.path(ValueId::new(RegionId::new(0), AtomId::new(2))).map(|path| path.views().len()),
            Some(2)
        );
        assert_eq!(analysis.path(ValueId::new(RegionId::new(0), AtomId::new(3))), None);
        assert_eq!(analysis.path(ValueId::new(RegionId::new(1), AtomId::new(0))), None);
    }

    #[test]
    fn test_reference_view_analysis_overlap() {
        // The row slice may overlap with the complete reference, and
        // indexing the single row of that slice selects exactly the slice's indices again. Values that are not
        // references, or that live in different regions, have no answer.
        let program = chain_program();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(0))
            ),
            Some(ReferenceViewOverlap::Same)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(1))
            ),
            Some(ReferenceViewOverlap::MayOverlap)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(1)),
                ValueId::new(RegionId::new(0), AtomId::new(2))
            ),
            Some(ReferenceViewOverlap::Same)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(3))
            ),
            None
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(1), AtomId::new(0))
            ),
            None
        );

        // Values of different roots are disjoint whatever their paths select.
        let mut builder = TestBuilder::new();
        let first = builder.add_input(reference_type([2]));
        let second = builder.add_input(reference_type([2]));
        let first_element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![first], None).unwrap()[0];
        let second_element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![second], None).unwrap()[0];
        let first_read = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first_element], None)
            .unwrap()[0];
        let second_read = builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![second_element], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![first_read, second_read],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(1))
            ),
            Some(ReferenceViewOverlap::Disjoint)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(2)),
                ValueId::new(RegionId::new(0), AtomId::new(3))
            ),
            Some(ReferenceViewOverlap::Disjoint)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(2))
            ),
            Some(ReferenceViewOverlap::MayOverlap)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(2)),
                ValueId::new(RegionId::new(0), AtomId::new(2))
            ),
            Some(ReferenceViewOverlap::Same)
        );

        // Symbolic indices compare by their bindings: two views through the same index input select the
        // same slot, while views through different inputs, or against a static index or the root, may overlap.
        let mut builder = ProgramBuilder::<TestValue, TestViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let index_value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let other = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let symbolic = TestViewOperation::Symbolic(1);
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, index_value], None).unwrap();
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, index_value], None).unwrap();
        builder.add_instruction(symbolic, Vec::new(), vec![vector, other], None).unwrap();
        builder
            .add_instruction(
                TestViewOperation::Native(ReferenceIndexOperation::new(0, 0).into()),
                Vec::new(),
                vec![vector],
                None,
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(3)),
                ValueId::new(RegionId::new(0), AtomId::new(4))
            ),
            Some(ReferenceViewOverlap::Same)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(3)),
                ValueId::new(RegionId::new(0), AtomId::new(5))
            ),
            Some(ReferenceViewOverlap::MayOverlap)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(3)),
                ValueId::new(RegionId::new(0), AtomId::new(6))
            ),
            Some(ReferenceViewOverlap::MayOverlap)
        );
        assert_eq!(
            analysis.overlap(
                region,
                ValueId::new(RegionId::new(0), AtomId::new(0)),
                ValueId::new(RegionId::new(0), AtomId::new(3))
            ),
            Some(ReferenceViewOverlap::MayOverlap)
        );
    }

    #[test]
    fn test_region_ref_reference_view_analysis() {
        let program = chain_program();
        let retained = program.entry_region_ref().reference_view_analysis(0).unwrap();
        assert_eq!(*retained, ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap());

        // A second request under the same capture scope is served the retained view analysis.
        assert!(Arc::ptr_eq(&program.entry_region_ref().reference_view_analysis(0).unwrap(), &retained));

        // A different capture count is validated instead of reusing the successful cached analysis.
        assert!(matches!(
            program.entry_region_ref().reference_view_analysis(2),
            Err(ReferenceViewAnalysisError::Analysis(ReferenceAnalysisError::InvalidCaptureScope { region, message }))
                if region == RegionId::new(0)
                    && message == "the capture prefix of 2 inputs exceeds the region's 1 inputs",
        ));
    }

    #[test]
    fn test_region_ref_reference_view_analysis_invalidates_rebased_imports() {
        /// Builds `f(matrix: ref<f32[2, 3]>) = read(matrix[row])`.
        fn branch(row: usize) -> TestProgram {
            let mut builder = TestBuilder::new();
            let matrix = builder.add_input(reference_type([2, 3]));
            let view = builder
                .add_instruction(ReferenceIndexOperation::new(0, row), Vec::new(), vec![matrix], None)
                .unwrap()[0];
            let snapshot =
                builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![view], None).unwrap()[0];
            builder.build(vec![snapshot], vec![Placeholder], vec![Placeholder]).unwrap()
        }

        // Program `first` attaches the row-0 branch as both branches of a condition, so the view analysis of its entry
        // `^1` records the row-0 view for the branch region `^0`.
        let mut builder = TestBuilder::new();
        let row_0 = builder.import_region(branch(0).entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let matrix = builder.add_input(reference_type([2, 3]));
        let output = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![row_0, row_0], vec![predicate, matrix], None)
            .unwrap()[0];
        let first = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let retained = first.entry_region_ref().reference_view_analysis(0).unwrap();
        assert_eq!(retained.analysis().region(), RegionId::new(1));
        assert_eq!(
            retained.path(ValueId::new(RegionId::new(0), AtomId::new(1))),
            Some(&TestPath::root().with_view(index(0, 0)))
        );

        // Re-sealing a copy of that entry into an arena whose region `^0` is the row-1 branch changes what the copy's
        // nested views select, so it must not be served the view analysis derived for the row-0 branch.
        let rebased = TestProgram::new(
            vec![Placeholder; 2],
            vec![Placeholder],
            vec![branch(1).entry_region().clone(), first.entry_region().clone()],
            RegionId::new(1),
        )
        .unwrap();
        let derived = rebased.entry_region_ref().reference_view_analysis(0).unwrap();
        assert!(!Arc::ptr_eq(&derived, &retained));
        assert_eq!(
            derived.path(ValueId::new(RegionId::new(0), AtomId::new(1))),
            Some(&TestPath::root().with_view(index(0, 1)))
        );
        assert_eq!(derived.path(ValueId::new(RegionId::new(1), AtomId::new(1))), Some(&TestPath::root()));

        // The source program keeps its own retained view analysis, because only the re-sealed copy was rebased.
        assert!(Arc::ptr_eq(&first.entry_region_ref().reference_view_analysis(0).unwrap(), &retained));
    }
}
