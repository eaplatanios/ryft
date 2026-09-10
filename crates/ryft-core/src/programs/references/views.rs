//! Contains machinery for representing and working with _reference views_ and paths layered on the structural
//! [`ReferenceAnalysis`]. The generic analysis records _that_ a reference-typed value is a narrowing view of its
//! [`ReferenceRoot`], through [`ReferenceAliasEdge`](crate::ReferenceAliasEdge)s of kind [`ReferenceAliasKind::View`],
//! but leaves the selection itself to the value family (e.g., an array view is an index or slice, while a downstream
//! family may split a register into halves). [`ReferenceViewOperation`] describes these selections for an operation
//! family, and [`ReferenceViewAnalysis`] composes the per-edge descriptions into a [`ReferenceViewPath`] for every
//! reference-typed value. Transforms that rebuild references (e.g., for tangent, cotangent, and residual
//! reconstruction) consult the view analysis and reapply descriptions through the same contract, so no transform ever
//! matches view operations by name, and downstream operations such as the array family's `reference_index` and
//! `reference_slice` operations are not special-cased anywhere.
//!
//! Everything in this module leverages static dispatch on the operation family `O`. View descriptions are owned data,
//! validation and reapplication are associated functions of the family, so the contract composes with the closed
//! operation enums that backends own.
//!
//! # Symbols And Bindings
//!
//! A view description may depend on values supplied to its [`Instruction`](crate::Instruction), such as a scalar array
//! index. Its [`ReferenceView::symbols`] function lists the positions of those inputs. Analysis binds each position to
//! the corresponding [`ValueId`]. The description and these bindings form a [`ReferenceViewStep`]. Static descriptions
//! have no symbols and carry empty bindings. Index values created by an enclosing loop enter as ordinary region
//! inputs, so analysis does not need loop-specific symbols or names.
//!
//! The binding type is a parameter of the path because the same path shape serves consumers that close symbols
//! differently: the view analysis binds program identities, an eager handle carries only static steps and uses
//! the uninhabited [`NoReferenceViewBinding`], and a discharge policy may close symbols over destination values.
//!
//! # Validating Views Against Transformed References
//!
//! A view description is validated against the _current_ source reference type before it is reapplied. A tangent or
//! cotangent root may have a different referent type from the primal root (e.g., a widened floating-point tangent
//! type), so a description that was valid on the primal root is re-checked against the transformed root rather than
//! assumed to transfer.
//!
//! # Batching Moves The Axis Through The Mapping
//!
//! [`ReferenceViewOperation::reapply_view`] rebuilds a description over a root with the same dimensions as the one it
//! was derived on, which is what tangent, cotangent, and residual reconstruction need. Batching is different in that it
//! inserts an axis into the packed root, and a primal description reapplied unchanged to a batched root would index or
//! slice the wrong axis. The contract therefore splits the two concerns. [`ReferenceView::batch`] is pure axis
//! arithmetic on the description (i.e., given the packed source type and the source's batch axis, it returns the
//! description that selects the same part of each item of the packed source together with the batch axis of the
//! derived reference). The shared rule [`batch_reference_view_operation`] then binds that batched description through
//! [`reapply_view`](ReferenceViewOperation::reapply_view) on the parent context, so every view operation of every
//! family batches through one rule and no operation carries the axis arithmetic itself.
//!
//! # Overlap Queries
//!
//! Two paths of one root may select the same part, provably disjoint parts, or parts whose overlap is not decidable
//! statically. [`ReferenceView::overlap`] answers that question for two closed paths of one root as a
//! [`ReferenceViewOverlap`]; [`ReferenceViewPath::overlap`] and [`ReferenceViewAnalysis::overlap`] expose it on paths
//! and on analyzed values. Equal symbol bindings identify the same value, but the view descriptions must also agree
//! for the selections to be identical. For example, array indexing can clamp the same index differently after two
//! different slices. The value family accounts for these semantics; generic consumers must treat [`MayOverlap`](
//! ReferenceViewOverlap::MayOverlap) conservatively. Proving disjoint selections does not establish independent
//! reference lifetimes or permit a transform to split one allocation into independently updated states.
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
    ReferenceAliasPosition, ReferenceAnalysis, ReferenceAnalysisError, ReferenceAnalysisTransformArguments,
    ReferenceRoot,
};
use crate::programs::regions::{Region, RegionRef};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId};

/// Error produced by [`ReferenceViewOperation::validate_view`] when a view description does not compose onto its
/// source reference type or does not derive the declared output reference type.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceViewValidationError {
    /// The description composes onto the source but derives a referent type that differs from the declared one.
    #[error("view declares referent type `{actual}` but derives referent type `{expected}` from its source")]
    TypeMismatch {
        /// Referent type derived by the description from the source.
        expected: String,

        /// Referent type declared by the view output.
        actual: String,
    },

    /// The description cannot be applied to the source reference type at all.
    #[error("invalid view composition: {message}")]
    InvalidComposition {
        /// Description of why the view is invalid for the source.
        message: String,
    },
}

/// Error produced by [`ReferenceViewAnalysis`] when the generic reference analysis fails or when a derived view path
/// cannot be reconciled with the program's declared reference types. Conversion to [`ProgramError`] preserves an
/// underlying [`ReferenceAnalysisError`] through its typed conversion. View-specific failures are preserved through
/// [`ReferenceError::ViewAnalysis`](crate::ReferenceError::ViewAnalysis). Invalid descriptions retain their
/// [`ReferenceViewValidationError`] as an error source; positions and symbols identify the exact declaration
/// that failed without duplicating the validation error's variants.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Error)]
pub enum ReferenceViewAnalysisError {
    /// The structural reference analysis rejected the region and its attached computation regions.
    #[error(transparent)]
    Analysis(#[from] ReferenceAnalysisError),

    /// An operation declares a view for an output but supplies no description for it.
    #[error("operation `{operation}` at {instruction} declares a reference view at {position} but describes no view")]
    MissingView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output whose view description is missing.
        position: ReferenceAliasPosition,
    },

    /// A view description cannot be applied to its source or derives a type different from the declared type.
    /// The underlying validation error retains the type mismatch or invalid-composition diagnostic.
    #[error("operation `{operation}` at {instruction} has an invalid view at {position}: {source}")]
    InvalidView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output described by the invalid view.
        position: ReferenceAliasPosition,

        /// Failure reported when validating the view against its source and declared output types.
        #[source]
        source: ReferenceViewValidationError,
    },

    /// A symbolic input position is out of range or names a reference rather than an index value.
    #[error(
        "operation `{operation}` at {instruction} describes a view at {position} through input {symbol}, but {message}"
    )]
    InvalidViewSymbol {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output whose view uses the invalid symbol.
        position: ReferenceAliasPosition,

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

// TODO(eaplatanios): Review from here onwards.

/// Uninhabited binding of paths that only ever carry static steps, such as the path of an eager array-reference
/// handle. Every step has empty bindings; consumers must reject descriptions that require symbols because no binding
/// value can be supplied for them.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub enum NoReferenceViewBinding {}

/// Overlap between the parts selected by two closed root-relative paths of one reference root, as decided statically by
/// [`ReferenceView::overlap`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ReferenceViewOverlap {
    /// The two paths provably select disjoint parts of the root.
    Disjoint,

    /// The two paths provably select exactly the same part of the root.
    Same,

    /// The two paths may overlap: their static selections intersect, or a selection depends on a symbol whose
    /// binding cannot prove the paths identical or disjoint.
    MayOverlap,
}

/// Owned description of one view step of a reference family, from a source reference to the reference it derives.
///
/// The bounds are what the retained [`ReferenceViewAnalysis`] needs to live in a region's transform cache and to be
/// revalidated against a fresh derivation, plus hashing so that paths of descriptions can key eager handles. A
/// description may depend on values outside itself, which it names through [`symbols`](Self::symbols); refer to the
/// module documentation for how the view analysis closes them.
pub trait ReferenceView: 'static + Clone + Debug + PartialEq + Eq + Hash + Send + Sync {
    /// Reference type family the description addresses.
    type Type: Type;

    /// Returns the symbols this description depends on, in the order their bindings and values are supplied to every
    /// consumer. Each symbol is a non-reference input's position in the describing instruction. Static descriptions
    /// return no symbols.
    fn symbols(&self) -> Vec<usize>;

    /// Moves the batch axis of a source reference through this mapping. The batch axis of a reference is an axis of
    /// its packed referent that the per-item view never sees, so the batched description must select the same part of
    /// each item of the packed source that this description selects of the unbatched one, and the derived reference has
    /// its own batch axis. This is pure axis arithmetic: the symbols of the description are untouched, and a replicated
    /// `batch_axis` returns the description unchanged and replicated.
    ///
    /// # Parameters
    ///
    ///   - `source`: Packed reference type of the batched source (i.e., the type with the batch axis inserted).
    ///   - `batch_axis`: Batch axis of the source, positioned in the packed referent of `source`.
    ///
    /// # Errors
    ///
    /// Returns a [`BatchingError`] when this family cannot carry `batch_axis` through the description (e.g., a family
    /// without axes rejects every mapped axis, and a static array slice cannot span a dynamically sized batch axis).
    fn batch(&self, source: &Self::Type, batch_axis: BatchAxis) -> Result<(Self, BatchAxis), BatchingError>;

    /// Returns the overlap between the parts that two closed root-relative paths `a` and `b` select of one root of
    /// type `root`. Equal symbol bindings identify the same value. Different bindings do not prove disjointness: for
    /// example, different input values may supply equal array indices. The empty path denotes the complete root, so it
    /// is [`Same`](ReferenceViewOverlap::Same) as itself and may overlap with any narrowing path. Paths are validated
    /// when they are derived, so implementations may treat a malformed path conservatively as
    /// [`MayOverlap`](ReferenceViewOverlap::MayOverlap) instead of failing.
    fn overlap(root: &Self::Type, a: &[ReferenceViewStep<Self>], b: &[ReferenceViewStep<Self>])
    -> ReferenceViewOverlap;
}

/// Static view contract of one operation family: the owned description of every view alias the family can derive,
/// its type-level validation, and its reapplication to another reference with compatible dimensions.
///
/// An operation family implements this trait once, and every transform that rebuilds references (tangent, cotangent,
/// and residual reconstruction) then reaches the family's views through it: [`ReferenceViewAnalysis`] composes the
/// descriptions into per-value [`ReferenceViewPath`]s, and reconstruction reapplies them step by step to the
/// transformed root. A tangent or cotangent root may have a different referent type from the primal root, so each
/// description is validated against the current transformed source type before it is reapplied. Batching does not
/// reapply a primal description unchanged, because a batched root has an extra axis; it first moves the batch axis
/// through the description with [`ReferenceView::batch`] and then reapplies the batched description, which is what
/// [`batch_reference_view_operation`] does for every view operation.
pub trait ReferenceViewOperation: Operation {
    /// Description of one view step of this family, addressing this family's reference types.
    type View: ReferenceView<Type = Self::Type>;

    /// Returns the description of the view this operation derives at output `output_index`, or [`None`] when that
    /// output is not a view. Exactly the outputs whose [`effects`](Operation::effects) declare a
    /// [`ReferenceAlias`](crate::programs::effects::ReferenceAlias) of kind [`ReferenceAliasKind::View`] return
    /// [`Some`]; an operation that declares such an alias but returns [`None`] is rejected by the view analysis with
    /// [`ReferenceViewAnalysisError::MissingView`]. Symbol positions refer to this instruction's non-reference inputs.
    fn reference_view(&self, output_index: usize) -> Option<Self::View>;

    /// Validates `view` as a step from the reference type `source` to the reference type `output`. Both types are the
    /// reference-typed atom types of the source and output values, so implementations project their referents.
    ///
    /// # Errors
    ///
    /// Returns [`ReferenceViewValidationError::InvalidComposition`] when `view` cannot be applied to `source` at all,
    /// and [`ReferenceViewValidationError::TypeMismatch`] when it derives a referent type other than the one `output`
    /// declares.
    fn validate_view(
        view: &Self::View,
        source: &Self::Type,
        output: &Self::Type,
    ) -> Result<(), ReferenceViewValidationError>;

    /// Stages `view` over the reference `source` through `context`, whose operation family is this one, and returns
    /// the derived reference. Used to rebuild tangent, cotangent, and residual views over a transformed root of
    /// compatible dimensions; callers validate the step with [`validate_view`](Self::validate_view) against the current
    /// source type first.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context of this operation family through which the view is staged.
    ///   - `view`: Description to reapply.
    ///   - `source`: Reference the view is applied to.
    ///   - `symbols`: One value per entry of [`view.symbols()`](ReferenceView::symbols), in that order. An
    ///     input position in the description corresponds to one value supplied here.
    ///
    /// # Errors
    ///
    /// Propagates the staging error of the context, and rejects a symbol count that differs from the description's or
    /// a description that cannot be reapplied.
    fn reapply_view<C: Context<Type = Self::Type, Operation = Self>>(
        context: &C,
        view: &Self::View,
        source: C::Value,
        symbols: &[C::Value],
    ) -> Result<C::Value, ProgramError>;
}

/// One closed view step of a [`ReferenceViewPath`]: a description together with one binding per symbol the description
/// reports, in [`ReferenceView::symbols`] order. Static descriptions carry empty bindings.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceViewStep<View, Binding = ValueId> {
    /// Refer to the documentation of [`Self::view`].
    view: View,

    /// Refer to the documentation of [`Self::bindings`].
    bindings: Vec<Binding>,
}

impl<View, Binding> ReferenceViewStep<View, Binding> {
    /// Returns the description of this step.
    #[inline]
    pub fn view(&self) -> &View {
        &self.view
    }

    /// Returns the binding of each symbol of the description, in the description's symbol order.
    #[inline]
    pub fn bindings(&self) -> &[Binding] {
        self.bindings.as_slice()
    }
}

/// Ordered closed view steps from a reference root to one derived reference-typed value.
///
/// The path stores only the steps, in root-to-value order; the root itself is a property of the structural
/// [`ReferenceAnalysis`]. The empty path is the identity and denotes the complete root. Complete root handles, capture
/// constants, and forwarded complete references carry it. Equality and
/// hashing distinguish different step sequences, not the values they were derived for. `Binding` is what each step's
/// symbols are closed over: the view analysis binds program identities ([`ValueId`]), and a path
/// that only ever carries static steps uses [`NoReferenceViewBinding`]. Refer to the module documentation for more
/// information.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Parameter)]
pub struct ReferenceViewPath<View, Binding = ValueId> {
    /// Refer to the documentation of [`Self::steps`].
    steps: Vec<ReferenceViewStep<View, Binding>>,
}

impl<View, Binding> ReferenceViewPath<View, Binding> {
    /// Returns the empty path denoting the complete root.
    #[inline]
    pub const fn root() -> Self {
        Self { steps: Vec::new() }
    }

    /// Returns the ordered closed steps applied from the root outward.
    #[inline]
    pub fn steps(&self) -> &[ReferenceViewStep<View, Binding>] {
        self.steps.as_slice()
    }

    /// Returns the ordered descriptions applied from the root outward, without their bindings.
    #[inline]
    pub fn views(&self) -> impl ExactSizeIterator<Item = &View> + DoubleEndedIterator {
        self.steps.iter().map(ReferenceViewStep::view)
    }

    /// Returns whether this path denotes the complete root.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.steps.is_empty()
    }

    /// Returns a copy of this path extended by one more step applied to its current end, closing the symbols of `view`
    /// over `bindings`. The caller must supply one binding per symbol in the description's symbol order; this generic
    /// container does not validate the description or its bindings.
    pub fn with_step(&self, view: View, bindings: Vec<Binding>) -> Self
    where
        View: Clone,
        Binding: Clone,
    {
        let mut steps = Vec::with_capacity(self.steps.len() + 1);
        steps.extend(self.steps.iter().cloned());
        steps.push(ReferenceViewStep { view, bindings });
        Self { steps }
    }

    /// Returns a copy of this path extended by one more static description applied to its current end. This is the
    /// shorthand of [`with_step`](Self::with_step) with no bindings; the caller must ensure `view` requires no symbols.
    #[inline]
    pub fn with_view(&self, view: View) -> Self
    where
        View: Clone,
        Binding: Clone,
    {
        self.with_step(view, Vec::new())
    }
}

impl<View: ReferenceView> ReferenceViewPath<View> {
    /// Returns the overlap between the parts this path and `other` select of one root of type `root`, through
    /// [`ReferenceView::overlap`]. Both paths must be root-relative paths of the same root; callers that compare
    /// analyzed values of one region use [`ReferenceViewAnalysis::overlap`], which checks the roots first, while this
    /// function serves callers that resolve roots across namespaces themselves.
    #[inline]
    pub fn overlap(&self, other: &Self, root: &View::Type) -> ReferenceViewOverlap {
        View::overlap(root, self.steps(), other.steps())
    }
}

impl<View, Binding> Default for ReferenceViewPath<View, Binding> {
    #[inline]
    fn default() -> Self {
        Self::root()
    }
}

/// Structural [`ReferenceAnalysis`] of a [`Region`] closure together with the [`ReferenceViewPath`] of every
/// reference-typed value in that closure, derived through the [`ReferenceViewOperation`] contract of the closure's
/// operation family.
///
/// Every reference-typed value has exactly one path. A root handle (a region input, an allocation, a capture constant,
/// or a forwarded region output) has the empty path, an identity alias copies the path of its source, and a view alias
/// copies the path of its source and appends the description its producing operation reports for that edge's output,
/// after that description was validated against the source and output reference types. Nested region inputs are
/// separate roots of the structural analysis and carry empty paths. Instructions in each region create views of those
/// roots, with any symbolic input positions bound to values in that region.
///
/// The view analysis is retained in the region's transform cache under exactly the cache identity of the structural
/// analysis (refer to the documentation of [`RegionRef::reference_view_analysis`]) and shares that analysis through an
/// [`Arc`] rather than re-deriving it. Transforms and other consumers invoke this validation explicitly on their
/// programs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceViewAnalysis<View> {
    /// Refer to the documentation of [`Self::analysis`].
    analysis: Arc<ReferenceAnalysis>,

    /// Refer to the documentation of [`Self::paths`].
    paths: BTreeMap<ValueId, ReferenceViewPath<View>>,
}

impl<View> ReferenceViewAnalysis<View> {
    /// Analyzes the complete closure of `region` and derives the [`ReferenceViewPath`] of every reference-typed value
    /// in it. The structural analysis is obtained through [`RegionRef::reference_analysis`], so it is shared with every
    /// other consumer of the same closure; this function itself is the uncached derivation of the view analysis, and
    /// [`RegionRef::reference_view_analysis`] is its retained counterpart. Refer to the documentation of
    /// [`RegionRef::reference_analysis`] for the meaning of `capture_count`.
    ///
    /// # Errors
    ///
    /// Returns the [`ReferenceAnalysisError`] of the structural analysis when the closure violates the reference
    /// model, and otherwise the first path derivation failure in canonical value order: an operation declaring a view
    /// alias without describing it, a description that is invalid for its source, or a declared output referent that
    /// differs from the derived one.
    pub fn new<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        capture_count: usize,
    ) -> Result<Self, ReferenceViewAnalysisError>
    where
        // Implied by `O::View`'s bounds, but the trait solver does not carry them through the projection equality.
        View: ReferenceView,
    {
        Self::new_with_arguments(
            region,
            &ReferenceAnalysisTransformArguments::new(region, Vec::new(), Some(capture_count), false),
        )
    }

    /// Derives the view analysis exactly like [`new`](Self::new), obtaining the structural analysis under the
    /// already-derived cache key `arguments` so that the closure is walked once per derivation.
    fn new_with_arguments<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        arguments: &ReferenceAnalysisTransformArguments,
    ) -> Result<Self, ReferenceViewAnalysisError>
    where
        // Implied by `O::View`'s bounds, but the trait solver does not carry them through the projection equality.
        View: ReferenceView,
    {
        let analysis = region.reference_analysis_impl(arguments)?;
        let mut paths = BTreeMap::new();
        analysis.values().try_for_each(|value| Self::derive_path(region, &analysis, &mut paths, value))?;
        Ok(Self { analysis, paths })
    }

    /// Returns the structural [`ReferenceAnalysis`] of the closure.
    #[inline]
    pub fn analysis(&self) -> &ReferenceAnalysis {
        &self.analysis
    }

    /// Returns the [`ReferenceViewPath`] from the root of the reference-typed `value` to the part it selects,
    /// or [`None`] when `value` is not a reference-typed value of the closure. Root handles carry the empty path, and
    /// so does a nested region input forwarded as a complete handle.
    #[inline]
    pub fn path(&self, value: ValueId) -> Option<&ReferenceViewPath<View>> {
        self.paths.get(&value)
    }

    /// Returns the [`ReferenceViewPath`] of every reference-typed value of the closure, in canonical [`ValueId`]
    /// order.
    #[inline]
    pub fn paths(&self) -> impl Iterator<Item = (ValueId, &ReferenceViewPath<View>)> + '_ {
        self.paths.iter().map(|(value, path)| (*value, path))
    }

    /// Returns the [`ReferenceViewOverlap`] between the parts that the reference-typed values `a` and `b` of one
    /// region select, or [`None`] when either is not a reference-typed value of the closure or the two values belong to
    /// different regions. Roots are region-relative (a nested region input is a root of its own namespace even when it
    /// carries a caller root), so only values of one region have comparable roots: values of different roots are
    /// [`Disjoint`](ReferenceViewOverlap::Disjoint), and values of one root delegate to [`ReferenceView::overlap`] with
    /// the type of the root's defining atom, read from `region`, the closure this view analysis was derived for. The view
    /// analysis retains no types itself, because the region's transform cache holds it behind a
    /// `Send + Sync` erasure that the value family's type is not required to satisfy. Callers that compare paths across
    /// namespaces resolve the roots themselves and use [`ReferenceViewPath::overlap`].
    pub fn overlap<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        &self,
        region: RegionRef<'_, V, O>,
        a: ValueId,
        b: ValueId,
    ) -> Option<ReferenceViewOverlap>
    where
        // Implied by `O::View`'s bounds, but the trait solver does not carry them through the projection equality.
        View: ReferenceView<Type = V::Type>,
    {
        if a.region() != b.region() {
            return None;
        }
        let root = self.analysis.root_of(a)?;
        if root != self.analysis.root_of(b)? {
            return Some(ReferenceViewOverlap::Disjoint);
        }
        // Every root is defined exactly once, by the input atom or the allocating instruction output that names it,
        // and every reference-typed value has a path, so once both roots resolved the remaining lookups cannot fail
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
        Some(self.paths[&a].overlap(&self.paths[&b], root_type.as_ref()))
    }

    /// Derives the path of `value`, recording each unresolved alias source before its dependent value. Instruction
    /// order guarantees acyclic aliases, but atom identifiers need not follow that order, so an explicit worklist
    /// avoids recursion proportional to the length of a valid alias chain.
    fn derive_path<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        analysis: &ReferenceAnalysis,
        paths: &mut BTreeMap<ValueId, ReferenceViewPath<View>>,
        value: ValueId,
    ) -> Result<(), ReferenceViewAnalysisError>
    where
        View: ReferenceView,
    {
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
                None => ReferenceViewPath::root(),
                Some(edge) => {
                    let mut path = paths[&edge.source()].clone();
                    if edge.kind() == ReferenceAliasKind::View {
                        path.steps.push(Self::derive_view_step(
                            region,
                            edge.instruction(),
                            edge.position(),
                            edge.source(),
                            value,
                        )?);
                    }
                    path
                }
            };
            paths.insert(value, path);
        }
        Ok(())
    }

    /// Validates one view description against its source and result reference types, then binds each symbol to the
    /// describing instruction's input value.
    fn derive_view_step<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        id: InstructionId,
        position: ReferenceAliasPosition,
        source: ValueId,
        value: ValueId,
    ) -> Result<ReferenceViewStep<View>, ReferenceViewAnalysisError>
    where
        View: ReferenceView,
    {
        // Structural analysis resolved both values and the instruction before recording this edge.
        let current = region.with_id(id.region()).unwrap();
        let instruction = &current.instructions()[id.index()];
        let operation = instruction.operation();
        let name = operation.name();
        let view = match position {
            ReferenceAliasPosition::Output(output_index) => operation.reference_view(output_index),
        }
        .ok_or(ReferenceViewAnalysisError::MissingView { operation: name, instruction: id, position })?;
        let atoms = current.atoms();
        let source_type = atoms[source.atom().index()].r#type();
        let output_type = region.with_id(value.region()).unwrap().atoms()[value.atom().index()].r#type();
        O::validate_view(&view, source_type.as_ref(), output_type.as_ref()).map_err(|source| {
            ReferenceViewAnalysisError::InvalidView { operation: name, instruction: id, position, source }
        })?;
        // Resolve each symbolic input position to the ordinary program value supplied to this instruction.
        // A reference cannot supply a scalar index; the operation family validates the remaining index type rules.
        let inputs = instruction.inputs();
        let mut bindings = Vec::new();
        for input_index in view.symbols() {
            let Some(atom) = inputs.get(input_index) else {
                return Err(ReferenceViewAnalysisError::InvalidViewSymbol {
                    operation: name,
                    instruction: id,
                    position,
                    symbol: input_index,
                    message: format!("the instruction has only {} inputs", inputs.len()),
                });
            };
            if atoms[atom.index()].r#type().is_reference() {
                return Err(ReferenceViewAnalysisError::InvalidViewSymbol {
                    operation: name,
                    instruction: id,
                    position,
                    symbol: input_index,
                    message: "that input is a reference rather than an index value".to_string(),
                });
            }
            bindings.push(ValueId::new(id.region(), *atom));
        }
        Ok(ReferenceViewStep { view, bindings })
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
                let analysis = ReferenceViewAnalysis::new_with_arguments(region, arguments)?;
                Ok(TransformArtifact::new(Vec::new(), Arc::new(analysis)))
            },
        )?;
        let (programs, analysis) = artifact.into_parts();
        assert!(programs.is_empty(), "reference view analysis transform retained a program");
        Ok(analysis)
    }
}

/// Batches one reference-view operation of the family of `C` through the [`ReferenceView`] contract: the shared
/// [`BatchableOperation`](crate::batching::BatchableOperation) rule of every operation whose effects declare only
/// [`View`](ReferenceAliasKind::View) aliases of one source input.
///
/// The rule reads the operation's [`effects`](Operation::effects) to find the single source input and the view outputs,
/// requires every other input (the inputs named by the views' symbols) to be replicated, and then, for each view output
/// in output order, moves the source's batch axis through the description with [`ReferenceView::batch`] and binds the
/// batched description over the packed source through [`ReferenceViewOperation::reapply_view`] on the parent context,
/// supplying the packed value of each input a symbol names. Each reapplication binds one operation on the parent, so an
/// operation with several view outputs (e.g., a family that splits a register into two halves) is bound once per
/// output, each time keeping the output the description denotes.
///
/// # Parameters
///
///   - `operation`: Member operation to batch, converted into the family operation `C::Operation` to reach the
///     family's view contract.
///   - `context`: Batching context whose parent the batched views are bound on.
///   - `inputs`: Batch carriers of the operation's inputs.
///
/// # Errors
///
/// Returns [`BatchingError::UnsupportedOperation`] when the operation derives no view, views more than one source
/// input, has outputs other than its views, has observable effects or attached regions, has a mapped non-source input
/// (batching a view through a mapped symbol is not supported). Propagates the [`BatchingError`] of
/// [`ReferenceView::batch`] and the errors of the
/// parent context's binding.
pub fn batch_reference_view_operation<C, P, O>(
    operation: &O,
    context: &BatchingContext<C, P>,
    inputs: &[P::Batch],
) -> Result<BatchedOutputs<C, P>, BatchingError>
where
    C: Context<Operation: ReferenceViewOperation + From<O>>,
    P: BatchingPolicy<C>,
    O: Clone,
{
    let operation = C::Operation::from(operation.clone());
    let name = operation.name();

    let effects = operation.effects();
    let not_a_view = |output_index: usize| BatchingError::UnsupportedOperation {
        message: format!(
            "`{name}` has reference output {output_index} that is not a view, so it cannot batch as a view operation",
        ),
    };
    if let Some(output_index) = effects.allocation_output_indices().next() {
        return Err(not_a_view(output_index));
    }
    let mut source_index = None;
    let mut output_indices = Vec::with_capacity(effects.reference_aliases().len());
    for alias in effects.reference_aliases() {
        let (output_index, input_index) = (alias.output_index(), alias.input_index());
        if alias.kind() != ReferenceAliasKind::View {
            return Err(not_a_view(output_index));
        }
        match source_index {
            None => source_index = Some(input_index),
            Some(source_index) if source_index == input_index => {}
            Some(source_index) => {
                return Err(BatchingError::UnsupportedOperation {
                    message: format!(
                        "`{name}` views inputs {source_index} and {input_index}, but a view operation views one source",
                    ),
                });
            }
        }
        output_indices.push(output_index);
    }
    let Some(source_index) = source_index else {
        return Err(BatchingError::UnsupportedOperation { message: format!("`{name}` derives no reference view") });
    };
    // Replaying descriptions preserves only the views themselves. Reject effects and attached computations whose
    // execution would otherwise be silently dropped by this rule.
    if effects.summary().has_observable_effects_when_unused() || !operation.region_slots().is_empty() {
        return Err(BatchingError::UnsupportedOperation {
            message: format!(
                "`{name}` has effects or attached regions that cannot be preserved by batching only its reference views",
            ),
        });
    }
    output_indices.sort_unstable();
    if output_indices.iter().enumerate().any(|(position, output_index)| position != *output_index) {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("`{name}` has outputs other than its reference views"),
        });
    }
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
                "`{name}` requires input {input_index} to be replicated; batching a reference view through a mapped \
                 index input is not supported",
            ),
        });
    }

    // Alias positions alone do not reveal a trailing non-reference output. Check the complete per-item signature
    // before emitting any views so the reconstructed outputs preserve the operation's full boundary.
    let input_types = inputs.iter().map(|input| P::unbatched_type(input).into_owned()).collect::<Vec<_>>();
    let output_types = operation.infer_output_types(&input_types, &[])?;
    if output_types.len() != output_indices.len() || output_types.iter().any(|r#type| !r#type.is_reference()) {
        return Err(BatchingError::UnsupportedOperation {
            message: format!("`{name}` has outputs other than its reference views"),
        });
    }
    let source_value = P::value(source);
    let source_axis = P::batch_axis(source);
    let source_type = source_value.r#type();
    let outputs = output_indices
        .into_iter()
        .map(|output_index| {
            let Some(view) = operation.reference_view(output_index) else {
                return Err(ProgramError::MalformedProgram(format!(
                    "operation `{name}` derives a reference view at output {output_index} but exposes no view transform",
                ))
                .into());
            };
            let symbols = view
                .symbols()
                .into_iter()
                .map(|input_index| {
                    inputs.get(input_index).map(|input| P::value(input).clone()).ok_or_else(|| {
                        BatchingError::from(ProgramError::MalformedProgram(format!(
                            "`{name}` describes output {output_index} through input {input_index} but was applied to {} inputs",
                            inputs.len(),
                        )))
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            let (view, output_axis) = view.batch(source_type.as_ref(), source_axis)?;
            let value = C::Operation::reapply_view(context.parent(), &view, source_value.clone(), symbols.as_slice())?;
            P::batch(value, output_axis)
        })
        .collect::<Result<Vec<_>, BatchingError>>()?;
    Ok(outputs.into())
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

    use pretty_assertions::assert_eq;

    use crate::arrays::{
        Array, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayIrOperation, ArrayIrType, ArrayIrValue,
        ArrayReferenceViewIndex, ArrayReferenceViewOperation, ArrayReferenceViewTransform, ArraySliceAxis, ArrayType,
        DataType, DimensionBounds, DimensionType, DimensionValue, DimensionVariable, REFERENCE_INDEX_OPERATION_NAME,
        ReferenceDynamicIndexOperation, ReferenceIndexOperation, ReferenceSliceOperation, reapply_array_reference_view,
    };
    use crate::contexts::{EagerContext, StagingContext};
    use crate::operations::{
        ConditionOperation, DynamicSliceOperation, DynamicUpdateSliceOperation, ReferenceFreezeOperation, ReferenceNew,
        ReferenceNewOperation, ReferenceRead, ReferenceReadOperation, ReferenceWriteOperation, ReshapeOperation,
        SliceOperation, UpdateSliceOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::{EffectClasses, Effects, ReferenceAccessMode, ReferenceAlias, ReferenceEffect};
    use crate::programs::instructions::Instruction;
    use crate::programs::programs::Program;
    use crate::programs::references::analysis::{ReferenceAliasEdge, ReferenceAliasPosition};
    use crate::programs::references::discharge::ReferenceSource;
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{
        InputRegionProvenance, OutputRegionProvenance, RegionId, RegionInterface, RegionSlot,
    };
    use crate::programs::types::TypeError;
    use crate::tracing::TracingContext;

    use super::*;

    type TestValue = ArrayIrValue<Array>;

    type TestOperation = ArrayIrOperation<Array>;

    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    type TestPath = ReferenceViewPath<ArrayReferenceViewTransform>;

    /// Returns an instruction identity in the test program arena.
    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    /// Returns a value identity in the test program arena.
    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    /// Returns a reference type over a statically shaped `f32` array.
    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    /// Returns a static index view for the given axis and index.
    fn index(axis: usize, index: usize) -> ArrayReferenceViewTransform {
        ArrayReferenceViewTransform::Index { axis, index: ArrayReferenceViewIndex::Static(index) }
    }

    /// Builds `f(matrix: ref<f32[2, 3]>) = read(matrix[0:1, 0:3][0])`, a two-step view chain over one root.
    fn chain_program() -> TestProgram {
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(reference_type([2, 3]));
        let axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let row =
            builder.add_instruction(ReferenceSliceOperation::new(axes), Vec::new(), vec![matrix], None).unwrap()[0];
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![row], None).unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        builder.build(vec![snapshot], vec![Placeholder], vec![Placeholder]).unwrap()
    }

    /// Array-IR family extended with one two-input view operation whose description selects index `symbol` on
    /// axis 0 of its reference input, `symbolic_view(reference: ref<f32[n]>, index: i64) -> ref<f32[]>`, and with
    /// operations exposing additional behavior to test the shared batching rule's validation.
    #[derive(Clone, Debug)]
    enum SymbolicViewOperation {
        Native(TestOperation),
        Symbolic(usize),
        AdditionalBehavior { reads: bool },
    }

    impl SymbolicViewOperation {
        /// Returns the view description used by the symbolic test operation.
        fn view(symbol: usize) -> ArrayReferenceViewTransform {
            ArrayReferenceViewTransform::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(symbol) }
        }
    }

    impl Operation for SymbolicViewOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Native(operation) => operation.name(),
                Self::Symbolic(_) => "symbolic_view",
                Self::AdditionalBehavior { .. } => "additional_behavior",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Native(operation) => operation.region_slots(),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => &[],
            }
        }

        fn infer_region_input_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
            match self {
                Self::Native(operation) => operation.infer_region_input_types(input_types, region_interfaces),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => Ok(Vec::new()),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            match self {
                Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
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

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<InputRegionProvenance> {
            match self {
                Self::Native(operation) => operation.input_region_provenance(region_index, input_index),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => None,
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Native(operation) => operation.output_region_provenance(output_index),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => Vec::new(),
            }
        }

        fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
            match self {
                Self::Native(operation) => operation.reference_output_identity_input(output_index),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => None,
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
                Self::Symbolic(_) => Cow::Owned(
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

    impl From<ReferenceIndexOperation> for SymbolicViewOperation {
        fn from(operation: ReferenceIndexOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl From<ReferenceDynamicIndexOperation> for SymbolicViewOperation {
        fn from(operation: ReferenceDynamicIndexOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl From<ReferenceSliceOperation> for SymbolicViewOperation {
        fn from(operation: ReferenceSliceOperation) -> Self {
            Self::Native(operation.into())
        }
    }

    impl ReferenceViewOperation for SymbolicViewOperation {
        type View = ArrayReferenceViewTransform;

        fn reference_view(&self, output_index: usize) -> Option<ArrayReferenceViewTransform> {
            match self {
                Self::Native(operation) => operation.reference_view(output_index),
                Self::Symbolic(symbol) if output_index == 0 => Some(Self::view(*symbol)),
                Self::AdditionalBehavior { .. } if output_index == 0 => Some(index(0, 0)),
                Self::Symbolic(_) | Self::AdditionalBehavior { .. } => None,
            }
        }

        fn validate_view(
            view: &ArrayReferenceViewTransform,
            source: &ArrayIrType,
            output: &ArrayIrType,
        ) -> Result<(), ReferenceViewValidationError> {
            TestOperation::validate_view(view, source, output)
        }

        fn reapply_view<C: Context<Type = ArrayIrType, Operation = Self>>(
            context: &C,
            view: &ArrayReferenceViewTransform,
            source: C::Value,
            symbols: &[C::Value],
        ) -> Result<C::Value, ProgramError> {
            reapply_array_reference_view(context, view, source, symbols)
        }
    }

    /// Builds `f(vector: ref<f32[2]>, index: i64) = read(symbolic_view(vector, index))`, whose view
    /// describes its index through `symbol`.
    fn symbolic_view_program(
        symbol: usize,
    ) -> Program<TestValue, SymbolicViewOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let index_value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let view = builder
            .add_instruction(SymbolicViewOperation::Symbolic(symbol), Vec::new(), vec![vector, index_value], None)
            .unwrap()[0];
        let snapshot = builder
            .add_instruction(
                SymbolicViewOperation::Native(ReferenceReadOperation::new().into()),
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
                instruction: id(0, 2),
                position: ReferenceAliasPosition::Output(0),
            }
            .to_string(),
            "operation `view` at ^0[2] declares a reference view at output 0 but describes no view",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidView {
                operation: "reference_index",
                instruction: id(0, 2),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 2),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 2),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 2),
                position: ReferenceAliasPosition::Output(0),
            }),
            ProgramError::Reference(crate::programs::references::ReferenceError::ViewAnalysis(Box::new(
                ReferenceViewAnalysisError::MissingView {
                    operation: "view",
                    instruction: id(0, 2),
                    position: ReferenceAliasPosition::Output(0),
                },
            ),)),
        );
    }

    #[test]
    fn test_reference_view_analysis_error_source() {
        use std::error::Error as _;

        let validation =
            ReferenceViewValidationError::TypeMismatch { expected: "f32[3]".to_string(), actual: "f32[2]".to_string() };
        let error = ReferenceViewAnalysisError::InvalidView {
            operation: "view",
            instruction: id(0, 0),
            position: ReferenceAliasPosition::Output(2),
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
    fn test_reference_view_path_root() {
        let root = TestPath::root();
        assert!(root.is_root());
        assert_eq!(root.steps(), &[]);
        assert_eq!(root.views().count(), 0);
        assert_eq!(root, TestPath::default());

        assert_eq!(format!("{root:?}"), "ReferenceViewPath { steps: [] }");
    }

    #[test]
    fn test_reference_view_path_steps() {
        let path = TestPath::root().with_view(index(0, 1));
        assert_eq!(path.steps().len(), 1);
        assert_eq!(path.steps()[0].view(), &index(0, 1));
        assert_eq!(path.steps()[0].bindings(), &[]);
    }

    #[test]
    fn test_reference_view_path_views() {
        let path = TestPath::root().with_view(index(0, 1)).with_view(index(0, 2));
        assert_eq!(path.views().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(path.views().rev().collect::<Vec<_>>(), vec![&index(0, 2), &index(0, 1)]);
    }

    #[test]
    fn test_reference_view_path_is_root() {
        assert!(TestPath::root().is_root());
        assert!(!TestPath::root().with_view(index(0, 1)).is_root());
    }

    #[test]
    fn test_reference_view_path_with_step() {
        let row = TestPath::root().with_view(index(0, 1));
        let symbolic = ArrayReferenceViewTransform::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) };
        let bound = row.with_step(symbolic.clone(), vec![value(0, 3)]);
        assert_eq!(bound.views().collect::<Vec<_>>(), vec![&index(0, 1), &symbolic]);
        assert_eq!(bound.steps()[1].bindings(), &[value(0, 3)]);
        assert_eq!(row.views().collect::<Vec<_>>(), vec![&index(0, 1)]);

        // Equal descriptions can select different indices when their source bindings differ.
        assert_eq!(bound, row.with_step(symbolic.clone(), vec![value(0, 3)]));
        assert_ne!(bound, row.with_step(symbolic.clone(), vec![value(0, 4)]));
        assert_ne!(bound, row.with_step(symbolic, vec![value(1, 0)]));
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
        let iteration = |region: usize| {
            TestPath::root().with_step(
                ArrayReferenceViewTransform::Index { axis: 0, index: ArrayReferenceViewIndex::Symbolic(1) },
                vec![value(region, 0)],
            )
        };
        assert_eq!(iteration(0).overlap(&iteration(0), &root), ReferenceViewOverlap::Same);
        assert_eq!(iteration(0).overlap(&TestPath::root(), &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(iteration(0).overlap(&row_1, &root), ReferenceViewOverlap::MayOverlap);
        assert_eq!(iteration(0).overlap(&iteration(1), &root), ReferenceViewOverlap::MayOverlap);
    }

    #[test]
    fn test_reference_view_analysis_new() {
        // The root carries the empty path, the row copies nothing but appends the slice, and the element appends the
        // index after the slice; the read output is not reference-typed and has no path.
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let slice = ArrayReferenceViewTransform::Slice {
            axes: vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)],
        };
        assert_eq!(analysis.path(value(0, 0)), Some(&TestPath::root()));
        assert_eq!(analysis.path(value(0, 1)), Some(&TestPath::root().with_view(slice.clone())));
        assert_eq!(analysis.path(value(0, 2)), Some(&TestPath::root().with_view(slice.clone()).with_view(index(0, 0))));
        assert_eq!(analysis.path(value(0, 3)), None);

        // Both alias edges record the output that defines the aliasing value, which is what the view analysis asked the
        // producing operation to describe.
        assert_eq!(
            analysis.analysis().alias(value(0, 1)),
            Some(ReferenceAliasEdge::new(
                id(0, 0),
                ReferenceAliasPosition::Output(0),
                value(0, 0),
                ReferenceAliasKind::View,
                true,
            )),
        );
        assert_eq!(
            analysis.analysis().alias(value(0, 2)),
            Some(ReferenceAliasEdge::new(
                id(0, 1),
                ReferenceAliasPosition::Output(0),
                value(0, 1),
                ReferenceAliasKind::View,
                true,
            )),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_handles_reverse_numbered_alias_chains() {
        let mut condition = TestBuilder::new();
        condition.add_input(reference_type([2]));
        let predicate = condition.add_constant(TestValue::Array(Array::scalar(false)));
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
        let predicate = condition.add_constant(TestValue::Array(Array::scalar(false)));
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
            analysis.analysis().alias(value(2, 3)),
            Some(ReferenceAliasEdge::new(
                id(2, 0),
                ReferenceAliasPosition::Output(1),
                value(2, 1),
                ReferenceAliasKind::Identity,
                false,
            )),
        );
        assert_eq!(
            analysis.paths().collect::<Vec<_>>(),
            vec![
                (value(0, 1), &TestPath::root()),
                (value(1, 1), &TestPath::root()),
                (value(1, 2), &TestPath::root().with_view(index(0, 1))),
                (value(2, 1), &TestPath::root()),
                (value(2, 3), &TestPath::root()),
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
                instruction: id(0, 0),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 0),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 0),
                root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
                external_source: ReferenceSource::Input { index: 0 },
            })),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_missing_views() {
        /// Array-IR family extended with one operation that declares a view alias but describes no view for it.
        #[allow(clippy::large_enum_variant)]
        #[derive(Clone, Debug)]
        enum UndescribedViewOperation {
            Native(TestOperation),
            View,
        }

        impl Operation for UndescribedViewOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                match self {
                    Self::Native(operation) => operation.name(),
                    Self::View => "undescribed_view",
                }
            }

            fn infer_output_types(
                &self,
                input_types: &[ArrayIrType],
                region_interfaces: &[RegionInterface<ArrayIrType>],
            ) -> Result<Vec<ArrayIrType>, TypeError> {
                match self {
                    Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
                    Self::View => Ok(vec![input_types[0].clone()]),
                }
            }

            fn effects(&self) -> Cow<'_, Effects> {
                match self {
                    Self::Native(operation) => operation.effects(),
                    Self::View => Cow::Owned(
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

        impl From<ReferenceIndexOperation> for UndescribedViewOperation {
            fn from(operation: ReferenceIndexOperation) -> Self {
                Self::Native(operation.into())
            }
        }

        impl From<ReferenceDynamicIndexOperation> for UndescribedViewOperation {
            fn from(operation: ReferenceDynamicIndexOperation) -> Self {
                Self::Native(operation.into())
            }
        }

        impl From<ReferenceSliceOperation> for UndescribedViewOperation {
            fn from(operation: ReferenceSliceOperation) -> Self {
                Self::Native(operation.into())
            }
        }

        impl ReferenceViewOperation for UndescribedViewOperation {
            type View = ArrayReferenceViewTransform;

            fn reference_view(&self, output_index: usize) -> Option<ArrayReferenceViewTransform> {
                match self {
                    Self::Native(operation) => operation.reference_view(output_index),
                    Self::View => None,
                }
            }

            fn validate_view(
                view: &ArrayReferenceViewTransform,
                source: &ArrayIrType,
                output: &ArrayIrType,
            ) -> Result<(), ReferenceViewValidationError> {
                TestOperation::validate_view(view, source, output)
            }

            fn reapply_view<C: Context<Type = ArrayIrType, Operation = Self>>(
                context: &C,
                view: &ArrayReferenceViewTransform,
                source: C::Value,
                symbols: &[C::Value],
            ) -> Result<C::Value, ProgramError> {
                reapply_array_reference_view(context, view, source, symbols)
            }
        }

        impl ArrayReferenceViewOperation for UndescribedViewOperation {
            fn from_reference_reshape(operation: ReshapeOperation) -> Self {
                Self::Native(TestOperation::from_reference_reshape(operation))
            }

            fn from_reference_slice(operation: SliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_slice(operation))
            }

            fn from_reference_update_slice(operation: UpdateSliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_update_slice(operation))
            }

            fn from_reference_dynamic_slice(operation: DynamicSliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_dynamic_slice(operation))
            }

            fn from_reference_dynamic_update_slice(operation: DynamicUpdateSliceOperation) -> Self {
                Self::Native(TestOperation::from_reference_dynamic_update_slice(operation))
            }
        }

        let mut builder = ProgramBuilder::<TestValue, UndescribedViewOperation>::new();
        let reference = builder.add_input(reference_type([2]));
        let view =
            builder.add_instruction(UndescribedViewOperation::View, Vec::new(), vec![reference], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                UndescribedViewOperation::Native(ReferenceReadOperation::new().into()),
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
                operation: "undescribed_view",
                instruction: id(0, 0),
                position: ReferenceAliasPosition::Output(0),
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_binds_input_symbols() {
        // The view analysis closes the description over the describing instruction: its input symbol binds to the index
        // input's identity, and the static read output has no path.
        let program = symbolic_view_program(1);
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let view = SymbolicViewOperation::view(1);
        assert_eq!(analysis.path(value(0, 0)), Some(&TestPath::root()));
        assert_eq!(analysis.path(value(0, 2)), Some(&TestPath::root().with_step(view.clone(), vec![value(0, 1)])),);
        assert_eq!(analysis.path(value(0, 3)), None);
        assert_eq!(analysis.path(value(0, 2)).map(|path| path.views().collect::<Vec<_>>()), Some(vec![&view]));
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
        assert_eq!(analysis.path(value(0, 0)), Some(&TestPath::root()));
        let path = analysis.path(value(0, 4)).unwrap();
        assert_eq!(path.steps().len(), 2);
        assert_eq!(path.steps()[0].bindings(), &[value(0, 1)]);
        assert_eq!(path.steps()[1].bindings(), &[value(0, 2)]);
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_invalid_symbols() {
        // An input symbol must name an input of the describing instruction.
        let program = symbolic_view_program(2);
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidViewSymbol {
                operation: "symbolic_view",
                instruction: id(0, 0),
                position: ReferenceAliasPosition::Output(0),
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
                instruction: id(0, 0),
                position: ReferenceAliasPosition::Output(0),
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
        assert!(analysis.analysis().is_view(value(0, 2)));

        // The structural analysis is the retained one, not a second derivation.
        let retained = program.entry_region_ref().reference_analysis(0).unwrap();
        assert!(std::ptr::eq(analysis.analysis(), &*retained));
    }

    #[test]
    fn test_reference_view_analysis_path() {
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(analysis.path(value(0, 0)), Some(&TestPath::root()));
        assert_eq!(analysis.path(value(0, 2)).map(|path| path.views().len()), Some(2));
        assert_eq!(analysis.path(value(0, 3)), None);
        assert_eq!(analysis.path(value(1, 0)), None);
    }

    #[test]
    fn test_reference_view_analysis_paths() {
        let program = chain_program();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(
            analysis.paths().map(|(value, path)| (value, path.views().len())).collect::<Vec<_>>(),
            vec![(value(0, 0), 0), (value(0, 1), 1), (value(0, 2), 2)],
        );
    }

    #[test]
    fn test_reference_view_analysis_overlap() {
        // Within one root both paths fold to root indices: the row slice may overlap with the complete root, and
        // indexing the single row of that slice selects exactly the slice's indices again. Values that are not
        // references, or that live in different regions, have no answer.
        let program = chain_program();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 0)), Some(ReferenceViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 1)), Some(ReferenceViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 1), value(0, 2)), Some(ReferenceViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 3)), None);
        assert_eq!(analysis.overlap(region, value(0, 0), value(1, 0)), None);

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
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 1)), Some(ReferenceViewOverlap::Disjoint));
        assert_eq!(analysis.overlap(region, value(0, 2), value(0, 3)), Some(ReferenceViewOverlap::Disjoint));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 2)), Some(ReferenceViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 2), value(0, 2)), Some(ReferenceViewOverlap::Same));

        // Symbolic indices compare by their bindings: two views through the same index input select the
        // same slot, while views through different inputs, or against a static index or the root, may overlap.
        let mut builder = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let index_value = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let other = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        let symbolic = SymbolicViewOperation::Symbolic(1);
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, index_value], None).unwrap();
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, index_value], None).unwrap();
        builder.add_instruction(symbolic, Vec::new(), vec![vector, other], None).unwrap();
        builder
            .add_instruction(
                SymbolicViewOperation::Native(ReferenceIndexOperation::new(0, 0).into()),
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
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 4)), Some(ReferenceViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 5)), Some(ReferenceViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 6)), Some(ReferenceViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 3)), Some(ReferenceViewOverlap::MayOverlap));
    }

    #[test]
    fn test_region_ref_reference_view_analysis() {
        let program = chain_program();
        let retained = program.entry_region_ref().reference_view_analysis(0).unwrap();
        assert_eq!(*retained, ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap());

        // A second request under the same capture scope is served the retained view analysis.
        assert!(Arc::ptr_eq(&program.entry_region_ref().reference_view_analysis(0).unwrap(), &retained));

        // A failed derivation is reported and not retained.
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
        assert_eq!(retained.path(value(0, 1)), Some(&TestPath::root().with_view(index(0, 0))));

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
        assert_eq!(derived.path(value(0, 1)), Some(&TestPath::root().with_view(index(0, 1))));
        assert_eq!(derived.path(value(1, 1)), Some(&TestPath::root()));

        // The source program keeps its own retained view analysis, because only the re-sealed copy was rebased.
        assert!(Arc::ptr_eq(&first.entry_region_ref().reference_view_analysis(0).unwrap(), &retained));
    }
    #[test]
    fn test_batch_reference_view_operation() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context =
            BatchingContext::<_, ArrayIrBatchingPolicy>::new(EagerContext::<TestValue, TestOperation>::new(), extent);
        let packed_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let reference = TestValue::Array(Array::from_f64s(packed_type, (0..6).map(f64::from).collect()))
            .reference_new()
            .unwrap();

        // A mapped source moves its batch axis through the description and binds the batched view on the parent: the
        // leading batch axis shifts the indexed per-item axis to packed axis 1 and the output keeps batch axis 0.
        let batch = ArrayIrBatch::new(reference.clone(), BatchAxis::new(0)).unwrap();
        let outputs = batch_reference_view_operation(&ReferenceIndexOperation::new(0, 2), &context, &[batch.clone()])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].batch_axis(), BatchAxis::new(0));
        assert_eq!(outputs[0].value().read(), Ok(TestValue::Array(Array::vector(vec![2.0f32, 5.0]))));

        // A replicated source is viewed unchanged and stays replicated.
        let replicated = ArrayIrBatch::replicated(reference);
        let outputs = batch_reference_view_operation(&ReferenceIndexOperation::new(0, 1), &context, &[replicated])
            .unwrap()
            .into_parts()
            .0;
        assert_eq!(outputs[0].batch_axis(), BatchAxis::replicated());
        assert_eq!(outputs[0].value().read(), Ok(TestValue::Array(Array::vector(vec![3.0f32, 4.0, 5.0]))));

        // Only view operations batch through the rule: an operation without a view alias and one with an allocation
        // output are rejected by name, and so is a mapped input other than the viewed source.
        assert_eq!(
            batch_reference_view_operation(
                &ReferenceReadOperation::<ArrayType, ArrayIrType>::new(),
                &context,
                &[batch.clone()],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`reference_read` derives no reference view".to_string(),
            }),
        );
        assert_eq!(
            batch_reference_view_operation(
                &ReferenceNewOperation::<ArrayType, ArrayIrType>::new(),
                &context,
                &[batch.clone()],
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`reference_new` has reference output 0 that is not a view, so it cannot batch as a view \
                          operation"
                    .to_string(),
            }),
        );
        assert_eq!(
            batch_reference_view_operation(&ReferenceIndexOperation::new(0, 1), &context, &[batch.clone(), batch])
                .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`reference_index` requires input 1 to be replicated; batching a reference view through a \
                          mapped index input is not supported"
                    .to_string(),
            }),
        );
    }

    #[test]
    fn test_batch_reference_view_operation_rejects_additional_behavior() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let parent = TracingContext::<TestValue, SymbolicViewOperation>::new();
        let extent = parent.lift(extent).unwrap();
        let input = ArrayIrBatch::replicated(parent.input(reference_type([2])));
        let context = BatchingContext::<_, ArrayIrBatchingPolicy>::new(parent.clone(), extent);

        // Replaying only the view would discard either the extra value result or the source read effect.
        assert_eq!(
            batch_reference_view_operation(
                &SymbolicViewOperation::AdditionalBehavior { reads: false },
                &context,
                &[input.clone()]
            )
            .err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`additional_behavior` has outputs other than its reference views".to_string(),
            }),
        );
        assert_eq!(
            batch_reference_view_operation(&SymbolicViewOperation::AdditionalBehavior { reads: true }, &context, &[input]).err(),
            Some(BatchingError::UnsupportedOperation {
                message: "`additional_behavior` has effects or attached regions that cannot be preserved by batching only its reference views".to_string(),
            }),
        );
        assert_eq!(parent.builder().borrow().instructions().len(), 0);
    }
}
