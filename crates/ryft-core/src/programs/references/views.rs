//! Value-family-generic static view contract and the view-path overlay over the structural [`ReferenceAnalysis`].
//!
//! The generic analysis records *that* a reference-typed value is a narrowing view of its root, through
//! [`ReferenceAliasEdge`](crate::programs::references::ReferenceAliasEdge)s of kind
//! [`ReferenceAliasKind::View`], but it deliberately stores no geometry: what a view selects is a property of the
//! value family (an array view is an index or slice, a downstream family may split a register into halves). This
//! module lifts that geometry into one static contract, [`ReferenceViewOperation`], that an operation family
//! implements once, and one overlay, [`ReferenceViewAnalysis`], that composes the per-edge descriptions into a
//! [`ReferenceViewPath`] for every reference-typed value. Transforms that rebuild references (tangent, cotangent, and
//! residual reconstruction) consult the overlay and reapply descriptions through the same contract, so no transform
//! ever matches view operations by name and the array family's `reference_index` and `reference_slice` are not
//! special-cased anywhere.
//!
//! Everything here is static dispatch on the operation family `O`: descriptions are owned data, validation and
//! reapplication are associated functions of the family, and no trait object or `Self`-outside-the-receiver appears,
//! so the contract composes with the closed operation enums that backends own.
//!
//! # Symbols And Bindings
//!
//! A description is a value of the family's [`ReferenceView`] type and may depend on coordinates that are not part of
//! the description itself: a traced operand of the describing instruction, or the iteration counter of a
//! region-carrying instruction whose region input the view describes. Such coordinates are *symbolic* in the
//! description and named relative to the describing operation by [`ViewSymbol`]s, in the order
//! [`ReferenceView::symbols`] reports them. The overlay *closes* every description over the instruction that created
//! it: each symbol becomes a [`ViewSymbolBinding`] (the program identity of the operand value, or the region whose
//! iteration counter is meant), and the description together with its bindings forms one [`ReferenceViewStep`] of a
//! [`ReferenceViewPath`]. Static descriptions report no symbols and carry empty bindings, so equality of static paths
//! is equality of their description sequences.
//!
//! The binding type is a parameter of the path because the same path shape serves consumers that close symbols
//! differently: the overlay binds program identities, an eager handle carries only static steps and uses the
//! uninhabited [`NoBinding`], and a discharge policy may close symbols over destination values.
//!
//! # Compatibility Is Geometry, Not Root Type
//!
//! A description is validated against the *current* source reference type before it is reapplied. A tangent or
//! cotangent root may have a different referent type from the primal root (e.g., a widened floating-point tangent
//! type), so a description that was valid on the primal root is re-checked against the transformed root rather than
//! assumed to transfer.
//!
//! # Batching Moves The Axis Through The Mapping
//!
//! [`ReferenceViewOperation::reapply_view`] rebuilds a description over a root of the same geometry as the one it was
//! derived on, which is what tangent, cotangent, and residual reconstruction need. Batching is different: it inserts an
//! axis into the packed root, and a primal description reapplied unchanged to a batched root would index or slice the
//! wrong axis. The contract therefore splits the two concerns. [`ReferenceView::batch`] is pure axis arithmetic on the
//! description: given the packed source type and the source's batch axis, it returns the description that selects the
//! same per-item coordinates of the packed source together with the batch axis of the derived reference. The shared
//! rule [`batch_reference_view_operation`] then binds that batched description through `reapply_view` on the parent
//! context, so every view operation of every family batches through one rule and no operation carries the axis
//! arithmetic itself.
//!
//! # Overlap Queries
//!
//! Two paths of one root may select the same coordinates, provably different coordinates, or coordinates whose
//! relation is not decidable statically. [`ReferenceView::overlap`] answers that question for two closed paths of one
//! root as a [`ViewOverlap`]; [`ReferenceViewPath::overlap`] and [`ReferenceViewAnalysis::overlap`] expose it on paths
//! and on analyzed values. Two symbolic coordinates are the same coordinate exactly when their bindings are equal, so a
//! path that selects a slot by one instruction's operand and a path that selects a slot by the same operand agree,
//! while two different operands, or an operand against a static coordinate, may overlap. The query is what lets a
//! transform admit two handles of one root side by side only when they are provably disjoint (e.g., a stacked
//! reference operand of a `scan` viewed per iteration next to a carry of the same root).

// TODO(eaplatanios): Review this module.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::hash::Hash;
use std::sync::Arc;

use thiserror::Error;

use crate::batching::{BatchAxis, BatchedOutputs, BatchingContext, BatchingError, BatchingPolicy};
use crate::contexts::Context;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::references::analysis::{
    ReferenceAliasOrigin, ReferenceAnalysis, ReferenceAnalysisError, ReferenceAnalysisTransformArguments, ReferenceRoot,
};
use crate::programs::references::semantics::{ReferenceAliasKind, ReferenceOutput};
use crate::programs::regions::{Region, RegionId, RegionRef};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::{Type, Typed};
use crate::programs::values::{Value, ValueId};

/// Error produced by [`ReferenceViewOperation::validate_view`] when one view description does not compose onto its
/// source reference type or does not derive the declared output reference type.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
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
/// cannot be reconciled with the program's declared reference types.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ReferenceViewAnalysisError {
    /// The generic reference analysis rejected the region closure.
    #[error(transparent)]
    Analysis(#[from] ReferenceAnalysisError),

    /// An operation declares a view alias in its reference semantics but describes no view for that output.
    #[error("operation `{operation}` at {instruction} derives a reference view but exposes no view transform")]
    MissingView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,
    },

    /// A view operation declares an output referent type that differs from the referent its description derives from
    /// the source referent.
    #[error(
        "operation `{operation}` at {instruction} declares view referent type `{actual}` but its transform derives \
         referent type `{expected}` from the source view"
    )]
    ViewTypeMismatch {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Referent type derived by the operation's description.
        expected: String,

        /// Referent type declared by the operation's output.
        actual: String,
    },

    /// A view operation's description cannot be applied to the reference type of its source.
    #[error("operation `{operation}` at {instruction} composes an invalid view transform: {message}")]
    InvalidViewComposition {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Description of why the view is invalid for the source.
        message: String,
    },

    /// A view operation's description depends on an operand symbol that names no non-reference operand of the
    /// describing instruction.
    #[error("operation `{operation}` at {instruction} describes a view through operand {operand_index}, but {message}")]
    InvalidViewSymbolOperand {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Operand named by the symbol.
        operand_index: usize,

        /// Description of why the operand cannot be a view coordinate.
        message: String,
    },

    /// A view operation describes an output view through the iteration symbol, which only describes region inputs.
    #[error(
        "operation `{operation}` at {instruction} describes output {output_index} through the iteration counter, but \
         an iteration symbol only describes region inputs"
    )]
    IterationSymbolAtOutput {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Output described through the iteration symbol.
        output_index: usize,
    },

    /// An operation declares through [`Operation::region_input_view_source`] that a region input is a boundary view
    /// it creates but describes no view for that input.
    #[error(
        "operation `{operation}` at {instruction} creates a boundary view for input {input_index} of region \
         {region_index} but exposes no view transform"
    )]
    MissingBoundaryView {
        /// Name of the operation.
        operation: &'static str,

        /// Instruction applying the operation.
        instruction: InstructionId,

        /// Position of the attached region among the instruction's regions.
        region_index: usize,

        /// Reference-typed input of the attached region.
        input_index: usize,
    },
}

impl From<ReferenceViewAnalysisError> for ProgramError {
    #[inline]
    fn from(error: ReferenceViewAnalysisError) -> Self {
        ProgramError::MalformedProgram(error.to_string())
    }
}

/// A coordinate that a view description depends on but does not contain, named relative to the operation that
/// describes the view.
///
/// A description lists its symbols through [`ReferenceView::symbols`], and every consumer that must resolve them
/// receives one value or binding per symbol in that order: the [`ReferenceViewAnalysis`] closes them into
/// [`ViewSymbolBinding`]s, and [`ReferenceViewOperation::reapply_view`] receives their values.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewSymbol {
    /// Operand `index` of the describing instruction, which must be a non-reference value (a traced coordinate).
    Operand(usize),

    /// The iteration counter of the region-carrying instruction whose region input the view describes. An iteration
    /// symbol never describes an instruction output, and a view depending on it is created by its operation at the
    /// region boundary rather than reapplied.
    Iteration,
}

impl Parameter for ViewSymbol {}

/// Program identity that a [`ViewSymbol`] is bound to once the [`ReferenceViewAnalysis`] closes a description over
/// the instruction that created it.
///
/// An [`Iteration`](ViewSymbol::Iteration) symbol binds to the region whose inputs the view describes rather than to
/// the instruction that attaches that region, so a region shared by several attaching instructions has one
/// attachment-independent path per nested input.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewSymbolBinding {
    /// The operand value named by an [`Operand`](ViewSymbol::Operand) symbol.
    Value(ValueId),

    /// The iteration counter of the region named by an [`Iteration`](ViewSymbol::Iteration) symbol.
    Iteration(RegionId),
}

impl Parameter for ViewSymbolBinding {}

/// Uninhabited binding of paths that only ever carry static steps, such as the path of an eager array-reference
/// handle: every step of such a path has empty bindings, and a description with symbols cannot be represented on it.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum NoBinding {}

impl Parameter for NoBinding {}

/// Coordinate-level relation between two closed root-relative paths of one reference root, as decided statically by
/// [`ReferenceView::overlap`].
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum ViewOverlap {
    /// The two paths provably select no common coordinate of the root.
    Disjoint,

    /// The two paths provably select exactly the same coordinates of the root.
    Same,

    /// The two paths may share coordinates: they select different but intersecting static coordinates, or at least
    /// one coordinate is symbolic and not provably equal to (or different from) its counterpart.
    MayOverlap,
}

/// Owned description of one view step of a reference family, from a source reference to the reference it derives.
///
/// The bounds are what the retained [`ReferenceViewAnalysis`] needs to live in a region's transform cache and to be
/// revalidated against a fresh derivation, plus hashing so that paths of descriptions can key eager handles. A
/// description may depend on coordinates outside itself, which it names through [`symbols`](Self::symbols); refer to
/// the module documentation for how the overlay closes them.
pub trait ReferenceView: 'static + Clone + Debug + PartialEq + Eq + Hash + Send + Sync {
    /// Reference type family the description addresses.
    type Type: Type;

    /// Returns the symbols this description depends on, in the order their bindings and values are supplied to every
    /// consumer. Static descriptions return no symbols.
    fn symbols(&self) -> Vec<ViewSymbol>;

    /// Moves the batch axis of a source reference through this mapping. The batch axis of a reference is an axis of
    /// its packed referent that the per-item view never sees, so the batched description must select the same
    /// per-item coordinates of the packed source that this description selects of the unbatched one, and the derived
    /// reference has its own batch axis. This is pure axis arithmetic: the symbols of the description are untouched,
    /// and a replicated `batch_axis` returns the description unchanged and replicated.
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

    /// Returns the relation between the coordinates that two closed root-relative paths `a` and `b` select of one
    /// root of type `root`. Two symbolic coordinates are the same coordinate iff their bindings are equal; a symbolic
    /// and a static coordinate on one axis, or two symbolic coordinates with different bindings, may overlap. The
    /// empty path denotes the complete root, so it is [`Same`](ViewOverlap::Same) as itself and may overlap with any
    /// narrowing path. Paths are validated when they are derived, so implementations may treat a malformed path
    /// conservatively as [`MayOverlap`](ViewOverlap::MayOverlap) instead of failing.
    fn overlap(root: &Self::Type, a: &[ReferenceViewStep<Self>], b: &[ReferenceViewStep<Self>]) -> ViewOverlap;
}

/// Static view contract of one operation family: the owned description of every view alias the family can derive,
/// its type-level validation, and its reapplication to another reference of compatible geometry.
///
/// An operation family implements this trait once, and every transform that rebuilds references (tangent, cotangent,
/// and residual reconstruction) then reaches the family's views through it: [`ReferenceViewAnalysis`] composes the
/// descriptions into per-value [`ReferenceViewPath`]s, and reconstruction reapplies them step by step to the
/// transformed root. Compatibility is geometry, not root type: a tangent or cotangent root may have a different
/// referent type from the primal root, so each description is validated against the current transformed source type
/// before it is reapplied. Batching does not reapply a primal description unchanged, because a batched root has an
/// extra axis; it first moves the batch axis through the description with [`ReferenceView::batch`] and then reapplies
/// the batched description, which is what [`batch_reference_view_operation`] does for every view operation.
pub trait ReferenceViewOperation: Operation {
    /// Description of one view step of this family, addressing this family's reference types.
    type View: ReferenceView<Type = Self::Type>;

    /// Returns the description of the view this operation derives at output `output_index`, or [`None`] when that
    /// output is not a view. Exactly the outputs whose [`reference_semantics`](Operation::reference_semantics) declare
    /// a [`ReferenceOutput::Alias`](crate::programs::references::ReferenceOutput::Alias) of kind
    /// [`ReferenceAliasKind::View`] return [`Some`]; an operation that declares such an alias but returns [`None`] is
    /// rejected by the overlay with [`ReferenceViewAnalysisError::MissingView`]. An output description may name
    /// [`ViewSymbol::Operand`] symbols but never [`ViewSymbol::Iteration`].
    fn reference_view(&self, output_index: usize) -> Option<Self::View>;

    /// Returns the description of the view this operation creates for the reference-typed input `input_index` of its
    /// attached region `region_index` (e.g., the per-iteration slice of a stacked reference operand), or [`None`] when
    /// that input is a complete handle of a caller root. Such a view is created by the operation at the region
    /// boundary, so its description may name [`ViewSymbol::Iteration`]. The default describes no boundary views.
    fn region_input_view(&self, region_index: usize, input_index: usize) -> Option<Self::View> {
        let _ = (region_index, input_index);
        None
    }

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
    /// compatible geometry; callers validate the step with [`validate_view`](Self::validate_view) against the current
    /// source type first.
    ///
    /// # Parameters
    ///
    ///   - `context`: Context of this operation family through which the view is staged.
    ///   - `view`: Description to reapply.
    ///   - `source`: Reference the view is applied to.
    ///   - `symbols`: One value per entry of [`view.symbols()`](ReferenceView::symbols), in that order. An
    ///     [`Iteration`](ViewSymbol::Iteration) symbol has no value: boundary views are created by their operation,
    ///     never reapplied, so a description naming one is rejected.
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
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceViewStep<View, Binding = ViewSymbolBinding> {
    /// Description of the step.
    view: View,

    /// Binding of each symbol of `view`, in the description's symbol order.
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

impl<View: Parameter, Binding: Parameter> Parameter for ReferenceViewStep<View, Binding> {}

/// Ordered closed view steps from a reference root to one derived reference-typed value.
///
/// The path stores only the steps, in root-to-value order; the root itself is a property of the structural
/// [`ReferenceAnalysis`]. The empty path is the identity and denotes the complete root, so root handles, region inputs,
/// capture constants, and forwarded region outputs all carry it. Equality and hashing distinguish different step
/// sequences, not the values they were derived for. `Binding` is what each step's symbols are closed over: the overlay
/// binds program identities ([`ViewSymbolBinding`]), and a path that only ever carries static steps uses
/// [`NoBinding`]. Refer to the module documentation for more information.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceViewPath<View, Binding = ViewSymbolBinding> {
    /// Steps applied from the root outward.
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

    /// Returns this path extended by one more step applied to its current end, closing the symbols of `view` over
    /// `bindings` (one per symbol, in the description's symbol order).
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

    /// Returns this path extended by one more static description applied to its current end; the shorthand of
    /// [`with_step`](Self::with_step) with no bindings.
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
    /// Returns the relation between the coordinates this path and `other` select of one root of type `root`, through
    /// [`ReferenceView::overlap`]. Both paths must be root-relative paths of the same root; callers that compare
    /// analyzed values of one region use [`ReferenceViewAnalysis::overlap`], which checks the roots first, while this
    /// function serves callers that resolve roots across namespaces themselves.
    #[inline]
    pub fn overlap(&self, other: &Self, root: &View::Type) -> ViewOverlap {
        View::overlap(root, self.steps(), other.steps())
    }
}

impl<View, Binding> Default for ReferenceViewPath<View, Binding> {
    #[inline]
    fn default() -> Self {
        Self::root()
    }
}

impl<View: Parameter, Binding: Parameter> Parameter for ReferenceViewPath<View, Binding> {}

/// Batches one reference-view operation of the family of `C` through the [`ReferenceView`] contract: the shared
/// [`BatchableOperation`](crate::batching::BatchableOperation) rule of every operation whose reference semantics
/// declare only [`View`](ReferenceAliasKind::View) aliases of one source operand.
///
/// The rule reads the operation's [`reference_semantics`](Operation::reference_semantics) to find the single source
/// operand and the view outputs, requires every other operand (the coordinate operands named by the views' symbols) to
/// be replicated, and then, for each view output in output order, moves the source's batch axis through the description
/// with [`ReferenceView::batch`] and binds the batched description over the packed source through
/// [`ReferenceViewOperation::reapply_view`] on the parent context, supplying the packed value of each operand a symbol
/// names. Each reapplication binds one operation on the parent, so an operation with several view outputs (e.g., a
/// family that splits a register into two halves) is bound once per output, each time keeping the output the
/// description denotes.
///
/// # Parameters
///
///   - `operation`: Member operation to batch, converted into the family operation `C::Operation` to reach the
///     family's view contract.
///   - `context`: Batching context whose parent the batched views are bound on.
///   - `inputs`: Batch carriers of the operation's operands.
///
/// # Errors
///
/// Returns [`BatchingError::UnsupportedOperation`] when the operation derives no view, views more than one source
/// operand, has reference outputs other than its views, has a mapped non-source operand (batching a view through a
/// mapped coordinate is not supported), or describes a view through the iteration counter, which only describes region
/// inputs. Propagates the [`BatchingError`] of [`ReferenceView::batch`] and the errors of the parent context's binding.
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
    let unsupported = |message: String| BatchingError::UnsupportedOperation { message };
    let semantics = operation.reference_semantics();
    let mut source_index = None;
    let mut output_indices = Vec::with_capacity(semantics.outputs().len());
    for output in semantics.outputs() {
        let ReferenceOutput::Alias { output_index, input_index, kind: ReferenceAliasKind::View } = *output else {
            return Err(unsupported(format!(
                "`{name}` has reference output {} that is not a view, so it cannot batch as a view operation",
                output.output_index(),
            )));
        };
        match source_index {
            None => source_index = Some(input_index),
            Some(source_index) if source_index == input_index => {}
            Some(source_index) => {
                return Err(unsupported(format!(
                    "`{name}` views operands {source_index} and {input_index}, but a view operation views one source",
                )));
            }
        }
        output_indices.push(output_index);
    }
    let Some(source_index) = source_index else {
        return Err(unsupported(format!("`{name}` derives no reference view")));
    };
    output_indices.sort_unstable();
    if output_indices.iter().enumerate().any(|(position, output_index)| position != *output_index) {
        return Err(unsupported(format!("`{name}` has outputs other than its reference views")));
    }
    let Some(source) = inputs.get(source_index) else {
        return Err(ProgramError::MalformedProgram(format!(
            "`{name}` views operand {source_index} but was applied to {} operands",
            inputs.len(),
        ))
        .into());
    };
    for (input_index, input) in inputs.iter().enumerate() {
        if input_index != source_index && !P::batch_axis(input).is_replicated() {
            return Err(unsupported(format!(
                "`{name}` requires operand {input_index} to be replicated; batching a reference view through a mapped \
                 coordinate operand is not supported",
            )));
        }
    }
    let source_value = P::value(source);
    let source_axis = P::batch_axis(source);
    let source_type = source_value.r#type();
    let mut outputs = Vec::with_capacity(output_indices.len());
    for output_index in output_indices {
        let Some(view) = operation.reference_view(output_index) else {
            return Err(ProgramError::MalformedProgram(format!(
                "operation `{name}` derives a reference view at output {output_index} but exposes no view transform",
            ))
            .into());
        };
        let symbols = view
            .symbols()
            .into_iter()
            .map(|symbol| match symbol {
                ViewSymbol::Operand(operand_index) => {
                    inputs.get(operand_index).map(|input| P::value(input).clone()).ok_or_else(|| {
                        BatchingError::from(ProgramError::MalformedProgram(format!(
                            "`{name}` describes output {output_index} through operand {operand_index} but was applied \
                             to {} operands",
                            inputs.len(),
                        )))
                    })
                }
                ViewSymbol::Iteration => Err(unsupported(format!(
                    "`{name}` describes output {output_index} through the iteration counter, which only describes \
                     region inputs",
                ))),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (view, output_axis) = view.batch(source_type.as_ref(), source_axis)?;
        let value = C::Operation::reapply_view(context.parent(), &view, source_value.clone(), symbols.as_slice())?;
        outputs.push(P::batch(value, output_axis)?);
    }
    Ok(outputs.into())
}

/// Structural [`ReferenceAnalysis`] of a [`Region`] closure together with the [`ReferenceViewPath`] of every
/// reference-typed value in that closure, derived through the [`ReferenceViewOperation`] contract of the closure's
/// operation family.
///
/// Every reference-typed value has exactly one path. A root handle (a region input, an allocation, a capture constant,
/// or a forwarded region output) has the empty path, an identity alias copies the path of its source, and a view alias
/// copies the path of its source and appends the description its producing operation reports for that edge's output,
/// after that description was validated against the source and output reference types. Nested region inputs are
/// separate roots of the structural analysis: an input forwarded as a complete handle (identity provenance) carries
/// the empty path, and an input the attaching operation creates as a boundary view (declared through
/// [`Operation::region_input_view_source`]) carries the single step that operation reports through
/// [`ReferenceViewOperation::region_input_view`], closed over the attached region, so its
/// [`Iteration`](ViewSymbol::Iteration) symbol binds to that region and the path is the same for every attachment of a
/// shared region. Either way the path is relative to the nested input itself, not to the caller root it is bound to.
///
/// The overlay is retained in the region's transform cache under exactly the cache identity of the structural analysis
/// (refer to the documentation of [`RegionRef::reference_view_analysis`]) and shares that analysis through an [`Arc`]
/// rather than re-deriving it. Like the structural analysis, it is kernel-owned validation infrastructure that
/// consumers invoke explicitly on the programs they own.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReferenceViewAnalysis<View> {
    /// Structural analysis of the closure.
    analysis: Arc<ReferenceAnalysis>,

    /// Path of every reference-typed value of the closure, in canonical value order.
    paths: BTreeMap<ValueId, ReferenceViewPath<View>>,
}

impl<View> ReferenceViewAnalysis<View> {
    /// Analyzes the complete closure of `region` and derives the [`ReferenceViewPath`] of every reference-typed value
    /// in it. The structural analysis is obtained through [`RegionRef::reference_analysis`], so it is shared with every
    /// other consumer of the same closure; this function itself is the uncached derivation of the overlay, and
    /// [`RegionRef::reference_view_analysis`] is its retained counterpart. Refer to the documentation of
    /// [`ReferenceAnalysis::new`] for the meaning of `capture_count`.
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
        Self::new_with_arguments(region, &ReferenceAnalysisTransformArguments::new(region, capture_count))
    }

    /// Derives the overlay exactly like [`new`](Self::new), obtaining the structural analysis under the already-derived
    /// cache key `arguments` so that the closure is walked once per derivation.
    pub(crate) fn new_with_arguments<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        arguments: &ReferenceAnalysisTransformArguments,
    ) -> Result<Self, ReferenceViewAnalysisError>
    where
        // Implied by `O::View`'s bounds, but the trait solver does not carry them through the projection equality.
        View: ReferenceView,
    {
        let analysis = region.reference_analysis_with_arguments(arguments)?;
        let mut paths = BTreeMap::new();
        for value in analysis.values() {
            Self::derive_path(region, &analysis, &mut paths, value)?;
        }
        Ok(Self { analysis, paths })
    }

    /// Derives the path of `value` into `paths`, deriving the path of its alias source first when needed, and returns
    /// it. Alias chains are acyclic and end at a root, so the recursion terminates, and a value is derived once.
    fn derive_path<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        region: RegionRef<'_, V, O>,
        analysis: &ReferenceAnalysis,
        paths: &mut BTreeMap<ValueId, ReferenceViewPath<View>>,
        value: ValueId,
    ) -> Result<ReferenceViewPath<View>, ReferenceViewAnalysisError>
    where
        View: ReferenceView,
    {
        if let Some(path) = paths.get(&value) {
            return Ok(path.clone());
        }
        let path = match analysis.alias(value) {
            None => ReferenceViewPath::root(),
            Some(edge) => {
                let source = Self::derive_path(region, analysis, paths, edge.source())?;
                match edge.kind() {
                    ReferenceAliasKind::Identity => source,
                    ReferenceAliasKind::View => {
                        // The structural analysis resolved every region of the closure and recorded this edge from an
                        // instruction of the region containing the edge's source, so none of the lookups can fail
                        // here. The described value lives in that same region for an output and in the attached
                        // region for a boundary view, so its type is read from the region of `value` rather than
                        // from the describing instruction's region.
                        let id = edge.instruction();
                        let current = region.with_id(id.region()).unwrap();
                        let instruction = &current.instructions()[id.index()];
                        let operation = instruction.operation();
                        let name = operation.name();
                        let origin = edge.origin();
                        let view = match origin {
                            ReferenceAliasOrigin::Output(output_index) => operation
                                .reference_view(output_index)
                                .ok_or(ReferenceViewAnalysisError::MissingView { operation: name, instruction: id })?,
                            ReferenceAliasOrigin::RegionInput { region_index, input_index } => operation
                                .region_input_view(region_index, input_index)
                                .ok_or(ReferenceViewAnalysisError::MissingBoundaryView {
                                    operation: name,
                                    instruction: id,
                                    region_index,
                                    input_index,
                                })?,
                        };
                        let atoms = current.atoms();
                        let source_type = atoms[edge.source().atom().index()].r#type();
                        let output_type =
                            region.with_id(value.region()).unwrap().atoms()[value.atom().index()].r#type();
                        O::validate_view(&view, source_type.as_ref(), output_type.as_ref()).map_err(
                            |error| match error {
                                ReferenceViewValidationError::TypeMismatch { expected, actual } => {
                                    ReferenceViewAnalysisError::ViewTypeMismatch {
                                        operation: name,
                                        instruction: id,
                                        expected,
                                        actual,
                                    }
                                }
                                ReferenceViewValidationError::InvalidComposition { message } => {
                                    ReferenceViewAnalysisError::InvalidViewComposition {
                                        operation: name,
                                        instruction: id,
                                        message,
                                    }
                                }
                            },
                        )?;
                        // Closing the description over the instruction that created it: an operand symbol binds to
                        // the named operand value, which must be a non-reference operand of this instruction, and an
                        // iteration symbol binds to the attached region whose input the view describes (never to the
                        // attaching instruction, so a shared region has one attachment-independent path per input)
                        // and never describes an instruction output.
                        let inputs = instruction.inputs();
                        let mut bindings = Vec::new();
                        for symbol in view.symbols() {
                            match symbol {
                                ViewSymbol::Operand(operand_index) => {
                                    let Some(atom) = inputs.get(operand_index) else {
                                        return Err(ReferenceViewAnalysisError::InvalidViewSymbolOperand {
                                            operation: name,
                                            instruction: id,
                                            operand_index,
                                            message: format!("the instruction has only {} operands", inputs.len()),
                                        });
                                    };
                                    if atoms[atom.index()].r#type().is_reference() {
                                        return Err(ReferenceViewAnalysisError::InvalidViewSymbolOperand {
                                            operation: name,
                                            instruction: id,
                                            operand_index,
                                            message: "that operand is a reference rather than a coordinate value"
                                                .to_string(),
                                        });
                                    }
                                    bindings.push(ViewSymbolBinding::Value(ValueId::new(id.region(), *atom)));
                                }
                                ViewSymbol::Iteration => match origin {
                                    ReferenceAliasOrigin::Output(output_index) => {
                                        return Err(ReferenceViewAnalysisError::IterationSymbolAtOutput {
                                            operation: name,
                                            instruction: id,
                                            output_index,
                                        });
                                    }
                                    ReferenceAliasOrigin::RegionInput { .. } => {
                                        bindings.push(ViewSymbolBinding::Iteration(value.region()));
                                    }
                                },
                            }
                        }
                        source.with_step(view, bindings)
                    }
                }
            }
        };
        paths.insert(value, path.clone());
        Ok(path)
    }

    /// Returns the structural [`ReferenceAnalysis`] of the closure.
    #[inline]
    pub fn analysis(&self) -> &ReferenceAnalysis {
        &self.analysis
    }

    /// Returns the [`ReferenceViewPath`] from the root of the reference-typed `value` to the coordinates it selects,
    /// or [`None`] when `value` is not a reference-typed value of the closure. Root handles carry the empty path, and
    /// so does a nested region input forwarded as a complete handle, while a nested region input created as a boundary
    /// view by its attaching operation carries that operation's step closed over the attached region.
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

    /// Returns the [`ViewOverlap`] between the coordinates that the reference-typed values `a` and `b` of one region
    /// select, or [`None`] when either is not a reference-typed value of the closure or the two values belong to
    /// different regions. Roots are region-relative (a nested region input is a root of its own namespace even when it
    /// carries a caller root), so only values of one region have comparable roots: values of different roots are
    /// [`Disjoint`](ViewOverlap::Disjoint), and values of one root delegate to [`ReferenceView::overlap`] with the type
    /// of the root's defining atom, read from `region`, the closure this overlay was derived for. The overlay retains
    /// no types itself, because the region's transform cache holds it behind a `Send + Sync` erasure that the value
    /// family's type is not required to satisfy. Callers that compare paths across namespaces resolve the roots
    /// themselves and use [`ReferenceViewPath::overlap`].
    pub fn overlap<V: Value, O: ReferenceViewOperation<Type = V::Type, View = View>>(
        &self,
        region: RegionRef<'_, V, O>,
        a: ValueId,
        b: ValueId,
    ) -> Option<ViewOverlap>
    where
        // Implied by `O::View`'s bounds, but the trait solver does not carry them through the projection equality.
        View: ReferenceView<Type = V::Type>,
    {
        if a.region() != b.region() {
            return None;
        }
        let root = self.analysis.root_of(a)?;
        if root != self.analysis.root_of(b)? {
            return Some(ViewOverlap::Disjoint);
        }
        // Every root is defined exactly once, by the input atom or the allocating instruction output that names it,
        // and every reference-typed value has a path, so once both roots resolved the remaining lookups cannot fail
        // for the region this overlay was derived for.
        let current = region.with_id(root.region()).ok()?;
        let atom = match root {
            ReferenceRoot::RegionInput { input_index, .. } => *current.input_ids().get(input_index)?,
            ReferenceRoot::Allocation { instruction, output_index } => {
                *current.instructions().get(instruction.index())?.outputs().get(output_index)?
            }
        };
        let root_type = current.atoms().get(atom.index())?.r#type();
        Some(self.paths[&a].overlap(&self.paths[&b], root_type.as_ref()))
    }
}

/// [`Region`] [`Transform`] marker for retained [`ReferenceViewAnalysis`] artifacts.
pub(crate) struct ReferenceViewAnalysisTransform;

impl<V: Value, O: ReferenceViewOperation<Type = V::Type>> Transform<Region<V, O>> for ReferenceViewAnalysisTransform {
    type Arguments = ReferenceAnalysisTransformArguments;
    type Artifact = TransformArtifact<V, O, Arc<ReferenceViewAnalysis<O::View>>>;

    const DEFAULT_CACHE_CAPACITY: usize = 2;
}

impl<'r, V: Value, O: ReferenceViewOperation<Type = V::Type>> RegionRef<'r, V, O> {
    /// Returns the [`ReferenceViewAnalysis`] of this [`Region`]'s closure, retained in the region's transform cache
    /// under exactly the cache identity of [`reference_analysis`](Self::reference_analysis): the same
    /// `capture_count` and closure region identifiers key both, so a topology-preserving import that renumbers regions
    /// derives its own overlay instead of being served paths keyed by another arena's identifiers, and repeated
    /// analysis of an unmoved region hits. The overlay shares the retained structural analysis rather than deriving a
    /// second one. Refer to the documentation of [`ReferenceViewAnalysis::new`] for the overlay itself; that function
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
        let arguments = ReferenceAnalysisTransformArguments::new(self, capture_count);
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

#[cfg(test)]
mod tests {
    use std::borrow::Cow;

    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::batching::{ArrayIrBatch, ArrayIrBatching};
    use crate::arrays::dimensions::DimensionValue;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{
        ArrayIrOperation, ArrayReferenceViewOperation, REFERENCE_INDEX_OPERATION_NAME, ReferenceIndexOperation,
        ReferenceSliceOperation, reapply_array_reference_view,
    };
    use crate::arrays::reference_views::{ArrayReferenceViewTransform, ViewIndex};
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{DimensionBounds, DimensionType, DimensionVariable};
    use crate::arrays::types::ir::ArrayIrType;
    use crate::contexts::EagerContext;
    use crate::operations::{
        ConditionOperation, ReshapeOperation, SCAN_OPERATION_NAME, ScanOperation, SliceOperation, UpdateSliceOperation,
        WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::instructions::Instruction;
    use crate::programs::programs::Program;
    use crate::programs::references::analysis::{ReferenceAliasEdge, ReferenceAliasOrigin};
    use crate::programs::references::operations::{
        ReferenceNew, ReferenceNewOperation, ReferenceRead, ReferenceReadOperation, ReferenceWriteOperation,
    };
    use crate::programs::references::semantics::{ReferenceOperationSemantics, ReferenceOutput};
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{OutputRegionProvenance, RegionId, RegionInterface, RegionSlot};
    use crate::programs::types::TypeError;

    use super::*;

    type TestValue = ArrayIrValue<Array>;

    type TestOperation = ArrayIrOperation<Array>;

    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    type TestProgram = Program<TestValue, TestOperation, Vec<TestValue>, Vec<TestValue>>;

    type TestPath = ReferenceViewPath<ArrayReferenceViewTransform>;

    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    fn index(axis: usize, index: usize) -> ArrayReferenceViewTransform {
        ArrayReferenceViewTransform::Index { axis, index: ViewIndex::Static(index) }
    }

    /// Returns the boundary view a `scan` creates for a stacked reference operand: its leading axis indexed by the
    /// iteration counter.
    fn iteration_view() -> ArrayReferenceViewTransform {
        ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Iteration) }
    }

    /// Builds a scan body `(carry: f32[], element: ref<f32[2]>) -> [carry]` that reads the per-iteration view
    /// `element` of a stacked reference operand.
    fn reading_scan_body() -> TestProgram {
        let mut body = TestBuilder::new();
        let carry = body.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let element = body.add_input(reference_type([2]));
        body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap();
        body.build(vec![carry], vec![Placeholder; 2], vec![Placeholder]).unwrap()
    }

    /// Builds `f(carry: f32[], stack: ref<f32[3, 2]>) = scan(body, carry, stack)` over the provided `body`, whose
    /// second input is the per-iteration view `ref<f32[2]>` the scan creates from the stacked reference operand. The
    /// body is region `^0` and the entry region is `^1`.
    fn stacked_scan_program(body: TestProgram) -> TestProgram {
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let stack = builder.add_input(reference_type([3, 2]));
        let outputs = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 3), vec![body], vec![carry, stack], None)
            .unwrap()
            .to_vec();
        builder.build(outputs, vec![Placeholder; 2], vec![Placeholder]).unwrap()
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

    /// Array-IR family extended with one two-operand view operation whose description selects coordinate `symbol` on
    /// axis 0 of its reference operand, `symbolic_view(reference: ref<f32[n]>, coordinate: f32) -> ref<f32[]>`, and
    /// with a `scan` that declares its stacked reference operands as boundary views of its body but describes none.
    #[derive(Clone, Debug)]
    enum SymbolicViewOperation {
        Native(TestOperation),
        Symbolic(ViewSymbol),
        UndescribedScan(ScanOperation<TestValue>),
    }

    impl SymbolicViewOperation {
        fn view(symbol: ViewSymbol) -> ArrayReferenceViewTransform {
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(symbol) }
        }
    }

    impl Operation for SymbolicViewOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Native(operation) => operation.name(),
                Self::Symbolic(_) => "symbolic_view",
                Self::UndescribedScan(_) => "undescribed_scan",
            }
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::Native(operation) => operation.region_slots(),
                Self::Symbolic(_) => &[],
                Self::UndescribedScan(operation) => operation.region_slots(),
            }
        }

        fn infer_region_input_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<Option<Vec<ArrayIrType>>>, TypeError> {
            match self {
                Self::Native(operation) => operation.infer_region_input_types(input_types, region_interfaces),
                Self::Symbolic(_) => Ok(Vec::new()),
                Self::UndescribedScan(operation) => operation.infer_region_input_types(input_types, region_interfaces),
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            match self {
                Self::Native(operation) => operation.infer_output_types(input_types, region_interfaces),
                Self::Symbolic(symbol) => {
                    let reference = <&ReferenceType<ArrayType>>::try_from(&input_types[0])?;
                    Ok(vec![ReferenceType::new(Self::view(*symbol).output_type(reference.referent())?).into()])
                }
                Self::UndescribedScan(operation) => operation.infer_output_types(input_types, region_interfaces),
            }
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::Native(operation) => operation.input_region_provenance(region_index, input_index),
                Self::Symbolic(_) => None,
                Self::UndescribedScan(operation) => operation.input_region_provenance(region_index, input_index),
            }
        }

        fn region_input_view_source(&self, region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::Native(operation) => operation.region_input_view_source(region_index, input_index),
                Self::Symbolic(_) => None,
                Self::UndescribedScan(operation) => operation.region_input_view_source(region_index, input_index),
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::Native(operation) => operation.output_region_provenance(output_index),
                Self::Symbolic(_) => Vec::new(),
                Self::UndescribedScan(operation) => operation.output_region_provenance(output_index),
            }
        }

        fn reference_output_identity_input(&self, output_index: usize) -> Option<usize> {
            match self {
                Self::Native(operation) => operation.reference_output_identity_input(output_index),
                Self::Symbolic(_) => None,
                Self::UndescribedScan(operation) => operation.reference_output_identity_input(output_index),
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            match self {
                Self::Native(operation) => operation.reference_semantics(),
                Self::Symbolic(_) => Cow::Owned(ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceOutput::Alias { output_index: 0, input_index: 0, kind: ReferenceAliasKind::View }],
                )),
                Self::UndescribedScan(operation) => operation.reference_semantics(),
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Native(operation) => operation.effects(),
                Self::Symbolic(_) => Effects::PURE,
                Self::UndescribedScan(operation) => operation.effects(),
            }
        }
    }

    impl From<ReferenceIndexOperation> for SymbolicViewOperation {
        fn from(operation: ReferenceIndexOperation) -> Self {
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
                Self::Symbolic(_) | Self::UndescribedScan(_) => None,
            }
        }

        // The undescribed scan deliberately keeps the default, which describes no boundary view.

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

    /// Builds `f(vector: ref<f32[2]>, coordinate: f32) = read(symbolic_view(vector, coordinate))`, whose view
    /// describes its coordinate through `symbol`.
    fn symbolic_view_program(
        symbol: ViewSymbol,
    ) -> Program<TestValue, SymbolicViewOperation, Vec<TestValue>, Vec<TestValue>> {
        let mut builder = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let coordinate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let view = builder
            .add_instruction(SymbolicViewOperation::Symbolic(symbol), Vec::new(), vec![vector, coordinate], None)
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
        let analysis = ReferenceAnalysisError::ReferenceConstant { region: RegionId::new(0), atom: AtomId::new(1) };
        assert_eq!(
            ReferenceViewAnalysisError::from(analysis.clone()),
            ReferenceViewAnalysisError::Analysis(analysis.clone())
        );
        assert_eq!(
            ReferenceViewAnalysisError::Analysis(analysis).to_string(),
            "region ^0 stores reference-typed constant %1 that names no capture; references enter a program only \
             through inputs and captures",
        );
        assert_eq!(
            ReferenceViewAnalysisError::MissingView { operation: "view", instruction: id(0, 2) }.to_string(),
            "operation `view` at ^0[2] derives a reference view but exposes no view transform",
        );
        assert_eq!(
            ReferenceViewAnalysisError::ViewTypeMismatch {
                operation: "reference_index",
                instruction: id(0, 2),
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }
            .to_string(),
            "operation `reference_index` at ^0[2] declares view referent type `f32[2]` but its transform derives \
             referent type `f32[3]` from the source view",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidViewComposition {
                operation: "reference_index",
                instruction: id(0, 2),
                message: "reference index axis 2 is out of bounds for rank 2".to_string(),
            }
            .to_string(),
            "operation `reference_index` at ^0[2] composes an invalid view transform: reference index axis 2 is out of \
             bounds for rank 2",
        );
        assert_eq!(
            ReferenceViewAnalysisError::InvalidViewSymbolOperand {
                operation: "symbolic_view",
                instruction: id(0, 2),
                operand_index: 3,
                message: "the instruction has only 2 operands".to_string(),
            }
            .to_string(),
            "operation `symbolic_view` at ^0[2] describes a view through operand 3, but the instruction has only 2 \
             operands",
        );
        assert_eq!(
            ReferenceViewAnalysisError::IterationSymbolAtOutput {
                operation: "symbolic_view",
                instruction: id(0, 2),
                output_index: 0,
            }
            .to_string(),
            "operation `symbolic_view` at ^0[2] describes output 0 through the iteration counter, but an iteration \
             symbol only describes region inputs",
        );
        assert_eq!(
            ReferenceViewAnalysisError::MissingBoundaryView {
                operation: "scan",
                instruction: id(1, 0),
                region_index: 0,
                input_index: 1,
            }
            .to_string(),
            "operation `scan` at ^1[0] creates a boundary view for input 1 of region 0 but exposes no view transform",
        );
        assert_eq!(
            ProgramError::from(ReferenceViewAnalysisError::MissingView { operation: "view", instruction: id(0, 2) }),
            ProgramError::MalformedProgram(
                "operation `view` at ^0[2] derives a reference view but exposes no view transform".to_string(),
            ),
        );
    }

    #[test]
    fn test_reference_view_path() {
        let root = TestPath::root();
        assert!(root.is_root());
        assert_eq!(root.steps(), &[]);
        assert_eq!(root.views().count(), 0);
        assert_eq!(root, TestPath::default());

        // Appending keeps root-to-value order and leaves the original path untouched. A static step carries no
        // bindings, and a symbolic step carries one binding per symbol of its description.
        let row = root.with_view(index(0, 1));
        let element = row.with_view(index(0, 2));
        assert!(!row.is_root());
        assert_eq!(row.views().collect::<Vec<_>>(), vec![&index(0, 1)]);
        assert_eq!(element.views().collect::<Vec<_>>(), vec![&index(0, 1), &index(0, 2)]);
        assert_eq!(element.steps().len(), 2);
        assert_eq!(element.steps()[1].view(), &index(0, 2));
        assert_eq!(element.steps()[1].bindings(), &[]);
        assert!(root.is_root());
        let symbolic =
            ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Symbolic(ViewSymbol::Operand(1)) };
        let bound = row.with_step(symbolic.clone(), vec![ViewSymbolBinding::Value(value(0, 3))]);
        assert_eq!(bound.views().collect::<Vec<_>>(), vec![&index(0, 1), &symbolic]);
        assert_eq!(bound.steps()[1].bindings(), &[ViewSymbolBinding::Value(value(0, 3))]);
        assert_ne!(bound, row.with_step(symbolic.clone(), vec![ViewSymbolBinding::Value(value(0, 4))]));
        assert_ne!(bound, row.with_step(symbolic, vec![ViewSymbolBinding::Iteration(RegionId::new(1))]));

        // Equality and rendering distinguish description sequences, not the values they were derived for.
        assert_eq!(row, TestPath::root().with_view(index(0, 1)));
        assert_ne!(row, element);
        assert_ne!(row, TestPath::root().with_view(index(1, 1)));
        assert_eq!(format!("{root:?}"), "ReferenceViewPath { steps: [] }");
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
        assert_eq!(row_0.overlap(&row_1, &root), ViewOverlap::Disjoint);
        assert_eq!(row_0.overlap(&TestPath::root().with_view(index(0, 0)), &root), ViewOverlap::Same);
        assert_eq!(TestPath::root().overlap(&row_0, &root), ViewOverlap::MayOverlap);
        assert_eq!(TestPath::root().overlap(&TestPath::root(), &root), ViewOverlap::Same);

        // A boundary view indexed by one region's iteration counter is the same as itself, may overlap with the
        // complete root, with any static row, and with the view of another region's iteration counter.
        let iteration = |region: usize| {
            TestPath::root().with_step(iteration_view(), vec![ViewSymbolBinding::Iteration(RegionId::new(region))])
        };
        assert_eq!(iteration(0).overlap(&iteration(0), &root), ViewOverlap::Same);
        assert_eq!(iteration(0).overlap(&TestPath::root(), &root), ViewOverlap::MayOverlap);
        assert_eq!(iteration(0).overlap(&row_1, &root), ViewOverlap::MayOverlap);
        assert_eq!(iteration(0).overlap(&iteration(1), &root), ViewOverlap::MayOverlap);
    }

    #[test]
    fn test_batch_reference_view_operation() {
        let extent = TestValue::Dimension(
            DimensionValue::new(DimensionType::new(DimensionVariable::new("batch", DimensionBounds::unbounded())), 2)
                .unwrap(),
        );
        let context =
            BatchingContext::<_, ArrayIrBatching>::new(EagerContext::<TestValue, TestOperation>::new(), extent);
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
        // output are rejected by name, and so is a mapped operand other than the viewed source.
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
                message: "`reference_index` requires operand 1 to be replicated; batching a reference view through a \
                          mapped coordinate operand is not supported"
                    .to_string(),
            }),
        );
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

        // Both alias edges record the output that defines the aliasing value, which is what the overlay asked the
        // producing operation to describe.
        assert_eq!(
            analysis.analysis().alias(value(0, 1)),
            Some(ReferenceAliasEdge::new(
                id(0, 0),
                ReferenceAliasOrigin::Output(0),
                value(0, 0),
                ReferenceAliasKind::View,
                true,
            )),
        );
        assert_eq!(
            analysis.analysis().alias(value(0, 2)),
            Some(ReferenceAliasEdge::new(
                id(0, 1),
                ReferenceAliasOrigin::Output(0),
                value(0, 1),
                ReferenceAliasKind::View,
                true,
            )),
        );
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
                ReferenceAliasOrigin::Output(1),
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
            Some(ReferenceViewAnalysisError::ViewTypeMismatch {
                operation: REFERENCE_INDEX_OPERATION_NAME,
                instruction: id(0, 0),
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_missing_views() {
        /// Array-IR family extended with one operation that declares a view alias but describes no view for it.
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

            fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
                match self {
                    Self::Native(operation) => operation.reference_semantics(),
                    Self::View => Cow::Owned(ReferenceOperationSemantics::new(
                        Vec::new(),
                        vec![ReferenceOutput::Alias {
                            output_index: 0,
                            input_index: 0,
                            kind: ReferenceAliasKind::View,
                        }],
                    )),
                }
            }

            fn effects(&self) -> Effects {
                match self {
                    Self::Native(operation) => operation.effects(),
                    Self::View => Effects::PURE,
                }
            }
        }

        impl From<ReferenceIndexOperation> for UndescribedViewOperation {
            fn from(operation: ReferenceIndexOperation) -> Self {
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
            Some(ReferenceViewAnalysisError::MissingView { operation: "undescribed_view", instruction: id(0, 0) }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_binds_operand_symbols() {
        // The overlay closes the description over the describing instruction: its operand symbol binds to the
        // coordinate input's identity, and the static read output has no path.
        let program = symbolic_view_program(ViewSymbol::Operand(1));
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let view = SymbolicViewOperation::view(ViewSymbol::Operand(1));
        assert_eq!(analysis.path(value(0, 0)), Some(&TestPath::root()));
        assert_eq!(
            analysis.path(value(0, 2)),
            Some(&TestPath::root().with_step(view.clone(), vec![ViewSymbolBinding::Value(value(0, 1))])),
        );
        assert_eq!(analysis.path(value(0, 3)), None);
        assert_eq!(analysis.path(value(0, 2)).map(|path| path.views().collect::<Vec<_>>()), Some(vec![&view]));
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_invalid_symbols() {
        // An operand symbol must name an operand of the describing instruction.
        let program = symbolic_view_program(ViewSymbol::Operand(2));
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidViewSymbolOperand {
                operation: "symbolic_view",
                instruction: id(0, 0),
                operand_index: 2,
                message: "the instruction has only 2 operands".to_string(),
            }),
        );

        // The named operand must be a coordinate value, not the viewed reference itself.
        let program = symbolic_view_program(ViewSymbol::Operand(0));
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::InvalidViewSymbolOperand {
                operation: "symbolic_view",
                instruction: id(0, 0),
                operand_index: 0,
                message: "that operand is a reference rather than a coordinate value".to_string(),
            }),
        );

        // An iteration symbol only describes region inputs, never an instruction output.
        let program = symbolic_view_program(ViewSymbol::Iteration);
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::IterationSymbolAtOutput {
                operation: "symbolic_view",
                instruction: id(0, 0),
                output_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_derives_boundary_views() {
        // The stacked reference operand enters the body as the view the scan creates at the boundary: the body input's
        // path is that single step, closed over the body region rather than over the attaching instruction, while the
        // operand itself stays a root of the entry region.
        let program = stacked_scan_program(reading_scan_body());
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let boundary =
            TestPath::root().with_step(iteration_view(), vec![ViewSymbolBinding::Iteration(RegionId::new(0))]);
        assert_eq!(
            analysis.analysis().alias(value(0, 1)),
            Some(ReferenceAliasEdge::new(
                id(1, 0),
                ReferenceAliasOrigin::RegionInput { region_index: 0, input_index: 1 },
                value(1, 1),
                ReferenceAliasKind::View,
                true,
            )),
        );
        assert_eq!(
            analysis.paths().collect::<Vec<_>>(),
            vec![(value(0, 1), &boundary), (value(1, 1), &TestPath::root())],
        );
        assert_eq!(
            analysis.path(value(0, 1)).map(|path| path.views().collect::<Vec<_>>()),
            Some(vec![&iteration_view()]),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_validates_boundary_views() {
        // The boundary view is validated against the operand type and the body input's own type, read from the body
        // region: indexing axis 0 of `f32[3, 2]` derives `f32[2]`, but this body declares its view input as `f32[3]`.
        let mut body = TestBuilder::new();
        let carry = body.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let element = body.add_input(reference_type([3]));
        body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap();
        let body = body.build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder]);
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.unwrap().entry_region_ref());
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let stack = builder.add_input(reference_type([3, 2]));
        let output = builder.add_variable(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        builder.add_instruction_unchecked(Instruction::new(
            ArrayIrOperation::Scan(ScanOperation::new(1, 3)),
            vec![carry, stack],
            vec![output],
            vec![body],
        ));
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::ViewTypeMismatch {
                operation: SCAN_OPERATION_NAME,
                instruction: id(1, 0),
                expected: "f32[2]".to_string(),
                actual: "f32[3]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_new_derives_shared_boundary_views_once() {
        // Two scans attaching one body derive one path for its view input: the edge names the first attachment, but
        // the path binds the iteration counter to the body region, so it does not depend on which scan attached it.
        let body = reading_scan_body();
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.entry_region_ref());
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let stack = builder.add_input(reference_type([3, 2]));
        let first = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 3), vec![body], vec![carry, stack], None)
            .unwrap()[0];
        let second = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 3), vec![body], vec![first, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let analysis = ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(
            analysis
                .analysis()
                .region_input_bindings()
                .iter()
                .map(|binding| binding.instruction())
                .collect::<Vec<_>>(),
            vec![id(1, 0), id(1, 1)],
        );
        assert_eq!(analysis.analysis().alias(value(0, 1)).map(|edge| edge.instruction()), Some(id(1, 0)));
        assert_eq!(
            analysis.paths().collect::<Vec<_>>(),
            vec![
                (
                    value(0, 1),
                    &TestPath::root().with_step(iteration_view(), vec![ViewSymbolBinding::Iteration(RegionId::new(0))]),
                ),
                (value(1, 1), &TestPath::root()),
            ],
        );
    }

    #[test]
    fn test_reference_view_analysis_new_rejects_undescribed_boundary_views() {
        // An operation that declares a region input as a boundary view must describe it.
        let mut body = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let carry = body.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let element = body.add_input(reference_type([2]));
        body.add_instruction(
            SymbolicViewOperation::Native(ReferenceReadOperation::new().into()),
            Vec::new(),
            vec![element],
            None,
        )
        .unwrap();
        let body = body.build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder]);
        let mut builder = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let body = builder.import_region(body.unwrap().entry_region_ref());
        let carry = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let stack = builder.add_input(reference_type([3, 2]));
        let output = builder
            .add_instruction(
                SymbolicViewOperation::UndescribedScan(ScanOperation::new(1, 3)),
                vec![body],
                vec![carry, stack],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ReferenceViewAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ReferenceViewAnalysisError::MissingBoundaryView {
                operation: "undescribed_scan",
                instruction: id(1, 0),
                region_index: 0,
                input_index: 1,
            }),
        );
    }

    #[test]
    fn test_reference_view_analysis_overlap_across_boundary_views() {
        // Roots are region-relative: inside the body, the carry and the boundary view are two different roots even
        // though the caller binds both to the same stacked reference, so the overlay reports them, and any views of
        // them, as disjoint. Relating them through the caller root is the caller's job (the scan discharge rule), which
        // compares the paths directly under the caller root's type and finds that a static row may overlap with the
        // per-iteration row.
        let mut body = TestBuilder::new();
        let carry = body.add_input(reference_type([3, 2]));
        let element = body.add_input(reference_type([2]));
        let row = body.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![carry], None).unwrap()[0];
        body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![row], None).unwrap();
        body.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap();
        let body = body.build::<Vec<TestValue>, Vec<TestValue>>(vec![carry], vec![Placeholder; 2], vec![Placeholder]);
        let mut builder = TestBuilder::new();
        let body = builder.import_region(body.unwrap().entry_region_ref());
        let stack = builder.add_input(reference_type([3, 2]));
        let output = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 3), vec![body], vec![stack, stack], None)
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        let caller_root = ReferenceRoot::RegionInput { region: RegionId::new(1), input_index: 0 };
        assert_eq!(
            analysis.analysis().region_input_bindings().iter().map(|binding| binding.root()).collect::<Vec<_>>(),
            vec![caller_root, caller_root],
        );
        assert_eq!(analysis.path(value(0, 2)), Some(&TestPath::root().with_view(index(0, 1))));
        assert_eq!(analysis.overlap(region, value(0, 1), value(0, 1)), Some(ViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 1)), Some(ViewOverlap::Disjoint));
        assert_eq!(analysis.overlap(region, value(0, 1), value(0, 2)), Some(ViewOverlap::Disjoint));
        let (boundary, row) = (analysis.path(value(0, 1)).unwrap(), analysis.path(value(0, 2)).unwrap());
        assert_eq!(boundary.overlap(row, &reference_type([3, 2])), ViewOverlap::MayOverlap);
        assert_eq!(boundary.overlap(&TestPath::root(), &reference_type([3, 2])), ViewOverlap::MayOverlap);
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
        // Within one root both paths fold to root coordinates: the row slice may overlap with the complete root, and
        // indexing the single row of that slice selects exactly the slice's coordinates again. Values that are not
        // references, or that live in different regions, have no answer.
        let program = chain_program();
        let region = program.entry_region_ref();
        let analysis = ReferenceViewAnalysis::new(region, 0).unwrap();
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 0)), Some(ViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 1)), Some(ViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 1), value(0, 2)), Some(ViewOverlap::Same));
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
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 1)), Some(ViewOverlap::Disjoint));
        assert_eq!(analysis.overlap(region, value(0, 2), value(0, 3)), Some(ViewOverlap::Disjoint));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 2)), Some(ViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 2), value(0, 2)), Some(ViewOverlap::Same));

        // Symbolic coordinates compare by their bindings: two views through the same coordinate operand select the
        // same slot, while views through different operands, or against a static coordinate or the root, may overlap.
        let mut builder = ProgramBuilder::<TestValue, SymbolicViewOperation>::new();
        let vector = builder.add_input(reference_type([2]));
        let coordinate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let other = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::F32)));
        let symbolic = SymbolicViewOperation::Symbolic(ViewSymbol::Operand(1));
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, coordinate], None).unwrap();
        builder.add_instruction(symbolic.clone(), Vec::new(), vec![vector, coordinate], None).unwrap();
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
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 4)), Some(ViewOverlap::Same));
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 5)), Some(ViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 3), value(0, 6)), Some(ViewOverlap::MayOverlap));
        assert_eq!(analysis.overlap(region, value(0, 0), value(0, 3)), Some(ViewOverlap::MayOverlap));
    }

    #[test]
    fn test_region_ref_reference_view_analysis() {
        let program = chain_program();
        let retained = program.entry_region_ref().reference_view_analysis(0).unwrap();
        assert_eq!(*retained, ReferenceViewAnalysis::new(program.entry_region_ref(), 0).unwrap());

        // A second request under the same capture scope is served the retained overlay.
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

        // Program `first` attaches the row-0 branch as both branches of a condition, so the overlay of its entry `^1`
        // records the row-0 view for the branch region `^0`.
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
        // nested views select, so it must not be served the overlay derived for the row-0 branch.
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

        // The source program keeps its own retained overlay, because only the re-sealed copy was rebased.
        assert!(Arc::ptr_eq(&first.entry_region_ref().reference_view_analysis(0).unwrap(), &retained));
    }
}
