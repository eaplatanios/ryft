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
//! # Compatibility Is Geometry, Not Root Type
//!
//! A description is validated against the *current* source reference type before it is reapplied. A tangent or
//! cotangent root may have a different referent type from the primal root (e.g., a widened floating-point tangent
//! type), so a description that was valid on the primal root is re-checked against the transformed root rather than
//! assumed to transfer.
//!
//! # Reapplication Is Not Batching
//!
//! [`ReferenceViewOperation::reapply_view`] exists only for tangent, cotangent, and residual reference reconstruction,
//! where the transformed root has the same geometry as the primal root. Batching inserts an axis into the root, and a
//! primal description reapplied unchanged to a batched root would index or slice the wrong axis, so each view
//! operation's own batching rule accounts for the inserted batch axis instead of going through this contract.
//!
//! # Overlap Queries
//!
//! Whether two paths select overlapping coordinates has no consumer in the current transform phases and is not part of
//! this contract; it is deferred until a consumer exists.

// TODO(eaplatanios): Review this module.

use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

use thiserror::Error;

use crate::contexts::Context;
use crate::parameters::Parameter;
use crate::programs::ProgramError;
use crate::programs::instructions::InstructionId;
use crate::programs::operations::Operation;
use crate::programs::references::analysis::{
    ReferenceAnalysis, ReferenceAnalysisError, ReferenceAnalysisTransformArguments,
};
use crate::programs::references::semantics::ReferenceAliasKind;
use crate::programs::regions::{Region, RegionRef};
use crate::programs::transforms::{Transform, TransformArtifact};
use crate::programs::types::Typed;
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
}

impl From<ReferenceViewAnalysisError> for ProgramError {
    #[inline]
    fn from(error: ReferenceViewAnalysisError) -> Self {
        ProgramError::MalformedProgram(error.to_string())
    }
}

/// Static view contract of one operation family: the owned description of every view alias the family can derive,
/// its type-level validation, and its reapplication to another reference of compatible geometry.
///
/// An operation family implements this trait once, and every transform that rebuilds references (tangent, cotangent,
/// and residual reconstruction) then reaches the family's views through it: [`ReferenceViewAnalysis`] composes the
/// descriptions into per-value [`ReferenceViewPath`]s, and reconstruction reapplies them step by step to the
/// transformed root. Compatibility is geometry, not root type: a tangent or cotangent root may have a different
/// referent type from the primal root, so each description is validated against the current transformed source type
/// before it is reapplied. Reapplication is not batching, because a primal description reapplied unchanged to a batched
/// root indexes or slices the wrong axis; each view operation's own batching rule accounts for the inserted batch axis.
pub trait ReferenceViewOperation: Operation {
    /// Owned description of one view step, from a source reference to the reference it derives. The bounds are what
    /// the retained [`ReferenceViewAnalysis`] needs to live in a region's transform cache and to be revalidated
    /// against a fresh derivation.
    type View: 'static + Clone + Debug + PartialEq + Eq + Send + Sync;

    /// Returns the description of the view this operation derives at output `output_index`, or [`None`] when that
    /// output is not a view. Exactly the outputs whose [`reference_semantics`](Operation::reference_semantics) declare
    /// a [`ReferenceOutput::Alias`](crate::programs::references::ReferenceOutput::Alias) of kind
    /// [`ReferenceAliasKind::View`] return [`Some`]; an operation that declares such an alias but returns [`None`] is
    /// rejected by the overlay with [`ReferenceViewAnalysisError::MissingView`].
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
    /// compatible geometry; callers validate the step with [`validate_view`](Self::validate_view) against the current
    /// source type first.
    ///
    /// # Errors
    ///
    /// Propagates the staging error of the context.
    fn reapply_view<C: Context<Type = Self::Type, Operation = Self>>(
        context: &C,
        view: &Self::View,
        source: C::Value,
    ) -> Result<C::Value, ProgramError>;
}

/// Ordered view descriptions from a reference root to one derived reference-typed value.
///
/// The path stores only the descriptions, in root-to-value order; the root itself is a property of the structural
/// [`ReferenceAnalysis`]. The empty path is the identity and denotes the complete root, so root handles, region inputs,
/// capture constants, and forwarded region outputs all carry it. Equality and hashing distinguish different
/// description sequences, not the values they were derived for.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReferenceViewPath<View> {
    /// Descriptions applied from the root outward.
    views: Vec<View>,
}

impl<View> ReferenceViewPath<View> {
    /// Returns the empty path denoting the complete root.
    #[inline]
    pub const fn root() -> Self {
        Self { views: Vec::new() }
    }

    /// Returns the ordered descriptions applied from the root outward.
    #[inline]
    pub fn views(&self) -> &[View] {
        self.views.as_slice()
    }

    /// Returns whether this path denotes the complete root.
    #[inline]
    pub fn is_root(&self) -> bool {
        self.views.is_empty()
    }

    /// Returns this path extended by one more description applied to its current end.
    pub fn with_view(&self, view: View) -> Self
    where
        View: Clone,
    {
        let mut views = Vec::with_capacity(self.views.len() + 1);
        views.extend(self.views.iter().cloned());
        views.push(view);
        Self { views }
    }
}

impl<View> Default for ReferenceViewPath<View> {
    #[inline]
    fn default() -> Self {
        Self::root()
    }
}

impl<View: Parameter> Parameter for ReferenceViewPath<View> {}

/// Structural [`ReferenceAnalysis`] of a [`Region`] closure together with the [`ReferenceViewPath`] of every
/// reference-typed value in that closure, derived through the [`ReferenceViewOperation`] contract of the closure's
/// operation family.
///
/// Every reference-typed value has exactly one path. A root handle (a region input, an allocation, a capture constant,
/// or a forwarded region output) has the empty path, an identity alias copies the path of its source, and a view alias
/// copies the path of its source and appends the description its producing operation reports for that edge's output,
/// after that description was validated against the source and output reference types. Nested region inputs are
/// separate roots of the structural analysis and therefore carry the empty path, consistent with the root-only region
/// boundary rule: the region that needs a view recreates it from the carried root.
///
/// The overlay is retained in the region's transform cache under exactly the cache identity of the structural analysis
/// (refer to the documentation of [`RegionRef::reference_view_analysis`]) and shares that analysis through an [`Arc`]
/// rather than re-deriving it. Like the structural analysis, it is kernel-owned validation infrastructure that consumers
/// invoke explicitly on the programs they own.
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
        View: Clone,
    {
        let analysis = region.reference_analysis(capture_count)?;
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
        View: Clone,
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
                        // instruction of the region containing `value`, so neither lookup can fail here.
                        let id = edge.instruction();
                        let current = region.with_id(id.region()).unwrap();
                        let operation = current.instructions()[id.index()].operation();
                        let name = operation.name();
                        let Some(view) = operation.reference_view(edge.output_index()) else {
                            return Err(ReferenceViewAnalysisError::MissingView { operation: name, instruction: id });
                        };
                        let atoms = current.atoms();
                        let source_type = atoms[edge.source().atom().index()].r#type();
                        let output_type = atoms[value.atom().index()].r#type();
                        O::validate_view(&view, source_type.as_ref(), output_type.as_ref()).map_err(|error| {
                            match error {
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
                            }
                        })?;
                        source.with_view(view)
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
    /// or [`None`] when `value` is not a reference-typed value of the closure. Root handles and nested region inputs
    /// carry the empty path.
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
                let analysis = ReferenceViewAnalysis::new(region, arguments.capture_count())?;
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
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{
        ArrayIrOperation, ArrayReferenceViewOperation, REFERENCE_INDEX_OPERATION_NAME, ReferenceIndexOperation,
        ReferenceSliceOperation, reapply_array_reference_view,
    };
    use crate::arrays::reference_views::ArrayReferenceViewTransform;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::ir::ArrayIrType;
    use crate::operations::{
        ConditionOperation, ReshapeOperation, SliceOperation, UpdateSliceOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::atoms::AtomId;
    use crate::programs::builders::ProgramBuilder;
    use crate::programs::effects::Effects;
    use crate::programs::instructions::Instruction;
    use crate::programs::programs::Program;
    use crate::programs::references::analysis::ReferenceAliasEdge;
    use crate::programs::references::operations::{ReferenceReadOperation, ReferenceWriteOperation};
    use crate::programs::references::semantics::{ReferenceOperationSemantics, ReferenceOutput};
    use crate::programs::references::types::ReferenceType;
    use crate::programs::regions::{RegionId, RegionInterface};
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
        ArrayReferenceViewTransform::Index { axis, index }
    }

    /// Builds `f(matrix: ref<f32[2, 3]>) = read(matrix[0:1, 0:3][0])`, a two-step view chain over one root.
    fn chain_program() -> TestProgram {
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(reference_type([2, 3]));
        let axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let row = builder.add_instruction(ReferenceSliceOperation::new(axes), Vec::new(), vec![matrix], None).unwrap()[0];
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![row], None).unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        builder.build(vec![snapshot], vec![Placeholder], vec![Placeholder]).unwrap()
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
        assert_eq!(root.views(), &[]);
        assert_eq!(root, TestPath::default());

        // Appending keeps root-to-value order and leaves the original path untouched.
        let row = root.with_view(index(0, 1));
        let element = row.with_view(index(0, 2));
        assert!(!row.is_root());
        assert_eq!(row.views(), &[index(0, 1)]);
        assert_eq!(element.views(), &[index(0, 1), index(0, 2)]);
        assert!(root.is_root());

        // Equality and rendering distinguish description sequences, not the values they were derived for.
        assert_eq!(row, TestPath::root().with_view(index(0, 1)));
        assert_ne!(row, element);
        assert_ne!(row, TestPath::root().with_view(index(1, 1)));
        assert_eq!(format!("{root:?}"), "ReferenceViewPath { views: [] }");
        assert_eq!(format!("{row:?}"), "ReferenceViewPath { views: [Index { axis: 0, index: 1 }] }");
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
            Some(ReferenceAliasEdge::new(id(0, 0), 0, value(0, 0), ReferenceAliasKind::View, true)),
        );
        assert_eq!(
            analysis.analysis().alias(value(0, 2)),
            Some(ReferenceAliasEdge::new(id(0, 1), 0, value(0, 1), ReferenceAliasKind::View, true)),
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
            Some(ReferenceAliasEdge::new(id(2, 0), 1, value(2, 1), ReferenceAliasKind::Identity, false)),
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
            ) -> Result<C::Value, ProgramError> {
                reapply_array_reference_view(context, view, source)
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
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![row_0, row_0],
                vec![predicate, matrix],
                None,
            )
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
