//! Array view overlay over the generic program-level reference analysis.
//!
//! [`ArrayReferenceAnalysis`] is the array instantiation of the value-family-generic
//! [`ReferenceViewAnalysis`]: it runs the generic
//! [`ReferenceAnalysis`](crate::ReferenceAnalysis) over a [`Region`](crate::Region) closure whose values are typed by
//! [`ArrayIrType`](crate::ArrayIrType) and derives, exactly once per reference-typed value, the [`ArrayReferenceView`]
//! that maps the value's canonical root to the coordinates the value selects. The generic analysis owns roots, alias
//! edges, accesses, capture scopes, nested-region bindings, and lifetime rules; the generic overlay adds the view
//! geometry, recovered from the alias edges through the operation family's
//! [`ReferenceViewOperation`](crate::ReferenceViewOperation) contract rather than by re-walking index and slice
//! operations by name; and this module only names the array specialization and its view-flavored accessors.
//!
//! The resulting view table is the one authoritative source of view geometry for every consumer of a validated
//! program: kernel-boundary validation, diagnostics, and lowering all read the same table instead of re-deriving views
//! independently. Like the generic analysis, it is *kernel-owned validation infrastructure* that consumers invoke
//! explicitly on the programs they own, not a standing lint that every program pays for: ordinary programs remain
//! validated by the construction-time alias tracking of [`ProgramBuilder`](crate::ProgramBuilder), by the eager
//! [`Reference`](crate::Reference) runtime, and by reference discharge.
//!
//! # View Derivation
//!
//! Every root handle (a region input, an allocation, a capture constant, or a provenance-forwarded region output) maps
//! to [`ArrayReferenceView::root`]. An identity edge copies the view of its source. A view edge composes the aliasing
//! operation's [`ArrayReferenceViewTransform`] onto the view of its source after
//! validating it with [`validate_array_reference_view`](crate::validate_array_reference_view), rejecting an operation
//! that declares a view alias but describes none, a transform that is invalid for the source referent, and a declared
//! output referent that differs from the derived one. Nested region inputs are separate roots of the generic analysis:
//! an input forwarded as a complete handle maps to [`ArrayReferenceView::root`], while an input the attaching
//! operation creates as a boundary view (e.g., the per-iteration slice of a stacked `scan` reference operand) maps to
//! that operation's single [`ArrayReferenceViewTransform`] closed over the attached region, whose iteration symbol is
//! bound to that region. Both views are relative to the nested input itself rather than to the caller root it is bound
//! to.

// TODO(eaplatanios): Review this module.

use crate::arrays::reference_views::{ArrayReferenceView, ArrayReferenceViewTransform};
use crate::programs::{ReferenceViewAnalysis, ReferenceViewAnalysisError, ValueId};

/// Error produced by [`ArrayReferenceAnalysis`] when the generic reference analysis fails or when a derived view
/// cannot be reconciled with the program's declared reference types. This is the generic
/// [`ReferenceViewAnalysisError`], whose view variants carry the array transform diagnostics produced by
/// [`validate_array_reference_view`](crate::validate_array_reference_view).
pub type ArrayReferenceAnalysisError = ReferenceViewAnalysisError;

/// Generic [`ReferenceAnalysis`](crate::ReferenceAnalysis) of an array-typed [`Region`](crate::Region) closure
/// together with the [`ArrayReferenceView`] of every reference-typed value in that closure: the array instantiation of
/// [`ReferenceViewAnalysis`]. This is the one authoritative view table that consumers of a validated program
/// (kernel-boundary validation, diagnostics, and lowering) read instead of re-walking index and slice operations, and
/// like the generic analysis it is kernel-owned validation infrastructure that consumers invoke explicitly rather than
/// a standing whole-program lint.
///
/// Every reference-typed value of the closure has exactly one view. Root handles and forwarded nested region inputs
/// map to [`ArrayReferenceView::root`], a nested region input created as a boundary view by its attaching operation
/// carries that operation's transform closed over the attached region, identity aliases copy the view of their
/// source, and view aliases compose the aliasing operation's [`ArrayReferenceViewTransform`] onto the view of their
/// source. The view of a value composed with the referent type of its root therefore reproduces exactly the referent
/// type the program declares for that value, which the analysis verifies while deriving the table. Construct it with
/// [`ReferenceViewAnalysis::new`] (uncached) or through
/// [`RegionRef::reference_view_analysis`](crate::RegionRef::reference_view_analysis) (retained), and read it through
/// [`view`](Self::view) and [`views`](Self::views) or the generic [`path`](ReferenceViewAnalysis::path) and
/// [`paths`](ReferenceViewAnalysis::paths).
pub type ArrayReferenceAnalysis = ReferenceViewAnalysis<ArrayReferenceViewTransform>;

impl ArrayReferenceAnalysis {
    /// Returns the [`ArrayReferenceView`] mapping the root of the reference-typed `value` to the coordinates it
    /// selects, or [`None`] when `value` is not a reference-typed value of the closure. Root handles and forwarded
    /// nested region inputs map to [`ArrayReferenceView::root`]; a nested region input created as a boundary view by
    /// its attaching operation carries that operation's transform closed over the attached region.
    #[inline]
    pub fn view(&self, value: ValueId) -> Option<&ArrayReferenceView> {
        self.path(value)
    }

    /// Returns the [`ArrayReferenceView`] of every reference-typed value of the closure, in canonical [`ValueId`]
    /// order.
    #[inline]
    pub fn views(&self) -> impl Iterator<Item = (ValueId, &ArrayReferenceView)> + '_ {
        self.paths()
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
    use crate::arrays::reference_views::ViewIndex;
    use crate::arrays::types::arrays::ArrayType;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::ir::ArrayIrType;
    use crate::contexts::Context;
    use crate::operations::{
        ConditionOperation, ReshapeOperation, SliceOperation, UpdateSliceOperation, WhileOperation,
    };
    use crate::parameters::Placeholder;
    use crate::programs::{
        AtomId, EffectClasses, Effects, Instruction, InstructionId, Operation, ProgramBuilder, ProgramError,
        ReferenceAccessMode, ReferenceAlias, ReferenceAliasEdge, ReferenceAliasKind, ReferenceAliasOrigin,
        ReferenceAnalysisError, ReferenceFreezeOperation, ReferenceReadOperation, ReferenceRoot, ReferenceSource,
        ReferenceType, ReferenceViewOperation, ReferenceViewValidationError, ReferenceWriteOperation, RegionId,
        RegionInterface, TypeError,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;

    type TestOperation = ArrayIrOperation<Array>;

    type TestBuilder = ProgramBuilder<TestValue, TestOperation>;

    fn id(region: usize, index: usize) -> InstructionId {
        InstructionId::new(RegionId::new(region), index)
    }

    fn value(region: usize, atom: usize) -> ValueId {
        ValueId::new(RegionId::new(region), AtomId::new(atom))
    }

    fn reference_type(dimensions: impl Into<Vec<usize>>) -> ArrayIrType {
        ArrayIrType::Reference(ReferenceType::new(ArrayType::new_static(DataType::F32, dimensions)))
    }

    #[test]
    fn test_array_reference_analysis_error() {
        let analysis = ReferenceAnalysisError::ReferenceConstant { region: RegionId::new(0), atom: AtomId::new(1) };
        assert_eq!(
            ArrayReferenceAnalysisError::Analysis(analysis.clone()).to_string(),
            "region ^0 stores reference-typed constant %1 that names no capture; references enter a program only \
             through inputs and captures",
        );
        assert_eq!(
            ArrayReferenceAnalysisError::from(analysis.clone()),
            ArrayReferenceAnalysisError::Analysis(analysis)
        );
        assert_eq!(
            ArrayReferenceAnalysisError::MissingView { operation: "view", instruction: id(0, 2) }.to_string(),
            "operation `view` at ^0[2] derives a reference view but exposes no view transform",
        );
        assert_eq!(
            ArrayReferenceAnalysisError::ViewTypeMismatch {
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
            ArrayReferenceAnalysisError::InvalidViewComposition {
                operation: "reference_index",
                instruction: id(0, 2),
                message: "reference index axis 2 is out of bounds for rank 2".to_string(),
            }
            .to_string(),
            "operation `reference_index` at ^0[2] composes an invalid view transform: reference index axis 2 is out of \
             bounds for rank 2",
        );
        assert_eq!(
            ProgramError::from(ArrayReferenceAnalysisError::MissingView { operation: "view", instruction: id(0, 2) }),
            ProgramError::MalformedProgram(
                "operation `view` at ^0[2] derives a reference view but exposes no view transform".to_string(),
            ),
        );
    }

    #[test]
    fn test_array_reference_analysis_new() {
        // A root matrix reference is narrowed to a row slice, then to one element of that row, while an overlapping
        // sibling view selects one column directly from the root; both leaves are read.
        let matrix_type = ArrayType::new_static(DataType::F32, [2, 3]);
        let mut builder = TestBuilder::new();
        let matrix = builder.add_input(ArrayIrType::Reference(ReferenceType::new(matrix_type.clone())));
        let row_axes = vec![ArraySliceAxis::new(0, 1, 1), ArraySliceAxis::new(0, 3, 1)];
        let row = builder
            .add_instruction(ReferenceSliceOperation::new(row_axes.clone()), Vec::new(), vec![matrix], None)
            .unwrap()[0];
        let element =
            builder.add_instruction(ReferenceIndexOperation::new(0, 0), Vec::new(), vec![row], None).unwrap()[0];
        let column =
            builder.add_instruction(ReferenceIndexOperation::new(1, 2), Vec::new(), vec![matrix], None).unwrap()[0];
        let element_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        let column_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![column], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![element_value, column_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let root = ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 };
        let row_view = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice { axes: row_axes.clone() });
        let element_view = row_view
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(0) });
        let column_view = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 1, index: ViewIndex::Static(2) });
        assert_eq!(analysis.view(value(0, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(0, 1)), Some(&row_view));
        assert_eq!(analysis.view(value(0, 2)), Some(&element_view));
        assert_eq!(analysis.view(value(0, 3)), Some(&column_view));
        assert_eq!(analysis.view(value(0, 4)), None);
        assert_eq!(analysis.view(value(0, 5)), None);
        assert_eq!(
            analysis.views().collect::<Vec<_>>(),
            vec![
                (value(0, 0), &ArrayReferenceView::root()),
                (value(0, 1), &row_view),
                (value(0, 2), &element_view),
                (value(0, 3), &column_view),
            ],
        );

        // Every view reproduces the referent type the program declares for its value, and the overlapping siblings
        // share one root while selecting different coordinates.
        assert_eq!(row_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [1, 3])));
        assert_eq!(element_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [3])));
        assert_eq!(column_view.output_type(&matrix_type), Ok(ArrayType::new_static(DataType::F32, [2])));
        assert_ne!(element_view, column_view);
        assert_eq!(analysis.analysis().root_of(value(0, 2)), Some(root));
        assert_eq!(analysis.analysis().root_of(value(0, 3)), Some(root));
        assert!(analysis.analysis().is_view(value(0, 2)));
        assert!(analysis.analysis().is_view(value(0, 3)));
        assert_eq!(analysis.analysis().external_source(root), Some(ReferenceSource::Input { index: 0 }));
    }

    #[test]
    fn test_array_reference_analysis_new_copies_identity_aliases_through_while_carries() {
        // The carried reference keeps its root view across the loop, while the body derives its own element view from
        // the carried root and writes through it.
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

        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
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
        assert_eq!(analysis.view(value(2, 1)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(2, 3)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(0, 1)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(1, 1)), Some(&ArrayReferenceView::root()));
        assert_eq!(
            analysis.view(value(1, 2)),
            Some(
                &ArrayReferenceView::root().with_transform_unchecked(ArrayReferenceViewTransform::Index {
                    axis: 0,
                    index: ViewIndex::Static(1)
                })
            ),
        );
        assert_eq!(
            analysis.views().map(|(value, _)| value).collect::<Vec<_>>(),
            vec![value(0, 1), value(1, 1), value(1, 2), value(2, 1), value(2, 3)],
        );
    }

    #[test]
    fn test_array_reference_analysis_new_maps_nested_region_inputs_to_roots() {
        // Both condition branches receive the caller's matrix root as a complete handle and derive their own row
        // views from it, so the branch inputs map to root views bound to the caller root.
        let make_branch = |row: usize| {
            let mut branch = TestBuilder::new();
            let matrix = branch.add_input(reference_type([2, 3]));
            let view = branch
                .add_instruction(ReferenceIndexOperation::new(0, row), Vec::new(), vec![matrix], None)
                .unwrap()[0];
            let snapshot =
                branch.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![view], None).unwrap()[0];
            branch
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };
        let mut builder = TestBuilder::new();
        let true_branch = builder.import_region(make_branch(0).entry_region_ref());
        let false_branch = builder.import_region(make_branch(1).entry_region_ref());
        let predicate = builder.add_input(ArrayIrType::Array(ArrayType::scalar(DataType::Boolean)));
        let matrix = builder.add_input(reference_type([2, 3]));
        let row = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![true_branch, false_branch],
                vec![predicate, matrix],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![row], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let root = ReferenceRoot::RegionInput { region: RegionId::new(2), input_index: 1 };
        assert_eq!(analysis.view(value(2, 1)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(0, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(value(1, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(
            analysis.view(value(0, 1)),
            Some(
                &ArrayReferenceView::root().with_transform_unchecked(ArrayReferenceViewTransform::Index {
                    axis: 0,
                    index: ViewIndex::Static(0)
                })
            ),
        );
        assert_eq!(
            analysis.view(value(1, 1)),
            Some(
                &ArrayReferenceView::root().with_transform_unchecked(ArrayReferenceViewTransform::Index {
                    axis: 0,
                    index: ViewIndex::Static(1)
                })
            ),
        );
        assert_eq!(
            analysis.analysis().region_input_bindings().iter().map(|binding| binding.root()).collect::<Vec<_>>(),
            vec![root, root],
        );
        assert_eq!(
            analysis.analysis().root_of(value(0, 1)),
            Some(ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 })
        );
    }

    #[test]
    fn test_array_reference_analysis_new_rejects_view_type_mismatches() {
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
            ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ArrayReferenceAnalysisError::ViewTypeMismatch {
                operation: REFERENCE_INDEX_OPERATION_NAME,
                instruction: id(0, 0),
                expected: "f32[3]".to_string(),
                actual: "f32[2]".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_reference_analysis_new_rejects_missing_view_transforms() {
        /// Array-IR family extended with one operation that declares a view alias but exposes no transform.
        #[derive(Clone, Debug)]
        enum ViewlessOperation {
            Native(TestOperation),
            View,
        }

        impl Operation for ViewlessOperation {
            type Type = ArrayIrType;

            fn name(&self) -> &'static str {
                match self {
                    Self::Native(operation) => operation.name(),
                    Self::View => "viewless_view",
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

        impl From<ReferenceIndexOperation> for ViewlessOperation {
            fn from(operation: ReferenceIndexOperation) -> Self {
                Self::Native(operation.into())
            }
        }

        impl From<ReferenceSliceOperation> for ViewlessOperation {
            fn from(operation: ReferenceSliceOperation) -> Self {
                Self::Native(operation.into())
            }
        }

        impl ReferenceViewOperation for ViewlessOperation {
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

        impl ArrayReferenceViewOperation for ViewlessOperation {
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

        let mut builder = ProgramBuilder::<TestValue, ViewlessOperation>::new();
        let reference = builder.add_input(reference_type([2]));
        let view = builder.add_instruction(ViewlessOperation::View, Vec::new(), vec![reference], None).unwrap()[0];
        let snapshot = builder
            .add_instruction(
                ViewlessOperation::Native(ReferenceReadOperation::new().into()),
                Vec::new(),
                vec![view],
                None,
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ArrayReferenceAnalysisError::MissingView { operation: "viewless_view", instruction: id(0, 0) }),
        );
    }

    #[test]
    fn test_array_reference_analysis_new_rejects_invalid_view_compositions() {
        // The unchecked instruction indexes axis 2 of a rank-2 referent, which no declared output type can repair.
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
            ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ArrayReferenceAnalysisError::InvalidViewComposition {
                operation: REFERENCE_INDEX_OPERATION_NAME,
                instruction: id(0, 0),
                message: "reference index axis 2 is out of bounds for rank 2".to_string(),
            }),
        );
    }

    #[test]
    fn test_array_reference_analysis_new_propagates_analysis_errors() {
        // Consuming an entry input violates the generic lifetime rules before any view is derived.
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([2]));
        let frozen =
            builder.add_instruction(ReferenceFreezeOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).err(),
            Some(ArrayReferenceAnalysisError::Analysis(ReferenceAnalysisError::ConsumeExternal {
                operation: "reference_freeze",
                instruction: id(0, 0),
                root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
                external_source: ReferenceSource::Input { index: 0 },
            })),
        );
    }

    #[test]
    fn test_array_reference_analysis_analysis() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([2]));
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let root = ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 };
        assert_eq!(analysis.analysis().region(), RegionId::new(0));
        assert_eq!(analysis.analysis().roots().collect::<Vec<_>>(), vec![root]);
        assert_eq!(analysis.analysis().access_modes(root).collect::<Vec<_>>(), vec![ReferenceAccessMode::Read]);
    }

    #[test]
    fn test_array_reference_analysis_view() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([2]));
        let element = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        assert_eq!(analysis.view(value(0, 0)), Some(&ArrayReferenceView::root()));
        assert_eq!(
            analysis.view(value(0, 1)),
            Some(
                &ArrayReferenceView::root().with_transform_unchecked(ArrayReferenceViewTransform::Index {
                    axis: 0,
                    index: ViewIndex::Static(1)
                })
            ),
        );
        assert_eq!(analysis.view(value(0, 2)), None);
        assert_eq!(analysis.view(value(1, 0)), None);
    }

    #[test]
    fn test_array_reference_analysis_views() {
        let mut builder = TestBuilder::new();
        let reference = builder.add_input(reference_type([2]));
        let element = builder
            .add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![reference], None)
            .unwrap()[0];
        let snapshot =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![element], None).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![snapshot], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = ArrayReferenceAnalysis::new(program.entry_region_ref(), 0).unwrap();
        let element_view = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: ViewIndex::Static(1) });
        assert_eq!(
            analysis.views().collect::<Vec<_>>(),
            vec![(value(0, 0), &ArrayReferenceView::root()), (value(0, 1), &element_view)],
        );
    }
}
