//! Array view overlay for generic reference analysis.
//!
//! [`ReferenceAnalysis`] owns value-family-independent roots, aliases, accesses, lifetimes, captures, and structured
//! boundary validation. [`ArrayReferenceAnalysis`] composes that artifact with the one authoritative root-relative
//! [`ArrayReferenceView`] table required by immutable array replay and preserved-reference kernels.

// TODO(eaplatanios): Review this module.
//  Also, is all of this specific to "array IR" or can some of it be moved to core?

use std::ops::Deref;

use crate::arrays::operations::ArrayReferenceOperation;
use crate::arrays::reference_views::{ArrayReferenceView, ArrayReferenceViewTransform};
use crate::arrays::types::arrays::ArrayType;
use crate::arrays::types::ir::ArrayIrType;
use crate::captures::CaptureConstant;
use crate::parameters::Parameterized;
use crate::programs::{
    AtomId, Program, ProgramError, ReferenceAlias, ReferenceAliasKind, ReferenceAnalysis, ReferenceAnalysisError,
    ReferenceType, RegionId, RegionRef, Typed, Value, ValueId,
};

/// Generic reference analysis augmented with validated array-reference views.
///
/// The generic analysis remains the sole owner of root and alias topology. This overlay derives array coordinates
/// exactly once from those alias edges and the operation family's [`ArrayReferenceOperation`] capability.
#[derive(Debug)]
pub struct ArrayReferenceAnalysis {
    /// Value-family-independent topology, lifetime, and access analysis.
    analysis: ReferenceAnalysis,

    /// Validated root-relative view for each reference value, indexed by region then atom.
    views: Vec<Vec<Option<ArrayReferenceView>>>,
}

impl ArrayReferenceAnalysis {
    /// Returns the value-family-independent reference analysis.
    #[inline]
    pub const fn analysis(&self) -> &ReferenceAnalysis {
        &self.analysis
    }

    /// Returns the validated root-relative array view for `value`, or [`None`] when it is not reference-typed or lies
    /// outside the analyzed closure.
    #[inline]
    pub fn view(&self, value: ValueId) -> Option<&ArrayReferenceView> {
        self.views.get(value.region().index())?.get(value.atom().index())?.as_ref()
    }

    /// Derives and validates array views from one generic analysis over the same source arena.
    fn from_analysis<V, O>(source: RegionRef<'_, V, O>, analysis: ReferenceAnalysis) -> Result<Self, ProgramError>
    where
        V: Value<Type = ArrayIrType>,
        O: ArrayReferenceOperation,
    {
        // Region inputs, captures, and allocations begin at their roots. Initializing every analyzed handle this way
        // also keeps table construction linear; the analyzer emits alias edges in program order, so the fold below
        // replaces each alias only after its source alias has received its final composed view.
        let mut views = source
            .arena()
            .iter()
            .enumerate()
            .map(|(region_index, region)| {
                region
                    .atoms()
                    .iter()
                    .enumerate()
                    .map(|(atom_index, _)| {
                        let value = ValueId::new(RegionId::new(region_index), AtomId::new(atom_index));
                        analysis.root(value).is_some().then(ArrayReferenceView::root)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        for alias in analysis.aliases() {
            let region = source.with_id(alias.instruction().region())?;
            let instruction = &region.instructions()[alias.instruction().index()];
            let source_view = views[alias.source().region().index()][alias.source().atom().index()].clone().unwrap();
            let view = match alias.kind() {
                ReferenceAliasKind::Identity => source_view,
                ReferenceAliasKind::View => {
                    let transform = instruction.operation().reference_view_transform().ok_or_else(|| {
                        ProgramError::MalformedProgram(format!(
                            "view alias operation `{}` does not expose a view transform",
                            instruction.operation().name(),
                        ))
                    })?;
                    validate_view_output_type(region, instruction.operation(), alias, &transform)?;
                    source_view.with_transform_unchecked(transform)
                }
            };
            views[alias.output().region().index()][alias.output().atom().index()] = Some(view);
        }

        Ok(Self { analysis, views })
    }
}

impl Deref for ArrayReferenceAnalysis {
    type Target = ReferenceAnalysis;

    #[inline]
    fn deref(&self) -> &Self::Target {
        self.analysis()
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceOperation,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Resolves generic reference topology and validates the root-relative array view for every reference value.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading entry inputs lifted from the program's capture table.
    pub fn analyze_array_references(&self, capture_count: usize) -> Result<ArrayReferenceAnalysis, ProgramError> {
        self.entry_region_ref().analyze_array_references(capture_count)
    }
}

impl<V, O, Input, Output> Program<V, O, Input, Output>
where
    V: CaptureConstant<Type = ArrayIrType>,
    O: ArrayReferenceOperation,
    Input: Parameterized<V>,
    Output: Parameterized<V>,
{
    /// Resolves generic reference topology and validates array views after entry captures have been lifted into the
    /// leading input prefix while attached regions may retain lexical capture constants.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading entry inputs lifted from the program's capture table.
    pub fn analyze_array_references_with_lifted_captures(
        &self,
        capture_count: usize,
    ) -> Result<ArrayReferenceAnalysis, ProgramError> {
        let analysis = self.analyze_references_with_lifted_captures(capture_count)?;
        ArrayReferenceAnalysis::from_analysis(self.entry_region_ref(), analysis)
    }
}

impl<V, O> RegionRef<'_, V, O>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceOperation,
{
    /// Resolves generic reference topology and validates root-relative array views in this attached-region closure.
    ///
    /// # Parameters
    ///
    ///   - `capture_count`: Number of leading root-region inputs lifted from the capture table.
    pub fn analyze_array_references(self, capture_count: usize) -> Result<ArrayReferenceAnalysis, ProgramError> {
        let analysis = self.analyze_references(capture_count)?;
        ArrayReferenceAnalysis::from_analysis(self, analysis)
    }
}

/// Validates the referent type produced by one array view alias.
fn validate_view_output_type<V, O>(
    region: RegionRef<'_, V, O>,
    operation: &O,
    alias: &ReferenceAlias,
    transform: &ArrayReferenceViewTransform,
) -> Result<(), ProgramError>
where
    V: Value<Type = ArrayIrType>,
    O: ArrayReferenceOperation,
{
    let input_type = region.atoms()[alias.source().atom().index()].r#type();
    let output_type = region.atoms()[alias.output().atom().index()].r#type();
    let input_reference = <&ReferenceType<ArrayType>>::try_from(input_type.as_ref())?;
    let expected = transform.output_type(input_reference.referent())?;
    let output_reference = <&ReferenceType<ArrayType>>::try_from(output_type.as_ref())?;
    if output_reference.referent() != &expected {
        return Err(ProgramError::custom(ReferenceAnalysisError::AliasTypeMismatch {
            instruction: alias.instruction(),
            operation: operation.name().to_string(),
            output_index: alias.output_index(),
            input_type: input_type.to_string(),
            output_type: output_type.to_string(),
        }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::borrow::Cow;
    use std::fmt::Display;

    use pretty_assertions::assert_eq;

    use crate::arrays::addressing::ArraySliceAxis;
    use crate::arrays::arrays::Array;
    use crate::arrays::ir::ArrayIrValue;
    use crate::arrays::operations::{ArrayIrOperation, ReferenceIndexOperation, ReferenceSliceOperation};
    use crate::arrays::reference_views::ArrayReference;
    use crate::arrays::types::data::DataType;
    use crate::arrays::types::dimensions::{Dimension, Shape};
    use crate::captures::CaptureReference;
    use crate::operations::{ConditionOperation, ScanOperation, WhileOperation};
    use crate::parameters::Placeholder;
    use crate::programs::{
        Effect, Effects, ExternalReferenceRoot, FreezeReferenceOperation, InstructionId, NewReferenceOperation,
        Operation, OutputRegionProvenance, ProgramBuilder, ReferenceAccess, ReferenceAccessMode, ReferenceInputAccess,
        ReferenceOperationSemantics, ReferenceOutputSemantics, ReferenceReadOperation, ReferenceRoot, ReferenceSource,
        ReferenceSwapOperation, ReferenceTransitiveAccess, RegionArena, RegionInterface, RegionSlot, TypeError,
    };

    use super::*;

    type TestValue = ArrayIrValue<Array>;
    type TestOperation = ArrayIrOperation<Array>;
    type Capture = CaptureReference<ArrayIrType>;
    type CaptureArray = CaptureReference<ArrayType>;
    type CaptureOperation = ArrayIrOperation<CaptureArray>;

    fn scalar_type() -> ArrayType {
        ArrayType::scalar(DataType::F32)
    }

    fn access_signature(access: &ReferenceAccess) -> (InstructionId, usize, ReferenceRoot, ReferenceAccessMode) {
        (access.instruction(), access.input_index(), access.root(), access.mode())
    }

    fn external_root_signature(external: &ExternalReferenceRoot) -> (ReferenceRoot, ReferenceSource) {
        (external.root(), external.source())
    }

    fn transitive_access_signature(access: &ReferenceTransitiveAccess) -> (ReferenceRoot, ReferenceAccessMode) {
        (access.root(), access.mode())
    }

    fn vector_type(size: usize) -> ArrayType {
        ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(size)]))
    }

    #[derive(Clone, Debug)]
    enum TestAliasOperation {
        View,
        Identity,
        MissingViewTransform,
        MismatchedViewType,
        Read,
    }

    impl Display for TestAliasOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for TestAliasOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::View => "test_view",
                Self::Identity => "test_identity_alias",
                Self::MissingViewTransform => "test_missing_view_transform",
                Self::MismatchedViewType => "test_mismatched_view_type",
                Self::Read => "test_read",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            _region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::Identity | Self::MissingViewTransform => vec![input_types[0].clone()],
                Self::View => vec![ReferenceType::new(scalar_type()).into()],
                Self::MismatchedViewType => vec![ReferenceType::new(vector_type(2)).into()],
                Self::Read => vec![scalar_type().into()],
            })
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            Cow::Owned(match self {
                Self::Identity => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::Identity,
                    }],
                    Vec::new(),
                ),
                Self::View | Self::MissingViewTransform | Self::MismatchedViewType => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::View,
                    }],
                    Vec::new(),
                ),
                Self::Read => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ),
            })
        }

        fn effects(&self) -> Effects {
            match self {
                Self::Read => Effects::single(Effect::OrderedState),
                _ => Effects::PURE,
            }
        }
    }

    impl ArrayReferenceOperation for TestAliasOperation {
        fn reference_view_transform(&self) -> Option<ArrayReferenceViewTransform> {
            matches!(self, Self::View | Self::MismatchedViewType)
                .then_some(ArrayReferenceViewTransform::Index { axis: 0, index: 0 })
        }
    }

    #[test]
    fn test_array_reference_analysis_records_root_index_slice_and_sibling_views() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let root = builder.add_input(ReferenceType::new(vector_type(4)).into());
        let slice = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(1, 3, 1)]), Vec::new(), vec![root])
            .unwrap()[0];
        let indexed = builder.add_instruction(ReferenceIndexOperation::new(0, 1), Vec::new(), vec![slice]).unwrap()[0];
        let sibling = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 3, 1)]), Vec::new(), vec![root])
            .unwrap()[0];
        let indexed_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![indexed]).unwrap()[0];
        let sibling_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![sibling]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![indexed_value, sibling_value],
                vec![Placeholder],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let root_view = ArrayReferenceView::root();
        let slice_view = root_view
            .clone()
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(1, 3, 1)] });
        let indexed_view = slice_view
            .clone()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 1 });
        let sibling_view = root_view
            .clone()
            .with_transform_unchecked(ArrayReferenceViewTransform::Slice { axes: vec![ArraySliceAxis::new(0, 3, 1)] });
        assert_eq!(analysis.view(ValueId::new(program.entry(), root)), Some(&root_view));
        assert_eq!(analysis.view(ValueId::new(program.entry(), slice)), Some(&slice_view));
        assert_eq!(analysis.view(ValueId::new(program.entry(), indexed)), Some(&indexed_view));
        assert_eq!(analysis.view(ValueId::new(program.entry(), sibling)), Some(&sibling_view));
        assert_ne!(
            analysis.view(ValueId::new(program.entry(), indexed)),
            analysis.view(ValueId::new(program.entry(), sibling))
        );
        assert_eq!(analysis.analysis().aliases().len(), 3);
    }

    #[test]
    fn test_array_reference_analysis_preserves_identity_alias_views() {
        let mut builder = ProgramBuilder::<TestValue, TestAliasOperation>::new();
        let root = builder.add_input(ReferenceType::new(vector_type(4)).into());
        let view = builder.add_instruction(TestAliasOperation::View, Vec::new(), vec![root]).unwrap()[0];
        let alias = builder.add_instruction(TestAliasOperation::Identity, Vec::new(), vec![view]).unwrap()[0];
        let output = builder.add_instruction(TestAliasOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let expected = ArrayReferenceView::root()
            .with_transform_unchecked(ArrayReferenceViewTransform::Index { axis: 0, index: 0 });
        assert_eq!(analysis.view(ValueId::new(program.entry(), root)), Some(&ArrayReferenceView::root()));
        assert_eq!(analysis.view(ValueId::new(program.entry(), view)), Some(&expected));
        assert_eq!(analysis.view(ValueId::new(program.entry(), alias)), Some(&expected));
        assert_eq!(analysis.alias(ValueId::new(program.entry(), alias)).unwrap().kind(), ReferenceAliasKind::Identity);
    }

    #[test]
    fn test_array_reference_analysis_rejects_missing_transform_and_view_type_mismatch() {
        let build = |operation| {
            let mut builder = ProgramBuilder::<TestValue, TestAliasOperation>::new();
            let root = builder.add_input(ReferenceType::new(vector_type(4)).into());
            let alias = builder.add_instruction(operation, Vec::new(), vec![root]).unwrap()[0];
            let output = builder.add_instruction(TestAliasOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![output], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        let missing = build(TestAliasOperation::MissingViewTransform);
        assert_eq!(
            missing.analyze_array_references(0).unwrap_err(),
            ProgramError::MalformedProgram(
                "view alias operation `test_missing_view_transform` does not expose a view transform".to_string(),
            ),
        );

        let mismatched = build(TestAliasOperation::MismatchedViewType);
        let error = mismatched.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::AliasTypeMismatch {
                instruction: InstructionId::new(mismatched.entry(), 0),
                operation: "test_mismatched_view_type".to_string(),
                output_index: 0,
                input_type: ReferenceType::new(vector_type(4)).to_string(),
                output_type: ReferenceType::new(vector_type(2)).to_string(),
            }),
        );
    }

    #[test]
    fn test_array_reference_analysis_enforces_condition_view_boundaries() {
        let root_type = vector_type(3);
        let view_type = vector_type(2);

        // A view cannot enter a condition region directly.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch_view = branch_builder.add_input(ReferenceType::new(view_type.clone()).into());
        let branch_value = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![branch_view])
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![branch_value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let root = builder.add_input(ReferenceType::new(root_type.clone()).into());
        let view = builder
            .add_instruction(ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]), Vec::new(), vec![root])
            .unwrap()[0];
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, view])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ViewRegionInputBoundary {
                instruction: InstructionId::new(program.entry(), 1),
                operation: "condition".to_string(),
                region_index: 0,
                input_index: 1,
            }),
        );
        let replay_error = program
            .interpret(vec![
                ArrayIrValue::Array(Array::scalar(true)),
                ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
            ])
            .unwrap_err();
        assert_eq!(replay_error.downcast_custom::<ReferenceAnalysisError>(), error.downcast_custom());

        // Passing the root and recreating the same view inside each branch is valid.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch_root = branch_builder.add_input(ReferenceType::new(root_type.clone()).into());
        let branch_view = branch_builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![branch_root],
            )
            .unwrap()[0];
        let branch_value = branch_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![branch_view])
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![branch_value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let root = builder.add_input(ReferenceType::new(root_type.clone()).into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, root])
            .unwrap()[0];
        let valid = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        valid.analyze_array_references(0).unwrap();

        // A view recreated inside a branch cannot escape through the condition's reference result.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch_root = branch_builder.add_input(ReferenceType::new(root_type.clone()).into());
        let branch_view = branch_builder
            .add_instruction(
                ReferenceSliceOperation::new(vec![ArraySliceAxis::new(0, 2, 1)]),
                Vec::new(),
                vec![branch_root],
            )
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![branch_view], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let root = builder.add_input(ReferenceType::new(root_type).into());
        let view = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, root])
            .unwrap()[0];
        let value = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![view]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ViewRegionOutputBoundary {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "condition".to_string(),
                region_index: 0,
                output_index: 0,
            }),
        );
        let replay_error = program
            .interpret(vec![
                ArrayIrValue::Array(Array::scalar(true)),
                ArrayIrValue::Reference(ArrayReference::new(Array::vector(vec![1.0_f32, 2.0, 3.0]))),
            ])
            .unwrap_err();
        assert_eq!(replay_error.downcast_custom::<ReferenceAnalysisError>(), error.downcast_custom());
    }

    /// Malformed reference operation descriptors used to pin each static-analysis diagnostic.
    #[derive(Clone, Debug)]
    enum MalformedReferenceOperation {
        /// Leaves a reference input unclassified.
        Unclassified,

        /// Produces a reference output without root or alias semantics.
        UnclassifiedOutput,

        /// Describes an access beyond the operand list.
        InvalidAccess,

        /// Describes a root beyond the result list.
        InvalidOutput,

        /// Describes a reference access on an array operand.
        NonReferenceAccess,

        /// Describes a new root on an array result.
        BadRoot,

        /// Describes an alias whose result type differs from its source type.
        BadAlias,

        /// Produces a valid fresh reference root.
        Root,

        /// Produces a valid alias of a reference operand.
        Alias,

        /// Reads a valid reference operand.
        Read,

        /// Consumes a valid reference operand.
        Consume,

        /// Describes reference state without declaring the ordered-state effect.
        MissingEffect,

        /// Forwards an operand into a computation region.
        ComputationCarrier,

        /// Calls a computation region and forwards its outputs to the caller.
        CallCarrier,

        /// Establishes a capture scope longer than the attached region input boundary.
        InvalidCaptureScopeCarrier,

        /// Declares an out-of-range attached-region source for a reference output.
        InvalidOutputSourceCarrier,

        /// Forwards one attached-region reference output through two implicit handles.
        DuplicateOutputCarrier,

        /// Reports an invalid source operand for a region input.
        InvalidSourceCarrier,

        /// Forwards an operand into a dormant rule region.
        RuleCarrier,
    }

    impl Display for MalformedReferenceOperation {
        fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            formatter.write_str(self.name())
        }
    }

    impl Operation for MalformedReferenceOperation {
        type Type = ArrayIrType;

        fn name(&self) -> &'static str {
            match self {
                Self::Unclassified => "unclassified_reference",
                Self::UnclassifiedOutput => "unclassified_reference_output",
                Self::InvalidAccess => "invalid_reference_access",
                Self::InvalidOutput => "invalid_reference_output",
                Self::NonReferenceAccess => "non_reference_access",
                Self::BadRoot => "bad_reference_root",
                Self::BadAlias => "bad_reference_alias",
                Self::Root => "test_reference_root",
                Self::Alias => "test_reference_alias",
                Self::Read => "test_reference_read",
                Self::Consume => "test_reference_consume",
                Self::MissingEffect => "test_missing_reference_effect",
                Self::ComputationCarrier => "test_computation_carrier",
                Self::CallCarrier => "test_call_carrier",
                Self::InvalidCaptureScopeCarrier => "test_invalid_capture_scope_carrier",
                Self::InvalidOutputSourceCarrier => "test_invalid_output_source_carrier",
                Self::DuplicateOutputCarrier => "test_duplicate_output_carrier",
                Self::InvalidSourceCarrier => "test_invalid_source_carrier",
                Self::RuleCarrier => "test_rule_carrier",
            }
        }

        fn infer_output_types(
            &self,
            input_types: &[ArrayIrType],
            region_interfaces: &[RegionInterface<ArrayIrType>],
        ) -> Result<Vec<ArrayIrType>, TypeError> {
            Ok(match self {
                Self::BadRoot | Self::Read => vec![scalar_type().into()],
                Self::Root | Self::MissingEffect | Self::InvalidOutput | Self::UnclassifiedOutput => {
                    vec![ReferenceType::new(scalar_type()).into()]
                }
                Self::Alias => vec![input_types[0].clone()],
                Self::CallCarrier | Self::InvalidCaptureScopeCarrier => region_interfaces[0].output_types().to_vec(),
                Self::InvalidOutputSourceCarrier => vec![region_interfaces[0].output_types()[0].clone()],
                Self::DuplicateOutputCarrier => vec![region_interfaces[0].output_types()[0].clone(); 2],
                Self::BadAlias => vec![ReferenceType::new(ArrayType::scalar(DataType::F64)).into()],
                Self::Consume => vec![scalar_type().into()],
                Self::Unclassified
                | Self::InvalidAccess
                | Self::NonReferenceAccess
                | Self::ComputationCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => Vec::new(),
            })
        }

        fn input_region_provenance(&self, region_index: usize, input_index: usize) -> Option<usize> {
            match self {
                Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                    if region_index == 0 =>
                {
                    Some(input_index)
                }
                Self::InvalidSourceCarrier if region_index == 0 => Some(1),
                _ => None,
            }
        }

        fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
            match self {
                Self::CallCarrier | Self::InvalidCaptureScopeCarrier => {
                    vec![OutputRegionProvenance { region_index: 0, output_index }]
                }
                Self::InvalidOutputSourceCarrier => {
                    vec![OutputRegionProvenance { region_index: 1, output_index: 0 }]
                }
                Self::DuplicateOutputCarrier => vec![OutputRegionProvenance { region_index: 0, output_index: 0 }],
                _ => Vec::new(),
            }
        }

        fn region_capture_input_count(&self, region_index: usize) -> Option<usize> {
            match self {
                Self::CallCarrier if region_index == 0 => Some(1),
                Self::InvalidCaptureScopeCarrier if region_index == 0 => Some(2),
                _ => None,
            }
        }

        fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
            Cow::Owned(match self {
                Self::Unclassified | Self::UnclassifiedOutput => ReferenceOperationSemantics::default(),
                Self::InvalidAccess => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(1, ReferenceAccessMode::Read)],
                ),
                Self::NonReferenceAccess => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ),
                Self::BadRoot => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::InvalidOutput => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 1 }],
                    Vec::new(),
                ),
                Self::Root => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::MissingEffect => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::NewRoot { output_index: 0 }],
                    Vec::new(),
                ),
                Self::Alias | Self::BadAlias => ReferenceOperationSemantics::new(
                    vec![ReferenceOutputSemantics::Alias {
                        output_index: 0,
                        input_index: 0,
                        kind: ReferenceAliasKind::Identity,
                    }],
                    Vec::new(),
                ),
                Self::Read => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Read)],
                ),
                Self::Consume => ReferenceOperationSemantics::new(
                    Vec::new(),
                    vec![ReferenceInputAccess::new(0, ReferenceAccessMode::Consume)],
                ),
                Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => ReferenceOperationSemantics::default(),
            })
        }

        fn region_slots(&self) -> &'static [RegionSlot] {
            match self {
                Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier => const { &[RegionSlot::computation("body")] },
                Self::RuleCarrier => const { &[RegionSlot::rule("rule")] },
                _ => &[],
            }
        }

        fn effects(&self) -> Effects {
            match self {
                Self::InvalidAccess
                | Self::NonReferenceAccess
                | Self::BadRoot
                | Self::InvalidOutput
                | Self::Root
                | Self::Read
                | Self::Consume => Effects::single(Effect::OrderedState),
                Self::Unclassified
                | Self::UnclassifiedOutput
                | Self::BadAlias
                | Self::Alias
                | Self::MissingEffect
                | Self::ComputationCarrier
                | Self::CallCarrier
                | Self::InvalidCaptureScopeCarrier
                | Self::InvalidOutputSourceCarrier
                | Self::DuplicateOutputCarrier
                | Self::InvalidSourceCarrier
                | Self::RuleCarrier => Effects::PURE,
            }
        }
    }

    impl ArrayReferenceOperation for MalformedReferenceOperation {}

    #[test]
    fn test_reference_analysis_resolves_local_roots_and_consumption() {
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![read, frozen], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(RegionId::new(0), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(RegionId::new(0), reference)), Some(root));
        assert_eq!(
            analysis.accesses().iter().map(access_signature).collect::<Vec<_>>(),
            &[
                (InstructionId::new(RegionId::new(0), 1), 0, root, ReferenceAccessMode::Read),
                (InstructionId::new(RegionId::new(0), 2), 0, root, ReferenceAccessMode::Consume),
            ],
        );
        assert!(analysis.external_roots().is_empty());

        // Every alias in the family observes root consumption, even when the invalid use is a plain read.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, read], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "reference_read".to_string(),
                input_index: 0,
                root,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_classifies_external_roots() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let captured = builder.add_input(reference_type.clone().into());
        let public = builder.add_input(reference_type.into());
        let captured_value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured]).unwrap()[0];
        let public_value = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![public]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(
                vec![captured_value, public_value],
                vec![Placeholder; 2],
                vec![Placeholder; 2],
            )
            .unwrap();

        let analysis = program.analyze_array_references(1).unwrap();
        assert_eq!(
            analysis.external_roots().iter().map(external_root_signature).collect::<Vec<_>>(),
            &[
                (
                    ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
                    ReferenceSource::Capture { index: 0 },
                ),
                (
                    ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 1 },
                    ReferenceSource::PublicInput { index: 0 },
                ),
            ],
        );

        // External roots are never consumable by user code in the initial lifecycle model.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(scalar_type()).into());
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![external]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "freeze_reference".to_string(),
                input_index: 0,
                root: ReferenceRoot::RegionInput { region: RegionId::new(0), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_region_reference_analysis_ignores_unreachable_sibling_regions() {
        let mut valid_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let array = valid_builder.add_input(scalar_type().into());
        let valid = valid_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![array], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut invalid_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = invalid_builder.add_input(ReferenceType::new(scalar_type()).into());
        let invalid = invalid_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let arena = RegionArena::from_regions(vec![
            valid.entry_region_ref().region().clone(),
            invalid.entry_region_ref().region().clone(),
        ])
        .unwrap();
        let valid = RegionRef::new(&arena, RegionId::new(0)).unwrap();
        let invalid = RegionRef::new(&arena, RegionId::new(1)).unwrap();
        // The reference-free root sees an arena whose sibling is illegal, yet its own closure still takes the
        // reference-free fast path; analyzing that sibling as the root reports the illegal escape.
        assert!(valid.analyze_array_references(0).unwrap().is_reference_free());
        let error = invalid.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: RegionId::new(1),
                output_index: 0,
                root: ReferenceRoot::RegionInput { region: RegionId::new(1), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_substitutes_condition_region_inputs() {
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value =
            branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let instruction = InstructionId::new(program.entry(), 0);
        let source_root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let branch_root = ReferenceRoot::RegionInput { region: branch, input_index: 0 };
        assert_eq!(analysis.accesses()[0].root(), branch_root);

        // Both branch invocations of the shared region substitute the same caller root, and a root that never crosses
        // the boundary has no formal counterpart in either invocation.
        assert_eq!(analysis.region_root_for_source(instruction, 0, source_root), Some(branch_root));
        assert_eq!(analysis.region_root_for_source(instruction, 1, source_root), Some(branch_root));
        assert_eq!(analysis.region_root_for_source(instruction, 0, branch_root), None);
    }

    #[test]
    fn test_reference_analysis_rejects_inconsistent_condition_reference_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut first_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = first_builder.add_input(reference_type.clone().into());
        first_builder.add_input(reference_type.clone().into());
        let first_branch = first_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut second_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        second_builder.add_input(reference_type.clone().into());
        let second = second_builder.add_input(reference_type.clone().into());
        let second_branch = second_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first_branch = builder.import_region(first_branch.entry_region_ref());
        let second_branch = builder.import_region(second_branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(
                ConditionOperation::new(),
                vec![first_branch, second_branch],
                vec![predicate, first, second],
            )
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 3], Vec::new())
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InconsistentRegionOutputRoots {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "condition".to_string(),
                output_index: 0,
                first_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
                second_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 2 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_invalid_or_duplicated_structured_reference_outputs() {
        let reference_type = ReferenceType::new(scalar_type());
        let make_body = || {
            let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
            let reference = builder.add_input(reference_type.clone().into());
            builder
                .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
                .unwrap()
        };

        // A declared attached-region source for a reference output must name an existing region and output.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let body = builder.import_region(make_body().entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidOutputSourceCarrier, vec![body], vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidRegionOutputSource {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_invalid_output_source_carrier".to_string(),
                output_index: 0,
                region_index: 1,
                region_output_index: 0,
            }),
        );

        // Two caller outputs cannot forward one attached-region reference output, because that would hand the caller
        // two unaliased handles to the same root.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let body = builder.import_region(make_body().entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::DuplicateOutputCarrier, vec![body], vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionOutputAlias {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_duplicate_output_carrier".to_string(),
                first_output_index: 0,
                second_output_index: 1,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_resolves_attached_lifted_capture_constants() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let captured = branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = branch_builder
            .add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![captured])
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();

        let ordinary_error = program.analyze_array_references(1).unwrap_err();
        assert_eq!(
            ordinary_error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceConstant { region: branch, atom: AtomId::new(0) }),
        );

        let analysis = program.analyze_array_references_with_lifted_captures(1).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(branch, captured)), Some(root));
        assert_eq!(analysis.accesses()[0].root(), root);
        assert_eq!(
            analysis.region_summary(branch).unwrap().iter().map(transitive_access_signature).collect::<Vec<_>>(),
            &[(root, ReferenceAccessMode::Read)],
        );
        assert_eq!(
            analysis
                .instruction_summary(InstructionId::new(program.entry(), 0))
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            &[(root, ReferenceAccessMode::Read)],
        );
    }

    #[test]
    fn test_reference_analysis_forwards_a_capture_root_out_of_a_doubly_nested_region() {
        // A root reaching a region only as a forwarded result of a doubly nested condition is still usable there.
        // Liveness is the complement of consumption, so every handle-producing path — including a root forwarded out
        // of an arbitrarily deep region — yields a live handle, even though the middle region neither declares the
        // root as a formal input nor names it with a capture constant of its own.
        let reference_type = ReferenceType::new(scalar_type());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut inner_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let captured = inner_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let inner = inner_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut middle_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let inner = middle_builder.import_region(inner.entry_region_ref());
        let inner_predicate = middle_builder.add_constant(Capture::new(1, predicate_type.clone().into()));
        let forwarded = middle_builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![inner, inner],
                vec![inner_predicate],
            )
            .unwrap()[0];
        let value =
            middle_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![forwarded]).unwrap()[0];
        let middle = middle_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let middle = builder.import_region(middle.entry_region_ref());
        builder.add_input(reference_type.into());
        let predicate = builder.add_input(predicate_type.into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![middle, middle],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references_with_lifted_captures(2).unwrap();
        let inner = program.regions()[middle.index()].instructions()[0].regions()[0];
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(inner, captured)), Some(root));
        assert_eq!(analysis.root(ValueId::new(middle, forwarded)), Some(root));
        assert_eq!(
            analysis.accesses().iter().map(access_signature).collect::<Vec<_>>(),
            &[(InstructionId::new(middle, 1), 0, root, ReferenceAccessMode::Read)],
        );

        // The summaries translate the same access outward through both region levels. The read is local to the middle
        // region, so the inner condition contributes nothing, and every enclosing summary names the entry root rather
        // than a region-local formal of the middle region.
        let expected = [(root, ReferenceAccessMode::Read)];
        assert_eq!(analysis.instruction_summary(InstructionId::new(middle, 0)), None);
        assert_eq!(
            analysis
                .instruction_summary(InstructionId::new(middle, 1))
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            expected,
        );
        assert_eq!(
            analysis
                .instruction_summary(InstructionId::new(program.entry(), 0))
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            expected,
        );
        assert_eq!(analysis.region_summary(inner).unwrap(), &[]);
        assert_eq!(
            analysis.region_summary(middle).unwrap().iter().map(transitive_access_signature).collect::<Vec<_>>(),
            expected,
        );
        assert_eq!(
            analysis
                .region_summary(program.entry())
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            expected,
        );
    }

    #[test]
    fn test_reference_analysis_deduplicates_shared_diamond_summaries() {
        let reference_type = ReferenceType::new(scalar_type());
        let predicate_type = ArrayType::scalar(DataType::Boolean);
        let mut leaf_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let reference = leaf_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value =
            leaf_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let mut shared = leaf_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder])
            .unwrap();

        // Each level attaches the same preceding region twice. A path-sensitive flattening would construct 2^64
        // identical access facts, while the canonical summary retains one fact per root/mode/source triple.
        for _ in 0..64 {
            let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
            let branch = builder.import_region(shared.entry_region_ref());
            let predicate = builder.add_constant(Capture::new(1, predicate_type.clone().into()));
            let value = builder
                .add_instruction(
                    ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                    vec![branch, branch],
                    vec![predicate],
                )
                .unwrap()[0];
            shared = builder.build::<Vec<Capture>, Vec<Capture>>(vec![value], Vec::new(), vec![Placeholder]).unwrap();
        }

        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(shared.entry_region_ref());
        builder.add_input(reference_type.into());
        let predicate = builder.add_input(predicate_type.into());
        let value = builder
            .add_instruction(
                ConditionOperation::<ArrayIrValue<CaptureArray>>::new(),
                vec![branch, branch],
                vec![predicate],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references_with_lifted_captures(2).unwrap();
        let instruction = InstructionId::new(program.entry(), 0);
        assert_eq!(analysis.instruction_summary(instruction).unwrap().len(), 1);
        assert_eq!(analysis.region_summary(program.entry()).unwrap().len(), 1);
    }

    #[test]
    fn test_reference_analysis_rejects_capture_index_beyond_the_lifted_prefix() {
        // A capture constant addresses its active lexical scope, so an index past that scope's addressable inputs is a
        // precise diagnostic rather than a silent out-of-range read.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(1, reference_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidReferenceCapture {
                region: branch,
                atom: AtomId::new(0),
                capture_index: 1,
                scope_input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_constant_type_mismatch() {
        // The constant and the lifted boundary it resolves against must declare exactly the same reference type, so a
        // differing referent element type cannot be silently reinterpreted.
        let reference_type = ReferenceType::new(scalar_type());
        let mismatched_type = ReferenceType::new(ArrayType::scalar(DataType::F64));
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(0, mismatched_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.clone().into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCaptureTypeMismatch {
                region: branch,
                atom: AtomId::new(0),
                capture_index: 0,
                capture_type: ArrayIrType::Reference(mismatched_type).to_string(),
                input_type: ArrayIrType::Reference(reference_type).to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_duplicate_capture_constant_aliases() {
        // Two constants naming one captured root would create parallel unaliased handles inside a single invocation,
        // which the lifecycle model forbids.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let branch = branch_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![branch], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateReferenceCaptureAlias {
                region: branch,
                first_atom: AtomId::new(0),
                second_atom: AtomId::new(1),
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_aliasing_a_formal_region_argument() {
        // A branch that receives a root as a formal argument cannot also reach that same root through a lexical capture
        // constant, because the two handles would alias one resource inside one invocation.
        let reference_type = ReferenceType::new(scalar_type());
        let mut branch_builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        branch_builder.add_input(reference_type.clone().into());
        let captured = branch_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value =
            branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![captured]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, CaptureOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        builder
            .add_instruction(ConditionOperation::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionCaptureAlias {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "condition".to_string(),
                region_index: 0,
                input_index: 0,
                capture_region: branch,
                capture_atom: captured,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_call_capture_scope_longer_than_its_region_boundary() {
        // A call-like operation establishes its callee's capture scope from that region's leading inputs, so a declared
        // prefix longer than the region boundary is rejected while the scope map is still being built.
        let reference_type = ReferenceType::new(scalar_type());
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        callee_builder.add_input(reference_type.clone().into());
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidCaptureScopeCarrier, vec![callee], vec![reference])
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidRegionCaptureCount {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_invalid_capture_scope_carrier".to_string(),
                region_index: 0,
                capture_count: 2,
                input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_ambiguous_shared_capture_scopes() {
        // A shared region reachable both through a scope-establishing call and through ordinary control flow would have
        // its capture constants resolve against two different lexical scopes. Resolving them would make legality
        // path-sensitive, so the shared region is rejected outright.
        let reference_type = ReferenceType::new(scalar_type());
        let mut shared_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        shared_builder.add_input(reference_type.clone().into());
        shared_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let shared = shared_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let shared = builder.import_region(shared.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![shared], vec![first])
            .unwrap();
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![shared], Vec::new())
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();

        let error = program.analyze_array_references_with_lifted_captures(2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::AmbiguousReferenceCaptureScope {
                region: shared,
                first_scope: shared,
                first_scope_input_count: 1,
                second_scope: program.entry(),
                second_scope_input_count: 2,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_propagates_while_and_scan_carry_roots() {
        let reference_type = ReferenceType::new(scalar_type());

        // A while loop's reference carry keeps the caller root, and the instruction summary reports the union of
        // the condition region's read and the body region's write in the caller's own namespace.
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition_reference = condition_builder.add_input(reference_type.clone().into());
        condition_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![condition_reference])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body_reference = body_builder.add_input(reference_type.clone().into());
        let replacement = body_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        body_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![body_reference, replacement])
            .unwrap();
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![body_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.clone().into());
        let final_reference =
            builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![reference]).unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), final_reference)), Some(root));
        assert_eq!(
            analysis
                .instruction_summary(InstructionId::new(program.entry(), 0))
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            &[(root, ReferenceAccessMode::Read), (root, ReferenceAccessMode::Write),],
        );

        // A scan carry behaves the same way, and a body that only reads produces exactly one summarized access.
        let mut scan_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let scan_reference = scan_body_builder.add_input(reference_type.clone().into());
        scan_body_builder
            .add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![scan_reference])
            .unwrap();
        let scan_body = scan_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![scan_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let scan_body = builder.import_region(scan_body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let final_reference = builder
            .add_instruction(ScanOperation::<TestValue>::new(1, 2), vec![scan_body], vec![reference])
            .unwrap()[0];
        let value =
            builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![final_reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_array_references(0).unwrap();
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), final_reference)), Some(root));
        assert_eq!(
            analysis
                .instruction_summary(InstructionId::new(program.entry(), 0))
                .unwrap()
                .iter()
                .map(transitive_access_signature)
                .collect::<Vec<_>>(),
            &[(root, ReferenceAccessMode::Read)],
        );
    }

    #[test]
    fn test_reference_analysis_rejects_permuted_while_reference_carries() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(reference_type.clone().into());
        condition_builder.add_input(reference_type.clone().into());
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        // A body that swaps the two carries has no consistent fixed point for either root and is rejected precisely.
        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = body_builder.add_input(reference_type.clone().into());
        let second = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![first, second]).unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCarryRootMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "while".to_string(),
                output_index: 0,
                input_index: 0,
                input_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                output_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_permuted_scan_reference_carries() {
        let reference_type = ReferenceType::new(scalar_type());

        // A body that swaps the two carries has no consistent fixed point for either root and is rejected precisely.
        let mut scan_body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = scan_body_builder.add_input(reference_type.clone().into());
        let second = scan_body_builder.add_input(reference_type.clone().into());
        let scan_body = scan_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![second, first], vec![Placeholder; 2], vec![Placeholder; 2])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let body = builder.import_region(scan_body.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder
            .add_instruction(ScanOperation::<TestValue>::new(2, 0), vec![body], vec![first, second])
            .unwrap();
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceCarryRootMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "scan".to_string(),
                output_index: 0,
                input_index: 0,
                input_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
                output_root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_mutation_of_while_condition_input() {
        let reference_type = ReferenceType::new(scalar_type());
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = condition_builder.add_input(reference_type.clone().into());
        let replacement = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        condition_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = body_builder.add_input(reference_type.clone().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![reference], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![reference]).unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();

        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 };
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ForbiddenRegionInputAccess {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "while".to_string(),
                region_index: 0,
                root,
                mode: ReferenceAccessMode::Write,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_allows_condition_local_reference_mutation() {
        let mut condition_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        condition_builder.add_input(scalar_type().into());
        let initial = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(1.0f32)));
        let reference =
            condition_builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let replacement = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(2.0f32)));
        condition_builder
            .add_instruction(ReferenceSwapOperation::new(), Vec::new(), vec![reference, replacement])
            .unwrap();
        let predicate = condition_builder.add_constant(ArrayIrValue::Array(Array::scalar(false)));
        let condition = condition_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![predicate], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut body_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let state = body_builder.add_input(scalar_type().into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let condition = builder.import_region(condition.entry_region_ref());
        let body = builder.import_region(body.entry_region_ref());
        let state = builder.add_input(scalar_type().into());
        let final_state =
            builder.add_instruction(WhileOperation::new(), vec![condition, body], vec![state]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![final_state], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        assert!(analysis.external_roots().is_empty());
        assert!(analysis.instruction_summary(InstructionId::new(program.entry(), 0)).is_none());
    }

    #[test]
    fn test_reference_analysis_rejects_duplicate_roots_within_one_region_invocation() {
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let first = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![first]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(
                ConditionOperation::<TestValue>::new(),
                vec![branch, branch],
                vec![predicate, reference, reference],
            )
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let instruction = InstructionId::new(program.entry(), 0);
        let root = ReferenceRoot::RegionInput { region: program.entry(), input_index: 1 };
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::DuplicateRegionInputAlias {
                instruction,
                operation: "condition".to_string(),
                region_index: 0,
                first_input_index: 0,
                second_input_index: 1,
                root,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_scopes_branch_local_allocations_to_their_creating_region() {
        // A root allocated inside a shared branch belongs to that branch invocation, never to the caller, so it stays a
        // region-local allocation root and never reaches the external boundary.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = branch_builder.add_input(scalar_type().into());
        let reference =
            branch_builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let value =
            branch_builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let initial = builder.add_input(scalar_type().into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, initial])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();

        let analysis = program.analyze_array_references(0).unwrap();
        let root = ReferenceRoot::Allocation { instruction: InstructionId::new(branch, 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(branch, reference)), Some(root));
        assert_eq!(analysis.accesses()[0].root(), root);
        assert!(analysis.external_roots().is_empty());
        assert!(analysis.region_summary(branch).unwrap().is_empty());
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_an_inherited_branch_root() {
        // A branch that receives a caller root as a formal argument only borrows it, so consuming it inside the branch
        // is reported against the branch's own region-input root.
        let mut branch_builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let reference = branch_builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = branch_builder
            .add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference])
            .unwrap()[0];
        let branch = branch_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let branch = builder.import_region(branch.entry_region_ref());
        let predicate = builder.add_input(ArrayType::scalar(DataType::Boolean).into());
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        let value = builder
            .add_instruction(ConditionOperation::<TestValue>::new(), vec![branch, branch], vec![predicate, reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder; 2], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(branch, 0),
                operation: "freeze_reference".to_string(),
                input_index: 0,
                root: ReferenceRoot::RegionInput { region: branch, input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_roots_escaping_a_region_output() {
        // References are second class in every region, so a nested region cannot return a resolved capture root through
        // its own output boundary even though the constant itself is legal there.
        let reference_type = ReferenceType::new(scalar_type());
        let mut child_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let captured = child_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let child = child_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![captured], Vec::new(), vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let child = builder.import_region(child.entry_region_ref());
        builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![child], Vec::new())
            .unwrap();
        let program = builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references_with_lifted_captures(1).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: child,
                output_index: 0,
                root: ReferenceRoot::RegionInput { region: program.entry(), input_index: 0 },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_constants_in_a_call_scope_root_region() {
        // A callee region is the root of its own capture scope, so a capture constant there would have to address that
        // region's own inputs; such a constant is rejected exactly like an immediate reference constant.
        let reference_type = ReferenceType::new(scalar_type());
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        callee_builder.add_input(reference_type.clone().into());
        let captured = callee_builder.add_constant(Capture::new(0, reference_type.clone().into()));
        let value = callee_builder
            .add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![captured])
            .unwrap()[0];
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let reference = builder.add_input(reference_type.into());
        let value = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![callee], vec![reference])
            .unwrap()[0];
        let program = builder
            .build::<Vec<Capture>, Vec<Capture>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceConstant { region: callee, atom: AtomId::new(1) }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_index_beyond_a_nested_call_scope() {
        // Inside a callee, capture indices address that callee's declared capture prefix rather than the entry
        // boundary, so the reported scope input count is the nested prefix length and not the caller's input count.
        let reference_type = ReferenceType::new(scalar_type());
        let mut child_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        child_builder.add_constant(Capture::new(1, reference_type.clone().into()));
        let child = child_builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), Vec::new(), Vec::new()).unwrap();
        let mut callee_builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let child = callee_builder.import_region(child.entry_region_ref());
        callee_builder.add_input(reference_type.clone().into());
        callee_builder.add_input(reference_type.clone().into());
        callee_builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![child], Vec::new())
            .unwrap();
        let callee = callee_builder
            .build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new())
            .unwrap();
        let mut builder = ProgramBuilder::<Capture, MalformedReferenceOperation>::new();
        let callee = builder.import_region(callee.entry_region_ref());
        let first = builder.add_input(reference_type.clone().into());
        let second = builder.add_input(reference_type.into());
        builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![callee], vec![first, second])
            .unwrap();
        let program =
            builder.build::<Vec<Capture>, Vec<Capture>>(Vec::new(), vec![Placeholder; 2], Vec::new()).unwrap();
        let child = program.regions()[callee.index()].instructions()[0].regions()[0];
        let error = program.analyze_array_references_with_lifted_captures(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidReferenceCapture {
                region: child,
                atom: AtomId::new(0),
                capture_index: 1,
                scope_input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_reference_state_in_nested_rule_closure() {
        let mut leaf_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = leaf_builder.add_input(scalar_type().into());
        leaf_builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap();
        let leaf = leaf_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        let mut rule_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let leaf = rule_builder.import_region(leaf.entry_region_ref());
        rule_builder
            .add_instruction(MalformedReferenceOperation::ComputationCarrier, vec![leaf], Vec::new())
            .unwrap();
        let rule = rule_builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let rule = builder.import_region(rule.entry_region_ref());
        builder.add_instruction(MalformedReferenceOperation::RuleCarrier, vec![rule], Vec::new()).unwrap();
        let program = builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), Vec::new(), Vec::new()).unwrap();

        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::RuleRegion { region: RegionId::new(0) }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_capture_count_exceeding_the_entry_input_count() {
        // The declared capture prefix is validated against the entry boundary before any operation semantics are
        // inspected, so the malformed operation in this fixture is never reached.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::MissingEffect, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(2).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidCaptureCount { capture_count: 2, input_count: 1 }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_stateful_semantics_without_ordered_state() {
        // Stateful reference semantics require the conservative ordered-state effect so that reference work can never
        // be reordered as if it were pure.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::MissingEffect, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::MissingOrderedState {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "test_missing_reference_effect".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_unclassified_reference_input() {
        // Every reference operand must be classified by the operation's own semantics descriptor.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder
            .add_instruction(MalformedReferenceOperation::Unclassified, Vec::new(), vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UnclassifiedInput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "unclassified_reference".to_string(),
                input_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_out_of_range_access_index() {
        // A declared access must name an operand that the instruction actually has.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidAccess, Vec::new(), vec![reference])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidAccessIndex {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "invalid_reference_access".to_string(),
                input_index: 1,
                input_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_access_on_a_non_reference_input() {
        // A declared access must land on a reference-typed operand.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let array = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::NonReferenceAccess, Vec::new(), vec![array])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::NonReferenceAccess {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "non_reference_access".to_string(),
                input_index: 0,
                input_type: "f32[]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_fresh_root_on_a_non_reference_output() {
        // A declared fresh root must land on a reference-typed result.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let array = builder.add_input(scalar_type().into());
        let bad = builder.add_instruction(MalformedReferenceOperation::BadRoot, Vec::new(), vec![array]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![bad], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::NonReferenceOutput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "bad_reference_root".to_string(),
                output_index: 0,
                output_type: "f32[]".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_out_of_range_root_output_index() {
        // A declared root must name a result that the instruction actually has.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::InvalidOutput, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidOutputIndex {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "invalid_reference_output".to_string(),
                output_index: 1,
                output_count: 1,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_unclassified_reference_output() {
        // Every reference-typed result must be classified as either a fresh root or an alias.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        builder
            .add_instruction(MalformedReferenceOperation::UnclassifiedOutput, Vec::new(), vec![initial])
            .unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UnclassifiedOutput {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "unclassified_reference_output".to_string(),
                output_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_alias_type_mismatch() {
        // An alias must preserve its source handle's exact reference type.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
        builder.add_instruction(MalformedReferenceOperation::BadAlias, Vec::new(), vec![reference]).unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::AliasTypeMismatch {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "bad_reference_alias".to_string(),
                output_index: 0,
                input_type: "ref<f32[]>".to_string(),
                output_type: "ref<f64[]>".to_string(),
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_an_alias_of_an_unresolved_source() {
        // An alias source must already carry a resolved handle. A non-reference operand never receives one, so the
        // alias is reported as unresolved instead of silently inventing a root for the reference-typed result.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let array = builder.add_input(scalar_type().into());
        builder.add_instruction(MalformedReferenceOperation::BadAlias, Vec::new(), vec![array]).unwrap();
        let program =
            builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UnresolvedAlias {
                instruction: InstructionId::new(program.entry(), 0),
                operation: "bad_reference_alias".to_string(),
                input_index: 0,
            }),
        );
    }

    #[test]
    fn test_reference_analysis_diagnostics_render_canonical_roots() {
        // Both canonical root kinds appear verbatim in user-facing diagnostics, so their rendering is pinned here
        // alongside the surrounding message rather than only through structural variant assertions.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let reference = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let read = builder.add_instruction(ReferenceReadOperation::new(), Vec::new(), vec![reference]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, read], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        assert_eq!(
            program.analyze_array_references(0).unwrap_err().to_string(),
            "reference input 0 of `reference_read` at `^0[2]` uses consumed root `^0[0] output 0`",
        );

        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let external = builder.add_input(ReferenceType::new(scalar_type()).into());
        let frozen = builder.add_instruction(FreezeReferenceOperation::new(), Vec::new(), vec![external]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
            .unwrap();
        assert_eq!(
            program.analyze_array_references(0).unwrap_err().to_string(),
            "`freeze_reference` at `^0[0]` can consume only a local root handle, but input 0 resolves to \
             `^0 input 0`",
        );
    }

    #[test]
    fn test_reference_analysis_rejects_missing_or_invalid_input_region_provenance() {
        let mut body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let body = body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new())
            .unwrap();

        // The table enumerates the two unusable region-input provenance contracts, one declaring no parent
        // provenance at all and one naming a parent operand that is not a reference operand.
        for (operation, expected) in [
            (
                MalformedReferenceOperation::ComputationCarrier,
                ReferenceAnalysisError::UnsupportedRegionInput {
                    instruction: InstructionId::new(RegionId::new(1), 0),
                    operation: "test_computation_carrier".to_string(),
                    region_index: 0,
                    input_index: 0,
                },
            ),
            (
                MalformedReferenceOperation::InvalidSourceCarrier,
                ReferenceAnalysisError::InvalidRegionInputSource {
                    instruction: InstructionId::new(RegionId::new(1), 0),
                    operation: "test_invalid_source_carrier".to_string(),
                    region_index: 0,
                    input_index: 0,
                    source_input_index: 1,
                },
            ),
        ] {
            let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
            let body = builder.import_region(body.entry_region_ref());
            let reference = builder.add_input(ReferenceType::new(scalar_type()).into());
            builder.add_instruction(operation, vec![body], vec![reference]).unwrap();
            let program =
                builder.build::<Vec<TestValue>, Vec<TestValue>>(Vec::new(), vec![Placeholder], Vec::new()).unwrap();
            let error = program.analyze_array_references(0).unwrap_err();
            assert_eq!(error.downcast_custom::<ReferenceAnalysisError>(), Some(&expected));
        }
    }

    #[test]
    fn test_reference_analysis_resolves_an_alias_to_its_source_root() {
        // An explicit alias resolves to its source's canonical root, so accesses through either handle name one root.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let value = builder.add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_array_references(0).unwrap();
        let canonical_root =
            ReferenceRoot::Allocation { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), root)), Some(canonical_root));
        assert_eq!(analysis.root(ValueId::new(program.entry(), alias)), Some(canonical_root));
        assert_eq!(analysis.accesses()[0].root(), canonical_root);
    }

    #[test]
    fn test_reference_analysis_rejects_reading_through_an_alias_of_a_consumed_root() {
        // Consuming the root invalidates every handle in its alias family, including reads through an alias.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let frozen = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let read = builder.add_instruction(MalformedReferenceOperation::Read, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![frozen, read], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 3),
                operation: "test_reference_read".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_a_root_twice() {
        // A root can be consumed at most once.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let first = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let second = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![root]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![first, second], vec![Placeholder], vec![Placeholder; 2])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::UseAfterConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_consuming_through_an_explicit_alias() {
        // Only an unaliased root handle may be consumed; an explicit alias never grants that right.
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let alias = builder.add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![root]).unwrap()[0];
        let value = builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![alias]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_forwards_call_roots_without_granting_alias_consumption() {
        // A structured call preserves a directly forwarded root handle, but it must not turn an explicit alias back
        // into a consumable root handle.
        let mut direct_body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let direct_reference = direct_body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let direct_body = direct_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![direct_reference], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let direct_body = builder.import_region(direct_body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let forwarded = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![direct_body], vec![root])
            .unwrap()[0];
        let value =
            builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![forwarded]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let analysis = program.analyze_array_references(0).unwrap();
        let forwarded_root =
            ReferenceRoot::Allocation { instruction: InstructionId::new(program.entry(), 0), output_index: 0 };
        assert_eq!(analysis.root(ValueId::new(program.entry(), forwarded)), Some(forwarded_root));

        let mut alias_body_builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let alias_reference = alias_body_builder.add_input(ReferenceType::new(scalar_type()).into());
        let alias = alias_body_builder
            .add_instruction(MalformedReferenceOperation::Alias, Vec::new(), vec![alias_reference])
            .unwrap()[0];
        let alias_body = alias_body_builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![alias], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let mut builder = ProgramBuilder::<TestValue, MalformedReferenceOperation>::new();
        let alias_body = builder.import_region(alias_body.entry_region_ref());
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(MalformedReferenceOperation::Root, Vec::new(), vec![initial]).unwrap()[0];
        let forwarded = builder
            .add_instruction(MalformedReferenceOperation::CallCarrier, vec![alias_body], vec![root])
            .unwrap()[0];
        let value =
            builder.add_instruction(MalformedReferenceOperation::Consume, Vec::new(), vec![forwarded]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![value], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::InvalidConsume {
                instruction: InstructionId::new(program.entry(), 2),
                operation: "test_reference_consume".to_string(),
                input_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }

    #[test]
    fn test_reference_analysis_rejects_a_local_allocation_escaping_the_root_region_output() {
        // A local allocation cannot escape through the analyzed root region's own output boundary.
        let mut builder = ProgramBuilder::<TestValue, TestOperation>::new();
        let initial = builder.add_input(scalar_type().into());
        let root = builder.add_instruction(NewReferenceOperation::new(), Vec::new(), vec![initial]).unwrap()[0];
        let program = builder
            .build::<Vec<TestValue>, Vec<TestValue>>(vec![root], vec![Placeholder], vec![Placeholder])
            .unwrap();
        let error = program.analyze_array_references(0).unwrap_err();
        assert_eq!(
            error.downcast_custom::<ReferenceAnalysisError>(),
            Some(&ReferenceAnalysisError::ReferenceOutput {
                region: program.entry(),
                output_index: 0,
                root: ReferenceRoot::Allocation {
                    instruction: InstructionId::new(program.entry(), 0),
                    output_index: 0,
                },
            }),
        );
    }
}
