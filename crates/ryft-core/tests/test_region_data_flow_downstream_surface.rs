//! Downstream proof that recurrent region data flow does not require a built-in loop payload.
//!
//! The carrier uses nonstandard region slots and noncontiguous carried inputs, including feedback across two
//! computation regions. Its execution-only predicate and bypass are declared through the public semantic contract.

// TODO(eaplatanios): Review this module.

use std::sync::{Arc, Mutex};

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::{
    Array, ArrayType, CompareOperation, ComparisonDirection, Concretizable, ConstantOperation, Context, DataType,
    Domain, DotDimensionNumbers, DotOperation, InterpretableOperation, InterpretationDriver, MulOperation, NoStorage,
    Operation, OperationFormatter, OutputRegionProvenance, PartiallyEvaluatableOperation, Placeholder, Program,
    ProgramBuilder, ProgramError, RegionDataFlow, RegionDataFlowBoundary, RegionDataFlowRegionBoundary,
    RegionDataFlowRule, RegionDataFlowSource, RegionDataFlowSources, RegionInterface, RegionLiveness, RegionSlot,
    ResidualCandidate, ResidualDecision, ResidualPolicy, ResidualPolicyError, ResidualPolicyReference,
    ResidualRejection, TagOperation, TypeError,
};

/// Small downstream operation family whose recurrent carrier has no scan or while variant.
#[derive(Clone, Debug, ryft_macros::Operation)]
#[ryft(crate = "ryft_core", type(ArrayType), constant(Array))]
enum DownstreamOperation {
    Constant(ConstantOperation<Array>),
    Tag(TagOperation<ArrayType>),
    Dot(DotOperation),
    Compare(CompareOperation<ArrayType>),
    Mul(MulOperation<ArrayType>),
    Recurrent(RecurrentOperation),
}

/// Two-step recurrence whose second result reaches its dot producer only through an earlier iteration.
#[derive(Copy, Clone, Debug)]
struct RecurrentOperation {
    /// Whether the operation returns its initial state without executing any attached computation.
    bypass: bool,

    /// Whether producer and execution metadata are deliberately absent at the operation hook.
    opaque: bool,
}

impl Operation for RecurrentOperation {
    type Type = ArrayType;

    fn name(&self) -> &'static str {
        "downstream.recurrent"
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        const {
            &[
                RegionSlot::computation("second_body"),
                RegionSlot::rule("dormant"),
                RegionSlot::computation("predicate"),
                RegionSlot::computation("first_body"),
            ]
        }
    }

    fn infer_output_types(
        &self,
        input_types: &[ArrayType],
        region_interfaces: &[RegionInterface<ArrayType>],
    ) -> Result<Vec<ArrayType>, TypeError> {
        if input_types.len() != 3 || region_interfaces.len() != 4 {
            return Err(TypeError::invalid("`downstream.recurrent` requires three inputs and four regions"));
        }
        let scalar = ArrayType::scalar(DataType::F64);
        if input_types != [scalar.clone(), scalar.clone(), scalar.clone()] {
            return Err(TypeError::invalid("`downstream.recurrent` inputs must be `f64` scalars"));
        }
        for (region_index, region) in region_interfaces.iter().enumerate() {
            if region.input_types() != input_types {
                return Err(TypeError::invalid("`downstream.recurrent` region inputs must be three `f64` scalars"));
            }
            let expected = match region_index {
                0 | 3 => vec![scalar.clone(), scalar.clone()],
                1 => vec![scalar.clone()],
                2 => vec![ArrayType::scalar(DataType::Boolean)],
                _ => unreachable!(),
            };
            if region.output_types() != expected {
                return Err(TypeError::invalid("`downstream.recurrent` region outputs have invalid types"));
            }
        }
        Ok(vec![scalar.clone(), scalar])
    }

    fn region_data_flow(&self) -> RegionDataFlow<'_> {
        if self.opaque { RegionDataFlow::Opaque } else { RegionDataFlow::Custom(self) }
    }

    fn render(&self, formatter: &mut std::fmt::Formatter<'_>, indentation: usize) -> std::fmt::Result {
        OperationFormatter::new(formatter, indentation, self.name())?
            .bracketed(|operation| operation.field("bypass", self.bypass))
    }
}

impl RegionDataFlowRule for RecurrentOperation {
    fn input_sources(
        &self,
        region_index: usize,
        input_index: usize,
        _boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        use RegionDataFlowSource::{InstructionInput, RegionOutput};

        let sources = match (region_index, input_index) {
            (3, 0) | (0, 1) => vec![InstructionInput(1)],
            (3, 1) => {
                vec![InstructionInput(0), RegionOutput(OutputRegionProvenance { region_index: 0, output_index: 0 })]
            }
            (3, 2) => {
                vec![InstructionInput(2), RegionOutput(OutputRegionProvenance { region_index: 0, output_index: 1 })]
            }
            (0, 0) => vec![RegionOutput(OutputRegionProvenance { region_index: 3, output_index: 0 })],
            (0, 2) => vec![RegionOutput(OutputRegionProvenance { region_index: 3, output_index: 1 })],
            (2, 0) => {
                vec![InstructionInput(0), RegionOutput(OutputRegionProvenance { region_index: 0, output_index: 0 })]
            }
            (2, 1) => vec![InstructionInput(1)],
            (2, 2) => {
                vec![InstructionInput(2), RegionOutput(OutputRegionProvenance { region_index: 0, output_index: 1 })]
            }
            _ => return Ok(RegionDataFlowSources::Unknown),
        };
        Ok(RegionDataFlowSources::Known(sources))
    }

    fn output_sources(
        &self,
        output_index: usize,
        _boundary: RegionDataFlowBoundary<'_>,
    ) -> Result<RegionDataFlowSources, ProgramError> {
        Ok(RegionDataFlowSources::Known(vec![if self.bypass {
            RegionDataFlowSource::InstructionInput(output_index * 2)
        } else {
            RegionDataFlowSource::RegionInput { region_index: 3, input_index: output_index + 1 }
        }]))
    }

    fn execution_demands(
        &self,
        _used_outputs: &[bool],
        _boundary: RegionDataFlowBoundary<'_>,
        _regions: &mut dyn RegionLiveness,
    ) -> Result<Vec<Option<Vec<bool>>>, ProgramError> {
        // Both state fields remain on this carrier's runtime boundary. The predicate executes without supplying a
        // result, whereas the dormant derivative rule never executes. A bypass executes none of these regions.
        Ok(if self.bypass {
            vec![None, None, None, None]
        } else {
            vec![Some(vec![true, true]), None, Some(vec![true]), Some(vec![true, true])]
        })
    }
}

impl<C> InterpretableOperation<C> for RecurrentOperation
where
    C: Domain<Type = ArrayType, Value: Concretizable<bool>>,
{
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        context: &C,
        driver: &D,
        inputs: &[C::Value],
    ) -> Result<Vec<C::Value>, ProgramError> {
        let mut first = inputs[0].clone();
        let mut second = inputs[2].clone();
        if !self.bypass {
            for _ in 0..2 {
                let predicate =
                    driver.interpret_region(context, 2, vec![first.clone(), inputs[1].clone(), second.clone()])?;
                if !predicate[0].concretize()? {
                    break;
                }
                let intermediate = driver.interpret_region(context, 3, vec![inputs[1].clone(), first, second])?;
                let result = driver.interpret_region(
                    context,
                    0,
                    vec![intermediate[0].clone(), inputs[1].clone(), intermediate[1].clone()],
                )?;
                first = result[0].clone();
                second = result[1].clone();
            }
        }
        Ok(vec![first, second])
    }
}

impl<C: Context<Type = ArrayType, Operation: From<RecurrentOperation>>> PartiallyEvaluatableOperation<C>
    for RecurrentOperation
{
}

/// Records producer order and optionally rejects a producer by its tag or operation name.
struct RecordingPolicy {
    /// Ordered producer lists of the residuals classified so far.
    producers: Arc<Mutex<Vec<Vec<String>>>>,

    /// Producer forbidden by this policy, or none when every residual is saved.
    rejected: Option<&'static str>,
}

impl ResidualPolicy<ArrayType> for RecordingPolicy {
    type Storage = NoStorage;

    fn name(&self) -> &str {
        "downstream_policy"
    }

    fn classify(
        &self,
        candidate: &ResidualCandidate<'_, ArrayType>,
    ) -> Result<ResidualDecision<NoStorage>, ResidualRejection> {
        let producers = candidate
            .producers()
            .iter()
            .map(|producer| {
                producer
                    .payload::<TagOperation<ArrayType>>()
                    .map_or_else(|| producer.name().to_owned(), |tag| tag.key().to_owned())
            })
            .collect::<Vec<_>>();
        self.producers.lock().expect("producer recording mutex poisoned").push(producers.clone());
        if self.rejected.is_some_and(|rejected| producers.iter().any(|producer| producer == rejected)) {
            Err(ResidualRejection::new("the declared producer cannot be replayed"))
        } else if self.rejected.is_some() {
            Ok(ResidualDecision::Recompute)
        } else {
            Ok(ResidualDecision::Save)
        }
    }
}

/// Public downstream program family used for execution and residual-placement assertions.
type DownstreamProgram = Program<Array, DownstreamOperation, Vec<Array>, Vec<Array>>;

/// Builds the custom carrier followed by a multiplication that needs its second result as a residual.
fn recurrent_program(operation: RecurrentOperation) -> DownstreamProgram {
    let scalar = ArrayType::scalar(DataType::F64);
    let mut first_body = ProgramBuilder::<Array, DownstreamOperation>::new();
    first_body.add_input(scalar.clone());
    let first = first_body.add_input(scalar.clone());
    let second = first_body.add_input(scalar.clone());
    let first_body = first_body
        .build::<Vec<Array>, Vec<Array>>(vec![second, first], vec![Placeholder; 3], vec![Placeholder; 2])
        .unwrap();

    let mut second_body = ProgramBuilder::<Array, DownstreamOperation>::new();
    second_body.add_input(scalar.clone());
    second_body.add_input(scalar.clone());
    let first = second_body.add_input(scalar.clone());
    let dot = second_body
        .add_instruction(
            DotOperation::new(DotDimensionNumbers::new(vec![], vec![], vec![], vec![])),
            Vec::new(),
            vec![first, first],
            None,
        )
        .unwrap()[0];
    let second_body = second_body
        .build::<Vec<Array>, Vec<Array>>(vec![dot, first], vec![Placeholder; 3], vec![Placeholder; 2])
        .unwrap();

    let mut predicate = ProgramBuilder::<Array, DownstreamOperation>::new();
    let first = predicate.add_input(scalar.clone());
    let threshold = predicate.add_input(scalar.clone());
    predicate.add_input(scalar.clone());
    let first = predicate
        .add_instruction(TagOperation::<ArrayType>::new("predicate_only"), Vec::new(), vec![first], None)
        .unwrap()[0];
    let output = predicate
        .add_instruction(
            CompareOperation::<ArrayType>::new(ComparisonDirection::LessThan),
            Vec::new(),
            vec![first, threshold],
            None,
        )
        .unwrap()[0];
    let predicate = predicate
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let mut dormant = ProgramBuilder::<Array, DownstreamOperation>::new();
    let first = dormant.add_input(scalar.clone());
    dormant.add_input(scalar.clone());
    dormant.add_input(scalar.clone());
    let output = dormant
        .add_instruction(TagOperation::<ArrayType>::new("dormant"), Vec::new(), vec![first], None)
        .unwrap()[0];
    let dormant = dormant
        .build::<Vec<Array>, Vec<Array>>(vec![output], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let mut builder = ProgramBuilder::<Array, DownstreamOperation>::new();
    let first = builder.add_input(scalar.clone());
    let threshold = builder.add_input(scalar.clone());
    let second = builder.add_input(scalar.clone());
    let unknown = builder.add_input(scalar);
    let first = builder
        .add_instruction(TagOperation::<ArrayType>::new("first"), Vec::new(), vec![first], None)
        .unwrap()[0];
    let second = builder
        .add_instruction(TagOperation::<ArrayType>::new("second"), Vec::new(), vec![second], None)
        .unwrap()[0];
    let second_body = builder.import_program(second_body);
    let dormant = builder.import_program(dormant);
    let predicate = builder.import_program(predicate);
    let first_body = builder.import_program(first_body);
    let outputs = builder
        .add_instruction(
            operation,
            vec![second_body, dormant, predicate, first_body],
            vec![first, threshold, second],
            None,
        )
        .unwrap()
        .to_vec();
    let output = builder
        .add_instruction(MulOperation::<ArrayType>::new(), Vec::new(), vec![outputs[1], unknown], None)
        .unwrap()[0];
    builder.build(vec![output], vec![Placeholder; 4], vec![Placeholder]).unwrap()
}

/// Pins the saved recurrence whose second output is the only residual edge.
fn check_saved_program(program: &DownstreamProgram) {
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[] .
            let %3:f64[] = tag [key=first] %0
                %4:f64[] = tag [key=second] %2
                %5:f64[], %6:f64[] = downstream.recurrent [bypass=false] %3 %1 %4 [
                    second_body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = dot [
                            dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                        ] %2 %2
                        in (%3, %2)
                    },
                    dormant={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = tag [key=dormant] %0
                        in (%3)
                    },
                    predicate={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = tag [key=predicate_only] %0
                            %4:bool[] = compare [direction=LessThan] %3 %1
                        in (%4)
                    },
                    first_body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        in (%2, %1)
                    },
                ]
            in (%6)"},
    );
}

/// Pins the entire runtime recurrence, including its attached dormant rule, when all work is recomputed.
fn check_recomputed_program(program: &DownstreamProgram, bypass: bool) {
    assert_eq!(
        program.to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[], %3:f64[] .
            let %4:f64[] = tag [key=first] %1
                %5:f64[] = tag [key=second] %3
                %6:f64[], %7:f64[] = downstream.recurrent [bypass=false] %4 %2 %5 [
                    second_body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = dot [
                            dimensions=(lhs_contracting=[], rhs_contracting=[], lhs_batching=[], rhs_batching=[]),
                        ] %2 %2
                        in (%3, %2)
                    },
                    dormant={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = tag [key=dormant] %0
                        in (%3)
                    },
                    predicate={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        let %3:f64[] = tag [key=predicate_only] %0
                            %4:bool[] = compare [direction=LessThan] %3 %1
                        in (%4)
                    },
                    first_body={
                        lambda %0:f64[], %1:f64[], %2:f64[] .
                        in (%2, %1)
                    },
                ]
                %8:f64[] = mul %7 %0
            in (%8)"}
        .replace("bypass=false", if bypass { "bypass=true" } else { "bypass=false" }),
    );
}

#[test]
fn test_downstream_region_data_flow() {
    /// This carrier's state boundary is fixed, so computing its demand does not inspect body liveness.
    struct UnusedLiveness;

    impl RegionLiveness for UnusedLiveness {
        fn used_region_inputs(
            &mut self,
            _region_index: usize,
            _used_outputs: &[bool],
        ) -> Result<Vec<bool>, ProgramError> {
            panic!("the custom carrier does not query body liveness")
        }
    }

    let boundaries = [
        RegionDataFlowRegionBoundary { input_count: 3, output_count: 2 },
        RegionDataFlowRegionBoundary { input_count: 3, output_count: 1 },
        RegionDataFlowRegionBoundary { input_count: 3, output_count: 1 },
        RegionDataFlowRegionBoundary { input_count: 3, output_count: 2 },
    ];
    let boundary = RegionDataFlowBoundary { input_count: 3, output_count: 2, regions: &boundaries };
    let operation = DownstreamOperation::Recurrent(RecurrentOperation { bypass: false, opaque: false });
    let data_flow = operation.region_data_flow();
    assert_eq!(
        data_flow.output_sources(&operation, 1, boundary),
        Ok(RegionDataFlowSources::Known(vec![RegionDataFlowSource::RegionInput { region_index: 3, input_index: 2 }])),
    );
    assert_eq!(
        data_flow.input_sources(&operation, 3, 1, boundary),
        Ok(RegionDataFlowSources::Known(vec![
            RegionDataFlowSource::InstructionInput(0),
            RegionDataFlowSource::RegionOutput(OutputRegionProvenance { region_index: 0, output_index: 0 }),
        ])),
    );
    assert_eq!(
        data_flow.execution_demands(&operation, &[false, true], boundary, &mut UnusedLiveness),
        Ok(vec![Some(vec![true, true]), None, Some(vec![true]), Some(vec![true, true])]),
    );
    let operation = DownstreamOperation::Recurrent(RecurrentOperation { bypass: true, opaque: false });
    let data_flow = operation.region_data_flow();
    assert_eq!(
        data_flow.output_sources(&operation, 1, boundary),
        Ok(RegionDataFlowSources::Known(vec![RegionDataFlowSource::InstructionInput(2)])),
    );
    assert_eq!(
        data_flow.execution_demands(&operation, &[false, true], boundary, &mut UnusedLiveness),
        Ok(vec![None, None, None, None]),
    );
}

#[test]
fn test_downstream_region_data_flow_interpretation() {
    let program = recurrent_program(RecurrentOperation { bypass: false, opaque: false });
    assert_eq!(
        program.interpret(vec![
            Array::scalar(3f64).unwrap(),
            Array::scalar(100f64).unwrap(),
            Array::scalar(5f64).unwrap(),
            Array::scalar(2f64).unwrap(),
        ]),
        Ok(vec![Array::scalar(18f64).unwrap()]),
    );
    // Even the executing carrier can return its initial state when the predicate stops it before the first round.
    assert_eq!(
        program.interpret(vec![
            Array::scalar(3f64).unwrap(),
            Array::scalar(2f64).unwrap(),
            Array::scalar(5f64).unwrap(),
            Array::scalar(2f64).unwrap(),
        ]),
        Ok(vec![Array::scalar(10f64).unwrap()]),
    );
}

#[test]
fn test_downstream_region_data_flow_residual_producers() {
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy = ResidualPolicyReference::new(RecordingPolicy { producers: producers.clone(), rejected: None });
    let partition = recurrent_program(RecurrentOperation { bypass: false, opaque: false })
        .partition(&[true, true, true, false])
        .unwrap();
    let placed = partition.with_residual_policy(&policy).unwrap();
    assert_eq!(*producers.lock().unwrap(), vec![vec!["second".to_owned(), "first".to_owned(), "dot".to_owned()]]);
    check_saved_program(placed.known_program());
    assert_eq!(
        placed.residual_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = mul %1 %0
            in (%2)"},
    );
    let known = placed
        .known_program()
        .interpret(vec![Array::scalar(3f64).unwrap(), Array::scalar(100f64).unwrap(), Array::scalar(5f64).unwrap()])
        .unwrap();
    assert_eq!(known, vec![Array::scalar(9f64).unwrap()]);
    assert_eq!(
        placed.residual_program().interpret(vec![Array::scalar(2f64).unwrap(), known[0].clone()]),
        Ok(vec![Array::scalar(18f64).unwrap()]),
    );
}

#[test]
fn test_downstream_region_data_flow_residual_producers_rejects_cross_region_feedback() {
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy = ResidualPolicyReference::new(RecordingPolicy { producers: producers.clone(), rejected: Some("dot") });
    assert_eq!(
        recurrent_program(RecurrentOperation { bypass: false, opaque: false })
            .partition(&[true, true, true, false])
            .unwrap()
            .with_residual_policy(&policy)
            .map(|_| ()),
        Err(ResidualPolicyError::Rejected {
            policy: "downstream_policy".to_owned(),
            rejection: ResidualRejection::new("the declared producer cannot be replayed"),
        }),
    );
    assert_eq!(*producers.lock().unwrap(), vec![vec!["second".to_owned(), "first".to_owned(), "dot".to_owned()]]);
}

#[test]
fn test_downstream_region_data_flow_residual_producers_bypass() {
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy = ResidualPolicyReference::new(RecordingPolicy { producers: producers.clone(), rejected: Some("dot") });
    let program = recurrent_program(RecurrentOperation { bypass: true, opaque: false });
    assert_eq!(
        program.interpret(vec![
            Array::scalar(3f64).unwrap(),
            Array::scalar(100f64).unwrap(),
            Array::scalar(5f64).unwrap(),
            Array::scalar(2f64).unwrap(),
        ]),
        Ok(vec![Array::scalar(10f64).unwrap()]),
    );
    let placed = program.partition(&[true, true, true, false]).unwrap().with_residual_policy(&policy).unwrap();
    assert_eq!(
        *producers.lock().unwrap(),
        vec![vec!["second".to_owned()], vec!["first".to_owned()], vec!["second".to_owned()]],
    );
    assert_eq!(
        placed.known_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[] .
            in (%0, %1, %2)"},
    );
    check_recomputed_program(placed.residual_program(), true);
}

#[test]
fn test_downstream_region_data_flow_residual_producers_opaque() {
    let operation = RecurrentOperation { bypass: false, opaque: true };
    assert!(matches!(DownstreamOperation::Recurrent(operation).region_data_flow(), RegionDataFlow::Opaque));
    let program = recurrent_program(operation);
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy = ResidualPolicyReference::new(RecordingPolicy { producers: producers.clone(), rejected: Some("dot") });

    // Explicit whole-operation replay can recompute an opaque carrier. Its undeclared internals do not contribute
    // producer candidates, so rejecting a body dot cannot manufacture a declaration the operation never supplied.
    let placed = program.partition(&[true, true, true, false]).unwrap().with_residual_policy(&policy).unwrap();
    assert_eq!(
        *producers.lock().unwrap(),
        vec![vec!["downstream.recurrent".to_owned()], vec!["first".to_owned()], vec!["second".to_owned()]],
    );
    assert_eq!(
        placed.known_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[] .
            in (%0, %1, %2)"},
    );
    check_recomputed_program(placed.residual_program(), false);

    // Split-preserving placement cannot approve replay from missing semantics, so the same policy saves the
    // opaque carrier's output and leaves its interior policy decisions intact.
    producers.lock().unwrap().clear();
    let placed = program.partition_with_residual_policy(&[true, true, true, false], &policy).unwrap();
    assert_eq!(*producers.lock().unwrap(), vec![vec!["downstream.recurrent".to_owned()]]);
    check_saved_program(placed.known_program());
    assert_eq!(
        placed.residual_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[] .
            let %2:f64[] = mul %1 %0
            in (%2)"},
    );
}

#[test]
fn test_downstream_region_data_flow_execution_demands_rejects_predicate() {
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy = ResidualPolicyReference::new(RecordingPolicy {
        producers: producers.clone(),
        rejected: Some("predicate_only"),
    });
    assert_eq!(
        recurrent_program(RecurrentOperation { bypass: false, opaque: false })
            .partition_with_residual_policy(&[true, true, true, false], &policy)
            .map(|_| ()),
        Err(ProgramError::from(ResidualPolicyError::Rejected {
            policy: "downstream_policy".to_owned(),
            rejection: ResidualRejection::new("the declared producer cannot be replayed"),
        })),
    );
    assert_eq!(
        *producers.lock().unwrap(),
        vec![
            vec!["second".to_owned(), "first".to_owned(), "dot".to_owned()],
            vec!["dot".to_owned()],
            vec!["predicate_only".to_owned()],
        ],
    );
}

#[test]
fn test_downstream_region_data_flow_execution_demands_ignores_dormant_rule() {
    let producers = Arc::new(Mutex::new(Vec::new()));
    let policy =
        ResidualPolicyReference::new(RecordingPolicy { producers: producers.clone(), rejected: Some("dormant") });
    let placed = recurrent_program(RecurrentOperation { bypass: false, opaque: false })
        .partition_with_residual_policy(&[true, true, true, false], &policy)
        .unwrap();
    assert_eq!(
        *producers.lock().unwrap(),
        vec![
            vec!["second".to_owned(), "first".to_owned(), "dot".to_owned()],
            vec!["dot".to_owned()],
            vec!["predicate_only".to_owned()],
            vec!["compare".to_owned()],
            vec!["first".to_owned()],
            vec!["second".to_owned()],
        ],
    );
    assert_eq!(
        placed.known_program().to_string(),
        indoc! {"
            lambda %0:f64[], %1:f64[], %2:f64[] .
            in (%0, %1, %2)"},
    );
    check_recomputed_program(placed.residual_program(), false);
}
