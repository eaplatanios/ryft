//! Downstream compile proof for the reference discharge extension surface.
//!
//! Everything in this file is written from the position of a backend crate outside `ryft-core`: a reference universe
//! of its own, a [`ReferenceDischargePolicy`] for that universe, and per-operation
//! [`ReferenceDischargeableOperation`] rules, all reaching `ryft-core` only through its public API. Because an
//! integration test is a separate crate, the compiler itself enforces the property this file exists to establish,
//! namely that a third-party reference universe can be discharged without any private `ryft-core` item.
//!
//! The universe is deliberately view-less, with a unit alias, which complements the composed-view universe covered by
//! the in-crate tests: together they pin both ends of the alias contract. It is also deliberately non-accumulating,
//! which makes it the standing proof of the policy's per-access capability granularity: it implements
//! [`ReferenceDischargePolicy`] and not [`ReferenceAccumulationPolicy`](ryft_core::ReferenceAccumulationPolicy), and
//! still discharges every program that reads and replaces. Only a program containing `reference_add_update` would
//! fail to discharge for it, and it would fail at compile time, scoped to exactly that operation.

use std::borrow::Cow;
use std::collections::BTreeSet;
use std::fmt::Display;

use indoc::indoc;
use pretty_assertions::assert_eq;

use ryft_core::macros::check_count;
use ryft_core::{
    Context, Domain, EagerContext, Effect, Effects, InterpretableOperation, InterpretationDriver, NoIdentity,
    Operation, OutputRegionProvenance, Parameter, Placeholder, Program, ProgramBuilder, ProgramError,
    RecursiveReferenceDischargeDriver, ReferenceAccessMode, ReferenceDischargeContext, ReferenceDischargeDriver,
    ReferenceDischargePolicy, ReferenceDischargeSite, ReferenceDischargeValue, ReferenceDischargeableOperation,
    ReferenceInput, ReferenceOperationSemantics, ReferenceOutput, ReferenceRegionDischargeBoundary,
    ReferenceRegionStateInsertion, ReferenceSource, ReferenceStateBinding, ReferenceType, RegionInterface, RegionSlot,
    Trace, Tracer, TracingContext, Type, TypeError, Typed, Value, discharge_reference_free_operation,
};

/// Destination universe of the downstream programs. Its dispatch domain is the constant-only eager context, which is
/// what a concrete backend value family looks like from outside `ryft-core`, and consequently [`RegisterValue`]
/// implements none of the operation-backed value capabilities.
type RegisterDestination = EagerContext<RegisterValue, RegisterOperation>;

/// Discharge context over the downstream destination universe.
type RegisterDischargeContext = ReferenceDischargeContext<RegisterDestination, RegisterReferenceDischarge>;

/// Carrier flowing through downstream discharge.
type RegisterDischargeValue = ReferenceDischargeValue<RegisterDestination, RegisterReferenceDischarge>;

/// Referent type of the downstream universe: one 64-bit integer register.
#[derive(Clone, Debug, PartialEq)]
struct RegisterType;

impl Display for RegisterType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("register")
    }
}

impl Parameter for RegisterType {}

impl Type for RegisterType {
    type Identity = NoIdentity;
    type Refinements = ();

    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn is_refined_by(&self, other: &Self) -> bool {
        self == other
    }

    fn is_scalar(&self) -> bool {
        true
    }

    fn is_complex(&self) -> bool {
        false
    }
}

/// Type universe of the downstream programs.
#[derive(Clone, Debug, PartialEq)]
enum RegisterIrType {
    Register(RegisterType),
    Reference(ReferenceType<RegisterType>),
}

impl Display for RegisterIrType {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Register(r#type) => Display::fmt(r#type, formatter),
            Self::Reference(r#type) => Display::fmt(r#type, formatter),
        }
    }
}

impl Parameter for RegisterIrType {}

impl Type for RegisterIrType {
    type Identity = NoIdentity;
    type Refinements = ();

    fn is_compatible_with(&self, other: &Self) -> bool {
        self == other
    }

    fn is_refined_by(&self, other: &Self) -> bool {
        self == other
    }

    fn is_scalar(&self) -> bool {
        matches!(self, Self::Register(_))
    }

    fn is_complex(&self) -> bool {
        false
    }

    fn is_reference(&self) -> bool {
        matches!(self, Self::Reference(_))
    }
}

/// Value universe of the downstream programs.
#[derive(Copy, Clone, Debug, PartialEq)]
struct RegisterValue(i64);

impl Display for RegisterValue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&self.0, formatter)
    }
}

impl Parameter for RegisterValue {}

impl Typed for RegisterValue {
    type Type = RegisterIrType;

    fn r#type(&self) -> Cow<'_, RegisterIrType> {
        Cow::Owned(RegisterIrType::Register(RegisterType))
    }
}

impl Value for RegisterValue {
    type DispatchDomain = EagerContext<Self>;
    type ExecutionDomain = EagerContext<Self>;

    fn dispatch_domain(&self) -> Self::DispatchDomain {
        EagerContext::new()
    }

    fn execution_domain(&self) -> Self::ExecutionDomain {
        EagerContext::new()
    }
}

/// View chain of the downstream universe. Registers have no interior structure, so every handle denotes its complete
/// root and the alias carries nothing.
#[derive(Copy, Clone, Debug, PartialEq)]
struct WholeRegister;

impl Parameter for WholeRegister {}

/// Reference discharge policy of the downstream universe.
#[derive(Copy, Clone, Debug)]
struct RegisterReferenceDischarge;

// The policy is generic over the destination value rather than pinned to `RegisterValue`, which is what lets one
// implementation serve an eager destination and a staging destination alike. A view-less universe needs no
// destination capability at all for reads and replacements, and this one declines accumulation entirely by not
// implementing `ReferenceAccumulationPolicy`.
impl<C: Domain<Type = RegisterIrType>> ReferenceDischargePolicy<C> for RegisterReferenceDischarge {
    type Referent = RegisterType;
    type Alias = WholeRegister;

    fn root_alias(_referent: &RegisterType) -> WholeRegister {
        WholeRegister
    }

    fn lift_reference_type(r#type: ReferenceType<RegisterType>) -> RegisterIrType {
        RegisterIrType::Reference(r#type)
    }

    fn lift_referent_type(referent: RegisterType) -> RegisterIrType {
        RegisterIrType::Register(referent)
    }

    fn project_reference_type(r#type: &RegisterIrType) -> Option<ReferenceType<RegisterType>> {
        match r#type {
            RegisterIrType::Reference(reference) => Some(reference.clone()),
            RegisterIrType::Register(_) => None,
        }
    }

    fn read(_context: &C, current: &C::Value, _alias: &WholeRegister) -> Result<C::Value, ProgramError> {
        Ok(current.clone())
    }

    fn write(
        _context: &C,
        _current: &C::Value,
        replacement: C::Value,
        _alias: &WholeRegister,
    ) -> Result<C::Value, ProgramError> {
        Ok(replacement)
    }

    fn replace(
        _context: &C,
        current: &C::Value,
        replacement: C::Value,
        _alias: &WholeRegister,
    ) -> Result<(C::Value, C::Value), ProgramError> {
        Ok((current.clone(), replacement))
    }
}

/// Operation family of the downstream universe.
#[derive(Copy, Clone, Debug)]
enum RegisterOperation {
    Negate,
    ReferenceNew,
    Read,
    Write,
    Swap,
    Freeze,
    Call,
}

impl Display for RegisterOperation {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.name())
    }
}

impl Operation for RegisterOperation {
    type Type = RegisterIrType;

    fn name(&self) -> &'static str {
        match self {
            Self::Negate => "register.negate",
            Self::ReferenceNew => "register.reference_new",
            Self::Read => "register.read",
            Self::Write => "register.write",
            Self::Swap => "register.swap",
            Self::Freeze => "register.freeze",
            Self::Call => "register.call",
        }
    }

    fn region_slots(&self) -> &'static [RegionSlot] {
        match self {
            Self::Call => const { &[RegionSlot::computation("callee")] },
            _ => &[],
        }
    }

    fn output_region_provenance(&self, output_index: usize) -> Vec<OutputRegionProvenance> {
        match self {
            Self::Call => vec![OutputRegionProvenance { region_index: 0, output_index }],
            _ => Vec::new(),
        }
    }

    fn allows_reference_access_through_region_input(&self, region_index: usize, mode: ReferenceAccessMode) -> bool {
        matches!(self, Self::Call) && region_index == 0 && !mode.is_consuming()
    }

    fn infer_output_types(
        &self,
        input_types: &[RegisterIrType],
        region_interfaces: &[RegionInterface<RegisterIrType>],
    ) -> Result<Vec<RegisterIrType>, TypeError> {
        let referent = || match input_types.first() {
            Some(RegisterIrType::Reference(reference)) => Ok(reference.referent().clone()),
            _ => Err(TypeError::invalid(format!("`{}` expects a reference operand", self.name()))),
        };
        match self {
            Self::Negate => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![RegisterIrType::Register(RegisterType)])
            }
            Self::ReferenceNew => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![RegisterIrType::Reference(ReferenceType::new(RegisterType))])
            }
            Self::Read | Self::Freeze => {
                check_count!("input", input_types, 1, TypeError);
                Ok(vec![RegisterIrType::Register(referent()?)])
            }
            Self::Write => {
                check_count!("input", input_types, 2, TypeError);
                referent()?;
                Ok(Vec::new())
            }
            Self::Swap => {
                check_count!("input", input_types, 2, TypeError);
                Ok(vec![RegisterIrType::Register(referent()?)])
            }
            Self::Call => match region_interfaces.first() {
                Some(interface) => Ok(interface.output_types().to_vec()),
                None => Err(TypeError::invalid("`register.call` expects one callee region")),
            },
        }
    }

    fn reference_semantics(&self) -> Cow<'_, ReferenceOperationSemantics> {
        match self {
            Self::Negate => Cow::Borrowed(ReferenceOperationSemantics::empty()),
            Self::ReferenceNew => Cow::Owned(ReferenceOperationSemantics::new(
                Vec::new(),
                vec![ReferenceOutput::Root { output_index: 0 }],
            )),
            Self::Read => Cow::Owned(ReferenceOperationSemantics::new(
                vec![ReferenceInput::new(0, ReferenceAccessMode::Read)],
                Vec::new(),
            )),
            Self::Write => Cow::Owned(ReferenceOperationSemantics::new(
                vec![ReferenceInput::new(0, ReferenceAccessMode::Write)],
                Vec::new(),
            )),
            Self::Swap => Cow::Owned(ReferenceOperationSemantics::new(
                vec![ReferenceInput::new(0, ReferenceAccessMode::ReadWrite)],
                Vec::new(),
            )),
            Self::Freeze => Cow::Owned(ReferenceOperationSemantics::new(
                vec![ReferenceInput::new(0, ReferenceAccessMode::Consume)],
                Vec::new(),
            )),
            // A structured operation declares no operation-local reference semantics: its accesses are summarized
            // transitively from the region closure it attaches.
            Self::Call => Cow::Borrowed(ReferenceOperationSemantics::empty()),
        }
    }

    fn effects(&self) -> Effects {
        match self {
            Self::Negate => Effects::PURE,
            _ => Effects::single(Effect::OrderedState),
        }
    }
}

impl<C: Domain<Type = RegisterIrType, Value = RegisterValue>> InterpretableOperation<C> for RegisterOperation {
    fn interpret<D: InterpretationDriver<C>>(
        &self,
        _context: &C,
        _driver: &D,
        inputs: &[RegisterValue],
    ) -> Result<Vec<RegisterValue>, ProgramError> {
        match self {
            Self::Negate => {
                check_count!("input", inputs, 1, ProgramError);
                Ok(vec![RegisterValue(-inputs[0].0)])
            }
            _ => Err(ProgramError::UnsupportedOperation {
                message: format!("`{}` must be discharged before interpretation", self.name()),
            }),
        }
    }
}

impl<C> ReferenceDischargeableOperation<C, RegisterReferenceDischarge> for RegisterOperation
where
    C: Context<Type = RegisterIrType, Operation: From<RegisterOperation>>,
{
    fn discharge_references<D: ReferenceDischargeDriver<C, RegisterReferenceDischarge>>(
        &self,
        context: &ReferenceDischargeContext<C, RegisterReferenceDischarge>,
        driver: &D,
        inputs: &[ReferenceDischargeValue<C, RegisterReferenceDischarge>],
    ) -> Result<Vec<ReferenceDischargeValue<C, RegisterReferenceDischarge>>, ProgramError> {
        // Access arms see only discharged roots: the dispatch path replays accesses to preserved roots verbatim
        // before any rule runs, so only the allocation arm still distinguishes selected from preserved.
        match self {
            Self::Negate => discharge_reference_free_operation(self, context, driver, inputs),
            Self::ReferenceNew => {
                check_count!("input", inputs, 1, ProgramError);
                let initial = inputs[0].expect_ordinary("an initial state")?.clone();
                if context.selects_allocation(driver.instruction(), 0) {
                    return Ok(vec![context.allocate_discharged(ReferenceType::new(RegisterType), initial)?]);
                }
                let mut outputs = context.parent().bind(*self, Vec::new(), std::slice::from_ref(&initial))?;
                check_count!("output", outputs, 1, ProgramError);
                Ok(vec![context.bind_preserved(ReferenceType::new(RegisterType), outputs.remove(0))?])
            }
            Self::Read => {
                check_count!("input", inputs, 1, ProgramError);
                let reference = inputs[0].expect_reference("a reference to read")?;
                Ok(vec![ReferenceDischargeValue::Ordinary(context.read(reference)?)])
            }
            Self::Write => {
                check_count!("input", inputs, 2, ProgramError);
                let reference = inputs[0].expect_reference("a reference to write")?;
                let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                context.write(reference, replacement)?;
                Ok(Vec::new())
            }
            Self::Swap => {
                check_count!("input", inputs, 2, ProgramError);
                let reference = inputs[0].expect_reference("a reference to replace")?;
                let replacement = inputs[1].expect_ordinary("a replacement value")?.clone();
                Ok(vec![ReferenceDischargeValue::Ordinary(context.replace(reference, replacement)?)])
            }
            Self::Freeze => {
                check_count!("input", inputs, 1, ProgramError);
                let reference = inputs[0].expect_reference("a reference to freeze")?;
                Ok(vec![ReferenceDischargeValue::Ordinary(context.consume(reference)?)])
            }
            // The hand-rolled structured widening a backend-owned region operation performs: summarize the closure,
            // widen the boundary with the reached state, rebuild the region in an isolated fork, validate the fork
            // against the summary's predictions, and merge every published successor state back. This is the same
            // shape a future kernel-call rule needs, expressed purely through the public discharge surface.
            Self::Call => {
                let region = driver.region(0)?;
                check_count!("input", region.input_ids(), inputs.len(), ProgramError);
                let mut declared = Vec::with_capacity(inputs.len());
                for input in inputs {
                    declared.push(context.operand_root(input, self.name())?);
                }
                let summary = context.region_summary(self, 0, region, declared.as_slice())?;
                if summary.output_roots().iter().any(Option::is_some) {
                    return Err(ProgramError::MalformedProgram(format!(
                        "`{}` does not return references from its callee",
                        self.name(),
                    )));
                }
                let operand_roots = declared.iter().copied().flatten().collect::<BTreeSet<_>>();
                let widening = context.state_widening(&summary, &operand_roots, self.name())?;
                let entering = widening.entering().to_vec();
                let source_output_count = region.output_ids().len();

                let fork = driver.discharge_region_program(
                    context,
                    0,
                    &ReferenceRegionDischargeBoundary::new(
                        self,
                        0,
                        declared,
                        ReferenceRegionStateInsertion::new(entering.clone(), inputs.len()),
                        ReferenceRegionStateInsertion::new(widening.published().to_vec(), source_output_count),
                    ),
                )?;
                fork.validate_predicted_mutations(widening.published(), self.name())?;
                fork.validate_predicted_output_roots(summary.output_roots(), self.name())?;

                let mut operands = Vec::with_capacity(inputs.len() + entering.len());
                for input in inputs {
                    operands.push(context.operand_value(input)?);
                }
                for root in &entering {
                    operands.push(context.discharged_state(*root)?);
                }
                let outputs = context.parent().bind(*self, vec![fork.into_program()], operands.as_slice())?;
                check_count!("output", outputs, source_output_count + widening.published().len(), ProgramError);

                let mut results = Vec::with_capacity(source_output_count);
                for (position, output) in outputs.into_iter().enumerate() {
                    if position < source_output_count {
                        results.push(ReferenceDischargeValue::Ordinary(output));
                    } else {
                        let root = widening.published()[position - source_output_count];
                        context.merge_boundary_state(&summary, widening.threaded(), root, output)?;
                    }
                }
                Ok(results)
            }
        }
    }
}

#[test]
fn test_downstream_reference_universe_discharges_through_the_public_surface() {
    // `f(initial, replacement) = (replaced value, frozen final state)`, written entirely in a reference universe that
    // `ryft-core` knows nothing about.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = builder.add_input(RegisterIrType::Register(RegisterType));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let root = builder.add_instruction(RegisterOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
    let replaced =
        builder.add_instruction(RegisterOperation::Swap, Vec::new(), vec![root, replacement], None).unwrap()[0];
    let snapshot = builder.add_instruction(RegisterOperation::Read, Vec::new(), vec![root], None).unwrap()[0];
    let frozen = builder.add_instruction(RegisterOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
    let program = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![replaced, snapshot, frozen],
            vec![Placeholder; 2],
            vec![Placeholder; 3],
        )
        .unwrap();

    // Discharging through the region driver rewrites every reference primitive into ordinary state threading, so the
    // downstream universe reaches the same outputs an eager reference execution would have produced.
    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let regions = [program];
    let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
    let inputs =
        vec![RegisterDischargeValue::Ordinary(RegisterValue(4)), RegisterDischargeValue::Ordinary(RegisterValue(3))];
    assert_eq!(
        driver.discharge_region(&context, 0, inputs),
        Ok(vec![
            RegisterDischargeValue::Ordinary(RegisterValue(4)),
            RegisterDischargeValue::Ordinary(RegisterValue(3)),
            RegisterDischargeValue::Ordinary(RegisterValue(3)),
        ]),
    );
    assert_eq!(context.live_roots(), Vec::new());
}

#[test]
fn test_downstream_reference_universe_discharges_into_a_staged_program() {
    // The same universe discharged against a staging destination, which is the shape production discharge uses: the
    // rewritten work is recorded into a destination program instead of executed.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = builder.add_input(RegisterIrType::Register(RegisterType));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let root = builder.add_instruction(RegisterOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
    let replaced =
        builder.add_instruction(RegisterOperation::Swap, Vec::new(), vec![root, replacement], None).unwrap()[0];
    let negated = builder.add_instruction(RegisterOperation::Negate, Vec::new(), vec![replaced], None).unwrap()[0];
    let frozen = builder.add_instruction(RegisterOperation::Freeze, Vec::new(), vec![root], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![negated, frozen],
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        )
        .unwrap();

    let discharge = |inputs: Vec<Tracer<TracingContext<RegisterValue, RegisterOperation>>>| {
        let context = ReferenceDischargeContext::new(inputs[0].context().clone());
        let carriers = inputs.into_iter().map(ReferenceDischargeValue::Ordinary).collect::<Vec<_>>();
        let regions = [source];
        let driver = RecursiveReferenceDischargeDriver::new(&regions, None);
        let outputs = driver.discharge_region(&context, 0, carriers)?;
        assert_eq!(context.live_roots(), Vec::new());
        outputs
            .iter()
            .map(|output| output.expect_ordinary("a discharged output").cloned())
            .collect::<Result<Vec<_>, _>>()
    };
    let (_, discharged): (_, Program<_, _, Vec<RegisterValue>, Vec<RegisterValue>>) =
        EagerContext::<RegisterValue, RegisterOperation>::trace(
            discharge,
            vec![RegisterIrType::Register(RegisterType); 2],
        )
        .unwrap();

    // Every reference primitive was normalized away, so the staged program threads the replacement directly and
    // records only the universe's one pure operation.
    assert_eq!(
        discharged.to_string(),
        indoc! {"
            lambda %0:register, %1:register .
            let %2:register = register.negate %0
            in (%2, %1)"},
    );
}

#[test]
fn test_downstream_program_level_discharge_threads_external_state_through_the_entry_boundary() {
    // `Program::discharge_references_with_policy` is the program-level entry point, and it is universe-generic: this
    // exercises it over the downstream universe, so nothing about the array universe can be load-bearing for it.
    // `f(counter, other, replacement) = replaced`, where `counter` is written and `other` is only read.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let counter = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let other = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let observed = builder.add_instruction(RegisterOperation::Read, Vec::new(), vec![other], None).unwrap()[0];
    let replaced = builder
        .add_instruction(RegisterOperation::Swap, Vec::new(), vec![counter, replacement], None)
        .unwrap()[0];
    let negated = builder.add_instruction(RegisterOperation::Negate, Vec::new(), vec![observed], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![replaced, negated],
            vec![Placeholder; 3],
            vec![Placeholder; 2],
        )
        .unwrap();

    // Each reference input keeps its boundary position and becomes an ordinary input carrying the referent's lifted
    // type, the public outputs are exactly the source outputs, and only the written root appends a hidden final-state
    // output after them.
    let discharged = source.discharge_references_with_policy::<RegisterReferenceDischarge>(1).unwrap();
    assert_eq!(discharged.public_output_count(), 2);
    assert_eq!(
        discharged.external_states(),
        &[
            ReferenceStateBinding::new(ReferenceSource::Capture { index: 0 }, Some(2)),
            ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, None),
        ],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register, %2:register .
            let %3:register = register.negate %1
            in (%0, %3, %2)"},
    );

    // A caller-owned root's holder belongs to the caller, so a program that consumes one is rejected by name.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let external = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let frozen = builder.add_instruction(RegisterOperation::Freeze, Vec::new(), vec![external], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
        .unwrap();
    assert_eq!(
        source.discharge_references_with_policy::<RegisterReferenceDischarge>(0).unwrap_err(),
        ProgramError::MalformedProgram(
            "reference discharge consumed external input 0, whose holder belongs to the caller".to_string(),
        ),
    );
}

#[test]
fn test_downstream_region_summary_exposes_exact_access_modes() {
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let read = builder.add_instruction(RegisterOperation::Read, Vec::new(), vec![reference], None).unwrap()[0];
    builder
        .add_instruction(RegisterOperation::Write, Vec::new(), vec![reference, replacement], None)
        .unwrap();
    let swapped = builder
        .add_instruction(RegisterOperation::Swap, Vec::new(), vec![reference, replacement], None)
        .unwrap()[0];
    let region = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(
            vec![read, swapped],
            vec![Placeholder; 2],
            vec![Placeholder; 2],
        )
        .unwrap();

    let context = RegisterDischargeContext::new(RegisterDestination::new());
    let reference = context.allocate_discharged(ReferenceType::new(RegisterType), RegisterValue(1)).unwrap();
    let root = reference.expect_reference("a downstream root").unwrap().root();
    let summary = context
        .region_summary(&RegisterOperation::Call, 0, region.entry_region_ref(), &[Some(root), None])
        .unwrap();

    assert_eq!(summary.accessed().collect::<Vec<_>>(), vec![root]);
    assert_eq!(
        summary.access_modes(root).collect::<Vec<_>>(),
        vec![ReferenceAccessMode::Read, ReferenceAccessMode::Write, ReferenceAccessMode::ReadWrite],
    );
    assert!(summary.has_access(root, ReferenceAccessMode::ReadWrite));
    assert!(!summary.has_access(root, ReferenceAccessMode::Accumulate));
    assert!(summary.is_mutated(root));
}

#[test]
fn test_downstream_partial_discharge_preserves_the_roots_it_was_not_asked_to_discharge() {
    // Partial discharge is reachable from downstream position too, and the register universe declines accumulation
    // and views entirely, so this is the minimal shape a backend needs: `f(counter, buffer, replacement) = replaced`,
    // where `counter` is selected and `buffer` stays a reference the rewritten program still reads and writes.
    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let counter = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let buffer = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = builder.add_input(RegisterIrType::Register(RegisterType));
    let observed = builder.add_instruction(RegisterOperation::Read, Vec::new(), vec![buffer], None).unwrap()[0];
    let replaced =
        builder.add_instruction(RegisterOperation::Swap, Vec::new(), vec![counter, observed], None).unwrap()[0];
    builder
        .add_instruction(RegisterOperation::Write, Vec::new(), vec![buffer, replacement], None)
        .unwrap();
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![replaced], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let sites = source.reference_discharge_sites(0).unwrap();
    assert_eq!(
        sites,
        vec![
            ReferenceDischargeSite::External(ReferenceSource::Input { index: 0 }),
            ReferenceDischargeSite::External(ReferenceSource::Input { index: 1 }),
        ],
    );
    let discharged = source
        .partially_discharge_references_with_policy::<RegisterReferenceDischarge>(0, &sites[..1])
        .unwrap();

    // The selected root became state at its own boundary position and publishes its final state as a hidden output;
    // the preserved root kept its reference type and reports no binding, and both of its accesses replayed verbatim.
    assert_eq!(discharged.public_output_count(), 1);
    assert_eq!(
        discharged.external_states(),
        &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:ref<register>, %2:register .
            let %3:register = register.read %1
                register.write %1 %2
            in (%0, %3)"},
    );
}

#[test]
fn test_downstream_structured_rule_discharges_through_the_region_boundary_api() {
    // The hand-rolled `register.call` rule exercises the complete structured surface from a third-party position:
    // region summaries, state widening, boundary construction, isolated region rebuilding, prediction validation,
    // and successor-state merging. The same caller root enters at two declared positions, so the rebuilt region must
    // preserve the aliasing: the write through position 0 is observed by the read through position 1.
    let mut callee = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let first = callee.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let second = callee.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let replacement = callee.add_input(RegisterIrType::Register(RegisterType));
    callee
        .add_instruction(RegisterOperation::Write, Vec::new(), vec![first, replacement], None)
        .unwrap();
    let observed = callee.add_instruction(RegisterOperation::Read, Vec::new(), vec![second], None).unwrap()[0];
    let callee = callee
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![observed], vec![Placeholder; 3], vec![Placeholder])
        .unwrap();

    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let reference = builder.add_input(RegisterIrType::Reference(ReferenceType::new(RegisterType)));
    let update = builder.add_input(RegisterIrType::Register(RegisterType));
    let region = builder.import_program(callee);
    let result = builder
        .add_instruction(RegisterOperation::Call, vec![region], vec![reference, reference, update], None)
        .unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![result], vec![Placeholder; 2], vec![Placeholder])
        .unwrap();

    let discharged = source.discharge_references_with_policy::<RegisterReferenceDischarge>(0).unwrap();
    assert_eq!(discharged.public_output_count(), 1);
    assert_eq!(
        discharged.external_states(),
        &[ReferenceStateBinding::new(ReferenceSource::Input { index: 0 }, Some(1))],
    );
    assert_eq!(
        discharged.program().to_string(),
        indoc! {"
            lambda %0:register, %1:register .
            let %2:register, %3:register = register.call %0 %0 %1 [
                callee={
                    lambda %0:register, %1:register, %2:register .
                    in (%2, %2)
                },
            ]
            in (%2, %3)"},
    );
}

#[test]
fn test_downstream_partial_selection_reaches_an_allocation_inside_a_structured_region() {
    // The allocation site sits inside the callee region, so whether it discharges is decided by the replay coordinate
    // the downstream rule's driver hands to `selects_allocation` inside the fork. An empty selection must preserve
    // the allocation inside the rebuilt region, and selecting the enumerated site must discharge it completely —
    // which is exactly the behavior a driver without a real `instruction()` coordinate would silently break.
    let mut callee = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let initial = callee.add_input(RegisterIrType::Register(RegisterType));
    let local = callee.add_instruction(RegisterOperation::ReferenceNew, Vec::new(), vec![initial], None).unwrap()[0];
    let frozen = callee.add_instruction(RegisterOperation::Freeze, Vec::new(), vec![local], None).unwrap()[0];
    let callee = callee
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![frozen], vec![Placeholder], vec![Placeholder])
        .unwrap();

    let mut builder = ProgramBuilder::<RegisterValue, RegisterOperation>::new();
    let input = builder.add_input(RegisterIrType::Register(RegisterType));
    let region = builder.import_program(callee);
    let result = builder.add_instruction(RegisterOperation::Call, vec![region], vec![input], None).unwrap()[0];
    let source = builder
        .build::<Vec<RegisterValue>, Vec<RegisterValue>>(vec![result], vec![Placeholder], vec![Placeholder])
        .unwrap();

    let preserved = source
        .clone()
        .partially_discharge_references_with_policy::<RegisterReferenceDischarge>(0, &[])
        .unwrap();
    assert_eq!(preserved.external_states(), &[]);
    assert_eq!(
        preserved.program().to_string(),
        indoc! {"
            lambda %0:register .
            let %1:register = register.call %0 [
                callee={
                    lambda %0:register .
                    let %1:ref<register> = register.reference_new %0
                        %2:register = register.freeze %1
                    in (%2)
                },
            ]
            in (%1)"},
    );

    let sites = source.reference_discharge_sites(0).unwrap();
    assert_eq!(sites.len(), 1);
    let full = source
        .partially_discharge_references_with_policy::<RegisterReferenceDischarge>(0, sites.as_slice())
        .unwrap()
        .try_into_full()
        .unwrap();
    assert_eq!(
        full.program().to_string(),
        indoc! {"
            lambda %0:register .
            let %1:register = register.call %0 [
                callee={
                    lambda %0:register .
                    in (%0)
                },
            ]
            in (%1)"},
    );
}

#[test]
fn test_downstream_reference_handles_cannot_be_fabricated() {
    // The tests above establish that the discharge surface is reachable from outside `ryft-core`. This is the
    // matching negative proof: a handle's checked construction cannot be bypassed from that same position, so a
    // third-party rule can read root identity, alias, derived type, and preserved value but can invent none of them.
    // The two cases are separate files because their rejections belong to different compiler passes, and only the
    // earlier one is reported when they share a file.
    let test_cases = trybuild::TestCases::new();
    test_cases.compile_fail("tests/reference_discharge/error_reference_handle_fabrication.rs");
    test_cases.compile_fail("tests/reference_discharge/error_reference_handle_private_fields.rs");
}
