//! Canonical grouping of logical XLA values into ordinary dense array leaves.
//!
//! A ragged value contributes storage followed by its canonical extent payloads. This grouping precedes the
//! executable signature's independent zero erasure, hidden dense sizes and effect-token projection.

use std::ops::Range;
use std::sync::Arc;

use ryft_core::macros::check_count;
use ryft_core::{
    Add, ArrayIrType, ArrayIrTypeRefinements, ArrayIrValue, ArrayType, CaptureReference, Context, DataType, Dimension,
    DimensionSize, DimensionVariable, GuardOperation, Operation, ProgramError, RaggedNew, RaggedNewOperation,
    RaggedType, RaggedValue, RaggedValueKind, RegionRef, ScanOperation, Tracer, TracingContext, TransposeOperation,
    Type, TypeError, TypeIdentity, TypeIdentityRenaming, TypeRefinements, Typed, ValueProjection, WhileOperation,
    ZeroOperation,
};

use crate::experimental::ops::{FlatXlaProgram, XlaConstant, XlaOperation};
use crate::operations::jit_call::JitCallOperation;
use crate::{XlaArray, XlaDomain, XlaValue};

/// Immutable logical-to-dense-leaf mapping shared by lowering, invocation, captures and result reconstruction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct XlaValueLayout {
    /// Public logical signature, including indexed identities and descriptor domains.
    logical_types: Arc<[ArrayIrType]>,
    /// Dense field signature before executable zero erasure and hidden sizes.
    leaf_types: Arc<[ArrayType]>,
    /// Contiguous field range contributed by each logical value.
    ranges: Arc<[Range<usize>]>,
}

/// Dense replay together with the logical boundary that its leaves implement. `capture_count` counts dense leaves.
pub(crate) struct GroupedXlaProgram {
    /// Ordinary dense program passed to the existing lowerer.
    pub(crate) program: FlatXlaProgram,
    /// Logical argument and lifted-capture grouping.
    pub(crate) input_layout: XlaValueLayout,
    /// Logical result grouping.
    pub(crate) output_layout: XlaValueLayout,
    /// Leading dense capture prefix, expanded by the input grouping.
    pub(crate) capture_count: usize,
}

/// Dense staging context used by final backend legalization.
type LeafContext = TracingContext<XlaConstant, XlaOperation>;
/// One canonical array field, or one unchanged structural member of an internal region.
type Leaf = Tracer<LeafContext>;
/// Ordered fields implementing one source logical value.
type Group = Vec<Leaf>;

/// Legalizes aggregate representation and transport once, before ordinary dense MLIR lowering. Dense programs keep
/// their original borrowed path and exact canonical rendering; external dense operation families retain singleton
/// lowering through the same existing dense backend interface.
pub(crate) fn legalize_ragged_program(
    program: &FlatXlaProgram,
    capture_count: usize,
) -> Result<Option<GroupedXlaProgram>, ProgramError> {
    if !program
        .regions()
        .iter()
        .any(|region| region.atoms().iter().any(|atom| matches!(atom.r#type().as_ref(), ArrayIrType::Ragged(_))))
    {
        return Ok(None);
    }
    if capture_count > program.input_count() {
        return Err(ProgramError::InvalidInputCount { expected: capture_count, actual: program.input_count() });
    }
    let input_layout = XlaValueLayout::new(&program.input_types())?;
    let output_layout = XlaValueLayout::new(&program.output_types())?;
    let capture_types = program.input_types()[..capture_count].to_vec();
    let program = legalize_region(program.entry_region_ref(), &capture_types)?;
    let capture_count = input_layout.prefix_leaf_count(capture_count);
    Ok(Some(GroupedXlaProgram { program, input_layout, output_layout, capture_count }))
}

/// Internal regions can also carry structural singleton members. Only ragged members expand.
fn leaf_types(r#type: &ArrayIrType) -> Vec<ArrayIrType> {
    match r#type {
        ArrayIrType::Ragged(r#type) => std::iter::once(r#type.storage_type())
            .chain(r#type.extent_types())
            .cloned()
            .map(ArrayIrType::Array)
            .collect(),
        _ => vec![r#type.clone()],
    }
}

/// Counts canonical fields without allocating their types.
fn leaf_count(r#type: &ArrayIrType) -> usize {
    match r#type {
        ArrayIrType::Ragged(r#type) => r#type.extent_types().len() + 1,
        _ => 1,
    }
}

/// Replays a borrowed region with its inherited lexical capture namespace and logical input groups.
fn legalize_region(
    region: RegionRef<'_, XlaConstant, XlaOperation>,
    capture_types: &[ArrayIrType],
) -> Result<FlatXlaProgram, ProgramError> {
    legalize_region_with_capture_prefix(region, capture_types, false)
}

/// Linear-call execution uses only its forward region. When represented as a final runtime call, that region's
/// enclosing captures become an explicit leading prefix so its lexical capture references keep the same meaning.
fn legalize_region_with_capture_prefix(
    region: RegionRef<'_, XlaConstant, XlaOperation>,
    capture_types: &[ArrayIrType],
    lift_captures: bool,
) -> Result<FlatXlaProgram, ProgramError> {
    let input_types = region.input_types();
    let prefix_count = if lift_captures { capture_types.iter().map(|r#type| leaf_count(r#type)).sum() } else { 0 };
    let dense_inputs = capture_types
        .iter()
        .take(if lift_captures { capture_types.len() } else { 0 })
        .chain(&input_types)
        .flat_map(leaf_types)
        .collect::<Vec<_>>();
    let (_, program) = LeafContext::trace_with_context_and_named_axes(
        |context, inputs: Vec<Leaf>| {
            let mut inputs = inputs.into_iter();
            inputs.by_ref().take(prefix_count).for_each(drop);
            let groups = input_types
                .iter()
                .map(|r#type| inputs.by_ref().take(leaf_count(r#type)).collect::<Group>())
                .collect::<Vec<_>>();
            let groups = guard_entry_groups(&input_types, groups)?;
            let outputs = region.interpret_with(
                groups,
                |_, constant| lift_constant(context, constant, capture_types),
                |instruction, inputs| {
                    context.invoke_with_provenance_origin(instruction.provenance().clone(), || {
                        let input_types = instruction
                            .inputs()
                            .iter()
                            .map(|id| region.atoms()[id.index()].r#type().into_owned())
                            .collect::<Vec<_>>();
                        let output_types = instruction
                            .outputs()
                            .iter()
                            .map(|id| region.atoms()[id.index()].r#type().into_owned())
                            .collect::<Vec<_>>();
                        let attached = instruction
                            .regions()
                            .iter()
                            .map(|id| RegionRef::new(region.arena(), *id))
                            .collect::<Result<Vec<_>, _>>()?;
                        legalize_instruction(
                            context,
                            instruction.operation(),
                            &input_types,
                            &output_types,
                            inputs,
                            &attached,
                            capture_types,
                        )
                    })
                },
            )?;
            Ok::<_, ProgramError>(outputs.into_iter().flatten().collect::<Vec<_>>())
        },
        dense_inputs,
        Vec::new(),
    )?;
    Ok(program)
}

/// Preserves the active lexical capture namespace while expanding captured aggregates into their field references.
fn lift_constant(
    context: &LeafContext,
    constant: &XlaConstant,
    capture_types: &[ArrayIrType],
) -> Result<Group, ProgramError> {
    if let XlaConstant::Captured(capture) = constant {
        let Some(expected) = capture_types.get(capture.index()) else {
            return Err(ProgramError::MalformedProgram(format!(
                "capture index {} is outside the enclosing prefix",
                capture.index()
            )));
        };
        if expected != capture.r#type().as_ref() {
            return Err(TypeError::invalid("captured value disagrees with its enclosing logical input type").into());
        }
        let offset = capture_types[..capture.index()].iter().map(|r#type| leaf_count(r#type)).sum::<usize>();
        leaf_types(expected)
            .into_iter()
            .enumerate()
            .map(|(index, r#type)| context.lift(XlaConstant::Captured(CaptureReference::new(offset + index, r#type))))
            .collect()
    } else {
        Ok(vec![context.lift(constant.clone())?])
    }
}

/// Forms a structural carrier over ordinary array tracers for the shared numerical checker.
fn core_ragged(
    r#type: &RaggedType,
    fields: &[Leaf],
) -> Result<ArrayIrValue<<Leaf as ValueProjection<ArrayType>>::Projected>, ProgramError> {
    let arrays = fields
        .iter()
        .cloned()
        .map(<Leaf as ValueProjection<ArrayType>>::into_projected)
        .collect::<Result<Vec<_>, _>>()?;
    let mut arrays = arrays.into_iter();
    Ok(ArrayIrValue::Ragged(RaggedValue::new(r#type.clone(), arrays.next().unwrap(), arrays.collect())?))
}

/// Installs the indexed obligations that untyped group replay cannot derive from its input vectors. The replay traces
/// every field at exactly its declared leaf type, so the fields prove all of their ordinary relationships by
/// construction. Only repeated indexed identities remain, and the shared exact-schema guard checks them together with
/// the dense fields of every array and ragged member.
fn guard_entry_groups(input_types: &[ArrayIrType], mut groups: Vec<Group>) -> Result<Vec<Group>, ProgramError> {
    if !input_types.iter().any(|r#type| matches!(r#type, ArrayIrType::Ragged(_))) {
        return Ok(groups);
    }
    let refinements = ArrayIrTypeRefinements::from_local_declarations(input_types, input_types)?;
    let indexed = input_types
        .iter()
        .filter_map(|r#type| <&RaggedType>::try_from(r#type).ok())
        .flat_map(|r#type| {
            let dimension = match r#type.value_kind() {
                RaggedValueKind::Array => None,
                RaggedValueKind::Dimension(dimension) => dimension.variable(),
            };
            r#type.ragged_dimensions().iter().map(|descriptor| descriptor.dimension()).chain(dimension)
        })
        .collect::<Vec<_>>();
    if !refinements.identities_needing_runtime_checks().iter().any(|identity| indexed.contains(&identity)) {
        return Ok(groups);
    }
    let members = input_types
        .iter()
        .enumerate()
        .filter_map(|(index, r#type)| matches!(r#type, ArrayIrType::Array(_) | ArrayIrType::Ragged(_)).then_some(index))
        .collect::<Vec<_>>();
    let types = members.iter().map(|&index| input_types[index].clone()).collect::<Vec<_>>();
    let pinned = input_types
        .iter()
        .flat_map(Type::identities)
        .map(|(_, identity)| identity.clone())
        .collect::<Vec<_>>();
    let operation = GuardOperation::new("XLA region entry", types.clone()).retaining(&types, &pinned)?;
    let values = members.iter().map(|&index| groups[index].clone()).collect::<Vec<_>>();
    for (index, value) in members.into_iter().zip(guard_array_groups(&operation, &types, &values)?) {
        groups[index] = value;
    }
    Ok(groups)
}

/// Uses the shared exact-schema guard over canonical carriers, retaining every checked field dependency.
fn guard_array_groups(
    operation: &GuardOperation<ArrayIrType>,
    input_types: &[ArrayIrType],
    inputs: &[Group],
) -> Result<Vec<Group>, ProgramError> {
    let values = input_types
        .iter()
        .zip(inputs)
        .map(|(r#type, fields)| match r#type {
            ArrayIrType::Array(_) => <Leaf as ValueProjection<ArrayType>>::into_projected(fields[0].clone())
                .map(ArrayIrValue::Array)
                .map_err(ProgramError::from),
            ArrayIrType::Ragged(r#type) => core_ragged(r#type, fields),
            _ => Err(ProgramError::UnsupportedOperation {
                message: "xla grouped guards support array and ragged members".into(),
            }),
        })
        .collect::<Result<Vec<_>, _>>()?;
    operation
        .interpret_ragged(&values)?
        .into_iter()
        .map(|value| match value {
            ArrayIrValue::Array(value) => Ok(vec![<Leaf as ValueProjection<ArrayType>>::from_projected(value)]),
            ArrayIrValue::Ragged(value) => Ok(std::iter::once(value.storage())
                .chain(value.extents())
                .cloned()
                .map(<Leaf as ValueProjection<ArrayType>>::from_projected)
                .collect()),
            _ => unreachable!("array/ragged guard inputs have array/ragged outputs"),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
/// Replays one logical instruction, expanding only supported representation and transport operations.
fn legalize_instruction(
    context: &LeafContext,
    operation: &XlaOperation,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    inputs: &[Group],
    regions: &[RegionRef<'_, XlaConstant, XlaOperation>],
    capture_types: &[ArrayIrType],
) -> Result<Vec<Group>, ProgramError> {
    match operation {
        XlaOperation::Guard(operation) if input_types.iter().any(|r#type| matches!(r#type, ArrayIrType::Ragged(_))) => {
            return guard_array_groups(operation, input_types, inputs);
        }
        XlaOperation::RaggedStorage(_) => return Ok(vec![vec![inputs[0][0].clone()]]),
        XlaOperation::RaggedExtents(operation) => {
            let r#type = <&RaggedType>::try_from(&input_types[0])?;
            let axis = operation
                .axis()
                .normalize(r#type.storage_type().rank())
                .map_err(|error| TypeError::invalid(error.to_string()))?;
            let occurrence = r#type
                .ragged_dimensions()
                .iter()
                .position(|descriptor| descriptor.axis() == axis)
                .ok_or_else(|| TypeError::invalid("ragged extent projection selected an ordinary axis"))?;
            return Ok(vec![vec![inputs[0][occurrence + 1].clone()]]);
        }
        XlaOperation::RaggedNew(operation) => {
            let count = operation.output_type().extent_types().len() + 1;
            let mut fields = inputs[..count].iter().map(|group| group[0].clone()).collect::<Vec<_>>();
            let declared = leaf_types(&operation.output_type().clone().into());
            // Preserve ordinary cross-field relationships before invoking the shared numerical checker. The direct
            // composite wrapper deliberately requires these relationships to have been established already.
            let actual = fields.iter().map(|field| field.r#type().into_owned()).collect::<Vec<_>>();
            if let Some((guard, members)) = GuardOperation::for_entry("ragged_new", &declared, &actual)? {
                let guarded = context.bind(
                    guard,
                    vec![],
                    &members.iter().map(|&index| fields[index].clone()).collect::<Vec<_>>(),
                )?;
                for (index, value) in members.into_iter().zip(guarded) {
                    fields[index] = value;
                }
            }
            let arrays = fields
                .into_iter()
                .map(<Leaf as ValueProjection<ArrayType>>::into_projected)
                .collect::<Result<Vec<_>, _>>()?;
            let arrays = arrays.into_iter().map(ArrayIrValue::Array).collect::<Vec<_>>();
            let result = if operation.has_identity_source() {
                let source = core_ragged(<&RaggedType>::try_from(&input_types[count])?, &inputs[count])?;
                arrays[0].ragged_new_with_identity_source(operation.output_type().clone(), &arrays[1..], &source)?
            } else {
                arrays[0].ragged_new(operation.output_type().clone(), &arrays[1..])?
            };
            let ArrayIrValue::Ragged(result) = result else {
                unreachable!("checked construction returns a ragged value")
            };
            return Ok(vec![
                std::iter::once(result.storage())
                    .chain(result.extents())
                    .cloned()
                    .map(<Leaf as ValueProjection<ArrayType>>::from_projected)
                    .collect(),
            ]);
        }
        XlaOperation::RaggedZero(_) => {
            let r#type = <&RaggedType>::try_from(&output_types[0])?;
            let dimensions = r#type
                .storage_type()
                .shape()
                .dimensions()
                .iter()
                .enumerate()
                .filter_map(|(axis, dimension)| matches!(dimension, Dimension::Dynamic(_)).then_some(axis))
                .map(|axis| inputs[0][0].dimension_size(axis))
                .collect::<Result<Vec<_>, _>>()?;
            let storage =
                context.bind(ZeroOperation::new(r#type.storage_type().clone()), vec![], &dimensions)?.remove(0);
            let mut fields = vec![vec![storage]];
            fields.extend(inputs[0][1..].iter().cloned().map(|field| vec![field]));
            fields.push(inputs[0].clone());
            let mut types =
                fields[..fields.len() - 1].iter().map(|field| field[0].r#type().into_owned()).collect::<Vec<_>>();
            types.push(input_types[0].clone());
            return legalize_instruction(
                context,
                &XlaOperation::RaggedNew(RaggedNewOperation::with_identity_source(r#type.clone())),
                &types,
                output_types,
                &fields,
                &[],
                capture_types,
            );
        }
        XlaOperation::RaggedAdd(_) => {
            let r#type = <&RaggedType>::try_from(&input_types[0])?;
            let mut fields = inputs[0].iter().cloned().map(|field| vec![field]).collect::<Vec<_>>();
            fields.push(inputs[1].clone());
            let mut types = leaf_types(&input_types[0]);
            types.push(input_types[1].clone());
            let constructor = XlaOperation::RaggedNew(RaggedNewOperation::with_identity_source(r#type.clone()));
            let checked =
                legalize_instruction(context, &constructor, &types, output_types, &fields, &[], capture_types)?
                    .remove(0);
            let storage = if r#type.storage_type().data_type() == DataType::Zero {
                checked[0].clone()
            } else {
                checked[0].add(&inputs[1][0])?
            };
            fields[0] = vec![storage];
            *fields.last_mut().unwrap() = checked;
            return legalize_instruction(context, &constructor, &types, output_types, &fields, &[], capture_types);
        }
        _ => {}
    }
    if matches!(operation, XlaOperation::LinearCall(_)) {
        let Some(forward) = regions.first() else {
            return Err(ProgramError::MalformedProgram("linear_call requires a forward region".into()));
        };
        let mut program = legalize_region_with_capture_prefix(*forward, capture_types, true)?;
        // A linear call's scalar defining output can retain its forward-region name. Give that definition a local
        // physical name before a JIT boundary republishes the original caller-owned result name.
        let callee_inputs = program.input_types();
        let callee_outputs = program.output_types();
        let mut unavailable = program.type_identity_signature().identities().to_vec();
        unavailable.extend(output_types.iter().flat_map(Type::identities).map(|(_, identity)| identity.clone()));
        let mut local_renaming = TypeIdentityRenaming::new();
        for (_, identity) in callee_outputs.iter().flat_map(Type::identities) {
            if local_renaming.target(identity).is_none()
                && !callee_inputs.iter().flat_map(Type::identities).any(|(_, input)| input == identity)
            {
                let local = local_renaming.insert_fresh(identity.clone(), &unavailable)?;
                unavailable.push(local);
            }
        }
        if !local_renaming.is_identity() {
            program = program.rename_type_identities(&local_renaming)?;
        }
        let captures = capture_types
            .iter()
            .flat_map(leaf_types)
            .enumerate()
            .map(|(index, r#type)| context.lift(XlaConstant::Captured(CaptureReference::new(index, r#type))))
            .collect::<Result<Vec<_>, _>>()?;
        let capture_count = captures.len();
        let flat = captures.into_iter().chain(inputs.iter().flatten().cloned()).collect::<Vec<_>>();
        let logical_inputs = capture_types.iter().chain(input_types).cloned().collect();
        let call = JitCallOperation::new(capture_count)
            .with_result_type_signature(
                &program.input_types(),
                &program.output_types(),
                output_types.iter().flat_map(leaf_types).collect(),
            )?
            .with_logical_signature(logical_inputs, output_types.to_vec());
        let outputs = context.bind(XlaOperation::JitCall(call), vec![program], &flat)?;
        return split_outputs(outputs, output_types);
    }
    if let XlaOperation::Scan(scan) = operation {
        let [body] = regions else {
            return Err(ProgramError::MalformedProgram("scan requires one body".into()));
        };
        return legalize_scan(context, scan, *body, input_types, output_types, inputs, capture_types);
    }
    let grouped_boundary =
        input_types.iter().chain(output_types).any(|r#type| matches!(r#type, ArrayIrType::Ragged(_)));
    if grouped_boundary && !matches!(operation, XlaOperation::While(_) | XlaOperation::JitCall(_)) {
        return Err(ProgramError::UnsupportedOperation {
            message: format!("xla grouped replay does not support ragged `{}`", operation.name()),
        });
    }
    let mut rewritten = match operation {
        // Indexed result binders belong to the logical signature. Dense state has no indexed identities; only the
        // semantic iteration bound crosses into the physical loop operation.
        XlaOperation::While(operation) => {
            XlaOperation::While(WhileOperation::<ArrayIrType>::new().with_iteration_bound(operation.iteration_bound())?)
        }
        _ => operation.clone(),
    };
    let programs = if let XlaOperation::JitCall(operation) = operation {
        let [callee] = regions else {
            return Err(ProgramError::MalformedProgram("jit_call requires one callee".into()));
        };
        let callee_inputs = callee.input_types();
        let captures = &callee_inputs[..operation.capture_count()];
        let capture_count = captures.iter().map(|r#type| leaf_count(r#type)).sum();
        let program = legalize_region(*callee, captures)?;
        rewritten = XlaOperation::JitCall(
            JitCallOperation::new(capture_count)
                .with_result_type_signature(
                    &program.input_types(),
                    &program.output_types(),
                    output_types.iter().flat_map(leaf_types).collect(),
                )?
                .with_logical_signature(callee_inputs, callee.output_types()),
        );
        vec![program]
    } else {
        regions
            .iter()
            .map(|region| legalize_region(*region, capture_types))
            .collect::<Result<Vec<_>, _>>()?
    };
    let flat = inputs.iter().flatten().cloned().collect::<Vec<_>>();
    let outputs = context.bind(rewritten, programs, &flat)?;
    split_outputs(outputs, output_types)
}

/// Checks flat result arity before partitioning results into canonical logical groups.
fn split_outputs(outputs: Vec<Leaf>, output_types: &[ArrayIrType]) -> Result<Vec<Group>, ProgramError> {
    let expected = output_types.iter().map(|r#type| leaf_count(r#type)).sum();
    if outputs.len() != expected {
        return Err(ProgramError::InvalidOutputCount { expected, actual: outputs.len() });
    }
    let mut outputs = outputs.into_iter();
    Ok(output_types.iter().map(|r#type| outputs.by_ref().take(leaf_count(r#type)).collect()).collect())
}

/// Uses the existing dense scan signature. Shared payloads become unchanged extra carries; indexed payloads move
/// their actual scan coordinate to the leading dense scan axis. The body and result use the canonical group order.
#[allow(clippy::too_many_arguments)]
fn legalize_scan(
    context: &LeafContext,
    scan: &ScanOperation<ArrayIrType>,
    body: RegionRef<'_, XlaConstant, XlaOperation>,
    input_types: &[ArrayIrType],
    output_types: &[ArrayIrType],
    inputs: &[Group],
    capture_types: &[ArrayIrType],
) -> Result<Vec<Group>, ProgramError> {
    let body_input_types = body.input_types();
    let operand_count = body_input_types.len() - 1;
    let body = legalize_region(body, capture_types)?;
    let body_leaf_types = body.input_types();
    let mut next = 1;
    let ranges = body_input_types[1..]
        .iter()
        .map(|r#type| {
            let start = next;
            next += leaf_count(r#type);
            start..next
        })
        .collect::<Vec<_>>();
    let carry_count = ranges.iter().take(scan.carry_count()).map(|range| range.len()).sum::<usize>();
    let mut shared = Vec::new();
    let mut stacks = Vec::new();
    for index in scan.carry_count()..operand_count {
        for field in 0..inputs[index].len() {
            let axis = match (&input_types[index], field) {
                (_, 0) => Some(0),
                (ArrayIrType::Ragged(r#type), field) => {
                    r#type.ragged_dimensions()[field - 1].extent_axes().iter().position(|&axis| axis == 0)
                }
                _ => Some(0),
            };
            match axis {
                Some(axis) => stacks.push((index, field, axis)),
                None => shared.push((index, field)),
            }
        }
    }
    let mut order = (0..1 + carry_count).collect::<Vec<_>>();
    order.extend(shared.iter().map(|&(index, field)| ranges[index].start + field));
    order.extend(stacks.iter().map(|&(index, field, _)| ranges[index].start + field));
    let wrapper_types = order.iter().map(|&index| body_leaf_types[index].clone()).collect::<Vec<_>>();
    let shared_count = shared.len();
    let (_, wrapper) = LeafContext::trace_with_context_and_named_axes(
        |context, inputs: Vec<Leaf>| {
            let mut canonical = vec![None; order.len()];
            for (&original, value) in order.iter().zip(&inputs) {
                canonical[original] = Some(value.clone());
            }
            let canonical = canonical
                .into_iter()
                .map(|value| value.expect("scan body input permutation covers every field"))
                .collect();
            let mut outputs = body.interpret_in_context(context, canonical)?;
            let stacked = outputs.split_off(carry_count);
            outputs.extend_from_slice(&inputs[1 + carry_count..1 + carry_count + shared_count]);
            outputs.extend(stacked);
            Ok(outputs)
        },
        wrapper_types,
        vec![],
    )?;
    let mut flat = inputs[..scan.carry_count()].iter().flatten().cloned().collect::<Vec<_>>();
    flat.extend(shared.iter().map(|&(index, field)| inputs[index][field].clone()));
    for &(index, field, axis) in &stacks {
        let value = &inputs[index][field];
        if axis == 0 {
            flat.push(value.clone());
        } else {
            let rank = <&ArrayType>::try_from(value.r#type().as_ref())?.rank();
            let permutation =
                std::iter::once(axis).chain((0..rank).filter(|&candidate| candidate != axis)).collect::<Vec<_>>();
            flat.push(
                context.bind(TransposeOperation::new(permutation), vec![], std::slice::from_ref(value))?.remove(0),
            );
        }
    }
    flat.extend(inputs[operand_count..].iter().flatten().cloned());
    let scan = ScanOperation::<ArrayIrType>::new(carry_count + shared_count, scan.length().clone())
        .with_reverse(scan.reverse())
        .with_unroll(scan.unroll())?;
    let mut outputs = context.bind(XlaOperation::Scan(scan), vec![wrapper], &flat)?;
    outputs.drain(carry_count..carry_count + shared_count);
    split_outputs(outputs, output_types)
}

impl XlaValueLayout {
    /// Validates an array/ragged boundary and records every logical value's contiguous field range.
    pub(crate) fn new(types: &[ArrayIrType]) -> Result<Self, ProgramError> {
        let mut leaf_types = Vec::new();
        let mut ranges = Vec::with_capacity(types.len());
        for r#type in types {
            let start = leaf_types.len();
            match r#type {
                ArrayIrType::Array(r#type) => leaf_types.push(r#type.clone()),
                ArrayIrType::Ragged(r#type) => {
                    leaf_types.push(r#type.storage_type().clone());
                    leaf_types.extend_from_slice(r#type.extent_types());
                }
                _ => {
                    return Err(ProgramError::UnsupportedOperation {
                        message: format!("xla array-leaf boundaries cannot represent type `{type}`"),
                    });
                }
            }
            ranges.push(start..leaf_types.len());
        }
        Ok(Self { logical_types: types.into(), leaf_types: leaf_types.into(), ranges: ranges.into() })
    }

    /// Returns the original logical boundary signature.
    pub(crate) fn logical_types(&self) -> &[ArrayIrType] {
        &self.logical_types
    }

    /// Returns the ordinary array signature before executable-specific projection.
    pub(crate) fn leaf_types(&self) -> &[ArrayType] {
        &self.leaf_types
    }

    /// Returns the canonical field range of logical value `index`.
    pub(crate) fn leaf_range(&self, index: usize) -> Range<usize> {
        self.ranges[index].clone()
    }

    /// Returns the total number of dense fields.
    pub(crate) fn leaf_count(&self) -> usize {
        self.leaf_types.len()
    }

    /// Returns the number of logical values.
    pub(crate) fn logical_count(&self) -> usize {
        self.logical_types.len()
    }

    /// Expands the first `count` logical values, as used by lifted capture prefixes.
    pub(crate) fn prefix_leaf_count(&self, count: usize) -> usize {
        if count == 0 { 0 } else { self.leaf_range(count - 1).end }
    }

    /// Expands logical donation requests, allowing the storage of array values to be donated but preserving every
    /// extent buffer. The storage of a ragged value of first-class dimension items holds the indexed dimension values
    /// themselves, so it is an extent buffer too. Eager projections share such handles with other live values without
    /// copying them, which means that donating one could invalidate the extents of another value.
    pub(crate) fn donation_flags(&self, flags: &[bool]) -> Result<Vec<bool>, ProgramError> {
        if flags.len() != self.logical_count() {
            return Err(ProgramError::InvalidInputCount { expected: self.logical_count(), actual: flags.len() });
        }
        let mut leaves = vec![false; self.leaf_count()];
        for ((index, donate), r#type) in flags.iter().enumerate().zip(self.logical_types.iter()) {
            let donatable = match r#type {
                ArrayIrType::Ragged(r#type) => matches!(r#type.value_kind(), RaggedValueKind::Array),
                _ => true,
            };
            leaves[self.leaf_range(index).start] = *donate && donatable;
        }
        Ok(leaves)
    }

    /// Borrows existing array handles; grouping never reads payloads or allocates device storage.
    pub(crate) fn flatten<'c>(&self, values: &[XlaValue<'c>]) -> Result<Vec<XlaArray<'c>>, ProgramError> {
        if values.len() != self.logical_count() {
            return Err(ProgramError::InvalidInputCount { expected: self.logical_count(), actual: values.len() });
        }
        let actual = self
            .logical_types
            .iter()
            .zip(values)
            .map(|(declared, value)| invocation_type(declared, value.r#type().as_ref()))
            .collect::<Result<Vec<_>, _>>()?;
        // A cached executable owns a formal input scope. Invocation admits a runtime scope with the same structure
        // under different identity names; repeated-family relationships still execute through the region's entry guard.
        ArrayIrTypeRefinements::from_local_declarations(self.logical_types.iter(), &actual)?;
        let mut leaves = Vec::with_capacity(self.leaf_count());
        for value in values {
            match value {
                XlaValue::Array(value) => leaves.push(value.clone()),
                XlaValue::Ragged(value) => {
                    leaves.push(value.storage().clone());
                    leaves.extend_from_slice(value.extents());
                }
                _ => return Err(TypeError::invalid("xla array-leaf invocation requires array or ragged values").into()),
            }
        }
        Ok(leaves)
    }

    /// Names the indexed families of this result signature for one execution. Input families referenced by results
    /// retain the corresponding runtime names; unreturned input families require no name choice. Output-local
    /// families, and input families supplied only by exact dimension items, receive one fresh name shared by every
    /// output occurrence of that family, so values returned by different executions never share a formal name.
    pub(crate) fn result_identity_renaming(
        &self,
        inputs: &Self,
        actual_inputs: &[ArrayIrType],
    ) -> Result<TypeIdentityRenaming<DimensionVariable>, ProgramError> {
        Ok(self.result_instantiation(inputs, actual_inputs)?.map(|(_, renaming)| renaming).unwrap_or_default())
    }

    /// Instantiates this result signature for one execution. Exact facts of the actual inputs replace the dimension
    /// items that they determine, just as eager dense values carry static extents: a result whose item family is
    /// supplied as `dimension<3>` has item `dimension<3>`, not the executable's formal family. The remaining indexed
    /// families are then named by [`Self::result_identity_renaming`]. Ragged descriptors and storage keep their
    /// representation, since an exact fact never erases a [`RaggedDimension`](ryft_core::RaggedDimension) entry.
    pub(crate) fn result_types(
        &self,
        inputs: &Self,
        actual_inputs: &[ArrayIrType],
    ) -> Result<Vec<ArrayIrType>, ProgramError> {
        let Some((refinements, renaming)) = self.result_instantiation(inputs, actual_inputs)? else {
            return Ok(self.logical_types.to_vec());
        };
        self.logical_types
            .iter()
            .map(|r#type| Ok(refinements.refine_declared(r#type, &[])?.rename_identities(&renaming)?))
            .collect()
    }

    /// Returns the input refinements and result renaming shared by [`Self::result_identity_renaming`] and
    /// [`Self::result_types`], or `None` when no result is ragged and the signature therefore needs no instantiation.
    fn result_instantiation(
        &self,
        inputs: &Self,
        actual_inputs: &[ArrayIrType],
    ) -> Result<Option<(ArrayIrTypeRefinements, TypeIdentityRenaming<DimensionVariable>)>, ProgramError> {
        check_count!("input", actual_inputs, inputs.logical_count(), ProgramError);
        if !self.logical_types.iter().any(|r#type| matches!(r#type, ArrayIrType::Ragged(_))) {
            return Ok(None);
        }
        let actual_inputs = inputs
            .logical_types
            .iter()
            .zip(actual_inputs)
            .map(|(declared, actual)| invocation_type(declared, actual))
            .collect::<Result<Vec<_>, _>>()?;
        let refinements = ArrayIrTypeRefinements::from_local_declarations(inputs.logical_types.iter(), &actual_inputs)?;
        let input_identities = indexed_identities(inputs.logical_types()).collect::<Vec<_>>();
        let output_identities = indexed_identities(self.logical_types()).collect::<Vec<_>>();
        let mut renaming = TypeIdentityRenaming::new();
        for class in refinements.identity_classes() {
            // Entry checks can relate several runtime families under one formal input name. Result reconstruction
            // needs a unique runtime name only when that formal indexed family actually occurs in a result.
            for identity in class
                .declared()
                .iter()
                .filter(|identity| input_identities.contains(identity) && output_identities.contains(identity))
            {
                match class.actual() {
                    [actual] => renaming.insert(identity.clone(), actual.clone())?,

                    // Exact dimension items have no runtime identity to forward. Refinement replaces the items
                    // themselves, and any descriptor that still names the family gets a per-execution name.
                    [] => renaming.insert(identity.clone(), identity.fresh())?,
                    _ => {
                        return Err(TypeError::invalid(
                            "a forwarded indexed result requires one unambiguous runtime input identity",
                        )
                        .into());
                    }
                }
            }
        }
        for identity in output_identities {
            if !input_identities.contains(&identity) && renaming.target(identity).is_none() {
                renaming.insert(identity.clone(), identity.fresh())?;
            }
        }
        Ok(Some((refinements, renaming)))
    }

    /// Reassembles handles at their actual dense physical types, retaining the logical indexed metadata of
    /// `result_types` (i.e., this signature as instantiated by [`Self::result_types`]) and the session.
    pub(crate) fn regroup<'c>(
        &self,
        leaves: Vec<XlaArray<'c>>,
        domain: &XlaDomain<'c>,
        result_types: &[ArrayIrType],
    ) -> Result<Vec<XlaValue<'c>>, ProgramError> {
        check_count!("output", result_types, self.logical_count(), ProgramError);
        if leaves.len() != self.leaf_count() {
            return Err(ProgramError::InvalidOutputCount { expected: self.leaf_count(), actual: leaves.len() });
        }
        let mut leaves = leaves.into_iter();
        let mut values = Vec::with_capacity(self.logical_count());
        for r#type in result_types {
            let storage = leaves.next().unwrap();
            match r#type {
                ArrayIrType::Array(_) => values.push(XlaValue::Array(storage)),
                ArrayIrType::Ragged(r#type) => {
                    let actual = RaggedType::new(
                        storage.r#type().into_owned(),
                        r#type.value_kind().clone(),
                        r#type.ragged_dimensions().to_vec(),
                    )?;
                    let extents = leaves.by_ref().take(actual.extent_types().len()).collect();
                    let value = RaggedValue::new(actual, storage, extents)?;
                    values.push(XlaValue::from_array_ir_value(ryft_core::ArrayIrValue::Ragged(value), domain)?);
                }
                _ => unreachable!("layout construction accepts only array and ragged boundaries"),
            }
        }
        Ok(values)
    }
}

/// Enumerates descriptor and dimension-item identities without interpreting padded storage axes as live extents.
fn indexed_identities(types: &[ArrayIrType]) -> impl Iterator<Item = &DimensionVariable> {
    types.iter().filter_map(|r#type| <&RaggedType>::try_from(r#type).ok()).flat_map(|r#type| {
        let item = match r#type.value_kind() {
            RaggedValueKind::Array => None,
            RaggedValueKind::Dimension(item) => item.variable(),
        };
        r#type.ragged_dimensions().iter().map(|dimension| dimension.dimension()).chain(item)
    })
}

/// Dense invocation already owns resharding. Compare logical relationships at the declared placement while leaving
/// each actual leaf handle and every non-sharding part of its type unchanged for the dense invocation path.
fn invocation_type(declared: &ArrayIrType, actual: &ArrayIrType) -> Result<ArrayIrType, ProgramError> {
    let sharding_error = |error: ryft_core::ShardingError| TypeError::invalid(error.to_string());
    match (declared, actual) {
        (ArrayIrType::Array(declared), ArrayIrType::Array(actual)) => {
            Ok(actual.clone().with_sharding(declared.sharding().cloned()).map_err(sharding_error)?.into())
        }
        (ArrayIrType::Ragged(declared), ArrayIrType::Ragged(actual)) => {
            let storage = actual
                .storage_type()
                .clone()
                .with_sharding(declared.storage_type().sharding().cloned())
                .map_err(sharding_error)?;
            Ok(RaggedType::new(storage, actual.value_kind().clone(), actual.ragged_dimensions().to_vec())?.into())
        }
        _ => Ok(actual.clone()),
    }
}

#[cfg(test)]
mod tests {
    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_core::{
        Array, ArrayBatch, ArrayBatchingPolicy, ArrayIrBatch, ArrayIrBatchingPolicy, ArrayOperation, BatchAxis,
        BatchingContext, BatchingPolicyProjection, CapturingContext, Compare, ComparisonDirection, CompilationDomain,
        CompiledFunctionDispatcher, DataType, Device, DeviceMesh, DifferentiableType, DimensionBounds,
        DimensionConstant, DimensionFromScalarOperation, DimensionToScalar, DimensionType, DimensionVariable,
        DynamicArrayExtentBatchingPolicy, EffectClass, EffectClasses, LinearCallOperation, LogicalMesh, MeshAxis,
        MeshAxisType, Print, ProjectedContext, RaggedArrayExtentBatchingPolicy, RaggedDimension, RaggedExtents,
        RaggedStorage, RaggedValueKind, RaggedZeroOperation, Reduce, ReductionKind, Shape, Sharding, StagedFunction,
        Value, call_function, differentiate_at, stage_function, try_jit_with_options,
    };

    use crate::experimental::domains::XlaDomainError;
    use crate::operations::debugging::with_captured_prints;
    use crate::operations::jit_call::collect_jit_call_functions;
    use crate::{FromPjrt, XlaOptions, XlaSession};

    use super::*;

    /// Compilation tracer whose capture values share the invocation's client lifetime.
    type RuntimeLeaf<'c> = ryft_core::CompilationTracer<XlaDomain<'c>>;

    /// Uses the shared CPU client's first device for exact single-device transport checks.
    fn cpu_mesh(client: &ryft_pjrt::Client<'_>) -> ryft_core::DeviceMesh {
        let logical = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let device = Device::from_pjrt(&client.addressable_devices().unwrap()[0]).unwrap();
        DeviceMesh::new(logical, vec![device]).unwrap()
    }

    /// Reads logical integer payloads through the existing materializer, preserving empty-output producer errors.
    fn read_i64(value: &XlaValue<'_>) -> Vec<i64> {
        let array = <XlaValue<'_> as ValueProjection<ArrayType>>::projected(value).unwrap();
        let bytes = crate::arrays_v0::host::materialize_dense_array_bytes(array).unwrap();
        crate::tests::values_from_bytes(bytes.as_slice())
    }

    /// Uploads checked rows `[0, 1]` over nonzero padded storage in the supplied session.
    fn runtime_ragged<'c>(domain: &XlaDomain<'c>) -> XlaValue<'c> {
        let storage = domain
            .lift(ryft_core::Array::matrix(2, 3, vec![7.0f32, 11.0, 101.0, 13.0, 103.0, 107.0]).unwrap().into())
            .unwrap();
        let extents = domain.lift(ryft_core::Array::vector(vec![0i64, 1]).unwrap().into()).unwrap();
        storage.ragged_new(representation(), &[extents]).unwrap()
    }

    /// Builds a counter-controlled body that increments extents under fresh local indexed identities.
    fn changing_extent_while_regions(state_types: Vec<ArrayIrType>) -> (FlatXlaProgram, FlatXlaProgram) {
        let (_, condition) = LeafContext::trace_with_context_and_named_axes(
            |context, input: Vec<Leaf>| {
                let two = context.lift(ryft_core::Array::scalar(2i64)?.into())?;
                Ok(vec![input[0].compare(&two, ComparisonDirection::LessThan)?])
            },
            state_types.clone(),
            vec![],
        )
        .unwrap();
        let input_type = <&RaggedType>::try_from(&state_types[1]).unwrap();
        let next_rows = DimensionVariable::new("next_rows", input_type.ragged_dimensions()[0].dimension().bounds());
        let output = RaggedType::new(
            input_type.storage_type().clone(),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, next_rows, vec![0])],
        )
        .unwrap();
        let (_, body) = LeafContext::trace_with_context_and_named_axes(
            |context, input: Vec<Leaf>| {
                let one = context.lift(ryft_core::Array::scalar(1i64)?.into())?;
                let counter = input[0].add(&one)?;
                let extents = input[1].ragged_extents(1)?.add(&one)?;
                let ragged = input[1].ragged_storage()?.ragged_new(output, &[extents])?;
                Ok(vec![counter, ragged])
            },
            state_types,
            vec![],
        )
        .unwrap();
        (condition, body)
    }

    #[test]
    fn test_eager_and_compiled_while_transport_changing_extents() {
        /// Keeps the compilation tracer and runtime value tied to the same client lifetime.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let initial = runtime_ragged(domain);
            let counter = domain.lift(ryft_core::Array::scalar(0i64).unwrap().into()).unwrap();
            let types = vec![counter.r#type().into_owned(), initial.r#type().into_owned()];
            let (condition, body) = changing_extent_while_regions(types);
            let operation = XlaOperation::While(
                WhileOperation::new().with_body_type_signature(&body.input_types(), &body.output_types()).unwrap(),
            );
            let compiled_operation = operation.clone();
            let compiled_regions = vec![condition.clone(), body.clone()];
            let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                try_jit_with_options(
                    domain,
                    move |(), inputs: Vec<RuntimeLeaf<'c>>| {
                        let context = inputs[0].domain();
                        let mut outputs =
                            context.bind(compiled_operation.clone(), compiled_regions.clone(), &inputs)?;
                        outputs.push(outputs[1].ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?);
                        Ok(outputs)
                    },
                    XlaOptions::new(mesh),
                );
            for (start, expected_extents, expected_sum) in [(0i64, vec![2i64, 3], 5i64), (2, vec![0, 1], 1)] {
                let counter = domain.lift(ryft_core::Array::scalar(start).unwrap().into()).unwrap();
                let eager = domain
                    .bind(operation.clone(), vec![condition.clone(), body.clone()], &[counter.clone(), initial.clone()])
                    .unwrap();
                assert_eq!(read_i64(&eager[0]), [2]);
                let extents = eager[1].ragged_extents(1).unwrap();
                assert_eq!(read_i64(&extents), expected_extents);
                assert_eq!(read_i64(&extents.reduce(&[0], ReductionKind::Sum).unwrap()), [expected_sum]);
                assert!(Arc::ptr_eq(eager[1].xla_domain().session(), domain.session()));
                let result = compiled.call((), vec![counter, initial.clone()]).unwrap();
                assert_eq!(read_i64(&result[0]), [2]);
                assert_eq!(read_i64(&result[1].ragged_extents(1).unwrap()), expected_extents);
                assert_eq!(read_i64(&result[2]), [expected_sum]);
                for output in [&eager[1], &result[1]] {
                    let output_type = output.r#type();
                    let input_type = initial.r#type();
                    let output_type = <&RaggedType>::try_from(output_type.as_ref()).unwrap();
                    let input_type = <&RaggedType>::try_from(input_type.as_ref()).unwrap();
                    assert_ne!(
                        output_type.ragged_dimensions()[0].dimension(),
                        input_type.ragged_dimensions()[0].dimension()
                    );
                }
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_jit_grouped_captures_inline_and_deduplicated_execution() {
        /// Keeps the compilation tracer and runtime value tied to the same client lifetime.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let source = runtime_ragged(domain);
            let offset = domain.lift(ryft_core::Array::scalar(2i64).unwrap().into()).unwrap();
            let input_types = vec![source.r#type().into_owned(), offset.r#type().into_owned()];
            for capture_constant in [false, true] {
                let (_, callee) = LeafContext::trace_with_context_and_named_axes(
                    |context, input: Vec<Leaf>| {
                        let capture = if capture_constant {
                            context.lift(XlaConstant::Captured(CaptureReference::new(0, input_types[0].clone())))?
                        } else {
                            input[0].clone()
                        };
                        Ok(vec![capture.clone(), capture.ragged_extents(1)?.add(&input[1])?])
                    },
                    input_types.clone(),
                    vec![],
                )
                .unwrap();
                let operation = XlaOperation::JitCall(JitCallOperation::new(1));
                let eager =
                    domain.bind(operation.clone(), vec![callee.clone()], &[source.clone(), offset.clone()]).unwrap();
                assert_eq!(read_i64(&eager[0].ragged_extents(1).unwrap()), [0, 1]);
                assert_eq!(read_i64(&eager[1]), [2, 3]);
                assert!(Arc::ptr_eq(eager[0].xla_domain().session(), domain.session()));
                let (_, parent) = LeafContext::trace_with_context_and_named_axes(
                    |context, input: Vec<Leaf>| {
                        let mut result = Vec::new();
                        for _ in 0..2 {
                            result.extend(context.bind(operation.clone(), vec![callee.clone()], &input)?);
                        }
                        Ok(result)
                    },
                    input_types.clone(),
                    vec![],
                )
                .unwrap();
                let grouped = legalize_ragged_program(&parent, 0).unwrap().unwrap();
                assert_eq!(
                    collect_jit_call_functions(&grouped.program).unwrap().functions.len(),
                    if capture_constant { 0 } else { 1 }
                );
                // Register concrete captures through the public staging API before crossing constant universes.
                // A manually fabricated XlaConstant capture reference has no table in the runtime-value trace.
                let captured_source = source.clone();
                let staged: StagedFunction<XlaDomain<'c>, Vec<ArrayIrType>, Vec<ArrayIrType>> = stage_function(
                    domain,
                    move |input: Vec<RuntimeLeaf<'c>>| {
                        let capture = if capture_constant {
                            let context = input[0].domain();
                            let reference = context.capture(captured_source).unwrap();
                            context.lift(reference).unwrap()
                        } else {
                            input[0].clone()
                        };
                        vec![capture.clone(), capture.ragged_extents(1).unwrap().add(&input[1]).unwrap()]
                    },
                    input_types.clone(),
                    XlaOptions::new(mesh.clone()),
                )
                .unwrap();
                let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                    try_jit_with_options(
                        domain,
                        move |(), inputs: Vec<RuntimeLeaf<'c>>| {
                            let context = inputs[0].domain();
                            let mut result = Vec::new();
                            for _ in 0..2 {
                                let call = staged.call_in_context(&context, inputs.clone())?;
                                let shape_sum = call[0].ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?;
                                let shifted_sum = call[1].reduce(&[0], ReductionKind::Sum)?;
                                result.push(shape_sum.add(&shifted_sum)?);
                                result.push(call[0].clone());
                            }
                            Ok(result)
                        },
                        XlaOptions::new(mesh.clone()),
                    );
                let result = compiled.call((), vec![source.clone(), offset.clone()]).unwrap();
                assert_eq!(read_i64(&result[0]), [6]);
                assert_eq!(read_i64(&result[2]), [6]);
                assert_eq!(read_i64(&result[1].ragged_extents(1).unwrap()), [0, 1]);
                assert_eq!(read_i64(&result[3].ragged_extents(1).unwrap()), [0, 1]);
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_scan_shared_and_permuted_nested_extent_transport() {
        /// Keeps the compilation tracer and runtime value tied to the same client lifetime.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            for (length, reverse) in [(2usize, false), (2, true), (0, false)] {
                let rows = DimensionVariable::new("stack_rows", DimensionBounds::non_negative(Some(3)).unwrap());
                let columns = DimensionVariable::new("stack_columns", DimensionBounds::non_negative(Some(4)).unwrap());
                let stacked = RaggedType::new(
                    ArrayType::new_static(DataType::F32, [length, 3, 2, 3]),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(2, rows, vec![1]), RaggedDimension::new(3, columns, vec![2, 0, 1])],
                )
                .unwrap();
                let storage = domain
                    .lift(
                        Array::from_elements(stacked.storage_type().clone(), &vec![7f32; length * 18]).unwrap().into(),
                    )
                    .unwrap();
                let row_values = domain.lift(Array::vector(vec![1i64, 2, 0]).unwrap().into()).unwrap();
                let columns = if length == 0 { vec![] } else { vec![1i64, 2, 0, 2, 1, 0, 999, 2, 999, 999, 1, 999] };
                let column_values = domain
                    .lift(Array::from_elements(stacked.extent_types()[1].clone(), &columns).unwrap().into())
                    .unwrap();
                let source = storage.ragged_new(stacked, &[row_values, column_values]).unwrap();
                let source_type = source.r#type();
                let source_type = <&RaggedType>::try_from(source_type.as_ref()).unwrap();
                let slice_storage = source_type.storage_type().without_dimension(0).unwrap().0;
                let slice = |prefix: &str| {
                    RaggedType::new(
                        slice_storage.clone(),
                        RaggedValueKind::Array,
                        vec![
                            RaggedDimension::new(
                                1,
                                DimensionVariable::new(
                                    format!("{prefix}_rows"),
                                    DimensionBounds::non_negative(Some(3)).unwrap(),
                                ),
                                vec![0],
                            ),
                            RaggedDimension::new(
                                2,
                                DimensionVariable::new(
                                    format!("{prefix}_columns"),
                                    DimensionBounds::non_negative(Some(4)).unwrap(),
                                ),
                                vec![1, 0],
                            ),
                        ],
                    )
                    .unwrap()
                };
                let input_type = RaggedType::new(
                    slice_storage.clone(),
                    RaggedValueKind::Array,
                    vec![
                        RaggedDimension::new(1, source_type.ragged_dimensions()[0].dimension().clone(), vec![0]),
                        RaggedDimension::new(2, source_type.ragged_dimensions()[1].dimension().clone(), vec![1, 0]),
                    ],
                )
                .unwrap();
                let output_type = slice("next");
                let (_, body) = LeafContext::trace(
                    |input: Vec<Leaf>| {
                        let storage = input[1].ragged_storage()?;
                        let rows = input[1].ragged_extents(1)?;
                        let columns = input[1].ragged_extents(2)?;
                        Ok(vec![storage.ragged_new(output_type, &[rows, columns])?])
                    },
                    vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64)), ArrayIrType::Ragged(input_type)],
                )
                .unwrap();
                let scan = ScanOperation::<ArrayIrType>::new(0, length)
                    .with_reverse(reverse)
                    .with_body_type_signature(&body.input_types(), &body.output_types())
                    .unwrap();
                let eager = domain
                    .bind(XlaOperation::Scan(scan.clone()), vec![body.clone()], std::slice::from_ref(&source))
                    .unwrap();
                let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), ArrayIrType, Vec<ArrayIrType>> =
                    try_jit_with_options(
                        domain,
                        move |(), input: RuntimeLeaf<'c>| {
                            let context = input.domain();
                            let mut result =
                                context.bind(XlaOperation::Scan(scan.clone()), vec![body.clone()], &[input])?;
                            result.push(result[0].ragged_extents(2)?.reduce(&[0, 1], ReductionKind::Sum)?);
                            let source = ArrayIrBatch::new(result[0].clone(), BatchAxis::new(1))?;
                            let projected =
                                <ArrayIrBatchingPolicy as BatchingPolicyProjection<_, ArrayType>>::project_batch(
                                    &context, &source,
                                )?;
                            let batching =
                                BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                                    ProjectedContext::new(context.clone()),
                                    context.dimension_constant(3)?,
                                );
                            // The child extent has a permuted scan coordinate; masking must include its row ancestor.
                            let masked = DynamicArrayExtentBatchingPolicy::mask_reduction_input(
                                &batching,
                                &projected,
                                &[3],
                                ReductionKind::Sum,
                            )?;
                            let reduced =
                                ArrayBatch::new(masked.value().reduce_sum(&[2, 3], None)?, BatchAxis::new(1))?;
                            result.push(
                                <ArrayIrBatchingPolicy as BatchingPolicyProjection<_, ArrayType>>::lift_batch(
                                    &context,
                                    &reduced,
                                    &[source],
                                )?
                                .into_value(),
                            );
                            Ok(result)
                        },
                        XlaOptions::new(mesh.clone()),
                    );
                let compiled = compiled.call((), source.clone()).unwrap();
                for result in [&eager[0], &compiled[0]] {
                    let rows = result.ragged_extents(2).unwrap();
                    let columns = result.ragged_extents(3).unwrap();
                    assert_eq!(read_i64(&rows), if length == 0 { vec![] } else { vec![1, 2, 0, 1, 2, 0] });
                    assert_eq!(
                        read_i64(&columns),
                        if length == 0 { vec![] } else { vec![1, 2, 0, 999, 2, 999, 2, 1, 0, 999, 1, 999] }
                    );
                    let r#type = result.r#type();
                    let r#type = <&RaggedType>::try_from(r#type.as_ref()).unwrap();
                    assert_eq!(r#type.ragged_dimensions()[0].extent_axes(), &[0, 1]);
                    assert_eq!(r#type.ragged_dimensions()[1].extent_axes(), &[0, 2, 1]);
                }
                assert_eq!(read_i64(&compiled[1]), [if length == 0 { 0 } else { 6 }]);
                let live_sums = <XlaValue<'_> as ValueProjection<ArrayType>>::projected(&compiled[2]).unwrap();
                assert_eq!(live_sums.r#type().shape().dimensions(), &[Dimension::Static(length), Dimension::Static(3)]);
                let bytes = live_sums
                    .addressable_shards()
                    .next()
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                assert_eq!(
                    crate::tests::values_from_bytes::<f32>(bytes.as_slice()),
                    if length == 0 { vec![] } else { vec![7f32, 28., 0., 14., 14., 0.] }
                );
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_dynamic_scan_runtime_length_reuses_compiled_dispatcher_on_cpu() {
        /// Uses one bounded signature and one dispatcher for nonempty and zero-trip physical values.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let length = DimensionVariable::new("runtime_scan_length", DimensionBounds::non_negative(Some(3)).unwrap());
            let rows = DimensionVariable::new("runtime_scan_rows", DimensionBounds::non_negative(Some(4)).unwrap());
            let declared = RaggedType::new(
                ArrayType::new(DataType::F32, Shape::new(vec![length.clone().into(), 2.into(), 3.into()])),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(2, rows.clone(), vec![0, 1])],
            )
            .unwrap();
            let slice = |name: &str| {
                RaggedType::new(
                    ArrayType::new_static(DataType::F32, [2, 3]),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(
                        1,
                        DimensionVariable::new(name, DimensionBounds::non_negative(Some(4)).unwrap()),
                        vec![0],
                    )],
                )
                .unwrap()
            };
            let output_type = slice("runtime_next_rows");
            let (_, body) = LeafContext::trace(
                |input: Vec<Leaf>| {
                    Ok(vec![input[1].ragged_storage()?.ragged_new(output_type, &[input[1].ragged_extents(1)?])?])
                },
                vec![
                    ArrayIrType::Array(ArrayType::scalar(DataType::I64)),
                    ArrayIrType::Ragged(
                        RaggedType::new(
                            ArrayType::new_static(DataType::F32, [2, 3]),
                            RaggedValueKind::Array,
                            vec![RaggedDimension::new(1, rows.clone(), vec![0])],
                        )
                        .unwrap(),
                    ),
                ],
            )
            .unwrap();
            let scan = ScanOperation::<ArrayIrType>::new(0, Dimension::Dynamic(length))
                .with_body_type_signature(&body.input_types(), &body.output_types())
                .unwrap();
            let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), ArrayIrType, Vec<ArrayIrType>> =
                try_jit_with_options(
                    domain,
                    move |(), input: RuntimeLeaf<'c>| {
                        let context = input.domain();
                        let length = input.ragged_storage()?.dimension_size(0)?;
                        let mut output =
                            context.bind(XlaOperation::Scan(scan.clone()), vec![body.clone()], &[input, length])?;
                        output.push(output[0].ragged_extents(2)?.reduce(&[0, 1], ReductionKind::Sum)?);
                        Ok(output)
                    },
                    XlaOptions::new(mesh.clone()),
                )
                .with_staged_input_types(Some(ArrayIrType::Ragged(declared)))
                .unwrap();
            for length in [2usize, 0] {
                let actual = RaggedType::new(
                    ArrayType::new_static(DataType::F32, [length, 2, 3]).replicated(&mesh).unwrap(),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(2, rows.clone(), vec![0, 1])],
                )
                .unwrap();
                let storage = XlaArray::from_host_buffer(
                    domain,
                    actual.storage_type().clone(),
                    mesh.clone(),
                    crate::tests::values_to_bytes(&vec![7f32; length * 6]),
                )
                .unwrap();
                let extents = if length == 0 { vec![] } else { vec![0i64, 1, 2, 3] };
                let payload = XlaArray::from_host_buffer(
                    domain,
                    actual.extent_types()[0].clone(),
                    mesh.clone(),
                    crate::tests::values_to_bytes(&extents),
                )
                .unwrap();
                let input = XlaValue::Ragged(RaggedValue::new(actual, storage, vec![payload]).unwrap());
                let output = compiled.call((), input).unwrap();
                let output_type = output[0].r#type();
                let output_type = <&RaggedType>::try_from(output_type.as_ref()).unwrap();
                assert_eq!(output_type.storage_type().dimension(0), Dimension::Static(length));
                let payload = output[0].ragged_extents(2).unwrap();
                let array = <XlaValue<'_> as ValueProjection<ArrayType>>::projected(&payload).unwrap();
                assert_eq!(array.shape().as_slice(), &[length, 2]);
                let buffer = array.addressable_shards().next().unwrap().buffer().unwrap();
                assert_eq!(buffer.unpadded_dimensions().unwrap(), &[length as u64, 2]);
                assert_eq!(read_i64(&payload), extents);
                assert_eq!(read_i64(&output[1]), [if length == 0 { 0 } else { 6 }]);
                assert_eq!(compiled.specialization_count(), 1);
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_eager_dynamic_scan_carry_omits_host_dimension_slot() {
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh);
        let input = runtime_ragged(&domain);
        let (_, body) = LeafContext::trace(
            |inputs: Vec<Leaf>| Ok(vec![inputs[1].clone()]),
            vec![ArrayType::scalar(DataType::I64).into(), input.r#type().into_owned()],
        )
        .unwrap();
        let length = DimensionVariable::new("length", DimensionBounds::non_negative(Some(3)).unwrap());
        let operation = ScanOperation::<ArrayIrType>::new(1, Dimension::Dynamic(length))
            .with_body_type_signature(&body.input_types(), &body.output_types())
            .unwrap();
        for length in [2usize, 0] {
            let extent = domain.lift(ryft_core::DimensionValue::constant(length).unwrap().into()).unwrap();
            let outputs = domain
                .bind(XlaOperation::Scan(operation.clone()), vec![body.clone()], &[input.clone(), extent])
                .unwrap();
            assert_eq!(outputs[0].r#type(), input.r#type());
            assert_eq!(read_i64(&outputs[0].ragged_extents(1).unwrap()), [0, 1]);
        }
    }

    /// Uses two independent rows with physical capacity three and a complete row-indexed extent payload.
    fn representation() -> RaggedType {
        let rows = DimensionVariable::new("rows", DimensionBounds::non_negative(Some(4)).unwrap());
        RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 3]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, rows, vec![0])],
        )
        .unwrap()
    }

    #[test]
    fn test_layout_orders_leaves_and_expands_donation() {
        let r#type = representation();
        let scalar = ArrayType::scalar(DataType::I64);
        let layout =
            XlaValueLayout::new(&[scalar.clone().into(), r#type.clone().into(), scalar.clone().into()]).unwrap();
        assert_eq!(
            layout.leaf_types(),
            &[scalar.clone(), r#type.storage_type().clone(), r#type.extent_types()[0].clone(), scalar]
        );
        assert_eq!(layout.leaf_range(1), 1..3);
        assert_eq!(layout.prefix_leaf_count(2), 3);
        assert_eq!(layout.donation_flags(&[true, true, false]).unwrap(), [true, true, false, false]);
        assert_eq!(layout.donation_flags(&[]), Err(ProgramError::InvalidInputCount { expected: 3, actual: 0 }));

        // The storage of first-class dimension items holds extents, so it stays live like every extent payload.
        let items = RaggedType::new(
            ArrayType::new_static(DataType::I64, [2]),
            RaggedValueKind::Dimension(ryft_core::DimensionType::new(
                "items",
                DimensionBounds::non_negative(Some(8)).unwrap(),
            )),
            vec![],
        )
        .unwrap();
        let layout = XlaValueLayout::new(&[items.clone().into(), r#type.clone().into()]).unwrap();
        assert_eq!(
            layout.leaf_types(),
            &[items.storage_type().clone(), r#type.storage_type().clone(), r#type.extent_types()[0].clone()]
        );
        assert_eq!(layout.donation_flags(&[true, true]).unwrap(), [false, true, false]);
    }

    #[test]
    fn test_layout_result_identity_renaming() {
        let declared = representation();
        let identity = declared.ragged_dimensions()[0].dimension().clone();
        let empty = XlaValueLayout::new(&[]).unwrap();
        let outputs = XlaValueLayout::new(&[declared.clone().into(), declared.clone().into()]).unwrap();
        let first = outputs.result_identity_renaming(&empty, &[]).unwrap();
        let second = outputs.result_identity_renaming(&empty, &[]).unwrap();
        assert_ne!(first.target(&identity), Some(&identity));
        assert_ne!(first.target(&identity), second.target(&identity));
        assert_eq!(first.replacements().len(), 1);
        let actual_identity = identity.fresh();
        let mut actual = TypeIdentityRenaming::new();
        actual.insert(identity.clone(), actual_identity.clone()).unwrap();
        let actual = declared.rename_identities(&actual).unwrap();
        let inputs = XlaValueLayout::new(&[declared.into()]).unwrap();
        let forwarded = outputs.result_identity_renaming(&inputs, &[actual.clone().into()]).unwrap();
        assert_eq!(forwarded.target(&identity), Some(&actual_identity));

        // Dense results have no indexed names to choose, even if entry checks relate independent input families.
        let dense = XlaValueLayout::new(&[ArrayType::scalar(DataType::I64).into()]).unwrap();
        let repeated =
            XlaValueLayout::new(&[inputs.logical_types()[0].clone(), inputs.logical_types()[0].clone()]).unwrap();
        assert!(
            dense
                .result_identity_renaming(&repeated, &[inputs.logical_types()[0].clone(), actual.into()])
                .unwrap()
                .is_identity()
        );
    }

    #[test]
    fn test_layout_result_identity_renaming_ignores_unreturned_input_families() {
        let declared = representation();
        let identity = declared.ragged_dimensions()[0].dimension().clone();
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(identity.clone(), identity.fresh()).unwrap();
        let actual = declared.rename_identities(&renaming).unwrap();
        let repeated = XlaValueLayout::new(&[declared.clone().into(), declared.clone().into()]).unwrap();
        let actual_inputs = vec![declared.clone().into(), actual.clone().into()];
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(identity.clone(), identity.fresh()).unwrap();
        let local = declared.rename_identities(&renaming).unwrap();
        let local_identity = local.ragged_dimensions()[0].dimension();
        let independent = XlaValueLayout::new(&[local.clone().into(), local.clone().into()]).unwrap();
        let first = independent.result_identity_renaming(&repeated, &actual_inputs).unwrap();
        let second = independent.result_identity_renaming(&repeated, &actual_inputs).unwrap();
        assert_eq!(first.replacements().len(), 1);
        assert_eq!(first.target(&identity), None);
        assert_ne!(first.target(local_identity), Some(local_identity));
        assert_ne!(first.target(local_identity), second.target(local_identity));

        // An unrelated ambiguous input class also must not block forwarding a different input family.
        let inputs =
            XlaValueLayout::new(&[declared.clone().into(), declared.clone().into(), local.clone().into()]).unwrap();
        let actual_local_identity = local_identity.fresh();
        let mut local_renaming = TypeIdentityRenaming::new();
        local_renaming.insert(local_identity.clone(), actual_local_identity.clone()).unwrap();
        let actual_local = local.rename_identities(&local_renaming).unwrap();
        let forwarded = independent
            .result_identity_renaming(&inputs, &[declared.clone().into(), actual.into(), actual_local.into()])
            .unwrap();
        assert_eq!(forwarded.replacements(), &[(local_identity.clone(), actual_local_identity)]);

        // Choosing a runtime name remains unsupported when the ambiguous family itself crosses the result boundary.
        let ambiguous = XlaValueLayout::new(&[declared.into()]).unwrap();
        assert_eq!(
            ambiguous.result_identity_renaming(&repeated, &actual_inputs),
            Err(TypeError::invalid("a forwarded indexed result requires one unambiguous runtime input identity").into()),
        );
    }

    #[test]
    fn test_layout_result_types() {
        let storage_type = ArrayType::new_static(DataType::I64, [2]);
        let items = |kind: ryft_core::DimensionType| {
            RaggedType::new(storage_type.clone(), RaggedValueKind::Dimension(kind), vec![]).unwrap()
        };
        let item = DimensionVariable::new("item", DimensionBounds::non_negative(Some(8)).unwrap());
        let declared = items(item.clone().into());
        let exact = |extent: usize| items(Dimension::Static(extent).into());
        let inputs = XlaValueLayout::new(&[declared.clone().into()]).unwrap();

        // Exact items replace the formal family in every aliasing result, independently for each execution.
        let aliases = XlaValueLayout::new(&[declared.clone().into(), declared.clone().into()]).unwrap();
        assert_eq!(aliases.result_types(&inputs, &[exact(3).into()]), Ok(vec![exact(3).into(), exact(3).into()]));
        assert_eq!(aliases.result_types(&inputs, &[exact(5).into()]), Ok(vec![exact(5).into(), exact(5).into()]));

        // A dynamic actual family is forwarded under its runtime name.
        let runtime = items(item.fresh().into());
        assert_eq!(
            aliases.result_types(&inputs, &[runtime.clone().into()]),
            Ok(vec![runtime.clone().into(), runtime.into()]),
        );

        // A declared exact item has no family to instantiate.
        let static_inputs = XlaValueLayout::new(&[exact(3).into()]).unwrap();
        assert_eq!(static_inputs.result_types(&static_inputs, &[exact(3).into()]), Ok(vec![exact(3).into()]));

        // Refinement never erases a descriptor naming the family. Each execution gives it one fresh shared name.
        let rows = RaggedType::new(
            ArrayType::new_static(DataType::F32, [2, 7]),
            RaggedValueKind::Array,
            vec![RaggedDimension::new(1, item.clone(), vec![0])],
        )
        .unwrap();
        let outputs = XlaValueLayout::new(&[declared.into(), rows.clone().into(), rows.clone().into()]).unwrap();
        let first = outputs.result_types(&inputs, &[exact(3).into()]).unwrap();
        let second = outputs.result_types(&inputs, &[exact(3).into()]).unwrap();
        let descriptor =
            |r#type: &ArrayIrType| <&RaggedType>::try_from(r#type).unwrap().ragged_dimensions()[0].dimension().clone();
        let first_rows = descriptor(&first[1]);
        assert_ne!(first_rows, item);
        assert_ne!(first_rows, descriptor(&second[1]));
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(item.clone(), first_rows).unwrap();
        let renamed = ArrayIrType::Ragged(rows.rename_identities(&renaming).unwrap());
        assert_eq!(first, vec![exact(3).into(), renamed.clone(), renamed]);
    }

    #[test]
    fn test_invocation_comparison_preserves_dense_resharding_contract() {
        let declared = representation();
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 1, MeshAxisType::Auto).unwrap()]).unwrap();
        let storage = declared.storage_type().clone().with_sharding(Sharding::replicated(mesh, 2)).unwrap();
        let actual =
            RaggedType::new(storage.clone(), declared.value_kind().clone(), declared.ragged_dimensions().to_vec())
                .unwrap();
        assert_eq!(
            invocation_type(&declared.clone().into(), &actual.clone().into()).unwrap(),
            ArrayIrType::Ragged(declared.clone())
        );
        assert_eq!(
            invocation_type(&declared.storage_type().clone().into(), &storage.clone().into()).unwrap(),
            ArrayIrType::Array(declared.storage_type().clone())
        );
        assert!(actual.storage_type().sharding().is_some());
        assert_eq!(actual.ragged_dimensions(), declared.ragged_dimensions());
    }

    #[test]
    fn test_dense_replay_keeps_original_program() {
        let (_, program) =
            LeafContext::trace(|input: Vec<Leaf>| Ok(input), vec![ArrayIrType::from(ArrayType::scalar(DataType::F32))])
                .unwrap();
        let rendered = program.to_string();
        assert!(legalize_ragged_program(&program, 0).unwrap().is_none());
        assert_eq!(program.to_string(), rendered);
    }

    #[test]
    fn test_checked_new_and_projection_replay_retains_checks() {
        let r#type = representation();
        let inputs = leaf_types(&r#type.clone().into());
        let (_, program) = LeafContext::trace(
            |input: Vec<Leaf>| {
                let ragged = input[0].ragged_new(r#type.clone(), &input[1..])?;
                Ok(vec![ragged.ragged_extents(-1)?, ragged.ragged_storage()?])
            },
            inputs,
        )
        .unwrap();
        let legalized = legalize_ragged_program(&program, 0).unwrap().unwrap();
        assert!(
            legalized.program.regions().iter().all(|region| region
                .atoms()
                .iter()
                .all(|atom| !matches!(atom.r#type().as_ref(), ArrayIrType::Ragged(_))))
        );
        assert!(legalized.program.regions().iter().any(|region| {
            region.instructions().iter().any(|instruction| {
                matches!(
                    instruction.operation(),
                    XlaOperation::Array(ArrayOperation::Assert(operation))
                        if operation.message() == "ragged extent on storage axis 1 is outside its declared bounds"
                )
            })
        }));
        assert_eq!(legalized.program.effects().classes(), EffectClasses::single(EffectClass::OrderedAssertion));
        assert!(legalized.program.regions().iter().all(|region| {
            region
                .instructions()
                .iter()
                .all(|instruction| !matches!(instruction.operation(), XlaOperation::RaggedNew(_)))
        }));
        assert_eq!(legalized.program.output_types(), program.output_types());
    }

    #[test]
    fn test_opaque_ragged_projection_uses_canonical_input_leaves() {
        let r#type = representation();
        let (_, program) = LeafContext::trace(
            |input: Vec<Leaf>| Ok(vec![input[0].ragged_extents(1)?, input[0].ragged_storage()?]),
            vec![ArrayIrType::Ragged(r#type.clone())],
        )
        .unwrap();
        let legalized = legalize_ragged_program(&program, 0).unwrap().unwrap();
        assert_eq!(legalized.program.input_types(), leaf_types(&r#type.into()));
        assert_eq!(
            legalized.program.output_ids(),
            &[legalized.program.input_ids()[1], legalized.program.input_ids()[0]]
        );
        assert_eq!(legalized.program.instructions().len(), 0);
    }

    #[test]
    fn test_jit_deduplication_retains_logical_group_schema() {
        let descriptor = |name, axes| {
            RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 2, 3]),
                RaggedValueKind::Array,
                vec![RaggedDimension::new(
                    2,
                    DimensionVariable::new(name, DimensionBounds::non_negative(Some(4)).unwrap()),
                    axes,
                )],
            )
            .unwrap()
        };
        let left = descriptor("left", vec![0]);
        let right = descriptor("right", vec![1]);
        assert_eq!(leaf_types(&left.clone().into()), leaf_types(&right.clone().into()));
        let identity = |r#type: RaggedType| {
            LeafContext::trace(|input: Vec<Leaf>| Ok(input), vec![ArrayIrType::Ragged(r#type)]).unwrap().1
        };
        let left_callee = identity(left.clone());
        let right_callee = identity(right.clone());
        let (_, program) = LeafContext::trace_with_context_and_named_axes(
            |context, input: Vec<Leaf>| {
                let mut outputs = Vec::new();
                for (index, callee) in [(0, &left_callee), (0, &left_callee), (1, &right_callee), (1, &right_callee)] {
                    outputs.extend(context.bind(
                        XlaOperation::JitCall(JitCallOperation::new(0)),
                        vec![callee.clone()],
                        &[input[index].clone()],
                    )?);
                }
                Ok(outputs)
            },
            vec![ArrayIrType::Ragged(left), ArrayIrType::Ragged(right)],
            vec![],
        )
        .unwrap();
        let legalized = legalize_ragged_program(&program, 0).unwrap().unwrap();
        let functions = collect_jit_call_functions(&legalized.program).unwrap();
        assert_eq!(functions.functions.len(), 2);
        assert_eq!(functions.order.len(), 2);
    }

    #[test]
    fn test_jit_local_result_identity_inference_and_renaming() {
        let r#type = representation();
        let (_, callee) = LeafContext::trace(
            |input: Vec<Leaf>| Ok(vec![input[0].ragged_new(r#type.clone(), &input[1..])?]),
            leaf_types(&r#type.clone().into()),
        )
        .unwrap();
        let inputs = callee.input_types();
        let outputs = callee.output_types();
        let left = JitCallOperation::new(0).with_callee_type_signature(&inputs, &outputs).unwrap();
        let right = JitCallOperation::new(0).with_callee_type_signature(&inputs, &outputs).unwrap();
        let interface = callee.interface();
        assert!(JitCallOperation::new(0).infer_output_types(&inputs, std::slice::from_ref(&interface)).is_err());
        let inferred = left.infer_output_types(&inputs, std::slice::from_ref(&interface)).unwrap();
        assert_eq!(left.infer_output_types(&inputs, std::slice::from_ref(&interface)).unwrap(), inferred);
        let other = right.infer_output_types(&inputs, std::slice::from_ref(&interface)).unwrap();
        let identity = |types: &[ArrayIrType]| {
            <&RaggedType>::try_from(&types[0]).unwrap().ragged_dimensions()[0].dimension().clone()
        };
        assert_ne!(identity(&inferred), identity(&other));
        assert_ne!(identity(&inferred), identity(&outputs));
        assert_eq!(left.defined_type_identities().as_ref(), &[identity(&inferred)]);
        let replacement = DimensionVariable::new("exported_rows", identity(&inferred).bounds());
        let mut renaming = TypeIdentityRenaming::new();
        renaming.insert(identity(&inferred), replacement.clone()).unwrap();
        let renamed = left.rename_type_identities(&renaming).unwrap();
        assert_eq!(identity(&renamed.infer_output_types(&inputs, &[interface]).unwrap()), replacement);
    }

    #[test]
    fn test_staged_jit_calls_keep_different_runtime_extent_families() {
        /// Keeps the compilation tracer and runtime value tied to the same client lifetime.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let storage = domain.lift(ryft_core::Array::matrix(2, 3, vec![7f32; 6]).unwrap().into()).unwrap();
            let left = domain.lift(ryft_core::Array::vector(vec![0i64, 1]).unwrap().into()).unwrap();
            let right = domain.lift(ryft_core::Array::vector(vec![2i64, 3]).unwrap().into()).unwrap();
            let types = vec![storage.r#type().into_owned(), left.r#type().into_owned()];
            let staged: StagedFunction<XlaDomain<'c>, Vec<ArrayIrType>, ArrayIrType> = stage_function(
                domain,
                |inputs: Vec<RuntimeLeaf<'c>>| inputs[0].ragged_new(representation(), &inputs[1..]).unwrap(),
                types,
                XlaOptions::new(mesh.clone()),
            )
            .unwrap();
            let parent: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                try_jit_with_options(
                    domain,
                    move |(), inputs: Vec<RuntimeLeaf<'c>>| {
                        let context = inputs[0].domain();
                        let left = staged.call_in_context(&context, vec![inputs[0].clone(), inputs[1].clone()])?;
                        let right = staged.call_in_context(&context, vec![inputs[0].clone(), inputs[2].clone()])?;
                        let left_sum = left.ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?;
                        let right_sum = right.ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?;
                        Ok(vec![left, right, left_sum, right_sum])
                    },
                    XlaOptions::new(mesh),
                );
            let outputs = parent.call((), vec![storage, left, right]).unwrap();
            assert_eq!(read_i64(&outputs[0].ragged_extents(1).unwrap()), [0, 1]);
            assert_eq!(read_i64(&outputs[1].ragged_extents(1).unwrap()), [2, 3]);
            assert_eq!(read_i64(&outputs[2]), [1]);
            assert_eq!(read_i64(&outputs[3]), [5]);
            let left = outputs[0].r#type();
            let right = outputs[1].r#type();
            let left = <&RaggedType>::try_from(left.as_ref()).unwrap();
            let right = <&RaggedType>::try_from(right.as_ref()).unwrap();
            assert_ne!(left.ragged_dimensions()[0].dimension(), right.ragged_dimensions()[0].dimension());
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_compiled_executions_own_local_identities_and_forward_runtime_input_identities() {
        /// Keeps the compilation tracer and runtime handles tied to the same CPU client.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let producer: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, ArrayIrType> =
                try_jit_with_options(
                    domain,
                    |(), inputs: Vec<RuntimeLeaf<'c>>| Ok(inputs[0].ragged_new(representation(), &inputs[1..])?),
                    XlaOptions::new(mesh.clone()),
                );
            let consumer: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                try_jit_with_options(
                    domain,
                    |(), inputs: Vec<RuntimeLeaf<'c>>| {
                        Ok(vec![
                            inputs[0].clone(),
                            inputs[0].clone(),
                            inputs[1].clone(),
                            inputs[0].ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?,
                            inputs[1].ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?,
                        ])
                    },
                    XlaOptions::new(mesh),
                );
            let storage = domain.lift(ryft_core::Array::matrix(2, 3, vec![7f32; 6]).unwrap().into()).unwrap();
            let left_extents = domain.lift(ryft_core::Array::vector(vec![1i64, 2]).unwrap().into()).unwrap();
            let right_extents = domain.lift(ryft_core::Array::vector(vec![3i64, 0]).unwrap().into()).unwrap();
            let left = producer.call((), vec![storage.clone(), left_extents]).unwrap();
            let right = producer.call((), vec![storage, right_extents]).unwrap();
            assert_ne!(left.r#type(), right.r#type());
            let first = consumer.call((), vec![left.clone(), right.clone()]).unwrap();
            assert_eq!(first[0].r#type(), left.r#type());
            assert_eq!(first[1].r#type(), left.r#type());
            assert_eq!(first[2].r#type(), right.r#type());
            assert_eq!(read_i64(&first[3]), [3]);
            assert_eq!(read_i64(&first[4]), [3]);
            let second = consumer.call((), vec![right.clone(), left.clone()]).unwrap();
            assert_eq!(second[0].r#type(), right.r#type());
            assert_eq!(second[2].r#type(), left.r#type());
            assert_eq!(read_i64(&second[0].ragged_extents(1).unwrap()), [3, 0]);
            assert_eq!(read_i64(&second[2].ragged_extents(1).unwrap()), [1, 2]);
            assert_eq!(producer.specialization_count(), 1);
            assert_eq!(consumer.specialization_count(), 1);
            assert_eq!(consumer.statistics().traces, 1);

            let before = domain.eager_dispatch_statistics();
            for value in [left, right] {
                let target = <&RaggedType>::try_from(value.r#type().as_ref()).unwrap().clone();
                let outputs = domain.bind(RaggedZeroOperation::new(target), vec![], &[value.clone()]).unwrap();
                assert_eq!(outputs[0].r#type(), value.r#type());
                assert_eq!(
                    read_i64(&outputs[0].ragged_extents(1).unwrap()),
                    read_i64(&value.ragged_extents(1).unwrap())
                );
            }
            let after = domain.eager_dispatch_statistics();
            assert_eq!(after.misses - before.misses, 1);
            assert_eq!(after.hits - before.hits, 1);
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_jit_call_repeated_alpha_inputs_preserve_entry_checks() {
        /// Compares one tracer passed twice with separate untrusted values under the same fresh input scope.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let left = runtime_ragged(domain);
            let actual_type = left.r#type();
            let actual = <&RaggedType>::try_from(actual_type.as_ref()).unwrap().clone();
            let rows = actual.ragged_dimensions()[0].dimension();
            let mut renaming = TypeIdentityRenaming::new();
            renaming.insert(rows.clone(), rows.fresh()).unwrap();
            let formal = actual.rename_identities(&renaming).unwrap();
            let extent = &actual.extent_types()[0];
            let (_, callee) = LeafContext::trace(
                |inputs: Vec<Leaf>| Ok(vec![inputs[0].ragged_extents(1)?, inputs[1].ragged_extents(1)?]),
                vec![ArrayIrType::Ragged(formal); 2],
            )
            .unwrap();
            for duplicate in [true, false] {
                let operation = JitCallOperation::new(0);
                let (_, parent) = LeafContext::trace_with_context_and_named_axes(
                    |context, inputs: Vec<Leaf>| {
                        let right = if duplicate { &inputs[0] } else { &inputs[1] };
                        context.bind(operation.clone(), vec![callee.clone()], &[inputs[0].clone(), right.clone()])
                    },
                    vec![ArrayIrType::Ragged(actual.clone()); 2],
                    vec![],
                )
                .unwrap();
                let right = if duplicate { "%0" } else { "%1" };
                assert_eq!(
                    parent.to_string(),
                    indoc::formatdoc! {"
                        lambda %0:{actual}, %1:{actual} .
                        let %2:{extent}, %3:{extent} = jit_call %0 {right} [
                            callee={{
                                lambda %0:{actual}, %1:{actual} .
                                let %2:{extent} = ragged_extents [axis=1] %0
                                    %3:{extent} = ragged_extents [axis=1] %1
                                in (%2, %3)
                            }},
                        ]
                        in (%2, %3)"},
                );
                let compiled_domain = domain.fork_effect_scope();
                let compiled_callee = callee.clone();
                let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                    try_jit_with_options(
                        &compiled_domain,
                        move |(), inputs: Vec<RuntimeLeaf<'c>>| {
                            let right = if duplicate { &inputs[0] } else { &inputs[1] };
                            Ok(inputs[0].domain().bind(
                                operation.clone(),
                                vec![compiled_callee.clone()],
                                &[inputs[0].clone(), right.clone()],
                            )?)
                        },
                        XlaOptions::new(mesh.clone()),
                    );
                let matching = left.clone();
                let outputs = compiled.call((), vec![left.clone(), matching]).unwrap();
                assert_eq!(read_i64(&outputs[0]), [0, 1]);
                assert_eq!(read_i64(&outputs[1]), [0, 1]);
                if !duplicate {
                    let XlaValue::Ragged(raw) = left.clone() else { panic!("expected raw ragged input") };
                    let extents = domain.lift(ryft_core::Array::vector(vec![2i64, 3]).unwrap().into()).unwrap();
                    let XlaValue::Array(extents) = extents else { panic!("expected extent array") };
                    let mismatched = XlaValue::Ragged(
                        RaggedValue::new(raw.r#type().as_ref().clone(), raw.storage().clone(), vec![extents]).unwrap(),
                    );
                    let error = compiled.call((), vec![left.clone(), mismatched]).unwrap_err();
                    assert!(
                        error.to_string().contains("guarded ragged occurrences of `rows` must have equal live extents")
                    );
                }
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_jit_call_aliased_dimension_items_have_one_owner() {
        /// Runs two independent calls exporting aliases of one callee-local dimension item family.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let storage_type = ArrayType::new_static(DataType::I64, [2]);
            let item = ryft_core::DimensionType::new("item", DimensionBounds::non_negative(Some(4)).unwrap());
            let raw = RaggedType::new(storage_type.clone(), RaggedValueKind::Dimension(item), vec![]).unwrap();
            let (_, callee) = LeafContext::trace(
                |inputs: Vec<Leaf>| {
                    let result = inputs[0].ragged_new(raw, &[])?;
                    Ok(vec![result.clone(), result])
                },
                vec![ArrayIrType::Array(storage_type.clone())],
            )
            .unwrap();
            let staged: StagedFunction<XlaDomain<'c>, Vec<ArrayIrType>, Vec<ArrayIrType>> = stage_function(
                domain,
                move |inputs: Vec<RuntimeLeaf<'c>>| {
                    let call = JitCallOperation::new(0)
                        .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                        .unwrap();
                    inputs[0].domain().bind(call, vec![callee.clone()], &inputs).unwrap()
                },
                vec![storage_type.into()],
                XlaOptions::new(mesh),
            )
            .unwrap();
            assert_eq!(
                staged.source_program().program().to_string(),
                indoc! {"
                    lambda %0:i64[2] .
                    let %1:ragged<i64[2]; item=dimension<item ∈ [0, 4)>>, %2:ragged<i64[2]; item=dimension<item ∈ [0, 4)>> = jit_call [
                        output_types=[ragged<i64[2]; item=dimension<item ∈ [0, 4)>>, ragged<i64[2]; item=dimension<item ∈ [0, 4)>>],
                    ] %0 [
                        callee={
                            lambda %0:i64[2] .
                            let %1:ragged<i64[2]; item=dimension<item ∈ [0, 4)>> = ragged_new \
                                [output_type=ragged<i64[2]; item=dimension<item ∈ [0, 4)>>] %0
                            in (%1, %1)
                        },
                    ]
                    in (%1, %2)"},
            );
            let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
            let storage = domain.lift(ryft_core::Array::vector(vec![1i64, 3]).unwrap().into()).unwrap();
            let first = call_function(domain, compiled.executable_function(), vec![storage.clone()]).unwrap();
            let second = call_function(domain, compiled.executable_function(), vec![storage]).unwrap();
            assert_eq!(first[0].r#type(), first[1].r#type());
            assert_eq!(second[0].r#type(), second[1].r#type());
            assert_ne!(first[0].r#type(), second[0].r#type());
            for result in first.iter().chain(&second) {
                assert_eq!(read_i64(&result.ragged_storage().unwrap()), [1, 3]);
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_compiled_independent_results_ignore_unreturned_input_families() {
        /// Uses one explicit formal equality class with independent runtime input names, checked at entry.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let formal = representation();
            let local = representation();
            let staged: StagedFunction<XlaDomain<'c>, Vec<ArrayIrType>, Vec<ArrayIrType>> = stage_function(
                domain,
                move |inputs: Vec<RuntimeLeaf<'c>>| {
                    let value = inputs[0]
                        .ragged_storage()
                        .unwrap()
                        .ragged_new(local, &[inputs[1].ragged_extents(1).unwrap()])
                        .unwrap();
                    vec![value.clone(), value]
                },
                vec![formal.clone().into(), formal.into()],
                XlaOptions::new(mesh),
            )
            .unwrap();
            assert_eq!(
                staged.source_program().program().to_string(),
                indoc! {"
                    lambda %0:ragged<f32[2, 3]; 1:rows over [0]>, %1:ragged<f32[2, 3]; 1:rows over [0]> .
                    let %2:f32[2, 3] = ragged_storage %0
                        %3:i64[2] = ragged_extents [axis=1] %1
                        %4:ragged<f32[2, 3]; 1:rows over [0]> = ragged_new \
                            [output_type=ragged<f32[2, 3]; 1:rows over [0]>] %2 %3
                    in (%4, %4)"},
            );
            let compiled = domain.compile(domain.lower(staged).unwrap()).unwrap();
            let left = runtime_ragged(domain);
            let right = runtime_ragged(domain);
            assert_ne!(left.r#type(), right.r#type());
            let first =
                call_function(domain, compiled.executable_function(), vec![left.clone(), right.clone()]).unwrap();
            let second = call_function(domain, compiled.executable_function(), vec![right, left.clone()]).unwrap();
            assert_eq!(first[0].r#type(), first[1].r#type());
            assert_eq!(second[0].r#type(), second[1].r#type());
            assert_ne!(first[0].r#type(), second[0].r#type());
            assert_ne!(first[0].r#type(), left.r#type());
            assert_eq!(read_i64(&first[0].ragged_extents(1).unwrap()), [0, 1]);
            assert_eq!(read_i64(&second[0].ragged_extents(1).unwrap()), [0, 1]);

            // Ignoring unreturned input names must preserve the formal input class's numerical entry check.
            let other_extents = domain.lift(ryft_core::Array::vector(vec![2i64, 3]).unwrap().into()).unwrap();
            let other = left.ragged_storage().unwrap().ragged_new(representation(), &[other_extents]).unwrap();
            let error = call_function(&domain.fork_effect_scope(), compiled.executable_function(), vec![left, other])
                .unwrap_err();
            assert!(error.to_string().contains("guarded ragged occurrences of `rows` must have equal live extents"));
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_static_dimension_item_results_carry_actual_kind() {
        /// Calls one declared dimension-item family with different exact items through compiled and eager dispatch.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let storage_type = ArrayType::new_static(DataType::I64, [2]);
            let item = DimensionType::new("item", DimensionBounds::non_negative(Some(8)).unwrap());
            let declared = RaggedType::new(storage_type.clone(), RaggedValueKind::Dimension(item), vec![]).unwrap();
            let items = |extent: usize| {
                let r#type = RaggedType::new(
                    storage_type.clone(),
                    RaggedValueKind::Dimension(Dimension::Static(extent).into()),
                    vec![],
                )
                .unwrap();
                let storage = domain.lift(ryft_core::Array::vector(vec![extent as i64; 2]).unwrap().into()).unwrap();
                storage.ragged_new(r#type, &[]).unwrap()
            };
            let three = items(3);
            let five = items(5);

            // One compiled specialization at the declared family returns each call's exact item kind, rather than
            // the executable's formal family, and forwards a dynamic family under its runtime name.
            let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), ArrayIrType, ArrayIrType> =
                try_jit_with_options(domain, |(), input: RuntimeLeaf<'c>| Ok(input), XlaOptions::new(mesh.clone()))
                    .with_staged_input_types(Some(ArrayIrType::Ragged(declared.clone())))
                    .unwrap();
            let compiled_three = compiled.call((), three.clone()).unwrap();
            let compiled_five = compiled.call((), five.clone()).unwrap();
            let dynamic_type = RaggedType::new(
                storage_type.clone(),
                RaggedValueKind::Dimension(DimensionType::new(
                    "runtime_item",
                    DimensionBounds::non_negative(Some(8)).unwrap(),
                )),
                vec![],
            )
            .unwrap();
            let dynamic = domain
                .lift(ryft_core::Array::vector(vec![1i64, 6]).unwrap().into())
                .unwrap()
                .ragged_new(dynamic_type, &[])
                .unwrap();
            let compiled_dynamic = compiled.call((), dynamic.clone()).unwrap();
            assert_eq!(compiled.specialization_count(), 1);
            assert_eq!(compiled_three.r#type(), three.r#type());
            assert_eq!(compiled_five.r#type(), five.r#type());
            assert_eq!(compiled_dynamic.r#type(), dynamic.r#type());
            for (result, items) in [(&compiled_three, [3, 3]), (&compiled_five, [5, 5]), (&compiled_dynamic, [1, 6])] {
                assert_eq!(read_i64(&result.ragged_storage().unwrap()), items);
                assert!(Arc::ptr_eq(result.xla_domain().session(), domain.session()));
            }

            // Results with different exact items do not share a formal name, so one repeated declared family rejects
            // them at dispatch, before any runtime entry check.
            let consumer: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                try_jit_with_options(domain, |(), inputs: Vec<RuntimeLeaf<'c>>| Ok(inputs), XlaOptions::new(mesh))
                    .with_staged_input_types(vec![
                        Some(ArrayIrType::Ragged(declared.clone())),
                        Some(ArrayIrType::Ragged(declared.clone())),
                    ])
                    .unwrap();
            assert!(matches!(
                consumer.call((), vec![compiled_three.clone(), compiled_five]),
                Err(XlaDomainError::Tracing(ProgramError::Type(error)))
                    if error == TypeError::custom(ryft_core::DimensionError::InputDimensionMismatch {
                        dimension: "item".to_owned(),
                        expected: 3,
                        actual: 5,
                    }),
            ));

            // Eager executables are compiled at the actual input types, so their exact items are already formal.
            let (_, callee) =
                LeafContext::trace(|inputs: Vec<Leaf>| Ok(inputs), vec![three.r#type().into_owned()]).unwrap();
            let call = JitCallOperation::new(0)
                .with_callee_type_signature(&callee.input_types(), &callee.output_types())
                .unwrap();
            let before = domain.eager_dispatch_statistics();
            for _ in 0..2 {
                let outputs = domain.bind(call.clone(), vec![callee.clone()], &[compiled_three.clone()]).unwrap();
                assert_eq!(outputs[0].r#type(), three.r#type());
                assert_eq!(read_i64(&outputs[0].ragged_storage().unwrap()), [3, 3]);
                assert!(Arc::ptr_eq(outputs[0].xla_domain().session(), domain.session()));
            }
            let after = domain.eager_dispatch_statistics();
            assert_eq!(after.misses - before.misses, 1);
            assert_eq!(after.hits - before.hits, 1);
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_jit_call_jvp_preserves_local_result_shape_relationships() {
        let r#type = representation();
        let input_types = vec![
            ArrayIrType::Array(r#type.storage_type().clone()),
            ArrayIrType::Array(r#type.extent_types()[0].clone()),
        ];
        let (_, callee) = LeafContext::trace(
            |inputs: Vec<Leaf>| Ok(vec![inputs[0].ragged_new(r#type.clone(), &inputs[1..])?]),
            input_types.clone(),
        )
        .unwrap();
        let operation = JitCallOperation::new(0)
            .with_callee_type_signature(&callee.input_types(), &callee.output_types())
            .unwrap();
        let (_, parent) = LeafContext::trace_with_context_and_named_axes(
            |context, inputs: Vec<Leaf>| context.bind(operation, vec![callee], &inputs),
            input_types,
            vec![],
        )
        .unwrap();
        let differentiated = parent.jvp().unwrap();
        let outputs = differentiated.output_types();
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[1], outputs[0].tangent().unwrap());
    }

    #[test]
    fn test_grouped_region_entry_checks_untrusted_extent_payloads_on_cpu() {
        /// Keeps both dispatch paths tied to one client while isolating failed assertion scopes.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            for length in [2, 0] {
                let rows = DimensionVariable::new("entry_rows", DimensionBounds::non_negative(Some(4)).unwrap());
                let r#type = RaggedType::new(
                    ArrayType::new_static(DataType::I64, [length, 3]).replicated(&mesh).unwrap(),
                    RaggedValueKind::Array,
                    vec![RaggedDimension::new(1, rows, vec![0])],
                )
                .unwrap();
                let raw = |extents: &[i64]| {
                    let storage = XlaArray::from_host_buffer(
                        domain,
                        r#type.storage_type().clone(),
                        mesh.clone(),
                        crate::tests::values_to_bytes(&vec![7i64; length * 3]),
                    )
                    .unwrap();
                    let extents = XlaArray::from_host_buffer(
                        domain,
                        r#type.extent_types()[0].clone(),
                        mesh.clone(),
                        crate::tests::values_to_bytes(extents),
                    )
                    .unwrap();
                    XlaValue::Ragged(RaggedValue::new(r#type.clone(), storage, vec![extents]).unwrap())
                };
                let (_, callee) = LeafContext::trace(
                    |input: Vec<Leaf>| {
                        Ok(vec![
                            input[0].ragged_storage()?.reduce(&[0, 1], ReductionKind::Sum)?,
                            input[1].ragged_extents(1)?.reduce(&[0], ReductionKind::Sum)?,
                        ])
                    },
                    vec![ArrayIrType::Ragged(r#type.clone()); 2],
                )
                .unwrap();
                let valid = if length == 0 { vec![] } else { vec![0i64, 1] };
                let cases = if length == 0 {
                    vec![(vec![], None)]
                } else {
                    vec![
                        (valid.clone(), None),
                        (
                            vec![2i64, 3],
                            Some("guarded ragged occurrences of `entry_rows` must have equal live extents"),
                        ),
                        (vec![-1i64, 1], Some("ragged extent on storage axis 1 is outside its declared bounds")),
                        (vec![0i64, 4], Some("ragged extent on storage axis 1 is outside its declared bounds")),
                    ]
                };
                for (other, expected_error) in cases {
                    let inputs = vec![raw(&valid), raw(&other)];
                    let operation = XlaOperation::JitCall(JitCallOperation::new(0));
                    let eager = domain.fork_effect_scope().bind(operation.clone(), vec![callee.clone()], &inputs);
                    let compiled_domain = domain.fork_effect_scope();
                    let compiled_callee = callee.clone();
                    let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                        try_jit_with_options(
                            &compiled_domain,
                            move |(), input: Vec<RuntimeLeaf<'c>>| {
                                Ok(input[0].domain().bind(operation.clone(), vec![compiled_callee.clone()], &input)?)
                            },
                            XlaOptions::new(mesh.clone()),
                        );
                    let compiled = compiled.call((), inputs);
                    if let Some(message) = expected_error {
                        let eager = eager.unwrap_err().to_string();
                        let compiled = compiled.unwrap_err().to_string();
                        assert!(eager.contains(message), "{eager}");
                        assert!(compiled.contains(message), "{compiled}");
                    } else {
                        let eager = eager.unwrap();
                        let compiled = compiled.unwrap();
                        for outputs in [&eager, &compiled] {
                            assert_eq!(read_i64(&outputs[0]), [if length == 0 { 0 } else { 42 }]);
                            assert_eq!(read_i64(&outputs[1]), [if length == 0 { 0 } else { 1 }]);
                        }
                    }
                }
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_grouped_region_entry_checks_order_effects_on_cpu() {
        /// Traces a JIT call and a while loop over two ragged values sharing one indexed identity. The callee and the
        /// loop body print after their ragged entry checks, between two prints of the enclosing program. Each `before`
        /// print observes a constant, so only the effect token chain orders it after the enclosing entry check.
        fn programs(ragged: ArrayIrType, counter: ArrayIrType) -> (FlatXlaProgram, FlatXlaProgram) {
            let ragged = vec![ragged; 2];
            let (_, callee) = LeafContext::trace(
                |input: Vec<Leaf>| {
                    Ok(vec![input[1].ragged_extents(1)?.print("callee")?.reduce(&[0], ReductionKind::Sum)?])
                },
                ragged.clone(),
            )
            .unwrap();
            let (_, call) = LeafContext::trace_with_context_and_named_axes(
                |context, input: Vec<Leaf>| {
                    context.lift(Array::scalar(7i64)?.into())?.print("before")?;
                    let output =
                        context.bind(XlaOperation::JitCall(JitCallOperation::new(0)), vec![callee.clone()], &input)?;
                    Ok(vec![output[0].clone().print("after")?])
                },
                ragged.clone(),
                vec![],
            )
            .unwrap();
            let state = std::iter::once(counter).chain(ragged).collect::<Vec<_>>();
            let (_, condition) = LeafContext::trace_with_context_and_named_axes(
                |context, input: Vec<Leaf>| {
                    let two = context.lift(Array::scalar(2i64)?.into())?;
                    Ok(vec![input[0].compare(&two, ComparisonDirection::LessThan)?])
                },
                state.clone(),
                vec![],
            )
            .unwrap();
            let (_, body) = LeafContext::trace_with_context_and_named_axes(
                |context, input: Vec<Leaf>| {
                    let one = context.lift(Array::scalar(1i64)?.into())?;
                    Ok(vec![input[0].clone().print("body")?.add(&one)?, input[1].clone(), input[2].clone()])
                },
                state.clone(),
                vec![],
            )
            .unwrap();
            let operation = XlaOperation::While(
                WhileOperation::new().with_body_type_signature(&body.input_types(), &body.output_types()).unwrap(),
            );
            let (_, r#loop) = LeafContext::trace_with_context_and_named_axes(
                |context, input: Vec<Leaf>| {
                    context.lift(Array::scalar(7i64)?.into())?.print("before")?;
                    let output = context.bind(operation, vec![condition, body], &input)?;
                    Ok(vec![output[0].clone().print("after")?])
                },
                state,
                vec![],
            )
            .unwrap();
            (call, r#loop)
        }

        /// Keeps compilation and execution on one client while isolating failed assertion scopes.
        fn run<'c>(domain: &XlaDomain<'c>) {
            let matching = runtime_ragged(domain);
            let storage = domain.lift(Array::matrix(2, 3, vec![7f32; 6]).unwrap().into()).unwrap();
            let extents = domain.lift(Array::vector(vec![2i64, 3]).unwrap().into()).unwrap();
            let mismatched = storage
                .ragged_new(<&RaggedType>::try_from(matching.r#type().as_ref()).unwrap().clone(), &[extents])
                .unwrap();
            // Checked executions now own independent families. Deliberately claim the first value's identity on
            // this structurally valid input so the region must check their unequal payloads before any effect.
            let XlaValue::Ragged(mismatched) = mismatched else { panic!("expected checked ragged storage") };
            let mismatched = XlaValue::Ragged(
                RaggedValue::new(
                    <&RaggedType>::try_from(matching.r#type().as_ref()).unwrap().clone(),
                    mismatched.storage().clone(),
                    mismatched.extents().to_vec(),
                )
                .unwrap(),
            );
            let counter = domain.lift(Array::scalar(0i64).unwrap().into()).unwrap();
            let (call, r#loop) = programs(matching.r#type().into_owned(), counter.r#type().into_owned());
            let invoke = |program: &FlatXlaProgram, inputs: &[XlaValue<'c>]| {
                with_captured_prints(|| {
                    domain.fork_effect_scope().bind(
                        XlaOperation::JitCall(JitCallOperation::new(0)),
                        vec![program.clone()],
                        inputs,
                    )
                })
            };
            let mismatch = "assertion failed: guarded ragged occurrences of `rows` must have equal live extents; \
                element [0], extent=2, other_extent=0; element [1], extent=3, other_extent=1";

            // Prints run in program order across the JIT boundary, and the callee print follows its entry check.
            let (outputs, lines) = invoke(&call, &[matching.clone(), matching.clone()]);
            assert_eq!(read_i64(&outputs.unwrap()[0]), [1]);
            assert_eq!(
                lines,
                [
                    "before: i64[] 0x0700000000000000",
                    "callee: i64[2] 0x00000000000000000100000000000000",
                    "after: i64[] 0x0100000000000000",
                ],
            );

            // A failed entry check stops every later effect, including the print of a constant.
            let (outputs, lines) = invoke(&call, &[matching.clone(), mismatched.clone()]);
            assert_eq!(outputs.unwrap_err().to_string(), mismatch);
            assert_eq!(lines, Vec::<String>::new());

            // The body print of every iteration follows that iteration's entry check, between the enclosing prints.
            let (outputs, lines) = invoke(&r#loop, &[counter.clone(), matching.clone(), matching.clone()]);
            assert_eq!(read_i64(&outputs.unwrap()[0]), [2]);
            assert_eq!(
                lines,
                [
                    "before: i64[] 0x0700000000000000",
                    "body: i64[] 0x0000000000000000",
                    "body: i64[] 0x0100000000000000",
                    "after: i64[] 0x0200000000000000",
                ],
            );
            let (outputs, lines) = invoke(&r#loop, &[counter, matching, mismatched]);
            assert_eq!(outputs.unwrap_err().to_string(), mismatch);
            assert_eq!(lines, Vec::<String>::new());
        }

        // Each region entry guards its fields and checks every live extent before the region's first effect.
        let (call, r#loop) =
            programs(ArrayIrType::Ragged(representation()), ArrayIrType::Array(ArrayType::scalar(DataType::I64)));
        assert_eq!(
            legalize_ragged_program(&call, 0).unwrap().unwrap().program.to_string(),
            indoc! {r#"
                lambda %0:f32[2, 3], %1:i64[2], %2:f32[2, 3], %3:i64[2] .
                let %4:f32[2, 3], %5:i64[2], %6:f32[2, 3], %7:i64[2] = guard [
                    boundary="XLA region entry",
                    declared_types=[f32[2, 3], i64[2], f32[2, 3], i64[2]],
                ] %0 %1 %2 %3
                    %8:i64[] = const literal<i64[]>[0000000000000000]
                    %9:i64[] = const literal<i64[]>[0300000000000000]
                    %10:bool[2] = compare [direction=GreaterThanOrEqual] %5 %8
                    %11:bool[2] = compare [direction=LessThanOrEqual] %5 %9
                    %12:bool[2] = and %10 %11
                    %13:i64[] = const literal<i64[]>[0000000000000000]
                    %14:i64[] = const literal<i64[]>[0300000000000000]
                    () = assert [
                        message="ragged extent on storage axis 1 is outside its declared bounds",
                        labels=["extent"],
                        failure_limit=8,
                    ] %12 %5
                    %15:bool[2] = compare [direction=GreaterThanOrEqual] %7 %13
                    %16:bool[2] = compare [direction=LessThanOrEqual] %7 %14
                    %17:bool[2] = and %15 %16
                    () = assert [
                        message="ragged extent on storage axis 1 is outside its declared bounds",
                        labels=["extent"],
                        failure_limit=8,
                    ] %17 %7
                    %18:bool[2] = compare [direction=Equal] %7 %5
                    () = assert [
                        message="guarded ragged occurrences of `rows` must have equal live extents",
                        labels=["extent", "other_extent"],
                        failure_limit=8,
                    ] %18 %7 %5
                    %19:f32[2, 3], %20:i64[2], %21:f32[2, 3], %22:i64[2] = guard [
                        boundary="XLA region entry",
                        declared_types=[f32[2, 3], i64[2], f32[2, 3], i64[2]],
                    ] %4 %5 %6 %7
                    %23:i64[] = const literal<i64[]>[0700000000000000]
                    %24:i64[] = print [label=before] %23
                    %25:i64[] = jit_call [
                        logical_inputs=[ragged<f32[2, 3]; 1:rows over [0]>, ragged<f32[2, 3]; 1:rows over [0]>],
                        logical_outputs=[i64[]],
                    ] %19 %20 %21 %22 [
                        callee={
                            lambda %0:f32[2, 3], %1:i64[2], %2:f32[2, 3], %3:i64[2] .
                            let %4:f32[2, 3], %5:i64[2], %6:f32[2, 3], %7:i64[2] = guard [
                                boundary="XLA region entry",
                                declared_types=[f32[2, 3], i64[2], f32[2, 3], i64[2]],
                            ] %0 %1 %2 %3
                                %8:i64[] = const literal<i64[]>[0000000000000000]
                                %9:i64[] = const literal<i64[]>[0300000000000000]
                                %10:bool[2] = compare [direction=GreaterThanOrEqual] %5 %8
                                %11:bool[2] = compare [direction=LessThanOrEqual] %5 %9
                                %12:bool[2] = and %10 %11
                                %13:i64[] = const literal<i64[]>[0000000000000000]
                                %14:i64[] = const literal<i64[]>[0300000000000000]
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %12 %5
                                %15:bool[2] = compare [direction=GreaterThanOrEqual] %7 %13
                                %16:bool[2] = compare [direction=LessThanOrEqual] %7 %14
                                %17:bool[2] = and %15 %16
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %17 %7
                                %18:bool[2] = compare [direction=Equal] %7 %5
                                () = assert [
                                    message="guarded ragged occurrences of `rows` must have equal live extents",
                                    labels=["extent", "other_extent"],
                                    failure_limit=8,
                                ] %18 %7 %5
                                %19:f32[2, 3], %20:i64[2], %21:f32[2, 3], %22:i64[2] = guard [
                                    boundary="XLA region entry",
                                    declared_types=[f32[2, 3], i64[2], f32[2, 3], i64[2]],
                                ] %4 %5 %6 %7
                                %23:i64[2] = print [label=callee] %22
                                %24:i64[] = reduce [kind=sum, axes=[0]] %23
                            in (%24)
                        },
                    ]
                    %26:i64[] = print [label=after] %25
                in (%26)"#},
        );
        assert_eq!(
            legalize_ragged_program(&r#loop, 0).unwrap().unwrap().program.to_string(),
            indoc! {r#"
                lambda %0:i64[], %1:f32[2, 3], %2:i64[2], %3:f32[2, 3], %4:i64[2] .
                let %5:i64[], %6:f32[2, 3], %7:i64[2], %8:f32[2, 3], %9:i64[2] = guard [
                    boundary="XLA region entry",
                    declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                ] %0 %1 %2 %3 %4
                    %10:i64[] = const literal<i64[]>[0000000000000000]
                    %11:i64[] = const literal<i64[]>[0300000000000000]
                    %12:bool[2] = compare [direction=GreaterThanOrEqual] %7 %10
                    %13:bool[2] = compare [direction=LessThanOrEqual] %7 %11
                    %14:bool[2] = and %12 %13
                    %15:i64[] = const literal<i64[]>[0000000000000000]
                    %16:i64[] = const literal<i64[]>[0300000000000000]
                    () = assert [
                        message="ragged extent on storage axis 1 is outside its declared bounds",
                        labels=["extent"],
                        failure_limit=8,
                    ] %14 %7
                    %17:bool[2] = compare [direction=GreaterThanOrEqual] %9 %15
                    %18:bool[2] = compare [direction=LessThanOrEqual] %9 %16
                    %19:bool[2] = and %17 %18
                    () = assert [
                        message="ragged extent on storage axis 1 is outside its declared bounds",
                        labels=["extent"],
                        failure_limit=8,
                    ] %19 %9
                    %20:bool[2] = compare [direction=Equal] %9 %7
                    () = assert [
                        message="guarded ragged occurrences of `rows` must have equal live extents",
                        labels=["extent", "other_extent"],
                        failure_limit=8,
                    ] %20 %9 %7
                    %21:i64[], %22:f32[2, 3], %23:i64[2], %24:f32[2, 3], %25:i64[2] = guard [
                        boundary="XLA region entry",
                        declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                    ] %5 %6 %7 %8 %9
                    %26:i64[] = const literal<i64[]>[0700000000000000]
                    %27:i64[] = print [label=before] %26
                    %28:i64[], %29:f32[2, 3], %30:i64[2], %31:f32[2, 3], %32:i64[2] = while %21 %22 %23 %24 %25 [
                        condition={
                            lambda %0:i64[], %1:f32[2, 3], %2:i64[2], %3:f32[2, 3], %4:i64[2] .
                            let %5:i64[], %6:f32[2, 3], %7:i64[2], %8:f32[2, 3], %9:i64[2] = guard [
                                boundary="XLA region entry",
                                declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                            ] %0 %1 %2 %3 %4
                                %10:i64[] = const literal<i64[]>[0000000000000000]
                                %11:i64[] = const literal<i64[]>[0300000000000000]
                                %12:bool[2] = compare [direction=GreaterThanOrEqual] %7 %10
                                %13:bool[2] = compare [direction=LessThanOrEqual] %7 %11
                                %14:bool[2] = and %12 %13
                                %15:i64[] = const literal<i64[]>[0000000000000000]
                                %16:i64[] = const literal<i64[]>[0300000000000000]
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %14 %7
                                %17:bool[2] = compare [direction=GreaterThanOrEqual] %9 %15
                                %18:bool[2] = compare [direction=LessThanOrEqual] %9 %16
                                %19:bool[2] = and %17 %18
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %19 %9
                                %20:bool[2] = compare [direction=Equal] %9 %7
                                () = assert [
                                    message="guarded ragged occurrences of `rows` must have equal live extents",
                                    labels=["extent", "other_extent"],
                                    failure_limit=8,
                                ] %20 %9 %7
                                %21:i64[], %22:f32[2, 3], %23:i64[2], %24:f32[2, 3], %25:i64[2] = guard [
                                    boundary="XLA region entry",
                                    declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                                ] %5 %6 %7 %8 %9
                                %26:i64[] = const literal<i64[]>[0200000000000000]
                                %27:bool[] = compare [direction=LessThan] %21 %26
                            in (%27)
                        },
                        body={
                            lambda %0:i64[], %1:f32[2, 3], %2:i64[2], %3:f32[2, 3], %4:i64[2] .
                            let %5:i64[], %6:f32[2, 3], %7:i64[2], %8:f32[2, 3], %9:i64[2] = guard [
                                boundary="XLA region entry",
                                declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                            ] %0 %1 %2 %3 %4
                                %10:i64[] = const literal<i64[]>[0000000000000000]
                                %11:i64[] = const literal<i64[]>[0300000000000000]
                                %12:bool[2] = compare [direction=GreaterThanOrEqual] %7 %10
                                %13:bool[2] = compare [direction=LessThanOrEqual] %7 %11
                                %14:bool[2] = and %12 %13
                                %15:i64[] = const literal<i64[]>[0000000000000000]
                                %16:i64[] = const literal<i64[]>[0300000000000000]
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %14 %7
                                %17:bool[2] = compare [direction=GreaterThanOrEqual] %9 %15
                                %18:bool[2] = compare [direction=LessThanOrEqual] %9 %16
                                %19:bool[2] = and %17 %18
                                () = assert [
                                    message="ragged extent on storage axis 1 is outside its declared bounds",
                                    labels=["extent"],
                                    failure_limit=8,
                                ] %19 %9
                                %20:bool[2] = compare [direction=Equal] %9 %7
                                () = assert [
                                    message="guarded ragged occurrences of `rows` must have equal live extents",
                                    labels=["extent", "other_extent"],
                                    failure_limit=8,
                                ] %20 %9 %7
                                %21:i64[], %22:f32[2, 3], %23:i64[2], %24:f32[2, 3], %25:i64[2] = guard [
                                    boundary="XLA region entry",
                                    declared_types=[i64[], f32[2, 3], i64[2], f32[2, 3], i64[2]],
                                ] %5 %6 %7 %8 %9
                                %26:i64[] = const literal<i64[]>[0100000000000000]
                                %27:i64[] = print [label=body] %21
                                %28:i64[] = add %27 %26
                            in (%28, %22, %23, %24, %25)
                        },
                    ]
                    %33:i64[] = print [label=after] %28
                in (%33)"#},
        );
        let client = crate::tests::execution_client();
        run(&crate::XlaSession::new(&client).domain());
    }

    #[test]
    fn test_nested_ragged_policy_masked_storage_reduction_on_cpu() {
        /// Keeps compilation and execution on one client while using the public batching-policy surface.
        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let r#type = RaggedType::new(
                ArrayType::new_static(DataType::F32, [2, 2, 3]).replicated(&mesh).unwrap(),
                RaggedValueKind::Array,
                vec![
                    RaggedDimension::new(
                        1,
                        DimensionVariable::new("rows", DimensionBounds::non_negative(Some(3)).unwrap()),
                        vec![0],
                    ),
                    RaggedDimension::new(
                        2,
                        DimensionVariable::new("columns", DimensionBounds::non_negative(Some(4)).unwrap()),
                        vec![0, 1],
                    ),
                ],
            )
            .unwrap();
            let (_, program) = LeafContext::trace(
                |value: Leaf| {
                    let parent = value.domain();
                    let source = ArrayIrBatch::new(value, BatchAxis::new(0))?;
                    let projected = <ArrayIrBatchingPolicy as BatchingPolicyProjection<_, ArrayType>>::project_batch(
                        &parent, &source,
                    )?;
                    let context =
                        BatchingContext::<_, ArrayBatchingPolicy<DynamicArrayExtentBatchingPolicy>>::with_policy(
                            ProjectedContext::new(parent.clone()),
                            parent.dimension_constant(2)?,
                        );
                    let masked = DynamicArrayExtentBatchingPolicy::mask_reduction_input(
                        &context,
                        &projected,
                        &[2],
                        ReductionKind::Sum,
                    )?;
                    let reduced = ArrayBatch::new(masked.value().reduce_sum(&[1, 2], None)?, BatchAxis::new(0))?;
                    Ok(<ArrayIrBatchingPolicy as BatchingPolicyProjection<_, ArrayType>>::lift_batch(
                        &parent,
                        &reduced,
                        &[source],
                    )?
                    .into_value())
                },
                ArrayIrType::Ragged(r#type.clone()),
            )
            .unwrap();
            let callee = program.into_flat_program();
            let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), ArrayIrType, ArrayIrType> =
                try_jit_with_options(
                    domain,
                    move |(), input: RuntimeLeaf<'c>| {
                        let mut output = input.domain().bind(
                            XlaOperation::JitCall(JitCallOperation::new(0)),
                            vec![callee.clone()],
                            &[input],
                        )?;
                        Ok(output.remove(0))
                    },
                    XlaOptions::new(mesh.clone()),
                );
            for (rows, columns, expected) in [
                (vec![1i64, 0], vec![2i64, 999, 999, 999], vec![3f32, 0.0]),
                (vec![0i64, 0], vec![999i64; 4], vec![0f32, 0.0]),
            ] {
                let storage = XlaArray::from_host_buffer(
                    domain,
                    r#type.storage_type().clone(),
                    mesh.clone(),
                    crate::tests::values_to_bytes(&[
                        1f32, 2.0, 91.0, 92.0, 93.0, 94.0, 95.0, 96.0, 97.0, 98.0, 99.0, 100.0,
                    ]),
                )
                .unwrap();
                let extents = r#type
                    .extent_types()
                    .iter()
                    .zip([rows, columns])
                    .map(|(r#type, values)| {
                        XlaArray::from_host_buffer(
                            domain,
                            r#type.clone(),
                            mesh.clone(),
                            crate::tests::values_to_bytes(&values),
                        )
                        .unwrap()
                    })
                    .collect::<Vec<_>>();
                let input = XlaValue::Ragged(RaggedValue::new(r#type.clone(), storage, extents).unwrap());
                let output = compiled.call((), input).unwrap();
                let output = <XlaValue<'c> as ValueProjection<ArrayType>>::projected(&output).unwrap();
                let bytes = output
                    .addressable_shards()
                    .next()
                    .unwrap()
                    .buffer()
                    .unwrap()
                    .copy_to_host(None)
                    .unwrap()
                    .r#await()
                    .unwrap();
                assert_eq!(crate::tests::values_from_bytes::<f32>(bytes.as_slice()), expected);
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = crate::XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }

    #[test]
    fn test_linear_call_final_replay_preserves_passive_extent_fields() {
        let r#type = representation();
        let (_, forward) =
            LeafContext::trace(|input: Vec<Leaf>| Ok(input), vec![ArrayIrType::Ragged(r#type.clone())]).unwrap();
        let transpose = forward.clone();
        let (_, program) = LeafContext::trace_with_context_and_named_axes(
            |context, input: Vec<Leaf>| {
                context.bind(LinearCallOperation::<ArrayIrType>::new(0), vec![forward, transpose], &input)
            },
            vec![ArrayIrType::Ragged(r#type.clone())],
            vec![],
        )
        .unwrap();
        let legalized = legalize_ragged_program(&program, 0).unwrap().unwrap();
        assert_eq!(legalized.program.input_types(), leaf_types(&r#type.clone().into()));
        assert_eq!(legalized.program.output_types(), leaf_types(&r#type.into()));
        assert!(matches!(legalized.program.instructions()[0].operation(), XlaOperation::JitCall(_)));
        assert!(!legalized.program.to_string().contains("linear_call"));
    }

    #[test]
    fn test_linear_call_final_replay_preserves_scalar_result_definition() {
        let inputs = vec![ArrayIrType::Ragged(representation()), ArrayIrType::Array(ArrayType::scalar(DataType::I64))];
        let dimension = DimensionType::new("computed", DimensionBounds::non_negative(Some(3)).unwrap());
        let (_, forward) = LeafContext::trace_with_context_and_named_axes(
            |context, inputs: Vec<Leaf>| {
                context.bind(DimensionFromScalarOperation::new(dimension.clone()), vec![], &inputs[1..])
            },
            inputs.clone(),
            vec![],
        )
        .unwrap();
        let transpose_inputs = inputs
            .iter()
            .cloned()
            .chain([ArrayIrType::Dimension(dimension.clone()).cotangent().unwrap()])
            .collect::<Vec<_>>();
        let (_, transpose) = LeafContext::trace(|_: Vec<Leaf>| Ok(Vec::<Leaf>::new()), transpose_inputs).unwrap();
        let (_, program) = LeafContext::trace_with_context_and_named_axes(
            |context, inputs: Vec<Leaf>| {
                let output = context.bind(LinearCallOperation::new(2), vec![forward, transpose], &inputs)?;
                Ok(vec![output[0].to_scalar()?])
            },
            inputs,
            vec![],
        )
        .unwrap();
        let legalized = legalize_ragged_program(&program, 0).unwrap().unwrap();
        let instruction = legalized
            .program
            .instructions()
            .iter()
            .find(|instruction| matches!(instruction.operation(), XlaOperation::JitCall(_)))
            .unwrap();
        let outputs = instruction
            .outputs()
            .iter()
            .map(|id| legalized.program.atoms()[id.index()].r#type().into_owned())
            .collect::<Vec<_>>();
        assert_eq!(outputs, vec![ArrayIrType::Dimension(dimension)]);
        assert_eq!(legalized.program.output_types(), vec![ArrayIrType::Array(ArrayType::scalar(DataType::I64))]);
    }

    #[test]
    fn test_ragged_reverse_accumulation_and_disconnected_zero_cpu() {
        fn read_f32(value: &XlaValue<'_>) -> Vec<f32> {
            let array = <XlaValue<'_> as ValueProjection<ArrayType>>::projected(value).unwrap();
            let bytes = array
                .addressable_shards()
                .next()
                .unwrap()
                .buffer()
                .unwrap()
                .copy_to_host(None)
                .unwrap()
                .r#await()
                .unwrap();
            crate::tests::values_from_bytes(bytes.as_slice())
        }

        fn run<'c>(domain: &XlaDomain<'c>, mesh: ryft_core::DeviceMesh) {
            let active = runtime_ragged(domain);
            let disconnected = runtime_ragged(domain);
            let seed = domain.lift(Array::matrix(2, 3, vec![1f32; 6]).unwrap().into()).unwrap();
            let (output, pullback) = differentiate_at(vec![active.clone(), disconnected.clone()])
                .vjp(|input| input[0].ragged_storage()?.add(&input[0].ragged_storage()?))
                .unwrap();
            assert_eq!(read_f32(&output), [14f32, 22., 202., 26., 206., 214.]);
            let transposed = pullback.transposed_program(&[]).unwrap().into_program();
            assert!(
                transposed
                    .instructions()
                    .iter()
                    .any(|instruction| matches!(instruction.operation(), XlaOperation::RaggedAdd(_)))
            );
            for residual in pullback.residuals() {
                if let ArrayIrType::Ragged(r#type) = residual.r#type().as_ref() {
                    assert_eq!(r#type.storage_type().data_type(), DataType::Zero);
                }
            }
            let eager = pullback.apply_materialized(seed.clone()).unwrap();
            let compiled: CompiledFunctionDispatcher<XlaDomain<'c>, _, (), Vec<ArrayIrType>, Vec<ArrayIrType>> =
                try_jit_with_options(
                    domain,
                    |(), input: Vec<RuntimeLeaf<'c>>| {
                        let (_, pullback) = differentiate_at(vec![input[0].clone(), input[1].clone()])
                            .vjp(|input| input[0].ragged_storage()?.add(&input[0].ragged_storage()?))
                            .map_err(ProgramError::from)?;
                        Ok(pullback.apply_materialized(input[2].clone()).map_err(ProgramError::from)?)
                    },
                    XlaOptions::new(mesh),
                );
            let compiled = compiled.call((), vec![active, disconnected, seed]).unwrap();
            for results in [&eager, &compiled] {
                // Explicit storage projection exposes padding too, so its adjoint is two on both live and padded cells.
                // The disconnected zero keeps its own indexed family and each complete extent payload.
                assert_eq!(read_f32(&results[0].ragged_storage().unwrap()), [2f32; 6]);
                assert_eq!(read_f32(&results[1].ragged_storage().unwrap()), [0f32; 6]);
                assert_eq!(read_i64(&results[0].ragged_extents(1).unwrap()), [0, 1]);
                assert_eq!(read_i64(&results[1].ragged_extents(1).unwrap()), [0, 1]);
            }
        }
        let client = crate::tests::execution_client();
        let mesh = cpu_mesh(&client);
        let domain = XlaSession::new(&client).domain().with_mesh(mesh.clone());
        run(&domain, mesh);
    }
}
