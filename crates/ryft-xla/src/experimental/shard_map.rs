use std::cell::RefCell;
use std::collections::{BTreeSet, HashMap, HashSet};
use std::fmt::Debug;

use ryft_core::{
    ArrayIrType, ArrayType, Atom, AtomId, Context, Dimension, Domain, DomainTracingContext, Instruction, LogicalMesh,
    MeshAxisType, NamedAxis, Operation, Parameter, ParameterError, Parameterized, ParameterizedFamily, Placeholder,
    ProgramError, ProgramStatistics, ProjectedValue, ReshardOperation, Shape, Sharding, ShardingConstraintOperation,
    ShardingDimension, ShardingError, Value, ValueProjection,
};
#[cfg(test)]
use ryft_core::{StagingContext, Typed};
#[cfg(test)]
use ryft_mlir::Block;
use ryft_mlir::Context as MlirContext;
use ryft_mlir::dialects::shardy::{
    DimensionShardingAttributeRef, ManualAxesAttributeRef, ReductionOperation, TensorShardingAttributeRef,
    TensorShardingPerValueAttributeRef,
};
use thiserror::Error;

use crate::experimental::domains::{XlaDomain, XlaTracer};
use crate::experimental::operations::ShardMapOperation;
use crate::experimental::ops::{XlaConstant, XlaOperation, XlaProgram, XlaProgramBuilder};
use crate::sharding::SHARDY_MESH_SYMBOL_NAME;

use super::lowering::LoweringError;

/// Error type for internal shard-map metadata validation and Shardy rendering.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub(crate) enum ShardMapError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a mesh used for `ShardMap` has no manual axes.
    #[error("shard_map requires at least one mesh axis with type manual")]
    MeshHasNoManualAxes,

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis '{free_axis_name}' \
         more major than manual axis '{manual_axis_name}'"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when a provided global shape rank does not match the sharding rank.
    #[error(
        "{value_kind} sharding #{value_index} has rank {partition_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, partition_rank: usize, shape_rank: usize },

    /// Error returned when a manual axis would require padding in the local body shape.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} has size {dimension_size}, \
         which is not divisible by manual partition count {manual_partition_count}"
    )]
    ManualAxisIntroducesPadding {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        dimension_size: usize,
        manual_partition_count: usize,
    },
}

/// Error type for tracing and lowering `xla::shard_map` bodies.
#[derive(Error, Clone, Debug, PartialEq, Eq)]
pub enum ShardMapTraceError {
    /// Underlying error returned by the mesh/sharding layer.
    #[error("{0}")]
    ShardingError(#[from] ShardingError),

    /// Error returned when a mesh used for `shard_map` has no manual axes.
    #[error("shard_map requires at least one mesh axis with type manual")]
    MeshHasNoManualAxes,

    /// Error returned when a partitioned dimension uses a free axis more major than a manual axis.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} uses free axis '{free_axis_name}' \
         more major than manual axis '{manual_axis_name}'"
    )]
    ManualAxisMustPrecedeFreeAxis {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        free_axis_name: String,
        manual_axis_name: String,
    },

    /// Error returned when a provided global shape rank does not match the sharding rank.
    #[error(
        "{value_kind} sharding #{value_index} has rank {partition_rank}, but the provided shape \
         has rank {shape_rank}"
    )]
    RankMismatch { value_kind: &'static str, value_index: usize, partition_rank: usize, shape_rank: usize },

    /// Error returned when a manual axis would require padding in the local body shape.
    #[error(
        "{value_kind} sharding #{value_index} dimension #{dimension} has size {dimension_size}, \
         which is not divisible by manual partition count {manual_partition_count}"
    )]
    ManualAxisIntroducesPadding {
        value_kind: &'static str,
        value_index: usize,
        dimension: usize,
        dimension_size: usize,
        manual_partition_count: usize,
    },

    /// Underlying tracing error returned while staging a shard-map body.
    #[error("{0}")]
    ProgramError(#[from] ProgramError),

    /// Underlying parameter-structure error returned while reparameterizing traced values.
    #[error("{0}")]
    ParameterError(#[from] ParameterError),

    /// Error returned when traced `shard_map` staging has non-empty outputs but no traced input
    /// leaf is available to supply the outer tracing context.
    #[error("traced shard_map with non-empty outputs requires at least one traced input leaf")]
    MissingTracedInvocationDomain,

    /// Error returned while building StableHLO/Shardy MLIR for a traced shard-map body.
    #[error("{message}")]
    LoweringFailure { message: String },

    /// Error returned when the number of global input types does not match the number of input shardings.
    #[error("got {actual} global input type(s), but shard_map expects {expected}")]
    InputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when the number of traced output types does not match the number of output shardings.
    #[error("traced body produced {actual} output type(s), but shard_map expects {expected}")]
    OutputTypeCountMismatch { expected: usize, actual: usize },

    /// Error returned when a traced shard-map type contains a dynamic dimension that is not supported yet.
    #[error("{value_kind} type #{value_index} dimension #{dimension} must be static for traced shard_map")]
    DynamicShapeNotSupported { value_kind: &'static str, value_index: usize, dimension: usize },

    /// Error returned when `check_vma=true` and one output still varies along an omitted manual axis.
    #[error(
        "output type #{output_index} still varies along manual axis '{axis_name}', but out_specs does not mention it"
    )]
    OutputVaryingManualAxisNotInOutSpecs { output_index: usize, axis_name: String },

    /// Error returned when one input manual-axis state does not match the corresponding `in_specs`.
    #[error("{value_kind} type #{value_index} has {state_kind} {actual:?}, but shard_map expects {expected:?}")]
    ShardingStateMismatch {
        value_kind: &'static str,
        value_index: usize,
        state_kind: &'static str,
        expected: Vec<String>,
        actual: Vec<String>,
    },

    /// Error returned when reconstructing a global output shape overflows `usize`.
    #[error("overflow while {context}")]
    Overflow { context: String },
}

impl From<LoweringError> for ShardMapTraceError {
    fn from(error: LoweringError) -> Self {
        match error {
            LoweringError::ShardMapError(error) => Self::from(error),
            LoweringError::ShardingError(error) => Self::ShardingError(error),
            error => Self::LoweringFailure { message: error.to_string() },
        }
    }
}

impl From<ShardMapError> for ShardMapTraceError {
    fn from(error: ShardMapError) -> Self {
        match error {
            ShardMapError::ShardingError(error) => Self::ShardingError(error),
            ShardMapError::MeshHasNoManualAxes => Self::MeshHasNoManualAxes,
            ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind,
                value_index,
                dimension,
                free_axis_name,
                manual_axis_name,
            } => Self::ManualAxisMustPrecedeFreeAxis {
                value_kind,
                value_index,
                dimension,
                free_axis_name,
                manual_axis_name,
            },
            ShardMapError::RankMismatch { value_kind, value_index, partition_rank, shape_rank } => {
                Self::RankMismatch { value_kind, value_index, partition_rank, shape_rank }
            }
            ShardMapError::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            } => Self::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            },
        }
    }
}

/// Default static tracer alias used by public XLA tracing helpers. This alias is public so that callers outside
/// this crate (e.g., tooling binaries) can annotate [`trace`] and [`shard_map`] closure parameters and pin the
/// generic parameters of [`shard_map`], whose tracer-valued regime cannot be inferred from call sites alone.
pub type ShardMapTracer = ProjectedValue<ArrayType, XlaTracer<'static>>;

/// Rebuilds an [`XlaProgram`] through [`XlaProgramBuilder`] using the public program-construction API, retyping the
/// source program's input/output parameter structures while preserving its atoms, instructions, and attached
/// instruction regions. The rebuilt program exposes the provided `output_ids` (in the source program's atom-id
/// space), which lets callers project the output boundary while copying everything else verbatim.
fn rebuild_xla_program_with_builder<Input, Output, SourceInput, SourceOutput>(
    source: &XlaProgram<SourceInput, SourceOutput>,
    output_ids: Vec<AtomId>,
    input_structure: Input::ParameterStructure,
    output_structure: Output::ParameterStructure,
) -> Result<XlaProgram<Input, Output>, ProgramError>
where
    Input: Parameterized<XlaConstant>,
    Output: Parameterized<XlaConstant>,
    SourceInput: Parameterized<XlaConstant>,
    SourceOutput: Parameterized<XlaConstant>,
{
    let atoms = source.atoms().to_vec();
    let input_ids = source.input_ids().to_vec();
    let instructions = source.instructions().to_vec();
    let mut builder = XlaProgramBuilder::new();
    let mut atom_id_mapping = vec![None; atoms.len()];
    let source_region_ids = instructions
        .iter()
        .flat_map(|instruction| instruction.regions().iter().copied())
        .collect::<Vec<_>>();
    let source_regions = source_region_ids
        .iter()
        .map(|region| source.region_ref(*region))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let imported_regions = builder.import_regions(source_regions.as_slice())?;
    let region_id_mapping = source_region_ids.into_iter().zip(imported_regions).collect::<HashMap<_, _>>();

    for input_id in input_ids {
        let input_atom = atoms.get(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
        let Atom::Variable(input_type) = input_atom else {
            return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()).into());
        };
        let mapped_input = builder.add_input(input_type.clone());
        let mapping_slot =
            atom_id_mapping.get_mut(input_id.index()).ok_or(ProgramError::UnboundAtomId { id: input_id })?;
        if mapping_slot.is_some() {
            return Err(
                ProgramError::MalformedProgram("program input atom was listed more than once".to_string()).into()
            );
        }
        *mapping_slot = Some(mapped_input);
    }

    for (atom_index, atom) in atoms.into_iter().enumerate() {
        if atom_id_mapping[atom_index].is_some() {
            continue;
        }
        let mapped_atom = match atom {
            Atom::Constant(value) => builder.add_constant(value),
            Atom::Variable(r#type) => builder.add_variable(r#type),
        };
        atom_id_mapping[atom_index] = Some(mapped_atom);
    }

    for instruction in instructions {
        let inputs = instruction
            .inputs()
            .iter()
            .copied()
            .map(|input| remap_atom_id(atom_id_mapping.as_slice(), input))
            .collect::<Result<Vec<_>, _>>()?;
        let outputs = instruction
            .outputs()
            .iter()
            .copied()
            .map(|output| remap_atom_id(atom_id_mapping.as_slice(), output))
            .collect::<Result<Vec<_>, _>>()?;
        // Re-attach the instruction's regions through the batch import above so shared descendants stay shared.
        let regions = instruction
            .regions()
            .iter()
            .map(|region| {
                region_id_mapping.get(region).copied().ok_or_else(|| {
                    ProgramError::MalformedProgram(format!("region {region} was not imported during rebuilding"))
                })
            })
            .collect::<Result<Vec<_>, ProgramError>>()?;
        builder.add_instruction_unchecked(Instruction::new(instruction.operation().clone(), inputs, outputs, regions));
    }

    let output_ids = output_ids
        .into_iter()
        .map(|output| remap_atom_id(atom_id_mapping.as_slice(), output))
        .collect::<Result<Vec<_>, _>>()?;
    builder.build(output_ids, input_structure, output_structure)
}

/// Returns the rebuilt [`AtomId`] corresponding to `atom_id`.
fn remap_atom_id(atom_id_mapping: &[Option<AtomId>], atom_id: AtomId) -> Result<AtomId, ProgramError> {
    atom_id_mapping
        .get(atom_id.index())
        .copied()
        .flatten()
        .ok_or(ProgramError::UnboundAtomId { id: atom_id })
}

/// Rewrites a traced XLA program to drop dead outputs of [`shard_map`](XlaOperation::ShardMap) instructions.
///
/// [`Program::simplified`](ryft_core::Program::simplified) is conservative for multi-output instructions: it keeps an
/// entire instruction whenever any of its outputs is live. The forward-mode `shard_map` rule
/// (`ShardMapOperation::jvp`) emits a primal `shard_map` producing `[primal_outputs..., residuals...]`, so when a
/// primal output is dead — as the discarded primal output of a `value_and_gradient` is — that conservative rule keeps
/// the dead output and its body computation. This pass walks the program and, for each `shard_map` instruction with at
/// least one dead output (per whole-program liveness), rebuilds it with its body projected to the live outputs through
/// [`FlatTracedShardMap::with_live_outputs`], allocating only the live output atoms. Dead `shard_map` outputs have no
/// uses by definition, so dropping them rewires nothing downstream. Every other instruction is copied verbatim; a
/// subsequent [`simplified`](ryft_core::Program::simplified) prunes any instructions left dead by the projection.
pub(crate) fn prune_dead_shard_map_outputs<ProgramInput, ProgramOutput>(
    program: &XlaProgram<ProgramInput, ProgramOutput>,
) -> Result<XlaProgram<ProgramInput, ProgramOutput>, ShardMapTraceError>
where
    ProgramInput: Parameterized<XlaConstant>,
    ProgramOutput: Parameterized<XlaConstant>,
{
    let live_atoms = program.live_sets();
    let live_atoms = live_atoms.atoms();
    let mut builder = XlaProgramBuilder::new();
    let mut atom_id_mapping = vec![None; program.atoms().len()];
    let source_region_ids = program
        .instructions()
        .iter()
        .filter(|instruction| {
            let has_dead_output =
                instruction.outputs().iter().any(|output| !live_atoms.get(output.index()).copied().unwrap_or(false));
            let has_live_output =
                instruction.outputs().iter().any(|output| live_atoms.get(output.index()).copied().unwrap_or(false));
            !matches!(instruction.operation(), XlaOperation::ShardMap(_) if has_dead_output && has_live_output)
        })
        .flat_map(|instruction| instruction.regions().iter().copied())
        .collect::<Vec<_>>();
    let source_regions = source_region_ids
        .iter()
        .map(|region| program.region_ref(*region))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    let imported_regions = builder.import_regions(source_regions.as_slice())?;
    let region_id_mapping = source_region_ids.into_iter().zip(imported_regions).collect::<HashMap<_, _>>();

    for input_id in program.input_ids().iter().copied() {
        let Atom::Variable(input_type) = &program.atoms()[input_id.index()] else {
            return Err(ProgramError::MalformedProgram("program input atom was not a variable".to_string()).into());
        };
        atom_id_mapping[input_id.index()] = Some(builder.add_input(input_type.clone()));
    }
    for (atom_index, atom) in program.atoms().iter().enumerate() {
        if atom_id_mapping[atom_index].is_some() {
            continue;
        }
        if let Atom::Constant(value) = atom {
            atom_id_mapping[atom_index] = Some(builder.add_constant(value.clone()));
        }
    }

    for instruction in program.instructions() {
        let inputs = instruction
            .inputs()
            .iter()
            .copied()
            .map(|input| remap_atom_id(atom_id_mapping.as_slice(), input))
            .collect::<Result<Vec<_>, ProgramError>>()?;
        let live_outputs = instruction
            .outputs()
            .iter()
            .map(|output| live_atoms.get(output.index()).copied().unwrap_or(false))
            .collect::<Vec<_>>();
        // Project a `shard_map` body only when some outputs are live and some are dead. A fully dead `shard_map` is left
        // intact for the subsequent `simplified` to drop wholesale, and a fully live one is copied verbatim. Every
        // other instruction re-attaches its regions through one batch import, preserving source sharing.
        let has_dead_output = live_outputs.iter().any(|&live| !live);
        let has_live_output = live_outputs.iter().any(|&live| live);
        let (operation, region_ids) = match instruction.operation() {
            XlaOperation::ShardMap(shard_map_op) if has_dead_output && has_live_output => {
                let body_program = program.region_ref(instruction.regions()[0])?.to_program();
                let local_input_types = body_program
                    .input_types()
                    .iter()
                    .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ProgramError::from)?;
                let local_output_types = body_program
                    .output_types()
                    .iter()
                    .map(|r#type| <&ArrayType>::try_from(r#type).cloned())
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(ProgramError::from)?;
                let body = FlatTracedShardMap::from_parts(
                    shard_map_op.shard_map().clone(),
                    shard_map_op.global_input_types().to_vec(),
                    local_input_types,
                    shard_map_op.global_output_types().to_vec(),
                    local_output_types,
                    body_program,
                );
                let projected = body.with_live_outputs(live_outputs.as_slice())?;
                let (projected_operation, projected_body) = ShardMapOperation::from_body(projected);
                (XlaOperation::ShardMap(Box::new(projected_operation)), vec![builder.import_program(projected_body)])
            }
            operation => {
                let region_ids = instruction
                    .regions()
                    .iter()
                    .map(|region| {
                        region_id_mapping.get(region).copied().ok_or_else(|| {
                            ProgramError::MalformedProgram(format!(
                                "region {region} was not imported during shard-map pruning",
                            ))
                        })
                    })
                    .collect::<Result<Vec<_>, ProgramError>>()?;
                (operation.clone(), region_ids)
            }
        };
        let new_outputs = builder.add_instruction(operation, region_ids, inputs)?.to_vec();
        let mut next_new_output = new_outputs.into_iter();
        for (output, live) in instruction.outputs().iter().copied().zip(live_outputs) {
            if live {
                atom_id_mapping[output.index()] = Some(next_new_output.next().ok_or_else(|| {
                    ProgramError::MalformedProgram("projected shard_map produced too few live outputs".to_string())
                })?);
            }
        }
    }

    let output_ids = program
        .output_ids()
        .iter()
        .copied()
        .map(|output| remap_atom_id(atom_id_mapping.as_slice(), output))
        .collect::<Result<Vec<_>, ProgramError>>()?;
    Ok(builder.build::<ProgramInput, ProgramOutput>(
        output_ids,
        program.input_structure().clone(),
        program.output_structure().clone(),
    )?)
}

/// Structured local-trace input produced by re-parameterizing a global input family over [`ShardMapTracer`] leaves.
pub type ShardMapLocalTraceInput<Input> = <Input as Parameterized<ArrayType>>::To<ShardMapTracer>;

/// Structured local-trace output produced by re-parameterizing a global output family over [`ShardMapTracer`] leaves.
pub type ShardMapLocalTraceOutput<Output> = <Output as Parameterized<ArrayType>>::To<ShardMapTracer>;

type ShardMapProgramParameters<P> = <P as Parameterized<ArrayType>>::To<ArrayIrType>;

type ShardMapProgramValues<P, V> = <ShardMapProgramParameters<P> as Parameterized<ArrayIrType>>::To<V>;

type ShardMapCapturedInput<Input> = ShardMapProgramValues<Input, XlaConstant>;

type ShardMapCapturedOutput<Output> = ShardMapProgramValues<Output, XlaConstant>;

/// Dispatch trait used by [`shard_map`] to select the appropriate tracing regime from the input leaf type. This
/// trait is public — although hidden from documentation — because [`shard_map`]'s return type projects through
/// [`Return`](Self::Return), so external callers could not invoke [`shard_map`] at all if the trait were private.
#[doc(hidden)]
pub trait ShardMapInvocationLeaf: Parameter + Sized {
    /// Return type produced by [`shard_map`] for the corresponding input leaf regime.
    type Return<Input: Parameterized<Self>, Output: Parameterized<ArrayType>>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;

    /// Invokes [`shard_map`] for one specific tracing regime.
    fn invoke<
        F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
        Input: Parameterized<Self>,
        Output: Parameterized<ArrayType>,
    >(
        function: F,
        inputs: Input,
        mesh: LogicalMesh,
        in_specs: Input::To<Sharding>,
        out_specs: Output::To<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self::Return<Input, Output>, ShardMapTraceError>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<Sharding>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<Sharding>
            + ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>
            + ParameterizedFamily<Self>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>;
}

/// Stages an arbitrary traced XLA function over global tensor types.
///
/// This is the general XLA tracing entry point used when callers want to compose `shard_map`
/// with other `tracing_v2` transforms such as `grad` and then lower the resulting whole program to
/// StableHLO/Shardy MLIR.
///
/// # Parameters
///
///   - `function`: Function to trace over global XLA values.
///   - `global_input_types`: Global input array types passed to the traced function.
#[allow(private_bounds, private_interfaces)]
pub fn trace<
    F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
>(
    function: F,
    global_input_types: Input,
) -> Result<TracedXlaProgram<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    let (global_output_types, program) = trace_xla_function(function, &global_input_types, Vec::new())?;
    Ok(TracedXlaProgram { global_input_types, global_output_types, program })
}

/// Binds one sharding-control operation per leaf of an XLA value tree, pairing each leaf with the correspondingly
/// structured [`Sharding`] and validating the requested rank before binding. Shared by [`reshard`] and
/// [`sharding_constraint`], which differ only in the array operation they bind per leaf.
fn bind_sharding_control_per_leaf<Input, Leaf, O>(
    input: Input,
    shardings: Input::To<Sharding>,
    make_operation: impl Fn(Sharding) -> O,
) -> Result<Input, ShardMapTraceError>
where
    Input: Parameterized<Leaf, To<Leaf> = Input>,
    Input::Family: ParameterizedFamily<Sharding>,
    Leaf: Value<Type = ArrayType>,
    Leaf::DispatchDomain: Context<Type = ArrayType>,
    <Leaf::DispatchDomain as Domain>::Operation: From<O>,
    O: Operation<Type = ArrayType>,
{
    fn bind_leaf<Leaf, O>(
        input: Leaf,
        sharding: Sharding,
        make_operation: &impl Fn(Sharding) -> O,
    ) -> Result<Leaf, ShardMapTraceError>
    where
        Leaf: Value<Type = ArrayType>,
        Leaf::DispatchDomain: Context<Type = ArrayType>,
        <Leaf::DispatchDomain as Domain>::Operation: From<O>,
        O: Operation<Type = ArrayType>,
    {
        let input_type = input.r#type();
        if sharding.rank() != input_type.rank() {
            return Err(ShardingError::ShardingRankMismatch {
                sharding_rank: sharding.rank(),
                array_rank: input_type.rank(),
            }
            .into());
        }
        Ok(input
            .dispatch_domain()
            .bind(make_operation(sharding), Vec::new(), std::slice::from_ref(&input))?
            .into_iter()
            .next()
            .expect("a sharding-control operation produces one output per input leaf"))
    }

    let structure = input.parameter_structure();
    let staged = input
        .into_parameters()
        .zip(shardings.into_parameters())
        .map(|(parameter, sharding)| bind_leaf(parameter, sharding, &make_operation))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, staged)?)
}

/// Reshards one traced XLA value tree to target shardings, a tracked sharding transition over the mesh's
/// [`Explicit`](ryft_core::arrays::MeshAxisType::Explicit) and [`Manual`](ryft_core::arrays::MeshAxisType::Manual)
/// axes.
///
/// This stages a [`ReshardOperation`] per leaf, the analogue of JAX's
/// [`jax.sharding.reshard`](https://docs.jax.dev/en/latest/jax.sharding.html): it behaves like the identity at the
/// value level while *replacing* each leaf's tracked [`Sharding`] with the requested one, and it differentiates as a
/// resharding (its transpose reshards the cotangent to the input's cotangent dual). To merely steer the compiler's
/// propagation over auto axes without tracking the result, use [`sharding_constraint`] instead.
///
/// Cross-mesh reshards are not representable inside a single staged program; for that case use the eager
/// [`Array::to_placement`](crate::Array::to_placement) outside the trace.
///
/// # Parameters
///
///   - `input`: Structured traced XLA value whose leaves will be resharded.
///   - `shardings`: Structured target shardings with the same leaf layout as `input`.
#[allow(private_bounds, private_interfaces)]
pub fn reshard<Input, Leaf>(input: Input, shardings: Input::To<Sharding>) -> Result<Input, ShardMapTraceError>
where
    Input: Parameterized<Leaf, To<Leaf> = Input>,
    Input::Family: ParameterizedFamily<Sharding>,
    Leaf: Value<Type = ArrayType>,
    Leaf::DispatchDomain: Context<Type = ArrayType>,
    <Leaf::DispatchDomain as Domain>::Operation: From<ReshardOperation>,
{
    bind_sharding_control_per_leaf(input, shardings, ReshardOperation::new)
}

/// Records sharding-propagation hints on one traced XLA value tree over the mesh's
/// [`Auto`](ryft_core::arrays::MeshAxisType::Auto) axes.
///
/// This stages a [`ShardingConstraintOperation`] per leaf,
/// mirroring [`jax.lax.with_sharding_constraint`](https://docs.jax.dev/en/latest/_autosummary/jax.lax.with_sharding_constraint.html):
/// it is the identity at both the value and type levels (each leaf's tracked sharding is unchanged) and only steers
/// the backend compiler's sharding propagation over auto mesh axes at lowering time, recording a concrete Shardy
/// `sdy.sharding_constraint` on each traced leaf. It is self-adjoint under differentiation. To perform a tracked
/// sharding transition over explicit or manual axes instead, use [`reshard`].
///
/// # Parameters
///
///   - `input`: Structured traced XLA value whose leaves will be hinted.
///   - `shardings`: Structured sharding hints with the same leaf layout as `input`.
#[allow(private_bounds, private_interfaces)]
pub fn sharding_constraint<Input, Leaf>(
    input: Input,
    shardings: Input::To<Sharding>,
) -> Result<Input, ShardMapTraceError>
where
    Input: Parameterized<Leaf, To<Leaf> = Input>,
    Input::Family: ParameterizedFamily<Sharding>,
    Leaf: Value<Type = ArrayType>,
    Leaf::DispatchDomain: Context<Type = ArrayType>,
    <Leaf::DispatchDomain as Domain>::Operation: From<ShardingConstraintOperation>,
{
    bind_sharding_control_per_leaf(input, shardings, ShardingConstraintOperation::new)
}

/// Stages a traced shard-map body over the provided mesh and shardings.
///
/// This is the ergonomic public entry point for traced XLA shard-map staging. It mirrors the
/// function-first shape of JAX's `shard_map` while adapting it to Rust and `tracing_v2` by
/// requiring explicit `global_input_types`.
///
/// Mesh axes whose type is [`Manual`](ryft_core::arrays::MeshAxisType::Manual) define the default
/// manual axes of the computation. Structured `in_specs` and `out_specs` follow the same
/// `Parameterized` layout as the corresponding input and output types. The body closure receives
/// only the traced local inputs, which lets common cases compile cleanly as `|x| ...` or
/// `|(lhs, rhs)| ...` without explicit tracer annotations.
///
/// # Parameters
///
///   - `function`: Body closure to trace over local shard-map values.
///   - `global_input_types`: Global input array types used to derive the local body argument types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_specs`: Structured shardings for the global inputs.
///   - `out_specs`: Structured shardings for the global outputs.
#[allow(private_bounds, private_interfaces)]
pub fn shard_map<
    F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<Leaf>,
    Output: Parameterized<ArrayType>,
    Leaf: ShardMapInvocationLeaf,
>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
) -> Result<<Leaf as ShardMapInvocationLeaf>::Return<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Sharding>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>
        + ParameterizedFamily<Leaf>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    shard_map_with_options(function, inputs, mesh, in_specs, out_specs, vec![], true)
}

/// Stages a traced shard-map body with one explicit manual-axis subset and `check_vma` mode.
///
/// `manual_axes` mirrors JAX's `axis_names`: when the list is empty, all mesh axes whose type is
/// [`Manual`](ryft_core::arrays::MeshAxisType::Manual) are active for this shard-map. `check_vma`
/// mirrors JAX's default output-validity check for omitted manual axes.
///
/// # Parameters
///
///   - `function`: Body closure to trace over local shard-map values.
///   - `global_input_types`: Global input array types used to derive the local body argument types.
///   - `mesh`: Logical mesh that the manual computation is defined over.
///   - `in_specs`: Structured shardings for the global inputs.
///   - `out_specs`: Structured shardings for the global outputs.
///   - `manual_axes`: Active manual mesh axes for this shard-map. An empty list means "all manual
///     mesh axes".
///   - `check_vma`: Whether to reject outputs that still vary along active manual axes omitted from
///     `out_specs`.
#[allow(private_bounds, private_interfaces)]
pub fn shard_map_with_options<
    F: FnOnce(ShardMapLocalTraceInput<Input::To<ArrayType>>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<Leaf>,
    Output: Parameterized<ArrayType>,
    Leaf: ShardMapInvocationLeaf,
>(
    function: F,
    inputs: Input,
    mesh: LogicalMesh,
    in_specs: Input::To<Sharding>,
    out_specs: Output::To<Sharding>,
    manual_axes: Vec<String>,
    check_vma: bool,
) -> Result<<Leaf as ShardMapInvocationLeaf>::Return<Input, Output>, ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<Sharding>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<Sharding>
        + ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>
        + ParameterizedFamily<Leaf>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    Leaf::invoke(function, inputs, mesh, in_specs, out_specs, manual_axes, check_vma)
}

/// Traced shard-map program backed by a staged `tracing_v2` program.
///
/// [`TracedShardMap`] extends internal shard-map metadata with both the traced local body program and the
/// reconstructed global/local boundary types, making it the main inspection and lowering handle
/// returned by [`shard_map`] and [`shard_map_with_options`].
#[allow(private_bounds, private_interfaces)]
pub struct TracedShardMap<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
{
    /// Manual SPMD metadata describing how the body is partitioned over the mesh.
    shard_map: ShardMap,

    /// Global input types supplied by the caller.
    global_input_types: Input,

    /// Local input types seen by the traced manual body.
    local_input_types: Input,

    /// Global output types reconstructed from the traced local outputs.
    global_output_types: Output,

    /// Local output types produced by the traced manual body.
    local_output_types: Output,

    /// Staged traced body specialized to abstract shard-map tensor leaves.
    program: XlaProgram<ShardMapCapturedInput<Input>, ShardMapCapturedOutput<Output>>,
}

/// Traced XLA program backed by a staged `tracing_v2` program.
#[allow(private_bounds, private_interfaces)]
pub struct TracedXlaProgram<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>
where
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
{
    /// Global input types supplied to the traced function.
    global_input_types: Input,

    /// Global output types inferred by tracing the function.
    global_output_types: Output,

    /// Staged traced XLA program specialized to abstract shard-map tensor leaves.
    program: XlaProgram<ShardMapCapturedInput<Input>, ShardMapCapturedOutput<Output>>,
}

/// Metadata describing one manual SPMD computation over a mesh.
///
/// A `ShardMap` stores the mesh plus the validated per-input and per-output shardings, the active
/// manual-axis subset, and whether JAX-style `check_vma` validation is enabled.
///
/// The public constructors accept [`Sharding`] values and project them into
/// traced/type-level semantics, so `Auto` mesh axes remain hidden while `Manual` axes still
/// drive the manual-computation body. When the surrounding mesh also contains free axes, the
/// rendered `in_shardings` and `out_shardings` dimensions stay open so Shardy can propagate those
/// free axes across the manual region.
///
/// This metadata is ultimately rendered into the three `sdy.manual_computation` attributes:
/// `in_shardings`, `out_shardings`, and `manual_axes`.
///
/// Reference: https://docs.jax.dev/en/latest/notebooks/shard_map.html.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ShardMap {
    /// Logical mesh that the manual computation is defined over.
    mesh: LogicalMesh,

    /// Validated shardings for each global input leaf.
    in_shardings: Vec<Sharding>,

    /// Validated shardings for each global output leaf.
    out_shardings: Vec<Sharding>,

    /// Active manual mesh axes for this shard-map invocation.
    manual_axes: Vec<String>,

    /// Whether to enforce JAX-style omitted-manual-axis output validation.
    check_vma: bool,
}

impl ShardMap {
    /// Creates a `ShardMap` with one explicit manual-axis selection and `check_vma` mode.
    ///
    /// When `manual_axes` is empty, every mesh axis with type
    /// [`Manual`](ryft_core::arrays::MeshAxisType::Manual) is treated as manual inside the body.
    /// The constructor returns [`ShardMapError::MeshHasNoManualAxes`] if the resulting active set
    /// is empty.
    ///
    /// # Parameters
    ///
    ///   - `mesh`: Logical mesh that the manual computation is defined over.
    ///   - `in_specs`: Per-input shardings for the global inputs.
    ///   - `out_specs`: Per-output shardings for the global outputs.
    ///   - `manual_axes`: Active manual mesh axes for this shard-map. An empty list means "all
    ///     manual mesh axes".
    ///   - `check_vma`: Whether to reject outputs that still vary along active manual axes omitted
    ///     from `out_specs`.
    pub(crate) fn new(
        mesh: LogicalMesh,
        in_specs: Vec<Sharding>,
        out_specs: Vec<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Result<Self, ShardMapError> {
        let manual_axes = normalize_manual_axes(&mesh, manual_axes)?;
        let in_shardings = build_shardings(&mesh, manual_axes.as_slice(), in_specs, "input")?;
        let out_shardings = build_shardings(&mesh, manual_axes.as_slice(), out_specs, "output")?;
        Ok(Self { mesh, in_shardings, out_shardings, manual_axes, check_vma })
    }

    /// Builds a shard map directly from already-validated shardings.
    pub(crate) fn from_shardings(
        mesh: LogicalMesh,
        in_shardings: Vec<Sharding>,
        out_shardings: Vec<Sharding>,
        manual_axes: Vec<String>,
        check_vma: bool,
    ) -> Self {
        Self { mesh, in_shardings, out_shardings, manual_axes, check_vma }
    }

    /// Returns the logical mesh of this manual computation.
    pub(crate) fn mesh(&self) -> &LogicalMesh {
        &self.mesh
    }

    /// Returns the validated per-input shardings.
    pub(crate) fn in_shardings(&self) -> &[Sharding] {
        self.in_shardings.as_slice()
    }

    /// Returns the validated per-output shardings.
    pub(crate) fn out_shardings(&self) -> &[Sharding] {
        self.out_shardings.as_slice()
    }

    /// Returns the active manual mesh axes for this shard-map.
    pub(crate) fn manual_axes(&self) -> &[String] {
        self.manual_axes.as_slice()
    }

    fn manual_axis_names(&self) -> HashSet<&str> {
        self.manual_axes.iter().map(String::as_str).collect()
    }

    pub(crate) fn check_vma(&self) -> bool {
        self.check_vma
    }

    /// Returns the local body shape for input `input_index`.
    ///
    /// The returned shape is the tensor shape seen inside the manual computation body for the
    /// corresponding global input. Only manual axes reduce the local shape; free axes remain
    /// global from the body's point of view.
    ///
    /// # Parameters
    ///
    ///   - `input_index`: Index of the input sharding to use.
    ///   - `global_shape`: Global input shape associated with that input.
    fn local_input_shape(&self, input_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        local_shape_for_sharding(
            &self.in_shardings[input_index],
            self.manual_axis_names(),
            global_shape,
            "input",
            input_index,
        )
    }

    /// Returns the local body shape for output `output_index`.
    ///
    /// # Parameters
    ///
    ///   - `output_index`: Index of the output sharding to use.
    ///   - `global_shape`: Global output shape associated with that output.
    #[cfg(test)]
    fn local_output_shape(&self, output_index: usize, global_shape: &[usize]) -> Result<Vec<usize>, ShardMapError> {
        local_shape_for_sharding(
            &self.out_shardings[output_index],
            self.manual_axis_names(),
            global_shape,
            "output",
            output_index,
        )
    }

    /// Renders the Shardy `in_shardings=[...]` attribute payload.
    ///
    /// The returned string is suitable for direct insertion into an `sdy.manual_computation`
    /// operation.
    ///
    #[cfg(test)]
    fn to_shardy_in_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.in_shardings.as_slice(), self.manual_axes())
    }

    /// Renders the Shardy `out_shardings=[...]` attribute payload.
    ///
    #[cfg(test)]
    fn to_shardy_out_shardings_attribute(&self) -> String {
        render_shardy_sharding_list(self.out_shardings.as_slice(), self.manual_axes())
    }

    /// Renders the Shardy `manual_axes={...}` attribute payload.
    #[cfg(test)]
    fn to_shardy_manual_axes_attribute(&self) -> String {
        render_shardy_axes(self.manual_axes())
    }

    /// Renders the three Shardy attributes attached to `sdy.manual_computation`.
    ///
    #[cfg(test)]
    fn to_shardy_manual_computation_attributes(&self) -> String {
        let in_shardings = self.to_shardy_in_shardings_attribute();
        let out_shardings = self.to_shardy_out_shardings_attribute();
        format!(
            "in_shardings={in_shardings} out_shardings={out_shardings} manual_axes={}",
            self.to_shardy_manual_axes_attribute()
        )
    }

    /// Builds the typed Shardy `in_shardings` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_in_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
        shardy_tensor_sharding_per_value(self.in_shardings.as_slice(), self.manual_axes(), context)
    }

    /// Builds the typed Shardy `out_shardings` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_out_shardings<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
        shardy_tensor_sharding_per_value(self.out_shardings.as_slice(), self.manual_axes(), context)
    }

    /// Builds the typed Shardy `manual_axes` attribute used by `sdy.manual_computation`.
    pub(crate) fn to_shardy_manual_axes<'c, 't>(
        &self,
        context: &'c MlirContext<'t>,
    ) -> Result<ManualAxesAttributeRef<'c, 't>, ryft_mlir::Error> {
        context.shardy_manual_axes(self.manual_axes())
    }

    /// Traces a shard-map body over local body tensor types using [`TracingContext::trace`].
    ///
    /// # Parameters
    ///
    ///   - `function`: Body closure to trace over local shard-map values.
    ///   - `global_input_types`: Global input array types in the same leaf order as the shard-map
    ///     input shardings.
    pub(crate) fn trace<
        F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
        Input: Parameterized<ArrayType>,
        Output: Parameterized<ArrayType>,
    >(
        &self,
        function: F,
        global_input_types: Input,
    ) -> Result<TracedShardMap<Input, Output>, ShardMapTraceError>
    where
        Input::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
        Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
    {
        let global_input_types = derive_global_input_types(self, &global_input_types)?;
        let local_input_types = derive_local_input_types(self, &global_input_types)?;
        let (local_output_types, program) =
            trace_xla_function(function, &local_input_types, shard_map_named_axes(self))?;
        let global_output_types = derive_global_output_types(self, &local_output_types)?;

        Ok(TracedShardMap {
            shard_map: self.clone(),
            global_input_types,
            local_input_types,
            global_output_types,
            local_output_types,
            program,
        })
    }
}

#[allow(private_bounds, private_interfaces)]
impl<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>> TracedShardMap<Input, Output>
where
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
{
    /// Returns the global input types used to derive the traced local body inputs.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced local body input types.
    pub fn local_input_types(&self) -> &Input {
        &self.local_input_types
    }

    /// Returns the traced local body output types.
    pub fn local_output_types(&self) -> &Output {
        &self.local_output_types
    }

    /// Returns the reconstructed global output types implied by the traced body and output shardings.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced shard-map.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, ShardMapTraceError> {
        let simplified_program = self.program.simplified()?;
        super::lowering::to_mlir_module(
            &self.shard_map,
            &simplified_program,
            &self.global_input_types,
            &self.local_input_types,
            &self.global_output_types,
            &self.local_output_types,
            function_name,
        )
        .map_err(ShardMapTraceError::from)
    }
}

#[allow(private_bounds, private_interfaces)]
impl<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>> TracedXlaProgram<Input, Output>
where
    Input::Family: ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
{
    /// Returns backend-neutral structural statistics for the staged traced XLA program backing this handle. The
    /// statistics describe the exact unsimplified traced program, so instructions whose outputs reach no program
    /// output are included in the reported counts. Refer to the documentation of
    /// [`Program::statistics`](ryft_core::Program::statistics) for the precise semantics of the reported statistics.
    pub fn statistics(&self) -> ProgramStatistics {
        self.program.statistics()
    }

    /// Returns the traced global input types.
    pub fn global_input_types(&self) -> &Input {
        &self.global_input_types
    }

    /// Returns the traced global output types.
    pub fn global_output_types(&self) -> &Output {
        &self.global_output_types
    }

    /// Renders a full StableHLO/Shardy MLIR module for this traced XLA program.
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    pub fn to_mlir_module<S: AsRef<str>>(&self, function_name: S) -> Result<String, ShardMapTraceError> {
        self.to_mlir_module_with_signature_shardings(function_name, None, None)
    }

    /// Same as [`Self::to_mlir_module`] but additionally attaches `sdy.sharding` attributes to the
    /// function's arguments and/or results when shardings are provided.
    ///
    /// This is what the XLA SPMD partitioner reads to drive boundary slicing of per-device output
    /// buffers, including for shapes whose dimensions are not divisible by the partition count
    /// (e.g. shape `[5]` on 2 partitions producing `[3]` + `[2]`).
    ///
    /// # Parameters
    ///
    ///   - `function_name`: Symbol name to use for the outer `func.func`.
    ///   - `arg_shardings`: Optional shardings to attach to each function argument. Must have the
    ///     same length as the global input types, or be `None`.
    ///   - `result_shardings`: Optional shardings to attach to each function result. Must have
    ///     the same length as the global output types, or be `None`.
    pub fn to_mlir_module_with_signature_shardings<S: AsRef<str>>(
        &self,
        function_name: S,
        arg_shardings: Option<&[Sharding]>,
        result_shardings: Option<&[Sharding]>,
    ) -> Result<String, ShardMapTraceError> {
        let pruned_program = prune_dead_shard_map_outputs(&self.program)?;
        let simplified_program = pruned_program.simplified()?;
        super::lowering::to_mlir_module_for_program(
            &simplified_program,
            &[],
            &self.global_input_types,
            &self.global_output_types,
            function_name,
            arg_shardings,
            result_shardings,
        )
        .map_err(ShardMapTraceError::from)
    }
}

/// Erased shard-map body payload used by nested higher-order shard-map ops.
#[derive(Clone, Debug)]
pub struct FlatTracedShardMap {
    /// Manual SPMD metadata carried by this erased shard-map body.
    shard_map: ShardMap,

    /// Global input types corresponding to the erased body inputs.
    global_input_types: Vec<ArrayType>,

    /// Local input types seen inside the erased shard-map body.
    local_input_types: Vec<ArrayType>,

    /// Global output types reconstructed from the erased body outputs.
    global_output_types: Vec<ArrayType>,

    /// Local output types produced inside the erased shard-map body.
    local_output_types: Vec<ArrayType>,

    /// Flattened staged program implementing the erased shard-map body.
    program: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
}

impl FlatTracedShardMap {
    /// Creates a new [`FlatTracedShardMap`] from explicit traced components.
    pub(crate) fn new(
        shard_map: ShardMap,
        global_input_types: Vec<ArrayType>,
        local_input_types: Vec<ArrayType>,
        global_output_types: Vec<ArrayType>,
        local_output_types: Vec<ArrayType>,
        program: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    ) -> Self {
        Self { shard_map, global_input_types, local_input_types, global_output_types, local_output_types, program }
    }

    /// Builds an erased shard-map body from explicit traced components.
    #[inline]
    pub(crate) fn from_parts(
        shard_map: ShardMap,
        global_input_types: Vec<ArrayType>,
        local_input_types: Vec<ArrayType>,
        global_output_types: Vec<ArrayType>,
        local_output_types: Vec<ArrayType>,
        program: XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>,
    ) -> Self {
        Self::new(shard_map, global_input_types, local_input_types, global_output_types, local_output_types, program)
    }

    /// Splits this erased body into the boundary metadata consumed by
    /// [`ShardMapOperation`](crate::experimental::operations::shard_map::ShardMapOperation) — the manual SPMD
    /// metadata plus the global boundary types — and the local body program that rides as the operation's attached
    /// `body` region. The local boundary types are dropped: the region program is authoritative for them.
    #[inline]
    pub(crate) fn into_operation_parts(
        self,
    ) -> (ShardMap, Vec<ArrayType>, Vec<ArrayType>, XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>>) {
        (self.shard_map, self.global_input_types, self.global_output_types, self.program)
    }

    /// Returns the global input types corresponding to the erased body inputs.
    #[inline]
    pub fn global_input_types(&self) -> &[ArrayType] {
        &self.global_input_types
    }

    /// Returns the local input types seen inside the erased shard-map body.
    #[inline]
    pub fn local_input_types(&self) -> &[ArrayType] {
        &self.local_input_types
    }

    /// Returns the global output types reconstructed from the erased body outputs.
    #[inline]
    pub fn global_output_types(&self) -> &[ArrayType] {
        &self.global_output_types
    }

    /// Returns the local output types produced inside the erased shard-map body.
    #[inline]
    pub fn local_output_types(&self) -> &[ArrayType] {
        &self.local_output_types
    }

    /// Returns the flattened staged program implementing the erased shard-map body.
    #[inline]
    pub fn program(&self) -> &XlaProgram<Vec<XlaConstant>, Vec<XlaConstant>> {
        &self.program
    }

    /// Builds an erased shard-map body from the typed traced representation.
    pub(crate) fn from_traced<Input: Parameterized<ArrayType>, Output: Parameterized<ArrayType>>(
        traced: &TracedShardMap<Input, Output>,
    ) -> Self
    where
        Input::Family:
            ParameterizedFamily<ArrayType> + ParameterizedFamily<ArrayIrType> + ParameterizedFamily<XlaConstant>,
        Output::Family: ParameterizedFamily<ArrayType>
            + ParameterizedFamily<ArrayIrType>
            + ParameterizedFamily<XlaConstant>
            + ParameterizedFamily<ShardMapTracer>,
    {
        let local_input_types = traced.local_input_types.parameters().cloned().collect::<Vec<_>>();
        let local_output_types = traced.local_output_types.parameters().cloned().collect::<Vec<_>>();
        let input_count = traced.program.input_ids().len();
        let output_count = traced.program.output_ids().len();
        let program = rebuild_xla_program_with_builder(
            &traced.program,
            traced.program.output_ids().to_vec(),
            vec![Placeholder; input_count],
            vec![Placeholder; output_count],
        )
        .expect("retyping a traced shard_map program should preserve valid program metadata");
        Self::from_parts(
            traced.shard_map.clone(),
            traced.global_input_types.parameters().cloned().collect::<Vec<_>>(),
            local_input_types,
            traced.global_output_types.parameters().cloned().collect::<Vec<_>>(),
            local_output_types,
            program,
        )
    }

    /// Returns a copy of this erased shard-map body restricted to the outputs selected by `live_outputs`, dropping the
    /// dead body outputs and their boundary metadata.
    ///
    /// `live_outputs` carries one flag per current global output. The retained outputs keep their global types, local
    /// types, and per-instance output shardings; the body program is rebuilt over the same input boundary with only the
    /// live output atoms exposed and then simplified, so the computation feeding the dropped outputs is pruned. The
    /// global and local input boundaries are unchanged so the operand signature seen by callers stays fixed.
    ///
    /// This backs dead-output elimination for forward-mode `shard_map`s, whose primal body carries the actual
    /// primal outputs alongside the residual edges; when a primal output is dead in a gradient-only program, this
    /// projects it away (see `prune_dead_shard_map_outputs`).
    ///
    /// # Parameters
    ///
    ///   - `live_outputs`: One flag per current global output; `true` keeps the output.
    pub(crate) fn with_live_outputs(&self, live_outputs: &[bool]) -> Result<Self, ShardMapTraceError> {
        if live_outputs.len() != self.global_output_types.len() {
            return Err(ShardMapTraceError::OutputTypeCountMismatch {
                expected: self.global_output_types.len(),
                actual: live_outputs.len(),
            });
        }
        let live_output_ids = self
            .program
            .output_ids()
            .iter()
            .copied()
            .zip(live_outputs.iter().copied())
            .filter_map(|(output_id, live)| live.then_some(output_id))
            .collect::<Vec<_>>();
        let live_output_count = live_output_ids.len();
        let program = rebuild_xla_program_with_builder::<Vec<XlaConstant>, Vec<XlaConstant>, _, _>(
            &self.program,
            live_output_ids,
            vec![Placeholder; self.program.input_ids().len()],
            vec![Placeholder; live_output_count],
        )?
        .simplified()?;
        let retain = |types: &[ArrayType]| {
            types
                .iter()
                .cloned()
                .zip(live_outputs.iter().copied())
                .filter_map(|(t, live)| live.then_some(t))
                .collect()
        };
        let live_out_shardings = self
            .shard_map
            .out_shardings()
            .iter()
            .cloned()
            .zip(live_outputs.iter().copied())
            .filter_map(|(sharding, live)| live.then_some(sharding))
            .collect::<Vec<_>>();
        let shard_map = ShardMap::from_shardings(
            self.shard_map.mesh().clone(),
            self.shard_map.in_shardings().to_vec(),
            live_out_shardings,
            self.shard_map.manual_axes().to_vec(),
            self.shard_map.check_vma(),
        );
        Ok(Self::from_parts(
            shard_map,
            self.global_input_types.clone(),
            self.local_input_types.clone(),
            retain(&self.global_output_types),
            retain(&self.local_output_types),
            program,
        ))
    }
}

fn axes_to_vec(axis_names: &BTreeSet<String>) -> Vec<String> {
    axis_names.iter().cloned().collect()
}

fn varying_axes(sharding: Option<&Sharding>) -> BTreeSet<String> {
    sharding.map(|sharding| sharding.varying_manual_axes().clone()).unwrap_or_default()
}

fn sharding_with_varying_manual_axes(
    sharding: &Sharding,
    varying_axes: BTreeSet<String>,
) -> Result<Sharding, ShardMapTraceError> {
    let varying_axes = varying_axes
        .into_iter()
        .filter(|axis_name| sharding.mesh().axis_type(axis_name) == Some(MeshAxisType::Manual))
        .collect::<BTreeSet<_>>();
    Ok(sharding.clone().with_varying_manual_axes(varying_axes)?)
}

fn axes_match(left: &BTreeSet<String>, right: &BTreeSet<String>) -> bool {
    left == right
}

fn validate_input_sharding_state(
    actual: Option<&Sharding>,
    expected: &Sharding,
    input_index: usize,
) -> Result<(), ShardMapTraceError> {
    let Some(actual) = actual else {
        return Ok(());
    };
    if !axes_match(actual.unreduced_axes(), expected.unreduced_axes()) {
        return Err(ShardMapTraceError::ShardingStateMismatch {
            value_kind: "input",
            value_index: input_index,
            state_kind: "unreduced axes",
            expected: axes_to_vec(expected.unreduced_axes()),
            actual: axes_to_vec(actual.unreduced_axes()),
        });
    }
    if !axes_match(actual.reduced_axes(), expected.reduced_axes()) {
        return Err(ShardMapTraceError::ShardingStateMismatch {
            value_kind: "input",
            value_index: input_index,
            state_kind: "reduced axes",
            expected: axes_to_vec(expected.reduced_axes()),
            actual: axes_to_vec(actual.reduced_axes()),
        });
    }
    Ok(())
}

fn spec_varying_axes(sharding: &Sharding, manual_axis_names: &HashSet<&str>) -> BTreeSet<String> {
    let mut varying_axes = BTreeSet::new();
    for partition_dimension in sharding.dimensions() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            for axis_name in axis_names {
                if manual_axis_names.contains(axis_name.as_str()) {
                    varying_axes.insert(axis_name.clone());
                }
            }
        }
    }
    varying_axes
}

fn derive_global_input_types<Input: Parameterized<ArrayType>>(
    shard_map: &ShardMap,
    global_input_types: &Input,
) -> Result<Input, ShardMapTraceError> {
    let global_input_type_count = global_input_types.parameter_count();
    if global_input_type_count != shard_map.in_shardings().len() {
        return Err(ShardMapTraceError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_type_count,
        });
    }

    let structure = global_input_types.parameter_structure();
    let global_input_types = global_input_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(input_index, global_input_type)| {
            let sharding = shard_map.in_shardings()[input_index].clone();
            validate_input_sharding_state(global_input_type.sharding(), &sharding, input_index)?;
            let varying_axes = varying_axes(global_input_type.sharding());
            let global_input_type = ArrayType::new(global_input_type.data_type(), global_input_type.shape().clone())
                .with_layout(global_input_type.layout().cloned())
                .with_sharding(sharding_with_varying_manual_axes(&sharding, varying_axes)?)?;
            Ok::<ArrayType, ShardMapTraceError>(global_input_type)
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, global_input_types)?)
}

fn derive_local_input_types<Input: Parameterized<ArrayType>>(
    shard_map: &ShardMap,
    global_input_types: &Input,
) -> Result<Input, ShardMapTraceError> {
    let global_input_type_count = global_input_types.parameter_count();
    if global_input_type_count != shard_map.in_shardings().len() {
        return Err(ShardMapTraceError::InputTypeCountMismatch {
            expected: shard_map.in_shardings().len(),
            actual: global_input_type_count,
        });
    }

    let manual_axis_names = shard_map.manual_axis_names();
    let structure = global_input_types.parameter_structure();
    let local_input_types = global_input_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(input_index, global_input_type)| {
            let global_shape = static_dimensions(&global_input_type, "input", input_index)?;
            let local_shape = shard_map.local_input_shape(input_index, &global_shape)?;
            let local_sharding = shard_map.in_shardings()[input_index].clone();
            let local_varying_axes = varying_axes(global_input_type.sharding())
                .union(&spec_varying_axes(&local_sharding, &manual_axis_names))
                .cloned()
                .collect();
            Ok::<ArrayType, ShardMapTraceError>(
                ArrayType::new(
                    global_input_type.data_type(),
                    Shape::new(local_shape.into_iter().map(Dimension::Static).collect()),
                )
                .with_layout(global_input_type.layout().cloned())
                .with_sharding(sharding_with_varying_manual_axes(&local_sharding, local_varying_axes)?)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Input::from_parameters(structure, local_input_types)?)
}

fn trace_xla_function<
    F: FnOnce(ShardMapLocalTraceInput<Input>) -> ShardMapLocalTraceOutput<Output>,
    Input: Parameterized<ArrayType>,
    Output: Parameterized<ArrayType>,
>(
    function: F,
    input_types: &Input,
    named_axes: Vec<(String, NamedAxis)>,
) -> Result<(Output, XlaProgram<ShardMapCapturedInput<Input>, ShardMapCapturedOutput<Output>>), ShardMapTraceError>
where
    Input::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::Family: ParameterizedFamily<ArrayType>
        + ParameterizedFamily<ArrayIrType>
        + ParameterizedFamily<XlaConstant>
        + ParameterizedFamily<ShardMapTracer>,
    Output::To<ShardMapTracer>: Parameterized<ShardMapTracer, To<ArrayType> = Output>,
{
    let input_structure = input_types.parameter_structure();
    let flat_input_types = input_types.parameters().cloned().map(ArrayIrType::from).collect::<Vec<_>>();
    let output_structure = RefCell::new(None);
    let (flat_output_types, program) = DomainTracingContext::<XlaDomain<'static>>::trace_with_named_axes(
        |input: Vec<XlaTracer<'static>>| {
            let input = input
                .into_iter()
                .map(|value| ValueProjection::<ArrayType>::into_projected(value).map_err(ProgramError::from))
                .collect::<Result<Vec<_>, _>>()?;
            let input = ShardMapLocalTraceInput::<Input>::from_parameters(input_structure.clone(), input)?;
            let output = function(input);
            output_structure.replace(Some(output.parameter_structure()));
            Ok(output.into_parameters().map(ProjectedValue::into_value).collect::<Vec<_>>())
        },
        flat_input_types,
        named_axes,
    )?;
    let output_structure = output_structure.into_inner().ok_or_else(|| {
        ProgramError::MalformedProgram("shard-map tracing completed without recording its output structure".to_string())
    })?;
    let output_parameters = flat_output_types
        .into_iter()
        .map(|r#type| <&ArrayType>::try_from(&r#type).cloned().map_err(ProgramError::from))
        .collect::<Result<Vec<_>, _>>()?;
    let output_types = Output::from_parameters(output_structure.clone(), output_parameters)?;
    let program = program.simplified()?.restructured::<ShardMapCapturedInput<Input>, ShardMapCapturedOutput<Output>>(
        input_structure,
        output_structure,
    )?;
    Ok((output_types, program))
}

/// Returns the named-axis bindings a `shard_map` body trace is seeded with: one [`NamedAxis::Mesh`] binding per
/// manual mesh axis of `shard_map`, so collectives staged inside the body resolve those axes by name.
fn shard_map_named_axes(shard_map: &ShardMap) -> Vec<(String, NamedAxis)> {
    let mesh = shard_map.mesh();
    shard_map
        .manual_axes()
        .iter()
        .map(|name| {
            let axis = mesh.axis_index(name).expect("manual axes are validated against the mesh at construction");
            let size = mesh.axis_size(name).expect("manual axes are validated against the mesh at construction");
            (name.clone(), NamedAxis::Mesh { axis, size })
        })
        .collect()
}

pub(crate) fn derive_global_output_types<Output: Parameterized<ArrayType>>(
    shard_map: &ShardMap,
    local_output_types: &Output,
) -> Result<Output, ShardMapTraceError> {
    let local_output_type_count = local_output_types.parameter_count();
    if local_output_type_count != shard_map.out_shardings().len() {
        return Err(ShardMapTraceError::OutputTypeCountMismatch {
            expected: shard_map.out_shardings().len(),
            actual: local_output_type_count,
        });
    }

    let manual_axis_names = shard_map.manual_axis_names();
    let structure = local_output_types.parameter_structure();
    let global_output_types = local_output_types
        .parameters()
        .cloned()
        .enumerate()
        .map(|(output_index, local_output_type)| {
            let local_shape = static_dimensions(&local_output_type, "output", output_index)?;
            let output_sharding = &shard_map.out_shardings()[output_index];
            let expected_current_varying_axes = spec_varying_axes(output_sharding, &manual_axis_names);
            let effective_local_varying_axes: BTreeSet<String> =
                varying_axes(local_output_type.sharding()).union(&expected_current_varying_axes).cloned().collect();
            if shard_map.check_vma() {
                let local_unreduced_axes =
                    local_output_type.sharding().map(|sharding| sharding.unreduced_axes().clone()).unwrap_or_default();
                let effective_local_unreduced_axes =
                    local_unreduced_axes.union(output_sharding.unreduced_axes()).cloned().collect();
                if !axes_match(&effective_local_unreduced_axes, output_sharding.unreduced_axes()) {
                    return Err(ShardMapTraceError::ShardingStateMismatch {
                        value_kind: "output",
                        value_index: output_index,
                        state_kind: "unreduced axes",
                        expected: axes_to_vec(output_sharding.unreduced_axes()),
                        actual: axes_to_vec(&local_unreduced_axes),
                    });
                }

                let local_reduced_axes =
                    local_output_type.sharding().map(|sharding| sharding.reduced_axes().clone()).unwrap_or_default();
                if !axes_match(&local_reduced_axes, output_sharding.reduced_axes()) {
                    return Err(ShardMapTraceError::ShardingStateMismatch {
                        value_kind: "output",
                        value_index: output_index,
                        state_kind: "reduced axes",
                        expected: axes_to_vec(output_sharding.reduced_axes()),
                        actual: axes_to_vec(&local_reduced_axes),
                    });
                }

                for axis_name in &effective_local_varying_axes {
                    if manual_axis_names.contains(axis_name.as_str())
                        && !expected_current_varying_axes.contains(axis_name)
                    {
                        return Err(ShardMapTraceError::OutputVaryingManualAxisNotInOutSpecs {
                            output_index,
                            axis_name: axis_name.clone(),
                        });
                    }
                }
            }
            let surviving_varying_axes = effective_local_varying_axes
                .into_iter()
                .filter(|axis_name| !manual_axis_names.contains(axis_name.as_str()))
                .collect::<BTreeSet<_>>();
            let global_shape =
                global_shape_for_sharding(output_sharding, &manual_axis_names, local_shape, output_index)?;
            Ok::<ArrayType, ShardMapTraceError>(
                ArrayType::new(
                    local_output_type.data_type(),
                    Shape::new(global_shape.into_iter().map(Dimension::Static).collect()),
                )
                .with_layout(local_output_type.layout().cloned())
                .with_sharding(sharding_with_varying_manual_axes(output_sharding, surviving_varying_axes)?)?,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Output::from_parameters(structure, global_output_types)?)
}

fn static_dimensions(
    array_type: &ArrayType,
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapTraceError> {
    if let Some(shape) = array_type.static_shape() {
        return Ok(shape.dimensions().to_vec());
    }

    let dimension = array_type
        .shape()
        .dimensions()
        .iter()
        .position(|size| !matches!(size, Dimension::Static(_)))
        .expect("array types without static shapes should have at least one dynamic dimension");
    Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind, value_index, dimension })
}

fn global_shape_for_sharding(
    sharding: &Sharding,
    manual_axis_names: &HashSet<&str>,
    local_shape: Vec<usize>,
    output_index: usize,
) -> Result<Vec<usize>, ShardMapTraceError> {
    if sharding.rank() != local_shape.len() {
        return Err(ShardMapTraceError::RankMismatch {
            value_kind: "output",
            value_index: output_index,
            partition_rank: sharding.rank(),
            shape_rank: local_shape.len(),
        });
    }

    sharding
        .dimensions()
        .iter()
        .zip(local_shape)
        .enumerate()
        .map(|(dimension, (partition_dimension, local_dimension_size))| {
            let manual_partition_count = match partition_dimension {
                ShardingDimension::Sharded(axis_names) => axis_names
                    .iter()
                    .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                    .try_fold(1usize, |partition_count, axis_name| {
                        let axis_size = sharding.mesh().axis_size(axis_name).ok_or_else(|| {
                            ShardMapTraceError::ShardingError(ShardingError::UnknownMeshAxisName {
                                name: axis_name.clone(),
                            })
                        })?;
                        partition_count.checked_mul(axis_size).ok_or_else(|| ShardMapTraceError::Overflow {
                            context: format!(
                                "computing global output shape for output #{output_index} dimension #{dimension}"
                            ),
                        })
                    })?,
                ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
            };

            local_dimension_size
                .checked_mul(manual_partition_count)
                .ok_or_else(|| ShardMapTraceError::Overflow {
                    context: format!("computing global output size for output #{output_index} dimension #{dimension}"),
                })
        })
        .collect()
}

fn build_shardings(
    mesh: &LogicalMesh,
    manual_axes: &[String],
    shardings: Vec<Sharding>,
    value_kind: &'static str,
) -> Result<Vec<Sharding>, ShardMapError> {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    shardings
        .into_iter()
        .enumerate()
        .map(|(value_index, sharding)| {
            if sharding.mesh() != mesh {
                return Err(ShardMapError::ShardingError(ShardingError::MeshMismatch {
                    expected: mesh.clone(),
                    actual: sharding.mesh().clone(),
                }));
            }
            validate_manual_axis_order(&sharding, &manual_axis_names, value_kind, value_index)?;
            Ok(sharding.without_auto_axes())
        })
        .collect()
}

fn normalize_manual_axes(mesh: &LogicalMesh, manual_axes: Vec<String>) -> Result<Vec<String>, ShardMapError> {
    let selected_manual_axes = if manual_axes.is_empty() {
        None
    } else {
        let mut selected_manual_axes = HashSet::new();
        for axis_name in manual_axes {
            if mesh.axis_index(axis_name.as_str()).is_none() {
                return Err(ShardMapError::ShardingError(ShardingError::UnknownMeshAxisName { name: axis_name }));
            }
            if mesh.axis_type(axis_name.as_str()) != Some(MeshAxisType::Manual) {
                return Err(ShardMapError::ShardingError(ShardingError::ExpectedManualMeshAxis { name: axis_name }));
            }
            selected_manual_axes.insert(axis_name);
        }
        Some(selected_manual_axes)
    };
    let manual_axes = mesh
        .axes()
        .iter()
        .filter_map(|axis| {
            (axis.r#type() == MeshAxisType::Manual
                && match &selected_manual_axes {
                    None => true,
                    Some(selected_manual_axes) => selected_manual_axes.contains(axis.name()),
                })
            .then(|| axis.name().to_string())
        })
        .collect::<Vec<_>>();
    if manual_axes.is_empty() {
        return Err(ShardMapError::MeshHasNoManualAxes);
    }
    Ok(manual_axes)
}

fn validate_manual_axis_order(
    sharding: &Sharding,
    manual_axes: &HashSet<&str>,
    value_kind: &'static str,
    value_index: usize,
) -> Result<(), ShardMapError> {
    for (dimension, partition_dimension) in sharding.dimensions().iter().enumerate() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            let mut first_free_axis: Option<&str> = None;
            for axis_name in axis_names {
                if manual_axes.contains(axis_name.as_str()) {
                    if let Some(free_axis_name) = first_free_axis {
                        return Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                            value_kind,
                            value_index,
                            dimension,
                            free_axis_name: free_axis_name.to_string(),
                            manual_axis_name: axis_name.clone(),
                        });
                    }
                } else if first_free_axis.is_none() {
                    first_free_axis = Some(axis_name.as_str());
                }
            }
        }
    }
    Ok(())
}

fn local_shape_for_sharding(
    sharding: &Sharding,
    manual_axis_names: HashSet<&str>,
    global_shape: &[usize],
    value_kind: &'static str,
    value_index: usize,
) -> Result<Vec<usize>, ShardMapError> {
    if sharding.rank() != global_shape.len() {
        return Err(ShardMapError::RankMismatch {
            value_kind,
            value_index,
            partition_rank: sharding.rank(),
            shape_rank: global_shape.len(),
        });
    }

    let mut local_shape = Vec::with_capacity(global_shape.len());
    for (dimension, (partition_dimension, dimension_size)) in
        sharding.dimensions().iter().zip(global_shape.iter().copied()).enumerate()
    {
        let manual_partition_count = match partition_dimension {
            ShardingDimension::Sharded(axis_names) => axis_names
                .iter()
                .filter(|axis_name| manual_axis_names.contains(axis_name.as_str()))
                .try_fold(1usize, |partition_count, axis_name| -> Result<usize, ShardMapError> {
                    let axis_size = sharding
                        .mesh()
                        .axis_size(axis_name)
                        .ok_or_else(|| ShardingError::UnknownMeshAxisName { name: axis_name.clone() })?;
                    Ok(partition_count * axis_size)
                })?,
            ShardingDimension::Replicated | ShardingDimension::Unconstrained => 1,
        };

        if dimension_size % manual_partition_count != 0 {
            return Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind,
                value_index,
                dimension,
                dimension_size,
                manual_partition_count,
            });
        }

        local_shape.push(dimension_size / manual_partition_count);
    }
    Ok(local_shape)
}

#[cfg(test)]
fn render_shardy_sharding_list(shardings: &[Sharding], manual_axes: &[String]) -> String {
    let mut result = String::from("[");
    for (sharding_index, sharding) in shardings.iter().enumerate() {
        if sharding_index > 0 {
            result.push_str(", ");
        }
        result.push_str(stripped_shardy_tensor_sharding(sharding, manual_axes).as_str());
    }
    result.push(']');
    result
}

fn shardy_tensor_sharding_per_value<'c, 't>(
    shardings: &[Sharding],
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingPerValueAttributeRef<'c, 't>, ryft_mlir::Error> {
    let shardings = shardings
        .iter()
        .map(|sharding| manual_computation_tensor_sharding(sharding, manual_axes, context))
        .collect::<Result<Vec<_>, _>>()?;
    context.shardy_tensor_sharding_per_value(shardings.as_slice())
}

fn manual_computation_tensor_sharding<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<TensorShardingAttributeRef<'c, 't>, ryft_mlir::Error> {
    let mesh_symbol_ref = context.flat_symbol_ref_attribute(SHARDY_MESH_SYMBOL_NAME);
    let dim_shardings = manual_computation_dimension_shardings(sharding, manual_axes, context)?;
    let replicated_axis_names = sharding.replicated_axes();
    let replicated_axes = replicated_axis_names
        .iter()
        .map(|axis_name| context.shardy_axis_ref(*axis_name, None))
        .collect::<Result<Vec<_>, _>>()?;
    let unreduced_axes = sharding
        .unreduced_axes()
        .iter()
        .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
        .collect::<Result<Vec<_>, _>>()?;
    context.shardy_tensor_sharding(
        mesh_symbol_ref,
        dim_shardings.as_slice(),
        replicated_axes.as_slice(),
        unreduced_axes.as_slice(),
        ReductionOperation::Sum,
    )
}

fn manual_computation_dimension_shardings<'c, 't>(
    sharding: &Sharding,
    manual_axes: &[String],
    context: &'c MlirContext<'t>,
) -> Result<Vec<DimensionShardingAttributeRef<'c, 't>>, ryft_mlir::Error> {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    let free_axis_names = sharding
        .mesh()
        .axes()
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name())).then_some(axis.name()))
        .collect::<HashSet<_>>();
    let mut used_axes = HashSet::new();
    for partition_dimension in sharding.dimensions() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            used_axes.extend(axis_names.iter().map(String::as_str));
        }
    }
    used_axes.extend(sharding.unreduced_axes().iter().map(String::as_str));
    used_axes.extend(sharding.reduced_axes().iter().map(String::as_str));
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    sharding
        .dimensions()
        .iter()
        .map(|partition_dimension| match partition_dimension {
            ShardingDimension::Replicated => context.shardy_dimension_sharding([], !has_unused_free_axes, None),
            ShardingDimension::Sharded(axis_names) => {
                let axes = axis_names
                    .iter()
                    .map(|axis_name| context.shardy_axis_ref(axis_name.as_str(), None))
                    .collect::<Result<Vec<_>, _>>()?;
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                context.shardy_dimension_sharding(axes, !(contains_free_axis || has_unused_free_axes), None)
            }
            ShardingDimension::Unconstrained => context.shardy_dimension_sharding([], false, None),
        })
        .collect()
}

#[cfg(test)]
fn stripped_shardy_tensor_sharding(sharding: &Sharding, manual_axes: &[String]) -> String {
    let mut result =
        format!("<@{SHARDY_MESH_SYMBOL_NAME}, {}>", render_manual_computation_dimensions(sharding, manual_axes));

    let replicated_axes = sharding.replicated_axes();
    result.pop();
    if !replicated_axes.is_empty() {
        result.push_str(", replicated={");
        for (axis_index, axis_name) in replicated_axes.iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    if !sharding.unreduced_axes().is_empty() {
        result.push_str(", unreduced={");
        for (axis_index, axis_name) in sharding.unreduced_axes().iter().enumerate() {
            if axis_index > 0 {
                result.push_str(", ");
            }
            result.push('"');
            result.push_str(escape_shardy_string(axis_name).as_str());
            result.push('"');
        }
        result.push('}');
    }

    result.push('>');
    result
}

#[cfg(test)]
fn render_manual_computation_dimensions(sharding: &Sharding, manual_axes: &[String]) -> String {
    let manual_axis_names = manual_axes.iter().map(String::as_str).collect::<HashSet<_>>();
    let free_axis_names = sharding
        .mesh()
        .axes()
        .iter()
        .filter_map(|axis| (!manual_axis_names.contains(axis.name())).then_some(axis.name()))
        .collect::<HashSet<_>>();
    let mut used_axes = HashSet::new();
    for partition_dimension in sharding.dimensions() {
        if let ShardingDimension::Sharded(axis_names) = partition_dimension {
            used_axes.extend(axis_names.iter().map(String::as_str));
        }
    }
    used_axes.extend(sharding.unreduced_axes().iter().map(String::as_str));
    used_axes.extend(sharding.reduced_axes().iter().map(String::as_str));
    let has_unused_free_axes = free_axis_names.iter().any(|axis_name| !used_axes.contains(axis_name));

    let mut result = String::from("[");
    for (dimension_index, partition_dimension) in sharding.dimensions().iter().enumerate() {
        if dimension_index > 0 {
            result.push_str(", ");
        }

        match partition_dimension {
            ShardingDimension::Replicated => {
                if has_unused_free_axes {
                    result.push_str("{?}");
                } else {
                    result.push_str("{}");
                }
            }
            ShardingDimension::Sharded(axis_names) => {
                let contains_free_axis =
                    axis_names.iter().any(|axis_name| free_axis_names.contains(axis_name.as_str()));
                result.push('{');
                for (axis_index, axis_name) in axis_names.iter().enumerate() {
                    if axis_index > 0 {
                        result.push_str(", ");
                    }
                    result.push('"');
                    result.push_str(escape_shardy_string(axis_name).as_str());
                    result.push('"');
                }
                if contains_free_axis || has_unused_free_axes {
                    result.push_str(", ?");
                }
                result.push('}');
            }
            ShardingDimension::Unconstrained => result.push_str("{?}"),
        }
    }
    result.push(']');
    result
}

#[cfg(test)]
fn render_shardy_axes<A: AsRef<str>>(axis_names: &[A]) -> String {
    let mut result = String::from("{");
    for (axis_index, axis_name) in axis_names.iter().enumerate() {
        if axis_index > 0 {
            result.push_str(", ");
        }
        result.push('"');
        result.push_str(escape_shardy_string(axis_name.as_ref()).as_str());
        result.push('"');
    }
    result.push('}');
    result
}

#[cfg(test)]
fn escape_shardy_string(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use std::collections::{BTreeMap, BTreeSet, HashMap};

    use indoc::indoc;
    use pretty_assertions::assert_eq;
    use ryft_pjrt::protos::{CompilationOptions, ExecutableCompilationOptions, Precision};
    use ryft_pjrt::{BufferType, ClientOptions, CpuClientOptions, Program, load_cpu_plugin};

    use crate::tests::{values_from_bytes, values_to_bytes};
    use crate::{Array, FromPjrt, ToMlir};
    use ryft_core::operations::collectives::{AllGather, AllGatherOutputVariance, CollectiveOptions};
    use ryft_core::{
        DataType, Device, DeviceMesh, DimensionBounds, DimensionVariable, Dot, DotDimensionNumbers, MeshAxis,
        MeshAxisType, RegionRole, Sharding, ShardingDimension, Sin,
    };

    use super::*;

    fn test_logical_mesh_2x2() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_data_model() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap()
    }

    fn test_logical_mesh_data_model_explicit() -> LogicalMesh {
        LogicalMesh::new(vec![
            MeshAxis::new("data", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("model", 4, MeshAxisType::Explicit).unwrap(),
        ])
        .unwrap()
    }

    fn static_sharded_array_type(data_type: DataType, global_shape: &[usize], sharding: Sharding) -> ArrayType {
        ArrayType::new(data_type, Shape::new(global_shape.iter().copied().map(Dimension::Static).collect()))
            .with_sharding(sharding)
            .unwrap()
    }

    fn test_sharding(mesh: &LogicalMesh, dimensions: Vec<ShardingDimension>, unreduced_axes: Vec<String>) -> Sharding {
        Sharding::new(mesh.clone(), dimensions).unwrap().with_unreduced_axes(unreduced_axes).unwrap()
    }

    fn test_sharding_with_varying(
        mesh: &LogicalMesh,
        dimensions: Vec<ShardingDimension>,
        unreduced_axes: Vec<String>,
        reduced_axes: Vec<String>,
        varying_manual_axes: Vec<String>,
    ) -> Sharding {
        Sharding::new(mesh.clone(), dimensions)
            .unwrap()
            .with_unreduced_axes(unreduced_axes)
            .unwrap()
            .with_reduced_axes(reduced_axes)
            .unwrap()
            .with_varying_manual_axes(varying_manual_axes)
            .unwrap()
    }

    fn test_spmd_compilation_options(partition_count: usize) -> CompilationOptions {
        CompilationOptions {
            argument_layouts: Vec::new(),
            parameter_is_tupled_arguments: false,
            executable_build_options: Some(ExecutableCompilationOptions {
                device_ordinal: -1,
                replica_count: 1,
                partition_count: partition_count as i64,
                use_spmd_partitioning: true,
                use_shardy_partitioner: true,
                ..Default::default()
            }),
            compile_portable_executable: false,
            profile_version: 0,
            serialized_multi_slice_configuration: Vec::new(),
            environment_option_overrides: HashMap::new(),
            target_config: None,
            allow_in_place_mlir_modification: false,
            matrix_unit_operand_precision: Precision::Default as i32,
        }
    }

    #[test]
    fn test_shard_map_uses_manual_axes_from_mesh() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x".to_string(), "y".to_string()].as_slice());
        assert_eq!(shard_map.in_shardings()[0].replicated_axes(), vec!["y"]);
        assert_eq!(shard_map.out_shardings()[0].replicated_axes(), vec!["y"]);
    }

    #[test]
    fn test_shard_map_can_select_manual_axis_subset() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], vec![])],
            Vec::new(),
            vec!["x".into()],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["x".to_string()].as_slice());
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![4]);
        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"x", "y", ?}]>]"#);
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"x"}"#);
    }

    #[test]
    fn test_shard_map_function_rejects_mesh_without_manual_axes() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let mesh = crate::tests::logical_mesh_2x2();
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        );

        assert!(matches!(result, Err(ShardMapTraceError::MeshHasNoManualAxes)));
    }

    #[test]
    fn test_shard_map_rejects_zero_input_traced_invocation_without_domain() {
        let mesh = test_logical_mesh_2x2();
        let result: Result<Vec<ShardMapTracer>, ShardMapTraceError> =
            shard_map::<_, Vec<ShardMapTracer>, Vec<ArrayType>, ShardMapTracer>(
                |_: Vec<ShardMapTracer>| -> Vec<ShardMapTracer> {
                    unreachable!("zero-input traced invocation should fail early")
                },
                Vec::<ShardMapTracer>::new(),
                mesh.clone(),
                Vec::<Sharding>::new(),
                vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            );

        assert!(matches!(result, Err(ShardMapTraceError::MissingTracedInvocationDomain)));
    }

    #[test]
    fn test_shard_map_rejects_free_axis_before_manual_axis() {
        let mesh = test_logical_mesh_data_model();
        let result = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["model", "data"])], vec![])],
            Vec::new(),
            vec![],
            true,
        );

        assert_eq!(
            result,
            Err(ShardMapError::ManualAxisMustPrecedeFreeAxis {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                free_axis_name: "model".to_string(),
                manual_axis_name: "data".to_string(),
            })
        );
    }

    #[test]
    fn test_shard_map_local_input_shape_for_all_manual_axes() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x", "y"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![4]);
    }

    #[test]
    fn test_shard_map_local_input_shape_for_mixed_manual_and_free_axes() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_input_shape(0, &[16]).unwrap(), vec![8]);
    }

    #[test]
    fn test_shard_map_local_output_shape() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(
                &mesh,
                vec![ShardingDimension::sharded(["data"]), ShardingDimension::replicated()],
                vec![],
            )],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.local_output_shape(0, &[32, 8]).unwrap(), vec![16, 8]);
    }

    #[test]
    fn test_shard_map_local_shape_rejects_padding_from_manual_axes() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 3, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[10]),
            Err(ShardMapError::ManualAxisIntroducesPadding {
                value_kind: "input",
                value_index: 0,
                dimension: 0,
                dimension_size: 10,
                manual_partition_count: 3,
            })
        );
    }

    #[test]
    fn test_shard_map_local_shape_rejects_rank_mismatch() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.local_input_shape(0, &[8, 4]),
            Err(ShardMapError::RankMismatch { value_kind: "input", value_index: 0, partition_rank: 1, shape_rank: 2 })
        );
    }

    #[test]
    fn test_shard_map_renders_in_shardings_attribute() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"x"}], replicated={"y"}>]"#);
    }

    #[test]
    fn test_shard_map_renders_free_axes_as_open_dimension_shardings() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_explicit_axes_in_traced_shardings() {
        let mesh = test_logical_mesh_data_model_explicit();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![])],
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_in_shardings_attribute(), r#"[<@mesh, [{"data", "model", ?}]>]"#);
    }

    #[test]
    fn test_shard_map_renders_out_shardings_attribute() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.to_shardy_out_shardings_attribute(), r#"[<@mesh, [{?}], replicated={"x"}>]"#);
    }

    #[test]
    fn test_shard_map_renders_manual_axes_attribute() {
        let shard_map = ShardMap::new(
            LogicalMesh::new(vec![
                MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
                MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            ])
            .unwrap(),
            Vec::new(),
            Vec::new(),
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(shard_map.manual_axes(), vec!["y".to_string()].as_slice());
        assert_eq!(shard_map.to_shardy_manual_axes_attribute(), r#"{"y"}"#);
    }

    #[test]
    fn test_shard_map_renders_manual_computation_attributes() {
        let mesh = test_logical_mesh_data_model();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], vec![])],
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["data"])], vec![])],
            vec![],
            true,
        )
        .unwrap();

        assert_eq!(
            shard_map.to_shardy_manual_computation_attributes(),
            r#"in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"}"#
        );
    }

    #[test]
    fn test_derive_local_input_types_adds_varying_axes_from_in_specs() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            Vec::new(),
            vec!["x".into()],
            true,
        )
        .unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["y".into()],
            ))
            .unwrap();
        let global_input_types = derive_global_input_types(&shard_map, &vec![global_input_type]).unwrap();
        let local_input_types = derive_local_input_types(&shard_map, &global_input_types).unwrap();

        assert_eq!(local_input_types[0].shape(), &Shape::new(vec![Dimension::Static(4)]));
        assert_eq!(
            local_input_types[0]
                .sharding()
                .expect("local shard_map input should keep sharding metadata")
                .varying_manual_axes(),
            &BTreeSet::from(["x".to_string(), "y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_input_types_rejects_mismatched_reduced_axes() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_reduced_axes(["x"])
            .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), vec![input_sharding], Vec::new(), vec!["x".into(), "y".into()], true).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_reduced_axes(["y"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            derive_global_input_types(&shard_map, &vec![global_input_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "input",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: vec!["y".to_string()],
            })
        );
    }

    #[test]
    fn test_derive_local_input_types_preserve_unreduced_and_reduced_axes_from_in_specs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("z", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap()
            .with_reduced_axes(["z"])
            .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            vec![input_sharding.clone()],
            Vec::new(),
            vec!["x".into(), "y".into(), "z".into()],
            true,
        )
        .unwrap();
        let global_input_types = derive_global_input_types(
            &shard_map,
            &vec![
                ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
                    .with_sharding(input_sharding)
                    .unwrap(),
            ],
        )
        .unwrap();
        let local_input_types = derive_local_input_types(&shard_map, &global_input_types).unwrap();

        assert_eq!(
            local_input_types[0]
                .sharding()
                .expect("local shard_map input should keep sharding metadata")
                .unreduced_axes(),
            &BTreeSet::from(["y".to_string()])
        );
        assert_eq!(
            local_input_types[0]
                .sharding()
                .expect("local shard_map input should keep sharding metadata")
                .reduced_axes(),
            &BTreeSet::from(["z".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_drops_active_manual_varying_axes_and_preserves_outer_ones() {
        let mesh = test_logical_mesh_2x2();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![])],
            vec!["x".into()],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into(), "y".into()],
            ))
            .unwrap();
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(global_output_types[0].shape(), &Shape::new(vec![Dimension::Static(8)]));
        assert_eq!(
            global_output_types[0]
                .sharding()
                .expect("global shard_map output should keep sharding metadata")
                .varying_manual_axes(),
            &BTreeSet::from(["y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_implicitly_adopts_unreduced_axes_from_out_specs() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
            .unwrap()
            .with_unreduced_axes(["y"])
            .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![output_sharding.clone()], vec!["x".into(), "y".into()], true)
                .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(
            global_output_types[0]
                .sharding()
                .expect("global shard_map output should keep sharding metadata")
                .unreduced_axes(),
            &BTreeSet::from(["y".to_string()])
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_extra_local_unreduced_axes_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()]).unwrap()],
            vec!["x".into(), "y".into()],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(
                Sharding::new(mesh, vec![ShardingDimension::replicated()])
                    .unwrap()
                    .with_unreduced_axes(["y"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "unreduced axes",
                expected: Vec::new(),
                actual: vec!["y".to_string()],
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_reduced_axis_mismatch_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::replicated()])
            .unwrap()
            .with_reduced_axes(["x"])
            .unwrap();
        let shard_map =
            ShardMap::new(mesh.clone(), Vec::new(), vec![output_sharding], vec!["x".into(), "y".into()], true).unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::ShardingStateMismatch {
                value_kind: "output",
                value_index: 0,
                state_kind: "reduced axes",
                expected: vec!["x".to_string()],
                actual: Vec::new(),
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_rejects_omitted_varying_manual_axis_when_check_vma_is_enabled() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            true,
        )
        .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into()],
            ))
            .unwrap();

        assert_eq!(
            derive_global_output_types(&shard_map, &vec![local_output_type]),
            Err(ShardMapTraceError::OutputVaryingManualAxisNotInOutSpecs {
                output_index: 0,
                axis_name: "x".to_string(),
            })
        );
    }

    #[test]
    fn test_derive_global_output_types_ignores_omitted_varying_manual_axis_when_check_vma_is_disabled() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap();
        let shard_map = ShardMap::new(
            mesh.clone(),
            Vec::new(),
            vec![test_sharding(&mesh, vec![ShardingDimension::replicated()], vec![])],
            vec![],
            false,
        )
        .unwrap();
        let local_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]))
            .with_sharding(test_sharding_with_varying(
                &mesh,
                vec![ShardingDimension::replicated()],
                vec![],
                vec![],
                vec!["x".into()],
            ))
            .unwrap();
        let global_output_types = derive_global_output_types(&shard_map, &vec![local_output_type]).unwrap();

        assert_eq!(
            global_output_types[0]
                .sharding()
                .expect("global shard_map output should keep sharding metadata")
                .varying_manual_axes(),
            &BTreeSet::<String>::new()
        );
    }

    #[test]
    fn test_array_type_display_renders_type() {
        let array_type = ArrayType::scalar(DataType::F32);

        assert_eq!(array_type.to_string(), "f32[]");
    }

    #[test]
    fn test_shard_map_trace_derives_types_and_renders_mlir() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding.clone(),
        )
        .unwrap();
        let expected_global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(input_sharding.clone())
            .unwrap();
        let expected_local_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])])
                    .unwrap()
                    .with_varying_manual_axes(["x"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(traced.global_input_types(), &expected_global_input_type);
        assert_eq!(traced.local_input_types(), &expected_local_input_type);
        assert_eq!(traced.local_output_types(), &expected_local_input_type);
        assert_eq!(traced.global_output_types(), &expected_global_input_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_hides_auto_axes_in_type_level_shardings() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)]));
        let mesh = test_logical_mesh_data_model();
        let input_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["data", "model"])], vec![]);
        let projected_sharding = input_sharding.without_auto_axes();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type.clone(),
            mesh.clone(),
            input_sharding.clone(),
            input_sharding,
        )
        .unwrap();
        let expected_global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)]))
            .with_sharding(projected_sharding.clone())
            .unwrap();
        let expected_local_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(
                Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["data"])])
                    .unwrap()
                    .with_varying_manual_axes(["data"])
                    .unwrap(),
            )
            .unwrap();

        assert_eq!(traced.global_input_types(), &expected_global_input_type);
        assert_eq!(traced.local_input_types(), &expected_local_input_type);
        assert_eq!(traced.global_output_types(), &expected_global_input_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["data"=2, "model"=4]>
                  func.func @main(%arg0: tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) -> (tensor<16xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"data"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"data", ?}]>] out_shardings=[<@mesh, [{"data", ?}]>] manual_axes={"data"} (%arg1: tensor<8xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<8xf32>
                      sdy.return %1 : tensor<8xf32>
                    } : (tensor<16xf32>) -> tensor<16xf32>
                    return %0 : tensor<16xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_can_render_nested_shard_maps() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let inner_mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Auto).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let outer_sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let inner_sharding = test_sharding(&inner_mesh, vec![ShardingDimension::sharded(["y"])], vec![]);
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            {
                let inner_mesh = inner_mesh.clone();
                let inner_sharding = inner_sharding.clone();
                move |x: ShardMapTracer| {
                    let nested: ShardMapTracer = shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                        |y: ShardMapTracer| y.clone() + y,
                        x.clone(),
                        inner_mesh.clone(),
                        inner_sharding.clone(),
                        inner_sharding.clone(),
                    )
                    .expect("nested shard_map should trace");
                    nested + x
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
            mesh,
            outer_sharding.clone(),
            outer_sharding,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", ?}]>] out_shardings=[<@mesh, [{"x", ?}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %1 = sdy.manual_computation(%arg1) in_shardings=[<@mesh, [{"y", ?}]>] out_shardings=[<@mesh, [{"y", ?}]>] manual_axes={"y"} (%arg2: tensor<2xf32>) {
                        %3 = stablehlo.add %arg2, %arg2 : tensor<2xf32>
                        sdy.return %3 : tensor<2xf32>
                      } : (tensor<4xf32>) -> tensor<4xf32>
                      %2 = stablehlo.add %1, %arg1 : tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_trace_rejects_dynamic_input_types() {
        let dynamic_input_type = ArrayType::new(
            DataType::F32,
            Shape::new(vec![DimensionVariable::new("dynamic", DimensionBounds::unbounded()).into()]),
        );
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let result: Result<TracedShardMap<ArrayType, ArrayType>, ShardMapTraceError> = shard_map(
            |x| x.clone() + x,
            dynamic_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        );

        assert!(matches!(
            result,
            Err(ShardMapTraceError::DynamicShapeNotSupported { value_kind: "input", value_index: 0, dimension: 0 })
        ));
    }

    #[test]
    fn test_shard_map_infers_single_input_closure_argument_type() {
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            mesh.clone(),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
            test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]),
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) -> (tensor<8xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}]>}) {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_traced_shard_map_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));
        let traced: TracedShardMap<ArrayType, ArrayType> = shard_map(
            |x| x.clone() + x,
            global_input_type,
            device_mesh.logical_mesh().clone(),
            sharding.clone(),
            sharding.clone(),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, *expected_values_by_device.get(&device_id).unwrap());
        }
    }

    #[test]
    fn test_traced_shard_map_matmul_renders_and_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(8) }))
            .expect("failed to create 8-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 8);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 8, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let lhs_sharding = Sharding::new(
            device_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let rhs_sharding = Sharding::replicated(device_mesh.logical_mesh().clone(), 2);
        let output_sharding = Sharding::new(
            device_mesh.logical_mesh().clone(),
            vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()],
        )
        .unwrap();
        let global_input_types = (
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8), Dimension::Static(4)])),
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4), Dimension::Static(2)])),
        );
        let traced: TracedShardMap<(ArrayType, ArrayType), ArrayType> = shard_map(
            |(lhs, rhs)| lhs.dot(&rhs, &DotDimensionNumbers::matmul()),
            global_input_types,
            device_mesh.logical_mesh().clone(),
            (lhs_sharding.clone(), rhs_sharding.clone()),
            output_sharding.clone(),
        )
        .unwrap();
        let mlir_program = traced.to_mlir_module("main").unwrap();

        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=8]>
                  func.func @main(%arg0: tensor<8x4xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}, %arg1: tensor<4x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{}, {}], replicated={"x"}>}) -> (tensor<8x2xf32> {sdy.sharding = #sdy.sharding<@mesh, [{"x"}, {}]>}) {
                    %0 = sdy.manual_computation(%arg0, %arg1) in_shardings=[<@mesh, [{"x"}, {}]>, <@mesh, [{}, {}], replicated={"x"}>] out_shardings=[<@mesh, [{"x"}, {}]>] manual_axes={"x"} (%arg2: tensor<1x4xf32>, %arg3: tensor<4x2xf32>) {
                      %1 = stablehlo.dot_general %arg2, %arg3, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x4xf32>, tensor<4x2xf32>) -> tensor<1x2xf32>
                      sdy.return %1 : tensor<1x2xf32>
                    } : (tensor<8x4xf32>, tensor<4x2xf32>) -> tensor<8x2xf32>
                    return %0 : tensor<8x2xf32>
                  }
                }
            "#}
        );

        let lhs_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(row_index, device)| {
                let row = row_index as f32;
                client
                    .buffer(
                        values_to_bytes::<f32>(&[row, row + 1.0, row + 2.0, row + 3.0]).as_slice(),
                        BufferType::F32,
                        [1u64, 4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let rhs_values = [1.0f32, 2.0, 0.0, 1.0, 1.0, 0.0, 2.0, 1.0];
        let rhs_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<f32>(rhs_values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64, 2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let lhs_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8, 4], lhs_sharding.clone()),
            device_mesh.clone(),
            lhs_buffers,
        )
        .unwrap();
        let rhs_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[4, 2], rhs_sharding),
            device_mesh,
            rhs_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(8)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 8);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let row_start_by_device = execution_device_ids
            .iter()
            .map(|device_id| {
                let row_start = lhs_array.device_shard(*device_id).unwrap().slice()[0].start;
                (*device_id, row_start)
            })
            .collect::<HashMap<_, _>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![lhs_array, rhs_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            let row = *row_start_by_device.get(&device_id).unwrap() as f32;
            assert_eq!(values[0], 4.0 * row + 8.0);
            assert_eq!(values[1], 4.0 * row + 4.0);
        }
    }

    #[test]
    fn test_trace_reshard_renders_mlir() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = test_sharding(&mesh, vec![ShardingDimension::sharded(["x"])], vec![]);
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    reshard(x.sin().unwrap(), sharding.clone()).expect("reshard should stage on traced XLA values")
                }
            },
            global_input_type.clone(),
        )
        .unwrap();

        let expected_output_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]))
            .with_sharding(sharding.clone())
            .unwrap();
        assert_eq!(traced.global_input_types(), &global_input_type);
        assert_eq!(traced.global_output_types(), &expected_output_type);
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = stablehlo.sine %arg0 : tensor<8xf32>
                    %1 = sdy.sharding_constraint %0 <@mesh, [{"x"}]> : tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_manual_computation_executes_end_to_end_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();

        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let shard_map = ShardMap::new(
            device_mesh.logical_mesh().clone(),
            vec![sharding.clone()],
            vec![sharding.clone()],
            vec![],
            true,
        )
        .unwrap();
        assert_eq!(shard_map.local_input_shape(0, &[8]).unwrap(), vec![2]);
        assert_eq!(shard_map.local_output_shape(0, &[8]).unwrap(), vec![2]);

        let context = MlirContext::new();
        let input_sharding = shard_map.in_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let output_sharding = shard_map.out_shardings()[0].to_mlir(context.unknown_location()).unwrap().to_string();
        let manual_computation_attributes = shard_map.to_shardy_manual_computation_attributes();
        let mesh_module = context.module(context.unknown_location()).unwrap();
        let mesh_operation = mesh_module
            .body()
            .unwrap()
            .append_operation(shard_map.mesh().to_mlir(context.unknown_location()).unwrap())
            .unwrap()
            .to_string();

        let mlir_program = format!(
            r#"
                module {{
                    {mesh_operation}
                    func.func @main(
                        %arg0: tensor<8xf32> {{sdy.sharding = {input_sharding}}}
                    ) -> (tensor<8xf32> {{sdy.sharding = {output_sharding}}}) {{
                        %0 = sdy.manual_computation(%arg0) {manual_computation_attributes} (%arg1: tensor<2xf32>) {{
                            %1 = stablehlo.add %arg1, %arg1 : tensor<2xf32>
                            sdy.return %1 : tensor<2xf32>
                        }} : (tensor<8xf32>) -> tensor<8xf32>
                        return %0 : tensor<8xf32>
                    }}
                }}
            "#
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();

        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let expected_values_by_device = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                (device.id().unwrap(), [device_index as f32 * 4.0 + 2.0, device_index as f32 * 4.0 + 4.0])
            })
            .collect::<HashMap<_, _>>();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();

        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        assert_eq!(outputs.len(), execution_device_ids.len());
        for (output, device_id) in outputs.into_iter().zip(execution_device_ids.iter().copied()) {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, *expected_values_by_device.get(&device_id).unwrap());
        }
    }

    #[test]
    fn test_shard_map_psum_lowers_to_all_reduce_and_executes_on_cpu() {
        use ryft_core::{Collective, CollectiveKind};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // `psum` over the manual mesh axis `"x"` resolves against the seeded body trace and lowers to a
        // `stablehlo.all_reduce` whose replica group spans the four devices along `"x"`, so every shard receives the
        // elementwise sum of all four local shards.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| local_x.collective("x", CollectiveKind::PSum).unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with psum should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 4);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2], [3, 4], [5, 6], [7, 8], so every device receives [1+3+5+7, 2+4+6+8] = [16, 20].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [16.0, 20.0]);
        }
    }

    #[test]
    fn test_shard_map_custom_call_lowers_inside_manual_region_and_executes_on_cpu() {
        use ryft_core::operations::custom_call::{CustomCall, CustomCallOperation};

        use crate::tests::{ADD_ONE_CUSTOM_CALL_TARGET, ensure_add_one_handler_registered};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        ensure_add_one_handler_registered(&client).unwrap();
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // The custom call executes per shard inside the manual region: its declared output type is the local
        // shard type, and the lowered `stablehlo.custom_call` appears inside the `sdy.manual_computation` body.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            let operation = CustomCallOperation::new(
                                ADD_ONE_CUSTOM_CALL_TARGET,
                                vec![local_x.r#type().into_owned()],
                            );
                            CustomCall::custom_call(&operation, std::slice::from_ref(&local_x)).unwrap().remove(0)
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with a custom call should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.custom_call @ryft.test.add_one(%arg1) {api_version = 4 : i32, backend_config = {}} : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4], so the kernel produces [2, 3] and [4, 5] respectively.
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [device_index as f32 * 2.0 + 2.0, device_index as f32 * 2.0 + 3.0]);
        }
    }

    #[test]
    fn test_shard_map_pmean_lowers_to_all_reduce_with_axis_size_division() {
        use ryft_core::{Collective, CollectiveKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| local_x.collective("x", CollectiveKind::PMean).unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with pmean should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %4 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %4 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      %cst = stablehlo.constant dense<4.000000e+00> : tensor<f32>
                      %2 = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<2xf32>
                      %3 = stablehlo.divide %1, %2 : tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_grouped_pmean_preserves_group_order_and_uses_group_divisor() {
        use ryft_core::{Collective, CollectiveKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x
                                .collective_with_axis_index_groups(
                                    "x",
                                    CollectiveKind::PMean,
                                    vec![vec![0, 2], vec![3, 1]],
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>"), "{module}",);
        assert!(module.contains("stablehlo.constant dense<2.000000e+00> : tensor<f32>"), "{module}");
    }

    #[test]
    fn test_batch_inside_shard_map_forwards_mesh_collective_to_all_reduce() {
        use ryft_core::{Batch, BatchAxis, BatchAxisSpecification, Collective, CollectiveKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // A named `batch` level inside the shard_map body binds `"b"`, while `psum` names the *mesh* axis `"x"`: the
        // batching rule forwards the collective through the batch level to the seeded base trace (which binds `"x"`),
        // so it lands in the body program on the batched physical value and lowers to the same `all_reduce` as a
        // direct `psum` — resolution composes across binder kinds.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            let context = local_x.dispatch_domain();
                            let summed: ShardMapTracer = Batch::batch(
                                &context,
                                |item| item.collective("x", CollectiveKind::PSum),
                                local_x,
                                BatchAxis::new(0),
                                BatchAxis::new(0),
                                BatchAxisSpecification::named("b"),
                            )
                            .expect("vmap inside shard_map should trace");
                            summed
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with vmapped psum should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        // The module is identical to the direct-psum module: the batch level forwarded the mesh collective untouched.
        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %2 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %2 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_collective_inside_condition_inside_shard_map_lowers_to_all_reduce() {
        use ryft_core::{
            CollectiveKind, CollectiveOperation, Compare, ComparisonDirection, ConditionOperation, Reduce,
            ReductionKind,
        };

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // The `psum` sits inside a `condition` branch inside the shard_map body, so it lowers through the nested
        // control-flow path: the threaded collective lowering state resolves the manual mesh axis inside the
        // `stablehlo.if` region and emits the same `all_reduce` as a body-level `psum` would.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let psum_branch = {
                                let mut builder = XlaProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type.clone()));
                                let output = builder
                                    .add_instruction(
                                        CollectiveOperation::new("x".to_string(), CollectiveKind::PSum),
                                        Vec::new(),
                                        vec![input],
                                    )
                                    .unwrap()[0];
                                builder.build(vec![output], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let identity_branch = {
                                let mut builder = XlaProgramBuilder::new();
                                let input = builder.add_input(ArrayIrType::Array(local_type));
                                builder.build(vec![input], vec![Placeholder], vec![Placeholder]).unwrap()
                            };
                            let predicate = local_x
                                .compare(&local_x, ComparisonDirection::Equal)
                                .unwrap()
                                .reduce(&[0], ReductionKind::Any);
                            let context = local_x.value().context().clone();
                            let mut outputs = context
                                .stage_operation(
                                    XlaOperation::Condition(ConditionOperation::new()),
                                    vec![psum_branch, identity_branch],
                                    &[predicate.into_value(), local_x.into_value()],
                                )
                                .unwrap();
                            ValueProjection::<ArrayType>::into_projected(outputs.remove(0))
                                .expect("condition output should remain an array")
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with psum inside a condition should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = stablehlo.compare EQ, %arg1, %arg1, FLOAT : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xi1>
                      %c = stablehlo.constant dense<false> : tensor<i1>
                      %2 = stablehlo.reduce(%1 init: %c) applies stablehlo.or across dimensions = [0] : (tensor<2xi1>, tensor<i1>) -> tensor<i1>
                      %3 = "stablehlo.if"(%2) ({
                        %4 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                        ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                          %5 = stablehlo.add %arg2, %arg3 : tensor<f32>
                          stablehlo.return %5 : tensor<f32>
                        }) : (tensor<2xf32>) -> tensor<2xf32>
                        stablehlo.return %4 : tensor<2xf32>
                      }, {
                        stablehlo.return %arg1 : tensor<2xf32>
                      }) : (tensor<i1>) -> tensor<2xf32>
                      sdy.return %3 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_two_shard_maps_with_collectives_receive_unique_channel_ids() {
        use ryft_core::{Collective, CollectiveKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // Two manual regions in one module each emit a channeled `all_reduce`: the module-scoped channel allocator
        // hands out distinct handles (1 and 2), which XLA requires across a module.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    let first = shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| local_x.collective("x", CollectiveKind::PSum).unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("first shard_map should trace");
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| local_x.collective("x", CollectiveKind::PSum).unwrap(),
                        first,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("second shard_map should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    %1 = sdy.manual_computation(%0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %2 = "stablehlo.all_reduce"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 2, type = 1>, replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64>, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %1 : tensor<8xf32>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_axis_index_lowers_to_partition_id_coordinate_and_executes_on_cpu() {
        use ryft_core::{AxisIndex, Broadcast};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 4);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(4)]));

        // `axis_index("x")` gives each device its own coordinate along the manual mesh axis `"x"`, added to the local
        // shard. The single-axis mesh has unit stride and full-size axis, so the coordinate is just `partition_id`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            // `axis_index` is a scalar per device; broadcast it to the shard shape so the add has
                            // shape-congruent operands (StableHLO has no implicit broadcasting).
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.dispatch_domain().axis_index("x").unwrap();
                            let index = index.broadcast(local_type, &[]).unwrap();
                            local_x + index
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with axis_index should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=4]>
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<4xui64> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<1xui64>) {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %3 = stablehlo.broadcast_in_dim %2, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      %4 = stablehlo.add %arg1, %3 : tensor<1xui64>
                      sdy.return %4 : tensor<1xui64>
                    } : (tensor<4xui64>) -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .map(|device| {
                client
                    .buffer(
                        values_to_bytes::<u64>(&[10u64]).as_slice(),
                        BufferType::U64,
                        [1u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::U64, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(4)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Each device d holds shard [10] and adds its coordinate d, so device d outputs [10 + d].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [u64; 1] = values_from_bytes::<u64>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [10 + device_index as u64]);
        }
    }

    #[test]
    fn test_shard_map_axis_index_of_major_mesh_axis_lowers_to_divide_and_remainder() {
        use ryft_core::{AxisIndex, Broadcast};

        // A 2x2 mesh: `axis_index("x")` addresses the major axis (row-major stride 2, size 2), so the device
        // coordinate is `(partition_id / 2) % 2` — exercising both the divide and the remainder that a single-axis
        // mesh skips.
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Manual).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x", "y"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::U64, Shape::new(vec![Dimension::Static(4)]));

        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            let local_type = local_x.r#type().into_owned();
                            let index = local_x.dispatch_domain().axis_index("x").unwrap();
                            index.broadcast(local_type, &[]).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with axis_index of the major axis should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        assert_eq!(
            traced.to_mlir_module("main").unwrap(),
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2, "y"=2]>
                  func.func @main(%arg0: tensor<4xui64>) -> tensor<4xui64> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x", "y"}]>] out_shardings=[<@mesh, [{"x", "y"}]>] manual_axes={"x", "y"} (%arg1: tensor<1xui64>) {
                      %1 = stablehlo.partition_id : tensor<ui32>
                      %2 = stablehlo.convert %1 : (tensor<ui32>) -> tensor<ui64>
                      %c = stablehlo.constant dense<2> : tensor<ui64>
                      %3 = stablehlo.divide %2, %c : tensor<ui64>
                      %c_0 = stablehlo.constant dense<2> : tensor<ui64>
                      %4 = stablehlo.remainder %3, %c_0 : tensor<ui64>
                      %5 = stablehlo.broadcast_in_dim %4, dims = [] : (tensor<ui64>) -> tensor<1xui64>
                      sdy.return %5 : tensor<1xui64>
                    } : (tensor<4xui64>) -> tensor<4xui64>
                    return %0 : tensor<4xui64>
                  }
                }
            "#}
        );
    }

    #[test]
    fn test_shard_map_collective_over_unbound_axis_is_rejected_at_trace_time() {
        use ryft_core::{AxisError, BatchingError, Collective, CollectiveKind};

        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // The shard_map body binds only the manual mesh axis `"x"`, so a collective naming `"y"` fails fast at
        // staging time with an unbound-axis error instead of tracing as a silent identity.
        let result: Result<TracedXlaProgram<ArrayType, ArrayType>, ShardMapTraceError> = trace(
            {
                let mesh = mesh.clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            let error = local_x.collective("y", CollectiveKind::PSum).unwrap_err();
                            assert!(matches!(
                                error.downcast_custom::<BatchingError>(),
                                Some(BatchingError::Axis(AxisError::UnboundAxisName { name })) if name == "y",
                            ));
                            local_x
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map should trace after handling the collective error")
                }
            },
            global_input_type,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_shard_map_all_gather_lowers_and_executes_on_cpu() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // `all_gather` over the manual mesh axis `"x"` extends the local shard from `f32[2]` to the full `f32[4]`
        // concatenation on every device. Its output still varies along the manual axis for VMA purposes (a replicated
        // out sharding is rejected with `OutputVaryingManualAxisNotInOutSpecs`, mirroring JAX's vma tracking), so the
        // output stays sharded over `"x"`, giving the global `f32[8]` concatenation of the per-device gathers. The
        // staged collective lowers to a channeled `stablehlo.all_gather` over the two devices along `"x"`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x
                                .all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::tiled(),
                                    AllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with all_gather should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %c = stablehlo.constant dense<2> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.multiply %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.all_gather"(%arg1) <{all_gather_dim = 0 : i64, channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, use_global_device_ids}> : (tensor<2xf32>) -> tensor<4xf32>
                      sdy.return %2 : tensor<4xf32>
                    } : (tensor<4xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4], so every device receives the full concatenation [1, 2, 3, 4].
        assert_eq!(outputs.len(), execution_device_ids.len());
        for output in outputs {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_shard_map_untiled_all_gather_lowers_rank_insertion() {
        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x
                                .all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::default(),
                                    AllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("stablehlo.broadcast_in_dim"), "{module}");
        assert!(
            module.contains("(tensor<2xf32>) -> tensor<1x2xf32>")
                && module.contains("(tensor<1x2xf32>) -> tensor<2x2xf32>"),
            "{module}",
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[4], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        for output in outputs {
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), vec![1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_shard_map_grouped_all_gather_preserves_group_order_across_mesh_coordinates() {
        let mesh = LogicalMesh::new(vec![
            MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap(),
            MeshAxis::new("y", 2, MeshAxisType::Auto).unwrap(),
        ])
        .unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x
                                .all_gather_with_options(
                                    "x",
                                    0,
                                    CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]),
                                    AllGatherOutputVariance::Varying,
                                )
                                .unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(
            module.contains("replica_groups = dense<[[0, 4], [6, 2], [1, 5], [7, 3]]> : tensor<4x2xi64>"),
            "{module}",
        );
        assert!(module.contains("(tensor<2xf32>) -> tensor<4xf32>"), "{module}");
    }

    #[test]
    fn test_shard_map_grouped_shape_changing_collectives_execute_on_cpu() {
        use ryft_core::operations::collectives::{AllToAll, PSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(4) }))
            .expect("failed to create 4-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, (ArrayType, ArrayType, ArrayType)> = trace(
            {
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, (ArrayType, ArrayType, ArrayType), _>(
                        |local_x: ShardMapTracer| {
                            let options =
                                CollectiveOptions::tiled().with_axis_index_groups(vec![vec![0, 2], vec![3, 1]]);
                            (
                                local_x
                                    .all_gather_with_options("x", 0, options.clone(), AllGatherOutputVariance::Varying)
                                    .unwrap(),
                                local_x.clone().psum_scatter_with_options("x", 0, options.clone()).unwrap(),
                                local_x.all_to_all_with_options("x", 0, 0, options).unwrap(),
                            )
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        (sharding.clone(), sharding.clone(), sharding.clone()),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(16)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert_eq!(module.matches("replica_groups = dense<[[0, 2], [3, 1]]> : tensor<2x2xi64>").count(), 3, "{module}",);

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let values = (0..4).map(|offset| (device_index * 4 + offset) as f32).collect::<Vec<_>>();
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[16], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(4))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        let expected_gather = [
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
            vec![0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0],
            vec![12.0, 13.0, 14.0, 15.0, 4.0, 5.0, 6.0, 7.0],
        ];
        let expected_scatter = [vec![8.0, 10.0], vec![20.0, 22.0], vec![12.0, 14.0], vec![16.0, 18.0]];
        let expected_all_to_all = [
            vec![0.0, 1.0, 8.0, 9.0],
            vec![14.0, 15.0, 6.0, 7.0],
            vec![2.0, 3.0, 10.0, 11.0],
            vec![12.0, 13.0, 4.0, 5.0],
        ];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 3);
            let actual = output
                .outputs
                .into_iter()
                .map(|buffer| {
                    let bytes = buffer.copy_to_host(None).unwrap().r#await().unwrap();
                    values_from_bytes::<f32>(bytes.as_slice())
                })
                .collect::<Vec<_>>();
            assert_eq!(
                actual,
                vec![
                    expected_gather[device_index].clone(),
                    expected_scatter[device_index].clone(),
                    expected_all_to_all[device_index].clone(),
                ],
            );
        }
    }

    #[test]
    fn test_shard_map_psum_scatter_lowers_and_executes_on_cpu() {
        use ryft_core::operations::collectives::{CollectiveOptions, PSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // `psum_scatter` over the manual mesh axis `"x"` sums the two local `f32[4]` shards elementwise and scatters
        // the sum, leaving each device with its own `f32[2]` chunk, so the sharded global output is `f32[4]`. The
        // staged collective lowers to a channeled `stablehlo.reduce_scatter` with a sum reduction.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x.psum_scatter_with_options("x", 0, CollectiveOptions::tiled()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with psum_scatter should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %c = stablehlo.constant dense<4> : tensor<i64>
                      %c_0 = stablehlo.constant dense<2> : tensor<i64>
                      %1 = stablehlo.divide %c, %c_0 : tensor<i64>
                      %2 = "stablehlo.reduce_scatter"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, scatter_dimension = 0 : i64, use_global_device_ids}> ({
                      ^bb0(%arg2: tensor<f32>, %arg3: tensor<f32>):
                        %3 = stablehlo.add %arg2, %arg3 : tensor<f32>
                        stablehlo.return %3 : tensor<f32>
                      }) : (tensor<4xf32>) -> tensor<2xf32>
                      sdy.return %2 : tensor<2xf32>
                    } : (tensor<8xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0f32 } else { 10.0f32 };
                let shard_values = [scale, scale * 2.0, scale * 3.0, scale * 4.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2, 3, 4] and [10, 20, 30, 40], so the elementwise sum is [11, 22, 33, 44]: device 0
        // receives the first chunk [11, 22] and device 1 the second chunk [33, 44].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[11.0f32, 22.0], [33.0, 44.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_shard_map_untiled_psum_scatter_lowers_rank_removal() {
        use ryft_core::operations::collectives::{CollectiveOptions, PSumScatter};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let output_sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x.psum_scatter_with_options("x", 0, CollectiveOptions::default()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(6)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("stablehlo.reduce_scatter"), "{module}");
        assert!(module.contains("stablehlo.reshape"), "{module}");
        assert!(module.contains("(tensor<2x3xf32>) -> tensor<1x3xf32>"), "{module}");

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let scale = if device_index == 0 { 1.0_f32 } else { 10.0_f32 };
                let values = [scale, scale * 2.0, scale * 3.0, scale * 4.0, scale * 5.0, scale * 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64, 3u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[2, 6], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let expected = [vec![11.0_f32, 22.0, 33.0], vec![44.0_f32, 55.0, 66.0]];
        for (output, expected) in outputs.into_iter().zip(expected) {
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
        }
    }

    #[test]
    fn test_shard_map_ppermute_lowers_and_executes_on_cpu() {
        use ryft_core::operations::collectives::Ppermute;

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(4)]));

        // `ppermute` over the manual mesh axis `"x"` with the rotation pairs [(0, 1), (1, 0)] swaps the two local
        // shards without changing their shapes. The staged collective lowers to a channeled
        // `stablehlo.collective_permute` with the axis-local pairs expanded to global device pairs.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| local_x.ppermute("x", vec![(0, 1), (1, 0)]).unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with ppermute should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<4xf32>) -> tensor<4xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<2xf32>) {
                      %1 = "stablehlo.collective_permute"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, source_target_pairs = dense<[[0, 1], [1, 0]]> : tensor<2x2xi64>}> : (tensor<2xf32>) -> tensor<2xf32>
                      sdy.return %1 : tensor<2xf32>
                    } : (tensor<4xf32>) -> tensor<4xf32>
                    return %0 : tensor<4xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let shard_values = [device_index as f32 * 2.0 + 1.0, device_index as f32 * 2.0 + 2.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [2u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[4], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2] and [3, 4]; the rotation swaps them, so device 0 receives [3, 4] and device 1 [1, 2].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[3.0f32, 4.0], [1.0, 2.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 2] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_shard_map_all_to_all_lowers_and_executes_on_cpu() {
        use ryft_core::operations::collectives::{AllToAll, CollectiveOptions};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        assert_eq!(client_devices.len(), 2);

        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let sharding =
            Sharding::new(device_mesh.logical_mesh().clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let global_input_type = ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)]));

        // `all_to_all` over the manual mesh axis `"x"` with split and concat both at axis 0 keeps the local `f32[4]`
        // shape: each device splits its shard into two chunks, keeps its own chunk, and receives the peer's matching
        // chunk. The staged collective lowers to a channeled `stablehlo.all_to_all`.
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = device_mesh.logical_mesh().clone();
                let sharding = sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x.all_to_all_with_options("x", 0, 0, CollectiveOptions::tiled()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .expect("shard_map with all_to_all should trace")
                }
            },
            global_input_type,
        )
        .unwrap();

        let mlir_program = traced.to_mlir_module("main").unwrap();
        assert_eq!(
            mlir_program,
            indoc! {r#"
                module {
                  sdy.mesh @mesh = <["x"=2]>
                  func.func @main(%arg0: tensor<8xf32>) -> tensor<8xf32> {
                    %0 = sdy.manual_computation(%arg0) in_shardings=[<@mesh, [{"x"}]>] out_shardings=[<@mesh, [{"x"}]>] manual_axes={"x"} (%arg1: tensor<4xf32>) {
                      %c = stablehlo.constant dense<4> : tensor<i64>
                      %1 = "stablehlo.all_to_all"(%arg1) <{channel_handle = #stablehlo.channel_handle<handle = 1, type = 1>, concat_dimension = 0 : i64, replica_groups = dense<[[0, 1]]> : tensor<1x2xi64>, split_count = 2 : i64, split_dimension = 0 : i64}> {use_global_device_ids} : (tensor<4xf32>) -> tensor<4xf32>
                      sdy.return %1 : tensor<4xf32>
                    } : (tensor<8xf32>) -> tensor<8xf32>
                    return %0 : tensor<8xf32>
                  }
                }
            "#}
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let base = device_index as f32 * 4.0;
                let shard_values = [base + 1.0, base + 2.0, base + 3.0, base + 4.0];
                client
                    .buffer(
                        values_to_bytes::<f32>(&shard_values).as_slice(),
                        BufferType::F32,
                        [4u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[8], sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let program = Program::Mlir { bytecode: mlir_program.into_bytes() };
        let executable = client.compile(&program, &test_spmd_compilation_options(2)).unwrap();

        let execution_devices = executable.addressable_devices().unwrap();
        assert_eq!(execution_devices.len(), 2);
        let execution_device_ids = execution_devices.iter().map(|device| device.id().unwrap()).collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();

        // Shards are [1, 2, 3, 4] and [5, 6, 7, 8]: the exchange leaves device 0 with the two first halves
        // [1, 2, 5, 6] and device 1 with the two second halves [3, 4, 7, 8].
        assert_eq!(outputs.len(), execution_device_ids.len());
        let expected_values_by_device = [[1.0f32, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]];
        for (device_index, output) in outputs.into_iter().enumerate() {
            assert_eq!(output.outputs.len(), 1);
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            let values: [f32; 4] = values_from_bytes::<f32>(output_bytes.as_slice()).try_into().unwrap();
            assert_eq!(values, expected_values_by_device[device_index]);
        }
    }

    #[test]
    fn test_shard_map_untiled_all_to_all_lowers_rank_exchange() {
        use ryft_core::operations::collectives::{AllToAll, CollectiveOptions};

        let plugin = load_cpu_plugin().unwrap();
        let client = plugin
            .client(ClientOptions::CPU(CpuClientOptions { device_count: Some(2) }))
            .expect("failed to create 2-device CPU client");
        let client_devices = client.addressable_devices().unwrap();
        let devices = client_devices.iter().map(|device| Device::from_pjrt(device).unwrap()).collect::<Vec<_>>();
        let device_mesh = DeviceMesh::new(
            LogicalMesh::new(vec![MeshAxis::new("x", 2, MeshAxisType::Manual).unwrap()]).unwrap(),
            devices,
        )
        .unwrap();
        let mesh = device_mesh.logical_mesh().clone();
        let input_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::replicated(), ShardingDimension::sharded(["x"])])
                .unwrap();
        let output_sharding =
            Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"]), ShardingDimension::replicated()])
                .unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let input_sharding = input_sharding.clone();
                let output_sharding = output_sharding.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, _, ArrayType, _>(
                        |local_x: ShardMapTracer| {
                            local_x.all_to_all_with_options("x", 0, 1, CollectiveOptions::default()).unwrap()
                        },
                        x,
                        mesh.clone(),
                        input_sharding.clone(),
                        output_sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(2), Dimension::Static(6)])),
        )
        .unwrap();

        let module = traced.to_mlir_module("main").unwrap();
        assert!(module.contains("stablehlo.broadcast_in_dim"), "{module}");
        assert!(module.contains("stablehlo.all_to_all"), "{module}");
        assert!(module.contains("stablehlo.reshape"), "{module}");
        assert!(
            module.contains("(tensor<2x3xf32>) -> tensor<2x3x1xf32>")
                && module.contains("(tensor<2x3x1xf32>) -> tensor<1x3x2xf32>"),
            "{module}",
        );

        let input_buffers = client_devices
            .iter()
            .enumerate()
            .map(|(device_index, device)| {
                let offset = device_index as f32 * 9.0;
                let values = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 4.0, offset + 5.0, offset + 6.0];
                client
                    .buffer(
                        values_to_bytes(values.as_slice()).as_slice(),
                        BufferType::F32,
                        [2u64, 3u64],
                        None,
                        device.clone(),
                        None,
                    )
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let input_array = Array::from_addressable_buffers(
            &client,
            static_sharded_array_type(DataType::F32, &[2, 6], input_sharding),
            device_mesh,
            input_buffers,
        )
        .unwrap();
        let executable = client
            .compile(&Program::Mlir { bytecode: module.into_bytes() }, &test_spmd_compilation_options(2))
            .unwrap();
        let execution_device_ids = executable
            .addressable_devices()
            .unwrap()
            .iter()
            .map(|device| device.id().unwrap())
            .collect::<Vec<_>>();
        let execute_arguments =
            Array::into_execute_arguments(vec![input_array], execution_device_ids.as_slice()).unwrap();
        let outputs = executable
            .execute(execute_arguments.as_execution_device_inputs(), Vec::new(), 0, None, Some(file!()), None, None)
            .unwrap()
            .block_until_ready()
            .unwrap();
        let expected = [vec![1.0_f32, 10.0, 2.0, 11.0, 3.0, 12.0], vec![4.0_f32, 13.0, 5.0, 14.0, 6.0, 15.0]];
        for (output, expected) in outputs.into_iter().zip(expected) {
            let output_bytes = output.outputs[0].copy_to_host(None).unwrap().r#await().unwrap();
            assert_eq!(values_from_bytes::<f32>(output_bytes.as_slice()), expected);
        }
    }

    /// Verifies that [`TracedXlaProgram::statistics`] delegates to the unsimplified staged program. The asserted
    /// numbers are cross-checked against `traced.program.to_string()`, which renders the entry region as one
    /// `shard_map` instruction attaching a single-`sin` body region.
    #[test]
    fn test_traced_xla_program_statistics_reports_unsimplified_program() {
        let mesh = LogicalMesh::new(vec![MeshAxis::new("x", 4, MeshAxisType::Manual).unwrap()]).unwrap();
        let sharding = Sharding::new(mesh.clone(), vec![ShardingDimension::sharded(["x"])]).unwrap();
        let traced: TracedXlaProgram<ArrayType, ArrayType> = trace(
            {
                let mesh = mesh.clone();
                move |x: ShardMapTracer| {
                    shard_map::<_, ShardMapTracer, ArrayType, ShardMapTracer>(
                        |local_x: ShardMapTracer| local_x.sin().unwrap(),
                        x,
                        mesh.clone(),
                        sharding.clone(),
                        sharding.clone(),
                    )
                    .unwrap()
                }
            },
            ArrayType::new(DataType::F32, Shape::new(vec![Dimension::Static(8)])),
        )
        .unwrap();

        let statistics = traced.statistics();
        assert_eq!(statistics.region_count(), 2);
        assert_eq!(statistics.entry_region_index(), 1);

        let body = &statistics.regions()[0];
        assert_eq!(body.input_count(), 1);
        assert_eq!(body.output_count(), 1);
        assert_eq!(body.instruction_count(), 1);
        assert_eq!(body.constant_count(), 0);
        assert_eq!(body.operation_counts(), &BTreeMap::from([("sin", 1usize)]));
        assert_eq!(body.maximum_output_dependency_depth(), 1);
        assert_eq!(body.attached_regions(), &[]);

        let entry = statistics.entry();
        assert_eq!(entry.input_count(), 1);
        assert_eq!(entry.output_count(), 1);
        assert_eq!(entry.instruction_count(), 1);
        assert_eq!(entry.operation_counts(), &BTreeMap::from([("shard_map", 1usize)]));
        assert_eq!(entry.maximum_output_dependency_depth(), 1);
        assert_eq!(entry.attached_regions().len(), 1);
        let edge = &entry.attached_regions()[0];
        assert_eq!(edge.instruction_index(), 0);
        assert_eq!(edge.operation(), "shard_map");
        assert_eq!(edge.region_slot(), "body");
        assert_eq!(edge.region_role(), RegionRole::Computation);
        assert_eq!(edge.region_index(), 0);
        assert_eq!(edge.label(), "shard_map.body");
    }
}
